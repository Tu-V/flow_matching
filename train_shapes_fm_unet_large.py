"""
Flow Matching training với UNet PHÓNG TO (~48M tham số, gấp ~5.3x baseline 9.1M),
để so sánh xem hallucination có giảm theo scale (nhiều tham số hơn, cùng kiến trúc
convolution/translation-equivariant) hay không — đối chứng với hướng đổi hẳn kiến trúc
(DiT/MLP) đã thử trước đó.

Khác biệt so với train_shapes_fm.py (baseline UNet 9.1M):
    model_channels        : 64  -> 128
    num_res_blocks        : 3   -> 4
    attention_resolutions : (2,) -> (1, 2, 4)   (attention ở mọi level, không chỉ 8x8)
    channel_mult giữ nguyên (1, 2, 2) — vẫn 3 level (16x16 -> 8x8 -> 4x4)

=> UNet params: ~48.3M (build_unet() in-file để verify).

Source distribution VẪN LÀ GAUSS liên tục (x_0 = torch.randn_like(x_1)), data vẫn là
5k ảnh 3x16x16 — giống hệt các script train khác, chỉ đổi CỠ kiến trúc, để so sánh công bằng.

Usage:
    python train_shapes_fm_unet_large.py --epochs 1000 --batch_size 128 --lr 1e-4

    # Resume từ checkpoint mới nhất
    python train_shapes_fm_unet_large.py --resume latest --epochs 1000

    # Tự chỉnh cỡ model (channel_mult / attention_resolutions truyền dạng "1,2,2")
    python train_shapes_fm_unet_large.py --model_channels 128 --num_res_blocks 4 \
        --channel_mult 1,2,2 --attention_resolutions 1,2,4

--epochs luôn là TỔNG số epoch mục tiêu (không phải số epoch thêm vào).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath          # noqa: E402
from flow_matching.solver import ODESolver              # noqa: E402
from flow_matching.utils import ModelWrapper            # noqa: E402
from models.unet import UNetModel                       # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    REPO_ROOT,
    "..",
    "neurips-2024-diffusion-model-hallucination",
    "simple-datasets",
    "simple-shapes-5k-16x16",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_unet_large_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

IMG_SIZE = 16
IN_CHANNELS = 3


# ── Dataset ───────────────────────────────────────────────────────────────────
class ShapesDataset(Dataset):
    """Loads all PNGs in a flat directory as 16x16 RGB tensors scaled to [-1, 1]."""

    def __init__(self, root: str):
        self.paths = sorted(
            [
                os.path.join(root, f)
                for f in os.listdir(root)
                if f.lower().endswith(".png")
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),                      # [0, 1]
                transforms.Normalize([0.5, 0.5, 0.5],      # → [-1, 1]
                                     [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ── Model wrapper for ODESolver ────────────────────────────────────────────────
class UNetVelocityWrapper(ModelWrapper):
    """
    Bridges ODESolver's call convention  (x=x, t=t)
    to UNetModel's forward signature     (x, timesteps, extra).
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t, extra=extras)


# ── Build UNet (phóng to) ────────────────────────────────────────────────────
def build_unet(args) -> UNetModel:
    return UNetModel(
        in_channels=IN_CHANNELS,
        model_channels=args.model_channels,
        out_channels=IN_CHANNELS,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(args.attention_resolutions),
        dropout=0.1,
        channel_mult=tuple(args.channel_mult),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=args.num_heads,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
        with_fourier_features=False,
    )


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Data ──
    dataset = ShapesDataset(DATA_DIR)
    print(f"Dataset size: {len(dataset)} images")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ── Model ──
    model = build_unet(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet (large): model_channels={args.model_channels}  num_res_blocks={args.num_res_blocks}  "
          f"channel_mult={tuple(args.channel_mult)}  attention_resolutions={tuple(args.attention_resolutions)}  "
          f"params={n_params:,}")

    # ── Flow matching path (giống hệt baseline) ──
    path = CondOTProbPath()

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # ── Resume từ checkpoint ──
    start_epoch = 1
    if args.resume:
        ckpt_path = _resolve_resume(args.resume)
        print(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {ckpt['epoch']}  loss={ckpt.get('loss', float('nan')):.5f}")
        if start_epoch > args.epochs:
            print(f"Đã train đủ {args.epochs} epochs rồi. Tăng --epochs nếu muốn train thêm.")
            return

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x_1 = batch.to(device)                          # [B, 3, 16, 16], in [-1,1]
            B = x_1.shape[0]

            x_0 = torch.randn_like(x_1)                     # Gaussian source (KHÔNG đổi so với baseline)
            t = torch.rand(B, device=device)                 # uniform t ∈ [0,1]

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t                           # target velocity

            u_pred = model(x_t, t, extra={})
            loss = torch.pow(u_pred - u_t, 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if epoch % args.log_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.5f}  lr={lr:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(CKPT_DIR, f"unet_large_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "arch": "unet_large",
                    "model_channels": args.model_channels,
                    "num_res_blocks": args.num_res_blocks,
                    "channel_mult": list(args.channel_mult),
                    "attention_resolutions": list(args.attention_resolutions),
                    "num_heads": args.num_heads,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            _sample_and_save(model, device, epoch, n_samples=64, steps=args.sample_steps)

    print("Training complete.")
    print(f"Checkpoints → {CKPT_DIR}")
    print(f"Samples     → {SAMPLE_DIR}")


# ── Sampling ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def _sample_and_save(model, device, epoch, n_samples=64, steps=100):
    model.eval()

    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    x_init = torch.randn(n_samples, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    x_gen = solver.sample(
        x_init=x_init,
        step_size=None,
        method="euler",
        time_grid=time_grid,
        return_intermediates=False,
    )

    x_gen = (x_gen.clamp(-1, 1) + 1) / 2
    out_path = os.path.join(SAMPLE_DIR, f"samples_epoch{epoch:04d}.png")
    save_image(x_gen, out_path, nrow=8, padding=1)
    print(f"  Samples saved: {out_path}")

    model.train()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_resume(resume: str) -> str:
    if resume == "latest":
        import glob
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
        if not ckpts:
            raise FileNotFoundError(f"Không có checkpoint trong {CKPT_DIR}")
        return ckpts[-1]
    return resume


def _parse_int_tuple(s: str) -> tuple:
    return tuple(int(x) for x in s.split(","))


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Flow Matching (UNet phóng to ~48M) trên simple-shapes-5k-16x16")
    p.add_argument("--epochs",       type=int,   default=1000,
                   help="Tổng số epoch mục tiêu (default: 1000, khớp UNet/DiT baseline)")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path checkpoint để resume, hoặc 'latest'")
    # UNet hyperparams (~48.3M params với default; baseline gốc 9.1M dùng
    # --model_channels 64 --num_res_blocks 3 --attention_resolutions 2)
    p.add_argument("--model_channels", type=int, default=128,
                   help="Phải chia hết cho 32 (GroupNorm32). default: 128")
    p.add_argument("--num_res_blocks", type=int, default=4)
    p.add_argument("--channel_mult", type=_parse_int_tuple, default=(1, 2, 2),
                   help="vd '1,2,2' -> 3 level (16x16 -> 8x8 -> 4x4)")
    p.add_argument("--attention_resolutions", type=_parse_int_tuple, default=(1, 2, 4),
                   help="downsample factor áp dụng attention, vd '1,2,4' -> attention ở mọi level")
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--save_every",   type=int,   default=50)
    p.add_argument("--sample_every", type=int,   default=50)
    p.add_argument("--sample_steps", type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
