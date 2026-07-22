"""
Flow Matching training CÓ CONDITIONING (class label) trên bộ shapes_5k_labeled/
(sinh bởi label_shapes_dataset.py — 7 lớp theo tập cột có shape, xem CLASS_DESC).

Dùng UNetModel với num_classes=7 (đúng khả năng conditioning có sẵn của
examples/image/models/unet.py — xem forward(): nếu extra["label"] có mặt thì cộng
label embedding vào time-embedding; nếu không có "label" trong extra thì tự dùng
"null embedding" (index num_classes) -> unconditional).

Classifier-free guidance (CFG) training: mỗi BATCH (không phải mỗi sample), với xác
suất --class_drop_prob (default 0.2, khớp examples/image/training/train_loop.py) sẽ
bỏ hẳn label (extra={}) để model học luôn cả nhánh unconditional -> cho phép CFG lúc
sample (xem CFGScaledModel, --sample_cfg_scale).

Kiến trúc: UNet ~7M (model_channels=64, num_res_blocks=2, channel_mult=(1,2,2),
attention_resolutions=(2,)) — nhỏ hơn baseline gốc 9.1M (num_res_blocks=3), theo đúng
yêu cầu "UNet ban đầu khoảng 7M".

Source distribution: Gauss liên tục x_0 = torch.randn_like(x_1), giống mọi bản UNet
trước đó — chỉ thêm conditioning, không đổi gì khác.

Usage:
    python train_shapes_fm_cond.py --epochs 1000 --batch_size 128 --lr 2e-4

    # Resume
    python train_shapes_fm_cond.py --resume latest --epochs 1000

--epochs luôn là TỔNG số epoch mục tiêu (không phải số epoch thêm vào).
"""

import argparse
import csv
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath           # noqa: E402
from flow_matching.solver import ODESolver                # noqa: E402
from flow_matching.utils import ModelWrapper              # noqa: E402
from models.unet import UNetModel                         # noqa: E402

LABELED_DATA_DIR = os.path.join(REPO_ROOT, "shapes_5k_labeled")
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_cond_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

IMG_SIZE = 16
IN_CHANNELS = 3
NUM_CLASSES = 7   # class 1..7 (class 0/empty không tồn tại trong data thật, xem stats.txt)

CLASS_DESC = {
    0: "chi cot 1 (triangle)",
    1: "chi cot 2 (square)",
    2: "chi cot 3 (pentagon)",
    3: "cot 1 + cot 2",
    4: "cot 1 + cot 3",
    5: "cot 2 + cot 3",
    6: "ca 3 cot",
}   # index 0-based (= class_user - 1), khớp label_emb index


# ── Dataset ───────────────────────────────────────────────────────────────────
class LabeledShapesDataset(Dataset):
    """
    Đọc shapes_5k_labeled/labels.csv (sinh bởi label_shapes_dataset.py).
    Trả về (image [-1,1], label 0-indexed 0..6)  — label_user (1..7) - 1.
    """

    def __init__(self, root: str):
        csv_path = os.path.join(root, "labels.csv")
        self.items = []   # list of (path, label0)
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                cls = int(row["class"])
                if cls == 0:
                    continue   # anomaly/empty, không dùng (thực tế = 0 ảnh)
                path = os.path.join(root, f"class_{cls}", row["filename"])
                self.items.append((path, cls - 1))

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label0 = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label0


# ── Model ─────────────────────────────────────────────────────────────────────
def build_unet() -> UNetModel:
    """~7M params: model_channels=64, num_res_blocks=2 (baseline gốc dùng 3 -> 9.1M)."""
    return UNetModel(
        in_channels=IN_CHANNELS, model_channels=64, out_channels=IN_CHANNELS,
        num_res_blocks=2, attention_resolutions=(2,), dropout=0.1,
        channel_mult=(1, 2, 2), conv_resample=True, dims=2,
        num_classes=NUM_CLASSES, use_checkpoint=False, num_heads=1,
        num_head_channels=-1, num_heads_upsample=-1,
        use_scale_shift_norm=True, resblock_updown=False,
        use_new_attention_order=True, with_fourier_features=False,
    )


class CFGScaledModel(ModelWrapper):
    """
    forward(x, t, cfg_scale=0.0, label=None):
      - label=None                -> unconditional (extra={})
      - label!=None, cfg_scale=0  -> conditional thuần (1 forward pass)
      - label!=None, cfg_scale!=0 -> classifier-free guidance (2 forward pass):
            result = (1+cfg_scale)*conditional - cfg_scale*unconditional
    (Giống examples/image/training/eval_loop.py, bỏ torch.cuda.amp.autocast() để
    chạy được trên MPS/CPU chứ không chỉ CUDA.)
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cfg_scale: float = 0.0, label: torch.Tensor = None) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if label is None:
            return self.model(x, t, extra={})
        if cfg_scale != 0.0:
            conditional = self.model(x, t, extra={"label": label})
            unconditional = self.model(x, t, extra={})
            return (1.0 + cfg_scale) * conditional - cfg_scale * unconditional
        return self.model(x, t, extra={"label": label})


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
    dataset = LabeledShapesDataset(LABELED_DATA_DIR)
    print(f"Dataset size: {len(dataset)} images  (num_classes={NUM_CLASSES})")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda", drop_last=True,
    )

    # ── Model ──
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet (conditional): params={n_params:,}  class_drop_prob={args.class_drop_prob}")

    path = CondOTProbPath()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

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

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for samples, labels in loader:
            x_1 = samples.to(device)
            labels = labels.to(device)
            B = x_1.shape[0]

            # CFG training: drop conditioning CẢ BATCH với xác suất class_drop_prob
            # (khớp convention examples/image/training/train_loop.py, không phải per-sample).
            if torch.rand(1).item() < args.class_drop_prob:
                conditioning = {}
            else:
                conditioning = {"label": labels}

            x_0 = torch.randn_like(x_1)
            t = torch.rand(B, device=device)

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            u_pred = model(x_t, t, extra=conditioning)
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
            ckpt_path = os.path.join(CKPT_DIR, f"unet_cond_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "arch": "unet_cond",
                    "num_classes": NUM_CLASSES,
                    "class_drop_prob": args.class_drop_prob,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            _sample_and_save(model, device, epoch, n_per_class=8,
                              steps=args.sample_steps, cfg_scale=args.sample_cfg_scale)

    print("Training complete.")
    print(f"Checkpoints → {CKPT_DIR}")
    print(f"Samples     → {SAMPLE_DIR}")


# ── Sampling (theo dõi tiến độ, 1 hàng/class) ──────────────────────────────────
@torch.no_grad()
def _sample_and_save(model, device, epoch, n_per_class=8, steps=100, cfg_scale=0.0):
    model.eval()

    wrapper = CFGScaledModel(model)
    solver = ODESolver(velocity_model=wrapper)

    labels_grid = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_class)  # [7*n]
    x_init = torch.randn(labels_grid.shape[0], IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    x_gen = solver.sample(
        x_init=x_init, step_size=None, method="euler", time_grid=time_grid,
        return_intermediates=False, cfg_scale=cfg_scale, label=labels_grid,
    )

    x_gen = (x_gen.clamp(-1, 1) + 1) / 2
    out_path = os.path.join(SAMPLE_DIR, f"samples_epoch{epoch:04d}.png")
    save_image(x_gen, out_path, nrow=n_per_class, padding=1)
    print(f"  Samples saved: {out_path}  (mỗi hàng = 1 class, thứ tự 0..{NUM_CLASSES-1}: "
          + ", ".join(CLASS_DESC[i] for i in range(NUM_CLASSES)) + ")")

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


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Flow Matching class-conditional (UNet ~7M) trên shapes_5k_labeled")
    p.add_argument("--epochs",       type=int,   default=1000,
                   help="Tổng số epoch mục tiêu (default: 1000)")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path checkpoint để resume, hoặc 'latest'")
    p.add_argument("--class_drop_prob", type=float, default=0.2,
                   help="Xác suất drop conditioning MỖI BATCH để train nhánh unconditional (CFG). default: 0.2")
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--save_every",   type=int,   default=50)
    p.add_argument("--sample_every", type=int,   default=50)
    p.add_argument("--sample_steps", type=int,   default=100)
    p.add_argument("--sample_cfg_scale", type=float, default=0.0,
                   help="CFG scale dùng khi sample preview lúc train (default: 0.0 = conditional thuần, không guidance)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
