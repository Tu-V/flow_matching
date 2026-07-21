"""
Flow Matching training với NGUỒN CỐ ĐỊNH (fixed discrete source).

Khác với train_shapes_fm.py (x_0 = torch.randn_like(x_1), Gauss liên tục,
sample lại hoàn toàn mới mỗi bước), script này:

  1. Sinh CỐ ĐỊNH một lần N_SOURCE=5000 điểm nguồn ~ N(0, I) (shape 3x16x16),
     lưu vào fixed_source_points.pt. Tập điểm này không đổi trong suốt quá
     trình train (kể cả khi resume).
  2. Mỗi bước train, x_0 cho một batch được lấy bằng cách chọn CHỈ SỐ ngẫu
     nhiên (uniform, có lặp) trong {0, ..., N_SOURCE-1} rồi index vào tập
     điểm cố định đó. Tức là:

         p_source = Uniform({z_1, ..., z_5000}),   z_i ~ N(0, I) (cố định)

     thay vì p_source = N(0, I) liên tục như trước.
  3. p_data vẫn là phân phối rời rạc đều trên 5000 ảnh shapes 3x16x16
     (ShapesDataset, mỗi ảnh 1 "điểm" p_data).

  => Cả nguồn và đích đều là phân phối rời rạc hữu hạn (5000 atoms mỗi bên).
     Việc ghép cặp (x_0, x_1) là NGẪU NHIÊN ĐỘC LẬP mỗi batch/epoch (không
     cố định 1-1 theo index) — đúng chuẩn conditional flow matching (CFM),
     chỉ khác là marginal của x_0 giờ có support hữu hạn thay vì toàn R^d.

Usage:
    # Train từ đầu (tự sinh + lưu 5000 điểm nguồn nếu chưa có)
    python train_shapes_fm_fixedsrc.py --epochs 500 --batch_size 128 --lr 1e-4

    # Resume từ checkpoint mới nhất (dùng lại đúng tập nguồn đã lưu)
    python train_shapes_fm_fixedsrc.py --resume latest --epochs 500

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
from flow_matching.solver import ODESolver             # noqa: E402
from flow_matching.utils import ModelWrapper           # noqa: E402
from models.unet import UNetModel                      # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    REPO_ROOT,
    "..",
    "neurips-2024-diffusion-model-hallucination",
    "simple-datasets",
    "simple-shapes-5k-16x16",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_fixedsrc_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
SOURCE_POINTS_PATH = os.path.join(OUTPUT_DIR, "fixed_source_points.pt")

DATA_SHAPE = (3, 16, 16)


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
                transforms.Resize((16, 16)),
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


# ── Fixed source points ────────────────────────────────────────────────────────
def load_or_create_source_points(
    path: str, n_source: int, seed: int, shape=DATA_SHAPE
) -> torch.Tensor:
    """
    Trả về tensor [n_source, *shape] ~ N(0, I), SINH VÀ LƯU 1 LẦN DUY NHẤT.
    Nếu file đã tồn tại, load lại nguyên vẹn (đảm bảo tập nguồn không đổi
    giữa các lần train/resume/sample).
    """
    if os.path.exists(path):
        pts = torch.load(path, weights_only=True)
        if pts.shape != (n_source, *shape):
            raise ValueError(
                f"fixed_source_points.pt hiện có shape {tuple(pts.shape)}, "
                f"khác với yêu cầu {(n_source, *shape)}. "
                f"Xoá file hoặc đổi --n_source / --source_seed nếu muốn sinh lại."
            )
        print(f"Loaded fixed source points: {path}  shape={tuple(pts.shape)}")
        return pts

    os.makedirs(os.path.dirname(path), exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(n_source, *shape, generator=g)
    torch.save(pts, path)
    print(f"Generated + saved {n_source} fixed source points (seed={seed}) -> {path}")
    return pts


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


# ── Build UNet ────────────────────────────────────────────────────────────────
def build_unet() -> UNetModel:
    """
    64 model_channels, 3 levels via channel_mult=[1,2,2],
    3 residual blocks per level, attention at 2× downsampling.
    For 16×16 input: levels run at 16×16 → 8×8 → 4×4.
    (Giống hệt kiến trúc trong train_shapes_fm.py để dễ so sánh.)
    """
    return UNetModel(
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=3,
        attention_resolutions=(2,),   # attention at 8×8
        dropout=0.1,
        channel_mult=(1, 2, 2),       # 3 levels
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
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

    # ── Data (target p_data — 5k ảnh shapes) ──
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

    # ── Fixed source points (p_source — 5000 điểm Gauss cố định) ──
    source_points = load_or_create_source_points(
        SOURCE_POINTS_PATH, args.n_source, args.source_seed
    ).to(device)
    print(f"Source distribution: Uniform over {args.n_source} fixed Gaussian points")

    # ── Model ──
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params:,}")

    # ── Flow matching path ──
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
    n_source = source_points.shape[0]
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x_1 = batch.to(device)                          # [B, 3, 16, 16], in [-1,1]
            B = x_1.shape[0]

            # x_0 ~ Uniform({z_1,...,z_5000}): chọn chỉ số ngẫu nhiên (có lặp)
            # MỚI mỗi bước -> ghép cặp (x_0, x_1) độc lập, đổi liên tục qua
            # các batch/epoch (đúng coupling độc lập của CFM), KHÔNG cố định
            # theo kiểu source[i] <-> data[i].
            idx0 = torch.randint(0, n_source, (B,), device=device)
            x_0 = source_points[idx0]

            t = torch.rand(B, device=device)                 # uniform t ∈ [0,1]

            # Sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t                           # target velocity

            # Predict velocity and compute MSE loss
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

        # ── Save checkpoint ──
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(CKPT_DIR, f"unet_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "n_source": n_source,
                    "source_points_path": SOURCE_POINTS_PATH,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        # ── Generate samples at milestones ──
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            _sample_and_save(model, source_points, device, epoch, n_samples=64,
                              steps=args.sample_steps)

    print("Training complete.")
    print(f"Checkpoints    → {CKPT_DIR}")
    print(f"Samples        → {SAMPLE_DIR}")
    print(f"Source points  → {SOURCE_POINTS_PATH}")


# ── Sampling (dùng trong lúc train để theo dõi tiến độ) ────────────────────────
@torch.no_grad()
def _sample_and_save(model, source_points, device, epoch, n_samples=64, steps=100):
    """Lấy x_init bằng cách sample UNIFORM (có lặp) từ tập nguồn cố định,
    rồi tích phân ODE — nhất quán với cách sample_fixedsrc_fm.py sẽ dùng
    sau khi train xong."""
    model.eval()

    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    n_source = source_points.shape[0]
    idx = torch.randint(0, n_source, (n_samples,), device=device)
    x_init = source_points[idx]

    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    x_gen = solver.sample(
        x_init=x_init,
        step_size=None,          # step_size=None → use time_grid spacing
        method="euler",
        time_grid=time_grid,
        return_intermediates=False,
    )

    # Rescale [-1, 1] → [0, 1] for saving
    x_gen = (x_gen.clamp(-1, 1) + 1) / 2

    out_path = os.path.join(SAMPLE_DIR, f"samples_epoch{epoch:04d}.png")
    save_image(x_gen, out_path, nrow=8, padding=1)
    print(f"  Samples saved: {out_path}")

    model.train()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_resume(resume: str) -> str:
    """'latest' → path checkpoint mới nhất; path khác giữ nguyên."""
    if resume == "latest":
        import glob
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
        if not ckpts:
            raise FileNotFoundError(f"Không có checkpoint trong {CKPT_DIR}")
        return ckpts[-1]
    return resume


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Flow Matching (fixed discrete source) trên simple-shapes-5k-16x16"
    )
    p.add_argument("--epochs",       type=int,   default=300,
                   help="Tổng số epoch mục tiêu (default: 300)")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path checkpoint để resume, hoặc 'latest'")
    p.add_argument("--n_source",     type=int,   default=5000,
                   help="Số điểm nguồn cố định (default: 5000)")
    p.add_argument("--source_seed",  type=int,   default=0,
                   help="Seed sinh tập điểm nguồn cố định (default: 0). "
                        "Chỉ dùng khi fixed_source_points.pt chưa tồn tại.")
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--save_every",   type=int,   default=50)
    p.add_argument("--sample_every", type=int,   default=50)
    p.add_argument("--sample_steps", type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
