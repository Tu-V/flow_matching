"""
Flow Matching training + Mahalanobis regularization (Plan B).

Giống hệt train_shapes_fm.py, nhưng cộng thêm 1 hinge-loss phạt khi model dự
đoán x1_pred (suy ra từ velocity tại x_t, t ngay trong batch training) lệch quá
xa vùng dữ liệu thật (đo bằng Mahalanobis distance, Sigma precompute 1 lần từ
toàn bộ 5000 ảnh thật).

Công thức (đã thống nhất với user):
  x1_pred(x_t, t) = x_t + (1-t) * v_theta(x_t, t)
  D_M(x)^2        = ‖ Wᵀ(vec(x) - mu) ‖^2      (W = V_k / sqrt(lambda_k), pseudo-inverse)
  L_reg           = E[ ReLU( D_M(x1_pred)^2 - tau ) ]
  tau             = percentile 99 của D_M^2 trên chính 5000 ảnh thật
  L               = L_FM + lambda_reg * L_reg

Usage:
    python train_shapes_fm_mahalanobis.py \\
        --data_dir ../neurips-2024-diffusion-model-hallucination/simple-datasets/simple-shapes-5k-16x16 \\
        --epochs 1000 --batch_size 128 --lr 2e-4 --lambda_reg 0.05 \\
        --out_dir shapes_fm_output_mahal_reg
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath          # noqa: E402
from flow_matching.solver import ODESolver              # noqa: E402
from flow_matching.utils import ModelWrapper             # noqa: E402
from models.unet import UNetModel                        # noqa: E402

from train_shapes_fm import build_unet, UNetVelocityWrapper   # noqa: E402
from mahalanobis_guided_sampling import load_dataset_images, MahalanobisTorch, IMG_SIZE  # noqa: E402


# ── Dataset (giống ShapesDataset gốc, nhưng data_dir truyền qua CLI) ───────────
class ShapesDataset(Dataset):
    def __init__(self, root: str):
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".png")
        )
        self.transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ── tau: percentile 99 của D_M^2 trên chính dataset thật ───────────────────────
@torch.no_grad()
def compute_tau(mahal, data_dir, device, percentile=99.0):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    imgs = []
    for p in paths:
        gray = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0   # [0,1]
        raw = gray[None, :, :] * 2 - 1                                          # -> [-1,1]
        imgs.append(np.repeat(raw, 3, axis=0))                                  # (3,H,W)
    X = torch.from_numpy(np.stack(imgs)).float().to(device)                     # (N,3,H,W)
    d2 = mahal.d2(X).cpu().numpy()
    tau = float(np.percentile(d2, percentile))
    print(f"  tau (p{percentile:.0f} của D_M^2 trên {len(paths)} ảnh thật) = {tau:.3f}  "
          f"(mean={d2.mean():.2f}, max={d2.max():.2f})")
    return tau


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Mahalanobis model (precompute 1 lần từ toàn bộ dataset thật) ──
    print(f"\nPrecompute Mahalanobis model từ {args.data_dir} ...")
    X_real = load_dataset_images(args.data_dir)
    mahal = MahalanobisTorch(X_real, device, rel_eps=args.rel_eps)
    tau = compute_tau(mahal, args.data_dir, device)

    # ── Data ──
    dataset = ShapesDataset(args.data_dir)
    print(f"\nDataset size: {len(dataset)} images")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda", drop_last=True,
    )

    # ── Model ──
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params:,}")

    path = CondOTProbPath()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    start_epoch = 1
    if args.resume:
        ckpt_path = _resolve_resume(args.resume, ckpt_dir)
        print(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {ckpt['epoch']}")
        if start_epoch > args.epochs:
            print(f"Đã train đủ {args.epochs} epochs rồi.")
            return

    print(f"\nTraining: epochs={args.epochs}  lambda_reg={args.lambda_reg}  tau={tau:.2f}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss_fm = 0.0
        epoch_loss_reg = 0.0
        n_active = 0
        n_batches = 0

        for batch in loader:
            x_1 = batch.to(device)
            B = x_1.shape[0]

            x_0 = torch.randn_like(x_1)
            t = torch.rand(B, device=device)

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            u_pred = model(x_t, t, extra={})
            loss_fm = torch.pow(u_pred - u_t, 2).mean()

            # ── Mahalanobis regularization ──
            x1_pred = x_t + (1 - t).view(-1, 1, 1, 1) * u_pred
            d2 = mahal.d2(x1_pred)                       # (B,)
            hinge = torch.relu(d2 - tau)
            loss_reg = hinge.mean()

            loss = loss_fm + args.lambda_reg * loss_reg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_fm += loss_fm.item()
            epoch_loss_reg += loss_reg.item()
            n_active += (hinge > 0).sum().item()
            n_batches += 1

        scheduler.step()
        avg_fm = epoch_loss_fm / n_batches
        avg_reg = epoch_loss_reg / n_batches
        active_rate = 100 * n_active / (n_batches * args.batch_size)

        if epoch % args.log_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{args.epochs}  L_FM={avg_fm:.5f}  "
                  f"L_reg={avg_reg:.4f}  active={active_rate:.1f}%  lr={lr:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(ckpt_dir, f"unet_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_fm,
                "loss_reg": avg_reg,
                "lambda_reg": args.lambda_reg,
                "tau": tau,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            _sample_and_save(model, device, epoch, sample_dir, n_samples=64,
                             steps=args.sample_steps)

    print("\nTraining complete.")
    print(f"Checkpoints -> {ckpt_dir}")
    print(f"Samples     -> {sample_dir}")


@torch.no_grad()
def _sample_and_save(model, device, epoch, sample_dir, n_samples=64, steps=100):
    model.eval()
    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    x_init = torch.randn(n_samples, 3, 16, 16, device=device)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
    x_gen = solver.sample(x_init=x_init, step_size=None, method="euler",
                          time_grid=time_grid, return_intermediates=False)
    x_gen = (x_gen.clamp(-1, 1) + 1) / 2

    out_path = os.path.join(sample_dir, f"samples_epoch{epoch:04d}.png")
    save_image(x_gen, out_path, nrow=8, padding=1)
    print(f"  Samples saved: {out_path}")
    model.train()


def _resolve_resume(resume, ckpt_dir):
    if resume == "latest":
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        if not ckpts:
            raise FileNotFoundError(f"Không có checkpoint trong {ckpt_dir}")
        return ckpts[-1]
    return resume


def parse_args():
    p = argparse.ArgumentParser(description="Flow Matching + Mahalanobis regularization")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="shapes_fm_output_mahal_reg")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_reg", type=float, default=0.05)
    p.add_argument("--rel_eps", type=float, default=1e-3)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=50)
    p.add_argument("--sample_steps", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
