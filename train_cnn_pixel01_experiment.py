"""
Train 1 model CNN (UNetModel, kiến trúc giống hệt build_unet() trong
train_anisotropic_p0_experiment.py) với p0 = N(0,I) ISOTROPIC thuần (không có nhánh
anisotropic), trên dataset simple-shapes-10k-64x64 thật.

KHÁC BIỆT DUY NHẤT so với train_anisotropic_p0_experiment.py: KHÔNG chuẩn hoá ảnh về
[-0.5, 0.5] (bỏ transforms.Normalize([0.5],[0.5])) -- giữ nguyên pixel value thô trong
[0, 1] (chỉ ToTensor()). Dataset x1 vì vậy nằm trong [0,1] thay vì [-1,1], ảnh hưởng tới
công thức flow matching (x_t, dx_t) và cách map output model về ảnh hiển thị (/255) lúc
sample -- clamp(0,1) thay vì clamp(-1,1) rồi +1)/2.

10,000 sample, 100,000 bước train, batch_size=128. Lưu checkpoint định kỳ + cuối cùng để
phân tích sau (KHÔNG chạy sampling/hallucination-rate ngay trong script này).

Usage:
    python train_cnn_pixel01_experiment.py
    python train_cnn_pixel01_experiment.py --quick
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath      # noqa: E402
from models.unet import UNetModel                    # noqa: E402

IN_CHANNELS = 1
DATASET_ROOT = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "simple-datasets",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "shape_exp", "64_64", "cnn_pixel01_isotropic_64x64_output")


# ── Dataset: KHÔNG Normalize -- giữ nguyên [0,1] ──────────────────────────────
class ShapesDataset01(Dataset):
    def __init__(self, root: str, img_size: int):
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".png")
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),   # -> [0,1], KHÔNG Normalize([0.5],[0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("L"))


def build_unet() -> UNetModel:
    return UNetModel(
        in_channels=IN_CHANNELS, model_channels=64, out_channels=IN_CHANNELS,
        num_res_blocks=3, attention_resolutions=(2,), dropout=0.1,
        channel_mult=(1, 2, 2), conv_resample=True, dims=2,
        num_classes=None, use_checkpoint=False, num_heads=1,
        num_head_channels=-1, num_heads_upsample=-1,
        use_scale_shift_norm=True, resblock_updown=False,
        use_new_attention_order=True, with_fourier_features=False,
    )


def save_checkpoint(model: UNetModel, ckpt_path: str, optimizer=None, scheduler=None, **extra):
    state = {"model_state_dict": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, ckpt_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--n_data", type=int, default=10000)
    p.add_argument("--total_steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    a = p.parse_args()
    if a.quick:
        a.total_steps, a.log_every, a.save_every = 300, 50, 10**9
    return a


def main():
    a = parse_args()
    device = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    label = f"{a.n_data // 1000}k" if a.n_data % 1000 == 0 else str(a.n_data)
    data_dir = os.path.join(DATASET_ROOT, f"simple-shapes-{label}-{a.img_size}x{a.img_size}")
    assert os.path.isdir(data_dir), f"Không thấy thư mục dataset: {data_dir}"
    dataset = ShapesDataset01(data_dir, a.img_size)
    print(f"Dataset: {len(dataset)} ảnh thật ({a.img_size}x{a.img_size}) — {data_dir}")
    print(f"  pixel range: KHÔNG chuẩn hoá, giữ nguyên [0,1] (chỉ ToTensor())")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet: params={n_params:,}")

    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.total_steps, eta_min=a.lr * 0.1)

    start_step = 1
    last_ckpt = os.path.join(ckpt_dir, "cnn_pixel01_isotropic_last.pt")
    if a.resume and os.path.isfile(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"  [resume] tiếp tục từ step {start_step}")

    n = len(dataset)
    print(f"  preloading {n} images to {device} ...")
    all_imgs = torch.stack([dataset[i] for i in range(n)]).to(device)   # [n,1,H,W], range [0,1]
    print(f"  X stats: min={all_imgs.min():.4f} max={all_imgs.max():.4f} mean={all_imgs.mean():.4f}")

    print(f"\n{'='*70}")
    print(f"n_data={n}  p0=isotropic  img_size={a.img_size}  params={n_params:,}  "
          f"total_steps={a.total_steps}  batch_size={a.batch_size}  pixel_range=[0,1]  "
          f"start_step={start_step}")
    print(f"{'='*70}")

    model.train()
    loss_window = []
    t0 = time.time()
    for step in range(start_step, a.total_steps + 1):
        idx = torch.randint(0, n, (a.batch_size,), device=device)
        x_1 = all_imgs[idx]
        B = x_1.shape[0]

        x_0 = torch.randn(B, IN_CHANNELS, a.img_size, a.img_size, device=device)   # isotropic N(0,I)
        t = torch.rand(B, device=device)

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        u_pred = model(path_sample.x_t, t, extra={})
        loss = torch.pow(u_pred - path_sample.dx_t, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_window.append(loss.item())
        if step % a.log_every == 0 or step == 1:
            avg = sum(loss_window) / len(loss_window)
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step {step:7d}/{a.total_steps}  loss={avg:.5f}  lr={lr:.2e}  "
                  f"elapsed={time.time()-t0:.0f}s")
            loss_window = []

        if (step % a.save_every == 0) and step != a.total_steps:
            step_ckpt = os.path.join(ckpt_dir, f"cnn_pixel01_isotropic_step{step:06d}.pt")
            save_checkpoint(model, step_ckpt, optimizer=optimizer, scheduler=scheduler,
                             n_data=n, p0_type="isotropic", img_size=a.img_size,
                             pixel_range="[0,1]", step=step, total_steps=a.total_steps,
                             loss=loss.item())
            save_checkpoint(model, last_ckpt, optimizer=optimizer, scheduler=scheduler,
                             n_data=n, p0_type="isotropic", img_size=a.img_size,
                             pixel_range="[0,1]", step=step, total_steps=a.total_steps,
                             loss=loss.item())
            print(f"  [checkpoint] Saved: {step_ckpt}")

    final_path = os.path.join(ckpt_dir, "cnn_pixel01_isotropic_final.pt")
    save_checkpoint(model, final_path, optimizer=optimizer, scheduler=scheduler,
                     n_data=n, p0_type="isotropic", img_size=a.img_size,
                     pixel_range="[0,1]", step=a.total_steps, total_steps=a.total_steps)
    if os.path.isfile(last_ckpt):
        os.remove(last_ckpt)
    print(f"\nĐã lưu checkpoint cuối: {final_path}")


if __name__ == "__main__":
    main()
