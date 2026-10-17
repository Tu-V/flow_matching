"""
Giống train_cnn_pixel01_experiment.py (p0 isotropic, pixel range KHÔNG chuẩn hoá [0,1],
dataset simple-shapes-10k-64x64, 100,000 bước, batch_size=128) nhưng ĐỔI KIẾN TRÚC sang
DiT (tái dùng đúng class DiT/DiTVelocityWrapper từ train_shapes_fm_dit.py, patch_size=4 ->
grid 16x16=256 token, ~7.4M tham số, tương đương quy mô UNet ~9.1M).

Train xong tự động sample --n_sample_total (mặc định 100,000) ảnh × --n_repeats (mặc định
5) lần lặp, tính mean/std hallucination rate bằng hallucination_detector.py dùng chung.

KHÁC BIỆT quan trọng so với train_anisotropic_p0_experiment_dit.py: KHÔNG
transforms.Normalize([0.5],[0.5]) -- giữ nguyên pixel [0,1]; lúc sample map output model
-> ảnh hiển thị bằng clamp(0,1) trực tiếp (KHÔNG (clamp(-1,1)+1)/2).

Usage:
    python train_dit_pixel01_experiment.py
    python train_dit_pixel01_experiment.py --quick
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath      # noqa: E402
from flow_matching.solver import ODESolver          # noqa: E402
from hallucination_detector import analyze_batch, summarize   # noqa: E402
from train_shapes_fm_dit import DiT, DiTVelocityWrapper        # noqa: E402  (tái dùng nguyên class DiT)

IN_CHANNELS = 1
DATASET_ROOT = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "simple-datasets",
)
# OUTPUT_DIR thật sự được gán lại trong main() theo --tag (để chạy nhiều cấu hình
# kiến trúc khác nhau mà không ghi đè lẫn nhau, vd bản lớn hơn dùng --tag _large)
OUTPUT_DIR = os.path.join(REPO_ROOT, "shape_exp", "64_64", "dit_pixel01_isotropic_64x64_output")


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


def build_dit(img_size: int, args) -> DiT:
    assert img_size % args.patch_size == 0, \
        f"img_size={img_size} phải chia hết cho patch_size={args.patch_size}"
    return DiT(
        img_size=img_size, patch_size=args.patch_size, in_channels=IN_CHANNELS,
        hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )


def save_checkpoint(model: DiT, ckpt_path: str, args, **extra):
    torch.save({
        "model_state_dict": model.state_dict(), "arch": "dit",
        "hidden_size": args.hidden_size, "depth": args.depth,
        "num_heads": args.num_heads, "patch_size": args.patch_size,
        "mlp_ratio": args.mlp_ratio, **extra,
    }, ckpt_path)


@torch.no_grad()
def sample_and_analyze(model: DiT, device, n_total: int, steps: int,
                        batch_size: int, img_size: int) -> dict:
    model.eval()
    wrapper = DiTVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    all_analyses = []
    n_done = 0
    t0 = time.time()
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = torch.randn(B, IN_CHANNELS, img_size, img_size, device=device)   # isotropic N(0,I)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        x_01 = x_final.clamp(0, 1)   # pixel range [0,1] -> clamp(0,1) trực tiếp
        x_01_rgb = x_01.repeat(1, 3, 1, 1) if x_01.shape[1] == 1 else x_01
        imgs_np = (x_01_rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        all_analyses.extend(analyze_batch(imgs_np))
        n_done += B
        print(f"    sampled {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", end="\r")
    print()
    return summarize(all_analyses)


def sample_and_analyze_repeated(model: DiT, device, n_total: int, steps: int,
                                 batch_size: int, n_repeats: int, img_size: int) -> dict:
    runs = []
    for rep in range(1, n_repeats + 1):
        print(f"  -- repeat {rep}/{n_repeats} --")
        s = sample_and_analyze(model, device, n_total, steps, batch_size, img_size)
        runs.append(s)
        print(f"     hall={100*s['hall_rate']:.4f}%  "
              f"(empty={100*s['n_empty']/s['n_total']:.4f}%  "
              f"double_col={100*s['n_double_col']/s['n_total']:.4f}%)")

    hall_rates   = np.array([r["hall_rate"] for r in runs]) * 100.0
    empty_rates  = np.array([r["n_empty"] / r["n_total"] for r in runs]) * 100.0
    double_rates = np.array([r["n_double_col"] / r["n_total"] for r in runs]) * 100.0

    return {
        "n_repeats": n_repeats, "n_total_per_repeat": n_total, "runs": runs,
        "hall_rate_pct_list":   hall_rates.tolist(),
        "empty_rate_pct_list":  empty_rates.tolist(),
        "double_rate_pct_list": double_rates.tolist(),
        "hall_rate_mean":   float(hall_rates.mean()),
        "hall_rate_std":    float(hall_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "empty_rate_mean":  float(empty_rates.mean()),
        "empty_rate_std":   float(empty_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "double_rate_mean": float(double_rates.mean()),
        "double_rate_std":  float(double_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--n_data", type=int, default=10000)
    p.add_argument("--total_steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=4,
                   help="64x64/patch=4 -> grid 16x16=256 token (~7.4M tham số, "
                        "nhẹ hơn patch=2 (1024 token))")
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--n_sample_total", type=int, default=100_000)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--tag", type=str, default="",
                   help="Hậu tố thêm vào tên thư mục output (vd '_large') để chạy nhiều "
                        "cấu hình kiến trúc khác nhau mà không ghi đè lẫn nhau")
    p.add_argument("--resume", action="store_true",
                   help="Tim checkpoint step*.pt gan nhat trong checkpoints/ va tiep tuc "
                        "tu do (chi khoi phuc model weight -- optimizer/scheduler state "
                        "KHONG duoc luu truoc do nen se reset, scheduler duoc fast-forward "
                        "cho khop LR tai step do)")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.total_steps, a.n_sample_total, a.n_repeats = 300, 500, 1
        a.log_every, a.save_every = 50, 10**9
    return a


def main():
    a = parse_args()
    device = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(REPO_ROOT, "shape_exp", "64_64",
                               f"dit_pixel01_isotropic_64x64_output{a.tag}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    label = f"{a.n_data // 1000}k" if a.n_data % 1000 == 0 else str(a.n_data)
    data_dir = os.path.join(DATASET_ROOT, f"simple-shapes-{label}-{a.img_size}x{a.img_size}")
    assert os.path.isdir(data_dir), f"Không thấy thư mục dataset: {data_dir}"
    dataset = ShapesDataset01(data_dir, a.img_size)
    print(f"Dataset: {len(dataset)} ảnh thật ({a.img_size}x{a.img_size}) — {data_dir}")
    print(f"  pixel range: KHÔNG chuẩn hoá, giữ nguyên [0,1] (chỉ ToTensor())")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = build_dit(a.img_size, a).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = model.x_embedder.num_patches
    print(f"DiT: hidden={a.hidden_size} depth={a.depth} heads={a.num_heads} "
          f"patch={a.patch_size} tokens={n_tokens}  params={n_params:,}")

    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.total_steps, eta_min=a.lr * 0.1)

    start_step = 1
    if a.resume:
        import glob
        import re
        step_ckpts = glob.glob(os.path.join(ckpt_dir, "dit_pixel01_isotropic_step*.pt"))
        if step_ckpts:
            def _step_num(p):
                m = re.search(r"step(\d+)\.pt$", p)
                return int(m.group(1)) if m else -1
            latest = max(step_ckpts, key=_step_num)
            latest_step = _step_num(latest)
            ck = torch.load(latest, map_location=device, weights_only=True)
            model.load_state_dict(ck["model_state_dict"])
            start_step = latest_step + 1
            for _ in range(latest_step):
                scheduler.step()
            print(f"  [resume] Tìm thấy {os.path.basename(latest)} (step={latest_step}) "
                  f"-> tiếp tục từ step {start_step}  (lr hiện tại={optimizer.param_groups[0]['lr']:.2e})")
        else:
            print("  [resume] Không tìm thấy checkpoint nào, train từ đầu")
        if start_step > a.total_steps:
            print(f"  [resume] start_step={start_step} > total_steps={a.total_steps}, "
                  f"đã train đủ -> bỏ qua training, dùng checkpoint hiện tại")

    n = len(dataset)
    print(f"  preloading {n} images to {device} ...")
    all_imgs = torch.stack([dataset[i] for i in range(n)]).to(device)   # [n,1,H,W], range [0,1]
    print(f"  X stats: min={all_imgs.min():.4f} max={all_imgs.max():.4f} mean={all_imgs.mean():.4f}")

    print(f"\n{'='*70}")
    print(f"n_data={n}  p0=isotropic  img_size={a.img_size}  arch=DiT  params={n_params:,}  "
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

        x_0 = torch.randn(B, IN_CHANNELS, a.img_size, a.img_size, device=device)
        t = torch.rand(B, device=device)

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        u_pred = model(path_sample.x_t, t)
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
            step_ckpt = os.path.join(ckpt_dir, f"dit_pixel01_isotropic_step{step:06d}.pt")
            save_checkpoint(model, step_ckpt, a, n_data=n, p0_type="isotropic",
                             img_size=a.img_size, pixel_range="[0,1]",
                             step=step, total_steps=a.total_steps, loss=loss.item())
            print(f"  [checkpoint] Saved: {step_ckpt}")

    del all_imgs
    final_path = os.path.join(ckpt_dir, "dit_pixel01_isotropic_final.pt")
    save_checkpoint(model, final_path, a, n_data=n, p0_type="isotropic",
                     img_size=a.img_size, pixel_range="[0,1]", step=a.total_steps,
                     total_steps=a.total_steps)
    print(f"\nĐã lưu checkpoint cuối: {final_path}")

    print(f"\nSampling {a.n_sample_total} ảnh x {a.n_repeats} lần lặp "
          f"(p0=isotropic, steps={a.sample_steps}) ...")
    s = sample_and_analyze_repeated(model, device, a.n_sample_total, a.sample_steps,
                                     a.sample_batch_size, a.n_repeats, a.img_size)

    print(f"\nhall={s['hall_rate_mean']:.4f}% ± {s['hall_rate_std']:.4f}%  "
          f"(empty={s['empty_rate_mean']:.4f}%±{s['empty_rate_std']:.4f}%  "
          f"double_col={s['double_rate_mean']:.4f}%±{s['double_rate_std']:.4f}%)")

    out_path = os.path.join(OUTPUT_DIR, "stats_pixel01_isotropic.txt")
    with open(out_path, "w") as f:
        f.write(f"DiT pixel01 isotropic  step={a.total_steps}  n_data={n}  "
                f"pixel_range=[0,1]  img_size={a.img_size}  params={n_params:,}\n")
        f.write(f"n_sample_total/repeat={a.n_sample_total}  n_repeats={a.n_repeats}  "
                f"sample_steps={a.sample_steps}\n\n")
        f.write("Per-repeat hallucination rate (%):\n")
        for rep, hr in enumerate(s["hall_rate_pct_list"], start=1):
            f.write(f"  repeat {rep}: hall={hr:.4f}%  empty={s['empty_rate_pct_list'][rep-1]:.4f}%  "
                    f"double_col={s['double_rate_pct_list'][rep-1]:.4f}%\n")
        f.write(f"\nHallucination : mean={s['hall_rate_mean']:.4f}%  std={s['hall_rate_std']:.4f}%\n")
        f.write(f"  empty img   : mean={s['empty_rate_mean']:.4f}%  std={s['empty_rate_std']:.4f}%\n")
        f.write(f"  double col  : mean={s['double_rate_mean']:.4f}%  std={s['double_rate_std']:.4f}%\n")
    print(f"\nĐã lưu: {out_path}")


if __name__ == "__main__":
    main()
