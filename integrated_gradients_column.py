"""
Integrated Gradients (IG) — giải thích ảnh hưởng của noise x_T lên TỪNG PIXEL
output ở 1 cột ảnh cụ thể (mặc định cột 1 = triangle).

Khác với việc gộp cả cột thành 1 scalar, ở đây MỖI pixel output trong cột target
(vd 16 hàng x 5 cột = 80 pixel với cột triangle) có 1 saliency map RIÊNG
(kích thước 3x16x16, giống noise) — attribution(pixel_out) -> toàn bộ noise.

Baseline: x' = 0 (noise "đen hết").
Input:    x  = noise_init.pt (x_T thật của case).
Target pixel (r,c): F_{r,c}(x_T) = x_0_gray(x_T)[r, c]   (giá trị grayscale
  output tại đúng pixel đó — grayscale = trung bình 3 kênh, khớp cách
  hallucination_detector xử lý ảnh).
ODE rollout dùng steps=25 (khớp toàn bộ phân tích trước đó).

Công thức IG (xấp xỉ Riemann, m bước) cho từng pixel target:
    IG_{r,c}(x)_i = x_i * (1/m) * sum_{k=1..m} dF_{r,c}(alpha_k * x) / dx_i

Cách tính: chạy 1 lần forward (m alpha, batched) qua ODE, rồi với MỖI pixel
target, gọi backward riêng (retain_graph=True) để lấy gradient riêng cho pixel
đó — tái sử dụng đúng 1 đồ thị tính toán, không phải chạy lại ODE 80 lần.

Usage:
    python integrated_gradients_column.py \\
        --case_dir shapes_fm_output/hallucination_analysis/traces/HALL_case_0109_idx92812 \\
        --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt \\
        --steps 25 --target_col 1 --m_steps 30
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from sample_shapes_fm import load_model, UNetVelocityWrapper, decode, to_uint8_numpy, _latest_ckpt   # noqa: E402
from hallucination_detector import analyze_image   # noqa: E402

IMG_SIZE = 16
ZONE_NAMES = ["triangle", "square", "pentagon"]
ZONE_SLICES = [(0, 5), (5, 10), (10, 15)]
ZOOM = 8


def upscale(img, z=ZOOM):
    return img.repeat(z, axis=0).repeat(z, axis=1)


def euler_rollout_batch_grad(wrapper, x_init_batch, steps, device):
    """Giống euler_rollout_batch nhưng giữ đồ thị tính toán (không no_grad)."""
    dt = 1.0 / steps
    x = x_init_batch
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)
        v = wrapper(x, t_tensor)
        x = x + dt * v
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--target_col", type=int, default=1, choices=[1, 2, 3],
                        help="Cột ẢNH cần giải thích: 1=triangle, 2=square, 3=pentagon")
    parser.add_argument("--m_steps", type=int, default=30,
                        help="Số điểm alpha xấp xỉ tích phân Riemann")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.case_dir, "integrated_gradients")
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt_path = args.ckpt or _latest_ckpt()
    model = load_model(ckpt_path, device)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapper = UNetVelocityWrapper(model)

    x_T = torch.load(os.path.join(args.case_dir, "noise_init.pt"), weights_only=True).to(device)  # (1,3,16,16)
    print(f"x_T shape: {tuple(x_T.shape)}")

    t0, t1 = ZONE_SLICES[args.target_col - 1]
    target_name = ZONE_NAMES[args.target_col - 1]
    col_w = t1 - t0
    print(f"Target: từng pixel output trong cột '{target_name}' (x[{t0}:{t1}]), "
          f"{IMG_SIZE}x{col_w} = {IMG_SIZE*col_w} pixel,  ODE steps={args.steps}")

    with torch.no_grad():
        x_0_ref = euler_rollout_batch_grad(wrapper, x_T, args.steps, device)
    img_ref = to_uint8_numpy(decode(x_0_ref)[0])
    r0 = analyze_image(img_ref)
    print(f"Reference x_0: hall={r0['is_hallucination']} type={r0['hall_type']} "
          f"blobs={r0['col_blobs']}")

    # ── Forward 1 lần cho m alpha (batched), giữ graph để backward nhiều lần ──
    m = args.m_steps
    alphas = torch.linspace(1.0 / m, 1.0, m, device=device).view(m, 1, 1, 1)
    x_T_rep = x_T.repeat(m, 1, 1, 1)
    x_alpha = (alphas * x_T_rep).clone().requires_grad_(True)

    print(f"\nForward ODE (m={m} alpha, steps={args.steps}) ...")
    x_0_alpha = euler_rollout_batch_grad(wrapper, x_alpha, args.steps, device)   # (m,3,16,16)
    x_0_alpha_gray = x_0_alpha.mean(dim=1)                                       # (m,16,16)

    # ── IG riêng cho từng pixel target (r, c) trong cột ────────────────────────
    print(f"Backward riêng cho {IMG_SIZE*col_w} pixel target ...")
    IG_per_pixel = np.zeros((IMG_SIZE, col_w, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    n_pixels = IMG_SIZE * col_w
    k = 0
    for r in range(IMG_SIZE):
        for c in range(col_w):
            k += 1
            is_last = (k == n_pixels)
            F_pixel = x_0_alpha_gray[:, r, t0 + c]              # (m,)
            grad_pixel = torch.autograd.grad(
                F_pixel.sum(), x_alpha, retain_graph=not is_last
            )[0]                                                 # (m,3,16,16)
            avg_grad = grad_pixel.mean(dim=0)                     # (3,16,16)
            IG_pixel = (x_T[0] * avg_grad).detach().cpu().numpy()
            IG_per_pixel[r, c] = IG_pixel

    print("Xong.")

    # ── Lưu số liệu đầy đủ ────────────────────────────────────────────────────
    np.savez(os.path.join(out_dir, f"ig_per_pixel_col{args.target_col}.npz"),
             IG_per_pixel=IG_per_pixel, target_col=args.target_col,
             target_name=target_name, t0=t0, t1=t1)
    print(f"Saved: {os.path.join(out_dir, f'ig_per_pixel_col{args.target_col}.npz')}  "
          f"shape={IG_per_pixel.shape}")

    # ── Vẽ: grid 16 hàng x col_w cột, mỗi ô là 1 saliency map |IG| (16x16) ────
    IG_mean_abs_per_pixel = np.abs(IG_per_pixel).mean(axis=2)   # (16,col_w,16,16) - TB 3 kênh

    vmax = IG_mean_abs_per_pixel.max()
    fig, axes = plt.subplots(IMG_SIZE, col_w, figsize=(col_w * 1.6, IMG_SIZE * 1.6))
    if col_w == 1:
        axes = axes.reshape(IMG_SIZE, 1)
    for r in range(IMG_SIZE):
        for c in range(col_w):
            ax = axes[r, c]
            ax.imshow(IG_mean_abs_per_pixel[r, c], cmap="hot", vmin=0, vmax=vmax)
            for x in [5, 10]:
                ax.axvline(x=x - 0.5, color="cyan", linewidth=0.5, alpha=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"r{r}", fontsize=6)
            if r == 0:
                ax.set_title(f"c{t0+c}", fontsize=6)
    fig.suptitle(f"{os.path.basename(args.case_dir)} — IG saliency map RIÊNG cho từng pixel "
                f"output cột '{target_name}'\n(mỗi ô nhỏ = |IG| trung bình 3 kênh, "
                f"kích thước 16x16, giải thích 1 pixel output)", fontsize=10)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, f"ig_per_pixel_grid_col{args.target_col}.png")
    plt.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {grid_path}")

    # ── Tóm tắt: trung bình |IG| qua toàn bộ pixel target, theo 3 vùng noise ──
    IG_overall = IG_mean_abs_per_pixel.mean(axis=(0, 1))    # (16,16) trung bình qua 80 pixel target
    zone_scores = {}
    for name_i, (s0, s1) in zip(ZONE_NAMES, ZONE_SLICES):
        zone_scores[name_i] = float(IG_overall[:, s0:s1].sum())
    total = sum(zone_scores.values())
    print(f"\nTrung bình |IG| qua toàn bộ {n_pixels} pixel target, theo vùng noise:")
    for name_i, score in zone_scores.items():
        print(f"  noise {name_i:10s}: {score:10.5f}  ({100*score/total:5.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axes[0].imshow(IG_overall, cmap="hot")
    for x in [5, 10]:
        axes[0].axvline(x=x - 0.5, color="cyan", linewidth=1)
    axes[0].set_title(f"|IG| trung bình qua {n_pixels} pixel target\n(cột '{target_name}')", fontsize=9)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(list(zone_scores.keys()), list(zone_scores.values()),
               color=["tomato" if n == target_name else "steelblue" for n in zone_scores])
    axes[1].set_title("Tổng |IG| theo vùng noise", fontsize=9)
    plt.tight_layout()
    summary_path = os.path.join(out_dir, f"ig_summary_col{args.target_col}.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
