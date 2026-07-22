"""
Xuất 1 file tổng hợp cho case cụ thể:
  1) noise đầu vào (3 kênh), intermediate images dọc ODE trajectory, ảnh cuối.
  2) Với mỗi pixel trong cột target (mặc định cột 1 = triangle): 1 cặp ảnh
     [ảnh cuối có đánh dấu vị trí pixel đó | saliency map IG của pixel đó].

Integrated Gradients: baseline = noise toàn 0 (ảnh đen), m_steps=50 (Riemann),
ODE rollout steps=25.

Không phân tích gì thêm — chỉ xuất hình để xem trực tiếp.

Usage:
    python ig_full_report.py \\
        --case_dir shapes_fm_output/hallucination_analysis/traces/HALL_case_0109_idx92812 \\
        --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt \\
        --steps 25 --target_col 1 --m_steps 50
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from sample_shapes_fm import load_model, UNetVelocityWrapper, decode, to_uint8_numpy, _latest_ckpt   # noqa: E402

IMG_SIZE = 16
ZONE_NAMES = ["triangle", "square", "pentagon"]
ZONE_SLICES = [(0, 5), (5, 10), (10, 15)]
ZOOM = 8


def upscale(img, z=ZOOM):
    return img.repeat(z, axis=0).repeat(z, axis=1)


def mark_pixel(img_uint8, r, c, zoom=ZOOM, color=(255, 0, 255)):
    """img_uint8: (16,16,3). Trả về ảnh upscale với khung màu quanh pixel (r,c)."""
    up = upscale(img_uint8, zoom).copy()
    y0, y1 = r * zoom, (r + 1) * zoom
    x0, x1 = c * zoom, (c + 1) * zoom
    up[y0:y1, x0:x0+2] = color
    up[y0:y1, x1-2:x1] = color
    up[y0:y0+2, x0:x1] = color
    up[y1-2:y1, x0:x1] = color
    return up


@torch.no_grad()
def euler_rollout_with_frames(wrapper, x_init, steps, device):
    dt = 1.0 / steps
    x = x_init
    frames = [to_uint8_numpy(decode(x)[0])]
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)
        v = wrapper(x, t_tensor)
        x = x + dt * v
        frames.append(to_uint8_numpy(decode(x)[0]))
    return frames


def euler_rollout_batch_grad(wrapper, x_init_batch, steps, device):
    dt = 1.0 / steps
    x = x_init_batch
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)
        v = wrapper(x, t_tensor)
        x = x + dt * v
    return x


@torch.no_grad()
def reverse_euler_rollout(wrapper, x1_init, steps, device):
    """Đảo ngược ODE: đi từ x_1 (ảnh, t=1) ngược về x_0 (noise, t=0).
    Flow matching ODE là dx/dt = v_theta(x,t), tích phân ngược chiều thời gian."""
    dt = 1.0 / steps
    x = x1_init
    for i in range(steps):
        t_val = 1.0 - i * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)
        v = wrapper(x, t_tensor)
        x = x - dt * v
    return x


def compute_mean_image_baseline(data_dir, wrapper, steps, device):
    """Ảnh trung bình của toàn bộ dataset thật -> invert ngược ODE -> noise baseline."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    assert paths, f"Không tìm thấy .png trong {data_dir}"
    arrs = [np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in paths]
    mean_gray = np.mean(arrs, axis=0)                      # (16,16) trong [0,1]
    mean_rgb = np.repeat(mean_gray[None, :, :], 3, axis=0)  # (3,16,16)
    x1 = torch.from_numpy((mean_rgb * 2 - 1).astype(np.float32)).unsqueeze(0).to(device)  # -> [-1,1]
    x0_baseline = reverse_euler_rollout(wrapper, x1, steps, device)
    return x0_baseline, mean_gray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--target_col", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--m_steps", type=int, default=50)
    parser.add_argument("--n_intermediate", type=int, default=8)
    parser.add_argument("--baseline", choices=["zero", "mean_image", "random_noise"],
                        default="zero")
    parser.add_argument("--data_dir",
                        default="../neurips-2024-diffusion-model-hallucination/simple-datasets/simple-shapes-5k-16x16",
                        help="Dùng khi --baseline mean_image")
    parser.add_argument("--baseline_seed", type=int, default=123,
                        help="Dùng khi --baseline random_noise")
    parser.add_argument("--out_path", default=None)
    args = parser.parse_args()

    out_path = args.out_path or os.path.join(
        args.case_dir, "integrated_gradients",
        f"ig_full_report_col{args.target_col}_baseline_{args.baseline}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt_path = args.ckpt or _latest_ckpt()
    model = load_model(ckpt_path, device)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapper = UNetVelocityWrapper(model)

    x_T = torch.load(os.path.join(args.case_dir, "noise_init.pt"), weights_only=True).to(device)

    t0, t1 = ZONE_SLICES[args.target_col - 1]
    col_w = t1 - t0

    # ── 1) noise, intermediate frames, ảnh cuối ──────────────────────────────
    frames = euler_rollout_with_frames(wrapper, x_T, args.steps, device)
    final_img = frames[-1]
    idxs = np.linspace(0, len(frames) - 1, args.n_intermediate).astype(int)
    inter_frames = [frames[i] for i in idxs]
    noise_np = x_T[0].detach().cpu().numpy()   # (3,16,16)

    # ── Chọn baseline x' cho IG ──────────────────────────────────────────────
    if args.baseline == "zero":
        x_baseline = torch.zeros_like(x_T)
    elif args.baseline == "mean_image":
        x_baseline, _ = compute_mean_image_baseline(args.data_dir, wrapper, args.steps, device)
    else:  # random_noise
        g = torch.Generator().manual_seed(args.baseline_seed)   # CPU generator, tránh vấn đề MPS
        x_baseline = torch.randn(x_T.shape, generator=g).to(device)
    print(f"Baseline: {args.baseline}  "
          f"norm={x_baseline.norm().item():.3f}  mean={x_baseline.mean().item():.3f}")

    # ── 2) Integrated Gradients riêng từng pixel cột target ───────────────────
    m = args.m_steps
    alphas = torch.linspace(1.0 / m, 1.0, m, device=device).view(m, 1, 1, 1)
    x_T_rep = x_T.repeat(m, 1, 1, 1)
    x_base_rep = x_baseline.repeat(m, 1, 1, 1)
    x_alpha = (x_base_rep + alphas * (x_T_rep - x_base_rep)).clone().requires_grad_(True)
    x_0_alpha = euler_rollout_batch_grad(wrapper, x_alpha, args.steps, device)
    x_0_alpha_gray = x_0_alpha.mean(dim=1)

    diff_x = (x_T[0] - x_baseline[0])   # (3,16,16)
    IG_per_pixel = np.zeros((IMG_SIZE, col_w, 16, 16), dtype=np.float32)   # |IG| TB 3 kênh
    n_pixels = IMG_SIZE * col_w
    k = 0
    for r in range(IMG_SIZE):
        for c in range(col_w):
            k += 1
            is_last = (k == n_pixels)
            F_pixel = x_0_alpha_gray[:, r, t0 + c]
            grad_pixel = torch.autograd.grad(F_pixel.sum(), x_alpha, retain_graph=not is_last)[0]
            avg_grad = grad_pixel.mean(dim=0)
            IG_pixel = (diff_x * avg_grad).detach().cpu().numpy()
            IG_per_pixel[r, c] = np.abs(IG_pixel).mean(axis=0)

    # ── Vẽ tổng hợp ────────────────────────────────────────────────────────
    n_top = 3 + args.n_intermediate + 1   # 3 kênh noise + intermediate + final
    fig = plt.figure(figsize=(max(n_top, col_w * 2) * 1.4, 2.3 + IMG_SIZE * 1.5))
    fig.suptitle(f"baseline = {args.baseline}", fontsize=10, y=0.995)
    gs_top = gridspec.GridSpec(1, n_top, figure=fig, top=0.96, bottom=0.88, left=0.02, right=0.98, wspace=0.15)

    for i in range(3):
        ax = fig.add_subplot(gs_top[0, i])
        ax.imshow(noise_np[i], cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_title(f"noise ch{i}", fontsize=6)
        ax.axis("off")
    for j, fi in enumerate(idxs):
        ax = fig.add_subplot(gs_top[0, 3 + j])
        ax.imshow(upscale(inter_frames[j]))
        ax.set_title(f"t-step {fi}", fontsize=6)
        ax.axis("off")
    ax = fig.add_subplot(gs_top[0, n_top - 1])
    ax.imshow(upscale(final_img))
    for x in [5 * ZOOM, 10 * ZOOM]:
        ax.axvline(x=x, color="red", linewidth=1)
    ax.set_title("final x_0", fontsize=6)
    ax.axis("off")

    # ── Grid dưới: mỗi hàng r, mỗi cột c -> [marked final | saliency map] ────
    gs_bot = gridspec.GridSpec(IMG_SIZE, col_w * 2, figure=fig,
                               top=0.85, bottom=0.02, left=0.02, right=0.93,
                               wspace=0.05, hspace=0.05)
    vmax = IG_per_pixel.max()
    im2 = None
    for r in range(IMG_SIZE):
        for c in range(col_w):
            marked = mark_pixel(final_img, r, t0 + c)
            ax1 = fig.add_subplot(gs_bot[r, 2 * c])
            ax1.imshow(marked)
            ax1.set_xticks([]); ax1.set_yticks([])
            if r == 0:
                ax1.set_title(f"pix(r{r},c{t0+c})\nfinal", fontsize=5)

            ax2 = fig.add_subplot(gs_bot[r, 2 * c + 1])
            im2 = ax2.imshow(IG_per_pixel[r, c], cmap="hot", vmin=0, vmax=vmax)
            ax2.set_xticks([]); ax2.set_yticks([])
            if r == 0:
                ax2.set_title("saliency", fontsize=5)

    cbar_ax = fig.add_axes([0.945, 0.02, 0.015, 0.85])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("|IG| attribution (0 = không ảnh hưởng)", fontsize=8)

    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
