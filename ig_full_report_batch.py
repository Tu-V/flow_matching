"""
Giống hệt ig_full_report.py (report đầy đủ: noise/intermediate/final + lưới
saliency map từng pixel cột target) nhưng chạy HÀNG LOẠT cho nhiều case cùng
lúc, gộp batch để chỉ cần 80 lần backward TỔNG CỘNG (không nhân theo số case) —
nhanh hơn nhiều so với chạy từng case riêng lẻ.

Với mỗi baseline trong {zero, mean_image, random_noise}: gộp N case x m alpha
thành 1 batch lớn (N*m), 1 lần forward ODE, rồi với mỗi trong 80 pixel target,
1 lần backward duy nhất cho CẢ batch (đúng theo tính độc lập giữa các sample
trong batch của mạng conv — không có nhiễu chéo giữa các case).

Usage:
    python ig_full_report_batch.py \\
        --traces_dir shapes_fm_output/hallucination_analysis/traces \\
        --case_prefix HALL_ \\
        --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt \\
        --steps 25 --target_col 1 --m_steps 50 \\
        --data_dir ../neurips-2024-diffusion-model-hallucination/simple-datasets/simple-shapes-5k-16x16 \\
        --skip_existing
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


def euler_rollout_batch_grad(wrapper, x_init_batch, steps, device, chunk=None):
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
    dt = 1.0 / steps
    x = x1_init
    for i in range(steps):
        t_val = 1.0 - i * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)
        v = wrapper(x, t_tensor)
        x = x - dt * v
    return x


def compute_mean_image_baseline(data_dir, wrapper, steps, device):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    assert paths, f"Không tìm thấy .png trong {data_dir}"
    arrs = [np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in paths]
    mean_gray = np.mean(arrs, axis=0)
    mean_rgb = np.repeat(mean_gray[None, :, :], 3, axis=0)
    x1 = torch.from_numpy((mean_rgb * 2 - 1).astype(np.float32)).unsqueeze(0).to(device)
    return reverse_euler_rollout(wrapper, x1, steps, device)


def get_baseline(kind, x_T_single, wrapper, steps, device, data_dir, seed):
    if kind == "zero":
        return torch.zeros_like(x_T_single)
    elif kind == "mean_image":
        return compute_mean_image_baseline(data_dir, wrapper, steps, device)
    else:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(x_T_single.shape, generator=g).to(device)


def save_case_report(case_name, out_dir, wrapper, x_T, final_img, inter_frames, idxs,
                     IG_per_pixel, t0, col_w, baseline_name, args, device):
    os.makedirs(out_dir, exist_ok=True)
    noise_np = x_T[0].detach().cpu().numpy()

    n_top = 3 + len(idxs) + 1
    fig = plt.figure(figsize=(max(n_top, col_w * 2) * 1.4, 2.3 + IMG_SIZE * 1.5))
    fig.suptitle(f"{case_name}  —  baseline = {baseline_name}", fontsize=10, y=0.995)
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

    out_path = os.path.join(out_dir, f"ig_full_report_col{args.target_col}_baseline_{baseline_name}.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_dir", required=True)
    parser.add_argument("--case_prefix", default="HALL_")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--target_col", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--m_steps", type=int, default=50)
    parser.add_argument("--n_intermediate", type=int, default=8)
    parser.add_argument("--baselines", default="zero,mean_image,random_noise")
    parser.add_argument("--data_dir",
                        default="../neurips-2024-diffusion-model-hallucination/simple-datasets/simple-shapes-5k-16x16")
    parser.add_argument("--baseline_seed", type=int, default=123)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--case_chunk_size", type=int, default=5,
                        help="Số case gộp batch mỗi lần (giảm nếu OOM)")
    args = parser.parse_args()

    baseline_list = args.baselines.split(",")

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

    case_names = sorted(d for d in os.listdir(args.traces_dir)
                        if d.startswith(args.case_prefix)
                        and os.path.isdir(os.path.join(args.traces_dir, d))
                        and os.path.exists(os.path.join(args.traces_dir, d, "noise_init.pt")))

    if args.skip_existing:
        remaining = []
        for c in case_names:
            done = all(
                os.path.exists(os.path.join(
                    args.traces_dir, c, "integrated_gradients",
                    f"ig_full_report_col{args.target_col}_baseline_{b}.png"))
                for b in baseline_list
            )
            if not done:
                remaining.append(c)
        print(f"{len(case_names)} case tổng, {len(remaining)} case còn thiếu báo cáo -> chỉ chạy các case này")
        case_names = remaining

    if not case_names:
        print("Không có case nào cần chạy.")
        return

    N = len(case_names)
    print(f"Chạy cho {N} case: {case_names}")

    x_T_all = torch.cat(
        [torch.load(os.path.join(args.traces_dir, c, "noise_init.pt"), weights_only=True)
         for c in case_names], dim=0
    ).to(device)   # (N,3,16,16)

    t0, t1 = ZONE_SLICES[args.target_col - 1]
    col_w = t1 - t0

    # ── Reference: intermediate frames + final image cho từng case (không cần batch lớn, làm riêng để lưu frames) ──
    print("Tính reference trajectory (intermediate frames) cho từng case ...")
    case_final_imgs = {}
    case_inter_frames = {}
    idxs_ref = None
    for j, c in enumerate(case_names):
        frames = euler_rollout_with_frames(wrapper, x_T_all[j:j+1], args.steps, device)
        if idxs_ref is None:
            idxs_ref = np.linspace(0, len(frames) - 1, args.n_intermediate).astype(int)
        case_final_imgs[c] = frames[-1]
        case_inter_frames[c] = [frames[i] for i in idxs_ref]

    chunk_size = args.case_chunk_size
    m = args.m_steps
    n_pixels = IMG_SIZE * col_w

    for baseline_name in baseline_list:
        print(f"\n{'='*60}\nBaseline: {baseline_name}\n{'='*60}")

        x_baseline_single = get_baseline(baseline_name, x_T_all[0:1], wrapper, args.steps,
                                         device, args.data_dir, args.baseline_seed)

        for chunk_start in range(0, N, chunk_size):
            chunk_names = case_names[chunk_start:chunk_start + chunk_size]
            x_T_chunk = x_T_all[chunk_start:chunk_start + chunk_size]
            n_chunk = x_T_chunk.shape[0]
            x_baseline_chunk = x_baseline_single.repeat(n_chunk, 1, 1, 1)

            print(f"\n-- chunk {chunk_start//chunk_size + 1}: {chunk_names} --")

            alphas = torch.linspace(1.0 / m, 1.0, m, device=device).view(1, m, 1, 1, 1)
            x_T_exp = x_T_chunk.unsqueeze(1)
            x_base_exp = x_baseline_chunk.unsqueeze(1)
            x_alpha_5d = x_base_exp + alphas * (x_T_exp - x_base_exp)
            x_alpha = x_alpha_5d.reshape(n_chunk * m, 3, IMG_SIZE, IMG_SIZE).clone().requires_grad_(True)

            print(f"Forward ODE batch=({n_chunk}x{m}={n_chunk*m}), steps={args.steps} ...")
            x_0_alpha = euler_rollout_batch_grad(wrapper, x_alpha, args.steps, device)
            x_0_alpha_gray = x_0_alpha.mean(dim=1)

            diff_x = (x_T_chunk - x_baseline_chunk)

            IG_per_pixel_chunk = np.zeros((n_chunk, IMG_SIZE, col_w, 16, 16), dtype=np.float32)
            k = 0
            print(f"Backward cho {n_pixels} pixel target (dùng chung cho {n_chunk} case trong chunk) ...")
            for r in range(IMG_SIZE):
                for c in range(col_w):
                    k += 1
                    is_last = (k == n_pixels)
                    F_pixel = x_0_alpha_gray[:, r, t0 + c]
                    grad_pixel = torch.autograd.grad(F_pixel.sum(), x_alpha, retain_graph=not is_last)[0]
                    grad_pixel = grad_pixel.view(n_chunk, m, 3, IMG_SIZE, IMG_SIZE)
                    avg_grad = grad_pixel.mean(dim=1)
                    IG = diff_x * avg_grad
                    IG_per_pixel_chunk[:, r, c] = IG.abs().mean(dim=1).detach().cpu().numpy()

            del x_alpha, x_0_alpha, x_0_alpha_gray
            if device.type == "mps":
                torch.mps.empty_cache()

            for j, c in enumerate(chunk_names):
                out_dir = os.path.join(args.traces_dir, c, "integrated_gradients")
                out_path = save_case_report(
                    c, out_dir, wrapper, x_T_all[chunk_start + j:chunk_start + j + 1],
                    case_final_imgs[c], case_inter_frames[c],
                    idxs_ref, IG_per_pixel_chunk[j], t0, col_w, baseline_name, args, device,
                )
                print(f"  saved: {out_path}")

    print("\nHoàn tất tất cả case và baseline.")


if __name__ == "__main__":
    main()
