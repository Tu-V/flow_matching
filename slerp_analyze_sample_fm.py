"""
Slerp analysis cho 1 case hallucination của Flow Matching (giống hệt phương pháp
đã áp dụng cho DDIM ở improved-diffusion/scripts/slerp_analyze_sample.py).

Dựng 2 điểm noise x1, x2 (cùng norm với x_noise = noise_init.pt của case đã biết)
sao cho slerp(x1, x2, alpha) = x_noise CHÍNH XÁC, rồi:
  1. Decode riêng x1, x2 qua ODE (Euler, 25 step) -> xem ảnh sinh ra là gì.
  2. Quét slerp(x1, x2, a) với a trong [0,1] -> vẽ toàn bộ ảnh dọc đường,
     xác nhận tại a=alpha ra đúng ảnh hallucination gốc.

Usage:
    python slerp_analyze_sample_fm.py \\
        --case_dir shapes_fm_output/hallucination_analysis/traces/case_0001_idx76 \\
        --steps 25 --alpha 0.5 --total_angle_deg 40 --n_alphas 101 --seed 0
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from sample_shapes_fm import (   # noqa: E402
    build_unet, load_model, UNetVelocityWrapper, sample_batch,
    decode, to_uint8_numpy, _latest_ckpt, CKPT_DIR,
)
from hallucination_detector import analyze_image   # noqa: E402

ZOOM = 8


def upscale(img, zoom=ZOOM):
    return img.repeat(zoom, axis=0).repeat(zoom, axis=1)


def random_orthogonal(x_noise_np, rng):
    flat = x_noise_np.reshape(-1)
    u = rng.standard_normal(flat.shape).astype(np.float32)
    u -= (u @ flat) / (flat @ flat) * flat
    u /= np.linalg.norm(u)
    return u.reshape(x_noise_np.shape)


def build_x1_x2(x_noise_np, alpha, total_angle_rad, u):
    r = np.linalg.norm(x_noise_np)
    n = x_noise_np / r
    phi1 = alpha * total_angle_rad
    phi2 = (1 - alpha) * total_angle_rad
    x1 = r * (np.cos(phi1) * n - np.sin(phi1) * u)
    x2 = r * (np.cos(phi2) * n + np.sin(phi2) * u)
    return x1.astype(np.float32), x2.astype(np.float32)


def slerp(x1, x2, a):
    f1, f2 = x1.reshape(-1), x2.reshape(-1)
    dot = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-6:
        return ((1 - a) * x1 + a * x2).astype(np.float32)
    w1 = np.sin((1 - a) * theta) / np.sin(theta)
    w2 = np.sin(a * theta) / np.sin(theta)
    return (w1 * x1 + w2 * x2).astype(np.float32)


@torch.no_grad()
def ode_decode(wrapper, x_np, steps, device):
    """x_np: (3,H,W) numpy float32 -> (H,W,3) uint8 numpy, qua ODE Euler determinstic."""
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)   # [1,3,H,W]
    x_final = sample_batch(wrapper, x, steps=steps, return_intermediates=False)
    x_01 = decode(x_final)   # [1,3,H,W] in [0,1]
    return to_uint8_numpy(x_01[0])


def save_noise_heatmap(noise_chw, path, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(noise_chw[0], cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True,
                        help="Thư mục case_XXXX_idxYYY chứa noise_init.pt")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--total_angle_deg", type=float, default=40.0)
    parser.add_argument("--n_alphas", type=int, default=101)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.case_dir, "slerp_x1x2")
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
    wrapper = UNetVelocityWrapper(model)

    # ── Load x_noise + build x1, x2 ─────────────────────────────────────────
    x_noise_t = torch.load(os.path.join(args.case_dir, "noise_init.pt"), weights_only=True)
    x_noise = x_noise_t[0].numpy()   # (3,H,W)

    rng = np.random.default_rng(args.seed)
    u = random_orthogonal(x_noise, rng)
    total_angle_rad = np.deg2rad(args.total_angle_deg)
    x1, x2 = build_x1_x2(x_noise, args.alpha, total_angle_rad, u)

    recon = slerp(x1, x2, args.alpha)
    err = np.linalg.norm(recon - x_noise)
    print(f"‖x_noise‖={np.linalg.norm(x_noise):.3f}  "
          f"‖x1‖={np.linalg.norm(x1):.3f}  ‖x2‖={np.linalg.norm(x2):.3f}  "
          f"‖slerp(x1,x2,alpha)-x_noise‖={err:.6f}  (phải ~0)")

    np.save(os.path.join(out_dir, "x_noise.npy"), x_noise)
    np.save(os.path.join(out_dir, "x1.npy"), x1)
    np.save(os.path.join(out_dir, "x2.npy"), x2)

    # ── Decode x_noise, x1, x2 riêng lẻ ──────────────────────────────────────
    print("Decoding x_noise, x1, x2 ...")
    img_noise = ode_decode(wrapper, x_noise, args.steps, device)
    img_x1    = ode_decode(wrapper, x1, args.steps, device)
    img_x2    = ode_decode(wrapper, x2, args.steps, device)

    r_noise = analyze_image(img_noise)
    r_x1    = analyze_image(img_x1)
    r_x2    = analyze_image(img_x2)
    print(f"  x_noise (a={args.alpha:.2f}): hall={r_noise['is_hallucination']} "
          f"type={r_noise['hall_type']} blobs={r_noise['col_blobs']}")
    print(f"  x1 (a=0)              : hall={r_x1['is_hallucination']} "
          f"type={r_x1['hall_type']} blobs={r_x1['col_blobs']}")
    print(f"  x2 (a=1)              : hall={r_x2['is_hallucination']} "
          f"type={r_x2['hall_type']} blobs={r_x2['col_blobs']}")

    for name, img in [("x_noise", img_noise), ("x1", img_x1), ("x2", img_x2)]:
        PILImage.fromarray(upscale(img)).save(os.path.join(out_dir, f"img_{name}.png"))
    for name, arr in [("x_noise", x_noise), ("x1", x1), ("x2", x2)]:
        save_noise_heatmap(arr, os.path.join(out_dir, f"heatmap_{name}.png"), name)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    for ax, name, img, r in [
        (axes[0], "x1 (a=0)", img_x1, r_x1),
        (axes[1], f"x_noise (a={args.alpha:.2f})", img_noise, r_noise),
        (axes[2], "x2 (a=1)", img_x2, r_x2),
    ]:
        ax.imshow(upscale(img))
        h_str = r["hall_type"] if r["is_hallucination"] else "ok"
        cb = r["col_blobs"]
        ax.set_title(f"{name}\n{h_str}  t{cb['triangle']}s{cb['square']}p{cb['pentagon']}",
                     fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("red" if r["is_hallucination"] else "green")
            spine.set_linewidth(3)
        ax.axis("off")
    fig.suptitle(f"[flow matching] slerp endpoints — total_angle={args.total_angle_deg}°", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "endpoints_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Full slerp sweep x1 -> x2 ────────────────────────────────────────────
    alphas = np.linspace(0.0, 1.0, args.n_alphas)
    records = []
    n_hall = 0
    print(f"\nSlerp sweep: {args.n_alphas} điểm, a in [0,1]  (x_noise tại a={args.alpha:.3f})")
    for i, a in enumerate(alphas):
        x_a = slerp(x1, x2, a)
        img_a = ode_decode(wrapper, x_a, args.steps, device)
        r_a = analyze_image(img_a)
        if r_a["is_hallucination"]:
            n_hall += 1
        records.append((float(a), r_a["is_hallucination"], r_a["hall_type"],
                        r_a["col_blobs"], img_a))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1:>4}/{args.n_alphas}]  a={a:.3f}  hall so far: {n_hall}")

    total = args.n_alphas
    print(f"\nResult: {n_hall}/{total} = {100*n_hall/total:.1f}% hallucination dọc slerp(x1,x2)")

    a_vals = [r[0] for r in records]
    hall_flags = [int(r[1]) for r in records]
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.fill_between(a_vals, hall_flags, step="mid", alpha=0.65, color="crimson",
                     label="hallucination")
    ax.axvline(x=args.alpha, color="blue", linestyle="--", linewidth=1.5,
               label=f"x_noise (a={args.alpha:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("a  (0=x1, 1=x2)")
    ax.set_ylabel("hall")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ok", "hall"])
    ax.set_title(
        f"[flow matching] slerp(x1,x2,a) — {n_hall}/{total} = {100*n_hall/total:.1f}% hallucinate  "
        f"|  total_angle={args.total_angle_deg}°"
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hallucination_profile.png"), dpi=150, bbox_inches="tight")
    plt.close()

    vis_targets = sorted(set(np.linspace(0.0, 1.0, 13).tolist() + [args.alpha]))
    key_idx = sorted(set(int(np.argmin(np.abs(np.array(a_vals) - v))) for v in vis_targets))

    n_vis = len(key_idx)
    fig, axes = plt.subplots(1, n_vis, figsize=(n_vis * 2.0, 3.2))
    for ax, idx in zip(axes, key_idx):
        a, is_h, htype, blobs, img_a = records[idx]
        ax.imshow(upscale(img_a))
        h_str = htype if is_h else "ok"
        cb = blobs
        marker = "  <- x_noise" if abs(a - args.alpha) < 1e-6 else ""
        ax.set_title(f"a={a:.2f}{marker}\n{h_str}\nt{cb['triangle']}s{cb['square']}p{cb['pentagon']}",
                     fontsize=6.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("red" if is_h else "green")
            spine.set_linewidth(3)
        ax.axis("off")
    fig.suptitle(f"[flow matching] slerp(x1,x2) grid  ({n_hall}/{total} hall)", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "interpolation_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()

    row_size = min(20, total)
    n_rows = (total + row_size - 1) // row_size
    fig, axes = plt.subplots(n_rows, row_size, figsize=(row_size * 0.6, n_rows * 0.75))
    axes = np.array(axes).reshape(n_rows, row_size)
    alpha_idx_marker = int(np.argmin(np.abs(np.array(a_vals) - args.alpha)))
    for idx in range(total):
        ri, ci = idx // row_size, idx % row_size
        ax = axes[ri, ci]
        _, is_h, _, _, img_a = records[idx]
        ax.imshow(img_a)
        color = "blue" if idx == alpha_idx_marker else ("red" if is_h else "lime")
        lw = 3 if idx == alpha_idx_marker else 2
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(lw)
        ax.axis("off")
    for idx in range(total, n_rows * row_size):
        ri, ci = idx // row_size, idx % row_size
        axes[ri, ci].axis("off")
    fig.suptitle(
        f"[flow matching] slerp(x1,x2) — all {total} samples\n"
        f"red=hall  green=ok  blue=x_noise  {n_hall}/{total} ({100*n_hall/total:.0f}%) hall",
        fontsize=9,
    )
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(out_dir, "all_samples_grid.png"), dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nFiles saved -> {out_dir}/")


if __name__ == "__main__":
    main()
