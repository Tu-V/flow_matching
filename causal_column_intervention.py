"""
Causal intervention analysis (Cách 2 — không cần gradient) cho 1 case cụ thể:
đo tác động nhân quả của từng cột noise x_T (triangle/square/pentagon zone)
lên từng cột ảnh cuối x_0.

Thuật toán:
  1. x_T (đã biết, từ noise_init.pt), chạy full ODE (steps cố định, vd 25)
     -> x_0 (reference).
  2. Với mỗi cột nhiễu i (triangle/square/pentagon):
       - Resample CHỈ cột i của x_T bằng noise mới độc lập ~ N(0,I)
         (giữ nguyên 2 cột còn lại) -> x_T~ (lặp K lần, Monte Carlo).
       - Chạy lại ODE (cùng steps) -> x_0~.
       - Delta_j = || (x_0~ - x_0)_{cột j} ||_2   cho j = 1,2,3.
       - Lấy trung bình Delta_j qua K lần resample -> S[i, j] = E[Delta_j | resample cột i].
  3. Kết quả: ma trận S 3x3 (hàng i = cột noise bị can thiệp, cột j = cột ảnh bị ảnh hưởng).

Đây là phép đo causal thuần (không dùng gradient/IG), độc lập với vấn đề
gradient vanish/saturate qua nhiều bước ODE composition.

Usage:
    python causal_column_intervention.py \\
        --case_dir shapes_fm_output/hallucination_analysis/traces/HALL_case_0109_idx92812 \\
        --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt \\
        --steps 25 --k_trials 30 --seed 0
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

from sample_shapes_fm import load_model, UNetVelocityWrapper, decode, to_uint8_numpy, _latest_ckpt   # noqa: E402
from hallucination_detector import analyze_image   # noqa: E402

IMG_SIZE = 16
ZONE_NAMES = ["triangle", "square", "pentagon"]
ZONE_SLICES = [(0, 5), (5, 10), (10, 15)]   # trên trục W (cột ảnh), bỏ padding col 15
ZOOM = 8


def upscale(img, z=ZOOM):
    return img.repeat(z, axis=0).repeat(z, axis=1)


@torch.no_grad()
def euler_rollout_batch(wrapper, x_init_batch, steps, device):
    """x_init_batch: (B,3,16,16) -> x_final: (B,3,16,16) (raw model scale, chưa decode)."""
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
    parser.add_argument("--k_trials", type=int, default=30,
                        help="Số lần Monte Carlo resample mỗi cột")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.case_dir, "causal_intervention")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)

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

    x_T = torch.load(os.path.join(args.case_dir, "noise_init.pt"), weights_only=True).to(device)
    print(f"x_T shape: {tuple(x_T.shape)}")

    # ── Reference: x_0 gốc ──────────────────────────────────────────────────
    x_0 = euler_rollout_batch(wrapper, x_T, args.steps, device)   # (1,3,16,16)
    img_0 = to_uint8_numpy(decode(x_0)[0])
    r0 = analyze_image(img_0)
    print(f"Reference x_0: hall={r0['is_hallucination']} type={r0['hall_type']} "
          f"blobs={r0['col_blobs']}")
    PILImage.fromarray(upscale(img_0)).save(os.path.join(out_dir, "reference_x0.png"))

    # ── Intervention: resample từng cột noise, đo Delta ──────────────────────
    K = args.k_trials
    S = np.zeros((3, 3))       # S[i,j] = E[|| (x0~-x0)_colj ||] khi resample col i cua noise
    S_std = np.zeros((3, 3))   # std để báo độ tin cậy Monte Carlo

    for i, (name_i, (s0, s1)) in enumerate(zip(ZONE_NAMES, ZONE_SLICES)):
        print(f"\nResample cột noise '{name_i}' (x[{s0}:{s1}]), K={K} trials ...")

        x_T_batch = x_T.repeat(K, 1, 1, 1).clone()          # (K,3,16,16)
        fresh_noise = torch.randn(K, 3, IMG_SIZE, (s1 - s0), device=device)
        x_T_batch[:, :, :, s0:s1] = fresh_noise             # chỉ thay cột i

        x_0_tilde = euler_rollout_batch(wrapper, x_T_batch, args.steps, device)  # (K,3,16,16)
        diff = x_0_tilde - x_0                                # (K,3,16,16), broadcast x_0 (1,3,16,16)

        for j, (name_j, (t0, t1)) in enumerate(zip(ZONE_NAMES, ZONE_SLICES)):
            d = diff[:, :, :, t0:t1]                          # (K,3,16,w_j)
            norms = d.flatten(1).norm(dim=1)                  # (K,)
            S[i, j] = norms.mean().item()
            S_std[i, j] = norms.std().item()

    print("\n=== Ma trận Causal S[i,j] = E[ ||Δ output cột j|| | resample cột noise i ] ===")
    header = "            " + "".join(f"{n:>12s}" for n in ZONE_NAMES)
    print(header)
    for i, name_i in enumerate(ZONE_NAMES):
        row = "".join(f"{S[i,j]:12.3f}" for j in range(3))
        print(f"noise {name_i:8s} {row}")

    print("\n(std Monte Carlo qua K trials, để tham khảo độ nhiễu ước lượng)")
    for i, name_i in enumerate(ZONE_NAMES):
        row = "".join(f"{S_std[i,j]:12.3f}" for j in range(3))
        print(f"noise {name_i:8s} {row}")

    # ── Chuẩn hoá theo hàng để so sánh trực quan (mỗi hàng: intervention nào ảnh hưởng đâu nhiều nhất) ──
    S_row_norm = S / S.sum(axis=1, keepdims=True)

    # ── Lưu số liệu ────────────────────────────────────────────────────────
    np.savez(os.path.join(out_dir, "causal_matrix.npz"), S=S, S_std=S_std, S_row_norm=S_row_norm)
    with open(os.path.join(out_dir, "causal_matrix.txt"), "w") as f:
        f.write(f"Case: {os.path.basename(args.case_dir)}\n")
        f.write(f"steps={args.steps}  K_trials={K}\n\n")
        f.write("S[i,j] = E[ ||delta output col j|| | resample noise col i ]  (raw)\n")
        f.write(header + "\n")
        for i, name_i in enumerate(ZONE_NAMES):
            f.write(f"noise {name_i:8s} " + "".join(f"{S[i,j]:12.3f}" for j in range(3)) + "\n")
        f.write("\nRow-normalized (moi hang chia tong hang):\n")
        f.write(header + "\n")
        for i, name_i in enumerate(ZONE_NAMES):
            f.write(f"noise {name_i:8s} " + "".join(f"{S_row_norm[i,j]:12.3f}" for j in range(3)) + "\n")

    # ── Vẽ heatmap ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    im = ax.imshow(S, cmap="viridis")
    ax.set_xticks(range(3)); ax.set_xticklabels(ZONE_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(ZONE_NAMES)
    ax.set_xlabel("cột ẢNH bị ảnh hưởng (Δ)")
    ax.set_ylabel("cột NOISE bị resample")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center",
                    color="white" if S[i,j] < S.max()*0.6 else "black", fontsize=10)
    ax.set_title("S[i,j] = E‖Δ output col j‖\n(raw, causal intervention)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(S_row_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(ZONE_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(ZONE_NAMES)
    ax.set_xlabel("cột ẢNH bị ảnh hưởng")
    ax.set_ylabel("cột NOISE bị resample")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{S_row_norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if S_row_norm[i,j] < 0.6 else "black", fontsize=10)
    ax.set_title("Row-normalized (tỉ lệ tác động\ntrong mỗi hàng)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{os.path.basename(args.case_dir)} — Causal column intervention "
                f"(steps={args.steps}, K={K})", fontsize=11)
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "causal_matrix_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {heatmap_path}")
    print(f"Saved: {os.path.join(out_dir, 'causal_matrix.txt')}")


if __name__ == "__main__":
    main()
