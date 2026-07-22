"""
Chạy causal column intervention (Cách 2) cho TOÀN BỘ case trong 1 thư mục traces/,
gộp lại theo loại lỗi cụ thể của từng case:
  - "empty"              : không có shape nào (0 shapes toàn ảnh)
  - "double_col_triangle": 2+ shape ở cột 1 (triangle)
  - "double_col_square"  : 2+ shape ở cột 2 (square)
  - "double_col_pentagon": 2+ shape ở cột 3 (pentagon)
  (1 case có thể rơi vào nhiều nhóm double_col cùng lúc nếu 2+ cột đều bị 2+ shape)

Với mỗi case: dùng chính x_T (noise_init.pt) của case đó, resample từng cột noise
K lần độc lập (giữ nguyên 2 cột kia), đo Delta_j = ||(x0~-x0)_col j||, trung bình
qua K lần -> ma trận causal S_case (3x3) riêng cho case đó.

Toàn bộ N case x 3 cột x K trial được GỘP THÀNH 1 BATCH LỚN để chạy nhanh
(chỉ 3 zone x steps forward-pass calls, không phải N x 3 x K lần).

Sau đó gộp trung bình S_case theo từng nhóm lỗi -> ra ma trận causal trung bình
đại diện cho mỗi loại lỗi.

Usage:
    python causal_intervention_batch_all.py \\
        --traces_dir shapes_fm_output/hallucination_analysis/traces \\
        --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt \\
        --steps 25 --k_trials 50 --seed 0
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


@torch.no_grad()
def euler_rollout_batch(wrapper, x_init_batch, steps, device, chunk=2000):
    """x_init_batch: (B,3,16,16) -> x_final (B,3,16,16), raw model scale.
    Chunk qua batch lớn để tránh tràn bộ nhớ MPS/CUDA."""
    outs = []
    for start in range(0, x_init_batch.shape[0], chunk):
        x = x_init_batch[start:start + chunk]
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i * dt
            t_tensor = torch.full((x.shape[0],), t_val, device=device)
            v = wrapper(x, t_tensor)
            x = x + dt * v
        outs.append(x)
    return torch.cat(outs, dim=0)


def classify_case(img_uint8):
    """Trả về list các nhãn lỗi mà case này thuộc về (có thể nhiều nếu 2+ cột đều lỗi)."""
    r = analyze_image(img_uint8)
    if r["hall_type"] == "empty":
        return ["empty"], r
    labels = []
    for name in ZONE_NAMES:
        if r["col_blobs"][name] >= 2:
            labels.append(f"double_col_{name}")
    if not labels:
        labels = ["none"]
    return labels, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--k_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.traces_dir),
                                           "causal_intervention_batch")
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

    case_names = sorted(d for d in os.listdir(args.traces_dir)
                        if os.path.isdir(os.path.join(args.traces_dir, d))
                        and os.path.exists(os.path.join(args.traces_dir, d, "noise_init.pt")))
    N = len(case_names)
    print(f"Tìm thấy {N} case trong {args.traces_dir}")

    x_T_all = torch.cat(
        [torch.load(os.path.join(args.traces_dir, c, "noise_init.pt"), weights_only=True)
         for c in case_names], dim=0
    ).to(device)   # (N,3,16,16)

    # ── reference x_0 cho toàn bộ N case (steps=25) ──────────────────────────
    print(f"Chạy reference ODE (steps={args.steps}) cho {N} case ...")
    x_0_all = euler_rollout_batch(wrapper, x_T_all, args.steps, device)   # (N,3,16,16)

    labels_per_case = []
    for j in range(N):
        img = to_uint8_numpy(decode(x_0_all[j:j+1])[0])
        labels, r = classify_case(img)
        labels_per_case.append(labels)
    print("Phân loại xong. Ví dụ 5 case đầu:",
          list(zip(case_names[:5], labels_per_case[:5])))

    # ── Causal intervention: gộp N case x K trial thành 1 batch lớn mỗi zone ──
    K = args.k_trials
    S_case = np.zeros((N, 3, 3))   # S_case[case, i, j]

    for i, (name_i, (s0, s1)) in enumerate(zip(ZONE_NAMES, ZONE_SLICES)):
        print(f"\nResample cột noise '{name_i}' cho {N} case x K={K} trial "
              f"(batch={N*K}) ...")

        x_T_rep = x_T_all.repeat_interleave(K, dim=0)         # (N*K,3,16,16)
        fresh = torch.randn(N * K, 3, IMG_SIZE, (s1 - s0), device=device)
        x_T_tilde = x_T_rep.clone()
        x_T_tilde[:, :, :, s0:s1] = fresh

        x_0_tilde = euler_rollout_batch(wrapper, x_T_tilde, args.steps, device)  # (N*K,3,16,16)
        x_0_rep = x_0_all.repeat_interleave(K, dim=0)          # (N*K,3,16,16)
        diff = x_0_tilde - x_0_rep

        diff = diff.view(N, K, 3, IMG_SIZE, IMG_SIZE)
        for j, (name_j, (t0, t1)) in enumerate(zip(ZONE_NAMES, ZONE_SLICES)):
            d = diff[:, :, :, :, t0:t1]                        # (N,K,3,16,w_j)
            norms = d.flatten(2).norm(dim=2)                   # (N,K)
            S_case[:, i, j] = norms.mean(dim=1).cpu().numpy()

    # ── Gộp theo nhóm lỗi ──────────────────────────────────────────────────
    from collections import defaultdict
    group_S = defaultdict(list)
    for j, labels in enumerate(labels_per_case):
        for lab in labels:
            group_S[lab].append(S_case[j])

    print(f"\n{'='*60}")
    print("Số case theo nhóm lỗi:")
    for lab in sorted(group_S.keys()):
        print(f"  {lab:22s}: {len(group_S[lab])}")

    group_mean = {lab: np.mean(np.stack(mats), axis=0) for lab, mats in group_S.items()}

    print(f"\nMa trận causal trung bình theo từng nhóm lỗi:")
    for lab in sorted(group_mean.keys()):
        S = group_mean[lab]
        print(f"\n-- {lab} (n={len(group_S[lab])}) --")
        header = "            " + "".join(f"{n:>12s}" for n in ZONE_NAMES)
        print(header)
        for i, name_i in enumerate(ZONE_NAMES):
            print(f"noise {name_i:8s} " + "".join(f"{S[i,j]:12.3f}" for j in range(3)))

    # ── Lưu số liệu ────────────────────────────────────────────────────────
    np.savez(os.path.join(out_dir, "causal_by_group.npz"),
             **{lab: mat for lab, mat in group_mean.items()},
             case_names=np.array(case_names, dtype=object),
             S_case=S_case,
             labels_per_case=np.array([",".join(l) for l in labels_per_case], dtype=object))

    with open(os.path.join(out_dir, "causal_by_group.txt"), "w") as f:
        f.write(f"N case = {N}, steps={args.steps}, K_trials={K}\n\n")
        for lab in sorted(group_mean.keys()):
            S = group_mean[lab]
            f.write(f"\n-- {lab} (n={len(group_S[lab])}) --\n")
            header = "            " + "".join(f"{n:>12s}" for n in ZONE_NAMES)
            f.write(header + "\n")
            for i, name_i in enumerate(ZONE_NAMES):
                f.write(f"noise {name_i:8s} " + "".join(f"{S[i,j]:12.3f}" for j in range(3)) + "\n")

    # ── Vẽ heatmap grid, mỗi panel 1 nhóm lỗi ────────────────────────────────
    labs = sorted(group_mean.keys())
    n_labs = len(labs)
    fig, axes = plt.subplots(1, n_labs, figsize=(n_labs * 4, 4.2))
    if n_labs == 1:
        axes = [axes]
    vmax = max(m.max() for m in group_mean.values())
    for ax, lab in zip(axes, labs):
        S = group_mean[lab]
        im = ax.imshow(S, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(3)); ax.set_xticklabels(ZONE_NAMES, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(ZONE_NAMES, fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center",
                        color="white" if S[i,j] < vmax*0.6 else "black", fontsize=8)
        ax.set_title(f"{lab}\n(n={len(group_S[lab])})", fontsize=9)
        ax.set_xlabel("cột ảnh bị ảnh hưởng", fontsize=7)
        ax.set_ylabel("cột noise resample", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle(f"Ma trận causal trung bình theo nhóm lỗi  "
                f"(steps={args.steps}, K={K}, N={N} case)", fontsize=11)
    heatmap_path = os.path.join(out_dir, "causal_by_group_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {heatmap_path}")
    print(f"Saved: {os.path.join(out_dir, 'causal_by_group.txt')}")
    print(f"Out dir: {out_dir}")


if __name__ == "__main__":
    main()
