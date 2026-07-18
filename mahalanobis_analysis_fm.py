"""
Kiểm tra ảnh sinh ra bởi Flow Matching có "tuân theo" covariance structure của
dataset thật hay không, bằng Mahalanobis distance.

  D_M(x)^2 = (x-mu)^T Sigma^-1 (x-mu)

mu, Sigma ước lượng từ dataset thật (simple-shapes-5k-16x16, ảnh xám, flatten
theo cột — giống hệt analyze_dataset_covariance.py ở repo neurips-2024-...).
Vì Sigma suy biến (cột padding variance=0, mỗi vùng shape rank thấp), dùng
pseudo-inverse qua eigen-decomposition, chỉ giữ eigenvalue đủ lớn.

So sánh 3 nhóm:
  1. Ảnh thật trong dataset (baseline — chính là dữ liệu ước lượng ra Sigma)
  2. Ảnh "normal" mới sinh bởi flow matching (không hallucinate)
  3. 12 case hallucination đã tìm được (case_XXXX_idxYYY/noise_init.pt)

Cũng tính Mahalanobis riêng cho từng vùng shape (triangle/square/pentagon) để
xem case hallucination lệch nhiều nhất ở vùng nào.

Usage:
    python mahalanobis_analysis_fm.py \\
        --data_dir ../neurips-2024-diffusion-model-hallucination/simple-datasets/simple-shapes-5k-16x16 \\
        --traces_dir shapes_fm_output/hallucination_analysis/traces \\
        --n_normal 300 --steps 25 \\
        --out_dir shapes_fm_output/hallucination_analysis/mahalanobis
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
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from sample_shapes_fm import (   # noqa: E402
    load_model, UNetVelocityWrapper, sample_batch, decode, to_uint8_numpy,
    _latest_ckpt,
)
from hallucination_detector import analyze_image   # noqa: E402

IMG_SIZE = 16
ZONE_NAMES = ["triangle", "square", "pentagon"]
ZONE_BOUNDS_COL = [0, 5, 10, 15]   # triangle/square/pentagon (bỏ padding)


def load_dataset_images(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    assert paths, f"Không tìm thấy .png trong {data_dir}"
    vecs = []
    for p in paths:
        arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        vecs.append(arr.flatten(order="F"))
    return np.stack(vecs, axis=0)   # (N, 256)


def rgb_uint8_to_vec(img_hwc_uint8):
    """(16,16,3) uint8 -> vector 256 chiều (grayscale, column-major, [0,1])."""
    gray = img_hwc_uint8.astype(np.float32).mean(axis=2) / 255.0   # (16,16)
    return gray.flatten(order="F")


class MahalanobisModel:
    """Pseudo-inverse Sigma qua eigen-decomposition, chỉ giữ eigenvalue lớn."""

    def __init__(self, X, rel_eps=1e-3):
        self.mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)   # tăng dần
        thresh = vals.max() * rel_eps
        mask = vals > thresh
        self.vals = vals[mask]
        self.vecs = vecs[:, mask]
        self.k = mask.sum()
        print(f"  [Mahalanobis] giữ {self.k}/{len(vals)} eigenvalue "
              f"(rel_eps={rel_eps}, thresh={thresh:.2e}, max={vals.max():.4f})")

    def d2(self, x):
        """x: (D,) hoặc (N,D) -> Mahalanobis^2 (scalar hoặc (N,))."""
        diff = x - self.mu
        proj = diff @ self.vecs                       # (..., k)
        return np.sum(proj ** 2 / self.vals, axis=-1)


@torch.no_grad()
def ode_decode(wrapper, x_np_or_th, steps, device):
    if isinstance(x_np_or_th, np.ndarray):
        x = torch.from_numpy(x_np_or_th).unsqueeze(0).to(device)
    else:
        x = x_np_or_th.to(device)
    x_final = sample_batch(wrapper, x, steps=steps, return_intermediates=False)
    x_01 = decode(x_final)
    return to_uint8_numpy(x_01[0])   # (16,16,3) uint8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--traces_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--n_normal", type=int, default=300,
                        help="Số sample flow-matching 'normal' mới sinh để làm baseline")
    parser.add_argument("--rel_eps", type=float, default=1e-3,
                        help="Ngưỡng tương đối giữ eigenvalue khi pseudo-invert Sigma")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── 1. Fit Mahalanobis model từ dataset thật ─────────────────────────────
    print(f"\nLoading real dataset: {args.data_dir}")
    X_real = load_dataset_images(args.data_dir)
    print(f"  N={X_real.shape[0]}, D={X_real.shape[1]}")
    mm_full = MahalanobisModel(X_real, rel_eps=args.rel_eps)

    # Model riêng cho từng vùng shape (80 chiều/vùng)
    zone_models = {}
    for zi, name in enumerate(ZONE_NAMES):
        lo, hi = ZONE_BOUNDS_COL[zi] * IMG_SIZE, ZONE_BOUNDS_COL[zi + 1] * IMG_SIZE
        zone_models[name] = MahalanobisModel(X_real[:, lo:hi], rel_eps=args.rel_eps)

    d2_real = mm_full.d2(X_real)   # (N,) — baseline "self-consistency" của data thật

    # ── 2. Model + sinh 'normal' baseline mới ────────────────────────────────
    ckpt_path = args.ckpt or _latest_ckpt()
    model = load_model(ckpt_path, device)
    wrapper = UNetVelocityWrapper(model)

    print(f"\nSinh {args.n_normal} sample flow-matching mới để làm baseline 'normal'...")
    normal_vecs = []
    tried = 0
    while len(normal_vecs) < args.n_normal and tried < args.n_normal * 5:
        x_init = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        img = ode_decode(wrapper, x_init, args.steps, device)
        r = analyze_image(img)
        tried += 1
        if not r["is_hallucination"]:
            normal_vecs.append(rgb_uint8_to_vec(img))
    X_normal = np.stack(normal_vecs, axis=0)
    print(f"  Lấy được {len(normal_vecs)} ảnh normal (thử {tried} lần)")
    d2_normal = mm_full.d2(X_normal)

    # ── 3. 12 case hallucination ──────────────────────────────────────────────
    case_dirs = sorted(d for d in os.listdir(args.traces_dir)
                       if os.path.isdir(os.path.join(args.traces_dir, d)))
    print(f"\nTìm thấy {len(case_dirs)} case hallucination trong {args.traces_dir}")

    case_results = []
    for cd in case_dirs:
        case_path = os.path.join(args.traces_dir, cd)
        noise = torch.load(os.path.join(case_path, "noise_init.pt"), weights_only=True)
        img = ode_decode(wrapper, noise, args.steps, device)
        r = analyze_image(img)
        vec = rgb_uint8_to_vec(img)
        d2_total = mm_full.d2(vec)

        zone_d2 = {}
        for zi, name in enumerate(ZONE_NAMES):
            lo, hi = ZONE_BOUNDS_COL[zi] * IMG_SIZE, ZONE_BOUNDS_COL[zi + 1] * IMG_SIZE
            zone_d2[name] = zone_models[name].d2(vec[lo:hi])

        case_results.append({
            "case": cd, "d2": d2_total, "zone_d2": zone_d2,
            "hall_type": r["hall_type"], "blobs": r["col_blobs"],
        })
        print(f"  {cd:22s}  D_M^2={d2_total:8.2f}  type={r['hall_type']:10s}  "
              f"zone_D2=" + "  ".join(f"{k}:{v:.1f}" for k, v in zone_d2.items()))

    # ── 4. So sánh phân phối ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"So sánh D_M^2 (tổng, k={mm_full.k} chiều giữ lại):")
    print(f"  Real data   : mean={d2_real.mean():8.2f}  median={np.median(d2_real):8.2f}  "
          f"p95={np.percentile(d2_real, 95):8.2f}  p99={np.percentile(d2_real, 99):8.2f}  "
          f"max={d2_real.max():8.2f}")
    print(f"  FM normal   : mean={d2_normal.mean():8.2f}  median={np.median(d2_normal):8.2f}  "
          f"p95={np.percentile(d2_normal, 95):8.2f}  p99={np.percentile(d2_normal, 99):8.2f}  "
          f"max={d2_normal.max():8.2f}")
    case_d2s = np.array([c["d2"] for c in case_results])
    print(f"  FM hall(12) : mean={case_d2s.mean():8.2f}  median={np.median(case_d2s):8.2f}  "
          f"min={case_d2s.min():8.2f}  max={case_d2s.max():8.2f}")

    n_above_p99 = sum(1 for d in case_d2s if d > np.percentile(d2_normal, 99))
    print(f"\n  {n_above_p99}/12 case hallucination có D_M^2 > p99 của FM-normal "
          f"({np.percentile(d2_normal, 99):.2f})")

    # ── 5. Vẽ ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins = np.linspace(0, max(d2_real.max(), d2_normal.max(), case_d2s.max()) * 1.05, 60)
    ax.hist(d2_real, bins=bins, alpha=0.5, density=True, label=f"real data (N={len(d2_real)})", color="steelblue")
    ax.hist(d2_normal, bins=bins, alpha=0.5, density=True, label=f"FM normal (N={len(d2_normal)})", color="seagreen")
    for c in case_results:
        ax.axvline(c["d2"], color="crimson", linewidth=1.2, alpha=0.85)
    ax.axvline(case_d2s[0], color="crimson", linewidth=1.2, alpha=0.85, label="FM hallucination (12 case)")
    ax.set_xlabel("Mahalanobis$^2$  D_M(x)")
    ax.set_ylabel("density")
    ax.set_title("Phân phối D_M² — real data vs FM-normal vs FM-hallucination")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [c["case"].replace("case_", "").replace("_idx", "\nidx") for c in case_results]
    colors = ["tomato" if c["hall_type"] == "double_col" else "orange" for c in case_results]
    ax.bar(range(len(case_results)), case_d2s, color=colors)
    ax.axhline(np.percentile(d2_normal, 99), color="seagreen", linestyle="--",
               label=f"FM-normal p99 ({np.percentile(d2_normal, 99):.1f})")
    ax.axhline(np.percentile(d2_normal, 50), color="seagreen", linestyle=":",
               label=f"FM-normal median ({np.median(d2_normal):.1f})")
    ax.set_xticks(range(len(case_results)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Mahalanobis$^2$")
    ax.set_title("D_M² từng case hallucination  (đỏ=double_col, cam=empty)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "mahalanobis_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # ── 6. Per-zone breakdown cho 12 case ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(case_results))
    width = 0.25
    for zi, name in enumerate(ZONE_NAMES):
        vals = [c["zone_d2"][name] for c in case_results]
        ax.bar(x + (zi - 1) * width, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Mahalanobis$^2$ (per-zone, 80-dim)")
    ax.set_title("D_M² theo từng vùng shape — 12 case hallucination")
    ax.legend()
    plt.tight_layout()
    out_path2 = os.path.join(args.out_dir, "mahalanobis_per_zone.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path2}")

    # ── 7. Lưu số liệu thô ─────────────────────────────────────────────────────
    np.savez(
        os.path.join(args.out_dir, "mahalanobis_raw.npz"),
        d2_real=d2_real, d2_normal=d2_normal,
        case_names=np.array([c["case"] for c in case_results]),
        case_d2=case_d2s,
    )
    print(f"Saved: {os.path.join(args.out_dir, 'mahalanobis_raw.npz')}")


if __name__ == "__main__":
    main()
