"""
Sampling-time Mahalanobis guidance cho Flow Matching (không train lại model).

Ý tưởng (gần với Discriminator Guidance / Manifold Constrained Gradient):
  Tại mỗi bước Euler, dùng velocity model dự đoán điểm cuối tạm thời
      x1_pred = x_t + (1 - t) * v_theta(x_t, t)
  rồi cộng thêm 1 "guidance velocity" kéo x1_pred về gần vùng dữ liệu thật
  (đo bằng Mahalanobis distance, dùng đúng Sigma đã ước lượng từ dataset thật):
      v_guided = v_theta(x_t, t)  -  w(t) * grad_{x_t} D_M(x1_pred)^2
  w(t) = w_max * t^power   (tăng dần về cuối trajectory, t: 0->1)

Usage:
    # calibrate: thử nhiều w_max trên N nhỏ, xem hallucination rate + ảnh
    python mahalanobis_guided_sampling.py --mode calibrate \\
        --data_dir ../neurips-2024-diffusion-model-hallucination/simple-datasets/simple-shapes-5k-16x16 \\
        --n_total 1000 --steps 25 --w_max_list 0,0.001,0.003,0.01,0.03,0.1

    # so sánh baseline vs guided trên cùng N lớn (paired, cùng seed)
    python mahalanobis_guided_sampling.py --mode compare \\
        --data_dir ../neurips-2024-diffusion-model-hallucination/simple-datasets/simple-shapes-5k-16x16 \\
        --n_total 100000 --steps 25 --w_max 0.01 --w_power 2.0
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid, save_image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from sample_shapes_fm import (   # noqa: E402
    load_model, UNetVelocityWrapper, decode, to_uint8_numpy, _latest_ckpt,
)
from hallucination_detector import analyze_batch, summarize   # noqa: E402

IMG_SIZE = 16


# ─────────────────────────────────────────────────────────────────────────────
# Mahalanobis model (torch, differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_images(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    assert paths, f"Không tìm thấy .png trong {data_dir}"
    vecs = []
    for p in paths:
        arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        vecs.append(arr.flatten(order="F"))
    return np.stack(vecs, axis=0)


class MahalanobisTorch:
    """Pseudo-inverse Sigma qua eigen-decomposition, dùng được với autograd."""

    def __init__(self, X, device, rel_eps=1e-3):
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        thresh = vals.max() * rel_eps
        mask = vals > thresh
        vals_k, vecs_k = vals[mask], vecs[:, mask]
        print(f"  [MahalanobisTorch] giữ {mask.sum()}/{len(vals)} eigenvalue "
              f"(rel_eps={rel_eps})")
        # W = V / sqrt(lambda)  ->  D_M^2(x) = || (x-mu) @ W ||^2
        W = vecs_k / np.sqrt(vals_k)[None, :]
        self.mu = torch.from_numpy(mu.astype(np.float32)).to(device)          # (256,)
        self.W  = torch.from_numpy(W.astype(np.float32)).to(device)           # (256,k)

    def d2(self, x1_pred_raw):
        """x1_pred_raw: (B,3,H,W) giá trị model output thô (~[-1,1]) -> D_M^2: (B,)"""
        x01 = (x1_pred_raw.clamp(-1, 1) + 1) / 2          # decode giống decode()
        gray = x01.mean(dim=1)                             # (B,H,W)
        vec = gray.transpose(1, 2).reshape(gray.shape[0], -1)   # column-major flatten (B,256)
        diff = vec - self.mu
        proj = diff @ self.W                                # (B,k)
        return (proj ** 2).sum(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Guided Euler sampler
# ─────────────────────────────────────────────────────────────────────────────

def guided_euler_sample(wrapper, x_init, steps, mahal=None, w_max=0.0, w_power=2.0,
                         eps=1e-8):
    """
    Euler integration t: 0->1, tuỳ chọn cộng Mahalanobis guidance mỗi step.

    Gradient của D_M^2 dao động biên độ rất khác nhau theo t (quan sát thực nghiệm:
    ~0.02 lúc t gần 0, đỉnh ~19 quanh t~0.2, rồi giảm còn ~0.6-2 lúc t gần 1) trong
    khi |v| gần như hằng số suốt trajectory. Nếu cộng thẳng w*grad, cùng 1 w sẽ tạo
    hiệu ứng rất khác nhau tùy t. Nên CHUẨN HOÁ: chỉ lấy HƯỚNG của gradient, rồi scale
    theo |v| của chính bước đó — w_t khi đó có nghĩa rõ ràng là "tỉ lệ % của |v| bị
    điều chỉnh theo hướng giảm Mahalanobis distance".
    """
    device = x_init.device
    dt = 1.0 / steps
    x = x_init
    for i in range(steps):
        t_val = i * dt
        w_t = w_max * (t_val ** w_power) if (mahal is not None and w_max > 0) else 0.0

        if w_t > 0:
            x = x.detach().requires_grad_(True)
            t_tensor = torch.full((x.shape[0],), t_val, device=device)
            v = wrapper(x, t_tensor)
            x1_pred = x + (1 - t_val) * v
            d2 = mahal.d2(x1_pred)
            grad = torch.autograd.grad(d2.sum(), x)[0]

            B = x.shape[0]
            v_norm = v.detach().flatten(1).norm(dim=1).view(B, 1, 1, 1)
            grad_norm = grad.flatten(1).norm(dim=1).clamp_min(eps).view(B, 1, 1, 1)
            correction = w_t * v_norm * (grad / grad_norm)

            v_guided = v.detach() - correction
            x = x.detach() + dt * v_guided
        else:
            with torch.no_grad():
                t_tensor = torch.full((x.shape[0],), t_val, device=device)
                v = wrapper(x, t_tensor)
                x = x + dt * v
    return x.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Sampling + hallucination check
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _decode_batch_to_uint8(x_raw):
    x01 = decode(x_raw)
    return np.stack([to_uint8_numpy(x01[i]) for i in range(x01.shape[0])])


def run_batch_sampling(wrapper, mahal, n_total, batch_size, steps, w_max, w_power,
                        seed, device):
    torch.manual_seed(seed)
    all_analyses = []
    n_done = 0
    t0 = time.time()
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = torch.randn(B, 3, IMG_SIZE, IMG_SIZE, device=device)
        x_final = guided_euler_sample(wrapper, x_init, steps, mahal, w_max, w_power)
        imgs = _decode_batch_to_uint8(x_final)
        all_analyses.extend(analyze_batch(imgs))
        n_done += B
        if n_done % (batch_size * 10) == 0 or n_done >= n_total:
            elapsed = time.time() - t0
            print(f"    [{n_done:7d}/{n_total}]  {n_done/elapsed:.1f} img/s  "
                  f"elapsed={elapsed:.0f}s", flush=True)
    return all_analyses


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def mode_calibrate(args, wrapper, mahal, device):
    w_max_list = [float(w) for w in args.w_max_list.split(",")]
    print(f"\nCalibrate: N={args.n_total}, steps={args.steps}, w_max_list={w_max_list}")

    results = []
    for w_max in w_max_list:
        print(f"\n-- w_max={w_max} --")
        analyses = run_batch_sampling(
            wrapper, mahal, args.n_total, args.batch_size, args.steps,
            w_max, args.w_power, seed=args.seed, device=device,
        )
        s = summarize(analyses)
        print(f"   hall={s['n_hall']}/{args.n_total} ({100*s['hall_rate']:.2f}%)  "
              f"empty={s['n_empty']}  double_col={s['n_double_col']}")
        results.append((w_max, s))

    print(f"\n{'w_max':>8} {'hall%':>8} {'n_hall':>7} {'empty':>7} {'double':>7}")
    for w_max, s in results:
        print(f"{w_max:>8} {100*s['hall_rate']:>7.2f}% {s['n_hall']:>7} "
              f"{s['n_empty']:>7} {s['n_double_col']:>7}")

    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([w for w, _ in results], [100 * s["hall_rate"] for _, s in results],
            marker="o", color="crimson")
    ax.set_xlabel("w_max")
    ax.set_ylabel("hallucination rate (%)")
    ax.set_title(f"Mahalanobis guidance calibration  (N={args.n_total}, steps={args.steps})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "calibration_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def mode_compare(args, wrapper, mahal, device):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n[Baseline]  w_max=0  N={args.n_total}")
    analyses_base = run_batch_sampling(
        wrapper, mahal, args.n_total, args.batch_size, args.steps,
        0.0, args.w_power, seed=args.seed, device=device,
    )
    s_base = summarize(analyses_base)

    print(f"\n[Guided]  w_max={args.w_max}  w_power={args.w_power}  N={args.n_total}")
    analyses_guided = run_batch_sampling(
        wrapper, mahal, args.n_total, args.batch_size, args.steps,
        args.w_max, args.w_power, seed=args.seed, device=device,
    )
    s_guided = summarize(analyses_guided)

    print(f"\n{'='*60}")
    print(f"Baseline (w_max=0)      : hall={s_base['n_hall']:5d}/{args.n_total}  "
          f"({100*s_base['hall_rate']:.3f}%)  empty={s_base['n_empty']}  "
          f"double_col={s_base['n_double_col']}")
    print(f"Guided   (w_max={args.w_max})  : hall={s_guided['n_hall']:5d}/{args.n_total}  "
          f"({100*s_guided['hall_rate']:.3f}%)  empty={s_guided['n_empty']}  "
          f"double_col={s_guided['n_double_col']}")
    rel_change = 100 * (s_guided["hall_rate"] - s_base["hall_rate"]) / max(s_base["hall_rate"], 1e-9)
    print(f"Relative change in hallucination rate: {rel_change:+.1f}%")

    stats_path = os.path.join(args.out_dir, "compare_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"N={args.n_total}  steps={args.steps}  seed={args.seed}  "
                f"w_max={args.w_max}  w_power={args.w_power}\n\n")
        f.write(f"Baseline : hall={s_base['n_hall']}/{args.n_total} "
                f"({100*s_base['hall_rate']:.3f}%)  empty={s_base['n_empty']}  "
                f"double_col={s_base['n_double_col']}\n")
        f.write(f"Guided   : hall={s_guided['n_hall']}/{args.n_total} "
                f"({100*s_guided['hall_rate']:.3f}%)  empty={s_guided['n_empty']}  "
                f"double_col={s_guided['n_double_col']}\n")
        f.write(f"Relative change: {rel_change:+.1f}%\n")
    print(f"\nSaved: {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "compare"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--n_total", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rel_eps", type=float, default=1e-3)
    parser.add_argument("--w_max_list", type=str, default="0,0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--w_max", type=float, default=0.01)
    parser.add_argument("--w_power", type=float, default=2.0)
    parser.add_argument("--out_dir", default="shapes_fm_output/mahalanobis_guidance")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading real dataset: {args.data_dir}")
    X_real = load_dataset_images(args.data_dir)
    mahal = MahalanobisTorch(X_real, device, rel_eps=args.rel_eps)

    ckpt_path = args.ckpt or _latest_ckpt()
    model = load_model(ckpt_path, device)
    for p in model.parameters():
        p.requires_grad_(False)   # chỉ cần grad theo x, không cần grad theo weight
    wrapper = UNetVelocityWrapper(model)

    if args.mode == "calibrate":
        mode_calibrate(args, wrapper, mahal, device)
    else:
        mode_compare(args, wrapper, mahal, device)


if __name__ == "__main__":
    main()
