"""
Sample n_total ảnh từ checkpoint cnn_pixel01_isotropic_final.pt (CNN/UNet, p0 isotropic,
train trên pixel range [0,1] KHÔNG chuẩn hoá [-0.5,0.5]) và tính tỉ lệ hallucination
(double_col / empty) bằng hallucination_detector.py dùng chung toàn repo.

KHÁC BIỆT so với các script cũ (shapes_fm_*, train_anisotropic_p0_experiment*.py): map
output model -> ảnh hiển thị dùng clamp(0,1) trực tiếp (KHÔNG (clamp(-1,1)+1)/2), vì
model này train trên x1 trong [0,1], không phải [-1,1].

Usage:
    python sample_analyze_cnn_pixel01.py
    python sample_analyze_cnn_pixel01.py --n_sample_total 100000 --n_repeats 5
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver         # noqa: E402
from flow_matching.utils import ModelWrapper        # noqa: E402
from models.unet import UNetModel                    # noqa: E402
from hallucination_detector import analyze_batch, summarize   # noqa: E402

IN_CHANNELS = 1
CKPT_PATH = os.path.join(
    REPO_ROOT, "shape_exp", "64_64", "cnn_pixel01_isotropic_64x64_output",
    "cnn_pixel01_isotropic_final.pt",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "shape_exp", "64_64", "cnn_pixel01_isotropic_64x64_output")


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


class UNetVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t, extra=extras)


@torch.no_grad()
def sample_and_analyze(model, device, n_total, steps, batch_size, img_size):
    model.eval()
    wrapper = UNetVelocityWrapper(model)
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
        # pixel range [0,1] -> clamp(0,1) trực tiếp, KHÔNG (clamp(-1,1)+1)/2
        x_01 = x_final.clamp(0, 1)
        x_01_rgb = x_01.repeat(1, 3, 1, 1) if x_01.shape[1] == 1 else x_01
        imgs_np = (x_01_rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        all_analyses.extend(analyze_batch(imgs_np))
        n_done += B
        print(f"    sampled {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", end="\r")
    print()
    return summarize(all_analyses)


def sample_and_analyze_repeated(model, device, n_total, steps, batch_size, n_repeats, img_size):
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
        "hall_rate_mean":   float(hall_rates.mean()),
        "hall_rate_std":    float(hall_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "empty_rate_mean":  float(empty_rates.mean()),
        "empty_rate_std":   float(empty_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "double_rate_mean": float(double_rates.mean()),
        "double_rate_std":  float(double_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "hall_rate_pct_list":   hall_rates.tolist(),
        "empty_rate_pct_list":  empty_rates.tolist(),
        "double_rate_pct_list": double_rates.tolist(),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sample_total", type=int, default=100_000)
    p.add_argument("--n_repeats", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    ck = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = build_unet().to(device)
    model.load_state_dict(ck["model_state_dict"])
    print(f"Loaded {CKPT_PATH}  (step={ck['step']}, n_data={ck['n_data']}, "
          f"pixel_range={ck['pixel_range']}, img_size={ck['img_size']})")

    print(f"\nSampling {a.n_sample_total} ảnh x {a.n_repeats} lần lặp "
          f"(p0=isotropic, steps={a.sample_steps}) ...")
    s = sample_and_analyze_repeated(model, device, a.n_sample_total, a.sample_steps,
                                     a.sample_batch_size, a.n_repeats, ck["img_size"])

    print(f"\nhall={s['hall_rate_mean']:.4f}% ± {s['hall_rate_std']:.4f}%  "
          f"(empty={s['empty_rate_mean']:.4f}%±{s['empty_rate_std']:.4f}%  "
          f"double_col={s['double_rate_mean']:.4f}%±{s['double_rate_std']:.4f}%)")

    out_path = os.path.join(OUTPUT_DIR, "stats_pixel01_isotropic.txt")
    with open(out_path, "w") as f:
        f.write(f"CNN pixel01 isotropic  step={ck['step']}  n_data={ck['n_data']}  "
                f"pixel_range={ck['pixel_range']}  img_size={ck['img_size']}\n")
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
