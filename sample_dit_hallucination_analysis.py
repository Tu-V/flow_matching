"""
Sampling + Hallucination Analysis cho model DiT (train bằng train_shapes_fm_dit.py).

Giống hệt pipeline phân tích của sample_shapes_fm.py (PASS 1 bulk sample + detect
hallucination, PASS 2 trace chi tiết các case hallucination), chỉ khác kiến trúc model:
DiT (attention theo patch + positional embedding tường minh) thay cho UNet.

Source distribution: Gauss liên tục x_init = torch.randn(B,3,16,16) — ĐÚNG như lúc
train_shapes_fm_dit.py (không phải tập nguồn cố định 5000 điểm của train_shapes_fm_fixedsrc.py).

Kiến trúc + hyperparams (hidden_size/depth/num_heads/patch_size/mlp_ratio) được đọc
thẳng từ checkpoint (đã lưu sẵn lúc train) — không cần truyền lại qua CLI.

Image layout (3 cột shape + 1 cột padding):
  col 0 [x 0:5]  : triangle
  col 1 [x 5:10] : square
  col 2 [x 10:15]: pentagon
  col 3 [x 15:16]: padding (ignored)

Hallucination: >= 2 shapes trong 1 cột

Usage:
    python sample_dit_hallucination_analysis.py                      # 100000 samples, steps=100
    python sample_dit_hallucination_analysis.py --n_total 100000 --steps 100
    python sample_dit_hallucination_analysis.py --ckpt shapes_fm_dit_output/checkpoints/dit_epoch1000.pt
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver                             # noqa: E402
from hallucination_detector import (                                   # noqa: E402
    analyze_batch, summarize, COLUMN_NAMES,
)
from train_shapes_fm_dit import (                                      # noqa: E402
    DiT, DiTVelocityWrapper, IMG_SIZE, IN_CHANNELS,
)

# ── Default dirs (khớp với train_shapes_fm_dit.py) ──────────────────────────────
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_dit_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> DiT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = DiT(
        img_size=IMG_SIZE,
        patch_size=ckpt["patch_size"],
        in_channels=IN_CHANNELS,
        hidden_size=ckpt["hidden_size"],
        depth=ckpt["depth"],
        num_heads=ckpt["num_heads"],
        mlp_ratio=ckpt["mlp_ratio"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", float("nan"))
    print(f"Loaded {os.path.basename(ckpt_path)}  (epoch={epoch}, loss={loss:.5f})")
    print(f"  arch=DiT  hidden={ckpt['hidden_size']}  depth={ckpt['depth']}  "
          f"heads={ckpt['num_heads']}  patch={ckpt['patch_size']}")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ODE sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_batch(wrapper: DiTVelocityWrapper, x_init: torch.Tensor,
                 steps: int, return_intermediates: bool = False):
    """
    Deterministic ODE integration (Euler).
    return_intermediates=True  →  shape [steps+1, B, 3, 16, 16]  (all timesteps)
    return_intermediates=False →  shape [B, 3, 16, 16]            (final only)
    """
    device    = x_init.device
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
    solver    = ODESolver(velocity_model=wrapper)
    out = solver.sample(
        x_init=x_init,
        step_size=None,
        method="euler",
        time_grid=time_grid,
        return_intermediates=return_intermediates,
    )
    return out   # already on device


def decode(t: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1] float, clipped."""
    return (t.clamp(-1, 1) + 1) / 2


def to_uint8_numpy(img_01: torch.Tensor) -> np.ndarray:
    """[3,16,16] float [0,1] → (16,16,3) uint8."""
    return (img_01.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

SCALE = 8   # 16→128 px per image

def add_red_dividers(img_tensor_01: torch.Tensor, scale: int = SCALE) -> torch.Tensor:
    """
    img_tensor_01: [3, H, W] float [0,1].
    Scale up and paint 2 red vertical columns (at x=5*scale and x=10*scale).
    Returns [3, H*scale, W*scale].
    """
    up = F.interpolate(img_tensor_01.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)
    for x in [5 * scale, 10 * scale]:
        up[0, :, x:x+2] = 1.0   # R
        up[1, :, x:x+2] = 0.0   # G
        up[2, :, x:x+2] = 0.0   # B
    return up


def make_labeled_grid(imgs_01: list, nrow: int = 10) -> torch.Tensor:
    """imgs_01: list of [3,16,16]. Returns grid tensor with red dividers."""
    scaled = [add_red_dividers(im) for im in imgs_01]
    return make_grid(torch.stack(scaled), nrow=nrow, padding=2, pad_value=0.5)


def print_channel0_steps(intermediates: torch.Tensor, n_print: int = 5):
    """intermediates: [steps+1, 3, 16, 16] (single sample, on cpu)."""
    T      = intermediates.shape[0]
    idxs   = np.linspace(0, T - 1, n_print, dtype=int)
    t_vals = np.linspace(0.0, 1.0, T)

    print("\n  ── Channel-0 evolution (raw latent, before [-1,1]→[0,1] decode) ──")
    for idx in idxs:
        arr = intermediates[idx, 0].cpu().numpy()   # (16, 16)
        print(f"\n  step={idx:3d}  t={t_vals[idx]:.2f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")
        rows = []
        for row in arr:
            rows.append("  " + " ".join(f"{v:+.2f}" for v in row))
        print("\n".join(rows))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    # ── device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── model ──
    ckpt_path = args.ckpt or _latest_ckpt()
    model     = load_model(ckpt_path, device)
    wrapper   = DiTVelocityWrapper(model)

    # ── dirs ── (suy ra từ checkpoint đang dùng, vd .../shapes_fm_dit_output/checkpoints/xxx.pt
    #             -> .../shapes_fm_dit_output/hallucination_analysis)
    run_root     = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    analysis_dir = os.path.join(run_root, "hallucination_analysis")
    hall_dir  = os.path.join(analysis_dir, "hallucinations")
    norm_dir  = os.path.join(analysis_dir, "normal")
    trace_dir = os.path.join(analysis_dir, "traces")
    for d in [hall_dir, norm_dir, trace_dir]:
        os.makedirs(d, exist_ok=True)
    print(f"Output -> {analysis_dir}")

    # ─────────────────────────────────────────────────────────────
    # PASS 1 — bulk sampling, detect hallucinations
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PASS 1: sampling {args.n_total} images (batch={args.batch_size}, steps={args.steps})")
    print(f"        x_init ~ N(0, I)  (Gaussian liên tục, giống lúc train)")
    print(f"{'='*60}")

    all_noises   = []  # each: [B, 3, 16, 16]
    all_finals   = []  # decoded [0,1]
    all_analyses = []

    n_done = 0
    batch_id = 0
    while n_done < args.n_total:
        B = min(args.batch_size, args.n_total - n_done)
        x_init = torch.randn(B, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)

        x_final = sample_batch(wrapper, x_init, steps=args.steps, return_intermediates=False)
        x_01    = decode(x_final)   # [B, 3, 16, 16] in [0,1]

        imgs_np = np.stack([to_uint8_numpy(x_01[i]) for i in range(B)])  # (B,16,16,3)
        analyses = analyze_batch(imgs_np)

        all_noises.append(x_init.cpu())
        all_finals.append(x_01.cpu())
        all_analyses.extend(analyses)

        n_done  += B
        batch_id += 1
        n_hall_batch = sum(1 for a in analyses if a["is_hallucination"])
        if batch_id % 5 == 0 or n_done == args.n_total:
            print(f"  [{n_done:6d}/{args.n_total}]  batch hall: {n_hall_batch}/{B}")

    all_noises_t = torch.cat(all_noises, dim=0)   # [N, 3, 16, 16]
    all_finals_t = torch.cat(all_finals, dim=0)   # [N, 3, 16, 16]

    s = summarize(all_analyses)
    hall_indices   = s["hall_indices"]
    norm_indices   = s["normal_indices"] if "normal_indices" in s else [
        i for i, a in enumerate(all_analyses) if not a["is_hallucination"]
    ]
    empty_indices  = s["empty_indices"]
    double_indices = s["double_col_indices"]
    n_hall, n_norm = s["n_hall"], s["n_normal"]
    n_empty, n_double = s["n_empty"], s["n_double_col"]

    print(f"\nResults: {n_hall} hallucinations / {args.n_total} total  ({100*n_hall/args.n_total:.2f}%)")
    print(f"  ├─ empty image (0 shapes)  : {n_empty}")
    print(f"  └─ double col (2+ in 1 col): {n_double}")
    for name in COLUMN_NAMES:
        print(f"       {name:10s}: {s['col_counts'][name]['2+']} images with 2+ shapes")

    # ─────────────────────────────────────────────────────────────
    # Save hallucination grids — tách riêng 2 loại
    # ─────────────────────────────────────────────────────────────
    print(f"\n── Saving hallucination grids ──")

    if empty_indices:
        imgs = [all_finals_t[i] for i in empty_indices[:200]]
        save_image(make_labeled_grid(imgs, nrow=10),
                   os.path.join(hall_dir, "grid_empty_images.png"))
        print(f"  grid_empty_images.png  ({len(imgs)} images)")

    if double_indices:
        imgs = [all_finals_t[i] for i in double_indices[:200]]
        save_image(make_labeled_grid(imgs, nrow=10),
                   os.path.join(hall_dir, "grid_double_col.png"))
        print(f"  grid_double_col.png  ({len(imgs)} images)")

    n_show_hall = min(n_hall, 200)
    if n_show_hall > 0:
        hall_imgs = [all_finals_t[i] for i in hall_indices[:n_show_hall]]
        grid = make_labeled_grid(hall_imgs, nrow=10)
        save_image(grid, os.path.join(hall_dir, "grid_all_hallucinations.png"))
        print(f"  grid_all_hallucinations.png  ({n_show_hall} images)")

    # ─────────────────────────────────────────────────────────────
    # Save normal grid
    # ─────────────────────────────────────────────────────────────
    print(f"\n── Saving normal grid ──")
    n_show_norm = min(n_norm, 200)
    if n_show_norm > 0:
        norm_imgs = [all_finals_t[i] for i in norm_indices[:n_show_norm]]
        grid = make_labeled_grid(norm_imgs, nrow=10)
        save_image(grid, os.path.join(norm_dir, "grid_normal.png"))
        print(f"  grid_normal.png  ({n_show_norm} images)")

    # ─────────────────────────────────────────────────────────────
    # PASS 2 — re-run hallucination cases with intermediates
    # ─────────────────────────────────────────────────────────────
    n_trace = min(n_hall, args.n_trace)
    if n_trace == 0:
        print("\nNo hallucinations to trace.")
    else:
        print(f"\n{'='*60}")
        print(f"PASS 2: tracing {n_trace} hallucination cases with full ODE intermediates")
        print(f"{'='*60}")

        for case_num, global_idx in enumerate(hall_indices[:n_trace]):
            x_init_single = all_noises_t[global_idx].unsqueeze(0).to(device)  # [1,3,16,16]
            analysis      = all_analyses[global_idx]

            print(f"\n[Case {case_num+1:03d} / {n_trace}]  sample_idx={global_idx}  "
                  f"score={analysis['score']}  "
                  f"blobs={analysis['col_blobs']}")

            intermediates = sample_batch(
                wrapper, x_init_single, steps=args.steps,
                return_intermediates=True,
            )  # [steps+1, 1, 3, 16, 16]
            intermediates_sq = intermediates[:, 0]  # [steps+1, 3, 16, 16]

            case_dir = os.path.join(trace_dir, f"case_{case_num+1:04d}_idx{global_idx}")
            os.makedirs(case_dir, exist_ok=True)

            torch.save(x_init_single.cpu(), os.path.join(case_dir, "noise_init.pt"))
            torch.save(intermediates_sq.cpu(), os.path.join(case_dir, "intermediates.pt"))

            final_img = decode(intermediates_sq[-1])   # [3,16,16] in [0,1]
            save_image(
                add_red_dividers(final_img),
                os.path.join(case_dir, "final_image.png"),
            )

            n_strip = 10
            step_idxs = np.linspace(0, args.steps, n_strip, dtype=int)
            strip_imgs = [add_red_dividers(decode(intermediates_sq[si])) for si in step_idxs]
            strip_grid = make_grid(torch.stack(strip_imgs), nrow=n_strip, padding=2, pad_value=0.3)
            save_image(strip_grid, os.path.join(case_dir, "progression_strip.png"))

            report_path = os.path.join(case_dir, "report.txt")
            with open(report_path, "w") as f:
                orig_stdout = sys.stdout
                sys.stdout = f

                print(f"Case {case_num+1}  |  sample_idx={global_idx}")
                print(f"Hallucination type : {analysis['hall_type']}")
                print(f"Hallucination score: {analysis['score']}")
                for name in COLUMN_NAMES:
                    nb = analysis['col_blobs'][name]
                    flag = " <-- 2+ shapes" if nb >= 2 else ""
                    print(f"  {name:10s}: {nb} shape(s){flag}")
                if analysis['hall_type'] == 'empty':
                    print(f"  *** EMPTY IMAGE — no shapes detected anywhere ***")
                print(f"\nFlow Matching is DETERMINISTIC.")
                print(f"  noise_init.pt  → re-run with this noise to reproduce exactly.")
                print(f"  intermediates  → {args.steps+1} steps × [3,16,16] float32")

                print_channel0_steps(intermediates_sq, n_print=6)

                sys.stdout = orig_stdout

            with open(report_path) as f:
                for line in f:
                    print("  " + line, end="")

        print(f"\nAll traces saved to: {trace_dir}")

    # ─────────────────────────────────────────────────────────────
    # Summary stats file
    # ─────────────────────────────────────────────────────────────
    stats_path = os.path.join(analysis_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Checkpoint : {os.path.basename(ckpt_path)}  (arch=DiT)\n")
        f.write(f"Source     : N(0, I) Gaussian liên tục\n")
        f.write(f"Total      : {args.n_total}\n")
        f.write(f"Steps      : {args.steps}\n")
        f.write(f"Hallucin.  : {n_hall} ({100*n_hall/args.n_total:.2f}%)\n")
        f.write(f"  empty img: {n_empty} ({100*n_empty/args.n_total:.2f}%)\n")
        f.write(f"  double col: {n_double} ({100*n_double/args.n_total:.2f}%)\n")
        f.write(f"Normal     : {n_norm} ({100*n_norm/args.n_total:.2f}%)\n\n")
        f.write("Per-column breakdown:\n")
        for name in COLUMN_NAMES:
            c0 = sum(1 for a in all_analyses if a["col_blobs"][name] == 0)
            c1 = sum(1 for a in all_analyses if a["col_blobs"][name] == 1)
            c2 = sum(1 for a in all_analyses if a["col_blobs"][name] >= 2)
            f.write(f"  {name:10s}: 0-shape={c0}  1-shape={c1}  2+-shape={c2}\n")
        f.write(f"\nHallucination indices (first 1000):\n")
        f.write(" ".join(str(i) for i in hall_indices[:1000]) + "\n")

    print(f"\nSummary: {stats_path}")
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Hallucination grids: {hall_dir}/")
    print(f"  Normal grid        : {norm_dir}/grid_normal.png")
    print(f"  Trace cases        : {trace_dir}/  ({n_trace} cases)")
    print(f"  Stats              : {stats_path}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def _latest_ckpt() -> str:
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {CKPT_DIR}")
    return ckpts[-1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, default=None,
                   help="Checkpoint path (default: latest trong shapes_fm_dit_output)")
    p.add_argument("--n_total",    type=int, default=100000,
                   help="Total samples to generate (default: 100000)")
    p.add_argument("--batch_size", type=int, default=512,
                   help="Batch size per forward pass (default: 512)")
    p.add_argument("--steps",      type=int, default=100,
                   help="ODE Euler steps (default: 100)")
    p.add_argument("--n_trace",    type=int, default=50,
                   help="Max hallucination cases to trace with full intermediates (default: 50)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
