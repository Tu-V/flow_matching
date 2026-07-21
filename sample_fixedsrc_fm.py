"""
Sampling từ model đã train bằng train_shapes_fm_fixedsrc.py.

Nguồn sinh mẫu là phân phối ĐỀU trên tập 5000 điểm Gauss CỐ ĐỊNH
(fixed_source_points.pt) — tức x_init cho ODE solver luôn được chọn bằng
cách lấy chỉ số ngẫu nhiên trong {0, ..., N_SOURCE-1} rồi index vào đúng
5000 điểm đã dùng lúc train (không sinh Gauss mới).

Flow Matching sinh mẫu là DETERMINISTIC (Euler ODE, không noise injection)
=> với cùng 1 điểm nguồn z_i, model luôn cho ra đúng 1 ảnh. Vì vậy script
này cho phép:
  - Sample ngẫu nhiên có lặp (mặc định)  : giống lúc train, có thể trùng index.
  - Sample không lặp (--no_replacement)  : mỗi điểm nguồn dùng tối đa 1 lần.
  - Dùng TOÀN BỘ 5000 điểm (--use_all)   : xem ánh xạ học được trên toàn bộ
    support của p_source, hữu ích để kiểm tra memorization / hallucination
    (5000 nguồn -> 5000 ảnh sinh ra, so với 5000 ảnh p_data gốc).

Usage:
    python sample_fixedsrc_fm.py --n_samples 64
    python sample_fixedsrc_fm.py --ckpt shapes_fm_fixedsrc_output/checkpoints/unet_epoch0300.pt
    python sample_fixedsrc_fm.py --use_all --batch_size 500 --steps 100
    python sample_fixedsrc_fm.py --n_samples 200 --no_replacement --seed 42
"""

import argparse
import glob
import os
import sys

import torch
from torchvision.utils import save_image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver                             # noqa: E402
from flow_matching.utils import ModelWrapper                           # noqa: E402
from models.unet import UNetModel                                      # noqa: E402

# ── Default dirs (khớp với train_shapes_fm_fixedsrc.py) ────────────────────────
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_fixedsrc_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SOURCE_POINTS_PATH = os.path.join(OUTPUT_DIR, "fixed_source_points.pt")
SAMPLE_OUT_DIR = os.path.join(OUTPUT_DIR, "eval_samples")


# ── Model ─────────────────────────────────────────────────────────────────────
def build_unet() -> UNetModel:
    return UNetModel(
        in_channels=3, model_channels=64, out_channels=3,
        num_res_blocks=3, attention_resolutions=(2,), dropout=0.1,
        channel_mult=(1, 2, 2), conv_resample=True, dims=2,
        num_classes=None, use_checkpoint=False, num_heads=1,
        num_head_channels=-1, num_heads_upsample=-1,
        use_scale_shift_norm=True, resblock_updown=False,
        use_new_attention_order=True, with_fourier_features=False,
    )


def load_model(ckpt_path: str, device: torch.device) -> UNetModel:
    model = build_unet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", float("nan"))
    print(f"Loaded {os.path.basename(ckpt_path)}  (epoch={epoch}, loss={loss:.5f})")
    model.eval()
    return model


class UNetVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t, extra=extras)


@torch.no_grad()
def sample_batch(wrapper: UNetVelocityWrapper, x_init: torch.Tensor, steps: int):
    device = x_init.device
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
    solver = ODESolver(velocity_model=wrapper)
    return solver.sample(
        x_init=x_init,
        step_size=None,
        method="euler",
        time_grid=time_grid,
        return_intermediates=False,
    )


def decode(t: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1] float, clipped."""
    return (t.clamp(-1, 1) + 1) / 2


# ── Chọn index nguồn ─────────────────────────────────────────────────────────
def pick_source_indices(n_source: int, n_samples: int, replacement: bool,
                         use_all: bool, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if use_all:
        return torch.arange(n_source)
    if replacement:
        # Uniform({0,...,n_source-1}) có lặp — đúng cách train_shapes_fm_fixedsrc.py
        # lấy x_0 mỗi batch.
        return torch.randint(0, n_source, (n_samples,), generator=g)
    if n_samples > n_source:
        raise ValueError(
            f"--no_replacement nhưng n_samples ({n_samples}) > n_source ({n_source})"
        )
    return torch.randperm(n_source, generator=g)[:n_samples]


# ── Main ──────────────────────────────────────────────────────────────────────
def run(args):
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

    source_path = args.source_points or SOURCE_POINTS_PATH
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Không tìm thấy tập điểm nguồn cố định: {source_path}\n"
            f"Hãy chạy train_shapes_fm_fixedsrc.py trước (nó tự sinh + lưu file này)."
        )
    source_points = torch.load(source_path, weights_only=True)
    n_source = source_points.shape[0]
    print(f"Fixed source points: {source_path}  (N={n_source}, shape={tuple(source_points.shape[1:])})")

    indices = pick_source_indices(
        n_source=n_source,
        n_samples=args.n_samples,
        replacement=not args.no_replacement,
        use_all=args.use_all,
        seed=args.seed,
    )
    n_total = indices.shape[0]
    n_unique = torch.unique(indices).numel()
    print(f"Sampling {n_total} images from Uniform(fixed source, N={n_source})"
          f"  [unique indices used: {n_unique}/{n_total}]")

    os.makedirs(SAMPLE_OUT_DIR, exist_ok=True)

    all_imgs = []
    for start in range(0, n_total, args.batch_size):
        batch_idx = indices[start:start + args.batch_size]
        x_init = source_points[batch_idx].to(device)
        x_final = sample_batch(wrapper, x_init, steps=args.steps)
        all_imgs.append(decode(x_final).cpu())
        print(f"  [{min(start + args.batch_size, n_total):5d}/{n_total}]")

    imgs = torch.cat(all_imgs, dim=0)   # [n_total, 3, 16, 16] in [0,1]

    tag = "all" if args.use_all else f"n{n_total}"
    grid_path = os.path.join(SAMPLE_OUT_DIR, f"grid_{tag}_seed{args.seed}.png")
    save_image(imgs, grid_path, nrow=args.nrow, padding=1)
    print(f"Saved grid  -> {grid_path}")

    if args.save_tensors:
        idx_path = os.path.join(SAMPLE_OUT_DIR, f"indices_{tag}_seed{args.seed}.pt")
        img_path = os.path.join(SAMPLE_OUT_DIR, f"images_{tag}_seed{args.seed}.pt")
        torch.save(indices, idx_path)
        torch.save(imgs, img_path)
        print(f"Saved indices -> {idx_path}")
        print(f"Saved images  -> {img_path}")


def _latest_ckpt() -> str:
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"Không có checkpoint trong {CKPT_DIR}")
    return ckpts[-1]


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample từ phân phối đều trên tập 5000 điểm nguồn cố định"
    )
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint path (default: latest trong shapes_fm_fixedsrc_output)")
    p.add_argument("--source_points", type=str, default=None,
                   help="Path fixed_source_points.pt (default: shapes_fm_fixedsrc_output/fixed_source_points.pt)")
    p.add_argument("--n_samples", type=int, default=64,
                   help="Số ảnh cần sinh (bỏ qua nếu dùng --use_all)")
    p.add_argument("--no_replacement", action="store_true",
                   help="Không lặp index nguồn (mặc định: có lặp, giống lúc train)")
    p.add_argument("--use_all", action="store_true",
                   help="Dùng toàn bộ N điểm nguồn (mỗi điểm đúng 1 lần), bỏ qua --n_samples/--no_replacement")
    p.add_argument("--steps", type=int, default=100, help="Số bước Euler ODE")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size khi chạy ODE solver")
    p.add_argument("--nrow", type=int, default=8, help="Số ảnh mỗi hàng trong grid output")
    p.add_argument("--seed", type=int, default=0, help="Seed cho việc chọn index nguồn")
    p.add_argument("--save_tensors", action="store_true",
                   help="Lưu thêm indices + images dạng .pt để trace lại (giống noise_init.pt trong sample_shapes_fm.py)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
