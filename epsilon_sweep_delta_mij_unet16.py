"""
Quét epsilon quanh vùng Δ (Thm 3.2, collapse -> "empty") và vùng M_ij (Thm 3.3,
two-index -> "double_col") của checkpoint UNet 16x16 ĐÃ TRAIN
shape_exp/unet_architecture/anisotropic_p0_output_16x16/checkpoints/unet_n5000_isotropic.pt.

Ý tưởng: x0 = (điểm Δ hoặc điểm M_ij) + epsilon * N(0,I)  (nhiễu TOÀN ẢNH, không chỉ ở vùng
template), epsilon quét từ 0 -> 1. Với mỗi epsilon: sample --n_sample_total ảnh, lặp lại
--n_repeats lần (x_init random khác nhau mỗi lần) để lấy mean/std của tỉ lệ hallucination.
Vùng Δ CHỈ đếm loại "empty" (blank-shape). Vùng M_ij CHỈ đếm loại "double_col" (2 shape).
Kỳ vọng: epsilon nhỏ -> gần đúng vùng lý thuyết -> tỉ lệ bắt được hallucination cao hơn.

Xác định 2 điểm neo trước khi quét (rẻ, không tốn nhiều compute):
  - Δ: c* = giá trị pixel trung bình của tập train thật (5000 ảnh, đã normalize [-1,1]).
       x0_delta = c* * ones(1,16,16).
  - M_ij: 2 "mode" = 2 vị trí triangle không chạm nhau trong CỘT 0 (hàng 1-3 và hàng 10-12,
       cột 0-5). x0_mij = 0 khắp ảnh, trừ 2 vùng đó = A0 (B0=0, khớp giá trị trung bình của
       N(0,I)). A0 được dò bằng 1 lưới thô (quét A, mỗi A thêm nhiễu nhỏ để đo tỉ lệ double_col,
       chọn A cho tỉ lệ cao nhất) trước khi chạy sweep epsilon chính.

Usage:
    python epsilon_sweep_delta_mij_unet16.py                 # full (100k sample x 5 repeat x 2 vùng x 15 eps)
    python epsilon_sweep_delta_mij_unet16.py --quick          # chạy thử cực nhanh (kiểm tra lỗi)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset                                            # noqa: E402
from torchvision import transforms                                              # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver                                      # noqa: E402
from flow_matching.utils import ModelWrapper                                    # noqa: E402
from models.unet import UNetModel                                               # noqa: E402
from hallucination_detector import analyze_batch                                # noqa: E402

IMG_SIZE = 16
IN_CHANNELS = 3   # checkpoint này (anisotropic_p0_output_16x16) được train TRƯỚC khi
                  # train_anisotropic_p0_experiment.py đổi sang 1 kênh grayscale -- vẫn
                  # là UNet 3 kênh RGB giống train_shapes_fm.py (đã verify shape checkpoint).
CKPT_PATH = os.path.join(
    REPO_ROOT, "shape_exp", "unet_architecture", "anisotropic_p0_output_16x16",
    "checkpoints", "unet_n5000_isotropic.pt",
)
DATASET_ROOT = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "simple-datasets",
)
OUT_DIR = os.path.join(
    REPO_ROOT, "shape_exp", "unet_architecture", "anisotropic_p0_output_16x16",
    "epsilon_sweep_delta_mij",
)
# 2 vùng triangle (cột 0, x in [0,5)) không chạm nhau: hàng 1-3 (trên) và hàng 10-12 (dưới)
GROUP_BOXES = [
    (slice(1, 4), slice(0, 5)),
    (slice(10, 13), slice(0, 5)),
]


def size_label(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


class ShapesDataset(Dataset):
    """Giống train_shapes_fm.py: RGB, resize 16x16, normalize [-1,1]."""

    def __init__(self, root: str, img_size: int):
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".png")
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


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


def decode_to_uint8_rgb(x1: torch.Tensor) -> np.ndarray:
    """x1: [N,3,H,W] thang model (~[-1,1]) -> uint8 RGB [N,H,W,3] cho hallucination_detector."""
    x01 = (x1.clamp(-1, 1) + 1) / 2
    return (x01.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def sample_from_x0(solver, x0: torch.Tensor, steps: int) -> torch.Tensor:
    tg = torch.linspace(0.0, 1.0, steps + 1, device=x0.device)
    return solver.sample(x_init=x0, step_size=None, method="euler", time_grid=tg,
                          return_intermediates=False)


def compute_c_star(device) -> float:
    label = size_label(5000)
    data_dir = os.path.join(DATASET_ROOT, f"simple-shapes-{label}-{IMG_SIZE}x{IMG_SIZE}")
    ds = ShapesDataset(data_dir, IMG_SIZE)
    imgs = torch.stack([ds[i] for i in range(len(ds))])
    return float(imgs.mean().item())


def calibrate_A0(solver, device, steps: int, n_calib: int, eps_calib: float, seed: int):
    """Quét A0 thô (B0=0 cố định), mỗi A0 thêm nhiễu nhỏ (eps_calib) để đo tỉ lệ double_col,
    chọn A0 cho tỉ lệ cao nhất. Rẻ: chỉ dùng n_calib mẫu/A0 (không phải n_sample_total)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    best_A, best_rate = 0.0, -1.0
    log_lines = []
    for A in torch.arange(-2.0, 2.01, 0.25):
        base = torch.zeros(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
        for y_sl, x_sl in GROUP_BOXES:
            base[:, :, y_sl, x_sl] = A
        noise = torch.randn(n_calib, IN_CHANNELS, IMG_SIZE, IMG_SIZE, generator=g) * eps_calib
        x0 = (base + noise).to(device)
        x1 = sample_from_x0(solver, x0, steps)
        analyses = analyze_batch(decode_to_uint8_rgb(x1))
        rate = 100.0 * sum(1 for an in analyses if an["hall_type"] == "double_col") / len(analyses)
        log_lines.append(f"  A0={A.item():+.2f}  B0=0  double_col%={rate:.1f}  (n={n_calib}, eps_calib={eps_calib})")
        if rate > best_rate:
            best_rate, best_A = rate, float(A.item())
    return best_A, best_rate, log_lines


def sweep_region(name: str, base_point: torch.Tensor, hall_type: str, solver, device,
                  steps: int, n_total: int, n_repeats: int, batch_size: int,
                  eps_list, seed: int, rep_fn):
    rows = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for eps in eps_list:
        is_exact = (eps == 0.0)
        n_rep_eff = 1 if is_exact else n_repeats
        n_tot_eff = 1 if is_exact else n_total
        rates, counts = [], []
        t0 = time.time()
        for rep in range(n_rep_eff):
            n_done, n_hit = 0, 0
            while n_done < n_tot_eff:
                B = min(batch_size, n_tot_eff - n_done)
                noise = torch.randn(B, IN_CHANNELS, IMG_SIZE, IMG_SIZE, generator=g) * eps
                x0 = (base_point + noise).to(device)
                x1 = sample_from_x0(solver, x0, steps)
                analyses = analyze_batch(decode_to_uint8_rgb(x1))
                n_hit += sum(1 for an in analyses if an["hall_type"] == hall_type)
                n_done += B
            counts.append(n_hit)
            rates.append(100.0 * n_hit / n_tot_eff)
        mean, std = float(np.mean(rates)), (float(np.std(rates, ddof=1)) if n_rep_eff > 1 else 0.0)
        count_mean = float(np.mean(counts))
        count_std = float(np.std(counts, ddof=1)) if n_rep_eff > 1 else 0.0
        rep_fn(f"  [{name}] eps={eps:<6g}  {hall_type} count mean={count_mean:8.1f} std={count_std:6.1f}  "
               f"({hall_type}%% mean={mean:6.2f}  std={std:5.2f})  "
               f"(n={n_tot_eff}, repeats={n_rep_eff}, counts={counts}, rates={[f'{r:.2f}' for r in rates]}, "
               f"elapsed={time.time()-t0:.0f}s)")
        rows.append({"region": name, "epsilon": eps, "hall_type": hall_type,
                     "n_total_per_repeat": n_tot_eff, "n_repeats": n_rep_eff,
                     "count_mean": count_mean, "count_std": count_std,
                     "rate_mean": mean, "rate_std": std,
                     "counts_list": ";".join(str(c) for c in counts),
                     "rates_list": ";".join(f"{r:.4f}" for r in rates)})
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--n_calib", type=int, default=50)
    p.add_argument("--eps_calib", type=float, default=0.1)
    p.add_argument("--eps_list", type=str,
                   default="0.0,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--quick", action="store_true", help="cấu hình siêu nhỏ để kiểm tra lỗi")
    a = p.parse_args()
    if a.quick:
        a.n_sample_total, a.n_repeats, a.n_calib = 20, 2, 10
        a.eps_list = "0.0,0.1,1.0"
        a.sample_steps = 10
    return a


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    os.makedirs(OUT_DIR, exist_ok=True)
    report_path = os.path.join(OUT_DIR, "results.txt")
    report = open(report_path, "w")

    def rep(s=""):
        print(s)
        report.write(s + "\n")
        report.flush()

    ck = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = build_unet().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    rep(f"Loaded {CKPT_PATH}  n_data={ck.get('n_data')}  p0_type={ck.get('p0_type')}  "
        f"total_steps={ck.get('total_steps')}")
    solver = ODESolver(velocity_model=UNetVelocityWrapper(model))

    # ── Xác định 2 điểm neo ──
    c_star = compute_c_star(device)
    rep(f"\n[calibration] c* (pixel trung bình tập train, thang [-1,1]) = {c_star:.4f}")
    best_A0, best_rate, calib_log = calibrate_A0(
        solver, device, args.sample_steps, args.n_calib, args.eps_calib, args.seed)
    rep("[calibration] Quét A0 (B0=0, 2 vùng triangle hàng 1-3 & 10-12, cột 0-5):")
    for line in calib_log:
        rep(line)
    rep(f"[calibration] => chọn A0={best_A0:+.2f}  (double_col%={best_rate:.1f} tại eps_calib={args.eps_calib})")

    delta_point = torch.full((1, IN_CHANNELS, IMG_SIZE, IMG_SIZE), c_star)
    mij_point = torch.zeros(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    for y_sl, x_sl in GROUP_BOXES:
        mij_point[:, :, y_sl, x_sl] = best_A0

    eps_list = [float(e) for e in args.eps_list.split(",")]
    rep(f"\neps_list = {eps_list}")

    rep(f"\n{'='*90}\n[Δ] x0 = c*·1 + eps·N(0,I)  -> chỉ đếm hall_type='empty'\n{'='*90}")
    rows = sweep_region("delta", delta_point, "empty", solver, device, args.sample_steps,
                         args.n_sample_total, args.n_repeats, args.sample_batch_size,
                         eps_list, args.seed + 1, rep)

    rep(f"\n{'='*90}\n[M_ij] x0 = (0 ngoài vùng, A0 trong 2 vùng) + eps·N(0,I)  "
        f"-> chỉ đếm hall_type='double_col'\n{'='*90}")
    rows += sweep_region("mij", mij_point, "double_col", solver, device, args.sample_steps,
                          args.n_sample_total, args.n_repeats, args.sample_batch_size,
                          eps_list, args.seed + 2, rep)

    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "epsilon", "hall_type", "n_total_per_repeat",
                                           "n_repeats", "count_mean", "count_std", "rate_mean",
                                           "rate_std", "counts_list", "rates_list"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    rep(f"\nĐã lưu: {report_path}, {csv_path}")
    report.close()


if __name__ == "__main__":
    main()
