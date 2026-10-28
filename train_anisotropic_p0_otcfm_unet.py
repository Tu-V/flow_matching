"""
OT-CFM (Tong et al. 2023, minibatch optimal-transport coupling -- xem
verify_otcfm_hypothesis.py cho bản toy 16 chiều) áp vào ĐÚNG thí nghiệm isotropic
vs anisotropic p0 (train_anisotropic_p0_experiment.py), kiến trúc UNet, trên bộ
simple-shapes-10k-16x16 (10K ảnh, grayscale, KHÔNG dùng bản labeled/guided).

Coupling (cách ghép (x_0,x_1) trước khi tính CondOT path target velocity), CHẠY CẢ
2 để so sánh, cùng kiến trúc/data/total_steps:
    independent : x_0, x_1 rút độc lập mỗi batch (chuẩn CFM, mọi script trước đây).
    ot          : trong MỖI batch B mẫu x_0 ~ p0 (isotropic hoặc anisotropic) và
                  B mẫu x_1 (data), giải bài toán OT CHÍNH XÁC (cost=||x0_i-x1_j||^2,
                  B<->B cân bằng, uniform weights) bằng Hungarian
                  (scipy.optimize.linear_sum_assignment), ghép lại x_1 theo hoán vị
                  tối ưu đó. Kỳ vọng: ít đường (x_0,x_1) "cắt nhau" trong minibatch
                  -> field mục tiêu ít mơ hồ hơn -> hallucination giảm, CẢ VỚI
                  isotropic p0 (không cần phá đối xứng p0 nữa).

Chạy 2x2 = 4 model (coupling x p0_type), mỗi model: train --total_steps step, rồi
sample --n_sample_total ảnh x --n_repeats lần, đếm double_col/empty bằng
hallucination_detector, in bảng so sánh 2x2.

CẢNH BÁO CHI PHÍ: 2 coupling x 2 p0_type x total_steps (mặc định 40,000) = 160,000
step + 4 x 5 x 100,000 ảnh sample. Chạy trên GPU.

Usage:
    python train_anisotropic_p0_otcfm_unet.py
    python train_anisotropic_p0_otcfm_unet.py --quick   # kiểm tra lỗi, cực nhanh
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath      # noqa: E402
from flow_matching.solver import ODESolver          # noqa: E402
from flow_matching.utils import ModelWrapper         # noqa: E402
from models.unet import UNetModel                    # noqa: E402
from hallucination_detector import analyze_batch, summarize   # noqa: E402

IMG_SIZE = 16
IN_CHANNELS = 1   # grayscale, giống train_anisotropic_p0_experiment.py (KHÔNG phải bản RGB labeled)
N_TRAIN = 10000

DATASET_ROOT = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "simple-datasets",
)
DATA_DIR = os.path.join(DATASET_ROOT, f"simple-shapes-10k-{IMG_SIZE}x{IMG_SIZE}")
OUTPUT_DIR = os.path.join(REPO_ROOT, "shape_exp", "unet_architecture", "anisotropic_p0_otcfm_output")


class ShapesDataset(Dataset):
    def __init__(self, root: str):
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".png")
        )
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("L"))


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


def make_sigma(epsilon: float, device) -> torch.Tensor:
    d = IN_CHANNELS * IMG_SIZE * IMG_SIZE
    idx = torch.arange(d, dtype=torch.float32, device=device)
    sigma = 1.0 + idx * epsilon
    return sigma.view(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


def sample_x0(B: int, sigma, device) -> torch.Tensor:
    z = torch.randn(B, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    return z if sigma is None else z * sigma


# ── Minibatch OT coupling: ghép lại x_1 theo hoán vị tối ưu (Hungarian, B<->B) ──
def ot_reorder(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    B = x0.shape[0]
    with torch.no_grad():
        cost = torch.cdist(x0.reshape(B, -1), x1.reshape(B, -1), p=2).pow(2).cpu().numpy()
    _, col_idx = linear_sum_assignment(cost)
    return x1[col_idx]


def train_one(all_imgs: torch.Tensor, p0_type: str, coupling: str, sigma, args, device) -> UNetModel:
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1)

    n = all_imgs.shape[0]
    print(f"\n{'='*70}\np0={p0_type}  coupling={coupling}  params={n_params:,}  "
          f"total_steps={args.total_steps}  batch_size={args.batch_size}\n{'='*70}")

    model.train()
    loss_window = []
    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        x_1 = all_imgs[idx]
        B = x_1.shape[0]

        x_0 = sample_x0(B, sigma, device)
        if coupling == "ot":
            x_1 = ot_reorder(x_0, x_1)

        t = torch.rand(B, device=device)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        u_pred = model(path_sample.x_t, t, extra={})
        loss = torch.pow(u_pred - path_sample.dx_t, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_window.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(loss_window) / len(loss_window)
            print(f"  [p0={p0_type} coupling={coupling}] step {step:7d}/{args.total_steps}  "
                  f"loss={avg:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"elapsed={time.time()-t0:.0f}s")
            loss_window = []

    if args.save_ckpt:
        ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"unet_n{N_TRAIN}_{p0_type}_{coupling}.pt")
        torch.save({"model_state_dict": model.state_dict(), "p0_type": p0_type,
                    "coupling": coupling, "epsilon": args.epsilon,
                    "total_steps": args.total_steps}, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    model.eval()
    return model


@torch.no_grad()
def sample_and_analyze(model: UNetModel, sigma, device, n_total: int, steps: int, batch_size: int) -> dict:
    model.eval()
    solver = ODESolver(velocity_model=UNetVelocityWrapper(model))
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    all_analyses = []
    n_done = 0
    t0 = time.time()
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = sample_x0(B, sigma, device)
        x_final = solver.sample(x_init=x_init, step_size=None, method="euler",
                                 time_grid=time_grid, return_intermediates=False)
        x_01 = (x_final.clamp(-1, 1) + 1) / 2
        x_01_rgb = x_01.repeat(1, 3, 1, 1)
        imgs_np = (x_01_rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        all_analyses.extend(analyze_batch(imgs_np))
        n_done += B
        print(f"    sampled {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", end="\r")
    print()
    return summarize(all_analyses)


def sample_and_analyze_repeated(model, sigma, device, n_total, steps, batch_size, n_repeats) -> dict:
    runs = []
    for rep in range(1, n_repeats + 1):
        print(f"  -- repeat {rep}/{n_repeats} --")
        s = sample_and_analyze(model, sigma, device, n_total, steps, batch_size)
        runs.append(s)
        print(f"     hall={100*s['hall_rate']:.4f}%  (empty={100*s['n_empty']/s['n_total']:.4f}%  "
              f"double_col={100*s['n_double_col']/s['n_total']:.4f}%)")
    hall_rates = np.array([r["hall_rate"] for r in runs]) * 100.0
    empty_rates = np.array([r["n_empty"] / r["n_total"] for r in runs]) * 100.0
    double_rates = np.array([r["n_double_col"] / r["n_total"] for r in runs]) * 100.0
    return {
        "hall_rate_pct_list": hall_rates.tolist(), "empty_rate_pct_list": empty_rates.tolist(),
        "double_rate_pct_list": double_rates.tolist(),
        "hall_rate_mean": float(hall_rates.mean()),
        "hall_rate_std": float(hall_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "empty_rate_mean": float(empty_rates.mean()),
        "empty_rate_std": float(empty_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "double_rate_mean": float(double_rates.mean()),
        "double_rate_std": float(double_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser(description="OT-CFM vs independent coupling x isotropic/anisotropic p0, UNet, 10K ảnh")
    p.add_argument("--couplings", type=str, default="independent,ot")
    p.add_argument("--p0_types", type=str, default="isotropic,anisotropic")
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--total_steps", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--save_ckpt", action="store_true", default=True)
    p.add_argument("--no_save_ckpt", dest="save_ckpt", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="cấu hình siêu nhỏ để kiểm tra lỗi")
    a = p.parse_args()
    if a.quick:
        a.total_steps, a.batch_size, a.log_every = 20, 16, 5
        a.n_sample_total, a.n_repeats, a.sample_steps, a.sample_batch_size = 20, 1, 5, 20
    return a


def main():
    args = parse_args()
    couplings = [s.strip() for s in args.couplings.split(",")]
    p0_types = [s.strip() for s in args.p0_types.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}  img_size={IMG_SIZE}x{IMG_SIZE} grayscale (D={IN_CHANNELS*IMG_SIZE*IMG_SIZE})")
    print(f"couplings: {couplings}  p0_types: {p0_types}")

    if not os.path.isdir(DATA_DIR):
        print(f"LỖI: không thấy {DATA_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = ShapesDataset(DATA_DIR)
    print(f"Dataset: {len(dataset)} ảnh thật ({DATA_DIR})")
    all_imgs_cpu = torch.stack([dataset[i] for i in range(len(dataset))])
    all_imgs = all_imgs_cpu.to(device)
    sigma_aniso = make_sigma(args.epsilon, device)

    results = {}
    for coupling in couplings:
        for p0_type in p0_types:
            torch.manual_seed(args.seed)
            sigma = sigma_aniso if p0_type == "anisotropic" else None
            model = train_one(all_imgs, p0_type, coupling, sigma, args, device)

            print(f"\nSampling {args.n_sample_total} ảnh × {args.n_repeats} lần lặp "
                  f"(p0={p0_type}, coupling={coupling}) ...")
            s = sample_and_analyze_repeated(model, sigma, device, args.n_sample_total,
                                             args.sample_steps, args.sample_batch_size, args.n_repeats)
            print(f"  p0={p0_type:12s} coupling={coupling:11s}  "
                  f"hall={s['hall_rate_mean']:.3f}%±{s['hall_rate_std']:.3f}%  "
                  f"(empty={s['empty_rate_mean']:.3f}%±{s['empty_rate_std']:.3f}%  "
                  f"double_col={s['double_rate_mean']:.3f}%±{s['double_rate_std']:.3f}%)")
            results[(coupling, p0_type)] = s

            run_dir = os.path.join(OUTPUT_DIR, f"n{N_TRAIN}_{p0_type}_{coupling}")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "stats.txt"), "w") as f:
                f.write(f"n_data={N_TRAIN}  p0_type={p0_type}  coupling={coupling}  "
                        f"img_size={IMG_SIZE}  epsilon={args.epsilon}\n")
                f.write(f"total_steps={args.total_steps}  n_sample_total/repeat={args.n_sample_total}  "
                        f"n_repeats={args.n_repeats}  sample_steps={args.sample_steps}\n\n")
                for rep, hr in enumerate(s["hall_rate_pct_list"], start=1):
                    f.write(f"  repeat {rep}: hall={hr:.4f}%  empty={s['empty_rate_pct_list'][rep-1]:.4f}%  "
                            f"double_col={s['double_rate_pct_list'][rep-1]:.4f}%\n")
                f.write(f"\nHallucination : mean={s['hall_rate_mean']:.4f}%  std={s['hall_rate_std']:.4f}%\n")
                f.write(f"  empty img   : mean={s['empty_rate_mean']:.4f}%  std={s['empty_rate_std']:.4f}%\n")
                f.write(f"  double col  : mean={s['double_rate_mean']:.4f}%  std={s['double_rate_std']:.4f}%\n")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\n{'='*90}\nTỔNG HỢP (UNet OT-CFM, n={N_TRAIN})\n{'='*90}")
    header = f"{'coupling':>11s} {'p0_type':>12s} {'hall% mean':>11s} {'hall% std':>10s} {'empty% mean':>12s} {'double_col% mean':>17s}"
    print(header)
    lines = [header]
    for coupling in couplings:
        for p0_type in p0_types:
            r = results[(coupling, p0_type)]
            line = (f"{coupling:>11s} {p0_type:>12s} {r['hall_rate_mean']:11.4f} {r['hall_rate_std']:10.4f} "
                    f"{r['empty_rate_mean']:12.4f} {r['double_rate_mean']:17.4f}")
            print(line)
            lines.append(line)

    lines.append("")
    for p0_type in p0_types:
        if ("independent", p0_type) in results and ("ot", p0_type) in results:
            ind, ot = results[("independent", p0_type)], results[("ot", p0_type)]
            rel = ("(giảm về 0)" if ot["hall_rate_mean"] == 0 else
                   f"({100*(ot['hall_rate_mean']-ind['hall_rate_mean'])/max(ind['hall_rate_mean'],1e-12):+.1f}% tương đối)")
            line = (f"p0={p0_type}: independent={ind['hall_rate_mean']:.4f}%±{ind['hall_rate_std']:.4f}%  ->  "
                    f"ot={ot['hall_rate_mean']:.4f}%±{ot['hall_rate_std']:.4f}%  {rel}")
            print(line)
            lines.append(line)

    summary_path = os.path.join(OUTPUT_DIR, "summary_otcfm_unet.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
