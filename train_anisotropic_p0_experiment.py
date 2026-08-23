"""
Kiểm chứng giả thuyết: hallucination (double-col / empty) trong flow matching trên
simple-shapes sinh ra vì p0 = N(0, I) ĐẲNG HƯỚNG (isotropic) — nhiều đường thẳng
(x_0, x_1) độc lập ngẫu nhiên "cắt nhau" đối xứng trong không gian nguồn khiến
network học ra field mơ hồ gần các điểm cắt. Phá đối xứng bằng:

    p0 = N(0, diag(sigma_1^2, ..., sigma_D^2)),   sigma_i = 1 + i*epsilon,
    i = 0..D-1,  D = 3*16*16 = 768 (thứ tự flatten kênh-major C,H,W)

thì hallucination được kỳ vọng biến mất/giảm mạnh, vì mỗi toạ độ nguồn giờ có
"quy mô" riêng biệt, phá vỡ tính đối xứng hoán vị của N(0,I) từng gây ra các cấu
trúc mode-averaging đối xứng (tương tự cơ chế "fixed-point subspace" đã kiểm chứng
ở bài toán toy 16 chiều trong cùng repo này).

Với MỖI cỡ dataset trong --dataset_sizes (mặc định 5000,10000,20000,50000), train
2 model CÙNG kiến trúc UNet (~9.1M, giống hệt train_shapes_fm.py) / CÙNG
--total_steps / CÙNG batch_size / CÙNG data — chỉ khác p0:
    isotropic   : x_0 ~ N(0, I)                                 (baseline chuẩn CFM)
    anisotropic : x_0 ~ N(0, diag(sigma_i^2)), sigma_i=1+i*epsilon  (phá đối xứng)

Sau khi train xong mỗi (size, p0_type): sample --n_sample_total (mặc định 100,000)
ảnh MỚI — x_init rút từ ĐÚNG p0 đã dùng lúc train — chạy hallucination_detector,
đếm double_col / empty / normal. In + lưu bảng tổng hợp so sánh isotropic vs
anisotropic qua từng cỡ dataset.

CẢNH BÁO CHI PHÍ: 4 cỡ × 2 p0 × total_steps (mặc định 40,000) = 320,000 gradient
step tổng + 8 × 100,000 ảnh sample (100 bước Euler/ảnh). Chạy trên GPU.

Usage:
    python train_anisotropic_p0_experiment.py
    python train_anisotropic_p0_experiment.py --dataset_sizes 5000,10000 --total_steps 20000
    python train_anisotropic_p0_experiment.py --epsilon 0.002 --p0_types isotropic,anisotropic
    python train_anisotropic_p0_experiment.py --p0_types anisotropic   # chỉ chạy nhánh phá đối xứng
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
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
IN_CHANNELS = 3
D = IN_CHANNELS * IMG_SIZE * IMG_SIZE   # 768

DATASET_ROOT = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "simple-datasets",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "anisotropic_p0_output")


def size_label(n: int) -> str:
    """5000 -> '5k', 10000 -> '10k', ... (khớp tên thư mục dataset có sẵn)."""
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


# ── Dataset ─────────────────────────────────────────────────────────────────
class ShapesDataset(Dataset):
    def __init__(self, root: str):
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".png")
        )
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


# ── Model (kiến trúc + wrapper giống hệt train_shapes_fm.py) ──────────────────
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


class UNetVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t, extra=extras)


# ── p0: isotropic vs anisotropic (sigma_i = 1 + i*epsilon) ───────────────────
def make_sigma(epsilon: float, device) -> torch.Tensor:
    """sigma_i = 1 + i*epsilon, i=0..D-1 (D=768=3*16*16, flatten C,H,W) -> [1,3,16,16]."""
    idx = torch.arange(D, dtype=torch.float32, device=device)
    sigma = 1.0 + idx * epsilon
    return sigma.view(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


def sample_x0(B: int, sigma, device) -> torch.Tensor:
    """sigma=None -> isotropic N(0,I). sigma=[1,3,16,16] -> N(0,diag(sigma^2))."""
    z = torch.randn(B, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    return z if sigma is None else z * sigma


# ── Train 1 model (n_data, p0_type) — theo STEP (không epoch) để so sánh công bằng
# giữa các cỡ dataset khác nhau, cùng tổng compute ──────────────────────────────
def train_one(dataset: ShapesDataset, p0_type: str, sigma, args, device) -> UNetModel:
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1
    )

    n = len(dataset)
    print(f"  preloading {n} images to {device} ...")
    all_imgs = torch.stack([dataset[i] for i in range(n)]).to(device)   # [n,3,16,16]

    print(f"\n{'='*70}")
    print(f"n_data={n}  p0={p0_type}  params={n_params:,}  total_steps={args.total_steps}  "
          f"batch_size={args.batch_size}")
    print(f"{'='*70}")

    model.train()
    loss_window = []
    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        x_1 = all_imgs[idx]
        B = x_1.shape[0]

        x_0 = sample_x0(B, sigma, device)
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
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [n={n} p0={p0_type}] step {step:7d}/{args.total_steps}  "
                  f"loss={avg:.5f}  lr={lr:.2e}  elapsed={time.time()-t0:.0f}s")
            loss_window = []

    del all_imgs
    return model


# ── Sample n_total ảnh (1 lần) + đếm hallucination ─────────────────────────────
@torch.no_grad()
def sample_and_analyze(model: UNetModel, sigma, device, n_total: int, steps: int,
                        batch_size: int) -> dict:
    model.eval()
    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    all_analyses = []
    n_done = 0
    t0 = time.time()
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = sample_x0(B, sigma, device)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        x_01 = (x_final.clamp(-1, 1) + 1) / 2
        imgs_np = (x_01.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        all_analyses.extend(analyze_batch(imgs_np))
        n_done += B
        print(f"    sampled {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", end="\r")
    print()
    return summarize(all_analyses)


# ── Lặp lại sampling n_repeats lần (mỗi lần n_total ảnh MỚI, x_init random khác
# nhau), tính mean/std của hallucination rate — đo độ nhiễu (variance) của ước
# lượng do số mẫu hữu hạn, không phải chỉ 1 con số duy nhất ──────────────────────
def sample_and_analyze_repeated(model: UNetModel, sigma, device, n_total: int,
                                 steps: int, batch_size: int, n_repeats: int) -> dict:
    runs = []
    for rep in range(1, n_repeats + 1):
        print(f"  -- repeat {rep}/{n_repeats} --")
        s = sample_and_analyze(model, sigma, device, n_total, steps, batch_size)
        runs.append(s)
        print(f"     hall={100*s['hall_rate']:.4f}%  "
              f"(empty={100*s['n_empty']/s['n_total']:.4f}%  "
              f"double_col={100*s['n_double_col']/s['n_total']:.4f}%)")

    hall_rates   = np.array([r["hall_rate"] for r in runs]) * 100.0
    empty_rates  = np.array([r["n_empty"] / r["n_total"] for r in runs]) * 100.0
    double_rates = np.array([r["n_double_col"] / r["n_total"] for r in runs]) * 100.0

    return {
        "n_repeats": n_repeats,
        "n_total_per_repeat": n_total,
        "runs": runs,
        "hall_rate_pct_list":   hall_rates.tolist(),
        "empty_rate_pct_list":  empty_rates.tolist(),
        "double_rate_pct_list": double_rates.tolist(),
        "hall_rate_mean":   float(hall_rates.mean()),
        "hall_rate_std":    float(hall_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "empty_rate_mean":  float(empty_rates.mean()),
        "empty_rate_std":   float(empty_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "double_rate_mean": float(double_rates.mean()),
        "double_rate_std":  float(double_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Isotropic vs anisotropic p0: hallucination rate")
    p.add_argument("--dataset_sizes", type=str, default="5000,10000,20000,50000")
    p.add_argument("--p0_types", type=str, default="isotropic,anisotropic")
    p.add_argument("--epsilon", type=float, default=0.001,
                   help="sigma_i = 1 + i*epsilon, i=0..767 (D=3*16*16). "
                        "default 0.001 -> sigma trai dai [1.0, 1.767]")
    p.add_argument("--total_steps", type=int, default=40000,
                   help="Tong so gradient step MOI (size,p0) — CO DINH de so sanh cong bang")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--n_repeats", type=int, default=5,
                   help="So lan lap lai sampling n_sample_total anh MOI (x_init random khac "
                        "nhau moi lan) de tinh mean/std cua hallucination rate")
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--save_ckpt", action="store_true", default=True)
    p.add_argument("--no_save_ckpt", dest="save_ckpt", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    sizes = [int(s) for s in args.dataset_sizes.split(",")]
    p0_types = [s.strip() for s in args.p0_types.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Dataset sizes: {sizes}")
    print(f"p0 types: {p0_types}")
    print(f"epsilon: {args.epsilon}  (sigma_max = 1 + {D-1}*{args.epsilon} = "
          f"{1 + (D-1)*args.epsilon:.4f})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    if args.save_ckpt:
        os.makedirs(ckpt_dir, exist_ok=True)

    sigma_aniso = make_sigma(args.epsilon, device)

    results = []   # list of dict: size, p0_type, stats
    for n_size in sizes:
        label = size_label(n_size)
        data_dir = os.path.join(DATASET_ROOT, f"simple-shapes-{label}-16x16")
        if not os.path.isdir(data_dir):
            print(f"BỎ QUA size={n_size}: không thấy thư mục {data_dir}")
            continue
        dataset = ShapesDataset(data_dir)
        print(f"\n### Dataset size={n_size} ({len(dataset)} ảnh thật) — {data_dir}")

        for p0_type in p0_types:
            sigma = sigma_aniso if p0_type == "anisotropic" else None

            model = train_one(dataset, p0_type, sigma, args, device)

            if args.save_ckpt:
                ckpt_path = os.path.join(ckpt_dir, f"unet_n{n_size}_{p0_type}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "n_data": n_size, "p0_type": p0_type, "epsilon": args.epsilon,
                    "total_steps": args.total_steps,
                }, ckpt_path)
                print(f"  Saved: {ckpt_path}")

            print(f"\nSampling {args.n_sample_total} ảnh × {args.n_repeats} lần lặp "
                  f"(n={n_size}, p0={p0_type}, steps={args.sample_steps}) ...")
            s = sample_and_analyze_repeated(model, sigma, device, args.n_sample_total,
                                             args.sample_steps, args.sample_batch_size,
                                             args.n_repeats)

            print(f"  n={n_size:6d}  p0={p0_type:12s}  "
                  f"hall={s['hall_rate_mean']:.3f}% ± {s['hall_rate_std']:.3f}%  "
                  f"(empty={s['empty_rate_mean']:.3f}%±{s['empty_rate_std']:.3f}%  "
                  f"double_col={s['double_rate_mean']:.3f}%±{s['double_rate_std']:.3f}%)")

            results.append({"n_data": n_size, "p0_type": p0_type, **s})

            # lưu stats.txt riêng cho run này (per-repeat + mean/std)
            run_dir = os.path.join(OUTPUT_DIR, f"n{n_size}_{p0_type}")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "stats.txt"), "w") as f:
                f.write(f"n_data={n_size}  p0_type={p0_type}  epsilon={args.epsilon}\n")
                f.write(f"total_steps={args.total_steps}  n_sample_total/repeat={args.n_sample_total}  "
                        f"n_repeats={args.n_repeats}  sample_steps={args.sample_steps}\n\n")
                f.write("Per-repeat hallucination rate (%):\n")
                for rep, hr in enumerate(s["hall_rate_pct_list"], start=1):
                    f.write(f"  repeat {rep}: hall={hr:.4f}%  "
                            f"empty={s['empty_rate_pct_list'][rep-1]:.4f}%  "
                            f"double_col={s['double_rate_pct_list'][rep-1]:.4f}%\n")
                f.write(f"\nHallucination : mean={s['hall_rate_mean']:.4f}%  std={s['hall_rate_std']:.4f}%\n")
                f.write(f"  empty img   : mean={s['empty_rate_mean']:.4f}%  std={s['empty_rate_std']:.4f}%\n")
                f.write(f"  double col  : mean={s['double_rate_mean']:.4f}%  std={s['double_rate_std']:.4f}%\n")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Bảng tổng hợp cuối ──
    print(f"\n{'='*80}")
    print(f"TỔNG HỢP  (n_sample_total={args.n_sample_total}/lần × {args.n_repeats} lần lặp, "
          f"sample_steps={args.sample_steps}, total_train_steps={args.total_steps}/run, "
          f"epsilon={args.epsilon})")
    print(f"{'='*80}")
    header = (f"{'n_data':>8s} {'p0_type':>12s} {'hall% mean':>11s} {'hall% std':>10s} "
              f"{'empty% mean':>12s} {'double_col% mean':>17s}")
    print(header)
    lines = [header]
    for r in results:
        line = (f"{r['n_data']:8d} {r['p0_type']:>12s} {r['hall_rate_mean']:11.4f} "
                f"{r['hall_rate_std']:10.4f} {r['empty_rate_mean']:12.4f} "
                f"{r['double_rate_mean']:17.4f}")
        print(line)
        lines.append(line)

    # so sánh trực tiếp isotropic vs anisotropic (nếu cả 2 đều chạy) theo từng size
    if "isotropic" in p0_types and "anisotropic" in p0_types:
        lines.append("")
        print()
        for n_size in sizes:
            iso = next((r for r in results if r["n_data"] == n_size and r["p0_type"] == "isotropic"), None)
            ani = next((r for r in results if r["n_data"] == n_size and r["p0_type"] == "anisotropic"), None)
            if iso and ani:
                rel = ("(giảm về 0)" if ani["hall_rate_mean"] == 0 else
                       f"({100*(ani['hall_rate_mean']-iso['hall_rate_mean'])/max(iso['hall_rate_mean'],1e-12):+.1f}% tương đối)")
                line = (f"n={n_size}: isotropic={iso['hall_rate_mean']:.4f}%±{iso['hall_rate_std']:.4f}%  ->  "
                        f"anisotropic={ani['hall_rate_mean']:.4f}%±{ani['hall_rate_std']:.4f}%  {rel}")
                print(line)
                lines.append(line)

    summary_path = os.path.join(OUTPUT_DIR, "summary_anisotropic_p0.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
