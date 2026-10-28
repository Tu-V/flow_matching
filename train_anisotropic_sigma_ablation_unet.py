"""
Ablation: quét epsilon (-> sigma_i = 1 + i*epsilon, xem train_anisotropic_p0_experiment.py)
trên UNet, bộ simple-shapes-10k-16x16 (10K ảnh, grayscale, train nhanh ở 16x16).
epsilon=0 chính là isotropic N(0,I) (sigma_i=1 với mọi i) -- KHÔNG cần tách riêng
"p0_types" như script gốc, chỉ 1 trục duy nhất: epsilon.

Mỗi epsilon trong --epsilon_list: train 1 model (--total_steps step), sample
--n_sample_total ảnh x --n_repeats lần, đếm double_col/empty bằng
hallucination_detector. Cuối cùng in bảng epsilon -> hall%, chỉ ra epsilon nào cho
hallucination THẤP NHẤT (tốt nhất).

CẢNH BÁO CHI PHÍ: len(--epsilon_list) x total_steps (mặc định 40,000) step + đó
nhiêu lần x 5 x 100,000 ảnh sample. Chạy trên GPU.

Usage:
    python train_anisotropic_sigma_ablation_unet.py
    python train_anisotropic_sigma_ablation_unet.py --epsilon_list 0,0.001,0.01
    python train_anisotropic_sigma_ablation_unet.py --quick   # kiểm tra lỗi, cực nhanh
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
IN_CHANNELS = 1

DATASET_ROOT = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "simple-datasets",
)
OUTPUT_DIR_BASE = os.path.join(REPO_ROOT, "shape_exp", "unet_architecture", "anisotropic_sigma_ablation_output")


def size_label(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


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
    """sigma_i = 1 + i*epsilon. epsilon=0 -> sigma toàn 1 (= isotropic N(0,I))."""
    d = IN_CHANNELS * IMG_SIZE * IMG_SIZE
    idx = torch.arange(d, dtype=torch.float32, device=device)
    sigma = 1.0 + idx * epsilon
    return sigma.view(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


def sample_x0(B: int, sigma: torch.Tensor, device) -> torch.Tensor:
    return torch.randn(B, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device) * sigma


def train_one(all_imgs: torch.Tensor, epsilon: float, sigma, args, device, n_train: int, output_dir: str) -> UNetModel:
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1)

    n = all_imgs.shape[0]
    sigma_max = float(sigma.max().item())
    print(f"\n{'='*70}\nepsilon={epsilon:g}  sigma_max={sigma_max:.4f}  params={n_params:,}  "
          f"total_steps={args.total_steps}  batch_size={args.batch_size}\n{'='*70}")

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
            print(f"  [eps={epsilon:g}] step {step:7d}/{args.total_steps}  loss={avg:.5f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  elapsed={time.time()-t0:.0f}s")
            loss_window = []

    if args.save_ckpt:
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"unet_n{n_train}_eps{epsilon:g}.pt")
        torch.save({"model_state_dict": model.state_dict(), "epsilon": epsilon,
                    "sigma_max": sigma_max, "total_steps": args.total_steps}, ckpt_path)
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
    p = argparse.ArgumentParser(description="Ablation: sigma (epsilon) nào cho hallucination thấp nhất, UNet")
    p.add_argument("--n_train", type=int, default=10000, help="5000 hoặc 10000 (dataset đã có sẵn ở 2 cỡ này)")
    p.add_argument("--epsilon_list", type=str, default="0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1")
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
        a.epsilon_list = "0,0.001,0.1"
        a.total_steps, a.batch_size, a.log_every = 20, 16, 5
        a.n_sample_total, a.n_repeats, a.sample_steps, a.sample_batch_size = 20, 1, 5, 20
    return a


def main():
    args = parse_args()
    n_train = args.n_train
    output_dir = OUTPUT_DIR_BASE if n_train == 10000 else f"{OUTPUT_DIR_BASE}_n{n_train}"
    data_dir = os.path.join(DATASET_ROOT, f"simple-shapes-{size_label(n_train)}-{IMG_SIZE}x{IMG_SIZE}")
    epsilons = [float(e) for e in args.epsilon_list.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}  img_size={IMG_SIZE}x{IMG_SIZE} grayscale (D={IN_CHANNELS*IMG_SIZE*IMG_SIZE})  n_train={n_train}")
    print(f"epsilon_list: {epsilons}")
    print(f"output_dir: {output_dir}")

    if not os.path.isdir(data_dir):
        print(f"LỖI: không thấy {data_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    dataset = ShapesDataset(data_dir)
    print(f"Dataset: {len(dataset)} ảnh thật ({data_dir})")
    all_imgs = torch.stack([dataset[i] for i in range(len(dataset))]).to(device)

    results = []
    for epsilon in epsilons:
        torch.manual_seed(args.seed)
        sigma = make_sigma(epsilon, device)
        model = train_one(all_imgs, epsilon, sigma, args, device, n_train, output_dir)

        print(f"\nSampling {args.n_sample_total} ảnh × {args.n_repeats} lần lặp (epsilon={epsilon:g}) ...")
        s = sample_and_analyze_repeated(model, sigma, device, args.n_sample_total,
                                         args.sample_steps, args.sample_batch_size, args.n_repeats)
        print(f"  epsilon={epsilon:<10g}  hall={s['hall_rate_mean']:.3f}%±{s['hall_rate_std']:.3f}%  "
              f"(empty={s['empty_rate_mean']:.3f}%±{s['empty_rate_std']:.3f}%  "
              f"double_col={s['double_rate_mean']:.3f}%±{s['double_rate_std']:.3f}%)")
        results.append({"epsilon": epsilon, "sigma_max": float(sigma.max().item()), **s})

        run_dir = os.path.join(output_dir, f"n{n_train}_eps{epsilon:g}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "stats.txt"), "w") as f:
            f.write(f"n_data={n_train}  epsilon={epsilon:g}  sigma_max={sigma.max().item():.4f}  "
                    f"img_size={IMG_SIZE}\n")
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

    print(f"\n{'='*90}\nTỔNG HỢP ABLATION epsilon/sigma (UNet, n={n_train})\n{'='*90}")
    header = f"{'epsilon':>10s} {'sigma_max':>10s} {'hall% mean':>11s} {'hall% std':>10s} {'empty% mean':>12s} {'double_col% mean':>17s}"
    print(header)
    lines = [header]
    for r in results:
        line = (f"{r['epsilon']:10g} {r['sigma_max']:10.4f} {r['hall_rate_mean']:11.4f} "
                f"{r['hall_rate_std']:10.4f} {r['empty_rate_mean']:12.4f} {r['double_rate_mean']:17.4f}")
        print(line)
        lines.append(line)

    best = min(results, key=lambda r: r["hall_rate_mean"])
    verdict = (f"\n=> epsilon TỐT NHẤT (hall% thấp nhất) = {best['epsilon']:g}  "
               f"(sigma_max={best['sigma_max']:.4f})  hall%={best['hall_rate_mean']:.4f}%±{best['hall_rate_std']:.4f}%")
    print(verdict)
    lines.append(verdict)

    summary_path = os.path.join(output_dir, "summary_sigma_ablation_unet.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
