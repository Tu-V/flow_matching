"""
Kiểm chứng lại giả thuyết isotropic-p0 (xem train_anisotropic_p0_experiment.py) nhưng
ĐỔI KIẾN TRÚC MẠNG từ UNet sang DiT (Diffusion Transformer, Peebles & Xie 2023) — tái sử
dụng ĐÚNG class DiT/DiTVelocityWrapper đã có sẵn trong train_shapes_fm_dit.py (patchify +
2D sin-cos positional embedding cố định + self-attention toàn cục + adaLN-Zero condition
theo t), KHÔNG viết lại. DiT không có translation-equivariance prior như conv (UNet) —
mỗi token biết vị trí tuyệt đối của nó qua positional embedding tường minh — nên đây cũng
là 1 cách tách bạch: hallucination giảm nhờ phá đối xứng p0 có còn đúng khi đổi hẳn kiến
trúc/prior không, hay là hiệu ứng riêng của UNet.

    p0 = N(0, diag(sigma_1^2, ..., sigma_D^2)),   sigma_i = 1 + i*epsilon,
    i = 0..D-1,  D = img_size*img_size (ảnh grayscale 1 kênh, flatten row-major H,W)
    isotropic   : x_0 ~ N(0, I)                                 (baseline chuẩn CFM)
    anisotropic : x_0 ~ N(0, diag(sigma_i^2)), sigma_i=1+i*epsilon  (phá đối xứng)

Hỗ trợ 2 độ phân giải --img_size 16 hoặc 64 (dataset 5k/10k đã có sẵn ở cả 2 độ phân
giải — 64x64 sinh bằng gen_simple_shapes_64x64.py). DiT dùng patch_size=2 mặc định:
  - 16x16 -> grid 8x8  = 64 token   (hidden=256, depth=6, heads=4 -> ~7.4M param)
  - 64x64 -> grid 32x32=1024 token  (attention O(N^2) ~ 1024^2, NẶNG hơn nhiều so
    với 16x16 — tăng --patch_size lên 4 (grid 16x16=256 token, ~7.4M param) nếu
    chậm/OOM, hoặc giảm --batch_size/--sample_batch_size).
hallucination_detector không cần sửa (đã verify ở 64x64, xem
train_anisotropic_p0_experiment.py). Ảnh 1 kênh grayscale (IN_CHANNELS=1), replicate
3 kênh CHỈ lúc gọi analyze_batch (hàm dùng chung không đổi).

Checkpoint lưu ĐỊNH KỲ trong lúc train (--save_every, mặc định 10,000 step, tên có
step) + checkpoint cuối cùng — tất cả trong anisotropic_p0_dit_output/checkpoints/.

Sau khi train xong mỗi (size, p0_type): sample --n_sample_total (mặc định 100,000)
ảnh × --n_repeats lần (mặc định 5, x_init random khác nhau mỗi lần) — tính mean/std
hallucination rate, không chỉ 1 con số. In + lưu bảng so sánh isotropic vs anisotropic.

CẢNH BÁO CHI PHÍ: mặc định 2 cỡ (5k,10k) × 2 p0 × total_steps (mặc định 40,000) =
160,000 gradient step + 4 × 5 × 100,000 ảnh sample. Chạy trên GPU.

Usage:
    python train_anisotropic_p0_experiment_dit.py
    python train_anisotropic_p0_experiment_dit.py --img_size 64 --patch_size 4
    python train_anisotropic_p0_experiment_dit.py --img_size 64 --epsilon 6.24e-05
    python train_anisotropic_p0_experiment_dit.py --hidden_size 512 --depth 10 --num_heads 8  # DiT lớn
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
from hallucination_detector import analyze_batch, summarize   # noqa: E402
from train_shapes_fm_dit import DiT, DiTVelocityWrapper        # noqa: E402  (tái dùng nguyên class DiT)

IN_CHANNELS = 1   # ảnh gốc vốn grayscale (PIL mode "L"), R=G=B luôn -> bỏ 3 kênh
                  # RGB dư thừa. hallucination_detector.py vẫn cần RGB 3 kênh nội bộ
                  # -> replicate lại ĐÚNG LÚC gọi analyze_batch, không đổi hàm chung.

DATASET_ROOT = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "simple-datasets",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "anisotropic_p0_dit_output")


def size_label(n: int) -> str:
    """5000 -> '5k', 10000 -> '10k', ... (khớp tên thư mục dataset có sẵn)."""
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


# ── Dataset (giống hệt train_anisotropic_p0_experiment.py) ────────────────────
class ShapesDataset(Dataset):
    def __init__(self, root: str, img_size: int):
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".png")
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("L"))


# ── Model: DiT (tái sử dụng class từ train_shapes_fm_dit.py) ───────────────────
def build_dit(img_size: int, args) -> DiT:
    assert img_size % args.patch_size == 0, \
        f"img_size={img_size} phải chia hết cho patch_size={args.patch_size}"
    return DiT(
        img_size=img_size, patch_size=args.patch_size, in_channels=IN_CHANNELS,
        hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )


# ── p0: isotropic vs anisotropic (sigma_i = 1 + i*epsilon) ───────────────────
def make_sigma(epsilon: float, img_size: int, device) -> torch.Tensor:
    """sigma_i = 1 + i*epsilon, i=0..D-1 (D=IN_CHANNELS*img_size*img_size, flatten
    H,W theo row-major vì IN_CHANNELS=1) -> [1,IN_CHANNELS,img_size,img_size]."""
    d = IN_CHANNELS * img_size * img_size
    idx = torch.arange(d, dtype=torch.float32, device=device)
    sigma = 1.0 + idx * epsilon
    return sigma.view(1, IN_CHANNELS, img_size, img_size)


def sample_x0(B: int, sigma, img_size: int, device) -> torch.Tensor:
    """sigma=None -> isotropic N(0,I). sigma=[1,IN_CHANNELS,H,W] -> N(0,diag(sigma^2))."""
    z = torch.randn(B, IN_CHANNELS, img_size, img_size, device=device)
    return z if sigma is None else z * sigma


def save_checkpoint(model: DiT, ckpt_path: str, args, **extra):
    torch.save({
        "model_state_dict": model.state_dict(), "arch": "dit",
        "hidden_size": args.hidden_size, "depth": args.depth,
        "num_heads": args.num_heads, "patch_size": args.patch_size,
        "mlp_ratio": args.mlp_ratio, **extra,
    }, ckpt_path)


# ── Train 1 model (n_data, p0_type) — theo STEP (không epoch) để so sánh công bằng
# giữa các cỡ dataset khác nhau, cùng tổng compute ──────────────────────────────
def train_one(dataset: ShapesDataset, p0_type: str, sigma, args, device,
              img_size: int, ckpt_dir: str, n_size: int) -> DiT:
    model = build_dit(img_size, args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = model.x_embedder.num_patches
    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1
    )

    n = len(dataset)
    print(f"  preloading {n} images ({img_size}x{img_size}) to {device} ...")
    all_imgs = torch.stack([dataset[i] for i in range(n)]).to(device)   # [n,1,H,W]

    print(f"\n{'='*70}")
    print(f"n_data={n}  p0={p0_type}  img_size={img_size}  arch=DiT  "
          f"hidden={args.hidden_size} depth={args.depth} heads={args.num_heads} "
          f"patch={args.patch_size} tokens={n_tokens}  params={n_params:,}  "
          f"total_steps={args.total_steps}  batch_size={args.batch_size}")
    print(f"{'='*70}")

    model.train()
    loss_window = []
    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        x_1 = all_imgs[idx]
        B = x_1.shape[0]

        x_0 = sample_x0(B, sigma, img_size, device)
        t = torch.rand(B, device=device)

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        u_pred = model(path_sample.x_t, t)   # DiT.forward(x,t) — không có "extra" như UNetModel
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

        # ── Lưu checkpoint ĐỊNH KỲ trong lúc train (không chỉ lúc xong) ────────
        if args.save_ckpt and (step % args.save_every == 0) and step != args.total_steps:
            ckpt_path = os.path.join(ckpt_dir, f"dit_n{n_size}_{p0_type}_img{img_size}_step{step:06d}.pt")
            save_checkpoint(model, ckpt_path, args, n_data=n_size, p0_type=p0_type,
                             img_size=img_size, epsilon=args.epsilon,
                             step=step, total_steps=args.total_steps, loss=loss.item())
            print(f"  [checkpoint] Saved: {ckpt_path}")

    del all_imgs
    return model


# ── Sample n_total ảnh (1 lần) + đếm hallucination ─────────────────────────────
@torch.no_grad()
def sample_and_analyze(model: DiT, sigma, device, n_total: int, steps: int,
                        batch_size: int, img_size: int) -> dict:
    model.eval()
    wrapper = DiTVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    all_analyses = []
    n_done = 0
    t0 = time.time()
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = sample_x0(B, sigma, img_size, device)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        x_01 = (x_final.clamp(-1, 1) + 1) / 2
        # hallucination_detector.analyze_batch cần (B,H,W,3) RGB uint8 — replicate
        # kênh grayscale (IN_CHANNELS=1) thành 3 kênh giống hệt nhau (R=G=B).
        x_01_rgb = x_01.repeat(1, 3, 1, 1) if x_01.shape[1] == 1 else x_01
        imgs_np = (x_01_rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        all_analyses.extend(analyze_batch(imgs_np))
        n_done += B
        print(f"    sampled {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", end="\r")
    print()
    return summarize(all_analyses)


# ── Lặp lại sampling n_repeats lần (mỗi lần n_total ảnh MỚI, x_init random khác
# nhau), tính mean/std của hallucination rate ──────────────────────────────────
def sample_and_analyze_repeated(model: DiT, sigma, device, n_total: int,
                                 steps: int, batch_size: int, n_repeats: int,
                                 img_size: int) -> dict:
    runs = []
    for rep in range(1, n_repeats + 1):
        print(f"  -- repeat {rep}/{n_repeats} --")
        s = sample_and_analyze(model, sigma, device, n_total, steps, batch_size, img_size)
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
    p = argparse.ArgumentParser(description="Isotropic vs anisotropic p0 với kiến trúc DiT")
    p.add_argument("--img_size", type=int, default=16,
                   help="Kich thuoc anh (16 hoac 64). Doi tuong dataset: "
                        "simple-shapes-{Nk}-{img_size}x{img_size}")
    p.add_argument("--dataset_sizes", type=str, default="5000,10000")
    p.add_argument("--p0_types", type=str, default="isotropic,anisotropic")
    p.add_argument("--epsilon", type=float, default=0.001,
                   help="sigma_i = 1 + i*epsilon, i=0..D-1 (D=img_size*img_size, 1 kenh). "
                        "default 0.001 -> voi img_size=16 (D=256), sigma trai dai [1.0, 1.255]")
    p.add_argument("--total_steps", type=int, default=40000,
                   help="Tong so gradient step MOI (size,p0) — CO DINH de so sanh cong bang")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=2000)
    # DiT hyperparams — mặc định "nhỏ" (~7.4M param), tương đương quy mô UNet
    # baseline (~9.1M) đã dùng ở train_anisotropic_p0_experiment.py, để so sánh
    # công bằng giữa 2 kiến trúc (không phải so sánh model to hơn).
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=2,
                   help="Phai chia het img_size. patch=2 o 64x64 -> 1024 token (nang) "
                        "-> tang len 4 (256 token) neu cham/OOM")
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--n_repeats", type=int, default=5,
                   help="So lan lap lai sampling n_sample_total anh MOI (x_init random khac "
                        "nhau moi lan) de tinh mean/std cua hallucination rate")
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--save_ckpt", action="store_true", default=True)
    p.add_argument("--no_save_ckpt", dest="save_ckpt", action="store_false")
    p.add_argument("--save_every", type=int, default=10000,
                   help="Luu checkpoint DINH KY moi save_every step trong luc train "
                        "(ngoai checkpoint cuoi cung luon duoc luu)")
    return p.parse_args()


def main():
    args = parse_args()
    sizes = [int(s) for s in args.dataset_sizes.split(",")]
    p0_types = [s.strip() for s in args.p0_types.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    d = IN_CHANNELS * args.img_size * args.img_size
    print(f"Device: {device}")
    print(f"Arch: DiT  hidden_size={args.hidden_size}  depth={args.depth}  "
          f"num_heads={args.num_heads}  patch_size={args.patch_size}")
    print(f"img_size: {args.img_size}x{args.img_size}  (D={d})")
    print(f"Dataset sizes: {sizes}")
    print(f"p0 types: {p0_types}")
    sigma_max = 1 + (d - 1) * args.epsilon
    print(f"epsilon: {args.epsilon}  (sigma_max = 1 + {d-1}*{args.epsilon} = {sigma_max:.4f})")
    if args.img_size != 16 and "anisotropic" in p0_types:
        d_ref16 = IN_CHANNELS * 16 * 16
        eps_equiv_16 = 0.001 * (d_ref16 - 1) / (d - 1)
        print(f"  LƯU Ý: sigma_max = 1+(D-1)*epsilon phụ thuộc D=IN_CHANNELS*img_size^2 -> cùng "
              f"epsilon ở img_size khác nhau cho độ méo p0 RẤT khác nhau (D ở đây={d}, so với "
              f"D={d_ref16} lúc img_size=16). Để có cùng sigma_max~1.255 như mặc định ở 16x16, "
              f"dùng --epsilon {eps_equiv_16:.2e}. sigma_max={sigma_max:.2f} hiện tại có thể quá "
              f"lớn (loss anisotropic sẽ cao bất thường, p0 bị méo quá mạnh) nếu không cố ý.")
    if args.img_size % args.patch_size != 0:
        raise ValueError(f"img_size={args.img_size} phải chia hết cho patch_size={args.patch_size}")

    n_tokens = (args.img_size // args.patch_size) ** 2
    print(f"n_tokens (grid {args.img_size//args.patch_size}x{args.img_size//args.patch_size}): {n_tokens}")
    if n_tokens > 256:
        approx_gib = args.batch_size * args.num_heads * n_tokens * n_tokens * 4 / (1024**3)
        print(f"  CẢNH BÁO OOM: self-attention là O(n_tokens^2) -> {n_tokens} token/ảnh với "
              f"batch_size={args.batch_size} có thể tốn RẤT NHIỀU VRAM (riêng 1 attention "
              f"matrix ~{approx_gib:.1f} GiB, nhân thêm nhiều lớp trung gian cần giữ lại cho "
              f"backward -> dễ CUDA OOM). Nếu OOM, thử 1 trong các cách sau:\n"
              f"    (a) tăng --patch_size (vd 4 -> {(args.img_size//4)**2} token, giảm ~"
              f"{(n_tokens/((args.img_size//4)**2))**2:.0f}x VRAM attention vì O(n^2)) — khuyến nghị trước tiên\n"
              f"    (b) giảm --batch_size (vd còn {max(1, args.batch_size//8)})\n"
              f"    (c) giảm --sample_batch_size tương ứng lúc sampling")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    if args.save_ckpt:
        os.makedirs(ckpt_dir, exist_ok=True)

    sigma_aniso = make_sigma(args.epsilon, args.img_size, device)

    results = []   # list of dict: size, p0_type, stats
    for n_size in sizes:
        label = size_label(n_size)
        data_dir = os.path.join(DATASET_ROOT, f"simple-shapes-{label}-{args.img_size}x{args.img_size}")
        if not os.path.isdir(data_dir):
            print(f"BỎ QUA size={n_size}: không thấy thư mục {data_dir}")
            continue
        dataset = ShapesDataset(data_dir, args.img_size)
        print(f"\n### Dataset size={n_size} ({len(dataset)} ảnh thật) — {data_dir}")

        for p0_type in p0_types:
            sigma = sigma_aniso if p0_type == "anisotropic" else None

            model = train_one(dataset, p0_type, sigma, args, device,
                               args.img_size, ckpt_dir, n_size)

            if args.save_ckpt:
                ckpt_path = os.path.join(ckpt_dir, f"dit_n{n_size}_{p0_type}_img{args.img_size}.pt")
                save_checkpoint(model, ckpt_path, args, n_data=n_size, p0_type=p0_type,
                                 img_size=args.img_size, epsilon=args.epsilon,
                                 total_steps=args.total_steps, step=args.total_steps)
                print(f"  Saved (final): {ckpt_path}")

            print(f"\nSampling {args.n_sample_total} ảnh × {args.n_repeats} lần lặp "
                  f"(n={n_size}, p0={p0_type}, steps={args.sample_steps}) ...")
            s = sample_and_analyze_repeated(model, sigma, device, args.n_sample_total,
                                             args.sample_steps, args.sample_batch_size,
                                             args.n_repeats, args.img_size)

            print(f"  n={n_size:6d}  p0={p0_type:12s}  "
                  f"hall={s['hall_rate_mean']:.3f}% ± {s['hall_rate_std']:.3f}%  "
                  f"(empty={s['empty_rate_mean']:.3f}%±{s['empty_rate_std']:.3f}%  "
                  f"double_col={s['double_rate_mean']:.3f}%±{s['double_rate_std']:.3f}%)")

            results.append({"n_data": n_size, "p0_type": p0_type, **s})

            # lưu stats.txt riêng cho run này (per-repeat + mean/std)
            run_dir = os.path.join(OUTPUT_DIR, f"n{n_size}_{p0_type}_img{args.img_size}")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "stats.txt"), "w") as f:
                f.write(f"n_data={n_size}  p0_type={p0_type}  img_size={args.img_size}  "
                        f"arch=DiT(hidden={args.hidden_size},depth={args.depth},"
                        f"heads={args.num_heads},patch={args.patch_size})  "
                        f"epsilon={args.epsilon}\n")
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
    print(f"TỔNG HỢP (DiT)  (n_sample_total={args.n_sample_total}/lần × {args.n_repeats} lần lặp, "
          f"sample_steps={args.sample_steps}, total_train_steps={args.total_steps}/run, "
          f"epsilon={args.epsilon}, img_size={args.img_size})")
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

    summary_path = os.path.join(OUTPUT_DIR, f"summary_anisotropic_p0_dit_img{args.img_size}.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
