"""
Kiểm chứng giả thuyết: hallucination (double-col / empty) trong flow matching trên
simple-shapes sinh ra vì p0 = N(0, I) ĐẲNG HƯỚNG (isotropic) — nhiều đường thẳng
(x_0, x_1) độc lập ngẫu nhiên "cắt nhau" đối xứng trong không gian nguồn khiến
network học ra field mơ hồ gần các điểm cắt. Phá đối xứng bằng:

    p0 = N(0, diag(sigma_1^2, ..., sigma_D^2)),   sigma_i = 1 + i*epsilon,
    i = 0..D-1,  D = img_size*img_size (ảnh grayscale 1 kênh, flatten row-major H,W)

thì hallucination được kỳ vọng biến mất/giảm mạnh, vì mỗi toạ độ nguồn giờ có
"quy mô" riêng biệt, phá vỡ tính đối xứng hoán vị của N(0,I) từng gây ra các cấu
trúc mode-averaging đối xứng (tương tự cơ chế "fixed-point subspace" đã kiểm chứng
ở bài toán toy 16 chiều trong cùng repo này).

Hỗ trợ 2 độ phân giải: --img_size 16 (mặc định) hoặc 64 — kiến trúc UNet
(build_unet, ~9.1M) không đổi vì fully-convolutional (image_size không hard-code
trong models/unet.py), dùng lại y hệt cho cả 2 độ phân giải. hallucination_detector
tự động scale (col slice = width//3, min_area scale theo (width/16)^2) nên KHÔNG
cần sửa gì để dùng đúng ở 64x64 — đã verify bằng số (30/30 ảnh mẫu khớp ground-truth
meta_data.npz). Dataset 64x64 (5k, 10k) được sinh mới bằng
../neurips-2024-diffusion-model-hallucination/shapes/gen_simple_shapes_64x64.py
(scale 4x logic của gen_simple_shapes_16x16.py, cùng layout 3 cột).
LƯU Ý: attention_resolutions=(2,) nghĩa là attention chạy ở độ phân giải
img_size/2 — ở 64x64 là 32x32 (1024 token), nặng hơn nhiều so với 16x16 (8x8),
sample_batch_size/batch_size nên giảm nếu OOM.

Ảnh đầu vào vốn grayscale (vẽ bằng PIL mode "L", không có màu — R=G=B luôn khi
convert("RGB")), nên pipeline này train/sample TRỰC TIẾP trên tensor 1 kênh
(IN_CHANNELS=1, shape [B,1,H,W] thay vì [B,3,H,W] dư thừa) — UNet in/out_channels
đổi 3->1 theo. hallucination_detector.py (dùng chung toàn repo) vẫn giữ contract
cũ (nhận RGB 3 kênh, tự cv2.cvtColor RGB2GRAY bên trong) nên KHÔNG bị đổi — ở
đây chỉ replicate kênh grayscale thành 3 kênh giống hệt nhau NGAY TRƯỚC khi gọi
analyze_batch, không ảnh hưởng script nào khác dùng chung hàm đó.

Với MỖI cỡ dataset trong --dataset_sizes (mặc định 5000,10000,20000,50000), train
2 model CÙNG kiến trúc UNet (~9.1M, giống hệt train_shapes_fm.py) / CÙNG
--total_steps / CÙNG batch_size / CÙNG data — chỉ khác p0:
    isotropic   : x_0 ~ N(0, I)                                 (baseline chuẩn CFM)
    anisotropic : x_0 ~ N(0, diag(sigma_i^2)), sigma_i=1+i*epsilon  (phá đối xứng)

Checkpoint được lưu ĐỊNH KỲ trong lúc train (mỗi --save_every step, mặc định
10,000 — file có step trong tên, không ghi đè) VÀ checkpoint cuối cùng (file
không có step, dùng để sample/tiếp tục phân tích) — tất cả trong
anisotropic_p0_output/checkpoints/.

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
    # 64x64, chỉ 5k/10k (dataset đã sinh sẵn), lưu checkpoint mỗi 5000 step:
    python train_anisotropic_p0_experiment.py --img_size 64 --dataset_sizes 5000,10000 \\
        --save_every 5000 --batch_size 64 --sample_batch_size 256
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

IN_CHANNELS = 1   # ảnh gốc vốn grayscale (vẽ bằng PIL mode "L"), R=G=B luôn -> bỏ
                  # 3 kênh RGB dư thừa, train/sample trực tiếp trên tensor 1 kênh
                  # (H,W thay vì 3,H,W). hallucination_detector.py vẫn cần RGB 3
                  # kênh (cv2.cvtColor RGB2GRAY) nên sẽ replicate lại đúng lúc gọi
                  # analyze_batch, không đổi hàm dùng chung đó.

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


# ── Model (kiến trúc + wrapper giống hệt train_shapes_fm.py, chỉ đổi in/out
# channels 3->1 để khớp ảnh grayscale) ─────────────────────────────────────────
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


def save_checkpoint(model: UNetModel, ckpt_path: str, **extra):
    torch.save({"model_state_dict": model.state_dict(), **extra}, ckpt_path)


# ── Train 1 model (n_data, p0_type) — theo STEP (không epoch) để so sánh công bằng
# giữa các cỡ dataset khác nhau, cùng tổng compute ──────────────────────────────
def train_one(dataset: ShapesDataset, p0_type: str, sigma, args, device,
              img_size: int, ckpt_dir: str, n_size: int) -> UNetModel:
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1
    )

    n = len(dataset)
    print(f"  preloading {n} images ({img_size}x{img_size}) to {device} ...")
    all_imgs = torch.stack([dataset[i] for i in range(n)]).to(device)   # [n,1,H,W]

    print(f"\n{'='*70}")
    print(f"n_data={n}  p0={p0_type}  img_size={img_size}  params={n_params:,}  "
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

        # ── Lưu checkpoint ĐỊNH KỲ trong lúc train (không chỉ lúc xong) — an toàn
        # cho run dài trên GPU (crash/OOM giữa chừng không mất hết), và cho phép
        # theo dõi tiến trình / trace sớm nếu cần ──────────────────────────────
        if args.save_ckpt and (step % args.save_every == 0) and step != args.total_steps:
            ckpt_path = os.path.join(ckpt_dir, f"unet_n{n_size}_{p0_type}_step{step:06d}.pt")
            save_checkpoint(model, ckpt_path, n_data=n_size, p0_type=p0_type,
                             img_size=img_size, epsilon=args.epsilon,
                             step=step, total_steps=args.total_steps, loss=loss.item())
            print(f"  [checkpoint] Saved: {ckpt_path}")

    del all_imgs
    return model


# ── Sample n_total ảnh (1 lần) + đếm hallucination ─────────────────────────────
@torch.no_grad()
def sample_and_analyze(model: UNetModel, sigma, device, n_total: int, steps: int,
                        batch_size: int, img_size: int) -> dict:
    model.eval()
    wrapper = UNetVelocityWrapper(model)
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
        # hallucination_detector.analyze_batch cần (B,H,W,3) RGB uint8 (dùng
        # cv2.cvtColor RGB2GRAY nội bộ) — replicate kênh grayscale (IN_CHANNELS=1)
        # thành 3 kênh giống hệt nhau (R=G=B), không đổi hàm dùng chung đó.
        x_01_rgb = x_01.repeat(1, 3, 1, 1) if x_01.shape[1] == 1 else x_01
        imgs_np = (x_01_rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        all_analyses.extend(analyze_batch(imgs_np))
        n_done += B
        print(f"    sampled {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", end="\r")
    print()
    return summarize(all_analyses)


# ── Lặp lại sampling n_repeats lần (mỗi lần n_total ảnh MỚI, x_init random khác
# nhau), tính mean/std của hallucination rate — đo độ nhiễu (variance) của ước
# lượng do số mẫu hữu hạn, không phải chỉ 1 con số duy nhất ──────────────────────
def sample_and_analyze_repeated(model: UNetModel, sigma, device, n_total: int,
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
    p = argparse.ArgumentParser(description="Isotropic vs anisotropic p0: hallucination rate")
    p.add_argument("--img_size", type=int, default=16,
                   help="Kich thuoc anh (16 hoac 64). Doi tuong dataset: "
                        "simple-shapes-{Nk}-{img_size}x{img_size}")
    p.add_argument("--dataset_sizes", type=str, default="5000,10000,20000,50000")
    p.add_argument("--p0_types", type=str, default="isotropic,anisotropic")
    p.add_argument("--epsilon", type=float, default=0.001,
                   help="sigma_i = 1 + i*epsilon, i=0..D-1 (D=img_size*img_size, 1 kenh). "
                        "default 0.001 -> voi img_size=16 (D=256), sigma trai dai [1.0, 1.255]")
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
              f"D={d_ref16} lúc img_size=16). Để có cùng sigma_max~1.767 như mặc định ở 16x16, "
              f"dùng --epsilon {eps_equiv_16:.2e}. sigma_max={sigma_max:.2f} hiện tại có thể quá "
              f"lớn (loss anisotropic sẽ cao bất thường, p0 bị méo quá mạnh) nếu không cố ý.")

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
                ckpt_path = os.path.join(ckpt_dir, f"unet_n{n_size}_{p0_type}.pt")
                save_checkpoint(model, ckpt_path, n_data=n_size, p0_type=p0_type,
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
            run_dir = os.path.join(OUTPUT_DIR, f"n{n_size}_{p0_type}")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "stats.txt"), "w") as f:
                f.write(f"n_data={n_size}  p0_type={p0_type}  img_size={args.img_size}  "
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
