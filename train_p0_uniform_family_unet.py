"""
So sánh 2 phân phối p0 KHÔNG PHẢI Gaussian, KIẾN TRÚC UNET — bản tách riêng từ
train_p0_uniform_family_experiment.py để chạy song song trên máy khác với bản DiT
(train_p0_uniform_family_dit.py); mỗi script tự chạy độc lập, không cần --archs:

    (1) uniform01  : x_0[i] ~ Uniform(0, 1) i.i.d MỌI chiều — cùng 1 phân phối cho
                     mọi chiều (đồng nhất, nhưng KHÔNG có tâm đối xứng/bất biến quay
                     như N(0,1); "baseline" của thí nghiệm này).
    (2) uniform_i2 : x_0[i] ~ Uniform((i/D)^2, ((i+1)/D)^2) ĐỘC LẬP theo từng
                     chiều i=0..D-1 (D=img_size*img_size, flatten row-major H,W,
                     ảnh grayscale 1 kênh) — CHUẨN HOÁ theo D^2 so với công thức
                     thô U[i^2,(i+1)^2] (D^2 lớn tới ~16.8 triệu ở 64x64 -> tràn
                     số/loss vô nghĩa nếu không chuẩn hoá). Vẫn giữ đúng hình dạng
                     tăng bậc 2 tương đối giữa các chiều (chiều đầu bin cực hẹp
                     quanh 0, chiều cuối bin rộng nhất) nhưng toàn bộ giá trị nén
                     về [0,1] — cùng thang đo với uniform01, không phụ thuộc D.

TÁI DÙNG nguyên build_unet/UNetVelocityWrapper/ShapesDataset từ
train_anisotropic_p0_experiment.py, không viết lại. Cả 2 độ phân giải (--img_sizes
16,64) và cả 2 cỡ dataset (--dataset_sizes 5000,10000) — mặc định chạy HẾT
2(p0)×2(img_size)×2(size) = 8 tổ hợp trong 1 lần gọi. Dataset chỉ load 1 lần cho
mỗi (img_size, size), tái dùng cho cả 2 p0 (không load lại 8 lần).

hallucination_detector không cần sửa (đã verify đúng ở cả 16x16 và 64x64 trong thí
nghiệm trước). Ảnh 1 kênh grayscale — replicate 3 kênh CHỈ lúc gọi analyze_batch.

Checkpoint lưu ĐỊNH KỲ (--save_every) + cuối cùng, tên có img_size+p0+size để KHÔNG
đụng nhau giữa các tổ hợp. Tự động RESUME nếu đã có checkpoint (--no_resume để tắt)
— bỏ qua hoàn toàn nếu đã train đủ total_steps.

--total_steps hỗ trợ RIÊNG theo img_size — mặc định "16:40000,64:100000" (16x16
train 40,000 step, 64x64 train 100,000 step/tổ hợp vì ảnh to hơn cần nhiều step
hơn); truyền 1 số duy nhất (vd "40000") để áp dụng đều cho mọi img_size.

CẢNH BÁO CHI PHÍ: mặc định 4 tổ hợp×40,000 (16x16) + 4 tổ hợp×100,000 (64x64) =
560,000 gradient step tổng + 8 × 100,000 ảnh × n_repeats (5) sample. Dùng
--img_sizes/--dataset_sizes/--p0_types để chạy 1 phần trước khi chạy full.

Usage (máy này chỉ chạy UNet; chạy song song train_p0_uniform_family_dit.py trên
máy còn lại):
    python train_p0_uniform_family_unet.py
    python train_p0_uniform_family_unet.py --img_sizes 16
    python train_p0_uniform_family_unet.py --total_steps 16:40000,64:100000
    python train_p0_uniform_family_unet.py --dataset_sizes 5000 --total_steps 5000  # test nhanh (moi size)
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath      # noqa: E402
from flow_matching.solver import ODESolver          # noqa: E402
from hallucination_detector import analyze_batch, summarize   # noqa: E402

from train_anisotropic_p0_experiment import (        # noqa: E402  (tái dùng, không viết lại)
    IN_CHANNELS, ShapesDataset, build_unet, UNetVelocityWrapper, size_label,
)

DATASET_ROOT = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "simple-datasets",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "p0_uniform_family_unet_output")


# ── p0 #1: Uniform(0,1) i.i.d mọi chiều ────────────────────────────────────────
def sample_x0_uniform01(B: int, img_size: int, device) -> torch.Tensor:
    return torch.rand(B, IN_CHANNELS, img_size, img_size, device=device)


# ── p0 #2: Uniform(i^2, (i+1)^2) độc lập theo từng chiều i=0..D-1, CHUẨN HOÁ
# theo D^2 để không tràn số ─────────────────────────────────────────────────
def make_uniform_i2_bounds(img_size: int, device):
    d = IN_CHANNELS * img_size * img_size
    idx = torch.arange(d, dtype=torch.float32, device=device)
    lo = ((idx / d) ** 2).view(1, IN_CHANNELS, img_size, img_size)
    hi = (((idx + 1) / d) ** 2).view(1, IN_CHANNELS, img_size, img_size)
    return lo, hi


def sample_x0_uniform_i2(B: int, lo: torch.Tensor, hi: torch.Tensor, device) -> torch.Tensor:
    u = torch.rand(B, *lo.shape[1:], device=device)
    return lo + u * (hi - lo)


def sample_x0(B: int, p0_type: str, bounds, img_size: int, device) -> torch.Tensor:
    if p0_type == "uniform01":
        return sample_x0_uniform01(B, img_size, device)
    elif p0_type == "uniform_i2":
        lo, hi = bounds
        return sample_x0_uniform_i2(B, lo, hi, device)
    raise ValueError(f"p0_type không hợp lệ: {p0_type}")


# ── Checkpoint (resume-aware) ──────────────────────────────────────────────────
def ckpt_stub(n_size: int, p0_type: str, img_size: int) -> str:
    return f"unet_n{n_size}_{p0_type}_img{img_size}"


def save_checkpoint(model, ckpt_path: str, optimizer=None, scheduler=None, **extra):
    state = {"model_state_dict": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, ckpt_path)


def find_resume_checkpoint(ckpt_dir: str, stub: str):
    best_path, best_step = None, -1
    final_path = os.path.join(ckpt_dir, f"{stub}.pt")
    if os.path.isfile(final_path):
        ckpt = torch.load(final_path, map_location="cpu", weights_only=True)
        step = ckpt.get("step", ckpt.get("total_steps", 0))
        if step > best_step:
            best_step, best_path = step, final_path
    for p in glob.glob(os.path.join(ckpt_dir, f"{stub}_step*.pt")):
        m = re.search(r"_step(\d+)\.pt$", p)
        if m and int(m.group(1)) > best_step:
            best_step, best_path = int(m.group(1)), p
    return (best_path, best_step) if best_path is not None else None


# ── Train 1 model (n_data, p0_type, img_size) — theo STEP, resume-aware ────────
def train_one(dataset: ShapesDataset, p0_type: str, bounds, args, device,
              img_size: int, ckpt_dir: str, n_size: int):
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1
    )

    stub = ckpt_stub(n_size, p0_type, img_size)

    start_step = 1
    if args.resume and args.save_ckpt:
        found = find_resume_checkpoint(ckpt_dir, stub)
        if found is not None:
            ckpt_path, ckpt_step = found
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_step = ckpt_step + 1
            print(f"  [resume] tìm thấy {os.path.basename(ckpt_path)} (step={ckpt_step}) "
                  f"-> tiếp tục từ step {start_step}")
            if start_step > args.total_steps:
                print(f"  [resume] đã train đủ {args.total_steps} step rồi -> BỎ QUA training.")
                return model

    n = len(dataset)
    print(f"  preloading {n} images ({img_size}x{img_size}) to {device} ...")
    all_imgs = torch.stack([dataset[i] for i in range(n)]).to(device)   # [n,1,H,W]

    print(f"\n{'='*70}")
    print(f"arch=unet  n_data={n}  p0={p0_type}  img_size={img_size}  params={n_params:,}  "
          f"total_steps={args.total_steps}  batch_size={args.batch_size}  start_step={start_step}")
    print(f"{'='*70}")

    model.train()
    loss_window = []
    t0 = time.time()
    for step in range(start_step, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        x_1 = all_imgs[idx]
        B = x_1.shape[0]

        x_0 = sample_x0(B, p0_type, bounds, img_size, device)
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
            print(f"  [unet n={n} p0={p0_type} img={img_size}] step {step:7d}/{args.total_steps}  "
                  f"loss={avg:.5g}  lr={lr:.2e}  elapsed={time.time()-t0:.0f}s")
            loss_window = []

        if args.save_ckpt and (step % args.save_every == 0) and step != args.total_steps:
            ckpt_path = os.path.join(ckpt_dir, f"{stub}_step{step:06d}.pt")
            save_checkpoint(model, ckpt_path, optimizer=optimizer, scheduler=scheduler,
                             n_data=n_size, p0_type=p0_type, img_size=img_size,
                             step=step, total_steps=args.total_steps, loss=loss.item())
            print(f"  [checkpoint] Saved: {ckpt_path}")

    if args.save_ckpt:
        final_path = os.path.join(ckpt_dir, f"{stub}.pt")
        save_checkpoint(model, final_path, optimizer=optimizer, scheduler=scheduler,
                         n_data=n_size, p0_type=p0_type, img_size=img_size,
                         step=args.total_steps, total_steps=args.total_steps)
        print(f"  Saved (final): {final_path}")

    del all_imgs
    return model


# ── Sample n_total ảnh (1 lần) + đếm hallucination ─────────────────────────────
@torch.no_grad()
def sample_and_analyze(model, p0_type: str, bounds, device, n_total: int,
                        steps: int, batch_size: int, img_size: int) -> dict:
    model.eval()
    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    all_analyses = []
    n_done = 0
    t0 = time.time()
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = sample_x0(B, p0_type, bounds, img_size, device)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        x_01 = (x_final.clamp(-1, 1) + 1) / 2
        x_01_rgb = x_01.repeat(1, 3, 1, 1) if x_01.shape[1] == 1 else x_01
        imgs_np = (x_01_rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        all_analyses.extend(analyze_batch(imgs_np))
        n_done += B
        print(f"    sampled {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", end="\r")
    print()
    return summarize(all_analyses)


def sample_and_analyze_repeated(model, p0_type: str, bounds, device, n_total: int,
                                 steps: int, batch_size: int, n_repeats: int, img_size: int) -> dict:
    runs = []
    for rep in range(1, n_repeats + 1):
        print(f"  -- repeat {rep}/{n_repeats} --")
        s = sample_and_analyze(model, p0_type, bounds, device, n_total, steps, batch_size, img_size)
        runs.append(s)
        print(f"     hall={100*s['hall_rate']:.4f}%  "
              f"(empty={100*s['n_empty']/s['n_total']:.4f}%  "
              f"double_col={100*s['n_double_col']/s['n_total']:.4f}%)")

    hall_rates   = np.array([r["hall_rate"] for r in runs]) * 100.0
    empty_rates  = np.array([r["n_empty"] / r["n_total"] for r in runs]) * 100.0
    double_rates = np.array([r["n_double_col"] / r["n_total"] for r in runs]) * 100.0

    return {
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
    p = argparse.ArgumentParser(description="p0 uniform01 vs uniform_i2 — UNet, 16/64x64, 5k/10k")
    p.add_argument("--img_sizes", type=str, default="16,64")
    p.add_argument("--dataset_sizes", type=str, default="5000,10000")
    p.add_argument("--p0_types", type=str, default="uniform01,uniform_i2")
    p.add_argument("--total_steps", type=str, default="16:40000,64:100000",
                   help="So gradient step. 1 so ap dung cho MOI img_size (vd '40000'), "
                        "hoac rieng tung img_size 'img:steps,...' (vd '16:40000,64:100000', "
                        "mac dinh). Ap dung deu cho moi (size,p0) trong CUNG 1 img_size.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--save_ckpt", action="store_true", default=True)
    p.add_argument("--no_save_ckpt", dest="save_ckpt", action="store_false")
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    return p.parse_args()


def parse_total_steps_spec(spec: str, img_sizes: list) -> dict:
    """'40000' -> {size:40000 cho moi size}. '16:40000,64:100000' -> rieng tung size."""
    if ":" in spec:
        d = {}
        for part in spec.split(","):
            k, v = part.split(":")
            d[int(k.strip())] = int(v.strip())
        missing = [isz for isz in img_sizes if isz not in d]
        if missing:
            raise ValueError(f"--total_steps thiếu img_size {missing}: {spec}")
        return d
    steps = int(spec)
    return {isz: steps for isz in img_sizes}


def main():
    args = parse_args()
    img_sizes = [int(s) for s in args.img_sizes.split(",")]
    sizes = [int(s) for s in args.dataset_sizes.split(",")]
    p0_types = [s.strip() for s in args.p0_types.split(",")]
    total_steps_by_size = parse_total_steps_spec(args.total_steps, img_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"arch=unet  img_sizes={img_sizes}  dataset_sizes={sizes}  p0_types={p0_types}")
    print(f"total_steps theo img_size: {total_steps_by_size}")
    n_combos = len(img_sizes) * len(sizes) * len(p0_types)
    total_grad_steps = sum(total_steps_by_size[isz] * len(sizes) * len(p0_types) for isz in img_sizes)
    print(f"Tổng số tổ hợp: {n_combos}  (tổng gradient step ước tính: {total_grad_steps:,} + "
          f"{n_combos}x{args.n_sample_total}x{args.n_repeats} ảnh sample)")

    if "uniform_i2" in p0_types:
        for isz in img_sizes:
            d = IN_CHANNELS * isz * isz
            bin0_hi = (1 / d) ** 2
            binlast_lo = ((d - 1) / d) ** 2
            print(f"  [uniform_i2 @ img_size={isz}] D={d}  chiều đầu ~ U[0, {bin0_hi:.2e}]  "
                  f"chiều cuối ~ U[{binlast_lo:.4f}, 1.0000]  (chuẩn hoá theo D^2, nằm gọn [0,1])")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    if args.save_ckpt:
        os.makedirs(ckpt_dir, exist_ok=True)

    results = []
    for img_size in img_sizes:
        args.total_steps = total_steps_by_size[img_size]
        bounds_i2 = make_uniform_i2_bounds(img_size, device) if "uniform_i2" in p0_types else None

        for n_size in sizes:
            label = size_label(n_size)
            data_dir = os.path.join(DATASET_ROOT, f"simple-shapes-{label}-{img_size}x{img_size}")
            if not os.path.isdir(data_dir):
                print(f"BỎ QUA img_size={img_size} size={n_size}: không thấy thư mục {data_dir}")
                continue
            dataset = ShapesDataset(data_dir, img_size)
            print(f"\n### img_size={img_size}  size={n_size} ({len(dataset)} ảnh thật) — {data_dir}")

            for p0_type in p0_types:
                bounds = bounds_i2 if p0_type == "uniform_i2" else None

                model = train_one(dataset, p0_type, bounds, args, device,
                                   img_size, ckpt_dir, n_size)

                print(f"\nSampling {args.n_sample_total} ảnh × {args.n_repeats} lần lặp "
                      f"(img={img_size}, n={n_size}, p0={p0_type}, steps={args.sample_steps}) ...")
                s = sample_and_analyze_repeated(model, p0_type, bounds, device,
                                                 args.n_sample_total, args.sample_steps,
                                                 args.sample_batch_size, args.n_repeats, img_size)

                print(f"  img={img_size:3d} n={n_size:6d}  p0={p0_type:12s}  "
                      f"hall={s['hall_rate_mean']:.3f}% ± {s['hall_rate_std']:.3f}%  "
                      f"(empty={s['empty_rate_mean']:.3f}%±{s['empty_rate_std']:.3f}%  "
                      f"double_col={s['double_rate_mean']:.3f}%±{s['double_rate_std']:.3f}%)")

                results.append({"img_size": img_size, "n_data": n_size, "p0_type": p0_type, **s})

                run_dir = os.path.join(OUTPUT_DIR, ckpt_stub(n_size, p0_type, img_size))
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "stats.txt"), "w") as f:
                    f.write(f"arch=unet  img_size={img_size}  n_data={n_size}  p0_type={p0_type}\n")
                    f.write(f"total_steps={args.total_steps}  n_sample_total/repeat="
                            f"{args.n_sample_total}  n_repeats={args.n_repeats}  "
                            f"sample_steps={args.sample_steps}\n\n")
                    f.write("Per-repeat hallucination rate (%):\n")
                    for rep, hr in enumerate(s["hall_rate_pct_list"], start=1):
                        f.write(f"  repeat {rep}: hall={hr:.4f}%  "
                                f"empty={s['empty_rate_pct_list'][rep-1]:.4f}%  "
                                f"double_col={s['double_rate_pct_list'][rep-1]:.4f}%\n")
                    f.write(f"\nHallucination : mean={s['hall_rate_mean']:.4f}%  "
                            f"std={s['hall_rate_std']:.4f}%\n")
                    f.write(f"  empty img   : mean={s['empty_rate_mean']:.4f}%  "
                            f"std={s['empty_rate_std']:.4f}%\n")
                    f.write(f"  double col  : mean={s['double_rate_mean']:.4f}%  "
                            f"std={s['double_rate_std']:.4f}%\n")

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # ── Bảng tổng hợp cuối ──
    print(f"\n{'='*80}")
    print(f"TỔNG HỢP (UNet)  (n_sample_total={args.n_sample_total}/lần × {args.n_repeats} lần lặp, "
          f"sample_steps={args.sample_steps}, total_steps theo img_size={total_steps_by_size})")
    print(f"{'='*80}")
    header = (f"{'img':>4s} {'n_data':>7s} {'p0_type':>10s} {'hall% mean':>11s} "
              f"{'hall% std':>10s} {'empty% mean':>12s} {'double_col% mean':>17s}")
    print(header)
    lines = [header]
    for r in results:
        line = (f"{r['img_size']:4d} {r['n_data']:7d} {r['p0_type']:>10s} "
                f"{r['hall_rate_mean']:11.4f} {r['hall_rate_std']:10.4f} "
                f"{r['empty_rate_mean']:12.4f} {r['double_rate_mean']:17.4f}")
        print(line)
        lines.append(line)

    if "uniform01" in p0_types and "uniform_i2" in p0_types:
        lines.append("")
        print()
        for img_size in img_sizes:
            for n_size in sizes:
                u01 = next((r for r in results if r["img_size"] == img_size
                            and r["n_data"] == n_size and r["p0_type"] == "uniform01"), None)
                ui2 = next((r for r in results if r["img_size"] == img_size
                            and r["n_data"] == n_size and r["p0_type"] == "uniform_i2"), None)
                if u01 and ui2:
                    rel = ("(giảm về 0)" if ui2["hall_rate_mean"] == 0 else
                           f"({100*(ui2['hall_rate_mean']-u01['hall_rate_mean'])/max(u01['hall_rate_mean'],1e-12):+.1f}% tương đối)")
                    line = (f"img={img_size} n={n_size}: "
                            f"uniform01={u01['hall_rate_mean']:.4f}%±{u01['hall_rate_std']:.4f}%  ->  "
                            f"uniform_i2={ui2['hall_rate_mean']:.4f}%±{ui2['hall_rate_std']:.4f}%  {rel}")
                    print(line)
                    lines.append(line)

    summary_path = os.path.join(OUTPUT_DIR, "summary_p0_uniform_family_unet.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
