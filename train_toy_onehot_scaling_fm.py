"""
Scaling experiment: CÙNG 1 kiến trúc MLP (cùng param count, xem ToyMLP trong
train_toy_onehot_fm.py), train trên các CỠ DATASET khác nhau (mặc định
5k/10k/20k/50k/100k/200k), nhưng CÙNG tổng số training STEP (không phải epoch —
vì dataset size khác nhau, số epoch không so sánh công bằng được) = 200,000 step
mặc định cho MỖI cỡ dataset.

Data: mỗi sample chỉ enable 1 trong 16 chiều (chọn ngẫu nhiên đều), giá trị chiều
đó ~ U[0.95, 1.0], các chiều còn lại = 0 — giống hệt scheme trong
train_toy_onehot_fm.py (import lại đúng make_dataset/ToyMLP để đảm bảo kiến trúc
và cách sinh data khớp 100%, không lệch giữa các lần chạy).

Sau khi train xong MỖI cỡ dataset: lưu checkpoint riêng, rồi sample 100,000 điểm
MỚI và đếm hallucination (ngưỡng active = giá trị > 0.5; >=2 chiều active = multi-
active/hallucination). Cuối cùng in + lưu 1 bảng tổng hợp so sánh tất cả các cỡ.

Câu hỏi cần trả lời: hallucination có giảm khi có NHIỀU DATA hơn (cùng model
capacity, cùng tổng compute train) không? Nếu KHÔNG giảm rõ rệt / giữ nguyên ở 1
mức, đó là bằng chứng hallucination không phải do THIẾU DATA, mà do chính hình học
đa-mode của target distribution (16 cụm one-hot tách biệt trong không gian 16
chiều) kết hợp coupling độc lập ngẫu nhiên trong flow matching — không lượng data
nào "sửa" được, vì ngay cả phân phối THẬT (vô hạn mẫu) cũng có cấu trúc crossing-
path giữa các mode y hệt.

CẢNH BÁO CHI PHÍ: 6 cỡ dataset x 200,000 step (mặc định) = 1.2 triệu step tổng —
nên chạy trên GPU cho nhanh (mô hình rất nhỏ ~95K param nên mỗi step rẻ, nhưng
tổng số step lớn).

Usage (khuyến khích chạy trên GPU):
    python train_toy_onehot_scaling_fm.py
    python train_toy_onehot_scaling_fm.py --dataset_sizes 5000,10000,20000,50000,100000,200000 --total_steps 200000
"""

import argparse
import os
import sys
from collections import Counter

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath     # noqa: E402
from flow_matching.solver import ODESolver         # noqa: E402

# Dùng lại ĐÚNG kiến trúc + cách sinh data từ bản gốc, đảm bảo tương đương 100%.
from train_toy_onehot_fm import (                   # noqa: E402
    DIM, ToyMLP, MLPVelocityWrapper, make_dataset, fmt_vec,
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "toy_onehot_scaling_output")


# ── Train 1 model trên 1 cỡ dataset, đếm theo STEP (không phải epoch) ─────────
def train_one(n_samples: int, args, device) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_data, _ = make_dataset(n_samples, DIM, seed=args.data_seed)
    X_data = X_data.to(device)

    model = ToyMLP(dim=DIM, hidden=args.hidden, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1
    )

    print(f"\n{'='*70}")
    print(f"n_samples={n_samples}  params={n_params:,}  total_steps={args.total_steps}  "
          f"batch_size={args.batch_size}")
    print(f"{'='*70}")

    n = X_data.shape[0]
    model.train()
    loss_window = []
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        x_1 = X_data[idx]
        B = x_1.shape[0]

        x_0 = torch.randn(B, DIM, device=device)
        t = torch.rand(B, device=device)

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        u_pred = model(path_sample.x_t, t)
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
            print(f"  [n={n_samples}] step {step:7d}/{args.total_steps}  loss={avg:.6f}  lr={lr:.2e}")
            loss_window = []

    ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_n{n_samples}.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "hidden": args.hidden, "depth": args.depth,
         "n_samples": n_samples, "total_steps": args.total_steps},
        ckpt_path,
    )
    print(f"  Saved: {ckpt_path}")
    return ckpt_path, model


# ── Sample + đếm hallucination (dùng lại đúng logic threshold 0.5 như bản gốc) ──
@torch.no_grad()
def analyze(model, device, n_total: int, steps: int, batch_size: int = 4096):
    model.eval()
    wrapper = MLPVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    active_counts = Counter()
    example_multi = []
    n_done = 0
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = torch.randn(B, DIM, device=device)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        active = x_final > 0.5
        n_active = active.sum(dim=1)
        for i, c in enumerate(n_active.tolist()):
            active_counts[c] += 1
            if c >= 2 and len(example_multi) < 3:
                example_multi.append(x_final[i].cpu().numpy().tolist())
        n_done += B

    n_zero = active_counts.get(0, 0)
    n_one = active_counts.get(1, 0)
    n_multi = sum(v for k, v in active_counts.items() if k >= 2)
    return {
        "n_total": n_total, "n_zero": n_zero, "n_one": n_one, "n_multi": n_multi,
        "active_counts": dict(active_counts), "example_multi": example_multi,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Scaling: hallucination vs dataset size, cung total step")
    p.add_argument("--dataset_sizes", type=str, default="5000,10000,20000,50000,100000,200000",
                   help="Danh sach cac co dataset, phan cach dau phay")
    p.add_argument("--total_steps", type=int, default=200000, help="Tong so gradient step MOI co dataset")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=5000)
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=4096)
    return p.parse_args()


def main():
    args = parse_args()
    sizes = [int(s) for s in args.dataset_sizes.split(",")]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Dataset sizes: {sizes}")
    print(f"Total steps / size: {args.total_steps}")

    results = []
    for n_samples in sizes:
        ckpt_path, model = train_one(n_samples, args, device)
        print(f"\nSampling {args.n_sample_total} vector moi (n_samples={n_samples}) ...")
        r = analyze(model, device, args.n_sample_total, args.sample_steps, args.sample_batch_size)
        r["n_samples"] = n_samples
        r["ckpt_path"] = ckpt_path
        results.append(r)

        print(f"  n_samples={n_samples:7d}  "
              f"clean={100*r['n_one']/r['n_total']:.2f}%  "
              f"collapsed={100*r['n_zero']/r['n_total']:.2f}%  "
              f"MULTI-ACTIVE={100*r['n_multi']/r['n_total']:.3f}%")
        if r["example_multi"]:
            print(f"  vi du multi-active:")
            for v in r["example_multi"]:
                print(f"    {fmt_vec(v)}")

    # ── Bảng tổng hợp cuối ──
    print(f"\n{'='*70}")
    print(f"TỔNG HỢP SCALING  ({args.n_sample_total} sample/co, total_steps={args.total_steps})")
    print(f"{'='*70}")
    header = f"{'n_samples':>10s} {'clean%':>8s} {'collapsed%':>11s} {'multi_active%':>14s}"
    print(header)
    lines = [header]
    for r in results:
        line = (f"{r['n_samples']:10d} {100*r['n_one']/r['n_total']:8.2f} "
                f"{100*r['n_zero']/r['n_total']:11.2f} {100*r['n_multi']/r['n_total']:14.3f}")
        print(line)
        lines.append(line)

    summary_path = os.path.join(OUTPUT_DIR, "summary_scaling.txt")
    with open(summary_path, "w") as f:
        f.write(f"total_steps/size: {args.total_steps}   n_sample_total: {args.n_sample_total}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
