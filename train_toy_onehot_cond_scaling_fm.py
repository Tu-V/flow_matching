"""
Scaling experiment CONDITIONING: kết hợp train_toy_onehot_cond_fm.py (conditioning +
CFG qua class_drop_prob) với train_toy_onehot_scaling_fm.py (train theo STEP cố định
qua nhiều cỡ dataset).

Cùng 1 kiến trúc ToyMLPCond (cùng param count — import lại từ train_toy_onehot_cond_fm.py
để đảm bảo giống hệt), train trên 6 cỡ TỔNG dataset khác nhau (mặc định
5k/10k/20k/50k/100k/200k, chia đều cho 16 class), CÙNG 200,000 step train / cỡ
dataset (không phải epoch), class_drop_prob=0.2 (CFG training — model học cả nhánh
unconditional).

Sau khi train xong MỖI cỡ dataset:
    - sample 100,000 điểm MỚI cho TỪNG class (16 x 100,000 = 1,600,000 sample)
    - sample 100,000 điểm unconditional (label=None)
    - đếm hallucination (ngưỡng active = giá trị > 0.5; >=2 chiều active = multi-active)

CẢNH BÁO CHI PHÍ: 6 cỡ dataset x (16 class x 100K + 100K unconditional)
= 6 x 1,700,000 = 10,200,000 sample tổng, mỗi sample 100 bước ODE. Model rất nhỏ
(~97K param, 16 chiều) nhưng tổng số sample RẤT LỚN -> NÊN CHẠY TRÊN GPU.

Usage (chạy trên GPU):
    python train_toy_onehot_cond_scaling_fm.py
    python train_toy_onehot_cond_scaling_fm.py --dataset_sizes 5000,10000,20000,50000,100000,200000 \
        --total_steps 200000 --class_drop_prob 0.2 --n_sample_per_run 100000
"""

import argparse
import os
import sys
import time
from collections import Counter

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath     # noqa: E402
from flow_matching.solver import ODESolver         # noqa: E402

# Dùng lại ĐÚNG kiến trúc + cách sinh data từ bản conditioning gốc.
from train_toy_onehot_cond_fm import (              # noqa: E402
    DIM, NUM_CLASSES, ToyMLPCond, MLPVelocityWrapper, make_dataset,
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "toy_onehot_cond_scaling_output")


# ── Train 1 model trên 1 cỡ dataset TỔNG, đếm theo STEP (không phải epoch) ────
def train_one(n_total_target: int, args, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_per_class = max(1, n_total_target // NUM_CLASSES)
    X_data, labels = make_dataset(n_per_class, DIM, NUM_CLASSES, seed=args.data_seed)
    X_data, labels = X_data.to(device), labels.to(device)
    n_actual = X_data.shape[0]

    model = ToyMLPCond(dim=DIM, num_classes=NUM_CLASSES, hidden=args.hidden, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1
    )

    print(f"\n{'='*70}")
    print(f"n_total_target={n_total_target}  (n_per_class={n_per_class}, n_actual={n_actual})  "
          f"params={n_params:,}  total_steps={args.total_steps}  class_drop_prob={args.class_drop_prob}")
    print(f"{'='*70}")

    n = n_actual
    model.train()
    loss_window = []
    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        x_1 = X_data[idx]
        y_1 = labels[idx]
        B = x_1.shape[0]

        if torch.rand(1).item() < args.class_drop_prob:
            label_in = None
        else:
            label_in = y_1

        x_0 = torch.randn(B, DIM, device=device)
        t = torch.rand(B, device=device)

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        u_pred = model(path_sample.x_t, t, label=label_in)
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
            print(f"  [n={n_total_target}] step {step:7d}/{args.total_steps}  loss={avg:.6f}  lr={lr:.2e}")
            loss_window = []

    print(f"  Train time: {time.time()-t0:.1f}s")

    ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_n{n_total_target}.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "hidden": args.hidden, "depth": args.depth,
         "n_total_target": n_total_target, "n_actual": n_actual, "total_steps": args.total_steps},
        ckpt_path,
    )
    print(f"  Saved: {ckpt_path}")
    return model, ckpt_path


# ── Sample 1 run (1 class cố định, hoặc unconditional) + đếm hallucination ─────
@torch.no_grad()
def sample_and_count(model, device, n_total: int, steps: int, label_val, batch_size: int):
    model.eval()
    wrapper = MLPVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    active_counts = Counter()
    n_correct = n_wrong = 0
    n_done = 0
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = torch.randn(B, DIM, device=device)
        label = (torch.full((B,), label_val, dtype=torch.long, device=device)
                 if label_val is not None else None)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False, label=label,
        )
        active = x_final > 0.5
        n_active = active.sum(dim=1)
        for i in range(B):
            k = int(n_active[i].item())
            active_counts[k] += 1
            if label_val is not None and k == 1:
                idx_active = int(active[i].nonzero()[0, 0].item())
                if idx_active == label_val:
                    n_correct += 1
                else:
                    n_wrong += 1
        n_done += B

    n_zero = active_counts.get(0, 0)
    n_one = active_counts.get(1, 0)
    n_multi = sum(v for k, v in active_counts.items() if k >= 2)
    return {"n_total": n_total, "n_zero": n_zero, "n_one": n_one, "n_multi": n_multi,
            "n_correct": n_correct, "n_wrong": n_wrong}


def parse_args():
    p = argparse.ArgumentParser(description="Scaling CONDITIONING: hallucination vs dataset size (+ class_drop_prob)")
    p.add_argument("--dataset_sizes", type=str, default="5000,10000,20000,50000,100000,200000")
    p.add_argument("--total_steps", type=int, default=200000, help="Tong so gradient step MOI co dataset")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--class_drop_prob", type=float, default=0.2)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10000)
    p.add_argument("--n_sample_per_run", type=int, default=100000,
                   help="So sample MOI class VA MOI lan unconditional (default: 100000)")
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=8192)
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
    print(f"Dataset sizes (tong, chia deu {NUM_CLASSES} class): {sizes}")
    print(f"Total steps / size: {args.total_steps}   class_drop_prob: {args.class_drop_prob}")
    print(f"Sample / class / lan chay: {args.n_sample_per_run}  "
          f"-> moi co dataset: {NUM_CLASSES} x {args.n_sample_per_run} (class) + "
          f"{args.n_sample_per_run} (unconditional) = "
          f"{NUM_CLASSES * args.n_sample_per_run + args.n_sample_per_run:,} sample")

    grand_summary = []
    for n_total_target in sizes:
        model, ckpt_path = train_one(n_total_target, args, device)

        print(f"\nSampling {args.n_sample_per_run}/class (x{NUM_CLASSES}) + "
              f"{args.n_sample_per_run} unconditional  (n_total_target={n_total_target}) ...")

        per_class_results = []
        for c in range(NUM_CLASSES):
            r = sample_and_count(model, device, args.n_sample_per_run, args.sample_steps, c,
                                 args.sample_batch_size)
            per_class_results.append(r)

        r_uncond = sample_and_count(model, device, args.n_sample_per_run, args.sample_steps, None,
                                    args.sample_batch_size)

        print(f"\n{'-'*70}")
        print(f"n_total_target={n_total_target}  KET QUA THEO CLASS")
        print(f"{'-'*70}")
        header = f"{'class':>5s} {'correct%':>9s} {'wrong_idx%':>11s} {'collapsed%':>11s} {'multi_active%':>14s}"
        print(header)
        for c, r in enumerate(per_class_results):
            n = r["n_total"]
            print(f"{c:5d} {100*r['n_correct']/n:9.2f} {100*r['n_wrong']/n:11.2f} "
                  f"{100*r['n_zero']/n:11.2f} {100*r['n_multi']/n:14.3f}")
        n = r_uncond["n_total"]
        print(f"{'uncond':>5s} {'-':>9s} {'-':>11s} {100*r_uncond['n_zero']/n:11.2f} "
              f"{100*r_uncond['n_multi']/n:14.3f}")

        avg_cond_multi = sum(r["n_multi"] for r in per_class_results) / (NUM_CLASSES * args.n_sample_per_run)
        avg_cond_correct = sum(r["n_correct"] for r in per_class_results) / (NUM_CLASSES * args.n_sample_per_run)
        uncond_multi = r_uncond["n_multi"] / r_uncond["n_total"]
        uncond_zero = r_uncond["n_zero"] / r_uncond["n_total"]

        print(f"\n  TB conditional: correct={100*avg_cond_correct:.2f}%  multi_active={100*avg_cond_multi:.3f}%")
        print(f"  Unconditional : collapsed={100*uncond_zero:.2f}%  multi_active={100*uncond_multi:.3f}%")

        grand_summary.append({
            "n_total_target": n_total_target, "ckpt_path": ckpt_path,
            "avg_cond_correct": avg_cond_correct, "avg_cond_multi": avg_cond_multi,
            "uncond_zero": uncond_zero, "uncond_multi": uncond_multi,
        })

    # ── Bảng tổng hợp cuối, so sánh giữa các cỡ dataset ──
    print(f"\n{'='*80}")
    print(f"TỔNG HỢP SCALING (CONDITIONING)  (total_steps={args.total_steps}, "
          f"class_drop_prob={args.class_drop_prob})")
    print(f"{'='*80}")
    header = (f"{'n_samples':>10s} {'cond_correct%':>14s} {'cond_multi_active%':>19s} "
              f"{'uncond_collapsed%':>18s} {'uncond_multi_active%':>21s}")
    print(header)
    lines = [header]
    for g in grand_summary:
        line = (f"{g['n_total_target']:10d} {100*g['avg_cond_correct']:14.2f} "
                f"{100*g['avg_cond_multi']:19.3f} {100*g['uncond_zero']:18.2f} "
                f"{100*g['uncond_multi']:21.3f}")
        print(line)
        lines.append(line)

    summary_path = os.path.join(OUTPUT_DIR, "summary_scaling_cond.txt")
    with open(summary_path, "w") as f:
        f.write(f"total_steps/size: {args.total_steps}   class_drop_prob: {args.class_drop_prob}   "
                f"n_sample/class/run: {args.n_sample_per_run}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
