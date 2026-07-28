"""
Kiểm chứng Đề xuất 4: minibatch OT coupling (OT-CFM, Tong et al. 2023 / tinh thần
Rectified Flow) có làm GIẢM tỉ lệ "2 index cao" (multi-active, >=2 chiều > 0.5) so
với coupling độc lập ngẫu nhiên (baseline flow matching chuẩn) hay không.

p1 MỚI (thay cho one-hot cũ ở train_toy_onehot_fm.py):
    - 1 chiều được chọn ngẫu nhiên đều trong 16 chiều: giá trị ~ U[0.95, 1.0]
    - 15 chiều còn lại: giá trị ~ U[0.0, 0.05]  (không còn = 0 tuyệt đối nữa)
Lý do đổi: với 15 chiều = 0 CHÍNH XÁC, một phần "signal" giúp network phân biệt
active/inactive có thể tới từ việc target ở các chiều đó luôn = 0 (dấu hiệu dễ học).
Thêm nhiễu U[0,0.05] loại bỏ chỗ dựa đó, buộc network phải thực sự học đúng field
gần separatrix thay vì học "0 luôn đúng".

Coupling (2 nhánh, CÙNG kiến trúc/data/total_steps, chỉ khác cách ghép (x_0,x_1)):
    independent : x_0 ~ N(0,I) và x_1 ~ batch data được rút ĐỘC LẬP mỗi batch
                  (chuẩn CFM, dùng cho toàn bộ codebase này từ trước tới giờ).
    ot          : trong MỖI batch, có B mẫu x_0 ~ N(0,I) và B mẫu x_1 (rút từ data).
                  Giải bài toán OT CHÍNH XÁC (cost = ||x0_i - x1_j||^2) giữa 2 tập
                  rời rạc B điểm trọng số đều -> đây là bài toán assignment cân bằng,
                  nghiệm tối ưu là 1 PHÉP HOÁN VỊ (linear_sum_assignment / Hungarian,
                  tương đương POT ot.emd trong trường hợp B=B, uniform weights, chỉ
                  nhanh hơn nhiều). Ghép lại x_1 theo hoán vị đó trước khi tính
                  path/target velocity. Kỳ vọng: ít đường (x_0,x_1) cắt nhau hơn trong
                  minibatch -> field mục tiêu "hiền" hơn gần separatrix -> multi-active
                  giảm.

Sau khi train xong 2 model: sample --n_sample_total (mặc định 100,000) mỗi model với
--sample_steps bước Euler (mặc định 2000, đủ mịn để loại trừ sai số rời rạc hoá theo
verify_step_count_hypothesis.py), đếm 0/1/>=2 chiều active (ngưỡng >0.5), in bảng so
sánh independent vs ot.

Usage:
    python verify_otcfm_hypothesis.py
    python verify_otcfm_hypothesis.py --n_samples 5000 --total_steps 40000 \
        --n_sample_total 100000 --sample_steps 2000
"""

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath      # noqa: E402
from flow_matching.solver import ODESolver          # noqa: E402

from train_toy_onehot_fm import ToyMLP, MLPVelocityWrapper, fmt_vec   # noqa: E402

DIM = 16
OUTPUT_DIR = os.path.join(REPO_ROOT, "toy_onehot_otcfm_output")


# ── p1 mới: active ~ U[0.95,1.0], inactive ~ U[0,0.05] ─────────────────────────
def make_dataset_noisy(n_samples: int, dim: int = DIM, seed: int = 0,
                        active_lo: float = 0.95, active_hi: float = 1.0,
                        inactive_lo: float = 0.0, inactive_hi: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, dim, (n_samples,), generator=g)
    X = torch.empty(n_samples, dim).uniform_(inactive_lo, inactive_hi, generator=g)
    active_vals = torch.empty(n_samples).uniform_(active_lo, active_hi, generator=g)
    X[torch.arange(n_samples), labels] = active_vals
    return X, labels


# ── Minibatch OT coupling: ghép lại x_1 theo hoán vị tối ưu (Hungarian) ────────
def ot_reorder(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Trả về x1 đã ghép lại sao cho sum ||x0_i - x1_{perm(i)}||^2 nhỏ nhất
    (nghiệm chính xác của bài toán OT cân bằng rời rạc B<->B, uniform weights)."""
    with torch.no_grad():
        cost = torch.cdist(x0, x1, p=2).pow(2).cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost)
    # row_idx luôn = [0..B-1] theo thứ tự tăng dần (tính chất của linear_sum_assignment)
    return x1[col_idx]


# ── Train 1 model với coupling chỉ định, đếm theo STEP (không epoch) ──────────
def train_one(coupling: str, X_data: torch.Tensor, args, device) -> nn.Module:
    model = ToyMLP(dim=DIM, hidden=args.hidden, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1
    )

    print(f"\n{'='*70}")
    print(f"coupling={coupling}  params={n_params:,}  total_steps={args.total_steps}  "
          f"batch_size={args.batch_size}")
    print(f"{'='*70}")

    n = X_data.shape[0]
    model.train()
    loss_window = []
    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        x_1 = X_data[idx]
        B = x_1.shape[0]

        x_0 = torch.randn(B, DIM, device=device)
        if coupling == "ot":
            x_1 = ot_reorder(x_0, x_1)

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
            dt = time.time() - t0
            print(f"  [{coupling}] step {step:7d}/{args.total_steps}  loss={avg:.6f}  "
                  f"lr={lr:.2e}  elapsed={dt:.0f}s")
            loss_window = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_{coupling}.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "hidden": args.hidden, "depth": args.depth,
         "coupling": coupling, "total_steps": args.total_steps},
        ckpt_path,
    )
    print(f"  Saved: {ckpt_path}")
    return model


# ── Sample + đếm multi-active (giống hệt logic train_toy_onehot_fm.analyze) ───
@torch.no_grad()
def analyze(model, device, n_total: int, steps: int, batch_size: int = 4096):
    model.eval()
    wrapper = MLPVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    active_counts = Counter()
    example_multi = []
    n_done = 0
    t0 = time.time()
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
            if c >= 2 and len(example_multi) < 5:
                example_multi.append(x_final[i].cpu().numpy().round(4).tolist())
        n_done += B
        print(f"    sampled {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", end="\r")
    print()

    n_zero = active_counts.get(0, 0)
    n_one = active_counts.get(1, 0)
    n_multi = sum(v for k, v in active_counts.items() if k >= 2)
    return {
        "n_total": n_total, "n_zero": n_zero, "n_one": n_one, "n_multi": n_multi,
        "active_counts": dict(active_counts), "example_multi": example_multi,
    }


def parse_args():
    p = argparse.ArgumentParser(description="OT-CFM vs independent coupling: multi-active rate")
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--log_every", type=int, default=5000)
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--sample_steps", type=int, default=2000)
    p.add_argument("--sample_batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    X_data, labels = make_dataset_noisy(args.n_samples, DIM, seed=args.data_seed)
    X_data = X_data.to(device)
    print(f"Dataset: {X_data.shape[0]} vector, active~U[0.95,1.0], inactive~U[0,0.05]")
    print(f"  ví dụ: {fmt_vec(X_data[0].cpu().tolist())}")

    results = {}
    for coupling in ["independent", "ot"]:
        torch.manual_seed(args.seed)
        model = train_one(coupling, X_data, args, device)
        print(f"\nSampling {args.n_sample_total} vector (coupling={coupling}, "
              f"steps={args.sample_steps}) ...")
        r = analyze(model, device, args.n_sample_total, args.sample_steps, args.sample_batch_size)
        results[coupling] = r
        print(f"  coupling={coupling:12s}  "
              f"clean={100*r['n_one']/r['n_total']:.3f}%  "
              f"collapsed={100*r['n_zero']/r['n_total']:.3f}%  "
              f"MULTI-ACTIVE={100*r['n_multi']/r['n_total']:.3f}%")
        if r["example_multi"]:
            print(f"  ví dụ multi-active:")
            for v in r["example_multi"]:
                print(f"    {fmt_vec(v)}")

    print(f"\n{'='*70}")
    print(f"TỔNG HỢP  (n_sample_total={args.n_sample_total}, sample_steps={args.sample_steps}, "
          f"total_train_steps={args.total_steps} mỗi nhánh)")
    print(f"{'='*70}")
    header = f"{'coupling':>12s} {'clean%':>8s} {'collapsed%':>11s} {'multi_active%':>14s}"
    print(header)
    lines = [header]
    for c in ["independent", "ot"]:
        r = results[c]
        line = (f"{c:>12s} {100*r['n_one']/r['n_total']:8.3f} "
                f"{100*r['n_zero']/r['n_total']:11.3f} {100*r['n_multi']/r['n_total']:14.3f}")
        print(line)
        lines.append(line)

    r_ind, r_ot = results["independent"], results["ot"]
    rate_ind = r_ind["n_multi"] / r_ind["n_total"]
    rate_ot = r_ot["n_multi"] / r_ot["n_total"]
    if rate_ind > 0:
        rel_change = 100 * (rate_ot - rate_ind) / rate_ind
        verdict = (f"OT coupling thay doi multi-active rate: {100*rate_ind:.3f}% -> {100*rate_ot:.3f}% "
                   f"({rel_change:+.1f}% tuong doi)")
    else:
        verdict = f"independent multi-active rate = 0, khong co gi de so sanh giam."
    print(f"\n{verdict}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "summary_otcfm.txt")
    with open(summary_path, "w") as f:
        f.write(f"n_samples={args.n_samples}  total_steps/nhanh={args.total_steps}  "
                f"n_sample_total={args.n_sample_total}  sample_steps={args.sample_steps}\n\n")
        f.write("\n".join(lines) + "\n\n")
        f.write(verdict + "\n")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
