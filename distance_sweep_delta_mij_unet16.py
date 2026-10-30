"""
Phiên bản THỨ 3 (chạy song song, độc lập) để tìm 1 cách đo "khoảng cách" sạch:
r thấp -> hallucination% cao, r tăng -> hallucination% giảm dần về 0 (mượt, không
bị artifact như 2 bản trước).

Khác với epsilon_sweep_delta_mij_unet16.py (uniform box per-pixel, eps KHÔNG phải
khoảng cách Euclid thật -- với D=768 chiều, 1 box cạnh eps có đường chéo dài gấp
~sqrt(D/3)~16x eps, méo so với "khoảng cách", gây double_col% bật ngược lên khi
eps lớn) và khác epsilon_sweep_delta_mij_unet16_gaussian.py (để A,B của M_ij tự do
theo marginal N(0,1/n_g) -- tại radius=0 vẫn có thể suy biến A~B~0, không đảm bảo
hallucination cao khi "khoảng cách" = 0):

Ở đây dùng ĐÚNG khoảng cách Euclid r = ||x0 - anchor||_2 tới 1 ANCHOR CỐ ĐỊNH, có
ý nghĩa (đã biết chắc nằm trong vùng hút):
  - Δ:   anchor = c* * ones(D)        (c* = pixel trung bình tập train)
  - M_ij: anchor = A0 ở 2 vùng triangle, B0=0 chỗ khác  (A0=1.0, đã biết nằm
          trong vùng hút double_col)

x0 = anchor + r * u,  u ~ Uniform trên mặt cầu unit S^{D-1} (u = z/||z||, z~N(0,I)).
r=0 -> đúng anchor (chắc chắn đúng hall_type). r tăng -> nhiễu tăng ĐÚNG theo
khoảng cách thật, không lệ thuộc box/subspace-floating.

Usage:
    python distance_sweep_delta_mij_unet16.py
    python distance_sweep_delta_mij_unet16.py --quick --device cpu
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from epsilon_sweep_delta_mij_unet16 import (  # noqa: E402
    IN_CHANNELS, IMG_SIZE, GROUP_BOXES, CKPT_PATH, REPO_ROOT,
    build_unet, UNetVelocityWrapper, decode_to_uint8_rgb, sample_from_x0,
    compute_c_star,
)
from flow_matching.solver import ODESolver                                      # noqa: E402
from hallucination_detector import analyze_batch                                # noqa: E402

D = IN_CHANNELS * IMG_SIZE * IMG_SIZE  # 768

OUT_DIR = os.path.join(
    REPO_ROOT, "shape_exp", "unet_architecture", "anisotropic_p0_output_16x16",
    "distance_sweep_delta_mij",
)


def sweep_region(name: str, anchor: torch.Tensor, hall_type: str, solver, device,
                  steps: int, n_total: int, n_repeats: int, batch_size: int,
                  r_list, seed: int, rep_fn):
    rows = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    anchor_flat = anchor.reshape(1, D)
    for r in r_list:
        is_exact = (r == 0.0)
        n_rep_eff = 1 if is_exact else n_repeats
        n_tot_eff = 1 if is_exact else n_total
        rates, counts = [], []
        t0 = time.time()
        for _ in range(n_rep_eff):
            n_done, n_hit = 0, 0
            while n_done < n_tot_eff:
                b = min(batch_size, n_tot_eff - n_done)
                if r == 0.0:
                    x0 = anchor.expand(b, -1, -1, -1).clone()
                else:
                    z = torch.randn(b, D, generator=g)
                    u = z / z.norm(dim=1, keepdim=True)
                    x0 = (anchor_flat + r * u).reshape(b, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
                x0 = x0.to(device)
                x1 = sample_from_x0(solver, x0, steps)
                analyses = analyze_batch(decode_to_uint8_rgb(x1))
                n_hit += sum(1 for an in analyses if an["hall_type"] == hall_type)
                n_done += b
            counts.append(n_hit)
            rates.append(100.0 * n_hit / n_tot_eff)
        mean = float(np.mean(rates))
        std = float(np.std(rates, ddof=1)) if n_rep_eff > 1 else 0.0
        count_mean = float(np.mean(counts))
        count_std = float(np.std(counts, ddof=1)) if n_rep_eff > 1 else 0.0
        rep_fn(f"  [{name}] r={r:<7g}  {hall_type} count mean={count_mean:8.1f} std={count_std:6.1f}  "
               f"({hall_type}%% mean={mean:6.2f}  std={std:5.2f})  "
               f"(n={n_tot_eff}, repeats={n_rep_eff}, counts={counts}, rates={[f'{x:.2f}' for x in rates]}, "
               f"elapsed={time.time()-t0:.0f}s)")
        rows.append({"region": name, "r": r, "hall_type": hall_type,
                     "n_total_per_repeat": n_tot_eff, "n_repeats": n_rep_eff,
                     "count_mean": count_mean, "count_std": count_std,
                     "rate_mean": mean, "rate_std": std,
                     "counts_list": ";".join(str(c) for c in counts),
                     "rates_list": ";".join(f"{x:.4f}" for x in rates)})
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--r_list", type=str,
                   default="0,0.5,1,2,3,5,7,10,15,20,27,35,50,100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--quick", action="store_true", help="cấu hình siêu nhỏ để kiểm tra lỗi")
    a = p.parse_args()
    if a.quick:
        a.n_sample_total, a.n_repeats = 20, 2
        a.r_list = "0,1,10,100"
        a.sample_steps = 10
    return a


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}  (D={D}, sqrt(D)~{D**0.5:.1f})")

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

    c_star = compute_c_star(device)
    a0 = 1.0
    rep(f"\n[anchor] Δ: c* = {c_star:.4f}")
    rep(f"[anchor] M_ij: A0={a0:+.2f}  B0=0.00  (2 vùng triangle hàng 1-3 & 10-12, cột 0-5)")

    delta_point = torch.full((1, IN_CHANNELS, IMG_SIZE, IMG_SIZE), c_star)
    mij_point = torch.zeros(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    for y_sl, x_sl in GROUP_BOXES:
        mij_point[:, :, y_sl, x_sl] = a0

    r_list = [float(x) for x in args.r_list.split(",")]
    rep(f"\nr_list = {r_list}   (x0 = anchor + r * hướng ngẫu nhiên đều trên mặt cầu S^{{D-1}})")

    rep(f"\n{'='*90}\n[Δ] -> chỉ đếm hall_type='empty'\n{'='*90}")
    rows = sweep_region("delta", delta_point, "empty", solver, device, args.sample_steps,
                         args.n_sample_total, args.n_repeats, args.sample_batch_size,
                         r_list, args.seed + 1, rep)

    rep(f"\n{'='*90}\n[M_ij] -> chỉ đếm hall_type='double_col'\n{'='*90}")
    rows += sweep_region("mij", mij_point, "double_col", solver, device, args.sample_steps,
                          args.n_sample_total, args.n_repeats, args.sample_batch_size,
                          r_list, args.seed + 2, rep)

    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "r", "hall_type", "n_total_per_repeat",
                                           "n_repeats", "count_mean", "count_std", "rate_mean",
                                           "rate_std", "counts_list", "rates_list"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    rep(f"\nĐã lưu: {report_path}, {csv_path}")
    report.close()


if __name__ == "__main__":
    main()
