"""
Phiên bản THỨ 4 (local-only), thay thế distance_sweep_delta_mij_unet16.py (bản 3)
cho vùng M_ij: thay vì ép CẢ ẢNH theo 1 anchor cố định (A0 ở box, B0=0 chỗ khác)
rồi cộng nhiễu r*u trên toàn D chiều -- khiến khi r lớn, TOÀN BỘ ảnh (kể cả cột
square/pentagon không liên quan) văng ra ngoài phân phối train cùng lúc, không
còn "trong phân phối" để so sánh công bằng với ảnh hallucination -- ở đây CHỈ ép
nhiễu lên đúng vùng box (nhóm S, 90px, 2 triangle-box) theo anchor+delta*u, còn
TOÀN BỘ phần còn lại của ảnh (nhóm T, 678px: cột square, cột pentagon, nền
quanh box) GIỮ NGUYÊN noise N(0,I) thật (không đổi, không ép hằng số).

Kiểm chứng trực quan (local, n=48, delta=10): ảnh "none" luôn có đúng 1 shape ở
cột triangle + 0-1 shape tự nhiên ở square/pentagon (sạch, như ảnh thật); ảnh
"double_col" luôn có đúng 2 shape ở cột triangle (lỗi đúng ý), còn square/
pentagon vẫn giữ 0-1 shape tự nhiên -- khác biệt duy nhất giữa 2 loại ảnh là
đúng tại vùng lỗi, không bị nhiễu lan sang các vùng khác.

Giới hạn đã biết: ở delta lớn (~>20 cho M_ij, do subspace box chỉ 90 chiều,
scale tự nhiên ~sqrt(90)~9.5), vùng box tự nó trở thành nhiễu cục bộ nhiều màu
(vượt xa thang giá trị train) và bị classifier double_col bắt nhầm -> double_col%
bật ngược lên lại (artifact, không phải hallucination thật). delta_list mặc định
cho mij dừng ở 50 để vẫn thấy xu hướng này (phục vụ thảo luận giới hạn), nhưng
phần dùng được/sạch chỉ nằm trong delta <= ~16-20.

Với Δ: vì Δ định nghĩa là TOÀN BỘ ảnh = hằng số (không có "vùng khác" để giữ
nguyên), nhóm S trùng với toàn ảnh (n_s=D) -> công thức local-only thu gọn về
đúng công thức bản 3 cũ (x0 = c* + delta*u trên toàn D chiều, không đổi).

Usage:
    python distance_sweep_delta_mij_local.py
    python distance_sweep_delta_mij_local.py --quick --device cpu
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from epsilon_sweep_delta_mij_unet16 import (  # noqa: E402
    IN_CHANNELS, IMG_SIZE, CKPT_PATH, REPO_ROOT,
    build_unet, UNetVelocityWrapper, decode_to_uint8_rgb, sample_from_x0,
    compute_c_star,
)
from epsilon_sweep_delta_mij_unet16_gaussian import build_group_masks        # noqa: E402
from flow_matching.solver import ODESolver                                   # noqa: E402
from hallucination_detector import analyze_batch                             # noqa: E402

D = IN_CHANNELS * IMG_SIZE * IMG_SIZE  # 768

OUT_DIR = os.path.join(
    REPO_ROOT, "shape_exp", "unet_architecture", "anisotropic_p0_output_16x16",
    "distance_sweep_delta_mij_local",
)


def sample_x0_local(name: str, n: int, delta: float, anchor_val: float, g: torch.Generator) -> torch.Tensor:
    masks = build_group_masks(name)
    mask_s = torch.from_numpy(masks[0])
    n_s = int(mask_s.sum())

    if name == "delta":
        # n_s == D -> toàn ảnh bị ép, không còn chỗ nào để giữ noise thật.
        z = torch.empty(n, D)
    else:
        # mij: khởi tạo noise N(0,I) thật cho TOÀN BỘ ảnh, sau đó chỉ ghi đè vùng S.
        z = torch.randn(n, D, generator=g)

    u = torch.randn(n, n_s, generator=g)
    u = u / u.norm(dim=1, keepdim=True)
    z[:, mask_s] = anchor_val + delta * u
    return z


def sweep_region(name: str, anchor_val: float, hall_type: str, solver, device,
                  steps: int, n_total: int, n_repeats: int, batch_size: int,
                  r_list, seed: int, rep_fn):
    rows = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for r in r_list:
        is_exact = (r == 0.0 and name == "delta")  # chỉ Δ ở r=0 mới là 1 điểm duy nhất (toàn ảnh = const)
        n_rep_eff = 1 if is_exact else n_repeats
        n_tot_eff = 1 if is_exact else n_total
        rates, counts = [], []
        t0 = time.time()
        for _ in range(n_rep_eff):
            n_done, n_hit = 0, 0
            while n_done < n_tot_eff:
                b = min(batch_size, n_tot_eff - n_done)
                x0 = sample_x0_local(name, b, r, anchor_val, g)
                x0 = x0.reshape(b, IN_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
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
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--r_list_delta", type=str,
                   default="0,0.5,1,2,3,5,7,10,15,20,27,35,50,60,70,80,90,100")
    p.add_argument("--r_list_mij", type=str,
                   default="0,0.5,1,2,3,5,7,10,13,16,20,25,30,40,50")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--quick", action="store_true", help="cấu hình siêu nhỏ để kiểm tra lỗi")
    a = p.parse_args()
    if a.quick:
        a.n_sample_total, a.n_repeats = 20, 2
        a.r_list_delta = "0,1,10,100"
        a.r_list_mij = "0,1,10,50"
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
    rep(f"\n[anchor] Δ: c* = {c_star:.4f}  (local-only == global, toàn ảnh là 1 nhóm)")
    rep(f"[anchor] M_ij: A0={a0:+.2f}  trên 2 vùng triangle (90px); phần còn lại (678px) "
        f"GIỮ NGUYÊN noise N(0,I) thật, không ép hằng số")

    r_list_delta = [float(x) for x in args.r_list_delta.split(",")]
    r_list_mij = [float(x) for x in args.r_list_mij.split(",")]
    rep(f"\nr_list_delta = {r_list_delta}")
    rep(f"r_list_mij   = {r_list_mij}")

    rep(f"\n{'='*90}\n[Δ] -> chỉ đếm hall_type='empty'\n{'='*90}")
    rows = sweep_region("delta", c_star, "empty", solver, device, args.sample_steps,
                         args.n_sample_total, args.n_repeats, args.sample_batch_size,
                         r_list_delta, args.seed + 1, rep)

    rep(f"\n{'='*90}\n[M_ij] -> chỉ đếm hall_type='double_col' (local-only: chỉ ép vùng box)\n{'='*90}")
    rows += sweep_region("mij", a0, "double_col", solver, device, args.sample_steps,
                          args.n_sample_total, args.n_repeats, args.sample_batch_size,
                          r_list_mij, args.seed + 2, rep)

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
