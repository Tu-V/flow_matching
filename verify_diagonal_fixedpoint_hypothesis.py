"""
Kiểm chứng giả thuyết: "0 chiều active" (collapsed — mọi toạ độ output đều gần 0)
xảy ra vì x_0 (noise ban đầu) nằm GẦN "fixed-point subspace" của phép đối xứng
hoán vị — đường chéo D = { x : x_1 = x_2 = ... = x_16 } = span((1,1,...,1)) —
dùng ĐÚNG ground-truth field vừa cài đặt trong verify_exact_field_multiactive.py
(không qua mạng, để loại trừ mọi nhiễu do xấp xỉ mạng).

Lý thuyết (Golubitsky–Stewart, "fixed-point subspace" trong lý thuyết hệ động lực
có đối xứng): p1 ở đây BẤT BIẾN dưới nhóm hoán vị S_16 (chọn chiều active đều ngẫu
nhiên, 15 chiều còn lại cùng phân phối). Vì coupling x_0~N(0,I) cũng bất biến hoán
vị và path CondOT tuyến tính, nên ground-truth field u_t LÀ EQUIVARIANT:
    u_t(P x) = P u_t(x)   với mọi hoán vị P, mọi t.
Hệ quả chuẩn: đường chéo D là TẬP BẤT BIẾN của dòng chảy ODE — nếu x_0 in D thì
x_t in D với MỌI t (vì P x_0 = x_0 với mọi P khi x_0 in D, nên P u_t(x_0) =
u_t(P x_0) = u_t(x_0), tức u_t(x_0) cũng bị mọi P cố định -> u_t(x_0) in D).
Trên D, u_t(c·1) = (E[x_1[d] | trên D] - c)/(1-t) với TẤT CẢ toạ độ d đối xứng như
nhau -> quỹ đạo hội tụ về c_final·1 với c_final ~ E_p1[x] = 0.084375 << 0.5 với
MỌI toạ độ -> "collapsed" (0 chiều active) đúng theo định nghĩa ngưỡng >0.5.

4 kiểm định trên field CHÍNH XÁC (glued từ verify_exact_field_multiactive.py):
  (1) Equivariance số học: u_t(Px) == P u_t(x) với hoán vị ngẫu nhiên.
  (2) Quỹ đạo xuất phát ĐÚNG trên đường chéo (x_0 = c·1): std giữa 16 toạ độ phải
      giữ ~0 (sai số máy) suốt quỹ đạo, và điểm cuối phải "collapsed".
  (3) "Bán kính hút" quanh đường chéo: nhiễu x_0 = c·1 + eps·v (v vuông góc 1) với
      eps tăng dần -> tìm ngưỡng eps mà quỹ đạo VẪN sụp (gần đường chéo) hay ĐàBREAK
      SYMMETRY thành 1-active (đường chéo hút hay đẩy theo phương ngang?).
  (4) Tương quan thực nghiệm: lấy 100k mẫu x_0~N(0,I) THẬT (không ép lên đường chéo),
      đo "khoảng cách tới đường chéo" d0 = std(x_0), chạy field 2000 bước, xem
      max(x_final) (độ "nhọn" của kết quả) có tương quan với d0 hay không -> nếu
      giả thuyết đúng, d0 nhỏ => max(x_final) nhỏ hơn hẳn (kết quả "nhoè", gần
      collapsed hơn), dù trong 100k mẫu ngẫu nhiên hiếm khi đủ gần để thực sự vượt
      ngưỡng 0.5 (vì D=16 chiều, xác suất x_0 rơi gần 1 đường thẳng 1 chiều là ~0).

Usage:
    python verify_diagonal_fixedpoint_hypothesis.py
"""

import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver                                   # noqa: E402
from verify_exact_field_multiactive import (                                  # noqa: E402
    DIM, ground_truth_velocity, ExactFieldWrapper, E_P1,
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "exact_field_multiactive_output")
torch.set_default_dtype(torch.float64)


def euler_rollout(x0: torch.Tensor, steps: int, track_std: bool = False):
    dt = 1.0 / steps
    x = x0.clone()
    stds = []
    for i in range(steps):
        t = i * dt
        u = ground_truth_velocity(x, t)
        if track_std:
            stds.append(x.std(dim=-1, unbiased=False).max().item())
        x = x + dt * u
    return x, stds


# ── (1) Equivariance ───────────────────────────────────────────────────────
def test_equivariance():
    print("=" * 70)
    print("(1) EQUIVARIANCE: u_t(P x) == P u_t(x) ?")
    print("=" * 70)
    torch.manual_seed(0)
    max_err = 0.0
    for trial in range(20):
        x = torch.randn(4, DIM)
        t = float(torch.rand(1).item()) * 0.9 + 0.05
        perm = torch.randperm(DIM)
        u_x = ground_truth_velocity(x, t)
        u_Px = ground_truth_velocity(x[:, perm], t)
        err = (u_Px - u_x[:, perm]).abs().max().item()
        max_err = max(max_err, err)
    print(f"  max |u_t(Px) - P u_t(x)| qua 20 phép thử ngẫu nhiên (t, P, x): {max_err:.3e}")
    print(f"  {'=> EQUIVARIANT (đúng lý thuyết)' if max_err < 1e-9 else '=> KHÔNG equivariant (SAI!)'}")


# ── (2) Quỹ đạo xuất phát đúng trên đường chéo ─────────────────────────────
def test_exact_diagonal(steps: int = 2000):
    print("\n" + "=" * 70)
    print(f"(2) QUỸ ĐẠO TRÊN ĐƯỜNG CHÉO (x_0 = c·1), steps={steps}")
    print("=" * 70)
    torch.manual_seed(1)
    cs = torch.randn(8).tolist()
    all_collapsed = True
    max_std_seen = 0.0
    for c in cs:
        x0 = torch.full((1, DIM), c)
        x_final, stds = euler_rollout(x0, steps, track_std=True)
        max_std = max(stds) if stds else 0.0
        max_std_seen = max(max_std_seen, max_std)
        vmax = x_final.max().item()
        collapsed = vmax <= 0.5
        all_collapsed &= collapsed
        print(f"  c={c:+.4f}  max_std_giua_16_toa_do_suot_traj={max_std:.2e}  "
              f"final_max_coord={vmax:.6f}  final_mean={x_final.mean().item():.6f}  "
              f"{'COLLAPSED' if collapsed else 'KHONG collapsed'}")
    print(f"\n  max std quan sát suốt mọi quỹ đạo (kỳ vọng ~0, chỉ do float64 rounding): "
          f"{max_std_seen:.3e}")
    print(f"  E_p1[x] lý thuyết (điểm hút trên đường chéo) = {E_P1:.6f}  (< 0.5 => luôn collapsed)")
    print(f"  {'=> Đúng: MỌI quỹ đạo trên đường chéo đều COLLAPSED' if all_collapsed else '=> SAI giả thuyết!'}")


# ── (3) Bán kính hút / đẩy quanh đường chéo ────────────────────────────────
def test_basin_radius(steps: int = 2000):
    print("\n" + "=" * 70)
    print(f"(3) ĐƯỜNG CHÉO HÚT HAY ĐẨY THEO PHƯƠNG NGANG? steps={steps}")
    print("=" * 70)
    torch.manual_seed(2)
    c = 0.0
    v = torch.randn(DIM)
    v = v - v.mean()          # chiếu vuông góc với (1,1,...,1)
    v = v / v.norm()
    eps_list = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    for eps in eps_list:
        x0 = (torch.full((1, DIM), c) + eps * v.unsqueeze(0))
        x_final, _ = euler_rollout(x0, steps)
        vmax = x_final.max().item()
        n_active = int((x_final > 0.5).sum().item())
        std0 = x0.std(dim=-1, unbiased=False).item()
        print(f"  eps={eps:8.1e}  std(x_0)={std0:.2e}  ->  final_max_coord={vmax:8.4f}  "
              f"n_active={n_active}  {'(sụp/gần đường chéo)' if n_active==0 else '(đã tách mode, symmetry-broken)'}")
    print("\n  => Nếu n_active chuyển 0 -> 1 khi eps tăng: đường chéo ĐẨY theo phương ngang "
          "(unstable transversally) — chỉ nhiễu eps~0 (đo được) mới thực sự collapse; "
          "một nhiễu hữu hạn dù rất nhỏ cũng đủ để symmetry-break và hội tụ về 1 mode.")


# ── (4) Tương quan thực nghiệm trên 100k mẫu ngẫu nhiên thật ───────────────
@torch.no_grad()
def test_random_sampling_correlation(n_total: int = 100000, steps: int = 2000, batch_size: int = 4096):
    print("\n" + "=" * 70)
    print(f"(4) TƯƠNG QUAN d0=std(x_0) vs max(x_final)  ({n_total} mẫu x_0~N(0,I) THẬT, steps={steps})")
    print("=" * 70)
    solver = ODESolver(velocity_model=ExactFieldWrapper())
    time_grid = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)

    d0_all, vmax_all, nactive_all = [], [], []
    n_done = 0
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = torch.randn(B, DIM)
        d0 = x_init.std(dim=-1, unbiased=False)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        vmax = x_final.max(dim=-1).values
        nact = (x_final > 0.5).sum(dim=-1)
        d0_all.append(d0)
        vmax_all.append(vmax)
        nactive_all.append(nact)
        n_done += B
        print(f"    {n_done}/{n_total}", end="\r")
    print()

    d0_all = torch.cat(d0_all).numpy()
    vmax_all = torch.cat(vmax_all).numpy()
    nactive_all = torch.cat(nactive_all).numpy()

    corr = np.corrcoef(d0_all, vmax_all)[0, 1]
    print(f"  Pearson corr(d0=std(x_0), max(x_final)) = {corr:.4f}  "
          f"(kỳ vọng DƯƠNG mạnh nếu giả thuyết đúng: x_0 xa đường chéo -> field 'chắc chắn' hơn -> peak cao hơn)")

    order = np.argsort(d0_all)
    n = len(d0_all)
    decile = max(1, n // 10)
    lo_idx, hi_idx = order[:decile], order[-decile:]
    print(f"\n  Decile d0 NHỎ NHẤT (gần đường chéo nhất, n={len(lo_idx)}):")
    print(f"    d0: mean={d0_all[lo_idx].mean():.4f}  max={d0_all[lo_idx].max():.4f}")
    print(f"    max(x_final): mean={vmax_all[lo_idx].mean():.4f}  min={vmax_all[lo_idx].min():.4f}  "
          f"max={vmax_all[lo_idx].max():.4f}")
    print(f"    n_active phân bố: {dict(zip(*np.unique(nactive_all[lo_idx], return_counts=True)))}")

    print(f"\n  Decile d0 LỚN NHẤT (xa đường chéo nhất, n={len(hi_idx)}):")
    print(f"    d0: mean={d0_all[hi_idx].mean():.4f}  min={d0_all[hi_idx].min():.4f}")
    print(f"    max(x_final): mean={vmax_all[hi_idx].mean():.4f}  min={vmax_all[hi_idx].min():.4f}  "
          f"max={vmax_all[hi_idx].max():.4f}")

    idx_min = int(np.argmin(d0_all))
    print(f"\n  Mẫu có d0 NHỎ NHẤT trong toàn bộ {n_total} mẫu: d0={d0_all[idx_min]:.5f}  "
          f"max(x_final)={vmax_all[idx_min]:.5f}  n_active={nactive_all[idx_min]}")
    print(f"  (so sánh: d0 trung bình của x_0~N(0,I_16) lý thuyết ~ sqrt((D-1)/D) = "
          f"{math.sqrt((DIM-1)/DIM):.4f})")

    n_collapsed = int((nactive_all == 0).sum())
    print(f"\n  Tổng số collapsed (0 active) trong {n_total} mẫu: {n_collapsed} "
          f"({100*n_collapsed/n_total:.5f}%)")

    report_path = os.path.join(OUTPUT_DIR, "diagonal_fixedpoint_correlation.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"n_total={n_total} steps={steps}\n")
        f.write(f"corr(d0, max(x_final)) = {corr:.4f}\n")
        f.write(f"n_collapsed = {n_collapsed} ({100*n_collapsed/n_total:.5f}%)\n")
        f.write(f"smallest d0 in sample = {d0_all[idx_min]:.5f}, "
                f"its max(x_final) = {vmax_all[idx_min]:.5f}, n_active = {nactive_all[idx_min]}\n")
        f.write(f"decile smallest-d0 mean max(x_final) = {vmax_all[lo_idx].mean():.4f}\n")
        f.write(f"decile largest-d0 mean max(x_final)  = {vmax_all[hi_idx].mean():.4f}\n")
    print(f"\n  Saved -> {report_path}")


if __name__ == "__main__":
    test_equivariance()
    test_exact_diagonal(steps=2000)
    test_basin_radius(steps=2000)
    test_random_sampling_correlation(n_total=100000, steps=2000)
