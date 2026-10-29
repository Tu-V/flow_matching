"""
Phiên bản THỨ 2 (chạy song song, độc lập) của epsilon_sweep_delta_mij_unet16.py:
thay vì lấy x0 ~ Uniform[anchor-eps, anchor+eps] quanh 1 điểm neo cố định, lấy
x0 ~ N(0,I_d) THẬT (đúng p0 lúc train) nhưng CÓ ĐIỀU KIỆN khoảng cách Euclid từ
x0 tới subspace Δ / M_ij <= radius. Không cần neo 1 điểm cụ thể: phần "song song
với subspace" (c cho Δ, hoặc (A,B) cho M_ij) tự lấy theo đúng marginal của nó
dưới N(0,I) (xem suy luận bên dưới) -- khớp đúng tinh thần "cả subspace là
fixed-point, điểm nào trong đó cũng tương đương".

Suy luận (cho 1 subspace được xác định bởi k nhóm toạ độ rời nhau, mỗi nhóm bị ép
bằng 1 hằng số -- Δ: 1 nhóm = toàn ảnh; M_ij: 2 nhóm = S (2 vùng triangle) và T
(phần còn lại)):
  - Giá trị trung bình mỗi nhóm A_g = mean_{i in g}(x_i) ~ N(0, 1/n_g), độc lập
    giữa các nhóm (vì các nhóm rời nhau, x~N(0,I)).
  - Phần dư r = x - sum_g A_g*1_g (phần "lệch khỏi subspace") có
    ||r||^2 ~ chi2_{d-k}, độc lập với các A_g.
  - Hướng của r (sau khi chuẩn hoá) là ngẫu nhiên đều trên mặt cầu unit của không
    gian dư (d-k chiều).
  -> lấy x0 ~ N(0,I) | ||r|| <= radius  <=>  lấy A_g như trên (không đổi), lấy
     hướng r ngẫu nhiên đều, và lấy ||r||=R ~ chi_{d-k} CẮT trên [0, radius].

d ở đây = 3*16*16 = 768 (IN_CHANNELS=3, IMG_SIZE=16) -> df dư rất lớn (767 cho Δ,
766 cho M_ij) -> chi2 cô đặc rất chặt quanh sqrt(df) ~ 27.7. Radius nhỏ (0.01-1)
nằm sâu trong vùng "mass underflow" của chi2.cdf (xác suất N(0,I) thật rơi gần
subspace đến mức đó là ~0) -- xử lý 3 chế độ số học để không crash và vẫn đúng:
  - radius=0: R=0 (chính xác, không cần sample).
  - logcdf(radius^2, df) rất âm (đuôi trái sâu): R ~ radius * U^(1/df) (xấp xỉ
    CHÍNH XÁC trong giới hạn này -- đây đúng là phân phối bán kính của 1 điểm
    ngẫu nhiên ĐỀU trong 1 hình cầu df chiều, vì mật độ chi2 gần 0 tỉ lệ x^(df/2-1),
    giống Jacobian toạ độ cầu).
  - logsf(radius^2, df) rất âm (đuôi phải sâu, radius >> sqrt(df)): lấy R từ chi2
    KHÔNG cắt (chi2.rvs) -- việc cắt gần như không ảnh hưởng gì.
  - còn lại: inverse-CDF chuẩn (mass=cdf, p~U(0,mass), R=sqrt(ppf(p))).

Usage:
    python epsilon_sweep_delta_mij_unet16_gaussian.py
    python epsilon_sweep_delta_mij_unet16_gaussian.py --quick --device cpu
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from scipy.stats import chi2

from epsilon_sweep_delta_mij_unet16 import (  # noqa: E402
    IN_CHANNELS, IMG_SIZE, GROUP_BOXES, CKPT_PATH, REPO_ROOT,
    build_unet, UNetVelocityWrapper, decode_to_uint8_rgb, sample_from_x0,
)
from flow_matching.solver import ODESolver                                      # noqa: E402
from hallucination_detector import analyze_batch                                # noqa: E402

D = IN_CHANNELS * IMG_SIZE * IMG_SIZE  # 768

OUT_DIR = os.path.join(
    REPO_ROOT, "shape_exp", "unet_architecture", "anisotropic_p0_output_16x16",
    "epsilon_sweep_delta_mij_gaussian",
)

UNDERFLOW_LOG = -40.0  # ngưỡng log(mass) / log(sf) coi là "sâu" (mass thật <1e-17)


def build_group_masks(name: str) -> list:
    shape = (IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    if name == "delta":
        return [np.ones(shape, dtype=bool).ravel()]
    if name == "mij":
        mask_s = np.zeros(shape, dtype=bool)
        for y_sl, x_sl in GROUP_BOXES:
            mask_s[:, y_sl, x_sl] = True
        return [mask_s.ravel(), (~mask_s).ravel()]
    raise ValueError(name)


def sample_radius(n: int, df: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    if radius <= 0:
        return np.zeros(n)
    x2 = radius ** 2
    logmass = chi2.logcdf(x2, df=df)
    logsf = chi2.logsf(x2, df=df)
    if logmass < UNDERFLOW_LOG:
        u = rng.uniform(size=n)
        return radius * u ** (1.0 / df)
    if logsf < UNDERFLOW_LOG:
        return np.sqrt(chi2.rvs(df=df, size=n, random_state=rng))
    mass = float(np.exp(logmass))
    p = rng.uniform(0.0, mass, size=n)
    return np.sqrt(chi2.ppf(p, df=df))


def sample_x0_gaussian_radius(name: str, n: int, radius: float, rng: np.random.Generator) -> torch.Tensor:
    masks = build_group_masks(name)
    df = D - len(masks)

    means_contrib = np.zeros((n, D), dtype=np.float64)
    for m in masks:
        n_g = int(m.sum())
        a_g = rng.normal(loc=0.0, scale=1.0 / np.sqrt(n_g), size=n)
        means_contrib[:, m] = a_g[:, None]

    z = rng.normal(size=(n, D))
    for m in masks:
        z[:, m] -= z[:, m].mean(axis=1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True)

    r = sample_radius(n, df, radius, rng)
    x0 = means_contrib + r[:, None] * z
    return torch.from_numpy(x0.reshape(n, IN_CHANNELS, IMG_SIZE, IMG_SIZE)).float()


def sweep_region_gaussian(name: str, hall_type: str, solver, device, steps: int,
                           n_total: int, n_repeats: int, batch_size: int,
                           radius_list, seed: int, rep_fn):
    rows = []
    rng = np.random.default_rng(seed)
    for radius in radius_list:
        is_exact = (radius == 0.0)
        n_rep_eff = 1 if is_exact else n_repeats
        n_tot_eff = 1 if is_exact else n_total
        rates, counts = [], []
        t0 = time.time()
        for _ in range(n_rep_eff):
            n_done, n_hit = 0, 0
            while n_done < n_tot_eff:
                b = min(batch_size, n_tot_eff - n_done)
                x0 = sample_x0_gaussian_radius(name, b, radius, rng).to(device)
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
        rep_fn(f"  [{name}] radius={radius:<6g}  {hall_type} count mean={count_mean:8.1f} std={count_std:6.1f}  "
               f"({hall_type}%% mean={mean:6.2f}  std={std:5.2f})  "
               f"(n={n_tot_eff}, repeats={n_rep_eff}, counts={counts}, rates={[f'{r:.2f}' for r in rates]}, "
               f"elapsed={time.time()-t0:.0f}s)")
        rows.append({"region": name, "radius": radius, "hall_type": hall_type,
                     "n_total_per_repeat": n_tot_eff, "n_repeats": n_rep_eff,
                     "count_mean": count_mean, "count_std": count_std,
                     "rate_mean": mean, "rate_std": std,
                     "counts_list": ";".join(str(c) for c in counts),
                     "rates_list": ";".join(f"{r:.4f}" for r in rates)})
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--radius_list", type=str, default="0,0.01,0.1,1,10,20,50,100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--quick", action="store_true", help="cấu hình siêu nhỏ để kiểm tra lỗi")
    a = p.parse_args()
    if a.quick:
        a.n_sample_total, a.n_repeats = 20, 2
        a.radius_list = "0,0.1,10,100"
        a.sample_steps = 10
    return a


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}  (D={D})")

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

    radius_list = [float(e) for e in args.radius_list.split(",")]
    rep(f"\nD={D}  df_delta={D-1}  df_mij={D-2}  (sqrt(df) ~ {D**0.5:.1f})")
    rep(f"radius_list = {radius_list}")
    rep("\nx0 ~ N(0,I_D) thật, điều kiện dist(x0, subspace) <= radius "
        "(radius=0 -> đúng tâm; radius lớn -> gần như N(0,I) không cắt)")

    rep(f"\n{'='*90}\n[Δ] -> chỉ đếm hall_type='empty'\n{'='*90}")
    rows = sweep_region_gaussian("delta", "empty", solver, device, args.sample_steps,
                                  args.n_sample_total, args.n_repeats, args.sample_batch_size,
                                  radius_list, args.seed + 1, rep)

    rep(f"\n{'='*90}\n[M_ij] -> chỉ đếm hall_type='double_col'\n{'='*90}")
    rows += sweep_region_gaussian("mij", "double_col", solver, device, args.sample_steps,
                                   args.n_sample_total, args.n_repeats, args.sample_batch_size,
                                   radius_list, args.seed + 2, rep)

    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "radius", "hall_type", "n_total_per_repeat",
                                           "n_repeats", "count_mean", "count_std", "rate_mean",
                                           "rate_std", "counts_list", "rates_list"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    rep(f"\nĐã lưu: {report_path}, {csv_path}")
    report.close()


if __name__ == "__main__":
    main()
