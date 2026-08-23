"""
Ground-truth (population) marginal velocity field cho p1 = "1 index cao ~U[0.95,1],
15 index còn lại ~U[0,0.05]" — suy ra CÔNG THỨC TƯỜNG MINH (không xấp xỉ bằng mạng,
không xấp xỉ bằng N=5000 mẫu rời rạc) rồi implement trực tiếp, sample 100k điểm với
2000 bước Euler (đủ mịn) và đếm "2 index cao" (multi-active).

Mục tiêu: tách bạch 2 nguồn gây multi-active đã biết trong codebase này —
  (1) sai số rời rạc hoá ODE (verify_step_count_hypothesis.py)   -> loại bằng 2000 bước
  (2) sai số xấp xỉ của MẠNG (network học sai field, đặc biệt gần separatrix)
  (3) sai số do DATASET HỮU HẠN (N=5000 mẫu rời rạc thay vì p1 liên tục thật, xem
      exact_marginal_field_analysis.py - N ảnh -> N "hố hút" rời rạc)
Field ở đây loại bỏ CẢ (2) và (3): dùng ĐÚNG p1 liên tục (vô hạn "mẫu"), không qua
mạng nơ-ron nào. Nếu multi-active vẫn > 0 ở đây -> là thuộc tính TOÁN HỌC của chính
quá trình flow matching (coupling độc lập ngẫu nhiên + CondOT path) với target p1
này, không phải lỗi network hay lỗi hữu hạn-N.

────────────────────────────────────────────────────────────────────────────
CÔNG THỨC p1
────────────────────────────────────────────────────────────────────────────
D = 16 chiều. Với mỗi mẫu: chọn k ~ Uniform{1..D} (đều, xác suất 1/D mỗi lớp), rồi

    p1(x) = (1/D) * sum_{k=1}^{D} p1_k(x)

    p1_k(x) = Unif(x_k; 0.95, 1.0) * prod_{j != k} Unif(x_j; 0, 0.05)

    Unif(y; a, b) = 1/(b-a)  nếu y in [a,b],  0  nếu ngược lại.

────────────────────────────────────────────────────────────────────────────
GROUND TRUTH VECTOR FIELD  (CondOT path, coupling độc lập x_0 ~ N(0,I_D))
────────────────────────────────────────────────────────────────────────────
Path: x_t = (1-t) x_0 + t x_1  =>  x_t | x_1 ~ N(t x_1, (1-t)^2 I_D)
Conditional velocity: u_t(x | x_1) = (x_1 - x) / (1-t)
Marginal (ground-truth) velocity:
    u_t(x) = ( E_{p1(x_1 | x_t=x)}[x_1] - x ) / (1-t)
    p1(x_1 | x_t=x)  âˆ  p1(x_1) * N(x; t x_1, (1-t)^2 I_D)

Vì p1 là mixture theo k của tích các Uniform ĐỘC LẬP theo từng chiều, và Gaussian
kernel cũng factorize theo chiều, nên (đặt s = 1-t, và với 1 khoảng [a,b] bất kỳ):

    Phi = CDF chuẩn tắc,  phi = pdf chuẩn tắc
    u1(x_d,a,b) = (t*b - x_d)/s,   u2(x_d,a,b) = (t*a - x_d)/s
    dPhi = Phi(u1) - Phi(u2)                      (∝ mật độ biên film theo chiều d)
    dphi = phi(u1) - phi(u2)

    log phi_d(x_d; a,b) = log(dPhi) - log(t*(b-a))          [log-mật độ 1 chiều]
    ratio_d(x_d; a,b)   = x_d/t - (s/t) * dphi/dPhi          [= E[z | x_d] với
                            z~Unif(a,b) "quan sát" qua kênh Gauss N(x_d; t*z, s^2),
                            đúng công thức mean truncated-normal]

Áp cho 2 giả thuyết mỗi chiều d: "active" (a,b)=(0.95,1) và "inactive" (a,b)=(0,0.05).
Khi cộng log-mật độ 16 chiều cho GIẢ THUYẾT "k là chiều active", phần 15 chiều
inactive-hypothesis GIỐNG HỆT NHAU giữa mọi k (đều dùng bound (0,0.05)) nên TRIỆT
TIÊU trong softmax chuẩn hoá theo k — hạ bậc phức tạp từ O(D^2) xuống O(D):

    score_d(x) = log phi_d(x_d; active) - log phi_d(x_d; inactive)     (D,)
    pi_d(x)    = softmax_d( score(x) )      = P(chiều d LÀ chiều active | x_t=x)
    E[x_1[d] | x_t=x] = pi_d(x) * ratio_d(x_d;active) + (1-pi_d(x)) * ratio_d(x_d;inactive)
    u_t(x)[d]  = ( E[x_1[d]|x_t=x] - x_d ) / (1-t)

Tại t->0: x_t=x_0 độc lập x_1 => posterior = prior => u_0(x) = E_{p1}[x_1] - x
(hằng số theo toạ độ, = (0.975 + 15*0.025)/16 = 0.084375 mỗi chiều) — dùng công
thức đóng này khi t < T_EPS để tránh 0/0 số học (không phải kỳ dị toán học thật).

Công thức trên đã được KIỂM CHỨNG BẰNG SỐ: (1) từng phần log-mật độ/ratio khớp
tuyệt đối với tích phân numerical (scipy.quad); (2) field đầy đủ (gồm cả bước rút
gọn softmax) khớp brute-force importance sampling (~20 triệu mẫu, ESS hàng triệu)
sai số ~1e-3..1e-4 (đúng bằng nhiễu Monte Carlo).

Usage:
    python verify_exact_field_multiactive.py --n_sample_total 100000 --sample_steps 2000
"""

import argparse
import math
import os
import sys
import time
from collections import Counter

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver     # noqa: E402
from flow_matching.utils import ModelWrapper    # noqa: E402
from train_toy_onehot_fm import fmt_vec          # noqa: E402

DIM = 16
ACTIVE_LO, ACTIVE_HI = 0.95, 1.0
INACTIVE_LO, INACTIVE_HI = 0.0, 0.05
T_EPS = 1e-4
OUTPUT_DIR = os.path.join(REPO_ROOT, "exact_field_multiactive_output")

E_P1 = (
    (ACTIVE_LO + ACTIVE_HI) / 2 + (DIM - 1) * (INACTIVE_LO + INACTIVE_HI) / 2
) / DIM   # = 0.084375, mean của p1 theo MỖI chiều (đối xứng)

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _std_normal_pdf(u: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * u * u) * INV_SQRT_2PI


def _log_dPhi(u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """log(Phi(u1)-Phi(u2)), u1>=u2, ổn định số học trên TOÀN BỘ trục thực.

    torch.special.ndtr(u) (CDF thường, không log) BÃO HOÀ về đúng 1.0 khi u chỉ
    mới ~8 (không cần u~39 mới underflow) — vì 1 - 1e-17 không phân biệt được
    với 1.0 trong float64 (epsilon máy ~2.2e-16) -> trừ trực tiếp ndtr(u1)-ndtr(u2)
    cho 0.0 dù giá trị thật ~1e-17 -> log(dPhi) sai hàng trăm đơn vị -> ratio nổ
    số (>1e280) chỉ sau vài trăm bước Euler (bug đã quan sát thực nghiệm).

    Thử "cứu" bằng log_ndtr (không bão hoà, đúng tới log~-700) + công thức
    log1p(-exp(lb-la)) ("logdiffexp" tiêu chuẩn) VẪN CHƯA ĐỦ: khi u1,u2 GẦN
    NHAU (trường hợp THỰC SỰ xảy ra ở đây, vì u1-u2 = t*(hi-lo)/s là hằng số cố
    định theo mỗi chiều, không phụ thuộc x — hoàn toàn có thể rất nhỏ so với 1),
    hiệu (lb-la) nhỏ hơn epsilon máy (~2.2e-16) khiến torch.exp(lb-la) LÀM TRÒN
    thành đúng 1.0 (mất hết thông tin), rồi log1p(-1.0) = log(0) = -inf.

    Fix ĐÚNG (chuẩn numerical: "log1mexp trick", Mächler 2012): dùng expm1
    (chính xác gần 0, KHÔNG bão hoà như exp) thay vì exp khi lb-la gần 0:
        x = lb - la  (<= 0)
        log(1 - exp(x)) = log(-expm1(x))   nếu x > -log(2)  (x gần 0, dùng expm1)
                         = log1p(-exp(x))   nếu x <= -log(2) (x rất âm, exp(x) nhỏ, an toàn)
    """
    la = torch.special.log_ndtr(u1)
    lb = torch.special.log_ndtr(u2)
    x = lb - la   # <= 0 (vì u1>=u2 => Phi(u1)>=Phi(u2) => la>=lb)
    near_zero = torch.log(-torch.expm1(x))      # x gần 0 (u1,u2 gần nhau) -> expm1 chính xác
    far_neg = torch.log1p(-torch.exp(x))        # x rất âm (u1,u2 cách xa) -> exp(x) nhỏ, an toàn
    log1mexp_x = torch.where(x > -math.log(2.0), near_zero, far_neg)
    return la + log1mexp_x


LOG_FLOOR = -700.0


def _logphi_and_ratio(x: torch.Tensor, t: float, s: float, lo: float, hi: float):
    u1 = (t * hi - x) / s
    u2 = (t * lo - x) / s
    log_dPhi = _log_dPhi(u1, u2)
    # FLOOR log_dPhi ở 1 giá trị HỮU HẠN (không phải chỉ clamp dPhi=exp(log_dPhi)
    # sau đó): nếu để log_dPhi tự nhiên = -inf (ca suy biến, VD x nằm ĐÚNG trên
    # đường chéo ở t gần 1, mọi giả thuyết active-cho-chiều-d đều có likelihood
    # ~0 GIỐNG HỆT NHAU tới từng bit) thì logphi=-inf cho MỌI d -> score=-inf mọi
    # d -> softmax(-inf,...,-inf) = NaN (0/0 thật). Floor log_dPhi ở -700 (dPhi
    # sàn ~1e-304, hữu hạn) giữ cho logphi luôn hữu hạn -> softmax vẫn tính được
    # đúng (khi TẤT CẢ d bị floor như nhau do đối xứng, softmax tự nhiên ra đều
    # 1/16 — đúng nghiệm giới hạn lý thuyết — không phải giá trị áp đặt tuỳ tiện).
    log_dPhi = torch.clamp(log_dPhi, min=LOG_FLOOR)
    dPhi = torch.exp(log_dPhi)
    dphi = _std_normal_pdf(u1) - _std_normal_pdf(u2)
    logphi = log_dPhi - math.log(t * (hi - lo))
    # QUAN TRỌNG (ổn định số học): ratio = x/t - (s/t)*dphi/dPhi bị catastrophic
    # cancellation khi t nhỏ (x/t và (s/t)*dphi/dPhi đều ~O(1/t), hiệu của chúng
    # chỉ ~O(1) -> mất hết chữ số có nghĩa, lan truyền/nổ số qua hàng nghìn bước
    # Euler). Gộp tử số TRƯỚC khi chia cho t đúng 1 lần -> tử số tự nhiên đã nhỏ
    # (~O(t)), không còn hiệu 2 số lớn.
    ratio = (x * dPhi - s * dphi) / (t * dPhi)
    return logphi, ratio


def ground_truth_velocity(x: torch.Tensor, t: float) -> torch.Tensor:
    """x: (B, DIM) float64. Trả về u_t(x): (B, DIM) float64 (công thức xem docstring)."""
    if t < T_EPS:
        return E_P1 - x
    s = 1.0 - t
    logphi_a, ratio_a = _logphi_and_ratio(x, t, s, ACTIVE_LO, ACTIVE_HI)
    logphi_i, ratio_i = _logphi_and_ratio(x, t, s, INACTIVE_LO, INACTIVE_HI)
    pi = torch.softmax(logphi_a - logphi_i, dim=1)     # (B, DIM)
    ex1 = pi * ratio_a + (1.0 - pi) * ratio_i
    return (ex1 - x) / s


class ExactFieldWrapper(ModelWrapper):
    def __init__(self):
        super().__init__(None)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        t_val = float(t.item()) if t.dim() == 0 else float(t.flatten()[0].item())
        return ground_truth_velocity(x, t_val)


@torch.no_grad()
def analyze(device, n_total: int, steps: int, batch_size: int = 4096):
    solver = ODESolver(velocity_model=ExactFieldWrapper())
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float64)

    active_counts = Counter()
    example_multi = []
    n_done = 0
    t0 = time.time()
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = torch.randn(B, DIM, device=device, dtype=torch.float64)
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
    p = argparse.ArgumentParser(description="Exact analytic ground-truth field: multi-active rate")
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--sample_steps", type=int, default=2000)
    p.add_argument("--sample_batch_size", type=int, default=4096)
    p.add_argument("--device", type=str, default="cpu",
                   help="mac định cpu vì MPS không hỗ trợ float64 (bắt buộc để field ổn định số học)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}  (float64)")
    print(f"p1: 1/{DIM} chiều active~U[{ACTIVE_LO},{ACTIVE_HI}], "
          f"{DIM-1}/{DIM} chiều inactive~U[{INACTIVE_LO},{INACTIVE_HI}]")
    print(f"E_p1[x] mỗi chiều = {E_P1:.6f}  (dùng làm field tại t->0)")

    print(f"\nSampling {args.n_sample_total} vector bằng GROUND-TRUTH FIELD "
          f"(không qua mạng, không qua N=5000 mẫu rời rạc), steps={args.sample_steps} ...")
    r = analyze(device, args.n_sample_total, args.sample_steps, args.sample_batch_size)

    print(f"\n{'='*60}")
    print(f"KẾT QUẢ  ({r['n_total']} sample, field=GROUND-TRUTH ANALYTIC, "
          f"steps={args.sample_steps}, ngưỡng active=>0.5)")
    print(f"{'='*60}")
    print(f"  0 chiều active   (collapsed)     : {r['n_zero']:6d}  ({100*r['n_zero']/r['n_total']:.4f}%)")
    print(f"  1 chiều active   (clean, đúng)   : {r['n_one']:6d}  ({100*r['n_one']/r['n_total']:.4f}%)")
    print(f"  >=2 chiều active (MULTI-ACTIVE)  : {r['n_multi']:6d}  ({100*r['n_multi']/r['n_total']:.4f}%)")
    print(f"\nPhân bố chi tiết số chiều active:")
    for k in sorted(r["active_counts"]):
        print(f"    {k} chiều: {r['active_counts'][k]:6d}  ({100*r['active_counts'][k]/r['n_total']:.4f}%)")
    if r["example_multi"]:
        print(f"\nVí dụ multi-active:")
        for v in r["example_multi"]:
            print(f"    {fmt_vec(v)}")

    if r["n_multi"] == 0:
        verdict = ("=> multi-active = 0/%d. Với field lý tưởng CHÍNH XÁC (không xấp xỉ mạng, "
                    "không xấp xỉ N hữu hạn) và 2000 bước Euler, '2 index cao' KHÔNG xảy ra "
                    "-> hiện tượng quan sát được ở model thật là do xấp xỉ mạng và/hoặc "
                    "N hữu hạn, không phải thuộc tính toán học của chính p1/coupling này." % r['n_total'])
    else:
        verdict = (f"=> multi-active = {r['n_multi']}/{r['n_total']} ({100*r['n_multi']/r['n_total']:.4f}%) "
                   f"ngay cả với field lý tưởng CHÍNH XÁC (đúng p1 liên tục, 2000 bước Euler) "
                   f"-> đây là thuộc tính TOÁN HỌC thật của flow matching với coupling độc lập "
                   f"trên p1 này (tồn tại vùng x_0 mà field 'lưỡng lự' giữa 2 mode và ODE hội tụ "
                   f"tới điểm có 2 chiều đều >0.5), KHÔNG phải lỗi xấp xỉ mạng hay lỗi N hữu hạn.")
    print(f"\n{verdict}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats_path = os.path.join(OUTPUT_DIR, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"N sample: {r['n_total']}   ODE steps: {args.sample_steps}   "
                f"field: GROUND-TRUTH ANALYTIC (khong qua mang)   nguong active: >0.5\n\n")
        f.write(f"0 chieu active   (collapsed)   : {r['n_zero']} ({100*r['n_zero']/r['n_total']:.4f}%)\n")
        f.write(f"1 chieu active   (clean)       : {r['n_one']} ({100*r['n_one']/r['n_total']:.4f}%)\n")
        f.write(f"2+ chieu active  (multi-active): {r['n_multi']} ({100*r['n_multi']/r['n_total']:.4f}%)\n\n")
        f.write("Phan bo chi tiet:\n")
        for k in sorted(r["active_counts"]):
            f.write(f"  {k} chieu: {r['active_counts'][k]} ({100*r['active_counts'][k]/r['n_total']:.4f}%)\n")
        if r["example_multi"]:
            f.write("\nVi du sample multi-active:\n")
            for v in r["example_multi"]:
                f.write(f"  {fmt_vec(v)}\n")
        f.write(f"\n{verdict}\n")
    print(f"\nStats -> {stats_path}")


if __name__ == "__main__":
    main()
