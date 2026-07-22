"""
So sánh định lượng: IG attribution (target = output cột 1/triangle) theo 3 vùng
noise, giữa nhóm case hallucination (HALL_) và nhóm case normal (không
hallucinate) — xem có pattern khác biệt nổi bật không (baseline=zero, m=50).

Gộp batch toàn bộ case để tính nhanh (giống ig_full_report_batch.py nhưng bỏ
phần vẽ ảnh, chỉ lấy số liệu tổng hợp theo vùng).
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from sample_shapes_fm import load_model, UNetVelocityWrapper, decode, to_uint8_numpy, _latest_ckpt   # noqa: E402
from hallucination_detector import analyze_image   # noqa: E402

IMG_SIZE = 16
ZONE_NAMES = ["triangle", "square", "pentagon"]
ZONE_SLICES = [(0, 5), (5, 10), (10, 15)]


def euler_rollout_batch_grad(wrapper, x_init_batch, steps, device):
    dt = 1.0 / steps
    x = x_init_batch
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)
        v = wrapper(x, t_tensor)
        x = x + dt * v
    return x


@torch.no_grad()
def euler_rollout_no_grad(wrapper, x_init_batch, steps, device):
    return euler_rollout_batch_grad(wrapper, x_init_batch, steps, device)


def main():
    steps = 25
    m = 50
    target_col = 1
    t0, t1 = ZONE_SLICES[target_col - 1]
    col_w = t1 - t0

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = load_model("shapes_fm_output/checkpoints/unet_epoch1000.pt", device)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapper = UNetVelocityWrapper(model)

    hall_cases = [
        "shapes_fm_output/hallucination_analysis/traces/HALL_case_0109_idx92812",
        "shapes_fm_output/hallucination_analysis/traces/HALL_case_0096_idx79065",
        "shapes_fm_output/hallucination_analysis/traces/HALL_case_0081_idx67253",
        "shapes_fm_output/hallucination_analysis/traces/HALL_case_0009_idx8329",
        "shapes_fm_output/hallucination_analysis/traces/HALL_case_0012_idx10116",
    ]
    normal_cases = [f"shapes_fm_output/hallucination_analysis/normal/sample_{i:02d}" for i in range(1, 6)]

    all_cases = hall_cases + normal_cases
    group = ["HALL"] * len(hall_cases) + ["normal"] * len(normal_cases)

    x_T_all = torch.cat(
        [torch.load(os.path.join(c, "noise_init.pt"), weights_only=True) for c in all_cases], dim=0
    ).to(device)
    N = x_T_all.shape[0]
    print(f"N case = {N}  ({len(hall_cases)} HALL + {len(normal_cases)} normal)")

    x_baseline = torch.zeros_like(x_T_all)

    alphas = torch.linspace(1.0 / m, 1.0, m, device=device).view(1, m, 1, 1, 1)
    x_T_exp = x_T_all.unsqueeze(1)
    x_base_exp = x_baseline.unsqueeze(1)
    x_alpha_5d = x_base_exp + alphas * (x_T_exp - x_base_exp)
    x_alpha = x_alpha_5d.reshape(N * m, 3, IMG_SIZE, IMG_SIZE).clone().requires_grad_(True)

    print(f"Forward ODE batch={N*m}, steps={steps} ...")
    x_0_alpha = euler_rollout_batch_grad(wrapper, x_alpha, steps, device)
    x_0_alpha_gray = x_0_alpha.mean(dim=1)

    diff_x = x_T_all - x_baseline

    IG_zone_score = np.zeros((N, 3))   # sum|IG| theo 3 vùng noise, gộp qua toàn bộ pixel cột target
    n_pixels = IMG_SIZE * col_w
    k = 0
    print(f"Backward cho {n_pixels} pixel target ...")
    for r in range(IMG_SIZE):
        for c in range(col_w):
            k += 1
            is_last = (k == n_pixels)
            F_pixel = x_0_alpha_gray[:, r, t0 + c]
            grad_pixel = torch.autograd.grad(F_pixel.sum(), x_alpha, retain_graph=not is_last)[0]
            grad_pixel = grad_pixel.view(N, m, 3, IMG_SIZE, IMG_SIZE)
            avg_grad = grad_pixel.mean(dim=1)
            IG = (diff_x * avg_grad).abs().mean(dim=1)   # (N,16,16) TB 3 kênh
            for zi, (s0, s1) in enumerate(ZONE_SLICES):
                IG_zone_score[:, zi] += IG[:, :, s0:s1].sum(dim=(1, 2)).detach().cpu().numpy()

    # ── Phân loại case (để in kèm) ─────────────────────────────────────────
    with torch.no_grad():
        x_0_all = euler_rollout_no_grad(wrapper, x_T_all, steps, device)
    labels = []
    for j in range(N):
        img = to_uint8_numpy(decode(x_0_all[j:j+1])[0])
        r = analyze_image(img)
        labels.append(r["hall_type"] if r["is_hallucination"] else "none")

    print(f"\n{'case':45s} {'group':7s} {'label':12s} {'triangle':>10s} {'square':>10s} {'pentagon':>10s}  %tri  %sq  %pent")
    for j, c in enumerate(all_cases):
        total = IG_zone_score[j].sum()
        pct = 100 * IG_zone_score[j] / total
        print(f"{os.path.basename(c):45s} {group[j]:7s} {labels[j]:12s} "
              f"{IG_zone_score[j,0]:10.4f} {IG_zone_score[j,1]:10.4f} {IG_zone_score[j,2]:10.4f}  "
              f"{pct[0]:4.1f} {pct[1]:4.1f} {pct[2]:4.1f}")

    # ── Gộp trung bình theo nhóm ───────────────────────────────────────────
    hall_idx = [j for j in range(N) if group[j] == "HALL"]
    norm_idx = [j for j in range(N) if group[j] == "normal"]

    hall_pct = np.array([100 * IG_zone_score[j] / IG_zone_score[j].sum() for j in hall_idx])
    norm_pct = np.array([100 * IG_zone_score[j] / IG_zone_score[j].sum() for j in norm_idx])

    print(f"\n{'='*70}")
    print(f"Trung bình %|IG| theo vùng noise (giải thích output cột triangle):")
    print(f"{'':10s} {'triangle':>10s} {'square':>10s} {'pentagon':>10s}")
    print(f"HALL   (n={len(hall_idx)})   " + "  ".join(f"{v:8.2f}%" for v in hall_pct.mean(axis=0)))
    print(f"normal (n={len(norm_idx)})   " + "  ".join(f"{v:8.2f}%" for v in norm_pct.mean(axis=0)))
    print(f"\nĐộ lệch chuẩn (để tham khảo mức nhiễu qua từng case):")
    print(f"HALL   std   " + "  ".join(f"{v:8.2f}%" for v in hall_pct.std(axis=0)))
    print(f"normal std   " + "  ".join(f"{v:8.2f}%" for v in norm_pct.std(axis=0)))

    np.savez("shapes_fm_output/hallucination_analysis/ig_compare_hall_vs_normal.npz",
             all_cases=np.array(all_cases), group=np.array(group), labels=np.array(labels),
             IG_zone_score=IG_zone_score)
    print(f"\nSaved: shapes_fm_output/hallucination_analysis/ig_compare_hall_vs_normal.npz")


if __name__ == "__main__":
    main()
