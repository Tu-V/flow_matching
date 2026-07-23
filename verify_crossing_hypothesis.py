"""
Kiểm chứng giả thuyết: sample multi-active (>=2 chiều active) xảy ra vì noise ban
đầu x_0 rơi vào vùng "cách đều" (tied) giữa ĐÚNG 2 mode, tách biệt rõ khỏi 14 mode
còn lại — không phải ngẫu nhiên.

Vì mode i nằm ở ~e_i (vector đơn vị chiều i), và CondOT path x_t=(1-t)x_0+t*x_1 là
đường THẲNG, nên ||x_0 - e_i||^2 = ||x_0||^2 - 2*x_0[i] + 1 — phần ||x_0||^2 và +1
không đổi theo i, nên XẾP HẠNG các mode theo khoảng cách Euclid tới x_0 TƯƠNG ĐƯƠNG
với xếp hạng THEO GIÁ TRỊ THÔ x_0[i] (giảm dần). Không cần tính khoảng cách thật,
chỉ cần sort 16 giá trị của x_0.

Với mỗi sample sinh ra, tính:
    top1, top2, top3 = 3 chiều x_0 có giá trị lớn nhất (mode "gần" x_0 nhất)
    margin_12 = x_0[top1] - x_0[top2]   (nhỏ => top1,top2 "cách đều"/tied)
    margin_23 = x_0[top2] - x_0[top3]   (lớn => top1,top2 tách biệt rõ khỏi phần còn lại)

Giả thuyết đúng nếu:
    1) multi-active có margin_12 nhỏ hơn HẲN so với clean (tied rõ rệt hơn).
    2) multi-active có margin_23 lớn (2 mode thắng tách biệt khỏi 14 mode còn lại).
    3) 2 chiều active ở OUTPUT trùng khớp với top1,top2 của x_0 (không phải ngẫu nhiên).

Usage:
    python verify_crossing_hypothesis.py --ckpt toy_onehot_fm_output/mlp_onehot.pt --n_total 100000
"""

import argparse
import os
import statistics
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver          # noqa: E402
from train_toy_onehot_fm import DIM, ToyMLP, MLPVelocityWrapper   # noqa: E402


@torch.no_grad()
def run(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = ToyMLP(dim=DIM, hidden=ckpt["hidden"], depth=ckpt["depth"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.ckpt}")

    wrapper = MLPVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, args.steps + 1, device=device)

    margin12_clean, margin23_clean = [], []
    margin12_multi, margin23_multi = [], []
    n_match_top2 = 0
    n_multi_total = 0
    example_rows = []

    n_done = 0
    while n_done < args.n_total:
        B = min(args.batch_size, args.n_total - n_done)
        x_init = torch.randn(B, DIM, device=device)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        active = x_final > 0.5
        n_active = active.sum(dim=1)

        sorted_vals, sorted_idx = torch.sort(x_init, dim=1, descending=True)   # (B,16)
        m12 = (sorted_vals[:, 0] - sorted_vals[:, 1]).cpu()
        m23 = (sorted_vals[:, 1] - sorted_vals[:, 2]).cpu()

        for i in range(B):
            k = int(n_active[i].item())
            if k == 1:
                margin12_clean.append(m12[i].item())
                margin23_clean.append(m23[i].item())
            elif k >= 2:
                n_multi_total += 1
                margin12_multi.append(m12[i].item())
                margin23_multi.append(m23[i].item())

                active_dims = set(active[i].nonzero().flatten().tolist())
                top2_dims = set(sorted_idx[i, :2].tolist())
                if active_dims == top2_dims:
                    n_match_top2 += 1

                if len(example_rows) < 8:
                    example_rows.append({
                        "active_dims": sorted(active_dims),
                        "x0_top2_dims": sorted(top2_dims),
                        "x0_top3_vals": sorted_vals[i, :3].cpu().tolist(),
                        "match": active_dims == top2_dims,
                    })

        n_done += B

    print(f"\n{'='*70}")
    print(f"KẾT QUẢ  ({args.n_total} sample)")
    print(f"{'='*70}")
    print(f"So sample multi-active: {n_multi_total}")
    print(f"So khop: {{2 chieu active o OUTPUT}} == {{top-2 mode gan x_0 nhat}}: "
          f"{n_match_top2}/{n_multi_total}  ({100*n_match_top2/max(n_multi_total,1):.1f}%)")

    print(f"\nmargin_12 = x0[top1] - x0[top2]  (nho => 2 mode dau TIED/canh nhau)")
    print(f"  clean (1 active)   : mean={statistics.mean(margin12_clean):.4f}  "
          f"median={statistics.median(margin12_clean):.4f}  n={len(margin12_clean)}")
    print(f"  multi-active (>=2) : mean={statistics.mean(margin12_multi):.4f}  "
          f"median={statistics.median(margin12_multi):.4f}  n={len(margin12_multi)}")

    print(f"\nmargin_23 = x0[top2] - x0[top3]  (lon => top1,top2 tach biet ro khoi phan con lai)")
    print(f"  clean (1 active)   : mean={statistics.mean(margin23_clean):.4f}  "
          f"median={statistics.median(margin23_clean):.4f}")
    print(f"  multi-active (>=2) : mean={statistics.mean(margin23_multi):.4f}  "
          f"median={statistics.median(margin23_multi):.4f}")

    print(f"\nVi du {len(example_rows)} sample multi-active (active_dims o OUTPUT vs top-2 mode gan x_0 nhat):")
    for r in example_rows:
        flag = "KHOP" if r["match"] else "khong khop"
        print(f"  active_dims={r['active_dims']}  x0_top2_dims={r['x0_top2_dims']}  "
              f"x0_top3_vals={[round(v,3) for v in r['x0_top3_vals']]}  -> {flag}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default=os.path.join(REPO_ROOT, "toy_onehot_fm_output", "mlp_onehot.pt"))
    p.add_argument("--n_total", type=int, default=100000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4096)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
