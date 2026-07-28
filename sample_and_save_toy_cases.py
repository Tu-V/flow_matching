"""
Sample từ 1 checkpoint toy one-hot (unconditional, ToyMLP — vd các checkpoint trong
toy_onehot_scaling_output/) và LƯU LẠI cả hallucination case (collapsed + multi-active)
lẫn normal case (clean, đúng 1 chiều active), để xem lại chi tiết sau — không chỉ in
thống kê tổng hợp như analyze() trong train_toy_onehot_fm.py.

Phân loại (ngưỡng active = giá trị > 0.5):
    collapsed    : 0 chiều active
    normal/clean : đúng 1 chiều active
    multi_active : >=2 chiều active
"hallucination" = collapsed HOẶC multi_active (khớp convention hallucination_detector
bên ảnh: empty + double-col).

Output (folder suy ra từ path checkpoint, vd .../toy_onehot_scaling_output/cases_ckpt_n200000/):
    hallucination_x_init.pt / hallucination_x_final.pt / hallucination_info.txt
    normal_x_init.pt        / normal_x_final.pt
    stats.txt

Usage:
    python sample_and_save_toy_cases.py --ckpt toy_onehot_scaling_output/ckpt_n200000.pt --n_total 100000
"""

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver                                  # noqa: E402
from train_toy_onehot_fm import DIM, ToyMLP, MLPVelocityWrapper, fmt_vec    # noqa: E402


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

    all_x_init, all_x_final = [], []
    n_done = 0
    print(f"\nSampling {args.n_total} vector (steps={args.steps}) ...")
    while n_done < args.n_total:
        B = min(args.batch_size, args.n_total - n_done)
        x_init = torch.randn(B, DIM, device=device)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        all_x_init.append(x_init.cpu())
        all_x_final.append(x_final.cpu())
        n_done += B

    x_init_all = torch.cat(all_x_init, dim=0)
    x_final_all = torch.cat(all_x_final, dim=0)

    active = x_final_all > 0.5
    n_active = active.sum(dim=1)

    collapsed_mask = n_active == 0
    normal_mask = n_active == 1
    multi_mask = n_active >= 2
    halluc_mask = collapsed_mask | multi_mask

    n_total = x_final_all.shape[0]
    n_collapsed = int(collapsed_mask.sum())
    n_normal = int(normal_mask.sum())
    n_multi = int(multi_mask.sum())
    n_halluc = int(halluc_mask.sum())

    print(f"\n{'='*60}")
    print(f"KET QUA ({n_total} sample)")
    print(f"{'='*60}")
    print(f"  normal (1 chieu active)        : {n_normal:6d}  ({100*n_normal/n_total:.3f}%)")
    print(f"  collapsed (0 chieu active)     : {n_collapsed:6d}  ({100*n_collapsed/n_total:.3f}%)")
    print(f"  multi-active (>=2 chieu)       : {n_multi:6d}  ({100*n_multi/n_total:.3f}%)")
    print(f"  TONG HALLUCINATION (collapsed+multi): {n_halluc:6d}  ({100*n_halluc/n_total:.3f}%)")

    # ── output dir suy ra tu ten checkpoint ──
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    ckpt_tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_dir = os.path.join(ckpt_dir, f"cases_{ckpt_tag}")
    os.makedirs(out_dir, exist_ok=True)

    halluc_idx = halluc_mask.nonzero().flatten()
    normal_idx = normal_mask.nonzero().flatten()

    torch.save(x_init_all[halluc_idx], os.path.join(out_dir, "hallucination_x_init.pt"))
    torch.save(x_final_all[halluc_idx], os.path.join(out_dir, "hallucination_x_final.pt"))
    torch.save(x_init_all[normal_idx], os.path.join(out_dir, "normal_x_init.pt"))
    torch.save(x_final_all[normal_idx], os.path.join(out_dir, "normal_x_final.pt"))

    # report chi tiet cho tung hallucination case (de doc, khong can load .pt)
    info_path = os.path.join(out_dir, "hallucination_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Tong sample: {n_total}   Hallucination: {n_halluc} ({100*n_halluc/n_total:.3f}%)\n\n")
        for rank, i in enumerate(halluc_idx.tolist()):
            k = int(n_active[i].item())
            typ = "collapsed" if k == 0 else f"multi_active(k={k})"
            f.write(f"[{rank:5d}] idx={i:6d}  type={typ}\n")
            f.write(f"  x_final: {fmt_vec(x_final_all[i].tolist())}\n")
            f.write(f"  x_init : {fmt_vec(x_init_all[i].tolist())}\n\n")
    print(f"\nSaved {n_halluc} hallucination case (x_init/x_final .pt + info.txt) -> {out_dir}")
    print(f"Saved {n_normal} normal case (x_init/x_final .pt) -> {out_dir}")

    stats_path = os.path.join(out_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"N sample: {n_total}   ODE steps: {args.steps}   nguong active: >0.5\n\n")
        f.write(f"normal (1 chieu active)   : {n_normal} ({100*n_normal/n_total:.3f}%)\n")
        f.write(f"collapsed (0 chieu active): {n_collapsed} ({100*n_collapsed/n_total:.3f}%)\n")
        f.write(f"multi-active (>=2 chieu)  : {n_multi} ({100*n_multi/n_total:.3f}%)\n")
        f.write(f"TONG HALLUCINATION        : {n_halluc} ({100*n_halluc/n_total:.3f}%)\n")
    print(f"Stats -> {stats_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n_total", type=int, default=100000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8192)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
