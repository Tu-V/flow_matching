"""
Chạy thí nghiệm multi-step (25,50,100,200,500,1000) cho TOÀN BỘ case hallucination
đã lưu trong 1 thư mục traces/ — xem hallucination nào là do lỗi rời rạc hoá ODE
(biến mất khi tăng step) và hallucination nào là lỗi model thật (còn ở step=1000).

Gộp toàn bộ case thành 1 batch để chạy nhanh (thay vì loop từng case).

Với mỗi case, mỗi steps: lưu final_stepsXXXX.png + progression_strip_stepsXXXX.png
(10 mốc t) vào <case_dir>/multi_step_predict/ — giống hệt layout của
multi_step_predict.py (single-case) nhưng chạy hàng loạt.

Ngoài ra lưu 1 CSV tổng hợp + 1 heatmap tổng quan (case x steps) + 1 biểu đồ
số lượng hallucination còn lại theo step.

Usage:
    python multi_step_batch_experiment.py \\
        --traces_dir shapes_fm_output/hallucination_analysis/traces \\
        --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt \\
        --steps_list 25,50,100,200,500,1000
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from sample_shapes_fm import load_model, UNetVelocityWrapper, decode, _latest_ckpt   # noqa: E402
from hallucination_detector import analyze_image   # noqa: E402

IMG_SIZE = 16
ZOOM = 8


def upscale(img, z=ZOOM):
    return img.repeat(z, axis=0).repeat(z, axis=1)


def to_uint8_batch(x01_batch):
    """x01_batch: (N,3,H,W) float [0,1] -> (N,H,W,3) uint8 numpy."""
    return (x01_batch.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def euler_rollout_batch(wrapper, x_init_batch, steps, device):
    """Trả về list length steps+1, mỗi phần tử (N,H,W,3) uint8."""
    dt = 1.0 / steps
    x = x_init_batch
    frame_batches = [to_uint8_batch(decode(x))]
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)
        v = wrapper(x, t_tensor)
        x = x + dt * v
        frame_batches.append(to_uint8_batch(decode(x)))
    return frame_batches


def save_progression_strip(frame_batches, case_idx, out_path, n_strip=10):
    T = len(frame_batches)
    idxs = np.linspace(0, T - 1, n_strip).astype(int)
    imgs = [upscale(frame_batches[t][case_idx]) for t in idxs]
    grid = np.concatenate(imgs, axis=1)
    PILImage.fromarray(grid).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--steps_list", default="25,50,100,200,500,1000")
    args = parser.parse_args()

    steps_list = [int(s) for s in args.steps_list.split(",")]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt_path = args.ckpt or _latest_ckpt()
    model = load_model(ckpt_path, device)
    wrapper = UNetVelocityWrapper(model)

    case_names = sorted(d for d in os.listdir(args.traces_dir)
                        if os.path.isdir(os.path.join(args.traces_dir, d))
                        and os.path.exists(os.path.join(args.traces_dir, d, "noise_init.pt")))
    print(f"Tìm thấy {len(case_names)} case trong {args.traces_dir}")

    noises = [torch.load(os.path.join(args.traces_dir, c, "noise_init.pt"), weights_only=True)
              for c in case_names]
    x_init_batch = torch.cat(noises, dim=0).to(device)   # (N,3,16,16)
    N = x_init_batch.shape[0]
    print(f"Batch shape: {tuple(x_init_batch.shape)}")

    # ── Đọc nhãn gốc (lúc sampling, thường là steps=25) để so sánh ──
    orig_labels = {}
    for c in case_names:
        rp = os.path.join(args.traces_dir, c, "report.txt")
        with open(rp) as f:
            text = f.read()
        import re
        m = re.search(r"Hallucination type\s*:\s*(\S+)", text)
        orig_labels[c] = m.group(1) if m else "?"

    rows = []   # (case, steps, is_hall, hall_type, t,s,p)
    for steps in steps_list:
        print(f"\n-- steps={steps} --  ({steps} Euler step x {N} case, batched) ...")
        frame_batches = euler_rollout_batch(wrapper, x_init_batch, steps, device)
        final_frames = frame_batches[-1]   # (N,H,W,3)

        n_hall = 0
        for j, case in enumerate(case_names):
            img = final_frames[j]
            r = analyze_image(img)
            if r["is_hallucination"]:
                n_hall += 1

            out_dir = os.path.join(args.traces_dir, case, "multi_step_predict")
            os.makedirs(out_dir, exist_ok=True)
            PILImage.fromarray(upscale(img)).save(
                os.path.join(out_dir, f"final_steps{steps:04d}.png"))
            save_progression_strip(
                frame_batches, j,
                os.path.join(out_dir, f"progression_strip_steps{steps:04d}.png"),
            )

            cb = r["col_blobs"]
            hall_type = r["hall_type"] if r["is_hallucination"] else "none"
            rows.append({
                "case": case, "steps": steps, "is_hall": r["is_hallucination"],
                "hall_type": hall_type, "triangle": cb["triangle"],
                "square": cb["square"], "pentagon": cb["pentagon"],
            })
        print(f"   hall còn lại: {n_hall}/{N}  ({100*n_hall/N:.1f}%)")

    # ── Lưu CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.traces_dir, "..", "multi_step_experiment.csv")
    csv_path = os.path.abspath(csv_path)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {csv_path}")

    # ── Phân loại từng case: resolved_early / resolved_late / persistent ─────
    by_case = {c: {} for c in case_names}
    for r in rows:
        by_case[r["case"]][r["steps"]] = r["is_hall"]

    categories = {}
    for c in case_names:
        hall_at = by_case[c]
        if hall_at[steps_list[-1]]:
            cat = "persistent (loi model that)"
        else:
            first_ok_step = next(s for s in steps_list if not hall_at[s])
            cat = f"resolved_by_step_{first_ok_step}"
        categories[c] = cat

    from collections import Counter
    cat_counts = Counter(categories.values())
    print("\nPhân loại theo mức step làm hết hallucination:")
    for k, v in sorted(cat_counts.items()):
        print(f"  {k:30s}: {v}")

    # ── Heatmap case x steps ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, max(6, N * 0.12)))
    mat = np.array([[1 if by_case[c][s] else 0 for s in steps_list] for c in case_names])
    ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(steps_list)))
    ax.set_xticklabels(steps_list)
    ax.set_xlabel("ODE steps")
    ax.set_ylabel(f"case ({N} tổng)")
    ax.set_yticks([])
    ax.set_title(f"Hallucination còn hay hết theo số step\n(đỏ=vẫn hallucinate, trắng=hết)")
    plt.tight_layout()
    heatmap_path = os.path.join(os.path.dirname(csv_path), "multi_step_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {heatmap_path}")

    # ── Đường cong số lượng hallucination còn lại theo step ───────────────────
    counts = [sum(1 for c in case_names if by_case[c][s]) for s in steps_list]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_list, counts, marker="o", color="crimson")
    for s, cnt in zip(steps_list, counts):
        ax.annotate(f"{cnt}", (s, cnt), textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xscale("log")
    ax.set_xticks(steps_list)
    ax.set_xticklabels(steps_list)
    ax.set_xlabel("ODE steps (Euler)")
    ax.set_ylabel(f"# case vẫn hallucinate / {N}")
    ax.set_title("Hallucination giảm dần khi tăng độ chính xác ODE solver")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(os.path.dirname(csv_path), "multi_step_curve.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {curve_path}")


if __name__ == "__main__":
    main()
