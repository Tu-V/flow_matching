"""
Chạy lại 1 noise_init.pt đã biết qua ODE Euler với nhiều số step khác nhau
(25, 50, 100, 200, 500, 1000), xem ảnh cuối cùng thay đổi thế nào khi tăng độ
chính xác tích phân — và so sánh step=25 với final_image.png đã lưu sẵn (phải
khớp gần như tuyệt đối vì cùng steps=25, cùng noise, flow matching là ODE
deterministic).

Với mỗi số step: lưu progression_strip (10 mốc t đều nhau) + final image riêng.

Usage:
    python multi_step_predict.py \\
        --case_dir shapes_fm_output/hallucination_analysis/traces/case_0003_idx1793 \\
        --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt \\
        --steps_list 25,50,100,200,500,1000
"""

import argparse
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

from sample_shapes_fm import (   # noqa: E402
    load_model, UNetVelocityWrapper, decode, to_uint8_numpy, _latest_ckpt,
)
from hallucination_detector import analyze_image   # noqa: E402

IMG_SIZE = 16
ZOOM = 8


def upscale(img, z=ZOOM):
    return img.repeat(z, axis=0).repeat(z, axis=1)


@torch.no_grad()
def euler_rollout(wrapper, x_init, steps, device):
    """Trả về list [steps+1] các frame (H,W,3) uint8, từ t=0 -> t=1."""
    dt = 1.0 / steps
    x = x_init
    frames = [to_uint8_numpy(decode(x)[0])]
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)
        v = wrapper(x, t_tensor)
        x = x + dt * v
        frames.append(to_uint8_numpy(decode(x)[0]))
    return frames


def save_progression_strip(frames, out_path, n_strip=10):
    idxs = np.linspace(0, len(frames) - 1, n_strip).astype(int)
    imgs = [upscale(frames[i]) for i in idxs]
    grid = np.concatenate(imgs, axis=1)
    PILImage.fromarray(grid).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--steps_list", default="25,50,100,200,500,1000")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    steps_list = [int(s) for s in args.steps_list.split(",")]
    out_dir = args.out_dir or os.path.join(args.case_dir, "multi_step_predict")
    os.makedirs(out_dir, exist_ok=True)

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

    x_init = torch.load(os.path.join(args.case_dir, "noise_init.pt"), weights_only=True).to(device)
    print(f"noise_init: shape={tuple(x_init.shape)}")

    # ── ảnh gốc đã lưu sẵn (steps=25 lúc sampling) để so sánh ──
    orig_final_path = os.path.join(args.case_dir, "final_image.png")
    orig_img = None
    if os.path.exists(orig_final_path):
        orig_pil = PILImage.open(orig_final_path).convert("RGB")
        orig_img = np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE), PILImage.NEAREST))
        r0 = analyze_image(orig_img)
        print(f"final_image.png (đã lưu) : hall={r0['is_hallucination']} "
              f"type={r0['hall_type']} blobs={r0['col_blobs']}")

    results = []
    for steps in steps_list:
        print(f"\n-- steps={steps} --")
        frames = euler_rollout(wrapper, x_init, steps, device)
        final_img = frames[-1]
        r = analyze_image(final_img)
        print(f"  hall={r['is_hallucination']}  type={r['hall_type']}  blobs={r['col_blobs']}")

        if orig_img is not None and steps == 25:
            diff = np.abs(orig_img.astype(int) - final_img.astype(int)).mean()
            print(f"  diff vs final_image.png (đã lưu) = {diff:.4f}  (kỳ vọng ~0, cùng steps=25)")

        PILImage.fromarray(upscale(final_img)).save(
            os.path.join(out_dir, f"final_steps{steps:04d}.png"))
        save_progression_strip(
            frames, os.path.join(out_dir, f"progression_strip_steps{steps:04d}.png"),
            n_strip=10,
        )
        results.append({"steps": steps, "img": final_img, "r": r})

    # ── Tổng hợp: 1 hàng, mỗi cột 1 số step ──
    fig, axes = plt.subplots(1, len(results), figsize=(len(results) * 2.2, 3))
    for ax, res in zip(axes, results):
        ax.imshow(upscale(res["img"]))
        r = res["r"]
        h_str = r["hall_type"] if r["is_hallucination"] else "ok"
        cb = r["col_blobs"]
        ax.set_title(f"steps={res['steps']}\n{h_str}\nt{cb['triangle']}s{cb['square']}p{cb['pentagon']}",
                     fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("red" if r["is_hallucination"] else "green")
            spine.set_linewidth(3)
        ax.axis("off")
    fig.suptitle(f"{os.path.basename(args.case_dir)} — final image theo số ODE step", fontsize=10)
    plt.tight_layout()
    summary_path = os.path.join(out_dir, "summary_by_steps.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved all files -> {out_dir}/")


if __name__ == "__main__":
    main()
