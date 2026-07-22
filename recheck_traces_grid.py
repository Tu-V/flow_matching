"""
Re-check toàn bộ case hallucination đã lưu (intermediates.pt có sẵn) bằng
hallucination_detector.py đã sửa (thêm MIN_ABS_BRIGHTNESS), so với nhãn CŨ
(đọc từ report.txt lúc sampling), rồi vẽ 1 grid tổng để kiểm tra bằng mắt.

Không sample lại gì cả — chỉ đọc lại intermediates.pt đã có.

Usage:
    python recheck_traces_grid.py \\
        --traces_dirs shapes_fm_output/hallucination_analysis/traces,shapes_fm_output_mahal_reg/hallucination_analysis/traces \\
        --out grid_recheck.png
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hallucination_detector import analyze_image   # noqa: E402


def parse_old_label(report_path):
    with open(report_path) as f:
        text = f.read()
    m = re.search(r"Hallucination type\s*:\s*(\S+)", text)
    return m.group(1) if m else "?"


def load_final_uint8(case_dir):
    inter = torch.load(os.path.join(case_dir, "intermediates.pt"), weights_only=True)
    final = inter[-1]
    img = ((final.clamp(-1, 1) + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return img


def upscale(img, z=6):
    return img.repeat(z, axis=0).repeat(z, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_dirs", required=True,
                        help="Comma-separated list of traces/ dirs")
    parser.add_argument("--out", default="grid_recheck.png")
    parser.add_argument("--row_size", type=int, default=15)
    args = parser.parse_args()

    records = []
    for traces_dir in args.traces_dirs.split(","):
        label_prefix = os.path.basename(os.path.dirname(os.path.dirname(traces_dir.rstrip("/"))))
        case_dirs = sorted(d for d in os.listdir(traces_dir)
                           if os.path.isdir(os.path.join(traces_dir, d)))
        for cd in case_dirs:
            case_path = os.path.join(traces_dir, cd)
            inter_path = os.path.join(case_path, "intermediates.pt")
            report_path = os.path.join(case_path, "report.txt")
            if not (os.path.exists(inter_path) and os.path.exists(report_path)):
                continue
            img = load_final_uint8(case_path)
            old_label = parse_old_label(report_path)
            r_new = analyze_image(img)
            new_label = r_new["hall_type"] if r_new["is_hallucination"] else "none"
            records.append({
                "src": label_prefix, "case": cd, "img": img,
                "old": old_label, "new": new_label, "blobs": r_new["col_blobs"],
            })

    print(f"Tổng {len(records)} case.")
    n_changed = sum(1 for r in records if r["old"] != r["new"])
    print(f"Đổi nhãn: {n_changed}/{len(records)}")
    from collections import Counter
    print("Old label counts:", Counter(r["old"] for r in records))
    print("New label counts:", Counter(r["new"] for r in records))
    print("Transition (old->new) counts:", Counter(f"{r['old']}->{r['new']}" for r in records))

    # ── Vẽ grid ──────────────────────────────────────────────────────────────
    row_size = args.row_size
    n = len(records)
    n_rows = (n + row_size - 1) // row_size
    fig, axes = plt.subplots(n_rows, row_size, figsize=(row_size * 1.6, n_rows * 1.9))
    axes = np.array(axes).reshape(n_rows, row_size)

    for idx, r in enumerate(records):
        ri, ci = idx // row_size, idx % row_size
        ax = axes[ri, ci]
        ax.imshow(upscale(r["img"]))
        changed = r["old"] != r["new"]
        color = "orange" if changed else ("crimson" if r["new"] != "none" else "lime")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        title = f"{r['src'][:4]}\nold:{r['old'][:6]}\nnew:{r['new'][:6]}"
        ax.set_title(title, fontsize=5.5)
        ax.set_xticks([]); ax.set_yticks([])

    for idx in range(n, n_rows * row_size):
        ri, ci = idx // row_size, idx % row_size
        axes[ri, ci].axis("off")

    fig.suptitle(
        f"Re-check {n} hallucination case (detector đã sửa MIN_ABS_BRIGHTNESS)\n"
        f"cam=đổi nhãn  đỏ=vẫn hallucinate (nhãn giữ nguyên)  xanh lá=hết hallucinate sau khi sửa\n"
        f"đổi nhãn: {n_changed}/{n}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
