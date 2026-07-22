"""
Gán nhãn cho bộ simple-shapes-5k-16x16 dựa trên CỘT NÀO có shape (dùng lại
hallucination_detector.analyze_image để đếm shape từng cột — không train gì cả,
chỉ xử lý dữ liệu để chuẩn bị cho bước train conditioning sau này).

Quy ước cột (khớp hallucination_detector.py): cột 1 = triangle, cột 2 = square,
cột 3 = pentagon.

Định nghĩa nhãn (dựa trên TẬP CỘT có >=1 shape, không quan tâm số lượng chính xác):
    class 1: chỉ cột 1
    class 2: chỉ cột 2
    class 3: chỉ cột 3
    class 4: cột 1 và cột 2
    class 5: cột 1 và cột 3
    class 6: cột 2 và cột 3
    class 7: cả 3 cột
    class 0: KHÔNG cột nào có shape (ảnh rỗng) — không nằm trong 7 lớp user yêu cầu,
             tách riêng để không làm sai lệch thống kê nếu dataset gốc có ảnh rỗng.

Output: shapes_5k_labeled/
    class_1/ .. class_7/   : ảnh copy nguyên bản (PNG gốc, không resize/convert)
    class_0_empty/         : (nếu có) ảnh không có shape nào — anomaly, tách riêng
    labels.csv             : filename, class, triangle_n, square_n, pentagon_n, double_col_flag
    stats.txt               : thống kê số lượng mỗi lớp

Usage:
    python label_shapes_dataset.py
"""

import csv
import os
import shutil
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from hallucination_detector import analyze_image, COLUMN_NAMES   # noqa: E402

SRC_DIR = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination",
    "simple-datasets", "simple-shapes-5k-16x16",
)
OUT_DIR = os.path.join(REPO_ROOT, "shapes_5k_labeled")

# (triangle_present, square_present, pentagon_present) -> class
CLASS_MAP = {
    (True,  False, False): 1,
    (False, True,  False): 2,
    (False, False, True):  3,
    (True,  True,  False): 4,
    (True,  False, True):  5,
    (False, True,  True):  6,
    (True,  True,  True):  7,
    (False, False, False): 0,   # rỗng — anomaly, không thuộc 7 lớp yêu cầu
}

CLASS_DESC = {
    0: "empty (KHONG co shape nao - anomaly, khong nam trong 7 class yeu cau)",
    1: "chi cot 1 (triangle)",
    2: "chi cot 2 (square)",
    3: "chi cot 3 (pentagon)",
    4: "cot 1 + cot 2 (triangle + square)",
    5: "cot 1 + cot 3 (triangle + pentagon)",
    6: "cot 2 + cot 3 (square + pentagon)",
    7: "ca 3 cot (triangle + square + pentagon)",
}


def run():
    paths = sorted(
        os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR) if f.lower().endswith(".png")
    )
    print(f"Nguồn: {SRC_DIR}")
    print(f"Số ảnh: {len(paths)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for c in range(8):
        os.makedirs(os.path.join(OUT_DIR, f"class_{c}_empty" if c == 0 else f"class_{c}"), exist_ok=True)

    rows = []
    class_counts = {c: 0 for c in range(8)}
    n_double_col_anomaly = 0

    for p in paths:
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.uint8)   # (16,16,3)
        analysis = analyze_image(arr)
        blobs = analysis["col_blobs"]   # {"triangle": n, "square": n, "pentagon": n}

        present = tuple(blobs[name] >= 1 for name in COLUMN_NAMES)   # (col1,col2,col3)
        cls = CLASS_MAP[present]
        has_double = any(n >= 2 for n in blobs.values())
        if has_double:
            n_double_col_anomaly += 1

        class_counts[cls] += 1
        fname = os.path.basename(p)
        dst_subdir = f"class_{cls}_empty" if cls == 0 else f"class_{cls}"
        shutil.copy2(p, os.path.join(OUT_DIR, dst_subdir, fname))

        rows.append({
            "filename": fname,
            "class": cls,
            "triangle_n": blobs["triangle"],
            "square_n": blobs["square"],
            "pentagon_n": blobs["pentagon"],
            "double_col_flag": int(has_double),
        })

    # ── labels.csv ──
    csv_path = os.path.join(OUT_DIR, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "class", "triangle_n", "square_n", "pentagon_n", "double_col_flag",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # ── stats.txt ──
    stats_path = os.path.join(OUT_DIR, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Nguồn: {SRC_DIR}\n")
        f.write(f"Tổng số ảnh: {len(paths)}\n\n")
        f.write("Định nghĩa nhãn (dựa trên tập cột có >=1 shape):\n")
        for c in range(8):
            f.write(f"  class {c}: {CLASS_DESC[c]}\n")
        f.write("\nThống kê số lượng mỗi lớp:\n")
        for c in range(8):
            pct = 100 * class_counts[c] / len(paths) if paths else 0
            f.write(f"  class {c}: {class_counts[c]:5d}  ({pct:5.2f}%)\n")
        f.write(f"\nAnomaly — ảnh có >=2 shape trong cùng 1 cột (double-col): {n_double_col_anomaly}\n")
        f.write("  (vẫn được gán nhãn theo tập cột có mặt như bình thường, chỉ đánh dấu double_col_flag=1 trong labels.csv)\n")

    print("\nThống kê số lượng mỗi lớp:")
    for c in range(8):
        pct = 100 * class_counts[c] / len(paths) if paths else 0
        tag = " (anomaly, không thuộc 7 class yêu cầu)" if c == 0 else ""
        print(f"  class {c}: {class_counts[c]:5d}  ({pct:5.2f}%){tag}")
    print(f"\nAnomaly double-col (>=2 shape/cột): {n_double_col_anomaly}")

    print(f"\nOutput -> {OUT_DIR}")
    print(f"  labels.csv -> {csv_path}")
    print(f"  stats.txt  -> {stats_path}")


if __name__ == "__main__":
    run()
