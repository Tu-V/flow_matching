"""
Giống label_shapes_dataset.py nhưng cho nguồn simple-shapes-10k-16x16 (10K ảnh) --
chuẩn bị dữ liệu cho train_anisotropic_p0_experiment_cond_{unet,dit}.py (guided
flow matching: so sánh isotropic vs anisotropic p0, có conditioning theo class).

Nhãn (giống bản 5k, dựa trên TẬP CỘT có >=1 shape):
    class 1: chỉ cột 1 (triangle)         class 5: cột 1 + cột 3
    class 2: chỉ cột 2 (square)           class 6: cột 2 + cột 3
    class 3: chỉ cột 3 (pentagon)         class 7: cả 3 cột
    class 4: cột 1 + cột 2
    class 0: KHÔNG cột nào có shape (rỗng) -- anomaly, tách riêng, không nằm trong 7 lớp.

Output: shapes_10k_labeled/
    class_1/ .. class_7/, class_0_empty/, labels.csv, stats.txt

Usage:
    python label_shapes_dataset_10k.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import label_shapes_dataset as base  # noqa: E402

base.SRC_DIR = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination",
    "simple-datasets", "simple-shapes-10k-16x16",
)
base.OUT_DIR = os.path.join(REPO_ROOT, "shapes_10k_labeled")

if __name__ == "__main__":
    base.run()
