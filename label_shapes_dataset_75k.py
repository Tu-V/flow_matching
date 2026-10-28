"""
Giống label_shapes_dataset_10k.py nhưng cho nguồn simple-shapes-75k-16x16 (75K ảnh,
sinh bởi gen_simple_shapes_75k_16x16.py) -- ~10,700 ảnh/class (75000/7), đủ "~10K
mỗi class" cho train_anisotropic_p0_experiment_cond_{unet,dit}.py.

Output: shapes_75k_labeled/
    class_1/ .. class_7/, class_0_empty/, labels.csv, stats.txt

Usage:
    python label_shapes_dataset_75k.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import label_shapes_dataset as base  # noqa: E402

base.SRC_DIR = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination",
    "simple-datasets", "simple-shapes-75k-16x16",
)
base.OUT_DIR = os.path.join(REPO_ROOT, "shapes_75k_labeled")

if __name__ == "__main__":
    base.run()
