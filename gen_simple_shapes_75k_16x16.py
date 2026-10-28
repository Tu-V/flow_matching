"""
Sinh thêm 1 bộ simple-shapes 16x16 LỚN HƠN (75K ảnh, thay vì 10K) để dùng cho
train_anisotropic_p0_experiment_cond_{unet,dit}.py (guided/CFG) -- 10K ảnh tổng chia
cho 7 class (mỗi class ~14.3%) chỉ cho ~1400 ảnh/class, hơi mỏng cho conditioning.
75K ảnh tổng -> ~10,700 ảnh/class (75000/7), đủ "~10K mỗi class" theo yêu cầu.

Tái dùng NGUYÊN các hàm vẽ (draw_triangle/draw_square/draw_pentagon, SHAPE_DRAWERS,
COLUMN_W/H, MARGIN, P_SHAPE) từ gen_simple_shapes_16x16.py (file gốc chỉ có logic
inline trong `if __name__=="__main__"`, không có hàm run() để gọi lại nên không thể
import trực tiếp -- import module để lấy lại các hằng số/hàm vẽ, rồi viết lại đúng
loop gốc với NUM_IMAGES_TO_GENERATE và SAVE_DIR khác).

Output: ../neurips-2024-diffusion-model-hallucination/simple-datasets/simple-shapes-75k-16x16/
    {i}.png (0..74999), meta_data.npz

Usage:
    python gen_simple_shapes_75k_16x16.py
"""

import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "shapes",
)
sys.path.insert(0, GEN_DIR)

import gen_simple_shapes_16x16 as gen   # noqa: E402  (tái dùng drawer functions + constants)

NUM_IMAGES_TO_GENERATE = 75_000
SAVE_DIR = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination",
    "simple-datasets", "simple-shapes-75k-16x16",
)


def run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    meta = {}
    count = 0
    attempts = 0

    while count < NUM_IMAGES_TO_GENERATE:
        attempts += 1
        cols, shapes_present = [], []

        for drawer in gen.SHAPE_DRAWERS:
            col_img = Image.new("L", (gen.COLUMN_W, gen.COLUMN_H), color=0)
            if random.random() <= gen.P_SHAPE:
                if drawer(ImageDraw.Draw(col_img), gen.COLUMN_W, gen.COLUMN_H):
                    shapes_present.append(drawer.__name__.replace("draw_", ""))
            cols.append(np.array(col_img))

        if not shapes_present:   # bỏ ảnh toàn đen, giống generator gốc
            continue

        canvas = np.zeros((gen.COLUMN_H, 16), dtype=np.uint8)
        canvas[:, : gen.COLUMN_W * 3] = np.concatenate(cols, axis=1)

        Image.fromarray(canvas).save(os.path.join(SAVE_DIR, f"{count}.png"))
        meta[count] = shapes_present
        count += 1

        if count % 10000 == 0:
            print(f"  ... {count}/{NUM_IMAGES_TO_GENERATE}")

    np.savez(os.path.join(SAVE_DIR, "meta_data.npz"), meta)

    print(f"\nGenerated {count} images in {attempts} attempts -> {SAVE_DIR}")
    print(f"  with triangle : {sum(1 for v in meta.values() if 'triangle' in v)}")
    print(f"  with square   : {sum(1 for v in meta.values() if 'square'   in v)}")
    print(f"  with pentagon : {sum(1 for v in meta.values() if 'pentagon' in v)}")
    print(f"  1-shape images: {sum(1 for v in meta.values() if len(v)==1)}")
    print(f"  2-shape images: {sum(1 for v in meta.values() if len(v)==2)}")
    print(f"  3-shape images: {sum(1 for v in meta.values() if len(v)==3)}")


if __name__ == "__main__":
    run()
