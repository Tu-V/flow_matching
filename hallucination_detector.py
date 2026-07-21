"""
Hallucination detector cho simple-shapes images.
Supports 16x16, 64x64, and any width (auto-scales column slices and min area).

Image layout:
  col 0  x[0 : W//3]       : triangle
  col 1  x[W//3 : 2*W//3]  : square
  col 2  x[2*W//3 : 3*W//3]: pentagon

Hallucination cases:
  1. double_col  — >= 2 shapes trong cùng 1 cột
  2. empty       — không có shape nào trong toàn bộ ảnh

Detection pipeline (per column):
  1. Contrast check  — cột flat → 0 shapes (background thuần)
  2. Otsu threshold  — tách foreground/background tự động
  3. connectedComponentsWithStats — đếm pixel thực từng blob
     (cv2.contourArea không dùng được vì shapes chỉ 3-9px,
      geometric area của outline nhỏ hơn pixel count thực)
  4. Đếm blob có pixel_count >= min_area (scales with image size)

Public API:
  analyze_image(img_uint8)         → dict
  analyze_batch(imgs_uint8)        → list[dict]
  is_hallucination(img_uint8)      → bool
  summarize(results)               → dict
"""

import cv2
import numpy as np

# ── Config (defaults for 16x16 backward compat) ───────────────────────────────
COLUMN_SLICES  = [(0, 5), (5, 10), (10, 15)]
COLUMN_NAMES   = ["triangle", "square", "pentagon"]
MIN_SHAPE_AREA = 3    # pixel count tối thiểu để tính là 1 shape thật (16x16)
MIN_CONTRAST   = 15   # max-min trong cột < ngưỡng này → cột trắng/đen thuần → 0 shape
MIN_ABS_BRIGHTNESS = 150   # cột phải có ít nhất 1 pixel đủ sáng (~trắng thật) mới
                            # được coi là CÓ THỂ chứa shape. Shape thật luôn có
                            # pixel gần 255; ảnh model sinh bị nhoè/nhiễu (noise mờ,
                            # không có shape) vẫn có thể đạt contrast tương đối > 15
                            # (vd max=51,min=6) dù không hề gần trắng — nếu chỉ xét
                            # contrast tương đối, Otsu sẽ tách nhiễu đó thành nhiều
                            # "blob" giả (double_col ảo). Check tuyệt đối này chặn
                            # trước khi vào Otsu.


def _get_col_config(img_width: int):
    """
    Compute column slices and min_area for any image width.
    Scales proportionally from 16x16 baseline (col_width=5, min_area=3).

    Returns:
        slices   : list of (x_start, x_end) for each of the 3 columns
        min_area : minimum pixel count to count as a real shape
    """
    col_width = img_width // 3
    slices = [
        (0,             col_width),
        (col_width,     2 * col_width),
        (2 * col_width, img_width),   # dùng img_width để không bỏ sót pixel cuối
    ]
    # Scale: 3 px @ 16px width → ~75 px @ 64px width
    min_area = max(3, int(3 * (img_width / 16) ** 2))
    return slices, min_area


# ── Core detection ────────────────────────────────────────────────────────────

def count_shapes_in_column(col_uint8: np.ndarray,
                           min_area: int = MIN_SHAPE_AREA) -> int:
    """
    Đếm số shape riêng biệt trong 1 cột grayscale.

    Args:
        col_uint8: (H, W) uint8, ví dụ (16, 5) or (64, 21).
        min_area : minimum pixel count to count as a real shape.

    Returns:
        Số shape đếm được (0, 1, 2, ...).
    """
    if int(col_uint8.max()) < MIN_ABS_BRIGHTNESS:
        return 0   # không có pixel nào đủ sáng để là shape thật -> chắc chắn 0 shape

    contrast = int(col_uint8.max()) - int(col_uint8.min())
    if contrast < MIN_CONTRAST:
        return 0

    _, binary = cv2.threshold(col_uint8, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    # stats[0] = background; stats[1:] = foreground blobs
    return sum(
        1 for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    )


def analyze_image(img_uint8: np.ndarray) -> dict:
    """
    Phân tích 1 ảnh (16x16, 64x64, or any size).

    Args:
        img_uint8: (H, W, 3) uint8 RGB.

    Returns:
        {
          "is_hallucination": bool,
          "hall_type": "none" | "empty" | "double_col",
          "score": int,          # 0 = clean; càng cao càng tệ
          "col_blobs": {         # số shape đếm được ở từng cột
              "triangle": int,
              "square":   int,
              "pentagon": int,
          }
        }
    """
    img_width = img_uint8.shape[1]
    col_slices, min_area = _get_col_config(img_width)

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    col_blobs = {
        name: count_shapes_in_column(gray[:, c0:c1], min_area)
        for (c0, c1), name in zip(col_slices, COLUMN_NAMES)
    }

    total_shapes = sum(col_blobs.values())
    double_col   = any(n >= 2 for n in col_blobs.values())
    empty_image  = (total_shapes == 0)
    is_hall      = double_col or empty_image

    score = sum(max(0, col_blobs[n] - 1) for n in COLUMN_NAMES)
    if empty_image:
        score += 1

    hall_type = "empty" if empty_image else ("double_col" if double_col else "none")

    return {
        "is_hallucination": is_hall,
        "hall_type":        hall_type,
        "score":            score,
        "col_blobs":        col_blobs,
    }


def analyze_batch(imgs_uint8: np.ndarray) -> list:
    """
    Phân tích batch ảnh.

    Args:
        imgs_uint8: (B, 16, 16, 3) uint8 RGB.

    Returns:
        List[dict] — mỗi phần tử là kết quả của analyze_image().
    """
    return [analyze_image(imgs_uint8[i]) for i in range(len(imgs_uint8))]


def is_hallucination(img_uint8: np.ndarray) -> bool:
    """Shorthand: chỉ trả về bool."""
    return analyze_image(img_uint8)["is_hallucination"]


# ── Summary helper ────────────────────────────────────────────────────────────

def summarize(results: list) -> dict:
    """
    Tổng hợp list kết quả từ analyze_image / analyze_batch.

    Returns:
        {
          "n_total":      int,
          "n_hall":       int,
          "n_empty":      int,
          "n_double_col": int,
          "n_normal":     int,
          "hall_rate":    float,   # 0.0 – 1.0
          "col_counts":   {name: {"0": int, "1": int, "2+": int}},
          "hall_indices": List[int],
          "empty_indices":      List[int],
          "double_col_indices": List[int],
        }
    """
    n = len(results)
    hall_idx   = [i for i, r in enumerate(results) if r["is_hallucination"]]
    empty_idx  = [i for i, r in enumerate(results) if r["hall_type"] == "empty"]
    double_idx = [i for i, r in enumerate(results) if r["hall_type"] == "double_col"]

    col_counts = {}
    for name in COLUMN_NAMES:
        blobs = [r["col_blobs"][name] for r in results]
        col_counts[name] = {
            "0":  sum(1 for b in blobs if b == 0),
            "1":  sum(1 for b in blobs if b == 1),
            "2+": sum(1 for b in blobs if b >= 2),
        }

    return {
        "n_total":            n,
        "n_hall":             len(hall_idx),
        "n_empty":            len(empty_idx),
        "n_double_col":       len(double_idx),
        "n_normal":           n - len(hall_idx),
        "hall_rate":          len(hall_idx) / n if n else 0.0,
        "col_counts":         col_counts,
        "hall_indices":       hall_idx,
        "empty_indices":      empty_idx,
        "double_col_indices": double_idx,
    }
