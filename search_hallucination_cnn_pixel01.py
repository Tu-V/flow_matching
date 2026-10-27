"""
Sample liên tục từ checkpoint cnn_pixel01_isotropic_final.pt (UNet, pixel range [0,1],
isotropic p0) cho đến khi tìm được CẢ 2 loại hallucination: "empty" (ảnh đen/trống) và
"double_col" (2 shape trong cùng 1 cột). Dừng ngay khi tìm thấy mỗi loại (lưu ngay lập
tức), tiếp tục tìm loại còn lại nếu chưa đủ cả 2. Lưu noise x0 (nguồn gây hallucination)
+ ảnh cuối x1, dưới cả 2 dạng .npy (giá trị float gốc, KHÔNG rescale) và .png (giá trị
pixel giữ nguyên [0,1] -> *255 chuẩn, KHÔNG auto-contrast) để REPRODUCE lại được chính
xác sau này (chỉ cần load lại .npy, feed vào ODESolver y hệt, sẽ ra lại đúng kết quả).

Usage:
    python search_hallucination_cnn_pixel01.py
    python search_hallucination_cnn_pixel01.py --max_samples 5000000 --batch_size 1000
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver                                  # noqa: E402
from models.unet import UNetModel                                            # noqa: E402
from flow_matching.utils import ModelWrapper                                 # noqa: E402
from hallucination_detector import analyze_batch                             # noqa: E402

IN_CHANNELS = 1
IMG_SIZE = 64
CKPT_PATH = os.path.join(
    REPO_ROOT, "shape_exp", "64_64", "cnn_pixel01_isotropic_64x64_output",
    "cnn_pixel01_isotropic_final.pt",
)
OUT_DIR = os.path.join(
    REPO_ROOT, "shape_exp", "64_64", "cnn_pixel01_isotropic_64x64_output", "hallucination_search"
)


def build_unet() -> UNetModel:
    return UNetModel(
        in_channels=IN_CHANNELS, model_channels=64, out_channels=IN_CHANNELS,
        num_res_blocks=3, attention_resolutions=(2,), dropout=0.1,
        channel_mult=(1, 2, 2), conv_resample=True, dims=2,
        num_classes=None, use_checkpoint=False, num_heads=1,
        num_head_channels=-1, num_heads_upsample=-1,
        use_scale_shift_norm=True, resblock_updown=False,
        use_new_attention_order=True, with_fourier_features=False,
    )


class UNetVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t, extra=extras)


def save_example(tag: str, x0: torch.Tensor, x1: torch.Tensor, global_idx: int):
    """x0, x1: [1, IMG_SIZE, IMG_SIZE] tensor CPU. Lưu .npy (float32 gốc, KHÔNG rescale)
    và .png (giữ nguyên [0,1] -> *255 chuẩn, KHÔNG auto-contrast)."""
    d = os.path.join(OUT_DIR, tag)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"noise_x0_g{global_idx}.npy"), x0.numpy())
    np.save(os.path.join(d, f"final_x1_g{global_idx}.npy"), x1.numpy())

    x1_01 = x1.clamp(0, 1)
    x1_u8 = (x1_01[0].numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(x1_u8, "L").save(os.path.join(d, f"final_x1_g{global_idx}.png"))

    # x0 KHONG nam trong [0,1] (la N(0,I)) -- luu PNG rieng de xem truc quan, ghi ro
    # trong ten file la "raw_clip01" de khong nham voi gia tri thuc (npy moi la chuan)
    x0_u8 = (x0.clamp(0, 1)[0].numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(x0_u8, "L").save(os.path.join(d, f"noise_x0_g{global_idx}_raw_clip01.png"))

    print(f"  !!! Đã lưu {tag} (global_idx={global_idx}) -> {d}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=20_000_000)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--log_every_batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    ck = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = build_unet().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"Loaded {CKPT_PATH}  (step={ck['step']}, pixel_range={ck.get('pixel_range')})")

    os.makedirs(OUT_DIR, exist_ok=True)
    solver = ODESolver(velocity_model=UNetVelocityWrapper(model))
    tg = torch.linspace(0.0, 1.0, a.sample_steps + 1, device=device)

    found_empty = False
    found_double = False
    n_done = 0
    t0 = time.time()
    torch.manual_seed(a.seed)

    with torch.no_grad():
        batch_i = 0
        while n_done < a.max_samples and not (found_empty and found_double):
            batch_i += 1
            x0 = torch.randn(a.batch_size, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
            x1 = solver.sample(x_init=x0, step_size=None, method="euler",
                                time_grid=tg, return_intermediates=False)
            x01 = x1.clamp(0, 1)
            rgb = x01.repeat(1, 3, 1, 1)
            imgs_np = (rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            analyses = analyze_batch(imgs_np)

            for k, an in enumerate(analyses):
                if an["is_hallucination"] and an["hall_type"] == "empty" and not found_empty:
                    save_example("empty", x0[k:k+1].cpu(), x1[k:k+1].cpu(), n_done + k)
                    found_empty = True
                if an["is_hallucination"] and an["hall_type"] == "double_col" and not found_double:
                    save_example("double_col", x0[k:k+1].cpu(), x1[k:k+1].cpu(), n_done + k)
                    found_double = True
                if found_empty and found_double:
                    break

            n_done += a.batch_size
            if batch_i % a.log_every_batches == 0:
                print(f"  đã sample {n_done:,}  (found_empty={found_empty}  "
                      f"found_double_col={found_double})  elapsed={time.time()-t0:.0f}s")

    print(f"\n{'='*70}")
    if found_empty and found_double:
        print(f"ĐÃ TÌM THẤY CẢ 2 LOẠI sau {n_done:,} sample ({time.time()-t0:.0f}s)")
    else:
        print(f"DỪNG sau {n_done:,} sample (đạt max_samples) -- "
              f"found_empty={found_empty}  found_double_col={found_double}")
    print(f"{'='*70}")

    with open(os.path.join(OUT_DIR, "search_log.txt"), "a") as f:
        f.write(f"seed={a.seed}  n_done={n_done}  found_empty={found_empty}  "
                f"found_double_col={found_double}  elapsed={time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
