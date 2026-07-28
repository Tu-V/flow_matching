"""
Kiểm chứng giả thuyết: hallucination "2 shape trong 1 cột" có phải do SỐ BƯỚC ODE
(Euler) không đủ mịn (sai số rời rạc hoá) hay không — bằng cách chạy lại ĐÚNG 1
noise_init.pt đã biết gây hallucination ở 100 bước, với số bước TĂNG DẦN
(100/200/300/500/700/1000), xem hallucination có biến mất khi discretization mịn
hơn hay vẫn giữ nguyên (=> không phải do thiếu bước, mà là bản chất continuous field
thật sự có 2 điểm hút).

Vì ODE là deterministic (cùng x_0 -> cùng nghiệm continuous "thật"), số bước Euler chỉ
là XẤP XỈ nghiệm đó. Nếu hallucination biến mất khi steps tăng -> đúng là do sai số
rời rạc hoá. Nếu vẫn y nguyên (2 shape) dù steps=1000 -> hallucination là thuộc tính
thật của velocity field đã học, không phải lỗi số học.

Usage:
    python verify_step_count_hypothesis.py \
        --case_dir shapes_fm_output/hallucination_analysis/traces/HALL_case_0010_idx9235 \
        --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver          # noqa: E402
from flow_matching.utils import ModelWrapper         # noqa: E402
from hallucination_detector import analyze_image     # noqa: E402
from models.unet import UNetModel                    # noqa: E402

IMG_SIZE = 16
SCALE = 8


def build_unet() -> UNetModel:
    """Kien truc UNet baseline (9.1M) — train_shapes_fm.py."""
    return UNetModel(
        in_channels=3, model_channels=64, out_channels=3,
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


def decode(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(-1, 1) + 1) / 2


def to_uint8_numpy(img_01: torch.Tensor):
    import numpy as np
    return (img_01.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def add_red_dividers(img_01: torch.Tensor, scale: int = SCALE) -> torch.Tensor:
    up = F.interpolate(img_01.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)
    for x in [5 * scale, 10 * scale]:
        up[0, :, x:x + 2] = 1.0
        up[1, :, x:x + 2] = 0.0
        up[2, :, x:x + 2] = 0.0
    return up


@torch.no_grad()
def run(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = build_unet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.ckpt}  (epoch={ckpt.get('epoch','?')})")
    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    x0 = torch.load(os.path.join(args.case_dir, "noise_init.pt"), weights_only=True).to(device)
    print(f"noise_init shape: {tuple(x0.shape)}")

    out_dir = os.path.join(args.case_dir, "step_count_experiment")
    os.makedirs(out_dir, exist_ok=True)

    imgs, results = [], []
    for steps in args.step_list:
        time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
        x_final = solver.sample(
            x_init=x0, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        img_01 = decode(x_final[0])
        analysis = analyze_image(to_uint8_numpy(img_01))
        imgs.append(add_red_dividers(img_01.cpu()))
        results.append({"steps": steps, "analysis": analysis})
        print(f"  steps={steps:5d}  col_blobs={analysis['col_blobs']}  "
              f"type={analysis['hall_type']}  score={analysis['score']}")

    grid = make_grid(torch.stack(imgs), nrow=len(args.step_list), padding=4, pad_value=0.5)
    grid_path = os.path.join(out_dir, "compare_step_counts.png")
    save_image(grid, grid_path)
    print(f"\nSaved: {grid_path}")

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"Case: {args.case_dir}\n")
        f.write(f"Checkpoint: {args.ckpt}\n\n")
        f.write(f"{'steps':>8s}  {'col_blobs':40s}  {'type':12s}  {'score':>5s}\n")
        for r in results:
            f.write(f"{r['steps']:8d}  {str(r['analysis']['col_blobs']):40s}  "
                    f"{r['analysis']['hall_type']:12s}  {r['analysis']['score']:5d}\n")

        # tom tat: hallucination con giu nguyen hay bien mat khi steps tang
        first_type = results[0]['analysis']['hall_type']
        last_type = results[-1]['analysis']['hall_type']
        f.write(f"\nsteps={results[0]['steps']}: {first_type}   ->   "
                f"steps={results[-1]['steps']}: {last_type}\n")
        if first_type != "none" and last_type == "none":
            f.write("=> Hallucination BIEN MAT khi tang so buoc -> co the do sai so roi rac hoa.\n")
        elif first_type != "none" and last_type != "none":
            f.write("=> Hallucination VAN GIU NGUYEN du tang so buoc -> KHONG phai do sai so roi rac "
                    "hoa, la thuoc tinh that cua velocity field da hoc.\n")
    print(f"Saved: {report_path}")

    print(f"\n{'='*70}")
    print(f"TOM TAT")
    print(f"{'='*70}")
    for r in results:
        print(f"  steps={r['steps']:5d}  ->  {r['analysis']['hall_type']:12s}  "
              f"col_blobs={r['analysis']['col_blobs']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str,
                   default=os.path.join(REPO_ROOT, "shapes_fm_output", "checkpoints", "unet_epoch1000.pt"))
    p.add_argument("--step_list", type=int, nargs="+", default=[100, 200, 300, 500, 700, 1000])
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
