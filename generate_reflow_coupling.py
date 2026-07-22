"""
Sinh coupling mới cho Rectified Flow (reflow) — biến thể "data-anchored".

Ý tưởng (Liu et al. 2022, "Flow Straightening and Fast Sampling via Rectified Flow"):
flow matching thường train với coupling ĐỘC LẬP NGẪU NHIÊN (x_0 ~ N(0,I) độc lập với
x_1 ~ p_data mỗi batch). Vì độc lập, rất nhiều đường thẳng nối (x_0, x_1) CẮT NHAU trong
không gian — tại các điểm giao nhau, target velocity của 2 cặp khác nhau lại khác nhau,
model buộc phải học ra 1 trường vector "trung bình hoá" ở đó. Đây là một giả thuyết khác
cho hallucination (bên cạnh giả thuyết translation-equivariance đã test bằng DiT/MLP):
không phải do kiến trúc, mà do bản chất quá trình train flow matching với coupling ngẫu
nhiên tạo crossing path.

Cách sinh coupling mới ở đây (KHÔNG giống hệt "reflow" gốc trong paper):
    x_1  = ẢNH THẬT trong 5k data (không phải ảnh model tự sinh ra)
    x_0  = chạy NGƯỢC ODE (t: 1 -> 0) từ x_1, dùng velocity field ĐÃ HỌC của model
           UNet baseline (shapes_fm_output/checkpoints/unet_epoch1000.pt, 9.1M)

=> 5000 cặp (x_0_i, x_1_i) MỚI, xác định 1-1 (deterministic), gần như không cắt nhau.

Khác với reflow gốc (x_1 = ảnh MODEL TỰ SINH từ x_0 ~ N(0,I) ngẫu nhiên): ở đây x_1 LUÔN
LÀ ẢNH THẬT — tránh việc model mới train tiếp trên chính hallucination mà model cũ đã
tạo ra (nếu dùng x_1 tự sinh, model mới có thể học lại/khuếch đại đúng những lỗi cũ).

Output:
    reflow_data/reflow_x0.pt   [5000, 3, 16, 16]  (nguồn mới, từ backward ODE)
    reflow_data/reflow_x1.pt   [5000, 3, 16, 16]  (ảnh thật, [-1,1], cùng thứ tự x0)

Usage:
    python generate_reflow_coupling.py --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt --steps 100
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver              # noqa: E402
from flow_matching.utils import ModelWrapper             # noqa: E402
from models.unet import UNetModel                        # noqa: E402

DATA_DIR = os.path.join(
    REPO_ROOT,
    "..",
    "neurips-2024-diffusion-model-hallucination",
    "simple-datasets",
    "simple-shapes-5k-16x16",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "reflow_data")

IMG_SIZE = 16
IN_CHANNELS = 3


class ShapesDataset(Dataset):
    """Loads all PNGs in a flat directory as 16x16 RGB tensors scaled to [-1, 1]."""

    def __init__(self, root: str):
        self.paths = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".png")]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_unet() -> UNetModel:
    """Kiến trúc UNet baseline y hệt train_shapes_fm.py (9.1M) — phải khớp checkpoint."""
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


def load_model(ckpt_path: str, device: torch.device) -> UNetModel:
    model = build_unet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {os.path.basename(ckpt_path)}  (epoch={ckpt.get('epoch','?')}, "
          f"loss={ckpt.get('loss', float('nan')):.5f}, params={n_params:,})")
    model.eval()
    return model


@torch.no_grad()
def run(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = load_model(args.ckpt, device)
    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    dataset = ShapesDataset(DATA_DIR)
    print(f"Dataset size: {len(dataset)} images")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # time_grid giảm dần 1.0 -> 0.0: ODESolver hỗ trợ tích phân ngược trực tiếp
    # ("May specify a descending time_grid to solve in the reverse direction.")
    time_grid = torch.linspace(1.0, 0.0, args.steps + 1, device=device)

    all_x0, all_x1 = [], []
    n_done = 0
    for batch in loader:
        x_1 = batch.to(device)                          # ảnh thật, [-1,1]
        x_0 = solver.sample(
            x_init=x_1,
            step_size=None,
            method="euler",
            time_grid=time_grid,
            return_intermediates=False,
        )
        all_x0.append(x_0.cpu())
        all_x1.append(x_1.cpu())
        n_done += x_1.shape[0]
        print(f"  backward ODE: [{n_done:5d}/{len(dataset)}]")

    x0 = torch.cat(all_x0, dim=0)
    x1 = torch.cat(all_x1, dim=0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x0_path = os.path.join(OUTPUT_DIR, "reflow_x0.pt")
    x1_path = os.path.join(OUTPUT_DIR, "reflow_x1.pt")
    torch.save(x0, x0_path)
    torch.save(x1, x1_path)

    print(f"\nSaved: {x0_path}  shape={tuple(x0.shape)}")
    print(f"Saved: {x1_path}  shape={tuple(x1.shape)}")
    print(f"\nSanity check — x_0 (backward ODE) vs N(0,I):")
    print(f"  mean={x0.mean().item():+.4f}  std={x0.std().item():.4f}  "
          f"(N(0,I) kỳ vọng mean=0.0000  std=1.0000)")
    print(f"  min={x0.min().item():+.4f}  max={x0.max().item():+.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default=os.path.join(REPO_ROOT, "shapes_fm_output", "checkpoints", "unet_epoch1000.pt"),
                   help="Checkpoint UNet baseline dùng để sinh coupling (default: baseline 9.1M epoch1000)")
    p.add_argument("--steps", type=int, default=100, help="Số bước Euler ODE (default: 100)")
    p.add_argument("--batch_size", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
