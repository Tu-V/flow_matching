"""
Rectified Flow (reflow) ĐÚNG BẢN GỐC (Liu et al. 2022) về cách sinh coupling ON-THE-FLY
(không đóng băng 1 lần như train_shapes_fm_reflow.py), NHƯNG đảo chiều so với paper gốc
theo yêu cầu:

    Paper gốc : x_0 ~ N(0,I) (sample mới), CHẠY XUÔI ODE (t:0->1) qua model đang có để
                sinh x_1 = data giả -> couple (x_0, x_1_giả). Rủi ro: x_1 giả có thể
                CHÍNH LÀ ảnh hallucination (model cũ vẫn hallucinate), train tiếp trên đó
                sẽ khuếch đại lại đúng lỗi cũ.
    Script này: x_1 = ẢNH THẬT (luôn lấy từ 5k data thật, không đổi), CHẠY NGƯỢC ODE
                (t:1->0, 100 bước) qua model ĐANG ĐƯỢC TRAIN (weight cập nhật liên tục)
                để sinh x_0 tương ứng -> couple (x_0_mới, x_1_thật). x_1 luôn sạch vì
                luôn là data thật.

"On-the-fly" ở đây nghĩa là: coupling KHÔNG đóng băng 1 lần từ 1 checkpoint cũ (khác
train_shapes_fm_reflow.py) mà được SINH LẠI ĐỊNH KỲ (mỗi --recouple_every epoch, mặc
định mỗi epoch) bằng ĐÚNG weight hiện tại của model đang train — giải quyết vấn đề
"overfit vào 5000 anchor cố định" đã quan sát thấy ở bản đóng-băng-1-lần (hallucination
tăng, không giảm). Weight cập nhật liên tục -> "điểm neo" x_0 cũng trôi theo mỗi lần
recouple -> model không còn khớp cứng vào 1 tập 5000 điểm bất biến nữa.

CẢNH BÁO CHI PHÍ: sinh lại coupling cho toàn bộ 5000 ảnh cần 100 bước Euler ODE (không
gradient) mỗi lần recouple. Với --recouple_every 1 (mỗi epoch), tổng chi phí ODE-backward
cho riêng việc sinh coupling gần bằng chi phí generate 5000*epochs ảnh — ĐẮT HƠN NHIỀU so
với 1 bước train thường. Cân nhắc tăng --recouple_every (vd 5, 10) nếu quá chậm.

Usage:
    python train_shapes_fm_reflow_online.py --epochs 1000 --batch_size 128 --lr 1e-4 \
        --recouple_every 1 --recouple_steps 100

    # Resume
    python train_shapes_fm_reflow_online.py --resume latest --epochs 1000

--epochs luôn là TỔNG số epoch mục tiêu (không phải số epoch thêm vào).
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath           # noqa: E402
from flow_matching.solver import ODESolver               # noqa: E402
from flow_matching.utils import ModelWrapper             # noqa: E402
from models.unet import UNetModel                        # noqa: E402

DATA_DIR = os.path.join(
    REPO_ROOT,
    "..",
    "neurips-2024-diffusion-model-hallucination",
    "simple-datasets",
    "simple-shapes-5k-16x16",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_reflow_online_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

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
    """Kiến trúc UNet baseline y hệt train_shapes_fm.py (9.1M) — phải khớp init_ckpt."""
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


@torch.no_grad()
def regenerate_coupling(model, solver, x1_all, device, steps, batch_size):
    """
    Chạy ngược ODE (t:1->0, `steps` bước Euler) qua weight HIỆN TẠI của `model` cho
    toàn bộ x1_all (5000 ảnh thật) để sinh x_0 mới. model.eval() để tắt dropout khi
    tích phân (bật lại model.train() trước khi return để không ảnh hưởng training loop).
    """
    model.eval()
    time_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)
    all_x0 = []
    for start in range(0, x1_all.shape[0], batch_size):
        x1_b = x1_all[start:start + batch_size].to(device)
        x0_b = solver.sample(
            x_init=x1_b, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        all_x0.append(x0_b.cpu())
    model.train()
    return torch.cat(all_x0, dim=0)


def train(args):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Data (x_1 pool cố định — 5000 ảnh thật, KHÔNG BAO GIỜ đổi) ──
    dataset = ShapesDataset(DATA_DIR)
    print(f"Dataset size: {len(dataset)} images")
    x1_all = torch.stack([dataset[i] for i in range(len(dataset))])  # [5000,3,16,16] cpu

    # ── Model ──
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet (reflow online): params={n_params:,}")

    path = CondOTProbPath()
    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    start_epoch = 1
    force_recouple = True   # luôn recouple ngay lần đầu (train mới hoặc sau resume)
    if args.resume:
        ckpt_path = _resolve_resume(args.resume)
        print(f"Resuming reflow-online training from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {ckpt['epoch']}  loss={ckpt.get('loss', float('nan')):.5f}")
        if start_epoch > args.epochs:
            print(f"Đã train đủ {args.epochs} epochs rồi. Tăng --epochs nếu muốn train thêm.")
            return
    else:
        print(f"Init từ baseline checkpoint: {args.init_ckpt}")
        init_ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(init_ckpt["model_state_dict"])
        print(f"  init epoch={init_ckpt.get('epoch','?')}  "
              f"loss={init_ckpt.get('loss', float('nan')):.5f}")

    current_x0 = None
    for epoch in range(start_epoch, args.epochs + 1):
        if force_recouple or (epoch - start_epoch) % args.recouple_every == 0:
            t0 = time.time()
            current_x0 = regenerate_coupling(
                model, solver, x1_all, device, args.recouple_steps, args.recouple_batch_size
            )
            dt = time.time() - t0
            print(f"  [epoch {epoch}] recoupled ({args.recouple_steps} bước ODE ngược, "
                  f"{dt:.1f}s)  x0 mean={current_x0.mean():.4f} std={current_x0.std():.4f}")
            force_recouple = False

        loader = DataLoader(
            TensorDataset(current_x0, x1_all),
            batch_size=args.batch_size, shuffle=True, drop_last=True,
        )

        model.train()
        epoch_loss = 0.0
        for x_0, x_1 in loader:
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            B = x_1.shape[0]

            t = torch.rand(B, device=device)
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            u_pred = model(x_t, t, extra={})
            loss = torch.pow(u_pred - u_t, 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if epoch % args.log_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.5f}  lr={lr:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(CKPT_DIR, f"unet_reflow_online_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "arch": "unet_reflow_online",
                    "init_ckpt": args.init_ckpt,
                    "recouple_every": args.recouple_every,
                    "recouple_steps": args.recouple_steps,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            _sample_and_save(model, device, epoch, n_samples=64, steps=args.sample_steps)

    print("Training complete.")
    print(f"Checkpoints → {CKPT_DIR}")
    print(f"Samples     → {SAMPLE_DIR}")


@torch.no_grad()
def _sample_and_save(model, device, epoch, n_samples=64, steps=100):
    """x_init ~ N(0,I) tươi — đúng lý thuyết Rectified Flow (giữ nguyên marginal 2 đầu)."""
    model.eval()

    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    x_init = torch.randn(n_samples, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    x_gen = solver.sample(
        x_init=x_init, step_size=None, method="euler",
        time_grid=time_grid, return_intermediates=False,
    )

    x_gen = (x_gen.clamp(-1, 1) + 1) / 2
    out_path = os.path.join(SAMPLE_DIR, f"samples_epoch{epoch:04d}.png")
    save_image(x_gen, out_path, nrow=8, padding=1)
    print(f"  Samples saved: {out_path}")

    model.train()


def _resolve_resume(resume: str) -> str:
    if resume == "latest":
        import glob
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
        if not ckpts:
            raise FileNotFoundError(f"Không có checkpoint trong {CKPT_DIR}")
        return ckpts[-1]
    return resume


def parse_args():
    p = argparse.ArgumentParser(
        description="Rectified Flow (reflow) on-the-fly, coupling sinh lại định kỳ từ weight hiện tại"
    )
    p.add_argument("--epochs",       type=int,   default=1000,
                   help="Tổng số epoch mục tiêu (default: 1000, khớp baseline)")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Nhỏ hơn baseline (2e-4) vì đây là fine-tune/warm-start")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path checkpoint reflow-online để resume, hoặc 'latest' (bỏ qua --init_ckpt)")
    p.add_argument("--init_ckpt",    type=str,
                   default=os.path.join(REPO_ROOT, "shapes_fm_output", "checkpoints", "unet_epoch1000.pt"),
                   help="Checkpoint UNet baseline để khởi tạo trọng số (warm-start)")
    # ── Recoupling on-the-fly ──
    p.add_argument("--recouple_every", type=int, default=1,
                   help="Sinh lại coupling (x_0 mới từ backward ODE qua weight hiện tại) mỗi "
                        "N epoch (default: 1 = mỗi epoch, ĐÚNG nghĩa on-the-fly nhưng RẤT TỐN "
                        "chi phí; tăng lên vd 5/10 nếu cần chạy nhanh hơn)")
    p.add_argument("--recouple_steps", type=int, default=100,
                   help="Số bước Euler ODE ngược để sinh coupling (default: 100)")
    p.add_argument("--recouple_batch_size", type=int, default=512,
                   help="Batch size khi chạy backward ODE sinh coupling (default: 512)")
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--save_every",   type=int,   default=50)
    p.add_argument("--sample_every", type=int,   default=50)
    p.add_argument("--sample_steps", type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
