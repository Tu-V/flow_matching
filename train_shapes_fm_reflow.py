"""
Train Rectified Flow (reflow) — retrain UNet trên coupling MỚI sinh bởi
generate_reflow_coupling.py, khởi tạo trọng số (warm-start) từ đúng model UNet
baseline đã dùng để sinh coupling đó.

Khác biệt DUY NHẤT so với train_shapes_fm.py (baseline, coupling độc lập ngẫu nhiên
x_0 ~ N(0,I) mới mỗi batch):
    - x_0 KHÔNG còn sample ngẫu nhiên độc lập mỗi batch nữa. Thay vào đó dùng ĐÚNG cặp
      (x_0_i, x_1_i) đã lưu sẵn trong reflow_data/ (x_0_i = kết quả chạy ngược ODE từ
      ảnh thật x_1_i qua model baseline — xem generate_reflow_coupling.py).
    - Model khởi tạo từ trọng số baseline (--init_ckpt), không train từ đầu.
Mọi thứ khác (kiến trúc UNet 9.1M, CondOTProbPath, ODESolver, optimizer, lịch
checkpoint/sample) giữ nguyên để so sánh công bằng.

Giả thuyết cần test: coupling độc lập ngẫu nhiên tạo ra nhiều đường thẳng (x_0,x_1)
CẮT NHAU trong không gian -> model phải học 1 velocity field "trung bình hoá" tại chỗ
giao nhau -> có thể là nguồn gốc hallucination (không liên quan gì tới kiến trúc, vì
DiT/MLP/UNet-large đều vẫn hallucinate). Coupling reflow gần như không cắt nhau (paths
thẳng hơn hẳn) -> nếu hallucination giảm mạnh sau reflow, giả thuyết này được củng cố.

Usage:
    # Bước 1 (đã chạy riêng): python generate_reflow_coupling.py
    # Bước 2:
    python train_shapes_fm_reflow.py --epochs 1000 --batch_size 128 --lr 1e-4

    # Resume reflow training đang dang dở (KHÁC với --init_ckpt, chỉ dùng 1 lần đầu)
    python train_shapes_fm_reflow.py --resume latest --epochs 1000

--epochs luôn là TỔNG số epoch mục tiêu (không phải số epoch thêm vào).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath           # noqa: E402
from flow_matching.solver import ODESolver               # noqa: E402
from flow_matching.utils import ModelWrapper             # noqa: E402
from models.unet import UNetModel                        # noqa: E402

REFLOW_DATA_DIR = os.path.join(REPO_ROOT, "reflow_data")
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_reflow_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

IMG_SIZE = 16
IN_CHANNELS = 3


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

    # ── Reflow coupling (x_0, x_1) cố định, đã sinh sẵn ──
    x0_path = os.path.join(REFLOW_DATA_DIR, "reflow_x0.pt")
    x1_path = os.path.join(REFLOW_DATA_DIR, "reflow_x1.pt")
    if not (os.path.exists(x0_path) and os.path.exists(x1_path)):
        raise FileNotFoundError(
            f"Chưa có reflow coupling. Chạy generate_reflow_coupling.py trước.\n"
            f"  thiếu: {x0_path if not os.path.exists(x0_path) else x1_path}"
        )
    reflow_x0 = torch.load(x0_path, weights_only=True)
    reflow_x1 = torch.load(x1_path, weights_only=True)
    assert reflow_x0.shape == reflow_x1.shape
    print(f"Reflow coupling: {reflow_x0.shape[0]} cặp (x_0, x_1) cố định  "
          f"(x_0 mean={reflow_x0.mean():.4f} std={reflow_x0.std():.4f})")

    dataset = TensorDataset(reflow_x0, reflow_x1)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ── Model ──
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet (reflow): params={n_params:,}")

    path = CondOTProbPath()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    start_epoch = 1
    if args.resume:
        # Resume 1 lần train reflow đang dang dở (đã init từ init_ckpt ở lần chạy đầu).
        ckpt_path = _resolve_resume(args.resume)
        print(f"Resuming reflow training from: {ckpt_path}")
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
        # Warm-start: khởi tạo trọng số từ model baseline đã dùng để sinh reflow coupling.
        print(f"Init từ baseline checkpoint: {args.init_ckpt}")
        init_ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(init_ckpt["model_state_dict"])
        print(f"  init epoch={init_ckpt.get('epoch','?')}  "
              f"loss={init_ckpt.get('loss', float('nan')):.5f}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x_0, x_1 in loader:
            x_0 = x_0.to(device)                              # ĐÃ ghép sẵn với x_1 tương ứng
            x_1 = x_1.to(device)
            B = x_1.shape[0]

            t = torch.rand(B, device=device)                   # uniform t ∈ [0,1]

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
            ckpt_path = os.path.join(CKPT_DIR, f"unet_reflow_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "arch": "unet_reflow",
                    "init_ckpt": args.init_ckpt,
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
    """
    x_init ~ N(0,I) tươi (KHÔNG dùng lại reflow_x0) — vì marginal của coupling reflow
    (backward ODE từ ảnh thật) vẫn xấp xỉ N(0,I) (xem sanity check trong
    generate_reflow_coupling.py), nên sample vẫn khởi động từ Gauss chuẩn, đúng lý
    thuyết Rectified Flow (giữ nguyên marginal 2 đầu, chỉ làm thẳng đường đi).
    """
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
    p = argparse.ArgumentParser(description="Rectified Flow (reflow) retrain trên UNet baseline")
    p.add_argument("--epochs",       type=int,   default=1000,
                   help="Tổng số epoch mục tiêu (default: 1000, khớp baseline)")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Nhỏ hơn baseline (2e-4) vì đây là fine-tune/warm-start, không train từ đầu")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path checkpoint reflow để resume, hoặc 'latest' (bỏ qua --init_ckpt)")
    p.add_argument("--init_ckpt",    type=str,
                   default=os.path.join(REPO_ROOT, "shapes_fm_output", "checkpoints", "unet_epoch1000.pt"),
                   help="Checkpoint UNet baseline để khởi tạo trọng số (warm-start), "
                        "PHẢI khớp đúng model đã dùng sinh reflow_data/")
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--save_every",   type=int,   default=50)
    p.add_argument("--sample_every", type=int,   default=50)
    p.add_argument("--sample_steps", type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
