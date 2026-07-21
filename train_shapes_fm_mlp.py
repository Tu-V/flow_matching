"""
Flow Matching training với kiến trúc MLP THUẦN (không convolution, không attention theo
patch) — phiên bản "cực đoan" nhất trong chuỗi so sánh kiến trúc để test giả thuyết
hallucination do translation-equivariance bias:

  - UNet (train_shapes_fm.py)      : convolution, weight-sharing đầy đủ theo không gian.
  - DiT  (train_shapes_fm_dit.py)  : attention theo patch + positional embedding tường
                                      minh, không weight-sharing theo không gian nhưng vẫn
                                      xử lý ảnh dạng chuỗi token có cấu trúc lưới.
  - MLP  (script này)               : coi cả ảnh 3x16x16=768 chiều là 1 vector phẳng DUY
                                      NHẤT, KHÔNG có khái niệm "patch"/"pixel lân cận" nào
                                      cả. Mỗi neuron ở lớp ẩn nhìn thấy toàn bộ 768 giá trị
                                      cùng lúc với trọng số RIÊNG cho từng vị trí -> triệt
                                      tiêu hoàn toàn translation-equivariance / weight-sharing
                                      theo không gian. Nếu hallucination (shape lệch cột)
                                      giảm mạnh so với UNet thì càng củng cố giả thuyết
                                      "positional bias của conv gây hallucination".

Kiến trúc: ResMLP với FiLM/adaLN-Zero conditioning theo t (giống tinh thần DiTBlock,
nhưng không có patch/attention — chỉ Linear + LayerNorm + residual trên 1 vector duy nhất).

Source distribution VẪN LÀ GAUSS liên tục (x_0 = torch.randn_like(x_1)), data vẫn là
5k ảnh 3x16x16 — giống hệt train_shapes_fm.py / train_shapes_fm_dit.py, chỉ đổi kiến trúc,
để so sánh công bằng.

Usage:
    python train_shapes_fm_mlp.py --epochs 1000 --batch_size 128 --lr 1e-4

    # Resume từ checkpoint mới nhất
    python train_shapes_fm_mlp.py --resume latest --epochs 1000

    # Đổi cỡ model (mặc định ~ tương đương param count với UNet/DiT baseline)
    python train_shapes_fm_mlp.py --hidden_size 384 --depth 6 --mlp_ratio 4.0

--epochs luôn là TỔNG số epoch mục tiêu (không phải số epoch thêm vào).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath          # noqa: E402
from flow_matching.solver import ODESolver              # noqa: E402
from flow_matching.utils import ModelWrapper            # noqa: E402
from models.nn import timestep_embedding                # noqa: E402  (dùng lại đúng hàm sin-cos t-embedding của UNet/DiT baseline, cho nhất quán)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    REPO_ROOT,
    "..",
    "neurips-2024-diffusion-model-hallucination",
    "simple-datasets",
    "simple-shapes-5k-16x16",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_mlp_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

IMG_SIZE = 16
IN_CHANNELS = 3
IN_DIM = IN_CHANNELS * IMG_SIZE * IMG_SIZE   # 768


# ── Dataset ───────────────────────────────────────────────────────────────────
class ShapesDataset(Dataset):
    """Loads all PNGs in a flat directory as 16x16 RGB tensors scaled to [-1, 1]."""

    def __init__(self, root: str):
        self.paths = sorted(
            [
                os.path.join(root, f)
                for f in os.listdir(root)
                if f.lower().endswith(".png")
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),                      # [0, 1]
                transforms.Normalize([0.5, 0.5, 0.5],      # → [-1, 1]
                                     [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ─────────────────────────────────────────────────────────────────────────────
# MLP building blocks
# ─────────────────────────────────────────────────────────────────────────────

def modulate(h: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return h * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """t (float, [0,1]) -> sinusoidal embedding -> MLP -> [B, hidden_size]."""

    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.freq_embed_size = freq_embed_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.freq_embed_size)
        return self.mlp(t_freq)


class ResMLPBlock(nn.Module):
    """
    Residual MLP block, FiLM/adaLN-Zero conditioning theo t (không patch, không attention
    — h là 1 vector duy nhất đại diện toàn bộ ảnh).
    """

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        # Zero-init modulation -> block bắt đầu train như identity (ổn định hơn).
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        h_mod = modulate(self.norm(h), shift, scale)
        return h + gate * self.fc2(self.act(self.fc1(h_mod)))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        h = modulate(self.norm_final(h), shift, scale)
        return self.linear(h)


# ── MLP model ─────────────────────────────────────────────────────────────────
class MLPFlow(nn.Module):
    """
    Ảnh [B,3,16,16] được flatten thành 1 vector [B,768] duy nhất -> Linear vào không
    gian ẩn -> N khối ResMLP (conditioning theo t bằng FiLM/adaLN-Zero) -> Linear ra
    lại 768 chiều -> reshape về [B,3,16,16]. KHÔNG có convolution/attention/patch nào.
    forward(x, t) -> velocity, cùng shape với x -> khớp trực tiếp interface ODESolver.
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        in_channels: int = IN_CHANNELS,
        hidden_size: int = 384,
        depth: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.in_dim = in_channels * img_size * img_size

        self.in_proj = nn.Linear(self.in_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [ResMLPBlock(hidden_size, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, self.in_dim)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # ResMLPBlock.adaLN_modulation và FinalLayer đã tự zero-init trong __init__.

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.in_proj(x.reshape(B, -1))
        c = self.t_embedder(t)
        for block in self.blocks:
            h = block(h, c)
        out = self.final_layer(h, c)
        return out.reshape(B, self.in_channels, self.img_size, self.img_size)


def build_mlp(args) -> MLPFlow:
    return MLPFlow(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        hidden_size=args.hidden_size,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
    )


# ── Model wrapper for ODESolver ────────────────────────────────────────────────
class MLPVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t)


# ── Training ──────────────────────────────────────────────────────────────────
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

    # ── Data ──
    dataset = ShapesDataset(DATA_DIR)
    print(f"Dataset size: {len(dataset)} images")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ── Model ──
    model = build_mlp(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP: hidden={args.hidden_size}  depth={args.depth}  mlp_ratio={args.mlp_ratio}  "
          f"in_dim={IN_DIM}  params={n_params:,}")

    # ── Flow matching path (giống hệt UNet/DiT baseline) ──
    path = CondOTProbPath()

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # ── Resume từ checkpoint ──
    start_epoch = 1
    if args.resume:
        ckpt_path = _resolve_resume(args.resume)
        print(f"Resuming from: {ckpt_path}")
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

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x_1 = batch.to(device)                          # [B, 3, 16, 16], in [-1,1]
            B = x_1.shape[0]

            x_0 = torch.randn_like(x_1)                     # Gaussian source (KHÔNG đổi so với baseline)
            t = torch.rand(B, device=device)                 # uniform t ∈ [0,1]

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t                           # target velocity

            u_pred = model(x_t, t)
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
            ckpt_path = os.path.join(CKPT_DIR, f"mlp_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "arch": "mlp",
                    "hidden_size": args.hidden_size,
                    "depth": args.depth,
                    "mlp_ratio": args.mlp_ratio,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            _sample_and_save(model, device, epoch, n_samples=64, steps=args.sample_steps)

    print("Training complete.")
    print(f"Checkpoints → {CKPT_DIR}")
    print(f"Samples     → {SAMPLE_DIR}")


# ── Sampling ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def _sample_and_save(model, device, epoch, n_samples=64, steps=100):
    model.eval()

    wrapper = MLPVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    x_init = torch.randn(n_samples, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    x_gen = solver.sample(
        x_init=x_init,
        step_size=None,
        method="euler",
        time_grid=time_grid,
        return_intermediates=False,
    )

    x_gen = (x_gen.clamp(-1, 1) + 1) / 2
    out_path = os.path.join(SAMPLE_DIR, f"samples_epoch{epoch:04d}.png")
    save_image(x_gen, out_path, nrow=8, padding=1)
    print(f"  Samples saved: {out_path}")

    model.train()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_resume(resume: str) -> str:
    if resume == "latest":
        import glob
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
        if not ckpts:
            raise FileNotFoundError(f"Không có checkpoint trong {CKPT_DIR}")
        return ckpts[-1]
    return resume


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Flow Matching (MLP thuần) trên simple-shapes-5k-16x16")
    p.add_argument("--epochs",       type=int,   default=1000,
                   help="Tổng số epoch mục tiêu (default: 1000, khớp UNet/DiT baseline)")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path checkpoint để resume, hoặc 'latest'")
    # MLP hyperparams (~10M params với default, cùng bậc với UNet/DiT baseline)
    p.add_argument("--hidden_size",  type=int,   default=384)
    p.add_argument("--depth",        type=int,   default=6)
    p.add_argument("--mlp_ratio",    type=float, default=4.0)
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--save_every",   type=int,   default=50)
    p.add_argument("--sample_every", type=int,   default=50)
    p.add_argument("--sample_steps", type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
