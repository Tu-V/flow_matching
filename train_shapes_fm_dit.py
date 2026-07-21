"""
Flow Matching training với kiến trúc DiT (Diffusion Transformer, Peebles & Xie 2023)
thay cho UNet, để test giả thuyết: hallucination (shape lệch cột / double-col) có phải
do bias translation-equivariance của convolution hay không.

  - DiT patchify ảnh thành token rời rạc + POSITIONAL EMBEDDING tường minh (2D sin-cos,
    cố định theo patch) + self-attention toàn cục. Không có weight-sharing theo không
    gian như conv -> không có translation-equivariance prior. Vị trí tuyệt đối (cột nào)
    là 1 phần thông tin đầu vào tường minh của mỗi token, khác hẳn UNet.
  - Conditioning theo t bằng adaLN-Zero (giống DiT gốc): mỗi block predict
    (shift, scale, gate) từ time-embedding, layer cuối init = 0 -> block khởi đầu như
    identity, giúp train ổn định ngay cả khi thay hẳn kiến trúc.

Source distribution VẪN LÀ GAUSS liên tục (x_0 = torch.randn_like(x_1)), giống hệt
train_shapes_fm.py gốc — chỉ đổi kiến trúc model, giữ nguyên mọi thứ khác (CondOTProbPath,
ODESolver, optimizer, cách train) để so sánh công bằng với UNet baseline.

Usage:
    python train_shapes_fm_dit.py --epochs 1000 --batch_size 128 --lr 1e-4

    # Resume từ checkpoint mới nhất
    python train_shapes_fm_dit.py --resume latest --epochs 1000

    # Đổi cỡ model (mặc định ~ tương đương param count với UNet baseline ~9M)
    python train_shapes_fm_dit.py --hidden_size 256 --depth 6 --num_heads 4 --patch_size 2

--epochs luôn là TỔNG số epoch mục tiêu (không phải số epoch thêm vào).
"""

import argparse
import math
import os
import sys

import numpy as np
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
from models.nn import timestep_embedding                # noqa: E402  (dùng lại đúng hàm sin-cos t-embedding của UNet baseline, cho nhất quán)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    REPO_ROOT,
    "..",
    "neurips-2024-diffusion-model-hallucination",
    "simple-datasets",
    "simple-shapes-5k-16x16",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_dit_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

IMG_SIZE = 16
IN_CHANNELS = 3


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
# DiT building blocks (theo kiến trúc gốc facebookresearch/DiT, rút gọn cho ảnh 16x16)
# ─────────────────────────────────────────────────────────────────────────────

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Ảnh [B,C,H,W] -> chuỗi token [B,N,hidden_size], N=(H/patch)*(W/patch)."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size phải chia hết cho patch_size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                    # [B, hidden, grid, grid]
        x = x.flatten(2).transpose(1, 2)     # [B, N, hidden]
        return x


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


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                    # each [B, heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """Transformer block với adaLN-Zero conditioning theo t (giống DiT gốc)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Zero-init modulation layer -> block bắt đầu train như identity (ổn định hơn).
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ── 2D sin-cos positional embedding (cố định, không train) ─────────────────────
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)          # grid[0]=w, grid[1]=h
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    return np.concatenate([emb_h, emb_w], axis=1)  # [grid_size*grid_size, embed_dim]


# ── DiT model ─────────────────────────────────────────────────────────────────
class DiT(nn.Module):
    """
    Diffusion/Flow Transformer, unconditional (chỉ điều kiện theo t).
    forward(x, t) -> velocity, cùng shape với x -> khớp trực tiếp interface
    ODESolver cần (không cần thêm 'extra' dict như UNetModel).
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        patch_size: int = 2,
        in_channels: int = IN_CHANNELS,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.patch_size = patch_size

        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self._init_weights()

    def _init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.x_embedder.proj.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # DiTBlock.adaLN_modulation và FinalLayer đã tự zero-init trong __init__.

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, p*p*C] -> [B, C, H, W]."""
        c = self.out_channels
        p = self.patch_size
        h = w = self.x_embedder.grid_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)


def build_dit(args) -> DiT:
    return DiT(
        img_size=IMG_SIZE,
        patch_size=args.patch_size,
        in_channels=IN_CHANNELS,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )


# ── Model wrapper for ODESolver ────────────────────────────────────────────────
class DiTVelocityWrapper(ModelWrapper):
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
    model = build_dit(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = model.x_embedder.num_patches
    print(f"DiT: hidden={args.hidden_size}  depth={args.depth}  heads={args.num_heads}  "
          f"patch={args.patch_size}  tokens={n_tokens}  params={n_params:,}")

    # ── Flow matching path (giống hệt UNet baseline) ──
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
            ckpt_path = os.path.join(CKPT_DIR, f"dit_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "arch": "dit",
                    "hidden_size": args.hidden_size,
                    "depth": args.depth,
                    "num_heads": args.num_heads,
                    "patch_size": args.patch_size,
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

    wrapper = DiTVelocityWrapper(model)
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
    p = argparse.ArgumentParser(description="Flow Matching (DiT) trên simple-shapes-5k-16x16")
    p.add_argument("--epochs",       type=int,   default=1000,
                   help="Tổng số epoch mục tiêu (default: 1000, khớp lần train UNet trước)")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path checkpoint để resume, hoặc 'latest'")
    # DiT hyperparams (~9M params với default, tương đương UNet baseline)
    p.add_argument("--hidden_size",  type=int,   default=256)
    p.add_argument("--depth",        type=int,   default=6)
    p.add_argument("--num_heads",    type=int,   default=4)
    p.add_argument("--patch_size",   type=int,   default=2,
                   help="Phải chia hết 16 (1, 2, 4, 8, 16). patch=2 -> 64 token.")
    p.add_argument("--mlp_ratio",    type=float, default=4.0)
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--save_every",   type=int,   default=50)
    p.add_argument("--sample_every", type=int,   default=50)
    p.add_argument("--sample_steps", type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
