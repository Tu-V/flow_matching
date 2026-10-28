"""
Bản GUIDED (class-conditional, CFG) của train_anisotropic_p0_experiment_dit.py: so
sánh hallucination (double-col/empty) isotropic vs anisotropic p0 trên DiT, train
CÓ CONDITIONING theo class (7 lớp = tập cột có shape, xem label_shapes_dataset_10k.py),
CFG training giống train_anisotropic_p0_experiment_cond_unet.py (drop label cả batch
với xác suất --class_drop_prob). Thêm 1 class DiTCond (tái sử dụng PatchEmbed,
TimestepEmbedder, DiTBlock, FinalLayer, get_2d_sincos_pos_embed từ train_shapes_fm_dit.py
-- class DiT gốc ở đó KHÔNG có conditioning nên không dùng lại được nguyên class, chỉ
tái dùng các building block) + LabelEmbedder (giống DiT gốc Peebles&Xie: nn.Embedding
num_classes+1, index cuối = null/unconditional cho CFG).

Dataset: shapes_10k_labeled/ (sinh bởi label_shapes_dataset_10k.py, 10K ảnh, 16x16, RGB).
CHỈ 10K ảnh theo yêu cầu, không quét nhiều cỡ dataset.

p0 = N(0, diag(sigma_i^2)), sigma_i = 1+i*epsilon, D = IN_CHANNELS*img_size^2 = 3*16*16=768
(RGB, giống bản UNet cond — khác bản DiT gốc grayscale D=256).

Usage:
    python train_anisotropic_p0_experiment_cond_dit.py
    python train_anisotropic_p0_experiment_cond_dit.py --quick   # kiểm tra lỗi, cực nhanh
"""

import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath      # noqa: E402
from flow_matching.solver import ODESolver          # noqa: E402
from flow_matching.utils import ModelWrapper         # noqa: E402
from hallucination_detector import analyze_batch, summarize   # noqa: E402
from train_shapes_fm_dit import (                   # noqa: E402  (tái dùng building block, KHÔNG đổi file này)
    PatchEmbed, TimestepEmbedder, DiTBlock, FinalLayer, get_2d_sincos_pos_embed,
)

IMG_SIZE = 16
IN_CHANNELS = 3
NUM_CLASSES = 7
N_TRAIN = 10000

LABELED_DATA_DIR = os.path.join(REPO_ROOT, "shapes_10k_labeled")
OUTPUT_DIR_BASE = os.path.join(REPO_ROOT, "shape_exp", "unet_architecture", "anisotropic_p0_cond_dit_output")
OUTPUT_DIR = OUTPUT_DIR_BASE   # bị main() ghi đè nếu --class_drop_prob khác mặc định (0.2)


# ── Dataset (giống bản UNet cond) ────────────────────────────────────────────
class LabeledShapesDataset(Dataset):
    def __init__(self, root: str):
        csv_path = os.path.join(root, "labels.csv")
        self.items = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                cls = int(row["class"])
                if cls == 0:
                    continue
                path = os.path.join(root, f"class_{cls}", row["filename"])
                self.items.append((path, cls - 1))
        self.labels = np.array([lb for _, lb in self.items], dtype=np.int64)
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label0 = self.items[idx]
        return self.transform(Image.open(path).convert("RGB")), label0


# ── LabelEmbedder (giống DiT gốc Peebles&Xie): index num_classes = null/CFG ──
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


# ── DiT có conditioning (tái dùng PatchEmbed/TimestepEmbedder/DiTBlock/FinalLayer) ──
class DiTCond(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, hidden_size: int,
                 depth: int, num_heads: int, num_classes: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
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
        nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        # DiTBlock.adaLN_modulation / FinalLayer đã tự zero-init trong __init__ (import).

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = self.x_embedder.grid_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        xe = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t)
        if label is None:
            label = torch.full((x.shape[0],), self.num_classes, dtype=torch.long, device=x.device)
        c = c + self.y_embedder(label)
        for block in self.blocks:
            xe = block(xe, c)
        xe = self.final_layer(xe, c)
        return self.unpatchify(xe)


def build_dit_cond(args) -> DiTCond:
    assert IMG_SIZE % args.patch_size == 0, \
        f"img_size={IMG_SIZE} phải chia hết cho patch_size={args.patch_size}"
    return DiTCond(
        img_size=IMG_SIZE, patch_size=args.patch_size, in_channels=IN_CHANNELS,
        hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads,
        num_classes=NUM_CLASSES, mlp_ratio=args.mlp_ratio,
    )


class CFGScaledModel(ModelWrapper):
    """label=None -> unconditional (null embedding). label!=None, cfg_scale=0 -> conditional
    thuần. label!=None, cfg_scale!=0 -> classifier-free guidance (2 forward pass)."""

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cfg_scale: float = 0.0, label: torch.Tensor = None) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if label is None:
            return self.model(x, t, label=None)
        if cfg_scale != 0.0:
            conditional = self.model(x, t, label=label)
            unconditional = self.model(x, t, label=None)
            return (1.0 + cfg_scale) * conditional - cfg_scale * unconditional
        return self.model(x, t, label=label)


# ── p0: isotropic vs anisotropic (sigma_i = 1 + i*epsilon), D = 3*16*16 = 768 ──
def make_sigma(epsilon: float, device) -> torch.Tensor:
    d = IN_CHANNELS * IMG_SIZE * IMG_SIZE
    idx = torch.arange(d, dtype=torch.float32, device=device)
    sigma = 1.0 + idx * epsilon
    return sigma.view(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


def sample_x0(B: int, sigma, device) -> torch.Tensor:
    z = torch.randn(B, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    return z if sigma is None else z * sigma


def save_checkpoint(model: DiTCond, ckpt_path: str, args, optimizer=None, **extra):
    state = {
        "model_state_dict": model.state_dict(), "img_size": IMG_SIZE,
        "hidden_size": args.hidden_size, "depth": args.depth, "num_heads": args.num_heads,
        "patch_size": args.patch_size, "mlp_ratio": args.mlp_ratio, "num_classes": NUM_CLASSES,
        **extra,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, ckpt_path)


def find_resume_checkpoint(ckpt_dir: str, p0_type: str):
    best_path, best_step = None, -1
    final_path = os.path.join(ckpt_dir, f"dit_cond_n{N_TRAIN}_{p0_type}.pt")
    if os.path.isfile(final_path):
        ck = torch.load(final_path, map_location="cpu", weights_only=True)
        best_step, best_path = ck.get("step", 0), final_path
    for p in glob.glob(os.path.join(ckpt_dir, f"dit_cond_n{N_TRAIN}_{p0_type}_step*.pt")):
        m = re.search(r"_step(\d+)\.pt$", p)
        if m and int(m.group(1)) > best_step:
            best_step, best_path = int(m.group(1)), p
    return (best_path, best_step) if best_path is not None else None


def train_one(dataset: LabeledShapesDataset, p0_type: str, sigma, args, device, ckpt_dir: str):
    model = build_dit_cond(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1)

    start_step = 1
    if args.resume and args.save_ckpt:
        found = find_resume_checkpoint(ckpt_dir, p0_type)
        if found is not None:
            ckpt_path, ckpt_step = found
            ck = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ck["model_state_dict"])
            if "optimizer_state_dict" in ck:
                optimizer.load_state_dict(ck["optimizer_state_dict"])
            start_step = ckpt_step + 1
            print(f"  [resume] {os.path.basename(ckpt_path)} (step={ckpt_step}) -> tiếp tục từ {start_step}")
            if start_step > args.total_steps:
                print(f"  [resume] đã đủ {args.total_steps} step -> bỏ qua training.")
                return model

    n = len(dataset)
    print(f"  preloading {n} images ({IMG_SIZE}x{IMG_SIZE}, RGB) to {device} ...")
    all_imgs = torch.stack([dataset[i][0] for i in range(n)]).to(device)
    all_labels = torch.from_numpy(dataset.labels).to(device)
    n_tokens = (IMG_SIZE // args.patch_size) ** 2

    print(f"\n{'='*70}\np0={p0_type}  hidden={args.hidden_size} depth={args.depth} "
          f"heads={args.num_heads} patch={args.patch_size} tokens={n_tokens}  "
          f"params={n_params:,}  total_steps={args.total_steps}  batch_size={args.batch_size}  "
          f"class_drop_prob={args.class_drop_prob}  start_step={start_step}\n{'='*70}")

    model.train()
    loss_window = []
    t0 = time.time()
    for step in range(start_step, args.total_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        x_1 = all_imgs[idx]
        labels = all_labels[idx]
        B = x_1.shape[0]

        cond_label = None if torch.rand(1).item() < args.class_drop_prob else labels

        x_0 = sample_x0(B, sigma, device)
        t = torch.rand(B, device=device)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        u_pred = model(path_sample.x_t, t, label=cond_label)
        loss = torch.pow(u_pred - path_sample.dx_t, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_window.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(loss_window) / len(loss_window)
            print(f"  [p0={p0_type}] step {step:7d}/{args.total_steps}  loss={avg:.5f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  elapsed={time.time()-t0:.0f}s")
            loss_window = []

        if args.save_ckpt and (step % args.save_every == 0) and step != args.total_steps:
            ckpt_path = os.path.join(ckpt_dir, f"dit_cond_n{N_TRAIN}_{p0_type}_step{step:06d}.pt")
            save_checkpoint(model, ckpt_path, args, optimizer=optimizer, p0_type=p0_type,
                             epsilon=args.epsilon, step=step, total_steps=args.total_steps,
                             loss=loss.item())
            print(f"  [checkpoint] Saved: {ckpt_path}")

    if args.save_ckpt:
        final_path = os.path.join(ckpt_dir, f"dit_cond_n{N_TRAIN}_{p0_type}.pt")
        save_checkpoint(model, final_path, args, optimizer=optimizer, p0_type=p0_type,
                         epsilon=args.epsilon, step=args.total_steps, total_steps=args.total_steps)
        print(f"  Saved (final): {final_path}")

    del all_imgs
    model.eval()
    return model


@torch.no_grad()
def sample_and_analyze(model: DiTCond, sigma, all_labels_np, cfg_scale, device,
                        n_total: int, steps: int, batch_size: int) -> dict:
    model.eval()
    wrapper = CFGScaledModel(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    all_analyses = []
    n_done = 0
    t0 = time.time()
    n_labels = len(all_labels_np)
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = sample_x0(B, sigma, device)
        labels = torch.from_numpy(all_labels_np[np.random.randint(0, n_labels, size=B)]).to(device)
        x_final = solver.sample(x_init=x_init, step_size=None, method="euler",
                                 time_grid=time_grid, return_intermediates=False,
                                 cfg_scale=cfg_scale, label=labels)
        x_01 = (x_final.clamp(-1, 1) + 1) / 2
        imgs_np = (x_01.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        all_analyses.extend(analyze_batch(imgs_np))
        n_done += B
        print(f"    sampled {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", end="\r")
    print()
    return summarize(all_analyses)


def sample_and_analyze_repeated(model, sigma, all_labels_np, cfg_scale, device,
                                 n_total: int, steps: int, batch_size: int, n_repeats: int) -> dict:
    runs = []
    for rep in range(1, n_repeats + 1):
        print(f"  -- repeat {rep}/{n_repeats} --")
        s = sample_and_analyze(model, sigma, all_labels_np, cfg_scale, device, n_total, steps, batch_size)
        runs.append(s)
        print(f"     hall={100*s['hall_rate']:.4f}%  (empty={100*s['n_empty']/s['n_total']:.4f}%  "
              f"double_col={100*s['n_double_col']/s['n_total']:.4f}%)")

    hall_rates = np.array([r["hall_rate"] for r in runs]) * 100.0
    empty_rates = np.array([r["n_empty"] / r["n_total"] for r in runs]) * 100.0
    double_rates = np.array([r["n_double_col"] / r["n_total"] for r in runs]) * 100.0
    return {
        "n_repeats": n_repeats, "n_total_per_repeat": n_total,
        "hall_rate_pct_list": hall_rates.tolist(), "empty_rate_pct_list": empty_rates.tolist(),
        "double_rate_pct_list": double_rates.tolist(),
        "hall_rate_mean": float(hall_rates.mean()),
        "hall_rate_std": float(hall_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "empty_rate_mean": float(empty_rates.mean()),
        "empty_rate_std": float(empty_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
        "double_rate_mean": float(double_rates.mean()),
        "double_rate_std": float(double_rates.std(ddof=1)) if n_repeats > 1 else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Guided (CFG) isotropic vs anisotropic p0, DiT, 10K ảnh")
    p.add_argument("--p0_types", type=str, default="isotropic,anisotropic")
    p.add_argument("--epsilon", type=float, default=0.001,
                   help="sigma_i = 1+i*epsilon, i=0..D-1, D=IN_CHANNELS*img_size^2=768 (RGB)")
    p.add_argument("--total_steps", type=int, default=40000, help="16x16 -> 40K step (quy ước)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--class_drop_prob", type=float, default=0.2)
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--sample_cfg_scale", type=float, default=0.0)
    p.add_argument("--save_ckpt", action="store_true", default=True)
    p.add_argument("--no_save_ckpt", dest="save_ckpt", action="store_false")
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--quick", action="store_true", help="cấu hình siêu nhỏ để kiểm tra lỗi")
    a = p.parse_args()
    if a.quick:
        a.total_steps, a.batch_size, a.log_every, a.save_every = 20, 16, 5, 10 ** 9
        a.n_sample_total, a.n_repeats, a.sample_steps, a.sample_batch_size = 20, 1, 5, 20
    return a


def main():
    global OUTPUT_DIR
    args = parse_args()
    if args.class_drop_prob != 0.2:
        OUTPUT_DIR = f"{OUTPUT_DIR_BASE}_dropprob{args.class_drop_prob:g}"
    p0_types = [s.strip() for s in args.p0_types.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    d = IN_CHANNELS * IMG_SIZE * IMG_SIZE
    print(f"Device: {device}  img_size={IMG_SIZE}x{IMG_SIZE} RGB (D={d})  num_classes={NUM_CLASSES}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"Arch: DiT  hidden_size={args.hidden_size}  depth={args.depth}  "
          f"num_heads={args.num_heads}  patch_size={args.patch_size}")
    print(f"p0 types: {p0_types}")

    if not os.path.isdir(LABELED_DATA_DIR):
        print(f"LỖI: không thấy {LABELED_DATA_DIR} -- chạy `python label_shapes_dataset_10k.py` trước.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    if args.save_ckpt:
        os.makedirs(ckpt_dir, exist_ok=True)

    dataset = LabeledShapesDataset(LABELED_DATA_DIR)
    print(f"Dataset: {len(dataset)} ảnh thật ({LABELED_DATA_DIR})")
    sigma_aniso = make_sigma(args.epsilon, device)

    results = []
    for p0_type in p0_types:
        sigma = sigma_aniso if p0_type == "anisotropic" else None
        model = train_one(dataset, p0_type, sigma, args, device, ckpt_dir)

        print(f"\nSampling {args.n_sample_total} ảnh × {args.n_repeats} lần lặp "
              f"(p0={p0_type}, cfg_scale={args.sample_cfg_scale}, steps={args.sample_steps}) ...")
        s = sample_and_analyze_repeated(model, sigma, dataset.labels, args.sample_cfg_scale, device,
                                         args.n_sample_total, args.sample_steps,
                                         args.sample_batch_size, args.n_repeats)
        print(f"  p0={p0_type:12s}  hall={s['hall_rate_mean']:.3f}%±{s['hall_rate_std']:.3f}%  "
              f"(empty={s['empty_rate_mean']:.3f}%±{s['empty_rate_std']:.3f}%  "
              f"double_col={s['double_rate_mean']:.3f}%±{s['double_rate_std']:.3f}%)")
        results.append({"p0_type": p0_type, **s})

        run_dir = os.path.join(OUTPUT_DIR, f"n{N_TRAIN}_{p0_type}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "stats.txt"), "w") as f:
            f.write(f"n_data={N_TRAIN}  p0_type={p0_type}  img_size={IMG_SIZE}  epsilon={args.epsilon}  "
                    f"num_classes={NUM_CLASSES}  sample_cfg_scale={args.sample_cfg_scale}\n")
            f.write(f"total_steps={args.total_steps}  n_sample_total/repeat={args.n_sample_total}  "
                    f"n_repeats={args.n_repeats}  sample_steps={args.sample_steps}\n\n")
            for rep, hr in enumerate(s["hall_rate_pct_list"], start=1):
                f.write(f"  repeat {rep}: hall={hr:.4f}%  empty={s['empty_rate_pct_list'][rep-1]:.4f}%  "
                        f"double_col={s['double_rate_pct_list'][rep-1]:.4f}%\n")
            f.write(f"\nHallucination : mean={s['hall_rate_mean']:.4f}%  std={s['hall_rate_std']:.4f}%\n")
            f.write(f"  empty img   : mean={s['empty_rate_mean']:.4f}%  std={s['empty_rate_std']:.4f}%\n")
            f.write(f"  double col  : mean={s['double_rate_mean']:.4f}%  std={s['double_rate_std']:.4f}%\n")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*80}\nTỔNG HỢP (guided DiT, n={N_TRAIN}, cfg_scale={args.sample_cfg_scale})\n{'='*80}")
    header = (f"{'p0_type':>12s} {'hall% mean':>11s} {'hall% std':>10s} "
              f"{'empty% mean':>12s} {'double_col% mean':>17s}")
    print(header)
    lines = [header]
    for r in results:
        line = (f"{r['p0_type']:>12s} {r['hall_rate_mean']:11.4f} {r['hall_rate_std']:10.4f} "
                f"{r['empty_rate_mean']:12.4f} {r['double_rate_mean']:17.4f}")
        print(line)
        lines.append(line)

    if "isotropic" in p0_types and "anisotropic" in p0_types:
        iso = next((r for r in results if r["p0_type"] == "isotropic"), None)
        ani = next((r for r in results if r["p0_type"] == "anisotropic"), None)
        if iso and ani:
            rel = ("(giảm về 0)" if ani["hall_rate_mean"] == 0 else
                   f"({100*(ani['hall_rate_mean']-iso['hall_rate_mean'])/max(iso['hall_rate_mean'],1e-12):+.1f}% tương đối)")
            line = (f"isotropic={iso['hall_rate_mean']:.4f}%±{iso['hall_rate_std']:.4f}%  ->  "
                    f"anisotropic={ani['hall_rate_mean']:.4f}%±{ani['hall_rate_std']:.4f}%  {rel}")
            print(line)
            lines.append(line)

    summary_path = os.path.join(OUTPUT_DIR, "summary_anisotropic_p0_cond_dit.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
