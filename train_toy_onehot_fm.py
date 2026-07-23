"""
Thử nghiệm TỐI GIẢN: flow matching từ Gauss(0, I_16) sang tập vector gần-one-hot 16
chiều (mỗi sample x có ĐÚNG 1 chiều được "enable", giá trị chiều đó ~ U[0.95, 1.0]
(không cố định = 1 nữa), các chiều còn lại = 0). Network là MLP đơn giản, KHÔNG có
cấu trúc không gian (không conv/attention/patch) vì data chỉ là vector phẳng.

Mục tiêu: kiểm tra xem hiện tượng "double-col hallucination" (2 shape trong 1 cột)
quan sát được ở bộ ảnh shapes có phải là ĐẶC THÙ của ảnh/kiến trúc hay không, bằng
cách tái hiện nó ở dạng tối giản nhất có thể — 1 vector 16 chiều, "hallucination"
= sinh ra vector có >=2 chiều cùng ~1 (thay vì đúng 1 chiều ~1 như toàn bộ data thật).
Nếu vẫn xảy ra ở đây, hiện tượng gần như chắc chắn là bản chất của flow matching với
coupling độc lập ngẫu nhiên (nhiều đường thẳng (x_0,x_1) cắt nhau trong không gian
16 chiều), không liên quan gì tới ảnh/CNN/positional embedding.

Dataset: X = 5000 vector, mỗi vector chọn ngẫu nhiên đều 1 trong 16 chiều, gán giá trị
~ U[0.95, 1.0] (có lặp giữa các chiều, không cần đủ 5000/16 mỗi lớp).
Source : x_0 ~ N(0, I_16) — độc lập ngẫu nhiên mỗi batch, giống hệt cách train
UNet baseline trên ảnh (train_shapes_fm.py), chỉ khác domain (vector thay vì ảnh).

Sau khi train xong, script TỰ sample --n_sample_total vector (mặc định 100000) và
thống kê theo ngưỡng 0.5 (giữa 0 và 1 — giá trị thật của one-hot):
    0 chiều > 0.5   : "collapsed" (analogue ảnh rỗng/empty)
    1 chiều > 0.5   : "clean" (đúng kỳ vọng one-hot)
    >=2 chiều > 0.5 : "multi-active" (analogue double-col hallucination)

Usage:
    python train_toy_onehot_fm.py --epochs 2000
"""

import argparse
import os
import sys
from collections import Counter

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath     # noqa: E402
from flow_matching.solver import ODESolver         # noqa: E402
from flow_matching.utils import ModelWrapper       # noqa: E402
from models.nn import timestep_embedding           # noqa: E402  (dùng lại đúng hàm sin-cos t-embedding đã dùng cho DiT/MLP ảnh)

DIM = 16
OUTPUT_DIR = os.path.join(REPO_ROOT, "toy_onehot_fm_output")
CKPT_PATH = os.path.join(OUTPUT_DIR, "mlp_onehot.pt")


# ── Dataset: mỗi sample chỉ enable 1 chiều ngẫu nhiên, giá trị ~ U[0.95, 1.0] ──
def make_dataset(n_samples: int, dim: int = DIM, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, dim, (n_samples,), generator=g)
    vals = torch.empty(n_samples).uniform_(0.95, 1.0, generator=g)
    X = torch.zeros(n_samples, dim)
    X[torch.arange(n_samples), labels] = vals
    return X, labels


# ── Model: MLP đơn giản, conditioning theo t bằng cộng embedding ───────────────
class ToyMLP(nn.Module):
    def __init__(self, dim: int = DIM, hidden: int = 128, depth: int = 4, t_embed_dim: int = 64):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Linear(dim, hidden)
        blocks = []
        for _ in range(depth):
            blocks += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t_e = timestep_embedding(t, self.t_embed_dim)
        h = self.in_proj(x) + self.t_mlp(t_e)
        h = self.blocks(h)
        return self.out_proj(h)


class MLPVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        return self.model(x, t)


# ── Train ───────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    X_data, labels = make_dataset(args.n_samples, DIM, seed=args.data_seed)
    X_data = X_data.to(device)
    print(f"Dataset: {X_data.shape[0]} vector one-hot {DIM} chiều  "
          f"(mỗi lớp trung bình ~{args.n_samples/DIM:.0f} sample)")

    model = ToyMLP(dim=DIM, hidden=args.hidden, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP: hidden={args.hidden} depth={args.depth}  params={n_params:,}")

    path = CondOTProbPath()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    n = X_data.shape[0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            x_1 = X_data[idx]
            B = x_1.shape[0]

            x_0 = torch.randn(B, DIM, device=device)
            t = torch.rand(B, device=device)

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            u_pred = model(path_sample.x_t, t)
            loss = torch.pow(u_pred - path_sample.dx_t, 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if epoch % args.log_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:5d}/{args.epochs}  loss={epoch_loss/n_batches:.6f}  lr={lr:.2e}")

    torch.save({"model_state_dict": model.state_dict(), "hidden": args.hidden, "depth": args.depth}, CKPT_PATH)
    print(f"\nSaved checkpoint: {CKPT_PATH}")

    analyze(model, device, args.n_sample_total, args.sample_steps, args.sample_batch_size)


def fmt_vec(v, decimals: int = 4) -> str:
    """Format list float thành chuỗi gọn đúng `decimals` chữ số sau dấu phẩy
    (numpy.round() chỉ làm tròn giá trị nhị phân, convert float32->float64 vẫn
    ra dãy số dài — phải format string tường minh mới gọn được)."""
    return "[" + ", ".join(f"{x:.{decimals}f}" for x in v) + "]"


# ── Sample + thống kê "double-active" ──────────────────────────────────────────
@torch.no_grad()
def analyze(model, device, n_total: int, steps: int, batch_size: int = 4096):
    model.eval()
    wrapper = MLPVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    active_counts = Counter()
    max_val_when_clean = []
    example_multi = []
    n_done = 0

    print(f"\nSampling {n_total} vector (steps={steps}) ...")
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = torch.randn(B, DIM, device=device)
        x_final = solver.sample(
            x_init=x_init, step_size=None, method="euler",
            time_grid=time_grid, return_intermediates=False,
        )
        active = x_final > 0.5
        n_active = active.sum(dim=1)

        for i, c in enumerate(n_active.tolist()):
            active_counts[c] += 1
            if c == 1:
                max_val_when_clean.append(x_final[i].max().item())
            elif c >= 2 and len(example_multi) < 5:
                example_multi.append(x_final[i].cpu().numpy().round(4).tolist())

        n_done += B

    n_zero  = active_counts.get(0, 0)
    n_one   = active_counts.get(1, 0)
    n_multi = sum(v for k, v in active_counts.items() if k >= 2)

    print(f"\n{'='*60}")
    print(f"KẾT QUẢ  ({n_total} sample, ngưỡng active = giá trị > 0.5)")
    print(f"{'='*60}")
    print(f"  0 chiều active   (collapsed)     : {n_zero:6d}  ({100*n_zero/n_total:.2f}%)")
    print(f"  1 chiều active   (clean, đúng)   : {n_one:6d}  ({100*n_one/n_total:.2f}%)")
    print(f"  >=2 chiều active (MULTI-ACTIVE)  : {n_multi:6d}  ({100*n_multi/n_total:.2f}%)")
    print(f"\nPhân bố chi tiết số chiều active:")
    for k in sorted(active_counts):
        print(f"    {k} chiều: {active_counts[k]:6d}  ({100*active_counts[k]/n_total:.3f}%)")
    if max_val_when_clean:
        import statistics
        print(f"\nKhi clean (1 chiều active): giá trị chiều đó trung bình = "
              f"{statistics.mean(max_val_when_clean):.4f}  (kỳ vọng ~1.0)")
    if example_multi:
        print(f"\nVí dụ {len(example_multi)} sample multi-active (giá trị đầy đủ 16 chiều):")
        for v in example_multi:
            print(f"    {fmt_vec(v)}")

    stats_path = os.path.join(OUTPUT_DIR, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"N sample: {n_total}   ODE steps: {steps}   nguong active: >0.5\n\n")
        f.write(f"0 chieu active   (collapsed)   : {n_zero} ({100*n_zero/n_total:.2f}%)\n")
        f.write(f"1 chieu active   (clean)       : {n_one} ({100*n_one/n_total:.2f}%)\n")
        f.write(f"2+ chieu active  (multi-active): {n_multi} ({100*n_multi/n_total:.2f}%)\n\n")
        f.write("Phan bo chi tiet:\n")
        for k in sorted(active_counts):
            f.write(f"  {k} chieu: {active_counts[k]} ({100*active_counts[k]/n_total:.3f}%)\n")
        if example_multi:
            f.write("\nVi du sample multi-active:\n")
            for v in example_multi:
                f.write(f"  {fmt_vec(v)}\n")
    print(f"\nStats -> {stats_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Toy flow matching: Gauss(0,I_16) -> one-hot vectors")
    p.add_argument("--n_samples", type=int, default=5000, help="Số sample data one-hot (default: 5000)")
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--n_sample_total", type=int, default=100000, help="Số vector sample để phân tích (default: 100000)")
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=4096)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
