"""
Bản CONDITIONING của thử nghiệm toy one-hot (train_toy_onehot_fm.py):

    - 16 class, class i <=> chiều thứ i được kích hoạt.
    - Data KHÔNG còn cố định = 1 nữa: giá trị ở chiều active là số ngẫu nhiên
      đều trong [0.95, 1.0] (các chiều còn lại = 0). Mỗi class lấy ~5000 mẫu
      (tổng dataset = 16 * n_per_class).
    - Model MLP có conditioning theo class (label embedding, giống cơ chế
      num_classes của UNetModel dùng cho ảnh — xem train_shapes_fm_cond.py).

Câu hỏi cần trả lời (nối tiếp phát hiện ở bộ ảnh: conditioning sửa được "sai cột"
nhưng KHÔNG sửa được "2 shape trong đúng cột"): khi ép model sinh đúng class c
(biết chắc chiều nào PHẢI active), sample ra có còn bị **>=2 chiều gần 1** không —
tức ngoài đúng chiều c, có chiều KHÁC cũng bị kích hoạt theo không (multi-active),
hay chỉ còn lỗi "sai chiều" (đúng 1 chiều active nhưng KHÔNG PHẢI chiều c)?

Phân loại mỗi sample sinh ra (ngưỡng active = giá trị > 0.5), theo đúng class đã ép:
    correct     : đúng 1 chiều active, VÀ đúng là chiều c được yêu cầu
    wrong_index : đúng 1 chiều active, nhưng KHÔNG PHẢI chiều c (conditioning thất bại)
    collapsed   : 0 chiều active
    multi_active: >=2 chiều active (bất kể có chứa đúng chiều c hay không)

Usage:
    python train_toy_onehot_cond_fm.py --epochs 500
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
from models.nn import timestep_embedding           # noqa: E402

DIM = 16
NUM_CLASSES = 16   # class i <=> chiều i
OUTPUT_DIR = os.path.join(REPO_ROOT, "toy_onehot_cond_fm_output")
CKPT_PATH = os.path.join(OUTPUT_DIR, "mlp_onehot_cond.pt")


# ── Dataset: mỗi class i -> n_per_class vector, chiều i ~ U[0.95,1], còn lại = 0 ──
def make_dataset(n_per_class: int, dim: int = DIM, num_classes: int = NUM_CLASSES, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X_list, y_list = [], []
    for c in range(num_classes):
        vals = torch.empty(n_per_class).uniform_(0.95, 1.0, generator=g)
        X_c = torch.zeros(n_per_class, dim)
        X_c[:, c] = vals
        X_list.append(X_c)
        y_list.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm]


# ── Model: MLP + label embedding (giống tinh thần num_classes của UNetModel) ────
class ToyMLPCond(nn.Module):
    def __init__(self, dim: int = DIM, num_classes: int = NUM_CLASSES,
                 hidden: int = 128, depth: int = 4, t_embed_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.t_embed_dim = t_embed_dim
        # +1 slot null/unconditional (padding_idx), giống UNetModel.label_emb
        self.label_emb = nn.Embedding(num_classes + 1, hidden, padding_idx=num_classes)
        self.in_proj = nn.Linear(dim, hidden)
        blocks = []
        for _ in range(depth):
            blocks += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t_e = timestep_embedding(t, self.t_embed_dim)
        h = self.in_proj(x) + self.t_mlp(t_e)
        if label is None:
            label = torch.full((x.shape[0],), self.num_classes, dtype=torch.long, device=x.device)
        h = h + self.label_emb(label)
        h = self.blocks(h)
        return self.out_proj(h)


class MLPVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor = None, **extras) -> torch.Tensor:
        return self.model(x, t, label=label)


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

    X_data, labels = make_dataset(args.n_per_class, DIM, NUM_CLASSES, seed=args.data_seed)
    X_data, labels = X_data.to(device), labels.to(device)
    print(f"Dataset: {X_data.shape[0]} sample  ({NUM_CLASSES} class x {args.n_per_class}/class)  "
          f"gia tri active ~ U[0.95, 1.0]")

    model = ToyMLPCond(dim=DIM, num_classes=NUM_CLASSES, hidden=args.hidden, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP conditional: hidden={args.hidden} depth={args.depth}  params={n_params:,}  "
          f"class_drop_prob={args.class_drop_prob}")

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
            y_1 = labels[idx]
            B = x_1.shape[0]

            if torch.rand(1).item() < args.class_drop_prob:
                label_in = None
            else:
                label_in = y_1

            x_0 = torch.randn(B, DIM, device=device)
            t = torch.rand(B, device=device)

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            u_pred = model(path_sample.x_t, t, label=label_in)
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

    torch.save({"model_state_dict": model.state_dict(), "hidden": args.hidden, "depth": args.depth},
               CKPT_PATH)
    print(f"\nSaved checkpoint: {CKPT_PATH}")

    analyze(model, device, args.n_sample_per_class, args.sample_steps, args.cfg_scale, args.sample_batch_size)


# ── Sample theo từng class + thống kê correct/wrong_index/collapsed/multi_active ──
@torch.no_grad()
def analyze(model, device, n_per_class: int, steps: int, cfg_scale: float = 0.0, batch_size: int = 4096):
    model.eval()
    wrapper = MLPVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    def sample_batch(B, label_val):
        x_init = torch.randn(B, DIM, device=device)
        label = torch.full((B,), label_val, dtype=torch.long, device=device)
        if cfg_scale != 0.0:
            def vel(x, t, **extra):
                cond = model(x, t, label=label)
                uncond = model(x, t, label=None)
                return (1.0 + cfg_scale) * cond - cfg_scale * uncond
            time_grid_local = time_grid
            x = x_init
            dt = 1.0 / steps
            for i in range(steps):
                t_tensor = torch.full((B,), i * dt, device=device)
                x = x + dt * vel(x, t_tensor)
            return x
        return solver.sample(x_init=x_init, step_size=None, method="euler",
                             time_grid=time_grid, return_intermediates=False, label=label)

    per_class_stats = []
    total = Counter()
    example_multi, example_wrong = [], []

    print(f"\nSampling {n_per_class} vector / class (x{NUM_CLASSES} class), steps={steps}, cfg_scale={cfg_scale} ...")
    for c in range(NUM_CLASSES):
        n_done = 0
        n_correct = n_wrong = n_collapsed = n_multi = 0
        while n_done < n_per_class:
            B = min(batch_size, n_per_class - n_done)
            x_final = sample_batch(B, c)
            active = x_final > 0.5
            n_active = active.sum(dim=1)

            for i in range(B):
                k = int(n_active[i].item())
                if k == 0:
                    n_collapsed += 1
                elif k == 1:
                    idx_active = int(active[i].nonzero()[0, 0].item())
                    if idx_active == c:
                        n_correct += 1
                    else:
                        n_wrong += 1
                        if len(example_wrong) < 5:
                            example_wrong.append((c, idx_active, round(x_final[i, idx_active].item(), 4)))
                else:
                    n_multi += 1
                    if len(example_multi) < 5:
                        example_multi.append((c, x_final[i].cpu().numpy().round(4).tolist()))
            n_done += B

        per_class_stats.append({
            "class": c, "n_total": n_per_class,
            "correct": n_correct, "wrong_index": n_wrong,
            "collapsed": n_collapsed, "multi_active": n_multi,
        })
        total["correct"] += n_correct
        total["wrong_index"] += n_wrong
        total["collapsed"] += n_collapsed
        total["multi_active"] += n_multi

    grand_total = n_per_class * NUM_CLASSES
    print(f"\n{'='*70}")
    print(f"KẾT QUẢ THEO CLASS  (ngưỡng active = giá trị > 0.5, {n_per_class} sample/class)")
    print(f"{'='*70}")
    header = f"{'class':>5s} {'correct%':>9s} {'wrong_idx%':>11s} {'collapsed%':>11s} {'multi_active%':>14s}"
    print(header)
    for s in per_class_stats:
        n = s["n_total"]
        print(f"{s['class']:5d} {100*s['correct']/n:9.2f} {100*s['wrong_index']/n:11.2f} "
              f"{100*s['collapsed']/n:11.2f} {100*s['multi_active']/n:14.2f}")

    print(f"\n{'='*70}")
    print(f"TỔNG HỢP  ({grand_total} sample)")
    print(f"{'='*70}")
    print(f"  correct      : {total['correct']:6d}  ({100*total['correct']/grand_total:.2f}%)")
    print(f"  wrong_index  : {total['wrong_index']:6d}  ({100*total['wrong_index']/grand_total:.2f}%)")
    print(f"  collapsed    : {total['collapsed']:6d}  ({100*total['collapsed']/grand_total:.2f}%)")
    print(f"  multi_active : {total['multi_active']:6d}  ({100*total['multi_active']/grand_total:.2f}%)")

    if example_multi:
        print(f"\nVí dụ multi_active (class ép, vector đầy đủ):")
        for c, v in example_multi:
            print(f"  class={c}: {v}")
    if example_wrong:
        print(f"\nVí dụ wrong_index (class ép, chiều thực tế active, giá trị):")
        for c, idx, val in example_wrong:
            print(f"  class={c} -> chiều active thực tế={idx}  (giá trị={val})")

    stats_path = os.path.join(OUTPUT_DIR, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"N sample/class: {n_per_class}   ODE steps: {steps}   cfg_scale: {cfg_scale}\n\n")
        f.write(header + "\n")
        for s in per_class_stats:
            n = s["n_total"]
            f.write(f"{s['class']:5d} {100*s['correct']/n:9.2f} {100*s['wrong_index']/n:11.2f} "
                    f"{100*s['collapsed']/n:11.2f} {100*s['multi_active']/n:14.2f}\n")
        f.write(f"\nTONG ({grand_total} sample):\n")
        f.write(f"  correct      : {total['correct']} ({100*total['correct']/grand_total:.2f}%)\n")
        f.write(f"  wrong_index  : {total['wrong_index']} ({100*total['wrong_index']/grand_total:.2f}%)\n")
        f.write(f"  collapsed    : {total['collapsed']} ({100*total['collapsed']/grand_total:.2f}%)\n")
        f.write(f"  multi_active : {total['multi_active']} ({100*total['multi_active']/grand_total:.2f}%)\n")
    print(f"\nStats -> {stats_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Toy conditional flow matching: Gauss(0,I_16) -> one-hot theo class")
    p.add_argument("--n_per_class", type=int, default=5000, help="Số sample data MỖI class (default: 5000)")
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--class_drop_prob", type=float, default=0.0,
                   help="Xac suat drop label moi batch de hoc nhanh unconditional (CFG). default: 0.0 (luon co label)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--n_sample_per_class", type=int, default=6250,
                   help="So vector sample MOI class de phan tich (default: 6250, x16 class = 100000)")
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=4096)
    p.add_argument("--cfg_scale", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
