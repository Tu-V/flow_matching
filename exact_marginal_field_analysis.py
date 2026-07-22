"""
So sánh velocity field ĐÃ HỌC (UNet) vs field LÝ TƯỞNG closed-form (marginal field
chính xác, suy ra trực tiếp từ N ảnh training thật, không qua mạng nơ-ron nào).

Field lý tưởng (CondOT, xem derivation trong tin nhắn):
    w_i(x_t,t)   = softmax_i( -||x_t - t*x_1^(i)||^2 / (2*(1-t)^2) )
    u_exact(x_t) = ( sum_i w_i * x_1^(i)  -  x_t ) / (1-t)

Chạy ODE bằng field NÀY (không phải UNet) từ đúng x_T của 1 case hallucination:
  - Nếu field lý tưởng CŨNG hallucinate  -> bản chất finite-N (mixture) gây ra,
    không phải lỗi riêng của UNet.
  - Nếu field lý tưởng ra ảnh sạch (hội tụ đúng 1 ảnh training)  -> hallucination
    là lỗi xấp xỉ của mạng, không phải thuộc tính vốn có của target.

Đồng thời track w_i(x_t,t) dọc trajectory -> định danh chính xác (các) ảnh
training có trọng số cao nhất, xem có đúng "2 ảnh bị trộn" không.

Usage:
    python exact_marginal_field_analysis.py \\
        --case_dir shapes_fm_output/hallucination_analysis/traces/HALL_case_0109_idx92812 \\
        --ckpt shapes_fm_output/checkpoints/unet_epoch1000.pt \\
        --data_dir ../neurips-2024-diffusion-model-hallucination/simple-datasets/simple-shapes-5k-16x16 \\
        --steps 25
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from sample_shapes_fm import load_model, UNetVelocityWrapper, decode, to_uint8_numpy, _latest_ckpt   # noqa: E402
from hallucination_detector import analyze_image   # noqa: E402

IMG_SIZE = 16
ZOOM = 8


def upscale(img, z=ZOOM):
    return img.repeat(z, axis=0).repeat(z, axis=1)


def load_training_images_model_scale(data_dir, device):
    """Trả về (N,3,16,16) tensor trong scale [-1,1] giống model, + list path."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    arrs = []
    for p in paths:
        gray = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        rgb = np.repeat(gray[None, :, :], 3, axis=0) * 2 - 1
        arrs.append(rgb)
    X1 = torch.from_numpy(np.stack(arrs).astype(np.float32)).to(device)   # (N,3,16,16)
    return X1, paths


@torch.no_grad()
def exact_marginal_velocity(x_t, t_val, X1_flat, X1_norm_sq):
    """
    x_t: (1, D) đã flatten.  X1_flat: (N, D).  X1_norm_sq: (N,) = ||x_1^(i)||^2.
    Trả về (u_exact: (1,D), weights: (N,)).
    """
    # ||x_t - t*x_1||^2 = ||x_t||^2 - 2t<x_t,x_1> + t^2 ||x_1||^2
    xt_norm_sq = (x_t ** 2).sum()
    cross = (x_t @ X1_flat.T).squeeze(0)               # (N,)
    dist_sq = xt_norm_sq - 2 * t_val * cross + (t_val ** 2) * X1_norm_sq
    logits = -dist_sq / (2 * (1 - t_val) ** 2 + 1e-12)
    weights = torch.softmax(logits, dim=0)              # (N,)
    x1_hat = (weights.unsqueeze(1) * X1_flat).sum(dim=0, keepdim=True)   # (1,D)
    u = (x1_hat - x_t) / (1 - t_val)
    return u, weights


def euler_rollout_exact(x_T_flat, steps, X1_flat, X1_norm_sq, device, track_weights=True):
    dt = 1.0 / steps
    x = x_T_flat.clone()
    weight_history = []
    for i in range(steps):
        t_val = i * dt
        u, w = exact_marginal_velocity(x, t_val, X1_flat, X1_norm_sq)
        if track_weights:
            weight_history.append(w.cpu().numpy())
        x = x + dt * u
    return x, weight_history


@torch.no_grad()
def euler_rollout_unet(wrapper, x_init, steps, device):
    dt = 1.0 / steps
    x = x_init
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)
        v = wrapper(x, t_tensor)
        x = x + dt * v
    return x


def to_model_uint8(x_flat_or_chw):
    """(3,16,16) hoặc flat (768,) tensor scale [-1,1] -> (16,16,3) uint8."""
    x = x_flat_or_chw.reshape(3, IMG_SIZE, IMG_SIZE)
    img01 = ((x.clamp(-1, 1) + 1) / 2)
    return (img01 * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.case_dir, "exact_marginal_field")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device: {device}")

    print(f"Loading training images from {args.data_dir} ...")
    X1_all, paths = load_training_images_model_scale(args.data_dir, device)   # (N,3,16,16)
    N = X1_all.shape[0]
    X1_flat = X1_all.reshape(N, -1)                     # (N, 768)
    X1_norm_sq = (X1_flat ** 2).sum(dim=1)               # (N,)
    print(f"N training images = {N}, D = {X1_flat.shape[1]}")

    ckpt_path = args.ckpt or _latest_ckpt()
    model = load_model(ckpt_path, device)
    wrapper = UNetVelocityWrapper(model)

    x_T = torch.load(os.path.join(args.case_dir, "noise_init.pt"), weights_only=True).to(device)  # (1,3,16,16)

    # ── (1) UNet đã học ──────────────────────────────────────────────────────
    x_0_unet = euler_rollout_unet(wrapper, x_T, args.steps, device)
    img_unet = to_uint8_numpy(decode(x_0_unet)[0])
    r_unet = analyze_image(img_unet)
    print(f"\n[UNet đã học]      hall={r_unet['is_hallucination']}  type={r_unet['hall_type']}  "
          f"blobs={r_unet['col_blobs']}")

    # ── (2) Field lý tưởng exact ──────────────────────────────────────────────
    x_T_flat = x_T.reshape(1, -1)
    x_0_exact_flat, weight_history = euler_rollout_exact(
        x_T_flat, args.steps, X1_flat, X1_norm_sq, device)
    img_exact = to_model_uint8(x_0_exact_flat[0])
    r_exact = analyze_image(img_exact)
    print(f"[Field lý tưởng]   hall={r_exact['is_hallucination']}  type={r_exact['hall_type']}  "
          f"blobs={r_exact['col_blobs']}")

    # ── Top-K ảnh training có trọng số cao nhất ở BƯỚC CUỐI (t gần 1 nhất) ────
    final_weights = weight_history[-1]
    top_idx = np.argsort(final_weights)[::-1][:args.top_k]
    print(f"\nTop {args.top_k} ảnh training có trọng số cao nhất (bước cuối, t={(args.steps-1)/args.steps:.2f}):")
    for rank, idx in enumerate(top_idx):
        print(f"  #{rank+1}: {os.path.basename(paths[idx])}  weight={final_weights[idx]:.4f}")

    # ── Vẽ ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, max(4, args.top_k), figsize=(max(4, args.top_k) * 2.2, 5.5))

    axes[0, 0].imshow(upscale(img_unet))
    axes[0, 0].set_title(f"UNet (đã học)\n{r_unet['hall_type']}  {r_unet['col_blobs']}", fontsize=8)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(upscale(img_exact))
    axes[0, 1].set_title(f"Field lý tưởng (exact)\n{r_exact['hall_type']}  {r_exact['col_blobs']}", fontsize=8)
    axes[0, 1].axis("off")

    for ax in axes[0, 2:]:
        ax.axis("off")

    for k in range(args.top_k):
        idx = top_idx[k]
        train_img = to_model_uint8(X1_flat[idx])
        axes[1, k].imshow(upscale(train_img))
        axes[1, k].set_title(f"train #{idx}\nw={final_weights[idx]:.3f}", fontsize=7)
        axes[1, k].axis("off")
    for ax in axes[1, args.top_k:]:
        ax.axis("off")

    fig.suptitle(f"{os.path.basename(args.case_dir)} — UNet vs Field lý tưởng (exact, N={N})", fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "unet_vs_exact.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_path}")

    # ── Đường cong entropy / effective-N của weight qua trajectory ───────────
    entropies = [-(w * np.log(w + 1e-12)).sum() for w in weight_history]
    eff_n = [np.exp(e) for e in entropies]   # "effective number of contributing images"
    max_w = [w.max() for w in weight_history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    t_vals = [i / args.steps for i in range(args.steps)]
    axes[0].plot(t_vals, eff_n, marker=".", color="steelblue")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("effective # ảnh đóng góp  (exp(entropy))")
    axes[0].set_title("Bao nhiêu ảnh training đang 'tranh chấp' tại mỗi t")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_vals, max_w, marker=".", color="crimson")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("max weight (ảnh mạnh nhất)")
    axes[1].set_title("Độ 'chắc chắn' của field lý tưởng theo t")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(out_dir, "weight_dynamics.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {curve_path}")

    np.savez(os.path.join(out_dir, "exact_field_data.npz"),
             img_unet=img_unet, img_exact=img_exact,
             final_weights=final_weights, top_idx=top_idx,
             eff_n=eff_n, max_w=max_w, paths=np.array(paths, dtype=object))
    print(f"Saved: {os.path.join(out_dir, 'exact_field_data.npz')}")


if __name__ == "__main__":
    main()
