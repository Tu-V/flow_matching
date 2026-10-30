"""
Table "giống Table 1" của arXiv:2210.12254 (Score-based Denoising Diffusion with
Non-Isotropic Gaussian Noise Models): FID / Precision / Recall ở nhiều số bước
sampling (1000, 100, 50, 20, 10), cho 4 checkpoint ĐÃ TRAIN ở 64x64, n=10000:
  - UNet  isotropic / anisotropic  (shape_exp/unet_architecture/anisotropic_p0_output_64x64)
  - DiT   isotropic / anisotropic  (shape_exp/64_64/anisotropic_p0_dit_64x64_output)

Cách tính (giống paper): Inception-v3 pool3 (2048-dim) feature trên ảnh thật (resize
64x64 -> 299x299, pytorch_fid tự làm qua resize_input=True) vs ảnh sinh ra (cùng số
lượng), rồi FID = Frechet distance giữa 2 phân phối Gauss khớp trên feature space,
Precision/Recall = k-NN manifold estimate (Kynkaanniemi et al. 2019, qua package prdc).

Cần: pip install pytorch-fid prdc

Usage:
    python eval_fid_pr_64x64.py                      # full (5 steps x 4 checkpoint)
    python eval_fid_pr_64x64.py --quick               # cấu hình siêu nhỏ để kiểm tra lỗi
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver                                       # noqa: E402
import train_anisotropic_p0_experiment as unet_exp                               # noqa: E402
import train_anisotropic_p0_experiment_dit as dit_exp                            # noqa: E402
from train_shapes_fm_dit import DiTVelocityWrapper                               # noqa: E402

from scipy import linalg                                                         # noqa: E402
from pytorch_fid.inception import InceptionV3                                    # noqa: E402
from prdc import compute_prdc                                                    # noqa: E402

IMG_SIZE = 64
N_TRAIN = 10000
NEAREST_K = 5  # k-NN cho Precision/Recall (Kynkaanniemi et al. style, qua package prdc)

CKPT_UNET_DIR = os.path.join(
    REPO_ROOT, "shape_exp", "unet_architecture", "anisotropic_p0_output_64x64", "checkpoints")
CKPT_DIT_DIR = os.path.join(
    REPO_ROOT, "shape_exp", "64_64", "anisotropic_p0_dit_64x64_output", "checkpoints")
DATA_DIR = os.path.join(
    REPO_ROOT, "..", "neurips-2024-diffusion-model-hallucination", "simple-datasets",
    f"simple-shapes-{N_TRAIN // 1000}k-{IMG_SIZE}x{IMG_SIZE}")
OUT_DIR = os.path.join(REPO_ROOT, "shape_exp", "64_64", "eval_fid_pr_64x64")


# ── load checkpoint -> (model, wrapper, epsilon) ────────────────────────────
def load_unet(p0_type: str, device):
    path = os.path.join(CKPT_UNET_DIR, f"unet_n{N_TRAIN}_{p0_type}.pt")
    ck = torch.load(path, map_location=device, weights_only=True)
    model = unet_exp.build_unet().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    wrapper = unet_exp.UNetVelocityWrapper(model)
    return wrapper, ck["epsilon"]


def load_dit(p0_type: str, device):
    path = os.path.join(CKPT_DIT_DIR, f"dit_n{N_TRAIN}_{p0_type}.pt")
    ck = torch.load(path, map_location=device, weights_only=True)

    class A:
        pass
    a = A()
    a.patch_size = ck["patch_size"]; a.hidden_size = ck["hidden_size"]
    a.depth = ck["depth"]; a.num_heads = ck["num_heads"]; a.mlp_ratio = ck["mlp_ratio"]
    model = dit_exp.build_dit(IMG_SIZE, a).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    wrapper = DiTVelocityWrapper(model)
    return wrapper, ck["epsilon"]


# ── sample n_total ảnh ở 1 mức steps, trả về tensor [n_total,3,H,W] in [0,1] ─
@torch.no_grad()
def sample_images(wrapper, sigma, device, n_total: int, steps: int, batch_size: int) -> torch.Tensor:
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
    out = []
    n_done = 0
    while n_done < n_total:
        B = min(batch_size, n_total - n_done)
        x_init = unet_exp.sample_x0(B, sigma, IMG_SIZE, device)
        x_final = solver.sample(x_init=x_init, step_size=None, method="euler",
                                 time_grid=time_grid, return_intermediates=False)
        x_01 = (x_final.clamp(-1, 1) + 1) / 2
        x_01_rgb = x_01.repeat(1, 3, 1, 1) if x_01.shape[1] == 1 else x_01
        out.append(x_01_rgb.cpu())
        n_done += B
    return torch.cat(out, dim=0)[:n_total]


# ── load ảnh thật (grayscale -> [0,1] RGB) ──────────────────────────────────
def load_real_images(n_max: int) -> torch.Tensor:
    ds = unet_exp.ShapesDataset(DATA_DIR, IMG_SIZE)
    n = min(n_max, len(ds))
    imgs = torch.stack([ds[i] for i in range(n)])          # [-1,1], 1 kênh
    x_01 = (imgs.clamp(-1, 1) + 1) / 2
    return x_01.repeat(1, 3, 1, 1)


# ── Inception feature extraction (batch) ────────────────────────────────────
@torch.no_grad()
def inception_features(inception, imgs_01: torch.Tensor, device, batch_size: int) -> np.ndarray:
    feats = []
    for i in range(0, imgs_01.shape[0], batch_size):
        batch = imgs_01[i:i + batch_size].to(device)
        f = inception(batch)[0]
        if f.shape[2] != 1 or f.shape[3] != 1:
            f = torch.nn.functional.adaptive_avg_pool2d(f, output_size=(1, 1))
        feats.append(f.squeeze(3).squeeze(2).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _sqrtm(mat: np.ndarray) -> np.ndarray:
    # scipy >=1.13 bỏ tham số `disp` của linalg.sqrtm (và trả thẳng ma trận
    # thay vì tuple (mat, errest)) -> tự xử lý cả 2 API để không phụ thuộc
    # version scipy cụ thể trên máy local vs Gadi.
    res = linalg.sqrtm(mat)
    return res[0] if isinstance(res, tuple) else res


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean = _sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def fid_from_features(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=5000, help="số ảnh sinh ra / checkpoint / steps")
    p.add_argument("--n_real", type=int, default=10000, help="số ảnh thật dùng làm reference")
    p.add_argument("--steps_list", type=str, default="1000,100,50,20,10")
    p.add_argument("--batch_size", type=int, default=128, help="batch size khi sample ODE")
    p.add_argument("--inception_batch_size", type=int, default=128)
    p.add_argument("--nearest_k", type=int, default=NEAREST_K)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--quick", action="store_true", help="cấu hình siêu nhỏ để kiểm tra lỗi")
    a = p.parse_args()
    if a.quick:
        a.n_samples, a.n_real = 16, 32
        a.steps_list = "5,2"
        a.batch_size, a.inception_batch_size = 16, 16
        a.nearest_k = 3
    return a


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")
    steps_list = [int(s) for s in args.steps_list.split(",")]
    out_dir = os.path.join(OUT_DIR, f"nreal{args.n_real}_nsamp{args.n_samples}")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "results.txt")
    report = open(report_path, "w")

    def rep(s=""):
        print(s)
        report.write(s + "\n")
        report.flush()

    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    rep(f"Loading {args.n_real} real images from {DATA_DIR} ...")
    real_imgs = load_real_images(args.n_real)
    real_feats = inception_features(inception, real_imgs, device, args.inception_batch_size)
    rep(f"real_feats: {real_feats.shape}")
    del real_imgs

    configs = [
        ("UNet", "isotropic", load_unet),
        ("UNet", "anisotropic", load_unet),
        ("DiT", "isotropic", load_dit),
        ("DiT", "anisotropic", load_dit),
    ]

    rows = []
    for arch, p0_type, loader in configs:
        wrapper, epsilon = loader(p0_type, device)
        sigma = None if p0_type == "isotropic" else unet_exp.make_sigma(epsilon, IMG_SIZE, device)
        rep(f"\n{'='*70}\n{arch}  p0={p0_type}  epsilon={epsilon}\n{'='*70}")
        for steps in steps_list:
            t0 = time.time()
            fake_imgs = sample_images(wrapper, sigma, device, args.n_samples, steps, args.batch_size)
            fake_feats = inception_features(inception, fake_imgs, device, args.inception_batch_size)
            del fake_imgs
            fid = fid_from_features(real_feats, fake_feats)
            prdc_res = compute_prdc(real_feats, fake_feats, nearest_k=args.nearest_k)
            rep(f"  steps={steps:<5d}  FID={fid:8.3f}  P={prdc_res['precision']:.3f}  "
                f"R={prdc_res['recall']:.3f}  (n_samples={args.n_samples}, "
                f"elapsed={time.time()-t0:.0f}s)")
            rows.append({"arch": arch, "p0_type": p0_type, "steps": steps,
                         "FID": fid, "precision": prdc_res["precision"],
                         "recall": prdc_res["recall"], "density": prdc_res["density"],
                         "coverage": prdc_res["coverage"], "n_samples": args.n_samples,
                         "n_real": real_feats.shape[0]})
        del wrapper

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "p0_type", "steps", "FID", "precision",
                                           "recall", "density", "coverage", "n_samples", "n_real"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    rep(f"\nĐã lưu: {report_path}, {csv_path}")
    report.close()


if __name__ == "__main__":
    main()
