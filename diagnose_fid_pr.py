"""
Chẩn đoán vì sao eval_fid_pr_64x64.py ra FID rất cao (~140-160) và Precision/Recall
~0 ở gần hết mọi config. Giả thuyết: Inception-v3 (train trên ảnh tự nhiên) khi nhìn
ảnh shape đen-trắng tổng hợp (rất đơn giản, ít texture) co cụm feature của ẢNH THẬT
lại rất chặt -> bán kính k-NN (dùng để tính Precision/Recall, Kynkaanniemi et al.)
gần như = 0 -> chỉ cần ảnh sinh ra lệch 1 chút cũng bị tính "ngoài manifold" dù về
hình học vẫn đúng.

Script này (chạy nhanh, n nhỏ, không cần qsub):
  1. Lấy n_real ảnh thật + n_fake ảnh sinh ra (1 checkpoint, 1 mức step).
  2. Tính Inception feature cho cả 2.
  3. In ra: phân phối khoảng cách k-NN GIỮA ảnh thật với nhau (bán kính "manifold thật"),
     và khoảng cách từ mỗi ảnh fake đến ảnh thật gần nhất.
  Nếu "khoảng cách fake->real gần nhất" >> "bán kính real->real k-NN" một cách áp đảo
  (vd gấp vài chục lần) -> xác nhận đúng là bán kính thật quá chặt (domain mismatch),
  không phải do model/script lỗi.

Usage:
    python diagnose_fid_pr.py --arch dit --p0_type anisotropic --steps 50 \
        --n_real 300 --n_fake 300 --device cpu
"""
import argparse

import numpy as np
import torch

import eval_fid_pr_64x64 as ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["unet", "dit"], default="dit")
    p.add_argument("--p0_type", choices=["isotropic", "anisotropic"], default="anisotropic")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n_real", type=int, default=300)
    p.add_argument("--n_fake", type=int, default=300)
    p.add_argument("--nearest_k", type=int, default=ev.NEAREST_K)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = p.parse_args()
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    loader = ev.load_unet if args.arch == "unet" else ev.load_dit
    wrapper, epsilon = loader(args.p0_type, device)
    sigma = None if args.p0_type == "isotropic" else ev.unet_exp.make_sigma(epsilon, ev.IMG_SIZE, device)

    print(f"Loading {args.n_real} real images ...")
    real_imgs = ev.load_real_images(args.n_real)
    print(f"Sampling {args.n_fake} fake images ({args.arch} {args.p0_type} steps={args.steps}) ...")
    fake_imgs = ev.sample_images(wrapper, sigma, device, args.n_fake, args.steps, batch_size=64)

    inception = ev.InceptionV3([ev.InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    real_feats = ev.inception_features(inception, real_imgs, device, batch_size=64)
    fake_feats = ev.inception_features(inception, fake_imgs, device, batch_size=64)
    print(f"real_feats: {real_feats.shape}  fake_feats: {fake_feats.shape}")

    # real -> real k-NN distance (bỏ chính nó, k=nearest_k)
    d_rr = np.linalg.norm(real_feats[:, None, :] - real_feats[None, :, :], axis=-1)
    np.fill_diagonal(d_rr, np.inf)
    knn_rr = np.sort(d_rr, axis=1)[:, args.nearest_k - 1]  # bán kính "manifold thật"

    # fake -> real nearest distance
    d_fr = np.linalg.norm(fake_feats[:, None, :] - real_feats[None, :, :], axis=-1)
    nn_fr = d_fr.min(axis=1)

    print(f"\n[real-real k={args.nearest_k}-NN radius]  "
          f"mean={knn_rr.mean():.3f}  median={np.median(knn_rr):.3f}  "
          f"p90={np.percentile(knn_rr, 90):.3f}  max={knn_rr.max():.3f}")
    print(f"[fake->real nearest dist]          "
          f"mean={nn_fr.mean():.3f}  median={np.median(nn_fr):.3f}  "
          f"p10={np.percentile(nn_fr, 10):.3f}  min={nn_fr.min():.3f}")
    frac_inside = float((nn_fr <= knn_rr.mean()).mean())
    print(f"\n% ảnh fake nằm trong bán kính TRUNG BÌNH của real-real k-NN: {frac_inside*100:.1f}%")
    print("(nếu tỉ lệ này rất thấp dù ảnh fake nhìn bằng mắt vẫn đúng hình -> xác nhận do")
    print(" domain mismatch của Inception feature với ảnh shape tổng hợp, không phải do model.)")

    prdc_res = ev.compute_prdc(real_feats, fake_feats, nearest_k=args.nearest_k)
    print(f"\n[compute_prdc thật, N={args.n_real}] "
          f"precision={prdc_res['precision']:.3f}  recall={prdc_res['recall']:.3f}  "
          f"density={prdc_res['density']:.3f}  coverage={prdc_res['coverage']:.3f}")


if __name__ == "__main__":
    main()
