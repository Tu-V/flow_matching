"""
Chạy LẠI riêng bước SAMPLING (không train lại) từ checkpoint đã có của
train_anisotropic_p0_experiment_dit.py, để kiểm tra 1 kết quả có ổn định/lặp lại
được không (vd n=5000 img16: isotropic=0.2518%, anisotropic=0.6178%, +145.4% --
ngược hướng mong đợi -- cần xem đây là nhiễu do sampling hay thật).

Load checkpoint CUỐI (dit_n{n_size}_{p0_type}_img{img_size}.pt) cho cả isotropic
và anisotropic, sample LẠI --n_sample_total ảnh x --n_repeats lần MỖI nhánh (x_init
random MỚI, seed khác lúc train), in + lưu kết quả riêng (không ghi đè
summary_anisotropic_p0_dit_img16.txt gốc).

Usage:
    python resample_dit16.py                       # n=5000, img=16 (mặc định)
    python resample_dit16.py --n_size 10000
    python resample_dit16.py --n_repeats 10 --seed 123
"""

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

import train_anisotropic_p0_experiment_dit as exp   # noqa: E402  (tái dùng build_dit/make_sigma/sample_and_analyze_repeated)

CKPT_DIR = os.path.join(exp.OUTPUT_DIR, "checkpoints")


class _Args:
    pass


def load_model(p0_type: str, n_size: int, img_size: int, device):
    ckpt_path = os.path.join(CKPT_DIR, f"dit_n{n_size}_{p0_type}_img{img_size}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    a = _Args()
    a.patch_size, a.hidden_size, a.depth = ck["patch_size"], ck["hidden_size"], ck["depth"]
    a.num_heads, a.mlp_ratio = ck["num_heads"], ck["mlp_ratio"]
    model = exp.build_dit(img_size, a).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_size", type=int, default=5000)
    p.add_argument("--img_size", type=int, default=16)
    p.add_argument("--n_sample_total", type=int, default=100000)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}  RESAMPLE n={args.n_size}  img_size={args.img_size}  "
          f"(seed={args.seed}, khác lúc train)")

    results = {}
    for p0_type in ["isotropic", "anisotropic"]:
        model, ck = load_model(p0_type, args.n_size, args.img_size, device)
        sigma = (exp.make_sigma(ck["epsilon"], args.img_size, device)
                 if p0_type == "anisotropic" else None)
        torch.manual_seed(args.seed)
        print(f"\n=== p0={p0_type}  (ckpt step={ck.get('step')}, epsilon={ck.get('epsilon')}) ===")
        s = exp.sample_and_analyze_repeated(model, sigma, device, args.n_sample_total,
                                             args.sample_steps, args.sample_batch_size,
                                             args.n_repeats, args.img_size)
        print(f"  n={args.n_size} p0={p0_type:12s}  hall={s['hall_rate_mean']:.4f}%±{s['hall_rate_std']:.4f}%  "
              f"(empty={s['empty_rate_mean']:.4f}%  double_col={s['double_rate_mean']:.4f}%)")
        results[p0_type] = s
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    iso, ani = results["isotropic"], results["anisotropic"]
    rel = ("(giảm về 0)" if ani["hall_rate_mean"] == 0 else
           f"({100*(ani['hall_rate_mean']-iso['hall_rate_mean'])/max(iso['hall_rate_mean'],1e-12):+.1f}% tương đối)")
    summary = (f"n={args.n_size}: isotropic={iso['hall_rate_mean']:.4f}%±{iso['hall_rate_std']:.4f}%  ->  "
               f"anisotropic={ani['hall_rate_mean']:.4f}%±{ani['hall_rate_std']:.4f}%  {rel}")
    print(f"\n{summary}")

    out_path = os.path.join(exp.OUTPUT_DIR, f"resample_n{args.n_size}_img{args.img_size}_seed{args.seed}.txt")
    with open(out_path, "w") as f:
        f.write(f"RESAMPLE (dùng lại checkpoint có sẵn, KHÔNG train lại)  n={args.n_size}  "
                f"img_size={args.img_size}  seed={args.seed}\n")
        f.write(f"n_sample_total={args.n_sample_total}  n_repeats={args.n_repeats}  "
                f"sample_steps={args.sample_steps}\n\n")
        for p0_type in ["isotropic", "anisotropic"]:
            s = results[p0_type]
            f.write(f"{p0_type}: hall%={s['hall_rate_mean']:.4f}±{s['hall_rate_std']:.4f}  "
                    f"empty%={s['empty_rate_mean']:.4f}  double_col%={s['double_rate_mean']:.4f}\n")
            for rep, hr in enumerate(s["hall_rate_pct_list"], start=1):
                f.write(f"    repeat {rep}: hall={hr:.4f}%  empty={s['empty_rate_pct_list'][rep-1]:.4f}%  "
                        f"double_col={s['double_rate_pct_list'][rep-1]:.4f}%\n")
        f.write(f"\n{summary}\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
