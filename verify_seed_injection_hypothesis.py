"""
Kiểm chứng giả thuyết: "chỉ cần tại 1 bước ODE nào đó, có 1 vùng pixel đã tăng lên
đủ cao (>=0.5-0.6), thì xung quanh vùng đó model sẽ tiếp tục hoàn thiện thành 1
shape đầy đủ" — tức việc "commit" vào 1 shape xảy ra CỤC BỘ (dựa vào tín hiệu local
đã hình thành), không phải quyết định toàn cục ngay từ đầu.

Cách làm (đúng theo yêu cầu):
    1. Lấy 1 case hallucination CÓ SẴN (double_col, class1, đã có 2 triangle ở cột 0)
       — dùng lại intermediates.pt đã trace sẵn (không train/generate lại từ đầu).
    2. Tại bước 50/100 (t=0.5), quan sát cột 0: đã có 2 vùng đang sáng lên (blob trên
       ~hàng 2-6, blob dưới ~hàng 13-15), còn hàng 7-12 vẫn là nền (chưa có gì).
    3. COPY patch của vùng đang sáng (blob trên) và PASTE vào giữa (hàng 8-12, vùng
       trống, KHÔNG trùng với 2 blob đã có) -> tạo "mầm" nhân tạo thứ 3.
    4. Tiếp tục chạy ODE từ bước 51 -> 100 (t=0.5 -> 1.0) với state đã sửa, dùng ĐÚNG
       model + conditioning (class1) như lúc trace gốc.
    5. Kiểm tra ảnh cuối: có hình thành đủ 3 shape trong cùng cột 0 không?

Nếu đúng giả thuyết: mầm nhân tạo ở hàng 8-12 cũng phát triển thành 1 triangle đầy đủ
-> ảnh cuối có 3 triangle trong cột 0 (thay vì 2) -> xác nhận cơ chế "commit cục bộ".

Usage:
    python verify_seed_injection_hypothesis.py \
        --case_dir shapes_fm_cond_output/hallucination_analysis_cond_unet_cond_epoch1000/class1/traces/case_0002_idx14270 \
        --ckpt shapes_fm_cond_output/checkpoints/unet_cond_epoch1000.pt
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver                # noqa: E402
from hallucination_detector import analyze_image, COLUMN_SLICES, COLUMN_NAMES   # noqa: E402
from train_shapes_fm_cond import build_unet, CFGScaledModel, CLASS_DESC   # noqa: E402
from train_shapes_fm_dit import DiT, DiTVelocityWrapper     # noqa: E402

IMG_SIZE = 16
SCALE = 8


def decode(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(-1, 1) + 1) / 2


def to_uint8_numpy(img_01: torch.Tensor):
    import numpy as np
    return (img_01.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def add_red_dividers(img_01: torch.Tensor, scale: int = SCALE) -> torch.Tensor:
    up = F.interpolate(img_01.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)
    for x in [5 * scale, 10 * scale]:
        up[0, :, x:x + 2] = 1.0
        up[1, :, x:x + 2] = 0.0
        up[2, :, x:x + 2] = 0.0
    return up


@torch.no_grad()
def continue_ode(wrapper, x_start: torch.Tensor, t_start: float, t_end: float,
                 n_steps: int, label, cfg_scale: float, device):
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.linspace(t_start, t_end, n_steps + 1, device=device)
    traj = solver.sample(
        x_init=x_start, step_size=None, method="euler", time_grid=time_grid,
        return_intermediates=True, cfg_scale=cfg_scale, label=label,
    )
    return traj   # (n_steps+1, B, 3, 16, 16)


def run(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if args.ckpt is None:
        if args.arch == "unet_cond":
            args.ckpt = os.path.join(REPO_ROOT, "shapes_fm_cond_output", "checkpoints", "unet_cond_epoch1000.pt")
        else:
            args.ckpt = os.path.join(REPO_ROOT, "shapes_fm_dit_large_output", "checkpoints", "dit_epoch1000.pt")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)

    if args.arch == "unet_cond":
        model = build_unet().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        wrapper = CFGScaledModel(model)
        class_idx = args.class_idx
        label = torch.tensor([class_idx], dtype=torch.long, device=device)
        print(f"Loaded {args.ckpt}  (epoch={ckpt.get('epoch','?')}, arch=unet_cond)")
        print(f"Class: {class_idx} ({CLASS_DESC[class_idx]})")
    else:   # dit — unconditional, khong co label
        model = DiT(
            img_size=IMG_SIZE, patch_size=ckpt["patch_size"], in_channels=3,
            hidden_size=ckpt["hidden_size"], depth=ckpt["depth"],
            num_heads=ckpt["num_heads"], mlp_ratio=ckpt["mlp_ratio"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        wrapper = DiTVelocityWrapper(model)
        label = None
        print(f"Loaded {args.ckpt}  (epoch={ckpt.get('epoch','?')}, arch=dit, "
              f"hidden={ckpt['hidden_size']}, depth={ckpt['depth']})")
        print("Model unconditional (khong co class label)")
    model.eval()

    # ── Load case gốc ──
    inter = torch.load(os.path.join(args.case_dir, "intermediates.pt"), weights_only=True)  # (101,3,16,16)
    n_steps_total = inter.shape[0] - 1
    print(f"intermediates: {inter.shape}  (n_steps_total={n_steps_total})")

    col_name = COLUMN_NAMES[args.col]
    col0, col1 = COLUMN_SLICES[args.col]
    print(f"Cot dang xet: {col_name}  (x[{col0}:{col1}])")

    out_dir = os.path.join(args.case_dir, "seed_injection_experiment")
    os.makedirs(out_dir, exist_ok=True)

    # ── Buoc step_mid: lay state, quan sat cot dang xet ──
    step_mid = args.step_mid
    t_mid = step_mid / n_steps_total
    x_mid = inter[step_mid].unsqueeze(0).clone()   # (1,3,16,16)
    gray_mid = x_mid[0].mean(dim=0)
    print(f"\nGray value cot {col_name} (x[{col0}:{col1}]) tai buoc {step_mid} (t={t_mid:.2f}):")
    for r in range(IMG_SIZE):
        print(f"  row {r:2d}: " + " ".join(f"{v.item():+.2f}" for v in gray_mid[r, col0:col1]))

    # ── Copy patch tu vung "dang sang" (blob, args.src_rows) sang vung trong
    #    (args.dst_rows), CA 3 kenh, KHONG chong lan voi vung da co gia tri cao ──
    src_r0, src_r1 = args.src_rows
    dst_r0, dst_r1 = args.dst_rows
    assert (src_r1 - src_r0) == (dst_r1 - dst_r0), "src/dst rows phai cung kich thuoc"

    x_mid_injected = x_mid.clone()
    patch = x_mid[:, :, src_r0:src_r1, col0:col1].clone()   # (1,3,h,w)
    x_mid_injected[:, :, dst_r0:dst_r1, col0:col1] = patch

    print(f"\nDa copy patch hang [{src_r0}:{src_r1}] -> [{dst_r0}:{dst_r1}] (cot {col_name})")

    # ── Luu anh so sanh truoc/sau khi inject (decode tam thoi de xem, chi de debug) ──
    img_before = decode(x_mid[0])
    img_after = decode(x_mid_injected[0])
    save_image(
        make_grid([add_red_dividers(img_before), add_red_dividers(img_after)], nrow=2, padding=4, pad_value=0.5),
        os.path.join(out_dir, f"step{step_mid}_before_after_injection.png"),
    )

    # ── Chay tiep ODE tu step_mid -> het, VOI STATE DA INJECT ──
    n_remaining = n_steps_total - step_mid
    print(f"\nChay tiep ODE: t={t_mid:.2f} -> 1.0  ({n_remaining} buoc con lai)  cfg_scale={args.cfg_scale}")
    traj_injected = continue_ode(
        wrapper, x_mid_injected.to(device), t_mid, 1.0, n_remaining,
        label, args.cfg_scale, device,
    )
    x_final_injected = traj_injected[-1, 0].cpu()

    # ── (Doi chieu) Chay tiep ODE KHONG inject, chi de xac nhan trung khop voi ban goc ──
    traj_control = continue_ode(
        wrapper, x_mid.to(device), t_mid, 1.0, n_remaining,
        label, args.cfg_scale, device,
    )
    x_final_control = traj_control[-1, 0].cpu()
    x_final_original = inter[-1]   # ban goc that (da trace san, khong inject gi)

    diff_control_vs_original = (x_final_control - x_final_original).abs().max().item()
    print(f"\n[Sanity check] max|control_replay - ban_goc_that| = {diff_control_vs_original:.6f}  "
          f"(cang gan 0 cang chung to replay dung logic goc)")

    # ── Phan tich shape ──
    img_final_injected_01 = decode(x_final_injected)
    img_final_original_01 = decode(x_final_original)
    a_injected = analyze_image(to_uint8_numpy(img_final_injected_01))
    a_original = analyze_image(to_uint8_numpy(img_final_original_01))

    print(f"\n{'='*60}")
    print(f"KET QUA")
    print(f"{'='*60}")
    print(f"Anh GOC (khong inject)   : col_blobs={a_original['col_blobs']}  "
          f"type={a_original['hall_type']}")
    print(f"Anh SAU KHI INJECT       : col_blobs={a_injected['col_blobs']}  "
          f"type={a_injected['hall_type']}")

    n_shape_before = a_original['col_blobs'][col_name]
    n_shape_after = a_injected['col_blobs'][col_name]
    if n_shape_after > n_shape_before:
        print(f"\n=> XAC NHAN gia thuyet: so {col_name} tang tu {n_shape_before} -> {n_shape_after} "
              f"sau khi inject 'mam' nhan tao vao vung trong.")
    else:
        print(f"\n=> KHONG xac nhan: so {col_name} KHONG tang ({n_shape_before} -> {n_shape_after}) "
              f"— vung inject co the da bi 'xoa' di trong qua trinh ODE con lai.")

    # ── Luu anh cuoi cung, so sanh ──
    grid = make_grid(
        [add_red_dividers(img_final_original_01), add_red_dividers(img_final_injected_01)],
        nrow=2, padding=4, pad_value=0.5,
    )
    save_image(grid, os.path.join(out_dir, "final_compare_original_vs_injected.png"))

    # ── Luu progression cua nhanh injected (step_mid -> 100), vai buoc dai dien ──
    n_strip = 10
    strip_idx = torch.linspace(0, n_remaining, n_strip, dtype=torch.long)
    strip_imgs = [add_red_dividers(decode(traj_injected[i, 0].cpu())) for i in strip_idx]
    save_image(make_grid(torch.stack(strip_imgs), nrow=n_strip, padding=2, pad_value=0.3),
              os.path.join(out_dir, "progression_after_injection.png"))

    torch.save(x_mid.cpu(), os.path.join(out_dir, f"step{step_mid}_original.pt"))
    torch.save(x_mid_injected.cpu(), os.path.join(out_dir, f"step{step_mid}_injected.pt"))
    torch.save(traj_injected.cpu(), os.path.join(out_dir, "trajectory_injected.pt"))
    torch.save(x_final_injected, os.path.join(out_dir, "final_injected.pt"))

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"Case goc: {args.case_dir}\n")
        f.write(f"Arch: {args.arch}\n")
        if args.arch == "unet_cond":
            f.write(f"Class: {class_idx} ({CLASS_DESC[class_idx]})\n")
        else:
            f.write("Unconditional (khong co class label)\n")
        f.write(f"Cot: {col_name} (x[{col0}:{col1}])\n")
        f.write(f"Inject tai step {step_mid}/{n_steps_total} (t={t_mid:.2f})\n")
        f.write(f"  src_rows={args.src_rows}  ->  dst_rows={args.dst_rows}  (cot {col_name})\n\n")
        f.write(f"Sanity check max|control_replay - ban_goc| = {diff_control_vs_original:.6f}\n\n")
        f.write(f"Anh GOC (khong inject)   : col_blobs={a_original['col_blobs']}  type={a_original['hall_type']}\n")
        f.write(f"Anh SAU KHI INJECT       : col_blobs={a_injected['col_blobs']}  type={a_injected['hall_type']}\n\n")
        if n_shape_after > n_shape_before:
            f.write(f"XAC NHAN gia thuyet: so {col_name} tang {n_shape_before} -> {n_shape_after}\n")
        else:
            f.write(f"KHONG xac nhan: so {col_name} khong tang ({n_shape_before} -> {n_shape_after})\n")

    print(f"\nSaved -> {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case_dir", type=str, required=True)
    p.add_argument("--arch", type=str, default="unet_cond", choices=["unet_cond", "dit"])
    p.add_argument("--ckpt", type=str, default=None,
                   help="default: unet_cond_epoch1000.pt hoac dit_epoch1000.pt tuy --arch")
    p.add_argument("--class_idx", type=int, default=0, help="0-indexed (class1 user -> 0), chi dung khi arch=unet_cond")
    p.add_argument("--col", type=int, default=0, choices=[0, 1, 2],
                   help="0=triangle, 1=square, 2=pentagon")
    p.add_argument("--step_mid", type=int, default=50)
    p.add_argument("--src_rows", type=int, nargs=2, default=[2, 7], help="[start,end) hang lay patch (vung dang sang)")
    p.add_argument("--dst_rows", type=int, nargs=2, default=[8, 13], help="[start,end) hang dan patch vao (vung trong)")
    p.add_argument("--cfg_scale", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
