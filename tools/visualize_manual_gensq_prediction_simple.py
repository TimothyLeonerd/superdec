#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    PointNetScaleExp,
    chamfer,
)
from tools.train_gensq_weighted_pca_fps_clean_lr import (
    WeightedPCANet,
    hard_pca_frame,
)
from tools.visualize_manual_gensq_prediction import (
    parse3,
    safe_sample_gensq,
    best_perm,
    make_points,
    subsample_np,
    set_equal,
    style_ax,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp-ckpt", required=True)
    ap.add_argument("--wpca-ckpt", required=True)

    ap.add_argument("--scale", required=True)
    ap.add_argument("--exp", required=True)

    ap.add_argument("--sampler", choices=["fps", "direct"], default="fps")
    ap.add_argument("--frame", choices=["weighted_pca", "hard_pca"], default="weighted_pca")
    ap.add_argument("--layout", choices=["overlay", "side_by_side"], default="overlay")

    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--candidate-n", type=int, default=2048)
    ap.add_argument("--surface-n", type=int, default=4096)
    ap.add_argument("--plot-n", type=int, default=2500)
    ap.add_argument("--dropout-keep", type=float, default=1.0)

    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=1.0)
    ap.add_argument("--exp-max", type=float, default=8.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    gt_scale_original = torch.tensor([parse3(args.scale)], dtype=torch.float32, device=device)
    gt_exp_original = torch.tensor([parse3(args.exp)], dtype=torch.float32, device=device)

    model = PointNetScaleExp(args.scale_min, args.scale_max, args.exp_min, args.exp_max).to(device)
    ckpt = torch.load(args.exp_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wpca = WeightedPCANet().to(device)
    wckpt = torch.load(args.wpca_ckpt, map_location=device)
    wpca.load_state_dict(wckpt["model"])
    wpca.eval()

    with torch.no_grad():
        world, gt_R = make_points(gt_scale_original, gt_exp_original, args, device)

        if args.frame == "weighted_pca":
            frame, _, _ = wpca(world)
        else:
            frame, _ = hard_pca_frame(world)

        canon = world @ frame

        perm, _ = best_perm(frame, gt_R)
        gt_scale = gt_scale_original.gather(1, perm)
        gt_exp = gt_exp_original.gather(1, perm)

        pred_scale, pred_exp = model(canon)

        scale_abs = (pred_scale - gt_scale).abs()[0]
        exp_abs = (pred_exp - gt_exp).abs()[0]

        scale_l1 = float(scale_abs.mean())
        exp_l1 = float(exp_abs.mean())
        scale_rel_pct = float((scale_abs / gt_scale[0].clamp_min(1e-8)).mean() * 100.0)

        dirs = fibonacci_sphere(args.surface_n, device)
        gt_surf = safe_sample_gensq(gt_scale, gt_exp, dirs)
        pred_surf = safe_sample_gensq(pred_scale, pred_exp, dirs)

        cd_gt_pred = float(chamfer(gt_surf, pred_surf)[0])
        cd_input_gt = float(chamfer(canon, gt_surf)[0])
        cd_input_pred = float(chamfer(canon, pred_surf)[0])

    gt_np = gt_surf[0].cpu().numpy()
    pred_np = pred_surf[0].cpu().numpy()
    gt_plot = subsample_np(gt_np, args.plot_n, args.seed + 1)
    pred_plot = subsample_np(pred_np, args.plot_n, args.seed + 2)

    print()
    print(f"saved: {out}")
    print(f"sampler={args.sampler}, frame={args.frame}, seed={args.seed}, dropout_keep={args.dropout_keep}")
    print()
    print(f"GT scale original: {gt_scale_original[0].cpu().numpy().round(5).tolist()}")
    print(f"GT exp original:   {gt_exp_original[0].cpu().numpy().round(5).tolist()}")
    print(f"GT scale matched:  {gt_scale[0].cpu().numpy().round(5).tolist()}")
    print(f"GT exp matched:    {gt_exp[0].cpu().numpy().round(5).tolist()}")
    print()
    print(f"Pred scale:        {pred_scale[0].cpu().numpy().round(5).tolist()}")
    print(f"Pred exp:          {pred_exp[0].cpu().numpy().round(5).tolist()}")
    print()
    print(f"scale_abs_err:     {scale_abs.cpu().numpy().round(5).tolist()}")
    print(f"exp_abs_err:       {exp_abs.cpu().numpy().round(5).tolist()}")
    print(f"scale_l1:          {scale_l1:.6f}")
    print(f"scale_rel_mean:    {scale_rel_pct:.2f}%")
    print(f"exp_l1:            {exp_l1:.6f}")
    print()
    print(f"cd_gt_pred_surface: {cd_gt_pred:.8f}")
    print(f"cd_input_gt:         {cd_input_gt:.8f}")
    print(f"cd_input_pred:       {cd_input_pred:.8f}")

    title = (
        f"scale_l1={scale_l1:.4f}, exp_l1={exp_l1:.4f}, "
        f"CD(gt,pred)={cd_gt_pred:.6f}"
    )

    if args.layout == "overlay":
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], s=3, alpha=0.40, label="GT")
        ax.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], s=3, alpha=0.40, label="pred")
        style_ax(ax, "GT vs predicted SQ")
        ax.legend()
        set_equal(ax, [gt_plot, pred_plot])
    else:
        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], s=3, alpha=0.45)
        style_ax(ax1, "GT SQ")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], s=3, alpha=0.45)
        style_ax(ax2, "Predicted SQ")

        for ax in [ax1, ax2]:
            set_equal(ax, [gt_plot, pred_plot])

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
