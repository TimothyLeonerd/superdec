#!/usr/bin/env python3
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    PointNetScaleExp,
    chamfer,
)
from tools.train_gensq_weighted_pca_fps_clean_lr import (
    WeightedPCANet,
    random_rotations,
    batched_fps,
    gather_points,
    hard_pca_frame,
    dropout_without_replacement,
)

PERMS = torch.tensor(list(itertools.permutations([0, 1, 2])), dtype=torch.long)


def parse3(s):
    vals = [float(x) for x in s.split(",")]
    assert len(vals) == 3
    return vals


def safe_sample_gensq(scale, exp, dirs, iters=80):
    """
    Bisection radial sampler for:
        |x/a|^r + |y/b|^s + |z/c|^t = 1

    scale: [B,3]
    exp:   [B,3]
    dirs:  [N,3] or [B,N,3], approximately unit directions
    """
    if dirs.dim() == 2:
        dirs = dirs[None].expand(scale.shape[0], dirs.shape[0], 3)

    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    B, N, _ = dirs.shape
    max_s = scale.max(dim=1).values[:, None]
    lo = torch.zeros(B, N, device=scale.device)
    hi = 4.0 * max_s.expand(B, N)

    a = scale[:, None, :]
    r = exp[:, None, :]

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pts = mid[..., None] * dirs
        val = ((pts.abs() / a.clamp_min(1e-12)) ** r).sum(dim=-1)
        hi = torch.where(val >= 1.0, mid, hi)
        lo = torch.where(val < 1.0, mid, lo)

    rho = 0.5 * (lo + hi)
    return rho[..., None] * dirs


def best_perm(frame, gt_R):
    device = frame.device
    perms = PERMS.to(device)
    dots = torch.einsum("bik,bil->bkl", frame, gt_R).abs().clamp(0.0, 1.0)

    scores = []
    for p in perms:
        d = dots[:, torch.arange(3, device=device), p]
        scores.append(d.pow(2).mean(dim=1))
    scores = torch.stack(scores, dim=1)

    best = scores.argmax(dim=1)
    return perms[best], dots


def make_points(scale, exp, args, device):
    B = 1

    if args.sampler == "direct":
        dirs = fibonacci_sphere(args.points, device)
        local = safe_sample_gensq(scale, exp, dirs)

    elif args.sampler == "fps":
        base_dirs = fibonacci_sphere(args.candidate_n, device)
        dir_R = random_rotations(B, device)
        dirs = base_dirs[None].expand(B, args.candidate_n, 3) @ dir_R.transpose(1, 2)
        candidates = safe_sample_gensq(scale, exp, dirs)
        idx = batched_fps(candidates, args.points)
        local = gather_points(candidates, idx)

    else:
        raise ValueError(args.sampler)

    gt_R = random_rotations(B, device)
    world = local @ gt_R.transpose(1, 2)

    if args.dropout_keep < 1.0:
        world = dropout_without_replacement(world, args.dropout_keep)

    return world, gt_R


def subsample_np(x, n, seed=0):
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]


def set_equal(ax, arrays, margin=1.10):
    pts = np.concatenate(arrays, axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    c = (lo + hi) / 2.0
    r = float((hi - lo).max() / 2.0) * margin
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def style_ax(ax, title):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exp-ckpt", required=True)
    ap.add_argument("--wpca-ckpt", required=True)

    ap.add_argument("--scale", required=True)
    ap.add_argument("--exp", required=True)

    ap.add_argument("--sampler", choices=["fps", "direct"], default="fps")
    ap.add_argument("--frame", choices=["weighted_pca", "hard_pca"], default="weighted_pca")
    ap.add_argument("--layout", choices=["overlay", "side_by_side", "all"], default="overlay")

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
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    scale = torch.tensor([parse3(args.scale)], dtype=torch.float32, device=device)
    exp = torch.tensor([parse3(args.exp)], dtype=torch.float32, device=device)

    model = PointNetScaleExp(args.scale_min, args.scale_max, args.exp_min, args.exp_max).to(device)
    ckpt = torch.load(args.exp_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wpca = WeightedPCANet().to(device)
    wckpt = torch.load(args.wpca_ckpt, map_location=device)
    wpca.load_state_dict(wckpt["model"])
    wpca.eval()

    with torch.no_grad():
        world, gt_R = make_points(scale, exp, args, device)

        if args.frame == "weighted_pca":
            frame, neff, maxw = wpca(world)
            neff_v = float(neff[0])
            maxw_v = float(maxw[0])
        else:
            frame, _ = hard_pca_frame(world)
            neff_v = float("nan")
            maxw_v = float("nan")

        canon = world @ frame

        perm, dots = best_perm(frame, gt_R)
        target_scale = scale.gather(1, perm)
        target_exp = exp.gather(1, perm)

        matched = dots.gather(2, perm[:, None, :]).diagonal(dim1=1, dim2=2)
        axis_angles = torch.rad2deg(torch.acos(matched.clamp(0.0, 1.0)))[0]

        pred_scale, pred_exp = model(canon)

        scale_abs = (pred_scale - target_scale).abs()[0]
        exp_abs = (pred_exp - target_exp).abs()[0]
        scale_l1 = float(scale_abs.mean())
        exp_l1 = float(exp_abs.mean())

        dirs = fibonacci_sphere(args.surface_n, device)
        gt_surf = safe_sample_gensq(target_scale, target_exp, dirs)
        pred_surf = safe_sample_gensq(pred_scale, pred_exp, dirs)

        cd_gt_pred = float(chamfer(gt_surf, pred_surf)[0])
        cd_input_gt = float(chamfer(canon, gt_surf)[0])
        cd_input_pred = float(chamfer(canon, pred_surf)[0])

    canon_np = canon[0].cpu().numpy()
    gt_np = gt_surf[0].cpu().numpy()
    pred_np = pred_surf[0].cpu().numpy()

    canon_plot = subsample_np(canon_np, args.plot_n, args.seed + 1)
    gt_plot = subsample_np(gt_np, args.plot_n, args.seed + 2)
    pred_plot = subsample_np(pred_np, args.plot_n, args.seed + 3)

    metrics = f"""sampler={args.sampler}
frame={args.frame}
layout={args.layout}
seed={args.seed}
dropout_keep={args.dropout_keep}

original_gt_scale={scale[0].cpu().numpy().tolist()}
original_gt_exp={exp[0].cpu().numpy().tolist()}

perm_canonical_to_gt={perm[0].cpu().numpy().tolist()}
matched_gt_scale={target_scale[0].cpu().numpy().tolist()}
matched_gt_exp={target_exp[0].cpu().numpy().tolist()}

pred_scale={pred_scale[0].cpu().numpy().tolist()}
pred_exp={pred_exp[0].cpu().numpy().tolist()}

scale_abs_error={scale_abs.cpu().numpy().tolist()}
exp_abs_error={exp_abs.cpu().numpy().tolist()}

scale_l1={scale_l1:.8f}
exp_l1={exp_l1:.8f}

cd_gt_pred_surface={cd_gt_pred:.8f}
cd_input_gt_surface={cd_input_gt:.8f}
cd_input_pred_surface={cd_input_pred:.8f}

axis_angles_deg={axis_angles.cpu().numpy().tolist()}
axis_angle_mean_deg={float(axis_angles.mean()):.6f}

neff={neff_v}
maxw={maxw_v}
"""
    (out / "metrics.txt").write_text(metrics)
    print(metrics)

    title = (
        f"scale_l1={scale_l1:.4f}, exp_l1={exp_l1:.4f}, "
        f"CD(gt,pred)={cd_gt_pred:.5f}, CD(input,pred)={cd_input_pred:.5f}"
    )

    if args.layout == "overlay":
        fig = plt.figure(figsize=(7, 6))
        fig.suptitle(title, fontsize=10)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], s=3, alpha=0.40, label="GT")
        ax.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], s=3, alpha=0.40, label="pred")
        style_ax(ax, "GT vs predicted SQ")
        ax.legend()
        set_equal(ax, [gt_plot, pred_plot])

    elif args.layout == "side_by_side":
        fig = plt.figure(figsize=(11, 5))
        fig.suptitle(title, fontsize=10)

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], s=3, alpha=0.45)
        style_ax(ax1, "GT SQ target")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], s=3, alpha=0.45)
        style_ax(ax2, "predicted SQ")

        for ax in [ax1, ax2]:
            set_equal(ax, [gt_plot, pred_plot])

    else:
        fig = plt.figure(figsize=(16, 5))
        fig.suptitle(title, fontsize=10)

        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax1.scatter(canon_plot[:, 0], canon_plot[:, 1], canon_plot[:, 2], s=3, alpha=0.45)
        style_ax(ax1, "canonical input")

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.scatter(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], s=3, alpha=0.45)
        style_ax(ax2, "GT SQ target")

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], s=3, alpha=0.45)
        style_ax(ax3, "predicted SQ")

        for ax in [ax1, ax2, ax3]:
            set_equal(ax, [canon_plot, gt_plot, pred_plot])

    plt.tight_layout()
    fig.savefig(out / "comparison.png", dpi=200)
    plt.close(fig)

    np.savez(
        out / "points_and_params.npz",
        canonical_input=canon_np,
        gt_surface=gt_np,
        pred_surface=pred_np,
        original_gt_scale=scale[0].cpu().numpy(),
        original_gt_exp=exp[0].cpu().numpy(),
        matched_gt_scale=target_scale[0].cpu().numpy(),
        matched_gt_exp=target_exp[0].cpu().numpy(),
        pred_scale=pred_scale[0].cpu().numpy(),
        pred_exp=pred_exp[0].cpu().numpy(),
    )

    print(f"Wrote {out / 'comparison.png'}")
    print(f"Wrote {out / 'metrics.txt'}")
    print(f"Wrote {out / 'points_and_params.npz'}")


if __name__ == "__main__":
    main()
