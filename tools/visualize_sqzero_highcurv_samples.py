import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from superdec.data.sqzero_lmdb import SQZeroLMDB
from superdec.loss.loss import sampling_from_parametric_space_to_equivalent_points
from superdec.loss.supervised_hungarian_loss import SupervisedHungarianLoss


def make_cfg(args):
    return OmegaConf.create({
        "trainer": {
            "augmentations": False,
        },
        "sqzero_lmdb": {
            "path": args.data_root,
            "train_split": args.split,
            "val_split": args.split,
            "n_points": args.n_points,
            "normalize": True,
            "normal_mode": "radial",
            "load_sidecars": True,
            "kmax": args.kmax,
        },
    })


def make_loss_cfg(args, sampler_type):
    return OmegaConf.create({
        "w_sup_exist": 2.0,
        "w_sup_assign": 1.0,
        "w_sup_surface": 10.0,
        "w_sup_shape_param": 0.0,
        "w_sup_shape_oracle_surface": 0.0,
        "w_sup_highcurv_pointwise_surface": 0.0,
        "w_sup_highcurv_chamfer_surface": 0.0,
        "w_sup_normal": 0.0,
        "w_sup_normal_dir": 0.0,

        "surface_target": "gt_surface",
        "surface_sampler_type": sampler_type,
        "surface_n_samples": args.uniform_samples,
        "surface_D_eta": args.surface_D_eta,
        "surface_D_omega": args.surface_D_omega,

        "surface_naive_n_theta": 12,
        "surface_naive_n_phi": 12,

        "highcurv_n_samples": args.highcurv_samples,
        "highcurv_eta_bins": args.eta_bins,
        "highcurv_omega_bins": args.omega_bins,
        "highcurv_alpha": args.alpha,
        "highcurv_uniform_mix": args.uniform_mix,
        "highcurv_jitter": args.jitter,
        "highcurv_probe_delta": args.probe_delta,

        "match_w_assign": 1.0,
        "match_w_exist": 0.0,
        "use_staged_surface": False,
        "debug_match_every": 0,
    })


def set_equal_axes(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def sample_equal_training_path(item, args):
    loss = SupervisedHungarianLoss(make_loss_cfg(args, "equal_distance"))

    K = int(item["K"])
    gt_scale = item["gt_scale"][:K].unsqueeze(0).float()
    gt_shape = item["gt_shape"][:K].unsqueeze(0).float()
    gt_rotate = item["gt_rotate"][:K].unsqueeze(0).float()
    gt_trans = item["gt_trans"][:K].unsqueeze(0).float()

    with torch.no_grad():
        gt_local, _ = sampling_from_parametric_space_to_equivalent_points(
            gt_scale,
            gt_shape,
            loss.surface_sampler,
        )
        gt_world = loss._local_to_world(gt_local, gt_rotate, gt_trans)[0]

    pts = gt_world.reshape(-1, 3).cpu().numpy()
    labels = np.repeat(np.arange(K), gt_world.shape[1])
    return pts, labels


def sample_highcurv_training_path(item, args):
    loss = SupervisedHungarianLoss(make_loss_cfg(args, "equal_distance"))

    if not hasattr(loss, "_sample_gt_highcurv_eta_omega"):
        raise RuntimeError(
            "Current supervised_hungarian_loss.py does not have "
            "_sample_gt_highcurv_eta_omega. Apply the high-curv patch first."
        )

    K = int(item["K"])
    gt_scale = item["gt_scale"][:K].unsqueeze(0).float()
    gt_shape = item["gt_shape"][:K].unsqueeze(0).float()
    gt_rotate = item["gt_rotate"][:K].unsqueeze(0).float()
    gt_trans = item["gt_trans"][:K].unsqueeze(0).float()

    with torch.no_grad():
        etas, omegas = loss._sample_gt_highcurv_eta_omega(gt_scale, gt_shape)
        gt_local, _ = loss._sq_local_from_eta_omega(
            gt_scale,
            gt_shape,
            etas,
            omegas,
        )
        gt_world = loss._local_to_world(gt_local, gt_rotate, gt_trans)[0]

    pts = gt_world.reshape(-1, 3).cpu().numpy()
    labels = np.repeat(np.arange(K), gt_world.shape[1])
    return pts, labels


def plot_panel(ax, input_points, sample_points, sample_labels, title):
    ax.scatter(
        input_points[:, 0],
        input_points[:, 1],
        input_points[:, 2],
        s=0.35,
        alpha=0.06,
        c="gray",
    )

    for k in sorted(np.unique(sample_labels)):
        m = sample_labels == k
        ax.scatter(
            sample_points[m, 0],
            sample_points[m, 1],
            sample_points[m, 2],
            s=10,
            alpha=0.95,
            label=f"SQ {k}",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--count", type=int, default=8)

    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--kmax", type=int, default=4)

    ap.add_argument("--uniform-samples", type=int, default=128)
    ap.add_argument("--surface-D-eta", type=float, default=0.05)
    ap.add_argument("--surface-D-omega", type=float, default=0.05)

    ap.add_argument("--highcurv-samples", type=int, default=128)
    ap.add_argument("--eta-bins", type=int, default=32)
    ap.add_argument("--omega-bins", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--uniform-mix", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=1.0)
    ap.add_argument("--probe-delta", type=float, default=0.05)

    ap.add_argument("--input-subsample", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SQZeroLMDB("val", make_cfg(args))

    for local_i, idx in enumerate(range(args.index, args.index + args.count)):
        item = ds[idx]
        key = ds.keys[idx]
        safe_key = key.replace(":", "_")

        input_points = item["points"].numpy()
        rng = np.random.default_rng(args.seed + idx)
        if input_points.shape[0] > args.input_subsample:
            sel = rng.choice(input_points.shape[0], args.input_subsample, replace=False)
            input_plot = input_points[sel]
        else:
            input_plot = input_points

        eq_pts, eq_labels = sample_equal_training_path(item, args)
        hc_pts, hc_labels = sample_highcurv_training_path(item, args)

        all_pts = np.concatenate([input_plot, eq_pts, hc_pts], axis=0)

        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(121, projection="3d")
        plot_panel(
            ax1,
            input_plot,
            eq_pts,
            eq_labels,
            f"{key}\nTraining equal-distance sampler, {args.uniform_samples}/SQ",
        )
        set_equal_axes(ax1, all_pts)

        ax2 = fig.add_subplot(122, projection="3d")
        plot_panel(
            ax2,
            input_plot,
            hc_pts,
            hc_labels,
            f"{key}\nGT high-curvature sampler, {args.highcurv_samples}/SQ",
        )
        set_equal_axes(ax2, all_pts)

        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        fig.tight_layout()
        out_png = out_dir / f"{local_i:03d}_{safe_key}_training_equal_vs_highcurv.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)

        np.savez_compressed(
            out_dir / f"{local_i:03d}_{safe_key}_training_equal_vs_highcurv.npz",
            key=key,
            input_points=input_plot,
            equal_points=eq_pts,
            equal_labels=eq_labels,
            highcurv_points=hc_pts,
            highcurv_labels=hc_labels,
        )

        print(f"Wrote {out_png}")

    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
