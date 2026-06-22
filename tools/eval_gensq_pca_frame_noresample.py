#!/usr/bin/env python3
import argparse, csv, itertools, math, random
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
    PointNetScaleExp,
    chamfer,
)

PERMS = list(itertools.permutations(range(3)))


def random_rotations(batch, device):
    q = torch.randn(batch, 4, device=device)
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)

    R = torch.empty(batch, 3, 3, device=device)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def pca_frame(points):
    # points: [B,N,3], centered object assumed, but covariance is computed mean-centered
    x = points - points.mean(dim=1, keepdim=True)
    cov = x.transpose(1, 2) @ x / max(points.shape[1] - 1, 1)

    evals, evecs = torch.linalg.eigh(cov)  # ascending, evecs columns
    order = evals.argsort(dim=1, descending=True)
    evecs = evecs.gather(2, order[:, None, :].expand(-1, 3, -1))

    # make proper right-handed frame
    det = torch.det(evecs)
    evecs[:, :, 2] *= torch.where(det < 0, -1.0, 1.0)[:, None]
    return evecs


def frame_perm_metrics(pred_R, gt_R):
    losses, angles = [], []
    for p in PERMS:
        gt = gt_R[:, :, list(p)]
        dot = (pred_R * gt).sum(dim=1).abs().clamp(0, 1)
        loss = (1.0 - dot.pow(2)).mean(dim=1)
        angle = torch.acos(dot) * (180.0 / math.pi)
        losses.append(loss)
        angles.append(angle)

    losses = torch.stack(losses, dim=1)
    angles = torch.stack(angles, dim=1)

    best_idx = losses.argmin(dim=1)
    best_angles = angles.gather(1, best_idx[:, None, None].expand(-1, 1, 3)).squeeze(1)
    best_loss = losses.gather(1, best_idx[:, None]).squeeze(1)
    return best_loss, best_idx, best_angles


def gather_perm_params(gt_scale, gt_exp, best_idx):
    perms = torch.tensor(PERMS, device=gt_scale.device, dtype=torch.long)
    idx = perms[best_idx]
    return gt_scale.gather(1, idx), gt_exp.gather(1, idx)


def resample_observed(world, mode, keep_frac=0.5):
    # Returns variable-sized observed point sets per mode/keep_frac.
    # No resampling with replacement; no duplicates are introduced.
    B, N, _ = world.shape
    device = world.device
    out = []

    for b in range(B):
        pts = world[b]

        if mode == "full":
            out.append(pts)
            continue

        k = max(8, int(round(keep_frac * N)))

        if mode == "dropout":
            idx = torch.randperm(N, device=device)[:k]

        elif mode == "viewcrop":
            # Keep the top-k points in a random viewing direction.
            v = torch.randn(3, device=device)
            v = v / v.norm().clamp_min(1e-8)
            score = pts @ v
            idx = torch.topk(score, k=k).indices

        else:
            raise ValueError(mode)

        out.append(pts[idx])

    return torch.stack(out, dim=0)


@torch.no_grad()
def make_dataset(n, points, scale_min, scale_max, exp_min, exp_max, device):
    dirs = fibonacci_sphere(points, device)
    scale = scale_min + (scale_max - scale_min) * torch.rand(n, 3, device=device)
    exp = exp_min + (exp_max - exp_min) * torch.rand(n, 3, device=device)
    gt_R = random_rotations(n, device)

    local = sample_generalized_surface(scale, exp, dirs)
    world = local @ gt_R.transpose(1, 2)

    return world, gt_R, scale, exp, dirs


def load_shape_net(path, device):
    ckpt = torch.load(path, map_location=device)
    a = ckpt.get("args", {})
    model = PointNetScaleExp(
        a.get("scale_min", 0.08),
        a.get("scale_max", 0.35),
        a.get("exp_min", 1.0),
        a.get("exp_max", 8.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def eval_mode(mode, world_full, gt_R, gt_scale, gt_exp, dirs, shape_net, args):
    obs = resample_observed(world_full, mode, args.keep_frac)

    pca_R = pca_frame(obs)
    frame_loss, best_idx, angles = frame_perm_metrics(pca_R, gt_R)
    gt_scale_p, gt_exp_p = gather_perm_params(gt_scale, gt_exp, best_idx)

    # PCA-canonicalized pipe
    obs_pca_local = obs @ pca_R
    pred_scale, pred_exp = shape_net(obs_pca_local)

    pred_local_surf = sample_generalized_surface(pred_scale, pred_exp, dirs)
    pred_world = pred_local_surf @ pca_R.transpose(1, 2)

    pca_scale_l1 = (pred_scale - gt_scale_p).abs().mean(dim=1)
    pca_exp_l1 = (pred_exp - gt_exp_p).abs().mean(dim=1)
    pca_cd_full = chamfer(pred_world, world_full)
    pca_cd_obs = chamfer(pred_world, obs)

    # GT-canonicalized pipe: upper bound for same observed input
    obs_gt_local = obs @ gt_R
    gt_pred_scale, gt_pred_exp = shape_net(obs_gt_local)

    gt_pred_local_surf = sample_generalized_surface(gt_pred_scale, gt_pred_exp, dirs)
    gt_pred_world = gt_pred_local_surf @ gt_R.transpose(1, 2)

    gt_scale_l1 = (gt_pred_scale - gt_scale).abs().mean(dim=1)
    gt_exp_l1 = (gt_pred_exp - gt_exp).abs().mean(dim=1)
    gt_cd_full = chamfer(gt_pred_world, world_full)
    gt_cd_obs = chamfer(gt_pred_world, obs)

    def mean_p95(x):
        return float(x.mean()), float(torch.quantile(x, 0.95))

    angle_mean, angle_p95 = mean_p95(angles.reshape(-1))
    pca_scale_mean, pca_scale_p95 = mean_p95(pca_scale_l1)
    pca_exp_mean, pca_exp_p95 = mean_p95(pca_exp_l1)
    pca_cd_full_mean, pca_cd_full_p95 = mean_p95(pca_cd_full)
    pca_cd_obs_mean, pca_cd_obs_p95 = mean_p95(pca_cd_obs)

    gt_scale_mean, gt_scale_p95 = mean_p95(gt_scale_l1)
    gt_exp_mean, gt_exp_p95 = mean_p95(gt_exp_l1)
    gt_cd_full_mean, gt_cd_full_p95 = mean_p95(gt_cd_full)
    gt_cd_obs_mean, gt_cd_obs_p95 = mean_p95(gt_cd_obs)

    return {
        "mode": mode,
        "pca_axis_angle_mean": angle_mean,
        "pca_axis_angle_p95": angle_p95,
        "pca_frame_loss_mean": float(frame_loss.mean()),
        "pca_pipe_scale_l1_mean": pca_scale_mean,
        "pca_pipe_scale_l1_p95": pca_scale_p95,
        "pca_pipe_exp_l1_mean": pca_exp_mean,
        "pca_pipe_exp_l1_p95": pca_exp_p95,
        "pca_pipe_cd_full_mean": pca_cd_full_mean,
        "pca_pipe_cd_full_p95": pca_cd_full_p95,
        "pca_pipe_cd_obs_mean": pca_cd_obs_mean,
        "pca_pipe_cd_obs_p95": pca_cd_obs_p95,
        "gtcanon_pipe_scale_l1_mean": gt_scale_mean,
        "gtcanon_pipe_scale_l1_p95": gt_scale_p95,
        "gtcanon_pipe_exp_l1_mean": gt_exp_mean,
        "gtcanon_pipe_exp_l1_p95": gt_exp_p95,
        "gtcanon_pipe_cd_full_mean": gt_cd_full_mean,
        "gtcanon_pipe_cd_full_p95": gt_cd_full_p95,
        "gtcanon_pipe_cd_obs_mean": gt_cd_obs_mean,
        "gtcanon_pipe_cd_obs_p95": gt_cd_obs_p95,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shape-ckpt", required=True)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=1.0)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--modes", nargs="+", default=["full", "dropout", "viewcrop"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    shape_net = load_shape_net(args.shape_ckpt, device)

    world, gt_R, gt_scale, gt_exp, dirs = make_dataset(
        args.val_n, args.points,
        args.scale_min, args.scale_max,
        args.exp_min, args.exp_max,
        device,
    )

    rows = []
    for mode in args.modes:
        print("Evaluating", mode, flush=True)
        row = eval_mode(mode, world, gt_R, gt_scale, gt_exp, dirs, shape_net, args)
        rows.append(row)
        for k, v in row.items():
            print(f"{k}: {v}")
        print()

    fields = list(rows[0].keys())
    with open(out / "pca_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("OUT", out)
    print(open(out / "pca_summary.csv").read())


if __name__ == "__main__":
    main()
