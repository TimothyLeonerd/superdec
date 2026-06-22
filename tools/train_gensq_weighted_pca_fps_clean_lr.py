#!/usr/bin/env python3
import argparse, csv, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
)

PERMS = torch.tensor([
    [0, 1, 2], [0, 2, 1], [1, 0, 2],
    [1, 2, 0], [2, 0, 1], [2, 1, 0],
], dtype=torch.long)


def random_rotations(batch, device):
    q = torch.randn(batch, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)
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


def batched_fps(points, out_n):
    B, M, _ = points.shape
    device = points.device
    selected = torch.empty(B, out_n, dtype=torch.long, device=device)
    min_d2 = torch.full((B, M), float("inf"), device=device)
    farthest = torch.randint(0, M, (B,), device=device)
    batch_arange = torch.arange(B, device=device)

    for j in range(out_n):
        selected[:, j] = farthest
        centroid = points[batch_arange, farthest].unsqueeze(1)
        d2 = ((points - centroid) ** 2).sum(dim=-1)
        min_d2 = torch.minimum(min_d2, d2)
        farthest = min_d2.argmax(dim=1)

    return selected


def gather_points(points, idx):
    B, _, C = points.shape
    return points.gather(1, idx[..., None].expand(B, idx.shape[1], C))


def project_per_sample_dirs(scale, exp, dirs, newton_iters=32):
    """
    Robust radial projection for generalized SQ directions.

        |x/A|^r + |y/B|^s + |z/C|^t = 1

    Uses bracketed bisection instead of Newton, so projected points cannot
    exceed the per-axis scale bounds.
    """
    B = scale.shape[0]

    dtype = scale.dtype
    device = scale.device

    u = dirs.to(device=device, dtype=dtype)
    if u.dim() == 2:
        u = u[None].expand(B, u.shape[0], 3)
    elif u.dim() == 3 and u.shape[0] == 1:
        u = u.expand(B, u.shape[1], 3)

    A = scale[:, None, :].clamp_min(1e-8)
    e = exp[:, None, :].clamp_min(0.05)

    abs_u = u.abs()

    huge = torch.full_like(abs_u, 1e8)
    rho_axis_hi = torch.where(abs_u > 1e-12, A / abs_u.clamp_min(1e-12), huge)
    rho_hi = rho_axis_hi.min(dim=-1, keepdim=True).values.clamp_min(1e-12)
    rho_lo = torch.zeros_like(rho_hi)

    coeff = (abs_u.clamp_min(1e-12) / A).pow(e)

    for _ in range(int(newton_iters)):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        f_mid = (coeff * rho_mid.clamp_min(1e-12).pow(e)).sum(dim=-1, keepdim=True) - 1.0

        too_high = f_mid >= 0.0
        rho_hi = torch.where(too_high, rho_mid, rho_hi)
        rho_lo = torch.where(too_high, rho_lo, rho_mid)

    rho = 0.5 * (rho_lo + rho_hi)
    return u * rho
@torch.no_grad()
def sample_world_batch(batch, points, candidate_n, sampler, args, device):
    scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(batch, 3, device=device)
    exp = args.exp_min + (args.exp_max - args.exp_min) * torch.rand(batch, 3, device=device)
    gt_R = random_rotations(batch, device)

    if sampler == "direct_fib":
        dirs = fibonacci_sphere(points, device)
        local = sample_generalized_surface(scale, exp, dirs)

    elif sampler == "fps":
        base_dirs = fibonacci_sphere(candidate_n, device)
        dir_R = random_rotations(batch, device)

        # Randomize candidate directions relative to the SQ's local axes.
        dirs = base_dirs[None].expand(batch, candidate_n, 3) @ dir_R.transpose(1, 2)

        candidates = project_per_sample_dirs(scale, exp, dirs)
        idx = batched_fps(candidates, points)
        local = gather_points(candidates, idx)

    else:
        raise ValueError(sampler)

    world = local @ gt_R.transpose(1, 2)
    return world, gt_R


@torch.no_grad()
def make_dataset(n, points, candidate_n, sampler, args, device, chunk, seed):
    old_cpu_state = torch.random.get_rng_state()
    old_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    old_py_state = random.getstate()

    set_seed(seed)

    pts_all, R_all = [], []
    for start in range(0, n, chunk):
        b = min(chunk, n - start)
        pts, R = sample_world_batch(b, points, candidate_n, sampler, args, device)
        pts_all.append(pts.cpu())
        R_all.append(R.cpu())
        print(f"generated {sampler}: {start+b}/{n}", flush=True)

    torch.random.set_rng_state(old_cpu_state)
    if old_cuda_state is not None:
        torch.cuda.set_rng_state_all(old_cuda_state)
    random.setstate(old_py_state)

    return torch.cat(pts_all, 0), torch.cat(R_all, 0)


def dropout_without_replacement(points, keep_frac):
    B, N, _ = points.shape
    device = points.device
    k = max(8, int(round(keep_frac * N)))

    if k >= N:
        return points

    out = []
    for b in range(B):
        idx = torch.randperm(N, device=device)[:k]
        out.append(points[b, idx])

    return torch.stack(out, dim=0)


def masked_mean(x, weights=None):
    if weights is None:
        return x.mean(dim=1)
    return (x * weights[..., None]).sum(dim=1)


def hard_pca_frame(points):
    mu = points.mean(dim=1)
    q = points - mu[:, None, :]
    cov = q.transpose(1, 2) @ q / points.shape[1]
    _, evecs = torch.linalg.eigh(cov)
    return evecs, mu


def frame_loss_and_angles(pred_R, gt_R):
    device = pred_R.device
    perms = PERMS.to(device)

    dots = torch.einsum("bik,bil->bkl", pred_R, gt_R).abs().clamp(0.0, 1.0)

    losses = []
    for p in perms:
        d = dots[:, torch.arange(3, device=device), p]
        losses.append((1.0 - d.pow(2)).mean(dim=1))

    loss_stack = torch.stack(losses, dim=1)
    best = loss_stack.argmin(dim=1)
    loss = loss_stack[torch.arange(pred_R.shape[0], device=device), best]

    angles = []
    for b in range(pred_R.shape[0]):
        p = perms[best[b]]
        d = dots[b, torch.arange(3, device=device), p]
        angles.append(torch.rad2deg(torch.acos(d)))

    return loss, torch.stack(angles, dim=0)


class WeightedPCANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.point = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Important: starts as exactly uniform weights, i.e. initially identical to hard PCA.
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def make_features(self, points):
        with torch.no_grad():
            R0, mu0 = hard_pca_frame(points)
            q = (points - mu0[:, None, :]) @ R0
            r = q.norm(dim=-1, keepdim=True)
            return torch.cat([q, q.abs(), q * q, r], dim=-1)

    def forward(self, points):
        feat = self.make_features(points)

        h = self.point(feat)
        g = h.max(dim=1).values
        g = g[:, None, :].expand(-1, points.shape[1], -1)

        logits = self.head(torch.cat([h, g], dim=-1)).squeeze(-1)
        w = F.softplus(logits) + 1e-8
        alpha = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)

        mu = (points * alpha[..., None]).sum(dim=1)
        q = points - mu[:, None, :]
        cov = (q * alpha[..., None]).transpose(1, 2) @ q

        _, evecs = torch.linalg.eigh(cov)

        n_eff = 1.0 / alpha.pow(2).sum(dim=1).clamp_min(1e-8)
        maxw = alpha.max(dim=1).values

        return evecs, n_eff, maxw


@torch.no_grad()
def eval_dataset(model, pts, gt_R, args, device, dropout_keep=None):
    model.eval()

    loader = DataLoader(
        TensorDataset(pts, gt_R),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    hard_angles_all = []
    weighted_angles_all = []
    losses_all = []
    neff_all = []
    maxw_all = []

    for x, R in loader:
        x = x.to(device)
        R = R.to(device)

        if dropout_keep is not None and dropout_keep < 1.0:
            x = dropout_without_replacement(x, dropout_keep)

        hard_R, _ = hard_pca_frame(x)
        _, hard_ang = frame_loss_and_angles(hard_R, R)

        pred_R, neff, maxw = model(x)
        loss, w_ang = frame_loss_and_angles(pred_R, R)

        hard_angles_all.append(hard_ang.cpu().reshape(-1))
        weighted_angles_all.append(w_ang.cpu().reshape(-1))
        losses_all.append(loss.cpu())
        neff_all.append(neff.cpu())
        maxw_all.append(maxw.cpu())

    hard = torch.cat(hard_angles_all)
    weighted = torch.cat(weighted_angles_all)
    losses = torch.cat(losses_all)
    neff = torch.cat(neff_all)
    maxw = torch.cat(maxw_all)

    return {
        "hard_mean": float(hard.mean()),
        "hard_p95": float(torch.quantile(hard, 0.95)),
        "weighted_mean": float(weighted.mean()),
        "weighted_p95": float(torch.quantile(weighted, 0.95)),
        "weighted_loss": float(losses.mean()),
        "neff_mean": float(neff.mean()),
        "neff_p10": float(torch.quantile(neff, 0.10)),
        "maxw_mean": float(maxw.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--candidate-n", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gen-chunk", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=1.0)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--dropout-keep", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print("Generating fixed train: fps2048_to_512", flush=True)
    train_pts, train_R = make_dataset(
        args.train_n,
        args.points,
        args.candidate_n,
        "fps",
        args,
        device,
        chunk=args.gen_chunk,
        seed=args.seed + 1000,
    )

    print("Generating fixed validation datasets", flush=True)

    val_direct512 = make_dataset(
        args.val_n, 512, args.candidate_n, "direct_fib",
        args, device, chunk=args.gen_chunk, seed=args.seed + 2000,
    )

    val_direct307 = make_dataset(
        args.val_n, 307, args.candidate_n, "direct_fib",
        args, device, chunk=args.gen_chunk, seed=args.seed + 3000,
    )

    val_fps512 = make_dataset(
        args.val_n, args.points, args.candidate_n, "fps",
        args, device, chunk=args.gen_chunk, seed=args.seed + 4000,
    )

    train_loader = DataLoader(
        TensorDataset(train_pts, train_R),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = WeightedPCANet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    drop_name = f"dropout{args.dropout_keep:g}"

    cases = [
        ("direct_fib512", val_direct512, None),
        ("fps2048_to_512", val_fps512, None),
        (f"{drop_name}_from_fib512", val_direct512, args.dropout_keep),
        (f"{drop_name}_from_fps512", val_fps512, args.dropout_keep),
        ("direct_fib307", val_direct307, None),
    ]

    fields = ["epoch", "lr", "train_loss"]
    for name, _, _ in cases:
        fields += [
            f"{name}_hard_mean", f"{name}_hard_p95",
            f"{name}_weighted_mean", f"{name}_weighted_p95",
            f"{name}_weighted_loss",
            f"{name}_neff_mean", f"{name}_neff_p10",
            f"{name}_maxw_mean",
        ]

    best_score = float("inf")
    best_state = None
    best_epoch = None

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            if args.lr_schedule == "cosine":
                t = (ep - 1) / max(1, args.epochs - 1)
                cur_lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * t))
            else:
                cur_lr = args.lr
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            model.train()

            total = 0.0
            seen = 0

            for x, R in train_loader:
                x = x.to(device)
                R = R.to(device)

                pred_R, _, _ = model(x)
                loss, _ = frame_loss_and_angles(pred_R, R)
                loss = loss.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += float(loss.detach()) * x.shape[0]
                seen += x.shape[0]

            row = {
                "epoch": ep,
                "lr": cur_lr,
                "train_loss": total / max(1, seen),
            }

            # Fixed eval dropout masks each epoch.
            old_cpu_state = torch.random.get_rng_state()
            old_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            old_py_state = random.getstate()
            set_seed(args.seed + 9000)

            for name, (pts, R), keep in cases:
                m = eval_dataset(model, pts, R, args, device, dropout_keep=keep)
                for k, v in m.items():
                    row[f"{name}_{k}"] = v

            torch.random.set_rng_state(old_cpu_state)
            if old_cuda_state is not None:
                torch.cuda.set_rng_state_all(old_cuda_state)
            random.setstate(old_py_state)

            writer.writerow(row)
            f.flush()

            score = (
                row["fps2048_to_512_weighted_mean"]
                + row[f"{drop_name}_from_fib512_weighted_mean"]
                + row[f"{drop_name}_from_fps512_weighted_mean"]
            ) / 3.0

            if score < best_score:
                best_score = score
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "model": best_state,
                        "args": vars(args),
                        "epoch": ep,
                        "score": score,
                    },
                    out / "best.pt",
                )

            print(
                f"ep={ep:03d} "
                f"lr={cur_lr:.2e} "
                f"train={row['train_loss']:.6f} "
                f"direct512={row['direct_fib512_weighted_mean']:.3f} "
                f"fps512={row['fps2048_to_512_weighted_mean']:.3f} "
                f"dropfib={row[f'{drop_name}_from_fib512_weighted_mean']:.3f} "
                f"dropfps={row[f'{drop_name}_from_fps512_weighted_mean']:.3f} "
                f"direct307={row['direct_fib307_weighted_mean']:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    lines = [f"best_epoch={best_epoch} score={best_score:.4f}"]

    set_seed(args.seed + 9000)

    for name, (pts, R), keep in cases:
        m = eval_dataset(model, pts, R, args, device, dropout_keep=keep)
        lines.append(
            f"{name:24s} "
            f"hard={m['hard_mean']:6.2f}/{m['hard_p95']:6.2f} "
            f"weighted={m['weighted_mean']:6.2f}/{m['weighted_p95']:6.2f} "
            f"loss={m['weighted_loss']:.6f} "
            f"neff={m['neff_mean']:6.1f}/{m['neff_p10']:6.1f} "
            f"maxw={m['maxw_mean']:.4f}"
        )

    (out / "summary_short.txt").write_text("
".join(lines) + "
")
    print("
".join(lines), flush=True)


if __name__ == "__main__":
    main()
