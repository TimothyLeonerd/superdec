#!/usr/bin/env python3
import argparse, csv, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import tools.train_gensq_unrolled_coupled_final_cd_stepidx as base


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


def fps_downsample(points, m):
    B, N, _ = points.shape
    device = points.device
    idx = torch.empty(B, m, dtype=torch.long, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch = torch.arange(B, device=device)
    dist = torch.full((B, N), float("inf"), device=device)

    for i in range(m):
        idx[:, i] = farthest
        c = points[batch, farthest][:, None, :]
        d = ((points - c) ** 2).sum(dim=-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(dim=1)

    return points[batch[:, None], idx]


def chamfer_sq(a, b):
    d = torch.cdist(a, b).pow(2)
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)


def perm_abs_cd_loss(X_local_pred, X_local_gt):
    """
    Sign-invariant via abs().
    Axis-permutation-invariant via min over 6 permutations.
    """
    P = PERMS.to(X_local_pred.device)

    pred_abs = X_local_pred.abs()
    gt_abs = X_local_gt.abs()

    losses = []
    for p in P:
        pred_p = pred_abs[..., p]
        losses.append(chamfer_sq(pred_p, gt_abs))

    return torch.stack(losses, dim=1).min(dim=1).values


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
        angles.append(torch.rad2deg(torch.acos(d.clamp(0.0, 1.0))))

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

        # Uniform weights at init => hard PCA.
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

        neff = 1.0 / alpha.pow(2).sum(dim=1).clamp_min(1e-8)
        maxw = alpha.max(dim=1).values

        return evecs, neff, maxw


@torch.no_grad()
def make_batch(args, B, device):
    scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    eps = base.sample_log_uniform(args.exp_min, args.exp_max, (B, 3), device)

    dirs = base.fibonacci_sphere(args.candidate_n, device)
    dirs = dirs[None, :, :].expand(B, args.candidate_n, 3)

    X_local_dense = base.sample_gensq_batched_dirs(scale, eps, dirs)
    X_local = fps_downsample(X_local_dense, args.points)

    R_gt = random_rotations(B, device)
    X_world = X_local @ R_gt.transpose(1, 2)

    return X_world, X_local, R_gt


@torch.no_grad()
def make_dataset(args, n, device, seed):
    old_cpu = torch.random.get_rng_state()
    old_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    old_py = random.getstate()

    base.set_seed(seed)

    Xw_all, Xl_all, R_all = [], [], []
    done = 0
    while done < n:
        b = min(args.gen_chunk, n - done)
        Xw, Xl, R = make_batch(args, b, device)
        Xw_all.append(Xw.cpu())
        Xl_all.append(Xl.cpu())
        R_all.append(R.cpu())
        done += b
        print(f"generated {done}/{n}", flush=True)

    torch.random.set_rng_state(old_cpu)
    if old_cuda is not None:
        torch.cuda.set_rng_state_all(old_cuda)
    random.setstate(old_py)

    return torch.cat(Xw_all), torch.cat(Xl_all), torch.cat(R_all)


def angle_stats(x):
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean()),
        "med": float(x.median()),
        "p90": float(torch.quantile(x, 0.90)),
        "p95": float(torch.quantile(x, 0.95)),
        "max": float(x.max()),
    }


@torch.no_grad()
def evaluate(model, Xw, Xl, Rgt, args, device):
    model.eval()

    loader = DataLoader(
        TensorDataset(Xw, Xl, Rgt),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    hard_ang_all = []
    wpca_ang_all = []
    cd_all = []
    neff_all = []
    maxw_all = []

    for xw, xl, r in loader:
        xw = xw.to(device)
        xl = xl.to(device)
        r = r.to(device)

        hard_R, _ = hard_pca_frame(xw)
        _, hard_ang = frame_loss_and_angles(hard_R, r)

        pred_R, neff, maxw = model(xw)
        _, wpca_ang = frame_loss_and_angles(pred_R, r)

        pred_local = xw @ pred_R
        cd = perm_abs_cd_loss(pred_local, xl)

        hard_ang_all.append(hard_ang.reshape(-1).cpu())
        wpca_ang_all.append(wpca_ang.reshape(-1).cpu())
        cd_all.append(cd.cpu())
        neff_all.append(neff.cpu())
        maxw_all.append(maxw.cpu())

    hard_ang = torch.cat(hard_ang_all)
    wpca_ang = torch.cat(wpca_ang_all)
    cd = torch.cat(cd_all)
    neff = torch.cat(neff_all)
    maxw = torch.cat(maxw_all)

    h = angle_stats(hard_ang)
    w = angle_stats(wpca_ang)

    return {
        "hard_mean": h["mean"],
        "hard_p95": h["p95"],
        "wpca_mean": w["mean"],
        "wpca_med": w["med"],
        "wpca_p90": w["p90"],
        "wpca_p95": w["p95"],
        "wpca_max": w["max"],
        "cd_mean": float(cd.mean()),
        "cd_p95": float(torch.quantile(cd, 0.95)),
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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gen-chunk", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=0.5)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    base.set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print("Generating train FPS2048→512", flush=True)
    train_Xw, train_Xl, train_R = make_dataset(args, args.train_n, device, args.seed + 1000)

    print("Generating val FPS2048→512", flush=True)
    val_Xw, val_Xl, val_R = make_dataset(args, args.val_n, device, args.seed + 2000)

    loader = DataLoader(
        TensorDataset(train_Xw, train_Xl, train_R),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = WeightedPCANet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fields = [
        "epoch", "train_cd",
        "hard_mean", "hard_p95",
        "wpca_mean", "wpca_med", "wpca_p90", "wpca_p95", "wpca_max",
        "cd_mean", "cd_p95",
        "neff_mean", "neff_p10", "maxw_mean",
    ]

    best = (float("inf"), None, None)

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            seen = 0

            for xw, xl, _ in loader:
                xw = xw.to(device)
                xl = xl.to(device)

                pred_R, _, _ = model(xw)
                pred_local = xw @ pred_R
                loss = perm_abs_cd_loss(pred_local, xl).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += float(loss.detach()) * xw.shape[0]
                seen += xw.shape[0]

            ev = evaluate(model, val_Xw, val_Xl, val_R, args, device)
            row = {"epoch": ep, "train_cd": total / max(1, seen), **ev}
            writer.writerow(row)
            f.flush()

            if ev["wpca_mean"] < best[0]:
                best = (ev["wpca_mean"], ep, ev)
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "ev": ev}, out / "best.pt")

            print(
                f"ep={ep:03d} train_cd={row['train_cd']:.8g} "
                f"hard={ev['hard_mean']:.3f}/{ev['hard_p95']:.3f} "
                f"wpca={ev['wpca_mean']:.3f}/{ev['wpca_p95']:.3f} "
                f"cd={ev['cd_mean']:.8g} "
                f"neff={ev['neff_mean']:.1f} maxw={ev['maxw_mean']:.4f}",
                flush=True,
            )

    lines = [
        "experiment=wpca_only_fps2048_to_512_cd_loss",
        "loss=sign_invariant_abs_cd_min_over_6_axis_permutations",
        f"points={args.points}",
        f"candidate_n={args.candidate_n}",
        f"exp_min={args.exp_min}",
        f"best_angle_epoch={best[1]} best_wpca_mean_angle={best[0]:.6f}",
        "",
    ]

    ev = best[2]
    for k, v in ev.items():
        lines.append(f"{k}={v}")

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
