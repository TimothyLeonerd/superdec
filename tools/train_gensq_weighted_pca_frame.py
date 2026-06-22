#!/usr/bin/env python3
import argparse, csv, itertools, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
)

PERMS = list(itertools.permutations(range(3)))


def random_rotations(batch, device):
    q = F.normalize(torch.randn(batch, 4, device=device), dim=-1)
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


def sample_batch(batch, points, args, device):
    dirs = fibonacci_sphere(points, device)
    scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(batch, 3, device=device)
    exp = args.exp_min + (args.exp_max - args.exp_min) * torch.rand(batch, 3, device=device)
    gt_R = random_rotations(batch, device)
    local = sample_generalized_surface(scale, exp, dirs)
    world = local @ gt_R.transpose(1, 2)
    return world, gt_R


def make_mask(points, mode, keep_frac, device, pts=None):
    B, N, _ = points.shape
    mask = torch.ones(B, N, dtype=torch.bool, device=device)

    if mode == "full":
        return mask

    k = max(8, int(round(keep_frac * N)))
    mask[:] = False

    for b in range(B):
        if mode == "dropout":
            idx = torch.randperm(N, device=device)[:k]
        elif mode == "viewcrop":
            v = torch.randn(3, device=device)
            v = v / v.norm().clamp_min(1e-8)
            score = points[b] @ v
            idx = torch.topk(score, k=k).indices
        else:
            raise ValueError(mode)
        mask[b, idx] = True

    return mask


def masked_mean(points, mask):
    w = mask.float()
    return (points * w[..., None]).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1.0)[..., None]


def pca_frame_masked(points, mask):
    mu = masked_mean(points, mask)
    x = points - mu
    w = mask.float()
    xw = x * w[..., None]
    denom = w.sum(dim=1).clamp_min(2.0) - 1.0
    cov = xw.transpose(1, 2) @ x / denom[:, None, None]

    evals, evecs = torch.linalg.eigh(cov)
    order = evals.argsort(dim=1, descending=True)
    evecs = evecs.gather(2, order[:, None, :].expand(-1, 3, -1))
    det = torch.det(evecs)
    evecs[:, :, 2] *= torch.where(det < 0, -1.0, 1.0)[:, None]
    return evecs


def frame_loss_and_angles(pred_R, gt_R):
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
    best = losses.argmin(dim=1)
    best_loss = losses.gather(1, best[:, None]).squeeze(1)
    best_angles = angles.gather(1, best[:, None, None].expand(-1, 1, 3)).squeeze(1)
    return best_loss, best_angles


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

    def make_features(self, points, mask):
        with torch.no_grad():
            R0 = pca_frame_masked(points, mask)
            mu = masked_mean(points, mask)
            q = (points - mu) @ R0
            r = q.norm(dim=-1, keepdim=True)
            feat = torch.cat([q, q.abs(), q * q, r], dim=-1)
        return feat

    def forward(self, points, mask):
        feat = self.make_features(points, mask)
        h = self.point(feat)
        h_masked = h.masked_fill(~mask[..., None], -1e9)
        g = h_masked.max(dim=1).values
        g = g[:, None, :].expand(-1, points.shape[1], -1)
        z = torch.cat([h, g], dim=-1)

        logits = self.head(z).squeeze(-1)
        weights = F.softplus(logits) + 1e-6
        weights = weights * mask.float()
        alpha = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        mu = (points * alpha[..., None]).sum(dim=1, keepdim=True)
        x = points - mu
        cov = (x * alpha[..., None]).transpose(1, 2) @ x

        evals, evecs = torch.linalg.eigh(cov)
        order = evals.argsort(dim=1, descending=True)
        evecs = evecs.gather(2, order[:, None, :].expand(-1, 3, -1))
        det = torch.det(evecs)
        evecs[:, :, 2] *= torch.where(det < 0, -1.0, 1.0)[:, None]

        n_eff = 1.0 / alpha.pow(2).sum(dim=1).clamp_min(1e-8)
        max_w = alpha.max(dim=1).values
        return evecs, n_eff, max_w


@torch.no_grad()
def eval_case(model, name, points, mode, keep_frac, args, device):
    world, gt_R = sample_batch(args.val_n, points, args, device)
    mask = make_mask(world, mode, keep_frac, device)

    hard_R = pca_frame_masked(world, mask)
    hard_loss, hard_ang = frame_loss_and_angles(hard_R, gt_R)

    pred_R, n_eff, max_w = model(world, mask)
    pred_loss, pred_ang = frame_loss_and_angles(pred_R, gt_R)

    def mean_p95(x):
        return float(x.mean()), float(torch.quantile(x, 0.95))

    hm, hp = mean_p95(hard_ang.reshape(-1))
    wm, wp = mean_p95(pred_ang.reshape(-1))
    ne_m, ne_p10 = float(n_eff.mean()), float(torch.quantile(n_eff, 0.10))

    return {
        "case": name,
        "points": points,
        "mode": mode,
        "keep_frac": keep_frac,
        "hard_axis_mean": hm,
        "hard_axis_p95": hp,
        "weighted_axis_mean": wm,
        "weighted_axis_p95": wp,
        "hard_frame_loss": float(hard_loss.mean()),
        "weighted_frame_loss": float(pred_loss.mean()),
        "n_eff_mean": ne_m,
        "n_eff_p10": ne_p10,
        "max_weight_mean": float(max_w.mean()),
    }


@torch.no_grad()
def quick_val(model, args, device):
    model.eval()
    rows = [
        eval_case(model, "full_512", 512, "full", 1.0, args, device),
        eval_case(model, "dropout_0.8", 512, "dropout", 0.8, args, device),
        eval_case(model, "viewcrop_0.8", 512, "viewcrop", 0.8, args, device),
    ]
    return rows


def train_mode_batch(world, device):
    r = random.random()
    if r < 0.34:
        return "full", 1.0, make_mask(world, "full", 1.0, device)
    elif r < 0.67:
        k = random.uniform(0.6, 1.0)
        return "dropout", k, make_mask(world, "dropout", k, device)
    else:
        k = random.uniform(0.6, 1.0)
        return "viewcrop", k, make_mask(world, "viewcrop", k, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=1.0)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = WeightedPCANet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = max(1, args.train_n // args.batch_size)
    best = 1e9

    metrics_path = out / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        fields = ["epoch", "train_loss", "full", "dropout08", "viewcrop08", "n_eff_mean", "n_eff_p10"]
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for _ in range(steps_per_epoch):
                world, gt_R = sample_batch(args.batch_size, args.points, args, device)
                _, _, mask = train_mode_batch(world, device)

                pred_R, n_eff, max_w = model(world, mask)
                loss_vec, _ = frame_loss_and_angles(pred_R, gt_R)
                loss = loss_vec.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += float(loss)

            vals = quick_val(model, args, device)
            vdict = {r["case"]: r for r in vals}
            score = vdict["dropout_0.8"]["weighted_axis_mean"] + vdict["viewcrop_0.8"]["weighted_axis_mean"]

            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "score": score}, out / "best.pt")

            row = {
                "epoch": ep,
                "train_loss": total / steps_per_epoch,
                "full": vdict["full_512"]["weighted_axis_mean"],
                "dropout08": vdict["dropout_0.8"]["weighted_axis_mean"],
                "viewcrop08": vdict["viewcrop_0.8"]["weighted_axis_mean"],
                "n_eff_mean": vdict["dropout_0.8"]["n_eff_mean"],
                "n_eff_p10": vdict["dropout_0.8"]["n_eff_p10"],
            }
            wr.writerow(row)
            f.flush()

            print(
                f"ep={ep:03d} loss={row['train_loss']:.5f} "
                f"full={row['full']:.2f} drop08={row['dropout08']:.2f} "
                f"view08={row['viewcrop08']:.2f} neff={row['n_eff_mean']:.1f}/{row['n_eff_p10']:.1f}",
                flush=True,
            )

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cases = [
        ("full_512", 512, "full", 1.0),
        ("direct_410", 410, "full", 1.0),
        ("direct_307", 307, "full", 1.0),
        ("dropout_0.8", 512, "dropout", 0.8),
        ("dropout_0.6", 512, "dropout", 0.6),
        ("viewcrop_0.8", 512, "viewcrop", 0.8),
        ("viewcrop_0.6", 512, "viewcrop", 0.6),
    ]

    rows = [eval_case(model, *c, args=args, device=device) for c in cases]

    with open(out / "eval_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    with open(out / "summary_short.txt", "w") as f:
        for r in rows:
            line = (
                f"{r['case']:<14s} "
                f"hard={r['hard_axis_mean']:6.2f}/{r['hard_axis_p95']:6.2f} "
                f"weighted={r['weighted_axis_mean']:6.2f}/{r['weighted_axis_p95']:6.2f} "
                f"neff={r['n_eff_mean']:6.1f}/{r['n_eff_p10']:6.1f} "
                f"maxw={r['max_weight_mean']:.4f}"
            )
            print(line)
            f.write(line + "\n")

    print("OUT", out)
    print((out / "summary_short.txt").read_text())


if __name__ == "__main__":
    main()
