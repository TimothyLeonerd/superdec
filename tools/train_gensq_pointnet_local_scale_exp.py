#!/usr/bin/env python3
import argparse, csv, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fibonacci_sphere(n, device, dtype=torch.float32):
    i = torch.arange(n, device=device, dtype=dtype)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = phi * i
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    return torch.stack([x, y, z], dim=-1)


def bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


def sample_generalized_surface(scale, exp, dirs, newton_iters=32):
    """
    Robust differentiable radial sampler for

        |x/A|^r + |y/B|^s + |z/C|^t = 1

    along directions `dirs`.

    Forward pass:
        bracketed bisection guarantees bounded valid samples.

    Backward pass:
        after bisection, apply one implicit Newton correction from the
        detached root. This gives gradients w.r.t. scale and exponents
        without allowing Newton overshoot in the forward pass.
    """
    B = scale.shape[0]
    S = dirs.shape[0]

    dtype = scale.dtype
    device = scale.device

    u = dirs.to(device=device, dtype=dtype)[None].expand(B, S, 3)
    A = scale[:, None, :].clamp_min(1e-8)
    e = exp[:, None, :].clamp_min(0.05)

    abs_u = u.abs().clamp_min(1e-12)

    # Coordinate validity requires rho <= A_i / |u_i|.
    rho_axis_hi = A / abs_u
    rho_hi_bound = rho_axis_hi.min(dim=-1, keepdim=True).values.clamp_min(1e-12)

    coeff = (abs_u / A).pow(e)

    # Safe forward root by bisection. The branch decisions are intentionally
    # treated as numerical root-finding, not as the gradient path.
    rho_lo = torch.zeros_like(rho_hi_bound)
    rho_hi = rho_hi_bound

    for _ in range(int(newton_iters)):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        f_mid = (coeff * rho_mid.clamp_min(1e-12).pow(e)).sum(dim=-1, keepdim=True) - 1.0
        too_high = f_mid >= 0.0
        rho_hi = torch.where(too_high, rho_mid, rho_hi)
        rho_lo = torch.where(too_high, rho_lo, rho_mid)

    rho_root = (0.5 * (rho_lo + rho_hi)).detach().clamp_min(1e-12)

    # Differentiable implicit correction:
    # F(rho, A, e) = sum_i (|u_i|/A_i)^e_i * rho^e_i - 1 = 0
    # rho = rho_root - F / dF_drho
    # At an accurate root this changes the forward value negligibly but
    # provides the correct local implicit gradient.
    rho_pow = rho_root.pow(e)
    F = (coeff * rho_pow).sum(dim=-1, keepdim=True) - 1.0
    dF = (coeff * e * rho_root.pow(e - 1.0)).sum(dim=-1, keepdim=True).clamp_min(1e-12)

    rho = rho_root - F / dF

    # Safety: bisection already gives an in-bounds root; this clamp should
    # almost never activate, but prevents numerical nonsense.
    rho = rho.clamp_min(1e-12)
    rho = torch.minimum(rho, rho_hi_bound)

    return u * rho
@torch.no_grad()
def make_dataset(n, points, scale_min, scale_max, exp_min, exp_max, device, chunk=2048):
    dirs = fibonacci_sphere(points, device)
    all_pts, all_scale, all_exp = [], [], []

    for start in range(0, n, chunk):
        b = min(chunk, n - start)

        scale = scale_min + (scale_max - scale_min) * torch.rand(b, 3, device=device)
        exp = exp_min + (exp_max - exp_min) * torch.rand(b, 3, device=device)

        pts = sample_generalized_surface(scale, exp, dirs)

        all_pts.append(pts.cpu())
        all_scale.append(scale.cpu())
        all_exp.append(exp.cpu())

    return torch.cat(all_pts, 0), torch.cat(all_scale, 0), torch.cat(all_exp, 0)


class PointNetScaleExp(nn.Module):
    def __init__(self, scale_min, scale_max, exp_min, exp_max):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.exp_min = exp_min
        self.exp_max = exp_max

        self.point = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.scale_head = nn.Linear(256, 3)
        self.exp_head = nn.Linear(256, 3)

    def forward(self, x):
        h = self.point(x)
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
        z = self.trunk(pooled)

        scale = bounded(self.scale_head(z), self.scale_min, self.scale_max)
        exp = bounded(self.exp_head(z), self.exp_min, self.exp_max)
        return scale, exp


def chamfer(a, b):
    d2 = torch.cdist(a, b).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


@torch.no_grad()
def evaluate(model, loader, dirs, args, device):
    model.eval()

    scale_ls, exp_ls, losses, cds = [], [], [], []

    for pts, gt_scale, gt_exp in loader:
        pts = pts.to(device)
        gt_scale = gt_scale.to(device)
        gt_exp = gt_exp.to(device)

        pred_scale, pred_exp = model(pts)

        scale_l1 = (pred_scale - gt_scale).abs().mean(dim=1)
        exp_l1 = (pred_exp - gt_exp).abs().mean(dim=1)

        scale_loss = scale_l1 / (args.scale_max - args.scale_min)
        exp_loss = exp_l1 / (args.exp_max - args.exp_min)
        loss = args.w_scale * scale_loss + args.w_exp * exp_loss

        pred_pts = sample_generalized_surface(pred_scale, pred_exp, dirs)
        cd = chamfer(pred_pts, pts)

        scale_ls.append(scale_l1.cpu())
        exp_ls.append(exp_l1.cpu())
        losses.append(loss.cpu())
        cds.append(cd.cpu())

    scale_l1 = torch.cat(scale_ls)
    exp_l1 = torch.cat(exp_ls)
    loss = torch.cat(losses)
    cd = torch.cat(cds)

    return {
        "loss": float(loss.mean()),
        "scale_l1_mean": float(scale_l1.mean()),
        "scale_l1_p95": float(torch.quantile(scale_l1, 0.95)),
        "exp_l1_mean": float(exp_l1.mean()),
        "exp_l1_p95": float(torch.quantile(exp_l1, 0.95)),
        "cd_mean": float(cd.mean()),
        "cd_p95": float(torch.quantile(cd, 0.95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-n", type=int, default=10000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=1.0)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--w-exp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs = fibonacci_sphere(args.points, device)

    print("Generating train data...", flush=True)
    train = make_dataset(args.train_n, args.points, args.scale_min, args.scale_max, args.exp_min, args.exp_max, device)

    print("Generating val data...", flush=True)
    val = make_dataset(args.val_n, args.points, args.scale_min, args.scale_max, args.exp_min, args.exp_max, device)

    train_loader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(*val), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PointNetScaleExp(args.scale_min, args.scale_max, args.exp_min, args.exp_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fields = [
        "epoch", "train_loss", "val_loss",
        "val_scale_l1_mean", "val_scale_l1_p95",
        "val_exp_l1_mean", "val_exp_l1_p95",
        "val_cd_mean", "val_cd_p95",
    ]

    best_cd = 1e9

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total, seen = 0.0, 0

            for pts, gt_scale, gt_exp in train_loader:
                pts = pts.to(device)
                gt_scale = gt_scale.to(device)
                gt_exp = gt_exp.to(device)

                pred_scale, pred_exp = model(pts)

                scale_l1 = (pred_scale - gt_scale).abs().mean(dim=1)
                exp_l1 = (pred_exp - gt_exp).abs().mean(dim=1)

                scale_loss = scale_l1 / (args.scale_max - args.scale_min)
                exp_loss = exp_l1 / (args.exp_max - args.exp_min)

                loss = (args.w_scale * scale_loss + args.w_exp * exp_loss).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += loss.item() * pts.shape[0]
                seen += pts.shape[0]

            va = evaluate(model, val_loader, dirs, args, device)

            row = {
                "epoch": ep,
                "train_loss": total / max(seen, 1),
                "val_loss": va["loss"],
                "val_scale_l1_mean": va["scale_l1_mean"],
                "val_scale_l1_p95": va["scale_l1_p95"],
                "val_exp_l1_mean": va["exp_l1_mean"],
                "val_exp_l1_p95": va["exp_l1_p95"],
                "val_cd_mean": va["cd_mean"],
                "val_cd_p95": va["cd_p95"],
            }
            writer.writerow(row)
            f.flush()

            if va["cd_mean"] < best_cd:
                best_cd = va["cd_mean"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": va}, out / "best.pt")

            print(
                f"Epoch {ep:03d}/{args.epochs} "
                f"scale={va['scale_l1_mean']:.5f}/{va['scale_l1_p95']:.5f} "
                f"exp={va['exp_l1_mean']:.5f}/{va['exp_l1_p95']:.5f} "
                f"cd={va['cd_mean']:.7g}/{va['cd_p95']:.7g}",
                flush=True,
            )

    print("OUT", out)
    print(open(out / "metrics.csv").read().splitlines()[-1])


if __name__ == "__main__":
    main()
