#!/usr/bin/env python3
import argparse, csv, itertools, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PERMS = list(itertools.permutations(range(3)))


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


def bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


def sample_generalized_surface(scale, exp, dirs, newton_iters=32):
    """
    Robust radial sampler for the generalized superquadric surface

        |x/A|^r + |y/B|^s + |z/C|^t = 1

    along directions `dirs`.

    `newton_iters` is kept for API compatibility, but is now used as the
    number of bracketed bisection iterations. This avoids Newton overshoot
    for large exponents / small scales and guarantees |x_i| <= scale_i.
    """
    B = scale.shape[0]
    S = dirs.shape[0]

    dtype = scale.dtype
    device = scale.device

    u = dirs.to(device=device, dtype=dtype)[None].expand(B, S, 3)
    A = scale[:, None, :].clamp_min(1e-8)
    e = exp[:, None, :].clamp_min(0.05)

    abs_u = u.abs()

    # For x = rho * u, coordinate validity requires
    # rho <= A_i / |u_i| for every nonzero direction component.
    # Thus the true root is bracketed in [0, min_i A_i/|u_i|].
    huge = torch.full_like(abs_u, 1e8)
    rho_axis_hi = torch.where(abs_u > 1e-12, A / abs_u.clamp_min(1e-12), huge)
    rho_hi = rho_axis_hi.min(dim=-1, keepdim=True).values.clamp_min(1e-12)
    rho_lo = torch.zeros_like(rho_hi)

    coeff = (abs_u.clamp_min(1e-12) / A).pow(e)

    for _ in range(int(newton_iters)):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        f_mid = (coeff * rho_mid.clamp_min(1e-12).pow(e)).sum(dim=-1, keepdim=True) - 1.0

        # f(rho) is monotone increasing. Keep the root bracketed.
        too_high = f_mid >= 0.0
        rho_hi = torch.where(too_high, rho_mid, rho_hi)
        rho_lo = torch.where(too_high, rho_lo, rho_mid)

    rho = 0.5 * (rho_lo + rho_hi)
    return u * rho
@torch.no_grad()
def make_dataset(n, points, scale_min, scale_max, exp_min, exp_max, device, chunk=2048):
    dirs = fibonacci_sphere(points, device)
    all_pts, all_R, all_scale, all_exp = [], [], [], []

    for start in range(0, n, chunk):
        b = min(chunk, n - start)
        scale = scale_min + (scale_max - scale_min) * torch.rand(b, 3, device=device)
        exp = exp_min + (exp_max - exp_min) * torch.rand(b, 3, device=device)
        R = random_rotations(b, device)

        local = sample_generalized_surface(scale, exp, dirs)
        world = local @ R.transpose(1, 2)

        all_pts.append(world.cpu())
        all_R.append(R.cpu())
        all_scale.append(scale.cpu())
        all_exp.append(exp.cpu())

    return (
        torch.cat(all_pts, dim=0),
        torch.cat(all_R, dim=0),
        torch.cat(all_scale, dim=0),
        torch.cat(all_exp, dim=0),
    )


class PointNetPerm(nn.Module):
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
        self.frame_head = nn.Linear(256, 9)
        self.scale_head = nn.Linear(256, 3)
        self.exp_head = nn.Linear(256, 3)

    def forward(self, x):
        h = self.point(x)
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
        z = self.trunk(pooled)

        M = self.frame_head(z).reshape(-1, 3, 3)
        U, _S, Vh = torch.linalg.svd(M)
        R = U @ Vh
        det = torch.det(R)
        D = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.shape[0], 3, 3).clone()
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)
        R = U @ D @ Vh

        scale = bounded(self.scale_head(z), self.scale_min, self.scale_max)
        exp = bounded(self.exp_head(z), self.exp_min, self.exp_max)
        return R, scale, exp


def perm_loss(pred_R, pred_scale, pred_exp, gt_R, gt_scale, gt_exp, args):
    all_total, all_axis, all_scale, all_exp = [], [], [], []

    for p in PERMS:
        gt_Rp = gt_R[:, :, list(p)]
        gt_sp = gt_scale[:, list(p)]
        gt_ep = gt_exp[:, list(p)]

        dot = (pred_R * gt_Rp).sum(dim=1).abs().clamp(0, 1)  # [B,3]
        axis_l = (1.0 - dot.pow(2)).mean(dim=1)

        scale_l = (pred_scale - gt_sp).abs().mean(dim=1) / (args.scale_max - args.scale_min)
        exp_l = (pred_exp - gt_ep).abs().mean(dim=1) / (args.exp_max - args.exp_min)

        total = args.w_axis * axis_l + args.w_scale * scale_l + args.w_exp * exp_l

        all_total.append(total)
        all_axis.append(axis_l)
        all_scale.append((pred_scale - gt_sp).abs().mean(dim=1))
        all_exp.append((pred_exp - gt_ep).abs().mean(dim=1))

    total = torch.stack(all_total, dim=1)  # [B,6]
    axis = torch.stack(all_axis, dim=1)
    scale = torch.stack(all_scale, dim=1)
    exp = torch.stack(all_exp, dim=1)

    best_idx = total.argmin(dim=1)
    best_total = total.gather(1, best_idx[:, None]).squeeze(1)
    best_axis = axis.gather(1, best_idx[:, None]).squeeze(1)
    best_scale = scale.gather(1, best_idx[:, None]).squeeze(1)
    best_exp = exp.gather(1, best_idx[:, None]).squeeze(1)

    return best_total, best_axis, best_scale, best_exp, best_idx


def chamfer(a, b):
    d2 = torch.cdist(a, b).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


@torch.no_grad()
def evaluate(model, loader, dirs, args, device):
    model.eval()
    losses, axis_ls, scale_ls, exp_ls, cds = [], [], [], [], []

    for pts, gt_R, gt_scale, gt_exp in loader:
        pts = pts.to(device)
        gt_R = gt_R.to(device)
        gt_scale = gt_scale.to(device)
        gt_exp = gt_exp.to(device)

        pred_R, pred_scale, pred_exp = model(pts)
        l, ax, sc, ex, _ = perm_loss(pred_R, pred_scale, pred_exp, gt_R, gt_scale, gt_exp, args)

        pred_local = sample_generalized_surface(pred_scale, pred_exp, dirs)
        pred_world = pred_local @ pred_R.transpose(1, 2)

        pred_local = sample_generalized_surface(pred_scale, pred_exp, dirs)
        pred_world = pred_local @ pred_R.transpose(1, 2)
        cd = chamfer(pred_world, pts)

        losses.append(l.cpu())
        axis_ls.append(ax.cpu())
        scale_ls.append(sc.cpu())
        exp_ls.append(ex.cpu())
        cds.append(cd.cpu())

    loss = torch.cat(losses)
    axis_l = torch.cat(axis_ls)
    scale_l = torch.cat(scale_ls)
    exp_l = torch.cat(exp_ls)
    cd = torch.cat(cds)

    angle = torch.asin(torch.sqrt(axis_l.clamp(0, 1))) * (180.0 / math.pi)

    return {
        "loss": float(loss.mean()),
        "axis_angle_mean": float(angle.mean()),
        "axis_angle_p95": float(torch.quantile(angle, 0.95)),
        "scale_l1_mean": float(scale_l.mean()),
        "scale_l1_p95": float(torch.quantile(scale_l, 0.95)),
        "exp_l1_mean": float(exp_l.mean()),
        "exp_l1_p95": float(torch.quantile(exp_l, 0.95)),
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
    ap.add_argument("--w-axis", type=float, default=1.0)
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

    model = PointNetPerm(args.scale_min, args.scale_max, args.exp_min, args.exp_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fields = [
        "epoch", "train_loss",
        "val_loss", "val_axis_angle_mean", "val_axis_angle_p95",
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

            for pts, gt_R, gt_scale, gt_exp in train_loader:
                pts = pts.to(device)
                gt_R = gt_R.to(device)
                gt_scale = gt_scale.to(device)
                gt_exp = gt_exp.to(device)

                pred_R, pred_scale, pred_exp = model(pts)
                loss_per, *_ = perm_loss(pred_R, pred_scale, pred_exp, gt_R, gt_scale, gt_exp, args)
                loss = loss_per.mean()

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
                "val_axis_angle_mean": va["axis_angle_mean"],
                "val_axis_angle_p95": va["axis_angle_p95"],
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
                f"axis={va['axis_angle_mean']:.2f}/{va['axis_angle_p95']:.2f} "
                f"scale={va['scale_l1_mean']:.4f} "
                f"exp={va['exp_l1_mean']:.4f} "
                f"cd={va['cd_mean']:.7g}",
                flush=True,
            )

    print("OUT", out)
    print(open(out / "metrics.csv").read().splitlines()[-1])


if __name__ == "__main__":
    main()
