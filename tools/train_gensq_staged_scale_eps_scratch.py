#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
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


def sample_log_uniform(lo, hi, shape, device):
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(shape, device=device))


def exp_to_raw(eps, lo, hi):
    t = ((eps - lo) / (hi - lo)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(t / (1.0 - t))


def raw_to_exp(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


def sample_gensq_batched_dirs(scale, exp, dirs_batched, iters=32):
    """
    Differentiable robust radial sampler.
      scale: [B,3]
      exp:   [B,3]
      dirs:  [B,S,3]
    """
    u = dirs_batched
    A = scale[:, None, :].clamp_min(1e-8)
    e = exp[:, None, :].clamp_min(0.05)
    abs_u = u.abs().clamp_min(1e-12)

    rho_hi_bound = (A / abs_u).min(dim=-1, keepdim=True).values.clamp_min(1e-12)
    rho_lo = torch.zeros_like(rho_hi_bound)
    rho_hi = rho_hi_bound

    coeff = (abs_u / A).pow(e)

    # robust forward bisection
    for _ in range(iters):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        f_mid = (coeff * rho_mid.clamp_min(1e-12).pow(e)).sum(dim=-1, keepdim=True) - 1.0
        too_high = f_mid >= 0.0
        rho_hi = torch.where(too_high, rho_mid, rho_hi)
        rho_lo = torch.where(too_high, rho_lo, rho_mid)

    # differentiable implicit correction around detached root
    rho_root = (0.5 * (rho_lo + rho_hi)).detach().clamp_min(1e-12)
    Fval = (coeff * rho_root.pow(e)).sum(dim=-1, keepdim=True) - 1.0
    dF = (coeff * e * rho_root.pow(e - 1.0)).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    rho = rho_root - Fval / dF
    rho = rho.clamp_min(1e-12)
    rho = torch.minimum(rho, rho_hi_bound)
    return u * rho


def chamfer(a, b):
    d = torch.cdist(a, b) ** 2
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)


def absmax_scale(X):
    return X.abs().amax(dim=1).clamp_min(1e-8)


def normalize_by_scale(X, scale):
    return X / scale[:, None, :].clamp_min(1e-8)


def pred_absmax_normalize(Y):
    return Y / Y.abs().amax(dim=1).clamp_min(1e-8)[:, None, :]


def ray_dirs(X):
    return F.normalize(X, p=2, dim=-1, eps=1e-12)


class EpsNet(nn.Module):
    """
    Absolute eps predictor.
    Per-point input: X_norm, raw_eps_prev, log_scale_est.
    """
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, X_norm, raw_eps_prev, log_scale):
        B, S, _ = X_norm.shape
        raw_rep = raw_eps_prev[:, None, :].expand(B, S, 3)
        scale_rep = log_scale[:, None, :].expand(B, S, 3)
        h = self.point(torch.cat([X_norm, raw_rep, scale_rep], dim=-1)).amax(dim=1)
        return self.head(h)


class ScaleNet(nn.Module):
    """
    Delta log-scale predictor.
    Per-point input: raw X, log_scale0, raw_eps.
    """
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, X, log_scale, raw_eps):
        B, S, _ = X.shape
        scale_rep = log_scale[:, None, :].expand(B, S, 3)
        eps_rep = raw_eps[:, None, :].expand(B, S, 3)
        h = self.point(torch.cat([X, scale_rep, eps_rep], dim=-1)).amax(dim=1)
        return self.head(h)


def make_batch(args, B, dirs_base, device):
    S = dirs_base.shape[0]
    scale_gt = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    eps_gt = sample_log_uniform(args.exp_min, args.exp_max, (B, 3), device)

    dirs = dirs_base[None, :, :].expand(B, S, 3)
    X = sample_gensq_batched_dirs(scale_gt, eps_gt, dirs)

    scale0 = absmax_scale(X)
    log_scale0 = torch.log(scale0)

    eps0 = torch.full((B, 3), args.eps0, device=device)
    raw_eps0 = exp_to_raw(eps0, args.exp_min, args.exp_max)

    return X, dirs, scale_gt, eps_gt, scale0, log_scale0, raw_eps0


def eps_ray_loss(X_norm, eps_pred, dirs_norm, exp_min, exp_max, pred_normalize):
    B = X_norm.shape[0]
    unit = torch.ones(B, 3, device=X_norm.device)
    Y = sample_gensq_batched_dirs(unit, eps_pred, dirs_norm)
    if pred_normalize:
        Y = pred_absmax_normalize(Y)
    return ((X_norm - Y) ** 2).sum(dim=-1).mean()


def forward_staged(args, eps_net, scale_net, batch):
    X, dirs, scale_gt, eps_gt, scale0, log_scale0, raw_eps0 = batch
    B = X.shape[0]

    # Stage 1: eps from bbox-normalized, ray-aligned input.
    Xn0 = normalize_by_scale(X, scale0)
    rays0 = ray_dirs(Xn0)
    raw_eps1 = eps_net(Xn0, raw_eps0, log_scale0)
    eps1 = raw_to_exp(raw_eps1, args.exp_min, args.exp_max)

    loss_eps1 = eps_ray_loss(
        Xn0, eps1, rays0, args.exp_min, args.exp_max, pred_normalize=args.pred_normalize
    )

    # Stage 2: scale refinement using eps1 as current shape estimate.
    # Detach eps1 so scale loss does not corrupt EpsNet into scale/shape compensation.
    delta_log_scale = scale_net(X, log_scale0, raw_eps1.detach())
    log_scale1 = log_scale0 + args.scale_alpha * delta_log_scale
    scale1 = torch.exp(log_scale1).clamp_min(1e-8)

    Y_scale = sample_gensq_batched_dirs(scale1, eps1.detach(), dirs)
    loss_scale_cd = chamfer(X, Y_scale).mean()

    # Keep scale refinement conservative around bbox.
    loss_scale_reg = ((log_scale1 - log_scale0) ** 2).mean()

    # Stage 3: eps refinement after scale update.
    Xn1 = normalize_by_scale(X, scale1.detach())
    rays1 = ray_dirs(Xn1)
    raw_eps2 = eps_net(Xn1, raw_eps1.detach(), log_scale1.detach())
    eps2 = raw_to_exp(raw_eps2, args.exp_min, args.exp_max)

    loss_eps2 = eps_ray_loss(
        Xn1, eps2, rays1, args.exp_min, args.exp_max, pred_normalize=args.pred_normalize
    )

    loss = (
        args.w_eps1 * loss_eps1
        + args.w_scale * loss_scale_cd
        + args.w_scale_reg * loss_scale_reg
        + args.w_eps2 * loss_eps2
    )

    return {
        "loss": loss,
        "loss_eps1": loss_eps1.detach(),
        "loss_scale_cd": loss_scale_cd.detach(),
        "loss_scale_reg": loss_scale_reg.detach(),
        "loss_eps2": loss_eps2.detach(),
        "scale0": scale0,
        "scale1": scale1,
        "eps1": eps1,
        "eps2": eps2,
        "raw_eps1": raw_eps1,
        "raw_eps2": raw_eps2,
    }


@torch.no_grad()
def summarize(x):
    x = x.detach().flatten()
    return (
        x.mean().item(),
        x.median().item(),
        torch.quantile(x, 0.90).item(),
        torch.quantile(x, 0.95).item(),
        x.max().item(),
    )


def fmt(name, s):
    return f"{name} mean/med/p90/p95/max = {s[0]:.8f} {s[1]:.8f} {s[2]:.8f} {s[3]:.8f} {s[4]:.8f}"


@torch.no_grad()
def eval_model(args, eps_net, scale_net, val_batch):
    X, dirs, scale_gt, eps_gt, scale0, log_scale0, raw_eps0 = val_batch

    out = forward_staged(args, eps_net, scale_net, val_batch)

    scale1 = out["scale1"]
    eps1 = out["eps1"]
    eps2 = out["eps2"]

    Y0e1 = sample_gensq_batched_dirs(scale0, eps1, dirs)
    Y1e1 = sample_gensq_batched_dirs(scale1, eps1, dirs)
    Y1e2 = sample_gensq_batched_dirs(scale1, eps2, dirs)

    cd0e1 = chamfer(X, Y0e1)
    cd1e1 = chamfer(X, Y1e1)
    cd1e2 = chamfer(X, Y1e2)

    return {
        "scale0_l1": summarize((scale0 - scale_gt).abs().mean(dim=1)),
        "scale1_l1": summarize((scale1 - scale_gt).abs().mean(dim=1)),
        "eps1_l1": summarize((eps1 - eps_gt).abs().mean(dim=1)),
        "eps2_l1": summarize((eps2 - eps_gt).abs().mean(dim=1)),
        "cd_scale0_eps1": summarize(cd0e1),
        "cd_scale1_eps1": summarize(cd1e1),
        "cd_scale1_eps2": summarize(cd1e2),
        "delta_log_scale": summarize((torch.log(scale1) - torch.log(scale0)).abs().mean(dim=1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-n", type=int, default=50000)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=0.5)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--eps0", type=float, default=2.0)
    ap.add_argument("--scale-alpha", type=float, default=0.5)
    ap.add_argument("--w-eps1", type=float, default=1.0)
    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--w-scale-reg", type=float, default=0.01)
    ap.add_argument("--w-eps2", type=float, default=1.0)
    ap.add_argument("--pred-normalize", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs_base = fibonacci_sphere(args.points, device)

    eps_net = EpsNet().to(device)
    scale_net = ScaleNet().to(device)

    opt = torch.optim.AdamW(
        list(eps_net.parameters()) + list(scale_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    val_batch = make_batch(args, args.val_n, dirs_base, device)

    metrics_path = out_dir / "metrics.csv"
    best = {"eps2": (float("inf"), -1, None), "cd1e2": (float("inf"), -1, None)}

    with metrics_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "epoch", "train_loss", "loss_eps1", "loss_scale_cd", "loss_scale_reg", "loss_eps2",
            "scale0_l1", "scale1_l1", "eps1_l1", "eps2_l1",
            "cd_scale0_eps1", "cd_scale1_eps1", "cd_scale1_eps2", "delta_log_scale",
        ])

        for ep in range(1, args.epochs + 1):
            eps_net.train()
            scale_net.train()

            losses = []
            le1s, lscs, lregs, le2s = [], [], [], []
            n_batches = math.ceil(args.train_n / args.batch_size)

            for _ in range(n_batches):
                batch = make_batch(args, args.batch_size, dirs_base, device)
                out = forward_staged(args, eps_net, scale_net, batch)
                loss = out["loss"]

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(eps_net.parameters()) + list(scale_net.parameters()), args.grad_clip)
                opt.step()

                losses.append(loss.item())
                le1s.append(out["loss_eps1"].item())
                lscs.append(out["loss_scale_cd"].item())
                lregs.append(out["loss_scale_reg"].item())
                le2s.append(out["loss_eps2"].item())

            eps_net.eval()
            scale_net.eval()
            ev = eval_model(args, eps_net, scale_net, val_batch)

            train_loss = sum(losses) / len(losses)
            row = [
                ep, train_loss,
                sum(le1s) / len(le1s),
                sum(lscs) / len(lscs),
                sum(lregs) / len(lregs),
                sum(le2s) / len(le2s),
                ev["scale0_l1"][0],
                ev["scale1_l1"][0],
                ev["eps1_l1"][0],
                ev["eps2_l1"][0],
                ev["cd_scale0_eps1"][0],
                ev["cd_scale1_eps1"][0],
                ev["cd_scale1_eps2"][0],
                ev["delta_log_scale"][0],
            ]
            wr.writerow(row)
            f.flush()

            if ev["eps2_l1"][0] < best["eps2"][0]:
                best["eps2"] = (ev["eps2_l1"][0], ep, ev)
            if ev["cd_scale1_eps2"][0] < best["cd1e2"][0]:
                best["cd1e2"] = (ev["cd_scale1_eps2"][0], ep, ev)

            print(
                f"ep={ep:03d} train={train_loss:.8f} "
                f"sc0={ev['scale0_l1'][0]:.5f} sc1={ev['scale1_l1'][0]:.5f} "
                f"eps1={ev['eps1_l1'][0]:.5f} eps2={ev['eps2_l1'][0]:.5f} "
                f"cd1e2={ev['cd_scale1_eps2'][0]:.8f} "
                f"dlog={ev['delta_log_scale'][0]:.5f}",
                flush=True,
            )

    lines = []
    lines.append("experiment=staged_scale_eps_scratch_A")
    lines.append(f"scale0=absmax_bbox")
    lines.append(f"eps0={args.eps0}")
    lines.append(f"scale_alpha={args.scale_alpha}")
    lines.append(f"pred_normalize={args.pred_normalize}")
    lines.append(f"weights eps1/scale/reg/eps2 = {args.w_eps1}/{args.w_scale}/{args.w_scale_reg}/{args.w_eps2}")
    lines.append(f"best_eps2_epoch={best['eps2'][1]} best_eps2_l1={best['eps2'][0]:.8f}")
    lines.append(f"best_cd1e2_epoch={best['cd1e2'][1]} best_cd1e2={best['cd1e2'][0]:.8f}")

    for tag in ["eps2", "cd1e2"]:
        val, ep, ev = best[tag]
        lines.append("")
        lines.append(f"== best_{tag}_epoch={ep} ==")
        for k in [
            "scale0_l1", "scale1_l1", "eps1_l1", "eps2_l1",
            "cd_scale0_eps1", "cd_scale1_eps1", "cd_scale1_eps2", "delta_log_scale",
        ]:
            lines.append(fmt(k, ev[k]))

    (out_dir / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("=== summary ===")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
