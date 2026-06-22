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
    return torch.exp(
        math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(shape, device=device)
    )


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

    # Bracketed forward solve.
    for _ in range(iters):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        f_mid = (coeff * rho_mid.clamp_min(1e-12).pow(e)).sum(dim=-1, keepdim=True) - 1.0
        too_high = f_mid >= 0.0
        rho_hi = torch.where(too_high, rho_mid, rho_hi)
        rho_lo = torch.where(too_high, rho_lo, rho_mid)

    # Differentiable implicit correction.
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


def normalize_points(X, mode, gt_scale=None):
    if mode == "gt":
        return X / gt_scale[:, None, :].clamp_min(1e-8)

    if mode == "absmax":
        denom = X.abs().amax(dim=1).clamp_min(1e-8)
        return X / denom[:, None, :]

    if mode == "half":
        mn = X.amin(dim=1)
        mx = X.amax(dim=1)
        center = 0.5 * (mn + mx)
        half = (0.5 * (mx - mn)).clamp_min(1e-8)
        return (X - center[:, None, :]) / half[:, None, :]

    raise ValueError(mode)


class EpsPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, X, raw_eps0):
        B, S, _ = X.shape
        raw_rep = raw_eps0[:, None, :].expand(B, S, 3)
        feat = torch.cat([X, raw_rep], dim=-1)
        h = self.point(feat).amax(dim=1)
        return self.head(h)


@torch.no_grad()
def summarize_vec(x):
    x = x.detach().flatten()
    return (
        x.mean().item(),
        x.median().item(),
        torch.quantile(x, 0.90).item(),
        torch.quantile(x, 0.95).item(),
        x.max().item(),
    )


def make_batch(args, B, dirs_base, device):
    S = dirs_base.shape[0]
    scale_gt = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    eps_gt = sample_log_uniform(args.exp_min, args.exp_max, (B, 3), device)
    eps0 = sample_log_uniform(args.exp_min, args.exp_max, (B, 3), device)
    raw_eps0 = exp_to_raw(eps0, args.exp_min, args.exp_max)

    dirs = dirs_base[None, :, :].expand(B, S, 3)
    X = sample_gensq_batched_dirs(scale_gt, eps_gt, dirs)

    X_norm = normalize_points(X, args.norm_mode, gt_scale=scale_gt)

    # Ray-align prediction to the normalized target cloud.
    dirs_norm = F.normalize(X_norm, p=2, dim=-1, eps=1e-12)

    return X_norm, dirs_norm, raw_eps0, eps_gt


def eval_model(args, model, val_data, device):
    X_norm, dirs_norm, raw_eps0, eps_gt = val_data
    B = X_norm.shape[0]
    unit_scale = torch.ones(B, 3, device=device)

    out = {}
    raw_cur = raw_eps0.clone()

    for step in range(args.eval_steps + 1):
        eps_cur = raw_to_exp(raw_cur, args.exp_min, args.exp_max)
        Y = sample_gensq_batched_dirs(unit_scale, eps_cur, dirs_norm)

        if args.pred_normalize:
            Y_cmp = normalize_points(Y, args.norm_mode, gt_scale=unit_scale)
        else:
            Y_cmp = Y

        cd = chamfer(X_norm, Y_cmp)
        eps_l1 = (eps_cur - eps_gt).abs().mean(dim=1)

        out[step] = {
            "cd": summarize_vec(cd),
            "eps": summarize_vec(eps_l1),
        }

        if step == args.eval_steps:
            break

        raw_cur = model(X_norm, raw_cur)

    return out


def metric_line(name, tup):
    return f"{name} mean/med/p90/p95/max = {tup[0]:.8f} {tup[1]:.8f} {tup[2]:.8f} {tup[3]:.8f} {tup[4]:.8f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--norm-mode", choices=["gt", "absmax", "half"], default="absmax")
    ap.add_argument("--pred-normalize", action="store_true")
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
    ap.add_argument("--eval-steps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs_base = fibonacci_sphere(args.points, device)

    model = EpsPointNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    val_data = make_batch(args, args.val_n, dirs_base, device)

    unit_scale_cache = None
    metrics_path = out / "metrics.csv"

    best_cd = (float("inf"), -1, None)
    best_eps = (float("inf"), -1, None)

    with metrics_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss", "step", "cd_mean", "eps_l1_mean"])

        for ep in range(1, args.epochs + 1):
            model.train()
            losses = []
            n_batches = math.ceil(args.train_n / args.batch_size)

            for _ in range(n_batches):
                B = args.batch_size
                X_norm, dirs_norm, raw_eps0, eps_gt = make_batch(args, B, dirs_base, device)

                raw_pred = model(X_norm, raw_eps0)
                eps_pred = raw_to_exp(raw_pred, args.exp_min, args.exp_max)

                unit_scale = torch.ones(B, 3, device=device)
                Y = sample_gensq_batched_dirs(unit_scale, eps_pred, dirs_norm)

                if args.pred_normalize:
                    Y_cmp = normalize_points(Y, args.norm_mode, gt_scale=unit_scale)
                else:
                    Y_cmp = Y

                # Ray-aligned pointwise loss. This is the point of the experiment.
                loss = ((X_norm - Y_cmp) ** 2).sum(dim=-1).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                ev = eval_model(args, model, val_data, device)

            train_loss = sum(losses) / len(losses)
            for step, d in ev.items():
                wr.writerow([ep, train_loss, step, d["cd"][0], d["eps"][0]])
            f.flush()

            cd1 = ev[1]["cd"][0]
            eps1 = ev[1]["eps"][0]

            if cd1 < best_cd[0]:
                best_cd = (cd1, ep, ev)
            if eps1 < best_eps[0]:
                best_eps = (eps1, ep, ev)

            print(
                f"ep={ep:03d} train={train_loss:.8f} "
                f"s1_cd={cd1:.8f} s1_eps={eps1:.8f}",
                flush=True,
            )

    lines = []
    lines.append(f"norm_mode={args.norm_mode}")
    lines.append(f"pred_normalize={args.pred_normalize}")
    lines.append(f"best_cd_epoch={best_cd[1]} best_cd_s1={best_cd[0]:.8f}")
    lines.append(f"best_eps_epoch={best_eps[1]} best_eps_s1={best_eps[0]:.8f}")

    for tag, (_, ep, ev) in [("best_cd", best_cd), ("best_eps", best_eps)]:
        lines.append(f"{tag}_epoch={ep}")
        for step in [0, 1, args.eval_steps]:
            lines.append(
                f"step {step}: "
                f"{metric_line('cd', ev[step]['cd'])} | "
                f"{metric_line('eps_l1', ev[step]['eps'])}"
            )

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("=== summary ===")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
