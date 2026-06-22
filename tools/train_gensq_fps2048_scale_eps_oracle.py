#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import tools.train_gensq_unrolled_coupled_final_cd_stepidx as base


def chamfer_sq(a, b):
    d = torch.cdist(a, b).pow(2)
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)


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


class EpsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, X_scaled, raw_eps, log_scale, step_t):
        B, S, _ = X_scaled.shape
        re = raw_eps[:, None, :].expand(B, S, 3)
        ls = log_scale[:, None, :].expand(B, S, 3)
        st = step_t[:, None, None].expand(B, S, 1)
        feat = torch.cat([X_scaled, re, ls, st], dim=-1)
        h = self.point(feat).amax(dim=1)
        return self.head(h)


class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, X, log_scale, raw_eps, step_t):
        B, S, _ = X.shape
        ls = log_scale[:, None, :].expand(B, S, 3)
        re = raw_eps[:, None, :].expand(B, S, 3)
        st = step_t[:, None, None].expand(B, S, 1)
        feat = torch.cat([X, ls, re, st], dim=-1)
        h = self.point(feat).amax(dim=1)
        return self.head(h)


def make_batch(args, B, dirs_pred, device):
    scale_gt = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    eps_gt = base.sample_log_uniform(args.exp_min, args.exp_max, (B, 3), device)

    dirs_src = base.fibonacci_sphere(args.source_points, device)
    dirs = dirs_src[None, :, :].expand(B, args.source_points, 3)
    X_dense = base.sample_gensq_batched_dirs(scale_gt, eps_gt, dirs)

    X = fps_downsample(X_dense, args.points)
    return X, scale_gt, eps_gt


def init_state(args, B, device):
    scale0 = math.sqrt(args.scale_min * args.scale_max)
    eps0 = math.sqrt(args.exp_min * args.exp_max)

    log_scale = torch.full((B, 3), math.log(scale0), device=device)
    eps0_t = torch.full((B, 3), eps0, device=device)
    raw_eps = base.exp_to_raw(eps0_t, args.exp_min, args.exp_max)
    return log_scale, raw_eps


def sample_pred(args, log_scale, raw_eps, dirs_pred):
    B = log_scale.shape[0]
    scale = log_scale.exp().clamp_min(1e-8)
    eps = base.raw_to_exp(raw_eps, args.exp_min, args.exp_max)
    dirs = dirs_pred[None, :, :].expand(B, dirs_pred.shape[0], 3)
    return base.sample_gensq_batched_dirs(scale, eps, dirs)


def unroll(args, X, scale_gt, eps_gt, eps_net, scale_net):
    B = X.shape[0]
    device = X.device
    log_scale, raw_eps = init_state(args, B, device)

    if args.mode == "eps":
        log_scale = scale_gt.log()
    if args.mode == "scale":
        raw_eps = base.exp_to_raw(eps_gt, args.exp_min, args.exp_max)

    hist = [(log_scale, raw_eps)]

    log_floor = math.log(args.scale_floor)
    log_ceil = math.log(args.scale_ceil)

    for k in range(args.unroll_steps):
        step_t = torch.full((B,), float(k) / max(args.unroll_steps - 1, 1), device=device)

        if args.mode in ["eps", "joint"]:
            scale_for_eps = log_scale.exp().detach().clamp_min(1e-8)
            X_scaled = X / scale_for_eps[:, None, :]
            d_eps = eps_net(X_scaled, raw_eps, log_scale.detach(), step_t)
            raw_eps = (raw_eps + args.eps_damp * d_eps).clamp(-args.raw_eps_clip, args.raw_eps_clip)

        if args.mode in ["scale", "joint"]:
            d_scale = scale_net(X, log_scale, raw_eps.detach(), step_t)
            log_scale = (log_scale + args.scale_damp * d_scale).clamp(log_floor, log_ceil)

        if args.mode == "eps":
            log_scale = scale_gt.log()
        if args.mode == "scale":
            raw_eps = base.exp_to_raw(eps_gt, args.exp_min, args.exp_max)

        hist.append((log_scale, raw_eps))

    return hist


def train_loss(args, X, scale_gt, eps_gt, hist, dirs_pred):
    log_scale, raw_eps = hist[-1]

    if args.mode == "eps":
        Y = sample_pred(args, scale_gt.log(), raw_eps, dirs_pred)
        return chamfer_sq(Y, X).mean()

    if args.mode == "scale":
        raw_gt = base.exp_to_raw(eps_gt, args.exp_min, args.exp_max)
        Y = sample_pred(args, log_scale, raw_gt, dirs_pred)
        return chamfer_sq(Y, X).mean()

    # joint: detach-routed like previous baseline
    Y_s = sample_pred(args, log_scale, raw_eps.detach(), dirs_pred)
    loss_s = chamfer_sq(Y_s, X).mean()
    Y_e = sample_pred(args, log_scale.detach(), raw_eps, dirs_pred)
    loss_e = chamfer_sq(Y_e, X).mean()
    return loss_s + loss_e


def stats(x):
    x = x.detach().float().cpu()
    return float(x.mean())


@torch.no_grad()
def evaluate(args, eps_net, scale_net, val_data, dirs_pred, device):
    eps_net.eval()
    scale_net.eval()

    X_all, scale_all, eps_all = val_data
    loader = DataLoader(TensorDataset(X_all, scale_all, eps_all), batch_size=args.batch_size)

    rows = [{"cd": [], "scale_l1": [], "eps_l1": [], "log_eps_l1": []} for _ in range(args.unroll_steps + 1)]

    for X, scale_gt, eps_gt in loader:
        X = X.to(device)
        scale_gt = scale_gt.to(device)
        eps_gt = eps_gt.to(device)

        hist = unroll(args, X, scale_gt, eps_gt, eps_net, scale_net)

        for j, (log_scale, raw_eps) in enumerate(hist):
            Y = sample_pred(args, log_scale, raw_eps, dirs_pred)
            cd = chamfer_sq(Y, X)

            scale = log_scale.exp().clamp_min(1e-8)
            eps = base.raw_to_exp(raw_eps, args.exp_min, args.exp_max)

            rows[j]["cd"].append(cd.cpu())
            rows[j]["scale_l1"].append((scale - scale_gt).abs().mean(dim=1).cpu())
            rows[j]["eps_l1"].append((eps - eps_gt).abs().mean(dim=1).cpu())
            rows[j]["log_eps_l1"].append((eps.log() - eps_gt.log()).abs().mean(dim=1).cpu())

    out = []
    for j in range(args.unroll_steps + 1):
        out.append({k: stats(torch.cat(v)) for k, v in rows[j].items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eps", "scale", "joint"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-n", type=int, default=50000)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--source-points", type=int, default=2048)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--scale-floor", type=float, default=0.02)
    ap.add_argument("--scale-ceil", type=float, default=0.80)
    ap.add_argument("--exp-min", type=float, default=0.5)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--raw-eps-clip", type=float, default=8.0)
    ap.add_argument("--unroll-steps", type=int, default=4)
    ap.add_argument("--scale-damp", type=float, default=0.5)
    ap.add_argument("--eps-damp", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    base.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dirs_pred = base.fibonacci_sphere(args.points, device)

    print(f"device={device} mode={args.mode}", flush=True)

    base.set_seed(args.seed + 2000)
    vals = []
    remaining = args.val_n
    while remaining > 0:
        b = min(args.batch_size, remaining)
        vals.append(tuple(t.cpu() for t in make_batch(args, b, dirs_pred, device)))
        remaining -= b
    val_data = tuple(torch.cat([v[i] for v in vals], dim=0) for i in range(3))

    eps_net = EpsNet().to(device)
    scale_net = ScaleNet().to(device)

    params = []
    if args.mode in ["eps", "joint"]:
        params += list(eps_net.parameters())
    if args.mode in ["scale", "joint"]:
        params += list(scale_net.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(args.train_n / args.batch_size)

    best_eps = (float("inf"), None, None)
    best_scale = (float("inf"), None, None)
    best_cd = (float("inf"), None, None)

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "cd", "scale_l1", "eps_l1", "log_eps_l1"])
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            eps_net.train()
            scale_net.train()
            total = 0.0
            seen = 0

            for _ in range(steps_per_epoch):
                b = min(args.batch_size, args.train_n - seen)
                if b <= 0:
                    break

                X, scale_gt, eps_gt = make_batch(args, b, dirs_pred, device)
                hist = unroll(args, X, scale_gt, eps_gt, eps_net, scale_net)
                loss = train_loss(args, X, scale_gt, eps_gt, hist, dirs_pred)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()

                total += float(loss.detach()) * b
                seen += b

            ev = evaluate(args, eps_net, scale_net, val_data, dirs_pred, device)
            final = ev[-1]

            row = {
                "epoch": ep,
                "train_loss": total / max(1, seen),
                "cd": final["cd"],
                "scale_l1": final["scale_l1"],
                "eps_l1": final["eps_l1"],
                "log_eps_l1": final["log_eps_l1"],
            }
            writer.writerow(row)
            f.flush()

            if row["cd"] < best_cd[0]:
                best_cd = (row["cd"], ep, ev)
            if row["scale_l1"] < best_scale[0]:
                best_scale = (row["scale_l1"], ep, ev)
            if row["eps_l1"] < best_eps[0]:
                best_eps = (row["eps_l1"], ep, ev)

            print(
                f"ep={ep:03d} train={row['train_loss']:.6g} "
                f"cd={row['cd']:.8g} scale={row['scale_l1']:.6g} "
                f"eps={row['eps_l1']:.6g} logeps={row['log_eps_l1']:.6g}",
                flush=True,
            )

    lines = [
        f"experiment=fps2048_to_512_{args.mode}_oracle",
        f"mode={args.mode}",
        f"source_points={args.source_points}",
        f"observed_points={args.points}",
        f"unroll_steps={args.unroll_steps}",
        f"best_cd_epoch={best_cd[1]} best_cd={best_cd[0]:.8f}",
        f"best_scale_epoch={best_scale[1]} best_scale_l1={best_scale[0]:.8f}",
        f"best_eps_epoch={best_eps[1]} best_eps_l1={best_eps[0]:.8f}",
        "",
    ]

    for name, best in [("best_cd", best_cd), ("best_scale", best_scale), ("best_eps", best_eps)]:
        _, ep, ev = best
        lines.append(f"{name}_epoch={ep}")
        for j, d in enumerate(ev):
            lines.append(
                f"step {j}: cd={d['cd']:.8f} "
                f"scale_l1={d['scale_l1']:.8f} "
                f"eps_l1={d['eps_l1']:.8f} "
                f"log_eps_l1={d['log_eps_l1']:.8f}"
            )
        lines.append("")

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
