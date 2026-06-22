#!/usr/bin/env python3
import argparse, csv, math, random
from pathlib import Path

import torch
import torch.nn as nn

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
    chamfer,
)


def bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


def inv_bounded(x, lo, hi, eps=1e-5):
    y = (x - lo) / (hi - lo)
    y = y.clamp(eps, 1.0 - eps)
    return torch.log(y / (1.0 - y))


def sample_log_uniform(lo, hi, shape, device):
    return torch.exp(
        math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(shape, device=device)
    )


@torch.no_grad()
def make_batch(batch_size, dirs, args, device):
    scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(batch_size, 3, device=device)

    eps_gt = sample_log_uniform(args.exp_min, args.exp_max, (batch_size, 3), device)
    eps0   = sample_log_uniform(args.exp_min, args.exp_max, (batch_size, 3), device)

    raw_eps0 = inv_bounded(eps0, args.exp_min, args.exp_max)

    X  = sample_generalized_surface(scale, eps_gt, dirs)
    Y0 = sample_generalized_surface(scale, eps0, dirs)

    return X, Y0, scale, eps_gt, eps0, raw_eps0


@torch.no_grad()
def make_fixed_dataset(n, dirs, args, device, chunk=512):
    xs, y0s, scales, eps_gts, eps0s, raw_eps0s = [], [], [], [], [], []

    for start in range(0, n, chunk):
        b = min(chunk, n - start)
        X, Y0, scale, eps_gt, eps0, raw_eps0 = make_batch(b, dirs, args, device)

        xs.append(X.cpu())
        y0s.append(Y0.cpu())
        scales.append(scale.cpu())
        eps_gts.append(eps_gt.cpu())
        eps0s.append(eps0.cpu())
        raw_eps0s.append(raw_eps0.cpu())

    return tuple(torch.cat(v, dim=0) for v in (xs, y0s, scales, eps_gts, eps0s, raw_eps0s))


class TaggedPointNetEpsUpdater(nn.Module):
    def __init__(self, input_mode="full"):
        super().__init__()
        assert input_mode in {"full", "xy", "xraw"}
        self.input_mode = input_mode

        self.use_y = input_mode in {"full", "xy"}
        self.use_raw_eps = input_mode in {"full", "xraw"}

        self.point = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
        )

        trunk_in = 1024 + (3 if self.use_raw_eps else 0)

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        self.head = nn.Linear(128, 3)

    def forward(self, X, Y, raw_eps):
        B, NX, _ = X.shape

        tag_X = torch.zeros(B, NX, 1, device=X.device, dtype=X.dtype)
        pts_X = torch.cat([X, tag_X], dim=-1)

        if self.use_y:
            NY = Y.shape[1]
            tag_Y = torch.ones(B, NY, 1, device=Y.device, dtype=Y.dtype)
            pts_Y = torch.cat([Y, tag_Y], dim=-1)
            pts = torch.cat([pts_X, pts_Y], dim=1)
        else:
            pts = pts_X

        h = self.point(pts)
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)

        if self.use_raw_eps:
            z = torch.cat([pooled, raw_eps], dim=-1)
        else:
            z = pooled

        delta_raw_eps = self.head(self.trunk(z))
        return delta_raw_eps


def stat(v):
    return {
        "mean": float(v.mean()),
        "median": float(v.median()),
        "p90": float(torch.quantile(v, 0.90)),
        "p95": float(torch.quantile(v, 0.95)),
        "max": float(v.max()),
    }


@torch.no_grad()
def evaluate(model, val_data, dirs, args, device):
    model.eval()

    X_all, Y0_all, scale_all, eps_gt_all, eps0_all, raw_eps0_all = val_data

    cd_steps = [[] for _ in range(args.eval_steps + 1)]
    eps_steps = [[] for _ in range(args.eval_steps + 1)]
    delta_steps = [[] for _ in range(args.eval_steps)]

    n = X_all.shape[0]

    for start in range(0, n, args.batch_size):
        sl = slice(start, min(n, start + args.batch_size))

        X = X_all[sl].to(device)
        Y = Y0_all[sl].to(device)
        scale = scale_all[sl].to(device)
        eps_gt = eps_gt_all[sl].to(device)
        eps = eps0_all[sl].to(device)
        raw_eps = raw_eps0_all[sl].to(device)

        for k in range(args.eval_steps + 1):
            cd = chamfer(Y, X)
            eps_l1 = (eps - eps_gt).abs().mean(dim=1)

            cd_steps[k].append(cd.cpu())
            eps_steps[k].append(eps_l1.cpu())

            if k == args.eval_steps:
                break

            delta_raw_eps = model(X, Y, raw_eps)
            delta_steps[k].append(delta_raw_eps.abs().mean(dim=1).cpu())

            raw_eps = raw_eps + delta_raw_eps
            eps = bounded(raw_eps, args.exp_min, args.exp_max)
            Y = sample_generalized_surface(scale, eps, dirs)

    cd_steps = [torch.cat(v, dim=0) for v in cd_steps]
    eps_steps = [torch.cat(v, dim=0) for v in eps_steps]
    delta_steps = [torch.cat(v, dim=0) for v in delta_steps]

    out = {}

    for k in range(args.eval_steps + 1):
        out[f"cd_s{k}"] = stat(cd_steps[k])
        out[f"eps_s{k}"] = stat(eps_steps[k])

        if k > 0:
            out[f"cd_improved_s{k}"] = float((cd_steps[k] < cd_steps[0]).float().mean())
            out[f"eps_improved_s{k}"] = float((eps_steps[k] < eps_steps[0]).float().mean())

    for k in range(args.eval_steps):
        out[f"delta_abs_s{k+1}"] = stat(delta_steps[k])

    return out


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
    ap.add_argument("--exp-min", type=float, default=0.25)
    ap.add_argument("--exp-max", type=float, default=8.0)

    ap.add_argument("--eval-steps", type=int, default=5)
    ap.add_argument("--input-mode", choices=["full", "xy", "xraw"], default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"args={vars(args)}", flush=True)

    dirs = fibonacci_sphere(args.points, device)

    print("Generating fixed validation data...", flush=True)
    val_data = make_fixed_dataset(args.val_n, dirs, args, device)

    model = TaggedPointNetEpsUpdater(input_mode=args.input_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(args.train_n / args.batch_size)

    best_cd_s1 = float("inf")
    best_eps_s1 = float("inf")
    best_val = None
    best_epoch_cd = None
    best_epoch_eps = None

    fields = [
        "epoch",
        "train_cd",
        "val_cd_s0_mean",
        "val_cd_s1_mean",
        f"val_cd_s{args.eval_steps}_mean",
        "val_eps_s0_mean",
        "val_eps_s1_mean",
        f"val_eps_s{args.eval_steps}_mean",
        "val_cd_improved_s1",
        "val_eps_improved_s1",
        "delta_abs_s1_mean",
    ]

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total_cd = 0.0
            seen = 0

            for _ in range(steps_per_epoch):
                b = min(args.batch_size, args.train_n - seen)
                if b <= 0:
                    break

                X, Y0, scale, eps_gt, eps0, raw_eps0 = make_batch(b, dirs, args, device)

                delta_raw_eps = model(X, Y0, raw_eps0)
                raw_eps1 = raw_eps0 + delta_raw_eps
                eps1 = bounded(raw_eps1, args.exp_min, args.exp_max)

                Y1 = sample_generalized_surface(scale, eps1, dirs)
                loss = chamfer(Y1, X).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                opt.step()

                total_cd += float(loss.detach()) * b
                seen += b

            train_cd = total_cd / max(1, seen)

            val = evaluate(model, val_data, dirs, args, device)

            row = {
                "epoch": ep,
                "train_cd": train_cd,
                "val_cd_s0_mean": val["cd_s0"]["mean"],
                "val_cd_s1_mean": val["cd_s1"]["mean"],
                f"val_cd_s{args.eval_steps}_mean": val[f"cd_s{args.eval_steps}"]["mean"],
                "val_eps_s0_mean": val["eps_s0"]["mean"],
                "val_eps_s1_mean": val["eps_s1"]["mean"],
                f"val_eps_s{args.eval_steps}_mean": val[f"eps_s{args.eval_steps}"]["mean"],
                "val_cd_improved_s1": val["cd_improved_s1"],
                "val_eps_improved_s1": val["eps_improved_s1"],
                "delta_abs_s1_mean": val["delta_abs_s1"]["mean"],
            }

            writer.writerow(row)
            f.flush()

            if row["val_cd_s1_mean"] < best_cd_s1:
                best_cd_s1 = row["val_cd_s1_mean"]
                best_epoch_cd = ep
                best_val = val
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "epoch": ep, "metric": best_cd_s1},
                    out / "best_cd.pt",
                )

            if row["val_eps_s1_mean"] < best_eps_s1:
                best_eps_s1 = row["val_eps_s1_mean"]
                best_epoch_eps = ep
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "epoch": ep, "metric": best_eps_s1},
                    out / "best_eps.pt",
                )

            print(
                f"ep={ep:03d} train_cd={train_cd:.8f} "
                f"cd s0/s1/s{args.eval_steps}="
                f"{row['val_cd_s0_mean']:.6f}/{row['val_cd_s1_mean']:.6f}/{row[f'val_cd_s{args.eval_steps}_mean']:.6f} "
                f"eps s0/s1/s{args.eval_steps}="
                f"{row['val_eps_s0_mean']:.4f}/{row['val_eps_s1_mean']:.4f}/{row[f'val_eps_s{args.eval_steps}_mean']:.4f} "
                f"imp_cd1={row['val_cd_improved_s1']:.3f} "
                f"imp_eps1={row['val_eps_improved_s1']:.3f} "
                f"|delta_raw|={row['delta_abs_s1_mean']:.3f}",
                flush=True,
            )

    lines = [
        f"best_cd_epoch={best_epoch_cd} best_cd_s1={best_cd_s1:.8f}",
        f"best_eps_epoch={best_epoch_eps} best_eps_s1={best_eps_s1:.8f}",
        "",
        "Best-CD checkpoint validation summary:",
    ]

    for k in range(args.eval_steps + 1):
        cd = best_val[f"cd_s{k}"]
        epv = best_val[f"eps_s{k}"]

        lines.append(
            f"step {k}: "
            f"cd mean/med/p90/p95/max = {cd['mean']:.8f} {cd['median']:.8f} {cd['p90']:.8f} {cd['p95']:.8f} {cd['max']:.8f} | "
            f"eps_l1 mean/med/p90/p95/max = {epv['mean']:.5f} {epv['median']:.5f} {epv['p90']:.5f} {epv['p95']:.5f} {epv['max']:.5f}"
        )

        if k > 0:
            lines.append(
                f"        improved vs step0: "
                f"cd={best_val[f'cd_improved_s{k}']:.3f}, "
                f"eps={best_val[f'eps_improved_s{k}']:.3f}"
            )

    for k in range(args.eval_steps):
        d = best_val[f"delta_abs_s{k+1}"]
        lines.append(
            f"delta step {k+1}: |delta_raw_eps| mean/p95/max = "
            f"{d['mean']:.5f} {d['p95']:.5f} {d['max']:.5f}"
        )

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
