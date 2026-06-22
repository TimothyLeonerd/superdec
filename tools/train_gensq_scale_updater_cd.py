#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

import torch
import torch.nn as nn

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
    chamfer,
)


def sample_log_uniform(lo, hi, shape, device):
    return torch.exp(
        math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(shape, device=device)
    )


@torch.no_grad()
def make_batch(batch_size, dirs, args, device):
    scale_gt = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(batch_size, 3, device=device)
    eps_gt = sample_log_uniform(args.exp_min, args.exp_max, (batch_size, 3), device)

    X = sample_generalized_surface(scale_gt, eps_gt, dirs)

    if args.init_mode == "bbox":
        scale0 = X.abs().amax(dim=1).clamp_min(1e-8)
    elif args.init_mode == "random":
        scale0 = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(batch_size, 3, device=device)
    else:
        raise ValueError(args.init_mode)

    log_scale0 = torch.log(scale0)

    return X, scale_gt, eps_gt, log_scale0


@torch.no_grad()
def make_fixed_dataset(n, dirs, args, device, chunk=512):
    xs, scale_gts, eps_gts, log_scale0s = [], [], [], []

    for start in range(0, n, chunk):
        b = min(chunk, n - start)
        X, scale_gt, eps_gt, log_scale0 = make_batch(b, dirs, args, device)

        xs.append(X.cpu())
        scale_gts.append(scale_gt.cpu())
        eps_gts.append(eps_gt.cpu())
        log_scale0s.append(log_scale0.cpu())

    return tuple(torch.cat(v, dim=0) for v in (xs, scale_gts, eps_gts, log_scale0s))


class XScaleUpdater(nn.Module):
    def __init__(self):
        super().__init__()

        self.point = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(1024 + 3, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        self.head = nn.Linear(128, 3)

    def forward(self, X, log_scale):
        B, N, _ = X.shape
        tag = torch.zeros(B, N, 1, device=X.device, dtype=X.dtype)
        pts = torch.cat([X, tag], dim=-1)

        h = self.point(pts)
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
        z = torch.cat([pooled, log_scale], dim=-1)

        delta_log_scale = self.head(self.trunk(z))
        return delta_log_scale


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

    X_all, scale_gt_all, eps_gt_all, log_scale0_all = val_data

    cd_steps = [[] for _ in range(args.eval_steps + 1)]
    scale_steps = [[] for _ in range(args.eval_steps + 1)]
    delta_steps = [[] for _ in range(args.eval_steps)]

    n = X_all.shape[0]

    for start in range(0, n, args.batch_size):
        sl = slice(start, min(n, start + args.batch_size))

        X = X_all[sl].to(device)
        scale_gt = scale_gt_all[sl].to(device)
        eps_gt = eps_gt_all[sl].to(device)
        log_scale = log_scale0_all[sl].to(device)

        for k in range(args.eval_steps + 1):
            scale = torch.exp(log_scale)
            Y = sample_generalized_surface(scale, eps_gt, dirs)

            cd_steps[k].append(chamfer(Y, X).cpu())
            scale_steps[k].append((scale - scale_gt).abs().mean(dim=1).cpu())

            if k == args.eval_steps:
                break

            delta_log_scale = model(X, log_scale)
            delta_steps[k].append(delta_log_scale.abs().mean(dim=1).cpu())

            log_scale = log_scale + delta_log_scale

    out = {}

    cd_steps = [torch.cat(v, dim=0) for v in cd_steps]
    scale_steps = [torch.cat(v, dim=0) for v in scale_steps]
    delta_steps = [torch.cat(v, dim=0) for v in delta_steps]

    for k in range(args.eval_steps + 1):
        out[f"cd_s{k}"] = stat(cd_steps[k])
        out[f"scale_s{k}"] = stat(scale_steps[k])
        if k > 0:
            out[f"cd_improved_s{k}"] = float((cd_steps[k] < cd_steps[0]).float().mean())
            out[f"scale_improved_s{k}"] = float((scale_steps[k] < scale_steps[0]).float().mean())

    for k in range(args.eval_steps):
        out[f"delta_s{k+1}"] = stat(delta_steps[k])

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init-mode", choices=["bbox", "random"], required=True)

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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs = fibonacci_sphere(args.points, device)

    print(f"device={device}", flush=True)
    print(f"args={vars(args)}", flush=True)

    print("Generating fixed validation data...", flush=True)
    val_data = make_fixed_dataset(args.val_n, dirs, args, device)

    model = XScaleUpdater().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(args.train_n / args.batch_size)

    fields = [
        "epoch",
        "train_cd",
        "val_cd_s0_mean",
        "val_cd_s1_mean",
        f"val_cd_s{args.eval_steps}_mean",
        "val_scale_s0_mean",
        "val_scale_s1_mean",
        f"val_scale_s{args.eval_steps}_mean",
        "val_cd_improved_s1",
        "val_scale_improved_s1",
        "delta_s1_mean",
    ]

    best_cd_s1 = float("inf")
    best_scale_s1 = float("inf")
    best_cd_val = None
    best_scale_val = None
    best_cd_epoch = None
    best_scale_epoch = None

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

                X, scale_gt, eps_gt, log_scale0 = make_batch(b, dirs, args, device)

                delta_log_scale = model(X, log_scale0)
                log_scale1 = log_scale0 + delta_log_scale
                scale1 = torch.exp(log_scale1)

                Y1 = sample_generalized_surface(scale1, eps_gt, dirs)
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
                "val_scale_s0_mean": val["scale_s0"]["mean"],
                "val_scale_s1_mean": val["scale_s1"]["mean"],
                f"val_scale_s{args.eval_steps}_mean": val[f"scale_s{args.eval_steps}"]["mean"],
                "val_cd_improved_s1": val["cd_improved_s1"],
                "val_scale_improved_s1": val["scale_improved_s1"],
                "delta_s1_mean": val["delta_s1"]["mean"],
            }

            writer.writerow(row)
            f.flush()

            if row["val_cd_s1_mean"] < best_cd_s1:
                best_cd_s1 = row["val_cd_s1_mean"]
                best_cd_epoch = ep
                best_cd_val = val
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, out / "best_cd.pt")

            if row["val_scale_s1_mean"] < best_scale_s1:
                best_scale_s1 = row["val_scale_s1_mean"]
                best_scale_epoch = ep
                best_scale_val = val
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, out / "best_scale.pt")

            print(
                f"ep={ep:03d} train_cd={train_cd:.8f} "
                f"cd s0/s1/s{args.eval_steps}="
                f"{row['val_cd_s0_mean']:.6f}/{row['val_cd_s1_mean']:.6f}/{row[f'val_cd_s{args.eval_steps}_mean']:.6f} "
                f"scale s0/s1/s{args.eval_steps}="
                f"{row['val_scale_s0_mean']:.5f}/{row['val_scale_s1_mean']:.5f}/{row[f'val_scale_s{args.eval_steps}_mean']:.5f} "
                f"imp_cd1={row['val_cd_improved_s1']:.3f} "
                f"imp_scale1={row['val_scale_improved_s1']:.3f} "
                f"|dlog_scale|={row['delta_s1_mean']:.3f}",
                flush=True,
            )

    def summary_block(name, epoch, val):
        lines = [f"{name}_epoch={epoch}"]
        for k in range(args.eval_steps + 1):
            cd = val[f"cd_s{k}"]
            sc = val[f"scale_s{k}"]
            lines.append(
                f"step {k}: cd mean/med/p90/p95/max = "
                f"{cd['mean']:.8f} {cd['median']:.8f} {cd['p90']:.8f} {cd['p95']:.8f} {cd['max']:.8f} | "
                f"scale_l1 mean/med/p90/p95/max = "
                f"{sc['mean']:.5f} {sc['median']:.5f} {sc['p90']:.5f} {sc['p95']:.5f} {sc['max']:.5f}"
            )
            if k > 0:
                lines.append(
                    f"        improved vs step0: cd={val[f'cd_improved_s{k}']:.3f}, "
                    f"scale={val[f'scale_improved_s{k}']:.3f}"
                )
        for k in range(args.eval_steps):
            d = val[f"delta_s{k+1}"]
            lines.append(
                f"delta step {k+1}: |delta_log_scale| mean/p95/max = "
                f"{d['mean']:.5f} {d['p95']:.5f} {d['max']:.5f}"
            )
        return lines

    lines = [
        f"init_mode={args.init_mode}",
        f"best_cd_epoch={best_cd_epoch} best_cd_s1={best_cd_s1:.8f}",
        f"best_scale_epoch={best_scale_epoch} best_scale_s1={best_scale_s1:.8f}",
        "",
        "Best-CD checkpoint:",
    ]
    lines += summary_block("best_cd", best_cd_epoch, best_cd_val)
    lines += ["", "Best-SCALE checkpoint:"]
    lines += summary_block("best_scale", best_scale_epoch, best_scale_val)

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
