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
    scale_gt = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(batch_size, 3, device=device)
    eps_gt = sample_log_uniform(args.exp_min, args.exp_max, (batch_size, 3), device)

    X = sample_generalized_surface(scale_gt, eps_gt, dirs)

    # Initial state: bbox scale, ellipsoid eps.
    scale0 = X.abs().amax(dim=1)
    log_scale0 = torch.log(scale0.clamp_min(1e-8))

    eps0 = torch.full_like(eps_gt, args.eps0)
    raw_eps0 = inv_bounded(eps0, args.exp_min, args.exp_max)

    return X, scale_gt, eps_gt, log_scale0, raw_eps0


@torch.no_grad()
def make_fixed_dataset(n, dirs, args, device, chunk=512):
    xs, scale_gts, eps_gts, log_scale0s, raw_eps0s = [], [], [], [], []
    for start in range(0, n, chunk):
        b = min(chunk, n - start)
        X, scale_gt, eps_gt, log_scale0, raw_eps0 = make_batch(b, dirs, args, device)
        xs.append(X.cpu())
        scale_gts.append(scale_gt.cpu())
        eps_gts.append(eps_gt.cpu())
        log_scale0s.append(log_scale0.cpu())
        raw_eps0s.append(raw_eps0.cpu())

    return tuple(torch.cat(v, dim=0) for v in (xs, scale_gts, eps_gts, log_scale0s, raw_eps0s))


class XParamUpdater(nn.Module):
    def __init__(self, out_dim=3):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1024 + 6, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, out_dim)

    def forward(self, X, log_scale, raw_eps):
        B, N, _ = X.shape
        tag = torch.zeros(B, N, 1, device=X.device, dtype=X.dtype)
        pts = torch.cat([X, tag], dim=-1)

        h = self.point(pts)
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
        z = torch.cat([pooled, log_scale, raw_eps], dim=-1)

        return self.head(self.trunk(z))


def stat(v):
    return {
        "mean": float(v.mean()),
        "median": float(v.median()),
        "p90": float(torch.quantile(v, 0.90)),
        "p95": float(torch.quantile(v, 0.95)),
        "max": float(v.max()),
    }


@torch.no_grad()
def eval_model(model_scale, model_eps, val_data, dirs, args, device):
    model_scale.eval()
    model_eps.eval()

    X_all, scale_gt_all, eps_gt_all, log_scale0_all, raw_eps0_all = val_data
    n = X_all.shape[0]

    cd_init, scale_init, eps_init = [], [], []
    cd_scale_steps = [[] for _ in range(args.n_loop)]
    cd_eps_steps = [[] for _ in range(args.n_loop)]
    scale_steps = [[] for _ in range(args.n_loop)]
    eps_steps = [[] for _ in range(args.n_loop)]
    dscale_steps = [[] for _ in range(args.n_loop)]
    deps_steps = [[] for _ in range(args.n_loop)]

    for start in range(0, n, args.batch_size):
        sl = slice(start, min(n, start + args.batch_size))

        X = X_all[sl].to(device)
        scale_gt = scale_gt_all[sl].to(device)
        eps_gt = eps_gt_all[sl].to(device)
        log_scale = log_scale0_all[sl].to(device)
        raw_eps = raw_eps0_all[sl].to(device)

        scale = torch.exp(log_scale)
        eps = bounded(raw_eps, args.exp_min, args.exp_max)
        Y = sample_generalized_surface(scale, eps, dirs)

        cd_init.append(chamfer(Y, X).cpu())
        scale_init.append((scale - scale_gt).abs().mean(dim=1).cpu())
        eps_init.append((eps - eps_gt).abs().mean(dim=1).cpu())

        for i in range(args.n_loop):
            eps_fixed = bounded(raw_eps, args.exp_min, args.exp_max).detach()

            dlog_scale = model_scale(X, log_scale, raw_eps)
            log_scale = (log_scale + dlog_scale).detach()
            scale = torch.exp(log_scale)

            Y_scale = sample_generalized_surface(scale, eps_fixed, dirs)
            cd_scale_steps[i].append(chamfer(Y_scale, X).cpu())
            scale_steps[i].append((scale - scale_gt).abs().mean(dim=1).cpu())
            eps_steps[i].append((eps_fixed - eps_gt).abs().mean(dim=1).cpu())
            dscale_steps[i].append(dlog_scale.abs().mean(dim=1).cpu())

            scale_fixed = scale.detach()

            draw_eps = model_eps(X, log_scale, raw_eps)
            raw_eps = (raw_eps + draw_eps).detach()
            eps = bounded(raw_eps, args.exp_min, args.exp_max)

            Y_eps = sample_generalized_surface(scale_fixed, eps, dirs)
            cd_eps_steps[i].append(chamfer(Y_eps, X).cpu())
            scale_steps[i].append((scale_fixed - scale_gt).abs().mean(dim=1).cpu())
            eps_steps[i].append((eps - eps_gt).abs().mean(dim=1).cpu())
            deps_steps[i].append(draw_eps.abs().mean(dim=1).cpu())

    out = {
        "init_cd": stat(torch.cat(cd_init)),
        "init_scale": stat(torch.cat(scale_init)),
        "init_eps": stat(torch.cat(eps_init)),
    }

    for i in range(args.n_loop):
        out[f"scale_step_{i+1}_cd"] = stat(torch.cat(cd_scale_steps[i]))
        out[f"eps_step_{i+1}_cd"] = stat(torch.cat(cd_eps_steps[i]))

        # For reporting, scale_steps/eps_steps have both after-scale and after-eps entries.
        out[f"after_scale_{i+1}_scale_l1"] = stat(torch.cat(scale_steps[i][0::2]))
        out[f"after_scale_{i+1}_eps_l1"] = stat(torch.cat(eps_steps[i][0::2]))
        out[f"after_eps_{i+1}_scale_l1"] = stat(torch.cat(scale_steps[i][1::2]))
        out[f"after_eps_{i+1}_eps_l1"] = stat(torch.cat(eps_steps[i][1::2]))

        out[f"delta_log_scale_{i+1}"] = stat(torch.cat(dscale_steps[i]))
        out[f"delta_raw_eps_{i+1}"] = stat(torch.cat(deps_steps[i]))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--train-n", type=int, default=50000)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-loop", type=int, default=4)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=0.25)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--eps0", type=float, default=2.0)

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

    model_scale = XParamUpdater(out_dim=3).to(device)
    model_eps = XParamUpdater(out_dim=3).to(device)

    opt_scale = torch.optim.AdamW(model_scale.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_eps = torch.optim.AdamW(model_eps.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(args.train_n / args.batch_size)

    fields = [
        "epoch",
        "train_scale_cd_last",
        "train_eps_cd_last",
        "val_init_cd",
        "val_init_scale_l1",
        "val_init_eps_l1",
    ]
    for i in range(args.n_loop):
        fields += [
            f"val_scale{i+1}_cd",
            f"val_eps{i+1}_cd",
            f"val_scale{i+1}_scale_l1",
            f"val_scale{i+1}_eps_l1",
            f"val_eps{i+1}_scale_l1",
            f"val_eps{i+1}_eps_l1",
            f"val_dlog_scale{i+1}",
            f"val_draw_eps{i+1}",
        ]

    best_final_cd = float("inf")
    best_final_eps = float("inf")
    best_cd_epoch = None
    best_eps_epoch = None
    best_cd_val = None
    best_eps_val = None

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model_scale.train()
            model_eps.train()

            total_scale_last = 0.0
            total_eps_last = 0.0
            seen = 0

            for _ in range(steps_per_epoch):
                b = min(args.batch_size, args.train_n - seen)
                if b <= 0:
                    break

                X, scale_gt, eps_gt, log_scale, raw_eps = make_batch(b, dirs, args, device)
                log_scale = log_scale.detach()
                raw_eps = raw_eps.detach()

                loss_scale_val = None
                loss_eps_val = None

                for i in range(args.n_loop):
                    # ---- scale block ----
                    eps_fixed = bounded(raw_eps, args.exp_min, args.exp_max).detach()

                    dlog_scale = model_scale(X, log_scale, raw_eps)
                    log_scale_next = log_scale + dlog_scale
                    scale_next = torch.exp(log_scale_next)

                    Y_scale = sample_generalized_surface(scale_next, eps_fixed, dirs)
                    loss_scale = chamfer(Y_scale, X).mean()

                    opt_scale.zero_grad(set_to_none=True)
                    loss_scale.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model_scale.parameters(), args.grad_clip)
                    opt_scale.step()

                    log_scale = log_scale_next.detach()
                    loss_scale_val = float(loss_scale.detach())

                    # ---- eps block ----
                    scale_fixed = torch.exp(log_scale).detach()

                    draw_eps = model_eps(X, log_scale, raw_eps)
                    raw_eps_next = raw_eps + draw_eps
                    eps_next = bounded(raw_eps_next, args.exp_min, args.exp_max)

                    Y_eps = sample_generalized_surface(scale_fixed, eps_next, dirs)
                    loss_eps = chamfer(Y_eps, X).mean()

                    opt_eps.zero_grad(set_to_none=True)
                    loss_eps.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model_eps.parameters(), args.grad_clip)
                    opt_eps.step()

                    raw_eps = raw_eps_next.detach()
                    loss_eps_val = float(loss_eps.detach())

                total_scale_last += loss_scale_val * b
                total_eps_last += loss_eps_val * b
                seen += b

            val = eval_model(model_scale, model_eps, val_data, dirs, args, device)

            row = {
                "epoch": ep,
                "train_scale_cd_last": total_scale_last / max(1, seen),
                "train_eps_cd_last": total_eps_last / max(1, seen),
                "val_init_cd": val["init_cd"]["mean"],
                "val_init_scale_l1": val["init_scale"]["mean"],
                "val_init_eps_l1": val["init_eps"]["mean"],
            }

            for i in range(args.n_loop):
                row.update({
                    f"val_scale{i+1}_cd": val[f"scale_step_{i+1}_cd"]["mean"],
                    f"val_eps{i+1}_cd": val[f"eps_step_{i+1}_cd"]["mean"],
                    f"val_scale{i+1}_scale_l1": val[f"after_scale_{i+1}_scale_l1"]["mean"],
                    f"val_scale{i+1}_eps_l1": val[f"after_scale_{i+1}_eps_l1"]["mean"],
                    f"val_eps{i+1}_scale_l1": val[f"after_eps_{i+1}_scale_l1"]["mean"],
                    f"val_eps{i+1}_eps_l1": val[f"after_eps_{i+1}_eps_l1"]["mean"],
                    f"val_dlog_scale{i+1}": val[f"delta_log_scale_{i+1}"]["mean"],
                    f"val_draw_eps{i+1}": val[f"delta_raw_eps_{i+1}"]["mean"],
                })

            writer.writerow(row)
            f.flush()

            final_cd = row[f"val_eps{args.n_loop}_cd"]
            final_eps = row[f"val_eps{args.n_loop}_eps_l1"]

            if final_cd < best_final_cd:
                best_final_cd = final_cd
                best_cd_epoch = ep
                best_cd_val = val
                torch.save(
                    {
                        "model_scale": model_scale.state_dict(),
                        "model_eps": model_eps.state_dict(),
                        "args": vars(args),
                        "epoch": ep,
                    },
                    out / "best_cd.pt",
                )

            if final_eps < best_final_eps:
                best_final_eps = final_eps
                best_eps_epoch = ep
                best_eps_val = val
                torch.save(
                    {
                        "model_scale": model_scale.state_dict(),
                        "model_eps": model_eps.state_dict(),
                        "args": vars(args),
                        "epoch": ep,
                    },
                    out / "best_eps.pt",
                )

            print(
                f"ep={ep:03d} "
                f"train last scale/eps cd={row['train_scale_cd_last']:.6f}/{row['train_eps_cd_last']:.6f} "
                f"init cd/sc/eps={row['val_init_cd']:.6f}/{row['val_init_scale_l1']:.4f}/{row['val_init_eps_l1']:.4f} "
                f"final cd/sc/eps={row[f'val_eps{args.n_loop}_cd']:.6f}/"
                f"{row[f'val_eps{args.n_loop}_scale_l1']:.4f}/"
                f"{row[f'val_eps{args.n_loop}_eps_l1']:.4f} "
                f"dscale1={row['val_dlog_scale1']:.3f} deps1={row['val_draw_eps1']:.3f}",
                flush=True,
            )

    def write_summary(name, epoch, val):
        lines = [f"{name}_epoch={epoch}"]
        lines.append(
            f"init: cd mean/p95/max = {val['init_cd']['mean']:.8f} {val['init_cd']['p95']:.8f} {val['init_cd']['max']:.8f} | "
            f"scale_l1 mean/p95/max = {val['init_scale']['mean']:.5f} {val['init_scale']['p95']:.5f} {val['init_scale']['max']:.5f} | "
            f"eps_l1 mean/p95/max = {val['init_eps']['mean']:.5f} {val['init_eps']['p95']:.5f} {val['init_eps']['max']:.5f}"
        )

        for i in range(args.n_loop):
            sc_cd = val[f"scale_step_{i+1}_cd"]
            ep_cd = val[f"eps_step_{i+1}_cd"]
            sc_l1 = val[f"after_scale_{i+1}_scale_l1"]
            ep_l1_sc = val[f"after_scale_{i+1}_eps_l1"]
            sc_l1_ep = val[f"after_eps_{i+1}_scale_l1"]
            ep_l1 = val[f"after_eps_{i+1}_eps_l1"]
            ds = val[f"delta_log_scale_{i+1}"]
            de = val[f"delta_raw_eps_{i+1}"]

            lines.append(
                f"scale {i+1}: cd mean/p95/max = {sc_cd['mean']:.8f} {sc_cd['p95']:.8f} {sc_cd['max']:.8f} | "
                f"scale_l1 mean/p95/max = {sc_l1['mean']:.5f} {sc_l1['p95']:.5f} {sc_l1['max']:.5f} | "
                f"eps_l1 mean/p95/max = {ep_l1_sc['mean']:.5f} {ep_l1_sc['p95']:.5f} {ep_l1_sc['max']:.5f} | "
                f"|dlog_scale| mean/p95/max = {ds['mean']:.5f} {ds['p95']:.5f} {ds['max']:.5f}"
            )
            lines.append(
                f"eps   {i+1}: cd mean/p95/max = {ep_cd['mean']:.8f} {ep_cd['p95']:.8f} {ep_cd['max']:.8f} | "
                f"scale_l1 mean/p95/max = {sc_l1_ep['mean']:.5f} {sc_l1_ep['p95']:.5f} {sc_l1_ep['max']:.5f} | "
                f"eps_l1 mean/p95/max = {ep_l1['mean']:.5f} {ep_l1['p95']:.5f} {ep_l1['max']:.5f} | "
                f"|draw_eps| mean/p95/max = {de['mean']:.5f} {de['p95']:.5f} {de['max']:.5f}"
            )
        return lines

    lines = [
        f"best_cd_epoch={best_cd_epoch} best_final_cd={best_final_cd:.8f}",
        f"best_eps_epoch={best_eps_epoch} best_final_eps={best_final_eps:.8f}",
        "",
        "Best-CD checkpoint:",
    ]
    lines += write_summary("best_cd", best_cd_epoch, best_cd_val)
    lines += ["", "Best-EPS checkpoint:"]
    lines += write_summary("best_eps", best_eps_epoch, best_eps_val)

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
