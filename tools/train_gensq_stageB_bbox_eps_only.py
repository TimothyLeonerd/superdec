#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn

from tools.train_gensq_eps_bboxnorm_rayaligned import (
    set_seed,
    fibonacci_sphere,
    exp_to_raw,
    raw_to_exp,
    summarize_vec,
)
from tools.train_gensq_stageA_bbox_eps_scale_eps import (
    EpsNet,
    make_batch,
    eps_ray_loss,
    full_cd_for_state,
    metric_line,
)


@torch.no_grad()
def eval_model(args, eps_net, val_data, dirs_base):
    X, scale_gt, eps_gt, scale0 = val_data
    log_scale0 = scale0.log()

    raw_eps0 = exp_to_raw(
        torch.full_like(eps_gt, args.eps0),
        args.exp_min,
        args.exp_max,
    )

    X0_norm = X / scale0[:, None, :].clamp_min(1e-8)
    raw_eps1 = eps_net(X0_norm, log_scale0)

    states = {
        0: (scale0, raw_eps0),
        1: (scale0, raw_eps1),
    }

    out = {}
    for step, (sc, re) in states.items():
        cd = full_cd_for_state(args, X, sc, re, dirs_base)
        eps = raw_to_exp(re, args.exp_min, args.exp_max)
        eps_l1 = (eps - eps_gt).abs().mean(dim=1)
        scale_l1 = (sc - scale_gt).abs().mean(dim=1)

        out[step] = {
            "cd": summarize_vec(cd),
            "eps": summarize_vec(eps_l1),
            "scale": summarize_vec(scale_l1),
        }

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
    ap.add_argument("--exp-min", type=float, default=0.5)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--eps0", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs_base = fibonacci_sphere(args.points, device)

    eps_net = EpsNet().to(device)
    opt = torch.optim.AdamW(
        eps_net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    val_data = make_batch(args, args.val_n, dirs_base, device)

    metrics_path = out / "metrics.csv"

    best_cd = (float("inf"), -1, None)
    best_eps = (float("inf"), -1, None)

    with metrics_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss", "step", "cd_mean", "scale_l1_mean", "eps_l1_mean"])

        for ep in range(1, args.epochs + 1):
            eps_net.train()
            losses = []
            n_batches = math.ceil(args.train_n / args.batch_size)

            for _ in range(n_batches):
                X, scale_gt, eps_gt, scale0 = make_batch(args, args.batch_size, dirs_base, device)
                log_scale0 = scale0.log()

                X0_norm = X / scale0[:, None, :].clamp_min(1e-8)
                raw_eps1 = eps_net(X0_norm, log_scale0)

                # The only training loss.
                # scale0 is fixed/detached; only EpsNet gets gradients.
                loss = eps_ray_loss(
                    args,
                    X,
                    scale0.detach(),
                    raw_eps1,
                    pred_normalize=True,
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(eps_net.parameters(), args.grad_clip)
                opt.step()

                losses.append(loss.item())

            eps_net.eval()
            with torch.no_grad():
                ev = eval_model(args, eps_net, val_data, dirs_base)

            train_loss = sum(losses) / len(losses)
            for step in [0, 1]:
                wr.writerow([
                    ep,
                    train_loss,
                    step,
                    ev[step]["cd"][0],
                    ev[step]["scale"][0],
                    ev[step]["eps"][0],
                ])
            f.flush()

            cd1 = ev[1]["cd"][0]
            eps1 = ev[1]["eps"][0]

            if cd1 < best_cd[0]:
                best_cd = (cd1, ep, ev)
            if eps1 < best_eps[0]:
                best_eps = (eps1, ep, ev)

            print(
                f"ep={ep:03d} train={train_loss:.8f} "
                f"s1 cd/scale/eps={cd1:.8f}/{ev[1]['scale'][0]:.8f}/{eps1:.8f}",
                flush=True,
            )

    lines = []
    lines.append("experiment=stageB_bbox_eps_only")
    lines.append(f"exp_min={args.exp_min}")
    lines.append("scale0=bbox_absmax_frozen")
    lines.append("loss=ray_aligned_bbox_normalized_eps_loss_prednorm_true")
    lines.append(f"best_cd_epoch={best_cd[1]} best_cd_s1={best_cd[0]:.8f}")
    lines.append(f"best_eps_epoch={best_eps[1]} best_eps_s1={best_eps[0]:.8f}")

    for tag, (_, ep, ev) in [("best_cd", best_cd), ("best_eps", best_eps)]:
        lines.append("")
        lines.append(f"{tag}_epoch={ep}")
        for step in [0, 1]:
            lines.append(
                f"step {step}: "
                f"{metric_line('cd', ev[step]['cd'])} | "
                f"{metric_line('scale_l1', ev[step]['scale'])} | "
                f"{metric_line('eps_l1', ev[step]['eps'])}"
            )

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("=== summary ===")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
