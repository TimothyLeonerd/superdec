#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.train_gensq_eps_bboxnorm_rayaligned import (
    set_seed,
    fibonacci_sphere,
    sample_log_uniform,
    exp_to_raw,
    raw_to_exp,
    sample_gensq_batched_dirs,
    chamfer,
    normalize_points,
    summarize_vec,
)


class EpsNet(nn.Module):
    """
    Absolute eps predictor:
      input per point = [X_norm(3), log_scale_est(3)]
      output = raw_eps(3)
    """
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

    def forward(self, X_norm, log_scale):
        B, S, _ = X_norm.shape
        ls = log_scale[:, None, :].expand(B, S, 3)
        feat = torch.cat([X_norm, ls], dim=-1)
        h = self.point(feat).amax(dim=1)
        return self.head(h)


class ScaleNet(nn.Module):
    """
    Scale updater:
      input per point = [X(3), log_scale0(3), raw_eps1(3)]
      output = delta_log_scale(3)
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
        ls = log_scale[:, None, :].expand(B, S, 3)
        re = raw_eps[:, None, :].expand(B, S, 3)
        feat = torch.cat([X, ls, re], dim=-1)
        h = self.point(feat).amax(dim=1)
        return self.head(h)


def bbox_scale_absmax(X):
    return X.abs().amax(dim=1).clamp_min(1e-8)


def make_batch(args, B, dirs_base, device):
    S = dirs_base.shape[0]
    scale_gt = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    eps_gt = sample_log_uniform(args.exp_min, args.exp_max, (B, 3), device)

    dirs = dirs_base[None, :, :].expand(B, S, 3)
    X = sample_gensq_batched_dirs(scale_gt, eps_gt, dirs)

    scale0 = bbox_scale_absmax(X)

    return X, scale_gt, eps_gt, scale0


def eps_ray_loss(args, X, scale_est, raw_eps_pred, pred_normalize):
    """
    Ray-aligned normalized eps loss.
    """
    B = X.shape[0]
    device = X.device

    X_norm = X / scale_est[:, None, :].clamp_min(1e-8)
    dirs_norm = F.normalize(X_norm, p=2, dim=-1, eps=1e-12)

    eps_pred = raw_to_exp(raw_eps_pred, args.exp_min, args.exp_max)
    unit_scale = torch.ones(B, 3, device=device)
    Y = sample_gensq_batched_dirs(unit_scale, eps_pred, dirs_norm)

    if pred_normalize:
        Y = normalize_points(Y, "absmax", gt_scale=unit_scale)

    return ((X_norm - Y) ** 2).sum(dim=-1).mean()


def full_cd_for_state(args, X, scale, raw_eps, dirs_base):
    B, S, _ = X.shape
    dirs = dirs_base[None, :, :].expand(B, S, 3)
    eps = raw_to_exp(raw_eps, args.exp_min, args.exp_max)
    Y = sample_gensq_batched_dirs(scale, eps, dirs)
    return chamfer(X, Y)


@torch.no_grad()
def eval_model(args, eps_net, scale_net, val_data, dirs_base):
    X, scale_gt, eps_gt, scale0 = val_data
    log_scale0 = scale0.log()

    raw_eps0 = exp_to_raw(
        torch.full_like(eps_gt, args.eps0),
        args.exp_min,
        args.exp_max,
    )

    # stage 0: bbox scale + constant eps0
    raw0 = raw_eps0
    scale_s0 = scale0

    # stage 1: eps from bbox-normalized cloud
    X0_norm = X / scale0[:, None, :].clamp_min(1e-8)
    raw_eps1 = eps_net(X0_norm, log_scale0)
    scale_s1 = scale0

    # stage 2: scale update conditioned on eps1
    delta_log_scale = scale_net(X, log_scale0, raw_eps1)
    log_scale1 = log_scale0 + args.scale_damp * delta_log_scale
    scale1 = log_scale1.exp().clamp_min(1e-8)
    raw_s2 = raw_eps1

    # stage 3: eps refinement using scale1-normalized cloud
    X1_norm = X / scale1[:, None, :].clamp_min(1e-8)
    raw_eps2 = eps_net(X1_norm, log_scale1)
    scale_s3 = scale1

    states = {
        0: (scale_s0, raw0),
        1: (scale_s1, raw_eps1),
        2: (scale1, raw_s2),
        3: (scale_s3, raw_eps2),
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

    out["delta"] = summarize_vec(delta_log_scale.abs().mean(dim=1))
    return out


def metric_line(name, tup):
    return f"{name} mean/med/p90/p95/max = {tup[0]:.8f} {tup[1]:.8f} {tup[2]:.8f} {tup[3]:.8f} {tup[4]:.8f}"


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
    ap.add_argument("--scale-damp", type=float, default=0.5)
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument("--loss-eps1-w", type=float, default=1.0)
    ap.add_argument("--loss-scale-w", type=float, default=1.0)
    ap.add_argument("--loss-eps2-w", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs_base = fibonacci_sphere(args.points, device)

    eps_net = EpsNet().to(device)
    scale_net = ScaleNet().to(device)

    opt = torch.optim.AdamW(
        list(eps_net.parameters()) + list(scale_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    val_data = make_batch(args, args.val_n, dirs_base, device)

    metrics_path = out / "metrics.csv"

    best_final_cd = (float("inf"), -1, None)
    best_final_eps = (float("inf"), -1, None)
    best_final_scale = (float("inf"), -1, None)

    with metrics_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "epoch", "train_loss",
            "loss_eps1", "loss_scale", "loss_eps2", "loss_scale_reg",
            "step", "cd_mean", "scale_l1_mean", "eps_l1_mean",
        ])

        for ep in range(1, args.epochs + 1):
            eps_net.train()
            scale_net.train()
            losses = []
            l_eps1s, l_scales, l_eps2s, l_regs = [], [], [], []

            n_batches = math.ceil(args.train_n / args.batch_size)

            for _ in range(n_batches):
                X, scale_gt, eps_gt, scale0 = make_batch(args, args.batch_size, dirs_base, device)
                log_scale0 = scale0.log()

                # eps1 from bbox-normalized cloud
                X0_norm = X / scale0[:, None, :].clamp_min(1e-8)
                raw_eps1 = eps_net(X0_norm, log_scale0)

                loss_eps1 = eps_ray_loss(
                    args,
                    X,
                    scale0.detach(),
                    raw_eps1,
                    pred_normalize=True,
                )

                # scale1 from X, bbox scale, eps1
                # Detach eps1 in scale loss to avoid scale/eps compensation gradients.
                delta_log_scale = scale_net(X, log_scale0.detach(), raw_eps1.detach())
                log_scale1 = log_scale0 + args.scale_damp * delta_log_scale
                scale1 = log_scale1.exp().clamp_min(1e-8)

                cd_scale = full_cd_for_state(args, X, scale1, raw_eps1.detach(), dirs_base).mean()
                scale_reg = ((log_scale1 - log_scale0) ** 2).mean()
                loss_scale = cd_scale + args.scale_reg * scale_reg

                # eps2 refinement using scale1-normalized cloud.
                # Detach scale1 so eps2 trains EpsNet, not scale through normalized target.
                X1_norm = X / scale1.detach()[:, None, :].clamp_min(1e-8)
                raw_eps2 = eps_net(X1_norm, log_scale1.detach())

                loss_eps2 = eps_ray_loss(
                    args,
                    X,
                    scale1.detach(),
                    raw_eps2,
                    pred_normalize=False,
                )

                loss = (
                    args.loss_eps1_w * loss_eps1
                    + args.loss_scale_w * loss_scale
                    + args.loss_eps2_w * loss_eps2
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(eps_net.parameters()) + list(scale_net.parameters()),
                    args.grad_clip,
                )
                opt.step()

                losses.append(loss.item())
                l_eps1s.append(loss_eps1.item())
                l_scales.append(cd_scale.item())
                l_eps2s.append(loss_eps2.item())
                l_regs.append(scale_reg.item())

            eps_net.eval()
            scale_net.eval()
            with torch.no_grad():
                ev = eval_model(args, eps_net, scale_net, val_data, dirs_base)

            train_loss = sum(losses) / len(losses)
            le1 = sum(l_eps1s) / len(l_eps1s)
            lsc = sum(l_scales) / len(l_scales)
            le2 = sum(l_eps2s) / len(l_eps2s)
            lrg = sum(l_regs) / len(l_regs)

            for step in [0, 1, 2, 3]:
                wr.writerow([
                    ep, train_loss, le1, lsc, le2, lrg,
                    step,
                    ev[step]["cd"][0],
                    ev[step]["scale"][0],
                    ev[step]["eps"][0],
                ])
            f.flush()

            final_cd = ev[3]["cd"][0]
            final_eps = ev[3]["eps"][0]
            final_scale = ev[3]["scale"][0]

            if final_cd < best_final_cd[0]:
                best_final_cd = (final_cd, ep, ev)
            if final_eps < best_final_eps[0]:
                best_final_eps = (final_eps, ep, ev)
            if final_scale < best_final_scale[0]:
                best_final_scale = (final_scale, ep, ev)

            print(
                f"ep={ep:03d} train={train_loss:.8f} "
                f"losses eps1/scale/eps2/reg={le1:.8f}/{lsc:.8f}/{le2:.8f}/{lrg:.8f} "
                f"final cd/scale/eps={final_cd:.8f}/{final_scale:.8f}/{final_eps:.8f}",
                flush=True,
            )

    lines = []
    lines.append("experiment=stageA_bbox_eps_scale_eps")
    lines.append(f"exp_min={args.exp_min}")
    lines.append(f"scale_damp={args.scale_damp}")
    lines.append(f"scale_reg={args.scale_reg}")
    lines.append(f"best_final_cd_epoch={best_final_cd[1]} best_final_cd_s3={best_final_cd[0]:.8f}")
    lines.append(f"best_final_eps_epoch={best_final_eps[1]} best_final_eps_s3={best_final_eps[0]:.8f}")
    lines.append(f"best_final_scale_epoch={best_final_scale[1]} best_final_scale_s3={best_final_scale[0]:.8f}")

    for tag, (_, ep, ev) in [
        ("best_final_cd", best_final_cd),
        ("best_final_eps", best_final_eps),
        ("best_final_scale", best_final_scale),
    ]:
        lines.append("")
        lines.append(f"{tag}_epoch={ep}")
        lines.append(f"delta_log_scale_abs_mean {metric_line('delta', ev['delta'])}")
        for step in [0, 1, 2, 3]:
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
