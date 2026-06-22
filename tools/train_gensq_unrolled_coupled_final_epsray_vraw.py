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
    sample_log_uniform,
    exp_to_raw,
    raw_to_exp,
    sample_gensq_batched_dirs,
    chamfer,
    summarize_vec,
)


class EpsUnrollNet(nn.Module):
    """
    Eps delta predictor.
    Input per point:
      X / current_scale, current raw_eps, current log_scale
    Output:
      delta_raw_eps
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

    def forward(self, X_scaled, raw_eps, log_scale):
        B, S, _ = X_scaled.shape
        re = raw_eps[:, None, :].expand(B, S, 3)
        ls = log_scale[:, None, :].expand(B, S, 3)
        feat = torch.cat([X_scaled, re, ls], dim=-1)
        h = self.point(feat).amax(dim=1)
        return self.head(h)


class ScaleUnrollNet(nn.Module):
    """
    Scale delta predictor with raw exponent-undo input.

    Per-point input:
      v_raw = sign(X) * |X|^(eps/2)
      current log_scale
      current raw_eps

    This keeps absolute size information while making the shape more ellipsoid-like.

    Input dim:
      v_raw(3) + log_scale(3) + raw_eps(3) = 9
    """
    def __init__(self, exp_min=0.5, exp_max=8.0):
        super().__init__()
        self.exp_min = exp_min
        self.exp_max = exp_max

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

        # Shape-correct raw X, preserving absolute scale.
        eps = raw_to_exp(raw_eps, self.exp_min, self.exp_max).detach()
        v_raw = torch.sign(X) * X.abs().clamp_min(1e-8).pow(eps[:, None, :] / 2.0)

        ls = log_scale[:, None, :].expand(B, S, 3)
        re = raw_eps[:, None, :].expand(B, S, 3)

        feat = torch.cat([v_raw, ls, re], dim=-1)
        h = self.point(feat).amax(dim=1)
        return self.head(h)

def make_batch(args, B, dirs_base, device):
    S = dirs_base.shape[0]
    scale_gt = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    eps_gt = sample_log_uniform(args.exp_min, args.exp_max, (B, 3), device)

    dirs = dirs_base[None, :, :].expand(B, S, 3)
    X = sample_gensq_batched_dirs(scale_gt, eps_gt, dirs)

    return X, scale_gt, eps_gt


def constant_initial_state(args, B, device):
    scale0_value = math.sqrt(args.scale_min * args.scale_max)
    eps0_value = math.sqrt(args.exp_min * args.exp_max)

    log_scale = torch.full((B, 3), math.log(scale0_value), device=device)
    eps0 = torch.full((B, 3), eps0_value, device=device)
    raw_eps = exp_to_raw(eps0, args.exp_min, args.exp_max)

    return log_scale, raw_eps


def unroll(args, X, eps_net, scale_net):
    B = X.shape[0]
    device = X.device

    log_scale, raw_eps = constant_initial_state(args, B, device)

    history = []
    history.append((log_scale, raw_eps))

    log_floor = math.log(args.scale_floor)
    log_ceil = math.log(args.scale_ceil)

    for _ in range(args.unroll_steps):
        scale = log_scale.exp().clamp_min(1e-8)

        # EpsNet sees the cloud in the coordinate system of current scale.
        # Detach scale for EpsNet input so eps loss does not update ScaleNet through X/scale.
        X_scaled = X / scale.detach()[:, None, :].clamp_min(1e-8)
        delta_raw_eps = eps_net(X_scaled, raw_eps, log_scale.detach())
        raw_eps_next = raw_eps + args.eps_damp * delta_raw_eps
        raw_eps_next = raw_eps_next.clamp(-args.raw_eps_clip, args.raw_eps_clip)

        # ScaleNet sees current eps as context, but detached.
        delta_log_scale = scale_net(X, log_scale, raw_eps.detach())
        log_scale_next = log_scale + args.scale_damp * delta_log_scale
        log_scale_next = log_scale_next.clamp(log_floor, log_ceil)

        raw_eps = raw_eps_next
        log_scale = log_scale_next
        history.append((log_scale, raw_eps))

    return history


def full_cd(args, X, scale, raw_eps, dirs_base):
    B, S, _ = X.shape
    dirs = dirs_base[None, :, :].expand(B, S, 3)
    eps = raw_to_exp(raw_eps, args.exp_min, args.exp_max)
    Y = sample_gensq_batched_dirs(scale, eps, dirs)
    return chamfer(X, Y)


def final_losses(args, X, log_scale_N, raw_eps_N, dirs_base):
    scale_N = log_scale_N.exp().clamp_min(1e-8)

    # Scale loss: full-space CD, eps detached.
    # This updates the ScaleNet path only.
    loss_scale = full_cd(args, X, scale_N, raw_eps_N.detach(), dirs_base).mean()

    # Eps loss: ray-aligned normalized pointwise loss, scale detached.
    # This updates the EpsNet path only.
    scale_det = scale_N.detach().clamp_min(1e-8)
    X_norm = X / scale_det[:, None, :]
    rays = X_norm / X_norm.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    eps_N = raw_to_exp(raw_eps_N, args.exp_min, args.exp_max)
    unit_scale = torch.ones_like(scale_det)
    Y_unit = sample_gensq_batched_dirs(unit_scale, eps_N, rays)

    loss_eps = ((X_norm - Y_unit) ** 2).sum(dim=-1).mean()

    loss_reg = torch.zeros((), device=X.device)

    return loss_scale, loss_eps, loss_reg


@torch.no_grad()
def eval_model(args, eps_net, scale_net, val_data, dirs_base):
    X, scale_gt, eps_gt = val_data
    hist = unroll(args, X, eps_net, scale_net)

    out = {}
    for step, (log_scale, raw_eps) in enumerate(hist):
        scale = log_scale.exp().clamp_min(1e-8)
        eps = raw_to_exp(raw_eps, args.exp_min, args.exp_max)

        cd = full_cd(args, X, scale, raw_eps, dirs_base)
        scale_l1 = (scale - scale_gt).abs().mean(dim=1)
        eps_l1 = (eps - eps_gt).abs().mean(dim=1)

        out[step] = {
            "cd": summarize_vec(cd),
            "scale": summarize_vec(scale_l1),
            "eps": summarize_vec(eps_l1),
        }

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
    ap.add_argument("--scale-floor", type=float, default=0.02)
    ap.add_argument("--scale-ceil", type=float, default=0.80)
    ap.add_argument("--exp-min", type=float, default=0.5)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--raw-eps-clip", type=float, default=8.0)
    ap.add_argument("--unroll-steps", type=int, default=4)
    ap.add_argument("--scale-damp", type=float, default=0.5)
    ap.add_argument("--eps-damp", type=float, default=0.5)
    ap.add_argument("--loss-scale-w", type=float, default=1.0)
    ap.add_argument("--loss-eps-w", type=float, default=1.0)
    ap.add_argument("--loss-reg-w", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs_base = fibonacci_sphere(args.points, device)

    eps_net = EpsUnrollNet().to(device)
    scale_net = ScaleUnrollNet(args.exp_min, args.exp_max).to(device)

    opt = torch.optim.AdamW(
        list(eps_net.parameters()) + list(scale_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    val_data = make_batch(args, args.val_n, dirs_base, device)

    best_final_cd = (float("inf"), -1, None)
    best_final_scale = (float("inf"), -1, None)
    best_final_eps = (float("inf"), -1, None)

    metrics_path = out / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "epoch",
            "train_loss",
            "loss_scale",
            "loss_eps",
            "step",
            "cd_mean",
            "scale_l1_mean",
            "eps_l1_mean",
        ])

        for ep in range(1, args.epochs + 1):
            eps_net.train()
            scale_net.train()

            losses, scale_losses, eps_losses = [], [], []
            n_batches = math.ceil(args.train_n / args.batch_size)

            for _ in range(n_batches):
                X, scale_gt, eps_gt = make_batch(args, args.batch_size, dirs_base, device)
                hist = unroll(args, X, eps_net, scale_net)
                log_scale_N, raw_eps_N = hist[-1]

                loss_scale, loss_eps, loss_reg = final_losses(
                    args, X, log_scale_N, raw_eps_N, dirs_base
                )

                loss = (
                    args.loss_scale_w * loss_scale
                    + args.loss_eps_w * loss_eps
                    + args.loss_reg_w * loss_reg
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(eps_net.parameters()) + list(scale_net.parameters()),
                    args.grad_clip,
                )
                opt.step()

                losses.append(loss.item())
                scale_losses.append(loss_scale.item())
                eps_losses.append(loss_eps.item())

            eps_net.eval()
            scale_net.eval()
            with torch.no_grad():
                ev = eval_model(args, eps_net, scale_net, val_data, dirs_base)

            train_loss = sum(losses) / len(losses)
            train_scale_loss = sum(scale_losses) / len(scale_losses)
            train_eps_loss = sum(eps_losses) / len(eps_losses)

            for step in range(args.unroll_steps + 1):
                wr.writerow([
                    ep,
                    train_loss,
                    train_scale_loss,
                    train_eps_loss,
                    step,
                    ev[step]["cd"][0],
                    ev[step]["scale"][0],
                    ev[step]["eps"][0],
                ])
            f.flush()

            final = ev[args.unroll_steps]
            final_cd = final["cd"][0]
            final_scale = final["scale"][0]
            final_eps = final["eps"][0]

            if final_cd < best_final_cd[0]:
                best_final_cd = (final_cd, ep, ev)
            if final_scale < best_final_scale[0]:
                best_final_scale = (final_scale, ep, ev)
            if final_eps < best_final_eps[0]:
                best_final_eps = (final_eps, ep, ev)

            print(
                f"ep={ep:03d} train={train_loss:.8f} "
                f"loss_scale/loss_eps={train_scale_loss:.8f}/{train_eps_loss:.8f} "
                f"final cd/scale/eps={final_cd:.8f}/{final_scale:.8f}/{final_eps:.8f}",
                flush=True,
            )

    scale0_value = math.sqrt(args.scale_min * args.scale_max)
    eps0_value = math.sqrt(args.exp_min * args.exp_max)

    lines = []
    lines.append("experiment=unrolled_coupled_final_epsray_vraw")
    lines.append(f"scale0_const_log_midpoint={scale0_value:.8f}")
    lines.append(f"eps0_const_log_midpoint={eps0_value:.8f}")
    lines.append(f"unroll_steps={args.unroll_steps}")
    lines.append(f"scale_damp={args.scale_damp}")
    lines.append(f"eps_damp={args.eps_damp}")
    lines.append(f"exp_min={args.exp_min}")
    lines.append(f"best_final_cd_epoch={best_final_cd[1]} best_final_cd_sN={best_final_cd[0]:.8f}")
    lines.append(f"best_final_scale_epoch={best_final_scale[1]} best_final_scale_sN={best_final_scale[0]:.8f}")
    lines.append(f"best_final_eps_epoch={best_final_eps[1]} best_final_eps_sN={best_final_eps[0]:.8f}")

    for tag, (_, ep, ev) in [
        ("best_final_cd", best_final_cd),
        ("best_final_scale", best_final_scale),
        ("best_final_eps", best_final_eps),
    ]:
        lines.append("")
        lines.append(f"{tag}_epoch={ep}")
        for step in range(args.unroll_steps + 1):
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
