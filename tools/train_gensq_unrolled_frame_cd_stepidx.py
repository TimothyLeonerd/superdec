#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import tools.train_gensq_unrolled_coupled_final_cd_stepidx as base

import torch.nn.functional as F

PERMS = torch.tensor([
    [0, 1, 2], [0, 2, 1], [1, 0, 2],
    [1, 2, 0], [2, 0, 1], [2, 1, 0],
], dtype=torch.long)


def random_rotations(batch, device):
    q = torch.randn(batch, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)

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


def hard_pca_frame(points):
    mu = points.mean(dim=1)
    q = points - mu[:, None, :]
    cov = q.transpose(1, 2) @ q / points.shape[1]
    _, evecs = torch.linalg.eigh(cov)
    return evecs, mu


def frame_loss_and_angles(pred_R, gt_R):
    device = pred_R.device
    perms = PERMS.to(device)

    # dots[b, k, l] = |pred axis k dot gt axis l|
    dots = torch.einsum("bik,bil->bkl", pred_R, gt_R).abs().clamp(0.0, 1.0)

    losses = []
    for pmt in perms:
        d = dots[:, torch.arange(3, device=device), pmt]
        losses.append((1.0 - d.pow(2)).mean(dim=1))

    loss_stack = torch.stack(losses, dim=1)
    best = loss_stack.argmin(dim=1)
    loss = loss_stack[torch.arange(pred_R.shape[0], device=device), best]

    angles = []
    for b in range(pred_R.shape[0]):
        pmt = perms[best[b]]
        d = dots[b, torch.arange(3, device=device), pmt]
        angles.append(torch.rad2deg(torch.acos(d.clamp(0.0, 1.0))))

    return loss, torch.stack(angles, dim=0)


class WeightedPCANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.point = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Starts as uniform weights, i.e. initially identical to hard PCA.
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def make_features(self, points):
        with torch.no_grad():
            R0, mu0 = hard_pca_frame(points)
            q = (points - mu0[:, None, :]) @ R0
            r = q.norm(dim=-1, keepdim=True)
            return torch.cat([q, q.abs(), q * q, r], dim=-1)

    def forward(self, points):
        feat = self.make_features(points)

        h = self.point(feat)
        g = h.max(dim=1).values
        g = g[:, None, :].expand(-1, points.shape[1], -1)

        logits = self.head(torch.cat([h, g], dim=-1)).squeeze(-1)
        w = F.softplus(logits) + 1e-8
        alpha = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)

        mu = (points * alpha[..., None]).sum(dim=1)
        q = points - mu[:, None, :]
        cov = (q * alpha[..., None]).transpose(1, 2) @ q

        _, evecs = torch.linalg.eigh(cov)

        n_eff = 1.0 / alpha.pow(2).sum(dim=1).clamp_min(1e-8)
        maxw = alpha.max(dim=1).values

        return evecs, n_eff, maxw


def chamfer_sq(a, b):
    d = torch.cdist(a, b).pow(2)
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)


class EpsUnrollNetR(nn.Module):
    """
    Eps update net with frame state.

    Per-point input:
      X_world @ R_i / scale_i : 3
      raw_eps_i              : 3
      log_scale_i            : 3
      R_i flattened          : 9
      step_t                 : 1
    total: 19
    """
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(19, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, X_scaled, raw_eps, log_scale, frame_flat, step_t):
        B, S, _ = X_scaled.shape
        re = raw_eps[:, None, :].expand(B, S, 3)
        ls = log_scale[:, None, :].expand(B, S, 3)
        rf = frame_flat[:, None, :].expand(B, S, 9)
        st = step_t[:, None, None].expand(B, S, 1)
        feat = torch.cat([X_scaled, re, ls, rf, st], dim=-1)
        h = self.point(feat).amax(dim=1)
        return self.head(h)


class ScaleUnrollNetR(nn.Module):
    """
    Scale update net with frame state.

    Per-point input:
      X_world @ R_i          : 3
      log_scale_i            : 3
      raw_eps_i              : 3
      R_i flattened          : 9
      step_t                 : 1
    total: 19
    """
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(19, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, X_canon, log_scale, raw_eps, frame_flat, step_t):
        B, S, _ = X_canon.shape
        ls = log_scale[:, None, :].expand(B, S, 3)
        re = raw_eps[:, None, :].expand(B, S, 3)
        rf = frame_flat[:, None, :].expand(B, S, 9)
        st = step_t[:, None, None].expand(B, S, 1)
        feat = torch.cat([X_canon, ls, re, rf, st], dim=-1)
        h = self.point(feat).amax(dim=1)
        return self.head(h)


def make_batch(args, B, dirs_base, device):
    S = dirs_base.shape[0]

    scale_gt = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    eps_gt = base.sample_log_uniform(args.exp_min, args.exp_max, (B, 3), device)

    dirs = dirs_base[None, :, :].expand(B, S, 3)
    X_local = base.sample_gensq_batched_dirs(scale_gt, eps_gt, dirs)

    R_gt = random_rotations(B, device)
    X_world = X_local @ R_gt.transpose(1, 2)

    return X_world, scale_gt, eps_gt, R_gt


def constant_initial_state(args, B, device):
    scale0_value = math.sqrt(args.scale_min * args.scale_max)
    eps0_value = math.sqrt(args.exp_min * args.exp_max)

    log_scale = torch.full((B, 3), math.log(scale0_value), device=device)
    eps0 = torch.full((B, 3), eps0_value, device=device)
    raw_eps = base.exp_to_raw(eps0, args.exp_min, args.exp_max)

    R0 = torch.eye(3, device=device)[None, :, :].expand(B, 3, 3).clone()
    return log_scale, raw_eps, R0


def full_cd_world(args, X_world, log_scale, raw_eps, frame_R, dirs_base):
    B = X_world.shape[0]
    scale = log_scale.exp().clamp_min(1e-8)
    eps = base.raw_to_exp(raw_eps, args.exp_min, args.exp_max)

    dirs = dirs_base[None, :, :].expand(B, dirs_base.shape[0], 3)
    Y_local = base.sample_gensq_batched_dirs(scale, eps, dirs)

    # frame_R maps world -> canonical by X @ frame_R.
    # Therefore local/canonical -> world is Y @ frame_R^T.
    Y_world = Y_local @ frame_R.transpose(1, 2)
    return chamfer_sq(Y_world, X_world)


def best_perm(pred_R, gt_R):
    device = pred_R.device
    perms = PERMS.to(device)
    dots = torch.einsum("bik,bil->bkl", pred_R, gt_R).abs().clamp(0.0, 1.0)

    scores = []
    for p in perms:
        d = dots[:, torch.arange(3, device=device), p]
        scores.append(d.sum(dim=1))
    scores = torch.stack(scores, dim=1)
    best = scores.argmax(dim=1)
    return perms[best]


def permute_targets(x, perm):
    return x.gather(1, perm)


def unroll(args, X_world, R_gt, eps_net, scale_net, frame_net):
    B = X_world.shape[0]
    device = X_world.device

    log_scale, raw_eps, frame_R = constant_initial_state(args, B, device)

    history = [(log_scale, raw_eps, frame_R)]
    frame_losses = []
    neffs = []
    maxws = []

    log_floor = math.log(args.scale_floor)
    log_ceil = math.log(args.scale_ceil)

    for k in range(args.unroll_steps):
        step_t = torch.full(
            (B,),
            float(k) / max(args.unroll_steps - 1, 1),
            device=device,
        )

        # Current canonical coordinates.
        frame_det = frame_R.detach()
        X_canon = X_world @ frame_det

        # FrameNet predicts residual frame in current coordinates:
        # target_delta = R_i^T R_gt, because X_world @ R_i @ target_delta = X_local.
        target_delta = frame_det.transpose(1, 2) @ R_gt
        delta_R, neff, maxw = frame_net(X_canon)

        fl, _ = frame_loss_and_angles(delta_R, target_delta)
        frame_losses.append(fl)
        neffs.append(neff)
        maxws.append(maxw)

        frame_flat = frame_det.reshape(B, 9)

        scale = log_scale.exp().clamp_min(1e-8)

        X_scaled = X_canon / scale.detach()[:, None, :].clamp_min(1e-8)
        delta_raw_eps = eps_net(X_scaled, raw_eps, log_scale.detach(), frame_flat, step_t)
        raw_eps_next = raw_eps + args.eps_damp * delta_raw_eps
        raw_eps_next = raw_eps_next.clamp(-args.raw_eps_clip, args.raw_eps_clip)

        delta_log_scale = scale_net(X_canon, log_scale, raw_eps.detach(), frame_flat, step_t)
        log_scale_next = log_scale + args.scale_damp * delta_log_scale
        log_scale_next = log_scale_next.clamp(log_floor, log_ceil)

        # Frame is trained by frame loss only, not by final Chamfer.
        frame_R_next = frame_det @ delta_R.detach()

        raw_eps = raw_eps_next
        log_scale = log_scale_next
        frame_R = frame_R_next

        history.append((log_scale, raw_eps, frame_R))

    return history, frame_losses, neffs, maxws


def final_losses(args, X_world, R_gt, history, frame_losses, dirs_base):
    log_scale_N, raw_eps_N, frame_R_N = history[-1]
    frame_det = frame_R_N.detach()

    loss_scale = full_cd_world(args, X_world, log_scale_N, raw_eps_N.detach(), frame_det, dirs_base).mean()
    loss_eps = full_cd_world(args, X_world, log_scale_N.detach(), raw_eps_N, frame_det, dirs_base).mean()

    loss_frame = torch.stack([x.mean() for x in frame_losses]).mean()

    return loss_scale, loss_eps, loss_frame


def tensor_stats(x):
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean()),
        "med": float(x.median()),
        "p90": float(torch.quantile(x, 0.90)),
        "p95": float(torch.quantile(x, 0.95)),
        "max": float(x.max()),
    }


@torch.no_grad()
def evaluate(args, eps_net, scale_net, frame_net, val_data, dirs_base, device):
    eps_net.eval()
    scale_net.eval()
    frame_net.eval()

    X_all, scale_all, eps_all, R_all = val_data

    loader = DataLoader(
        TensorDataset(X_all, scale_all, eps_all, R_all),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    per_step = [
        {"cd": [], "scale_l1": [], "eps_l1": [], "frame_angle": []}
        for _ in range(args.unroll_steps + 1)
    ]
    frame_loss_steps = [[] for _ in range(args.unroll_steps)]
    neff_steps = [[] for _ in range(args.unroll_steps)]
    maxw_steps = [[] for _ in range(args.unroll_steps)]

    for X, scale_gt, eps_gt, R_gt in loader:
        X = X.to(device)
        scale_gt = scale_gt.to(device)
        eps_gt = eps_gt.to(device)
        R_gt = R_gt.to(device)

        history, frame_losses, neffs, maxws = unroll(args, X, R_gt, eps_net, scale_net, frame_net)

        for j, (log_scale, raw_eps, frame_R) in enumerate(history):
            cd = full_cd_world(args, X, log_scale, raw_eps, frame_R, dirs_base)

            perm = best_perm(frame_R, R_gt)
            scale_t = permute_targets(scale_gt, perm)
            eps_t = permute_targets(eps_gt, perm)

            scale = log_scale.exp().clamp_min(1e-8)
            eps = base.raw_to_exp(raw_eps, args.exp_min, args.exp_max)

            scale_l1 = (scale - scale_t).abs().mean(dim=1)
            eps_l1 = (eps - eps_t).abs().mean(dim=1)

            _, ang = frame_loss_and_angles(frame_R, R_gt)
            frame_ang = ang.reshape(-1)

            per_step[j]["cd"].append(cd.cpu())
            per_step[j]["scale_l1"].append(scale_l1.cpu())
            per_step[j]["eps_l1"].append(eps_l1.cpu())
            per_step[j]["frame_angle"].append(frame_ang.cpu())

        for k in range(args.unroll_steps):
            frame_loss_steps[k].append(frame_losses[k].cpu())
            neff_steps[k].append(neffs[k].cpu())
            maxw_steps[k].append(maxws[k].cpu())

    out = {"steps": []}
    for j in range(args.unroll_steps + 1):
        d = {}
        for name in ["cd", "scale_l1", "eps_l1", "frame_angle"]:
            d[name] = tensor_stats(torch.cat(per_step[j][name]))
        out["steps"].append(d)

    out["frame_loss_mean"] = float(torch.cat([torch.cat(x) for x in frame_loss_steps]).mean())
    out["neff_mean"] = float(torch.cat([torch.cat(x) for x in neff_steps]).mean())
    out["maxw_mean"] = float(torch.cat([torch.cat(x) for x in maxw_steps]).mean())
    return out


def format_stats(s):
    return f"{s['mean']:.8f} {s['med']:.8f} {s['p90']:.8f} {s['p95']:.8f} {s['max']:.8f}"


def summary_for_epoch(tag, epoch, ev):
    lines = [f"{tag}_epoch={epoch}"]
    for j, d in enumerate(ev["steps"]):
        lines.append(
            f"step {j}: "
            f"cd mean/med/p90/p95/max = {format_stats(d['cd'])} | "
            f"scale_l1 mean/med/p90/p95/max = {format_stats(d['scale_l1'])} | "
            f"eps_l1 mean/med/p90/p95/max = {format_stats(d['eps_l1'])} | "
            f"frame_angle_deg mean/med/p90/p95/max = {format_stats(d['frame_angle'])}"
        )
    lines.append(
        f"frame_loss_mean={ev['frame_loss_mean']:.8f} "
        f"neff_mean={ev['neff_mean']:.4f} "
        f"maxw_mean={ev['maxw_mean']:.6f}"
    )
    return lines


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
    ap.add_argument("--loss-frame-w", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    base.set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    dirs_base = base.fibonacci_sphere(args.points, device)

    print("Generating fixed validation set with random rotations", flush=True)
    val_batches = []
    remaining = args.val_n
    base.set_seed(args.seed + 2000)
    while remaining > 0:
        b = min(args.batch_size, remaining)
        val_batches.append(tuple(x.detach().cpu() for x in make_batch(args, b, dirs_base, device)))
        remaining -= b

    val_data = tuple(torch.cat([vb[i] for vb in val_batches], dim=0) for i in range(4))

    eps_net = EpsUnrollNetR().to(device)
    scale_net = ScaleUnrollNetR().to(device)
    frame_net = WeightedPCANet().to(device)

    opt = torch.optim.AdamW(
        list(eps_net.parameters()) + list(scale_net.parameters()) + list(frame_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(args.train_n / args.batch_size)

    fields = [
        "epoch",
        "train_loss",
        "train_scale_loss",
        "train_eps_loss",
        "train_frame_loss",
        "final_cd",
        "final_scale_l1",
        "final_eps_l1",
        "final_frame_angle",
        "frame_loss_mean",
        "neff_mean",
        "maxw_mean",
    ]

    best_cd = (float("inf"), None, None)
    best_scale = (float("inf"), None, None)
    best_eps = (float("inf"), None, None)
    best_frame = (float("inf"), None, None)

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            eps_net.train()
            scale_net.train()
            frame_net.train()

            total = total_s = total_e = total_f = 0.0
            seen = 0

            for _ in range(steps_per_epoch):
                b = min(args.batch_size, args.train_n - seen)
                if b <= 0:
                    break

                X, scale_gt, eps_gt, R_gt = make_batch(args, b, dirs_base, device)

                history, frame_losses, _, _ = unroll(args, X, R_gt, eps_net, scale_net, frame_net)
                loss_scale, loss_eps, loss_frame = final_losses(args, X, R_gt, history, frame_losses, dirs_base)

                loss = (
                    args.loss_scale_w * loss_scale
                    + args.loss_eps_w * loss_eps
                    + args.loss_frame_w * loss_frame
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(eps_net.parameters()) + list(scale_net.parameters()) + list(frame_net.parameters()),
                        args.grad_clip,
                    )
                opt.step()

                total += float(loss.detach()) * b
                total_s += float(loss_scale.detach()) * b
                total_e += float(loss_eps.detach()) * b
                total_f += float(loss_frame.detach()) * b
                seen += b

            ev = evaluate(args, eps_net, scale_net, frame_net, val_data, dirs_base, device)
            final = ev["steps"][-1]

            row = {
                "epoch": ep,
                "train_loss": total / max(1, seen),
                "train_scale_loss": total_s / max(1, seen),
                "train_eps_loss": total_e / max(1, seen),
                "train_frame_loss": total_f / max(1, seen),
                "final_cd": final["cd"]["mean"],
                "final_scale_l1": final["scale_l1"]["mean"],
                "final_eps_l1": final["eps_l1"]["mean"],
                "final_frame_angle": final["frame_angle"]["mean"],
                "frame_loss_mean": ev["frame_loss_mean"],
                "neff_mean": ev["neff_mean"],
                "maxw_mean": ev["maxw_mean"],
            }
            writer.writerow(row)
            f.flush()

            if row["final_cd"] < best_cd[0]:
                best_cd = (row["final_cd"], ep, ev)
            if row["final_scale_l1"] < best_scale[0]:
                best_scale = (row["final_scale_l1"], ep, ev)
            if row["final_eps_l1"] < best_eps[0]:
                best_eps = (row["final_eps_l1"], ep, ev)
            if row["final_frame_angle"] < best_frame[0]:
                best_frame = (row["final_frame_angle"], ep, ev)

            print(
                f"ep={ep:03d} "
                f"train={row['train_loss']:.6g} "
                f"cd={row['final_cd']:.8g} "
                f"scale={row['final_scale_l1']:.6g} "
                f"eps={row['final_eps_l1']:.6g} "
                f"frame={row['final_frame_angle']:.3f}deg "
                f"floss={row['frame_loss_mean']:.6g} "
                f"neff={row['neff_mean']:.1f} "
                f"maxw={row['maxw_mean']:.4f}",
                flush=True,
            )

    lines = [
        "experiment=unrolled_frame_cd_stepidx",
        "data=random_rotated_gensq",
        "frame_net=WeightedPCANet",
        "frame_loss=6way_axis_loss_on_delta_frame",
        "frame_update=R_next = R_current @ delta_R",
        "frame_grad=frame_loss_only_CD_uses_detached_frame",
        f"scale0_const_log_midpoint={math.sqrt(args.scale_min * args.scale_max):.8f}",
        f"eps0_const_log_midpoint={math.sqrt(args.exp_min * args.exp_max):.8f}",
        f"unroll_steps={args.unroll_steps}",
        f"scale_damp={args.scale_damp}",
        f"eps_damp={args.eps_damp}",
        f"exp_min={args.exp_min}",
        f"best_final_cd_epoch={best_cd[1]} best_final_cd_sN={best_cd[0]:.8f}",
        f"best_final_scale_epoch={best_scale[1]} best_final_scale_sN={best_scale[0]:.8f}",
        f"best_final_eps_epoch={best_eps[1]} best_final_eps_sN={best_eps[0]:.8f}",
        f"best_final_frame_epoch={best_frame[1]} best_final_frame_angle_sN={best_frame[0]:.8f}",
        "",
    ]

    for tag, best in [
        ("best_final_cd", best_cd),
        ("best_final_scale", best_scale),
        ("best_final_eps", best_eps),
        ("best_final_frame", best_frame),
    ]:
        _, ep, ev = best
        lines.extend(summary_for_epoch(tag, ep, ev))
        lines.append("")

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
