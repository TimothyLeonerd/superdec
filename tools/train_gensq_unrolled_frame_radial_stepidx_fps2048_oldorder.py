#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import tools.train_gensq_unrolled_coupled_final_cd_stepidx as base
import tools.train_gensq_unrolled_frame_cd_stepidx_fps2048 as oldframe


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


def make_batch(args, B, device):
    scale_gt = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    eps_gt = base.sample_log_uniform(args.exp_min, args.exp_max, (B, 3), device)

    dirs = base.fibonacci_sphere(args.source_points, device)
    dirs = dirs[None, :, :].expand(B, args.source_points, 3)

    X_local_dense = base.sample_gensq_batched_dirs(scale_gt, eps_gt, dirs)
    X_local = fps_downsample(X_local_dense, args.points)

    R_gt = random_rotations(B, device)
    X_world = X_local @ R_gt.transpose(1, 2)

    return X_world, scale_gt, eps_gt, R_gt


def init_state(args, B, device):
    scale0 = math.sqrt(args.scale_min * args.scale_max)
    eps0 = math.sqrt(args.exp_min * args.exp_max)
    log_scale = torch.full((B, 3), math.log(scale0), device=device)
    raw_eps = base.exp_to_raw(torch.full((B, 3), eps0, device=device), args.exp_min, args.exp_max)
    frame_R = torch.eye(3, device=device)[None].expand(B, 3, 3).clone()
    return log_scale, raw_eps, frame_R


def sample_pred(args, log_scale, raw_eps, dirs_obs):
    scale = log_scale.exp().clamp_min(1e-8)
    eps = base.raw_to_exp(raw_eps, args.exp_min, args.exp_max)
    return base.sample_gensq_batched_dirs(scale, eps, dirs_obs)


def radial_mse(Y, X):
    return ((Y - X).pow(2).sum(dim=-1)).mean(dim=1)


def unroll(args, X_world, R_gt, frame_net, eps_net, scale_net):
    B = X_world.shape[0]
    device = X_world.device
    log_scale, raw_eps, frame_R = init_state(args, B, device)

    hist = [(log_scale, raw_eps, frame_R)]
    frame_losses = []

    log_floor = math.log(args.scale_floor)
    log_ceil = math.log(args.scale_ceil)

    for k in range(args.unroll_steps):
        step_t = torch.full((B,), float(k) / max(args.unroll_steps - 1, 1), device=device)

        # OLD ORDER:
        # scale/eps use the current frame first
        X_canon = X_world @ frame_R.detach()
        dirs_obs = F.normalize(X_canon, dim=-1, eps=1e-8)

        scale_for_eps = log_scale.exp().detach().clamp_min(1e-8)
        X_scaled = X_canon / scale_for_eps[:, None, :]
        d_eps = eps_net(X_scaled, raw_eps, log_scale.detach(), step_t)
        raw_eps = (raw_eps + args.eps_damp * d_eps).clamp(-args.raw_eps_clip, args.raw_eps_clip)

        d_scale = scale_net(X_canon, log_scale, raw_eps.detach(), step_t)
        log_scale = (log_scale + args.scale_damp * d_scale).clamp(log_floor, log_ceil)

        # then update frame, as in the previous full FPS run
        target_delta = frame_R.detach().transpose(1, 2) @ R_gt
        delta_R, neff, maxw = frame_net(X_canon)
        floss, _ = oldframe.frame_loss_and_angles(delta_R, target_delta)
        frame_losses.append(floss.mean())

        frame_R = frame_R.detach() @ delta_R.detach()

        hist.append((log_scale, raw_eps, frame_R))

    return hist, frame_losses


def train_loss(args, X_world, R_gt, hist, frame_losses):
    log_scale, raw_eps, frame_R = hist[-1]
    X_canon = X_world @ frame_R.detach()
    dirs_obs = F.normalize(X_canon, dim=-1, eps=1e-8)

    Y_s = sample_pred(args, log_scale, raw_eps.detach(), dirs_obs)
    loss_scale = radial_mse(Y_s, X_canon).mean()

    Y_e = sample_pred(args, log_scale.detach(), raw_eps, dirs_obs)
    loss_eps = radial_mse(Y_e, X_canon).mean()

    loss_frame = torch.stack(frame_losses).mean()
    return loss_scale + loss_eps + args.frame_w * loss_frame, loss_scale, loss_eps, loss_frame


def best_perm_from_frames(pred_R, gt_R):
    device = pred_R.device
    perms = oldframe.PERMS.to(device)
    dots = torch.einsum("bik,bil->bkl", pred_R, gt_R).abs().clamp(0.0, 1.0)
    losses = []
    for p in perms:
        d = dots[:, torch.arange(3, device=device), p]
        losses.append((1.0 - d.pow(2)).mean(dim=1))
    loss_stack = torch.stack(losses, dim=1)
    best = loss_stack.argmin(dim=1)
    return perms[best]

def permute_gt_to_pred_axes(x_gt, perm):
    return torch.gather(x_gt, 1, perm)

def mean_or_nan(xs):
    if len(xs) == 0:
        return float("nan")
    x = torch.cat(xs).float()
    if x.numel() == 0:
        return float("nan")
    return float(x.mean())


def q_or_nan(xs, q):
    if len(xs) == 0:
        return float("nan")
    x = torch.cat(xs).float()
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x, q))


def append_eps_bins(store, eps, eps_gt):
    err = (eps - eps_gt).abs().reshape(-1).detach().cpu()
    logerr = (eps.clamp_min(1e-8).log() - eps_gt.clamp_min(1e-8).log()).abs().reshape(-1).detach().cpu()
    gt = eps_gt.reshape(-1).detach().cpu()

    bins = [
        ("0p5_1", 0.5, 1.0),
        ("1_2", 1.0, 2.0),
        ("2_4", 2.0, 4.0),
        ("4_8", 4.0, 8.000001),
    ]

    for name, lo, hi in bins:
        m = (gt >= lo) & (gt < hi)
        store[f"eps_l1_gt_{name}"].append(err[m])
        store[f"logeps_l1_gt_{name}"].append(logerr[m])


@torch.no_grad()
def evaluate(args, frame_net, eps_net, scale_net, val_data, device):
    frame_net.eval()
    eps_net.eval()
    scale_net.eval()

    X_all, scale_all, eps_all, R_all = val_data
    loader = DataLoader(TensorDataset(X_all, scale_all, eps_all, R_all), batch_size=args.batch_size)

    keys = [
        "radial", "scale_l1", "eps_l1", "log_eps_l1", "rel_eps_l1", "scale_l1_aligned", "eps_l1_aligned", "log_eps_l1_aligned", "rel_eps_l1_aligned",
        "frame_angle",
        "eps_l1_gt_0p5_1", "eps_l1_gt_1_2", "eps_l1_gt_2_4", "eps_l1_gt_4_8",
        "logeps_l1_gt_0p5_1", "logeps_l1_gt_1_2", "logeps_l1_gt_2_4", "logeps_l1_gt_4_8",
    ]

    rows = [{k: [] for k in keys} for _ in range(args.unroll_steps + 1)]

    for X, scale_gt, eps_gt, R_gt in loader:
        X = X.to(device)
        scale_gt = scale_gt.to(device)
        eps_gt = eps_gt.to(device)
        R_gt = R_gt.to(device)

        hist, _ = unroll(args, X, R_gt, frame_net, eps_net, scale_net)

        for j, (log_scale, raw_eps, frame_R) in enumerate(hist):
            X_canon = X @ frame_R.detach()
            dirs_obs = F.normalize(X_canon, dim=-1, eps=1e-8)
            Y = sample_pred(args, log_scale, raw_eps, dirs_obs)

            radial = radial_mse(Y, X_canon)
            scale = log_scale.exp().clamp_min(1e-8)
            eps = base.raw_to_exp(raw_eps, args.exp_min, args.exp_max)

            _, angle = oldframe.frame_loss_and_angles(frame_R, R_gt)
            perm = best_perm_from_frames(frame_R, R_gt)
            scale_gt_a = permute_gt_to_pred_axes(scale_gt, perm)
            eps_gt_a = permute_gt_to_pred_axes(eps_gt, perm)

            rows[j]["radial"].append(radial.detach().cpu())

            # raw unaligned diagnostics
            rows[j]["scale_l1"].append((scale - scale_gt).abs().mean(dim=1).detach().cpu())
            rows[j]["eps_l1"].append((eps - eps_gt).abs().mean(dim=1).detach().cpu())
            rows[j]["log_eps_l1"].append((eps.log() - eps_gt.log()).abs().mean(dim=1).detach().cpu())
            rows[j]["rel_eps_l1"].append(((eps - eps_gt).abs() / eps_gt.clamp_min(1e-8)).mean(dim=1).detach().cpu())

            # frame-permutation-aligned diagnostics
            rows[j]["scale_l1_aligned"].append((scale - scale_gt_a).abs().mean(dim=1).detach().cpu())
            rows[j]["eps_l1_aligned"].append((eps - eps_gt_a).abs().mean(dim=1).detach().cpu())
            rows[j]["log_eps_l1_aligned"].append((eps.log() - eps_gt_a.log()).abs().mean(dim=1).detach().cpu())
            rows[j]["rel_eps_l1_aligned"].append(((eps - eps_gt_a).abs() / eps_gt_a.clamp_min(1e-8)).mean(dim=1).detach().cpu())

            rows[j]["frame_angle"].append(angle.reshape(angle.shape[0], -1).mean(dim=1).detach().cpu())

            append_eps_bins(rows[j], eps, eps_gt_a)

    out = []
    for j in range(args.unroll_steps + 1):
        d = {}
        for k in keys:
            d[k] = mean_or_nan(rows[j][k])
        d["frame_angle_p95"] = q_or_nan(rows[j]["frame_angle"], 0.95)
        out.append(d)

    return out


def format_step(j, d):
    return (
        f"step {j}: radial={d['radial']:.10f} "
        f"scale_l1={d['scale_l1']:.8f} "
        f"eps_l1={d['eps_l1']:.8f} "
        f"log_eps_l1={d['log_eps_l1']:.8f} "
        f"rel_eps_l1={d['rel_eps_l1']:.8f} "
        f"frame_angle={d['frame_angle']:.6f} "
        f"frame_p95={d['frame_angle_p95']:.6f} "
        f"eps_bins=[0.5-1:{d['eps_l1_gt_0p5_1']:.6f}, "
        f"1-2:{d['eps_l1_gt_1_2']:.6f}, "
        f"2-4:{d['eps_l1_gt_2_4']:.6f}, "
        f"4-8:{d['eps_l1_gt_4_8']:.6f}] "
        f"logeps_bins=[0.5-1:{d['logeps_l1_gt_0p5_1']:.6f}, "
        f"1-2:{d['logeps_l1_gt_1_2']:.6f}, "
        f"2-4:{d['logeps_l1_gt_2_4']:.6f}, "
        f"4-8:{d['logeps_l1_gt_4_8']:.6f}]"
    )


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--frame-w", type=float, default=1.0)
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

    print(f"device={device}", flush=True)

    base.set_seed(args.seed + 2000)
    vals = []
    remaining = args.val_n
    while remaining > 0:
        b = min(args.batch_size, remaining)
        vals.append(tuple(t.cpu() for t in make_batch(args, b, device)))
        remaining -= b
    val_data = tuple(torch.cat([v[i] for v in vals], dim=0) for i in range(4))

    frame_net = oldframe.WeightedPCANet().to(device)
    eps_net = EpsNet().to(device)
    scale_net = ScaleNet().to(device)

    params = list(frame_net.parameters()) + list(eps_net.parameters()) + list(scale_net.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(args.train_n / args.batch_size)

    fieldnames = [
        "epoch", "train_loss", "train_scale_loss", "train_eps_loss", "train_frame_loss",
        "radial", "scale_l1", "eps_l1", "log_eps_l1", "rel_eps_l1", "scale_l1_aligned", "eps_l1_aligned", "log_eps_l1_aligned", "rel_eps_l1_aligned",
        "frame_angle", "frame_angle_p95", "scale_l1_aligned", "eps_l1_aligned", "log_eps_l1_aligned", "rel_eps_l1_aligned",
        "eps_l1_gt_0p5_1", "eps_l1_gt_1_2", "eps_l1_gt_2_4", "eps_l1_gt_4_8",
        "logeps_l1_gt_0p5_1", "logeps_l1_gt_1_2", "logeps_l1_gt_2_4", "logeps_l1_gt_4_8",
    ]

    best_eps = (float("inf"), None, None)
    best_scale = (float("inf"), None, None)
    best_radial = (float("inf"), None, None)
    best_frame = (float("inf"), None, None)

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            frame_net.train()
            eps_net.train()
            scale_net.train()

            totals = {"loss": 0.0, "scale": 0.0, "eps": 0.0, "frame": 0.0}
            seen = 0

            for _ in range(steps_per_epoch):
                b = min(args.batch_size, args.train_n - seen)
                if b <= 0:
                    break

                X, scale_gt, eps_gt, R_gt = make_batch(args, b, device)
                hist, frame_losses = unroll(args, X, R_gt, frame_net, eps_net, scale_net)
                loss, ls, le, lf = train_loss(args, X, R_gt, hist, frame_losses)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()

                totals["loss"] += float(loss.detach()) * b
                totals["scale"] += float(ls.detach()) * b
                totals["eps"] += float(le.detach()) * b
                totals["frame"] += float(lf.detach()) * b
                seen += b

            ev = evaluate(args, frame_net, eps_net, scale_net, val_data, device)
            final = ev[-1]

            row = {
                "epoch": ep,
                "train_loss": totals["loss"] / max(1, seen),
                "train_scale_loss": totals["scale"] / max(1, seen),
                "train_eps_loss": totals["eps"] / max(1, seen),
                "train_frame_loss": totals["frame"] / max(1, seen),
                **final,
            }
            writer.writerow(row)
            f.flush()

            if row["radial"] < best_radial[0]:
                best_radial = (row["radial"], ep, ev)
            if row["scale_l1_aligned"] < best_scale[0]:
                best_scale = (row["scale_l1_aligned"], ep, ev)
            if row["eps_l1_aligned"] < best_eps[0]:
                best_eps = (row["eps_l1_aligned"], ep, ev)
            if row["frame_angle"] < best_frame[0]:
                best_frame = (row["frame_angle"], ep, ev)

            print(
                f"ep={ep:03d} train={row['train_loss']:.8g} "
                f"rad={row['radial']:.8g} scale={row['scale_l1']:.6g} "
                f"eps={row['eps_l1']:.6g} epsA={row['eps_l1_aligned']:.6g} logepsA={row['log_eps_l1_aligned']:.6g} scaleA={row['scale_l1_aligned']:.6g} "
                f"frame={row['frame_angle']:.4f}/{row['frame_angle_p95']:.4f} "
                f"epsbin[0.5-1]={row['eps_l1_gt_0p5_1']:.4f} "
                f"[1-2]={row['eps_l1_gt_1_2']:.4f} "
                f"[2-4]={row['eps_l1_gt_2_4']:.4f} "
                f"[4-8]={row['eps_l1_gt_4_8']:.4f}",
                flush=True,
            )

    lines = [
        "experiment=unrolled_frame_radial_stepidx_fps2048_to_512_oldorder",
        "data=random_rotated_gensq_fps2048_to_512",
        "frame_net=WeightedPCANet",
        "frame_loss=6way_axis_loss_on_delta_frame",
        "scale_eps_loss=observed_ray_radial_mse",
        "frame_first_ordering=False_old_order_like_previous_run",
        "prediction_rays=normalize(X_world @ frame_R)",
        f"source_points={args.source_points}",
        f"observed_points={args.points}",
        f"unroll_steps={args.unroll_steps}",
        f"best_final_radial_epoch={best_radial[1]} best_final_radial={best_radial[0]:.10f}",
        f"best_final_scale_epoch={best_scale[1]} best_final_scale_l1_aligned={best_scale[0]:.10f}",
        f"best_final_eps_epoch={best_eps[1]} best_final_eps_l1_aligned={best_eps[0]:.10f}",
        f"best_final_frame_epoch={best_frame[1]} best_final_frame_angle={best_frame[0]:.10f}",
        "",
    ]

    for name, best in [
        ("best_radial", best_radial),
        ("best_scale", best_scale),
        ("best_eps", best_eps),
        ("best_frame", best_frame),
    ]:
        _, ep, ev = best
        lines.append(f"{name}_epoch={ep}")
        for j, d in enumerate(ev):
            lines.append(format_step(j, d))
        lines.append("")

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
