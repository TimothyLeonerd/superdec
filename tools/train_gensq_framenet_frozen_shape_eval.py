#!/usr/bin/env python3
import argparse, csv, itertools, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
    PointNetScaleExp,
    chamfer,
)

PERMS = list(itertools.permutations(range(3)))


def random_rotations(batch, device):
    q = torch.randn(batch, 4, device=device)
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)

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


@torch.no_grad()
def make_dataset(n, points, scale_min, scale_max, exp_min, exp_max, device, chunk=2048):
    dirs = fibonacci_sphere(points, device)
    all_world, all_R, all_scale, all_exp = [], [], [], []

    for start in range(0, n, chunk):
        b = min(chunk, n - start)

        scale = scale_min + (scale_max - scale_min) * torch.rand(b, 3, device=device)
        exp = exp_min + (exp_max - exp_min) * torch.rand(b, 3, device=device)
        R = random_rotations(b, device)

        local = sample_generalized_surface(scale, exp, dirs)
        world = local @ R.transpose(1, 2)

        all_world.append(world.cpu())
        all_R.append(R.cpu())
        all_scale.append(scale.cpu())
        all_exp.append(exp.cpu())

    return (
        torch.cat(all_world, 0),
        torch.cat(all_R, 0),
        torch.cat(all_scale, 0),
        torch.cat(all_exp, 0),
    )


class FrameNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.frame_head = nn.Linear(256, 9)

    def forward(self, x):
        h = self.point(x)
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
        z = self.trunk(pooled)

        M = self.frame_head(z).reshape(-1, 3, 3)
        U, _S, Vh = torch.linalg.svd(M)

        R = U @ Vh
        det = torch.det(R)

        D = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.shape[0], 3, 3).clone()
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

        return U @ D @ Vh


def frame_perm_loss(pred_R, gt_R):
    losses, angles = [], []

    for p in PERMS:
        gt = gt_R[:, :, list(p)]

        dot = (pred_R * gt).sum(dim=1).abs().clamp(0, 1)  # [B,3], column-wise
        loss = (1.0 - dot.pow(2)).mean(dim=1)
        angle = torch.acos(dot) * (180.0 / math.pi)

        losses.append(loss)
        angles.append(angle)

    losses = torch.stack(losses, dim=1)  # [B,6]
    angles = torch.stack(angles, dim=1)  # [B,6,3]

    best_idx = losses.argmin(dim=1)
    best_loss = losses.gather(1, best_idx[:, None]).squeeze(1)
    best_angles = angles.gather(1, best_idx[:, None, None].expand(-1, 1, 3)).squeeze(1)

    return best_loss, best_idx, best_angles


def gather_perm_params(gt_scale, gt_exp, best_idx):
    perms = torch.tensor(PERMS, device=gt_scale.device, dtype=torch.long)
    idx = perms[best_idx]  # [B,3]

    gt_scale_p = gt_scale.gather(1, idx)
    gt_exp_p = gt_exp.gather(1, idx)

    return gt_scale_p, gt_exp_p


@torch.no_grad()
def evaluate(frame_net, shape_net, loader, dirs, args, device):
    frame_net.eval()
    shape_net.eval()

    vals = {
        "frame_axis_angle": [],
        "gtcanon_scale_l1": [],
        "gtcanon_exp_l1": [],
        "gtcanon_cd": [],
        "predcanon_scale_l1": [],
        "predcanon_exp_l1": [],
        "predcanon_cd": [],
    }

    for world, gt_R, gt_scale, gt_exp in loader:
        world = world.to(device)
        gt_R = gt_R.to(device)
        gt_scale = gt_scale.to(device)
        gt_exp = gt_exp.to(device)

        pred_R = frame_net(world)
        _frame_loss, best_idx, best_angles = frame_perm_loss(pred_R, gt_R)

        vals["frame_axis_angle"].append(best_angles.reshape(-1).cpu())

        # Upper bound: GT canonicalization
        gt_local = world @ gt_R
        s_gtc, e_gtc = shape_net(gt_local)

        pred_local_gtc = sample_generalized_surface(s_gtc, e_gtc, dirs)
        pred_world_gtc = pred_local_gtc @ gt_R.transpose(1, 2)

        vals["gtcanon_scale_l1"].append((s_gtc - gt_scale).abs().mean(dim=1).cpu())
        vals["gtcanon_exp_l1"].append((e_gtc - gt_exp).abs().mean(dim=1).cpu())
        vals["gtcanon_cd"].append(chamfer(pred_world_gtc, world).cpu())

        # Actual frozen pipeline: predicted canonicalization
        pred_local = world @ pred_R
        s_pred, e_pred = shape_net(pred_local)

        gt_scale_p, gt_exp_p = gather_perm_params(gt_scale, gt_exp, best_idx)

        pred_local_surf = sample_generalized_surface(s_pred, e_pred, dirs)
        pred_world = pred_local_surf @ pred_R.transpose(1, 2)

        vals["predcanon_scale_l1"].append((s_pred - gt_scale_p).abs().mean(dim=1).cpu())
        vals["predcanon_exp_l1"].append((e_pred - gt_exp_p).abs().mean(dim=1).cpu())
        vals["predcanon_cd"].append(chamfer(pred_world, world).cpu())

    out = {}
    for k, xs in vals.items():
        x = torch.cat(xs)
        out[k + "_mean"] = float(x.mean())
        out[k + "_p95"] = float(torch.quantile(x, 0.95))
    return out


def load_shape_net(path, device):
    ckpt = torch.load(path, map_location=device)
    a = ckpt.get("args", {})

    model = PointNetScaleExp(
        a.get("scale_min", 0.08),
        a.get("scale_max", 0.35),
        a.get("exp_min", 1.0),
        a.get("exp_max", 8.0),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shape-ckpt", required=True)
    ap.add_argument("--train-n", type=int, default=10000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=1.0)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs = fibonacci_sphere(args.points, device)

    print("Loading frozen ShapeNet:", args.shape_ckpt, flush=True)
    shape_net = load_shape_net(args.shape_ckpt, device)

    print("Generating train data...", flush=True)
    train = make_dataset(args.train_n, args.points, args.scale_min, args.scale_max, args.exp_min, args.exp_max, device)

    print("Generating val data...", flush=True)
    val = make_dataset(args.val_n, args.points, args.scale_min, args.scale_max, args.exp_min, args.exp_max, device)

    train_loader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(*val), batch_size=args.batch_size, shuffle=False, num_workers=0)

    frame_net = FrameNet().to(device)
    opt = torch.optim.AdamW(frame_net.parameters(), lr=args.lr, weight_decay=1e-4)

    fields = [
        "epoch", "train_frame_loss",
        "frame_axis_angle_mean", "frame_axis_angle_p95",
        "gtcanon_scale_l1_mean", "gtcanon_exp_l1_mean", "gtcanon_cd_mean",
        "predcanon_scale_l1_mean", "predcanon_exp_l1_mean", "predcanon_cd_mean",
        "predcanon_cd_p95",
    ]

    best_pred_cd = 1e9

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            frame_net.train()
            total, seen = 0.0, 0

            for world, gt_R, _gt_scale, _gt_exp in train_loader:
                world = world.to(device)
                gt_R = gt_R.to(device)

                pred_R = frame_net(world)
                loss_per, _best_idx, _angles = frame_perm_loss(pred_R, gt_R)
                loss = loss_per.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += loss.item() * world.shape[0]
                seen += world.shape[0]

            va = evaluate(frame_net, shape_net, val_loader, dirs, args, device)

            row = {
                "epoch": ep,
                "train_frame_loss": total / max(seen, 1),
                "frame_axis_angle_mean": va["frame_axis_angle_mean"],
                "frame_axis_angle_p95": va["frame_axis_angle_p95"],
                "gtcanon_scale_l1_mean": va["gtcanon_scale_l1_mean"],
                "gtcanon_exp_l1_mean": va["gtcanon_exp_l1_mean"],
                "gtcanon_cd_mean": va["gtcanon_cd_mean"],
                "predcanon_scale_l1_mean": va["predcanon_scale_l1_mean"],
                "predcanon_exp_l1_mean": va["predcanon_exp_l1_mean"],
                "predcanon_cd_mean": va["predcanon_cd_mean"],
                "predcanon_cd_p95": va["predcanon_cd_p95"],
            }
            writer.writerow(row)
            f.flush()

            if va["predcanon_cd_mean"] < best_pred_cd:
                best_pred_cd = va["predcanon_cd_mean"]
                torch.save(
                    {"model": frame_net.state_dict(), "args": vars(args), "epoch": ep, "val": va},
                    out / "best_framenet.pt",
                )

            print(
                f"Epoch {ep:03d}/{args.epochs} "
                f"frame={va['frame_axis_angle_mean']:.2f}/{va['frame_axis_angle_p95']:.2f}deg "
                f"GTcd={va['gtcanon_cd_mean']:.6g} "
                f"PREDcd={va['predcanon_cd_mean']:.6g}/{va['predcanon_cd_p95']:.6g} "
                f"PREDexp={va['predcanon_exp_l1_mean']:.4f}",
                flush=True,
            )

    print("OUT", out)
    print(open(out / "metrics.csv").read().splitlines()[-1])


if __name__ == "__main__":
    main()
