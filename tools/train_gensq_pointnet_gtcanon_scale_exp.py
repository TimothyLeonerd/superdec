#!/usr/bin/env python3
import argparse, csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
    PointNetScaleExp,
    evaluate,
)


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
def make_dataset_gtcanon(n, points, scale_min, scale_max, exp_min, exp_max, device, chunk=2048):
    dirs = fibonacci_sphere(points, device)
    all_pts, all_scale, all_exp = [], [], []
    max_err = 0.0

    for start in range(0, n, chunk):
        b = min(chunk, n - start)

        scale = scale_min + (scale_max - scale_min) * torch.rand(b, 3, device=device)
        exp = exp_min + (exp_max - exp_min) * torch.rand(b, 3, device=device)

        local = sample_generalized_surface(scale, exp, dirs)
        R = random_rotations(b, device)

        # convention: world = local @ R.T
        world = local @ R.transpose(1, 2)

        # GT-frame canonicalization check: local_recovered = world @ R
        canon = world @ R

        max_err = max(max_err, float((canon - local).abs().max()))

        all_pts.append(canon.cpu())
        all_scale.append(scale.cpu())
        all_exp.append(exp.cpu())

    print(f"max |GT-canon - original-local| = {max_err:.3e}", flush=True)

    return torch.cat(all_pts, 0), torch.cat(all_scale, 0), torch.cat(all_exp, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
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
    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--w-exp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs = fibonacci_sphere(args.points, device)

    print("Generating train data with rotate → GT-canonicalize...", flush=True)
    train = make_dataset_gtcanon(
        args.train_n, args.points,
        args.scale_min, args.scale_max,
        args.exp_min, args.exp_max,
        device,
    )

    print("Generating val data with rotate → GT-canonicalize...", flush=True)
    val = make_dataset_gtcanon(
        args.val_n, args.points,
        args.scale_min, args.scale_max,
        args.exp_min, args.exp_max,
        device,
    )

    train_loader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(*val), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PointNetScaleExp(args.scale_min, args.scale_max, args.exp_min, args.exp_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fields = [
        "epoch", "train_loss", "val_loss",
        "val_scale_l1_mean", "val_scale_l1_p95",
        "val_exp_l1_mean", "val_exp_l1_p95",
        "val_cd_mean", "val_cd_p95",
    ]

    best_cd = 1e9

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total, seen = 0.0, 0

            for pts, gt_scale, gt_exp in train_loader:
                pts = pts.to(device)
                gt_scale = gt_scale.to(device)
                gt_exp = gt_exp.to(device)

                pred_scale, pred_exp = model(pts)

                scale_l1 = (pred_scale - gt_scale).abs().mean(dim=1)
                exp_l1 = (pred_exp - gt_exp).abs().mean(dim=1)

                scale_loss = scale_l1 / (args.scale_max - args.scale_min)
                exp_loss = exp_l1 / (args.exp_max - args.exp_min)

                loss = (args.w_scale * scale_loss + args.w_exp * exp_loss).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += loss.item() * pts.shape[0]
                seen += pts.shape[0]

            va = evaluate(model, val_loader, dirs, args, device)

            row = {
                "epoch": ep,
                "train_loss": total / max(seen, 1),
                "val_loss": va["loss"],
                "val_scale_l1_mean": va["scale_l1_mean"],
                "val_scale_l1_p95": va["scale_l1_p95"],
                "val_exp_l1_mean": va["exp_l1_mean"],
                "val_exp_l1_p95": va["exp_l1_p95"],
                "val_cd_mean": va["cd_mean"],
                "val_cd_p95": va["cd_p95"],
            }
            writer.writerow(row)
            f.flush()

            if va["cd_mean"] < best_cd:
                best_cd = va["cd_mean"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": va}, out / "best.pt")

            print(
                f"Epoch {ep:03d}/{args.epochs} "
                f"scale={va['scale_l1_mean']:.5f}/{va['scale_l1_p95']:.5f} "
                f"exp={va['exp_l1_mean']:.5f}/{va['exp_l1_p95']:.5f} "
                f"cd={va['cd_mean']:.7g}/{va['cd_p95']:.7g}",
                flush=True,
            )

    print("OUT", out)
    print(open(out / "metrics.csv").read().splitlines()[-1])


if __name__ == "__main__":
    main()
