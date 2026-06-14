#!/usr/bin/env python3
import argparse, csv, math, random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ConfigNode(SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)


def make_cfg(args, split, max_samples):
    return ConfigNode(
        sqzero_lmdb=ConfigNode(
            path=args.data_root,
            n_points=args.n_points,
            normalize=True,
            load_sidecars=True,
            normal_mode="radial",
            kmax=args.kmax,
            train_split=args.train_split,
            val_split=args.val_split,
            max_train_samples=max_samples if split == "train" else None,
            max_val_samples=max_samples if split == "val" else None,
        ),
        trainer=ConfigNode(augmentations=False),
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_primitive_tensor(args, split, max_objects):
    from superdec.data.sqzero_lmdb import SQZeroLMDB

    cfg = make_cfg(args, split, max_objects)
    ds = SQZeroLMDB(split=split, cfg=cfg)

    X, Y_eps, Y_z = [], [], []
    rng = np.random.default_rng(args.seed + (0 if split == "train" else 10000))

    for i in range(len(ds)):
        item = ds[i]
        pts = item["points"].float()
        labels = item["labels"].long()
        K = int(item["K"].item())

        gt_scale = item["gt_scale"].float()
        gt_shape = item["gt_shape"].float()
        gt_rotate = item["gt_rotate"].float()
        gt_trans = item["gt_trans"].float()

        for k in range(K):
            idx = torch.where(labels == k)[0]
            if int(idx.numel()) < args.min_visible_points:
                continue

            idx_np = idx.cpu().numpy()
            chosen = rng.choice(idx_np, size=args.prim_points, replace=len(idx_np) < args.prim_points)
            p = pts[torch.from_numpy(chosen).long()]

            # No GT rotation. Center using GT translation, normalize by scalar size only.
            size = torch.clamp(gt_scale[k].mean(), min=1e-6)
            x = (p - gt_trans[k]) / size

            z = gt_rotate[k, :, 2]
            z = F.normalize(z, dim=0)

            X.append(x)
            Y_eps.append(gt_shape[k])
            Y_z.append(z)

    X = torch.stack(X)
    Y_eps = torch.stack(Y_eps)
    Y_z = torch.stack(Y_z)
    print(f"[{split}] primitives={len(X)} X={tuple(X.shape)}")
    return X, Y_eps, Y_z


class PointNetZAxisEps(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        h = self.point(x)
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
        raw = self.head(pooled)
        eps = 0.1 + 1.8 * torch.sigmoid(raw[:, :2])
        z = F.normalize(raw[:, 2:], dim=-1)
        return eps, z


def z_axis_loss(z_pred, z_gt):
    dot = torch.sum(z_pred * z_gt, dim=-1).clamp(-1, 1)
    return 1.0 - dot * dot


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    eps_errs, z_losses = [], []
    for x, eps, z in loader:
        x, eps, z = x.to(device), eps.to(device), z.to(device)
        pred_eps, pred_z = model(x)
        eps_errs.append((pred_eps - eps).abs().cpu())
        z_losses.append(z_axis_loss(pred_z, z).cpu())

    e = torch.cat(eps_errs)
    zl = torch.cat(z_losses)
    angles = torch.asin(torch.sqrt(torch.clamp(zl, 0, 1))) * (180.0 / math.pi)

    return {
        "eps_l1_mean": float(e.mean()),
        "eps_l1_median": float(e.mean(dim=1).median()),
        "eps_l1_p95": float(torch.quantile(e.mean(dim=1), 0.95)),
        "eps1_l1": float(e[:, 0].mean()),
        "eps2_l1": float(e[:, 1].mean()),
        "z_loss_mean": float(zl.mean()),
        "z_angle_mean": float(angles.mean()),
        "z_angle_median": float(angles.median()),
        "z_angle_p90": float(torch.quantile(angles, 0.90)),
        "z_angle_p95": float(torch.quantile(angles, 0.95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-split", default="train.txt")
    ap.add_argument("--val-split", default="test.txt")
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--prim-points", type=int, default=256)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--min-visible-points", type=int, default=32)
    ap.add_argument("--max-train-objects", type=int, default=9000)
    ap.add_argument("--max-val-objects", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-z", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Xtr, Etr, Ztr = build_primitive_tensor(args, "train", args.max_train_objects)
    Xva, Eva, Zva = build_primitive_tensor(args, "val", args.max_val_objects)

    train_loader = DataLoader(TensorDataset(Xtr, Etr, Ztr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, Eva, Zva), batch_size=args.batch_size, shuffle=False)

    model = PointNetZAxisEps().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fields = [
        "epoch", "train_loss",
        "train_eps_l1_mean", "train_z_angle_mean", "train_z_angle_p95",
        "val_eps_l1_mean", "val_eps_l1_median", "val_eps_l1_p95",
        "val_eps1_l1", "val_eps2_l1",
        "val_z_loss_mean", "val_z_angle_mean", "val_z_angle_median",
        "val_z_angle_p90", "val_z_angle_p95",
    ]

    best = 999
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total, seen = 0.0, 0
            for x, eps, z in train_loader:
                x, eps, z = x.to(device), eps.to(device), z.to(device)
                pred_eps, pred_z = model(x)
                loss_eps = (pred_eps - eps).abs().mean()
                loss_z = z_axis_loss(pred_z, z).mean()
                loss = loss_eps + args.w_z * loss_z

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += loss.item() * x.shape[0]
                seen += x.shape[0]

            tr = evaluate(model, train_loader, device)
            va = evaluate(model, val_loader, device)

            row = {
                "epoch": ep,
                "train_loss": total / max(seen, 1),
                "train_eps_l1_mean": tr["eps_l1_mean"],
                "train_z_angle_mean": tr["z_angle_mean"],
                "train_z_angle_p95": tr["z_angle_p95"],
                "val_eps_l1_mean": va["eps_l1_mean"],
                "val_eps_l1_median": va["eps_l1_median"],
                "val_eps_l1_p95": va["eps_l1_p95"],
                "val_eps1_l1": va["eps1_l1"],
                "val_eps2_l1": va["eps2_l1"],
                "val_z_loss_mean": va["z_loss_mean"],
                "val_z_angle_mean": va["z_angle_mean"],
                "val_z_angle_median": va["z_angle_median"],
                "val_z_angle_p90": va["z_angle_p90"],
                "val_z_angle_p95": va["z_angle_p95"],
            }
            writer.writerow(row)
            f.flush()

            if va["eps_l1_mean"] + 0.01 * va["z_angle_mean"] < best:
                best = va["eps_l1_mean"] + 0.01 * va["z_angle_mean"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": va}, out / "best.pt")

            print(
                f"Epoch {ep:03d}/{args.epochs} "
                f"val_eps={va['eps_l1_mean']:.5f} "
                f"val_z_angle={va['z_angle_mean']:.2f} "
                f"val_z_p95={va['z_angle_p95']:.2f}",
                flush=True,
            )

    print(f"OUT={out}")
    print(open(out / "metrics.csv").read().splitlines()[-1])


if __name__ == "__main__":
    main()
