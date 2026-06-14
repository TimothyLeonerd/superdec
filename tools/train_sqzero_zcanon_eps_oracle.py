#!/usr/bin/env python3
import argparse, csv, random
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


def frame_from_z(z):
    """Return local-to-world frame Q with Q[:,2] = z.
    Uses only z plus a fixed helper; yaw is arbitrary, not GT yaw.
    Row-vector local coords are world @ Q.
    """
    z = F.normalize(z, dim=0)
    helper = torch.tensor([0.0, 0.0, 1.0], device=z.device, dtype=z.dtype)
    if torch.abs(torch.dot(helper, z)) > 0.9:
        helper = torch.tensor([1.0, 0.0, 0.0], device=z.device, dtype=z.dtype)

    x = helper - torch.dot(helper, z) * z
    x = F.normalize(x, dim=0)
    y = torch.cross(z, x, dim=0)
    y = F.normalize(y, dim=0)
    x = torch.cross(y, z, dim=0)
    x = F.normalize(x, dim=0)

    return torch.stack([x, y, z], dim=1)  # columns = local axes in world


def build_primitive_tensor(args, split, max_objects):
    from superdec.data.sqzero_lmdb import SQZeroLMDB

    cfg = make_cfg(args, split, max_objects)
    ds = SQZeroLMDB(split=split, cfg=cfg)

    X, Y = [], []
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

            # GT center + scalar size normalization.
            size = torch.clamp(gt_scale[k].mean(), min=1e-6)
            centered = (p - gt_trans[k]) / size

            # Use only GT z-axis, not full GT rotation.
            z_gt = gt_rotate[k, :, 2]
            Q = frame_from_z(z_gt)
            local_zcanon = centered @ Q

            X.append(local_zcanon)
            Y.append(gt_shape[k])

    X = torch.stack(X)
    Y = torch.stack(Y)
    print(f"[{split}] primitives={len(X)} X={tuple(X.shape)} Y={tuple(Y.shape)}")
    return X, Y


class CanonicalPointNet(nn.Module):
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
            nn.Linear(128, 2),
        )

    def forward(self, x):
        h = self.point(x)
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
        raw = self.head(pooled)
        return 0.1 + 1.8 * torch.sigmoid(raw)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errs = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        errs.append((pred - y).abs().cpu())
    e = torch.cat(errs)
    per = e.mean(dim=1)
    return {
        "shape_l1_mean": float(per.mean()),
        "shape_l1_median": float(per.median()),
        "shape_l1_p90": float(torch.quantile(per, 0.90)),
        "shape_l1_p95": float(torch.quantile(per, 0.95)),
        "eps1_l1": float(e[:, 0].mean()),
        "eps2_l1": float(e[:, 1].mean()),
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Xtr, Ytr = build_primitive_tensor(args, "train", args.max_train_objects)
    Xva, Yva = build_primitive_tensor(args, "val", args.max_val_objects)

    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, Yva), batch_size=args.batch_size, shuffle=False)

    model = CanonicalPointNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fields = [
        "epoch", "train_loss",
        "train_shape_l1_mean", "train_shape_l1_p95",
        "val_shape_l1_mean", "val_shape_l1_median", "val_shape_l1_p90", "val_shape_l1_p95",
        "val_eps1_l1", "val_eps2_l1",
    ]

    best = 999
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total, seen = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = (pred - y).abs().mean()
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
                "train_shape_l1_mean": tr["shape_l1_mean"],
                "train_shape_l1_p95": tr["shape_l1_p95"],
                "val_shape_l1_mean": va["shape_l1_mean"],
                "val_shape_l1_median": va["shape_l1_median"],
                "val_shape_l1_p90": va["shape_l1_p90"],
                "val_shape_l1_p95": va["shape_l1_p95"],
                "val_eps1_l1": va["eps1_l1"],
                "val_eps2_l1": va["eps2_l1"],
            }
            writer.writerow(row)
            f.flush()

            if va["shape_l1_mean"] < best:
                best = va["shape_l1_mean"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": va}, out / "best.pt")

            print(
                f"Epoch {ep:03d}/{args.epochs} "
                f"train_l1={tr['shape_l1_mean']:.5f} "
                f"val_l1={va['shape_l1_mean']:.5f} "
                f"val_p95={va['shape_l1_p95']:.5f}",
                flush=True,
            )

    print(f"OUT={out}")
    print(open(out / "metrics.csv").read().splitlines()[-1])


if __name__ == "__main__":
    main()
