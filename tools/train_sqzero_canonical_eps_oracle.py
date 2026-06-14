#!/usr/bin/env python3
import argparse, csv, random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
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

    X, Y = [], []
    metas = []

    rng = np.random.default_rng(args.seed + (0 if split == "train" else 10000))

    for i in range(len(ds)):
        item = ds[i]
        pts = item["points"].float()          # [N,3], normalized world
        labels = item["labels"].long()        # [N]
        K = int(item["K"].item())

        gt_scale = item["gt_scale"].float()   # [Kmax,3]
        gt_shape = item["gt_shape"].float()   # [Kmax,2]
        gt_rotate = item["gt_rotate"].float() # [Kmax,3,3], local-to-world
        gt_trans = item["gt_trans"].float()   # [Kmax,3]

        for k in range(K):
            idx = torch.where(labels == k)[0]
            nvis = int(idx.numel())
            if nvis < args.min_visible_points:
                continue

            idx_np = idx.cpu().numpy()
            replace = len(idx_np) < args.prim_points
            chosen = rng.choice(idx_np, size=args.prim_points, replace=replace)
            p = pts[torch.from_numpy(chosen).long()]

            # row-vector convention:
            # world = local @ R.T + t
            # local = (world - t) @ R
            local = (p - gt_trans[k]) @ gt_rotate[k]
            local_norm = local / torch.clamp(gt_scale[k], min=1e-6)

            X.append(local_norm)
            Y.append(gt_shape[k])
            metas.append((item["model_id"], k, nvis))

    X = torch.stack(X, dim=0)
    Y = torch.stack(Y, dim=0)

    print(f"[{split}] primitives={len(X)} X={tuple(X.shape)} Y={tuple(Y.shape)}")
    return X, Y, metas


class CanonicalPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        h = self.point(x)              # [B,N,256]
        h_max = h.max(dim=1).values
        h_mean = h.mean(dim=1)
        raw = self.global_mlp(torch.cat([h_max, h_mean], dim=-1))
        return 0.1 + 1.8 * torch.sigmoid(raw)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    abs_errs = []
    mse_sum = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        abs_errs.append((pred - y).abs().cpu())
        mse_sum += ((pred - y) ** 2).sum().item()
        n += y.numel()
    e = torch.cat(abs_errs, dim=0)
    return {
        "shape_l1_mean": float(e.mean()),
        "shape_l1_median": float(e.mean(dim=1).median()),
        "eps1_l1": float(e[:, 0].mean()),
        "eps2_l1": float(e[:, 1].mean()),
        "shape_l1_p90": float(torch.quantile(e.mean(dim=1), 0.90)),
        "shape_l1_p95": float(torch.quantile(e.mean(dim=1), 0.95)),
        "mse": mse_sum / max(n, 1),
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

    Xtr, Ytr, _ = build_primitive_tensor(args, "train", args.max_train_objects)
    Xva, Yva, _ = build_primitive_tensor(args, "val", args.max_val_objects)

    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, Yva), batch_size=args.batch_size, shuffle=False)

    model = CanonicalPointNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    csv_path = out / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss",
            "train_shape_l1_mean", "train_shape_l1_p95",
            "val_shape_l1_mean", "val_shape_l1_median", "val_shape_l1_p90", "val_shape_l1_p95",
            "val_eps1_l1", "val_eps2_l1", "val_mse",
        ])
        writer.writeheader()

        best = 999.0
        for ep in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            seen = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = (pred - y).abs().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * x.shape[0]
                seen += x.shape[0]

            train_m = evaluate(model, train_loader, device)
            val_m = evaluate(model, val_loader, device)
            row = {
                "epoch": ep,
                "train_loss": total / max(seen, 1),
                "train_shape_l1_mean": train_m["shape_l1_mean"],
                "train_shape_l1_p95": train_m["shape_l1_p95"],
                "val_shape_l1_mean": val_m["shape_l1_mean"],
                "val_shape_l1_median": val_m["shape_l1_median"],
                "val_shape_l1_p90": val_m["shape_l1_p90"],
                "val_shape_l1_p95": val_m["shape_l1_p95"],
                "val_eps1_l1": val_m["eps1_l1"],
                "val_eps2_l1": val_m["eps2_l1"],
                "val_mse": val_m["mse"],
            }
            writer.writerow(row)
            f.flush()

            if val_m["shape_l1_mean"] < best:
                best = val_m["shape_l1_mean"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": val_m}, out / "best.pt")

            print(
                f"Epoch {ep:03d}/{args.epochs} "
                f"train_l1={train_m['shape_l1_mean']:.5f} "
                f"val_l1={val_m['shape_l1_mean']:.5f} "
                f"val_p95={val_m['shape_l1_p95']:.5f} "
                f"eps1={val_m['eps1_l1']:.5f} eps2={val_m['eps2_l1']:.5f}",
                flush=True,
            )

    print(f"Wrote {csv_path}")
    print("Final:")
    print(open(csv_path).read().splitlines()[-1])


if __name__ == "__main__":
    main()
