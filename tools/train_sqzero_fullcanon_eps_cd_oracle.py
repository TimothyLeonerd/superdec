#!/usr/bin/env python3
import argparse, csv, random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ConfigNode(SimpleNamespace):
    def __contains__(self, key): return hasattr(self, key)


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
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def fexp(x, p):
    return torch.sign(x) * torch.abs(x).clamp_min(1e-8).pow(p)


def make_unit_sq_surface(eps, n_eta=16, n_omega=32):
    # eps: [B,2], output [B,S,3], unit scale, canonical frame
    device, dtype = eps.device, eps.dtype
    eta0 = -np.pi / 2 + np.pi / (2 * n_eta)
    omega0 = -np.pi + np.pi / n_omega
    eta = eta0 + (np.pi / n_eta) * torch.arange(n_eta, device=device, dtype=dtype)
    omega = omega0 + (2 * np.pi / n_omega) * torch.arange(n_omega, device=device, dtype=dtype)
    eta, omega = torch.meshgrid(eta, omega, indexing="ij")
    eta = eta.reshape(1, -1)
    omega = omega.reshape(1, -1)

    e1 = eps[:, 0:1]
    e2 = eps[:, 1:2]
    x = fexp(torch.cos(eta), e1) * fexp(torch.cos(omega), e2)
    y = fexp(torch.cos(eta), e1) * fexp(torch.sin(omega), e2)
    z = fexp(torch.sin(eta), e1)
    return torch.stack([x, y, z], dim=-1)


def chamfer_sq(a, b):
    # a,b: [B,S,3]
    d2 = torch.cdist(a, b).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


def build_tensors(args, split, max_objects):
    from superdec.data.sqzero_lmdb import SQZeroLMDB
    ds = SQZeroLMDB(split=split, cfg=make_cfg(args, split, max_objects))

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

            # Full GT canonicalization:
            # local = (world - t) @ R, then componentwise divide by local scale.
            local = (p - gt_trans[k]) @ gt_rotate[k]
            local = local / torch.clamp(gt_scale[k], min=1e-6)

            X.append(local)
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
def evaluate(model, loader, device, n_eta, n_omega):
    model.eval()
    eps_errs, cds = [], []
    for x, eps_gt in loader:
        x, eps_gt = x.to(device), eps_gt.to(device)
        eps_pred = model(x)
        pred_surf = make_unit_sq_surface(eps_pred, n_eta, n_omega)
        gt_surf = make_unit_sq_surface(eps_gt, n_eta, n_omega)
        cds.append(chamfer_sq(pred_surf, gt_surf).cpu())
        eps_errs.append((eps_pred - eps_gt).abs().cpu())

    e = torch.cat(eps_errs)
    per = e.mean(dim=1)
    cd = torch.cat(cds)
    return {
        "eps_l1_mean": float(per.mean()),
        "eps_l1_median": float(per.median()),
        "eps_l1_p90": float(torch.quantile(per, 0.90)),
        "eps_l1_p95": float(torch.quantile(per, 0.95)),
        "eps1_l1": float(e[:, 0].mean()),
        "eps2_l1": float(e[:, 1].mean()),
        "cd_mean": float(cd.mean()),
        "cd_p95": float(torch.quantile(cd, 0.95)),
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
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--surf-eta", type=int, default=16)
    ap.add_argument("--surf-omega", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Xtr, Ytr = build_tensors(args, "train", args.max_train_objects)
    Xva, Yva = build_tensors(args, "val", args.max_val_objects)

    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, Yva), batch_size=args.batch_size, shuffle=False)

    model = CanonicalPointNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fields = [
        "epoch", "train_cd",
        "val_cd_mean", "val_cd_p95",
        "val_eps_l1_mean", "val_eps_l1_median", "val_eps_l1_p90", "val_eps_l1_p95",
        "val_eps1_l1", "val_eps2_l1",
    ]

    best = 999
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total, seen = 0.0, 0
            for x, eps_gt in train_loader:
                x, eps_gt = x.to(device), eps_gt.to(device)
                eps_pred = model(x)
                pred_surf = make_unit_sq_surface(eps_pred, args.surf_eta, args.surf_omega)
                gt_surf = make_unit_sq_surface(eps_gt, args.surf_eta, args.surf_omega)
                loss = chamfer_sq(pred_surf, gt_surf).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += loss.item() * x.shape[0]
                seen += x.shape[0]

            va = evaluate(model, val_loader, device, args.surf_eta, args.surf_omega)
            row = {
                "epoch": ep,
                "train_cd": total / max(seen, 1),
                "val_cd_mean": va["cd_mean"],
                "val_cd_p95": va["cd_p95"],
                "val_eps_l1_mean": va["eps_l1_mean"],
                "val_eps_l1_median": va["eps_l1_median"],
                "val_eps_l1_p90": va["eps_l1_p90"],
                "val_eps_l1_p95": va["eps_l1_p95"],
                "val_eps1_l1": va["eps1_l1"],
                "val_eps2_l1": va["eps2_l1"],
            }
            writer.writerow(row); f.flush()

            if va["cd_mean"] < best:
                best = va["cd_mean"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": va}, out / "best.pt")

            print(
                f"Epoch {ep:03d}/{args.epochs} "
                f"val_cd={va['cd_mean']:.8g} "
                f"val_eps={va['eps_l1_mean']:.5f} "
                f"val_p95={va['eps_l1_p95']:.5f}",
                flush=True,
            )

    print(f"OUT={out}")
    print(open(out / "metrics.csv").read().splitlines()[-1])


if __name__ == "__main__":
    main()
