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
    device, dtype = eps.device, eps.dtype
    eta0 = -np.pi / 2 + np.pi / (2 * n_eta)
    omega0 = -np.pi + np.pi / n_omega
    eta = eta0 + (np.pi / n_eta) * torch.arange(n_eta, device=device, dtype=dtype)
    omega = omega0 + (2 * np.pi / n_omega) * torch.arange(n_omega, device=device, dtype=dtype)
    eta, omega = torch.meshgrid(eta, omega, indexing="ij")
    eta = eta.reshape(1, -1)
    omega = omega.reshape(1, -1)
    e1, e2 = eps[:, 0:1], eps[:, 1:2]
    x = fexp(torch.cos(eta), e1) * fexp(torch.cos(omega), e2)
    y = fexp(torch.cos(eta), e1) * fexp(torch.sin(omega), e2)
    z = fexp(torch.sin(eta), e1)
    return torch.stack([x, y, z], dim=-1)


def chamfer_sq(a, b):
    d2 = torch.cdist(a, b).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


def frame_from_z_batch(z):
    z = F.normalize(z, dim=-1)
    B = z.shape[0]
    helper0 = torch.tensor([0., 0., 1.], device=z.device, dtype=z.dtype).expand(B, 3)
    helper1 = torch.tensor([1., 0., 0.], device=z.device, dtype=z.dtype).expand(B, 3)
    use_alt = (helper0 * z).sum(dim=-1).abs() > 0.9
    helper = torch.where(use_alt[:, None], helper1, helper0)

    x = helper - ((helper * z).sum(dim=-1, keepdim=True) * z)
    x = F.normalize(x, dim=-1)
    y = torch.cross(z, x, dim=-1)
    y = F.normalize(y, dim=-1)
    x = torch.cross(y, z, dim=-1)
    x = F.normalize(x, dim=-1)

    return torch.stack([x, y, z], dim=-1)


def build_tensors(args, split, max_objects):
    from superdec.data.sqzero_lmdb import SQZeroLMDB
    ds = SQZeroLMDB(split=split, cfg=make_cfg(args, split, max_objects))

    X, Yeps, Yz = [], [], []
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

            size = torch.clamp(gt_scale[k].mean(), min=1e-6)
            x = (p - gt_trans[k]) / size

            X.append(x)
            Yeps.append(gt_shape[k])
            Yz.append(F.normalize(gt_rotate[k, :, 2], dim=0))

    return torch.stack(X), torch.stack(Yeps), torch.stack(Yz)


class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )

    def forward(self, x):
        h = self.point(x)
        return torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)


class PredZCanonEpsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_enc = PointNetEncoder()
        self.z_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.eps_enc = PointNetEncoder()
        self.eps_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        z = F.normalize(self.z_head(self.z_enc(x)), dim=-1)
        Q = frame_from_z_batch(z)
        x_canon = torch.einsum("bnd,bdk->bnk", x, Q)
        raw_eps = self.eps_head(self.eps_enc(x_canon))
        eps = 0.1 + 1.8 * torch.sigmoid(raw_eps)
        return eps, z


def z_axis_loss(z_pred, z_gt):
    dot = (z_pred * z_gt).sum(dim=-1).clamp(-1, 1)
    return 1.0 - dot * dot


@torch.no_grad()
def evaluate(model, loader, device, n_eta, n_omega):
    model.eval()
    eps_errs, cds, z_losses = [], [], []
    for x, eps_gt, z_gt in loader:
        x, eps_gt, z_gt = x.to(device), eps_gt.to(device), z_gt.to(device)
        eps_pred, z_pred = model(x)
        pred_surf = make_unit_sq_surface(eps_pred, n_eta, n_omega)
        gt_surf = make_unit_sq_surface(eps_gt, n_eta, n_omega)
        cds.append(chamfer_sq(pred_surf, gt_surf).cpu())
        eps_errs.append((eps_pred - eps_gt).abs().cpu())
        z_losses.append(z_axis_loss(z_pred, z_gt).cpu())

    e = torch.cat(eps_errs)
    per = e.mean(dim=1)
    cd = torch.cat(cds)
    zl = torch.cat(z_losses)
    angles = torch.asin(torch.sqrt(torch.clamp(zl, 0, 1))) * (180.0 / math.pi)

    return {
        "cd_mean": float(cd.mean()),
        "cd_p95": float(torch.quantile(cd, 0.95)),
        "eps_l1_mean": float(per.mean()),
        "eps_l1_median": float(per.median()),
        "eps_l1_p90": float(torch.quantile(per, 0.90)),
        "eps_l1_p95": float(torch.quantile(per, 0.95)),
        "eps1_l1": float(e[:, 0].mean()),
        "eps2_l1": float(e[:, 1].mean()),
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

    Xtr, Etr, Ztr = build_tensors(args, "train", args.max_train_objects)
    Xva, Eva, Zva = build_tensors(args, "val", args.max_val_objects)
    print(f"[train] {tuple(Xtr.shape)} [val] {tuple(Xva.shape)}")

    train_loader = DataLoader(TensorDataset(Xtr, Etr, Ztr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, Eva, Zva), batch_size=args.batch_size, shuffle=False)

    model = PredZCanonEpsNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fields = [
        "epoch", "train_cd",
        "val_cd_mean", "val_cd_p95",
        "val_eps_l1_mean", "val_eps_l1_median", "val_eps_l1_p90", "val_eps_l1_p95",
        "val_eps1_l1", "val_eps2_l1",
        "val_z_angle_mean", "val_z_angle_median", "val_z_angle_p90", "val_z_angle_p95",
    ]

    best = 999
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total, seen = 0.0, 0

            for x, eps_gt, _z_gt in train_loader:
                x, eps_gt = x.to(device), eps_gt.to(device)
                eps_pred, _z_pred = model(x)
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
                "val_z_angle_mean": va["z_angle_mean"],
                "val_z_angle_median": va["z_angle_median"],
                "val_z_angle_p90": va["z_angle_p90"],
                "val_z_angle_p95": va["z_angle_p95"],
            }
            writer.writerow(row); f.flush()

            if va["cd_mean"] < best:
                best = va["cd_mean"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": va}, out / "best.pt")

            print(
                f"Epoch {ep:03d}/{args.epochs} "
                f"val_cd={va['cd_mean']:.8g} "
                f"val_eps={va['eps_l1_mean']:.5f} "
                f"val_p95={va['eps_l1_p95']:.5f} "
                f"z_mean={va['z_angle_mean']:.2f}",
                flush=True,
            )

    print(f"OUT={out}")
    print(open(out / "metrics.csv").read().splitlines()[-1])


if __name__ == "__main__":
    main()
