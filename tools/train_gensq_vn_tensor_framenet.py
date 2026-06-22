#!/usr/bin/env python3
import argparse, csv, itertools, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from third_party.vnn.models.vn_layers import VNLinearLeakyReLU
from third_party.vnn.models.utils.vn_dgcnn_util import get_graph_feature_cross

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
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


def project_to_so3(M):
    U, _S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    det = torch.det(R)

    D = torch.eye(3, device=M.device, dtype=M.dtype).expand(M.shape[0], 3, 3).clone()
    D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

    return U @ D @ Vh


def second_order_pool(x):
    # x: [B,C,3,N]
    # returns normalized tensor: [B,3,3]
    T = torch.einsum("bcin,bcjn->bij", x, x)
    T = T / max(x.shape[1] * x.shape[-1], 1)
    T = 0.5 * (T + T.transpose(1, 2))
    T = T / T.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]
    return T


@torch.no_grad()
def raw_cov_tensor(points):
    x = points - points.mean(dim=1, keepdim=True)
    T = x.transpose(1, 2) @ x / max(points.shape[1] - 1, 1)
    T = 0.5 * (T + T.transpose(1, 2))
    T = T / T.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]
    return T


class VNTensorFrameNet(nn.Module):
    def __init__(self, n_knn=20, hidden=128, use_raw_cov=False):
        super().__init__()
        self.n_knn = n_knn
        self.use_raw_cov = use_raw_cov

        c1 = 32
        c2 = 64
        c3 = 96

        self.conv_pos = VNLinearLeakyReLU(3, c1, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(c1, c1, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(c1, c2, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(c2, c3, dim=4, negative_slope=0.0)

        n_tensors = 3 + int(use_raw_cov)
        in_dim = 9 * n_tensors

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, points):
        # points: [B,N,3]
        x0 = points.transpose(1, 2).unsqueeze(1)  # [B,1,3,N]

        g = get_graph_feature_cross(x0, k=self.n_knn)  # [B,3,3,N,k]
        x = self.conv_pos(g).mean(dim=-1)              # [B,C,3,N]

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        tensors = [
            second_order_pool(x1),
            second_order_pool(x2),
            second_order_pool(x3),
        ]

        if self.use_raw_cov:
            tensors.append(raw_cov_tensor(points))

        feat = torch.cat([T.flatten(1) for T in tensors], dim=1)
        M = self.head(feat).reshape(-1, 3, 3)
        R = project_to_so3(M)
        return R


def frame_perm_loss(pred_R, gt_R):
    losses, angles = [], []

    for p in PERMS:
        gt = gt_R[:, :, list(p)]
        dot = (pred_R * gt).sum(dim=1).abs().clamp(0, 1)
        loss = (1.0 - dot.pow(2)).mean(dim=1)
        angle = torch.acos(dot) * (180.0 / math.pi)
        losses.append(loss)
        angles.append(angle)

    losses = torch.stack(losses, dim=1)
    angles = torch.stack(angles, dim=1)

    best_idx = losses.argmin(dim=1)
    best_loss = losses.gather(1, best_idx[:, None]).squeeze(1)
    best_angles = angles.gather(1, best_idx[:, None, None].expand(-1, 1, 3)).squeeze(1)

    return best_loss, best_idx, best_angles


@torch.no_grad()
def pca_frame(points):
    x = points - points.mean(dim=1, keepdim=True)
    cov = x.transpose(1, 2) @ x / max(points.shape[1] - 1, 1)

    evals, evecs = torch.linalg.eigh(cov)
    order = evals.argsort(dim=1, descending=True)
    evecs = evecs.gather(2, order[:, None, :].expand(-1, 3, -1))

    det = torch.det(evecs)
    evecs[:, :, 2] *= torch.where(det < 0, -1.0, 1.0)[:, None]
    return evecs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_all, angle_all = [], []
    pca_loss_all, pca_angle_all = [], []

    for world, gt_R, _scale, _exp in loader:
        world = world.to(device)
        gt_R = gt_R.to(device)

        pred_R = model(world)
        loss, _idx, angles = frame_perm_loss(pred_R, gt_R)

        pca_R = pca_frame(world)
        pca_loss, _pca_idx, pca_angles = frame_perm_loss(pca_R, gt_R)

        loss_all.append(loss.cpu())
        angle_all.append(angles.reshape(-1).cpu())
        pca_loss_all.append(pca_loss.cpu())
        pca_angle_all.append(pca_angles.reshape(-1).cpu())

    loss = torch.cat(loss_all)
    angle = torch.cat(angle_all)
    pca_loss = torch.cat(pca_loss_all)
    pca_angle = torch.cat(pca_angle_all)

    return {
        "frame_loss_mean": float(loss.mean()),
        "axis_angle_mean": float(angle.mean()),
        "axis_angle_p95": float(torch.quantile(angle, 0.95)),
        "pca_frame_loss_mean": float(pca_loss.mean()),
        "pca_axis_angle_mean": float(pca_angle.mean()),
        "pca_axis_angle_p95": float(torch.quantile(pca_angle, 0.95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-n", type=int, default=10000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-knn", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--use-raw-cov", action="store_true")
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
    if device.type != "cuda":
        raise RuntimeError("vendored get_graph_feature_cross expects CUDA")

    print("Generating train data...", flush=True)
    train = make_dataset(
        args.train_n, args.points,
        args.scale_min, args.scale_max,
        args.exp_min, args.exp_max,
        device,
    )

    print("Generating val data...", flush=True)
    val = make_dataset(
        args.val_n, args.points,
        args.scale_min, args.scale_max,
        args.exp_min, args.exp_max,
        device,
    )

    train_loader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(*val), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VNTensorFrameNet(
        n_knn=args.n_knn,
        hidden=args.hidden,
        use_raw_cov=args.use_raw_cov,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fields = [
        "epoch", "train_loss",
        "val_frame_loss_mean",
        "val_axis_angle_mean", "val_axis_angle_p95",
        "pca_frame_loss_mean",
        "pca_axis_angle_mean", "pca_axis_angle_p95",
    ]

    best = 1e9

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total, seen = 0.0, 0

            for world, gt_R, _scale, _exp in train_loader:
                world = world.to(device)
                gt_R = gt_R.to(device)

                pred_R = model(world)
                loss_per, _best_idx, _angles = frame_perm_loss(pred_R, gt_R)
                loss = loss_per.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += loss.item() * world.shape[0]
                seen += world.shape[0]

            va = evaluate(model, val_loader, device)

            row = {
                "epoch": ep,
                "train_loss": total / max(seen, 1),
                "val_frame_loss_mean": va["frame_loss_mean"],
                "val_axis_angle_mean": va["axis_angle_mean"],
                "val_axis_angle_p95": va["axis_angle_p95"],
                "pca_frame_loss_mean": va["pca_frame_loss_mean"],
                "pca_axis_angle_mean": va["pca_axis_angle_mean"],
                "pca_axis_angle_p95": va["pca_axis_angle_p95"],
            }
            writer.writerow(row)
            f.flush()

            if va["axis_angle_mean"] < best:
                best = va["axis_angle_mean"]
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": va},
                    out / "best.pt",
                )

            print(
                f"Epoch {ep:03d}/{args.epochs} "
                f"loss={row['train_loss']:.6f} "
                f"axis={va['axis_angle_mean']:.3f}/{va['axis_angle_p95']:.3f}deg "
                f"pca={va['pca_axis_angle_mean']:.3f}/{va['pca_axis_angle_p95']:.3f}deg",
                flush=True,
            )

    print("OUT", out)
    print(open(out / "metrics.csv").read().splitlines()[-1])


if __name__ == "__main__":
    main()
