#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

import torch
import torch.nn.functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def rodrigues(rotvec):
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-8)
    k = rotvec / theta
    kx, ky, kz = k.unbind(-1)
    z = torch.zeros_like(kx)

    K = torch.stack([
        z, -kz, ky,
        kz, z, -kx,
        -ky, kx, z,
    ], dim=-1).reshape(-1, 3, 3)

    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(rotvec.shape[0], 3, 3)
    st = torch.sin(theta).view(-1, 1, 1)
    ct = torch.cos(theta).view(-1, 1, 1)
    return I + st * K + (1.0 - ct) * (K @ K)


def bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


def inv_sigmoid_param(x, lo, hi):
    y = (x - lo) / (hi - lo)
    y = torch.clamp(y, 1e-5, 1 - 1e-5)
    return torch.log(y / (1 - y))


def fexp(x, p):
    return torch.sign(x) * torch.abs(x).clamp_min(1e-8).pow(p)


def make_dirs_grid(surface_n, device):
    # choose approximately 16 x 32 = 512 by default
    n_eta = int(round(math.sqrt(surface_n / 2)))
    n_omega = max(8, surface_n // max(1, n_eta))
    eta0 = -math.pi / 2 + math.pi / (2 * n_eta)
    omega0 = -math.pi + math.pi / n_omega

    eta = eta0 + (math.pi / n_eta) * torch.arange(n_eta, device=device)
    omega = omega0 + (2 * math.pi / n_omega) * torch.arange(n_omega, device=device)
    eta, omega = torch.meshgrid(eta, omega, indexing="ij")
    return eta.reshape(-1), omega.reshape(-1)


def sample_classical_sq(scale, eps, eta, omega):
    # scale [B,3], eps [B,2], eta/omega [S]
    e1 = eps[:, 0:1]
    e2 = eps[:, 1:2]

    ce = torch.cos(eta)[None, :]
    se = torch.sin(eta)[None, :]
    co = torch.cos(omega)[None, :]
    so = torch.sin(omega)[None, :]

    x = scale[:, 0:1] * fexp(ce, e1) * fexp(co, e2)
    y = scale[:, 1:2] * fexp(ce, e1) * fexp(so, e2)
    z = scale[:, 2:3] * fexp(se, e1)

    return torch.stack([x, y, z], dim=-1)


def chamfer(a, b):
    d2 = torch.cdist(a, b).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


def geodesic_deg(Ra, Rb):
    R = Ra.transpose(1, 2) @ Rb
    tr = R.diagonal(dim1=1, dim2=2).sum(dim=1)
    c = ((tr - 1.0) / 2.0).clamp(-1, 1)
    return torch.acos(c) * (180.0 / math.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--surface-n", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--eps-min", type=float, default=0.3)
    ap.add_argument("--eps-max", type=float, default=1.7)
    ap.add_argument("--init", choices=["random", "neutral"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eta, omega = make_dirs_grid(args.surface_n, device)

    B = args.n

    gt_scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    gt_eps = args.eps_min + (args.eps_max - args.eps_min) * torch.rand(B, 2, device=device)
    gt_R = random_rotations(B, device)

    with torch.no_grad():
        gt_local = sample_classical_sq(gt_scale, gt_eps, eta, omega)
        gt_world = gt_local @ gt_R.transpose(1, 2)

    init_R = random_rotations(B, device)
    raw_rot = torch.zeros(B, 3, device=device, requires_grad=True)

    if args.init == "neutral":
        init_eps = torch.ones(B, 2, device=device)
    else:
        init_eps = args.eps_min + (args.eps_max - args.eps_min) * torch.rand(B, 2, device=device)

    init_scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)

    raw_eps = inv_sigmoid_param(init_eps, args.eps_min, args.eps_max).detach().clone().requires_grad_(True)
    raw_scale = inv_sigmoid_param(init_scale, args.scale_min, args.scale_max).detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([
        {"params": [raw_rot], "lr": args.lr},
        {"params": [raw_eps], "lr": args.lr},
        {"params": [raw_scale], "lr": args.lr},
    ])

    for step in range(1, args.steps + 1):
        pred_eps = bounded(raw_eps, args.eps_min, args.eps_max)
        pred_scale = bounded(raw_scale, args.scale_min, args.scale_max)
        pred_R = rodrigues(raw_rot) @ init_R

        pred_local = sample_classical_sq(pred_scale, pred_eps, eta, omega)
        pred_world = pred_local @ pred_R.transpose(1, 2)

        losses = chamfer(pred_world, gt_world)
        loss = losses.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            with torch.no_grad():
                eps_l1 = (pred_eps - gt_eps).abs().mean(dim=1)
                scale_l1 = (pred_scale - gt_scale).abs().mean(dim=1)
                rot = geodesic_deg(pred_R, gt_R)
                print(
                    f"step {step:04d}/{args.steps} "
                    f"loss={loss.item():.8g} "
                    f"loss_p95={torch.quantile(losses, .95).item():.8g} "
                    f"eps_l1={eps_l1.mean().item():.5f} "
                    f"scale_l1={scale_l1.mean().item():.5f} "
                    f"rot_mean={rot.mean().item():.2f} "
                    f"rot_p95={torch.quantile(rot, .95).item():.2f}",
                    flush=True,
                )

    with torch.no_grad():
        pred_eps = bounded(raw_eps, args.eps_min, args.eps_max)
        pred_scale = bounded(raw_scale, args.scale_min, args.scale_max)
        pred_R = rodrigues(raw_rot) @ init_R

        pred_local = sample_classical_sq(pred_scale, pred_eps, eta, omega)
        pred_world = pred_local @ pred_R.transpose(1, 2)

        losses = chamfer(pred_world, gt_world)
        eps_l1 = (pred_eps - gt_eps).abs().mean(dim=1)
        scale_l1 = (pred_scale - gt_scale).abs().mean(dim=1)
        rot = geodesic_deg(pred_R, gt_R)

    details = out / "details.csv"
    with open(details, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "loss", "eps_l1", "scale_l1", "rot_geodesic_deg",
            "gt_A", "gt_B", "gt_C", "gt_eps1", "gt_eps2",
            "pred_A", "pred_B", "pred_C", "pred_eps1", "pred_eps2",
        ])
        for i in range(B):
            w.writerow([
                i,
                float(losses[i].cpu()),
                float(eps_l1[i].cpu()),
                float(scale_l1[i].cpu()),
                float(rot[i].cpu()),
                *[float(x) for x in gt_scale[i].cpu()],
                *[float(x) for x in gt_eps[i].cpu()],
                *[float(x) for x in pred_scale[i].cpu()],
                *[float(x) for x in pred_eps[i].cpu()],
            ])

    summary = {
        "n": B,
        "loss_mean": float(losses.mean().cpu()),
        "loss_median": float(losses.median().cpu()),
        "loss_p90": float(torch.quantile(losses, .90).cpu()),
        "loss_p95": float(torch.quantile(losses, .95).cpu()),
        "loss_max": float(losses.max().cpu()),
        "succ_loss_lt_1e-6": float((losses < 1e-6).float().mean().cpu()),
        "succ_loss_lt_1e-5": float((losses < 1e-5).float().mean().cpu()),
        "succ_loss_lt_1e-4": float((losses < 1e-4).float().mean().cpu()),
        "eps_l1_mean": float(eps_l1.mean().cpu()),
        "eps_l1_p95": float(torch.quantile(eps_l1, .95).cpu()),
        "scale_l1_mean": float(scale_l1.mean().cpu()),
        "scale_l1_p95": float(torch.quantile(scale_l1, .95).cpu()),
        "rotdeg_mean": float(rot.mean().cpu()),
        "rotdeg_p95": float(torch.quantile(rot, .95).cpu()),
        "rotdeg_max": float(rot.max().cpu()),
    }

    summary_path = out / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("Wrote", details)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
