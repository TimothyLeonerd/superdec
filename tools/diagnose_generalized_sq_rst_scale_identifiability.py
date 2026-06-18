#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

import torch
import torch.nn.functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fibonacci_sphere(n, device, dtype=torch.float32):
    i = torch.arange(n, device=device, dtype=dtype)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = phi * i
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    return torch.stack([x, y, z], dim=-1)


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


def sample_generalized_surface(scale, exp, dirs, newton_iters=32):
    """
    Robust radial sampler for the generalized superquadric surface

        |x/A|^r + |y/B|^s + |z/C|^t = 1

    along directions `dirs`.

    `newton_iters` is kept for API compatibility, but is now used as the
    number of bracketed bisection iterations. This avoids Newton overshoot
    for large exponents / small scales and guarantees |x_i| <= scale_i.
    """
    B = scale.shape[0]
    S = dirs.shape[0]

    dtype = scale.dtype
    device = scale.device

    u = dirs.to(device=device, dtype=dtype)[None].expand(B, S, 3)
    A = scale[:, None, :].clamp_min(1e-8)
    e = exp[:, None, :].clamp_min(0.05)

    abs_u = u.abs()

    # For x = rho * u, coordinate validity requires
    # rho <= A_i / |u_i| for every nonzero direction component.
    # Thus the true root is bracketed in [0, min_i A_i/|u_i|].
    huge = torch.full_like(abs_u, 1e8)
    rho_axis_hi = torch.where(abs_u > 1e-12, A / abs_u.clamp_min(1e-12), huge)
    rho_hi = rho_axis_hi.min(dim=-1, keepdim=True).values.clamp_min(1e-12)
    rho_lo = torch.zeros_like(rho_hi)

    coeff = (abs_u.clamp_min(1e-12) / A).pow(e)

    for _ in range(int(newton_iters)):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        f_mid = (coeff * rho_mid.clamp_min(1e-12).pow(e)).sum(dim=-1, keepdim=True) - 1.0

        # f(rho) is monotone increasing. Keep the root bracketed.
        too_high = f_mid >= 0.0
        rho_hi = torch.where(too_high, rho_mid, rho_hi)
        rho_lo = torch.where(too_high, rho_lo, rho_mid)

    rho = 0.5 * (rho_lo + rho_hi)
    return u * rho
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
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=0.3)
    ap.add_argument("--exp-max", type=float, default=1.7)
    ap.add_argument("--init", choices=["random", "neutral"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs = fibonacci_sphere(args.surface_n, device)

    B = args.n

    gt_scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    gt_exp = args.exp_min + (args.exp_max - args.exp_min) * torch.rand(B, 3, device=device)
    gt_R = random_rotations(B, device)

    with torch.no_grad():
        gt_local = sample_generalized_surface(gt_scale, gt_exp, dirs)
        gt_world = gt_local @ gt_R.transpose(1, 2)

    init_R = random_rotations(B, device)
    raw_rot = torch.zeros(B, 3, device=device, requires_grad=True)

    if args.init == "neutral":
        init_exp = torch.ones(B, 3, device=device)
    else:
        init_exp = args.exp_min + (args.exp_max - args.exp_min) * torch.rand(B, 3, device=device)

    raw_exp = inv_sigmoid_param(init_exp, args.exp_min, args.exp_max).detach().clone().requires_grad_(True)

    init_scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    raw_scale = inv_sigmoid_param(init_scale, args.scale_min, args.scale_max).detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([
        {"params": [raw_rot], "lr": args.lr},
        {"params": [raw_exp], "lr": args.lr},
        {"params": [raw_scale], "lr": args.lr},
    ])

    for step in range(1, args.steps + 1):
        pred_exp = bounded(raw_exp, args.exp_min, args.exp_max)
        pred_scale = bounded(raw_scale, args.scale_min, args.scale_max)
        dR = rodrigues(raw_rot)
        pred_R = dR @ init_R

        pred_local = sample_generalized_surface(pred_scale, pred_exp, dirs)
        pred_world = pred_local @ pred_R.transpose(1, 2)

        losses = chamfer(pred_world, gt_world)
        loss = losses.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            with torch.no_grad():
                exp_l1 = (pred_exp - gt_exp).abs().mean(dim=1)
                scale_l1 = (pred_scale - gt_scale).abs().mean(dim=1)
                rot = geodesic_deg(pred_R, gt_R)
                print(
                    f"step {step:04d}/{args.steps} "
                    f"loss={loss.item():.8g} "
                    f"loss_p95={torch.quantile(losses, .95).item():.8g} "
                    f"exp_l1={exp_l1.mean().item():.5f} "
                    f"scale_l1={scale_l1.mean().item():.5f} "
                    f"rot_mean={rot.mean().item():.2f} rot_p95={torch.quantile(rot, .95).item():.2f}",
                    flush=True,
                )

    with torch.no_grad():
        pred_exp = bounded(raw_exp, args.exp_min, args.exp_max)
        pred_scale = bounded(raw_scale, args.scale_min, args.scale_max)
        pred_R = rodrigues(raw_rot) @ init_R
        pred_local = sample_generalized_surface(pred_scale, pred_exp, dirs)
        pred_world = pred_local @ pred_R.transpose(1, 2)
        losses = chamfer(pred_world, gt_world)
        exp_l1 = (pred_exp - gt_exp).abs().mean(dim=1)
        scale_l1 = (pred_scale - gt_scale).abs().mean(dim=1)
        rot = geodesic_deg(pred_R, gt_R)

    details = out / "details.csv"
    with open(details, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "loss", "exp_l1", "rot_geodesic_deg",
            "gt_A", "gt_B", "gt_C", "gt_r", "gt_s", "gt_t",
            "pred_A", "pred_B", "pred_C", "pred_r", "pred_s", "pred_t",
        ])
        for i in range(B):
            w.writerow([
                i,
                float(losses[i].cpu()),
                float(exp_l1[i].cpu()),
                float(rot[i].cpu()),
                *[float(x) for x in gt_scale[i].cpu()],
                *[float(x) for x in gt_exp[i].cpu()],
                *[float(x) for x in pred_scale[i].cpu()],
                *[float(x) for x in pred_exp[i].cpu()],
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
        "exp_l1_mean": float(exp_l1.mean().cpu()),
        "exp_l1_p95": float(torch.quantile(exp_l1, .95).cpu()),
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
