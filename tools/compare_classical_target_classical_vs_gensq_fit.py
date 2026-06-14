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


def sample_classical_ray(scale, eps, dirs):
    # classical SQ implicit:
    # ((|x/A|^(2/e2)+|y/B|^(2/e2))^(e2/e1) + |z/C|^(2/e1)) = 1
    B = scale.shape[0]
    S = dirs.shape[0]
    u = dirs[None].expand(B, S, 3)

    A = scale[:, None, 0].clamp_min(1e-8)
    Bsc = scale[:, None, 1].clamp_min(1e-8)
    C = scale[:, None, 2].clamp_min(1e-8)
    e1 = eps[:, None, 0].clamp_min(1e-4)
    e2 = eps[:, None, 1].clamp_min(1e-4)

    ux = u[..., 0].abs().clamp_min(1e-8)
    uy = u[..., 1].abs().clamp_min(1e-8)
    uz = u[..., 2].abs().clamp_min(1e-8)

    xy = (ux / A).pow(2.0 / e2) + (uy / Bsc).pow(2.0 / e2)
    coeff = xy.pow(e2 / e1) + (uz / C).pow(2.0 / e1)
    rho = coeff.clamp_min(1e-12).pow(-e1 / 2.0)

    return u * rho[..., None]


def sample_generalized_surface(scale, exp, dirs, newton_iters=12):
    # |x/A|^r + |y/B|^s + |z/C|^t = 1
    B = scale.shape[0]
    S = dirs.shape[0]
    u = dirs[None].expand(B, S, 3)
    A = scale[:, None, :].clamp_min(1e-6)
    e = exp[:, None, :].clamp_min(0.05)

    coeff = (u.abs().clamp_min(1e-8) / A).pow(e)

    em = e.mean(dim=-1, keepdim=True)
    rho = coeff.sum(dim=-1, keepdim=True).clamp_min(1e-8).pow(-1.0 / em).clamp(1e-4, 10.0)

    for _ in range(newton_iters):
        f = (coeff * rho.pow(e)).sum(dim=-1, keepdim=True) - 1.0
        df = (coeff * e * rho.pow(e - 1.0)).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rho = (rho - f / df).clamp(1e-4, 10.0)

    return u * rho


def chamfer(a, b):
    d2 = torch.cdist(a, b).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


def geodesic_deg(Ra, Rb):
    R = Ra.transpose(1, 2) @ Rb
    tr = R.diagonal(dim1=1, dim2=2).sum(dim=1)
    c = ((tr - 1.0) / 2.0).clamp(-1, 1)
    return torch.acos(c) * (180.0 / math.pi)


def summarize(losses):
    return {
        "loss_mean": float(losses.mean().cpu()),
        "loss_median": float(losses.median().cpu()),
        "loss_p90": float(torch.quantile(losses, .90).cpu()),
        "loss_p95": float(torch.quantile(losses, .95).cpu()),
        "loss_max": float(losses.max().cpu()),
        "succ_loss_lt_1e-6": float((losses < 1e-6).float().mean().cpu()),
        "succ_loss_lt_1e-5": float((losses < 1e-5).float().mean().cpu()),
        "succ_loss_lt_1e-4": float((losses < 1e-4).float().mean().cpu()),
    }


def fit_classical(args, gt_world, gt_scale, gt_eps, gt_R, dirs, out):
    device = gt_world.device
    B = gt_world.shape[0]

    init_R = random_rotations(B, device)
    init_scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    init_eps = args.eps_min + (args.eps_max - args.eps_min) * torch.rand(B, 2, device=device)

    raw_rot = torch.zeros(B, 3, device=device, requires_grad=True)
    raw_scale = inv_sigmoid_param(init_scale, args.scale_min, args.scale_max).detach().clone().requires_grad_(True)
    raw_eps = inv_sigmoid_param(init_eps, args.eps_min, args.eps_max).detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([raw_rot, raw_scale, raw_eps], lr=args.lr)

    for step in range(1, args.steps + 1):
        pred_scale = bounded(raw_scale, args.scale_min, args.scale_max)
        pred_eps = bounded(raw_eps, args.eps_min, args.eps_max)
        pred_R = rodrigues(raw_rot) @ init_R
        pred_local = sample_classical_ray(pred_scale, pred_eps, dirs)
        pred_world = pred_local @ pred_R.transpose(1, 2)
        losses = chamfer(pred_world, gt_world)
        loss = losses.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 200 == 0 or step == args.steps:
            print(f"[classical] step {step:04d} loss={loss.item():.8g} p95={torch.quantile(losses,.95).item():.8g}", flush=True)

    with torch.no_grad():
        pred_scale = bounded(raw_scale, args.scale_min, args.scale_max)
        pred_eps = bounded(raw_eps, args.eps_min, args.eps_max)
        pred_R = rodrigues(raw_rot) @ init_R
        pred_world = sample_classical_ray(pred_scale, pred_eps, dirs) @ pred_R.transpose(1, 2)
        losses = chamfer(pred_world, gt_world)
        eps_l1 = (pred_eps - gt_eps).abs().mean(dim=1)
        scale_l1 = (pred_scale - gt_scale).abs().mean(dim=1)
        rot = geodesic_deg(pred_R, gt_R)

    with open(out / "details_classical.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","loss","eps_l1","scale_l1","rot_geodesic_deg",
                    "gt_A","gt_B","gt_C","gt_eps1","gt_eps2",
                    "pred_A","pred_B","pred_C","pred_eps1","pred_eps2"])
        for i in range(B):
            w.writerow([i, float(losses[i].cpu()), float(eps_l1[i].cpu()), float(scale_l1[i].cpu()), float(rot[i].cpu()),
                        *[float(x) for x in gt_scale[i].cpu()],
                        *[float(x) for x in gt_eps[i].cpu()],
                        *[float(x) for x in pred_scale[i].cpu()],
                        *[float(x) for x in pred_eps[i].cpu()]])
    extra = {
        "eps_l1_mean": float(eps_l1.mean().cpu()),
        "scale_l1_mean": float(scale_l1.mean().cpu()),
        "rotdeg_mean": float(rot.mean().cpu()),
    }
    return losses, extra


def fit_gensq(args, gt_world, gt_scale, gt_eps, gt_R, dirs, out):
    device = gt_world.device
    B = gt_world.shape[0]

    init_R = random_rotations(B, device)
    init_scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(B, 3, device=device)
    init_exp = torch.full((B, 3), 2.0, device=device)

    raw_rot = torch.zeros(B, 3, device=device, requires_grad=True)
    raw_scale = inv_sigmoid_param(init_scale, args.scale_min, args.scale_max).detach().clone().requires_grad_(True)
    raw_exp = inv_sigmoid_param(init_exp, args.exp_min, args.exp_max).detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([raw_rot, raw_scale, raw_exp], lr=args.lr)

    for step in range(1, args.steps + 1):
        pred_scale = bounded(raw_scale, args.scale_min, args.scale_max)
        pred_exp = bounded(raw_exp, args.exp_min, args.exp_max)
        pred_R = rodrigues(raw_rot) @ init_R
        pred_local = sample_generalized_surface(pred_scale, pred_exp, dirs)
        pred_world = pred_local @ pred_R.transpose(1, 2)
        losses = chamfer(pred_world, gt_world)
        loss = losses.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 200 == 0 or step == args.steps:
            print(f"[gensq]     step {step:04d} loss={loss.item():.8g} p95={torch.quantile(losses,.95).item():.8g}", flush=True)

    with torch.no_grad():
        pred_scale = bounded(raw_scale, args.scale_min, args.scale_max)
        pred_exp = bounded(raw_exp, args.exp_min, args.exp_max)
        pred_R = rodrigues(raw_rot) @ init_R
        pred_world = sample_generalized_surface(pred_scale, pred_exp, dirs) @ pred_R.transpose(1, 2)
        losses = chamfer(pred_world, gt_world)
        scale_l1 = (pred_scale - gt_scale).abs().mean(dim=1)
        rot = geodesic_deg(pred_R, gt_R)

    with open(out / "details_gensq.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","loss","scale_l1","rot_geodesic_deg",
                    "gt_A","gt_B","gt_C","gt_eps1","gt_eps2",
                    "pred_A","pred_B","pred_C","pred_r","pred_s","pred_t"])
        for i in range(B):
            w.writerow([i, float(losses[i].cpu()), float(scale_l1[i].cpu()), float(rot[i].cpu()),
                        *[float(x) for x in gt_scale[i].cpu()],
                        *[float(x) for x in gt_eps[i].cpu()],
                        *[float(x) for x in pred_scale[i].cpu()],
                        *[float(x) for x in pred_exp[i].cpu()]])
    extra = {
        "scale_l1_mean": float(scale_l1.mean().cpu()),
        "rotdeg_mean": float(rot.mean().cpu()),
    }
    return losses, extra


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
    ap.add_argument("--exp-min", type=float, default=0.3)
    ap.add_argument("--exp-max", type=float, default=1.7)
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
    gt_eps = args.eps_min + (args.eps_max - args.eps_min) * torch.rand(B, 2, device=device)
    gt_R = random_rotations(B, device)

    with torch.no_grad():
        gt_local = sample_classical_ray(gt_scale, gt_eps, dirs)
        gt_world = gt_local @ gt_R.transpose(1, 2)

    classical_losses, classical_extra = fit_classical(args, gt_world, gt_scale, gt_eps, gt_R, dirs, out)
    gensq_losses, gensq_extra = fit_gensq(args, gt_world, gt_scale, gt_eps, gt_R, dirs, out)

    rows = []
    for name, losses, extra in [
        ("classical_fit", classical_losses, classical_extra),
        ("gensq_fit", gensq_losses, gensq_extra),
    ]:
        row = {"model": name, "n": B}
        row.update(summarize(losses))
        row.update(extra)
        rows.append(row)

    fields = sorted(set(k for r in rows for k in r.keys()))
    fields = ["model", "n"] + [f for f in fields if f not in ("model", "n")]

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n=== SUMMARY ===")
    for r in rows:
        print(r)
    print("Wrote", out / "summary.csv")


if __name__ == "__main__":
    main()
