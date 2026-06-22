#!/usr/bin/env python3
import math
import torch

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
    chamfer,
)


def sample_log_uniform(lo, hi, shape, device):
    return torch.exp(
        math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(shape, device=device)
    )


@torch.no_grad()
def sample_generalized_surface_batched_dirs(scale, exp, dirs_batched, iters=32):
    """
    Same robust bisection sampler, but accepts per-sample dirs:
        scale:        [B,3]
        exp:          [B,3]
        dirs_batched: [B,S,3]
    """
    B, S, _ = dirs_batched.shape
    device = scale.device
    dtype = scale.dtype

    u = dirs_batched.to(device=device, dtype=dtype)
    A = scale[:, None, :].clamp_min(1e-8)
    e = exp[:, None, :].clamp_min(0.05)

    abs_u = u.abs().clamp_min(1e-12)

    rho_axis_hi = A / abs_u
    rho_hi = rho_axis_hi.min(dim=-1, keepdim=True).values.clamp_min(1e-12)
    rho_lo = torch.zeros_like(rho_hi)

    coeff = (abs_u / A).pow(e)

    for _ in range(iters):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        f_mid = (coeff * rho_mid.clamp_min(1e-12).pow(e)).sum(dim=-1, keepdim=True) - 1.0
        too_high = f_mid >= 0.0
        rho_hi = torch.where(too_high, rho_mid, rho_hi)
        rho_lo = torch.where(too_high, rho_lo, rho_mid)

    rho = 0.5 * (rho_lo + rho_hi)
    return u * rho


def stat(name, x):
    x = x.detach().cpu()
    print(
        f"{name:36s} "
        f"mean={x.mean():.8f} med={x.median():.8f} "
        f"p90={torch.quantile(x,0.90):.8f} "
        f"p95={torch.quantile(x,0.95):.8f} "
        f"max={x.max():.8f}"
    )


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2000
    S = 512

    scale_min, scale_max = 0.08, 0.35
    exp_min, exp_max = 0.5, 8.0

    dirs = fibonacci_sphere(S, device)

    scale = scale_min + (scale_max - scale_min) * torch.rand(B, 3, device=device)
    eps = sample_log_uniform(exp_min, exp_max, (B, 3), device)

    unit_scale = torch.ones_like(scale)

    # Original anisotropic-scale samples.
    X = sample_generalized_surface(scale, eps, dirs)

    # Divide by true scale.
    X_div_gt = X / scale[:, None, :]

    # Unit-scale samples using the same original directions.
    Y_unit_same_dirs = sample_generalized_surface(unit_scale, eps, dirs)

    # Correct transformed directions after componentwise scale division.
    dirs_trans = dirs[None, :, :] / scale[:, None, :]
    dirs_trans = dirs_trans / dirs_trans.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    Y_unit_trans_dirs = sample_generalized_surface_batched_dirs(unit_scale, eps, dirs_trans)

    print("=== CD comparisons ===")
    stat("CD(X/scale_gt, unit same dirs)", chamfer(X_div_gt, Y_unit_same_dirs))
    stat("CD(X/scale_gt, unit transformed dirs)", chamfer(X_div_gt, Y_unit_trans_dirs))
    print()

    print("=== pointwise L2 comparisons ===")
    pw_same = ((X_div_gt - Y_unit_same_dirs) ** 2).sum(dim=-1).sqrt().mean(dim=1)
    pw_trans = ((X_div_gt - Y_unit_trans_dirs) ** 2).sum(dim=-1).sqrt().mean(dim=1)
    stat("pointwise L2 same dirs", pw_same)
    stat("pointwise L2 transformed dirs", pw_trans)
    print()

    print("=== max pointwise L2 comparisons ===")
    pw_same_max = ((X_div_gt - Y_unit_same_dirs) ** 2).sum(dim=-1).sqrt().max(dim=1).values
    pw_trans_max = ((X_div_gt - Y_unit_trans_dirs) ** 2).sum(dim=-1).sqrt().max(dim=1).values
    stat("max pointwise L2 same dirs", pw_same_max)
    stat("max pointwise L2 transformed dirs", pw_trans_max)
    print()

    print("=== bbox sanity ===")
    stat("maxabs X/scale_gt", X_div_gt.abs().amax(dim=1).reshape(-1))
    stat("maxabs unit same dirs", Y_unit_same_dirs.abs().amax(dim=1).reshape(-1))
    stat("maxabs unit transformed dirs", Y_unit_trans_dirs.abs().amax(dim=1).reshape(-1))

    print()
    print("Expected:")
    print("  same dirs: nonzero mismatch, because componentwise scale division changes ray directions")
    print("  transformed dirs: near-zero pointwise/CD mismatch")


if __name__ == "__main__":
    main()
