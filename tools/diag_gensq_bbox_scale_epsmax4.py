#!/usr/bin/env python3
import math
import torch

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
)


def sample_log_uniform(lo, hi, shape, device):
    return torch.exp(
        math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(shape, device=device)
    )


def summarize(name, x):
    x = x.detach().cpu()
    print(
        f"{name:28s} "
        f"mean={x.mean():.6f} med={x.median():.6f} "
        f"p90={torch.quantile(x,0.90):.6f} "
        f"p95={torch.quantile(x,0.95):.6f} "
        f"min={x.min():.6f} max={x.max():.6f}"
    )


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 5000
    P = 512
    scale_min, scale_max = 0.08, 0.35
    exp_min, exp_max = 0.25, 4.0

    dirs = fibonacci_sphere(P, device)

    scale_gt = scale_min + (scale_max - scale_min) * torch.rand(B, 3, device=device)
    eps_gt = sample_log_uniform(exp_min, exp_max, (B, 3), device)

    X = sample_generalized_surface(scale_gt, eps_gt, dirs)

    mn = X.amin(dim=1)
    mx = X.amax(dim=1)

    absmax = X.abs().amax(dim=1)
    half = 0.5 * (mx - mn)
    center = 0.5 * (mx + mn)

    print("=== bbox vs GT scale ratios ===")
    summarize("absmax / scale_gt", (absmax / scale_gt).reshape(-1))
    summarize("half_extent / scale_gt", (half / scale_gt).reshape(-1))
    summarize("abs(absmax-half)", (absmax - half).abs().reshape(-1))
    summarize("abs(center)", center.abs().reshape(-1))
    print()

    print("=== normalized with absmax: X / absmax ===")
    X_abs = X / absmax[:, None, :]
    mn_abs = X_abs.amin(dim=1)
    mx_abs = X_abs.amax(dim=1)
    maxabs_abs = X_abs.abs().amax(dim=1)

    summarize("min after absmax norm", mn_abs.reshape(-1))
    summarize("max after absmax norm", mx_abs.reshape(-1))
    summarize("maxabs after absmax norm", maxabs_abs.reshape(-1))
    summarize("abs(max-1)", (mx_abs - 1.0).abs().reshape(-1))
    summarize("abs(min+1)", (mn_abs + 1.0).abs().reshape(-1))
    print()

    print("=== normalized with centered half bbox: (X-center)/half ===")
    X_half = (X - center[:, None, :]) / half[:, None, :].clamp_min(1e-8)
    mn_half = X_half.amin(dim=1)
    mx_half = X_half.amax(dim=1)
    maxabs_half = X_half.abs().amax(dim=1)

    summarize("min after half norm", mn_half.reshape(-1))
    summarize("max after half norm", mx_half.reshape(-1))
    summarize("maxabs after half norm", maxabs_half.reshape(-1))
    summarize("abs(max-1)", (mx_half - 1.0).abs().reshape(-1))
    summarize("abs(min+1)", (mn_half + 1.0).abs().reshape(-1))
    print()

    print("=== example worst absmax/scale_gt ratios ===")
    ratio_err = (absmax / scale_gt - 1.0).abs().mean(dim=1)
    idx = torch.topk(ratio_err, 10).indices

    for j in idx.tolist():
        print()
        print(f"sample {j}")
        print(f"scale_gt = {scale_gt[j].detach().cpu().numpy()}")
        print(f"eps_gt   = {eps_gt[j].detach().cpu().numpy()}")
        print(f"min      = {mn[j].detach().cpu().numpy()}")
        print(f"max      = {mx[j].detach().cpu().numpy()}")
        print(f"absmax   = {absmax[j].detach().cpu().numpy()}")
        print(f"half     = {half[j].detach().cpu().numpy()}")
        print(f"center   = {center[j].detach().cpu().numpy()}")
        print(f"absmax/scale = {(absmax[j]/scale_gt[j]).detach().cpu().numpy()}")
        print(f"half/scale   = {(half[j]/scale_gt[j]).detach().cpu().numpy()}")


if __name__ == "__main__":
    main()
