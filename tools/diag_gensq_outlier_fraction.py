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
    x = x.detach().float().cpu()
    print(
        f"{name:34s} "
        f"mean={x.mean():.6f} med={x.median():.6f} "
        f"p90={torch.quantile(x,0.90):.6f} "
        f"p95={torch.quantile(x,0.95):.6f} "
        f"p99={torch.quantile(x,0.99):.6f} "
        f"max={x.max():.6f}"
    )


def run(exp_max):
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 5000
    P = 512
    scale_min, scale_max = 0.08, 0.35
    exp_min = 0.25

    dirs = fibonacci_sphere(P, device)

    scale_gt = scale_min + (scale_max - scale_min) * torch.rand(B, 3, device=device)
    eps_gt = sample_log_uniform(exp_min, exp_max, (B, 3), device)

    X = sample_generalized_surface(scale_gt, eps_gt, dirs)

    ratio = X.abs() / scale_gt[:, None, :]
    max_ratio_per_point = ratio.max(dim=-1).values

    print()
    print(f"================ eps_max={exp_max} ================")

    summarize("point max(|x|/scale)", max_ratio_per_point.reshape(-1))

    for thr in [1.001, 1.01, 1.05, 1.10, 1.20, 2.0]:
        invalid_point = max_ratio_per_point > thr
        invalid_frac_per_cloud = invalid_point.float().mean(dim=1)
        keep_frac_per_cloud = 1.0 - invalid_frac_per_cloud

        print()
        print(f"threshold {thr}")
        summarize("invalid point fraction/cloud", invalid_frac_per_cloud)
        summarize("kept point fraction/cloud", keep_frac_per_cloud)
        print(f"clouds with >=1 invalid point: {(invalid_frac_per_cloud > 0).float().mean().item():.4f}")
        print(f"clouds with >5% invalid:       {(invalid_frac_per_cloud > 0.05).float().mean().item():.4f}")
        print(f"clouds with >20% invalid:      {(invalid_frac_per_cloud > 0.20).float().mean().item():.4f}")

    # Show worst few clouds.
    invalid_frac = (max_ratio_per_point > 1.05).float().mean(dim=1)
    idx = torch.topk(invalid_frac, 10).indices

    print()
    print("=== worst clouds at threshold 1.05 ===")
    for j in idx.tolist():
        print()
        print(f"sample {j}")
        print(f"invalid_frac={invalid_frac[j].item():.4f}")
        print(f"scale_gt={scale_gt[j].detach().cpu().numpy()}")
        print(f"eps_gt={eps_gt[j].detach().cpu().numpy()}")
        print(f"raw min={X[j].amin(dim=0).detach().cpu().numpy()}")
        print(f"raw max={X[j].amax(dim=0).detach().cpu().numpy()}")
        print(f"max ratio={max_ratio_per_point[j].max().item():.4f}")


def main():
    run(8.0)
    run(4.0)
    run(3.0)
    run(2.0)


if __name__ == "__main__":
    main()
