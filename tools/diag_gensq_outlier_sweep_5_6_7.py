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


def q(x, p):
    return float(torch.quantile(x.detach().float().cpu(), p))


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
    point_max_ratio = ratio.max(dim=-1).values
    cloud_max_ratio = point_max_ratio.max(dim=1).values

    print()
    print(f"================ eps_max={exp_max} ================")
    print(
        "point max(|x|/scale): "
        f"mean={point_max_ratio.mean().item():.4f} "
        f"p95={q(point_max_ratio.reshape(-1),0.95):.4f} "
        f"p99={q(point_max_ratio.reshape(-1),0.99):.4f} "
        f"max={point_max_ratio.max().item():.4f}"
    )
    print(
        "cloud max ratio:       "
        f"mean={cloud_max_ratio.mean().item():.4f} "
        f"p90={q(cloud_max_ratio,0.90):.4f} "
        f"p95={q(cloud_max_ratio,0.95):.4f} "
        f"p99={q(cloud_max_ratio,0.99):.4f} "
        f"max={cloud_max_ratio.max().item():.4f}"
    )

    for thr in [1.05, 1.2, 2.0]:
        invalid_frac = (point_max_ratio > thr).float().mean(dim=1)

        print(
            f"thr={thr:<4} "
            f"invalid_frac mean={invalid_frac.mean().item():.5f} "
            f"p95={q(invalid_frac,0.95):.5f} "
            f"p99={q(invalid_frac,0.99):.5f} "
            f"max={invalid_frac.max().item():.5f} | "
            f"clouds >=1 invalid={(invalid_frac > 0).float().mean().item():.4f} "
            f">5%={(invalid_frac > 0.05).float().mean().item():.4f} "
            f">20%={(invalid_frac > 0.20).float().mean().item():.4f}"
        )

    # worst cloud only
    invalid_frac = (point_max_ratio > 1.05).float().mean(dim=1)
    j = int(torch.argmax(invalid_frac).item())

    print("worst @thr=1.05:")
    print(f"  invalid_frac={invalid_frac[j].item():.5f}")
    print(f"  scale_gt={scale_gt[j].detach().cpu().numpy()}")
    print(f"  eps_gt={eps_gt[j].detach().cpu().numpy()}")
    print(f"  raw min={X[j].amin(dim=0).detach().cpu().numpy()}")
    print(f"  raw max={X[j].amax(dim=0).detach().cpu().numpy()}")
    print(f"  max_ratio={point_max_ratio[j].max().item():.5f}")


def main():
    for exp_max in [5.0, 6.0, 7.0]:
        run(exp_max)


if __name__ == "__main__":
    main()
