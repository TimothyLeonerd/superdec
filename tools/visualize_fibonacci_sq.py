import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fibonacci_sphere(n: int) -> np.ndarray:
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    pts = []
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / n
        rho = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i

        x = np.cos(theta) * rho
        z = np.sin(theta) * rho
        pts.append([x, y, z])

    return np.asarray(pts)


def apply_dropout(pts: np.ndarray, keep_frac: float, seed: int):
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"--keep-frac must be in (0, 1], got {keep_frac}")

    n = pts.shape[0]
    k = max(1, int(round(keep_frac * n)))

    if k >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=k, replace=False)
        idx = np.sort(idx)

    return pts[idx], idx


def generalized_sq_ray_intersection(
    dirs: np.ndarray,
    scale=(1.0, 1.0, 1.0),
    exponents=(2.0, 2.0, 2.0),
    iters=64,
) -> np.ndarray:
    A, B, C = scale
    r, s, t = exponents

    lo = np.zeros((dirs.shape[0],), dtype=np.float64)
    hi = np.ones((dirs.shape[0],), dtype=np.float64)

    def F(lam):
        p = dirs * lam[:, None]
        return (
            np.abs(p[:, 0] / A) ** r
            + np.abs(p[:, 1] / B) ** s
            + np.abs(p[:, 2] / C) ** t
        )

    while np.any(F(hi) < 1.0):
        hi *= 2.0

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        inside = F(mid) < 1.0
        lo[inside] = mid[inside]
        hi[~inside] = mid[~inside]

    lam = 0.5 * (lo + hi)
    return dirs * lam[:, None]


def set_equal_3d(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * (maxs - mins).max()

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_points(ax, pts, title):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 1], s=14)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y vertical")
    ax.set_zlabel("z")
    set_equal_3d(ax, pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--keep-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs_full = fibonacci_sphere(args.n)
    dirs, idx = apply_dropout(dirs_full, args.keep_frac, args.seed)

    sphere = generalized_sq_ray_intersection(
        dirs,
        scale=(1.0, 1.0, 1.0),
        exponents=(2.0, 2.0, 2.0),
    )

    boxy_sq = generalized_sq_ray_intersection(
        dirs,
        scale=(1.2, 0.8, 0.6),
        exponents=(8.0, 8.0, 8.0),
    )

    kept = dirs.shape[0]
    title_suffix = f"n={args.n}, keep_frac={args.keep_frac}, kept={kept}, seed={args.seed}"

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_points(ax1, sphere, f"Sphere\n{title_suffix}")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_points(ax2, boxy_sq, f"Boxy generalized SQ\n{title_suffix}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"saved to {args.out}")
    print(f"original_n={args.n}")
    print(f"keep_frac={args.keep_frac}")
    print(f"kept={kept}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
