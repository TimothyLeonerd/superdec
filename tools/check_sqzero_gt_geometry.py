#!/usr/bin/env python3
import hydra
import torch
from torch.utils.data import DataLoader

from superdec.data.sqzero_lmdb import SQZeroLMDB


def sq_implicit_value(points, labels, gt_scale, gt_shape, gt_rotate, gt_trans):
    """Evaluate SQ implicit function for each point against its own GT label.

    Args:
        points:    [B, N, 3] normalized points
        labels:    [B, N] primitive id per point
        gt_scale:  [B, Kmax, 3]
        gt_shape:  [B, Kmax, 2]
        gt_rotate: [B, Kmax, 3, 3], local-to-world rotation
        gt_trans:  [B, Kmax, 3]

    Returns:
        F: [B, N], where ideal surface points satisfy F ~= 1
    """
    B, N, _ = points.shape
    device = points.device

    bidx = torch.arange(B, device=device)[:, None]

    scale = gt_scale[bidx, labels]      # [B, N, 3]
    shape = gt_shape[bidx, labels]      # [B, N, 2]
    rotate = gt_rotate[bidx, labels]    # [B, N, 3, 3]
    trans = gt_trans[bidx, labels]      # [B, N, 3]

    centered = points - trans

    # SuperDec / SQ-Zero convention:
    #   p_world = R @ p_local + t
    # so:
    #   p_local = R.T @ (p_world - t)
    p_local = torch.einsum(
        "bnij,bnj->bni",
        rotate.transpose(-1, -2),
        centered,
    )

    eps = 1e-8
    a = torch.clamp(scale, min=eps)
    e1 = torch.clamp(shape[..., 0], min=eps)
    e2 = torch.clamp(shape[..., 1], min=eps)

    x = torch.abs(p_local[..., 0] / a[..., 0])
    y = torch.abs(p_local[..., 1] / a[..., 1])
    z = torch.abs(p_local[..., 2] / a[..., 2])

    xy = torch.pow(
        torch.pow(x, 2.0 / e2) + torch.pow(y, 2.0 / e2),
        e2 / e1,
    )
    zz = torch.pow(z, 2.0 / e1)

    return xy + zz


def print_stats(name, values):
    values = values.detach().cpu().float()
    q = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99]))
    print(
        f"{name}: "
        f"mean={values.mean().item():.6e}, "
        f"median={q[0].item():.6e}, "
        f"p90={q[1].item():.6e}, "
        f"p99={q[2].item():.6e}, "
        f"max={values.max().item():.6e}"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    ds = SQZeroLMDB(split="train", cfg=cfg)

    loader = DataLoader(
        ds,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    batch = next(iter(loader))

    required = [
        "points",
        "labels",
        "gt_scale",
        "gt_shape",
        "gt_rotate",
        "gt_trans",
        "valid_mask",
        "K",
    ]
    for k in required:
        if k not in batch:
            raise KeyError(f"Missing batch key: {k}")

    points = batch["points"].float()
    labels = batch["labels"].long()
    gt_scale = batch["gt_scale"].float()
    gt_shape = batch["gt_shape"].float()
    gt_rotate = batch["gt_rotate"].float()
    gt_trans = batch["gt_trans"].float()
    K = batch["K"].long()

    F = sq_implicit_value(
        points=points,
        labels=labels,
        gt_scale=gt_scale,
        gt_shape=gt_shape,
        gt_rotate=gt_rotate,
        gt_trans=gt_trans,
    )

    abs_residual = torch.abs(F - 1.0)

    print("=== Shapes ===")
    print("points:", tuple(points.shape))
    print("labels:", tuple(labels.shape))
    print("gt_scale:", tuple(gt_scale.shape))
    print("gt_shape:", tuple(gt_shape.shape))
    print("gt_rotate:", tuple(gt_rotate.shape))
    print("gt_trans:", tuple(gt_trans.shape))
    print("K:", K.tolist())

    print("\n=== Overall implicit residual |F - 1| ===")
    print_stats("all points", abs_residual)

    print("\n=== Per-sample / per-primitive residuals ===")
    '''
    B = points.shape[0]
    for b in range(B):
        print(f"sample {b}, K={int(K[b].item())}")
        for k in range(int(K[b].item())):
            mask = labels[b] == k
            count = int(mask.sum().item())
            if count == 0:
                print(f"  primitive {k}: no points")
                continue
            r = abs_residual[b, mask]
            print(
                f"  primitive {k}: "
                f"n={count:4d}, "
                f"mean={r.mean().item():.6e}, "
                f"p99={torch.quantile(r.cpu(), 0.99).item():.6e}, "
                f"max={r.max().item():.6e}"
            )

    # Optional sanity contrast: for samples with K > 1, evaluate points against
    # the next primitive label. This should usually be much worse.
    wrong_residuals = []
    for b in range(B):
        k = int(K[b].item())
        if k <= 1:
            continue
        wrong_labels = (labels[b] + 1) % k
        F_wrong = sq_implicit_value(
            points=points[b:b+1],
            labels=wrong_labels[None, :],
            gt_scale=gt_scale[b:b+1],
            gt_shape=gt_shape[b:b+1],
            gt_rotate=gt_rotate[b:b+1],
            gt_trans=gt_trans[b:b+1],
        )
        wrong_residuals.append(torch.abs(F_wrong - 1.0).reshape(-1))
        

    if wrong_residuals:
        wrong_residuals = torch.cat(wrong_residuals)
        print("\n=== Contrast check: residual against wrong primitive |F_wrong - 1| ===")
        print_stats("wrong primitive", wrong_residuals)

    print("\n=== PASS condition ===")
    print(
        "For correct normalization/convention, the correct-label residual should be "
        "near numerical precision and much smaller than the wrong-primitive residual."
    )


if __name__ == "__main__":
    main()
