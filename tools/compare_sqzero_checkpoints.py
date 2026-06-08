#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from superdec.superdec import SuperDec
from superdec.lm_optimization.lm_optimizer import LMOptimizer
from superdec.data.sqzero_lmdb import SQZeroLMDB
from superdec.loss.loss import sampling_from_parametric_space_to_equivalent_points
from superdec.loss.sampler import EqualDistanceSamplerSQ


def infer_config_path(ckpt_path: Path) -> Path:
    cfg = ckpt_path.parent / "config.yaml"
    if not cfg.exists():
        raise FileNotFoundError(
            f"Could not infer config path for {ckpt_path}. "
            f"Expected {cfg}. Use --cfg-a/--cfg-b explicitly."
        )
    return cfg


def load_checkpoint_state(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    cleaned = {}
    for k, v in state.items():
        cleaned[k[len("module."):] if k.startswith("module.") else k] = v

    return cleaned


def load_model(ckpt_path: Path, cfg_path: Path, device: torch.device, use_lm: bool = False):
    cfg = OmegaConf.load(cfg_path)

    model = SuperDec(cfg.superdec).to(device)
    state = load_checkpoint_state(ckpt_path)

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARN] Missing keys for {ckpt_path}:")
        for k in missing[:20]:
            print(f"  {k}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")

    if unexpected:
        print(f"[WARN] Unexpected keys for {ckpt_path}:")
        for k in unexpected[:20]:
            print(f"  {k}")
        if len(unexpected) > 20:
            print(f"  ... {len(unexpected) - 20} more")

    if use_lm:
        model.lm_optimizer = LMOptimizer().to(device)
        model.lm_optimization = True
        print(f"[INFO] LM optimization enabled for {ckpt_path}")
    else:
        model.lm_optimization = False

    model.eval()
    return model, cfg


def make_eval_cfg(base_cfg, args):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    if "sqzero_lmdb" not in cfg:
        cfg.sqzero_lmdb = {}

    cfg.sqzero_lmdb.path = str(args.data_root)
    cfg.sqzero_lmdb.train_split = args.split
    cfg.sqzero_lmdb.val_split = args.split
    cfg.sqzero_lmdb.n_points = args.n_points
    cfg.sqzero_lmdb.normalize = True
    cfg.sqzero_lmdb.normal_mode = args.normal_mode
    cfg.sqzero_lmdb.load_sidecars = True
    cfg.sqzero_lmdb.kmax = args.kmax

    if args.max_samples is not None and args.max_samples > 0:
        cfg.sqzero_lmdb.max_val_samples = args.max_samples
        cfg.sqzero_lmdb.max_train_samples = args.max_samples
    else:
        cfg.sqzero_lmdb.max_val_samples = None
        cfg.sqzero_lmdb.max_train_samples = None

    if "trainer" not in cfg:
        cfg.trainer = {}
    cfg.trainer.augmentations = False

    return cfg


def local_to_world(local_points, rotate, trans):
    """local_points [B,P,S,3], rotate [B,P,3,3], trans [B,P,3]."""
    return torch.einsum("bpij,bpsj->bpsi", rotate, local_points) + trans.unsqueeze(2)


def chamfer_squared(x, y):
    """Symmetric squared Chamfer distance between x [Nx,3] and y [Ny,3]."""
    if x.numel() == 0 or y.numel() == 0:
        return x.new_tensor(float("nan"))

    d = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2.0)[0]
    d2 = d * d
    return d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean()


def maybe_subsample_points(points, max_points):
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points

    # Deterministic subsampling for reproducible evaluation.
    idx = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=max_points,
        device=points.device,
    ).long()
    return points[idx]


def sq_implicit_value(points, scale, shape, rotate, trans, eps=1e-8):
    """Evaluate SQ implicit function.

    Args:
        points: [N,3] world/object-normalized points.
        scale:  [M,3]
        shape:  [M,2], eps1, eps2
        rotate: [M,3,3], local-to-world rotation
        trans:  [M,3]

    Returns:
        f: [M,N], inside if f < 1.
    """
    if scale.numel() == 0:
        return points.new_empty((0, points.shape[0]))

    centered = points.unsqueeze(0) - trans.unsqueeze(1)  # [M,N,3]

    # world -> local:
    # p_world = R @ p_local + t
    # p_local = R.T @ (p_world - t)
    local = torch.einsum(
        "mij,mnj->mni",
        rotate.transpose(-1, -2),
        centered,
    )

    a = torch.clamp(scale, min=eps)
    e1 = torch.clamp(shape[:, 0], min=eps)
    e2 = torch.clamp(shape[:, 1], min=eps)

    x = torch.abs(local[..., 0] / a[:, 0:1])
    y = torch.abs(local[..., 1] / a[:, 1:2])
    z = torch.abs(local[..., 2] / a[:, 2:3])

    xy = torch.pow(
        torch.pow(x, 2.0 / e2[:, None]) + torch.pow(y, 2.0 / e2[:, None]),
        e2[:, None] / e1[:, None],
    )
    zz = torch.pow(z, 2.0 / e1[:, None])

    return xy + zz


def visible_surface_union(
    surfaces_by_prim,
    scale,
    shape,
    rotate,
    trans,
    inside_threshold=1.0,
    eps=1e-8,
):
    """Remove hidden/internal SQ surface points.

    Args:
        surfaces_by_prim: [M,S,3], sampled surface points for selected SQs.
        scale/shape/rotate/trans: selected SQ params.
        inside_threshold: remove point from SQ p if it has f < threshold
            for any other selected SQ q.

    Returns:
        visible_points: [Nv,3]
        raw_points: [M*S,3]
        visible_fraction: Nv / (M*S)
        empty_visible: 1.0 if no visible point survived, else 0.0
    """
    M = surfaces_by_prim.shape[0]
    raw = surfaces_by_prim.reshape(-1, 3)

    if M <= 1:
        return raw, raw, 1.0, 0.0

    visible_chunks = []

    for p in range(M):
        pts_p = surfaces_by_prim[p]  # [S,3]

        f = sq_implicit_value(
            points=pts_p,
            scale=scale,
            shape=shape,
            rotate=rotate,
            trans=trans,
            eps=eps,
        )  # [M,S]

        # Ignore self-SQ when testing occlusion/hiddenness.
        other_mask = torch.ones(M, dtype=torch.bool, device=pts_p.device)
        other_mask[p] = False

        inside_any_other = (f[other_mask] < inside_threshold).any(dim=0)
        keep = ~inside_any_other

        if keep.any():
            visible_chunks.append(pts_p[keep])

    if visible_chunks:
        visible = torch.cat(visible_chunks, dim=0)
        empty = 0.0
    else:
        visible = raw.new_empty((0, 3))
        empty = 1.0

    visible_fraction = float(visible.shape[0]) / max(float(raw.shape[0]), 1.0)
    return visible, raw, visible_fraction, empty


@torch.no_grad()
def hungarian_assignment_match(
    assign_b,
    exist_b,
    labels_b,
    K_b,
    eps=1e-8,
    match_w_assign=1.0,
    match_w_exist=0.0,
):
    """Return matched_slots[k] = predicted slot p for GT primitive k."""
    _, P = assign_b.shape
    K = int(K_b)

    cost = assign_b.new_zeros((P, K))

    for k in range(K):
        mask = labels_b == k
        if int(mask.sum().item()) == 0:
            cost[:, k] = 1e6
            continue

        probs = assign_b[mask, :]  # [Nk, P]
        assign_cost = -torch.log(probs.clamp_min(eps)).mean(dim=0)
        exist_cost = -torch.log(exist_b.clamp_min(eps))

        cost[:, k] = match_w_assign * assign_cost + match_w_exist * exist_cost

    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

    matched = np.empty((K,), dtype=np.int64)
    for p, k in zip(row_ind, col_ind):
        matched[k] = p

    return torch.as_tensor(matched, dtype=torch.long, device=assign_b.device)


def update_sum(sums, counts, key, value, n=1):
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    value = float(value)
    if not math.isfinite(value):
        return
    sums[key] = sums.get(key, 0.0) + value * n
    counts[key] = counts.get(key, 0) + n


def selected_surface_sets(pred_world_b, scale_b, shape_b, rotate_b, trans_b, idx, args):
    """Return visible/raw surface union for selected predicted slots."""
    idx = idx.long()
    surf = pred_world_b[idx]
    scale = scale_b[idx]
    shape = shape_b[idx]
    rotate = rotate_b[idx]
    trans = trans_b[idx]

    visible, raw, frac, empty = visible_surface_union(
        surf,
        scale,
        shape,
        rotate,
        trans,
        inside_threshold=args.inside_threshold,
        eps=args.eps,
    )
    return visible, raw, frac, empty


@torch.no_grad()
def evaluate_model(model, name, loader, sampler, args, device):
    sums = {}
    counts = {}

    n_samples = 0

    for batch in tqdm(loader, desc=f"eval {name}", disable=args.quiet):
        points = batch["points"].to(device).float()      # [B,N,3]
        labels = batch["labels"].to(device).long()       # [B,N]
        K = batch["K"].to(device).long()                 # [B]

        gt_scale = batch["gt_scale"].to(device).float()
        gt_shape = batch["gt_shape"].to(device).float()
        gt_rotate = batch["gt_rotate"].to(device).float()
        gt_trans = batch["gt_trans"].to(device).float()

        out = model(points)

        assign = out["assign_matrix"]                    # [B,N,P]
        exist = out["exist"].squeeze(-1)                 # [B,P]
        scale = out["scale"]                             # [B,P,3]
        shape = out["shape"]                             # [B,P,2]
        rotate = out["rotate"]                           # [B,P,3,3]
        trans = out["trans"]                             # [B,P,3]

        B, N, P = assign.shape
        n_samples += B

        # Sample predicted primitive surfaces once per batch.
        pred_local, _ = sampling_from_parametric_space_to_equivalent_points(
            scale,
            shape,
            sampler,
        )
        pred_world = local_to_world(pred_local, rotate, trans)  # [B,P,S,3]

        for b in range(B):
            K_b = int(K[b].item())
            input_pts = maybe_subsample_points(points[b], args.object_cd_max_input_points)

            # ------------------------------------------------------------
            # Ground-truth visible/raw object surface baselines.
            # ------------------------------------------------------------
            gt_local_b, _ = sampling_from_parametric_space_to_equivalent_points(
                gt_scale[b:b + 1, :K_b, :],
                gt_shape[b:b + 1, :K_b, :],
                sampler,
            )
            gt_world_b = local_to_world(
                gt_local_b,
                gt_rotate[b:b + 1, :K_b, :, :],
                gt_trans[b:b + 1, :K_b, :],
            )[0]  # [K_b,S,3]

            gt_visible, gt_raw, gt_vis_frac, gt_empty = visible_surface_union(
                gt_world_b,
                gt_scale[b, :K_b],
                gt_shape[b, :K_b],
                gt_rotate[b, :K_b],
                gt_trans[b, :K_b],
                inside_threshold=args.inside_threshold,
                eps=args.eps,
            )

            update_sum(sums, counts, "gt_cd_visible_to_input", chamfer_squared(input_pts, gt_visible))
            update_sum(sums, counts, "gt_cd_raw_to_input", chamfer_squared(input_pts, gt_raw))
            update_sum(sums, counts, "gt_visible_fraction", gt_vis_frac)
            update_sum(sums, counts, "gt_visible_empty_frac", gt_empty)

            # ------------------------------------------------------------
            # Supervised decomposition metrics.
            # ------------------------------------------------------------
            matched_slots = hungarian_assignment_match(
                assign_b=assign[b],
                exist_b=exist[b],
                labels_b=labels[b],
                K_b=K_b,
                eps=args.eps,
                match_w_assign=args.match_w_assign,
                match_w_exist=args.match_w_exist,
            )

            exist_target = torch.zeros_like(exist[b])
            exist_target[matched_slots] = 1.0

            exist_bce = F.binary_cross_entropy(
                exist[b],
                exist_target,
                reduction="mean",
            )

            target_slot = matched_slots[labels[b]]
            point_idx = torch.arange(N, device=device)
            chosen = assign[b, point_idx, target_slot]
            assign_nll = -torch.log(chosen.clamp_min(args.eps)).mean()

            pred_slot = assign[b].argmax(dim=1)
            assign_acc = (pred_slot == target_slot).float().mean()

            hard_count = (exist[b] > args.exist_threshold).sum().float()
            soft_count = exist[b].sum()
            true_count = torch.as_tensor(float(K_b), device=device)
            count_acc = (hard_count == true_count).float()

            matched_vals = exist[b][exist_target > 0.5]
            unmatched_vals = exist[b][exist_target < 0.5]

            update_sum(sums, counts, "decomp_exist_bce", exist_bce)
            update_sum(sums, counts, "decomp_assign_nll", assign_nll)
            update_sum(sums, counts, "decomp_assign_acc", assign_acc)
            update_sum(sums, counts, "decomp_count_acc", count_acc)
            update_sum(sums, counts, "pred_count_hard", hard_count)
            update_sum(sums, counts, "pred_count_soft", soft_count)
            update_sum(sums, counts, "true_count", true_count)

            if matched_vals.numel() > 0:
                update_sum(sums, counts, "exist_matched_mean", matched_vals.mean())
            if unmatched_vals.numel() > 0:
                update_sum(sums, counts, "exist_unmatched_mean", unmatched_vals.mean())

            # ------------------------------------------------------------
            # Predicted object geometry metrics.
            #
            # active: model's operational active set, exist > threshold
            # topK:   force model to use the true number of primitives
            # all:    use all predicted slots
            #
            # For each set, report:
            #   raw_to_input      = full sampled SQ surfaces vs input cloud
            #   visible_to_input  = hidden/internal surfaces removed vs input cloud
            #   visible_to_gt     = hidden/internal pred surface vs hidden/internal GT surface
            # ------------------------------------------------------------
            all_idx = torch.arange(P, device=device)

            active_idx = torch.nonzero(
                exist[b] > args.exist_threshold,
                as_tuple=False,
            ).flatten()

            active_empty = 0.0
            if active_idx.numel() == 0:
                active_empty = 1.0
                active_idx = torch.topk(exist[b], k=1).indices

            topk = max(1, min(K_b, P))
            topk_idx = torch.topk(exist[b], k=topk).indices

            selections = {
                "active": active_idx,
                "topK": topk_idx,
                "all": all_idx,
            }

            update_sum(sums, counts, "active_empty_frac", active_empty)

            for sel_name, sel_idx in selections.items():
                pred_visible, pred_raw, pred_vis_frac, pred_empty = selected_surface_sets(
                    pred_world_b=pred_world[b],
                    scale_b=scale[b],
                    shape_b=shape[b],
                    rotate_b=rotate[b],
                    trans_b=trans[b],
                    idx=sel_idx,
                    args=args,
                )

                update_sum(sums, counts, f"{sel_name}_slots_mean", float(sel_idx.numel()))
                update_sum(sums, counts, f"{sel_name}_visible_fraction", pred_vis_frac)
                update_sum(sums, counts, f"{sel_name}_visible_empty_frac", pred_empty)

                update_sum(
                    sums,
                    counts,
                    f"pred_cd_{sel_name}_raw_to_input",
                    chamfer_squared(input_pts, pred_raw),
                )
                update_sum(
                    sums,
                    counts,
                    f"pred_cd_{sel_name}_visible_to_input",
                    chamfer_squared(input_pts, pred_visible),
                )
                update_sum(
                    sums,
                    counts,
                    f"pred_cd_{sel_name}_visible_to_gt_visible",
                    chamfer_squared(gt_visible, pred_visible),
                )

    summary = {
        "name": name,
        "n_samples": n_samples,
    }

    for key in sorted(sums.keys()):
        summary[key] = sums[key] / max(counts[key], 1)

    return summary


def write_summary_csv(rows, path: Path):
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_summary(rows):
    print("\n=== Summary ===")
    for row in rows:
        print(f"\n[{row['name']}] n={row['n_samples']}")

        important_order = [
            # GT baseline
            "gt_cd_visible_to_input",
            "gt_cd_raw_to_input",
            "gt_visible_fraction",
            # pred object -> input
            "pred_cd_active_visible_to_input",
            "pred_cd_active_raw_to_input",
            "pred_cd_topK_visible_to_input",
            "pred_cd_topK_raw_to_input",
            "pred_cd_all_visible_to_input",
            "pred_cd_all_raw_to_input",
            # pred object -> GT object
            "pred_cd_active_visible_to_gt_visible",
            "pred_cd_topK_visible_to_gt_visible",
            "pred_cd_all_visible_to_gt_visible",
            # visibility diagnostics
            "active_slots_mean",
            "topK_slots_mean",
            "all_slots_mean",
            "active_visible_fraction",
            "topK_visible_fraction",
            "all_visible_fraction",
            "active_empty_frac",
            # decomposition
            "decomp_assign_acc",
            "decomp_assign_nll",
            "decomp_count_acc",
            "decomp_exist_bce",
            "pred_count_hard",
            "pred_count_soft",
            "true_count",
            "exist_matched_mean",
            "exist_unmatched_mean",
        ]

        printed = set()
        for k in important_order:
            if k in row:
                v = row[k]
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
                printed.add(k)

        for k, v in row.items():
            if k in printed or k in {"name", "n_samples"}:
                continue
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt-a", type=Path, required=True)
    ap.add_argument("--ckpt-b", type=Path, required=True)
    ap.add_argument("--name-a", type=str, default="model_a")
    ap.add_argument("--name-b", type=str, default="model_b")

    ap.add_argument("--use-lm-a", action="store_true")
    ap.add_argument("--use-lm-b", action="store_true")

    ap.add_argument("--cfg-a", type=Path, default=None)
    ap.add_argument("--cfg-b", type=Path, default=None)

    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test.txt")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--normal-mode", type=str, default="radial")
    ap.add_argument("--max-samples", type=int, default=None)

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--surface-n-samples", type=int, default=1024)
    ap.add_argument("--surface-D-eta", type=float, default=0.05)
    ap.add_argument("--surface-D-omega", type=float, default=0.05)

    ap.add_argument("--object-cd-max-input-points", type=int, default=4096)
    ap.add_argument("--exist-threshold", type=float, default=0.5)

    # Visibility removal matches SQ-Zero generation idea:
    # remove point from SQ p if it lies inside another SQ q.
    # Generation used f < 1.0 as inside for the older naive path.
    ap.add_argument("--inside-threshold", type=float, default=1.0)

    ap.add_argument("--match-w-assign", type=float, default=1.0)
    ap.add_argument("--match-w-exist", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=1e-8)

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg_a_path = args.cfg_a or infer_config_path(args.ckpt_a)
    cfg_b_path = args.cfg_b or infer_config_path(args.ckpt_b)

    print(f"Loading A: {args.name_a}")
    print(f"  ckpt: {args.ckpt_a}")
    print(f"  cfg:  {cfg_a_path}")

    print(f"Loading B: {args.name_b}")
    print(f"  ckpt: {args.ckpt_b}")
    print(f"  cfg:  {cfg_b_path}")

    model_a, cfg_a = load_model(args.ckpt_a, cfg_a_path, device, use_lm=args.use_lm_a)
    model_b, cfg_b = load_model(args.ckpt_b, cfg_b_path, device, use_lm=args.use_lm_b)

    eval_cfg = make_eval_cfg(cfg_a, args)
    dataset = SQZeroLMDB(split="val", cfg=eval_cfg)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    sampler = EqualDistanceSamplerSQ(
        n_samples=args.surface_n_samples,
        D_eta=args.surface_D_eta,
        D_omega=args.surface_D_omega,
    )

    print("Dataset:")
    print(f"  root: {args.data_root}")
    print(f"  split: {args.split}")
    print(f"  len: {len(dataset)}")
    print("Object CD:")
    print(f"  surface_n_samples per primitive: {args.surface_n_samples}")
    print(f"  max input points: {args.object_cd_max_input_points}")
    print(f"  exist threshold: {args.exist_threshold}")
    print(f"  inside threshold for visibility removal: {args.inside_threshold}")

    summary_a = evaluate_model(model_a, args.name_a, loader, sampler, args, device)
    summary_b = evaluate_model(model_b, args.name_b, loader, sampler, args, device)

    rows = [summary_a, summary_b]

    json_path = args.out_dir / "summary.json"
    csv_path = args.out_dir / "summary.csv"

    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    write_summary_csv(rows, csv_path)

    print_summary(rows)

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()