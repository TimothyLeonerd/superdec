#!/usr/bin/env python3
import argparse
import csv
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
from superdec.data.sqzero_lmdb import SQZeroLMDB
from superdec.loss.loss import sampling_from_parametric_space_to_equivalent_points
from superdec.loss.sampler import EqualDistanceSamplerSQ


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


def load_model(ckpt_path: Path, cfg_path: Path, device):
    cfg = OmegaConf.load(cfg_path)
    model = SuperDec(cfg.superdec).to(device)

    state = load_checkpoint_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARN] {ckpt_path}: missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"  missing: {k}")

    if unexpected:
        print(f"[WARN] {ckpt_path}: unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  unexpected: {k}")

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
        cfg.sqzero_lmdb.max_train_samples = args.max_samples
        cfg.sqzero_lmdb.max_val_samples = args.max_samples
    else:
        cfg.sqzero_lmdb.max_train_samples = None
        cfg.sqzero_lmdb.max_val_samples = None

    if "trainer" not in cfg:
        cfg.trainer = {}
    cfg.trainer.augmentations = False

    return cfg


def local_to_world(local_points, rotate, trans):
    return torch.einsum("bpij,bpsj->bpsi", rotate, local_points) + trans.unsqueeze(2)


def sq_implicit_value(points, scale, shape, rotate, trans, eps=1e-8):
    if scale.numel() == 0:
        return points.new_empty((0, points.shape[0]))

    centered = points.unsqueeze(0) - trans.unsqueeze(1)
    local = torch.einsum("mij,mnj->mni", rotate.transpose(-1, -2), centered)

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


def visible_surface_union(surfaces_by_prim, scale, shape, rotate, trans, inside_threshold=1.0, eps=1e-8):
    M = surfaces_by_prim.shape[0]
    raw = surfaces_by_prim.reshape(-1, 3)

    if M <= 1:
        return raw, raw, 1.0, 0.0

    chunks = []

    for p in range(M):
        pts_p = surfaces_by_prim[p]
        f = sq_implicit_value(
            points=pts_p,
            scale=scale,
            shape=shape,
            rotate=rotate,
            trans=trans,
            eps=eps,
        )

        other = torch.ones(M, dtype=torch.bool, device=pts_p.device)
        other[p] = False

        inside_any_other = (f[other] < inside_threshold).any(dim=0)
        keep = ~inside_any_other

        if keep.any():
            chunks.append(pts_p[keep])

    if chunks:
        visible = torch.cat(chunks, dim=0)
        empty = 0.0
    else:
        visible = raw.new_empty((0, 3))
        empty = 1.0

    frac = float(visible.shape[0]) / max(float(raw.shape[0]), 1.0)
    return visible, raw, frac, empty


def chamfer_squared(x, y):
    if x.numel() == 0 or y.numel() == 0:
        return x.new_tensor(float("nan"))

    d = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2.0)[0]
    d2 = d * d
    return d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean()


def maybe_subsample_points(points, max_points):
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = torch.linspace(0, points.shape[0] - 1, steps=max_points, device=points.device).long()
    return points[idx]


@torch.no_grad()
def hungarian_assignment_match(assign_b, exist_b, labels_b, K_b, eps=1e-8, match_w_assign=1.0, match_w_exist=0.0):
    _, P = assign_b.shape
    K = int(K_b)
    cost = assign_b.new_zeros((P, K))

    for k in range(K):
        mask = labels_b == k
        if int(mask.sum().item()) == 0:
            cost[:, k] = 1e6
            continue

        probs = assign_b[mask, :]
        assign_cost = -torch.log(probs.clamp_min(eps)).mean(dim=0)
        exist_cost = -torch.log(exist_b.clamp_min(eps))
        cost[:, k] = match_w_assign * assign_cost + match_w_exist * exist_cost

    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

    matched = np.empty((K,), dtype=np.int64)
    for p, k in zip(row_ind, col_ind):
        matched[k] = p

    return torch.as_tensor(matched, dtype=torch.long, device=assign_b.device)


def select_slots(exist_b, K_b, P, mode, threshold):
    if mode == "active":
        idx = torch.nonzero(exist_b > threshold, as_tuple=False).flatten()
        if idx.numel() == 0:
            idx = torch.topk(exist_b, k=1).indices
        return idx

    if mode == "topK":
        k = max(1, min(int(K_b), P))
        return torch.topk(exist_b, k=k).indices

    if mode == "all":
        return torch.arange(P, device=exist_b.device)

    raise ValueError(mode)


def selected_visible_and_raw(pred_world_b, scale_b, shape_b, rotate_b, trans_b, idx, args):
    surf = pred_world_b[idx]
    scale = scale_b[idx]
    shape = shape_b[idx]
    rotate = rotate_b[idx]
    trans = trans_b[idx]

    return visible_surface_union(
        surf,
        scale,
        shape,
        rotate,
        trans,
        inside_threshold=args.inside_threshold,
        eps=args.eps,
    )


def scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def get_model_id(batch, b, global_index):
    model_ids = batch.get("model_id", None)
    if model_ids is None:
        return f"idx{global_index}"
    if isinstance(model_ids, (list, tuple)):
        return str(model_ids[b])
    return str(model_ids)


@torch.no_grad()
def gt_visible_surface_for_sample(batch, b, sampler, args, device):
    points = batch["points"].to(device).float()
    K = batch["K"].to(device).long()

    gt_scale = batch["gt_scale"].to(device).float()
    gt_shape = batch["gt_shape"].to(device).float()
    gt_rotate = batch["gt_rotate"].to(device).float()
    gt_trans = batch["gt_trans"].to(device).float()

    K_b = int(K[b].item())

    gt_local, _ = sampling_from_parametric_space_to_equivalent_points(
        gt_scale[b:b + 1, :K_b, :],
        gt_shape[b:b + 1, :K_b, :],
        sampler,
    )

    gt_world = local_to_world(
        gt_local,
        gt_rotate[b:b + 1, :K_b, :, :],
        gt_trans[b:b + 1, :K_b, :],
    )[0]

    gt_visible, gt_raw, gt_frac, gt_empty = visible_surface_union(
        gt_world,
        gt_scale[b, :K_b],
        gt_shape[b, :K_b],
        gt_rotate[b, :K_b],
        gt_trans[b, :K_b],
        inside_threshold=args.inside_threshold,
        eps=args.eps,
    )

    input_pts = maybe_subsample_points(points[b], args.object_cd_max_input_points)

    return {
        "gt_cd_visible_to_input": scalar(chamfer_squared(input_pts, gt_visible)),
        "gt_cd_raw_to_input": scalar(chamfer_squared(input_pts, gt_raw)),
        "gt_visible_fraction": gt_frac,
        "gt_visible_empty_frac": gt_empty,
        "gt_visible": gt_visible,
        "gt_raw": gt_raw,
        "input_pts": input_pts,
    }


@torch.no_grad()
def eval_one_model_on_batch(model, batch, sampler, args, device):
    points = batch["points"].to(device).float()
    labels = batch["labels"].to(device).long()
    K = batch["K"].to(device).long()

    out = model(points)

    assign = out["assign_matrix"]
    exist = out["exist"].squeeze(-1)
    scale = out["scale"]
    shape = out["shape"]
    rotate = out["rotate"]
    trans = out["trans"]

    pred_local, _ = sampling_from_parametric_space_to_equivalent_points(scale, shape, sampler)
    pred_world = local_to_world(pred_local, rotate, trans)

    B, N, P = assign.shape
    rows = []

    for b in range(B):
        K_b = int(K[b].item())
        input_pts = maybe_subsample_points(points[b], args.object_cd_max_input_points)

        gt_info = gt_visible_surface_for_sample(batch, b, sampler, args, device)
        gt_visible = gt_info["gt_visible"]

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

        target_slot = matched_slots[labels[b]]
        point_idx = torch.arange(N, device=device)
        chosen = assign[b, point_idx, target_slot]

        pred_slot = assign[b].argmax(dim=1)

        row = {
            "K": K_b,
            "decomp_assign_acc": scalar((pred_slot == target_slot).float().mean()),
            "decomp_assign_nll": scalar(-torch.log(chosen.clamp_min(args.eps)).mean()),
            "decomp_count_acc": scalar(((exist[b] > args.exist_threshold).sum().float() == float(K_b)).float()),
            "decomp_exist_bce": scalar(F.binary_cross_entropy(exist[b], exist_target, reduction="mean")),
            "pred_count_hard": scalar((exist[b] > args.exist_threshold).sum().float()),
            "pred_count_soft": scalar(exist[b].sum()),
            "exist_matched_mean": scalar(exist[b][exist_target > 0.5].mean()),
        }

        unmatched = exist[b][exist_target < 0.5]
        row["exist_unmatched_mean"] = scalar(unmatched.mean()) if unmatched.numel() else float("nan")

        for mode in ["active", "topK", "all"]:
            idx = select_slots(
                exist_b=exist[b],
                K_b=K_b,
                P=P,
                mode=mode,
                threshold=args.exist_threshold,
            )

            pred_visible, pred_raw, pred_frac, pred_empty = selected_visible_and_raw(
                pred_world_b=pred_world[b],
                scale_b=scale[b],
                shape_b=shape[b],
                rotate_b=rotate[b],
                trans_b=trans[b],
                idx=idx,
                args=args,
            )

            row[f"{mode}_slots"] = int(idx.numel())
            row[f"{mode}_visible_fraction"] = pred_frac
            row[f"{mode}_visible_empty"] = pred_empty
            row[f"pred_cd_{mode}_visible_to_input"] = scalar(chamfer_squared(input_pts, pred_visible))
            row[f"pred_cd_{mode}_raw_to_input"] = scalar(chamfer_squared(input_pts, pred_raw))
            row[f"pred_cd_{mode}_visible_to_gt_visible"] = scalar(chamfer_squared(gt_visible, pred_visible))

        rows.append(row)

    return rows


def write_csv(rows, path):
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_pair_rankings(wide_rows, names, out_dir, metric, top_n):
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            key_a = f"{a}__{metric}"
            key_b = f"{b}__{metric}"

            valid = []
            for r in wide_rows:
                va = float(r[key_a])
                vb = float(r[key_b])
                if math.isfinite(va) and math.isfinite(vb):
                    rr = dict(r)
                    rr[f"diff_{b}_minus_{a}"] = vb - va
                    valid.append(rr)

            # Negative means b better than a. Positive means a better than b.
            b_better = sorted(valid, key=lambda r: r[f"diff_{b}_minus_{a}"])[:top_n]
            a_better = sorted(valid, key=lambda r: r[f"diff_{b}_minus_{a}"], reverse=True)[:top_n]

            for label, rows in [(f"{b}_better_than_{a}", b_better), (f"{a}_better_than_{b}", a_better)]:
                path = out_dir / f"{metric}__{label}.txt"
                with path.open("w") as f:
                    f.write(f"Ranking for metric: {metric}\n")
                    f.write(f"Pair: {a} vs {b}\n")
                    f.write(f"diff = {b} - {a}; lower metric is better\n\n")
                    for r in rows:
                        f.write(
                            f"sample_index={r['sample_index']} "
                            f"model_id={r['model_id']} "
                            f"K={r['K']} "
                            f"{a}={float(r[key_a]):.8f} "
                            f"{b}={float(r[key_b]):.8f} "
                            f"diff={float(r[f'diff_{b}_minus_{a}']):.8f}\n"
                        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--ckpts", nargs="+", type=Path, required=True)
    ap.add_argument("--cfgs", nargs="+", type=Path, required=True)

    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test.txt")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--normal-mode", type=str, default="radial")
    ap.add_argument("--max-samples", type=int, default=None)

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--surface-n-samples", type=int, default=1024)
    ap.add_argument("--surface-D-eta", type=float, default=0.05)
    ap.add_argument("--surface-D-omega", type=float, default=0.05)

    ap.add_argument("--object-cd-max-input-points", type=int, default=4096)
    ap.add_argument("--exist-threshold", type=float, default=0.5)
    ap.add_argument("--inside-threshold", type=float, default=1.0)

    ap.add_argument("--match-w-assign", type=float, default=1.0)
    ap.add_argument("--match-w-exist", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=1e-8)

    ap.add_argument("--ranking-metric", type=str, default="pred_cd_active_visible_to_input")
    ap.add_argument("--top-n", type=int, default=25)

    args = ap.parse_args()

    if not (len(args.names) == len(args.ckpts) == len(args.cfgs)):
        raise ValueError("--names, --ckpts, and --cfgs must have the same length.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = []
    cfg0 = None

    for name, ckpt, cfg_path in zip(args.names, args.ckpts, args.cfgs):
        print(f"Loading {name}:")
        print(f"  ckpt={ckpt}")
        print(f"  cfg={cfg_path}")
        model, cfg = load_model(ckpt, cfg_path, device)
        models.append((name, model))
        if cfg0 is None:
            cfg0 = cfg

    eval_cfg = make_eval_cfg(cfg0, args)
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

    wide_rows = []
    global_index = 0

    for batch in tqdm(loader, desc="batches"):
        B = batch["points"].shape[0]

        base_rows = []
        for b in range(B):
            gt_info = gt_visible_surface_for_sample(batch, b, sampler, args, device)
            model_id = get_model_id(batch, b, global_index + b)
            K_b = int(batch["K"][b].item())

            base_rows.append({
                "sample_index": global_index + b,
                "model_id": model_id,
                "K": K_b,
                "gt_cd_visible_to_input": gt_info["gt_cd_visible_to_input"],
                "gt_cd_raw_to_input": gt_info["gt_cd_raw_to_input"],
                "gt_visible_fraction": gt_info["gt_visible_fraction"],
            })

        for name, model in models:
            model_rows = eval_one_model_on_batch(model, batch, sampler, args, device)
            for b, mr in enumerate(model_rows):
                for k, v in mr.items():
                    if k == "K":
                        continue
                    base_rows[b][f"{name}__{k}"] = v

        wide_rows.extend(base_rows)
        global_index += B

    metrics_csv = args.out_dir / "per_sample_metrics.csv"
    write_csv(wide_rows, metrics_csv)
    print(f"Wrote: {metrics_csv}")

    rank_dir = args.out_dir / "rankings"
    write_pair_rankings(
        wide_rows=wide_rows,
        names=args.names,
        out_dir=rank_dir,
        metric=args.ranking_metric,
        top_n=args.top_n,
    )
    print(f"Wrote rankings to: {rank_dir}")

    print("\nMost useful ranking files:")
    print(f"  {rank_dir}/{args.ranking_metric}__<modelA>_better_than_<modelB>.txt")
    print("\nUse the listed sample_index values with plot_sqzero_model_comparison.py.")


if __name__ == "__main__":
    main()
