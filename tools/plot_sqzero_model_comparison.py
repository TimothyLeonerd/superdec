#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

from superdec.superdec import SuperDec
from superdec.data.sqzero_lmdb import SQZeroLMDB
from superdec.loss.loss import sampling_from_parametric_space_to_equivalent_points
from superdec.loss.sampler import EqualDistanceSamplerSQ


PALETTE = np.array([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
], dtype=np.uint8)


def colors_for_ids(ids):
    ids_np = np.asarray(ids, dtype=np.int64)
    return PALETTE[ids_np % len(PALETTE)]


def infer_config_path(ckpt_path: Path) -> Path:
    cfg = ckpt_path.parent / "config.yaml"
    if not cfg.exists():
        raise FileNotFoundError(
            f"Could not infer config path for {ckpt_path}. Expected {cfg}."
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


def load_model(ckpt_path: Path, cfg_path: Path, device: torch.device):
    cfg = OmegaConf.load(cfg_path)
    model = SuperDec(cfg.superdec).to(device)

    state = load_checkpoint_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARN] Missing keys for {ckpt_path}: {len(missing)}")
        for k in missing[:10]:
            print(f"  missing: {k}")

    if unexpected:
        print(f"[WARN] Unexpected keys for {ckpt_path}: {len(unexpected)}")
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
    cfg.sqzero_lmdb.max_train_samples = None
    cfg.sqzero_lmdb.max_val_samples = None

    if "trainer" not in cfg:
        cfg.trainer = {}
    cfg.trainer.augmentations = False

    return cfg


def local_to_world(local_points, rotate, trans):
    """local_points [B,P,S,3], rotate [B,P,3,3], trans [B,P,3]."""
    return torch.einsum("bpij,bpsj->bpsi", rotate, local_points) + trans.unsqueeze(2)


def sq_implicit_value(points, scale, shape, rotate, trans, eps=1e-8):
    """Evaluate SQ implicit function.

    Args:
        points: [N,3]
        scale:  [M,3]
        shape:  [M,2]
        rotate: [M,3,3], local-to-world
        trans:  [M,3]

    Returns:
        f: [M,N], inside if f < 1.
    """
    if scale.numel() == 0:
        return points.new_empty((0, points.shape[0]))

    centered = points.unsqueeze(0) - trans.unsqueeze(1)  # [M,N,3]
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


def visible_surface_points(
    surfaces_by_prim,
    scale,
    shape,
    rotate,
    trans,
    inside_threshold=1.0,
    eps=1e-8,
):
    """Remove surface points that lie inside another selected SQ.

    Args:
        surfaces_by_prim: [M,S,3]

    Returns:
        visible_points: [Nv,3]
        visible_local_ids: [Nv], ids in selected-local primitive index
        raw_points: [M*S,3]
        raw_local_ids: [M*S]
    """
    M, S, _ = surfaces_by_prim.shape

    raw_points = surfaces_by_prim.reshape(-1, 3)
    raw_ids = torch.arange(M, device=surfaces_by_prim.device).repeat_interleave(S)

    if M <= 1:
        return raw_points, raw_ids, raw_points, raw_ids

    visible_chunks = []
    id_chunks = []

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

        other = torch.ones(M, dtype=torch.bool, device=pts_p.device)
        other[p] = False

        inside_any_other = (f[other] < inside_threshold).any(dim=0)
        keep = ~inside_any_other

        if keep.any():
            visible_chunks.append(pts_p[keep])
            id_chunks.append(torch.full(
                (int(keep.sum().item()),),
                p,
                device=pts_p.device,
                dtype=torch.long,
            ))

    if visible_chunks:
        visible = torch.cat(visible_chunks, dim=0)
        visible_ids = torch.cat(id_chunks, dim=0)
    else:
        visible = raw_points.new_empty((0, 3))
        visible_ids = raw_ids.new_empty((0,))

    return visible, visible_ids, raw_points, raw_ids


def hungarian_assignment_match(assign_b, exist_b, labels_b, K_b, eps=1e-8, match_w_assign=1.0, match_w_exist=0.0):
    """Return matched_slots[k] = predicted slot p for GT primitive k."""
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


def select_slots(exist, K, mode, threshold):
    P = exist.shape[0]

    if mode == "active":
        idx = torch.nonzero(exist > threshold, as_tuple=False).flatten()
        if idx.numel() == 0:
            idx = torch.topk(exist, k=1).indices
        return idx

    if mode == "topK":
        k = max(1, min(int(K), P))
        return torch.topk(exist, k=k).indices

    if mode == "all":
        return torch.arange(P, device=exist.device)

    raise ValueError(f"Unsupported slot mode: {mode}")


def get_sample(dataset, args):
    if args.model_id is not None:
        keys = getattr(dataset, "keys", None)
        if keys is None:
            raise RuntimeError("Dataset has no .keys attribute; cannot select by model_id.")

        try:
            index = list(keys).index(args.model_id)
        except ValueError:
            raise ValueError(f"model_id {args.model_id!r} not found in split.")
    else:
        index = int(args.sample_index)

    sample = dataset[index]
    return index, sample


def to_device_sample(sample, device):
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def run_model(model, points_b, sampler):
    out = model(points_b)

    pred_local, _ = sampling_from_parametric_space_to_equivalent_points(
        out["scale"],
        out["shape"],
        sampler,
    )

    pred_world = local_to_world(
        pred_local,
        out["rotate"],
        out["trans"],
    )

    return out, pred_world


def remap_local_ids(local_ids, local_to_color_id):
    if local_ids.numel() == 0:
        return local_ids

    mapped = torch.empty_like(local_ids)
    for local_idx, color_id in enumerate(local_to_color_id):
        mapped[local_ids == local_idx] = int(color_id)
    return mapped


def build_pred_surface_dict(out, pred_world, K, labels, args):
    if out["exist"].ndim == 3:
        exist = out["exist"][0].squeeze(-1)
    else:
        exist = out["exist"][0]

    idx = select_slots(exist, K=K, mode=args.slot_mode, threshold=args.exist_threshold)

    surf = pred_world[0, idx]
    scale = out["scale"][0, idx]
    shape = out["shape"][0, idx]
    rotate = out["rotate"][0, idx]
    trans = out["trans"][0, idx]

    visible_pts, visible_local_ids, raw_pts, raw_local_ids = visible_surface_points(
        surfaces_by_prim=surf,
        scale=scale,
        shape=shape,
        rotate=rotate,
        trans=trans,
        inside_threshold=args.inside_threshold,
        eps=args.eps,
    )

    if args.color_pred_by == "slot":
        # Use selected-local slot index colors.
        visible_ids = visible_local_ids
        raw_ids = raw_local_ids
    elif args.color_pred_by == "matched_gt":
        assign = out["assign_matrix"][0]  # [N,P]
        matched_slots = hungarian_assignment_match(
            assign_b=assign,
            exist_b=exist,
            labels_b=labels,
            K_b=K,
            eps=args.eps,
            match_w_assign=1.0,
            match_w_exist=0.0,
        )  # [K], gt -> pred slot

        pred_slot_to_gt = {}
        for gt_id, pred_slot in enumerate(matched_slots.detach().cpu().tolist()):
            pred_slot_to_gt[int(pred_slot)] = int(gt_id)

        local_to_color_id = []
        for pred_slot in idx.detach().cpu().tolist():
            # Unmatched predicted slots get colors after the GT range.
            color_id = pred_slot_to_gt.get(int(pred_slot), int(K) + int(pred_slot))
            local_to_color_id.append(color_id)

        visible_ids = remap_local_ids(visible_local_ids, local_to_color_id)
        raw_ids = remap_local_ids(raw_local_ids, local_to_color_id)
    else:
        raise ValueError(f"Unsupported color_pred_by: {args.color_pred_by}")

    result = {
        "selected_slots": idx.detach().cpu().tolist(),
        "exist": exist.detach().cpu().tolist(),
        "visible_points": visible_pts,
        "visible_ids": visible_ids,
        "raw_points": raw_pts,
        "raw_ids": raw_ids,
    }
    return result


def build_gt_surface(sample, sampler, args):
    gt_scale = sample["gt_scale"].unsqueeze(0)
    gt_shape = sample["gt_shape"].unsqueeze(0)
    gt_rotate = sample["gt_rotate"].unsqueeze(0)
    gt_trans = sample["gt_trans"].unsqueeze(0)
    K = int(sample["K"].item())

    gt_local, _ = sampling_from_parametric_space_to_equivalent_points(
        gt_scale[:, :K, :],
        gt_shape[:, :K, :],
        sampler,
    )

    gt_world = local_to_world(
        gt_local,
        gt_rotate[:, :K, :, :],
        gt_trans[:, :K, :],
    )[0]

    visible_pts, visible_ids, raw_pts, raw_ids = visible_surface_points(
        surfaces_by_prim=gt_world,
        scale=gt_scale[0, :K],
        shape=gt_shape[0, :K],
        rotate=gt_rotate[0, :K],
        trans=gt_trans[0, :K],
        inside_threshold=args.inside_threshold,
        eps=args.eps,
    )

    return {
        "visible_points": visible_pts,
        "visible_ids": visible_ids,
        "raw_points": raw_pts,
        "raw_ids": raw_ids,
    }


def choose_points(surface_dict, visibility):
    if visibility == "raw":
        return surface_dict["raw_points"], surface_dict["raw_ids"]
    if visibility == "visible":
        return surface_dict["visible_points"], surface_dict["visible_ids"]
    raise ValueError(f"Unsupported visibility: {visibility}")


def downsample_for_plot(points, ids, max_points):
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points, ids

    idx = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=max_points,
        device=points.device,
    ).long()
    return points[idx], ids[idx]


def set_axes_equal(ax, xyz_all):
    xyz = np.asarray(xyz_all)
    if xyz.size == 0:
        return

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-3)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    ax.set_box_aspect((1, 1, 1))


def scatter_panel(ax, points, ids, title, point_size):
    points_np = points.detach().cpu().numpy()
    ids_np = ids.detach().cpu().numpy().astype(np.int64)

    if points_np.shape[0] == 0:
        ax.set_title(title + "\n(empty)")
        return points_np

    colors = colors_for_ids(ids_np) / 255.0

    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        c=colors,
        s=point_size,
        depthshade=False,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return points_np


def make_plot(
    out_png,
    input_points,
    input_labels,
    gt_surface,
    pred_a,
    pred_b,
    name_a,
    name_b,
    visibility,
    args,
):
    fig = plt.figure(figsize=(18, 5))

    axes = [
        fig.add_subplot(1, 4, 1, projection="3d"),
        fig.add_subplot(1, 4, 2, projection="3d"),
        fig.add_subplot(1, 4, 3, projection="3d"),
        fig.add_subplot(1, 4, 4, projection="3d"),
    ]

    xyz_for_limits = []

    pts_input, ids_input = downsample_for_plot(
        input_points,
        input_labels,
        args.max_plot_points,
    )
    xyz_for_limits.append(scatter_panel(
        axes[0],
        pts_input,
        ids_input,
        "input point cloud\nGT labels",
        args.input_point_size,
    ))

    gt_pts, gt_ids = choose_points(gt_surface, visibility)
    gt_pts, gt_ids = downsample_for_plot(gt_pts, gt_ids, args.max_plot_surface_points)
    xyz_for_limits.append(scatter_panel(
        axes[1],
        gt_pts,
        gt_ids,
        f"GT SQs\n{visibility}",
        args.surface_point_size,
    ))

    a_pts, a_ids = choose_points(pred_a, visibility)
    a_pts, a_ids = downsample_for_plot(a_pts, a_ids, args.max_plot_surface_points)
    xyz_for_limits.append(scatter_panel(
        axes[2],
        a_pts,
        a_ids,
        f"{name_a}\n{args.slot_mode}, {visibility}",
        args.surface_point_size,
    ))

    b_pts, b_ids = choose_points(pred_b, visibility)
    b_pts, b_ids = downsample_for_plot(b_pts, b_ids, args.max_plot_surface_points)
    xyz_for_limits.append(scatter_panel(
        axes[3],
        b_pts,
        b_ids,
        f"{name_b}\n{args.slot_mode}, {visibility}",
        args.surface_point_size,
    ))

    nonempty = [x for x in xyz_for_limits if x.size > 0]
    if nonempty:
        xyz_all = np.concatenate(nonempty, axis=0)
        for ax in axes:
            set_axes_equal(ax, xyz_all)
            ax.view_init(elev=args.elev, azim=args.azim)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt-a", type=Path, required=True)
    ap.add_argument("--ckpt-b", type=Path, required=True)
    ap.add_argument("--name-a", type=str, default="original")
    ap.add_argument("--name-b", type=str, default="supervised")
    ap.add_argument("--cfg-a", type=Path, default=None)
    ap.add_argument("--cfg-b", type=Path, default=None)

    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test.txt")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--model-id", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--normal-mode", type=str, default="radial")

    ap.add_argument("--slot-mode", choices=["active", "topK", "all"], default="active")
    ap.add_argument("--surface-visibility", choices=["visible", "raw", "both"], default="visible")
    ap.add_argument("--color-pred-by", choices=["slot", "matched_gt"], default="slot")

    ap.add_argument("--surface-n-samples", type=int, default=2048)
    ap.add_argument("--surface-D-eta", type=float, default=0.05)
    ap.add_argument("--surface-D-omega", type=float, default=0.05)

    ap.add_argument("--exist-threshold", type=float, default=0.5)
    ap.add_argument("--inside-threshold", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-8)

    ap.add_argument("--max-plot-points", type=int, default=4096)
    ap.add_argument("--max-plot-surface-points", type=int, default=12000)
    ap.add_argument("--input-point-size", type=float, default=1.0)
    ap.add_argument("--surface-point-size", type=float, default=1.0)

    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=45.0)
    ap.add_argument("--dpi", type=int, default=180)

    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg_a_path = args.cfg_a or infer_config_path(args.ckpt_a)
    cfg_b_path = args.cfg_b or infer_config_path(args.ckpt_b)

    model_a, cfg_a = load_model(args.ckpt_a, cfg_a_path, device)
    model_b, cfg_b = load_model(args.ckpt_b, cfg_b_path, device)

    eval_cfg = make_eval_cfg(cfg_a, args)
    dataset = SQZeroLMDB(split="val", cfg=eval_cfg)

    sample_index, sample_raw = get_sample(dataset, args)
    sample = to_device_sample(sample_raw, device)

    model_id = sample.get("model_id", f"index_{sample_index}")
    K = int(sample["K"].item())

    print(f"Selected sample_index={sample_index}")
    print(f"model_id={model_id}")
    print(f"K={K}")

    points = sample["points"].float()
    labels = sample["labels"].long()

    points_b = points.unsqueeze(0)

    sampler = EqualDistanceSamplerSQ(
        n_samples=args.surface_n_samples,
        D_eta=args.surface_D_eta,
        D_omega=args.surface_D_omega,
    )

    with torch.no_grad():
        out_a, pred_world_a = run_model(model_a, points_b, sampler)
        out_b, pred_world_b = run_model(model_b, points_b, sampler)

    pred_a = build_pred_surface_dict(out_a, pred_world_a, K=K, labels=labels, args=args)
    pred_b = build_pred_surface_dict(out_b, pred_world_b, K=K, labels=labels, args=args)
    gt_surface = build_gt_surface(sample, sampler, args)

    visibility_modes = (
        ["visible", "raw"]
        if args.surface_visibility == "both"
        else [args.surface_visibility]
    )

    safe_model_id = str(model_id).replace(":", "_").replace("/", "_")
    stem = f"{safe_model_id}_idx{sample_index}_{args.slot_mode}_{args.color_pred_by}"

    for visibility in visibility_modes:
        out_png = args.out_dir / f"{stem}_{visibility}.png"
        make_plot(
            out_png=out_png,
            input_points=points,
            input_labels=labels,
            gt_surface=gt_surface,
            pred_a=pred_a,
            pred_b=pred_b,
            name_a=args.name_a,
            name_b=args.name_b,
            visibility=visibility,
            args=args,
        )
        print(f"Wrote PNG: {out_png}")


if __name__ == "__main__":
    main()