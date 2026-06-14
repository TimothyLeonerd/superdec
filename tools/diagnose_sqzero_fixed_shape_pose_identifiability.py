#!/usr/bin/env python3
"""SQ-Zero eps/scale/translation/rotation identifiability diagnostic.

This script does NOT train SuperDec. It directly optimizes per-primitive SQ
parameters with PyTorch autograd.

Main diagnostic question:
    If eps, scale, translation, and rotation are free, can Chamfer still recover
    eps, or can pose/scale compensate for wrong eps while keeping low loss?

It reads the existing SQ-Zero LMDB sidecars, extracts individual GT primitives,
and optimizes eps, optionally scale/translation/rotation, from one or more
initializations. It also writes optional eps-loss landscape plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


class ConfigNode(SimpleNamespace):
    """Tiny config object compatible with SQZeroLMDB's attribute and 'in' checks."""

    def __contains__(self, key: str) -> bool:  # used by get_transforms()
        return hasattr(self, key)


@dataclass
class PrimitiveRecord:
    global_prim_index: int
    batch_sample_index: int
    model_id: str
    primitive_k: int
    K: int
    visible_points_in_sample: int
    scale: torch.Tensor  # [3], normalized SuperDec frame
    shape: torch.Tensor  # [2]
    rotate: torch.Tensor  # [3, 3]
    trans: torch.Tensor  # [3]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_init_pairs(text: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Bad --init-eps-pairs entry {chunk!r}; expected e.g. '1.0,1.0;0.3,0.3'."
            )
        pairs.append((float(parts[0]), float(parts[1])))
    if not pairs:
        raise ValueError("--init-eps-pairs produced no pairs.")
    return pairs


def parse_scale_triples(text: str) -> List[Tuple[float, float, float]]:
    triples: List[Tuple[float, float, float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Bad --init-scale-mults entry {chunk!r}; expected e.g. '1,1,1;0.7,0.7,0.7'."
            )
        triples.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not triples:
        raise ValueError("--init-scale-mults produced no triples.")
    return triples


def parse_trans_offsets(text: str) -> List[Tuple[float, float, float]]:
    offsets: List[Tuple[float, float, float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Bad --init-trans-offsets entry {chunk!r}; expected e.g. '0.04,0,0;-0.04,0.04,0'."
            )
        offsets.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not offsets:
        raise ValueError("--init-trans-offsets produced no offsets.")
    return offsets


def parse_rotvecs_deg(text: str) -> List[Tuple[float, float, float]]:
    """Parse semicolon-separated rotation-vector initializations in degrees.

    Each triple is an axis-angle vector expressed as degrees around x/y/z in
    the GT/world coordinate basis. It is converted to radians later.
    Example: '15,0,0;0,15,0;0,0,15'.
    """
    triples: List[Tuple[float, float, float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Bad --init-rotvecs-deg entry {chunk!r}; expected e.g. '15,0,0;0,15,0'."
            )
        triples.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not triples:
        raise ValueError("--init-rotvecs-deg produced no triples.")
    return triples


def make_cfg(args: argparse.Namespace) -> ConfigNode:
    return ConfigNode(
        sqzero_lmdb=ConfigNode(
            path=args.data_root,
            n_points=args.n_points,
            normalize=True,
            load_sidecars=True,
            normal_mode=args.normal_mode,
            kmax=args.kmax,
            train_split=args.train_split,
            val_split=args.val_split,
            max_train_samples=args.max_samples if args.split == "train" else None,
            max_val_samples=args.max_samples if args.split in {"val", "test"} else None,
        ),
        trainer=ConfigNode(augmentations=False),
    )


def fixed_eta_omega_grid(
    n_eta: int,
    n_omega: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cell-center eta/omega parameter grid, flattened to [S]."""
    if n_eta <= 0 or n_omega <= 0:
        raise ValueError("n_eta and n_omega must be positive.")

    d_eta = math.pi / n_eta
    d_omega = 2.0 * math.pi / n_omega

    eta = torch.linspace(
        -math.pi / 2.0 + d_eta / 2.0,
        math.pi / 2.0 - d_eta / 2.0,
        n_eta,
        device=device,
        dtype=dtype,
    )
    omega = torch.linspace(
        -math.pi + d_omega / 2.0,
        math.pi - d_omega / 2.0,
        n_omega,
        device=device,
        dtype=dtype,
    )
    eta_grid, omega_grid = torch.meshgrid(eta, omega, indexing="ij")
    return eta_grid.reshape(-1), omega_grid.reshape(-1)


def fexp(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.clamp(torch.abs(x), min=1e-8) ** p)


def sq_local_points_fixed_grid(
    scale: torch.Tensor,
    eps: torch.Tensor,
    eta: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a single SQ in local coordinates on a fixed eta/omega grid.

    Args:
        scale: [3]
        eps:   [2]
        eta:   [S]
        omega: [S]
    Returns:
        points: [S, 3]
    """
    a1, a2, a3 = scale[0], scale[1], scale[2]
    e1, e2 = eps[0], eps[1]

    x = a1 * fexp(torch.cos(eta), e1) * fexp(torch.cos(omega), e2)
    y = a2 * fexp(torch.cos(eta), e1) * fexp(torch.sin(omega), e2)
    z = a3 * fexp(torch.sin(eta), e1)

    # Match the numerical guard used in superdec.loss.loss.
    tiny = x.new_tensor(1e-6)
    x = ((x > 0).float() * 2 - 1) * torch.maximum(torch.abs(x), tiny)
    y = ((y > 0).float() * 2 - 1) * torch.maximum(torch.abs(y), tiny)
    z = ((z > 0).float() * 2 - 1) * torch.maximum(torch.abs(z), tiny)

    return torch.stack([x, y, z], dim=-1)


def rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
    """Differentiable axis-angle vector -> rotation matrix.

    rotvec: [3], radians. Returns [3,3].
    """
    theta = torch.linalg.norm(rotvec)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)

    # Stable small-angle behavior: first-order approximation near zero.
    if float(theta.detach().cpu().item()) < 1e-8:
        kx, ky, kz = rotvec[0], rotvec[1], rotvec[2]
        K = torch.stack([
            torch.stack([rotvec.new_tensor(0.0), -kz, ky]),
            torch.stack([kz, rotvec.new_tensor(0.0), -kx]),
            torch.stack([-ky, kx, rotvec.new_tensor(0.0)]),
        ])
        return eye + K

    axis = rotvec / theta
    kx, ky, kz = axis[0], axis[1], axis[2]
    K = torch.stack([
        torch.stack([rotvec.new_tensor(0.0), -kz, ky]),
        torch.stack([kz, rotvec.new_tensor(0.0), -kx]),
        torch.stack([-ky, kx, rotvec.new_tensor(0.0)]),
    ])
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def rotation_geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_rel = R_pred @ R_gt.transpose(0, 1)
    cos = (torch.trace(R_rel) - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(cos))


def local_to_world(points: torch.Tensor, rotate: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """points [S,3], rotate local-to-world [3,3], trans [3]."""
    return points @ rotate.transpose(0, 1) + trans


def sq_points(
    scale: torch.Tensor,
    eps: torch.Tensor,
    rotate: torch.Tensor,
    trans: torch.Tensor,
    sampler: str,
    eta: torch.Tensor | None,
    omega: torch.Tensor | None,
    surface_n_samples: int,
    surface_D_eta: float,
    surface_D_omega: float,
) -> torch.Tensor:
    """Sample one SQ surface in world/normalized coordinates."""
    if sampler == "fixed_grid":
        assert eta is not None and omega is not None
        local = sq_local_points_fixed_grid(scale, eps, eta, omega)
        return local_to_world(local, rotate, trans)

    if sampler == "equal_distance":
        # Lazy import: fixed_grid mode should not require the compiled fast_sampler extension.
        from superdec.loss.loss import sampling_from_parametric_space_to_equivalent_points
        from superdec.loss.sampler import EqualDistanceSamplerSQ

        sq_sampler = EqualDistanceSamplerSQ(
            n_samples=surface_n_samples,
            D_eta=surface_D_eta,
            D_omega=surface_D_omega,
        )
        local, _ = sampling_from_parametric_space_to_equivalent_points(
            scale.view(1, 1, 3),
            eps.view(1, 1, 2),
            sq_sampler,
        )
        local = local[0, 0]
        return local_to_world(local, rotate, trans)

    raise ValueError(f"Unknown sampler: {sampler!r}")


def chamfer_squared(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2.0)[0] ** 2
    return d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean()


def surface_loss(x: torch.Tensor, y: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "chamfer":
        return chamfer_squared(x, y)
    if loss_type == "pointwise":
        if x.shape != y.shape:
            raise ValueError(f"Pointwise loss needs same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
        return ((x - y) ** 2).sum(dim=-1).mean()
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


def eps_from_raw(raw: torch.Tensor, eps_min: float, eps_max: float) -> torch.Tensor:
    return eps_min + (eps_max - eps_min) * torch.sigmoid(raw)


def raw_from_eps(eps: torch.Tensor, eps_min: float, eps_max: float) -> torch.Tensor:
    z = (eps - eps_min) / (eps_max - eps_min)
    z = torch.clamp(z, 1e-5, 1.0 - 1e-5)
    return torch.log(z / (1.0 - z))


def bounded_from_raw(raw: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    return min_value + (max_value - min_value) * torch.sigmoid(raw)


def raw_from_bounded(x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    z = (x - min_value) / (max_value - min_value)
    z = torch.clamp(z, 1e-5, 1.0 - 1e-5)
    return torch.log(z / (1.0 - z))


def load_primitives(args: argparse.Namespace, device: torch.device) -> List[PrimitiveRecord]:
    from superdec.data.sqzero_lmdb import SQZeroLMDB

    cfg = make_cfg(args)
    split_for_dataset = "val" if args.split == "test" else args.split
    ds = SQZeroLMDB(split=split_for_dataset, cfg=cfg)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    records: List[PrimitiveRecord] = []
    global_prim_idx = 0

    for batch in loader:
        labels = batch["labels"].long()
        K = batch["K"].long()
        model_ids = batch.get("model_id", None)
        B = labels.shape[0]

        for b in range(B):
            model_id = str(model_ids[b]) if model_ids is not None else f"sample_{len(records)}"
            K_b = int(K[b].item())
            for k in range(K_b):
                visible_count = int((labels[b] == k).sum().item())
                if visible_count < args.min_visible_points:
                    continue

                records.append(
                    PrimitiveRecord(
                        global_prim_index=global_prim_idx,
                        batch_sample_index=len(records),
                        model_id=model_id,
                        primitive_k=k,
                        K=K_b,
                        visible_points_in_sample=visible_count,
                        scale=batch["gt_scale"][b, k].float().to(device),
                        shape=batch["gt_shape"][b, k].float().to(device),
                        rotate=batch["gt_rotate"][b, k].float().to(device),
                        trans=batch["gt_trans"][b, k].float().to(device),
                    )
                )
                global_prim_idx += 1

                if args.max_primitives > 0 and len(records) >= args.max_primitives:
                    return records

    return records


def optimize_one(
    rec: PrimitiveRecord,
    init_pair: Tuple[float, float],
    init_scale_mult: Tuple[float, float, float],
    init_trans_offset: Tuple[float, float, float],
    init_rotvec_deg: Tuple[float, float, float],
    args: argparse.Namespace,
    eta: torch.Tensor | None,
    omega: torch.Tensor | None,
) -> dict:
    gt_eps = rec.shape

    with torch.no_grad():
        gt_surface = sq_points(
            rec.scale,
            gt_eps,
            rec.rotate,
            rec.trans,
            sampler=args.sampler,
            eta=eta,
            omega=omega,
            surface_n_samples=args.surface_n_samples,
            surface_D_eta=args.surface_D_eta,
            surface_D_omega=args.surface_D_omega,
        ).detach()

    # Fixed-shape diagnostic: eps is not optimized.
    # We keep init_eps for logging, but force it to GT shape.
    init_eps = gt_eps.detach().clone()
    raw_eps = None

    params: List[torch.Tensor] = []

    init_scale_mult_t = torch.tensor(init_scale_mult, device=rec.scale.device, dtype=rec.scale.dtype)
    init_scale = torch.clamp(rec.scale.detach() * init_scale_mult_t, min=args.scale_min, max=args.scale_max)
    raw_scale = None
    if args.optimize_scale:
        raw_scale = raw_from_bounded(init_scale, args.scale_min, args.scale_max).detach().clone().requires_grad_(True)
        params.append(raw_scale)

    init_trans_offset_t = torch.tensor(init_trans_offset, device=rec.trans.device, dtype=rec.trans.dtype)
    init_trans = rec.trans.detach() + init_trans_offset_t
    raw_trans = None
    if args.optimize_trans:
        # Translation is optimized directly/unbounded. This is intentional for the diagnostic:
        # we want to see whether the surface loss pulls an imperfect translation back to GT
        # or permits a wrong pose/shape tradeoff.
        raw_trans = init_trans.detach().clone().requires_grad_(True)
        params.append(raw_trans)

    init_rotvec = torch.deg2rad(
        torch.tensor(init_rotvec_deg, device=rec.rotate.device, dtype=rec.rotate.dtype)
    )
    raw_rotvec = None
    if args.optimize_rotate:
        # Rotation is optimized as a world-frame delta applied to GT rotation:
        # R_pred = exp(rotvec) @ R_gt.
        raw_rotvec = init_rotvec.detach().clone().requires_grad_(True)
        params.append(raw_rotvec)

    opt = torch.optim.Adam(params, lr=args.lr)

    def current_scale() -> torch.Tensor:
        if raw_scale is None:
            return rec.scale
        return bounded_from_raw(raw_scale, args.scale_min, args.scale_max)

    def current_trans() -> torch.Tensor:
        if raw_trans is None:
            return rec.trans
        return raw_trans

    def current_rotate() -> torch.Tensor:
        if raw_rotvec is None:
            return rec.rotate
        return rodrigues(raw_rotvec) @ rec.rotate

    loss_initial = None
    best = {
        "loss": float("inf"),
        "eps": None,
        "scale": None,
        "trans": None,
        "rotate": None,
        "step": -1,
    }

    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        pred_eps = gt_eps
        pred_surface = sq_points(
            current_scale(),
            pred_eps,
            current_rotate(),
            current_trans(),
            sampler=args.sampler,
            eta=eta,
            omega=omega,
            surface_n_samples=args.surface_n_samples,
            surface_D_eta=args.surface_D_eta,
            surface_D_omega=args.surface_D_omega,
        )
        loss = surface_loss(pred_surface, gt_surface, args.optim_loss)

        if step == 0:
            loss_initial = float(loss.detach().cpu().item())

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best["loss"]:
            best["loss"] = loss_value
            best["eps"] = pred_eps.detach().clone()
            best["scale"] = current_scale().detach().clone()
            best["trans"] = current_trans().detach().clone()
            best["rotate"] = current_rotate().detach().clone()
            best["step"] = step

        if step == args.steps:
            break
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        opt.step()

    final_eps = best["eps"]
    final_scale = best["scale"]
    final_trans = best["trans"]
    final_rotate = best["rotate"]
    assert (
        final_eps is not None
        and final_scale is not None
        and final_trans is not None
        and final_rotate is not None
        and loss_initial is not None
    )

    init_l1 = torch.abs(init_eps - gt_eps).mean().detach().cpu().item()
    final_l1 = torch.abs(final_eps - gt_eps).mean().detach().cpu().item()
    scale_l1 = torch.abs(final_scale - rec.scale).mean().detach().cpu().item()
    init_trans_l2 = torch.linalg.norm(init_trans - rec.trans).detach().cpu().item()
    trans_l2 = torch.linalg.norm(final_trans - rec.trans).detach().cpu().item()
    init_rotate = rodrigues(init_rotvec) @ rec.rotate
    init_rot_deg = rotation_geodesic_deg(init_rotate, rec.rotate).detach().cpu().item()
    final_rot_deg = rotation_geodesic_deg(final_rotate, rec.rotate).detach().cpu().item()

    return {
        "global_prim_index": rec.global_prim_index,
        "model_id": rec.model_id,
        "primitive_k": rec.primitive_k,
        "K": rec.K,
        "visible_points_in_sample": rec.visible_points_in_sample,
        "gt_scale_x": float(rec.scale[0].detach().cpu().item()),
        "gt_scale_y": float(rec.scale[1].detach().cpu().item()),
        "gt_scale_z": float(rec.scale[2].detach().cpu().item()),
        "gt_trans_x": float(rec.trans[0].detach().cpu().item()),
        "gt_trans_y": float(rec.trans[1].detach().cpu().item()),
        "gt_trans_z": float(rec.trans[2].detach().cpu().item()),
        "gt_eps1": float(gt_eps[0].detach().cpu().item()),
        "gt_eps2": float(gt_eps[1].detach().cpu().item()),
        "init_eps1": float(init_eps[0].detach().cpu().item()),
        "init_eps2": float(init_eps[1].detach().cpu().item()),
        "init_shape_l1": float(init_l1),
        "init_scale_x": float(init_scale[0].detach().cpu().item()),
        "init_scale_y": float(init_scale[1].detach().cpu().item()),
        "init_scale_z": float(init_scale[2].detach().cpu().item()),
        "init_scale_l1_vs_gt": float(torch.abs(init_scale - rec.scale).mean().detach().cpu().item()),
        "init_trans_x": float(init_trans[0].detach().cpu().item()),
        "init_trans_y": float(init_trans[1].detach().cpu().item()),
        "init_trans_z": float(init_trans[2].detach().cpu().item()),
        "init_trans_l2_vs_gt": float(init_trans_l2),
        "init_rotvec_x_deg": float(init_rotvec_deg[0]),
        "init_rotvec_y_deg": float(init_rotvec_deg[1]),
        "init_rotvec_z_deg": float(init_rotvec_deg[2]),
        "init_rot_geodesic_deg_vs_gt": float(init_rot_deg),
        "final_eps1": float(final_eps[0].detach().cpu().item()),
        "final_eps2": float(final_eps[1].detach().cpu().item()),
        "final_shape_l1": float(final_l1),
        "loss_initial": float(loss_initial),
        "loss_final_best": float(best["loss"]),
        "best_step": int(best["step"]),
        "final_scale_x": float(final_scale[0].detach().cpu().item()),
        "final_scale_y": float(final_scale[1].detach().cpu().item()),
        "final_scale_z": float(final_scale[2].detach().cpu().item()),
        "scale_l1_vs_gt": float(scale_l1),
        "final_trans_x": float(final_trans[0].detach().cpu().item()),
        "final_trans_y": float(final_trans[1].detach().cpu().item()),
        "final_trans_z": float(final_trans[2].detach().cpu().item()),
        "trans_l2_vs_gt": float(trans_l2),
        "rot_geodesic_deg_vs_gt": float(final_rot_deg),
    }

def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> List[dict]:
    """Summarize one selected row per primitive.

    For --optimize-scale, the key selection is by final loss, because that is
    the actual objective the optimizer/model would prefer. We also report a
    shape-selected view to separate "can some init recover eps" from "does the
    Chamfer optimum prefer correct eps".
    """
    best_by_loss = {}
    best_by_shape = {}
    for r in rows:
        key = int(r["global_prim_index"])
        if key not in best_by_loss or float(r["loss_final_best"]) < float(best_by_loss[key]["loss_final_best"]):
            best_by_loss[key] = r
        if key not in best_by_shape or float(r["final_shape_l1"]) < float(best_by_shape[key]["final_shape_l1"]):
            best_by_shape[key] = r

    def stat(name: str, values: Iterable[float]) -> dict:
        arr = np.asarray(list(values), dtype=np.float64)
        return {
            "metric": name,
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(np.max(arr)),
        }

    loss_rows = list(best_by_loss.values())
    shape_rows = list(best_by_shape.values())
    return [
        stat("loss_selected_final_shape_l1", (float(r["final_shape_l1"]) for r in loss_rows)),
        stat("loss_selected_scale_l1_vs_gt", (float(r["scale_l1_vs_gt"]) for r in loss_rows)),
        stat("loss_selected_trans_l2_vs_gt", (float(r["trans_l2_vs_gt"]) for r in loss_rows)),
        stat("loss_selected_rot_geodesic_deg_vs_gt", (float(r["rot_geodesic_deg_vs_gt"]) for r in loss_rows)),
        stat("loss_selected_loss_final", (float(r["loss_final_best"]) for r in loss_rows)),
        stat("shape_selected_final_shape_l1", (float(r["final_shape_l1"]) for r in shape_rows)),
        stat("shape_selected_scale_l1_vs_gt", (float(r["scale_l1_vs_gt"]) for r in shape_rows)),
        stat("shape_selected_trans_l2_vs_gt", (float(r["trans_l2_vs_gt"]) for r in shape_rows)),
        stat("shape_selected_rot_geodesic_deg_vs_gt", (float(r["rot_geodesic_deg_vs_gt"]) for r in shape_rows)),
        stat("shape_selected_loss_final", (float(r["loss_final_best"]) for r in shape_rows)),
        stat("loss_selected_init_shape_l1", (float(r["init_shape_l1"]) for r in loss_rows)),
        stat("loss_selected_init_scale_l1_vs_gt", (float(r["init_scale_l1_vs_gt"]) for r in loss_rows)),
        stat("loss_selected_init_trans_l2_vs_gt", (float(r["init_trans_l2_vs_gt"]) for r in loss_rows)),
        stat("loss_selected_init_rot_geodesic_deg_vs_gt", (float(r["init_rot_geodesic_deg_vs_gt"]) for r in loss_rows)),
    ]


def select_best_rows(rows: Sequence[dict], key_name: str) -> List[dict]:
    if key_name not in {"loss_final_best", "final_shape_l1"}:
        raise ValueError(key_name)
    best = {}
    for r in rows:
        prim = int(r["global_prim_index"])
        if prim not in best or float(r[key_name]) < float(best[prim][key_name]):
            best[prim] = r
    return sorted(best.values(), key=lambda r: int(r["global_prim_index"]))


def maybe_write_landscapes(
    records: Sequence[PrimitiveRecord],
    args: argparse.Namespace,
    eta: torch.Tensor | None,
    omega: torch.Tensor | None,
    out_dir: Path,
) -> None:
    selected_records = list(records)
    if args.landscape_prim_indices.strip():
        wanted = {int(x.strip()) for x in args.landscape_prim_indices.split(",") if x.strip()}
        selected_records = [r for r in records if r.global_prim_index in wanted]
        missing = sorted(wanted - {r.global_prim_index for r in selected_records})
        if missing:
            print(f"Warning: requested landscape prim indices not loaded: {missing}", flush=True)
    else:
        selected_records = selected_records[: args.landscape_count]

    if not selected_records or args.landscape_count == 0:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    landscape_dir = out_dir / "landscapes"
    landscape_dir.mkdir(parents=True, exist_ok=True)

    eps_values = torch.linspace(
        args.eps_min,
        args.eps_max,
        args.landscape_grid_steps,
        device=selected_records[0].scale.device,
        dtype=selected_records[0].scale.dtype,
    )

    for rec in selected_records:
        with torch.no_grad():
            gt_surface = sq_points(
                rec.scale,
                rec.shape,
                rec.rotate,
                rec.trans,
                sampler=args.sampler,
                eta=eta,
                omega=omega,
                surface_n_samples=args.surface_n_samples,
                surface_D_eta=args.surface_D_eta,
                surface_D_omega=args.surface_D_omega,
            ).detach()

            grid = torch.empty(
                (args.landscape_grid_steps, args.landscape_grid_steps),
                device=rec.scale.device,
                dtype=rec.scale.dtype,
            )

            for i, e1 in enumerate(eps_values):
                for j, e2 in enumerate(eps_values):
                    eps = torch.stack([e1, e2])
                    pred_surface = sq_points(
                        rec.scale,
                        eps,
                        rec.rotate,
                        rec.trans,
                        sampler=args.sampler,
                        eta=eta,
                        omega=omega,
                        surface_n_samples=args.surface_n_samples,
                        surface_D_eta=args.surface_D_eta,
                        surface_D_omega=args.surface_D_omega,
                    )
                    grid[i, j] = surface_loss(pred_surface, gt_surface, args.landscape_loss)

            grid_np = grid.detach().cpu().numpy()
            eps_np = eps_values.detach().cpu().numpy()
            best_idx = np.unravel_index(np.argmin(grid_np), grid_np.shape)
            best_eps1 = float(eps_np[best_idx[0]])
            best_eps2 = float(eps_np[best_idx[1]])

        csv_path = landscape_dir / f"landscape_prim{rec.global_prim_index:05d}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["eps1\\eps2", *[f"{v:.6f}" for v in eps_np]])
            for i, e1 in enumerate(eps_np):
                writer.writerow([f"{e1:.6f}", *[f"{x:.8e}" for x in grid_np[i]]])

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(
            grid_np,
            origin="lower",
            extent=[args.eps_min, args.eps_max, args.eps_min, args.eps_max],
            aspect="auto",
        )
        ax.scatter(
            [float(rec.shape[1].detach().cpu().item())],
            [float(rec.shape[0].detach().cpu().item())],
            marker="x",
            s=60,
            label="GT eps",
        )
        ax.scatter([best_eps2], [best_eps1], marker="o", s=35, label="best grid")
        ax.set_xlabel("eps_2")
        ax.set_ylabel("eps_1")
        ax.set_title(
            f"prim {rec.global_prim_index} | {rec.model_id} | k={rec.primitive_k}\n"
            f"landscape loss={args.landscape_loss} sampler={args.sampler}"
        )
        ax.legend(loc="best")
        fig.colorbar(im, ax=ax, label="loss")
        fig.tight_layout()
        fig.savefig(landscape_dir / f"landscape_prim{rec.global_prim_index:05d}.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--train-split", default="train.txt")
    parser.add_argument("--val-split", default="test.txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-points", type=int, default=4096)
    parser.add_argument("--kmax", type=int, default=4)
    parser.add_argument("--normal-mode", default="radial")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional dataset sample cap before primitive extraction.")
    parser.add_argument("--max-primitives", type=int, default=256)
    parser.add_argument("--min-visible-points", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--sampler", default="fixed_grid", choices=["fixed_grid", "equal_distance"])
    parser.add_argument("--grid-n-eta", type=int, default=32)
    parser.add_argument("--grid-n-omega", type=int, default=64)
    parser.add_argument("--surface-n-samples", type=int, default=1024)
    parser.add_argument("--surface-D-eta", type=float, default=0.05)
    parser.add_argument("--surface-D-omega", type=float, default=0.05)

    parser.add_argument("--optim-loss", default="chamfer", choices=["chamfer", "pointwise"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--eps-min", type=float, default=0.1)
    parser.add_argument("--eps-max", type=float, default=1.9)
    parser.add_argument(
        "--init-eps-pairs",
        default="1.0,1.0;0.3,0.3;1.7,1.7;0.3,1.7;1.7,0.3",
        help="Semicolon-separated eps initializations.",
    )
    parser.add_argument("--optimize-scale", action="store_true")
    parser.add_argument(
        "--init-scale-mults",
        default="1,1,1",
        help="Semicolon-separated scale multipliers used when --optimize-scale is set.",
    )
    parser.add_argument("--scale-min", type=float, default=1e-4)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--optimize-trans", action="store_true")
    parser.add_argument(
        "--init-trans-offsets",
        default="0,0,0",
        help="Semicolon-separated translation offsets in normalized coordinates, e.g. '0.04,0,0;-0.04,0.04,0'.",
    )
    parser.add_argument("--optimize-rotate", action="store_true")
    parser.add_argument(
        "--init-rotvecs-deg",
        default="0,0,0",
        help="Semicolon-separated world-frame axis-angle rotation-vector initializations in degrees, e.g. '15,0,0;0,15,0'.",
    )

    parser.add_argument("--landscape-count", type=int, default=8)
    parser.add_argument("--landscape-prim-indices", default="", help="Comma-separated global primitive indices to plot instead of first N.")
    parser.add_argument("--landscape-grid-steps", type=int, default=41)
    parser.add_argument("--landscape-loss", default="pointwise", choices=["chamfer", "pointwise"])

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.optim_loss == "pointwise" and args.sampler != "fixed_grid":
        raise ValueError("--optim-loss pointwise is only meaningful with --sampler fixed_grid.")
    if args.landscape_loss == "pointwise" and args.sampler != "fixed_grid":
        raise ValueError("--landscape-loss pointwise is only meaningful with --sampler fixed_grid.")

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eta = omega = None
    if args.sampler == "fixed_grid":
        eta, omega = fixed_eta_omega_grid(
            args.grid_n_eta,
            args.grid_n_omega,
            device=device,
            dtype=torch.float32,
        )

    print("=== SQ-Zero eps/scale/translation/rotation identifiability diagnostic ===", flush=True)
    print(f"data_root={args.data_root}", flush=True)
    print(f"split={args.split} sampler={args.sampler} optim_loss={args.optim_loss}", flush=True)
    print(f"optimize_scale={args.optimize_scale} init_scale_mults={args.init_scale_mults}", flush=True)
    print(f"optimize_trans={args.optimize_trans} init_trans_offsets={args.init_trans_offsets}", flush=True)
    print(f"optimize_rotate={args.optimize_rotate} init_rotvecs_deg={args.init_rotvecs_deg}", flush=True)
    print(f"device={device} max_primitives={args.max_primitives}", flush=True)

    records = load_primitives(args, device=device)
    if not records:
        raise RuntimeError("No primitives loaded. Check data root/split/min-visible-points.")
    print(f"Loaded {len(records)} primitive records.", flush=True)

    init_pairs = parse_init_pairs(args.init_eps_pairs)
    scale_mults = parse_scale_triples(args.init_scale_mults) if args.optimize_scale else [(1.0, 1.0, 1.0)]
    trans_offsets = parse_trans_offsets(args.init_trans_offsets) if args.optimize_trans else [(0.0, 0.0, 0.0)]
    rotvecs_deg = parse_rotvecs_deg(args.init_rotvecs_deg) if args.optimize_rotate else [(0.0, 0.0, 0.0)]
    print(
        f"eps_initializations={len(init_pairs)} scale_initializations={len(scale_mults)} "
        f"trans_initializations={len(trans_offsets)} rot_initializations={len(rotvecs_deg)}",
        flush=True,
    )

    rows: List[dict] = []
    for rec_i, rec in enumerate(records):
        if rec_i % 25 == 0:
            print(f"Optimizing primitive {rec_i + 1}/{len(records)}", flush=True)
        for init_pair in init_pairs:
            for init_scale_mult in scale_mults:
                for init_trans_offset in trans_offsets:
                    for init_rotvec_deg in rotvecs_deg:
                        rows.append(
                            optimize_one(
                                rec,
                                init_pair,
                                init_scale_mult,
                                init_trans_offset,
                                init_rotvec_deg,
                                args,
                                eta=eta,
                                omega=omega,
                            )
                        )

    write_csv(out_dir / "eps_optimization_rows.csv", rows)
    write_csv(out_dir / "eps_optimization_best_by_loss.csv", select_best_rows(rows, "loss_final_best"))
    write_csv(out_dir / "eps_optimization_best_by_shape.csv", select_best_rows(rows, "final_shape_l1"))
    summary_rows = summarize(rows)
    write_csv(out_dir / "eps_optimization_summary.csv", summary_rows)
    maybe_write_landscapes(records, args, eta=eta, omega=omega, out_dir=out_dir)

    print("\n=== Summary ===", flush=True)
    for r in summary_rows:
        print(
            f"{r['metric']}: n={r['n']} mean={r['mean']:.6f} "
            f"median={r['median']:.6f} p90={r['p90']:.6f} max={r['max']:.6f}",
            flush=True,
        )
    print(f"\nWrote: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
