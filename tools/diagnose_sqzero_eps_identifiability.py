#!/usr/bin/env python3
"""Fixed-pose single-superquadric eps identifiability diagnostic for SQ-Zero.

This script does NOT train SuperDec.  It asks a narrower question:

    If GT scale / rotation / translation are fixed, can the same surface loss
    recover the GT superquadric exponents eps_1, eps_2?

It reads the existing SQ-Zero LMDB sidecars, extracts individual GT primitives,
and optimizes only eps from one or more initializations.  Optionally it also
writes eps-loss landscape plots for a few primitives.

Default mode uses a fixed eta/omega grid, because that removes sampler movement
as a confound.  A training-like equal-distance sampler mode is also provided for
closer comparison to the current supervised-Hungarian GT-surface loss.
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

    init_eps = torch.tensor(init_pair, device=rec.scale.device, dtype=rec.scale.dtype)
    raw_eps = raw_from_eps(init_eps, args.eps_min, args.eps_max).detach().clone().requires_grad_(True)

    params: List[torch.Tensor] = [raw_eps]
    log_scale = None
    if args.optimize_scale:
        log_scale = torch.log(torch.clamp(rec.scale.detach(), min=1e-5)).clone().requires_grad_(True)
        params.append(log_scale)

    opt = torch.optim.Adam(params, lr=args.lr)

    def current_scale() -> torch.Tensor:
        if log_scale is None:
            return rec.scale
        return torch.exp(log_scale).clamp(min=args.scale_min, max=args.scale_max)

    loss_initial = None
    best = {
        "loss": float("inf"),
        "eps": None,
        "scale": None,
        "step": -1,
    }

    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        pred_eps = eps_from_raw(raw_eps, args.eps_min, args.eps_max)
        pred_surface = sq_points(
            current_scale(),
            pred_eps,
            rec.rotate,
            rec.trans,
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
            best["step"] = step

        if step == args.steps:
            break
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        opt.step()

    final_eps = best["eps"]
    final_scale = best["scale"]
    assert final_eps is not None and final_scale is not None and loss_initial is not None

    init_l1 = torch.abs(init_eps - gt_eps).mean().detach().cpu().item()
    final_l1 = torch.abs(final_eps - gt_eps).mean().detach().cpu().item()
    scale_l1 = torch.abs(final_scale - rec.scale).mean().detach().cpu().item()

    return {
        "global_prim_index": rec.global_prim_index,
        "model_id": rec.model_id,
        "primitive_k": rec.primitive_k,
        "K": rec.K,
        "visible_points_in_sample": rec.visible_points_in_sample,
        "gt_scale_x": float(rec.scale[0].detach().cpu().item()),
        "gt_scale_y": float(rec.scale[1].detach().cpu().item()),
        "gt_scale_z": float(rec.scale[2].detach().cpu().item()),
        "gt_eps1": float(gt_eps[0].detach().cpu().item()),
        "gt_eps2": float(gt_eps[1].detach().cpu().item()),
        "init_eps1": float(init_eps[0].detach().cpu().item()),
        "init_eps2": float(init_eps[1].detach().cpu().item()),
        "init_shape_l1": float(init_l1),
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
    # one best row per primitive by final loss
    best_by_prim = {}
    for r in rows:
        key = int(r["global_prim_index"])
        if key not in best_by_prim or float(r["loss_final_best"]) < float(best_by_prim[key]["loss_final_best"]):
            best_by_prim[key] = r
    best_rows = list(best_by_prim.values())

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

    return [
        stat("best_final_shape_l1", (float(r["final_shape_l1"]) for r in best_rows)),
        stat("best_init_shape_l1", (float(r["init_shape_l1"]) for r in best_rows)),
        stat("best_loss_final", (float(r["loss_final_best"]) for r in best_rows)),
        stat("best_loss_initial", (float(r["loss_initial"]) for r in best_rows)),
        stat("best_scale_l1_vs_gt", (float(r["scale_l1_vs_gt"]) for r in best_rows)),
    ]


def maybe_write_landscapes(
    records: Sequence[PrimitiveRecord],
    args: argparse.Namespace,
    eta: torch.Tensor | None,
    omega: torch.Tensor | None,
    out_dir: Path,
) -> None:
    if args.landscape_count <= 0:
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
        device=records[0].scale.device,
        dtype=records[0].scale.dtype,
    )

    for rec in records[: args.landscape_count]:
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
    parser.add_argument("--scale-min", type=float, default=1e-4)
    parser.add_argument("--scale-max", type=float, default=2.0)

    parser.add_argument("--landscape-count", type=int, default=8)
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

    print("=== SQ-Zero fixed-pose eps identifiability diagnostic ===", flush=True)
    print(f"data_root={args.data_root}", flush=True)
    print(f"split={args.split} sampler={args.sampler} optim_loss={args.optim_loss}", flush=True)
    print(f"device={device} max_primitives={args.max_primitives}", flush=True)

    records = load_primitives(args, device=device)
    if not records:
        raise RuntimeError("No primitives loaded. Check data root/split/min-visible-points.")
    print(f"Loaded {len(records)} primitive records.", flush=True)

    init_pairs = parse_init_pairs(args.init_eps_pairs)
    rows: List[dict] = []
    for rec_i, rec in enumerate(records):
        if rec_i % 25 == 0:
            print(f"Optimizing primitive {rec_i + 1}/{len(records)}", flush=True)
        for init_pair in init_pairs:
            rows.append(optimize_one(rec, init_pair, args, eta=eta, omega=omega))

    write_csv(out_dir / "eps_optimization_rows.csv", rows)
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
