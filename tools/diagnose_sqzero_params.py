#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from superdec.superdec import SuperDec
from superdec.lm_optimization.lm_optimizer import LMOptimizer
from superdec.data.sqzero_lmdb import SQZeroLMDB


PARAM_NAMES = [
    "scale_x", "scale_y", "scale_z",
    "eps_1", "eps_2",
    "trans_x", "trans_y", "trans_z",
]


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


def load_model(ckpt_path: Path, cfg_path: Path, device, use_lm: bool = False):
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
        cfg.sqzero_lmdb.max_train_samples = args.max_samples
        cfg.sqzero_lmdb.max_val_samples = args.max_samples
    else:
        cfg.sqzero_lmdb.max_train_samples = None
        cfg.sqzero_lmdb.max_val_samples = None

    if "trainer" not in cfg:
        cfg.trainer = {}
    cfg.trainer.augmentations = False

    return cfg


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


def rotation_geodesic_deg(R_pred, R_gt):
    R_rel = R_pred.transpose(-1, -2) @ R_gt
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = torch.clamp((trace - 1.0) / 2.0, min=-1.0, max=1.0)
    angle = torch.acos(cos)
    return angle * (180.0 / math.pi)


def get_model_id(batch, b, global_index):
    model_ids = batch.get("model_id", None)
    if model_ids is None:
        return f"idx{global_index}"
    if isinstance(model_ids, (list, tuple)):
        return str(model_ids[b])
    return str(model_ids)


def scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


@torch.no_grad()
def eval_model_params(model, model_name, batch, global_start_index, args, device):
    points = batch["points"].to(device).float()
    labels = batch["labels"].to(device).long()
    K = batch["K"].to(device).long()

    gt_scale = batch["gt_scale"].to(device).float()
    gt_shape = batch["gt_shape"].to(device).float()
    gt_rotate = batch["gt_rotate"].to(device).float()
    gt_trans = batch["gt_trans"].to(device).float()

    out = model(points)

    assign = out["assign_matrix"]
    exist = out["exist"].squeeze(-1)

    pred_scale = out["scale"]
    pred_shape = out["shape"]
    pred_rotate = out["rotate"]
    pred_trans = out["trans"]

    B, _, _ = assign.shape
    rows = []

    for b in range(B):
        K_b = int(K[b].item())
        matched_slots = hungarian_assignment_match(
            assign_b=assign[b],
            exist_b=exist[b],
            labels_b=labels[b],
            K_b=K_b,
            eps=args.eps,
            match_w_assign=args.match_w_assign,
            match_w_exist=args.match_w_exist,
        )

        sample_index = global_start_index + b
        model_id = get_model_id(batch, b, sample_index)

        for gt_id in range(K_b):
            slot = int(matched_slots[gt_id].item())

            ps = pred_scale[b, slot]
            gs = gt_scale[b, gt_id]

            psh = pred_shape[b, slot]
            gsh = gt_shape[b, gt_id]

            pt = pred_trans[b, slot]
            gt = gt_trans[b, gt_id]

            pr = pred_rotate[b, slot]
            gr = gt_rotate[b, gt_id]

            rot_deg = rotation_geodesic_deg(pr, gr)

            scale_abs = torch.abs(ps - gs)
            shape_abs = torch.abs(psh - gsh)
            trans_abs = torch.abs(pt - gt)

            row = {
                "model": model_name,
                "sample_index": sample_index,
                "model_id": model_id,
                "K": K_b,
                "gt_id": gt_id,
                "pred_slot": slot,
                "exist_prob": scalar(exist[b, slot]),

                "gt_scale_x": scalar(gs[0]),
                "gt_scale_y": scalar(gs[1]),
                "gt_scale_z": scalar(gs[2]),
                "pred_scale_x": scalar(ps[0]),
                "pred_scale_y": scalar(ps[1]),
                "pred_scale_z": scalar(ps[2]),

                "gt_eps_1": scalar(gsh[0]),
                "gt_eps_2": scalar(gsh[1]),
                "pred_eps_1": scalar(psh[0]),
                "pred_eps_2": scalar(psh[1]),

                "gt_trans_x": scalar(gt[0]),
                "gt_trans_y": scalar(gt[1]),
                "gt_trans_z": scalar(gt[2]),
                "pred_trans_x": scalar(pt[0]),
                "pred_trans_y": scalar(pt[1]),
                "pred_trans_z": scalar(pt[2]),

                "scale_l1": scalar(scale_abs.mean()),
                "shape_l1": scalar(shape_abs.mean()),
                "trans_l1": scalar(trans_abs.mean()),
                "trans_l2": scalar(torch.linalg.norm(pt - gt)),
                "rot_geodesic_deg": scalar(rot_deg),
            }

            for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
                row[f"scale_{axis}_abs_err"] = scalar(scale_abs[idx])
                row[f"trans_{axis}_abs_err"] = scalar(trans_abs[idx])

            row["eps_1_abs_err"] = scalar(shape_abs[0])
            row["eps_2_abs_err"] = scalar(shape_abs[1])

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


def numeric_values(rows, model, key):
    vals = []
    for r in rows:
        if r["model"] != model:
            continue
        v = r.get(key, None)
        if v is None:
            continue
        v = float(v)
        if math.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=np.float64)


def summarize(rows, models, out_path):
    keys = [
        "scale_l1",
        "shape_l1",
        "trans_l1",
        "trans_l2",
        "rot_geodesic_deg",
        "scale_x_abs_err",
        "scale_y_abs_err",
        "scale_z_abs_err",
        "eps_1_abs_err",
        "eps_2_abs_err",
        "trans_x_abs_err",
        "trans_y_abs_err",
        "trans_z_abs_err",
        "exist_prob",
    ]

    summary_rows = []

    for model in models:
        row = {"model": model}

        for key in keys:
            vals = numeric_values(rows, model, key)
            if vals.size == 0:
                continue

            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_median"] = float(np.median(vals))
            row[f"{key}_p90"] = float(np.quantile(vals, 0.90))
            row[f"{key}_p99"] = float(np.quantile(vals, 0.99))

        summary_rows.append(row)

    write_csv(summary_rows, out_path)

    print("\n=== Parameter diagnostics summary ===")
    for row in summary_rows:
        print(f"\n[{row['model']}]")
        for key in ["scale_l1", "shape_l1", "trans_l2", "rot_geodesic_deg"]:
            m = row.get(f"{key}_mean", None)
            med = row.get(f"{key}_median", None)
            p90 = row.get(f"{key}_p90", None)
            if m is not None:
                print(f"  {key}: mean={m:.6f}, median={med:.6f}, p90={p90:.6f}")


def plot_error_hist(rows, models, key, out_path):
    plt.figure(figsize=(9, 5))

    for model in models:
        vals = numeric_values(rows, model, key)
        if vals.size == 0:
            continue
        plt.hist(vals, bins=50, alpha=0.45, label=model, density=True)

    plt.xlabel(key)
    plt.ylabel("density")
    plt.title(key)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_pred_vs_gt(rows, models, gt_key, pred_key, out_path):
    plt.figure(figsize=(6, 6))

    all_vals = []

    for model in models:
        xs = []
        ys = []
        for r in rows:
            if r["model"] != model:
                continue
            xs.append(float(r[gt_key]))
            ys.append(float(r[pred_key]))

        if not xs:
            continue

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        all_vals.append(xs)
        all_vals.append(ys)

        plt.scatter(xs, ys, s=3, alpha=0.35, label=model)

    if all_vals:
        all_cat = np.concatenate(all_vals)
        lo = float(np.min(all_cat))
        hi = float(np.max(all_cat))
        pad = 0.05 * max(hi - lo, 1e-6)
        lo -= pad
        hi += pad
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)

    plt.xlabel(gt_key)
    plt.ylabel(pred_key)
    plt.title(f"{pred_key} vs {gt_key}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_plots(rows, models, out_dir, plot_mode="all"):
    if plot_mode == "none":
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if plot_mode == "all":
        for key in [
            "scale_l1",
            "shape_l1",
            "trans_l2",
            "rot_geodesic_deg",
            "eps_1_abs_err",
            "eps_2_abs_err",
            "exist_prob",
        ]:
            plot_error_hist(rows, models, key, plot_dir / f"hist_{key}.png")

    pairs = [
        ("gt_scale_x", "pred_scale_x"),
        ("gt_scale_y", "pred_scale_y"),
        ("gt_scale_z", "pred_scale_z"),
        ("gt_eps_1", "pred_eps_1"),
        ("gt_eps_2", "pred_eps_2"),
        ("gt_trans_x", "pred_trans_x"),
        ("gt_trans_y", "pred_trans_y"),
        ("gt_trans_z", "pred_trans_z"),
    ]

    for gt_key, pred_key in pairs:
        plot_pred_vs_gt(rows, models, gt_key, pred_key, plot_dir / f"scatter_{pred_key}_vs_{gt_key}.png")

    print(f"Wrote {plot_mode} plots to: {plot_dir}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--ckpts", nargs="+", type=Path, required=True)
    ap.add_argument("--cfgs", nargs="+", type=Path, required=True)
    ap.add_argument("--lm-names", nargs="*", default=[],
                    help="Model names for which LM optimization should be enabled.")

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
    ap.add_argument("--no-detail-csv", action="store_true")
    ap.add_argument("--plot-mode", choices=["all", "scatter", "none"], default="all")

    ap.add_argument("--match-w-assign", type=float, default=1.0)
    ap.add_argument("--match-w-exist", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=1e-8)

    args = ap.parse_args()

    if not (len(args.names) == len(args.ckpts) == len(args.cfgs)):
        raise ValueError("--names, --ckpts, and --cfgs must have the same length.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = []
    cfg0 = None

    lm_names = set(args.lm_names)

    for name, ckpt, cfg_path in zip(args.names, args.ckpts, args.cfgs):
        print(f"Loading {name}:")
        print(f"  ckpt={ckpt}")
        print(f"  cfg={cfg_path}")
        use_lm = name in lm_names
        model, cfg = load_model(ckpt, cfg_path, device, use_lm=use_lm)
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

    all_rows = []
    global_index = 0

    for batch in tqdm(loader, desc="batches", disable=args.quiet):
        B = batch["points"].shape[0]

        for name, model in models:
            rows = eval_model_params(
                model=model,
                model_name=name,
                batch=batch,
                global_start_index=global_index,
                args=args,
                device=device,
            )
            all_rows.extend(rows)

        global_index += B

    detail_csv = args.out_dir / "matched_param_details.csv"
    summary_csv = args.out_dir / "matched_param_summary.csv"

    if not args.no_detail_csv:
        write_csv(all_rows, detail_csv)
        print(f"Wrote: {detail_csv}")
    else:
        print("Skipped matched_param_details.csv (--no-detail-csv)")

    summarize(all_rows, args.names, summary_csv)
    make_plots(all_rows, args.names, args.out_dir, plot_mode=args.plot_mode)

    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
