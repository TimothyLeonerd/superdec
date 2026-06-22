#!/usr/bin/env python3
import argparse, csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    fibonacci_sphere,
    sample_generalized_surface,
    PointNetScaleExp,
    chamfer,
)
from tools.train_gensq_weighted_pca_fps_clean_lr import (
    WeightedPCANet,
    random_rotations,
    batched_fps,
    gather_points,
    project_per_sample_dirs,
    dropout_without_replacement,
    hard_pca_frame,
    frame_loss_and_angles,
    PERMS,
)


@torch.no_grad()
def sample_eval_dataset(n, points, candidate_n, sampler, args, device, chunk):
    all_world, all_scale, all_exp, all_R = [], [], [], []

    for start in range(0, n, chunk):
        b = min(chunk, n - start)

        scale = args.scale_min + (args.scale_max - args.scale_min) * torch.rand(b, 3, device=device)
        exp = args.exp_min + (args.exp_max - args.exp_min) * torch.rand(b, 3, device=device)
        gt_R = random_rotations(b, device)

        if sampler == "direct_fib":
            dirs = fibonacci_sphere(points, device)
            local = sample_generalized_surface(scale, exp, dirs)

        elif sampler == "fps":
            base_dirs = fibonacci_sphere(candidate_n, device)
            dir_R = random_rotations(b, device)
            dirs = base_dirs[None].expand(b, candidate_n, 3) @ dir_R.transpose(1, 2)
            candidates = project_per_sample_dirs(scale, exp, dirs)
            idx = batched_fps(candidates, points)
            local = gather_points(candidates, idx)

        else:
            raise ValueError(sampler)

        world = local @ gt_R.transpose(1, 2)

        all_world.append(world.cpu())
        all_scale.append(scale.cpu())
        all_exp.append(exp.cpu())
        all_R.append(gt_R.cpu())

    return (
        torch.cat(all_world, 0),
        torch.cat(all_scale, 0),
        torch.cat(all_exp, 0),
        torch.cat(all_R, 0),
    )


def best_perm(frame, gt_R):
    device = frame.device
    perms = PERMS.to(device)
    dots = torch.einsum("bik,bil->bkl", frame, gt_R).abs().clamp(0.0, 1.0)

    scores = []
    for p in perms:
        d = dots[:, torch.arange(3, device=device), p]
        scores.append(d.pow(2).mean(dim=1))

    scores = torch.stack(scores, dim=1)
    best = scores.argmax(dim=1)
    return perms[best]


def permute_targets(x, perm):
    return x.gather(1, perm)


@torch.no_grad()
def eval_case(case_name, dataset, exp_model, wpca_model, args, device, dropout_keep=None):
    world, gt_scale, gt_exp, gt_R = dataset

    loader = DataLoader(
        TensorDataset(world, gt_scale, gt_exp, gt_R),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    accum = {}

    def add(method, key, val):
        accum.setdefault(method, {}).setdefault(key, []).append(val.detach().cpu())

    for world_b, scale_b, exp_b, R_b in loader:
        world_b = world_b.to(device)
        scale_b = scale_b.to(device)
        exp_b = exp_b.to(device)
        R_b = R_b.to(device)

        if dropout_keep is not None and dropout_keep < 1.0:
            world_b = dropout_without_replacement(world_b, dropout_keep)

        frames = {}

        frames["gt"] = R_b

        hard_R, _ = hard_pca_frame(world_b)
        frames["hard_pca"] = hard_R

        w_R, neff, maxw = wpca_model(world_b)
        frames["weighted_pca"] = w_R
        add("weighted_pca", "neff", neff)
        add("weighted_pca", "maxw", maxw)

        for method, frame in frames.items():
            canon = world_b @ frame

            if method == "gt":
                aligned_scale = scale_b
                aligned_exp = exp_b
                axis_loss = torch.zeros(world_b.shape[0], device=device)
                axis_ang = torch.zeros(world_b.shape[0], 3, device=device)
            else:
                perm = best_perm(frame, R_b)
                aligned_scale = permute_targets(scale_b, perm)
                aligned_exp = permute_targets(exp_b, perm)
                axis_loss, axis_ang = frame_loss_and_angles(frame, R_b)

            pred_scale, pred_exp = exp_model(canon)

            scale_l1 = (pred_scale - aligned_scale).abs().mean(dim=1)
            exp_l1 = (pred_exp - aligned_exp).abs().mean(dim=1)
            loss = (
                scale_l1 / (args.scale_max - args.scale_min)
                + exp_l1 / (args.exp_max - args.exp_min)
            )

            dirs = fibonacci_sphere(canon.shape[1], device)
            pred_pts = sample_generalized_surface(pred_scale, pred_exp, dirs)
            cd = chamfer(pred_pts, canon)

            add(method, "axis_angle", axis_ang.reshape(-1))
            add(method, "axis_loss", axis_loss)
            add(method, "scale_l1", scale_l1)
            add(method, "exp_l1", exp_l1)
            add(method, "loss", loss)
            add(method, "cd", cd)

    rows = []
    for method, d in accum.items():
        row = {"case": case_name, "method": method}
        for key in ["axis_angle", "axis_loss", "scale_l1", "exp_l1", "loss", "cd", "neff", "maxw"]:
            if key not in d:
                continue
            v = torch.cat(d[key])
            row[f"{key}_mean"] = float(v.mean())
            row[f"{key}_p95"] = float(torch.quantile(v, 0.95))
        rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exp-ckpt", required=True)
    ap.add_argument("--wpca-ckpt", required=True)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--candidate-n", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gen-chunk", type=int, default=256)
    ap.add_argument("--scale-min", type=float, default=0.08)
    ap.add_argument("--scale-max", type=float, default=0.35)
    ap.add_argument("--exp-min", type=float, default=1.0)
    ap.add_argument("--exp-max", type=float, default=8.0)
    ap.add_argument("--dropout-keep", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    exp_model = PointNetScaleExp(args.scale_min, args.scale_max, args.exp_min, args.exp_max).to(device)
    exp_ckpt = torch.load(args.exp_ckpt, map_location=device)
    exp_model.load_state_dict(exp_ckpt["model"])
    exp_model.eval()
    print(f"loaded exp/scale: {args.exp_ckpt}", flush=True)

    wpca_model = WeightedPCANet().to(device)
    wpca_ckpt = torch.load(args.wpca_ckpt, map_location=device)
    wpca_model.load_state_dict(wpca_ckpt["model"])
    wpca_model.eval()
    print(f"loaded wpca: {args.wpca_ckpt}", flush=True)

    cases = [
        ("direct_fib512", "direct_fib", 512, None),
        ("fps2048_to_512", "fps", 512, None),
        (f"dropout{args.dropout_keep:g}_from_fib512", "direct_fib", 512, args.dropout_keep),
        (f"dropout{args.dropout_keep:g}_from_fps512", "fps", 512, args.dropout_keep),
        ("direct_fib307", "direct_fib", 307, None),
    ]

    all_rows = []

    for i, (name, sampler, points, keep) in enumerate(cases):
        print(f"\n=== generating {name} ===", flush=True)
        set_seed(args.seed + 1000 + i)
        data = sample_eval_dataset(
            args.val_n,
            points,
            args.candidate_n,
            sampler,
            args,
            device,
            args.gen_chunk,
        )

        print(f"=== evaluating {name} ===", flush=True)
        rows = eval_case(name, data, exp_model, wpca_model, args, device, dropout_keep=keep)
        all_rows.extend(rows)

        for r in rows:
            print(
                f"{r['case']:26s} {r['method']:12s} "
                f"axis={r.get('axis_angle_mean', 0.0):6.2f}/{r.get('axis_angle_p95', 0.0):6.2f} "
                f"scale={r['scale_l1_mean']:.5f}/{r['scale_l1_p95']:.5f} "
                f"exp={r['exp_l1_mean']:.5f}/{r['exp_l1_p95']:.5f} "
                f"cd={r['cd_mean']:.6f}/{r['cd_p95']:.6f}",
                flush=True,
            )

    fields = sorted({k for r in all_rows for k in r.keys()})
    fields = ["case", "method"] + [f for f in fields if f not in ("case", "method")]

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    lines = []
    for r in all_rows:
        lines.append(
            f"{r['case']:26s} {r['method']:12s} "
            f"axis={r.get('axis_angle_mean', 0.0):6.2f}/{r.get('axis_angle_p95', 0.0):6.2f} "
            f"scale={r['scale_l1_mean']:.5f}/{r['scale_l1_p95']:.5f} "
            f"exp={r['exp_l1_mean']:.5f}/{r['exp_l1_p95']:.5f} "
            f"loss={r['loss_mean']:.5f} "
            f"cd={r['cd_mean']:.6f}/{r['cd_p95']:.6f}"
        )

    (out / "summary_short.txt").write_text("\n".join(lines) + "\n")
    print("\n=== wrote ===")
    print(out / "summary.csv")
    print(out / "summary_short.txt")


if __name__ == "__main__":
    main()
