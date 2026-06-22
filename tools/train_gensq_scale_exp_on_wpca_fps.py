#!/usr/bin/env python3
import argparse, csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    PointNetScaleExp,
    chamfer,
    fibonacci_sphere,
    sample_generalized_surface,
)

from tools.train_gensq_weighted_pca_fps_clean_lr import WeightedPCANet
from tools.eval_gensq_downstream_frame_swap import (
    sample_eval_dataset,
    best_perm,
    permute_targets,
    eval_case,
)


@torch.no_grad()
def canonicalize_with_wpca(dataset, wpca_model, args, device, name):
    world, gt_scale, gt_exp, gt_R = dataset

    loader = DataLoader(
        TensorDataset(world, gt_scale, gt_exp, gt_R),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    canons, scales, exps, axis_angles = [], [], [], []

    for bi, (x, scale, exp, R) in enumerate(loader):
        x = x.to(device)
        scale = scale.to(device)
        exp = exp.to(device)
        R = R.to(device)

        frame, _, _ = wpca_model(x)
        canon = x @ frame

        perm = best_perm(frame, R)
        scale_t = permute_targets(scale, perm)
        exp_t = permute_targets(exp, perm)

        dots = torch.einsum("bik,bil->bkl", frame, R).abs().clamp(0.0, 1.0)
        # rough diagnostic only: best matched axis angles
        matched = dots.gather(2, perm[:, None, :]).diagonal(dim1=1, dim2=2)
        ang = torch.rad2deg(torch.acos(matched.clamp(0.0, 1.0)))

        canons.append(canon.cpu())
        scales.append(scale_t.cpu())
        exps.append(exp_t.cpu())
        axis_angles.append(ang.cpu().reshape(-1))

        if (bi + 1) % 20 == 0:
            print(f"{name}: canonicalized batch {bi+1}", flush=True)

    axis = torch.cat(axis_angles)
    print(
        f"{name}: wpca axis mean={axis.mean():.3f} p95={torch.quantile(axis, 0.95):.3f}",
        flush=True,
    )

    return torch.cat(canons, 0), torch.cat(scales, 0), torch.cat(exps, 0)


def compute_loss(pred_scale, pred_exp, scale_t, exp_t, args):
    scale_l1 = (pred_scale - scale_t).abs().mean(dim=1)
    exp_l1 = (pred_exp - exp_t).abs().mean(dim=1)
    loss = (
        scale_l1 / (args.scale_max - args.scale_min)
        + exp_l1 / (args.exp_max - args.exp_min)
    )
    return loss, scale_l1, exp_l1


@torch.no_grad()
def eval_canonical_dataset(model, dataset, args, device):
    canon, scale_t, exp_t = dataset
    loader = DataLoader(
        TensorDataset(canon, scale_t, exp_t),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    losses, scale_l1s, exp_l1s, cds = [], [], [], []

    model.eval()
    for x, scale, exp in loader:
        x = x.to(device)
        scale = scale.to(device)
        exp = exp.to(device)

        pred_scale, pred_exp = model(x)
        loss, scale_l1, exp_l1 = compute_loss(pred_scale, pred_exp, scale, exp, args)

        dirs = fibonacci_sphere(x.shape[1], device)
        pred_pts = sample_generalized_surface(pred_scale, pred_exp, dirs)
        cd = chamfer(pred_pts, x)

        losses.append(loss.cpu())
        scale_l1s.append(scale_l1.cpu())
        exp_l1s.append(exp_l1.cpu())
        cds.append(cd.cpu())

    losses = torch.cat(losses)
    scale_l1s = torch.cat(scale_l1s)
    exp_l1s = torch.cat(exp_l1s)
    cds = torch.cat(cds)

    return {
        "loss": float(losses.mean()),
        "scale_l1": float(scale_l1s.mean()),
        "exp_l1": float(exp_l1s.mean()),
        "cd": float(cds.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wpca-ckpt", required=True)

    ap.add_argument("--train-n", type=int, default=50000)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=1000)

    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--candidate-n", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gen-chunk", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

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

    wpca_model = WeightedPCANet().to(device)
    wpca_ckpt = torch.load(args.wpca_ckpt, map_location=device)
    wpca_model.load_state_dict(wpca_ckpt["model"])
    wpca_model.eval()
    print(f"loaded wpca: {args.wpca_ckpt}", flush=True)

    print("Generating train fps2048_to_512", flush=True)
    set_seed(args.seed + 1000)
    train_raw = sample_eval_dataset(
        args.train_n, args.points, args.candidate_n, "fps",
        args, device, args.gen_chunk,
    )

    print("Generating val fps2048_to_512", flush=True)
    set_seed(args.seed + 2000)
    val_raw = sample_eval_dataset(
        args.val_n, args.points, args.candidate_n, "fps",
        args, device, args.gen_chunk,
    )

    print("Canonicalizing train with frozen weighted PCA", flush=True)
    train_canon = canonicalize_with_wpca(train_raw, wpca_model, args, device, "train")

    print("Canonicalizing val with frozen weighted PCA", flush=True)
    val_canon = canonicalize_with_wpca(val_raw, wpca_model, args, device, "val")

    train_loader = DataLoader(
        TensorDataset(*train_canon),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = PointNetScaleExp(args.scale_min, args.scale_max, args.exp_min, args.exp_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = None

    metrics_path = out / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "val_loss",
                "val_scale_l1",
                "val_exp_l1",
                "val_cd",
            ],
        )
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            seen = 0

            for x, scale, exp in train_loader:
                x = x.to(device)
                scale = scale.to(device)
                exp = exp.to(device)

                pred_scale, pred_exp = model(x)
                loss_vec, _, _ = compute_loss(pred_scale, pred_exp, scale, exp, args)
                loss = loss_vec.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += float(loss.detach()) * x.shape[0]
                seen += x.shape[0]

            train_loss = total / max(1, seen)
            val = eval_canonical_dataset(model, val_canon, args, device)

            row = {
                "epoch": ep,
                "lr": args.lr,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_scale_l1": val["scale_l1"],
                "val_exp_l1": val["exp_l1"],
                "val_cd": val["cd"],
            }
            writer.writerow(row)
            f.flush()

            if val["loss"] < best_val:
                best_val = val["loss"]
                best_epoch = ep
                torch.save(
                    {
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "args": vars(args),
                        "epoch": ep,
                        "val_loss": best_val,
                    },
                    out / "best.pt",
                )

            print(
                f"ep={ep:03d} "
                f"train={train_loss:.6f} "
                f"val={val['loss']:.6f} "
                f"scale={val['scale_l1']:.5f} "
                f"exp={val['exp_l1']:.5f} "
                f"cd={val['cd']:.6f}",
                flush=True,
            )

    print(f"best_epoch={best_epoch} best_val={best_val:.6f}", flush=True)

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Final downstream eval on all cases", flush=True)

    cases = [
        ("direct_fib512", "direct_fib", 512, None),
        ("fps2048_to_512", "fps", 512, None),
        (f"dropout{args.dropout_keep:g}_from_fib512", "direct_fib", 512, args.dropout_keep),
        (f"dropout{args.dropout_keep:g}_from_fps512", "fps", 512, args.dropout_keep),
        ("direct_fib307", "direct_fib", 307, None),
    ]

    all_rows = []

    for i, (name, sampler, points, keep) in enumerate(cases):
        print(f"\n=== generating final eval {name} ===", flush=True)
        set_seed(args.seed + 3000 + i)
        data = sample_eval_dataset(
            args.eval_n, points, args.candidate_n, sampler,
            args, device, args.gen_chunk,
        )

        print(f"=== evaluating final eval {name} ===", flush=True)
        rows = eval_case(name, data, model, wpca_model, args, device, dropout_keep=keep)
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

    lines = [f"best_epoch={best_epoch} best_val={best_val:.6f}"]
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

    print("\n=== summary ===")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
