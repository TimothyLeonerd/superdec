#!/usr/bin/env python3
import argparse, csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.train_gensq_pointnet_local_scale_exp import (
    set_seed,
    PointNetScaleExp,
    fibonacci_sphere,
    sample_generalized_surface,
    chamfer,
)
from tools.train_gensq_weighted_pca_fps_clean_lr import WeightedPCANet
from tools.eval_gensq_downstream_frame_swap import sample_eval_dataset, eval_case
from tools.train_gensq_scale_exp_on_wpca_fps import canonicalize_with_wpca, eval_canonical_dataset


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
    ap.add_argument("--batch-size", type=int, default=128)
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
    train_raw = sample_eval_dataset(args.train_n, args.points, args.candidate_n, "fps", args, device, args.gen_chunk)

    print("Generating val fps2048_to_512", flush=True)
    set_seed(args.seed + 2000)
    val_raw = sample_eval_dataset(args.val_n, args.points, args.candidate_n, "fps", args, device, args.gen_chunk)

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

    best_val_cd = float("inf")
    best_epoch = None

    metrics_path = out / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_cd",
                "val_cd",
                "val_param_loss",
                "val_scale_l1",
                "val_exp_l1",
            ],
        )
        writer.writeheader()

        for ep in range(1, args.epochs + 1):
            model.train()
            total_cd = 0.0
            seen = 0

            for x, scale, exp in train_loader:
                x = x.to(device)

                pred_scale, pred_exp = model(x)
                dirs = fibonacci_sphere(x.shape[1], device)
                pred_pts = sample_generalized_surface(pred_scale, pred_exp, dirs)

                loss = chamfer(pred_pts, x).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_cd += float(loss.detach()) * x.shape[0]
                seen += x.shape[0]

            train_cd = total_cd / max(1, seen)

            val = eval_canonical_dataset(model, val_canon, args, device)
            val_cd = val["cd"]

            writer.writerow({
                "epoch": ep,
                "lr": args.lr,
                "train_cd": train_cd,
                "val_cd": val_cd,
                "val_param_loss": val["loss"],
                "val_scale_l1": val["scale_l1"],
                "val_exp_l1": val["exp_l1"],
            })
            f.flush()

            if val_cd < best_val_cd:
                best_val_cd = val_cd
                best_epoch = ep
                torch.save(
                    {
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "args": vars(args),
                        "epoch": ep,
                        "val_cd": best_val_cd,
                    },
                    out / "best.pt",
                )

            print(
                f"ep={ep:03d} "
                f"train_cd={train_cd:.8f} "
                f"val_cd={val_cd:.8f} "
                f"val_scale={val['scale_l1']:.5f} "
                f"val_exp={val['exp_l1']:.5f} "
                f"val_param_loss={val['loss']:.6f}",
                flush=True,
            )

    print(f"best_epoch={best_epoch} best_val_cd={best_val_cd:.8f}", flush=True)

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
        data = sample_eval_dataset(args.eval_n, points, args.candidate_n, sampler, args, device, args.gen_chunk)

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

    lines = [f"best_epoch={best_epoch} best_val_cd={best_val_cd:.8f}"]
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
