#!/usr/bin/env python3
import argparse, csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.train_gensq_pointnet_local_scale_exp import set_seed
from tools.train_gensq_weighted_pca_fps_clean_lr import (
    WeightedPCANet,
    hard_pca_frame,
    dropout_without_replacement,
    frame_loss_and_angles,
    PERMS,
)
from tools.eval_gensq_downstream_frame_swap import sample_eval_dataset


PERM_LABELS = ["012", "021", "102", "120", "201", "210"]


def best_perm_index(frame, gt_R):
    device = frame.device
    perms = PERMS.to(device)

    dots = torch.einsum("bik,bil->bkl", frame, gt_R).abs().clamp(0.0, 1.0)

    scores = []
    for p in perms:
        d = dots[:, torch.arange(3, device=device), p]
        scores.append(d.pow(2).mean(dim=1))

    scores = torch.stack(scores, dim=1)
    return scores.argmax(dim=1), dots


def rank_indices(x):
    # ranks original GT axes by value: 0=smallest, 1=middle, 2=largest
    order = x.argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(3, device=x.device)[None].expand_as(order))
    return ranks


@torch.no_grad()
def analyze_method(name, frame_fn, dataset, args, device, dropout_keep=None):
    world, scale, exp, gt_R = dataset

    loader = DataLoader(
        TensorDataset(world, scale, exp, gt_R),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    perm_counts = torch.zeros(6, dtype=torch.long)
    scale_rank_counts = torch.zeros(3, 3, dtype=torch.long)
    exp_rank_counts = torch.zeros(3, 3, dtype=torch.long)

    matched_scale_sum = torch.zeros(3)
    matched_exp_sum = torch.zeros(3)
    n_samples = 0

    all_angles = []

    for x, s, e, R in loader:
        x = x.to(device)
        s = s.to(device)
        e = e.to(device)
        R = R.to(device)

        if dropout_keep is not None and dropout_keep < 1.0:
            x = dropout_without_replacement(x, dropout_keep)

        frame = frame_fn(x)

        best_idx, _ = best_perm_index(frame, R)
        perms = PERMS.to(device)[best_idx]

        _, angles = frame_loss_and_angles(frame, R)
        all_angles.append(angles.cpu().reshape(-1))

        perm_counts += torch.bincount(best_idx.cpu(), minlength=6)

        s_ranks = rank_indices(s)
        e_ranks = rank_indices(e)

        matched_s = s.gather(1, perms)
        matched_e = e.gather(1, perms)
        matched_s_rank = s_ranks.gather(1, perms)
        matched_e_rank = e_ranks.gather(1, perms)

        matched_scale_sum += matched_s.cpu().sum(dim=0)
        matched_exp_sum += matched_e.cpu().sum(dim=0)

        for ca in range(3):
            scale_rank_counts[ca] += torch.bincount(matched_s_rank[:, ca].cpu(), minlength=3)
            exp_rank_counts[ca] += torch.bincount(matched_e_rank[:, ca].cpu(), minlength=3)

        n_samples += x.shape[0]

    angles = torch.cat(all_angles)

    result = {
        "method": name,
        "n": n_samples,
        "axis_mean": float(angles.mean()),
        "axis_p95": float(torch.quantile(angles, 0.95)),
        "perm_counts": perm_counts,
        "scale_rank_counts": scale_rank_counts,
        "exp_rank_counts": exp_rank_counts,
        "matched_scale_mean": matched_scale_sum / n_samples,
        "matched_exp_mean": matched_exp_sum / n_samples,
    }

    return result


def print_result(case_name, result):
    n = result["n"]

    print()
    print(f"=== {case_name} / {result['method']} ===")
    print(f"axis_mean={result['axis_mean']:.3f} axis_p95={result['axis_p95']:.3f} n={n}")

    print("Permutation counts; tuple means canonical axes 0,1,2 -> GT axes:")
    for label, count in zip(PERM_LABELS, result["perm_counts"].tolist()):
        print(f"  {label}: {count:5d}  {100.0 * count / n:6.2f}%")

    print("Canonical axis -> GT scale rank counts [small, mid, large]:")
    for ca in range(3):
        row = result["scale_rank_counts"][ca].tolist()
        pct = [100.0 * v / n for v in row]
        print(f"  canon {ca}: {row}  pct={[round(x, 2) for x in pct]}")

    print("Canonical axis -> GT exponent rank counts [small, mid, large]:")
    for ca in range(3):
        row = result["exp_rank_counts"][ca].tolist()
        pct = [100.0 * v / n for v in row]
        print(f"  canon {ca}: {row}  pct={[round(x, 2) for x in pct]}")

    print(
        "Mean GT scale matched to canonical axes:",
        " ".join(f"{x:.4f}" for x in result["matched_scale_mean"].tolist()),
    )
    print(
        "Mean GT exponent matched to canonical axes:",
        " ".join(f"{x:.4f}" for x in result["matched_exp_mean"].tolist()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wpca-ckpt", required=True)
    ap.add_argument("--val-n", type=int, default=3000)
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
    print(f"device={device}")

    wpca = WeightedPCANet().to(device)
    ckpt = torch.load(args.wpca_ckpt, map_location=device)
    wpca.load_state_dict(ckpt["model"])
    wpca.eval()

    cases = [
        ("direct_fib512", "direct_fib", 512, None),
        ("fps2048_to_512", "fps", 512, None),
        (f"dropout{args.dropout_keep:g}_from_fib512", "direct_fib", 512, args.dropout_keep),
        (f"dropout{args.dropout_keep:g}_from_fps512", "fps", 512, args.dropout_keep),
        ("direct_fib307", "direct_fib", 307, None),
    ]

    text_lines = []
    csv_rows = []

    for i, (case_name, sampler, points, keep) in enumerate(cases):
        print(f"\nGenerating {case_name}")
        set_seed(args.seed + 5000 + i)
        dataset = sample_eval_dataset(
            args.val_n,
            points,
            args.candidate_n,
            sampler,
            args,
            device,
            args.gen_chunk,
        )

        methods = [
            ("hard_pca", lambda x: hard_pca_frame(x)[0]),
            ("weighted_pca", lambda x: wpca(x)[0]),
        ]

        for method_name, fn in methods:
            res = analyze_method(method_name, fn, dataset, args, device, dropout_keep=keep)
            print_result(case_name, res)

            text_lines.append(f"\n=== {case_name} / {method_name} ===")
            text_lines.append(f"axis_mean={res['axis_mean']:.3f} axis_p95={res['axis_p95']:.3f} n={res['n']}")
            text_lines.append("perm_counts=" + " ".join(
                f"{lab}:{cnt}" for lab, cnt in zip(PERM_LABELS, res["perm_counts"].tolist())
            ))
            text_lines.append("scale_rank_counts=" + str(res["scale_rank_counts"].tolist()))
            text_lines.append("exp_rank_counts=" + str(res["exp_rank_counts"].tolist()))
            text_lines.append("matched_scale_mean=" + " ".join(f"{x:.4f}" for x in res["matched_scale_mean"].tolist()))
            text_lines.append("matched_exp_mean=" + " ".join(f"{x:.4f}" for x in res["matched_exp_mean"].tolist()))

            row = {
                "case": case_name,
                "method": method_name,
                "n": res["n"],
                "axis_mean": res["axis_mean"],
                "axis_p95": res["axis_p95"],
            }
            for lab, cnt in zip(PERM_LABELS, res["perm_counts"].tolist()):
                row[f"perm_{lab}_count"] = cnt
                row[f"perm_{lab}_pct"] = 100.0 * cnt / res["n"]
            for ca in range(3):
                for rank, rank_name in enumerate(["small", "mid", "large"]):
                    row[f"canon{ca}_scale_rank_{rank_name}_pct"] = 100.0 * int(res["scale_rank_counts"][ca, rank]) / res["n"]
                    row[f"canon{ca}_exp_rank_{rank_name}_pct"] = 100.0 * int(res["exp_rank_counts"][ca, rank]) / res["n"]
                row[f"canon{ca}_matched_scale_mean"] = float(res["matched_scale_mean"][ca])
                row[f"canon{ca}_matched_exp_mean"] = float(res["matched_exp_mean"][ca])
            csv_rows.append(row)

    (out / "summary_short.txt").write_text("\n".join(text_lines) + "\n")

    fields = sorted({k for r in csv_rows for k in r.keys()})
    fields = ["case", "method", "n", "axis_mean", "axis_p95"] + [
        f for f in fields if f not in ("case", "method", "n", "axis_mean", "axis_p95")
    ]

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    print("\nWrote:")
    print(out / "summary_short.txt")
    print(out / "summary.csv")


if __name__ == "__main__":
    main()
