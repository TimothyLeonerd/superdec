#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PHASE_RE = re.compile(r"(Epoch|Eval)\s+(\d+)/(\d+):")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

DEBUG_RE = re.compile(
    r"\[HungarianDebug\]\s+"
    r"call=(?P<call>\d+)\s+"
    r"phase=(?P<phase>\w+)\s+"
    r"sample=(?P<sample>\d+)\s+"
    r"(?:model_id=(?P<model_id>\S+)\s+)?"
    r"K=(?P<K>\d+)\s+"
    r"matches=\[(?P<matches>[^\]]*)\]\s+"
    r"exist=\[(?P<exist>[^\]]*)\]\s+"
    r"soft_count=(?P<soft_count>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"pred_count=(?P<pred_count>\d+)\s+"
    r"assign_acc=(?P<assign_acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

MATCH_RE = re.compile(r"gt(?P<gt>\d+)->slot(?P<slot>\d+)")


def parse_metric_rows(path: Path):
    rows = []
    counters = {}

    for line in path.read_text(errors="replace").splitlines():
        m = PHASE_RE.search(line)
        if not m:
            continue

        phase = "train" if m.group(1) == "Epoch" else "eval"
        epoch = int(m.group(2))

        kvs = dict(KV_RE.findall(line))
        if not kvs:
            continue

        key = (phase, epoch)
        counters[key] = counters.get(key, 0) + 1

        row = {
            "phase": phase,
            "epoch": epoch,
            "step_in_epoch": counters[key],
        }

        for k, v in kvs.items():
            row[k] = float(v)

        rows.append(row)

    return rows


def parse_debug_rows(path: Path):
    rows = []

    for line in path.read_text(errors="replace").splitlines():
        m = DEBUG_RE.search(line)
        if not m:
            continue

        matches_text = m.group("matches")
        exist_text = m.group("exist")

        matches = {}
        for mm in MATCH_RE.finditer(matches_text):
            gt = int(mm.group("gt"))
            slot = int(mm.group("slot"))
            matches[gt] = slot

        exist = []
        if exist_text.strip():
            exist = [float(x.strip()) for x in exist_text.split(",") if x.strip()]

        row = {
            "call": int(m.group("call")),
            "phase": m.group("phase"),
            "sample": int(m.group("sample")),
            "model_id": m.group("model_id") or "",
            "K": int(m.group("K")),
            "soft_count": float(m.group("soft_count")),
            "pred_count": int(m.group("pred_count")),
            "assign_acc": float(m.group("assign_acc")),
            "matches_repr": matches_text,
            "exist_repr": exist_text,
        }

        for gt, slot in matches.items():
            row[f"match_gt{gt}"] = slot

        for i, value in enumerate(exist):
            row[f"exist_slot{i}"] = value

        rows.append(row)

    return rows


def numeric_keys(rows):
    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in {"epoch", "step_in_epoch"}:
                keys.add(k)
    return sorted(keys)


def write_csv(rows, out_csv: Path):
    if not rows:
        return

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    preferred = [
        "phase",
        "epoch",
        "step_in_epoch",
        "call",
        "sample",
        "model_id",
        "K",
        "soft_count",
        "pred_count",
        "assign_acc",
    ]

    fields = [k for k in preferred if k in all_keys]
    fields += sorted(k for k in all_keys if k not in fields)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_epoch_mean_rows(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["phase"], r["epoch"])].append(r)

    out = []

    for (phase, epoch), group in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        keys = numeric_keys(group)

        row = {
            "phase": phase,
            "epoch": epoch,
            "n_batches": len(group),
        }

        for k in keys:
            vals = [g[k] for g in group if k in g and math.isfinite(float(g[k]))]
            if vals:
                row[k] = sum(vals) / len(vals)

        out.append(row)

    return out


def get_xy_raw(rows, metric, phase):
    phase_rows = [r for r in rows if r.get("phase") == phase and metric in r]
    if not phase_rows:
        return [], []

    max_step_by_epoch = {}
    for r in phase_rows:
        e = r["epoch"]
        max_step_by_epoch[e] = max(max_step_by_epoch.get(e, 1), r["step_in_epoch"])

    xs = []
    ys = []
    for r in phase_rows:
        e = r["epoch"]
        denom = max_step_by_epoch[e]
        x = e + (r["step_in_epoch"] - 1) / max(denom, 1)
        xs.append(x)
        ys.append(r[metric])

    return xs, ys


def get_xy_epoch(rows, metric, phase):
    phase_rows = [r for r in rows if r.get("phase") == phase and metric in r]
    if not phase_rows:
        return [], []

    xs = [r["epoch"] for r in phase_rows]
    ys = [r[metric] for r in phase_rows]
    return xs, ys


def plot_metric_raw(rows, metric: str, out_png: Path):
    plt.figure(figsize=(11, 5))

    for phase in ["train", "eval"]:
        xs, ys = get_xy_raw(rows, metric, phase)
        if xs:
            plt.plot(xs, ys, label=phase, linewidth=1, alpha=0.85)

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(metric + " raw per-batch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_metric_epoch_mean(rows, metric: str, out_png: Path, y_limit=None, log_y=False):
    plt.figure(figsize=(11, 5))

    for phase in ["train", "eval"]:
        xs, ys = get_xy_epoch(rows, metric, phase)
        if xs:
            plt.plot(xs, ys, label=phase, linewidth=2)

    if log_y:
        plt.yscale("log")

    if y_limit is not None:
        plt.ylim(y_limit)

    plt.xlabel("epoch")
    plt.ylabel(metric)
    title = metric + " epoch mean"
    if log_y:
        title += " log-y"
    if y_limit is not None:
        title += " zoom"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[max(0, min(idx, len(values) - 1))]


def zoom_limit_for_metric(epoch_rows, metric, skip_first_epochs=3, q=0.95):
    vals = []
    for r in epoch_rows:
        if r.get("epoch", 0) <= skip_first_epochs:
            continue
        if metric in r and math.isfinite(float(r[metric])):
            vals.append(float(r[metric]))

    if not vals:
        return None

    p = percentile(vals, q)
    if p is None:
        return None

    ymax = max(p * 1.15, 1e-8)
    ymin = 0.0

    return (ymin, ymax)


def should_make_log_plot(metric):
    return metric == "all" or metric.endswith("_loss") or "loss" in metric


def plot_debug_metric(debug_rows, metric: str, out_png: Path):
    plt.figure(figsize=(11, 5))

    for phase in ["train", "eval"]:
        phase_rows = [r for r in debug_rows if r.get("phase") == phase and metric in r]
        if not phase_rows:
            continue

        xs = [r["call"] for r in phase_rows]
        ys = [r[metric] for r in phase_rows]
        plt.plot(xs, ys, marker="o", linewidth=1, markersize=3, label=phase)

    plt.xlabel("forward call")
    plt.ylabel(metric)
    plt.title(f"debug/{metric}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--zoom-skip-first-epochs", type=int, default=3)
    ap.add_argument("--zoom-quantile", type=float, default=0.95)
    args = ap.parse_args()

    logfile = args.logfile
    if args.out_dir is None:
        out_dir = logfile.with_suffix("").parent / (logfile.with_suffix("").name + "_plots")
    else:
        out_dir = args.out_dir

    raw_dir = out_dir / "raw"
    epoch_dir = out_dir / "epoch_mean"
    zoom_dir = out_dir / "epoch_mean_zoom"
    log_dir = out_dir / "epoch_mean_log"
    debug_dir = out_dir / "debug"

    for d in [out_dir, raw_dir, epoch_dir, zoom_dir, log_dir, debug_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metric_rows = parse_metric_rows(logfile)
    debug_rows = parse_debug_rows(logfile)

    if not metric_rows and not debug_rows:
        raise RuntimeError(f"No metric/debug rows found in {logfile}")

    if metric_rows:
        metrics_csv = out_dir / "metrics_raw.csv"
        write_csv(metric_rows, metrics_csv)

        epoch_rows = make_epoch_mean_rows(metric_rows)
        epoch_csv = out_dir / "metrics_epoch_mean.csv"
        write_csv(epoch_rows, epoch_csv)

        all_metrics = sorted(
            {
                k
                for r in metric_rows
                for k in r.keys()
                if k not in {"phase", "epoch", "step_in_epoch"}
            }
        )

        for metric in all_metrics:
            plot_metric_raw(metric_rows, metric, raw_dir / f"{metric}_raw.png")
            plot_metric_epoch_mean(epoch_rows, metric, epoch_dir / f"{metric}_epoch_mean.png")

            y_limit = zoom_limit_for_metric(
                epoch_rows,
                metric,
                skip_first_epochs=args.zoom_skip_first_epochs,
                q=args.zoom_quantile,
            )
            if y_limit is not None:
                plot_metric_epoch_mean(
                    epoch_rows,
                    metric,
                    zoom_dir / f"{metric}_epoch_mean_zoom.png",
                    y_limit=y_limit,
                )

            if should_make_log_plot(metric):
                # Avoid log(0) display issues by relying on matplotlib;
                # exact zeros will simply not be shown well, which is fine for loss diagnostics.
                plot_metric_epoch_mean(
                    epoch_rows,
                    metric,
                    log_dir / f"{metric}_epoch_mean_log.png",
                    log_y=True,
                )

        print(f"Parsed metric rows: {len(metric_rows)}")
        print(f"Wrote raw CSV: {metrics_csv}")
        print(f"Wrote epoch-mean CSV: {epoch_csv}")
        print(f"Wrote raw plots to: {raw_dir}")
        print(f"Wrote epoch-mean plots to: {epoch_dir}")
        print(f"Wrote zoomed epoch-mean plots to: {zoom_dir}")
        print(f"Wrote log-y epoch-mean plots to: {log_dir}")
        print("Metrics:")
        for m in all_metrics:
            print(f"  {m}")

    if debug_rows:
        debug_csv = out_dir / "hungarian_debug.csv"
        write_csv(debug_rows, debug_csv)

        debug_metrics = sorted(
            {
                k
                for r in debug_rows
                for k in r.keys()
                if (
                    k.startswith("match_gt")
                    or k.startswith("exist_slot")
                    or k in {"soft_count", "pred_count", "assign_acc"}
                )
            }
        )

        for metric in debug_metrics:
            plot_debug_metric(debug_rows, metric, debug_dir / f"debug_{metric}.png")

        print(f"Parsed Hungarian debug rows: {len(debug_rows)}")
        print(f"Wrote debug CSV: {debug_csv}")
        print(f"Wrote debug plots to: {debug_dir}")

    print(f"Wrote all outputs to: {out_dir}")


if __name__ == "__main__":
    main()