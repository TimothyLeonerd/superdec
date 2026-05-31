#!/usr/bin/env python3
import argparse
import ast
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


PHASE_RE = re.compile(r"(Epoch|Eval)\s+(\d+)/(\d+):")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

DEBUG_RE = re.compile(
    r"\[HungarianDebug\]\s+"
    r"call=(?P<call>\d+)\s+"
    r"phase=(?P<phase>\w+)\s+"
    r"sample=(?P<sample>\d+)\s+"
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


def plot_metric(rows, metric: str, out_png: Path):
    plt.figure(figsize=(10, 5))

    for phase in ["train", "eval"]:
        phase_rows = [r for r in rows if r.get("phase") == phase and metric in r]
        if not phase_rows:
            continue

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

        plt.plot(xs, ys, label=phase)

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_debug_metric(debug_rows, metric: str, out_png: Path):
    plt.figure(figsize=(10, 5))

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
    args = ap.parse_args()

    logfile = args.logfile
    if args.out_dir is None:
        out_dir = logfile.with_suffix("").parent / (logfile.with_suffix("").name + "_plots")
    else:
        out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = parse_metric_rows(logfile)
    debug_rows = parse_debug_rows(logfile)

    if not metric_rows and not debug_rows:
        raise RuntimeError(f"No metric/debug rows found in {logfile}")

    if metric_rows:
        metrics_csv = out_dir / "metrics.csv"
        write_csv(metric_rows, metrics_csv)

        all_metrics = sorted(
            {
                k
                for r in metric_rows
                for k in r.keys()
                if k not in {"phase", "epoch", "step_in_epoch"}
            }
        )

        for metric in all_metrics:
            plot_metric(metric_rows, metric, out_dir / f"{metric}.png")

        print(f"Parsed metric rows: {len(metric_rows)}")
        print(f"Wrote CSV: {metrics_csv}")
        print("Metric plots:")
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
            plot_debug_metric(debug_rows, metric, out_dir / f"debug_{metric}.png")

        print(f"Parsed Hungarian debug rows: {len(debug_rows)}")
        print(f"Wrote debug CSV: {debug_csv}")
        print("Debug plots:")
        for m in debug_metrics:
            print(f"  {m}")

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
