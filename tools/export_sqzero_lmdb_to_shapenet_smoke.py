#!/usr/bin/env python3
import argparse
import io
import json
import shutil
from pathlib import Path

import lmdb
import numpy as np


def load_npy(txn, key: str):
    value = txn.get(key.encode("utf-8"))
    if value is None:
        raise KeyError(f"LMDB key not found: {key}")
    return np.load(io.BytesIO(value), allow_pickle=False)


def read_split(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def make_4096(points: np.ndarray, rng: np.random.Generator):
    """SuperDec ShapeNet loader samples 4096 points without replacement.
    Our SQ-Zero LMDB has 2048, so write 4096 by sampling with replacement once.
    """
    points = points.astype(np.float32, copy=False)
    n = points.shape[0]
    if n >= 4096:
        idx = rng.choice(n, 4096, replace=False)
    else:
        idx = rng.choice(n, 4096, replace=True)
    return points[idx].astype(np.float32, copy=False)


def fake_normals(points: np.ndarray):
    """Smoke-run normals.

    We will set loss.w_cub=0 for the first run, so normals are only needed
    because the SuperDec trainer expects a 'normals' array. Unit radial
    normals are safer than all-zero normals if some code inspects them.
    """
    centered = points - points.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centered, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (centered / denom).astype(np.float32)


def export_split(txn, keys, out_cat: Path, split_name: str, n_items: int, rng):
    chosen = keys[:n_items]
    model_ids = []
    manifest = []

    for i, key in enumerate(chosen):
        model_id = f"{split_name}_{i:06d}"
        model_ids.append(model_id)

        points = load_npy(txn, key)
        points = make_4096(points, rng)
        normals = fake_normals(points)

        model_dir = out_cat / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            model_dir / "pointcloud.npz",
            points=points.astype(np.float32),
            normals=normals.astype(np.float32),
        )

        manifest.append(
            {
                "model_id": model_id,
                "source_key": key,
                "points_shape": list(points.shape),
            }
        )

    (out_cat / f"{split_name}.lst").write_text("\n".join(model_ids) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb-root", required=True,
                    help="SQ-Zero root, e.g. /hnvme/.../data/SQ_Zero/1k_naive")
    ap.add_argument("--out-root", required=True,
                    help="Output root in SuperDec ShapeNet-like format")
    ap.add_argument("--category", default="sqzero")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    lmdb_root = Path(args.lmdb_root)
    out_root = Path(args.out_root)
    out_cat = out_root / args.category

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_cat.mkdir(parents=True, exist_ok=True)

    train_keys = read_split(lmdb_root / "train.txt")
    # Your SQ-Zero generator wrote test.txt; SuperDec expects val.lst.
    val_keys = read_split(lmdb_root / "test.txt")

    shard_dirs = sorted([p for p in lmdb_root.iterdir()
                         if p.is_dir() and (p / "data.mdb").exists()])
    if len(shard_dirs) != 1:
        raise RuntimeError(
            f"This smoke exporter expects exactly one LMDB shard; found {len(shard_dirs)}: {shard_dirs}"
        )

    rng = np.random.default_rng(args.seed)

    env = lmdb.open(
        str(shard_dirs[0]),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )

    with env.begin(write=False) as txn:
        train_manifest = export_split(txn, train_keys, out_cat, "train", args.n_train, rng)
        val_manifest = export_split(txn, val_keys, out_cat, "val", args.n_val, rng)

    env.close()

    summary = {
        "source_lmdb_root": str(lmdb_root),
        "out_root": str(out_root),
        "category": args.category,
        "n_train": len(train_manifest),
        "n_val": len(val_manifest),
        "train_lst": str(out_cat / "train.lst"),
        "val_lst": str(out_cat / "val.lst"),
        "note": "ShapeNet-style smoke dataset for unchanged SuperDec training.",
    }
    (out_root / "sqzero_smoke_manifest.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "train": train_manifest,
                "val": val_manifest,
            },
            indent=2,
        )
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
