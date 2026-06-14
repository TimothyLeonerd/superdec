#!/usr/bin/env python3
import argparse, csv, math, importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class ConfigNode(SimpleNamespace):
    def __contains__(self, key): return hasattr(self, key)


def make_cfg(args):
    return ConfigNode(
        sqzero_lmdb=ConfigNode(
            path=args.data_root,
            n_points=args.n_points,
            normalize=True,
            load_sidecars=True,
            normal_mode="radial",
            kmax=args.kmax,
            train_split="train.txt",
            val_split=args.split,
            max_train_samples=None,
            max_val_samples=args.max_samples,
        ),
        trainer=ConfigNode(augmentations=False),
    )


def load_model(repo, ckpt_path, device):
    p = Path(repo) / "tools/train_sqzero_predz_canon_eps_cd_zsup.py"
    spec = importlib.util.spec_from_file_location("zsup", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.PredZCanonEpsNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model


def z_angle_deg(z_pred, z_gt):
    dot = (z_pred * z_gt).sum(dim=-1).abs().clamp(0, 1)
    return torch.acos(dot) * (180.0 / math.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test.txt")
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--prim-points", type=int, default=256)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--min-visible-points", type=int, default=32)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from superdec.data.sqzero_lmdb import SQZeroLMDB

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.repo, args.ckpt, device)

    ds = SQZeroLMDB(split="val", cfg=make_cfg(args))
    rng = np.random.default_rng(args.seed)

    rows = []

    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="samples"):
            item = ds[i]
            pts = item["points"].float()
            labels = item["labels"].long()
            K = int(item["K"].item())
            gt_scale = item["gt_scale"].float()
            gt_shape = item["gt_shape"].float()
            gt_rotate = item["gt_rotate"].float()
            gt_trans = item["gt_trans"].float()

            for k in range(K):
                idx = torch.where(labels == k)[0]
                visible = int(idx.numel())
                if visible < args.min_visible_points:
                    continue

                chosen = rng.choice(idx.cpu().numpy(), size=args.prim_points, replace=visible < args.prim_points)
                p = pts[torch.from_numpy(chosen).long()]
                size = torch.clamp(gt_scale[k].mean(), min=1e-6)
                x = ((p - gt_trans[k]) / size).unsqueeze(0).to(device)

                _eps, z_pred = model(x)
                z_pred = z_pred[0].cpu()
                z_gt = F.normalize(gt_rotate[k, :, 2], dim=0)

                ang = float(z_angle_deg(z_pred, z_gt))

                sx, sy, sz = [float(v) for v in gt_scale[k]]
                sm = (sx + sy + sz) / 3.0
                xy_mean = 0.5 * (sx + sy)

                rows.append({
                    "sample": i,
                    "prim": k,
                    "z_angle": ang,
                    "visible_points": visible,
                    "scale_x": sx,
                    "scale_y": sy,
                    "scale_z": sz,
                    "scale_mean": sm,
                    "scale_aniso": (max(sx, sy, sz) - min(sx, sy, sz)) / max(sm, 1e-8),
                    "z_distinct": abs(sz - xy_mean) / max(sm, 1e-8),
                    "xy_aniso": abs(sx - sy) / max(xy_mean, 1e-8),
                    "eps1": float(gt_shape[k, 0]),
                    "eps2": float(gt_shape[k, 1]),
                })

    csv_path = out / "z_ambiguity_details.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def corr(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(np.corrcoef(a, b)[0, 1])

    z = np.array([r["z_angle"] for r in rows])
    props = ["visible_points", "scale_aniso", "z_distinct", "xy_aniso", "eps1", "eps2"]

    print("n", len(rows))
    print("z_angle mean/median/p90/p95:", np.mean(z), np.median(z), np.quantile(z, .9), np.quantile(z, .95))
    print("\ncorrelations with z_angle:")
    for p in props:
        print(p, corr([r[p] for r in rows], z))

    print("\nquantile bins:")
    for p in ["scale_aniso", "z_distinct", "xy_aniso"]:
        vals = np.array([r[p] for r in rows])
        qs = np.quantile(vals, [0, .25, .5, .75, 1.0])
        print("\n", p, qs)
        for lo, hi in zip(qs[:-1], qs[1:]):
            mask = (vals >= lo) & (vals <= hi)
            print(f"  {lo:.4f}-{hi:.4f}: n={mask.sum()} z_mean={z[mask].mean():.2f} z_p95={np.quantile(z[mask], .95):.2f}")

    print("wrote", csv_path)


if __name__ == "__main__":
    main()
