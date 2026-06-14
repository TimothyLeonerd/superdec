#!/usr/bin/env bash
set -euo pipefail

EVAL=${1:-/hnvme/workspace/v123be13-WS/superdec_eval}
VARIANTS=(rotvec quat rot6d z6d lie)

ARGS=()
echo "=== Selected latest run dirs ==="
for v in "${VARIANTS[@]}"; do
  d=$(ls -td "$EVAL"/eps_scale_trans_rot_${v}_16min_* 2>/dev/null | head -1 || true)
  if [ -z "$d" ]; then
    echo "MISSING: $v"
    exit 1
  fi
  echo "$v -> $d"
  ARGS+=("$v=$d")
done

python3 - "${ARGS[@]}" <<'PY'
import sys, csv, math
from pathlib import Path
from collections import defaultdict

def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

def f(x):
    try: return float(x)
    except Exception: return float("nan")

def q(vals, qq):
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals: return float("nan")
    i = (len(vals)-1)*qq
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    if lo == hi: return vals[lo]
    return vals[lo]*(hi-i) + vals[hi]*(i-lo)

def stats(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return dict(
        n=len(vals),
        mean=sum(vals)/len(vals) if vals else float("nan"),
        median=q(vals,0.5),
        p90=q(vals,0.9),
        p95=q(vals,0.95),
        max=max(vals) if vals else float("nan"),
    )

def fmt(x):
    if isinstance(x, int): return str(x)
    if not math.isfinite(x): return "nan"
    return f"{x:.6g}"

runs = {}
for arg in sys.argv[1:]:
    v, d = arg.split("=", 1)
    runs[v] = Path(d)

best_loss = {}
best_shape = {}
all_rows = {}
for v, d in runs.items():
    best_loss[v] = read_csv(d/"eps_optimization_best_by_loss.csv")
    best_shape[v] = read_csv(d/"eps_optimization_best_by_shape.csv")
    all_rows[v] = read_csv(d/"eps_optimization_rows.csv")

print("\n=== PRIMARY: best-by-loss, i.e. lowest Chamfer per primitive ===")
print("variant,n,loss_mean,loss_median,loss_p90,loss_p95,loss_max,"
      "succ_loss<1e-6,succ_loss<1e-5,succ_loss<1e-4,succ_loss<1e-3,"
      "shape_mean,shape_p95,rotdeg_p95,rotdeg_max,trans_p95,beststep_median")
for v, rows in best_loss.items():
    losses = [f(r["loss_final_best"]) for r in rows]
    shapes = [f(r["final_shape_l1"]) for r in rows]
    rots = [f(r["rot_geodesic_deg_vs_gt"]) for r in rows]
    trans = [f(r["trans_l2_vs_gt"]) for r in rows]
    steps = [f(r["best_step"]) for r in rows]
    s_loss = stats(losses)
    line = [
        v, s_loss["n"], s_loss["mean"], s_loss["median"], s_loss["p90"], s_loss["p95"], s_loss["max"],
        sum(x < 1e-6 for x in losses)/len(losses),
        sum(x < 1e-5 for x in losses)/len(losses),
        sum(x < 1e-4 for x in losses)/len(losses),
        sum(x < 1e-3 for x in losses)/len(losses),
        stats(shapes)["mean"], stats(shapes)["p95"],
        stats(rots)["p95"], stats(rots)["max"],
        stats(trans)["p95"], stats(steps)["median"],
    ]
    print(",".join(fmt(x) if not isinstance(x,str) else x for x in line))

print("\n=== SECONDARY: best-by-shape, to see if correct-ish eps exists but Chamfer rejects it ===")
print("variant,n,shape_mean,shape_median,shape_p95,loss_mean,loss_p95,rotdeg_p95")
for v, rows in best_shape.items():
    line = [
        v, len(rows),
        stats([f(r["final_shape_l1"]) for r in rows])["mean"],
        stats([f(r["final_shape_l1"]) for r in rows])["median"],
        stats([f(r["final_shape_l1"]) for r in rows])["p95"],
        stats([f(r["loss_final_best"]) for r in rows])["mean"],
        stats([f(r["loss_final_best"]) for r in rows])["p95"],
        stats([f(r["rot_geodesic_deg_vs_gt"]) for r in rows])["p95"],
    ]
    print(",".join(fmt(x) if not isinstance(x,str) else x for x in line))

print("\n=== Win counts: which parametrization gives lowest loss per primitive ===")
by_prim = defaultdict(dict)
meta = {}
for v, rows in best_loss.items():
    for r in rows:
        k = int(r["global_prim_index"])
        by_prim[k][v] = r
        meta[k] = r

wins = defaultdict(int)
for k, d in by_prim.items():
    winner = min(d, key=lambda v: f(d[v]["loss_final_best"]))
    wins[winner] += 1
for v in runs:
    print(f"{v},{wins[v]}")

print("\n=== Hard cases: high best-achievable loss even after trying all parametrizations ===")
print("prim,model_id,k,gt_eps1,gt_eps2,min_loss,best_variant,rotvec_loss,quat_loss,rot6d_loss,z6d_loss")
hard = []
for k, d in by_prim.items():
    if len(d) != len(runs): continue
    best_v = min(d, key=lambda v: f(d[v]["loss_final_best"]))
    hard.append((f(d[best_v]["loss_final_best"]), k, best_v))
for minloss, k, best_v in sorted(hard, reverse=True)[:20]:
    r = meta[k]
    vals = [f(by_prim[k][v]["loss_final_best"]) for v in ["rotvec","quat","rot6d","z6d","lie"]]
    print(",".join([str(k), r["model_id"], r["primitive_k"], fmt(f(r["gt_eps1"])), fmt(f(r["gt_eps2"])),
                    fmt(minloss), best_v] + [fmt(x) for x in vals]))

print("\n=== Disagreement cases: one parametrization escapes much better than another ===")
print("prim,model_id,k,best_variant,best_loss,worst_loss,ratio,rotvec,quat,rot6d,z6d")
dis = []
for k, d in by_prim.items():
    if len(d) != len(runs): continue
    vals = {v:f(d[v]["loss_final_best"]) for v in runs}
    mn = min(vals.values())
    mx = max(vals.values())
    ratio = mx / max(mn, 1e-12)
    dis.append((ratio, k, vals))
for ratio, k, vals in sorted(dis, reverse=True)[:20]:
    r = meta[k]
    best_v = min(vals, key=vals.get)
    print(",".join([str(k), r["model_id"], r["primitive_k"], best_v,
                    fmt(vals[best_v]), fmt(max(vals.values())), fmt(ratio)] +
                   [fmt(vals[v]) for v in ["rotvec","quat","rot6d","z6d","lie"]]))

print("\n=== Primitive 23, if present ===")
print("variant,loss,shape_l1,rot_deg,scale_l1,trans_l2,best_step,final_eps1,final_eps2")
for v in ["rotvec","quat","rot6d","z6d","lie"]:
    rows = [r for r in best_loss[v] if int(r["global_prim_index"]) == 23]
    for r in rows:
        print(",".join([
            v,
            fmt(f(r["loss_final_best"])),
            fmt(f(r["final_shape_l1"])),
            fmt(f(r["rot_geodesic_deg_vs_gt"])),
            fmt(f(r["scale_l1_vs_gt"])),
            fmt(f(r["trans_l2_vs_gt"])),
            r["best_step"],
            fmt(f(r["final_eps1"])),
            fmt(f(r["final_eps2"])),
        ]))
PY
