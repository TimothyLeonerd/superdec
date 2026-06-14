#!/bin/bash -l
#SBATCH --job-name=sqzero-fixedshape
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

export WS=/hnvme/workspace/v123be13-WS
export REPO="$WS/superdec"
export SIF="$WS/containers/superdec_torch251_cuda124.sif"
export DATA="$WS/data/SQ_Zero/sqzero_sup_easy_10k"
export EVAL="$WS/superdec_eval"

STAMP=$(date +"%Y%m%d_%H%M%S")
OUT="$EVAL/fixed_shape_pose_ident_${STAMP}"
mkdir -p "$OUT"

cd "$REPO"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/diagnose_sqzero_fixed_shape_pose_identifiability.py" \
    --data-root "$DATA" \
    --split test \
    --out-dir "$OUT" \
    --n-points 4096 \
    --kmax 4 \
    --normal-mode radial \
    --batch-size 32 \
    --num-workers 4 \
    --max-primitives 64 \
    --min-visible-points 32 \
    --sampler fixed_grid \
    --grid-n-eta 20 \
    --grid-n-omega 40 \
    --optim-loss chamfer \
    --steps 250 \
    --lr 0.03 \
    --init-eps-pairs "1.0,1.0" \
    --optimize-scale \
    --scale-min 0.05 \
    --scale-max 1.5 \
    --init-scale-mults "1,1,1" \
    --optimize-trans \
    --init-trans-offsets "0,0,0;0.04,0,0" \
    --optimize-rotate \
    --init-rotvecs-deg "0,0,0;15,0,0;0,15,0;0,0,15;-15,10,5;30,0,0" \
    --landscape-count 0 \
    --device cuda \
    --seed 0

echo "Wrote: $OUT"
column -s, -t < "$OUT/eps_optimization_summary.csv"

echo "=== Worst pose errors ==="
python - <<PY
import csv
path = "$OUT/eps_optimization_best_by_loss.csv"
cols = [
    "global_prim_index","model_id","primitive_k","K","visible_points_in_sample",
    "gt_eps1","gt_eps2","final_shape_l1",
    "scale_l1_vs_gt","trans_l2_vs_gt","rot_geodesic_deg_vs_gt",
    "loss_final_best","best_step",
]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: float(r["rot_geodesic_deg_vs_gt"]), reverse=True)
print(",".join(cols))
for r in rows[:20]:
    print(",".join(str(r[c]) for c in cols))
PY
