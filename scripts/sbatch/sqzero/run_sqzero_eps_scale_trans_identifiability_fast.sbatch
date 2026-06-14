#!/bin/bash -l
#SBATCH --job-name=sqzero-eps-strans
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

export WS=/hnvme/workspace/v123be13-WS
export REPO="$WS/superdec"
export SIF="$WS/containers/superdec_torch251_cuda124.sif"
export DATA="$WS/data/SQ_Zero/sqzero_sup_easy_10k"
export EVAL="$WS/superdec_eval"
export LOGS="$WS/logs"

mkdir -p "$EVAL" "$LOGS"

STAMP=$(date +"%Y%m%d_%H%M%S")
OUT="$EVAL/eps_scale_trans_ident_fast_${STAMP}"
mkdir -p "$OUT"

cd "$REPO"

echo "=== Job info ==="
date
hostname
pwd
nvidia-smi || true

echo "=== Git state ==="
git branch --show-current || true
git log --oneline --decorate -5 || true
git status --short || true

echo "=== Compile check ==="
apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python -m py_compile "$REPO/tools/diagnose_sqzero_eps_scale_trans_identifiability.py"

echo "=== Running fast eps+scale+translation identifiability diagnostic ==="
echo "Output: $OUT"

# Fast mode relative to previous eps+scale run:
# - 128 primitives instead of 256
# - 3 eps inits instead of 5
# - 1 scale init instead of 3
# - 3 nonzero translation offsets
# - 24x48 grid instead of 32x64
# - 200 steps instead of 300
apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/diagnose_sqzero_eps_scale_trans_identifiability.py" \
    --data-root "$DATA" \
    --split test \
    --out-dir "$OUT" \
    --n-points 4096 \
    --kmax 4 \
    --normal-mode radial \
    --batch-size 32 \
    --num-workers 4 \
    --max-primitives 128 \
    --min-visible-points 32 \
    --sampler fixed_grid \
    --grid-n-eta 24 \
    --grid-n-omega 48 \
    --optim-loss chamfer \
    --steps 200 \
    --lr 0.05 \
    --eps-min 0.1 \
    --eps-max 1.9 \
    --optimize-scale \
    --scale-min 0.05 \
    --scale-max 1.5 \
    --init-scale-mults "1,1,1" \
    --optimize-trans \
    --init-trans-offsets "0.04,0,0;-0.04,0.04,0;0,0,-0.04" \
    --init-eps-pairs "1.0,1.0;0.3,0.3;1.7,1.7" \
    --landscape-count 0 \
    --device cuda \
    --seed 0

echo "=== Output files ==="
find "$OUT" -maxdepth 2 -type f | sort

echo "=== Summary ==="
column -s, -t < "$OUT/eps_optimization_summary.csv" || cat "$OUT/eps_optimization_summary.csv"

echo "=== Worst loss-selected eps errors ==="
apptainer exec --nv --bind "$WS:$WS" "$SIF" python - <<PY
import csv
out = "$OUT"
path = f"{out}/eps_optimization_best_by_loss.csv"
cols = [
    "global_prim_index", "model_id", "primitive_k", "K", "visible_points_in_sample",
    "gt_eps1", "gt_eps2", "init_eps1", "init_eps2",
    "init_scale_l1_vs_gt", "init_trans_l2_vs_gt",
    "final_eps1", "final_eps2", "final_shape_l1",
    "scale_l1_vs_gt", "trans_l2_vs_gt",
    "loss_final_best", "best_step",
]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: float(r["final_shape_l1"]), reverse=True)
print(",".join(cols))
for r in rows[:15]:
    print(",".join(str(r[c]) for c in cols))
PY

echo "Wrote: $OUT"
