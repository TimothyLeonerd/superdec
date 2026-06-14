#!/bin/bash -l
#SBATCH --job-name=sqzero-eps-ident
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

export WS=/hnvme/workspace/v123be13-WS
export REPO="$WS/superdec"
export SIF="$WS/containers/superdec_torch251_cuda124.sif"
export DATA="$WS/data/SQ_Zero/sqzero_sup_easy_10k"
export EVAL="$WS/superdec_eval"
export LOGS="$WS/logs"

mkdir -p "$LOGS" "$EVAL"
cd "$REPO"

echo "=== Job info ==="
date
hostname
pwd
nvidia-smi || true

echo "=== Repo state ==="
git branch --show-current || true
git log --oneline --decorate -5 || true
git status --short || true

echo "=== Compile check ==="
apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python -m py_compile "$REPO/tools/diagnose_sqzero_eps_identifiability.py"

STAMP=$(date +"%Y%m%d_%H%M%S")
OUT="$EVAL/eps_ident_fixed_grid_${STAMP}"
mkdir -p "$OUT"

echo "=== Running fixed-grid fixed-pose eps identifiability diagnostic ==="
apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/diagnose_sqzero_eps_identifiability.py" \
    --data-root "$DATA" \
    --split test \
    --out-dir "$OUT" \
    --n-points 4096 \
    --kmax 4 \
    --normal-mode radial \
    --batch-size 32 \
    --num-workers 0 \
    --max-primitives 256 \
    --min-visible-points 32 \
    --sampler fixed_grid \
    --grid-n-eta 32 \
    --grid-n-omega 64 \
    --optim-loss chamfer \
    --steps 300 \
    --lr 0.05 \
    --init-eps-pairs "1.0,1.0;0.3,0.3;1.7,1.7;0.3,1.7;1.7,0.3" \
    --landscape-count 8 \
    --landscape-grid-steps 41 \
    --landscape-loss pointwise \
    --seed 0

echo "=== Outputs ==="
find "$OUT" -maxdepth 2 -type f | sort

echo "=== Summary CSV ==="
column -s, -t < "$OUT/eps_optimization_summary.csv" || cat "$OUT/eps_optimization_summary.csv"

echo "Done: $OUT"
