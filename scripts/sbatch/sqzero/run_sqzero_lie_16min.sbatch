#!/bin/bash -l
#SBATCH --job-name=sqzero-lie16
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:35:00
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
OUT="$EVAL/eps_scale_trans_rot_lie_16min_${STAMP}"
mkdir -p "$OUT"

cd "$REPO"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python -m py_compile "$REPO/tools/diagnose_sqzero_eps_scale_trans_rot_identifiability.py"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/diagnose_sqzero_eps_scale_trans_rot_identifiability.py" \
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
    --eps-min 0.1 \
    --eps-max 1.9 \
    --optimize-scale \
    --scale-min 0.05 \
    --scale-max 1.5 \
    --init-scale-mults "1,1,1" \
    --optimize-trans \
    --init-trans-offsets "0.04,0,0" \
    --optimize-rotate \
    --rotation-param lie \
    --lie-commit-every 5 \
    --init-rotvecs-deg "0,0,0;15,0,0;0,15,0;0,0,15;-15,10,5;30,0,0" \
    --init-eps-pairs "1.0,1.0;0.3,0.3;1.7,1.7" \
    --landscape-count 0 \
    --device cuda \
    --seed 0

echo "Wrote: $OUT"
column -s, -t < "$OUT/eps_optimization_summary.csv"
