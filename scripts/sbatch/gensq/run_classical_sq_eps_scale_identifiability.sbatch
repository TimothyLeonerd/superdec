#!/bin/bash -l
#SBATCH --job-name=classsq-eps-scale
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"
OUT="$WS/superdec_eval/classsq_eps_scale_ident_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$WS/logs" "$OUT"
cd "$REPO"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python -m py_compile "$REPO/tools/diagnose_classical_sq_eps_scale_identifiability.py"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/diagnose_classical_sq_eps_scale_identifiability.py" \
    --out-dir "$OUT" \
    --n 64 \
    --surface-n 512 \
    --steps 2000 \
    --lr 0.03 \
    --eps-min 0.3 \
    --eps-max 1.7 \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --init random \
    --device cuda \
    --seed 0

echo "OUT=$OUT"
column -s, -t < "$OUT/summary.csv"
