#!/bin/bash -l
#SBATCH --job-name=classgt-vs-gensq
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
OUT="$WS/superdec_eval/classical_target_classical_vs_gensq_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$WS/logs" "$OUT"
cd "$REPO"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python -m py_compile "$REPO/tools/compare_classical_target_classical_vs_gensq_fit.py"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/compare_classical_target_classical_vs_gensq_fit.py" \
    --out-dir "$OUT" \
    --n 64 \
    --surface-n 512 \
    --steps 2000 \
    --lr 0.03 \
    --eps-min 0.3 \
    --eps-max 1.7 \
    --exp-min 1.0 \
    --exp-max 8.0 \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --device cuda \
    --seed 0

echo "OUT=$OUT"
column -s, -t < "$OUT/summary.csv"
