#!/bin/bash -l
#SBATCH --job-name=zcanon-noise
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --array=0-4
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.err

set -euo pipefail

NOISES=(2 5 9 14 20)
NOISE="${NOISES[$SLURM_ARRAY_TASK_ID]}"
NTAG="${NOISE//./_}"

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"
DATA="$WS/data/SQ_Zero/sqzero_sup_easy_10k"
OUT="$WS/superdec_eval/zcanon_eps_noise_${NTAG}deg_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$WS/logs" "$OUT"
cd "$REPO"

echo "NOISE=$NOISE"
echo "OUT=$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python -m py_compile "$REPO/tools/train_sqzero_zcanon_eps_noise_oracle.py"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_sqzero_zcanon_eps_noise_oracle.py" \
    --data-root "$DATA" \
    --out-dir "$OUT" \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --prim-points 256 \
    --min-visible-points 32 \
    --max-train-objects 9000 \
    --max-val-objects 1000 \
    --z-noise-deg "$NOISE" \
    --device cuda \
    --seed 0

echo "OUT=$OUT"
tail -n 5 "$OUT/metrics.csv"
