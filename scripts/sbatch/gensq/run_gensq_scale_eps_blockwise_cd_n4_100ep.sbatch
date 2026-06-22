#!/bin/bash -l
#SBATCH --job-name=gensq-block-n4
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"

OUT="$WS/superdec_eval/gensq_scale_eps_blockwise_cd_n4_e025_8_100ep_${SLURM_JOB_ID}"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

echo "OUT=$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_gensq_scale_eps_blockwise_cd.py" \
    --out-dir "$OUT" \
    --train-n 50000 \
    --val-n 2000 \
    --points 512 \
    --epochs 100 \
    --batch-size 128 \
    --n-loop 4 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --exp-min 0.25 \
    --exp-max 8.0 \
    --eps0 2.0 \
    --device cuda \
    --seed 0

echo "=== summary ==="
cat "$OUT/summary_short.txt"

echo "=== last metrics ==="
tail -n 10 "$OUT/metrics.csv"
