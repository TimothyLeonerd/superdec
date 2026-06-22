#!/bin/bash -l
#SBATCH --job-name=gensq-fps-scale
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"

mkdir -p "$WS/logs" "$WS/superdec_eval"
cd "$REPO"

OUT="$WS/superdec_eval/gensq_fps2048_to_512_scale_oracle_expmin05_N4_100ep_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "OUT=$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_gensq_fps2048_scale_eps_oracle.py" \
    --mode scale \
    --out-dir "$OUT" \
    --train-n 50000 \
    --val-n 2000 \
    --source-points 2048 \
    --points 512 \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --scale-floor 0.02 \
    --scale-ceil 0.80 \
    --exp-min 0.5 \
    --exp-max 8.0 \
    --unroll-steps 4 \
    --scale-damp 0.5 \
    --eps-damp 0.5 \
    --device cuda \
    --seed 0

echo "=== summary ==="
cat "$OUT/summary_short.txt"

echo "=== last metrics ==="
tail -n 30 "$OUT/metrics.csv"
