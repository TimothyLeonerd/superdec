#!/bin/bash -l
#SBATCH --job-name=gensq-wpca
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"
OUT="$WS/superdec_eval/gensq_weighted_pca_frame_20ep_${SLURM_JOB_ID}"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_gensq_weighted_pca_frame.py" \
    --out-dir "$OUT" \
    --train-n 20000 \
    --val-n 1000 \
    --points 512 \
    --epochs 20 \
    --batch-size 256 \
    --lr 1e-3 \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --exp-min 1.0 \
    --exp-max 8.0 \
    --device cuda \
    --seed 0

cat "$OUT/summary_short.txt"
