#!/bin/bash -l
#SBATCH --job-name=gensq-pca-directN
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --array=0-4
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"
SHAPE_CKPT="$WS/superdec_eval/gensq_pointnet_gtcanon_scale_exp_20260613_190624/best.pt"

POINTS=(512 461 410 358 307)
LABELS=(1.0 0.9 0.8 0.7 0.6)

PTS="${POINTS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"

RUN_ROOT="$WS/superdec_eval/gensq_pca_direct_points_grid_${SLURM_ARRAY_JOB_ID}"
OUT="$RUN_ROOT/direct_points_${PTS}_frac_${LABEL}"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

echo "PTS=$PTS LABEL=$LABEL OUT=$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/eval_gensq_pca_frame.py" \
    --out-dir "$OUT" \
    --shape-ckpt "$SHAPE_CKPT" \
    --val-n 1000 \
    --points "$PTS" \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --exp-min 1.0 \
    --exp-max 8.0 \
    --keep-frac 1.0 \
    --modes full \
    --device cuda \
    --seed 0
