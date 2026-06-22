#!/bin/bash -l
#SBATCH --job-name=gensq-pca-grid
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --array=0-8
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"
SHAPE_CKPT="$WS/superdec_eval/gensq_pointnet_gtcanon_scale_exp_20260613_190624/best.pt"

MODES=(full dropout dropout dropout dropout viewcrop viewcrop viewcrop viewcrop)
KEEPS=(1.0  0.9     0.8     0.7     0.6     0.9      0.8      0.7      0.6)

MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
KEEP="${KEEPS[$SLURM_ARRAY_TASK_ID]}"

RUN_ROOT="$WS/superdec_eval/gensq_pca_corruption_grid_090_060_${SLURM_ARRAY_JOB_ID}"
OUT="$RUN_ROOT/${MODE}_keep_${KEEP}"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

echo "MODE=$MODE"
echo "KEEP=$KEEP"
echo "OUT=$OUT"
echo "SHAPE_CKPT=$SHAPE_CKPT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/eval_gensq_pca_frame_keepgrid.py" \
    --out-dir "$OUT" \
    --shape-ckpt "$SHAPE_CKPT" \
    --val-n 1000 \
    --points 512 \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --exp-min 1.0 \
    --exp-max 8.0 \
    --keep-frac "$KEEP" \
    --modes "$MODE" \
    --device cuda \
    --seed 0

column -s, -t < "$OUT/pca_summary.csv"
