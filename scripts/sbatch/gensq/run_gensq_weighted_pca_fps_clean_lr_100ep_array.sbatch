#!/bin/bash -l
#SBATCH --job-name=gensq-wpca-lr100
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG="A_const_1e-3"
    LR="1e-3"
    LR_SCHEDULE="constant"
    LR_MIN="1e-4"
    ;;
  1)
    TAG="B_cos_1e-3_to_1e-4"
    LR="1e-3"
    LR_SCHEDULE="cosine"
    LR_MIN="1e-4"
    ;;
  2)
    TAG="C_const_3e-4"
    LR="3e-4"
    LR_SCHEDULE="constant"
    LR_MIN="1e-4"
    ;;
  *)
    echo "bad array id ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

OUT="$WS/superdec_eval/gensq_weighted_pca_fps_clean_100ep_${TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

echo "TAG=$TAG"
echo "LR=$LR"
echo "LR_SCHEDULE=$LR_SCHEDULE"
echo "LR_MIN=$LR_MIN"
echo "OUT=$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_gensq_weighted_pca_fps_clean_lr.py" \
    --out-dir "$OUT" \
    --train-n 20000 \
    --val-n 1000 \
    --points 512 \
    --candidate-n 2048 \
    --epochs 100 \
    --batch-size 256 \
    --gen-chunk 256 \
    --lr "$LR" \
    --lr-schedule "$LR_SCHEDULE" \
    --lr-min "$LR_MIN" \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --exp-min 1.0 \
    --exp-max 8.0 \
    --dropout-keep 0.8 \
    --device cuda \
    --seed 0

echo "=== summary ==="
cat "$OUT/summary_short.txt"

echo "=== last metrics ==="
tail -n 5 "$OUT/metrics.csv"
