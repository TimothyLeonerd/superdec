#!/bin/bash -l
#SBATCH --job-name=gensq-eps-raybbox
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"

mkdir -p "$WS/logs" "$WS/superdec_eval"
cd "$REPO"

COMMON_ARGS=(
  --train-n 50000
  --val-n 2000
  --points 512
  --epochs 100
  --batch-size 128
  --lr 1e-3
  --weight-decay 1e-4
  --grad-clip 1.0
  --scale-min 0.08
  --scale-max 0.35
  --exp-min 0.5
  --exp-max 8.0
  --eval-steps 5
  --device cuda
  --seed 0
)

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  NAME="absmax_no_prednorm"
  OUT="$WS/superdec_eval/gensq_samplerfix_expmin05_eps_bboxnorm_rayaligned_absmax_noprednorm_100ep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  EXTRA=()
elif [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then
  NAME="absmax_with_prednorm"
  OUT="$WS/superdec_eval/gensq_samplerfix_expmin05_eps_bboxnorm_rayaligned_absmax_prednorm_100ep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  EXTRA=(--pred-normalize)
else
  echo "bad task id"
  exit 1
fi

echo "RUN=$NAME"
echo "OUT=$OUT"
mkdir -p "$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_gensq_eps_bboxnorm_rayaligned.py" \
    --out-dir "$OUT" \
    --norm-mode absmax \
    "${EXTRA[@]}" \
    "${COMMON_ARGS[@]}"

echo "=== summary ==="
cat "$OUT/summary_short.txt"

echo "=== last metrics ==="
tail -n 10 "$OUT/metrics.csv"
