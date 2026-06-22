#!/bin/bash -l
#SBATCH --job-name=gensq-fix-rerun
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"

mkdir -p "$WS/logs" "$WS/superdec_eval"
cd "$REPO"

echo "TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Checking patched sampler:"
grep -n "Robust radial sampler\|rho_axis_hi" "$REPO/tools/train_gensq_pointnet_local_scale_exp.py"

COMMON_ENV=(apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF")

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  NAME="scale_bbox"
  OUT="$WS/superdec_eval/gensq_samplerfix_scale_updater_cd_bbox_e025_8_100ep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  mkdir -p "$OUT"
  echo "RUN=$NAME"
  echo "OUT=$OUT"

  "${COMMON_ENV[@]}" \
    python "$REPO/tools/train_gensq_scale_updater_cd.py" \
      --out-dir "$OUT" \
      --init-mode bbox \
      --train-n 50000 \
      --val-n 2000 \
      --points 512 \
      --epochs 100 \
      --batch-size 128 \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --grad-clip 1.0 \
      --scale-min 0.08 \
      --scale-max 0.35 \
      --exp-min 0.25 \
      --exp-max 8.0 \
      --eval-steps 5 \
      --device cuda \
      --seed 0

elif [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then
  NAME="eps_unit_gt_scale"
  OUT="$WS/superdec_eval/gensq_samplerfix_eps_xraw_absolute_unit_scale_e025_8_100ep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  mkdir -p "$OUT"
  echo "RUN=$NAME"
  echo "OUT=$OUT"

  "${COMMON_ENV[@]}" \
    python "$REPO/tools/train_gensq_eps_updater_xraw_predmode.py" \
      --out-dir "$OUT" \
      --pred-mode absolute \
      --train-n 50000 \
      --val-n 2000 \
      --points 512 \
      --epochs 100 \
      --batch-size 128 \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --grad-clip 1.0 \
      --scale-min 1.0 \
      --scale-max 1.0 \
      --exp-min 0.25 \
      --exp-max 8.0 \
      --eval-steps 5 \
      --device cuda \
      --seed 0

elif [ "$SLURM_ARRAY_TASK_ID" = "2" ]; then
  NAME="eps_bbox_absmax"
  OUT="$WS/superdec_eval/gensq_samplerfix_eps_xraw_absolute_bboxnorm_absmax_e025_8_100ep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  mkdir -p "$OUT"
  echo "RUN=$NAME"
  echo "OUT=$OUT"

  "${COMMON_ENV[@]}" \
    python "$REPO/tools/train_gensq_eps_updater_xraw_bboxnorm.py" \
      --out-dir "$OUT" \
      --pred-mode absolute \
      --norm-mode absmax \
      --train-n 50000 \
      --val-n 2000 \
      --points 512 \
      --epochs 100 \
      --batch-size 128 \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --grad-clip 1.0 \
      --scale-min 0.08 \
      --scale-max 0.35 \
      --exp-min 0.25 \
      --exp-max 8.0 \
      --eval-steps 5 \
      --device cuda \
      --seed 0

elif [ "$SLURM_ARRAY_TASK_ID" = "3" ]; then
  NAME="eps_bbox_half"
  OUT="$WS/superdec_eval/gensq_samplerfix_eps_xraw_absolute_bboxnorm_half_e025_8_100ep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  mkdir -p "$OUT"
  echo "RUN=$NAME"
  echo "OUT=$OUT"

  "${COMMON_ENV[@]}" \
    python "$REPO/tools/train_gensq_eps_updater_xraw_bboxnorm.py" \
      --out-dir "$OUT" \
      --pred-mode absolute \
      --norm-mode half \
      --train-n 50000 \
      --val-n 2000 \
      --points 512 \
      --epochs 100 \
      --batch-size 128 \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --grad-clip 1.0 \
      --scale-min 0.08 \
      --scale-max 0.35 \
      --exp-min 0.25 \
      --exp-max 8.0 \
      --eval-steps 5 \
      --device cuda \
      --seed 0
fi

echo "=== summary ==="
cat "$OUT/summary_short.txt"

echo "=== last metrics ==="
tail -n 10 "$OUT/metrics.csv"
