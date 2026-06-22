#!/bin/bash -l
#SBATCH --job-name=gensq-randscale-epsraw
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

echo "TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Checking sampler:"
grep -n "Robust differentiable radial sampler\|rho_root" "$REPO/tools/train_gensq_pointnet_local_scale_exp.py"

COMMON=(apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF")

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  NAME="scale_random_init"
  OUT="$WS/superdec_eval/gensq_samplerfix_expmin05_scale_updater_cd_random_100ep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

  mkdir -p "$OUT"
  echo "RUN=$NAME"
  echo "OUT=$OUT"

  "${COMMON[@]}" \
    python "$REPO/tools/train_gensq_scale_updater_cd.py" \
      --out-dir "$OUT" \
      --init-mode random \
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
      --exp-min 0.5 \
      --exp-max 8.0 \
      --eval-steps 5 \
      --device cuda \
      --seed 0

elif [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then
  NAME="eps_raw_varying_scale"
  OUT="$WS/superdec_eval/gensq_samplerfix_expmin05_eps_xraw_absolute_varyscale_100ep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

  mkdir -p "$OUT"
  echo "RUN=$NAME"
  echo "OUT=$OUT"

  "${COMMON[@]}" \
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
      --scale-min 0.08 \
      --scale-max 0.35 \
      --exp-min 0.5 \
      --exp-max 8.0 \
      --eval-steps 5 \
      --device cuda \
      --seed 0
fi

echo "=== summary ==="
cat "$OUT/summary_short.txt"

echo "=== last metrics ==="
tail -n 10 "$OUT/metrics.csv"
