#!/bin/bash -l
#SBATCH --job-name=gensq-scale-e05
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"

OUT="$WS/superdec_eval/gensq_samplerfix_expmin05_scale_updater_cd_bbox_100ep_${SLURM_JOB_ID}"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

echo "OUT=$OUT"
echo "Checking sampler:"
grep -n "Robust differentiable radial sampler\|rho_root" "$REPO/tools/train_gensq_pointnet_local_scale_exp.py"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
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
    --exp-min 0.5 \
    --exp-max 8.0 \
    --eval-steps 5 \
    --device cuda \
    --seed 0

echo "=== summary ==="
cat "$OUT/summary_short.txt"

echo "=== last metrics ==="
tail -n 10 "$OUT/metrics.csv"
