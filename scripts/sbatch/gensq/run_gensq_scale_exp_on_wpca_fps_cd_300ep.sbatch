#!/bin/bash -l
#SBATCH --job-name=gensq-exp-cd
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"

WPCA_CKPT="$WS/superdec_eval/gensq_weighted_pca_fps_clean_100ep_A_const_1e-3_506337_0/best.pt"
OUT="$WS/superdec_eval/gensq_scale_exp_on_wpca_fps_cd_300ep_${SLURM_JOB_ID}"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

echo "WPCA_CKPT=$WPCA_CKPT"
echo "OUT=$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_gensq_scale_exp_on_wpca_fps_cd.py" \
    --out-dir "$OUT" \
    --wpca-ckpt "$WPCA_CKPT" \
    --train-n 50000 \
    --val-n 2000 \
    --eval-n 1000 \
    --points 512 \
    --candidate-n 2048 \
    --epochs 300 \
    --batch-size 128 \
    --gen-chunk 256 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
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
tail -n 10 "$OUT/metrics.csv"
