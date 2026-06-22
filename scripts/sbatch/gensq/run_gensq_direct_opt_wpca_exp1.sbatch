#!/bin/bash -l
#SBATCH --job-name=gensq-direct-opt
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"

WPCA_CKPT="$WS/superdec_eval/gensq_weighted_pca_fps_clean_100ep_A_const_1e-3_506337_0/best.pt"
OUT="$WS/superdec_eval/gensq_direct_opt_wpca_exp1_${SLURM_JOB_ID}"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/optimize_gensq_scale_exp_direct_wpca.py" \
    --out-dir "$OUT" \
    --wpca-ckpt "$WPCA_CKPT" \
    --steps 700 \
    --lr 0.03 \
    --points 512 \
    --candidate-n 2048 \
    --surface-n 2048 \
    --eval-surface-n 4096 \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --exp-min 1.0 \
    --exp-max 8.0 \
    --seed 0 \
    --device cuda

echo "=== summary ==="
cat "$OUT/summary_short.txt"
