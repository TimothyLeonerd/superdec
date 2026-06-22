#!/bin/bash -l
#SBATCH --job-name=gensq-down-A
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"

EXP_CKPT="$WS/superdec_eval/gensq_pointnet_gtcanon_scale_exp_20260613_190624/best.pt"
WPCA_CKPT="$WS/superdec_eval/gensq_weighted_pca_fps_clean_100ep_A_const_1e-3_506337_0/best.pt"
OUT="$WS/superdec_eval/gensq_downstream_frame_swap_A_${SLURM_JOB_ID}"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

echo "EXP_CKPT=$EXP_CKPT"
echo "WPCA_CKPT=$WPCA_CKPT"
echo "OUT=$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/eval_gensq_downstream_frame_swap.py" \
    --out-dir "$OUT" \
    --exp-ckpt "$EXP_CKPT" \
    --wpca-ckpt "$WPCA_CKPT" \
    --val-n 1000 \
    --points 512 \
    --candidate-n 2048 \
    --batch-size 256 \
    --gen-chunk 256 \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --exp-min 1.0 \
    --exp-max 8.0 \
    --dropout-keep 0.8 \
    --device cuda \
    --seed 0

echo "=== summary ==="
cat "$OUT/summary_short.txt"

echo "=== csv ==="
column -s, -t < "$OUT/summary.csv"
