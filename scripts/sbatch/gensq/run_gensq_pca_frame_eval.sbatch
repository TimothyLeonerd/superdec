#!/bin/bash -l
#SBATCH --job-name=gensq-pca-frame
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"
SHAPE_CKPT="/hnvme/workspace/v123be13-WS/superdec_eval/gensq_pointnet_gtcanon_scale_exp_20260613_190624/best.pt"
OUT="$WS/superdec_eval/gensq_pca_frame_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT" "$WS/logs"
cd "$REPO"

echo "SHAPE_CKPT=$SHAPE_CKPT"
echo "OUT=$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF"   python "$REPO/tools/eval_gensq_pca_frame.py"     --out-dir "$OUT"     --shape-ckpt "$SHAPE_CKPT"     --val-n 1000     --points 512     --scale-min 0.08     --scale-max 0.35     --exp-min 1.0     --exp-max 8.0     --keep-frac 0.5     --modes full dropout viewcrop     --device cuda     --seed 0

column -s, -t < "$OUT/pca_summary.csv"
