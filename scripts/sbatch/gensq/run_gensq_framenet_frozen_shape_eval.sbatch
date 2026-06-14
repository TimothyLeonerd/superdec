#!/bin/bash -l
#SBATCH --job-name=gensq-frame-freeze
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"
SHAPE_CKPT="/hnvme/workspace/v123be13-WS/superdec_eval/gensq_pointnet_gtcanon_scale_exp_20260613_190624/best.pt"
OUT="$WS/superdec_eval/gensq_framenet_frozen_shape_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$WS/logs" "$OUT"
cd "$REPO"

echo "SHAPE_CKPT=$SHAPE_CKPT"
echo "OUT=$OUT"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF"   python -m py_compile "$REPO/tools/train_gensq_framenet_frozen_shape_eval.py"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF"   python "$REPO/tools/train_gensq_framenet_frozen_shape_eval.py"     --out-dir "$OUT"     --shape-ckpt "$SHAPE_CKPT"     --train-n 10000     --val-n 1000     --points 512     --epochs 50     --batch-size 128     --lr 0.001     --scale-min 0.08     --scale-max 0.35     --exp-min 1.0     --exp-max 8.0     --device cuda     --seed 0

echo "OUT=$OUT"
tail -n 20 "$OUT/metrics.csv"
