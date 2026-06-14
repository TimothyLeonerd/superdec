#!/bin/bash -l
#SBATCH --job-name=gensq-pn-perm
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
OUT="$WS/superdec_eval/gensq_pointnet_perm_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$WS/logs" "$OUT"
cd "$REPO"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python -m py_compile "$REPO/tools/train_gensq_pointnet_perm.py"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_gensq_pointnet_perm.py" \
    --out-dir "$OUT" \
    --train-n 10000 \
    --val-n 1000 \
    --points 512 \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.001 \
    --scale-min 0.08 \
    --scale-max 0.35 \
    --exp-min 1.0 \
    --exp-max 8.0 \
    --w-axis 1.0 \
    --w-scale 1.0 \
    --w-exp 1.0 \
    --device cuda \
    --seed 0

echo "OUT=$OUT"
tail -n 20 "$OUT/metrics.csv"
