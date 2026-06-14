#!/bin/bash -l
#SBATCH --job-name=predz-cd-zsup-hi
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%A_%a.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"
DATA="$WS/data/SQ_Zero/sqzero_sup_easy_10k"

W_LIST=(5.0 10.0)
W="${W_LIST[$SLURM_ARRAY_TASK_ID]}"
SAFE="${W/./_}"

OUT="$WS/superdec_eval/predz_canon_eps_cd_zsup_w${SAFE}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$WS/logs" "$OUT"
cd "$REPO"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_sqzero_predz_canon_eps_cd_zsup.py" \
    --data-root "$DATA" \
    --out-dir "$OUT" \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.001 \
    --prim-points 256 \
    --surf-eta 16 \
    --surf-omega 32 \
    --min-visible-points 32 \
    --max-train-objects 9000 \
    --max-val-objects 1000 \
    --w-z "$W" \
    --device cuda \
    --seed 0

echo "OUT=$OUT"
tail -n 20 "$OUT/metrics.csv"
