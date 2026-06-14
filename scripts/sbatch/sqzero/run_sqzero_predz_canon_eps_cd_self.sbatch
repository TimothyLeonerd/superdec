#!/bin/bash -l
#SBATCH --job-name=predz-cd-self
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err

set -euo pipefail

WS=/hnvme/workspace/v123be13-WS
REPO="$WS/superdec"
SIF="$WS/containers/superdec_torch251_cuda124.sif"
DATA="$WS/data/SQ_Zero/sqzero_sup_easy_10k"
OUT="$WS/superdec_eval/predz_canon_eps_cd_self_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$WS/logs" "$OUT"
cd "$REPO"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python -m py_compile "$REPO/tools/train_sqzero_predz_canon_eps_cd_self.py"

apptainer exec --nv --bind "$WS:$WS" --env PYTHONPATH="$REPO" "$SIF" \
  python "$REPO/tools/train_sqzero_predz_canon_eps_cd_self.py" \
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
    --device cuda \
    --seed 0

echo "OUT=$OUT"
tail -n 20 "$OUT/metrics.csv"
