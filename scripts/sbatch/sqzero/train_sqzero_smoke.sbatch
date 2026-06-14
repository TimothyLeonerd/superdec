#!/bin/bash -l
#SBATCH --job-name=superdec-sqzero-smoke
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/hnvme/workspace/v123be13-WS/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/v123be13-WS/logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

export WS=/hnvme/workspace/v123be13-WS
export REPO="$WS/superdec"
export SIF="$WS/containers/superdec_torch251_cuda124.sif"
export DATA="$WS/data/SQ_Zero_ShapeNetSmoke"
export OUT="$WS/superdec_runs"

mkdir -p "$WS/logs" "$OUT"

cd "$REPO"

echo "=== Environment ==="
hostname
date
echo "WS=$WS"
echo "REPO=$REPO"
echo "SIF=$SIF"
echo "DATA=$DATA"
echo "OUT=$OUT"
ls -lh "$SIF"
find "$DATA" -maxdepth 3 -type f | head -20

echo "=== Start tiny SuperDec training ==="
date

apptainer exec --nv \
  --bind "$WS:$WS" \
  --env PYTHONPATH="$REPO" \
  "$SIF" \
  bash -lc '
    set -euo pipefail
    cd "'"$REPO"'"

    export TORCH_CUDA_ARCH_LIST=9.0
    export PYTHONUNBUFFERED=1
    export WANDB_MODE=disabled

    python train/train.py \
      run_name=sqzero_smoke_001 \
      dataset=shapenet \
      shapenet.path="'"$DATA"'" \
      shapenet.categories=null \
      shapenet.normalize=true \
      trainer.save_path="'"$OUT"'" \
      trainer.num_epochs=1 \
      trainer.batch_size=2 \
      trainer.num_workers=0 \
      trainer.save_every_n_epochs=1 \
      trainer.augmentations=false \
      optimizer.enable_scheduler=false \
      loss.n_samples=64 \
      loss.w_cub=0.0 \
      loss.w_cd=1.0 \
      loss.w_ext=0.01 \
      loss.w_sps=0.1 \
      superdec.decoder.n_queries=8
  '

echo "=== Finished tiny SuperDec training ==="
date

echo "=== Output files ==="
find "$OUT" -maxdepth 4 -type f -print | sort | tail -50
