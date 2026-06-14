#!/bin/bash -l
#SBATCH --job-name=superdec-lmdb-smoke
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
export DATA="$WS/data/SQ_Zero/1k_naive"
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
ls -l "$DATA"/train.txt "$DATA"/test.txt
find "$DATA" -maxdepth 2 -type f | head -20

echo "=== Check local code changes ==="
grep -Rni "class SQZeroLMDB" superdec/data/sqzero_lmdb.py
grep -Rni "sqzero_lmdb" train/utils.py configs/train.yaml

echo "=== Start native LMDB SuperDec smoke training ==="
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
      run_name=sqzero_lmdb_smoke_001 \
      dataset=sqzero_lmdb \
      sqzero_lmdb.path="'"$DATA"'" \
      sqzero_lmdb.n_points=4096 \
      sqzero_lmdb.normalize=true \
      sqzero_lmdb.normal_mode=radial \
      sqzero_lmdb.load_sidecars=false \
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

echo "=== Finished native LMDB smoke training ==="
date

echo "=== Output files ==="
find "$OUT/sqzero_lmdb_smoke_001" -maxdepth 3 -type f -print | sort || true

echo "=== Done ==="
date
