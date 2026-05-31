#!/bin/bash -l
#SBATCH --job-name=superdec-suphung-smoke
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
export DATA="$WS/data/SQ_Zero/sqzero_sup_easy_10k"
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

echo "=== Check code ==="
grep -Rni "class SupervisedHungarianLoss" superdec/loss/supervised_hungarian_loss.py
grep -Rni "supervised_hungarian" train/utils.py configs/train.yaml
grep -Rni "requires_batch" train/trainer.py

echo "=== Start supervised Hungarian smoke training ==="
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
    export HYDRA_FULL_ERROR=1

    python train/train.py \
      run_name=sqzero_suphung_smoke_001 \
      dataset=sqzero_lmdb \
      sqzero_lmdb.path="'"$DATA"'" \
      sqzero_lmdb.n_points=4096 \
      sqzero_lmdb.normalize=true \
      sqzero_lmdb.normal_mode=radial \
      sqzero_lmdb.load_sidecars=true \
      sqzero_lmdb.kmax=4 \
      trainer.save_path="'"$OUT"'" \
      trainer.num_epochs=1 \
      trainer.batch_size=2 \
      trainer.num_workers=0 \
      trainer.save_every_n_epochs=1 \
      trainer.augmentations=false \
      optimizer.enable_scheduler=false \
      loss.type=supervised_hungarian \
      loss.w_sup_exist=1.0 \
      loss.w_sup_assign=1.0 \
      loss.match_w_assign=1.0 \
      loss.match_w_exist=0.1 \
      superdec.decoder.n_queries=4
  '

echo "=== Finished supervised Hungarian smoke training ==="
date

echo "=== Output files ==="
find "$OUT/sqzero_suphung_smoke_001" -maxdepth 3 -type f -print | sort || true

echo "=== Done ==="
date
