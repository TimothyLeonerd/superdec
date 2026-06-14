#!/bin/bash -l
#SBATCH --job-name=train_sqzero_suphung_surface_easy10k_150ep_surface_15_0
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
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
wc -l "$DATA"/train.txt "$DATA"/test.txt

echo "=== Check code ==="
grep -Rni "class SupervisedHungarianLoss" superdec/loss/supervised_hungarian_loss.py
grep -Rni "w_sup_surface\|surface_n_samples\|_compute_surface_loss" superdec/loss/supervised_hungarian_loss.py configs/train.yaml
grep -Rni "grad_clip_norm" train/trainer.py configs/train.yaml
grep -Rni "dropout=getattr" superdec/superdec.py || true

echo "=== Start supervised Hungarian + surface real train/val on easy10k, 150 epochs ==="
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
      run_name=train_sqzero_suphung_surface_easy10k_150ep_surface_15_0 \
      dataset=sqzero_lmdb \
      sqzero_lmdb.path="'"$DATA"'" \
      sqzero_lmdb.train_split=train.txt \
      sqzero_lmdb.val_split=test.txt \
      sqzero_lmdb.n_points=4096 \
      sqzero_lmdb.normalize=true \
      sqzero_lmdb.normal_mode=radial \
      sqzero_lmdb.load_sidecars=true \
      sqzero_lmdb.kmax=4 \
      trainer.save_path="'"$OUT"'" \
      trainer.num_epochs=150 \
      trainer.batch_size=32 \
      trainer.num_workers=4 \
      trainer.save_every_n_epochs=25 \
      trainer.augmentations=false \
      trainer.grad_clip_norm=1.0 \
      optimizer.enable_scheduler=false \
      optimizer.lr=0.0001 \
      loss.type=supervised_hungarian \
      loss.w_sup_exist=2.0 \
      loss.w_sup_assign=1.0 \
      loss.w_sup_surface=10.0 \
      loss.surface_n_samples=128 \
      loss.surface_D_eta=0.05 \
      loss.surface_D_omega=0.05 \
      loss.match_w_assign=1.0 \
      loss.match_w_exist=0.0 \
      loss.debug_match_every=0 \
      loss.debug_match_max_samples=4 \
      superdec.decoder.n_queries=4 \
      superdec.decoder.dropout=0.0
  '

echo "=== Finished supervised Hungarian + surface real train/val on easy10k, 150 epochs ==="
date

echo "=== Output files ==="
find "$OUT/train_sqzero_suphung_surface_easy10k_150ep_surface_15_0" -maxdepth 3 -type f -print | sort || true

echo "=== Done ==="
date
