#!/bin/bash -l
#SBATCH --job-name=train_sqzero_staged_strict_v1_r020_w5
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

export RUN_NAME="train_sqzero_staged_strict_v1_easy10k_150ep_r020_w5"

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
echo "RUN_NAME=$RUN_NAME"
ls -lh "$SIF"
ls -l "$DATA"/train.txt "$DATA"/test.txt
wc -l "$DATA"/train.txt "$DATA"/test.txt

echo "=== Git state ==="
git status --short || true
git branch --show-current || true
git log --oneline --decorate -5 || true

echo "=== Check staged code/config ==="
grep -Rni "staged_params" superdec/superdec.py configs/train.yaml
grep -Rni "use_staged_surface\|stage1_fixed_radius\|w_stage1_surface\|w_stage2_surface\|w_stage3_surface" \
  superdec/loss/supervised_hungarian_loss.py configs/train.yaml
grep -Rni "staged_outdicts\|_make_stage1\|_make_stage2\|_make_stage3\|_compute_staged_surface_losses" \
  superdec/loss/supervised_hungarian_loss.py superdec/superdec.py
grep -Rni "grad_clip_norm" train/trainer.py configs/train.yaml

echo "=== Check optimizer.step count around training loop ==="
grep -Rni "optimizer.step" train/trainer.py

echo "=== Compile check ==="
apptainer exec --nv \
  --bind "$WS:$WS" \
  --env PYTHONPATH="$REPO" \
  "$SIF" \
  bash -lc '
    set -euo pipefail
    cd "'"$REPO"'"
    python -m py_compile \
      superdec/superdec.py \
      superdec/loss/supervised_hungarian_loss.py \
      train/trainer.py
  '

echo "=== Start strict staged v1 radius 0.20, equal stage weights 5.0 ==="
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
      run_name="'"$RUN_NAME"'" \
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
      loss.match_w_assign=1.0 \
      loss.match_w_exist=0.0 \
      loss.surface_target=gt_surface \
      loss.surface_n_samples=128 \
      loss.surface_D_eta=0.05 \
      loss.surface_D_omega=0.05 \
      loss.w_sup_surface=0.0 \
      loss.w_sup_shape_param=0.0 \
      loss.use_staged_surface=true \
      loss.stage1_fixed_radius=0.20 \
      loss.w_stage1_surface=5.0 \
      loss.w_stage2_surface=5.0 \
      loss.w_stage3_surface=5.0 \
      loss.debug_match_every=0 \
      loss.debug_match_max_samples=4 \
      superdec.decoder.n_queries=4 \
      superdec.decoder.n_layers=3 \
      superdec.decoder.staged_params=true \
      superdec.decoder.dropout=0.0
  '

echo "=== Finished strict staged v1 radius 0.20 ==="
date

echo "=== Output files ==="
find "$OUT/$RUN_NAME" -maxdepth 3 -type f -print | sort || true

echo "=== Saved config staged lines ==="
grep -E "staged_params|use_staged_surface|stage1_fixed_radius|w_stage|w_sup_surface|w_sup_shape_param|n_queries|n_layers|dropout" \
  "$OUT/$RUN_NAME/config.yaml" || true

echo "=== Done ==="
date
