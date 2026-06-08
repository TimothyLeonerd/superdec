#!/bin/bash -l
#SBATCH --job-name=sdec-nsp-w1
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
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
export RUN_NAME="train_sqzero_suphung_gtsurf_normals_spatial_signed_w1_150ep_001"

mkdir -p "$WS/logs" "$OUT"
cd "$REPO"

echo "=== Checks ==="
hostname
date
echo "RUN_NAME=$RUN_NAME"
grep -Rni "normal_mode == \"sidecar\"" superdec/data/sqzero_lmdb.py
grep -Rni "w_sup_normal\|w_sup_normal_dir\|_compute_part_normal_loss\|_compute_normal_direction_loss" \
  superdec/loss/supervised_hungarian_loss.py configs/train.yaml || true
grep -Rni "normal_gate\|sup_normal_gate\|gate_sum" \
  superdec/loss/supervised_hungarian_loss.py configs/train.yaml && exit 1 || true
grep -Rni "optimizer.step" train/trainer.py

apptainer exec \
  --bind "$WS:$WS" \
  --env PYTHONPATH="$REPO" \
  "$SIF" \
  python -m py_compile \
    "$REPO/superdec/data/sqzero_lmdb.py" \
    "$REPO/superdec/loss/supervised_hungarian_loss.py" \
    "$REPO/train/trainer.py"

echo "=== Train: gtsurf_w10 + spatial signed normal w1 ==="
apptainer exec --nv \
  --bind "$WS:$WS" \
  --env PYTHONPATH="$REPO" \
  "$SIF" \
  bash -lc '
    set -euo pipefail
    cd "'"$REPO"'"
    export PYTHONUNBUFFERED=1
    export HYDRA_FULL_ERROR=1
    export WANDB_MODE=disabled

    python train/train.py \
      run_name="'"$RUN_NAME"'" \
      dataset=sqzero_lmdb \
      sqzero_lmdb.path="'"$DATA"'" \
      sqzero_lmdb.train_split=train.txt \
      sqzero_lmdb.val_split=test.txt \
      sqzero_lmdb.n_points=4096 \
      sqzero_lmdb.normalize=true \
      sqzero_lmdb.normal_mode=sidecar \
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
      loss.w_sup_normal=1.0 \
      loss.w_sup_normal_dir=0.0 \
      loss.normal_unsigned=false \
      loss.normal_max_points_per_part=512 \
      loss.normal_eps=1.0e-8 \
      loss.surface_target=gt_surface \
      loss.surface_n_samples=128 \
      loss.surface_D_eta=0.05 \
      loss.surface_D_omega=0.05 \
      loss.match_w_assign=1.0 \
      loss.match_w_exist=0.0 \
      loss.w_sup_shape_param=0.0 \
      loss.use_staged_surface=false \
      loss.debug_match_every=0 \
      superdec.decoder.n_queries=4 \
      superdec.decoder.n_layers=3 \
      superdec.decoder.staged_params=false \
      superdec.decoder.dropout=0.0
  '

echo "=== Done ==="
date
find "$OUT/$RUN_NAME" -maxdepth 2 -type f -print | sort || true
