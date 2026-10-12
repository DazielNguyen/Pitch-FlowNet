#!/bin/bash
set -e
source <WSL_DEV_HOME>/miniconda3/bin/activate edgetts

echo "--- syncing code ---"
rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    <WINDOWS_WORKSPACE>/PitchFlowNet/ <WSL_DEV_HOME>/PitchFlowNet/

cd <WSL_DEV_HOME>/PitchFlowNet/src/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m piper_train \
    --dataset-dir <WSL_DEV_HOME>/data_preprocessed \
    --default_root_dir <WSL_DEV_HOME>/01_Baseline_BigVgan_VITS2_FO \
    --accelerator gpu \
    --devices 2 \
    --strategy ddp_find_unused_parameters_true \
    --precision bf16-mixed \
    --batch-size 16 \
    --num-val-examples 100 \
    --num-test-examples 500 \
    --max_epochs 1100 \
    --checkpoint-epochs 1 \
    --seed 1234 \
    --use-bigvgan true \
    --use-vits2 true \
    --use-f0 true \
    --grad-clip 1.0 \
    --resume_from_checkpoint <WSL_DEV_HOME>/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_1/checkpoints/last.ckpt
