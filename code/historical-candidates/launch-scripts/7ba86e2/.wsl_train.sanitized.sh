#!/bin/bash
set -e
source <WSL_DEV_HOME>/miniconda3/bin/activate edgetts

echo "--- syncing code ---"
rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    <WINDOWS_WORKSPACE>/EdgeTTS/ <WSL_DEV_HOME>/EdgeTTS/

cd <WSL_DEV_HOME>/EdgeTTS/src/python

python -m piper_train \
    --dataset-dir <WSL_DEV_HOME>/data_preprocessed \
    --batch-size 16 \
    --validation-split 0.05 \
    --num-test-examples 5 \
    --max_epochs 10000 \
    --checkpoint-epochs 1 \
    --resume_from_checkpoint <WSL_DEV_HOME>/data_preprocessed/lightning_logs/version_8/checkpoints/last.ckpt
