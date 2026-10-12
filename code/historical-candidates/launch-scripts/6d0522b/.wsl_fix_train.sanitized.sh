#!/bin/bash
set -e
source <WSL_DEV_HOME>/miniconda3/bin/activate edgetts

echo "--- syncing code ---"
rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    <WINDOWS_WORKSPACE>/PitchFlowNet/ <WSL_DEV_HOME>/PitchFlowNet/

echo "--- installing tensorboard ---"
pip install -q "tensorboard>=2.13,<3"

echo "--- launching training ---"
cd <WSL_DEV_HOME>/PitchFlowNet/src/python
python -m piper_train \
    --dataset-dir <WSL_DEV_HOME>/data_preprocessed \
    --batch-size 8 \
    --validation-split 0.05 \
    --num-test-examples 5 \
    --max_epochs 10000 \
    --checkpoint-epochs 1
