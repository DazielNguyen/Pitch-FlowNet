#!/bin/bash
set -e
source <WSL_DEV_HOME>/miniconda3/bin/activate edgetts
REPO=<WSL_DEV_HOME>/EdgeTTS

echo "--- syncing code from Windows ---"
rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    <WINDOWS_WORKSPACE>/EdgeTTS/ "$REPO/"

echo "--- syncing dataset from Windows (this can take a while, 3.6GB) ---"
mkdir -p <WSL_DEV_HOME>/data
rsync -a --info=progress2 \
    <WINDOWS_WORKSPACE>/EdgeTTS/data/ <WSL_DEV_HOME>/data/

echo "--- running preprocess ---"
cd "$REPO/src/python"
python -m piper_train.preprocess \
    --input-dir <WSL_DEV_HOME>/data \
    --output-dir <WSL_DEV_HOME>/data_preprocessed \
    --language en-us \
    --sample-rate 22050 \
    --dataset-format ljspeech \
    --single-speaker \
    --max-workers 16

echo "--- done ---"
ls -la <WSL_DEV_HOME>/data_preprocessed
wc -l <WSL_DEV_HOME>/data_preprocessed/dataset.jsonl
