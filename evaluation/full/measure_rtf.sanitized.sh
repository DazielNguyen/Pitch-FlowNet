#!/bin/bash
# Measure CPU real-time factor for a checkpoint.
# Usage: bash measure_rtf.sh [checkpoint_path]
set -e
source <WSL_DEV_HOME>/miniconda3/bin/activate edgetts

CKPT="${1:-<WSL_DEV_HOME>/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt}"
SENTENCES="<WSL_DEV_HOME>/test_sentences_500.txt"

echo "Checkpoint : $CKPT"
echo "Sentences  : $SENTENCES"

rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    <WINDOWS_WORKSPACE>/EdgeTTS/ <WSL_DEV_HOME>/EdgeTTS/

cd <WSL_DEV_HOME>/EdgeTTS/src/python

python -m piper_train.measure_rtf \
    --checkpoint "$CKPT" \
    --sentences-file "$SENTENCES" \
    --num-sentences 20 \
    --language vi
