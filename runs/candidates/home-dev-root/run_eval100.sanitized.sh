#!/bin/bash
source <WSL_DEV_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate edgetts
mkdir -p <WSL_DEV_HOME>/eval_100samples
cd <WSL_DEV_HOME>/EdgeTTS/src/python
CUDA_VISIBLE_DEVICES='' python3 -m piper_train.eval_harness   --checkpoint <WSL_DEV_HOME>/baseline_run/lightning_logs/version_2/checkpoints/best-epoch=861-val_loss_mel=20.1489.ckpt   --checkpoint <WSL_DEV_HOME>/data_preprocessed/lightning_logs/version_12/checkpoints/best-epoch=863-val_loss_mel=19.5701.ckpt   --dataset-jsonl <WSL_DEV_HOME>/data_preprocessed/dataset.jsonl   --num-dataset-samples 100   --language en-us   --save-wavs   --output <WSL_DEV_HOME>/eval_100samples   > <WSL_DEV_HOME>/eval_100samples/run.log 2>&1
echo 'DONE' >> <WSL_DEV_HOME>/eval_100samples/run.log
