#!/bin/bash
source <WSL_DEV_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate edgetts
mkdir -p <WSL_DEV_HOME>/bigvgan_run

cd <WSL_DEV_HOME>/EdgeTTS/src/python
python3 -m piper_train   --dataset-dir <WSL_DEV_HOME>/data_preprocessed   --accelerator gpu   --devices 2   --strategy ddp_find_unused_parameters_true   --precision bf16-mixed   --batch-size 16   --validation-split 0.05   --default_root_dir <WSL_DEV_HOME>/bigvgan_run   --checkpoint-epochs 1   --use-snake true   --use-mrd true   --use-dur-disc false   --use-transformer-flows false   --use-f0 false   > <WSL_DEV_HOME>/bigvgan_run/train.log 2>&1
