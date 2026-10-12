#!/bin/bash
cd <WSL_DEV_HOME>/EdgeTTS/src/python
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

CONFIG_E="<WSL_DEV_HOME>/06_Config_E_BigVGAN_VITS2/lightning_logs/version_1/checkpoints/best-epoch=434-val_loss_mel=19.9303.ckpt"
CONFIG_H="<WSL_DEV_HOME>/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt"

setsid <WSL_DEV_HOME>/miniconda3/envs/edgetts/bin/python -m piper_train.eval_harness \
    --checkpoint "$CONFIG_E" \
    --checkpoint "$CONFIG_H" \
    --dataset-jsonl <WSL_DEV_HOME>/data_preprocessed/dataset.jsonl \
    --num-dataset-samples 500 \
    --seed 42 \
    --whisper-model base.en \
    --output <WSL_DEV_HOME>/eval_f0_ablation_E_vs_H \
    < /dev/null > <WSL_DEV_HOME>/eval_f0_ablation_E_vs_H.log 2>&1 &
BGPID=$!
echo "bg_pid: $BGPID" > <WSL_DEV_HOME>/eval_f0_ablation_diag.txt
disown -a
sleep 10
{
  ps -p "$BGPID" -o pid,stat,cmd 2>&1
  echo "---log tail---"
  tail -30 <WSL_DEV_HOME>/eval_f0_ablation_E_vs_H.log
} >> <WSL_DEV_HOME>/eval_f0_ablation_diag.txt 2>&1
