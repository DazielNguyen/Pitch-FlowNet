#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:?usage: resume_full_training.sh OUTPUT_DIR CHECKPOINT [DATASET_DIR]}"
checkpoint="${2:?usage: resume_full_training.sh OUTPUT_DIR CHECKPOINT [DATASET_DIR]}"
dataset_dir="${3:-${project_root}/preprocessed}"
python_bin="${BANHMI_PYTHON:-<WSL_DEV_HOME>/miniconda3/envs/banhmi-tts/bin/python}"
batch_size="${BANHMI_BATCH_SIZE:-16}"
steps="${BANHMI_STEPS:-430100}"
steps_per_epoch="${BANHMI_STEPS_PER_EPOCH:-391}"

if [[ ! -d "${output_dir}" ]]; then
    echo "output directory not found: ${output_dir}" >&2
    exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "resume checkpoint not found: ${checkpoint}" >&2
    exit 2
fi
if [[ ! -f "${dataset_dir}/splits/train.jsonl" ]]; then
    echo "training manifest not found: ${dataset_dir}/splits/train.jsonl" >&2
    exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 2
fi

env PYTHONPATH="${project_root}/src/python" "${python_bin}" -c \
    'from banhmi_tts.model.alignment import compiled_maximum_path_available; raise SystemExit(0 if compiled_maximum_path_available() else "compiled MAS backend is unavailable; run scripts/build_monotonic_align.sh")'

printf '%s\n' "$$" > "${output_dir}/resume-launcher.pid"
{
    echo
    echo "===== RESUME $(date --iso-8601=seconds) ====="
    echo "resume_checkpoint=${checkpoint}"
    echo "target_steps=${steps}"
    echo "batch_size_per_gpu=${batch_size}"
} >> "${output_dir}/train.log"

exec env \
    PYTHONPATH="${project_root}/src/python" \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0,1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "${python_bin}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node=2 \
        -m banhmi_tts.training \
        --dataset-dir "${dataset_dir}" \
        --output-dir "${output_dir}" \
        --steps "${steps}" \
        --batch-size "${batch_size}" \
        --num-workers 8 \
        --checkpoint-every "${steps_per_epoch}" \
        --validation-every "${steps_per_epoch}" \
        --validation-batches 9 \
        --sample-every "${steps_per_epoch}" \
        --c-kl 1.0 \
        --generator-gradient-clip 1.0 \
        --discriminator-gradient-clip 1.0 \
        --generator-activation snake_beta \
        --duration-predictor stochastic \
        --edge-discriminator \
        --resume "${checkpoint}" \
        >> "${output_dir}/train.log" 2>&1
