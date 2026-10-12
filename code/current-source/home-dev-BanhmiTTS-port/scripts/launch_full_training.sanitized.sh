#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:?usage: launch_full_training.sh OUTPUT_DIR [DATASET_DIR]}"
dataset_dir="${2:-${project_root}/preprocessed}"
python_bin="${BANHMI_PYTHON:-<WSL_DEV_HOME>/miniconda3/envs/banhmi-tts/bin/python}"
batch_size="${BANHMI_BATCH_SIZE:-16}"
epochs="${BANHMI_EPOCHS:-1100}"

if [[ -e "${output_dir}" ]]; then
    echo "output directory already exists: ${output_dir}" >&2
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

steps_per_epoch="$(
    env PYTHONPATH="${project_root}/src/python" "${python_bin}" -c \
        'import sys; from pathlib import Path; from banhmi_tts.data import BanhmiDataset, LengthBucketBatchSampler; dataset = BanhmiDataset(Path(sys.argv[1]) / "splits" / "train.jsonl"); sampler = LengthBucketBatchSampler(dataset.spectrogram_lengths, batch_size=int(sys.argv[2]), seed=1234, num_replicas=2, rank=0); print(len(sampler))' \
        "${dataset_dir}" "${batch_size}"
)"
steps="${BANHMI_STEPS:-$((epochs * steps_per_epoch))}"

mkdir -p "${output_dir}"

printf '%s\n' "$$" > "${output_dir}/launcher.pid"

echo "output_dir=${output_dir}"
echo "launcher_pid=$$"
echo "epochs=${epochs}"
echo "steps_per_epoch=${steps_per_epoch}"
echo "steps=${steps}"
echo "batch_size_per_gpu=${batch_size}"

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
        --generator-activation snake_beta \
        --duration-predictor stochastic \
        --edge-discriminator \
        > "${output_dir}/train.log" 2>&1
