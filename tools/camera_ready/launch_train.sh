#!/usr/bin/env bash
# Launch one camera-ready training run on the single GPU of the training machine (inside WSL).
#
#   tools/camera_ready/launch_train.sh <config.json> [--resume] [--limit-train-batches N] [--stop-after-epochs N] [--execute]
#
# Without --execute it only checks the inputs and prints the command (dry run).
# With --execute it starts the run in its own tmux session cr-<run_id> and writes the log to
# $CR_OUTPUTS/logs/<run_id>-<UTC time>.log.
#
# It refuses to start when:
#   - the config, the dataset files or the split manifest are missing;
#   - the config still carries a placeholder budget (protocol.budget_label starts with "PLACEHOLDER");
#   - the run directory exists (fresh start) or lacks last.ckpt (--resume);
#   - another camera-ready training process or the tmux session cr-<run_id> already exists;
#   - the GPU is busy (memory or utilisation above the thresholds below).
#
# Environment (defaults in brackets):
#   CR_DATASET_DIR [$HOME/data_preprocessed]   CR_OUTPUTS [$HOME/outputs]
#   CR_PYTHON [$HOME/miniforge3/envs/pfn-cr/bin/python]
#   CR_GPU_MAX_USED_MIB [3000]   CR_GPU_MAX_UTIL [10]   (the display alone uses about 1.2 GB)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC="$REPO/code/camera-ready-source/src/python"
SPLIT="$REPO/manifests/camera-ready/split-manifest.json"
DATASET_DIR=${CR_DATASET_DIR:-$HOME/data_preprocessed}
OUTPUTS=${CR_OUTPUTS:-$HOME/outputs}
PY=${CR_PYTHON:-$HOME/miniforge3/envs/pfn-cr/bin/python}
MAX_USED=${CR_GPU_MAX_USED_MIB:-3000}
MAX_UTIL=${CR_GPU_MAX_UTIL:-10}

[ $# -ge 1 ] || { sed -n '2,20p' "$0"; exit 2; }
CONFIG=$(realpath "$1"); shift
EXECUTE=0; RESUME=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1 ;;
    --resume) RESUME=1; EXTRA+=(--resume) ;;
    --limit-train-batches|--stop-after-epochs) EXTRA+=("$1" "$2"); shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

problems=()
[ -f "$CONFIG" ] || problems+=("config not found: $CONFIG")
for f in dataset.jsonl config.json; do [ -f "$DATASET_DIR/$f" ] || problems+=("missing $DATASET_DIR/$f"); done
[ -f "$SPLIT" ] || problems+=("missing split manifest $SPLIT")
[ -x "$PY" ] || problems+=("python not found: $PY")

RUN_ID=""
if [ -f "$CONFIG" ]; then
  RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$CONFIG")
  LABEL=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol"].get("budget_label",""))' "$CONFIG")
  case "$LABEL" in PLACEHOLDER*) problems+=("placeholder budget in $CONFIG: decide max_g_updates first (T08)") ;; esac
fi
RUN_DIR="$OUTPUTS/runs/$RUN_ID"
SESSION="cr-$RUN_ID"

if [ -n "$RUN_ID" ]; then
  if [ "$RESUME" = 1 ]; then
    [ -f "$RUN_DIR/checkpoints/last.ckpt" ] || problems+=("--resume needs $RUN_DIR/checkpoints/last.ckpt")
  else
    [ ! -e "$RUN_DIR" ] || problems+=("run directory exists: $RUN_DIR (use --resume for the same run)")
  fi
  ! tmux has-session -t "$SESSION" 2>/dev/null || problems+=("tmux session $SESSION already exists")
fi
if pgrep -f "piper_train.camera_ready.train" >/dev/null; then
  problems+=("another camera-ready training process is running: $(pgrep -af piper_train.camera_ready.train | head -1)")
fi

GPU_STATE=$(nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
if [ -z "$GPU_STATE" ]; then
  problems+=("nvidia-smi gave no GPU")
else
  read -r used util < <(echo "$GPU_STATE" | head -1 | awk -F', ' '{print $3, $4}')
  if [ "$used" -gt "$MAX_USED" ] || [ "$util" -gt "$MAX_UTIL" ]; then
    problems+=("GPU busy: ${used} MiB used, ${util} % utilisation (limits ${MAX_USED} MiB, ${MAX_UTIL} %)")
  fi
fi

CMD=(env CUDA_VISIBLE_DEVICES=0 "$PY" -m piper_train.camera_ready.train
     --config "$CONFIG" --dataset-dir "$DATASET_DIR" --split-manifest "$SPLIT" --run-dir "$RUN_DIR" "${EXTRA[@]}")

echo "run_id:   $RUN_ID"
echo "run dir:  $RUN_DIR"
echo "gpu:      $GPU_STATE"
echo "command:  (cd $SRC && ${CMD[*]})"
if [ ${#problems[@]} -gt 0 ]; then
  printf 'REFUSED:\n'; printf '  - %s\n' "${problems[@]}"; exit 1
fi

# The training entry point validates the inputs once more (hashes, split, config keys) without creating anything.
(cd "$SRC" && "${CMD[@]}" --dry-run >/dev/null) || { echo "REFUSED: train.py --dry-run failed"; exit 1; }
echo "inputs:   ok (train.py --dry-run)"

if [ "$EXECUTE" = 0 ]; then
  echo "DRY RUN: nothing started. Add --execute to launch in tmux session $SESSION."
  exit 0
fi

mkdir -p "$OUTPUTS/logs"
LOG="$OUTPUTS/logs/$RUN_ID-$(date -u +%Y%m%dT%H%M%SZ).log"
printf -v QCMD '%q ' "${CMD[@]}"
tmux new-session -d -s "$SESSION" "cd $(printf %q "$SRC") && { echo \"# start \$(date -Is) $QCMD\"; $QCMD; echo \"# exit \$? \$(date -Is)\"; } > $(printf %q "$LOG") 2>&1"
echo "STARTED in tmux session $SESSION; log: $LOG"
