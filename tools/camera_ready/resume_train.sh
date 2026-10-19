#!/usr/bin/env bash
# Resume an interrupted camera-ready run after a crash, a reboot or a power loss (inside WSL).
#
#   tools/camera_ready/resume_train.sh <config.json> [--fallback] [--execute]
#
# 1. Refuses while the run is still alive (tmux session cr-<run_id> or a training process).
# 2. Checks <run dir>/checkpoints/last.ckpt with check_ckpt.py (zip CRC, safe load, counters, finite).
# 3. If last.ckpt is damaged (the machine stopped while it was being written), it reports the newest
#    epoch-*.ckpt. That file is written just before last.ckpt, so one of the two is always complete.
#    With --fallback, the damaged last.ckpt is kept as last.ckpt.damaged-<UTC time>, and the epoch
#    file is copied to last.ckpt.
# 4. Hands over to launch_train.sh --resume: a dry run, or the real resume with --execute.
#
# Never starts a run from scratch and never runs two copies of a run.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUTPUTS=${CR_OUTPUTS:-$HOME/outputs}
PY=${CR_PYTHON:-$HOME/miniforge3/envs/pfn-cr/bin/python}

[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 2; }
CONFIG=$(realpath "$1"); shift
FALLBACK=0; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --fallback) FALLBACK=1 ;;
    --execute) PASS+=(--execute) ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$CONFIG")
RUN_DIR="$OUTPUTS/runs/$RUN_ID"
CK="$RUN_DIR/checkpoints"
[ -d "$CK" ] || { echo "REFUSED: no checkpoint directory $CK (nothing to resume; a fresh start uses launch_train.sh)"; exit 1; }
if tmux has-session -t "cr-$RUN_ID" 2>/dev/null; then
  echo "REFUSED: tmux session cr-$RUN_ID still exists; the run may still be alive (tmux attach -t cr-$RUN_ID)"; exit 1
fi
if pgrep -f "piper_train.camera_ready.train.*$RUN_ID" >/dev/null; then
  echo "REFUSED: a training process for $RUN_ID is still running"; exit 1
fi

echo "checking $CK/last.ckpt"
if "$PY" "$HERE/check_ckpt.py" "$CK/last.ckpt"; then
  :
else
  NEWEST=$(ls -t "$CK"/epoch-*.ckpt 2>/dev/null | head -1 || true)
  [ -n "$NEWEST" ] || { echo "REFUSED: last.ckpt is damaged and no epoch-*.ckpt exists"; exit 1; }
  echo "last.ckpt is damaged; newest epoch checkpoint:"
  "$PY" "$HERE/check_ckpt.py" "$NEWEST" || { echo "REFUSED: $NEWEST is damaged too"; exit 1; }
  if [ "$FALLBACK" = 0 ]; then
    echo "REFUSED: rerun with --fallback to keep the damaged file aside and resume from $(basename "$NEWEST")"; exit 1
  fi
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  [ -f "$CK/last.ckpt" ] && mv "$CK/last.ckpt" "$CK/last.ckpt.damaged-$STAMP"
  cp --preserve=timestamps "$NEWEST" "$CK/last.ckpt"
  echo "{\"time_utc\": \"$STAMP\", \"event\": \"resume_fallback\", \"from\": \"$(basename "$NEWEST")\", \"damaged\": \"last.ckpt.damaged-$STAMP\"}" >> "$RUN_DIR/checkpoint_ledger.jsonl"
  echo "last.ckpt replaced by $(basename "$NEWEST"); recorded in checkpoint_ledger.jsonl"
fi

exec bash "$HERE/launch_train.sh" "$CONFIG" --resume "${PASS[@]}"
