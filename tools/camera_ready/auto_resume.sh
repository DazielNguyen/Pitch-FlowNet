#!/usr/bin/env bash
# Watchdog: resume a camera-ready run automatically after the known CUDA "unknown error" (inside WSL).
#
#   tools/camera_ready/auto_resume.sh <config.json> [--gpu N] [--stop-after-epochs N] [--max-resumes 5] [--interval 120] [--once] [--dry-run]
#   tools/camera_ready/auto_resume.sh <config.json> --start [--gpu N] [--stop-after-epochs N] [--max-resumes 5]   # in tmux cr-watch-<run_id>
#
# --stop-after-epochs N is passed to every resume, for a run that must end at epoch N (e.g. PFN-Full-275ep ended at 200).
#
# Standing permission of the author (2026-10-21) for this one failure: on MardeusPC the display GPU resets
# (Windows LiveKernelEvent 141) and a run can die with "CUDA error: unknown error". Every <interval> seconds:
#   - the run's tmux session cr-<run_id> is alive               -> nothing to do;
#   - its newest launch log ends with "# exit 0"                -> the run is finished: finalize_run.sh archives its best and
#                                                                  last checkpoints (read-only copies); the watchdog stops;
#   - it ends with a non-zero exit and the CUDA unknown error   -> wait 60 s, check CUDA with a small matmul,
#     then resume_train.sh --execute (which checks last.ckpt); at most --max-resumes times in total;
#   - anything else (another error, a damaged last.ckpt, a failed CUDA check, the limit reached)
#                                                                -> stop and leave it to the author.
# Each decision is written to $CR_OUTPUTS/logs/watchdog-<run_id>.log; each automatic resume is also appended to
# the run's checkpoint_ledger.jsonl. It never uses --fallback, never restarts from scratch, never starts a 2nd copy.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUTPUTS=${CR_OUTPUTS:-$HOME/outputs}
PY=${CR_PYTHON:-$HOME/miniforge3/envs/pfn-cr/bin/python}

[ $# -ge 1 ] || { sed -n '2,21p' "$0"; exit 2; }
CONFIG=$(realpath "$1"); shift
GPU=0; MAX=5; INTERVAL=120; ONCE=0; DRY=0; START=0; STOPAT=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=$2; shift ;;
    --stop-after-epochs) STOPAT=(--stop-after-epochs "$2"); shift ;;
    --max-resumes) MAX=$2; shift ;;
    --interval) INTERVAL=$2; shift ;;
    --once) ONCE=1 ;;
    --dry-run) DRY=1 ;;
    --start) START=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$CONFIG")
RUN_DIR="$OUTPUTS/runs/$RUN_ID"
SESSION="cr-$RUN_ID"
WLOG="$OUTPUTS/logs/watchdog-$RUN_ID.log"
mkdir -p "$OUTPUTS/logs"

if [ "$START" = 1 ]; then
  W="cr-watch-$RUN_ID"
  if tmux has-session -t "$W" 2>/dev/null; then echo "already running: tmux session $W (log $WLOG)"; exit 0; fi
  tmux new-session -d -s "$W" "bash $(printf %q "$0") $(printf %q "$CONFIG") --gpu $GPU ${STOPAT[*]} --max-resumes $MAX --interval $INTERVAL"
  echo "started: tmux session $W, log $WLOG"
  exit 0
fi

say() { echo "$(date -Is) $*" | tee -a "$WLOG"; }
count_resumes() { grep -c '"event": "auto_resume"' "$RUN_DIR/checkpoint_ledger.jsonl" 2>/dev/null || true; }

say "watchdog for $RUN_ID on GPU $GPU${STOPAT:+ (${STOPAT[*]})}: max $MAX automatic resumes, every ${INTERVAL}s$([ $DRY = 1 ] && echo ', DRY RUN')"
while true; do
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    [ "$ONCE" = 1 ] && { say "run alive (tmux $SESSION)"; exit 0; }
    sleep "$INTERVAL"; continue
  fi
  LOG=$(ls -t "$OUTPUTS/logs/$RUN_ID-"*.log 2>/dev/null | head -1)
  [ -n "$LOG" ] || { say "STOP: no launch log for $RUN_ID"; exit 1; }
  EXIT=$(grep -ao '# exit [0-9]*' "$LOG" | tail -1 | awk '{print $3}')
  if [ "$EXIT" = 0 ]; then
    say "run finished (exit 0 in $(basename "$LOG")); archiving its best and last checkpoints"
    EXP=(); [ ${#STOPAT[@]} -gt 0 ] && EXP=(--expect-epochs "${STOPAT[1]}")
    if [ "$DRY" = 1 ]; then say "DRY RUN: would run finalize_run.sh $(basename "$CONFIG") ${EXP[*]}"; exit 0; fi
    if bash "$HERE/finalize_run.sh" "$CONFIG" "${EXP[@]}" >>"$WLOG" 2>&1; then say "finalized: $OUTPUTS/final/$RUN_ID; watchdog stops"
    else say "STOP: finalize_run.sh refused (see above); author needed"; exit 1; fi
    exit 0
  fi
  if [ -z "$EXIT" ]; then say "STOP: $(basename "$LOG") has no exit line and the session is gone (WSL restart?); author needed"; exit 1; fi
  if ! tail -c 20000 "$LOG" | grep -aq "CUDA error: unknown error"; then
    say "STOP: exit $EXIT without the CUDA unknown error; author needed. Last error: $(tr '\r' '\n' < "$LOG" | grep -aE 'Error' | tail -1 | cut -c1-160)"
    exit 1
  fi
  N=$(count_resumes)
  if [ "$N" -ge "$MAX" ]; then say "STOP: $N automatic resumes already (limit $MAX); author needed"; exit 1; fi
  say "run died with the CUDA unknown error (exit $EXIT, $(basename "$LOG")); automatic resume $((N + 1)) of $MAX after 60 s"
  [ "$DRY" = 1 ] || sleep 60
  if ! "$PY" -c "import torch; x=torch.randn(2048,2048,device='cuda'); assert torch.isfinite((x@x).sum()).item()" >>"$WLOG" 2>&1; then
    say "STOP: CUDA check failed after the error; author needed"; exit 1
  fi
  if [ "$DRY" = 1 ]; then
    say "DRY RUN: would run resume_train.sh $(basename "$CONFIG") --gpu $GPU --execute; its dry run says:"
    bash "$HERE/resume_train.sh" "$CONFIG" --gpu "$GPU" "${STOPAT[@]}" 2>&1 | grep -v '^command:' | tee -a "$WLOG"
    exit 0
  fi
  echo "{\"time_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"event\": \"auto_resume\", \"n\": $((N + 1)), \"after_log\": \"$(basename "$LOG")\", \"reason\": \"CUDA error: unknown error\"}" >> "$RUN_DIR/checkpoint_ledger.jsonl"
  if bash "$HERE/resume_train.sh" "$CONFIG" --gpu "$GPU" "${STOPAT[@]}" --execute >>"$WLOG" 2>&1; then
    say "resumed (see resume_train.sh output above)"
  else
    say "STOP: resume_train.sh refused (damaged last.ckpt or other guard); author needed"; exit 1
  fi
  [ "$ONCE" = 1 ] && exit 0
  sleep "$INTERVAL"
done
