#!/usr/bin/env bash
# Run the locked evaluation (docs/camera-ready/eval-protocol.md) for one or more checkpoints (inside WSL).
#
#   tools/camera_ready/launch_eval.sh [--split test|validation] [--threads 4] [--limit N] [--execute] LABEL=CHECKPOINT [...]
#
# Without --execute it checks the inputs and prints the commands (dry run). With --execute each LABEL runs in its
# own tmux session cr-eval-<LABEL>, at nice 10, with its log in $CR_OUTPUTS/logs/eval-<LABEL>-<UTC>.log and its
# results in $CR_OUTPUTS/eval/<LABEL>/. It refuses a missing checkpoint, an existing result folder or session, a
# missing evaluation environment, and the test split while eval-protocol.md is absent.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUTS=${CR_OUTPUTS:-$HOME/outputs}
PY=${CR_EVAL_PYTHON:-$HOME/miniforge3/envs/pfn-eval/bin/python}
SPLIT=test; THREADS=4; LIMIT=0; EXECUTE=0; JOBS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --split) SPLIT=$2; shift ;;
    --threads) THREADS=$2; shift ;;
    --limit) LIMIT=$2; shift ;;
    --execute) EXECUTE=1 ;;
    *=*) JOBS+=("$1") ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[ ${#JOBS[@]} -gt 0 ] || { sed -n '2,10p' "$0"; exit 2; }

problems=()
[ -x "$PY" ] || problems+=("evaluation python not found: $PY")
"$PY" -c "import whisper, jiwer, utmosv2" 2>/dev/null || problems+=("whisper/jiwer/utmosv2 not importable in $PY")
[ "$SPLIT" != test ] || [ -f "$REPO/docs/camera-ready/eval-protocol.md" ] || problems+=("test split needs docs/camera-ready/eval-protocol.md (locked protocol)")
for job in "${JOBS[@]}"; do
  label=${job%%=*}; ckpt=${job#*=}
  [ -f "$ckpt" ] || problems+=("$label: checkpoint not found: $ckpt")
  [ ! -e "$OUTPUTS/eval/$label" ] || problems+=("$label: result folder exists: $OUTPUTS/eval/$label")
  ! tmux has-session -t "cr-eval-$label" 2>/dev/null || problems+=("$label: tmux session cr-eval-$label exists")
done

for job in "${JOBS[@]}"; do
  label=${job%%=*}; ckpt=${job#*=}
  echo "$label: $ckpt  ->  $OUTPUTS/eval/$label  (split $SPLIT, threads $THREADS$([ "$LIMIT" != 0 ] && echo ", limit $LIMIT"))"
done
if [ ${#problems[@]} -gt 0 ]; then printf 'REFUSED:\n'; printf '  - %s\n' "${problems[@]}"; exit 1; fi
if [ "$EXECUTE" = 0 ]; then echo "DRY RUN: nothing started. Add --execute to run."; exit 0; fi

mkdir -p "$OUTPUTS/eval" "$OUTPUTS/logs"
for job in "${JOBS[@]}"; do
  label=${job%%=*}; ckpt=${job#*=}
  log="$OUTPUTS/logs/eval-$label-$(date -u +%Y%m%dT%H%M%SZ).log"
  cmd=(nice -n10 "$PY" "$REPO/tools/camera_ready/evaluate_like_history.py" evaluate --checkpoint "$ckpt" --label "$label"
       --output "$OUTPUTS/eval" --split "$SPLIT" --threads "$THREADS" --limit "$LIMIT")
  printf -v q '%q ' "${cmd[@]}"
  tmux new-session -d -s "cr-eval-$label" "cd $(printf %q "$REPO") && { echo \"# start \$(date -Is) $q\"; $q; echo \"# exit \$? \$(date -Is)\"; } > $(printf %q "$log") 2>&1"
  echo "STARTED $label in tmux cr-eval-$label; log $log"
done
