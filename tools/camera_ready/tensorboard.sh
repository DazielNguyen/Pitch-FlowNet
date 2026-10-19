#!/usr/bin/env bash
# Start TensorBoard for all camera-ready runs in its own tmux session (inside WSL); idempotent.
#
#   tools/camera_ready/tensorboard.sh [start|stop|status]
#
# Serves $CR_OUTPUTS/runs on 127.0.0.1:$CR_TB_PORT (default 6006). WSL forwards localhost, so the
# Windows browser opens it at http://localhost:6006. TensorBoard only reads the event files.
set -euo pipefail
OUTPUTS=${CR_OUTPUTS:-$HOME/outputs}
PORT=${CR_TB_PORT:-6006}
TB=${CR_TENSORBOARD:-$HOME/miniforge3/envs/pfn-cr/bin/tensorboard}
SESSION=cr-tensorboard

case "${1:-start}" in
  start)
    if tmux has-session -t "$SESSION" 2>/dev/null; then echo "already running: tmux session $SESSION, http://localhost:$PORT"; exit 0; fi
    mkdir -p "$OUTPUTS/runs" "$OUTPUTS/logs"
    tmux new-session -d -s "$SESSION" \
      "nice -n10 $TB --logdir $OUTPUTS/runs --host 127.0.0.1 --port $PORT --reload_interval 60 > $OUTPUTS/logs/tensorboard.log 2>&1"
    echo "started: tmux session $SESSION, http://localhost:$PORT (log $OUTPUTS/logs/tensorboard.log)" ;;
  stop)
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped" || echo "not running" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "running: http://localhost:$PORT" || echo "not running" ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac
