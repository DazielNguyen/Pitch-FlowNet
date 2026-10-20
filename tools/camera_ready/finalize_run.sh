#!/usr/bin/env bash
# Archive the best and the last checkpoint of a finished camera-ready run (inside WSL).
#
#   tools/camera_ready/finalize_run.sh <config.json> [--expect-epochs N]
#
# For the run of <config.json> it:
#   1. checks that the newest attempt in run_manifest.json is "completed" (and, with --expect-epochs N, that it
#      completed N epochs), and that best-*.ckpt and last.ckpt pass check_ckpt.py;
#   2. copies them (real copies, not hard links: Lightning rewrites last.ckpt in place) to
#      $CR_OUTPUTS/final/<run_id>/, the best one under its own name (best-epoch=...ckpt) and the last one as
#      last-epoch=<E>.ckpt (E = 0-based index of the last epoch: a 200-epoch run ends with last-epoch=199.ckpt);
#   3. writes SHA256SUMS and final.json (run, commit, epochs, G updates, best score, hashes) and makes everything
#      read-only.
# Read-only on the run directory. Refuses if the final folder already exists.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUTPUTS=${CR_OUTPUTS:-$HOME/outputs}
PY=${CR_PYTHON:-$HOME/miniforge3/envs/pfn-cr/bin/python}

[ $# -ge 1 ] || { sed -n '2,14p' "$0"; exit 2; }
CONFIG=$(realpath "$1"); shift
EXPECT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --expect-epochs) EXPECT=$2; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$CONFIG")
RUN_DIR="$OUTPUTS/runs/$RUN_ID"; CK="$RUN_DIR/checkpoints"; DEST="$OUTPUTS/final/$RUN_ID"
[ ! -e "$DEST" ] || { echo "REFUSED: $DEST exists (already finalized)"; exit 1; }
[ -f "$RUN_DIR/run_manifest.json" ] || { echo "REFUSED: no run_manifest.json in $RUN_DIR"; exit 1; }

"$PY" - "$RUN_DIR/run_manifest.json" "$EXPECT" <<'P' || exit 1
import json, sys
m = json.load(open(sys.argv[1])); a = m["attempts"][-1]; want = sys.argv[2]
if a.get("status") != "completed":
    sys.exit(f"REFUSED: newest attempt status is {a.get('status')!r}, not 'completed'")
if want and int(a.get("epochs_completed", -1)) != int(want):
    sys.exit(f"REFUSED: newest attempt completed {a.get('epochs_completed')} epochs, expected {want}")
print(f"manifest: completed, {a.get('epochs_completed')} epochs, {a.get('g_updates')} G updates, best {a.get('best_checkpoint')}")
P

BEST=$(ls "$CK"/best-*.ckpt 2>/dev/null)
[ "$(echo "$BEST" | grep -c .)" = 1 ] || { echo "REFUSED: expected exactly one best-*.ckpt in $CK"; exit 1; }
"$PY" "$HERE/check_ckpt.py" "$BEST" "$CK/last.ckpt" || { echo "REFUSED: a checkpoint failed check_ckpt.py"; exit 1; }
LAST_EPOCH=$("$PY" -c 'import sys,torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["epoch"])' "$CK/last.ckpt")

mkdir -p "$DEST"
cp --preserve=timestamps "$BEST" "$DEST/$(basename "$BEST")"
cp --preserve=timestamps "$CK/last.ckpt" "$DEST/last-epoch=$LAST_EPOCH.ckpt"
cp --preserve=timestamps "$RUN_DIR/run_manifest.json" "$RUN_DIR/checkpoint_ledger.jsonl" "$RUN_DIR/val_diagnostics.jsonl" "$DEST/"
(cd "$DEST" && sha256sum best-*.ckpt last-epoch=*.ckpt run_manifest.json checkpoint_ledger.jsonl val_diagnostics.jsonl > SHA256SUMS)
# the copies must equal the sources byte for byte
for f in "$BEST" "$CK/last.ckpt"; do
  src=$(sha256sum "$f" | cut -d' ' -f1)
  grep -q "^$src " "$DEST/SHA256SUMS" || { echo "ABORT: copy of $f does not match its source"; exit 1; }
done
"$PY" - "$DEST" "$RUN_DIR/run_manifest.json" "$CONFIG" <<'P'
import json, sys, pathlib, datetime
dest = pathlib.Path(sys.argv[1]); m = json.load(open(sys.argv[2])); a = m["attempts"][-1]
sums = dict(reversed(l.split("  ", 1)) for l in (dest / "SHA256SUMS").read_text().splitlines())
doc = {"run_id": m["run_id"], "config": sys.argv[3], "config_sha256": m["config_sha256"],
       "source_commit": m["source"]["git"]["head"], "source_tree_sha256": m["source"]["tree_sha256"],
       "finalized_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "attempts": len(m["attempts"]), "epochs_completed": a.get("epochs_completed"), "g_updates": a.get("g_updates"),
       "samples_seen_global": a.get("samples_seen_global"), "best_checkpoint": a.get("best_checkpoint"),
       "best_val_mel_l1_global": a.get("best_score"), "files_sha256": {k.strip(): v for k, v in sums.items()}}
(dest / "final.json").write_text(json.dumps(doc, indent=1) + "\n")
print(json.dumps({k: doc[k] for k in ("run_id", "epochs_completed", "g_updates", "best_checkpoint", "best_val_mel_l1_global")}))
P
chmod a-w "$DEST"/* && chmod a-w "$DEST"
echo "FINALIZED: $DEST (read-only)"; ls -la "$DEST"
