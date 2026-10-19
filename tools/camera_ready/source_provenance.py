"""Hash the workspace source tree and compare it with the record of the historical candidate.

    python tools/camera_ready/source_provenance.py --stage pre|post --output <json>

Run from the repository root. Read-only except for the output file.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

CANDIDATE = "code/historical-candidates/edgetts-badf381/src/python/"
WORK = Path("code/camera-ready-source/src/python")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pre", "post"], required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    core = {e["destination"]: e for e in json.loads(Path("manifests/file-manifest-core.json").read_text())["entries"]}
    rows, differs, missing_in_record = [], [], []
    for p in sorted(x for x in WORK.rglob("*") if x.is_file() and "__pycache__" not in x.parts and x.suffix not in (".so", ".c", ".pyc")
                    and "build" not in x.parts and ".egg-info" not in str(x)):
        rel = p.relative_to(WORK).as_posix()
        rec = core.get(CANDIDATE + rel)
        h = sha(p)
        row = {"path": rel, "sha256": h, "record_sha256": rec["destination_sha256"] if rec else None,
               "record_origin": rec["origin"] if rec else None}
        row["equals_record"] = (h == row["record_sha256"]) if rec else None
        if rec is None:
            missing_in_record.append(rel)
        elif h != rec["destination_sha256"]:
            differs.append(rel)
        rows.append(row)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"stage": a.stage, "repo_head": head, "candidate": "EdgeTTS commit badf381 (copied from " + CANDIDATE + ")",
           "excluded_from_copy": ["piper_train/extract_test_sentences.py (contains machine paths; not used by training)",
                                  "piper_train/vits/monotonic_align/core.c (Cython output; rebuilt from core.pyx)",
                                  "piper_train/norm_audio/models/silero_vad.onnx (preprocessing only)", "*.so, __pycache__"],
           "files": len(rows), "equal_to_record": sum(1 for r in rows if r["equals_record"]),
           "differ_from_record": differs, "not_in_record": missing_in_record, "rows": rows}
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("stage", "files", "equal_to_record", "differ_from_record", "not_in_record")}, indent=1))


if __name__ == "__main__":
    main()
