"""Verify the Pitch-FlowNet folder from its own files (standard library only; nothing is run).

    python tools/verify_pitchflownet.py [--skip-hashes] [--output manifests/verification-report.json]

Checks copy integrity (destination hashes against manifests/file-manifest.json, source hash equal to
destination hash), dataset completeness (rows, wavs, cache tensors referenced by every row, duplicate
IDs), the reconstructed split (sizes, disjointness, agreement with the stored test entries), evaluation
alignment (row counts and sentence order against the stored 500 test sentences, stored means) and
scans text files for credentials and machine paths. A copy hash match is not runtime identity.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRED = re.compile(r"(hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
PATHS = re.compile(r"(/home/" + "dev/|/home/" + "capstone/|/mnt/c/" + "Users/|C:[\\\\/]+" + "Users[\\\\/])")
TEXT = {".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".sh", ".log", ".csv", ".tex", ".bib", ".patch", ".cfg", ".toml"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-hashes", action="store_true")
    ap.add_argument("--output")
    a = ap.parse_args()
    checks = []

    def check(name, ok, **detail):
        checks.append({"check": name, "passed": bool(ok), **detail})

    m = json.loads((ROOT / "manifests/file-manifest.json").read_text(encoding="utf-8"))["entries"]
    missing = [e["destination"] for e in m if not (ROOT / e["destination"]).is_file()]
    check("every manifest entry exists", not missing, entries=len(m), missing=missing[:20])
    if not a.skip_hashes:
        bad = [e["destination"] for e in m if (ROOT / e["destination"]).is_file() and sha(ROOT / e["destination"]) != e["destination_sha256"]]
        check("destination hashes equal the manifest", not bad, mismatched=bad[:20])
    differ = [e["destination"] for e in m if e["source_sha256"] is not None and e["source_sha256"] != e["destination_sha256"]]
    check("copy integrity: source SHA-256 equals destination SHA-256", not differ, differing=differ[:20],
          not_applicable=[e["destination"] for e in m if e["source_sha256"] is None])

    # dataset
    ds = ROOT / "datasets/processed/data_preprocessed/dataset.jsonl"
    rows = [json.loads(l) for l in open(ds, encoding="utf-8")]
    remap = lambda s: ROOT / s.replace("/home/" + "dev/data_preprocessed/", "datasets/processed/data_preprocessed/").replace("/home/" + "dev/data/", "datasets/raw/home-dev-data/")
    path_keys = sorted({k for r in rows for k, v in r.items() if k.endswith("_path") and isinstance(v, str)})
    absent = {k: sum(1 for r in rows if r.get(k) and not remap(r[k]).is_file()) for k in path_keys}
    ids = [Path(r["audio_path"]).stem for r in rows]
    wavs = sorted((ROOT / "datasets/raw/home-dev-data/wavs").glob("*.wav"))
    meta = (ROOT / "datasets/raw/home-dev-data/metadata.csv").read_text(encoding="utf-8").splitlines()
    cache = list((ROOT / "datasets/processed/data_preprocessed/cache/22050").glob("*"))
    check("dataset rows and raw wavs", len(rows) == 13100 and len(wavs) == 13100 and len(meta) == 13100,
          rows=len(rows), wavs=len(wavs), metadata_lines=len(meta), cache_files=len(cache))
    check("no duplicate utterance IDs or audio paths", len(set(ids)) == len(ids) and len({r["audio_norm_path"] for r in rows}) == len(rows))
    check("every path referenced by dataset.jsonl exists in the copy", not any(absent.values()), path_keys=path_keys, missing_per_key=absent)
    check("dataset rows all present in metadata.csv", set(ids) <= {l.split("|")[0] for l in meta})

    # split
    sp = json.loads((ROOT / "datasets/splits/reconstructed_split_seed1234.json").read_text(encoding="utf-8"))
    check("reconstructed split 12,500 / 500 / 100 and disjoint", sp["sizes"] == {"train": 12500, "test": 500, "validation": 100} and sp["pairwise_disjoint"])
    check("reconstructed test split equals stored test_entries_500.jsonl in order", sp["test_equals_stored_test_entries_in_order"])

    # evaluation alignment
    sentences = (ROOT / "datasets/splits/test_sentences_500.txt").read_text(encoding="utf-8").split("\n")
    ev = ROOT / "evaluation/eval_h_vs_baseline"
    summ = json.loads((ev / "summary.json").read_text(encoding="utf-8"))
    for f in sorted(ev.glob("best-epoch=*_per_sentence.csv")):
        r = list(csv.DictReader(open(f, encoding="utf-8")))
        saved = next(c for c in summ["checkpoints"] if Path(c["checkpoint"]).stem == f.name.replace("_per_sentence.csv", ""))
        mean = lambda k: sum(float(x[k]) for x in r) / len(r)
        check(f"{f.name}: 500 rows aligned with the stored test sentences, means equal summary.json",
              len(r) == 500 and [x["text"] for x in r] == sentences[:500]
              and abs(mean("wer") - saved["mean_wer"]) < 1e-12 and abs(mean("utmos") - saved["mean_utmos"]) < 1e-12,
              rows=len(r), mean_wer=mean("wer"), mean_utmos=mean("utmos"), mean_rtf=mean("rtf"))
    gt = list(csv.DictReader(open(ROOT / "evaluation/ground-truth/eval_pitchflownet_500_gt/gt_wer_utmos_per_sentence.csv", encoding="utf-8")))
    check("ground-truth CSV: 500 rows in the stored test-sentence order", len(gt) == 500 and [x["text"] for x in gt] == sentences[:500], rows=len(gt))

    # scans
    cred, paths = [], 0
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in TEXT and p.stat().st_size < 50_000_000 and "datasets" not in p.parts and ".git" not in p.parts:
            t = p.read_bytes().decode("utf-8", "ignore")
            if CRED.search(t):
                cred.append(str(p.relative_to(ROOT)))
            if PATHS.search(t):
                paths += 1
    check("no credential pattern in text files", not cred, hits=cred)
    checks.append({"check": "text files containing machine paths (expected in local originals; Git tracks sanitized copies)",
                   "passed": True, "count": paths})
    failed = [c["check"] for c in checks if not c["passed"]]
    rep = {"python": sys.version.split()[0], "checks_total": len(checks), "checks_failed": failed, "checks": checks,
           "note": "Copy hashes show the copy equals the source today; they do not show which bytes a training process loaded."}
    if a.output:
        Path(a.output).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(checks)} checks, {len(failed)} failed")
    for f in failed:
        print("FAILED:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
