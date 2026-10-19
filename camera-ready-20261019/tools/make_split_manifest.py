"""Write the camera-ready split manifest: ordered stable utterance IDs per split, with hashes.

    python camera-ready-20261019/tools/make_split_manifest.py \
        --dataset <data_preprocessed>/dataset.jsonl --wav-dir <raw LJ Speech wavs> \
        --stored-test <test_entries_500.jsonl> --output camera-ready-20261019/split-manifest.json

Membership is the reconstruction of the historical split: random_split over dataset.jsonl rows in
file order, order [train, test, validation], torch.Generator().manual_seed(1234). It is checked
against the stored 500 test entries. From here on the split is read from this manifest by ID, so
changing the training seed can no longer change the split. Read-only on all inputs.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import random_split


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--wav-dir", required=True, type=Path)
    ap.add_argument("--stored-test", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--num-val", type=int, default=100)
    ap.add_argument("--num-test", type=int, default=500)
    ap.add_argument("--max-phoneme-ids", type=int, default=400)
    ap.add_argument("--num-preview", type=int, default=5)
    a = ap.parse_args()
    if a.output.exists():
        ap.error("output exists; refusing to overwrite")
    raw = a.dataset.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    kept = [i for i, r in enumerate(rows) if len(r["phoneme_ids"]) <= a.max_phoneme_ids]
    n = len(kept)
    tr, te, va = random_split(kept, [n - a.num_val - a.num_test, a.num_test, a.num_val],
                              generator=torch.Generator().manual_seed(a.split_seed))
    parts = {"train": [kept[i] for i in tr.indices], "validation": [kept[i] for i in va.indices],
             "test": [kept[i] for i in te.indices]}
    stored = [json.loads(line)["audio_norm_path"] for line in open(a.stored_test, encoding="utf-8") if line.strip()]

    def uid(i):
        return Path(rows[i]["audio_path"]).stem

    wav_sha = {i: sha_file(a.wav_dir / f"{uid(i)}.wav") for i in kept}
    out_parts = {k: [{"id": uid(i), "row": i, "raw_wav_sha256": wav_sha[i]} for i in v] for k, v in parts.items()}
    ids = {k: [e["id"] for e in v] for k, v in out_parts.items()}
    hashes = {k: [e["raw_wav_sha256"] for e in v] for k, v in out_parts.items()}
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    id_overlap = {f"{x}&{y}": sorted(set(ids[x]) & set(ids[y])) for x, y in pairs}
    hash_overlap = {f"{x}&{y}": sorted(set(hashes[x]) & set(hashes[y])) for x, y in pairs}
    text_split = collections.defaultdict(set)
    for k, v in parts.items():
        for i in v:
            text_split[rows[i]["text"]].add(k)
    shared = sorted(({"text": t, "splits": sorted(s)} for t, s in text_split.items() if len(s) > 1), key=lambda d: d["text"])
    checks = {
        "sizes": {k: len(v) for k, v in parts.items()},
        "ids_unique_within_each_split": all(len(set(v)) == len(v) for v in ids.values()),
        "ids_disjoint_across_splits": all(not v for v in id_overlap.values()),
        "raw_wav_hashes_disjoint_across_splits": all(not v for v in hash_overlap.values()),
        "test_equals_stored_test_entries_in_order": [rows[i]["audio_norm_path"] for i in parts["test"]] == stored,
        "rows_excluded_by_max_phoneme_ids": len(rows) - n,
        "transcripts_shared_across_splits": len(shared),
    }
    doc = {
        "schema": "camera-ready split manifest v1",
        "classification": "RECONSTRUCTED (historical membership, verified against the stored test entries)",
        "dataset_jsonl_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows),
        "rule": {"split_seed": a.split_seed, "order": ["train", "test", "validation"],
                 "generator": "torch.Generator().manual_seed", "torch_version_used_for_reconstruction": torch.__version__,
                 "max_phoneme_ids": a.max_phoneme_ids},
        "id_definition": "file stem of audio_path in dataset.jsonl (the LJ Speech utterance ID)",
        "preview_ids": ids["validation"][: a.num_preview],
        "preview_rule": f"first {a.num_preview} validation IDs in manifest order; test IDs are never used during training",
        "checks": checks,
        "overlaps": {"ids": id_overlap, "raw_wav_sha256": hash_overlap},
        "transcripts_shared_across_splits": shared,
        "splits": out_parts,
    }
    a.output.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "preview_ids": doc["preview_ids"]}, indent=1))


if __name__ == "__main__":
    main()
