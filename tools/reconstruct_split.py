"""Reconstruct the train/test/validation membership used by the PitchFlowNet-candidate runs.

    python tools/reconstruct_split.py --dataset datasets/processed/data_preprocessed/dataset.jsonl \
        --test-entries datasets/splits/test_entries_500.jsonl --output datasets/splits/reconstructed_split_seed1234.json

Rule taken from piper_train/vits/lightning.py (_load_datasets) in the candidate sources:
rows of dataset.jsonl in file order (max_phoneme_ids=400 removes nothing here: longest row 399),
then torch.utils.data.random_split(rows, [n - 100 - 500, 500, 100],
generator=torch.Generator().manual_seed(1234)), i.e. the order train, TEST, validation.
This is a RECONSTRUCTION with the currently installed torch; it is checked against the
stored test_entries_500.jsonl and does not re-run any preprocessing or training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import random_split


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--test-entries", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-val", type=int, default=100)
    p.add_argument("--num-test", type=int, default=500)
    p.add_argument("--max-phoneme-ids", type=int, default=400)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists")
    rows = [json.loads(l) for l in open(a.dataset, encoding="utf-8") if l.strip()]
    kept = [i for i, r in enumerate(rows) if len(r["phoneme_ids"]) <= a.max_phoneme_ids]
    n = len(kept)
    train, test, val = random_split(kept, [n - a.num_val - a.num_test, a.num_test, a.num_val],
                                    generator=torch.Generator().manual_seed(a.seed))
    part = {"train": [kept[i] for i in train.indices], "test": [kept[i] for i in test.indices], "validation": [kept[i] for i in val.indices]}
    key = lambda i: rows[i]["audio_norm_path"]
    stored = [json.loads(l)["audio_norm_path"] for l in open(a.test_entries, encoding="utf-8") if l.strip()]
    sets = {k: set(map(key, v)) for k, v in part.items()}
    wav = lambda i: Path(rows[i]["audio_path"]).stem
    out = {
        "classification": "RECONSTRUCTED",
        "rule": "random_split(rows in file order, [train, test, validation]) with torch.Generator().manual_seed(seed)",
        "torch_version_used_now": torch.__version__,
        "dataset_sha256": hashlib.sha256(a.dataset.read_bytes()).hexdigest(), "rows": len(rows),
        "rows_after_max_phoneme_ids": n, "seed": a.seed,
        "sizes": {k: len(v) for k, v in part.items()},
        "pairwise_disjoint": not (sets["train"] & sets["test"] or sets["train"] & sets["validation"] or sets["test"] & sets["validation"]),
        "test_equals_stored_test_entries_in_order": [key(i) for i in part["test"]] == stored,
        "test_equals_stored_test_entries_as_set": set(stored) == sets["test"],
        "partitions": {k: [{"row": i, "lj_id": wav(i), "audio_norm_path": key(i)} for i in v] for k, v in part.items()},
    }
    a.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({k: out[k] for k in ("sizes", "pairwise_disjoint", "test_equals_stored_test_entries_in_order", "test_equals_stored_test_entries_as_set")})


if __name__ == "__main__":
    main()
