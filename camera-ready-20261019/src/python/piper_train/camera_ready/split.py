"""Read the camera-ready split manifest and bind it to dataset.jsonl rows by stable utterance ID."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitManifest:
    path: Path
    sha256: str
    dataset_sha256: str
    rows: Dict[str, List[int]]   # split -> dataset.jsonl row indices, in manifest order
    ids: Dict[str, List[str]]    # split -> utterance IDs, in manifest order
    preview_ids: List[str]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_split_manifest(path: Union[str, Path], dataset_jsonl: Union[str, Path]) -> SplitManifest:
    """Load the manifest and check it against dataset.jsonl.

    Raises ValueError when the dataset hash differs, when a row does not carry the expected ID,
    when IDs overlap between splits, or when a preview ID is not a validation ID.
    """
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    ds_raw = Path(dataset_jsonl).read_bytes()
    ds_sha = sha256_bytes(ds_raw)
    if ds_sha != doc["dataset_jsonl_sha256"]:
        raise ValueError(f"dataset.jsonl SHA-256 {ds_sha} differs from the manifest ({doc['dataset_jsonl_sha256']})")
    lines = [json.loads(line) for line in ds_raw.decode("utf-8").splitlines() if line.strip()]
    stem = [Path(r["audio_path"]).stem for r in lines]
    rows, ids = {}, {}
    for split in SPLITS:
        entries = doc["splits"][split]
        r = [int(e["row"]) for e in entries]
        i = [str(e["id"]) for e in entries]
        for row, uid in zip(r, i):
            if row >= len(stem) or stem[row] != uid:
                raise ValueError(f"{split}: row {row} is {stem[row] if row < len(stem) else 'missing'}, manifest says {uid}")
        rows[split], ids[split] = r, i
    seen = [x for split in SPLITS for x in ids[split]]
    if len(seen) != len(set(seen)):
        raise ValueError("utterance IDs overlap between splits")
    preview = list(doc.get("preview_ids", []))
    if not set(preview) <= set(ids["validation"]):
        raise ValueError("preview IDs must be validation IDs")
    return SplitManifest(path=path, sha256=sha256_bytes(raw), dataset_sha256=ds_sha, rows=rows, ids=ids, preview_ids=preview)
