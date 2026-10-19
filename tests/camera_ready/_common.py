"""Shared paths and helpers for the camera-ready tests (standard-library unittest).

Inputs are read-only. Environment overrides:
  CR_DATASET_DIR   preprocessed dataset directory (dataset.jsonl, config.json, cache/)
  CR_STORED_TEST   stored test_entries_500.jsonl of the historical runs
  CR_CKPT_BASELINE, CR_CKPT_FULL   historical checkpoints (epoch 773 / 1079)
  CR_TEST_OUTPUT   directory for test runs (outside Git)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]               # repository root
SRC = REPO / "code" / "camera-ready-source" / "src" / "python"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HOME = Path.home()  # the WSL account that holds the historical runs and data
DATASET_DIR = Path(os.environ.get("CR_DATASET_DIR", str(HOME / "data_preprocessed")))
STORED_TEST = Path(os.environ.get("CR_STORED_TEST", str(HOME / "test_entries_500.jsonl")))
SPLIT_MANIFEST = REPO / "manifests" / "camera-ready" / "split-manifest.json"
CKPT_BASELINE = Path(os.environ.get("CR_CKPT_BASELINE", str(
    HOME / "02_Baseline_VanillaVITS/lightning_logs/version_0/checkpoints/best-epoch=773-val_loss_mel=20.2158.ckpt")))
CKPT_FULL = Path(os.environ.get("CR_CKPT_FULL", str(
    HOME / "01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt")))
TEST_OUTPUT = Path(os.environ.get("CR_TEST_OUTPUT", str(REPO.parent / "outputs" / "tests")))


def dataset_config() -> dict:
    return json.loads((DATASET_DIR / "config.json").read_text(encoding="utf-8"))


def small_manifest(n_train=4, n_val=3, n_test=2, n_preview=2) -> Path:
    """Write a manifest with the first entries of each split of the real manifest (same dataset hash)."""
    doc = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    doc["splits"] = {"train": doc["splits"]["train"][:n_train], "validation": doc["splits"]["validation"][:n_val],
                     "test": doc["splits"]["test"][:n_test]}
    doc["preview_ids"] = [e["id"] for e in doc["splits"]["validation"][:n_preview]]
    doc["note"] = "test subset of split-manifest.json"
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="manifest-", suffix=".json", dir=TEST_OUTPUT)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return Path(path)


def new_run_dir(tag: str) -> Path:
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=f"{tag}-", dir=TEST_OUTPUT))
    d.rmdir()  # train.py requires a run directory that does not exist yet
    return d


def base_model_kwargs(**flags) -> dict:
    cfg = dataset_config()
    kw = dict(num_symbols=int(cfg["num_symbols"]), num_speakers=int(cfg["num_speakers"]),
              sample_rate=int(cfg["audio"]["sample_rate"]), dataset=None, resblock="2")
    kw.update(flags)
    return kw
