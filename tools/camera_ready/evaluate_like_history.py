"""Evaluate one checkpoint with the historical evaluation of the paper, made reproducible.

    python tools/camera_ready/evaluate_like_history.py evaluate --checkpoint <ckpt> --label <name> --output <dir>
           [--split test|validation] [--limit N] [--seed 2026] [--threads 4] [--no-mcd] [--no-f0]
    python tools/camera_ready/evaluate_like_history.py compare --output <dir> --labels A B [C ...]

Run it in the evaluation environment (pfn-eval: openai-whisper, jiwer, UTMOSv2 at the historical versions).

What is measured is exactly what the historical evaluation measured, by calling its own functions in
piper_train/eval_harness.py and piper_train/compute_reference_metrics.py:
  - per utterance: synthesis on the CPU with noise_scale 0.667, length_scale 1.0, noise_w 0.8; WER with
    Whisper base.en after the harness's text normalisation; UTMOSv2 (a predicted score, not MOS); RTF;
  - per checkpoint: mean/median WER and UTMOSv2, mean RTF; F0 RMSE/correlation of the F0 predictor (models
    with the F0 group only); MCD (DTW on 24 MFCCs) and ground-truth UTMOSv2 against the reference audio;
  - compare: two-sided Wilcoxon signed-rank tests on WER and UTMOSv2 for every pair (SciPy defaults).
Four things differ from the historical runs, and each is recorded in protocol.json:
  1. checkpoints load with torch.load(weights_only=True) through camera_ready/ckpt_io.py and a strict
     state-dict load (no missing or unexpected tensors), instead of load_from_checkpoint(strict=False);
  2. synthesis is seeded per utterance (seed = (base * 1,000,003 + index) mod 2^31), so a rerun gives the
     same audio; the historical synthesis was not seeded (two runs of the epoch-1079 model differed);
  3. the sentences come from the split manifest; for the test split they are checked line by line against
     datasets/splits/test_sentences_500.txt, the historical sentence file, and the F0 predictor metric uses
     the dataset rows of that split (the historical run did not record which rows it used);
  4. UTMOSv2 scoring is seeded per utterance: its predict() takes a random crop of the audio
     (utmosv2/dataset/_utils.py, np.random.randint), so the same audio scored twice differed by up to ~0.18;
     numpy, random and torch are seeded before each call (default num_repetitions=1, as historically).
The test split is used only when the author has locked the protocol (docs/camera-ready/eval-protocol.md).
"""
import argparse
import csv
import random
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "camera-ready-source" / "src" / "python"
sys.path.insert(0, str(SRC))

ARCH_KEYS = ("num_symbols", "num_speakers", "sample_rate", "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
             "upsample_rates", "upsample_initial_channel", "upsample_kernel_sizes", "filter_length", "hop_length",
             "win_length", "mel_channels", "inter_channels", "hidden_channels", "filter_channels", "n_heads", "n_layers",
             "kernel_size", "p_dropout", "segment_size", "use_sdp", "gin_channels", "use_bigvgan", "use_vits2", "use_f0")
SCALES = [0.667, 1.0, 0.8]          # historical defaults of eval_harness (noise_scale, length_scale, noise_w)
SAMPLE_RATE = 22050
HISTORICAL_SENTENCES = ROOT / "datasets" / "splits" / "test_sentences_500.txt"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    out = {}
    for p in ("torch", "torchaudio", "openai-whisper", "jiwer", "transformers", "timm", "scipy", "librosa", "utmosv2",
              "numpy", "piper-phonemize"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def split_rows(split: str, limit: int):
    from piper_train.camera_ready.split import load_split_manifest
    dataset = pathlib.Path(os.environ.get("CR_DATASET_DIR", str(pathlib.Path.home() / "data_preprocessed"))) / "dataset.jsonl"
    manifest_path = ROOT / "manifests" / "camera-ready" / "split-manifest.json"
    m = load_split_manifest(manifest_path, dataset)          # checks the dataset hash
    lines = dataset.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(lines[r]) for r in m.rows[split]]
    ids = list(m.ids[split])
    if split == "test":
        hist = [s.strip() for s in HISTORICAL_SENTENCES.read_text(encoding="utf-8").splitlines() if s.strip()]
        if [r["text"].strip() for r in rows] != hist:
            raise SystemExit("test texts differ from datasets/splits/test_sentences_500.txt")
    if limit:
        rows, ids = rows[:limit], ids[:limit]
    texts = [r["text"] for r in rows]
    if len(set(texts)) != len(texts):
        raise SystemExit("duplicate texts in the split; the per-utterance seed needs unique texts")
    return rows, ids, {"manifest_sha256": m.sha256, "dataset_sha256": m.dataset_sha256, "dataset_jsonl": str(dataset)}


class SeededUTMOS:
    """UTMOSv2 model whose predict() is seeded per call (call k uses seed base + k)."""

    def __init__(self, model, base: int):
        self.model, self.base, self.calls = model, base, 0

    def predict(self, *args, **kwargs):
        import numpy as np
        import torch
        s = (self.base + self.calls) % (2 ** 31)
        self.calls += 1
        random.seed(s), np.random.seed(s), torch.manual_seed(s)
        return self.model.predict(*args, **kwargs)


# Runs piper_train.compute_reference_metrics in a child process with a seeded UTMOSv2 (ground-truth UTMOS).
REFERENCE_RUNNER = """
import sys, utmosv2
sys.path.insert(0, sys.argv[1]); base = int(sys.argv[2])
sys.path.insert(0, sys.argv[3])
from evaluate_like_history import SeededUTMOS
_create = utmosv2.create_model
utmosv2.create_model = lambda *a, **k: SeededUTMOS(_create(*a, **k), base)
from piper_train import compute_reference_metrics as crm
crm.utmosv2 = utmosv2
sys.argv = ["compute_reference_metrics"] + sys.argv[4:]
crm.main()
"""


def load_model(path: pathlib.Path):
    from piper_train.camera_ready.ckpt_io import load_checkpoint
    from piper_train.vits.lightning import VitsModel
    ck = load_checkpoint(path)
    hp = ck["hyper_parameters"]
    semantics = "strict" if "camera_ready" in ck else "legacy"
    model = VitsModel(dataset=None, group_semantics=semantics, **{k: hp[k] for k in ARCH_KEYS if k in hp})
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=True)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing {list(missing)[:3]}, unexpected {list(unexpected)[:3]}")
    info = {"group_semantics": semantics, "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
            "camera_ready": {k: v for k, v in ck.get("camera_ready", {}).items() if k != "units"},
            "flags": {k: hp.get(k) for k in ("use_bigvgan", "use_vits2", "use_f0")}}
    return model, info


def cmd_evaluate(a):
    import torch
    torch.set_num_threads(a.threads)
    from piper_train import eval_harness as eh
    import utmosv2
    import whisper

    out = a.output / a.label
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; use a new label or output folder")
    out.mkdir(parents=True, exist_ok=True)
    rows, ids, split_info = split_rows(a.split, a.limit)
    texts = [r["text"] for r in rows]
    index = {t: i for i, t in enumerate(texts)}
    seed_of = lambda i: (a.seed * 1_000_003 + i) % (2 ** 31)

    model, ckpt_info = load_model(a.checkpoint)
    # The harness loads with load_from_checkpoint(strict=False); give it the strictly loaded model instead.
    eh.VitsModel = type("StrictLoaded", (), {"load_from_checkpoint": staticmethod(lambda *x, **k: model)})
    orig_synth, orig_f0 = eh.synthesize, eh.collect_f0_pairs

    def seeded_synthesize(m, text, *args, **kwargs):
        torch.manual_seed(seed_of(index[text]))
        return orig_synth(m, text, *args, **kwargs)

    def seeded_f0(m, entries):
        torch.manual_seed(seed_of(10 ** 6))
        return orig_f0(m, entries)

    eh.synthesize, eh.collect_f0_pairs = seeded_synthesize, seeded_f0

    protocol = {
        "tool": "tools/camera_ready/evaluate_like_history.py", "argv": sys.argv, "label": a.label,
        "checkpoint": str(a.checkpoint), "checkpoint_sha256": sha256(a.checkpoint), "checkpoint_info": ckpt_info,
        "split": a.split, "n": len(texts), "ids_first_last": [ids[0], ids[-1]], **split_info,
        "scales": SCALES, "sample_rate": SAMPLE_RATE, "whisper_model": "base.en", "devices": "cpu (as historical)",
        "seed": {"base": a.seed, "rule": "(base * 1000003 + index) mod 2^31 before each synthesis; F0 metric: index 10^6; "
                 "UTMOSv2 of utterance k: base * 1000003 + 5000000 + k; ground-truth UTMOSv2 of utterance k: base * 1000003 + 8000000 + k"},
        "threads": a.threads, "mcd": not a.no_mcd, "f0_predictor_metric": not a.no_f0,
        "versions": versions(), "python": platform.python_version(), "platform": platform.platform(),
        "loadavg_at_start": os.getloadavg(), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rtf_note": "RTF is measured inline as in the historical harness; it is only comparable on an idle machine",
        "differences_from_historical": ["strict weights_only checkpoint load", "per-utterance synthesis seed",
                                        "sentences from the split manifest (checked against the historical file)",
                                        "UTMOSv2 seeded per utterance (its predict() crops the audio at random)"],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n")

    whisper_model = whisper.load_model("base.en", device="cpu")
    utmos_model = SeededUTMOS(utmosv2.create_model(pretrained=True, device="cpu"), a.seed * 1_000_003 + 5_000_000)
    f0_entries = rows if (not a.no_f0 and ckpt_info["flags"]["use_f0"]) else None
    t0 = time.time()
    report = eh.evaluate_checkpoint(a.checkpoint, texts, whisper_model, utmos_model, "en-us", SCALES, SAMPLE_RATE,
                                    out / "wavs", dataset_entries=f0_entries)
    eh.write_report(report, out)
    summary = {"label": a.label, "checkpoint": report.checkpoint, "mean_wer": report.mean_wer,
               "median_wer": report.median_wer, "mean_utmos": report.mean_utmos, "median_utmos": report.median_utmos,
               "mean_rtf": report.mean_rtf, "f0_rmse": report.f0_rmse, "f0_corr": report.f0_corr,
               "num_sentences": len(report.results), "eval_seconds": round(time.time() - t0, 1)}

    if not a.no_mcd:
        (out / "reference.jsonl").write_text("".join(json.dumps({"text": r["text"], "audio_norm_path": r["audio_norm_path"]}) + "\n" for r in rows))
        subprocess.run([sys.executable, "-c", REFERENCE_RUNNER, str(SRC), str(a.seed * 1_000_003 + 8_000_000),
                        str(pathlib.Path(__file__).resolve().parent), "--gen-wav-dir", str(out / "wavs"),
                        "--test-jsonl", str(out / "reference.jsonl"), "--output", str(out / "mcd"),
                        "--sample-rate", str(SAMPLE_RATE)], cwd=SRC, check=True)
        summary["mcd"] = json.loads((out / "mcd" / "mcd_gtmos_summary.json").read_text())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def cmd_compare(a):
    from scipy.stats import wilcoxon
    data = {}
    for label in a.labels:
        csvs = list((a.output / label).glob("*_per_sentence.csv"))
        if len(csvs) != 1:
            raise SystemExit(f"{a.output / label}: expected one per-sentence CSV, found {len(csvs)}")
        with open(csvs[0], encoding="utf-8") as f:
            data[label] = list(csv.DictReader(f))
    texts = {k: [r["text"] for r in v] for k, v in data.items()}
    if len({tuple(t) for t in texts.values()}) != 1:
        raise SystemExit("the runs were not evaluated on the same sentences in the same order")
    tests = []
    for i, la in enumerate(a.labels):
        for lb in a.labels[i + 1:]:
            e = {"a": la, "b": lb, "n": len(data[la])}
            for metric in ("wer", "utmos"):
                xa = [float(r[metric]) for r in data[la]]
                xb = [float(r[metric]) for r in data[lb]]
                e[f"mean_{metric}_a"], e[f"mean_{metric}_b"] = sum(xa) / len(xa), sum(xb) / len(xb)
                try:
                    s, p = wilcoxon(xa, xb)
                    e[f"{metric}_wilcoxon_stat"], e[f"{metric}_wilcoxon_p"] = float(s), float(p)
                except ValueError as exc:
                    e[f"{metric}_wilcoxon_error"] = str(exc)
            tests.append(e)
    path = a.output / f"compare-{'_'.join(a.labels)}.json"
    path.write_text(json.dumps({"method": "two-sided Wilcoxon signed-rank, SciPy defaults, per utterance (as eval_harness)",
                                "paired_tests": tests}, indent=1) + "\n")
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", type=pathlib.Path, required=True)
    e.add_argument("--label", required=True)
    e.add_argument("--output", type=pathlib.Path, required=True)
    e.add_argument("--split", choices=("test", "validation"), default="validation")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--seed", type=int, default=2026)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--no-mcd", action="store_true")
    e.add_argument("--no-f0", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--output", type=pathlib.Path, required=True)
    c.add_argument("--labels", nargs="+", required=True)
    a = ap.parse_args()
    cmd_evaluate(a) if a.cmd == "evaluate" else cmd_compare(a)


if __name__ == "__main__":
    main()
