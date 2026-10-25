"""Pitch intervention test (T14; protocol agreed with the author on 2026-10-25, before any run).

    python tools/camera_ready/pitch_shift.py --final-dir <CR_OUTPUTS>/final --eval-dir <CR_OUTPUTS>/eval \
           --out <CR_OUTPUTS>/tests/pitch-shift-<YYYYMMDD> [--limit 100] [--workers 8]

Question: does the decoder use its F0 conditioning, and can the pitch of the waveform be steered through it?
  - Models: the best checkpoints of the four F = 1 cells, and the historical Full checkpoint (epoch 1079).
  - Hook: dec.forward is wrapped so that the F0 it receives is multiplied by 2^(s/12), or replaced by None ("off",
    no F0 conditioning). The decoder draws no random numbers, so with the evaluation's per-utterance seed the
    latent z, the durations and the noise stay the same in every condition. Nothing in code/ is changed.
  - Conditions: s = -2, 0, +2 semitones, and off. Sentences: the first --limit test utterances (default 100).
  - Check: at s = 0 the durations equal the evaluation's, and where the evaluation's WAVs exist on this machine the
    WAVs are byte-identical. The historical checkpoint was evaluated without seeds, so it has no such check.
  - F0: pyworld harvest, 60-600 Hz, 5 ms frames, the same settings for every condition. The audio lengths are
    equal across conditions, so frames pair one to one (no alignment).
  - Per utterance and condition c: achieved shift = median over frames voiced in both c and 0 of
    12 * log2(F0_c / F0_0). Per model: mean, median and paired-bootstrap 95 % CI over utterances (10,000, seed 2026);
    slope = (shift(+2) - shift(-2)) / 4; voicing agreement with s = 0.
  - Interpretation, fixed before the run: slope >= 0.8 with a CI excluding 0 is "controllable"; 0.2 <= slope < 0.8
    with a CI excluding 0 is "partly controllable"; otherwise "not controllable". An "off" condition clearly
    different from s = 0 shows that the decoder uses the conditioning.
Outputs: <out>/wavs/<label>/<condition>/sent_NNNN.wav, <out>/records.jsonl, <out>/summary.json, <out>/protocol.json.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "camera-ready-source" / "src" / "python"))

CONDITIONS = ("-2", "0", "+2", "off")
HARVEST = {"f0_floor": 60.0, "f0_ceil": 600.0, "frame_period": 5.0}
HISTORICAL = ("Historical-Full-1079", pathlib.Path.home() / "01_Baseline_BigVgan_VITS2_FO" / "lightning_logs" / "version_2" /
              "checkpoints" / "best-epoch=1079-val_loss_mel=19.2943.ckpt",
              "8e6eacface1105c329ada0ccbec4c7a5cce88e86ac08797092ab30c1698a4edf")   # data-manifest.json, ckpt_full_1079


def install_f0_hook(dec):
    """Wrap dec.forward; returns a function that sets the condition ("off" or semitones as a string)."""
    original = dec.forward
    state = {"cond": "0"}

    def forward(x, g=None, f0=None):
        if state["cond"] == "off":
            f0 = None
        elif f0 is not None:
            f0 = f0 * (2.0 ** (float(state["cond"]) / 12.0))
        return original(x, g=g, f0=f0)

    dec.forward = forward
    return lambda cond: state.__setitem__("cond", cond)


def shift_semitones(f0_c, f0_0):
    """Median of 12*log2(f0_c / f0_0) over frames voiced in both, and the voicing agreement; None if no frame."""
    import numpy as np
    n = min(len(f0_c), len(f0_0))
    a, b = np.asarray(f0_c[:n], dtype=float), np.asarray(f0_0[:n], dtype=float)
    both = (a > 0) & (b > 0)
    agree = float(np.mean((a > 0) == (b > 0))) if n else None
    if not both.any():
        return None, agree
    return float(np.median(12.0 * np.log2(a[both] / b[both]))), agree


def interpret(slope, lo, hi):
    if lo > 0 or hi < 0:
        if slope >= 0.8:
            return "controllable"
        if slope >= 0.2:
            return "partly controllable"
    return "not controllable"


def boot_mean_ci(x, idx):
    import numpy as np
    v = np.asarray(x, dtype=float)
    m = v[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def summarize(records, resamples=10000, seed=2026):
    """records: one per (label, utterance) with shift and voicing agreement per condition. Per label: statistics."""
    import numpy as np
    out = {}
    labels = sorted({r["label"] for r in records}, key=[r["label"] for r in records].index)
    for label in labels:
        rs = [r for r in records if r["label"] == label]
        ok = [r for r in rs if all(r["shift"][c] is not None for c in ("-2", "+2", "off"))]
        idx = np.random.default_rng(seed).integers(0, len(ok), size=(resamples, len(ok)))
        row = {"n_utterances": len(rs), "n_used": len(ok)}
        for c in ("-2", "+2", "off"):
            s = [r["shift"][c] for r in ok]
            mean, lo, hi = boot_mean_ci(s, idx)
            row[c] = {"mean_shift_st": mean, "ci95": [lo, hi], "median_shift_st": float(np.median(s)),
                      "voicing_agreement": float(np.mean([r["voicing"][c] for r in ok]))}
        slope = [(r["shift"]["+2"] - r["shift"]["-2"]) / 4.0 for r in ok]
        mean, lo, hi = boot_mean_ci(slope, idx)
        row["slope"] = {"mean": mean, "ci95": [lo, hi], "interpretation": interpret(mean, lo, hi)}
        row["off_mean_abs_shift_st"] = float(np.mean([abs(r["shift"]["off"]) for r in ok]))
        out[label] = row
    return out


# ---------------------------------------------------------------- run

def f0_of(path):
    import numpy as np
    import pyworld
    import soundfile
    x, sr = soundfile.read(str(path), dtype="float64")
    f0, _t = pyworld.harvest(np.ascontiguousarray(x), sr, **HARVEST)
    return f0.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-dir", type=pathlib.Path, required=True)
    ap.add_argument("--eval-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import csv
    from multiprocessing import Pool
    import torch
    torch.set_num_threads(4)
    import ablation8_analysis as A
    import evaluate_like_history as elh
    from piper_train import eval_harness as eh

    if a.out.exists():
        raise SystemExit(f"{a.out} exists; use a new folder")
    a.out.mkdir(parents=True)
    rows, ids, _ = elh.split_rows("test", a.limit)
    models = []
    for label, bits in A.CELLS.items():
        if bits[2]:
            fin = a.final_dir / A.RUNS[label]
            name = json.loads((fin / "final.json").read_text())["best_checkpoint"]
            sums = {p.split(None, 1)[1].strip(): p.split(None, 1)[0] for p in (fin / "SHA256SUMS").read_text().splitlines() if p}
            models.append((label, fin / name, sums[name]))
    models.append(HISTORICAL)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    protocol = {"tool": "tools/camera_ready/pitch_shift.py", "commit": commit, "dirty": [d for d in dirty if d],
                "argv": sys.argv, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "conditions": CONDITIONS, "harvest": HARVEST, "scales": elh.SCALES, "seed_base": 2026, "threads": 4,
                "n": len(rows), "ids_first_last": [ids[0], ids[-1]], "versions": elh.versions(),
                "models": [{"label": l, "checkpoint": str(p), "expected_sha256": h} for l, p, h in models],
                "interpretation": "slope >= 0.8 and CI excluding 0: controllable; 0.2 <= slope < 0.8 and CI excluding 0: "
                                  "partly controllable; otherwise not controllable (fixed before the run)"}
    (a.out / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n")
    seed_of = lambda i: (2026 * 1_000_003 + i) % (2 ** 31)
    checks, wavs = [], []
    for label, ckpt, expected in models:
        if A.sha256(ckpt) != expected:
            raise SystemExit(f"{ckpt}: SHA-256 differs from the record")
        model, info = elh.load_model(ckpt)
        model.eval()
        with torch.no_grad():
            model.model_g.dec.remove_weight_norm()           # as the evaluation does
        set_cond = install_f0_hook(model.model_g.dec)
        durations = {}
        for cond in CONDITIONS:
            set_cond(cond)
            d = a.out / "wavs" / label / cond
            d.mkdir(parents=True)
            for i, r in enumerate(rows):
                torch.manual_seed(seed_of(i))
                audio, _sec, dur, _rtf = eh.synthesize(model, r["text"], "en-us", elh.SCALES, elh.SAMPLE_RATE)
                eh.write_wav(str(d / f"sent_{i:04d}.wav"), elh.SAMPLE_RATE, eh.audio_float_to_int16(audio))
                durations.setdefault(cond, []).append(dur)
                wavs.append((label, cond, i, d / f"sent_{i:04d}.wav"))
        equal_len = all(durations[c] == durations["0"] for c in CONDITIONS)
        check = {"label": label, "epoch": info["epoch"], "lengths_equal_across_conditions": equal_len}
        evd = a.eval_dir / label
        if evd.is_dir():
            per = next(evd.glob("*_per_sentence.csv"))
            ev = [float(x["audio_duration_sec"]) for x in csv.DictReader(open(per, encoding="utf-8"))][:len(rows)]
            check["durations_equal_evaluation"] = sum(abs(x - y) < 1e-6 for x, y in zip(durations["0"], ev))
            if (evd / "wavs").is_dir():
                check["wav_bytes_equal_evaluation"] = sum(
                    (evd / "wavs" / f"sent_{i:04d}.wav").read_bytes() == (a.out / "wavs" / label / "0" / f"sent_{i:04d}.wav").read_bytes()
                    for i in range(len(rows)))
        checks.append(check)
        print(json.dumps(check), flush=True)
        if not equal_len or check.get("durations_equal_evaluation", len(rows)) != len(rows) or \
                check.get("wav_bytes_equal_evaluation", len(rows)) != len(rows):
            raise SystemExit(f"{label}: reproduction check failed: {check}")
    with Pool(a.workers) as pool:
        f0s = pool.map(f0_of, [w[3] for w in wavs], chunksize=8)
    f0_by = {(l, c, i): f for (l, c, i, _p), f in zip(wavs, f0s)}
    records = []
    for label, _ckpt, _h in models:
        for i in range(len(rows)):
            shift, voicing = {}, {}
            for c in ("-2", "+2", "off"):
                shift[c], voicing[c] = shift_semitones(f0_by[(label, c, i)], f0_by[(label, "0", i)])
            records.append({"label": label, "utterance": i, "id": ids[i], "shift": shift, "voicing": voicing})
    with open(a.out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(a.out / "f0.jsonl", "w", encoding="utf-8") as f:
        for (l, c, i), v in f0_by.items():
            f.write(json.dumps({"label": l, "condition": c, "utterance": i, "f0": [round(x, 3) for x in v]}) + "\n")
    summary = summarize(records)
    (a.out / "summary.json").write_text(json.dumps({"checks": checks, "results": summary,
                                                    "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    for label, row in summary.items():
        print(f"{label}: slope {row['slope']['mean']:.3f} {row['slope']['ci95']} -> {row['slope']['interpretation']}; "
              f"+2 {row['+2']['mean_shift_st']:+.3f}, -2 {row['-2']['mean_shift_st']:+.3f}, off {row['off']['mean_shift_st']:+.3f} st")


if __name__ == "__main__":
    main()
