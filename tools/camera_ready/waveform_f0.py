"""Waveform F0 against the recordings (T13; protocol agreed with the author on 2026-10-25, before any run).

    python tools/camera_ready/waveform_f0.py --final-dir <CR_OUTPUTS>/final --eval-dir <CR_OUTPUTS>/eval \
           --out <CR_OUTPUTS>/tests/waveform-f0-<YYYYMMDD> [--limit 0] [--workers 8]

  - Models: the best checkpoints of the eight ablation cells and the historical pair (773, 1079).
  - Audio: the test utterances synthesized again with the evaluation's per-utterance seeds and scales. Where the
    evaluation's WAVs exist on this machine they must be byte-identical; the durations of the eight cells must
    equal the evaluation's. The historical pair is synthesized with seeds for the first time.
  - Reference: the normalised audio of the cache (audio_norm_path), the reference of the harness's MCD.
  - F0: pyworld harvest, 60-600 Hz, 5 ms frames, the same settings for synthesized and reference audio.
  - Alignment: DTW exactly as the harness's MCD (24 MFCC C1-C24, n_fft 1024, hop 256, normalised by the mean and
    standard deviation of the reference frames, euclidean); each MFCC frame takes the F0 frame nearest in time.
    Sensitivity: a linear time stretch. DTW on F0 itself is not used (it would inflate the agreement).
  - Per utterance, on the aligned frame pairs: F0 RMSE in cents and mean offset in cents (pairs voiced in both),
    correlation of log-F0 (same pairs, >= 10 pairs), voicing decision error VDE, gross pitch error GPE
    (|F0_syn / F0_ref - 1| > 0.2 among pairs voiced in both), F0 frame error FFE, coverage (pairs voiced in both).
  - Per model: mean with a bootstrap 95 % CI over utterances (10,000, seed 2026). Paired: the four F edges of the
    cube, Full vs Baseline and 1079 vs 773; mean difference (on - off) with a paired-bootstrap CI, two-sided
    Wilcoxon (SciPy defaults), Holm within each metric across these six.
Outputs: <out>/wavs/<label>/sent_NNNN.wav, records.jsonl, summary.json, protocol.json.
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

SR = 22050
HARVEST = {"f0_floor": 60.0, "f0_ceil": 600.0, "frame_period": 5.0}
MFCC = {"n_mfcc": 25, "n_fft": 1024, "hop_length": 256}      # C0 dropped, as compute_reference_metrics.extract_mfcc
METRICS = ("rmse_cents", "offset_cents", "corr_logf0", "vde", "gpe", "ffe", "coverage")
HIST = pathlib.Path.home()
HISTORICAL = [
    ("Historical-Baseline-773", HIST / "02_Baseline_VanillaVITS/lightning_logs/version_0/checkpoints/best-epoch=773-val_loss_mel=20.2158.ckpt",
     "54d867192b673c5f493ee937db06ff3bba6a1c0192969131407e3a795245aac1"),
    ("Historical-Full-1079", HIST / "01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt",
     "8e6eacface1105c329ada0ccbec4c7a5cce88e86ac08797092ab30c1698a4edf"),
]                                                              # data-manifest.json, ckpt_baseline_773 / ckpt_full_1079
COMPARISONS = [("F", "PFN-Baseline-200ep-best", "PFN-Baseline-F0-200ep-best"),
               ("F", "PFN-Baseline-VITS2-200ep-best", "PFN-NoBigVGAN-200ep-best"),
               ("F", "PFN-Baseline-BigVGAN-200ep-best", "PFN-NoVITS2-200ep-best"),
               ("F", "PFN-NoF0-200ep-best", "PFN-Full-200ep-best"),
               ("BVF", "PFN-Baseline-200ep-best", "PFN-Full-200ep-best"),
               ("historical", "Historical-Baseline-773", "Historical-Full-1079")]


def pair_metrics(f0_ref, f0_syn):
    """Metrics on aligned pairs (arrays of the same length; 0 = unvoiced)."""
    import numpy as np
    r, s = np.asarray(f0_ref, dtype=float), np.asarray(f0_syn, dtype=float)
    vr, vs = r > 0, s > 0
    both = vr & vs
    n, nb = len(r), int(both.sum())
    out = {"pairs": n, "vde": float(np.mean(vr != vs)) if n else None, "coverage": nb / n if n else None}
    if nb:
        cents = 1200.0 * np.log2(s[both] / r[both])
        gross = np.abs(s[both] / r[both] - 1.0) > 0.2
        out.update(rmse_cents=float(np.sqrt(np.mean(cents ** 2))), offset_cents=float(np.mean(cents)),
                   gpe=float(np.mean(gross)), ffe=float((np.sum(vr != vs) + np.sum(gross)) / n))
        lr, ls = np.log(r[both]), np.log(s[both])
        out["corr_logf0"] = float(np.corrcoef(lr, ls)[0, 1]) if nb >= 10 and lr.std() > 0 and ls.std() > 0 else None
    else:
        out.update(rmse_cents=None, offset_cents=None, gpe=None, ffe=float(np.mean(vr != vs)) if n else None, corr_logf0=None)
    return out


def frames_to_f0_index(n_mfcc_frames, n_f0):
    """Nearest 5 ms F0 frame for each MFCC frame (centred at k * hop / sr)."""
    import numpy as np
    t = np.arange(n_mfcc_frames) * MFCC["hop_length"] / SR
    return np.clip(np.rint(t / (HARVEST["frame_period"] / 1000.0)).astype(int), 0, n_f0 - 1)


def aligned_pairs(f0_ref, mf_ref, f0_syn, mf_syn, mean, std, method):
    """F0 pairs along the MFCC DTW path (as the MCD of the harness) or along a linear time stretch."""
    import librosa
    import numpy as np
    ir, is_ = frames_to_f0_index(len(mf_ref), len(f0_ref)), frames_to_f0_index(len(mf_syn), len(f0_syn))
    if method == "dtw":
        _D, wp = librosa.sequence.dtw(((mf_syn - mean) / std).T, ((mf_ref - mean) / std).T, metric="euclidean")
        wp = wp[::-1]
        syn_k, ref_k = wp[:, 0], wp[:, 1]
    else:
        ref_k = np.arange(len(mf_ref))
        syn_k = np.rint(ref_k * (len(mf_syn) - 1) / max(len(mf_ref) - 1, 1)).astype(int)
    return np.asarray(f0_ref)[ir[ref_k]], np.asarray(f0_syn)[is_[syn_k]]


def features(audio):
    import librosa
    import numpy as np
    import pyworld
    x = np.ascontiguousarray(audio, dtype=np.float64)
    f0, _t = pyworld.harvest(x, SR, **HARVEST)
    mf = librosa.feature.mfcc(y=np.asarray(audio, dtype=np.float32), sr=SR, **MFCC)[1:].T
    return f0, mf


def _job(args):
    """One synthesized utterance against its reference: metrics for both alignments."""
    import soundfile
    label, i, wav, ref_feat, mean, std = args
    a, sr = soundfile.read(str(wav), dtype="float32")
    assert sr == SR
    f0_s, mf_s = features(a)
    f0_r, mf_r = ref_feat
    return {"label": label, "utterance": i,
            **{m: pair_metrics(*aligned_pairs(f0_r, mf_r, f0_s, mf_s, mean, std, m)) for m in ("dtw", "linear")}}


def _ref_job(path):
    from piper_train.camera_ready.ckpt_io import load_checkpoint
    audio = load_checkpoint(path).numpy().squeeze()
    return features(audio)


def summarize(records, labels, resamples=10000, seed=2026):
    import numpy as np
    from scipy.stats import wilcoxon
    sys.path.insert(0, str(HERE))
    from ablation8_analysis import holm
    by = {(r["label"], r["utterance"]): r for r in records}
    n = max(r["utterance"] for r in records) + 1
    idx = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    out = {"models": {}, "comparisons": {}}
    for align in ("dtw", "linear"):
        for label in labels:
            row = {}
            for m in METRICS:
                v = np.array([by[(label, i)][align][m] if by[(label, i)][align][m] is not None else np.nan for i in range(n)])
                ok = ~np.isnan(v)
                boot = np.nanmean(np.where(ok[idx], v[idx], np.nan), axis=1)
                row[m] = {"mean": float(np.nanmean(v)), "ci95": [float(x) for x in np.nanpercentile(boot, [2.5, 97.5])],
                          "n": int(ok.sum())}
            out["models"].setdefault(align, {})[label] = row
        comps = []
        for kind, off, on in COMPARISONS:
            c = {"kind": kind, "off": off, "on": on}
            for m in METRICS:
                a = np.array([by[(off, i)][align][m] for i in range(n)], dtype=float)
                b = np.array([by[(on, i)][align][m] for i in range(n)], dtype=float)
                ok = ~(np.isnan(a) | np.isnan(b))
                d = (b - a)[ok]
                j = np.random.default_rng(seed).integers(0, len(d), size=(resamples, len(d)))
                lo, hi = np.percentile(d[j].mean(axis=1), [2.5, 97.5])
                try:
                    p = float(wilcoxon(b[ok], a[ok])[1])
                except ValueError:
                    p = None
                c[m] = {"mean_diff_on_minus_off": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(ok.sum()), "p_raw": p}
            comps.append(c)
        for m in METRICS:
            for c, q in zip(comps, holm([c[m]["p_raw"] for c in comps])):
                c[m]["p_holm"] = q
        out["comparisons"][align] = comps
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-dir", type=pathlib.Path, required=True)
    ap.add_argument("--eval-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import csv
    from multiprocessing import Pool
    import numpy as np
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
    for label in A.CELLS:
        fin = a.final_dir / A.RUNS[label]
        name = json.loads((fin / "final.json").read_text())["best_checkpoint"]
        sums = {p.split(None, 1)[1].strip(): p.split(None, 1)[0] for p in (fin / "SHA256SUMS").read_text().splitlines() if p}
        models.append((label, fin / name, sums[name]))
    models += HISTORICAL
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (a.out / "protocol.json").write_text(json.dumps({
        "tool": "tools/camera_ready/waveform_f0.py", "commit": commit, "argv": sys.argv,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": len(rows), "ids_first_last": [ids[0], ids[-1]],
        "harvest": HARVEST, "mfcc": MFCC, "scales": elh.SCALES, "seed_base": 2026, "threads": 4, "versions": elh.versions(),
        "reference": "audio_norm_path of the dataset rows (the MCD reference)", "comparisons": COMPARISONS,
        "models": [{"label": l, "checkpoint": str(p), "expected_sha256": h} for l, p, h in models]}, indent=1) + "\n")
    seed_of = lambda i: (2026 * 1_000_003 + i) % (2 ** 31)
    checks, jobs = [], []
    for label, ckpt, expected in models:
        if A.sha256(ckpt) != expected:
            raise SystemExit(f"{ckpt}: SHA-256 differs from the record")
        model, info = elh.load_model(ckpt)
        model.eval()
        with torch.no_grad():
            model.model_g.dec.remove_weight_norm()           # as the evaluation does
        d = a.out / "wavs" / label
        d.mkdir(parents=True)
        durs = []
        for i, r in enumerate(rows):
            torch.manual_seed(seed_of(i))
            audio, _sec, dur, _rtf = eh.synthesize(model, r["text"], "en-us", elh.SCALES, elh.SAMPLE_RATE)
            eh.write_wav(str(d / f"sent_{i:04d}.wav"), elh.SAMPLE_RATE, eh.audio_float_to_int16(audio))
            durs.append(dur)
            jobs.append((label, i, d / f"sent_{i:04d}.wav"))
        check = {"label": label, "epoch": info["epoch"], "semantics": info["group_semantics"]}
        evd = a.eval_dir / label
        if evd.is_dir():
            per = next(evd.glob("*_per_sentence.csv"))
            ev = [float(x["audio_duration_sec"]) for x in csv.DictReader(open(per, encoding="utf-8"))][:len(rows)]
            check["durations_equal_evaluation"] = sum(abs(x - y) < 1e-6 for x, y in zip(durs, ev))
            if (evd / "wavs").is_dir():
                check["wav_bytes_equal_evaluation"] = sum((evd / "wavs" / f"sent_{i:04d}.wav").read_bytes() ==
                                                          (d / f"sent_{i:04d}.wav").read_bytes() for i in range(len(rows)))
        checks.append(check)
        print(json.dumps(check), flush=True)
        if check.get("durations_equal_evaluation", len(rows)) != len(rows) or check.get("wav_bytes_equal_evaluation", len(rows)) != len(rows):
            raise SystemExit(f"{label}: reproduction check failed: {check}")
        del model
    with Pool(a.workers) as pool:
        ref_feats = pool.map(_ref_job, [r["audio_norm_path"] for r in rows], chunksize=4)
        allmf = np.concatenate([mf for _f0, mf in ref_feats])
        mean, std = allmf.mean(axis=0), allmf.std(axis=0)   # as compute_reference_metrics.compute_gt_mfcc_stats
        self_check = [_job_self(ref_feats[i], mean, std) for i in range(min(5, len(rows)))]
        records = pool.map(_job, [(l, i, w, ref_feats[i], mean, std) for l, i, w in jobs], chunksize=4)
    with open(a.out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarize(records, [m[0] for m in models])
    (a.out / "summary.json").write_text(json.dumps({"checks": checks, "reference_self_check": self_check, **summary,
                                                    "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    for label in [m[0] for m in models]:
        r = summary["models"]["dtw"][label]
        print(f"{label}: RMSE {r['rmse_cents']['mean']:.1f} c, corr {r['corr_logf0']['mean']:.3f}, VDE {r['vde']['mean']:.3f}, "
              f"GPE {r['gpe']['mean']:.3f}, offset {r['offset_cents']['mean']:+.1f} c")


def _job_self(ref_feat, mean, std):
    """Sanity check: a reference aligned with itself must give 0 cents and no voicing error."""
    f0, mf = ref_feat
    return pair_metrics(*aligned_pairs(f0, mf, f0, mf, mean, std, "dtw"))


if __name__ == "__main__":
    main()
