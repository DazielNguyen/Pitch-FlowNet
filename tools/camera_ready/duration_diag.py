"""Duration diagnostics of the eight ablation checkpoints (2026-10-24). No audio is written.

    python tools/camera_ready/duration_diag.py --final-dir <CR_OUTPUTS>/final --eval-dir <CR_OUTPUTS>/eval \
           --out <CR_OUTPUTS>/tests/duration-diag-<YYYYMMDD>

Only the text encoder, the duration predictor and (for 2) the posterior encoder, flow and MAS run; not the decoder.
  1. Predicted durations of the 500 test texts for the best and the last checkpoint of every run, with
     noise_scale_w 0.8 seeded per utterance as the evaluation. For the best checkpoint this must reproduce the
     evaluation's audio_duration_sec exactly; that is checked.
  2. Predicted durations against the MAS durations of the 100 validation utterances (teacher forced: posterior
     encoder on the cached spectrogram, flow, MAS without noise), best checkpoint.
  3. As 1 and 2 with noise_scale_w 0 (the stochastic duration predictor without its noise).
Reference lengths are the frames of the cached spectrograms. Nothing here selects a checkpoint: the checkpoints are
the fixed best (minimum val_mel_l1_global) and the last (epoch 199) of each run.
"""
import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "camera-ready-source" / "src" / "python"))

NOISE_W = (0.8, 0.0)
SEED_BASE = 2026


def seed_of(i: int) -> int:
    return (SEED_BASE * 1_000_003 + i) % (2 ** 31)          # the evaluation's per-utterance synthesis seed


def predicted_frames(model_g, ids, noise_w, seed):
    """Frames per phoneme exactly as SynthesizerTrn.infer computes them (length_scale 1)."""
    import torch
    torch.manual_seed(seed)
    x = torch.LongTensor(ids).unsqueeze(0)
    x, _m, _logs, x_mask = model_g.enc_p(x, torch.LongTensor([len(ids)]))
    if model_g.use_sdp:
        logw = model_g.dp(x, x_mask, g=None, reverse=True, noise_scale=noise_w)
    else:
        logw = model_g.dp(x, x_mask, g=None)
    logw = torch.nan_to_num(logw, nan=0.0, posinf=6.0, neginf=-6.0).clamp(-6, 6)
    w = torch.nan_to_num(torch.exp(logw) * x_mask, nan=1.0, posinf=403.0, neginf=0.0)
    return torch.ceil(w).clamp(min=0, max=500)[0, 0].long().tolist()


def mas_frames(model_g, ids, spec, seed):
    """MAS durations per phoneme on the real spectrogram, as the training forward computes them, MAS noise off."""
    import torch
    from piper_train.vits import monotonic_align
    torch.manual_seed(seed)
    x = torch.LongTensor(ids).unsqueeze(0)
    x, m_p, logs_p, x_mask = model_g.enc_p(x, torch.LongTensor([len(ids)]))
    y = spec.unsqueeze(0)
    z, _m_q, _logs_q, y_mask = model_g.enc_q(y, torch.LongTensor([spec.shape[-1]]))
    z_p = model_g.flow(z, y_mask)
    s_p_sq_r = torch.exp(-2 * logs_p)
    neg_cent = (torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
                + torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
                + torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
                + torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True))
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
    return attn.sum(1)[0].long().tolist()


def summarize(records, ref_frames):
    """Per (label, checkpoint, split, noise_w): total predicted / reference frames, median per-utterance ratio and,
    where MAS durations exist, the mean absolute log ratio per phoneme (predicted against MAS)."""
    rows = []
    for r in records:
        pred = [sum(p) for p in r["pred"]]
        ref = ref_frames[r["split"]]
        row = {"label": r["label"], "checkpoint": r["checkpoint"], "epoch": r["epoch"], "split": r["split"],
               "noise_w": r["noise_w"], "n": len(pred), "total_ratio_to_reference": sum(pred) / sum(ref),
               "median_utt_ratio": statistics.median(p / q for p, q in zip(pred, ref))}
        if r.get("mas"):
            logs = [abs(math.log(max(a, 1) / max(b, 1))) for p, m in zip(r["pred"], r["mas"]) for a, b in zip(p, m)]
            row["mean_abs_log_ratio_to_mas"] = statistics.fmean(logs)
            row["total_ratio_to_mas"] = sum(pred) / sum(sum(m) for m in r["mas"])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-dir", type=pathlib.Path, required=True)
    ap.add_argument("--eval-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    import ablation8_analysis as A
    import evaluate_like_history as elh
    from piper_train import eval_harness as eh
    from piper_train.camera_ready.ckpt_io import load_checkpoint

    if a.out.exists():
        raise SystemExit(f"{a.out} exists; use a new folder")
    a.out.mkdir(parents=True)
    test_rows, test_ids, _ = elh.split_rows("test", 0)
    val_rows, val_ids, _ = elh.split_rows("validation", 0)
    specs = {k: [load_checkpoint(r["audio_spec_path"]) for r in rows] for k, rows in (("test", test_rows), ("validation", val_rows))}
    ref_frames = {k: [int(s.shape[-1]) for s in v] for k, v in specs.items()}
    test_phonemes = [eh.text_to_phoneme_ids(r["text"], "en-us") for r in test_rows]   # as synthesize() does
    records, checks = [], []
    for label in A.CELLS:
        fin = a.final_dir / A.RUNS[label]
        final = json.loads((fin / "final.json").read_text())
        sums = {p.split(None, 1)[1].strip(): p.split(None, 1)[0] for p in (fin / "SHA256SUMS").read_text().splitlines() if p}
        for which, name in (("best", final["best_checkpoint"]), ("last", "last-epoch=199.ckpt")):
            ckpt = fin / name
            if A.sha256(ckpt) != sums.get(name):
                raise SystemExit(f"{ckpt}: SHA-256 differs from SHA256SUMS")
            model, info = elh.load_model(ckpt)
            model.eval()
            g = model.model_g
            g.mas_noise_scale = 0.0
            hop = int(model.hparams.hop_length)
            with torch.no_grad():
                for nw in NOISE_W:
                    pred = [predicted_frames(g, ids, nw, seed_of(i)) for i, ids in enumerate(test_phonemes)]
                    records.append({"label": label, "checkpoint": which, "epoch": info["epoch"], "split": "test",
                                    "noise_w": nw, "pred": pred})
                    if which == "best" and nw == 0.8:
                        per = next((a.eval_dir / label).glob("*_per_sentence.csv"))
                        ev = [float(r["audio_duration_sec"]) for r in csv.DictReader(open(per, encoding="utf-8"))]
                        same = sum(abs(sum(p) * hop / elh.SAMPLE_RATE - e) < 1e-6 for p, e in zip(pred, ev))
                        checks.append({"label": label, "check": "best, noise_w 0.8 reproduces the evaluation durations",
                                       "matching": same, "of": len(ev)})
                    if which == "best":
                        vp = [predicted_frames(g, r["phoneme_ids"], nw, seed_of(10 ** 7 + i)) for i, r in enumerate(val_rows)]
                        vm = [mas_frames(g, r["phoneme_ids"], s, seed_of(2 * 10 ** 7 + i)) for i, (r, s) in enumerate(zip(val_rows, specs["validation"]))]
                        records.append({"label": label, "checkpoint": which, "epoch": info["epoch"], "split": "validation",
                                        "noise_w": nw, "pred": vp, "mas": vm})
            print(f"{label} {which} (epoch {info['epoch']}) done", flush=True)
    with open(a.out / "durations.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    rows = summarize(records, ref_frames)
    with open(a.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    (a.out / "summary.json").write_text(json.dumps({
        "tool": "tools/camera_ready/duration_diag.py", "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": "frames of the cached spectrograms (audio_spec_path) of each split",
        "test_ids_first_last": [test_ids[0], test_ids[-1]], "validation_n": len(val_ids),
        "checks": checks, "rows": rows}, indent=1) + "\n")
    bad = [c for c in checks if c["matching"] != c["of"]]
    print(json.dumps(checks))
    if bad:
        raise SystemExit(f"{len(bad)} labels do not reproduce the evaluation durations")


if __name__ == "__main__":
    main()
