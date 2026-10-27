"""Final statistics of the camera-ready comparisons, from per-sentence files that exist (nothing is synthesized).

    python tools/camera_ready/ablation_final.py --eval-dir <CR_OUTPUTS>/eval --tests-dir <CR_OUTPUTS>/tests \
           --final-dir <CR_OUTPUTS>/final

Method, the manuscript's (section 4.1): per sentence the paired difference on - off; mean, 95 % BCa bootstrap CI
from 20,000 resamples (seed 1234), paired Cohen's d_z = mean / sd(ddof=1), two-sided Wilcoxon signed-rank (SciPy
defaults). Holm-adjusted p-values within each family below; the families were fixed in this analysis, after the
results existed:
  H    the paper's main pair, PitchFlowNet (epoch 1079) vs Piper (epoch 773), historical evaluation records
       (evaluation/eval_h_vs_baseline/): WER, UTMOSv2, MCD; Holm across the three.
  A    the 200-epoch ablation: the 12 cube edges and Full vs Baseline; Holm within each metric across the 13.
  P13  waveform F0 against the recordings (T13, DTW): the four F edges, Full vs Baseline, 1079 vs 773; Holm within
       each metric across the six.
  P14  pitch intervention (T14): slope per model, with a BCa CI; descriptive, no test.
RTF has no test: three benchmark rounds per cell, median and range. The inference parameter count of the historical
pair is recomputed from the checkpoints (weights_only) under the manuscript's definition.
Outputs: <eval-dir>/ablation-final-stats.{json,csv} and <eval-dir>/ablation-final-tables.md.
"""
import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "camera-ready-source" / "src" / "python"))

RESAMPLES, SEED = 20000, 1234
HIST_DIR = ROOT / "evaluation" / "eval_h_vs_baseline"
HIST = {"Piper-773": ("best-epoch=773-val_loss_mel=20.2158_per_sentence.csv", "mcd_baseline"),
        "PitchFlowNet-1079": ("best-epoch=1079-val_loss_mel=19.2943_per_sentence.csv", "mcd_configH")}
HIST_CKPT = {"Piper-773": pathlib.Path.home() / "02_Baseline_VanillaVITS/lightning_logs/version_0/checkpoints/best-epoch=773-val_loss_mel=20.2158.ckpt",
             "PitchFlowNet-1079": pathlib.Path.home() / "01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt"}
PAPER_PARAMS_M = {"Piper-773": 15.66, "PitchFlowNet-1079": 17.21}       # manuscript, Table final-results
P13_METRICS = ("rmse_cents", "corr_logf0", "vde", "gpe", "offset_cents")


def paired_stats(on, off, resamples=RESAMPLES, seed=SEED):
    """Mean of on - off, BCa CI, d_z and Wilcoxon p over the pairs where both are finite."""
    import numpy as np
    from scipy.stats import bootstrap, wilcoxon
    a, b = np.asarray(off, dtype=float), np.asarray(on, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    out = {"n": int(ok.sum()), "mean_off": float(a[ok].mean()), "mean_on": float(b[ok].mean()), "diff": float(d.mean())}
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    out["d_z"] = float(d.mean() / sd) if sd > 0 else None
    if sd > 0:
        ci = bootstrap((d,), np.mean, n_resamples=resamples, method="BCa", random_state=seed).confidence_interval
        out["ci95"] = [float(ci.low), float(ci.high)]
    else:
        out["ci95"] = [float(d.mean()), float(d.mean())]
    try:
        out["p_raw"] = float(wilcoxon(b[ok], a[ok])[1])
    except ValueError:
        out["p_raw"] = None
    return out


def add_holm(rows, family_key):
    """Holm within groups of rows that share row[family_key]; m is recorded."""
    from ablation8_analysis import holm
    groups = {}
    for r in rows:
        groups.setdefault(r[family_key], []).append(r)
    for g in groups.values():
        for r, q in zip(g, holm([r["p_raw"] for r in g])):
            r["p_holm"], r["holm_m"] = q, len(g)


def inference_params(state_dict):
    """The manuscript's count: the generator without the posterior encoder (enc_q), the posterior branch of the
    stochastic duration predictor (dp.post_*) and the flow the SDP drops at inference (dp.flows.1, "remove a useless
    vflow" in its reverse pass), with the decoder's weight norm folded (weight_g not counted)."""
    keep = lambda k: (k.startswith("model_g.") and not k.startswith("model_g.enc_q.") and not k.startswith("model_g.dp.post_")
                      and not k.startswith("model_g.dp.flows.1.") and not (k.startswith("model_g.dec.") and k.endswith("weight_g")))
    tensors = {k: v for k, v in state_dict.items() if keep(k)}
    return {"elements": int(sum(v.numel() for v in tensors.values())),
            "fp32_bytes": int(sum(v.numel() * 4 for v in tensors.values()))}


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=pathlib.Path, required=True)
    ap.add_argument("--tests-dir", type=pathlib.Path, required=True)
    ap.add_argument("--final-dir", type=pathlib.Path, required=True)
    a = ap.parse_args()
    import numpy as np
    import ablation8_analysis as A

    cells = A.load_all(a.eval_dir)
    ps = {label: A.per_sentence(d) for label, d in cells.items()}
    texts = [r["text"] for r in next(iter(cells.values()))["rows"]]
    hist = {}
    for label, (per, mcd) in HIST.items():
        rows, mrows = read_rows(HIST_DIR / per), read_rows(HIST_DIR / mcd / "mcd_gtmos.csv")
        if [r["text"] for r in rows] != texts or [r["text"] for r in mrows] != texts:
            raise SystemExit(f"{label}: the historical sentences differ from the ablation's or their order differs")
        hist[label] = {"wer": [float(r["wer"]) for r in rows], "utmos": [float(r["utmos"]) for r in rows],
                       "mcd": [float(r["mcd"]) for r in mrows]}
    rows = []
    # H: the main pair
    for metric in ("wer", "utmos", "mcd"):
        s = paired_stats(hist["PitchFlowNet-1079"][metric], hist["Piper-773"][metric])
        rows.append({"family": "H", "group": "H", "comparison": "Piper-773 -> PitchFlowNet-1079", "metric": metric, **s})
    # A: the ablation
    for factor, off, on in A.cube_edges():
        for metric in A.METRICS:
            s = paired_stats(ps[on][metric], ps[off][metric])
            rows.append({"family": "A", "group": f"A:{metric}", "comparison": f"{factor}: {off} -> {on}", "metric": metric, **s})
    # P13: waveform F0 (DTW), per-utterance records of T13
    import waveform_f0 as W
    rec = [json.loads(l) for l in (a.tests_dir / "waveform-f0-20261025" / "records.jsonl").read_text().splitlines() if l]
    by = {(r["label"], r["utterance"]): r["dtw"] for r in rec}
    n = max(r["utterance"] for r in rec) + 1
    for kind, off, on in W.COMPARISONS:
        for metric in P13_METRICS:
            val = lambda lab: [by[(lab, i)][metric] if by[(lab, i)][metric] is not None else np.nan for i in range(n)]
            s = paired_stats(val(on), val(off))
            rows.append({"family": "P13", "group": f"P13:{metric}", "comparison": f"{kind}: {off} -> {on}", "metric": metric, **s})
    for fam in ("H", "A", "P13"):
        add_holm([r for r in rows if r["family"] == fam], "group")
    # P14: pitch intervention slope per model (descriptive)
    p14 = {}
    pr = [json.loads(l) for l in (a.tests_dir / "pitch-shift-20261025" / "records.jsonl").read_text().splitlines() if l]
    for label in dict.fromkeys(r["label"] for r in pr):
        sl = [(r["shift"]["+2"] - r["shift"]["-2"]) / 4.0 for r in pr if r["label"] == label
              and r["shift"]["+2"] is not None and r["shift"]["-2"] is not None]
        s = paired_stats(sl, [0.0] * len(sl))
        p14[label] = {"n": s["n"], "slope": s["diff"], "ci95": s["ci95"]}
    # RTF of every cell (no test)
    rtf = {}
    for f in ("ablation8-rtf.json", "historical-pair-rtf.json", "historical-pair-rtf-t12.json"):
        for r in json.loads((a.eval_dir / f).read_text())["results"]:
            rtf.setdefault(r["label"], {})[str(r["threads"])] = {k: r[k] for k in ("model_corpus_rtf_median", "model_corpus_rtf_min",
                                                                                     "model_corpus_rtf_max", "e2e_corpus_rtf_median")}
    # parameter check of the historical pair
    from piper_train.camera_ready.ckpt_io import load_checkpoint
    params = {label: {**inference_params(load_checkpoint(p)["state_dict"]), "manuscript_M": PAPER_PARAMS_M[label]}
              for label, p in HIST_CKPT.items()}
    doc = {"method": {"difference": "per sentence, on - off", "ci": f"BCa, {RESAMPLES} resamples, seed {SEED}",
                      "effect_size": "paired Cohen's d_z = mean / sd(ddof=1)", "test": "two-sided Wilcoxon, SciPy defaults",
                      "families": {"H": "main pair, WER/UTMOSv2/MCD, Holm m=3", "A": "ablation, 13 pairs, Holm within metric",
                                   "P13": "waveform F0 (DTW), 6 comparisons, Holm within metric", "P14": "slopes, descriptive"},
                      "note": "families fixed after the results existed"},
           "comparisons": rows, "p14_slopes": p14, "rtf": rtf, "historical_params": params}
    (a.eval_dir / "ablation-final-stats.json").write_text(json.dumps(doc, indent=1) + "\n")
    fields = ["family", "comparison", "metric", "n", "mean_off", "mean_on", "diff", "ci_low", "ci_high", "d_z", "p_raw", "p_holm", "holm_m"]
    with open(a.eval_dir / "ablation-final-stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r.get(k) for k in fields}, "ci_low": r["ci95"][0], "ci_high": r["ci95"][1]})
    (a.eval_dir / "ablation-final-tables.md").write_text(tables(rows, p14, rtf, params))
    print(f"{len(rows)} comparisons; wrote ablation-final-stats.json/.csv and ablation-final-tables.md")


def _p(x):
    return "—" if x is None else (f"{x:.1e}" if x < 1e-3 else f"{x:.3f}")


def tables(rows, p14, rtf, params):
    short = lambda l: l.replace("PFN-", "").replace("-200ep-best", "").replace("Historical-", "Hist. ")
    scale = {"wer": (100.0, 2, "WER (pp)"), "utmos": (1.0, 3, "UTMOSv2"), "mcd": (1.0, 3, "MCD"), "rmse_cents": (1.0, 1, "F0 RMSE (cents)"),
             "corr_logf0": (1.0, 3, "log-F0 corr"), "vde": (1.0, 3, "VDE"), "gpe": (1.0, 3, "GPE"), "offset_cents": (1.0, 1, "F0 offset (cents)")}
    out = []

    def table(title, sel):
        out.append(f"\n#### {title}\n")
        out.append("| Comparison (off → on) | Metric | Mean off | Mean on | Diff (on − off) [95 % BCa CI] | d_z | p (raw) | p (Holm) | m |")
        out.append("|---|---|---:|---:|---|---:|---:|---:|---:|")
        for r in sel:
            k, nd, name = scale[r["metric"]]
            c = r["comparison"]
            for lab in sorted({x for x in c.replace(":", " ").split() if x.startswith(("PFN-", "Historical-"))}, key=len, reverse=True):
                c = c.replace(lab, short(lab))
            dz = "—" if r["d_z"] is None else f"{r['d_z']:+.3f}"
            out.append(f"| {c} | {name} | {r['mean_off']*k:.{nd}f} | {r['mean_on']*k:.{nd}f} | {r['diff']*k:+.{nd}f} "
                       f"[{r['ci95'][0]*k:+.{nd}f}, {r['ci95'][1]*k:+.{nd}f}] | {dz} | {_p(r['p_raw'])} | {_p(r['p_holm'])} | {r['holm_m']} |")

    table("H. Main pair: PitchFlowNet (epoch 1079) vs Piper (epoch 773), historical evaluation", [r for r in rows if r["family"] == "H"])
    for metric in ("utmos", "wer", "mcd"):
        table(f"A. Ablation, {scale[metric][2]}: 12 cube edges and Full vs Baseline (200 epochs)",
              [r for r in rows if r["family"] == "A" and r["metric"] == metric])
    table("P13. Waveform F0 against the recordings (DTW)", [r for r in rows if r["family"] == "P13"])
    out.append("\n#### P14. Pitch intervention: slope of the achieved over the requested shift (descriptive)\n")
    out.append("| Model | n | Slope [95 % BCa CI] |")
    out.append("|---|---:|---|")
    for lab, s in p14.items():
        out.append(f"| {short(lab)} | {s['n']} | {s['slope']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] |")
    out.append("\n#### RTF, CPU benchmark (model RTF, median [min, max] of three rounds)\n")
    out.append("| Model | 1 thread | 2 threads | 12 threads |")
    out.append("|---|---|---|---|")
    for lab, t in rtf.items():
        f = lambda th: "—" if th not in t else (f"{t[th]['model_corpus_rtf_median']:.4f} [{t[th]['model_corpus_rtf_min']:.4f}, "
                                                f"{t[th]['model_corpus_rtf_max']:.4f}]")
        out.append(f"| {short(lab)} | {f('1')} | {f('2')} | {f('12')} |")
    out.append("\n#### Inference parameters of the historical pair, recomputed\n")
    out.append("| Model | Elements | Millions | Manuscript (M) | FP32 bytes (MB, 10^6) |")
    out.append("|---|---:|---:|---:|---:|")
    for lab, p in params.items():
        out.append(f"| {lab} | {p['elements']:,} | {p['elements']/1e6:.2f} | {p['manuscript_M']:.2f} | {p['fp32_bytes']/1e6:.2f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
