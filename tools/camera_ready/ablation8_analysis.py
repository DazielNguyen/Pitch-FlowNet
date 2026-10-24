"""Analyse the eight 200-epoch ablation runs (2^3 cells of B = BigVGAN, V = VITS2, F = F0) from scores that exist.

    python tools/camera_ready/ablation8_analysis.py all --eval-dir <CR_OUTPUTS>/eval --final-dir <CR_OUTPUTS>/final
           [--seed 2026] [--resamples 10000]
    subcommands: check | table | pairs | effects | cost | all   (same options)

Nothing is synthesized, transcribed or re-scored. The inputs are, per label, the files written by
evaluate_like_history.py evaluate (summary.json, protocol.json, <ckpt>_per_sentence.csv with WER and UTMOSv2,
mcd/mcd_gtmos.csv with MCD) and, per run, final/<run_id>/ (SHA256SUMS, final.json, run_manifest.json, ledger).
Checkpoints are opened only for the parameter count, with torch.load(weights_only=True) via camera_ready/ckpt_io.py.

  check    consistency of the eight evaluations (sizes, split, checkpoint hashes, sentences and their order,
           protocol settings, reference UTMOSv2) and the config differences between cube neighbours; stops on a failure
  table    one row per label: bits, best epoch, WER, UTMOSv2, MCD, F0, recorded RTF, training GPU
  pairs    the 12 cube edges and Full vs Baseline: paired difference (on - off), paired-bootstrap 95 % CI of the
           mean difference, two-sided Wilcoxon signed-rank (SciPy defaults, as compare), Holm within each metric
  effects  factorial effects in effect coding, Shapley attribution of the Full - Baseline gap, F0 metrics as 2x2
  cost     stored tensor elements, checkpoint size, training time per machine, recorded RTF, Pareto sets

UTMOSv2 is a predicted score, not MOS. A non-significant difference does not establish equivalence. One training
seed per cell: the sentence-level bootstrap and tests do not cover training variance.
Outputs: <eval-dir>/ablation8-*.{json,csv,md}.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "camera-ready-source" / "src" / "python"
sys.path.insert(0, str(SRC))

# label -> (B, V, F); run_id of each label. The Full reference ran with the 275-epoch config, stopped at epoch 200.
CELLS = {
    "PFN-Full-200ep-best": (1, 1, 1), "PFN-NoF0-200ep-best": (1, 1, 0), "PFN-NoVITS2-200ep-best": (1, 0, 1),
    "PFN-NoBigVGAN-200ep-best": (0, 1, 1), "PFN-Baseline-BigVGAN-200ep-best": (1, 0, 0),
    "PFN-Baseline-VITS2-200ep-best": (0, 1, 0), "PFN-Baseline-F0-200ep-best": (0, 0, 1), "PFN-Baseline-200ep-best": (0, 0, 0),
}
RUNS = {l: ("PFN-Full-275ep-s1234" if l == "PFN-Full-200ep-best" else l.replace("-200ep-best", "-200ep-s1234")) for l in CELLS}
BY_BITS = {bits: label for label, bits in CELLS.items()}
FACTORS = ("B", "V", "F")
METRICS = ("wer", "utmos", "mcd")          # lower is better for wer and mcd, higher for utmos
HIGHER_BETTER = {"wer": False, "utmos": True, "mcd": False}
REF_UTMOS, REF_TOL = 3.8308, 5e-4
TREE_SHA256 = "1ee13cb4efe39da5657398021f3b85eb3a2a3eb5b89a1f1392dc33c4dfe90a52"
# protocol.json keys that must be identical for all labels (platform, argv, paths and times may differ)
SAME_PROTOCOL = ("tool", "split", "n", "ids_first_last", "manifest_sha256", "dataset_sha256", "scales", "sample_rate",
                 "whisper_model", "devices", "seed", "threads", "mcd", "f0_predictor_metric", "versions", "python",
                 "differences_from_historical")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cube_edges():
    """The 12 edges of the cube as (factor, off_label, on_label), plus Full vs Baseline as ("BVF", ...)."""
    edges = []
    for k, name in enumerate(FACTORS):
        for rest in itertools.product((0, 1), repeat=2):
            off, on = list(rest), list(rest)
            off.insert(k, 0), on.insert(k, 1)
            edges.append((name, BY_BITS[tuple(off)], BY_BITS[tuple(on)]))
    return edges + [("BVF", BY_BITS[(0, 0, 0)], BY_BITS[(1, 1, 1)])]


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order (None stays None)."""
    idx = sorted((i for i, p in enumerate(pvalues) if p is not None), key=lambda i: pvalues[i])
    m, out, running = len(idx), [None] * len(pvalues), 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


def bootstrap_indices(n: int, resamples: int, seed: int):
    import numpy as np
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def mean_ci(values, idx):
    """Mean of per-sentence values and the percentile 95 % CI of the mean over the bootstrap index matrix."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    boot = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def contrast(bits_of_cell, term):
    """Effect-coding sign (+1/-1) of a cell for a term such as "B", "BV" or "BVF"."""
    s = 1
    for name in term:
        s *= 1 if bits_of_cell[FACTORS.index(name)] else -1
    return s


EFFECT_TERMS = ("B", "V", "F", "BV", "BF", "VF", "BVF")


def factorial_effects(y):
    """y: {bits: per-sentence array}. Per-sentence effects in the classical 2^3 definition:
    effect(term) = (1/4) * sum over the 8 cells of sign(term, cell) * y(cell); a main effect is then
    mean(y | factor on) - mean(y | factor off), and an interaction is half the change of the lower-order effect."""
    import numpy as np
    return {t: sum(contrast(b, t) * np.asarray(v, dtype=float) for b, v in y.items()) / 4.0 for t in EFFECT_TERMS}


def shapley(y):
    """Exact Shapley values of B, V, F for the value v(S) = y(cell with the components in S on).
    They sum to y(1,1,1) - y(0,0,0). Works on scalars or per-sentence arrays."""
    n, out = len(FACTORS), {}
    for k, name in enumerate(FACTORS):
        total = 0
        for rest in itertools.product((0, 1), repeat=n - 1):
            size = sum(rest)
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            off, on = list(rest), list(rest)
            off.insert(k, 0), on.insert(k, 1)
            total = total + weight * (y[tuple(on)] - y[tuple(off)])
        out[name] = total
    return out


def pareto(points, minimise):
    """Names of the non-dominated points. points: {name: (x1, x2, ...)}; minimise: per coordinate True/False."""
    def better_eq(a, b):
        return all((ai <= bi) if m else (ai >= bi) for ai, bi, m in zip(a, b, minimise))
    return sorted(n for n, p in points.items()
                  if not any(o != n and better_eq(q, p) and q != p for o, q in points.items()))


# ---------------------------------------------------------------- inputs

def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_label(eval_dir: pathlib.Path, label: str) -> dict:
    d = eval_dir / label
    per = list(d.glob("*_per_sentence.csv"))
    if len(per) != 1:
        raise SystemExit(f"{d}: expected one per-sentence CSV, found {len(per)}")
    return {"dir": d, "summary": json.loads((d / "summary.json").read_text()),
            "protocol": json.loads((d / "protocol.json").read_text()),
            "per_csv": per[0], "rows": read_csv(per[0]), "mcd_rows": read_csv(d / "mcd" / "mcd_gtmos.csv"),
            "reference_sha256": sha256(d / "reference.jsonl")}


def per_sentence(data: dict) -> dict:
    return {"wer": [float(r["wer"]) for r in data["rows"]], "utmos": [float(r["utmos"]) for r in data["rows"]],
            "mcd": [float(r["mcd"]) for r in data["mcd_rows"]]}


def load_all(eval_dir):
    return {label: load_label(eval_dir, label) for label in CELLS}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, doc):
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- check

def cmd_check(a, data=None):
    data = data or load_all(a.eval_dir)
    checks, failed = [], []

    def ok(name, label, cond, detail=""):
        checks.append({"check": name, "label": label, "ok": bool(cond), "detail": detail})
        if not cond:
            failed.append(f"{label}: {name} {detail}")

    first = next(iter(CELLS))
    ref_texts = [r["text"] for r in data[first]["rows"]]
    ref_gt = [float(r["gt_utmos"]) for r in data[first]["mcd_rows"]]
    for label, bits in CELLS.items():
        d, s, p = data[label], data[label]["summary"], data[label]["protocol"]
        fin = a.final_dir / RUNS[label]
        sums = dict(reversed(line.split(None, 1)) for line in (fin / "SHA256SUMS").read_text().splitlines() if line)
        sums = {k.strip(): v for k, v in sums.items()}
        final = json.loads((fin / "final.json").read_text())
        best = pathlib.Path(p["checkpoint"]).name
        ok("num_sentences 500", label, s["num_sentences"] == 500 and len(d["rows"]) == 500)
        ok("mcd n 500, n_skipped 0", label, s["mcd"]["n"] == 500 and s["mcd"]["n_skipped"] == 0 and len(d["mcd_rows"]) == 500)
        ok("split test", label, p["split"] == "test")
        ok("checkpoint is the final best", label, best == final["best_checkpoint"] and best.startswith("best-epoch="), best)
        ok("checkpoint_sha256 equals final SHA256SUMS", label, p["checkpoint_sha256"] == sums.get(best))
        ok("final: 200 epochs, 156,400 G updates, training tree", label,
           final["epochs_completed"] == 200 and final["g_updates"] == 156400 and final["source_tree_sha256"] == TREE_SHA256)
        ok("checkpoint flags equal the cell", label,
           tuple(int(bool(p["checkpoint_info"]["flags"][k])) for k in ("use_bigvgan", "use_vits2", "use_f0")) == bits)
        ok("strict group semantics, seeds 1234/1234/2026", label, p["checkpoint_info"]["group_semantics"] == "strict" and
           {k: p["checkpoint_info"]["camera_ready"][k] for k in ("train_seed", "data_seed", "val_seed")} ==
           {"train_seed": 1234, "data_seed": 1234, "val_seed": 2026})
        ok("reference UTMOSv2 3.8308", label, abs(s["mcd"]["mean_gt_utmos"] - REF_UTMOS) < REF_TOL, f'{s["mcd"]["mean_gt_utmos"]:.6f}')
        ok("same sentences in the same order (per-sentence CSV)", label, [r["text"] for r in d["rows"]] == ref_texts)
        ok("same sentences in the same order (MCD CSV)", label,
           [r["text"] for r in d["mcd_rows"]] == ref_texts and [int(r["idx"]) for r in d["mcd_rows"]] == list(range(500)))
        ok("reference.jsonl identical", label, d["reference_sha256"] == data[first]["reference_sha256"])
        gt_diff = max(abs(float(r["gt_utmos"]) - g) for r, g in zip(d["mcd_rows"], ref_gt))
        ok("per-sentence reference UTMOSv2 identical (CSV, 4 decimals)", label, gt_diff <= 1e-4, f"max |diff| {gt_diff:.4g}")
        same = [k for k in SAME_PROTOCOL if p.get(k) != data[first]["protocol"].get(k)]
        ok("protocol settings identical", label, not same, ", ".join(same))
        ps = per_sentence(d)
        ok("summary means equal the per-sentence means", label,
           abs(sum(ps["wer"]) / 500 - s["mean_wer"]) < 1e-9 and abs(sum(ps["utmos"]) / 500 - s["mean_utmos"]) < 1e-9 and
           abs(sum(ps["mcd"]) / 500 - s["mcd"]["mean_mcd"]) < 1e-3)

    # The test sentences are the test split of the split manifest, in its order.
    try:
        from piper_train.camera_ready.split import load_split_manifest
        dataset = pathlib.Path(data[first]["protocol"]["dataset_jsonl"])
        m = load_split_manifest(ROOT / "manifests" / "camera-ready" / "split-manifest.json", dataset)
        rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
        refs = [json.loads(line) for line in (data[first]["dir"] / "reference.jsonl").read_text().splitlines() if line]
        ok("reference audio = test split of split-manifest.json, in order", "all",
           [rows[r]["audio_norm_path"] for r in m.rows["test"]] == [x["audio_norm_path"] for x in refs])
        test_ids = list(m.ids["test"])
    except (OSError, KeyError, ValueError) as exc:
        ok("reference audio = test split of split-manifest.json, in order", "all", False, repr(exc))
        test_ids = None

    tool = ROOT / "tools" / "camera_ready" / "evaluate_like_history.py"
    log = subprocess.run(["git", "log", "--format=%h", "--", str(tool)], cwd=ROOT, capture_output=True, text=True).stdout.split()
    ok("evaluation tool has a single version in Git", "all", len(log) == 1, " ".join(log))

    configs = {label: flatten(json.loads((a.final_dir / RUNS[label] / "run_manifest.json").read_text())["config"]) for label in CELLS}
    expected = {"run_id", "device_note", "tier", "notes", "note", "groups.use_bigvgan", "groups.use_vits2", "groups.use_f0",
                "model.use_bigvgan", "model.use_vits2", "model.use_f0", "protocol.max_g_updates", "protocol.budget_label"}
    config_diffs = []
    for factor, off, on in cube_edges():
        keys = sorted(k for k in set(configs[off]) | set(configs[on]) if configs[off].get(k) != configs[on].get(k))
        config_diffs.append({"factor": factor, "off": off, "on": on, "differing_keys": keys,
                             "unexpected": [k for k in keys if k not in expected and not k.startswith("groups.")]})
    full_budget = {k: configs["PFN-Full-200ep-best"].get(k) for k in ("protocol.max_g_updates", "trainer.max_epochs")}

    env = {}
    for label in CELLS:
        man = json.loads((a.final_dir / RUNS[label] / "run_manifest.json").read_text())
        env[label] = {"gpu": man["devices"]["torch"][0]["name"], "source_commit": man["source"]["git"]["head"][:7],
                      "dirty_paths": man["source"]["git"]["dirty_paths"], "tree_sha256": man["source"]["tree_sha256"],
                      "torch": man["environment"].get("torch"), "attempts": len(man["attempts"]),
                      "eval_platform": data[label]["protocol"]["platform"]}

    doc = {"labels": RUNS, "cells": {k: dict(zip(FACTORS, v)) for k, v in CELLS.items()}, "checks": checks,
           "failed": failed, "test_ids_first_last": [test_ids[0], test_ids[-1]] if test_ids else None,
           "evaluation_tool": {"path": "tools/camera_ready/evaluate_like_history.py", "commits": log, "sha256": sha256(tool)},
           "config_differences_between_cube_neighbours": config_diffs,
           "full_reference_budget": {**full_budget, "note": "the Full run used its 275-epoch config and stopped at epoch 200 "
                                     "(--stop-after-epochs 200); see the budget-dependence note in ablation8-results.md"},
           "runs": env}
    write_json(a.eval_dir / "ablation8-check.json", doc)
    print(f"checks: {len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        raise SystemExit("FAILED:\n  " + "\n  ".join(failed))
    return doc


# ---------------------------------------------------------------- table

def cmd_table(a, data=None):
    data = data or load_all(a.eval_dir)
    rows = []
    for label, (b, v, f) in sorted(CELLS.items(), key=lambda kv: kv[1], reverse=True):
        s, p = data[label]["summary"], data[label]["protocol"]
        man = json.loads((a.final_dir / RUNS[label] / "run_manifest.json").read_text())
        rows.append({"label": label, "B": b, "V": v, "F": f, "best_epoch": p["checkpoint_info"]["epoch"],
                     "wer_mean": s["mean_wer"], "wer_median": s["median_wer"], "utmos_mean": s["mean_utmos"],
                     "utmos_median": s["median_utmos"], "mcd_mean": s["mcd"]["mean_mcd"], "mcd_median": s["mcd"]["median_mcd"],
                     "f0_rmse": s.get("f0_rmse"), "f0_corr": s.get("f0_corr"), "mean_rtf_recorded": s["mean_rtf"],
                     "training_gpu": man["devices"]["torch"][0]["name"],
                     "source": f"eval/{label}/summary.json, protocol.json; final/{RUNS[label]}/run_manifest.json"})
    write_csv(a.eval_dir / "ablation8-table.csv", rows)
    fmt = lambda x, n: "n/a" if x is None else f"{x:.{n}f}"
    md = ["| Label | B | V | F | Best epoch | WER mean | WER median | UTMOSv2 mean | UTMOSv2 median | MCD mean | F0 RMSE | F0 corr | Recorded RTF | Training GPU |",
          "|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        md.append(f"| `{r['label']}` | {r['B']} | {r['V']} | {r['F']} | {r['best_epoch']} | {fmt(r['wer_mean'], 4)} | "
                  f"{fmt(r['wer_median'], 4)} | {fmt(r['utmos_mean'], 4)} | {fmt(r['utmos_median'], 4)} | {fmt(r['mcd_mean'], 3)} | "
                  f"{fmt(r['f0_rmse'], 4)} | {fmt(r['f0_corr'], 4)} | {fmt(r['mean_rtf_recorded'], 4)} | {r['training_gpu']} |")
    (a.eval_dir / "ablation8-table.md").write_text("\n".join(md) + "\n")
    return rows


# ---------------------------------------------------------------- pairs

def find_compare(eval_dir, la, lb):
    for x, y in ((la, lb), (lb, la)):
        f = eval_dir / f"compare-{x}_{y}.json"
        if f.is_file():
            for t in json.loads(f.read_text())["paired_tests"]:
                if {t["a"], t["b"]} == {la, lb}:
                    return f.name, t
    return None, None


def cmd_pairs(a, data=None):
    import numpy as np
    from scipy.stats import wilcoxon
    data = data or load_all(a.eval_dir)
    ps = {label: per_sentence(d) for label, d in data.items()}
    idx = bootstrap_indices(500, a.resamples, a.seed)
    out = []
    for factor, off, on in cube_edges():
        cfile, ctest = find_compare(a.eval_dir, off, on)
        for metric in METRICS:
            diff = np.asarray(ps[on][metric]) - np.asarray(ps[off][metric])
            mean, lo, hi = mean_ci(diff, idx)
            try:
                stat, p = (float(x) for x in wilcoxon(ps[on][metric], ps[off][metric]))
            except ValueError:
                stat, p = None, None
            agree = None
            if ctest is not None and metric in ("wer", "utmos"):
                agree = ctest.get(f"{metric}_wilcoxon_p") is not None and abs(ctest[f"{metric}_wilcoxon_p"] - p) <= 1e-12 * max(1.0, p)
            out.append({"factor": factor, "off": off, "on": on, "metric": metric, "n": 500,
                        "mean_off": float(np.mean(ps[off][metric])), "mean_on": float(np.mean(ps[on][metric])),
                        "mean_diff_on_minus_off": mean, "ci95_low": lo, "ci95_high": hi, "median_diff": float(np.median(diff)),
                        "wilcoxon_stat": stat, "p_raw": p, "compare_file": cfile, "compare_p_agrees": agree})
    for metric in METRICS:
        rows = [r for r in out if r["metric"] == metric]
        for r, q in zip(rows, holm([r["p_raw"] for r in rows])):
            r["p_holm"] = q
    write_csv(a.eval_dir / "ablation8-pairs.csv", out)
    write_json(a.eval_dir / "ablation8-pairs.json", {
        "method": {"difference": "per sentence, on - off (for Full vs Baseline: Full - Baseline)",
                   "ci": f"percentile 95 % CI of the mean difference, paired bootstrap over sentences, {a.resamples} resamples, seed {a.seed}",
                   "test": "two-sided Wilcoxon signed-rank, SciPy defaults (zero differences dropped), as evaluate_like_history.py compare",
                   "multiplicity": "Holm within each metric across the 13 pairs",
                   "compare_files": "WER and UTMOSv2 p-values are checked against compare-*.json written by evaluate_like_history.py compare"},
        "pairs": out})
    missing = [r for r in out if r["metric"] != "mcd" and r["compare_p_agrees"] is not True]
    if missing:
        raise SystemExit(f"{len(missing)} WER/UTMOSv2 tests have no matching compare-*.json p-value: "
                         + ", ".join(sorted({f'{r["off"]}/{r["on"]}' for r in missing})))
    return out


# ---------------------------------------------------------------- effects

def cmd_effects(a, data=None):
    import numpy as np
    data = data or load_all(a.eval_dir)
    idx = bootstrap_indices(500, a.resamples, a.seed)
    doc, rows = {"effects": {}, "shapley": {}, "f0_2x2": {}}, []
    for metric in METRICS:
        y = {bits: np.asarray(per_sentence(data[label])[metric]) for label, bits in CELLS.items()}
        for term, v in factorial_effects(y).items():
            mean, lo, hi = mean_ci(v, idx)
            doc["effects"].setdefault(metric, {})[term] = {"effect": mean, "ci95_low": lo, "ci95_high": hi}
            rows.append({"kind": "factorial effect", "metric": metric, "term": term, "value": mean, "ci95_low": lo,
                         "ci95_high": hi, "pct_of_gap": None})
        gap = y[(1, 1, 1)] - y[(0, 0, 0)]
        gap_mean, gap_lo, gap_hi = mean_ci(gap, idx)
        sh = shapley(y)
        doc["shapley"][metric] = {"gap_full_minus_baseline": {"value": gap_mean, "ci95_low": gap_lo, "ci95_high": gap_hi}}
        for name, v in sh.items():
            mean, lo, hi = mean_ci(v, idx)
            pct = 100.0 * mean / gap_mean if gap_mean else None
            doc["shapley"][metric][name] = {"value": mean, "ci95_low": lo, "ci95_high": hi, "pct_of_gap": pct}
            rows.append({"kind": "shapley", "metric": metric, "term": name, "value": mean, "ci95_low": lo, "ci95_high": hi,
                         "pct_of_gap": pct})
        rows.append({"kind": "gap Full - Baseline", "metric": metric, "term": "BVF", "value": gap_mean, "ci95_low": gap_lo,
                     "ci95_high": gap_hi, "pct_of_gap": 100.0})
        total = sum(doc["shapley"][metric][n]["value"] for n in FACTORS)
        if abs(total - gap_mean) > 1e-9 * max(1.0, abs(gap_mean)):
            raise SystemExit(f"Shapley values of {metric} do not sum to the gap")
    for b, v in itertools.product((1, 0), repeat=2):
        s = data[BY_BITS[(b, v, 1)]]["summary"]
        doc["f0_2x2"][f"B={b},V={v}"] = {"label": BY_BITS[(b, v, 1)], "f0_rmse": s["f0_rmse"], "f0_corr": s["f0_corr"]}
    doc["method"] = {
        "coding": "effect coding, -1 = off, +1 = on; effect(term) = (1/4) sum_cells sign(term, cell) * y(cell), per sentence",
        "reading": "a main effect is mean(on) - mean(off), averaged over the other two factors; a two-way interaction "
                   "X x Y is half the difference between the effect of X with Y on and with Y off",
        "shapley": "exact over the 8 cells; v(S) = mean metric of the cell with the components in S on; sums to Full - Baseline",
        "ci": f"percentile 95 % CI, paired bootstrap over sentences, {a.resamples} resamples, seed {a.seed}",
        "f0": "F0-predictor RMSE and correlation exist only for F = 1 and only as summary values (no per-sentence values), "
              "so they are reported as a 2x2 over B and V without a CI or test"}
    write_json(a.eval_dir / "ablation8-effects.json", doc)
    write_csv(a.eval_dir / "ablation8-effects.csv", rows)
    return doc


# ---------------------------------------------------------------- cost

def parse_utc(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def training_cost(final_dir: pathlib.Path, run_id: str) -> dict:
    """Wall-clock and active training time from run_manifest.json attempts and the checkpoint ledger.
    An attempt without an end time (it died) ends at its last checkpoint event (save or remove), not at the
    watchdog's auto_resume entry, which is written after the wait; the lost partial epoch is not counted."""
    man = json.loads((final_dir / run_id / "run_manifest.json").read_text())
    ledger = [json.loads(line) for line in (final_dir / run_id / "checkpoint_ledger.jsonl").read_text().splitlines() if line]
    times = sorted(parse_utc(r["time_utc"]) for r in ledger if r.get("event") in ("save", "remove"))
    starts = [parse_utc(x["start_utc"]) for x in man["attempts"]]
    active = 0.0
    for i, st in enumerate(starts):
        a_end = man["attempts"][i].get("end_utc")
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        end = parse_utc(a_end) if a_end else max((t for t in times if t >= st and (nxt is None or t < nxt)), default=st)
        active += (end - st).total_seconds()
    end_all = parse_utc(man["attempts"][-1]["end_utc"])
    saves = [(r["epoch"], parse_utc(r["time_utc"])) for r in ledger if r.get("file") == "last.ckpt" and r.get("event") == "save"]
    # epoch durations from consecutive last.ckpt saves of the same attempt (none across an interruption)
    durations = sorted((t2 - t1).total_seconds() / 60 for (e1, t1), (e2, t2) in zip(saves, saves[1:])
                       if e2 == e1 + 1 and not any(t1 < st <= t2 for st in starts))
    auto = sum(1 for r in ledger if r.get("event") == "auto_resume")
    return {"gpu": man["devices"]["torch"][0]["name"], "attempts": len(starts), "auto_resumes": auto,
            "other_resumes": len(starts) - 1 - auto, "wall_clock_h": (end_all - starts[0]).total_seconds() / 3600,
            "active_gpu_h": active / 3600, "g_updates_per_s": 156400 / active,
            "median_epoch_min": durations[len(durations) // 2] if durations else None}


def model_size(ckpt: pathlib.Path) -> dict:
    from piper_train.camera_ready.ckpt_io import load_checkpoint
    sd = load_checkpoint(ckpt)["state_dict"]
    count = lambda pred: int(sum(v.numel() for k, v in sd.items() if pred(k)))
    return {"elements_inference_path": count(lambda k: k.startswith("model_g.") and not k.startswith("model_g.enc_q.")),
            "elements_generator": count(lambda k: k.startswith("model_g.")),
            "elements_all": count(lambda k: True), "checkpoint_bytes": ckpt.stat().st_size}


def cmd_cost(a, data=None):
    data = data or load_all(a.eval_dir)
    rows = []
    for label, (b, v, f) in sorted(CELLS.items(), key=lambda kv: kv[1], reverse=True):
        ckpt = a.final_dir / RUNS[label] / pathlib.Path(data[label]["protocol"]["checkpoint"]).name
        s = data[label]["summary"]
        rows.append({"label": label, "B": b, "V": v, "F": f, **model_size(ckpt), **training_cost(a.final_dir, RUNS[label]),
                     "mean_rtf_recorded": s["mean_rtf"], "wer_mean": s["mean_wer"], "utmos_mean": s["mean_utmos"],
                     "mcd_mean": s["mcd"]["mean_mcd"]})
    write_csv(a.eval_dir / "ablation8-cost.csv", rows)
    fronts = {}
    for metric in METRICS:
        for cost in ("elements_inference_path", "mean_rtf_recorded"):
            pts = {r["label"]: (r[f"{metric}_mean"], r[cost]) for r in rows}
            fronts[f"{metric} vs {cost}"] = pareto(pts, (not HIGHER_BETTER[metric], True))
    write_json(a.eval_dir / "ablation8-cost.json", {
        "pareto_sets": fronts,
        "notes": {"elements": "stored state_dict tensor elements (weight norm stores g and v); the inference path is the "
                              "generator without the posterior encoder enc_q, which only training uses",
                  "training": "per machine; RTX 5070 and RTX 4070 Ti times are not comparable with each other; active time "
                              "ends a dead attempt at its last ledger event",
                  "rtf": "recorded inline during evaluation on different, not idle machines: indicative only, not reportable"},
        "rows": rows})
    return rows


def cmd_all(a):
    data = load_all(a.eval_dir)
    cmd_check(a, data)
    cmd_table(a, data)
    cmd_pairs(a, data)
    cmd_effects(a, data)
    cmd_cost(a, data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "table", "pairs", "effects", "cost", "all"))
    ap.add_argument("--eval-dir", type=pathlib.Path, required=True)
    ap.add_argument("--final-dir", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resamples", type=int, default=10000)
    a = ap.parse_args()
    {"check": cmd_check, "table": cmd_table, "pairs": cmd_pairs, "effects": cmd_effects, "cost": cmd_cost, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    main()
