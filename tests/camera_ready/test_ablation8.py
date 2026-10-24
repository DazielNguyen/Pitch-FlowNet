"""The eight-cell ablation analysis: cube edges, Holm, factorial effects, Shapley, Pareto, bootstrap, and the
pairs/effects commands on synthetic per-sentence files (no real result is read)."""
import argparse
import csv
import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _common import REPO

_spec = importlib.util.spec_from_file_location("ablation8_analysis", REPO / "tools" / "camera_ready" / "ablation8_analysis.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def additive(base, gains, inter=None):
    """Cell values base + sum of gains of the factors on (+ an optional B x V term when both are on)."""
    y = {}
    for bits in itertools.product((0, 1), repeat=3):
        v = base + sum(g * b for g, b in zip(gains, bits))
        if inter and bits[0] and bits[1]:
            v += inter
        y[bits] = v
    return y


class PureFunctions(unittest.TestCase):
    def test_cube_edges(self):
        edges = A.cube_edges()
        self.assertEqual(len(edges), 13)
        for factor, off, on in edges[:12]:
            diff = [i for i, (x, y) in enumerate(zip(A.CELLS[off], A.CELLS[on])) if x != y]
            self.assertEqual(diff, [A.FACTORS.index(factor)])
            self.assertEqual(A.CELLS[on][diff[0]], 1)
        self.assertEqual(edges[12][1:], ("PFN-Baseline-200ep-best", "PFN-Full-200ep-best"))
        self.assertEqual(len({(o, n) for _, o, n in edges}), 13)

    def test_holm(self):
        # textbook example: p = 0.01, 0.04, 0.03, 0.005 -> 0.03, 0.06, 0.06, 0.02
        np.testing.assert_allclose(A.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(A.holm([0.5, None, 0.9])[1], None)
        self.assertEqual(A.holm([0.6, 0.9]), [1.0, 1.0])

    def test_factorial_effects_of_an_additive_model(self):
        eff = A.factorial_effects(additive(10.0, (1.0, -2.0, 0.5)))
        for term, want in zip(A.EFFECT_TERMS, (1.0, -2.0, 0.5, 0, 0, 0, 0)):
            self.assertAlmostEqual(float(eff[term]), want, places=12, msg=term)

    def test_factorial_interaction(self):
        # a B x V term of +4 only when both are on: half the change of the B effect (0 -> 4) is 2
        eff = A.factorial_effects(additive(0.0, (0.0, 0.0, 0.0), inter=4.0))
        self.assertAlmostEqual(float(eff["BV"]), 2.0)
        self.assertAlmostEqual(float(eff["B"]), 2.0)
        self.assertAlmostEqual(float(eff["BVF"]), 0.0)

    def test_shapley(self):
        y = additive(3.0, (1.0, -2.0, 0.5), inter=4.0)
        sh = A.shapley(y)
        self.assertAlmostEqual(sum(sh.values()), y[(1, 1, 1)] - y[(0, 0, 0)])
        self.assertAlmostEqual(sh["F"], 0.5)                 # no interaction with F: its own gain
        self.assertAlmostEqual(sh["B"] - sh["V"], 3.0)       # the B x V term is split equally between B and V
        arr = A.shapley({k: np.full(5, v) for k, v in y.items()})
        np.testing.assert_allclose(arr["B"], sh["B"])

    def test_pareto(self):
        pts = {"a": (1.0, 10.0), "b": (2.0, 5.0), "c": (3.0, 6.0), "d": (1.0, 10.0)}
        self.assertEqual(A.pareto(pts, (True, True)), ["a", "b", "d"])
        self.assertEqual(A.pareto({"a": (3.5, 1.0), "b": (3.0, 2.0)}, (False, True)), ["a"])

    def test_bootstrap_is_seeded_and_covers_the_mean(self):
        x = np.random.default_rng(0).normal(1.0, 1.0, 500)
        i1, i2 = A.bootstrap_indices(500, 2000, 7), A.bootstrap_indices(500, 2000, 7)
        np.testing.assert_array_equal(i1, i2)
        mean, lo, hi = A.mean_ci(x, i1)
        self.assertAlmostEqual(mean, float(x.mean()))
        self.assertLess(lo, mean)
        self.assertLess(mean, hi)


class SyntheticRun(unittest.TestCase):
    """pairs and effects on fake per-sentence files; the WER/UTMOSv2 p-values must match a compare-*.json."""

    @classmethod
    def setUpClass(cls):
        from scipy.stats import wilcoxon
        cls.tmp = Path(tempfile.mkdtemp(prefix="ablation8-"))
        rng = np.random.default_rng(1)
        texts = [f"sentence {i}" for i in range(500)]
        cls.values = {}
        for label, bits in A.CELLS.items():
            d = cls.tmp / label
            (d / "mcd").mkdir(parents=True)
            wer = np.clip(0.12 - 0.01 * bits[0] + rng.normal(0, 0.05, 500), 0, 1)
            utmos = 3.0 + 0.2 * bits[1] + 0.1 * bits[2] + rng.normal(0, 0.3, 500)
            mcd = 25.0 + 0.3 * bits[0] * bits[2] + rng.normal(0, 1.0, 500)
            cls.values[label] = {"wer": wer, "utmos": utmos, "mcd": mcd}
            with open(d / "x_per_sentence.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["text", "wer", "utmos", "infer_sec", "audio_duration_sec", "rtf"])
                w.writerows([t, a, b, 0.1, 1.0, 0.1] for t, a, b in zip(texts, wer, utmos))
            with open(d / "mcd" / "mcd_gtmos.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["idx", "text", "mcd", "gt_utmos"])
                w.writerows([i, t, m, 3.8] for i, (t, m) in enumerate(zip(texts, mcd)))
            (d / "summary.json").write_text(json.dumps({"f0_rmse": 0.5 if bits[2] else None, "f0_corr": 0.2 if bits[2] else None}))
            (d / "protocol.json").write_text("{}")
            (d / "reference.jsonl").write_text("")
        for _, off, on in A.cube_edges():
            t = {"a": off, "b": on, "n": 500}
            for m in ("wer", "utmos"):
                t[f"{m}_wilcoxon_p"] = float(wilcoxon(cls.values[off][m], cls.values[on][m])[1])
            (cls.tmp / f"compare-{off}_{on}.json").write_text(json.dumps({"paired_tests": [t]}))
        cls.args = argparse.Namespace(eval_dir=cls.tmp, final_dir=cls.tmp, seed=3, resamples=500)

    def test_pairs(self):
        out = A.cmd_pairs(self.args)
        self.assertEqual(len(out), 13 * 3)
        self.assertTrue(all(r["compare_p_agrees"] for r in out if r["metric"] != "mcd"))
        for metric in A.METRICS:
            rows = [r for r in out if r["metric"] == metric]
            self.assertTrue(all(r["p_holm"] >= r["p_raw"] for r in rows))
        r = next(r for r in out if r["metric"] == "wer" and r["off"] == "PFN-Baseline-200ep-best" and r["factor"] == "B")
        want = float(np.mean(self.values[r["on"]]["wer"] - self.values[r["off"]]["wer"]))
        self.assertAlmostEqual(r["mean_diff_on_minus_off"], want)
        self.assertLessEqual(r["ci95_low"], want)
        self.assertGreaterEqual(r["ci95_high"], want)
        self.assertTrue((self.tmp / "ablation8-pairs.csv").is_file())

    def test_effects(self):
        doc = A.cmd_effects(self.args)
        for metric in A.METRICS:
            sh = doc["shapley"][metric]
            self.assertAlmostEqual(sum(sh[n]["value"] for n in A.FACTORS), sh["gap_full_minus_baseline"]["value"])
        self.assertEqual(set(doc["f0_2x2"]), {"B=1,V=1", "B=1,V=0", "B=0,V=1", "B=0,V=0"})
        self.assertTrue((self.tmp / "ablation8-effects.json").is_file())


if __name__ == "__main__":
    unittest.main()
