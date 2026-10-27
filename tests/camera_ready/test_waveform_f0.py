"""The waveform F0 metrics, frame mapping, alignment and summary (synthetic signals; no checkpoint)."""
import importlib.util
import unittest

import numpy as np

from _common import REPO

_spec = importlib.util.spec_from_file_location("waveform_f0", REPO / "tools" / "camera_ready" / "waveform_f0.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)


class PairMetrics(unittest.TestCase):
    def test_known_values(self):
        ref = np.array([0.0, 100.0, 200.0, 100.0, 0.0])
        syn = np.array([0.0, 100.0 * 2 ** (100 / 1200), 200.0 * 2 ** (100 / 1200), 0.0, 150.0])
        m = W.pair_metrics(ref, syn)
        self.assertAlmostEqual(m["rmse_cents"], 100.0)
        self.assertAlmostEqual(m["offset_cents"], 100.0)
        self.assertAlmostEqual(m["vde"], 2 / 5)
        self.assertAlmostEqual(m["coverage"], 2 / 5)
        self.assertAlmostEqual(m["gpe"], 0.0)                 # 100 cents is a 5.9 % error, under 20 %
        self.assertAlmostEqual(m["ffe"], 2 / 5)
        self.assertIsNone(m["corr_logf0"])                    # fewer than 10 pairs

    def test_gross_errors_and_correlation(self):
        ref = np.linspace(100, 200, 20)
        syn = ref * 1.5
        m = W.pair_metrics(ref, syn)
        self.assertAlmostEqual(m["gpe"], 1.0)
        self.assertAlmostEqual(m["corr_logf0"], 1.0)
        self.assertAlmostEqual(m["vde"], 0.0)


class Alignment(unittest.TestCase):
    def test_frame_mapping(self):
        idx = W.frames_to_f0_index(4, 100)
        np.testing.assert_array_equal(idx, np.rint(np.arange(4) * 256 / 22050 / 0.005).astype(int))
        self.assertEqual(W.frames_to_f0_index(10 ** 4, 50).max(), 49)

    def test_self_alignment_is_exact(self):
        rng = np.random.default_rng(0)
        mf = rng.normal(size=(40, 24))
        f0 = np.where(rng.random(200) > 0.3, rng.uniform(100, 300, 200), 0.0)
        for method in ("dtw", "linear"):
            r, s = W.aligned_pairs(f0, mf, f0, mf, mf.mean(0), mf.std(0), method)
            np.testing.assert_array_equal(r, s)
            m = W.pair_metrics(r, s)
            self.assertAlmostEqual(m["rmse_cents"], 0.0)
            self.assertAlmostEqual(m["vde"], 0.0)


class Summary(unittest.TestCase):
    def test_comparisons_and_holm(self):
        rng = np.random.default_rng(1)
        labels = sorted({l for c in W.COMPARISONS for l in c[1:]})
        recs = []
        for label in labels:
            bump = 50.0 if label == "Historical-Full-1079" else 0.0
            for i in range(30):
                m = {k: float(v) for k, v in zip(W.METRICS, rng.normal(0.5, 0.1, len(W.METRICS)))}
                m["rmse_cents"] = 200.0 + bump + rng.normal(0, 5)
                recs.append({"label": label, "utterance": i, "dtw": m, "linear": m})
        s = W.summarize(recs, labels, resamples=300)
        hist = next(c for c in s["comparisons"]["dtw"] if c["kind"] == "historical")
        self.assertGreater(hist["rmse_cents"]["mean_diff_on_minus_off"], 30)
        self.assertLess(hist["rmse_cents"]["p_raw"], 1e-3)
        self.assertTrue(all(c[m]["p_holm"] >= c[m]["p_raw"] for c in s["comparisons"]["dtw"] for m in W.METRICS))
        self.assertEqual(set(s["models"]["dtw"]), set(labels))


if __name__ == "__main__":
    unittest.main()
