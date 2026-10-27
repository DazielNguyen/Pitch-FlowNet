"""Final statistics: paired statistics, Holm by family, the inference parameter count (synthetic data only)."""
import importlib.util
import sys
import unittest

import numpy as np
import torch

from _common import REPO

sys.path.insert(0, str(REPO / "tools" / "camera_ready"))
_spec = importlib.util.spec_from_file_location("ablation_final", REPO / "tools" / "camera_ready" / "ablation_final.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)


class PairedStats(unittest.TestCase):
    def test_shift_and_effect_size(self):
        rng = np.random.default_rng(0)
        off = rng.normal(3.0, 0.3, 500)
        on = off + 0.4 + rng.normal(0, 0.3, 500)
        s = F.paired_stats(on, off, resamples=2000)
        self.assertAlmostEqual(s["diff"], float(np.mean(on - off)))
        self.assertLess(s["ci95"][0], s["diff"])
        self.assertGreater(s["ci95"][1], s["diff"])
        self.assertAlmostEqual(s["d_z"], float(np.mean(on - off) / np.std(on - off, ddof=1)))
        self.assertLess(s["p_raw"], 1e-10)

    def test_nan_pairs_dropped_and_constant_difference(self):
        s = F.paired_stats([1.0, 2.0, np.nan, 4.0], [0.0, 1.0, 5.0, 3.0], resamples=200)
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["ci95"], [1.0, 1.0])
        self.assertIsNone(s["d_z"])

    def test_holm_by_group(self):
        rows = [{"group": "x", "p_raw": 0.01}, {"group": "x", "p_raw": 0.04}, {"group": "y", "p_raw": 0.04}]
        F.add_holm(rows, "group")
        self.assertEqual([r["holm_m"] for r in rows], [2, 2, 1])
        self.assertAlmostEqual(rows[0]["p_holm"], 0.02)
        self.assertAlmostEqual(rows[1]["p_holm"], 0.04)
        self.assertAlmostEqual(rows[2]["p_holm"], 0.04)


class Params(unittest.TestCase):
    def test_inference_definition(self):
        z = lambda *s: torch.zeros(*s)
        sd = {"model_g.enc_p.w": z(10), "model_g.enc_q.w": z(100), "model_g.dp.post_pre.w": z(7), "model_g.dp.flows.1.w": z(5),
              "model_g.dp.flows.2.w": z(3), "model_g.dp.flows.10.w": z(2), "model_g.dec.conv.weight_g": z(4),
              "model_g.dec.conv.weight_v": z(40), "model_g.flow.x.weight_g": z(1), "model_d.disc.w": z(1000)}
        p = F.inference_params(sd)
        self.assertEqual(p["elements"], 10 + 3 + 2 + 40 + 1)        # enc_p, dp.flows.2, dp.flows.10, dec weight_v, flow g
        self.assertEqual(p["fp32_bytes"], 4 * p["elements"])


if __name__ == "__main__":
    unittest.main()
