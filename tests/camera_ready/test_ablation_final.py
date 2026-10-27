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


class PaperTables(unittest.TestCase):
    def test_leave_one_out_sign_and_rows(self):
        rows = []
        for metric in ("utmos", "wer", "mcd"):
            rows.append({"comparison": "Piper-773 -> PitchFlowNet-1079", "metric": metric, "mean_off": 1.0, "mean_on": 2.0,
                         "diff": 1.0, "ci95": [0.5, 1.5], "d_z": 1.0, "p_raw": 1e-5, "p_holm": 3e-5})
            for _, comp, _ in F.PAPER_ROWS:
                rows.append({"comparison": comp, "metric": metric, "diff": 0.3, "ci95": [0.1, 0.5], "p_holm": 0.02})
        md = F.paper_tables(rows)
        self.assertIn("| w/o BigVGAN (leave-one-out) | -0.300 [-0.500, -0.100] | 0.020 |", md)
        self.assertIn("| + BigVGAN (add-one) | +0.300 [+0.100, +0.500] | 0.020 |", md)
        self.assertIn("\\times10^{-5}", md)
        self.assertEqual(md.count("\\begin{tabular}"), 2)


class F0Tables(unittest.TestCase):
    def test_units_and_sections(self):
        self.assertAlmostEqual(F.CENTS_PER_LN, 1731.234, places=3)
        f0 = {"predictor_metric": [{"model": "PFN-Full-200ep-best", "protocol": "p", "rmse_ln": 0.5, "rmse_cents": 0.5 * F.CENTS_PER_LN,
                                    "corr": 0.2, "source": "s"}],
              "main_historical_evaluation_f0": [None, None],
              "waveform": {"PFN-Full-200ep-best": {m: {"mean": 1.0, "ci95": [0.5, 1.5], "n": 3} for m in ("rmse_cents", "corr_logf0", "vde", "gpe")}}}
        md = F.f0_tables(f0)
        self.assertIn("| Full | p | 0.5000 | 866 | 0.2000 | `s` |", md)
        self.assertIn("F0-b.", md)
        self.assertIn("[None, None]", md)


class ExtraTables(unittest.TestCase):
    def test_rows(self):
        ci = {"effect": 0.1, "ci95": [0.0, 0.2]}
        sv = {"value": 0.1, "ci95": [0.0, 0.2], "pct_of_gap": 50.0}
        x = {"eight": [{"label": "PFN-Full-200ep-best", "B": 1, "V": 1, "F": 1, "best_epoch": 188, "wer_mean": 0.1135, "wer_median": 0.0741,
                        "utmos_mean": 3.5, "utmos_median": 3.5, "mcd_mean": 25.6, "f0_rmse_ln": None, "f0_corr": None, "audio_sec": 10.0, "gpu": "RTX 5070"}],
             "reference_audio_sec": 12.0,
             "effects": {m: {t: ci for t in ("B", "V", "F", "BV", "BF", "VF", "BVF")} for m in ("wer", "utmos", "mcd")},
             "shapley": {m: {"gap": {"value": 0.2, "ci95": [0.1, 0.3]}, "B": sv, "V": sv, "F": sv} for m in ("wer", "utmos", "mcd")},
             "runs": [{"label": "PFN-Full-200ep-best", "gpu": "RTX 5070", "attempts": 6, "auto_resumes": 2, "end_utc": "2026-10-21T03:08:19+00:00",
                       "best_epoch": 188, "best_val": 21.3, "val_epoch_199": 21.4, "slope_150_199_per_10": -0.07}],
             "nof0_over_full": {"median": 1.4, "p10": 1.25, "p90": 1.57, "share_above_1_1": 0.986},
             "duration_diag": None,
             "pitch_intervention": {"PFN-Full-200ep-best": {"n": 100, "+2": {"mean": 0.02, "ci95": [0.01, 0.03], "voicing": 0.9},
                                                            "-2": {"mean": -0.02, "ci95": [-0.03, -0.01], "voicing": 0.9},
                                                            "off": {"mean": -0.08, "ci95": [-0.09, -0.07], "voicing": 0.8},
                                                            "slope": {"mean": 0.008, "ci95": [0.007, 0.009]}, "reading": "not controllable"}},
             "cost_rows": [], "pareto": {"wer vs size": ["PFN-Baseline-200ep-best"]}, "historical_records": [], "historical_mcd": {}}
        md = F.extra_tables(x)
        self.assertIn("| Full | 1 | 1 | 1 | 188 | 11.35 | 7.41 |", md)
        self.assertIn("| Full | RTX 5070 | 6 | 2 | 2026-10-21 03:08 | 188 |", md)
        self.assertIn("not controllable", md)
        self.assertIn("- wer vs size: Baseline", md)
        self.assertIn("98.6 %", md)


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
