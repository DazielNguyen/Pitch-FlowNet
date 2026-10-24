"""The duration diagnostic's aggregation (no model is run)."""
import importlib.util
import math
import unittest

from _common import REPO

_spec = importlib.util.spec_from_file_location("duration_diag", REPO / "tools" / "camera_ready" / "duration_diag.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


class Summary(unittest.TestCase):
    def test_ratios(self):
        records = [{"label": "x", "checkpoint": "best", "epoch": 9, "split": "test", "noise_w": 0.8,
                    "pred": [[2, 2], [3, 3]]},
                   {"label": "x", "checkpoint": "best", "epoch": 9, "split": "validation", "noise_w": 0.0,
                    "pred": [[2, 4]], "mas": [[2, 2]]}]
        rows = D.summarize(records, {"test": [8, 6], "validation": [4]})
        t, v = rows
        self.assertAlmostEqual(t["total_ratio_to_reference"], 10 / 14)
        self.assertAlmostEqual(t["median_utt_ratio"], (4 / 8 + 6 / 6) / 2)
        self.assertNotIn("total_ratio_to_mas", t)
        self.assertAlmostEqual(v["total_ratio_to_mas"], 6 / 4)
        self.assertAlmostEqual(v["mean_abs_log_ratio_to_mas"], (0 + math.log(2)) / 2)

    def test_seed_is_the_evaluation_seed(self):
        self.assertEqual(D.seed_of(0), (2026 * 1_000_003) % 2 ** 31)


if __name__ == "__main__":
    unittest.main()
