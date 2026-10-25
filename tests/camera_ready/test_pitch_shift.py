"""The pitch intervention test's hook, shift metric, summary and interpretation (no checkpoint is used)."""
import importlib.util
import unittest

import numpy as np
import torch

from _common import REPO

_spec = importlib.util.spec_from_file_location("pitch_shift", REPO / "tools" / "camera_ready" / "pitch_shift.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


class EchoDecoder(torch.nn.Module):
    """Returns the F0 it receives (or a marker when it receives None)."""

    def forward(self, x, g=None, f0=None):
        return torch.full((1,), -1.0) if f0 is None else f0


class Hook(unittest.TestCase):
    def test_scales_or_removes_f0(self):
        dec = EchoDecoder()
        set_cond = P.install_f0_hook(dec)
        f0 = torch.tensor([100.0, 200.0])
        set_cond("0")
        torch.testing.assert_close(dec(None, f0=f0), f0)
        set_cond("+2")
        torch.testing.assert_close(dec(None, f0=f0), f0 * 2 ** (2 / 12))
        set_cond("-2")
        torch.testing.assert_close(dec(None, f0=f0), f0 * 2 ** (-2 / 12))
        set_cond("off")
        self.assertEqual(float(dec(None, f0=f0)), -1.0)


class Metric(unittest.TestCase):
    def test_shift_on_voiced_frames(self):
        f0_0 = np.array([0.0, 100.0, 200.0, 150.0, 0.0])
        f0_c = np.array([0.0, 100.0 * 2 ** (2 / 12), 200.0 * 2 ** (2 / 12), 0.0, 120.0])
        s, agree = P.shift_semitones(f0_c, f0_0)
        self.assertAlmostEqual(s, 2.0)
        self.assertAlmostEqual(agree, 3 / 5)
        self.assertEqual(P.shift_semitones([0.0, 0.0], [0.0, 0.0])[0], None)

    def test_interpretation(self):
        self.assertEqual(P.interpret(0.95, 0.9, 1.0), "controllable")
        self.assertEqual(P.interpret(0.5, 0.4, 0.6), "partly controllable")
        self.assertEqual(P.interpret(0.5, -0.1, 1.1), "not controllable")
        self.assertEqual(P.interpret(0.1, 0.05, 0.15), "not controllable")

    def test_summary(self):
        recs = [{"label": "m", "shift": {"-2": -2.0 + e, "+2": 2.0 + e, "off": 0.1}, "voicing": {"-2": 1, "+2": 1, "off": 0.9}}
                for e in (-0.1, 0.0, 0.1)]
        row = P.summarize(recs, resamples=200)["m"]
        self.assertAlmostEqual(row["slope"]["mean"], 1.0)
        self.assertEqual(row["slope"]["interpretation"], "controllable")
        self.assertAlmostEqual(row["+2"]["mean_shift_st"], 2.0)
        self.assertAlmostEqual(row["off_mean_abs_shift_st"], 0.1)


if __name__ == "__main__":
    unittest.main()
