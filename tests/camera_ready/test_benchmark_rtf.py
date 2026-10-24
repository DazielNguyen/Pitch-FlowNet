"""The RTF benchmark's schedule, idle checks and aggregation (no model is run)."""
import importlib.util
import unittest

from _common import REPO

_spec = importlib.util.spec_from_file_location("benchmark_rtf", REPO / "tools" / "camera_ready" / "benchmark_rtf.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


class Schedule(unittest.TestCase):
    def test_rotation_and_size(self):
        labels = ["a", "b", "c", "d"]
        self.assertEqual(B.round_order(labels, 0), labels)
        self.assertEqual(B.round_order(labels, 1), ["b", "c", "d", "a"])
        plan = B.schedule(labels, rounds=3, threads=(1, 2))
        self.assertEqual(len(plan), 3 * 4 * 2)
        self.assertEqual(len(set(plan)), len(plan))
        self.assertEqual(plan[:2], [(0, "a", 1), (0, "a", 2)])
        self.assertEqual({B.CPUSET[t].count(",") + 1 for t in B.THREADS}, set(B.THREADS))


class IdleChecks(unittest.TestCase):
    def test_refusals(self):
        busy = {"loadavg": (1.5, 1.0, 1.0), "tmux_sessions": ["cr-rtf", "cr-eval-x"], "gpus": ["0, 50, 4000", "1, 2, 1500"]}
        p = B.idle_problems(busy)
        self.assertEqual(len(p), 3)
        self.assertIn("cr-eval-x", p[1])
        idle = {"loadavg": (0.2, 0.1, 0.0), "tmux_sessions": ["cr-rtf"], "gpus": ["0, 0, 4", "1, 3, 1600"]}
        self.assertEqual(B.idle_problems(idle), [])


class Summary(unittest.TestCase):
    def rec(self, label, t, r, model, e2e, audio, samples):
        return {"label": label, "threads": t, "round": r, "model_sec": model, "e2e_sec": e2e, "audio_sec": audio,
                "samples": samples}

    def test_corpus_and_per_utterance_rtf(self):
        recs = [self.rec("x", 1, 0, [1.0, 1.0], [1.5, 1.5], [10.0, 2.0], [1, 2]),
                self.rec("x", 1, 1, [2.0, 2.0], [2.5, 2.5], [10.0, 2.0], [1, 2]),
                self.rec("x", 1, 2, [3.0, 3.0], [3.5, 3.5], [10.0, 2.0], [1, 2]),
                self.rec("x", 2, 0, [0.5, 0.5], [1.0, 1.0], [10.0, 2.0], [1, 3])]
        rows = {(r["label"], r["threads"]): r for r in B.summarize(recs)}
        r1 = rows[("x", 1)]
        self.assertEqual(r1["rounds"], 3)
        self.assertAlmostEqual(r1["model_corpus_rtf_median"], 4.0 / 12.0)        # round 1: (2 + 2) / 12
        self.assertAlmostEqual(r1["model_corpus_rtf_min"], 2.0 / 12.0)
        self.assertAlmostEqual(r1["model_corpus_rtf_max"], 6.0 / 12.0)
        self.assertAlmostEqual(r1["model_mean_utt_rtf_median"], (2 / 10 + 2 / 2) / 2)
        self.assertAlmostEqual(r1["e2e_corpus_rtf_median"], 5.0 / 12.0)
        self.assertEqual(r1["audio_sec_total"], 12.0)
        self.assertFalse(r1["audio_lengths_identical_across_runs"])               # the T = 2 run gave other lengths


if __name__ == "__main__":
    unittest.main()
