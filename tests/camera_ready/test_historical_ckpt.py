"""Historical checkpoints load strictly (weights_only=True, allowlist PosixPath only) in legacy mode,
and the camera-ready strict architecture of H-new has the historical full model's parameters."""
import unittest

import torch

from _common import CKPT_BASELINE, CKPT_FULL, base_model_kwargs
from piper_train.camera_ready.ckpt_io import load_checkpoint
from piper_train.vits.lightning import VitsModel

torch.set_num_threads(4)

ARCH_KEYS = ("num_symbols", "num_speakers", "sample_rate", "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
             "upsample_rates", "upsample_initial_channel", "upsample_kernel_sizes", "filter_length", "hop_length",
             "win_length", "mel_channels", "inter_channels", "hidden_channels", "filter_channels", "n_heads", "n_layers",
             "kernel_size", "p_dropout", "segment_size", "use_sdp", "gin_channels", "use_bigvgan", "use_vits2", "use_f0")


def model_from(hp, semantics):
    kw = {k: hp[k] for k in ARCH_KEYS if k in hp}
    return VitsModel(dataset=None, group_semantics=semantics, **kw)


def shapes(sd):
    return {k: tuple(v.shape) for k, v in sd.items()}


class HistoricalCheckpoints(unittest.TestCase):
    def _check(self, path, flags):
        if not path.is_file():
            self.skipTest(f"{path} not present")
        ckpt = load_checkpoint(path)
        hp = ckpt["hyper_parameters"]
        self.assertEqual((hp["use_bigvgan"], hp["use_vits2"], hp["use_f0"]), flags)
        legacy = model_from(hp, "legacy")
        missing, unexpected = legacy.load_state_dict(ckpt["state_dict"], strict=True)
        self.assertEqual((list(missing), list(unexpected)), ([], []))
        return ckpt, hp

    def test_baseline_773_strict_legacy_load(self):
        ckpt, hp = self._check(CKPT_BASELINE, (False, False, False))
        strict = model_from(hp, "strict")
        hist, new = shapes(ckpt["state_dict"]), shapes(strict.state_dict())
        only_hist = sorted(set(hist) - set(new))
        # Strict B=0 drops exactly the ResBlock Snake parameters (36 alpha/beta tensors), nothing else.
        self.assertEqual(len(only_hist), 36)
        self.assertTrue(all(".resblocks." in k and ".snakes." in k for k in only_hist))
        self.assertEqual(set(new) - set(hist), set())

    def test_full_1079_matches_strict_H_new(self):
        ckpt, hp = self._check(CKPT_FULL, (True, True, True))
        strict = model_from(hp, "strict")
        self.assertEqual(shapes(strict.state_dict()), shapes(ckpt["state_dict"]))

    def test_unlisted_globals_still_refused(self):
        if not CKPT_FULL.is_file():
            self.skipTest("checkpoint not present")
        with self.assertRaises(Exception):
            torch.load(CKPT_FULL, map_location="cpu", weights_only=True)  # PosixPath not allowlisted here


if __name__ == "__main__":
    unittest.main()
