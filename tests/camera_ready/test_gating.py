"""Group flags B (BigVGAN), V (VITS2), F (F0): module tree, parameters and loss terms per configuration."""
import itertools
import unittest

import torch

import _common  # noqa: F401  (sets sys.path)
from _common import base_model_kwargs
from piper_train.vits import modules
from piper_train.vits.lightning import VitsModel

torch.set_num_threads(4)


def build(b, v, f, semantics="strict"):
    torch.manual_seed(0)
    return VitsModel(**base_model_kwargs(use_bigvgan=bool(b), use_vits2=bool(v), use_f0=bool(f)), group_semantics=semantics)


def keys(module):
    return list(module.state_dict().keys())


def snake_count(module):
    return sum(isinstance(m, modules.Snake1d) for m in module.modules())


def synthetic_batch(n_vocab=256):
    g = torch.Generator().manual_seed(1)
    t_text, t_frames = 12, 48
    x = torch.randint(1, n_vocab, (1, t_text), generator=g)
    spec = torch.rand(1, 513, t_frames, generator=g)
    f0 = torch.rand(1, t_frames, generator=g) * 200 + 80
    return x, torch.LongTensor([t_text]), spec, torch.LongTensor([t_frames]), f0


class GroupGating(unittest.TestCase):
    def test_every_combination_has_exactly_its_groups(self):
        for b, v, f in itertools.product((0, 1), repeat=3):
            with self.subTest(B=b, V=v, F=f):
                m = build(b, v, f)
                dec_keys = keys(m.model_g.dec)
                flow_keys = keys(m.model_g.flow)
                g_keys = keys(m.model_g)
                # B: every periodic activation of the decoder and the MRD
                self.assertEqual(snake_count(m.model_g.dec) > 0, bool(b))
                self.assertEqual(any("snakes" in k for k in dec_keys), bool(b))
                self.assertEqual(any(k.startswith(("pre_up_snakes", "final_snake")) for k in dec_keys), bool(b))
                self.assertEqual(m.model_d_mrd is not None, bool(b))
                self.assertTrue(all(rb.use_snake == bool(b) for rb in m.model_g.dec.resblocks))
                # V: attention in the flow couplings, duration discriminator, noised MAS
                self.assertEqual(any(".attn." in k for k in flow_keys), bool(v))
                self.assertEqual(m.model_d_dur is not None, bool(v))
                self.assertEqual(m.model_g.use_noised_mas, bool(v))
                # F: phoneme F0 predictor and decoder conditioning
                self.assertEqual(any(k.startswith("f0_predictor.") for k in g_keys), bool(f))
                self.assertEqual(any(k.startswith("f0_cond.") for k in dec_keys), bool(f))

    def test_legacy_semantics_keep_resblock_snake(self):
        m = build(0, 0, 0, semantics="legacy")
        self.assertTrue(any("resblocks." in k and "snakes" in k for k in keys(m.model_g.dec)))
        self.assertFalse(any(k.startswith(("pre_up_snakes", "final_snake")) for k in keys(m.model_g.dec)))
        self.assertIsNone(m.model_d_mrd)

    def test_f0_loss_and_conditioning_follow_F(self):
        for f in (0, 1):
            with self.subTest(F=f):
                m = build(1, 1, f).eval()
                x, xl, spec, sl, f0 = synthetic_batch()
                torch.manual_seed(3)
                out = m.model_g(x, xl, spec, sl, None, f0=f0)
                l_f0 = out[7][3]
                if f:
                    self.assertGreater(float(l_f0.detach()), 0.0)
                    self.assertTrue(l_f0.requires_grad)
                else:
                    self.assertEqual(float(l_f0.sum()), 0.0)
                    self.assertFalse(l_f0.requires_grad)

    def test_decoder_without_snake_runs_and_differs(self):
        z = torch.randn(1, 192, 32)
        outs = {}
        for b in (0, 1):
            torch.manual_seed(5)
            dec = build(b, 0, 0).model_g.dec.eval()
            with torch.no_grad():
                outs[b] = dec(z)
            self.assertTrue(torch.isfinite(outs[b]).all())
        self.assertEqual(outs[0].shape, outs[1].shape)
        self.assertFalse(torch.allclose(outs[0], outs[1]))

    def test_invalid_semantics_rejected(self):
        with self.assertRaises(ValueError):
            VitsModel(**base_model_kwargs(), group_semantics="other")


if __name__ == "__main__":
    unittest.main()
