import sys
sys.path.insert(0, '.')
import torch.nn as nn
from piper_train.vits.models import (
    SynthesizerTrn, MultiPeriodDiscriminator, MultiResolutionDiscriminator,
    DurationDiscriminator,
)

CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
LINEAR_TYPES = (nn.Linear,)

def count_modules(model, label, indent="  "):
    total_conv = 0
    print(f"{label}:")
    for name, sub in model.named_children():
        n_conv = sum(1 for _, m in sub.named_modules() if isinstance(m, CONV_TYPES))
        n_lin = sum(1 for _, m in sub.named_modules() if isinstance(m, LINEAR_TYPES))
        total_conv += n_conv
        extra = f" (+{n_lin} Linear)" if n_lin else ""
        print(f"{indent}{name}: {n_conv}{extra}")
    n_lin_top = sum(1 for _, m in model.named_modules() if isinstance(m, LINEAR_TYPES))
    print(f"{indent}TOTAL conv: {total_conv}  |  TOTAL linear: {n_lin_top}")
    print()

common = dict(
    n_vocab=256, spec_channels=1024 // 2 + 1, segment_size=8192 // 256,
    inter_channels=192, hidden_channels=192, filter_channels=768, n_heads=2,
    n_layers=6, kernel_size=3, p_dropout=0.1, resblock="2",
    resblock_kernel_sizes=(3, 5, 7), resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
    upsample_rates=(8, 8, 4), upsample_initial_channel=256, upsample_kernel_sizes=(16, 16, 8),
    n_speakers=1, gin_channels=0,
)

print("################ Config A (02_Baseline_VanillaVITS) ################\n")
mpd = MultiPeriodDiscriminator(use_spectral_norm=False)
count_modules(mpd, "model_d (MultiPeriodDiscriminator)")
print("  -> model_d_mrd: None (use_bigvgan=False)")
print("  -> model_d_dur: None (use_vits2=False)\n")

print("################ Config H (01_Baseline_BigVgan_VITS2_FO) ################\n")
count_modules(mpd, "model_d (MultiPeriodDiscriminator) -- identical, not gated by any flag")
mrd = MultiResolutionDiscriminator(use_spectral_norm=False)
count_modules(mrd, "model_d_mrd (MultiResolutionDiscriminator)")
dur_disc = DurationDiscriminator(in_channels=192, filter_channels=192, kernel_size=3, p_dropout=0.1, gin_channels=0)
count_modules(dur_disc, "model_d_dur (DurationDiscriminator)")

# Per-submodule breakdown detail
print("################ Detail: MultiPeriodDiscriminator sub-discriminators ################\n")
for i, d in enumerate(mpd.discriminators):
    n = sum(1 for _, m in d.named_modules() if isinstance(m, CONV_TYPES))
    print(f"  discriminators[{i}] ({type(d).__name__}): {n} conv")
print()

print("################ Detail: MultiResolutionDiscriminator sub-discriminators ################\n")
for i, d in enumerate(mrd.discriminators):
    n = sum(1 for _, m in d.named_modules() if isinstance(m, CONV_TYPES))
    print(f"  discriminators[{i}] ({type(d).__name__}, resolution={d.resolution}): {n} conv")
