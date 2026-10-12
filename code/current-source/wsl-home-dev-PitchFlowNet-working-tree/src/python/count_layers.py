import sys
sys.path.insert(0, '.')
import torch.nn as nn
from piper_train.vits.models import SynthesizerTrn

CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)

common = dict(
    n_vocab=256,
    spec_channels=1024 // 2 + 1,
    segment_size=8192 // 256,
    inter_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.1,
    resblock="2",
    resblock_kernel_sizes=(3, 5, 7),
    resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
    upsample_rates=(8, 8, 4),
    upsample_initial_channel=256,
    upsample_kernel_sizes=(16, 16, 8),
    n_speakers=1,
    gin_channels=0,
)

def count(model, label):
    print(f"=== {label} ===")
    total = 0
    for name, sub in model.named_children():
        n = sum(1 for _, m in sub.named_modules() if isinstance(m, CONV_TYPES))
        total += n
        print(f"  {name}: {n}")
    print(f"  TOTAL (model_g): {total}")
    print()

configA = SynthesizerTrn(**common, use_sdp=True, use_snake=False, use_f0=False, use_transformer_flows=False, use_noised_mas=False)
count(configA, "Config A (02_Baseline_VanillaVITS: vits2=F,bigvgan=F,f0=F)")

configH = SynthesizerTrn(**common, use_sdp=True, use_snake=True, use_f0=True, use_transformer_flows=True, use_noised_mas=True)
count(configH, "Config H (01_Baseline_BigVgan_VITS2_FO: vits2=T,bigvgan=T,f0=T)")
