# Manuscript crosswalk: PitchFlowNet (SOICT 2026, submission 4478)

This file compares the manuscript's statements with stored records, item by item.
- The manuscript details were provided as leads; they are not treated as evidence.
- Paths are relative to this folder.
- No result here was produced by re-running anything. Each item is checked against the stored records only.

**Status values:**
- `MATCH`: the stored record shows the stated value.
- `DIFFERS`: the stored record shows something else.
- `PARTIAL`: part of the statement is supported.
- `MISSING`: no record was found.
- `UNRESOLVED`: the records conflict, or the cause is unknown.

**Evidence classes:**
- `CR` (contemporaneous record);
- `DA` (direct artifact);
- `HSC` (historical source candidate);
- `CS` (current source);
- `REC` (reconstructed);
- `AA` (author account).

## Run identity

| Manuscript | Record | Status |
|---|---|---|
| Piper baseline retrained, epoch about 773, val mel 20.2158 | `runs/baseline/…/best-epoch=773-val_loss_mel=20.2158.ckpt` (DA). The static scan gives epoch 773 and global step 605,268. The TensorBoard value at epoch 773 is 20.2158 (CR). Run `02_Baseline_VanillaVITS` | MATCH |
| Full model (PitchFlowNet), epoch about 1079, val mel 19.2943 | `runs/full/…/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt` (DA). Epoch 1079, global step 844,560. TensorBoard at epoch 1079 shows 19.2943 (CR) | MATCH |
| Selected checkpoint = minimum validation mel loss | ModelCheckpoint is configured with `monitor="val_loss_mel"`, `mode="min"` and `save_top_k=1` (HSC). Yet the logged rank-0 series have lower values later: baseline 19.9053 at epoch 996, full 19.0860 at epoch 641. A candidate cause is that `val_loss_mel` is logged per rank without `sync_dist` under DDP, while checkpointing mixes rank-local decisions; this is not proven | UNRESOLVED |
| "Piper baseline" is not the published Piper checkpoint or the 2026-10 `Piper_no_VITS2_cpn` run | It is the project's own EdgeTTS run, trained from scratch on 2026-07-09 → 07-13 (hparams, log, CR) | MATCH (identity) |
| Same split membership for both models | Both hparams name the same `dataset.jsonl`, seed 1234 and 100 val / 500 test (CR). The TensorBoard audio examples, which are the first five test items, are identical in both runs (CR). The reconstructed split (REC) equals the stored `test_entries_500.jsonl` in order | MATCH (reconstructed + corroborated; no run-time split file) |

## Components

| Manuscript | Record | Status |
|---|---|---|
| Baseline has none of the full model's waveform/VITS2/F0 additions | hparams: `use_bigvgan`, `use_vits2` and `use_f0` all false (CR). The checkpoint has no flow attention, duration discriminator, MRD or F0 names (DA). **However, the checkpoint holds 36 Snake parameters in the decoder ResBlocks**, and `ResBlock2` at the candidate commits always builds `Snake1d` (HSC 6033be5; an on/off switch arrived only in `8f07b63`, 2026-07-25). The baseline checkpoint has no Snake parameters outside the ResBlocks, whereas the full model has 44 (the extra 8 sit before the upsampling layers and `conv_post`); which activation the baseline used there is not re-derived here | DIFFERS (Snake activations inside the baseline ResBlocks) |
| HiFi-GAN ResBlock2, upsampling 8/8/4 | `resblock '2'`, upsample rates 8/8/4, kernels 16/16/8 in both hparams (CR) | MATCH |
| SnakeBeta | Full model: 44 Snake parameters in the decoder (DA). The candidate code class is `Snake1d` with alpha/beta. It was renamed `SnakeBeta` on 2026-07-25 (`8f07b63`) | PARTIAL (name in code at run time was Snake1d) |
| MRD with 3 STFT resolutions | `model_d_mrd.*`: 82 names (DA). Resolution count from code (HSC) | MATCH (presence); resolutions not re-derived here |
| Self-attention in flow coupling | 144 flow-attention names (DA) | MATCH |
| Duration discriminator | `model_d_dur.*`: 33 names (DA) | MATCH |
| MAS noise `max(0, 0.01 − step × 2e-6)` | hparams `mas_noise_scale_decay` 2e-6 (CR). `use_noised_mas` and `mas_noise_scale=0.01` exist from `badf381` (HSC). They are **absent at 6033be5**, so the code that ran was uncommitted at launch | PARTIAL (formula not verified in the source that ran) |
| Phoneme-level F0 predictor (2 × Conv1d, hidden 256, kernel 3, 1×1 out) and `f0_cond` zero-initialised | `F0Predictor` (17 names) and `f0_cond` (3 names) in the checkpoint (DA). `f0_cond` is zero-initialised and the predictor exists in the candidate `models.py` (HSC). Layer sizes not re-derived here | PARTIAL |
| Training uses ground-truth frame F0; inference uses predicted F0 and durations | Candidate `models.py`: training averages frame F0 per phoneme for the loss and conditions the decoder on ground-truth F0 slices; inference expands the predicted log-F0 by the alignment and exponentiates it (HSC) | MATCH in candidate source only |
| Backbone sizes (192/768, 6 layers, 2 heads, kernel 3; posterior 16 layers k5; flow 4 × WN 4 k5) | Text-encoder values in hparams (CR). Posterior/flow sizes are hard-coded in the candidate code (HSC) | PARTIAL |

## Data and training

| Manuscript | Record | Status |
|---|---|---|
| LJSpeech 13,100 utterances | `datasets/raw/home-dev-data/wavs`: 13,100 WAVs, `metadata.csv` 13,100 lines. `dataset.jsonl`: 13,100 rows (DA) | MATCH |
| 12,500 / 100 / 500, random permutation | `random_split` seed 1234, order train/test/val (HSC). Reconstructed (REC), and its test part equals the stored test entries | MATCH (reconstructed) |
| espeak-ng en-us, `QUOTE_NONE`, Silero VAD, 22.05 kHz, normalisation | `QUOTE_NONE` in `preprocess.py`; Silero VAD and pyworld in `norm_audio` (HSC). `preprocess.log` (CR). Dataset `config.json` (DA) | PARTIAL (source candidates; preprocessing run 2026-07-01) |
| FFT 1024, hop 256, 513 linear bins, 80 mel | hparams (CR) | MATCH |
| WORLD DIO + StoneMask F0 | `pyworld.dio` + `pyworld.stonemask` in candidate `norm_audio` (HSC). F0 tensors in the cache (DA) | MATCH in candidate source |
| Segment 8192 samples / 32 latent frames | `segment_size` 8192, hop 256 (CR) | MATCH |
| From scratch, 1,100 epochs | Baseline: from scratch, `max_epochs` 1100, "max_epochs=1100 reached" (CR). Full: also 1,100 epochs, but **resumed twice**. `version_2` resumed from `version_0/last.ckpt`; `version_0/1` no longer exist. Earlier attempts in `train.log` were aborted or killed (CR) | PARTIAL (full model not a single uninterrupted run) |
| 2 × RTX 4070 Ti, DDP/NCCL, bf16 | `devices '2'`, `ddp_find_unused_parameters_true`, `bf16-mixed` (CR). NCCL in the log (CR). GPU model not in these run records (AA; see `history/environment`) | PARTIAL |
| Batch 16/GPU (32 total), seed 1234 | hparams (CR) | MATCH |
| AdamW 2e-4, (0.8, 0.99), eps 1e-9; clip 1.0; LR decay 0.999875; c_mel 45 | hparams (CR) | MATCH |
| Manual optimization, generator before discriminator | EdgeTTS `training_step` (HSC) | MATCH in candidate source |
| Recorded training duration: baseline 77.26 h | TensorBoard first-to-last wall time 77.26 h (CR) | MATCH |
| Recorded training duration: full 103.31 h | `version_2` TensorBoard covers 99.47 h (epochs 43–1099). The `version_0` events are missing | MISSING (cannot be reproduced from what remains) |

## Evaluation (500 test sentences)

| Manuscript | Record | Status |
|---|---|---|
| WER 9.17% / 8.64% | `evaluation/eval_h_vs_baseline/summary.json`: 0.091710 / 0.086383. Recomputed from the per-sentence CSVs (CR) | MATCH |
| UTMOSv2 3.366 / 3.807 | 3.3656 / 3.8070 (same files) | MATCH |
| RTF 0.0559 / 0.0568 | 0.05591 / 0.05681 (same files) | MATCH |
| Params 15.66 M / 17.21 M; size 62.80 / 69.00 MB | Baseline inference path 15,664,667 (REC: 2026-09-23 audit, `history/prior-audits-2026-09-23/question2_parameter_counts…`). Full-model value and byte sizes are in the same audit files; not re-derived here | PARTIAL (later audit, not a run record) |
| Ground truth WER 6.67%, UTMOSv2 3.830 | `evaluation/ground-truth/eval_pitchflownet_500_gt/gt_wer_utmos_summary.json` (2026-09-24, CR): 0.066694 / 3.8301 | MATCH |
| Paired two-sided Wilcoxon; WER p = 0.115 | `summary.json` WER p 0.114604 (CR). UTMOS p 1.65e-79, reported as "< 0.001" | MATCH |
| UTMOS diff +0.441, BCa CI [0.416, 0.468], dz 1.482; WER diff −0.533 pp, BCa [−1.318, 0.232], dz −0.061; BCa 20,000 resamples, seed 1234 | **No script or output found** in the scanned scope (EdgeTTS, `<WSL_DEV_HOME>`, `paper_revisions`, `Compact`). The mean differences follow from the stored means (3.8070 − 3.3656 = 0.441; 8.638 − 9.171 = −0.533 pp) | MISSING (BCa/dz computation record) |
| Other evaluations of the same checkpoint | `evaluation/other/eval_results/baseline_full` (2026-07-10: 8.85% / 3.827 / RTF 0.0712); `eval_h_vs_baseline/f0_configH` and `f0_configH_v2`; `evaluation/full/eval_pitchflownet_500` (2026-09-24: 9.03% / 3.818 / 0.0687); F0 ablation (8.33% / 3.796). The values differ between evaluation runs (synthesis randomness and settings) | Recorded; not substituted |
| Timing on Intel Core i7-14700; inference only; thread/batch settings unrecorded | CPU from the author (AA, `history/notes`, kept local and not tracked). The workstation record of 2026-10-12 (`history/environment/system_2026-10-12.sanitized.txt`) lists an i7-14700; it is a current record, not a run-time one. The harness times the model call (HSC `eval_harness.py`). No thread setting recorded | PARTIAL / MISSING (threads) |

## Ablations

- **Config E** (BigVGAN + VITS2 without F0; `06_Config_E_BigVGAN_VITS2`): `grad_clip` 0.5, resumed from epoch 219, stopped at epoch 508. Best is epoch 434. One 500-sentence evaluation exists (`evaluation/other/eval_f0_ablation_E_vs_H`).
  - This is **not** a complete, matched ablation: it uses a different clipping value and a shorter budget.
- **Config H to 773** (`09_Config_H_Full_ep773`): stopped after about 2 minutes (one 1.2 MB event file, no checkpoint).
- **`baseline_run`** (2026-07-03): a 6-minute start with no checkpoint.
- **No complete ablation exists in the scanned records.**
