# Final statistics of the camera-ready comparisons

**Scope.** SOICT 2026 paper 4478, camera-ready, 2026-10-28. The measurement conditions were added the same day.
- **What this page holds:** the statistics of every comparison that the paper reports or should report, in one method:
  - the main pair, PitchFlowNet (epoch 1079) against Piper (epoch 773);
  - the 2^3 ablation at 200 epochs;
  - the pitch analyses (T13, T14);
  - the training runs, the duration diagnostics, size and cost, and the historical evaluation records.
- **Ready-to-use tables:**
  - Table F1 for the main result;
  - Table F2 for the ablation;
  - the appendix tables below them.
- **No new score:** nothing was trained, synthesized or re-scored for this page; T13, T14, T15 and the duration diagnostics were run earlier, each under a protocol fixed before its run. Every number is computed from per-sentence files that exist, by `tools/camera_ready/ablation_final.py` (commit `f343ca3`; the X tables are also in `eval/ablation-final-extra.md`).
- **The manuscript:** it is not edited here. Proposed wording is in the last section.

**Reading the numbers:**
- UTMOSv2 is a predicted score, not MOS.
- A non-significant difference is not equivalence.
- Significance is never read from a table of means; every claim below has its paired test.

## Summary

**State on 2026-10-28.** This page gathers every camera-ready result in one method (the manuscript's): paired differences, BCa CIs and Holm-adjusted Wilcoxon p-values. It covers:
- the main pair;
- the eight 200-epoch ablation cells;
- the pitch analyses (T13, T14);
- the CPU RTF benchmark (T15);
- the duration diagnostics;
- the measurement conditions of every table.

**Main pair (1,100 epochs, historical evaluation; Table F1):**
- **UTMOSv2:** PitchFlowNet raises it by +0.441 [0.416, 0.468] over Piper, p (Holm) 4.9 × 10⁻⁷⁹.
- **WER:** the change is −0.53 pp [−1.32, +0.23], p 0.115, p (Holm) 0.229. Not significant.
- **MCD:** −0.064 [−0.175, +0.046], p (Holm) 0.229.
- **RTF:** with declared threads, PitchFlowNet is 12.5 %, 12.6 % and 10.7 % slower at 1, 2 and 12 threads. The ≈ 1.6 % of the manuscript is not reproduced.
- **Size:** the inference parameters match the manuscript (17.21 M against 15.66 M).

**Ablation (200 epochs, one seed per cell; Table F2, X2, X3):**
- **Full vs Baseline:**
  - UTMOSv2 +0.575 [0.546, 0.604], p (Holm) 6.7 × 10⁻⁸⁰;
  - WER +0.54 pp [−0.41, +1.51], p (Holm) 0.975;
  - MCD +0.065 [−0.022, +0.148], p (Holm) 0.052.
- **Removing one group from Full,** effect on UTMOSv2:
  - BigVGAN: −0.271 [−0.299, −0.244];
  - F0: −0.327 [−0.357, −0.297];
  - VITS2: +0.003 [−0.023, +0.029], p (Holm) 0.595.
- **Shapley shares of the UTMOSv2 gain:** BigVGAN 56 %, VITS2 21 %, F0 23 %.
- **Interactions are strong:** B×V is −0.231 [−0.247, −0.216].

**Pitch (sections F0 and Pitch intervention):**
- **The F0 conditioning does not steer the waveform pitch** (answer to Reviewer 1 in "Response to Reviewer 1"). The decoder uses it: the waveform changes when it is scaled or removed. A requested ±2 semitones moves the output pitch by only 0.4–1.1 % of the request (slope 0.0043–0.0114), in all F0 models including 1079.
- **The F0 group does not bring the waveform pitch closer to the recordings.** Alone it adds +56.9 cents of RMSE. 1079 and 773 do not differ: +1.0 cents [−5.6, +7.5], p (Holm) 1.0.

**Durations:**
- **NoF0 synthesizes about 40 % more audio** than the other cells.
- **The cause** is the stochastic part of its duration predictor: the test length is 1.300 of the reference at noise_w 0.8, against 0.879 at 0.
- **The checkpoint-selection metric** is a teacher-forced reconstruction loss and does not see durations.

**Recommended system:** Full. NoVITS2 is a smaller and faster alternative whose UTMOSv2 does not differ significantly from Full's; this is not equivalence.

**Limits:**
- one seed per cell;
- two GPU types;
- all runs still improving at 200 epochs;
- the Holm families, the RTF protocol and T13/T14 were defined after the results existed;
- UTMOSv2 is predicted, not MOS;
- MCD is not in dB;
- a non-significant difference is not equivalence.

## Method

**The manuscript's own method** (section 4.1), applied to every comparison:
- **Difference:** per sentence, on − off.
- **Interval:** mean with a 95 % BCa bootstrap CI over sentences (20,000 resamples, seed 1234).
- **Effect size:** paired Cohen's d_z = mean / sd(ddof = 1).
- **Test:** two-sided Wilcoxon signed-rank, SciPy defaults.
- **Check:** for the main pair this reproduces the manuscript's numbers to every reported digit. They are UTMOSv2 +0.441 [0.416, 0.468], d_z 1.482; WER −0.533 pp [−1.318, 0.232], d_z −0.061, p = 0.115.

**Holm correction within families.** The CIs are not adjusted; the Holm correction adjusts the p-values only. The families were fixed in this analysis, after the results existed:

| Family | Comparisons | Holm across |
|---|---|---|
| **H**, main pair | PitchFlowNet 1079 vs Piper 773: WER, UTMOSv2, MCD, from the historical evaluation records (`evaluation/eval_h_vs_baseline/`) | the 3 metrics (m = 3) |
| **A**, ablation | 12 cube edges and Full vs Baseline (200 epochs), for each of WER, UTMOSv2 and MCD | the 13 pairs, within each metric (m = 13) |
| **P13**, waveform F0 | the four F edges, Full vs Baseline, 1079 vs 773 (DTW alignment) | the 6 comparisons, within each metric (m = 6) |
| **P14**, pitch intervention | slope per model | none (descriptive, against bands fixed in advance) |

**Differences from `ablation8-results.md`:**
- That page used percentile CIs from 10,000 resamples (seed 2026). This page uses the manuscript's BCa CIs.
- **How much the bounds move:** at most 0.001 for UTMOSv2 and 0.005 for MCD. For WER they move by up to 0.74 pp, because the per-sentence WER differences are skewed; BCa corrects for that skew.
- Means and p-values are identical.

**Sources:**
- per-sentence files of the 8 cells: `<WSL_DEV_HOME>/outputs-pfn-cr/eval/<label>/`;
- the historical pair: `evaluation/eval_h_vs_baseline/`, tracked;
- T13 records: `tests/waveform-f0-20261025/records.jsonl`; T14 records: `tests/pitch-shift-20261025/records.jsonl`;
- RTF: `eval/ablation8-rtf.json`, `historical-pair-rtf.json`, `historical-pair-rtf-t12.json`;
- checkpoints: `final/` and the historical checkpoint files.
- Outputs: `eval/ablation-final-stats.{json,csv}`, `ablation-final-tables.md`, `ablation-final-paper.md`. Their hashes are in `manifests/camera-ready/ablation8-derived.json`.

## Measurement conditions

### Two protocols, kept apart

The main pair and the ablation were trained and evaluated under different protocols. **A number of one column is never compared with a number of the other:** the difference would mix training budget, devices, batch, seeding and machine load with any component effect.

| | **1,100-epoch pair** (the paper's main result: Piper 773, PitchFlowNet 1079) | **200-epoch ablation** (eight cells, camera-ready) |
|---|---|---|
| Training | 1,100 epochs, 2 × RTX 4070 Ti, DDP, batch 16 per GPU (32), bf16, seed 1234 (hparams; `evidence/manuscript-crosswalk.md`) | 200 epochs = 156,400 G updates, 1 GPU (RTX 5070 or RTX 4070 Ti), batch 16, bf16, seeds 1234 / 1234 / 2026 (`experiment-plan.md`) |
| Checkpoint | minimum validation mel loss: epochs 773 and 1079. The selection under DDP is UNRESOLVED in the crosswalk | minimum `val_mel_l1_global` on the 100 validation utterances: epochs 188–199 |
| Evaluation record | the historical evaluation, `evaluation/eval_h_vs_baseline/` | the protocol locked on 21/10 (`eval-protocol.md`), `eval/<label>/` |
| CPU of the quality evaluation | Intel Core i7-14700 (manuscript) | Intel Core i7-14700 for the four Machine A cells; Intel Core i5-14400F for the four Machine B cells |
| Threads | **not recorded**. The harness sets none, so the PyTorch default applied; it is 12 in that environment today, but this is not proven for the run | **4** (`torch.set_num_threads(4)`, `threads` in every `protocol.json`) |
| Batch | 1: one utterance per synthesis call (`eval_harness.synthesize`) | 1, the same call |
| Synthesis | noise 0.667, length 1.0, noise_w 0.8, **unseeded** | the same scales, **seeded per utterance**: (2026 × 1,000,003 + i) mod 2³¹ |
| Checkpoint load | `load_from_checkpoint(strict=False)` | `torch.load(weights_only=True)`, strict state dict |
| Text frontend | piper-phonemize (espeak-ng, en-us) | the same |
| ASR for WER | Whisper `base.en` on the CPU, audio at 16 kHz, `jiwer` per utterance, harness normalisation | the same |
| UTMOSv2 | `create_model(pretrained=True)`, CPU, **unseeded** (random crop) | the same model, **seeded per utterance** |
| Software | environment `edgetts`, known only from a snapshot of 2026-10-12, taken after the run: torch 2.12.1, torchaudio 2.11.0, openai-whisper 20250625, jiwer 4.0.0, transformers 5.12.1, timm 1.0.27, scipy 1.15.3, librosa 0.11.0, numpy 2.2.6, piper-phonemize 1.1.0, UTMOSv2 commit `cc2700db` (`history/environment/`) | environment `pfn-eval` on both machines, with the same versions; recorded in every `protocol.json` and identical for the eight labels |
| Inline RTF of that run | 0.0559 / 0.0568, mean per utterance; threads and load unknown, so **not reportable** | recorded at 4 threads while other work ran, so **not reportable** |
| Reference UTMOSv2 (MCD module) | 3.8146 and 3.8181, unseeded (`mcd_baseline`, `mcd_configH`) | 3.8308, seeded, identical for every label |

**Measurements made later on the historical checkpoints use the camera-ready conventions** (strict `weights_only` load, the evaluation's seeds): the RTF benchmark, T13 and T14. They describe the 1,100-epoch checkpoints, not the historical evaluation.

### CPU RTF benchmark

**Tool:** `tools/camera_ready/benchmark_rtf.py`, at commit `a67caf3` for the eight cells; the historical pair was added at `5ff1c2b` and `7c08080`. Outputs: `eval/ablation8-rtf.json`, `historical-pair-rtf.json`, `historical-pair-rtf-t12.json`.

| Item | Setting |
|---|---|
| Machine | Machine A: **Intel Core i7-14700**, Windows 11 Pro, WSL 2 (Ubuntu 24.04.4, kernel 6.18.33.1). WSL exposes 24 vCPUs as 12 virtual cores × 2, with 62.8 GiB RAM. The GPUs are present but not used (`device: cpu`) |
| Threads | **1, 2 and 12** intra-op threads; 1 interop thread; `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set to the same count |
| Pinning | `taskset` to vCPU 0 (1 thread), vCPUs 0 and 2 (2 threads), vCPUs 0, 2, …, 22 (12 threads): one vCPU per virtual core. Hyper-V schedules the vCPUs on host cores, so P-core or E-core placement is **not controlled** |
| Batch | **1**: one utterance per call, no padding |
| Sentences | 10 validation utterances as warm-up, not timed; then the **first 100 test utterances** (610–859 s of audio per model) |
| Rounds | **3**; model order rotated each round; every (round, model, threads) cell in its own process |
| Timing | **model-only**: the harness's window around the forward pass. **End-to-end**: adds espeak phonemization and tensor building. RTF = Σ time / Σ audio duration (corpus RTF); the median and the range over the rounds are reported |
| Synthesis | noise 0.667, length 1.0, noise_w 0.8, with the evaluation's seeds |
| Checkpoint load | `torch.load(weights_only=True)`, strict state dict (legacy group semantics for the historical pair). SHA-256 checked before use |
| Machine state | refuses to start with a load average ≥ 1, another tmux session or GPU work |
| Software | environment `pfn-eval`, unchanged after 2026-10-22 20:27 +07, i.e. before every run: Python 3.10.20; **torch 2.12.1+cu130 on the CPU** (ATen OpenMP backend, Intel oneMKL 2024.2, oneDNN 3.11.2); numpy 2.2.6; piper-phonemize 1.1.0; pytorch-lightning 2.6.5 |

| Run | Models | Threads | Time (+07) | Load average at start |
|---|---|---|---|---:|
| 1 | eight 200-epoch cells | 1, 2 | 2026-10-24 20:31–21:12 | 0.51 |
| 2 | Piper 773, PitchFlowNet 1079 | 1, 2 | 2026-10-28 02:59–03:10 | 0.06 |
| 3 | Piper 773, PitchFlowNet 1079 | 12 | 2026-10-28 03:15–03:20 | 0.00 |

**Limits:**
- the placement on P-cores or E-cores is not controlled;
- the 12-thread rounds raise the load for each other (load average up to 9.8 before a later round);
- 100 of the 500 test sentences are timed;
- one machine.

### MCD as computed here

The MCD of both protocols comes from `piper_train/compute_reference_metrics.py`:

MCD = (10·√2 / ln 10) · (1/|P|) · Σ_{(i,j)∈P} ‖ ẑ_syn,i − ẑ_ref,j ‖₂

- **Features:** librosa MFCCs, C1–C24 (`n_mfcc` 25 with C0 dropped), `n_fft` 1024, hop 256, at 22,050 Hz. The other librosa defaults apply: 128 mel bands, Hann window, centred frames, log-power, DCT-II.
- **Normalisation:** ẑ = (c − μ) / σ per coefficient. μ and σ are the mean and standard deviation over all frames of the 500 reference utterances, with σ increased by 10⁻⁸.
- **Alignment:** P is the DTW path between the two normalised sequences (`librosa.sequence.dtw`, euclidean). The accumulated cost is divided by the path length |P|.
- **Audio:**
  - synthesized: the WAV, read at 22,050 Hz;
  - reference: the normalised training audio of the cache (`audio_norm_path`), not the raw LJSpeech WAV.
- **Consequence:** this is **not** the usual MCD in dB on mel-cepstra (for example SPTK or WORLD) without normalisation. The coefficients are z-normalised, so the values (≈ 25–27) are not in dB and cannot be compared with MCD values in the literature. They compare models within this study only.
- **If the paper reports MCD,** it should state this definition. The manuscript of 2026-09-24 does not report MCD.

### Ground-truth WER and UTMOSv2 (Table 1 of the manuscript)

**Source.**
- The manuscript's Ground Truth row, WER 6.67 % and UTMOSv2 3.830, comes from `evaluation/ground-truth/eval_pitchflownet_500_gt/gt_wer_utmos_summary.json`, of 2026-09-24: mean WER 0.066694, mean UTMOSv2 3.8301, 500 sentences. The run log is `eval_pitchflownet_500_gt_run.sanitized.log`.
- The crosswalk marks it MATCH. The sentences and their order equal the test split.

**What is not recorded:**
- The script of that run is not in the repository; only its log and outputs are.
- Whether it scored the raw LJSpeech WAVs or the normalised cache audio is not recorded.
- Its UTMOSv2 was not seeded, since it is the historical setup.

**Four scorings of the same 500 references exist:**

| Scoring | Audio | Seeded | Mean UTMOSv2 |
|---|---|---|---:|
| Ground-truth run, 2026-09-24 | not recorded | no | 3.8301 |
| MCD module, Piper 773 evaluation | normalised cache audio | no | 3.8146 |
| MCD module, PitchFlowNet 1079 evaluation | normalised cache audio | no | 3.8181 |
| MCD module, locked protocol (all eight cells) | normalised cache audio | yes | 3.8308 |

**How the four scorings agree:**
- The means agree within 0.016.
- Per sentence they correlate only at r = 0.61–0.67, with a mean absolute difference of about 0.17–0.18. That holds even between the two unseeded scorings of identical audio: UTMOSv2's random crop changes per-sentence scores by up to about 0.87.
- The 3.830 of the manuscript is therefore one unseeded scoring. Its mean lies within 0.016 of the other three scorings (3.8146–3.8308).

**Proposed for the paper:**
- report 3.830 as recorded, or 3.831 from the seeded locked protocol, and say which;
- the Ground Truth WER (6.67 %) exists only from the 2026-09-24 run.

## Training runs and the consistency of the evaluations

### The eight 200-epoch runs

#### X4. Training runs (final/<run_id>/)

| Model | GPU | Attempts | Watchdog resumes | End (UTC) | Best epoch | Best val_mel_l1_global | Value at epoch 199 | Slope, epochs 150–199 (per 10 epochs) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Full | RTX 5070 | 6 | 2 | 2026-10-21 03:08 | 188 | 21.3153 | 21.3991 | -0.0747 |
| NoF0 | RTX 5070 | 4 | 3 | 2026-10-22 10:18 | 192 | 21.1037 | 21.2832 | -0.0809 |
| NoVITS2 | RTX 4070 Ti | 1 | 0 | 2026-10-22 16:16 | 188 | 21.2745 | 21.4670 | -0.0727 |
| NoBigVGAN | RTX 4070 Ti | 1 | 0 | 2026-10-22 17:12 | 199 | 22.3429 | 22.3429 | -0.0755 |
| Baseline-BigVGAN | RTX 4070 Ti | 1 | 0 | 2026-10-23 16:09 | 199 | 21.0612 | 21.0612 | -0.1166 |
| Baseline-VITS2 | RTX 5070 | 1 | 0 | 2026-10-23 11:26 | 188 | 22.2346 | 22.7031 | -0.0887 |
| Baseline-F0 | RTX 5070 | 1 | 0 | 2026-10-24 09:24 | 195 | 22.5705 | 23.1776 | -0.1085 |
| Baseline | RTX 4070 Ti | 1 | 0 | 2026-10-23 13:20 | 194 | 22.1978 | 22.5097 | -0.0463 |

**What the table shows:**
- **Interruptions:**
  - Full had six attempts: four `CUDA error: unknown error` interruptions (two resumed by hand, two by the watchdog) and one stop for the budget change.
  - NoF0 had three interruptions, all resumed by the watchdog.
  - The six other runs ran in one attempt each.
  - Resume reproduces an uninterrupted run on the CPU (`test_train_cpu.py`). GPU training is not bit-exact.
- **Undertraining.** Every run was still improving:
  - `val_mel_l1_global` falls by 0.046–0.117 per 10 epochs over epochs 150–199;
  - every best epoch is in the last 25 epochs (188–199);
  - NoBigVGAN and Baseline-BigVGAN have their best at the last epoch, 199.
  - 200 epochs are 18.2 % of the historical epochs.
- **The selection metric does not see text-to-speech quality.**
  - `val_mel_l1_global` is the training forward pass: the decoder reconstructs a random segment from the posterior encoder's latent of the real spectrogram.
  - It does not pass through the text encoder, the flow or the duration predictor.
  - The best epoch is thus the best for decoder reconstruction. NoF0, whose duration predictor stretches the speech, still has the second-lowest value (21.10).
  - The rule was fixed before any test result (`experiment-plan.md`, decision (e)).

### Consistency of the eight evaluations

`ablation8_analysis.py check`: **122 of 122 checks passed** (`eval/ablation8-check.json`). For every label:
- **Sizes and split:** 500 sentences; MCD n 500 with none skipped; split test.
- **Checkpoint:**
  - it is the final best, and its SHA-256 equals `final/<run_id>/SHA256SUMS`;
  - 200 epochs and 156,400 G updates;
  - training tree `1ee13cb4…`;
  - the flags equal the cell; strict group semantics; seeds 1234 / 1234 / 2026.
- **Same sentences:** the same 500 sentences in the same order; `reference.jsonl` is byte-identical, and it is the test split of `split-manifest.json`.
- **Reference UTMOSv2:** mean 3.8308, identical per sentence on both machines.
- **Protocol:** identical settings, including the package versions. The evaluation tool has a single version in Git (`f773be2`).
- **Configs:** between cube neighbours they differ only in the group flags, `run_id`, `tier` and `device_note`. The edges to Full also differ in `max_g_updates` and `budget_label`.
- **The budget of Full:** nothing in its 275-epoch config depends on the planned epoch count. The LR decays per epoch with a fixed gamma, the MAS noise decays per G update, `warmup_epochs` is unused and `max_epochs` is 1110 everywhere. Its first 200 epochs follow the schedule of a 200-epoch run.

## Claims of the abstract and conclusion, checked

The claims are from `evidence/manuscript/final/main_draft.tex`, the draft of 2026-09-24 on this machine. A later draft may differ.

| # | Claim | Result | Status | Action proposed |
|---:|---|---|---|---|
| 1 | "mean UTMOS increases from 3.366 to 3.807" | +0.441 [0.416, 0.468], d_z 1.482, p (Holm, m = 3) 4.9 × 10⁻⁷⁹ (Table F1) | **Supported** | Keep; say "predicted (UTMOSv2)" |
| 2 | "mean WER decreases from 9.17 % to 8.64 % without statistical significance" | −0.53 pp [−1.32, +0.23], p 0.115, p (Holm) 0.229 | **Supported**: not significant, CI includes 0 | Keep. Do not word it as "comparable" or "equal" |
| 3 | "Mean model-inference RTF … increases from 0.0559 to 0.0568" | The two values are those of the historical record, whose thread count was not recorded. On the idle CPU with declared threads (three rounds each), Full is slower by 12.5 % at 1 thread (0.0789 → 0.0888), 12.6 % at 2 (0.0514 → 0.0579) and 10.7 % at 12 (0.0460 → 0.0509) | **Not reproduced.** The values match the record, but the ≈ 1.6 % increase does not hold under declared threads | Replace with the declared-thread values (appendix, RTF table), and change "approximately 1.6 %" in the introduction too |
| 4 | "17.21 million inference parameters, compared with 15.66 million" | Recounted from the checkpoints: 17,207,008 and 15,664,671 | **Match** under the manuscript's definition: without the posterior encoder, the SDP's posterior branch and the SDP flow dropped at inference; decoder weight norm folded | Keep. The definition is in section 4.3 |
| 5 | Table: size 69.00 / 62.80 MB | FP32 bytes of the counted tensors: 68.83 / 62.66 MB | **Partial**: 0.17 / 0.14 MB lower. The manuscript says that it includes retained buffers; these were not re-derived | Keep, or give the counted value with its definition |
| 6 | "component ablations … remain necessary to assess attribution" | All eight cells exist at 200 epochs, one seed (Table F2) | **Outdated** | Replace with the ablation result (draft below). Multiple seeds and listening tests are still missing |
| 7 | Conclusion: "component attribution … remain open"; section 4.3: "the proposed ablations … have no measured results" | as 6 | **Outdated** | as 6 |

**Not claimed, but needed for an honest reading of "explicit F0 conditioning":**
- **Pitch is not steered by the F0 branch** (P14, `ablation8-results.md` section 10). A ±2-semitone change of the decoder's F0 input moves the output pitch by 0.004–0.011 of the request, in every F0 model including 1079.
- **The F0 branch does not bring the waveform pitch closer to the recordings** (P13). 1079 and 773 do not differ:
  - F0 RMSE +1.0 cents [−5.6, +7.5], p (Holm) 1.0;
  - log-F0 correlation −0.005 [−0.020, +0.010], p (Holm) 1.0.
- **Suggestion:** if the paper describes F0 conditioning as pitch modelling, add these limits.

## Tables for the paper

### Table F1. PitchFlowNet vs Piper (historical evaluation, 500 test utterances)

| Metric | Piper | PitchFlowNet | Diff [95 % BCa CI] | d_z | p (Wilcoxon) | p (Holm, m = 3) |
|---|---:|---:|---|---:|---:|---:|
| UTMOSv2 | 3.366 | 3.807 | +0.441 [+0.416, +0.468] | +1.482 | 1.6e-79 | 4.9e-79 |
| WER (%) | 9.17 | 8.64 | -0.53 [-1.32, +0.23] | -0.061 | 0.115 | 0.229 |
| MCD | 26.064 | 26.000 | -0.064 [-0.175, +0.046] | -0.051 | 0.117 | 0.229 |

```latex
\begin{tabular}{lrrcrrr}
\toprule
Metric & Piper & PitchFlowNet & $\Delta$ [95\% BCa CI] & $d_z$ & $p$ & $p_{\mathrm{Holm}}$ \\
\midrule
UTMOSv2 & 3.366 & 3.807 & +0.441 [+0.416, +0.468] & +1.482 & $1.6\times10^{-79}$ & $4.9\times10^{-79}$ \\
WER (\%) & 9.17 & 8.64 & -0.53 [-1.32, +0.23] & -0.061 & 0.115 & 0.229 \\
MCD & 26.064 & 26.000 & -0.064 [-0.175, +0.046] & -0.051 & 0.117 & 0.229 \\
\bottomrule
\end{tabular}
```

**Measurement, Table F1:**
- the historical evaluation of the 1,100-epoch checkpoints, 500 test utterances;
- CPU synthesis, batch 1, thread count not recorded;
- synthesis and UTMOSv2 unseeded;
- Whisper `base.en`;
- MCD as defined in "MCD as computed here".

Suggested caption: *"Paired differences (PitchFlowNet − Piper) over 500 test utterances; 95 % BCa bootstrap CIs (20,000 resamples); two-sided Wilcoxon signed-rank tests, Holm-adjusted across the three metrics. Synthesis on CPU, one utterance per call. UTMOSv2 is a predicted score. MCD is a DTW distance on z-normalised MFCCs (C1–C24), not in dB."*

### Table F2. Ablation contrasts (200 epochs, 500 test utterances; Holm across the 13 pairs within each metric)

Leave-one-out rows give ablated − Full (the change when the component is removed); add-one rows give with − without.

| Contrast | ΔUTMOSv2 [95 % CI] | p (Holm) | ΔWER pp [95 % CI] | p (Holm) | ΔMCD [95 % CI] | p (Holm) |
|---|---|---:|---|---:|---|---:|
| w/o BigVGAN (leave-one-out) | -0.271 [-0.299, -0.244] | 6.2e-51 | +0.79 [-0.21, +1.81] | 0.816 | +0.285 [+0.195, +0.376] | 1.3e-07 |
| w/o VITS2 (leave-one-out) | +0.003 [-0.023, +0.029] | 0.595 | -0.34 [-1.36, +1.51] | 0.415 | +0.122 [+0.042, +0.200] | 0.013 |
| w/o F0 (leave-one-out) | -0.327 [-0.357, -0.297] | 1.7e-57 | +1.67 [+0.46, +3.39] | 0.085 | -0.063 [-0.149, +0.026] | 0.144 |
| + BigVGAN (add-one) | +0.554 [+0.523, +0.585] | 1.8e-78 | -0.73 [-1.60, +0.11] | 0.816 | +0.171 [+0.087, +0.252] | 2.6e-05 |
| + VITS2 (add-one) | +0.432 [+0.401, +0.462] | 2.9e-69 | +0.23 [-0.71, +1.16] | 1.000 | +0.050 [-0.029, +0.126] | 0.398 |
| + F0 (add-one) | +0.119 [+0.084, +0.155] | 2.6e-10 | +2.67 [+1.16, +5.52] | 0.085 | +1.155 [+1.048, +1.263] | 2.0e-57 |
| Full vs Baseline | +0.575 [+0.546, +0.604] | 6.7e-80 | +0.54 [-0.41, +1.51] | 0.975 | +0.065 [-0.022, +0.148] | 0.052 |

```latex
\begin{tabular}{lcccccc}
\toprule
Contrast & $\Delta$UTMOSv2 & $p_{\mathrm{Holm}}$ & $\Delta$WER (pp) & $p_{\mathrm{Holm}}$ & $\Delta$MCD & $p_{\mathrm{Holm}}$ \\
\midrule
w/o BigVGAN (leave-one-out) & -0.271 [-0.299, -0.244] & $6.2\times10^{-51}$ & +0.79 [-0.21, +1.81] & 0.816 & +0.285 [+0.195, +0.376] & $1.3\times10^{-7}$ \\
w/o VITS2 (leave-one-out) & +0.003 [-0.023, +0.029] & 0.595 & -0.34 [-1.36, +1.51] & 0.415 & +0.122 [+0.042, +0.200] & 0.013 \\
w/o F0 (leave-one-out) & -0.327 [-0.357, -0.297] & $1.7\times10^{-57}$ & +1.67 [+0.46, +3.39] & 0.085 & -0.063 [-0.149, +0.026] & 0.144 \\
+ BigVGAN (add-one) & +0.554 [+0.523, +0.585] & $1.8\times10^{-78}$ & -0.73 [-1.60, +0.11] & 0.816 & +0.171 [+0.087, +0.252] & $2.6\times10^{-5}$ \\
+ VITS2 (add-one) & +0.432 [+0.401, +0.462] & $2.9\times10^{-69}$ & +0.23 [-0.71, +1.16] & 1.000 & +0.050 [-0.029, +0.126] & 0.398 \\
+ F0 (add-one) & +0.119 [+0.084, +0.155] & $2.6\times10^{-10}$ & +2.67 [+1.16, +5.52] & 0.085 & +1.155 [+1.048, +1.263] & $2.0\times10^{-57}$ \\
Full vs Baseline & +0.575 [+0.546, +0.604] & $6.7\times10^{-80}$ & +0.54 [-0.41, +1.51] & 0.975 & +0.065 [-0.022, +0.148] & 0.052 \\
\bottomrule
\end{tabular}
```


**Reading Table F2:**
- **Intervals and p-values are corrected differently.** The intervals are not adjusted; the p-values are Holm-adjusted across the 13 pairs within each metric.
  - So a row can have a CI that excludes 0 and a p (Holm) above 0.05. An example is w/o F0 on WER: +1.67 pp [+0.46, +3.39], p (Holm) 0.085.
  - Such a row is not significant after correction.
- **Training conditions:** one training seed per cell, two GPU types. The machine is aliased with the V terms (`ablation8-results.md`, section 4).
- **Budget:** all runs were still improving at 200 epochs.

**Measurement, Table F2:**
- the locked protocol of 21/10, applied to the 200-epoch checkpoints;
- CPU synthesis, batch 1, **4 threads**;
- Intel Core i7-14700 (the Machine A cells) or i5-14400F (the Machine B cells); the scores do not depend on the CPU, and the reference UTMOSv2 is identical per sentence on both machines;
- synthesis and UTMOSv2 seeded per utterance;
- `pfn-eval` with the package versions above.
- **Do not compare with Table F1:** it is a different protocol.

Suggested caption: *"Ablation at a matched 200-epoch budget (one seed per configuration). Paired differences over 500 test utterances with 95 % BCa CIs; Holm-adjusted Wilcoxon p-values across the 13 contrasts of the 2³ design within each metric. Leave-one-out rows give ablated − full."*

## The eight models and the contribution of each component

### The eight cells

#### X1. The eight cells (locked protocol, 500 test utterances)

| Model | B | V | F | Best epoch | WER mean (%) | WER median (%) | UTMOSv2 mean | UTMOSv2 median | MCD mean | F0-pred. RMSE (ln Hz) | F0-pred. r | Audio (s) | Training GPU |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Full | 1 | 1 | 1 | 188 | 11.35 | 7.41 | 3.516 | 3.520 | 25.638 | 0.5399 | 0.1861 | 3009.9 | RTX 5070 |
| NoF0 | 1 | 1 | 0 | 192 | 13.02 | 8.70 | 3.189 | 3.203 | 25.576 | n/a | n/a | 4219.7 | RTX 5070 |
| NoVITS2 | 1 | 0 | 1 | 188 | 11.01 | 5.88 | 3.519 | 3.545 | 25.760 | 0.5827 | 0.1992 | 3036.8 | RTX 4070 Ti |
| Baseline-BigVGAN | 1 | 0 | 0 | 199 | 10.09 | 6.67 | 3.495 | 3.513 | 25.744 | n/a | n/a | 3000.0 | RTX 4070 Ti |
| NoBigVGAN | 0 | 1 | 1 | 199 | 12.14 | 7.69 | 3.244 | 3.270 | 25.923 | 0.4813 | 0.1784 | 2951.6 | RTX 4070 Ti |
| Baseline-VITS2 | 0 | 1 | 0 | 188 | 11.05 | 7.14 | 3.372 | 3.382 | 25.623 | n/a | n/a | 3057.6 | RTX 5070 |
| Baseline-F0 | 0 | 0 | 1 | 195 | 13.49 | 7.14 | 3.060 | 3.096 | 26.728 | 0.6200 | 0.1923 | 3070.6 | RTX 5070 |
| Baseline | 0 | 0 | 0 | 194 | 10.82 | 6.67 | 2.941 | 2.958 | 25.573 | n/a | n/a | 3076.9 | RTX 4070 Ti |

Reference test audio: 3256.2 s (500 raw WAVs, header durations).

- **Columns:**
  - WER is in percent.
  - UTMOSv2 is a predicted score.
  - MCD follows the definition in "MCD as computed here".
  - "F0-pred." is the F0-predictor metric (a), not the waveform pitch. See "F0: predictor log-F0 and waveform pitch".
  - "Audio" is the total synthesized duration of the 500 sentences; see "The long audio of NoF0".
- **Not shown:** the inline RTF of these evaluations is not reportable, so it is omitted. The CPU benchmark is in the appendix.

### Factorial effects and Shapley values

**Effect coding.** Per sentence, effect(term) = ¼ Σ over the 8 cells of sign(term, cell) × y(cell), with −1 = off and +1 = on.
- A main effect is mean(on) − mean(off), averaged over the other two groups.
- A two-way interaction X×Y is half the difference between the effect of X with Y on and with Y off.
- The CIs are BCa over sentences (20,000 resamples, seed 1234).

#### X2. Factorial effects, effect coding (mean over sentences [95 % BCa CI])

| Term | WER (pp) | UTMOSv2 | MCD |
|---|---|---|---|
| B | -0.50 [-1.13, +0.12] | +0.275 [+0.260, +0.291] | -0.282 [-0.334, -0.233] |
| V | +0.54 [-0.15, +1.12] | +0.077 [+0.062, +0.092] | -0.261 [-0.311, -0.214] |
| F | +0.76 [+0.19, +1.45] | +0.086 [+0.069, +0.102] | +0.384 [+0.336, +0.430] |
| B×V | +1.10 [+0.55, +1.96] | -0.231 [-0.247, -0.216] | +0.116 [+0.073, +0.159] |
| B×F | -1.13 [-2.01, -0.55] | +0.090 [+0.074, +0.106] | -0.344 [-0.394, -0.294] |
| V×F | -1.04 [-2.44, -0.34] | +0.014 [-0.001, +0.030] | -0.202 [-0.247, -0.154] |
| B×V×F | -0.25 [-0.85, +0.35] | +0.138 [+0.122, +0.153] | +0.225 [+0.179, +0.273] |

**Shapley attribution.** It is exact over the 8 cells: v(S) is the mean metric of the cell with the groups in S on, and the three values sum to Full − Baseline. Where the gap's CI includes 0 (WER, MCD), the percentages are not meaningful.

#### X3. Shapley attribution of the Full − Baseline gap [95 % BCa CI]; % of the gap

| Metric | Gap | B | V | F |
|---|---|---|---|---|
| UTMOSv2 | +0.575 [+0.546, +0.604] | +0.321 [+0.305, +0.337], 56 % | +0.123 [+0.108, +0.138], 21 % | +0.131 [+0.114, +0.149], 23 % |
| WER (pp) | +0.54 [-0.41, +1.51] | -0.59 [-1.16, -0.02], -109 % | +0.46 [-0.27, +1.02], 84 % | +0.67 [+0.05, +1.44], 125 % |
| MCD | +0.065 [-0.022, +0.148] | -0.207 [-0.259, -0.157], -318 % | -0.186 [-0.234, -0.139], -286 % | +0.459 [+0.409, +0.508], 703 % |

### The training machine aliases some effects

The two GPU types are not spread evenly over the cube:
- the RTX 5070 trained 111, 110, 010 and 001;
- the RTX 4070 Ti trained 101, 011, 100 and 000.

Coded ±1, the machine equals ½(V + B×V − V×F + B×V×F).

What this means:
- **Not biased by a machine offset:** if training on one GPU type shifted a metric by δ, the B, F and B×F effects would not be biased.
- **Biased by a machine offset:** V, B×V and B×V×F would each be biased by +δ/2, and V×F by −δ/2.
- **Shapley values:** δ/6 would be added to B and F, and 2δ/3 to V.
- **Edges that cross GPU types:** every V edge, Full vs Baseline, two of the four B edges and two of the four F edges.

δ cannot be estimated from these runs.

### Each component

The numbers below are from Tables F2, X2 and X3 and the appendix tables A (Holm across the 13 pairs within each metric).

- **B, BigVGAN.**
  - **UTMOSv2:** the largest effect: main effect +0.275 [+0.260, +0.291], and 56 % of the Full − Baseline gap.
  - **Where it helps:** three of its four edges raise UTMOSv2: +0.554 [+0.523, +0.585] on Baseline, +0.459 on Baseline-F0, and +0.271 from NoBigVGAN to Full.
  - **Where it hurts:** on Baseline-VITS2 → NoF0 it lowers UTMOSv2 by −0.184 [−0.216, −0.149], p (Holm) 5.8 × 10⁻²⁴. This is the B×V interaction.
  - **MCD:** main effect −0.282 [−0.334, −0.233]. On Baseline alone, however, it raises MCD by +0.171 [+0.087, +0.252].
  - **WER:** main effect −0.50 pp [−1.13, +0.12]. One edge is significant after Holm: Baseline-F0 → NoVITS2, −2.48 pp [−5.13, −1.25], p (Holm) 0.037.
  - **Machine and size:** it is balanced over the machines. It adds 3,648 inference-path elements.
- **V, VITS2.**
  - **UTMOSv2:** main effect +0.077 [+0.062, +0.092], but it depends strongly on B:
    - +0.432 [+0.401, +0.462] on Baseline;
    - +0.184 on Baseline-F0;
    - −0.306 [−0.338, −0.274] with B only;
    - −0.003 [−0.029, +0.023], p (Holm) 0.595, with B and F. This is not equivalence.
  - **WER:** on Baseline-BigVGAN it raises WER by +2.93 pp [+1.80, +4.64], p (Holm) 8.2 × 10⁻⁵.
  - **MCD:** main effect −0.261 [−0.311, −0.214]; NoVITS2 → Full is −0.122 [−0.200, −0.042], p (Holm) 0.013.
  - **Reliability:** it is the effect most exposed to the machine (above).
  - **Size:** +1,195,008 inference-path elements (+7.3 %).
- **F, F0.**
  - **UTMOSv2:** main effect +0.086 [+0.069, +0.102]. Its sign depends on the others:
    - +0.119 on Baseline;
    - −0.128 on Baseline-VITS2;
    - +0.024 [−0.005, +0.053] on Baseline-BigVGAN, p (Holm) 0.209;
    - +0.327 [+0.297, +0.357] from NoF0 to Full.
  - **MCD:** main effect +0.384 [+0.336, +0.430]; +1.155 [+1.048, +1.263] on Baseline.
  - **WER:** main effect +0.76 pp [+0.19, +1.45]. Neither of its two largest edges is significant after Holm:
    - +2.67 pp alone, p (Holm) 0.085;
    - −1.67 pp from NoF0 to Full, p (Holm) 0.085.
  - **Size:** +346,369 inference-path elements (+2.1 %).
- **Interactions.**
  - **UTMOSv2:** every term except V×F (+0.014 [−0.001, +0.030]) has a CI that excludes 0. B×V is −0.231, B×F +0.090, B×V×F +0.138.
  - **MCD:** all seven CIs exclude 0.
  - **WER:** B×V, B×F and V×F exclude 0, at about ±1 pp.
  - **Caveats:** one seed per cell, so training variance is not in these CIs; B×V and B×V×F are partly aliased with the machine.

## F0: predictor log-F0 and waveform pitch

**Two different F0 measurements exist.** They answer different questions and must not be mixed:

| | **(a) F0-predictor metric** | **(b) Waveform F0** (T13) |
|---|---|---|
| What is evaluated | the **F0 predictor's output**: the log-F0 it predicts per input token, before any synthesis | the **pitch of the synthesized waveform** |
| Level | **input token**: the `phoneme_ids` with the interleaved pad tokens and BOS/EOS, pooled over all tokens of all utterances | **frame**: harvest F0 at 5 ms, taken at the MFCC frames (11.6 ms) along the DTW path, per utterance |
| Reference | the cached frame F0 of the recording (WORLD DIO + StoneMask at hop 256, unvoiced frames filled by log-linear interpolation), **averaged over the segments given by the model's own predicted durations** (SDP, noise_w 0.8) | harvest F0 (60–600 Hz) of the normalised reference audio, aligned by MFCC-DTW as the MCD |
| Voicing | none: the reference is continuous (no unvoiced frames) | voiced in both (RMSE, offset, correlation); VDE, GPE, FFE for voicing and gross errors |
| Unit | RMSE in **natural-log Hz** (1 unit = 1,731 cents); Pearson r of log-F0 | RMSE and offset in **cents**; Pearson r of log-F0; VDE, GPE, FFE as fractions |
| Statistic | **one pooled value per model**; no per-utterance values, so **no CI and no test** | per-utterance values; BCa CIs; Wilcoxon with Holm (P13) |
| Models | F = 1 only (it needs the predictor) | all ten |
| Protocol | the locked protocol (`eval-protocol.md`) for the 200-epoch cells. The historical 1079 runs also computed it, on rows that were not recorded | supplementary, after the results; protocol agreed on 25/10 |
| Code | `piper_train/eval_harness.py`, `collect_f0_pairs` | `tools/camera_ready/waveform_f0.py` |

**Caveats of (a), read from the code:**
1. **The reference carries duration error.** The reference segments come from the predicted durations, not from the alignment of the recording. A duration error therefore shifts every later segment of the utterance, and the metric mixes duration and F0 errors. The function's docstring says "the same projection used during training via MAS", but training uses the MAS durations, so it is not the same projection.
2. **Pad tokens count.** About half of the tokens are the interleaved pad tokens, and BOS/EOS are included.
3. **Some tokens get a placeholder reference.** When the predicted durations run past the recording's frames, the remaining tokens get the reference 1 Hz (log 0), an extreme value.
4. **No voicing.** The reference is interpolated through unvoiced and silent frames.
5. **One pooled number per model,** so there is no CI and no test.
6. **The historical values do not compare with the locked ones.** The historical runs (0.34 ln Hz, r = 0.33) used unrecorded rows and unseeded durations, after 1,100 epochs.

**(a) Results.** `eval/ablation-final-stats.json` → `f0.predictor_metric`.

| Model | Protocol | RMSE (ln Hz) | RMSE (cents) | Pearson r (log-F0) | Source |
|---|---|---:|---:|---:|---|
| Full | locked (200 epochs), the 500 test rows, seeded | 0.5399 | 935 | 0.1861 | `eval/PFN-Full-200ep-best/summary.json` |
| NoVITS2 | locked (200 epochs), the 500 test rows, seeded | 0.5827 | 1009 | 0.1992 | `eval/PFN-NoVITS2-200ep-best/summary.json` |
| NoBigVGAN | locked (200 epochs), the 500 test rows, seeded | 0.4813 | 833 | 0.1784 | `eval/PFN-NoBigVGAN-200ep-best/summary.json` |
| Baseline-F0 | locked (200 epochs), the 500 test rows, seeded | 0.6200 | 1073 | 0.1923 | `eval/PFN-Baseline-F0-200ep-best/summary.json` |
| PitchFlowNet-1079, historical run f0_configH_v2 | historical, rows not recorded, unseeded | 0.3420 | 592 | 0.3320 | `evaluation/eval_h_vs_baseline/f0_configH_v2/summary.sanitized.json` |
| PitchFlowNet-1079, historical run eval_f0_ablation_E_vs_H | historical, rows not recorded, unseeded | 0.3411 | 591 | 0.3337 | `evaluation/other/eval_f0_ablation_E_vs_H/summary.sanitized.json` |

- **The paper's main evaluation computed no F0 metric:** `evaluation/eval_h_vs_baseline/summary.json` has `f0_rmse = null` for both 773 and 1079.
- **Piper 773 has no F0 predictor,** so (a) does not exist for it.

**(b) Results.** `eval/ablation-final-stats.json` → `f0.waveform`. Mean over 500 utterances, with the 95 % BCa CI:

| Model | F0 RMSE (cents) | log-F0 corr | VDE | GPE |
|---|---|---|---|---|
| Full | 439.1 [432.4, 445.9] | 0.338 [0.323, 0.353] | 0.159 [0.154, 0.163] | 0.372 [0.363, 0.381] |
| NoF0 | 445.9 [439.7, 452.2] | 0.352 [0.337, 0.366] | 0.162 [0.158, 0.166] | 0.372 [0.364, 0.380] |
| NoVITS2 | 451.1 [444.2, 457.9] | 0.339 [0.324, 0.353] | 0.157 [0.153, 0.161] | 0.375 [0.366, 0.383] |
| NoBigVGAN | 448.0 [440.8, 455.4] | 0.361 [0.346, 0.377] | 0.156 [0.152, 0.161] | 0.388 [0.379, 0.397] |
| Baseline-BigVGAN | 451.9 [445.4, 458.7] | 0.350 [0.335, 0.365] | 0.158 [0.154, 0.162] | 0.388 [0.379, 0.396] |
| Baseline-VITS2 | 447.7 [441.2, 454.4] | 0.356 [0.340, 0.371] | 0.160 [0.156, 0.164] | 0.381 [0.372, 0.389] |
| Baseline-F0 | 513.0 [505.0, 521.2] | 0.316 [0.300, 0.332] | 0.168 [0.164, 0.172] | 0.432 [0.422, 0.441] |
| Baseline | 456.1 [449.6, 462.8] | 0.339 [0.324, 0.354] | 0.160 [0.156, 0.164] | 0.395 [0.386, 0.404] |
| Hist. Baseline-773 | 451.6 [444.9, 458.4] | 0.374 [0.359, 0.389] | 0.155 [0.151, 0.159] | 0.385 [0.376, 0.394] |
| Hist. Full-1079 | 452.6 [445.4, 459.6] | 0.369 [0.353, 0.384] | 0.155 [0.151, 0.160] | 0.380 [0.371, 0.390] |

- The paired comparisons of (b) are table P13 in the appendix.

**Reading:**
- **The two RMSEs are not the same quantity.** (a), 833–1073 cents in the 200-epoch cells, is far larger than (b), 439–513 cents. The reason is (a)'s construction: duration-driven segments, pad tokens and an interpolated reference. A smaller (a) does not mean a more accurate pitch.
- **(a) says nothing about the waveform.** On (b), the F = 1 cells are not closer to the recordings than the F = 0 cells (P13). Scaling the predicted F0 does not move the output pitch (P14).
- **The manuscript (draft of 2026-09-24) reports neither (a) nor (b),** so it has no F0 sentence to correct.
- **Sentences corrected in this repository** (this commit):
  - `experiment-plan.md`, "Evaluation": it listed waveform F0 and the pitch-shift test as part of the locked evaluation. It now says that the lock kept the historical metrics only, and that T13, T14 and T15 were added later.
  - `ablation8-results.md`, sections 2, 4, 6 and 9: the F0 RMSE there is now named as the F0-predictor metric in ln Hz and distinguished from the waveform pitch.
- **Not changed:** `eval-protocol.md` stays as locked. Its row "F0 predictor metric" is correct but brief; this section gives the full definition.
- **If the paper reports F0 numbers,** use (b) for pitch accuracy and state its definition. Report (a) only as a property of the predictor, with caveats 1–3.

## Pitch intervention: can the F0 conditioning steer the pitch? (T14)

**Question.** Does changing the F0 that the decoder receives change the pitch of the output?

**Protocol.** Fixed on 2026-10-25 before the run; supplementary, outside the locked protocol.
- **Tool:** `tools/camera_ready/pitch_shift.py`, commit `524708f`.
- **Intervention:** a hook on `dec.forward` scales the decoder's F0 input by 2^(s/12) for s = −2, 0, +2 semitones, or removes it ("off").
  - With the evaluation's per-utterance seeds, the latent z, the durations and the noise are identical in every condition.
  - No file under `code/` was changed.
- **Models and sentences:** the best checkpoints of the four F = 1 cells, and PitchFlowNet 1079; the first 100 test utterances; CPU, 4 threads, batch 1.
- **F0 extraction:** pyworld harvest (60–600 Hz, 5 ms), with equal lengths across conditions, so frames pair one to one.
- **Metric:** per utterance, the median of 12·log₂(F0_c / F0_0) over the frames voiced in both; slope = (shift(+2) − shift(−2)) / 4.
- **Interpretation bands, fixed before the run:**
  - slope ≥ 0.8 with a CI excluding 0: controllable;
  - 0.2–0.8 with a CI excluding 0: partly controllable;
  - otherwise: not controllable.

**Checks:**
- At s = 0 every model reproduced the evaluation's durations, 100 of 100. For NoVITS2 and NoBigVGAN, whose evaluation WAVs are on this machine, the WAVs were byte-identical, 100 of 100.
- The hook acts: the waveform differs from s = 0 by a median relative RMS of 0.09–0.22 for ±2 semitones, and 0.59–0.90 for "off". The relative RMS includes phase differences, so it is not a perceptual measure.

### X7. Pitch intervention: achieved shift in semitones [95 % BCa CI], 100 test utterances

| Model | +2 requested | −2 requested | Off | Slope | Reading (bands fixed in advance) | Voicing agreement +2 / −2 / off |
|---|---|---|---|---|---|---|
| Full | +0.018 [+0.017, +0.019] | -0.014 [-0.015, -0.014] | -0.078 [-0.083, -0.073] | 0.0081 [0.0078, 0.0085] | not controllable | 0.935 / 0.932 / 0.903 |
| NoVITS2 | +0.020 [+0.019, +0.021] | -0.017 [-0.018, -0.016] | -0.059 [-0.065, -0.055] | 0.0092 [0.0088, 0.0096] | not controllable | 0.929 / 0.927 / 0.890 |
| NoBigVGAN | +0.009 [+0.008, +0.010] | -0.008 [-0.009, -0.008] | -0.152 [-0.162, -0.144] | 0.0043 [0.0041, 0.0046] | not controllable | 0.943 / 0.942 / 0.871 |
| Baseline-F0 | +0.013 [+0.012, +0.014] | -0.012 [-0.012, -0.011] | -0.423 [-0.523, -0.366] | 0.0062 [0.0058, 0.0065] | not controllable | 0.926 / 0.927 / 0.857 |
| Hist. Full-1079 | +0.023 [+0.022, +0.024] | -0.022 [-0.024, -0.021] | -0.009 [-0.013, -0.006] | 0.0114 [0.0109, 0.0120] | not controllable | 0.926 / 0.923 / 0.873 |

**Reading:**
- **The pitch of the waveform does not follow the F0 conditioning.** A requested ±2 semitones moves it by 0.008–0.023 semitones. By the fixed bands, it is **not controllable** in every model, including the paper's PitchFlowNet 1079.
- **The decoder uses the conditioning, but not for pitch:**
  - the waveform changes clearly when F0 is scaled and strongly when it is removed;
  - removing the conditioning moves the pitch by only −0.009 to −0.423 semitones.
  - In `SynthesizerTrn.infer` the predicted F0 reaches only the decoder, so the pitch is presumably carried by the latent from the prior and the flow. That was not tested (below).
- **Not tested:**
  - shifting the latent;
  - larger shifts;
  - the quality of the shifted audio.

### Response to Reviewer 1: does the decoder use the F0 conditioning?

**Question.** Does the decoder use the F0 it receives, and does that F0 set the pitch of the output?

**Test.** The intervention above (T14), with the existing checkpoints:
- only the decoder's F0 input is changed: scaled by ±2 semitones, or removed;
- the latent z, the durations and the noise are held fixed by the evaluation's per-utterance seeds;
- the pitch is measured on the output waveform (harvest), on 100 test utterances.

**Answer, in two parts:**
- **The decoder uses the conditioning.** Changing it changes the waveform: the median relative RMS difference from s = 0 is 0.09–0.22 for ±2 semitones, and 0.59–0.90 when the conditioning is removed.
- **The conditioning does not set the pitch.**
  - A requested ±2 semitones moves the output pitch by 0.008–0.023 semitones: slope 0.0043–0.0114, with every 95 % BCa CI inside 0.0041–0.0120.
  - Removing the conditioning moves the pitch by −0.009 to −0.423 semitones.
  - By the bands fixed before the run, the pitch is not controllable through the conditioning in any of the five models. That includes PitchFlowNet 1079: slope 0.0114 [0.0109, 0.0120].
- **The waveform F0 against the recordings agrees (P13):**
  - the F0 group does not bring the waveform pitch closer to the recordings;
  - PitchFlowNet 1079 and Piper 773 do not differ: +1.0 cents [−5.6, +7.5], p (Holm) 1.0. This is not equivalence.

**What this test does not show:**
- the effect of larger shifts, or of shifting the latent;
- the perceptual quality of the shifted audio;
- whether the conditioning helps through another route. The F0 group's UTMOSv2 gain with BigVGAN and VITS2 is a predicted score, from one seed per cell.

**Suggested response text** (to adapt):

*"We tested whether the decoder uses the F0 conditioning by intervening on it in the trained checkpoints. With the latent, the durations and the noise held fixed, we scaled the decoder's F0 input by ±2 semitones or removed it, and measured the pitch of the output waveform on 100 test utterances. The conditioning changes the waveform (median relative RMS difference 0.09–0.22 for ±2 semitones, 0.59–0.90 when removed), but the output pitch moves by only 0.008–0.023 semitones for a requested ±2 semitones (slope 0.0114 [0.0109, 0.0120] for PitchFlowNet). The decoder therefore uses the F0 signal, but the pitch of the output is not controlled by it."*

If this text is used, any sentence of the manuscript that presents the F0 branch as controlling pitch has to change to match. The manuscript is not edited here.

## The long audio of NoF0

**What the durations show.** The total synthesized durations are in X1. Seven cells produce 2,952–3,077 s for the 500 sentences, against 3,256.2 s of reference audio. NoF0 produces 4,219.7 s.

### X5. Synthesized duration, NoF0 against Full (per sentence)

Median ratio 1.402; 10th–90th percentile 1.252–1.565; share of sentences more than 10 % longer 98.6 %.

**The stretch is about uniform:**
- the excess grows with length (≈ 0.385 × Full duration + 0.10 s);
- it does not relate to punctuation (r = −0.06);
- so NoF0 speaks more slowly throughout; it does not add pauses.
- **A possible effect on quality:** within NoF0, the more stretched sentences score lower on UTMOSv2 (r = −0.24). This is a plausible contributor to its low UTMOSv2, not a tested cause.

**Duration diagnostics** (run on 24/10).
- **Tool:** `tools/camera_ready/duration_diag.py`, commit `c8d3144`. Results: `tests/duration-diag-20261024/`.
- **What ran:** only the text encoder and the duration predictor, and for the validation part the posterior encoder, the flow and MAS. No audio was written.
- **Check:** with noise_w 0.8 and the evaluation's seeds, every best checkpoint reproduced the evaluation's durations, 500 of 500.

### X6. Duration diagnostics: predicted length / reference frames (test) and predicted / MAS (validation)

| Model | Best epoch | Test, best, noise_w 0.8 | Test, best, noise_w 0 | Test, last, noise_w 0.8 | Test, last, noise_w 0 | Validation pred/MAS, 0.8 | Validation pred/MAS, 0 | Per-phoneme mean abs log ratio, 0.8 / 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full | 188 | 0.927 | 0.800 | 0.923 | 0.813 | 0.945 | 0.815 | 0.603 / 0.490 |
| NoF0 | 192 | 1.300 | 0.879 | 1.221 | 0.751 | 1.327 | 0.893 | 0.719 / 0.515 |
| NoVITS2 | 188 | 0.935 | 0.824 | 0.925 | 0.825 | 0.943 | 0.833 | 0.584 / 0.480 |
| Baseline-BigVGAN | 199 | 0.924 | 0.817 | 0.924 | 0.817 | 0.941 | 0.831 | 0.576 / 0.478 |
| NoBigVGAN | 199 | 0.909 | 0.776 | 0.909 | 0.776 | 0.918 | 0.788 | 0.608 / 0.496 |
| Baseline-VITS2 | 188 | 0.942 | 0.825 | 0.980 | 0.869 | 0.950 | 0.833 | 0.599 / 0.492 |
| Baseline-F0 | 195 | 0.946 | 0.855 | 0.974 | 0.879 | 0.955 | 0.866 | 0.603 / 0.486 |
| Baseline | 194 | 0.948 | 0.843 | 0.955 | 0.848 | 0.955 | 0.859 | 0.595 / 0.490 |

**Reading:**
- **The stretch comes from the stochastic part of NoF0's duration predictor.**
  - Without noise, its test length is 0.879 of the reference, near the others (0.776–0.855).
  - With the evaluation's noise_w 0.8 it is 1.300: the noise lengthens NoF0 by 1.48×, against 1.11–1.17× for the others.
- **On validation,** against its own MAS alignment, it predicts 1.33× the aligned length (others 0.92–0.96), with the largest per-phoneme error (0.719 against 0.576–0.608).
- **It is also present at epoch 199:** 1.221 with noise.
- **The cause is in the learned weights, not in the code.** `use_f0` does not change the inference duration path, and only one of the four F = 0 cells is stretched.
  - So it is not a general effect of removing F0.
  - With one seed per cell, a run-specific outcome cannot be separated from an interaction.
- **The evaluation is still valid for its protocol:** it used noise_w 0.8, the historical default, so NoF0's scores stand under that protocol. No re-evaluation at another noise_w was run.

## Cost against quality

### Size and training cost

#### X8. Size and training cost of the eight cells (eval/ablation8-cost.json)

| Model | Inference-path elements | Generator | Whole model | Checkpoint (bytes) | GPU | Wall clock (h) | Active GPU-h | G updates / s | Median epoch (min) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Full | 17,970,805 | 25,208,822 | 72,794,737 | 874,731,173 | RTX 5070 | 31.65 | 28.75 | 1.511 | 8.47 |
| NoF0 | 17,624,436 | 24,862,453 | 72,448,368 | 870,560,277 | RTX 5070 | 28.93 | 28.45 | 1.527 | 8.42 |
| NoVITS2 | 16,775,797 | 24,013,814 | 71,043,888 | 853,606,869 | RTX 4070 Ti | 21.71 | 21.71 | 2.001 | 6.50 |
| Baseline-BigVGAN | 16,429,428 | 23,667,445 | 70,697,519 | 849,436,037 | RTX 4070 Ti | 20.56 | 20.56 | 2.113 | 6.23 |
| NoBigVGAN | 17,967,157 | 25,205,174 | 72,508,147 | 871,187,440 | RTX 4070 Ti | 22.65 | 22.65 | 1.918 | 6.82 |
| Baseline-VITS2 | 17,620,788 | 24,858,805 | 72,161,778 | 867,016,544 | RTX 5070 | 25.10 | 25.10 | 1.731 | 7.42 |
| Baseline-F0 | 16,772,149 | 24,010,166 | 70,757,298 | 850,063,072 | RTX 5070 | 21.88 | 21.88 | 1.985 | 6.42 |
| Baseline | 16,425,780 | 23,663,797 | 70,410,929 | 845,892,176 | RTX 4070 Ti | 17.75 | 17.75 | 2.447 | 5.30 |

**Columns:**
- **Elements:** stored `state_dict` tensor elements, loaded with `weights_only=True`. The inference path is the generator without the posterior encoder.
  - This differs from the manuscript's definition, which is used for the historical pair in the appendix (17.21 M, 15.66 M): that one also leaves out the SDP's unused branches and folds the decoder's weight norm.
- **Training time:**
  - per machine; RTX 5070 and RTX 4070 Ti times are not comparable with each other;
  - the active time ends a dead attempt at its last checkpoint event;
  - the RTX 5070 drives a display, and evaluations sometimes ran beside training.

**Inference cost.** It is the CPU RTF of the benchmark: appendix, RTF table, with its measurement conditions. At 1 thread:
- Baseline 0.0682;
- B adds about 16 % (Baseline-BigVGAN 0.0790);
- V about 11 % (Baseline-VITS2 0.0758);
- F about 1 % (Baseline-F0 0.0690);
- Full 0.0880, about 29 % slower than Baseline;
- NoVITS2 0.0795, about 10 % faster than Full.

### Pareto sets

#### X9. Pareto sets of mean quality against cost

- wer vs elements_inference_path: Baseline, Baseline-BigVGAN
- wer vs mean_rtf_recorded: Baseline, Baseline-BigVGAN, Baseline-F0
- wer vs rtf_cpu_1t_model: Baseline, Baseline-BigVGAN
- utmos vs elements_inference_path: Baseline, Baseline-BigVGAN, NoVITS2
- utmos vs mean_rtf_recorded: Baseline-F0, Full, NoVITS2
- utmos vs rtf_cpu_1t_model: Baseline, Baseline-BigVGAN, Baseline-F0, Baseline-VITS2, NoVITS2
- mcd vs elements_inference_path: Baseline
- mcd vs mean_rtf_recorded: Baseline, Baseline-F0
- mcd vs rtf_cpu_1t_model: Baseline

**Reading:**
- **The top UTMOSv2 group:** NoVITS2 (3.519), Full (3.516) and Baseline-BigVGAN (3.495).
- **Tested differences inside it:**
  - NoVITS2 vs Full: −0.003, p (Holm) 0.595;
  - Baseline-BigVGAN vs NoVITS2: +0.024, p (Holm) 0.209.
  - Neither is significant, and neither shows equality.
  - Full vs Baseline-BigVGAN is not a cube edge, so it was not tested.
- **The UTMOSv2 front against the 1-thread CPU RTF** holds Baseline, Baseline-F0, Baseline-VITS2, Baseline-BigVGAN and NoVITS2. Full is not on it, because NoVITS2 has a higher mean at a lower RTF. That is a ranking of point estimates, not a test.
- **The fronts against `mean_rtf_recorded`** use the RTF recorded inline during the evaluations, on two machines that were not idle; it is not reportable. They are listed for completeness only; use the CPU RTF fronts.
- **Baseline** is on every MCD and WER front, but its UTMOSv2 (2.941) is the lowest.

## Historical evaluation records of the main pair

These are copies of the records, not new evaluations; they are not comparable with the 200-epoch cells (Measurement conditions).

### X10. Historical evaluation records of the main pair (unseeded; the paper's evaluation)

| Run | Checkpoint | WER mean / median (%) | UTMOSv2 mean / median | Recorded RTF | F0-pred. RMSE (ln Hz) / r | Source |
|---|---|---|---|---:|---|---|
| main evaluation | PitchFlowNet-1079 | 8.64 / 4.76 | 3.807 / 3.813 | 0.0568 | — | `evaluation/eval_h_vs_baseline/summary.sanitized.json` |
| main evaluation | Piper-773 | 9.17 / 5.26 | 3.366 / 3.394 | 0.0559 | — | `evaluation/eval_h_vs_baseline/summary.sanitized.json` |
| second run of 1079 | PitchFlowNet-1079 | 8.43 / 4.55 | 3.824 / 3.835 | 1.3375 | — | `evaluation/eval_h_vs_baseline/f0_configH/summary.sanitized.json` |
| third run of 1079 | PitchFlowNet-1079 | 8.76 / 4.35 | 3.813 / 3.833 | 0.0476 | 0.342 / 0.332 | `evaluation/eval_h_vs_baseline/f0_configH_v2/summary.sanitized.json` |

- MCD, Piper-773: mean 26.064, median 25.904; reference UTMOSv2 3.8146; n 500, skipped 0
- MCD, PitchFlowNet-1079: mean 26.000, median 25.953; reference UTMOSv2 3.8181; n 500, skipped 0

- **Run-to-run spread:** the three runs of 1079 differ by up to 0.33 pp WER and 0.017 UTMOSv2, because the historical synthesis and UTMOSv2 were unseeded.
- **RTF of the second run:** its recorded RTF (1.3375) differs by more than 20× from the others. The record does not say why. It is not used anywhere.

## Recommended configuration

**At this budget, no configuration is best on all three metrics.** If one has to be chosen, **Full (B + V + F)** is the defensible system to report. The reasons:
- **UTMOSv2:** it is in the top group.
- **MCD:** it is lower than NoVITS2's: −0.122 [−0.200, −0.042], p (Holm) 0.013.
- **WER:** it does not differ significantly from its neighbours or from Baseline.

The record should also say:
- **BigVGAN carries the largest share** of the predicted-naturalness gain, at almost no inference-path size.
- **VITS2 adds no measurable UTMOSv2 or WER** once BigVGAN and F0 are present.
  - NoVITS2 is therefore a smaller (−6.7 % inference-path elements) and faster (0.0795 against 0.0880 at 1 thread) alternative.
  - Its UTMOSv2 does not differ significantly from Full's (−0.003 [−0.029, +0.023] for NoVITS2 → Full); this does not show that they are equal.
- **The F0 group helps predicted naturalness only together with BigVGAN and VITS2.** Its effect is not through pitch accuracy (sections F0 and Pitch intervention).

These statements hold for 200 epochs and one seed per cell; the runs were still improving.

## Limits

- **One training seed per configuration.** The bootstrap CIs and Wilcoxon tests cover variation across sentences, not across training runs. Interactions are estimated from single runs.
- **Two training machines.** The machine is aliased with V, B×V, V×F and B×V×F. Every V comparison and Full vs Baseline cross GPU types.
- **Undertraining.** Every run's best epoch is in the last 25 of 200, and the validation curves still fall. 200 epochs are 18.2 % of the historical epochs.
- **The selection metric** is a teacher-forced reconstruction loss; it does not see the duration predictor.
- **Interrupted runs.** Full had four CUDA interruptions and a budget switch; NoF0 had three. GPU training is not bit-exact across a resume.
- **Analysis choices made after the results:** the Holm families, the CI method of the ablation tables (made to match the manuscript), the RTF protocol, T13, T14 and the duration diagnostics. The CIs are not adjusted for multiplicity.
- **Metrics:**
  - UTMOSv2 is a predicted score; its per-sentence values carry random-crop noise when unseeded.
  - MCD is a DTW distance on z-normalised MFCCs, not in dB.
  - The F0-predictor metric mixes duration and F0 errors.
  - A non-significant difference is not equivalence.
- **RTF:** one WSL machine, with P-core or E-core placement not controlled, and 100 of 500 sentences. The historical inline RTFs have no thread count.
- **Two protocols:** the main pair and the ablation come from different training and evaluation protocols. Numbers across them are never compared.
- **No human listening test.**

## Appendix tables

Every comparison, all metrics; generated in the same run.

### H. Main pair: PitchFlowNet (epoch 1079) vs Piper (epoch 773), historical evaluation

| Comparison (off → on) | Metric | Mean off | Mean on | Diff (on − off) [95 % BCa CI] | d_z | p (raw) | p (Holm) | m |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Piper-773 -> PitchFlowNet-1079 | WER (pp) | 9.17 | 8.64 | -0.53 [-1.32, +0.23] | -0.061 | 0.115 | 0.229 | 3 |
| Piper-773 -> PitchFlowNet-1079 | UTMOSv2 | 3.366 | 3.807 | +0.441 [+0.416, +0.468] | +1.482 | 1.6e-79 | 4.9e-79 | 3 |
| Piper-773 -> PitchFlowNet-1079 | MCD | 26.064 | 26.000 | -0.064 [-0.175, +0.046] | -0.051 | 0.117 | 0.229 | 3 |

### A. Ablation, UTMOSv2: 12 cube edges and Full vs Baseline (200 epochs)

| Comparison (off → on) | Metric | Mean off | Mean on | Diff (on − off) [95 % BCa CI] | d_z | p (raw) | p (Holm) | m |
|---|---|---:|---:|---|---:|---:|---:|---:|
| B: Baseline -> Baseline-BigVGAN | UTMOSv2 | 2.941 | 3.495 | +0.554 [+0.523, +0.585] | +1.576 | 1.5e-79 | 1.8e-78 | 13 |
| B: Baseline-F0 -> NoVITS2 | UTMOSv2 | 3.060 | 3.519 | +0.459 [+0.424, +0.493] | +1.160 | 4.5e-69 | 4.5e-68 | 13 |
| B: Baseline-VITS2 -> NoF0 | UTMOSv2 | 3.372 | 3.189 | -0.184 [-0.216, -0.149] | -0.487 | 9.6e-25 | 5.8e-24 | 13 |
| B: NoBigVGAN -> Full | UTMOSv2 | 3.244 | 3.516 | +0.271 [+0.244, +0.299] | +0.856 | 7.7e-52 | 6.2e-51 | 13 |
| V: Baseline -> Baseline-VITS2 | UTMOSv2 | 2.941 | 3.372 | +0.432 [+0.401, +0.462] | +1.244 | 2.6e-70 | 2.9e-69 | 13 |
| V: Baseline-F0 -> NoBigVGAN | UTMOSv2 | 3.060 | 3.244 | +0.184 [+0.150, +0.219] | +0.462 | 3.1e-21 | 1.5e-20 | 13 |
| V: Baseline-BigVGAN -> NoF0 | UTMOSv2 | 3.495 | 3.189 | -0.306 [-0.338, -0.274] | -0.838 | 5.3e-50 | 3.7e-49 | 13 |
| V: NoVITS2 -> Full | UTMOSv2 | 3.519 | 3.516 | -0.003 [-0.029, +0.023] | -0.010 | 0.595 | 0.595 | 13 |
| F: Baseline -> Baseline-F0 | UTMOSv2 | 2.941 | 3.060 | +0.119 [+0.084, +0.155] | +0.290 | 8.8e-11 | 2.6e-10 | 13 |
| F: Baseline-VITS2 -> NoBigVGAN | UTMOSv2 | 3.372 | 3.244 | -0.128 [-0.159, -0.097] | -0.358 | 1.9e-14 | 7.7e-14 | 13 |
| F: Baseline-BigVGAN -> NoVITS2 | UTMOSv2 | 3.495 | 3.519 | +0.024 [-0.005, +0.053] | +0.072 | 0.105 | 0.209 | 13 |
| F: NoF0 -> Full | UTMOSv2 | 3.189 | 3.516 | +0.327 [+0.297, +0.357] | +0.946 | 1.9e-58 | 1.7e-57 | 13 |
| BVF: Baseline -> Full | UTMOSv2 | 2.941 | 3.516 | +0.575 [+0.546, +0.604] | +1.738 | 5.1e-81 | 6.7e-80 | 13 |

### A. Ablation, WER (pp): 12 cube edges and Full vs Baseline (200 epochs)

| Comparison (off → on) | Metric | Mean off | Mean on | Diff (on − off) [95 % BCa CI] | d_z | p (raw) | p (Holm) | m |
|---|---|---:|---:|---|---:|---:|---:|---:|
| B: Baseline -> Baseline-BigVGAN | WER (pp) | 10.82 | 10.09 | -0.73 [-1.60, +0.11] | -0.074 | 0.136 | 0.816 | 13 |
| B: Baseline-F0 -> NoVITS2 | WER (pp) | 13.49 | 11.01 | -2.48 [-5.13, -1.25] | -0.125 | 0.003 | 0.037 | 13 |
| B: Baseline-VITS2 -> NoF0 | WER (pp) | 11.05 | 13.02 | +1.97 [+0.78, +3.71] | +0.122 | 0.011 | 0.099 | 13 |
| B: NoBigVGAN -> Full | WER (pp) | 12.14 | 11.35 | -0.79 [-1.81, +0.21] | -0.069 | 0.142 | 0.816 | 13 |
| V: Baseline -> Baseline-VITS2 | WER (pp) | 10.82 | 11.05 | +0.23 [-0.71, +1.16] | +0.022 | 0.543 | 1.000 | 13 |
| V: Baseline-F0 -> NoBigVGAN | WER (pp) | 13.49 | 12.14 | -1.35 [-4.19, +0.12] | -0.060 | 0.989 | 1.000 | 13 |
| V: Baseline-BigVGAN -> NoF0 | WER (pp) | 10.09 | 13.02 | +2.93 [+1.80, +4.64] | +0.187 | 6.3e-06 | 8.2e-05 | 13 |
| V: NoVITS2 -> Full | WER (pp) | 11.01 | 11.35 | +0.34 [-1.51, +1.36] | +0.022 | 0.059 | 0.415 | 13 |
| F: Baseline -> Baseline-F0 | WER (pp) | 10.82 | 13.49 | +2.67 [+1.16, +5.52] | +0.117 | 0.008 | 0.085 | 13 |
| F: Baseline-VITS2 -> NoBigVGAN | WER (pp) | 11.05 | 12.14 | +1.09 [+0.14, +2.06] | +0.101 | 0.034 | 0.275 | 13 |
| F: Baseline-BigVGAN -> NoVITS2 | WER (pp) | 10.09 | 11.01 | +0.93 [-0.04, +2.72] | +0.064 | 0.221 | 0.883 | 13 |
| F: NoF0 -> Full | WER (pp) | 13.02 | 11.35 | -1.67 [-3.39, -0.46] | -0.102 | 0.008 | 0.085 | 13 |
| BVF: Baseline -> Full | WER (pp) | 10.82 | 11.35 | +0.54 [-0.41, +1.51] | +0.049 | 0.325 | 0.975 | 13 |

### A. Ablation, MCD: 12 cube edges and Full vs Baseline (200 epochs)

| Comparison (off → on) | Metric | Mean off | Mean on | Diff (on − off) [95 % BCa CI] | d_z | p (raw) | p (Holm) | m |
|---|---|---:|---:|---|---:|---:|---:|---:|
| B: Baseline -> Baseline-BigVGAN | MCD | 25.573 | 25.744 | +0.171 [+0.087, +0.252] | +0.182 | 3.2e-06 | 2.6e-05 | 13 |
| B: Baseline-F0 -> NoVITS2 | MCD | 26.728 | 25.760 | -0.968 [-1.087, -0.856] | -0.732 | 2.0e-42 | 2.4e-41 | 13 |
| B: Baseline-VITS2 -> NoF0 | MCD | 25.623 | 25.576 | -0.047 [-0.137, +0.043] | -0.046 | 0.222 | 0.444 | 13 |
| B: NoBigVGAN -> Full | MCD | 25.923 | 25.638 | -0.285 [-0.376, -0.195] | -0.275 | 1.4e-08 | 1.3e-07 | 13 |
| V: Baseline -> Baseline-VITS2 | MCD | 25.573 | 25.623 | +0.050 [-0.029, +0.126] | +0.056 | 0.133 | 0.398 | 13 |
| V: Baseline-F0 -> NoBigVGAN | MCD | 26.728 | 25.923 | -0.805 [-0.914, -0.690] | -0.631 | 2.6e-37 | 2.8e-36 | 13 |
| V: Baseline-BigVGAN -> NoF0 | MCD | 25.744 | 25.576 | -0.168 [-0.269, -0.071] | -0.149 | 0.001 | 0.009 | 13 |
| V: NoVITS2 -> Full | MCD | 25.760 | 25.638 | -0.122 [-0.200, -0.042] | -0.135 | 0.002 | 0.013 | 13 |
| F: Baseline -> Baseline-F0 | MCD | 25.573 | 26.728 | +1.155 [+1.048, +1.263] | +0.943 | 1.5e-58 | 2.0e-57 | 13 |
| F: Baseline-VITS2 -> NoBigVGAN | MCD | 25.623 | 25.923 | +0.301 [+0.205, +0.401] | +0.271 | 3.9e-09 | 3.9e-08 | 13 |
| F: Baseline-BigVGAN -> NoVITS2 | MCD | 25.744 | 25.760 | +0.016 [-0.072, +0.104] | +0.016 | 0.403 | 0.444 | 13 |
| F: NoF0 -> Full | MCD | 25.576 | 25.638 | +0.063 [-0.026, +0.149] | +0.063 | 0.036 | 0.144 | 13 |
| BVF: Baseline -> Full | MCD | 25.573 | 25.638 | +0.065 [-0.022, +0.148] | +0.067 | 0.010 | 0.052 | 13 |

### P13. Waveform F0 against the recordings (DTW)

| Comparison (off → on) | Metric | Mean off | Mean on | Diff (on − off) [95 % BCa CI] | d_z | p (raw) | p (Holm) | m |
|---|---|---:|---:|---|---:|---:|---:|---:|
| F: Baseline -> Baseline-F0 | F0 RMSE (cents) | 456.1 | 513.0 | +56.9 [+49.3, +64.5] | +0.656 | 2.0e-38 | 1.2e-37 | 6 |
| F: Baseline -> Baseline-F0 | log-F0 corr | 0.339 | 0.316 | -0.023 [-0.040, -0.006] | -0.116 | 0.002 | 0.011 | 6 |
| F: Baseline -> Baseline-F0 | VDE | 0.160 | 0.168 | +0.008 [+0.004, +0.012] | +0.181 | 1.9e-05 | 1.2e-04 | 6 |
| F: Baseline -> Baseline-F0 | GPE | 0.395 | 0.432 | +0.037 [+0.027, +0.046] | +0.340 | 5.0e-15 | 3.0e-14 | 6 |
| F: Baseline -> Baseline-F0 | F0 offset (cents) | 15.5 | 50.7 | +35.2 [+24.3, +46.4] | +0.279 | 7.3e-10 | 7.3e-10 | 6 |
| F: Baseline-VITS2 -> NoBigVGAN | F0 RMSE (cents) | 447.7 | 448.0 | +0.3 [-6.0, +6.5] | +0.004 | 0.639 | 1.000 | 6 |
| F: Baseline-VITS2 -> NoBigVGAN | log-F0 corr | 0.356 | 0.361 | +0.005 [-0.010, +0.020] | +0.032 | 0.217 | 0.650 | 6 |
| F: Baseline-VITS2 -> NoBigVGAN | VDE | 0.160 | 0.156 | -0.003 [-0.007, +0.001] | -0.073 | 0.098 | 0.488 | 6 |
| F: Baseline-VITS2 -> NoBigVGAN | GPE | 0.381 | 0.388 | +0.007 [-0.001, +0.015] | +0.078 | 0.049 | 0.147 | 6 |
| F: Baseline-VITS2 -> NoBigVGAN | F0 offset (cents) | -15.7 | 33.9 | +49.7 [+41.0, +58.3] | +0.502 | 1.8e-25 | 9.2e-25 | 6 |
| F: Baseline-BigVGAN -> NoVITS2 | F0 RMSE (cents) | 451.9 | 451.1 | -0.8 [-7.3, +5.5] | -0.011 | 0.937 | 1.000 | 6 |
| F: Baseline-BigVGAN -> NoVITS2 | log-F0 corr | 0.350 | 0.339 | -0.012 [-0.027, +0.004] | -0.067 | 0.067 | 0.333 | 6 |
| F: Baseline-BigVGAN -> NoVITS2 | VDE | 0.158 | 0.157 | -0.001 [-0.005, +0.003] | -0.031 | 0.120 | 0.488 | 6 |
| F: Baseline-BigVGAN -> NoVITS2 | GPE | 0.388 | 0.375 | -0.013 [-0.021, -0.005] | -0.149 | 0.004 | 0.017 | 6 |
| F: Baseline-BigVGAN -> NoVITS2 | F0 offset (cents) | 24.7 | -49.5 | -74.2 [-82.9, -65.7] | -0.755 | 3.7e-45 | 2.2e-44 | 6 |
| F: NoF0 -> Full | F0 RMSE (cents) | 445.9 | 439.1 | -6.8 [-13.0, -0.9] | -0.099 | 0.021 | 0.083 | 6 |
| F: NoF0 -> Full | log-F0 corr | 0.352 | 0.338 | -0.013 [-0.029, +0.002] | -0.075 | 0.075 | 0.333 | 6 |
| F: NoF0 -> Full | VDE | 0.162 | 0.159 | -0.003 [-0.007, +0.001] | -0.066 | 0.135 | 0.488 | 6 |
| F: NoF0 -> Full | GPE | 0.372 | 0.372 | +0.000 [-0.008, +0.008] | +0.001 | 0.714 | 0.714 | 6 |
| F: NoF0 -> Full | F0 offset (cents) | -4.0 | -32.8 | -28.7 [-36.4, -20.9] | -0.321 | 9.5e-12 | 1.9e-11 | 6 |
| BVF: Baseline -> Full | F0 RMSE (cents) | 456.1 | 439.1 | -17.0 [-23.0, -11.0] | -0.248 | 2.0e-08 | 9.9e-08 | 6 |
| BVF: Baseline -> Full | log-F0 corr | 0.339 | 0.338 | -0.001 [-0.017, +0.016] | -0.004 | 0.657 | 1.000 | 6 |
| BVF: Baseline -> Full | VDE | 0.160 | 0.159 | -0.001 [-0.005, +0.003] | -0.030 | 0.501 | 1.000 | 6 |
| BVF: Baseline -> Full | GPE | 0.395 | 0.372 | -0.023 [-0.030, -0.015] | -0.260 | 5.0e-09 | 2.5e-08 | 6 |
| BVF: Baseline -> Full | F0 offset (cents) | 15.5 | -32.8 | -48.2 [-56.7, -39.9] | -0.505 | 8.8e-24 | 3.5e-23 | 6 |
| historical: Hist. Baseline-773 -> Hist. Full-1079 | F0 RMSE (cents) | 451.6 | 452.6 | +1.0 [-5.6, +7.5] | +0.013 | 0.609 | 1.000 | 6 |
| historical: Hist. Baseline-773 -> Hist. Full-1079 | log-F0 corr | 0.374 | 0.369 | -0.005 [-0.020, +0.010] | -0.031 | 0.602 | 1.000 | 6 |
| historical: Hist. Baseline-773 -> Hist. Full-1079 | VDE | 0.155 | 0.155 | +0.000 [-0.004, +0.004] | +0.008 | 0.949 | 1.000 | 6 |
| historical: Hist. Baseline-773 -> Hist. Full-1079 | GPE | 0.385 | 0.380 | -0.004 [-0.013, +0.004] | -0.047 | 0.243 | 0.487 | 6 |
| historical: Hist. Baseline-773 -> Hist. Full-1079 | F0 offset (cents) | 21.3 | -24.3 | -45.6 [-55.2, -36.3] | -0.425 | 2.0e-19 | 6.0e-19 | 6 |

### P14. Pitch intervention: slope of the achieved over the requested shift (descriptive)

| Model | n | Slope [95 % BCa CI] |
|---|---:|---|
| Full | 100 | 0.0081 [0.0078, 0.0085] |
| NoVITS2 | 100 | 0.0092 [0.0088, 0.0096] |
| NoBigVGAN | 100 | 0.0043 [0.0041, 0.0046] |
| Baseline-F0 | 100 | 0.0062 [0.0058, 0.0065] |
| Hist. Full-1079 | 100 | 0.0114 [0.0109, 0.0120] |

### RTF, CPU benchmark (model RTF, median [min, max] of three rounds)

| Model | 1 thread | 2 threads | 12 threads |
|---|---|---|---|
| Baseline | 0.0682 [0.0674, 0.0683] | 0.0442 [0.0438, 0.0443] | — |
| Baseline-BigVGAN | 0.0790 [0.0785, 0.0796] | 0.0517 [0.0514, 0.0520] | — |
| Baseline-F0 | 0.0690 [0.0688, 0.0692] | 0.0449 [0.0446, 0.0454] | — |
| Baseline-VITS2 | 0.0758 [0.0757, 0.0759] | 0.0495 [0.0494, 0.0501] | — |
| Full | 0.0880 [0.0861, 0.0881] | 0.0567 [0.0560, 0.0584] | — |
| NoBigVGAN | 0.0766 [0.0761, 0.0769] | 0.0495 [0.0495, 0.0498] | — |
| NoF0 | 0.0917 [0.0914, 0.0924] | 0.0600 [0.0599, 0.0611] | — |
| NoVITS2 | 0.0795 [0.0791, 0.0801] | 0.0521 [0.0520, 0.0526] | — |
| Hist. Baseline-773 | 0.0789 [0.0772, 0.0803] | 0.0514 [0.0509, 0.0521] | 0.0460 [0.0457, 0.0462] |
| Hist. Full-1079 | 0.0888 [0.0873, 0.0895] | 0.0579 [0.0575, 0.0588] | 0.0509 [0.0502, 0.0526] |

**Measurement, RTF table:** see "CPU RTF benchmark".
- Intel Core i7-14700 under WSL 2, idle;
- CPU only, **batch 1**, **1 / 2 / 12 intra-op threads** with 1 interop thread, pinned one per virtual core;
- 10 warm-up and **100 test utterances**, 3 rounds, model-only time;
- torch 2.12.1 (OpenMP, oneMKL 2024.2, oneDNN 3.11.2), Python 3.10.20.

Suggested caption: *"Model-inference RTF on an Intel Core i7-14700 CPU (WSL 2), batch size 1, with 1, 2 or 12 intra-op threads; median [min, max] over three rounds of 100 test utterances after 10 warm-up utterances."*

### Inference parameters of the historical pair, recomputed

| Model | Elements | Millions | Manuscript (M) | FP32 bytes (MB, 10^6) |
|---|---:|---:|---:|---:|
| Piper-773 | 15,664,671 | 15.66 | 15.66 | 62.66 |
| PitchFlowNet-1079 | 17,207,008 | 17.21 | 17.21 | 68.83 |

**Notes on the appendix tables:**
- **RTF:**
  - It is the model-only RTF on Machine A, idle (WSL, Intel Core i7-14700). Threads are declared, pinned one per virtual core; the placement on P-cores or E-cores is not controlled.
  - It is the median over three rounds; the 8 cells used 100 test utterances.
  - No test is applied: three rounds give no sentence-level pairing.
  - The two historical values recorded in the paper (0.0559, 0.0568) have no recorded thread count. They are not in this table.
- **P13 numbers:** they use the historical pair synthesized with seeds (T13). They are not the historical evaluation.
- **Source of every number:** `eval/ablation-final-stats.json`.

## Draft text for the manuscript

Proposals only. Every number has its row above.

> **Abstract (replacing the RTF sentence and the last sentence).** "… Mean UTMOS increases from 3.366 to 3.807 (paired difference +0.441, 95 % CI [0.416, 0.468], Holm-adjusted p < 10⁻⁷⁸), whereas the decrease in mean WER from 9.17 % to 8.64 % is not statistically significant (−0.53 pp, 95 % CI [−1.32, 0.23], p = 0.115). On an Intel Core i7-14700 CPU with declared threads, the complete model is 11–13 % slower than Piper (model-inference RTF 0.0888 vs 0.0789 at one thread). A 2³ ablation of the three component groups at a matched 200-epoch budget attributes 56 % of the UTMOSv2 gain to the BigVGAN-style decoder, 21 % to the VITS2 refinements and 23 % to F0 conditioning, with strong interactions. Multiple training seeds and human listening tests remain necessary."

> **Results (ablation).** "Relative to the 200-epoch full model, removing the BigVGAN-style decoder lowers UTMOSv2 by 0.271 (95 % CI [0.244, 0.299], Holm-adjusted p < 10⁻⁵⁰) and removing F0 conditioning lowers it by 0.327 ([0.297, 0.357], p < 10⁻⁵⁶). Removing the VITS2 refinements does not change it detectably (+0.003, [−0.023, 0.029], p = 0.60), which does not establish equivalence. Added alone to the baseline, each group raises UTMOSv2 (BigVGAN +0.554, VITS2 +0.432, F0 +0.119; all p < 10⁻⁹). No contrast changes WER significantly after correction."

> **Limitations (pitch).** "Scaling the decoder's F0 input by ±2 semitones moves the output pitch by less than 2 % of the request, and the F0 branch does not bring the waveform pitch contour closer to the recordings; the F0 conditioning therefore does not provide explicit pitch control in these checkpoints."

Before using these sentences, check:
- "Holm-adjusted p < 10⁻⁷⁸": 4.9 × 10⁻⁷⁹ (F1).
- "p < 10⁻⁵⁰": 6.2 × 10⁻⁵¹. "p < 10⁻⁵⁶": 1.7 × 10⁻⁵⁷. "p < 10⁻⁹": the add-one rows are at most 2.6 × 10⁻¹⁰ (F2).
- "56 / 21 / 23 %": Shapley values in `ablation8-results.md`, section 4.
- "less than 2 %": the slopes are at most 0.0114, with a CI upper bound of 0.0120 (P14).
- "No contrast changes WER significantly": the smallest WER p (Holm) among the F2 rows is 0.085.
  - Among all 13 pairs, two WER edges are significant: Baseline-F0 → NoVITS2, p (Holm) 0.037, and Baseline-BigVGAN → NoF0, p (Holm) 8.2 × 10⁻⁵.
  - If the sentence speaks of the 13 pairs, it must say "except two".
