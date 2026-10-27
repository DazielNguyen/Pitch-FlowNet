# Ablation results: eight 200-epoch runs over BigVGAN, VITS2 and F0

**Scope.** SOICT 2026 paper 4478, camera-ready. This page analyses the eight cells of the 2^3 ablation (`experiment-plan.md`). The cells combine three groups:
- B = BigVGAN;
- V = VITS2;
- F = F0.

The analysis was done on the PC on 2026-10-24.
- **No new score.** Nothing was trained, synthesized, transcribed or re-scored for it. Every number below comes from the evaluation files written under the locked protocol (`eval-protocol.md`) and from the run records.
- **Main result of the paper.** It is still the historical pair (Baseline epoch 773, Full epoch 1079). These eight runs are a separate, budget-matched ablation at 200 epochs and one seed per cell.
- **Scope of the conclusions.** They hold for this budget only; they are not results at the historical 1,100-epoch budget.

**Reading the numbers:**
- UTMOSv2 is a **predicted** score, not MOS.
- A non-significant difference does **not** establish equivalence.
- WER differences are in percentage points (pp).

## Summary

**State on 2026-10-25.** All eight 200-epoch cells are trained and evaluated under the locked protocol, analysed, and benchmarked for CPU RTF. The long NoF0 audio has been traced to its duration predictor. The historical pair is reported as recorded, and its CPU RTF was measured again on 2026-10-28 (section 9). Pitch control through the F0 conditioning (section 10) and the waveform F0 against the recordings (section 11) have been tested.

**Main numbers** (test split, 500 utterances; 95 % CIs from the paired bootstrap; Holm across 13 pairs):
- **Full vs Baseline (200 epochs):**
  - UTMOSv2 +0.575 [+0.546, +0.604], p (Holm) 6.7 × 10⁻⁸⁰;
  - WER +0.54 pp [−0.42, +1.52], p (Holm) 0.98;
  - MCD +0.065 [−0.020, +0.148], p (Holm) 0.052.
- **Shapley shares of the UTMOSv2 gain:** BigVGAN 56 %, VITS2 21 %, F0 23 %.
- **Strong interactions on UTMOSv2.** B×V is −0.231: BigVGAN and VITS2 substitute for each other. F0 gains most with both (+0.327 from NoF0 to Full).
- **Top UTMOSv2 group:** NoVITS2 3.519, Full 3.516, Baseline-BigVGAN 3.495. The tested differences inside it are not significant.
- **Cost:**
  - CPU RTF at 1 thread: Baseline 0.0682, Baseline-BigVGAN 0.0790, NoVITS2 0.0795, Full 0.0880.
  - Inference-path size: Full is 9.4 % larger than Baseline.
- **NoF0 synthesizes about 40 % more audio** than the other cells. This comes from the stochastic part of its duration predictor at `noise_scale_w` 0.8 (section 8). The checkpoint-selection metric cannot see it (section 7).
- **Recommended system:** Full. NoVITS2 is a smaller and faster alternative with the same predicted naturalness within the test's resolution (section 6).
- **Historical pair, as recorded:** Full 1079 against Baseline 773 gives WER 8.64 % vs 9.17 % (p = 0.115) and UTMOSv2 3.807 vs 3.366 (p = 1.6 × 10⁻⁷⁹). It is not comparable with the ablation (section 9).
  - Timed again on 28/10 with the CPU protocol of section 5 (section 9): Full 0.0888 / 0.0579 and Baseline 0.0789 / 0.0514 at 1 / 2 threads, so Full is about 12.5 % slower. The recorded RTFs, 0.0568 vs 0.0559, have no recorded thread count.
- **Pitch is not controllable through the F0 conditioning** (section 10).
  - Scaling the decoder's F0 input by ±2 semitones, with z, durations and noise fixed, moves the output pitch by only 0.008–0.023 semitones (slope 0.004–0.011). This holds in all four F = 1 cells and in the historical Full model.
  - The decoder uses the conditioning, since the waveform changes, but the pitch comes from the latent.
- **Waveform F0 against the recordings** (section 11, DTW as the MCD):
  - every model has a log-F0 correlation of 0.32–0.37 and an F0 RMSE of 439–513 cents;
  - the F0 group does not bring the pitch closer to the recordings: alone it makes it worse (RMSE +57 cents), and in the other cells RMSE, correlation, VDE and GPE do not change significantly after Holm, except GPE −0.013 with BigVGAN;
  - the historical 1079 and 773 do not differ;
  - Full vs Baseline improves RMSE by 17 cents.

**Limits** (section 7):
- one seed per cell;
- two GPU types;
- all runs still improving at 200 epochs;
- Holm and the RTF protocol chosen after the results existed;
- a selection metric blind to durations;
- UTMOSv2 is predicted, not MOS;
- a non-significant difference is not equivalence.

## Sources

| What | Where (outside Git unless stated) |
|---|---|
| Evaluations, one folder per label | `<WSL_DEV_HOME>/outputs-pfn-cr/eval/<label>/`: `summary.json`, `protocol.json`, `*_per_sentence.csv` (WER, UTMOSv2), `mcd/mcd_gtmos.csv` (MCD, reference UTMOSv2), `reference.jsonl` |
| Final checkpoints and run records | `<WSL_DEV_HOME>/outputs-pfn-cr/final/<run_id>/` (read-only, `SHA256SUMS`) |
| Pairwise Wilcoxon tests of WER and UTMOSv2 | `eval/compare-<off>_<on>.json`, from `tools/camera_ready/evaluate_like_history.py compare` (commit `f773be2`) |
| Analysis tool | `tools/camera_ready/ablation8_analysis.py` (in Git, last changed in commit `f939d4b`), tests `tests/camera_ready/test_ablation8.py` |
| CPU RTF benchmark (T15) | `tools/camera_ready/benchmark_rtf.py` (commit `a67caf3`), tests `tests/camera_ready/test_benchmark_rtf.py` |
| Duration diagnostics | `tools/camera_ready/duration_diag.py` (commit `c8d3144`), tests `tests/camera_ready/test_duration_diag.py`; results in `tests/duration-diag-20261024/` |
| Historical pair, as recorded | `evaluation/eval_h_vs_baseline/` (in Git, sanitized copies) |
| CPU RTF benchmark of the historical pair (28/10) | `tools/camera_ready/benchmark_rtf.py historical`, tests `tests/camera_ready/test_benchmark_rtf.py`; results `eval/historical-pair-rtf.{json,csv}`, `historical-pair-rtf-runs.jsonl` |
| Pitch intervention (T14) | `tools/camera_ready/pitch_shift.py` (commit `524708f`), tests `tests/camera_ready/test_pitch_shift.py`; results in `tests/pitch-shift-20261025/` |
| Waveform F0 (T13) | `tools/camera_ready/waveform_f0.py` (commit `ac6e68a`), tests `tests/camera_ready/test_waveform_f0.py`; results in `tests/waveform-f0-20261025/` |
| Derived files | `eval/ablation8-check.json`, `ablation8-table.{csv,md}`, `ablation8-pairs.{csv,json}`, `ablation8-effects.{csv,json}`, `ablation8-cost.{csv,json}`, `ablation8-rtf.{csv,json}`, `ablation8-rtf-runs.jsonl` |
| Hashes of every source and derived file | `manifests/camera-ready/ablation8-derived.json` (in Git) |
| Import of the four MardeusPC runs | `runs-Daziel.md`, "Import from MardeusPC" |

**Labels and training machines:**
- MardeusPC, RTX 5070: `PFN-Full-200ep-best` (run `PFN-Full-275ep-s1234`, ended at epoch 200), `PFN-NoF0`, `PFN-Baseline-VITS2`, `PFN-Baseline-F0`.
- The PC, RTX 4070 Ti: `PFN-NoBigVGAN`, `PFN-NoVITS2`, `PFN-Baseline-BigVGAN`, `PFN-Baseline`.

Each run was evaluated on the machine that trained it.

## 1. Consistency of the eight evaluations

`ablation8_analysis.py check`: **122 of 122 checks passed** (`eval/ablation8-check.json`). For every label:
- **Sizes and split:**
  - `num_sentences` 500;
  - MCD `n` 500 and `n_skipped` 0;
  - split `test`.
- **Checkpoint:**
  - it is the final best checkpoint, and its `checkpoint_sha256` equals the line in `final/<run_id>/SHA256SUMS`;
  - `final.json` has 200 epochs, 156,400 G updates and training tree `1ee13cb4…`;
  - the checkpoint flags equal the cell;
  - strict group semantics; seeds 1234 / 1234 / 2026.
- **Same sentences:**
  - the same 500 sentences in the same order, in the per-sentence CSV and in the MCD CSV;
  - `reference.jsonl` is byte-identical;
  - its audio is the test split of `split-manifest.json`, in order (first and last ID `LJ040-0106`, `LJ021-0050`).
- **Reference UTMOSv2:** mean 3.8308. The per-sentence values are identical in all eight files, to the four decimals the CSV keeps. The scoring pipeline gave the same scores on both machines for the same audio.
- **Protocol:** identical settings in `protocol.json`: tool, seeds, synthesis scales, sample rate, Whisper `base.en`, devices, threads, MCD and F0 flags, package versions including UTMOSv2 1.3.1.dev0.
- **Evaluation tool:** it has one version in Git (`f773be2`, SHA-256 `314c79cc…`), so all eight evaluations used the same code.
- **Summary means** equal the means of the per-sentence values.

**Config differences between cube neighbours.** Read from each run's `run_manifest.json`, the config that actually ran. Every edge differs only in:
- the group flags (`model.use_*`, `groups.*`);
- `run_id`, `tier` and `device_note`;
- for the four edges that end in Full, also `protocol.max_g_updates` and `protocol.budget_label`.

No model, optimiser, data or seed setting differs.

**Does anything in the Full config depend on its 275-epoch plan?** The source was checked; nothing does:
- The LR schedule is `ExponentialLR(gamma = 0.999875)`, stepped once per epoch. It does not depend on the total.
- The MAS noise decays per G update (`4e-6`).
- `warmup_epochs` is declared but not used by the training code.
- `trainer.max_epochs` is 1110 in all eight configs.
- `max_g_updates` (215,050 for Full, 156,400 for the others) is read only by `GUpdateBudget`, which stops training. The Full run stopped at epoch 200 through `--stop-after-epochs 200`.
- The first 200 epochs of the Full run therefore follow the same schedule as a 200-epoch run.

**How the Full run differs: interruptions.**
- `PFN-Full` had 6 attempts:
  - four `CUDA error: unknown error` interruptions, two of them resumed by the watchdog;
  - one stop for the budget change.
- `PFN-NoF0` had 4 attempts, with 3 watchdog resumes. The other six runs had one attempt each.
- Resume was checked to reproduce an uninterrupted run on the CPU (`test_train_cpu.py`, PLAN O1) and in the GPU pilot (`eta.md`). On the GPU it is not bit-exact, because training is not deterministic there.

## 2. The eight models

`eval/ablation8-table.md`, from each `summary.json` and `protocol.json`:

| Label | B | V | F | Best epoch | WER mean | WER median | UTMOSv2 mean | UTMOSv2 median | MCD mean | F0 RMSE | F0 corr | Recorded RTF | Audio (s) | Training GPU |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `PFN-Full-200ep-best` | 1 | 1 | 1 | 188 | 0.1135 | 0.0741 | 3.5156 | 3.5204 | 25.638 | 0.5399 | 0.1861 | 0.0682 | 3009.9 | NVIDIA GeForce RTX 5070 |
| `PFN-NoF0-200ep-best` | 1 | 1 | 0 | 192 | 0.1302 | 0.0870 | 3.1887 | 3.2026 | 25.576 | n/a | n/a | 0.0926 | 4219.7 | NVIDIA GeForce RTX 5070 |
| `PFN-NoVITS2-200ep-best` | 1 | 0 | 1 | 188 | 0.1101 | 0.0588 | 3.5185 | 3.5452 | 25.760 | 0.5827 | 0.1992 | 0.0722 | 3036.8 | NVIDIA GeForce RTX 4070 Ti |
| `PFN-Baseline-BigVGAN-200ep-best` | 1 | 0 | 0 | 199 | 0.1009 | 0.0667 | 3.4945 | 3.5125 | 25.744 | n/a | n/a | 0.0737 | 3000.0 | NVIDIA GeForce RTX 4070 Ti |
| `PFN-NoBigVGAN-200ep-best` | 0 | 1 | 1 | 199 | 0.1214 | 0.0769 | 3.2442 | 3.2703 | 25.923 | 0.4813 | 0.1784 | 0.0709 | 2951.6 | NVIDIA GeForce RTX 4070 Ti |
| `PFN-Baseline-VITS2-200ep-best` | 0 | 1 | 0 | 188 | 0.1105 | 0.0714 | 3.3723 | 3.3817 | 25.623 | n/a | n/a | 0.0775 | 3057.6 | NVIDIA GeForce RTX 5070 |
| `PFN-Baseline-F0-200ep-best` | 0 | 0 | 1 | 195 | 0.1349 | 0.0714 | 3.0599 | 3.0956 | 26.728 | 0.6200 | 0.1923 | 0.0529 | 3070.6 | NVIDIA GeForce RTX 5070 |
| `PFN-Baseline-200ep-best` | 0 | 0 | 0 | 194 | 0.1082 | 0.0667 | 2.9407 | 2.9585 | 25.573 | n/a | n/a | 0.0593 | 3076.9 | NVIDIA GeForce RTX 4070 Ti |

Reference test audio (the 500 raw WAVs of the split, header durations): 3256.2 s.

- WER is a fraction here (0.1135 = 11.35 %).
- F0 RMSE and correlation are the F0-predictor metric (phoneme-level log-F0 against the cached reference). They exist only for F = 1.
- **The recorded RTF is not a reportable RTF:** it was measured inline, on two machines, with other work running. The idle-machine CPU benchmark is in section 5.
- **Audio** is the total duration synthesized for the 500 sentences (sum of `audio_duration_sec`).
  - Seven cells produce 2,952–3,077 s.
  - **`PFN-NoF0` produces 4,219.7 s, about 40 % more for the same text.** This is longer than the reference recordings (3,256.2 s, which include their leading and trailing silence).
  - The CPU benchmark reproduced the same lengths exactly (section 5). This is an observation; its cause, for example the duration predictor or pauses, was not examined. It may bear on NoF0's WER and UTMOSv2, which are the highest WER and the lowest UTMOSv2 of the four cells with BigVGAN.

## 3. Pairwise tests: the 12 cube edges and Full vs Baseline

`eval/ablation8-pairs.{csv,json}`. Method:
- **Difference:** per sentence, on − off. For Full vs Baseline it is Full − Baseline.
- **CI:** percentile 95 % CI of the mean difference, from a paired bootstrap over sentences (10,000 resamples, seed 2026).
- **Test:** two-sided Wilcoxon signed-rank with SciPy defaults, so zero differences are dropped, as in `compare`.
  - For WER and UTMOSv2 the p-values equal those of `compare-*.json` (checked for all 26).
  - MCD is not in `compare`, so the tool tests it with the same call.
- **Holm correction** within each metric across the 13 pairs. It was specified in the author's task of 24/10, **after all eight evaluations existed**, so it was not fixed in advance.
- **Median WER difference:** it is 0.00 for every pair, because most sentences have the same WER in both models.

#### WER, percentage points

| Switched | Off → on | Mean off | Mean on | Mean diff (on − off) | 95 % CI | Median diff | p (raw) | p (Holm) |
|---|---|---:|---:|---:|---|---:|---:|---:|
| B | Baseline → Baseline-BigVGAN | 10.82 | 10.09 | -0.73 | [-1.59, +0.13] | +0.00 | 0.136 | 0.816 |
| B | Baseline-F0 → NoVITS2 | 13.49 | 11.01 | -2.48 | [-4.39, -1.00] | +0.00 | 0.003 | 0.037 |
| B | Baseline-VITS2 → NoF0 | 11.05 | 13.02 | +1.97 | [+0.65, +3.53] | +0.00 | 0.011 | 0.099 |
| B | NoBigVGAN → Full | 12.14 | 11.35 | -0.79 | [-1.77, +0.22] | +0.00 | 0.142 | 0.816 |
| V | Baseline → Baseline-VITS2 | 10.82 | 11.05 | +0.23 | [-0.68, +1.15] | +0.00 | 0.543 | 1.000 |
| V | Baseline-F0 → NoBigVGAN | 13.49 | 12.14 | -1.35 | [-3.48, +0.38] | +0.00 | 0.989 | 1.000 |
| V | Baseline-BigVGAN → NoF0 | 10.09 | 13.02 | +2.93 | [+1.67, +4.45] | +0.00 | 6.3e-06 | 8.2e-05 |
| V | NoVITS2 → Full | 11.01 | 11.35 | +0.34 | [-1.14, +1.56] | +0.00 | 0.059 | 0.415 |
| F | Baseline → Baseline-F0 | 10.82 | 13.49 | +2.67 | [+0.90, +4.86] | +0.00 | 0.008 | 0.085 |
| F | Baseline-VITS2 → NoBigVGAN | 11.05 | 12.14 | +1.09 | [+0.15, +2.01] | +0.00 | 0.034 | 0.275 |
| F | Baseline-BigVGAN → NoVITS2 | 10.09 | 11.01 | +0.93 | [-0.20, +2.36] | +0.00 | 0.221 | 0.883 |
| F | NoF0 → Full | 13.02 | 11.35 | -1.67 | [-3.22, -0.30] | +0.00 | 0.008 | 0.085 |
| BVF | Baseline → Full | 10.82 | 11.35 | +0.54 | [-0.42, +1.52] | +0.00 | 0.325 | 0.975 |

#### UTMOSv2 (predicted)

| Switched | Off → on | Mean off | Mean on | Mean diff (on − off) | 95 % CI | Median diff | p (raw) | p (Holm) |
|---|---|---:|---:|---:|---|---:|---:|---:|
| B | Baseline → Baseline-BigVGAN | 2.941 | 3.495 | +0.554 | [+0.523, +0.584] | +0.566 | 1.5e-79 | 1.8e-78 |
| B | Baseline-F0 → NoVITS2 | 3.060 | 3.519 | +0.459 | [+0.424, +0.494] | +0.435 | 4.5e-69 | 4.5e-68 |
| B | Baseline-VITS2 → NoF0 | 3.372 | 3.189 | -0.184 | [-0.216, -0.151] | -0.189 | 9.6e-25 | 5.8e-24 |
| B | NoBigVGAN → Full | 3.244 | 3.516 | +0.271 | [+0.243, +0.299] | +0.263 | 7.7e-52 | 6.2e-51 |
| V | Baseline → Baseline-VITS2 | 2.941 | 3.372 | +0.432 | [+0.401, +0.463] | +0.440 | 2.6e-70 | 2.9e-69 |
| V | Baseline-F0 → NoBigVGAN | 3.060 | 3.244 | +0.184 | [+0.149, +0.219] | +0.172 | 3.1e-21 | 1.5e-20 |
| V | Baseline-BigVGAN → NoF0 | 3.495 | 3.189 | -0.306 | [-0.337, -0.274] | -0.313 | 5.3e-50 | 3.7e-49 |
| V | NoVITS2 → Full | 3.519 | 3.516 | -0.003 | [-0.029, +0.023] | -0.006 | 0.595 | 0.595 |
| F | Baseline → Baseline-F0 | 2.941 | 3.060 | +0.119 | [+0.083, +0.155] | +0.133 | 8.8e-11 | 2.6e-10 |
| F | Baseline-VITS2 → NoBigVGAN | 3.372 | 3.244 | -0.128 | [-0.159, -0.097] | -0.132 | 1.9e-14 | 7.7e-14 |
| F | Baseline-BigVGAN → NoVITS2 | 3.495 | 3.519 | +0.024 | [-0.005, +0.053] | +0.026 | 0.105 | 0.209 |
| F | NoF0 → Full | 3.189 | 3.516 | +0.327 | [+0.296, +0.356] | +0.322 | 1.9e-58 | 1.7e-57 |
| BVF | Baseline → Full | 2.941 | 3.516 | +0.575 | [+0.546, +0.604] | +0.560 | 5.1e-81 | 6.7e-80 |

#### MCD

| Switched | Off → on | Mean off | Mean on | Mean diff (on − off) | 95 % CI | Median diff | p (raw) | p (Holm) |
|---|---|---:|---:|---:|---|---:|---:|---:|
| B | Baseline → Baseline-BigVGAN | 25.573 | 25.744 | +0.171 | [+0.089, +0.253] | +0.171 | 3.2e-06 | 2.6e-05 |
| B | Baseline-F0 → NoVITS2 | 26.728 | 25.760 | -0.968 | [-1.083, -0.852] | -0.902 | 2.0e-42 | 2.4e-41 |
| B | Baseline-VITS2 → NoF0 | 25.623 | 25.576 | -0.047 | [-0.138, +0.043] | -0.067 | 0.222 | 0.444 |
| B | NoBigVGAN → Full | 25.923 | 25.638 | -0.285 | [-0.377, -0.197] | -0.209 | 1.4e-08 | 1.3e-07 |
| V | Baseline → Baseline-VITS2 | 25.573 | 25.623 | +0.050 | [-0.027, +0.128] | +0.038 | 0.133 | 0.398 |
| V | Baseline-F0 → NoBigVGAN | 26.728 | 25.923 | -0.805 | [-0.917, -0.692] | -0.770 | 2.6e-37 | 2.8e-36 |
| V | Baseline-BigVGAN → NoF0 | 25.744 | 25.576 | -0.168 | [-0.267, -0.068] | -0.143 | 0.001 | 0.009 |
| V | NoVITS2 → Full | 25.760 | 25.638 | -0.122 | [-0.202, -0.043] | -0.134 | 0.002 | 0.013 |
| F | Baseline → Baseline-F0 | 25.573 | 26.728 | +1.155 | [+1.051, +1.264] | +1.119 | 1.5e-58 | 2.0e-57 |
| F | Baseline-VITS2 → NoBigVGAN | 25.623 | 25.923 | +0.301 | [+0.204, +0.399] | +0.252 | 3.9e-09 | 3.9e-08 |
| F | Baseline-BigVGAN → NoVITS2 | 25.744 | 25.760 | +0.016 | [-0.073, +0.106] | +0.077 | 0.403 | 0.444 |
| F | NoF0 → Full | 25.576 | 25.638 | +0.063 | [-0.026, +0.148] | +0.149 | 0.036 | 0.144 |
| BVF | Baseline → Full | 25.573 | 25.638 | +0.065 | [-0.020, +0.148] | +0.132 | 0.010 | 0.052 |

## 4. How much each component contributes

`eval/ablation8-effects.{csv,json}`.

**Factorial effects** use effect coding (−1 = off, +1 = on). Per sentence, effect(term) = ¼ Σ over the 8 cells of sign(term, cell) × y(cell); the reported value is the mean over sentences.
- A **main effect** is mean(on) − mean(off), averaged over the other two factors.
- A **two-way interaction** X×Y is half the difference between the effect of X with Y on and with Y off.
- **CIs** come from the paired bootstrap over sentences, as above.

#### Factorial effects

| Term | WER, pp [95 % CI] | UTMOSv2 [95 % CI] | MCD [95 % CI] |
|---|---|---|---|
| B | -0.50 [-1.12, +0.12] | +0.275 [+0.260, +0.291] | -0.282 [-0.334, -0.231] |
| V | +0.54 [-0.09, +1.16] | +0.077 [+0.062, +0.092] | -0.261 [-0.310, -0.213] |
| F | +0.76 [+0.15, +1.42] | +0.086 [+0.070, +0.102] | +0.384 [+0.336, +0.431] |
| B×V | +1.10 [+0.48, +1.82] | -0.231 [-0.247, -0.216] | +0.116 [+0.073, +0.160] |
| B×F | -1.13 [-1.87, -0.48] | +0.090 [+0.074, +0.106] | -0.344 [-0.392, -0.296] |
| V×F | -1.04 [-2.10, -0.22] | +0.014 [-0.002, +0.029] | -0.202 [-0.248, -0.157] |
| B×V×F | -0.25 [-0.85, +0.34] | +0.138 [+0.122, +0.154] | +0.225 [+0.178, +0.273] |

**Shapley attribution.** This splits the Full − Baseline gap among B, V and F.
- It is exact over the 8 cells: v(S) is the mean metric of the cell with the components in S on, and the three values sum to the gap.
- "%" is the share of the gap.
- **Where the gap's CI includes 0 (WER, MCD), the percentages are not meaningful:** they divide by a gap that may be zero, so they are given only for completeness.

#### Shapley attribution of the Full − Baseline gap

| Metric | Gap Full − Baseline [95 % CI] | B | V | F |
|---|---|---|---|---|
| WER, percentage points | +0.54 [-0.42, +1.52] | -0.59 [-1.16, -0.02], -109 % | +0.46 [-0.20, +1.08], 84 % | +0.67 [+0.01, +1.41], 125 % |
| UTMOSv2 (predicted) | +0.575 [+0.546, +0.604] | +0.321 [+0.305, +0.337], 56 % | +0.123 [+0.108, +0.138], 21 % | +0.131 [+0.114, +0.149], 23 % |
| MCD | +0.065 [-0.020, +0.148] | -0.207 [-0.259, -0.157], -318 % | -0.186 [-0.234, -0.139], -286 % | +0.459 [+0.409, +0.509], 703 % |

**F0-predictor metrics** (F = 1 only). These are summary values only: no per-sentence F0 values exist, so there is no CI or test.

| | V = 1 | V = 0 |
|---|---|---|
| B = 1 | RMSE 0.540, corr 0.186 (Full) | RMSE 0.583, corr 0.199 (NoVITS2) |
| B = 0 | RMSE 0.481, corr 0.178 (NoBigVGAN) | RMSE 0.620, corr 0.192 (Baseline-F0) |

### The training machine and the effects

The two GPU types are not spread evenly over the cube:
- the RTX 5070 trained 111, 110, 010 and 001;
- the RTX 4070 Ti trained 101, 011, 100 and 000.

Coded as ±1, the machine is orthogonal to the B, F and B×F contrasts. It equals ½(V + B×V − V×F + B×V×F).

What this means:
- **Effects that a machine offset does not bias:** if training on one GPU type shifted a metric by a constant δ, the B, F and B×F effects would not be biased.
- **Effects that it does bias:** V, B×V and B×V×F would each be biased by +δ/2, and V×F by −δ/2.
- **Edges that cross GPU types:** every V edge, two of the four B edges, two of the four F edges, and Full vs Baseline.
- **Shapley values:** a machine offset would add δ/6 to the B and F values and 2δ/3 to the V value.

δ cannot be estimated from these runs: one seed per cell, and the machine is aliased with interactions. The evaluation side agrees across the machines (identical per-sentence reference UTMOSv2). The training side is the open part.

### B: BigVGAN

- **UTMOSv2:** B has the largest effect: main effect +0.275 [+0.260, +0.291], and 56 % of the Full − Baseline gap (+0.321 of +0.575).
- **Where it helps:** three of its four edges raise UTMOSv2:
  - +0.554 on Baseline;
  - +0.459 on Baseline-F0;
  - +0.271 on NoBigVGAN → Full.
  - All have p (Holm) ≤ 6.2 × 10⁻⁵¹.
- **Where it hurts:** on Baseline-VITS2 → NoF0 it lowers UTMOSv2 by −0.184 [−0.216, −0.151], p (Holm) 5.8 × 10⁻²⁴. This is the B×V interaction, −0.231 [−0.247, −0.216].
- **MCD:** main effect −0.282 [−0.334, −0.231]. On Baseline alone, however, B raises MCD by +0.171 [+0.089, +0.253], p (Holm) 2.6 × 10⁻⁵.
- **WER:** main effect −0.50 pp [−1.12, +0.12]. Only one edge is significant after Holm: Baseline-F0 → NoVITS2, −2.48 pp, p (Holm) 0.037.
- **Machine:** B's main effect is balanced over the two GPU types.
- **Size:** it adds 3,648 elements to the inference path (Baseline → Baseline-BigVGAN; Snake activations in the decoder under strict gating). Most of its 286,590 extra elements in the whole model are the multi-resolution discriminator, which only training uses.

### V: VITS2

- **UTMOSv2:** main effect +0.077 [+0.062, +0.092], but **it depends strongly on B**:
  - without B it helps: +0.432 on Baseline and +0.184 on Baseline-F0;
  - with B it does not: −0.306 [−0.337, −0.274] on Baseline-BigVGAN, p (Holm) 3.7 × 10⁻⁴⁹;
  - with B and F: −0.003 [−0.029, +0.023], p (Holm) 0.60.
  - So in Full, V adds no measurable UTMOSv2. This is not evidence that it adds nothing.
- **WER:** main effect +0.54 pp [−0.09, +1.16]. On Baseline-BigVGAN, V raises WER by +2.93 pp [+1.67, +4.45], p (Holm) 8.2 × 10⁻⁵.
- **MCD:** main effect −0.261 [−0.310, −0.213]; NoVITS2 → Full is −0.122 [−0.202, −0.043], p (Holm) 0.013.
- **Shapley:** 21 % of the UTMOSv2 gap.
- **Reliability:** **every V comparison crosses GPU types, and V is the effect most exposed to the machine** (above). Its estimates are the least reliable of the three.
- **Size:** about 1.19 M elements in the inference path (+7.3 % over Baseline).

### F: F0

- **UTMOSv2:** main effect +0.086 [+0.070, +0.102], and 23 % of the gap. Its sign depends on the other groups:
  - +0.119 on Baseline;
  - −0.128 on Baseline-VITS2;
  - +0.024 [−0.005, +0.053] on Baseline-BigVGAN, p (Holm) 0.21;
  - **+0.327 [+0.296, +0.356] on NoF0 → Full**, p (Holm) 1.7 × 10⁻⁵⁷.
  - F gains most when B and V are both present (B×F +0.090, B×V×F +0.138).
- **MCD:** F raises it: main effect +0.384 [+0.336, +0.431], and +1.155 on Baseline, p (Holm) 2.0 × 10⁻⁵⁷. With B and V present the change is +0.063 [−0.026, +0.148], p (Holm) 0.14.
- **WER:**
  - main effect +0.76 pp [+0.15, +1.42];
  - +2.67 pp alone, p (Holm) 0.085;
  - −1.67 pp [−3.22, −0.30] for NoF0 → Full, p (Holm) 0.085;
  - neither edge is significant after Holm.
- **F0-predictor RMSE:** it is lowest with V on (0.481 without B, 0.540 with B) and highest in Baseline-F0 (0.620). The correlation is low in all four cells (0.18–0.20). No test is possible on these values.
- **Machine:** F's main effect is balanced over the two GPU types.
- **Size:** about 0.35 M elements (the F0 predictor, +2.1 %).

### Interactions

- **UTMOSv2:** all interactions except V×F (+0.014 [−0.002, +0.029]) have CIs that exclude 0.
  - The largest is B×V, −0.231: B and V substitute for each other.
  - B×F (+0.090) and B×V×F (+0.138) are positive: F helps more when B is on, and most when B and V are both on.
- **MCD:** all seven CIs exclude 0.
- **WER:** B×V, B×F and V×F have CIs that exclude 0, all about ±1 pp.
- **Caveat:** with one seed per cell, each interaction is estimated from single training runs. **Training variance is not in these CIs.** B×V and B×V×F are also partly aliased with the machine.

## 5. Cost against quality

`eval/ablation8-cost.{csv,json}`.
- **Sizes** are stored `state_dict` tensor elements, loaded with `weights_only=True`; weight norm stores both g and v.
- **The inference path** is the generator without the posterior encoder `enc_q`, which only training uses.

#### Size

| Label | B | V | F | Inference path (elements) | Generator | Whole model | Checkpoint (bytes) |
|---|:-:|:-:|:-:|---:|---:|---:|---:|
| Full | 1 | 1 | 1 | 17,970,805 | 25,208,822 | 72,794,737 | 874,731,173 |
| NoF0 | 1 | 1 | 0 | 17,624,436 | 24,862,453 | 72,448,368 | 870,560,277 |
| NoVITS2 | 1 | 0 | 1 | 16,775,797 | 24,013,814 | 71,043,888 | 853,606,869 |
| Baseline-BigVGAN | 1 | 0 | 0 | 16,429,428 | 23,667,445 | 70,697,519 | 849,436,037 |
| NoBigVGAN | 0 | 1 | 1 | 17,967,157 | 25,205,174 | 72,508,147 | 871,187,440 |
| Baseline-VITS2 | 0 | 1 | 0 | 17,620,788 | 24,858,805 | 72,161,778 | 867,016,544 |
| Baseline-F0 | 0 | 0 | 1 | 16,772,149 | 24,010,166 | 70,757,298 | 850,063,072 |
| Baseline | 0 | 0 | 0 | 16,425,780 | 23,663,797 | 70,410,929 | 845,892,176 |

**Training cost**, per machine; RTX 5070 and RTX 4070 Ti times are **not** comparable with each other. Within a machine they are indicative only:
- the two PC runs on each pair of GPUs shared the CPU;
- on the RTX 5070, evaluations and other runs sometimes ran beside a training run;
- that GPU also drives the display.

Columns:
- **Active time:** a dead attempt ends at its last checkpoint event.
- **Median epoch:** the median over consecutive epochs of the same attempt.

#### Training cost, per machine

| Label | GPU | Attempts | Watchdog resumes | Other resumes | Wall clock (h) | Active (GPU-h) | G updates / s | Median epoch (min) | Recorded RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | RTX 4070 Ti | 1 | 0 | 0 | 17.75 | 17.75 | 2.447 | 5.30 | 0.0593 |
| Baseline-BigVGAN | RTX 4070 Ti | 1 | 0 | 0 | 20.56 | 20.56 | 2.113 | 6.23 | 0.0737 |
| NoBigVGAN | RTX 4070 Ti | 1 | 0 | 0 | 22.65 | 22.65 | 1.918 | 6.82 | 0.0709 |
| NoVITS2 | RTX 4070 Ti | 1 | 0 | 0 | 21.71 | 21.71 | 2.001 | 6.50 | 0.0722 |
| Baseline-F0 | RTX 5070 | 1 | 0 | 0 | 21.88 | 21.88 | 1.985 | 6.42 | 0.0529 |
| Baseline-VITS2 | RTX 5070 | 1 | 0 | 0 | 25.10 | 25.10 | 1.731 | 7.42 | 0.0775 |
| Full | RTX 5070 | 6 | 2 | 3 | 31.65 | 28.75 | 1.511 | 8.47 | 0.0682 |
| NoF0 | RTX 5070 | 4 | 3 | 0 | 28.93 | 28.45 | 1.527 | 8.42 | 0.0926 |

**Inference cost, recorded inline.** The recorded `mean_rtf` was measured inline during evaluation, on two different machines that were not idle. It is **indicative only and not reportable as RTF.**

**Inference cost, CPU benchmark (T15).** All eight best checkpoints were measured on the PC, idle, on 2026-10-24 20:31–21:12 +07, with `tools/camera_ready/benchmark_rtf.py` (commit `a67caf3`; log `logs/rtf-ablation8-20261024.log`). Files: `eval/ablation8-rtf.{json,csv}`, `ablation8-rtf-runs.jsonl`.

Protocol (agreed with the author on 24/10):
- **Threads:** CPU only, 1 and 2 threads, one interop thread.
  - The process was pinned with `taskset` to vCPU 0 (1 thread) and to vCPUs 0 and 2 (2 threads).
  - Under WSL, Hyper-V schedules the vCPUs on host cores, so P-core or E-core placement was not controlled.
- **Synthesis:** the same path as the evaluation:
  - `weights_only` strict load, decoder weight norm removed;
  - `eval_harness.synthesize` at 0.667 / 1.0 / 0.8;
  - the evaluation's per-utterance seeds.
- **Sentences:** 10 validation utterances as warm-up, then the first 100 test utterances (610–859 s of audio per model).
- **Rounds:** three, with the model order rotated. Each (round, model, threads) cell ran in its own process.
- **Reported value:** the corpus RTF, i.e. Σ time / Σ audio duration, as the median over the three rounds, with the range in brackets.
  - Model-only time is the harness's own window (the model forward).
  - End-to-end adds espeak phonemization and tensor building.

| Label | B V F | 1 thread: model RTF, median [min, max] | 1 thread: end-to-end | 2 threads: model RTF, median [min, max] | 2 threads: end-to-end | Audio lengths equal in all 6 runs |
|---|:-:|---|---:|---|---:|:-:|
| Full | 1 1 1 | 0.0880 [0.0861, 0.0881] | 0.0881 | 0.0567 [0.0560, 0.0584] | 0.0568 | yes |
| NoF0 | 1 1 0 | 0.0917 [0.0914, 0.0924] | 0.0918 | 0.0600 [0.0599, 0.0611] | 0.0601 | yes |
| NoVITS2 | 1 0 1 | 0.0795 [0.0791, 0.0801] | 0.0796 | 0.0521 [0.0520, 0.0526] | 0.0522 | yes |
| Baseline-BigVGAN | 1 0 0 | 0.0790 [0.0785, 0.0796] | 0.0791 | 0.0517 [0.0514, 0.0520] | 0.0518 | yes |
| NoBigVGAN | 0 1 1 | 0.0766 [0.0761, 0.0769] | 0.0767 | 0.0495 [0.0495, 0.0498] | 0.0496 | yes |
| Baseline-VITS2 | 0 1 0 | 0.0758 [0.0757, 0.0759] | 0.0759 | 0.0495 [0.0494, 0.0501] | 0.0496 | yes |
| Baseline-F0 | 0 0 1 | 0.0690 [0.0688, 0.0692] | 0.0691 | 0.0449 [0.0446, 0.0454] | 0.0450 | yes |
| Baseline | 0 0 0 | 0.0682 [0.0674, 0.0683] | 0.0683 | 0.0442 [0.0438, 0.0443] | 0.0443 | yes |

**Checks on the benchmark:**
- all 48 runs had 100 utterances, the requested thread count and affinity;
- at the start, no other tmux session ran and the load average was 0.51. The load before later runs, up to 1.75, came from the benchmark's own previous run;
- the round-to-round range is at most 4.2 % of the median (Full, 2 threads), and under 2.5 % for every other cell;
- the synthesized lengths were identical across the 6 runs of each model and equal to the evaluation's, so the benchmark timed the same audio that was scored.

**Reading:**
- **Relative to Baseline, at 1 thread:**
  - B adds about 16 % to the model RTF (0.0790 vs 0.0682);
  - V adds about 11 % (0.0758);
  - F adds about 1 % (0.0690);
  - Full is about 29 % slower (0.0880).
- **NoVITS2** (0.0795) is about 10 % faster than Full.
- **Two threads** give a 1.53–1.55× speed-up for every model.
- **End-to-end ≈ model-only:** phonemization adds about 0.0001 RTF.
- **The RTF of NoF0 divides by its longer audio** (section 2). Per sentence, it spends more time than the RTF suggests.
- **What these values are:** CPU RTFs of one idle WSL machine at 1–2 threads. They are comparable among the eight models, not with other hardware or with RTFs measured differently.

**Pareto sets** of mean quality against cost:

- WER vs inference-path size: Baseline, Baseline-BigVGAN
- WER vs recorded RTF: Baseline, Baseline-BigVGAN, Baseline-F0
- WER vs CPU RTF (1 thread): Baseline, Baseline-BigVGAN
- UTMOSv2 vs inference-path size: Baseline, Baseline-BigVGAN, NoVITS2
- UTMOSv2 vs recorded RTF: Baseline-F0, Full, NoVITS2
- UTMOSv2 vs CPU RTF (1 thread): Baseline, Baseline-BigVGAN, Baseline-F0, Baseline-VITS2, NoVITS2
- MCD vs inference-path size: Baseline
- MCD vs recorded RTF: Baseline, Baseline-F0
- MCD vs CPU RTF (1 thread): Baseline

**Reading:**
- **The top UTMOSv2 group** has three cells: NoVITS2 (3.519), Full (3.516) and Baseline-BigVGAN (3.495).
  - Two of their differences were tested: NoVITS2 vs Full, −0.003 (Holm 0.60), and Baseline-BigVGAN → NoVITS2, +0.024 (Holm 0.21).
  - Neither is significant. **This does not show that they are equal.**
  - Full vs Baseline-BigVGAN is not a cube edge and was not tested.
- **Size:** Baseline-BigVGAN is the smallest of the three; it is B alone, 3,648 elements more than Baseline. NoVITS2 is 2.1 % larger than Baseline, Full 9.4 %.
- **Speed:** at 1 thread, Baseline-BigVGAN and NoVITS2 have nearly the same CPU RTF (0.0790, 0.0795); Full is slower (0.0880). Both of them are on the UTMOSv2 front against the CPU RTF. Full is not, because NoVITS2 has a higher mean UTMOSv2 at a lower RTF. That difference in UTMOSv2 (−0.003 for Full) is not significant, so the front is a ranking of point estimates.
- **Baseline** is on every MCD and WER front. Its UTMOSv2 (2.941) is the lowest of the eight.

## 6. Recommended configuration

**At this budget, the data do not single out one best configuration on all three metrics.**

If one has to be chosen from these runs, **Full (B + V + F) is the defensible choice for the system the paper reports.** The reasons:
- **UTMOSv2:** it is in the top group.
- **MCD:** it is lower than NoVITS2's, −0.122, p (Holm) 0.013.
- **F0 RMSE:** it is lower than NoVITS2's (0.540 vs 0.583, untested).
- **WER:** it is not significantly different from its neighbours or from Baseline.

The record should also say:
- **BigVGAN carries the largest share of the predicted-naturalness gain.** It does so with almost no inference-path cost.
- **VITS2 adds no measurable UTMOSv2 or WER once BigVGAN and F0 are present.** Its only significant gain in Full is MCD.
  - NoVITS2 is therefore a smaller and faster alternative: −6.7 % inference-path elements, and a CPU RTF of 0.0795 against 0.0880 at 1 thread. Its predicted naturalness is the same within the test's resolution.
  - V is also the estimate most exposed to the machine confound.
- **The F0 group helps predicted naturalness only together with BigVGAN and VITS2.** On its own it worsens MCD and WER.

These are statements about 200 epochs and one seed per cell. All eight runs were still improving at the end:
- `val_mel_l1_global` fell by 0.05–0.12 per 10 epochs over epochs 150–199;
- every best epoch lies in the last 25 epochs (188–199);
- NoBigVGAN and Baseline-BigVGAN have their best at epoch 199, the last epoch.

So the ranking may change with longer training.

## 7. Limits

- **One seed per configuration.** The sentence-level bootstrap and the Wilcoxon tests cover variation across sentences, not across training runs. Small differences, and all interactions, may be within training noise.
- **Two training machines,** a hardware confound. The machine is aliased with V, B×V, V×F and B×V×F (section 4). Every V comparison and the Full vs Baseline comparison cross GPU types.
- **The Full run had four CUDA interruptions** and a budget switch, and NoF0 had three interruptions. Resume is reproducible on the CPU, but GPU training is not bit-exact.
- **Undertraining.** Every run's best epoch is in the last 25 of 200, and the validation curves still fall. NoBigVGAN and Baseline-BigVGAN have their best at the last epoch (199). 200 epochs are 18.2 % of the historical epochs.
- **Multiple comparisons.** The Holm correction (13 pairs per metric) was specified after all evaluations existed. The other derived quantities have CIs but no correction: factorial effects, Shapley values, interactions.
- **Metrics:**
  - UTMOSv2 is a predicted score, not MOS.
  - MCD is the harness's DTW over 24 MFCCs, with large absolute values (≈ 25–27).
  - F0 metrics exist only as summaries.
  - A non-significant difference is not equivalence.
- **RTF:**
  - The CPU benchmark was run on one WSL machine without control of P-core or E-core placement, at 1–2 threads, on 100 of the 500 test sentences.
  - Its values compare the eight models with each other only.
  - The inline values of the evaluations are not reportable.
- **Audio length of NoF0:** its synthesized audio is about 40 % longer than that of the other cells (section 8).
- **The checkpoint-selection metric does not see text-to-speech quality.**
  - `val_mel_l1_global` is computed with the training forward. The decoder reconstructs a random segment from the posterior encoder's latent of the real spectrogram.
  - It does not pass through the text encoder, the flow or the duration predictor.
  - The best epoch of each run is therefore the best for decoder reconstruction, not for synthesis from text. A run whose duration predictor drifts, as NoF0's appears to, can still have a low validation value. NoF0's 21.10 is the second lowest of the eight, after Baseline-BigVGAN's 21.06.
  - This holds for all eight runs. It was fixed before any test result (decision (e) of `experiment-plan.md`).

## 8. The long audio of NoF0: what is known

Read-only analysis of the files above (per-sentence `audio_duration_sec`, the raw test WAVs, the source code), on 24/10.

**What the data show:**
- **NoF0 speaks more slowly, about uniformly.**
  - Per sentence, its duration over Full has median 1.40 (10th–90th percentile 1.25–1.57); 98.6 % of the sentences are more than 10 % longer.
  - The excess grows with length (excess ≈ 0.385 × Full duration + 0.10 s).
  - It does not correlate with punctuation (r = −0.06). So it is a slower rate throughout, not added pauses.
- **The other cells agree with each other.** The other seven cells produce 0.91–0.95 of the reference duration (median per sentence 0.91–0.95). NoF0 produces 1.30.
- **Within NoF0, the most stretched sentences score lower on UTMOSv2:**
  - r = −0.24 between stretch and UTMOSv2;
  - UTMOSv2 3.121 in the top quartile of stretch, against 3.292 in the bottom quartile.
  - The longer audio is a plausible contributor to NoF0's low UTMOSv2. A causal link was not tested.

**What the code shows:**
- In `SynthesizerTrn.infer`, the durations come from the stochastic duration predictor applied to the text-encoder output (`use_sdp: true` in both configs; `noise_scale_w` 0.8).
- `use_f0` does not change that path. It adds the F0 predictor, read from the same encoder output and trained with its own loss, and the F0 conditioning of the decoder.
- So the stretch is in the learned weights of the NoF0 run, not in a different inference path. Removing F0 can only act on durations indirectly: through training, the F0 loss also shapes the shared encoder output.

**What cannot be decided from these runs:**
- Only one of the four F = 0 cells is stretched. Baseline, Baseline-BigVGAN and Baseline-VITS2 have normal durations.
- So this is not a general effect of removing F0.
- With one seed per cell, a run-specific outcome cannot be separated from an interaction of F with B and V.

### Duration diagnostics (run on 24/10 after the author's OK)

**How they ran.**
- Tool: `tools/camera_ready/duration_diag.py` (commit `c8d3144`), 2026-10-24 22:10–22:19 +07, CPU.
- Results outside Git: `<WSL_DEV_HOME>/outputs-pfn-cr/tests/duration-diag-20261024/` (`summary.{json,csv}`, `durations.jsonl`). Log: `logs/duration-diag-20261024.log`.
- **Only part of the model ran:** the text encoder and the duration predictor, and for the teacher-forced part also the posterior encoder, the flow and MAS. The decoder did not run and no audio was written.
- **No selection:** the checkpoints are the fixed best and last (epoch 199) of each run.
- **Reference lengths** are the frames of the cached spectrograms.
- **Check:** with `noise_scale_w` 0.8 and the evaluation's seeds, the best checkpoint of every run reproduced the evaluation's `audio_duration_sec` **for 500 of 500 sentences**. The diagnostic therefore measures the same durations that were scored.

**1 and 3. Predicted length of the 500 test texts, as a fraction of the reference frames:**

| Label | B V F | Best epoch | Best, noise_w 0.8 | Best, noise_w 0 | Last (199), noise_w 0.8 | Last (199), noise_w 0 |
|---|:-:|---:|---:|---:|---:|---:|
| Full | 1 1 1 | 188 | 0.927 | 0.800 | 0.923 | 0.813 |
| NoF0 | 1 1 0 | 192 | 1.300 | 0.879 | 1.221 | 0.751 |
| NoVITS2 | 1 0 1 | 188 | 0.935 | 0.824 | 0.925 | 0.825 |
| Baseline-BigVGAN | 1 0 0 | 199 | 0.924 | 0.817 | 0.924 | 0.817 |
| NoBigVGAN | 0 1 1 | 199 | 0.909 | 0.776 | 0.909 | 0.776 |
| Baseline-VITS2 | 0 1 0 | 188 | 0.942 | 0.825 | 0.980 | 0.869 |
| Baseline-F0 | 0 0 1 | 195 | 0.946 | 0.855 | 0.974 | 0.879 |
| Baseline | 0 0 0 | 194 | 0.948 | 0.843 | 0.955 | 0.848 |

**2 and 3. Validation (100 utterances, best checkpoint): predicted durations against the MAS durations of the same model:**

| Label | B V F | Predicted / MAS, noise_w 0.8 | Predicted / MAS, noise_w 0 | Mean abs log ratio per phoneme, noise_w 0.8 | Same, noise_w 0 |
|---|:-:|---:|---:|---:|---:|
| Full | 1 1 1 | 0.945 | 0.815 | 0.603 | 0.490 |
| NoF0 | 1 1 0 | 1.327 | 0.893 | 0.719 | 0.515 |
| NoVITS2 | 1 0 1 | 0.943 | 0.833 | 0.584 | 0.480 |
| Baseline-BigVGAN | 1 0 0 | 0.941 | 0.831 | 0.576 | 0.478 |
| NoBigVGAN | 0 1 1 | 0.918 | 0.788 | 0.608 | 0.496 |
| Baseline-VITS2 | 0 1 0 | 0.950 | 0.833 | 0.599 | 0.492 |
| Baseline-F0 | 0 0 1 | 0.955 | 0.866 | 0.603 | 0.486 |
| Baseline | 0 0 0 | 0.955 | 0.859 | 0.595 | 0.490 |

**What they show:**
- **The stretch comes from the stochastic part of NoF0's duration predictor.**
  - Without its noise (`noise_scale_w` 0), NoF0's predicted length is 0.879 of the reference. That is at the upper end of the other cells (0.776–0.855) but close to them.
  - With the evaluation's `noise_scale_w` 0.8 it is 1.300.
  - So the noise lengthens NoF0 by a factor of 1.48. For the other cells it is 1.11–1.17.
- **NoF0's predictor maps the same noise to much longer durations.** On validation, against its own MAS alignment:
  - it predicts 1.33 times the aligned length, against 0.92–0.96 for the others;
  - its per-phoneme error is the largest (0.719, against 0.576–0.608).
  - Without noise, it is 0.893 times the aligned length, and its per-phoneme error is close to the others (0.515, against 0.478–0.496).
- **It is not peculiar to the best epoch.** The last checkpoint (epoch 199) of NoF0 is also stretched (1.221 with noise), while its noise-free length is short (0.751).
- **The other runs barely change** between the best and the last checkpoint (at most 0.044, Baseline-VITS2 without noise).
- **What is left unexplained:** why only this run's stochastic duration predictor learned such a wide or skewed distribution. With one seed per cell it cannot be told whether this is run-specific or an interaction of F with B and V.
- **All cells speak faster than the recordings without noise** (0.78–0.88). The reference frames include the recordings' leading and trailing silence.
- **What the evaluation measured:** it used `noise_scale_w` 0.8, the historical default of the locked protocol. NoF0's scores are valid for that protocol. Part of NoF0's low UTMOSv2 and high WER plausibly comes from this sampling behaviour; a causal link was not tested.
- **No re-evaluation** at another `noise_scale_w` was run. It would be outside the locked protocol.

## 9. The historical pair, as recorded

**This section copies the historical evaluation of the paper's main pair; it is not a new evaluation.** The pair was not re-evaluated under the locked protocol on this machine, and no record of such a re-evaluation exists in this repository.

**Source:** `evaluation/eval_h_vs_baseline/`, tracked as sanitized copies of contemporaneous records:
- `summary.sanitized.json`;
- `mcd_baseline/` and `mcd_configH/` (`mcd_gtmos_summary.json`);
- two further runs of the epoch-1079 checkpoint, in `f0_configH/` and `f0_configH_v2/`.

| Checkpoint | WER mean / median | UTMOSv2 mean / median (predicted) | MCD mean / median | Reference UTMOSv2 | Recorded RTF | Source |
|---|---|---|---|---:|---:|---|
| Full, epoch 1079 | 0.0864 / 0.0476 | 3.8070 / 3.8134 | 26.000 / 25.953 | 3.8181 | 0.0568 | `summary`, `mcd_configH` |
| Baseline, epoch 773 | 0.0917 / 0.0526 | 3.3656 / 3.3945 | 26.064 / 25.904 | 3.8146 | 0.0559 | `summary`, `mcd_baseline` |
| Full, epoch 1079, second run | 0.0843 / 0.0455 | 3.8244 / 3.8351 | | | | `f0_configH` |
| Full, epoch 1079, third run | 0.0876 / 0.0435 | 3.8130 / 3.8332 | | | | `f0_configH_v2`, F0 RMSE 0.342, corr 0.332 |

- **Paired Wilcoxon, 1079 vs 773** (`summary.sanitized.json`): WER p = 0.115, UTMOSv2 p = 1.6 × 10⁻⁷⁹.
- **Run-to-run variation:** the three runs of the same checkpoint differ by up to 0.33 pp WER and 0.017 UTMOSv2. The historical synthesis and UTMOSv2 were not seeded (`eval-protocol.md`).

**Why these numbers are not comparable with sections 2–6:**
- **The evaluation differs** in four ways:
  - unseeded synthesis;
  - unseeded UTMOSv2: the reference UTMOSv2 is 3.815–3.818 here, against 3.8308 under the locked protocol;
  - `load_from_checkpoint(strict=False)`;
  - F0 rows not recorded.
- **The training differs:** 1,100 epochs on 2 GPUs with batch 2 × 16, against 200 epochs on 1 GPU with batch 16.
- **Do not take** a difference between a historical number and an ablation number as an effect of any component.

**CPU benchmark of the historical pair (2026-10-28, new measurement).** The recorded RTFs above (0.0568 and 0.0559) have no recorded thread count or device setting; the harness ran on the CPU with PyTorch's default threads. The author asked on 28/10 for the pair to be timed again with the protocol of section 5.
- **Tool:** `tools/camera_ready/benchmark_rtf.py historical` (commit `5ff1c2b`; log `logs/rtf-historical-20261028-b.log`; files `eval/historical-pair-rtf.{json,csv}`, `historical-pair-rtf-runs.jsonl`, outside Git).
  - Each checkpoint's file name and SHA-256 were checked against `manifests/checkpoint-manifest.json` before the run.
  - A first launch at 02:48 was refused by the idle check (load average 1.14, left by a smoke run); its log is `logs/rtf-historical-20261028.log`.
- **Protocol:** that of section 5:
  - CPU, 1 and 2 threads pinned to vCPU 0 and to vCPUs 0 and 2, one interop thread;
  - 10 validation utterances as warm-up, then the first 100 of the same 500 test sentences;
  - three rounds with the order rotated, each cell in its own process.
  - Run on the idle PC, 02:59–03:09 +07; load average 0.06 at the start, no other tmux session.
- **Differences from the historical evaluation:**
  - The checkpoints load through `evaluate_like_history.load_model` in "legacy" group semantics with a strict state dict, not through `load_from_checkpoint(strict=False)`.
  - Synthesis is seeded as in the locked protocol, where the historical evaluation was unseeded.

| Checkpoint | 1 thread: model RTF, median [min, max] | 1 thread: end-to-end | 2 threads: model RTF, median [min, max] | 2 threads: end-to-end | Audio of the 100 utterances | Lengths equal in all 6 runs |
|---|---|---:|---|---:|---:|:-:|
| Full, epoch 1079 | 0.0888 [0.0873, 0.0895] | 0.0889 | 0.0579 [0.0575, 0.0588] | 0.0580 | 599.4 s | yes |
| Baseline, epoch 773 | 0.0789 [0.0772, 0.0803] | 0.0790 | 0.0514 [0.0509, 0.0521] | 0.0515 | 638.1 s | yes |

- **Checks:** all 12 runs had 100 utterances, the requested thread count and affinity. The round-to-round range is at most 3.9 % of the median (Baseline, 1 thread).
- **Full against Baseline:** Full is about 12.5 % slower at 1 thread and 12.6 % at 2 threads. The recorded values (0.0568 vs 0.0559, +1.6 %) do not show this gap; their thread count and machine state are not known.
- **Two threads** give a 1.53× speed-up for both checkpoints, as in section 5.
- **Against section 5:**
  - The historical Full (0.0888 / 0.0579) is close to the 200-epoch Full (0.0880 / 0.0567).
  - The historical Baseline (0.0789 / 0.0514) is slower than the 200-epoch Baseline (0.0682 / 0.0442).
  - This is consistent with the historical baseline's decoder keeping Snake in its ResBlocks: `checkpoint-manifest.json` lists 36 decoder Snake parameters for epoch 773. The "strict" Baseline of the ablation has none. This was not profiled.
  - The same caveat as above holds: training and evaluation differ, so these are timings of two different model versions, not an effect of any component.

## 10. Pitch control through the F0 conditioning (T14)

**Question.**
- The F0-predictor RMSE and correlation (sections 2, 4) measure the predictor only.
- They do not show the pitch of the waveform, or that the decoder uses its F0 conditioning.
- This test asks whether changing the F0 that the decoder receives changes the pitch of the output.

**Protocol.** Agreed with the author on 2026-10-25 before any run. It is a supplementary analysis, outside the protocol locked on 21/10.
- **Tool:** `tools/camera_ready/pitch_shift.py`, commit `524708f`.
- **Intervention:** a hook on `dec.forward` scales the decoder's F0 input by 2^(s/12) for s = −2, 0, +2 semitones, or removes it ("off").
  - With the evaluation's per-utterance seeds, the latent z, the durations and the noise are identical in every condition.
  - No file under `code/` was changed.
- **Models:** the best checkpoints of Full, NoVITS2, NoBigVGAN and Baseline-F0, and the historical Full (epoch 1079).
- **Sentences:** the first 100 test utterances.
- **F0 extraction:** pyworld `harvest` (60–600 Hz, 5 ms frames), with the same settings for all conditions. Lengths are equal, so frames pair one to one.
- **Metrics:**
  - per utterance, the median of 12·log₂(F0_c / F0_0) over frames voiced in both;
  - per model, the paired-bootstrap 95 % CI (10,000 resamples, seed 2026);
  - slope = (shift(+2) − shift(−2)) / 4.
- **Interpretation bands, fixed before the run:**
  - slope ≥ 0.8 with a CI excluding 0: controllable;
  - 0.2–0.8 with a CI excluding 0: partly controllable;
  - otherwise: not controllable.

**Run and checks:**
- **Run:** 2026-10-25 17:25–17:37 +07, CPU. 2,000 WAVs plus F0, records and summary are outside Git, in `<WSL_DEV_HOME>/outputs-pfn-cr/tests/pitch-shift-20261025/` (538 MB). Log: `logs/pitch-shift-20261025.log`.
- **Reproduction:** at s = 0 every model reproduced the evaluation's durations, 100 of 100.
  - For NoVITS2 and NoBigVGAN, whose evaluation WAVs are on this machine, the WAVs were byte-identical, 100 of 100.
  - The historical checkpoint was evaluated without seeds, so it has only the check that lengths are equal across conditions.
- **The hook acts on the decoder:** the waveform differs from s = 0 by a median relative RMS of 0.09–0.22 for ±2 semitones, and 0.59–0.90 for "off". The relative RMS includes phase differences, so it is not a perceptual measure.

| Model | +2 st requested: achieved shift [95 % CI] | −2 st: achieved [95 % CI] | Slope [95 % CI] | Reading (fixed bands) | Off: shift [95 % CI] | Voicing agreement with s = 0: +2 / −2 / off |
|---|---|---|---|---|---|---|
| Full | +0.018 [+0.017, +0.019] | −0.014 [−0.015, −0.013] | 0.0081 [0.0078, 0.0085] | not controllable | −0.078 [−0.083, −0.073] | 0.935 / 0.932 / 0.903 |
| NoVITS2 | +0.020 [+0.019, +0.021] | −0.017 [−0.018, −0.016] | 0.0092 [0.0088, 0.0096] | not controllable | −0.059 [−0.065, −0.055] | 0.929 / 0.927 / 0.890 |
| NoBigVGAN | +0.009 [+0.008, +0.010] | −0.008 [−0.009, −0.008] | 0.0043 [0.0041, 0.0045] | not controllable | −0.152 [−0.162, −0.143] | 0.943 / 0.942 / 0.871 |
| Baseline-F0 | +0.013 [+0.012, +0.014] | −0.012 [−0.012, −0.011] | 0.0062 [0.0058, 0.0065] | not controllable | −0.423 [−0.505, −0.356] | 0.926 / 0.927 / 0.857 |
| Historical Full 1079 | +0.023 [+0.022, +0.024] | −0.022 [−0.024, −0.021] | 0.0114 [0.0109, 0.0120] | not controllable | −0.009 [−0.013, −0.006] | 0.926 / 0.923 / 0.873 |

All values are in semitones. Every model used 100 of 100 utterances.

**What it shows:**
- **The pitch of the waveform does not follow the F0 conditioning.**
  - A requested ±2 semitones moves the output pitch by 0.008–0.023 semitones. That is about 0.4–1.1 % of the request, in every model including the paper's historical Full model.
  - The slopes are significantly above 0 but far below the 0.2 band. By the bands fixed in advance, the pitch is **not controllable** through this path.
- **The decoder does use the conditioning,** but not for pitch:
  - the waveform changes clearly under ±2 semitones and very strongly under "off";
  - removing the conditioning moves the pitch by only −0.01 to −0.42 semitones;
  - it lowers the voicing agreement to 0.86–0.90.
- **Where the pitch comes from:** in this architecture the predicted F0 reaches only the decoder (`SynthesizerTrn.infer`). The pitch of the output therefore comes from the latent z produced by the prior and the flow, not from the explicit F0.
- **Voicing agreement under ±2** is only 0.92–0.94 although the pitch hardly moves. The harvest voicing decision reacts to small waveform changes, so this agreement is not by itself evidence of a change in voicing.

**What it does not show:**
- **Not every form of pitch control is ruled out.** Shifting the latent z, or training with a stronger F0 path, was not tested.
- **Larger shifts** and the quality of the shifted audio (UTMOSv2, WER) were not measured.
- **The F0 path may affect other properties:** this test does not show that the F0 path has no benefit. The UTMOSv2 effects of F (section 4) may come through effects other than pitch.

## 11. Waveform F0 against the recordings (T13)

**Question.** How close is the pitch of the synthesized waveform to the pitch of the recording of the same sentence, and does the F0 group bring it closer? Review 1 asks for waveform F0 metrics.

**Protocol.** Agreed with the author on 2026-10-25 before the run. It is a supplementary analysis, outside the protocol locked on 21/10.
- **Tool:** `tools/camera_ready/waveform_f0.py`, commit `ac6e68a`.
- **Models:** the eight cells (best checkpoints) and the historical pair (773, 1079).
- **Audio:** the 500 test utterances, synthesized with the evaluation's seeds and scales.
  - The historical pair is synthesized with seeds here for the first time. Its numbers are not those of the historical evaluation.
- **Reference:** the normalised audio of the cache, the same reference as the MCD.
- **F0 extraction:** pyworld `harvest` (60–600 Hz, 5 ms) for both the synthesized and the reference audio.
- **Alignment (primary):** DTW exactly as the harness's MCD (24 MFCC, n_fft 1024, hop 256, normalised by the reference frames, euclidean). Each MFCC frame takes the nearest F0 frame.
- **Alignment (sensitivity):** a linear time stretch.
- **Metrics, per utterance:**
  - F0 RMSE and mean offset, in cents (pairs voiced in both);
  - log-F0 correlation;
  - VDE (voicing decision error);
  - GPE (gross pitch error, more than 20 %);
  - FFE (F0 frame error);
  - coverage.
- **Statistics:**
  - per model, bootstrap 95 % CIs over utterances;
  - six paired comparisons: the four F edges, Full vs Baseline, and 1079 vs 773;
  - paired-bootstrap CIs of the difference (on − off);
  - two-sided Wilcoxon, Holm within each metric across the six.
- **Interpretation:** numbers only. A non-significant difference is not equivalence.

**Run and checks:**
- **Run:** 2026-10-25 17:57–18:35 +07, CPU. 5,000 WAVs, records and summary are outside Git, in `<WSL_DEV_HOME>/outputs-pfn-cr/tests/waveform-f0-20261025/` (1.4 GB). Log: `logs/waveform-f0-20261025.log`.
- **Reproduction:** the eight cells reproduced the evaluation's durations, 500 of 500. Baseline, Baseline-BigVGAN, NoBigVGAN and NoVITS2 also reproduced the evaluation's WAV bytes, 500 of 500.
- **Self-check:** a reference aligned with itself gave 0 cents and no voicing error.
- **Smoke-run check of the high error level.** Five utterances, three models; this sanity check was not a step of the protocol.
  - The synthesized and reference F0 ranges agree (about 150–300 Hz).
  - Only 1–8 % of aligned frames are more than 1,000 cents apart, so octave errors are not the cause.
  - Shifting the synthesized track by ±30 ms changes the RMSE only moderately.
  - The high level therefore reflects contours that differ from the particular recording, not a measurement fault.

#### Per model, DTW alignment (primary)

| Model | B V F | F0 RMSE (cents) | Offset (cents) | log-F0 corr | VDE | GPE | FFE | Coverage |
|---|:-:|---|---|---|---|---|---|---:|
| Full | 1 1 1 | 439.1 [432.5, 445.7] | -32.8 [-43.9, -21.8] | 0.338 [0.323, 0.353] | 0.159 [0.154, 0.163] | 0.372 [0.363, 0.381] | 0.441 [0.433, 0.449] | 0.763 |
| NoF0 | 1 1 0 | 445.9 [439.8, 452.2] | -4.0 [-14.5, +6.6] | 0.352 [0.337, 0.366] | 0.162 [0.158, 0.166] | 0.372 [0.364, 0.380] | 0.440 [0.433, 0.447] | 0.751 |
| NoVITS2 | 1 0 1 | 451.1 [444.3, 457.9] | -49.5 [-60.6, -38.1] | 0.339 [0.324, 0.353] | 0.157 [0.153, 0.161] | 0.375 [0.366, 0.383] | 0.438 [0.431, 0.446] | 0.755 |
| Baseline-BigVGAN | 1 0 0 | 451.9 [445.3, 458.6] | +24.7 [+13.4, +36.0] | 0.350 [0.335, 0.365] | 0.158 [0.154, 0.162] | 0.388 [0.379, 0.396] | 0.451 [0.443, 0.458] | 0.757 |
| NoBigVGAN | 0 1 1 | 448.0 [440.9, 455.2] | +33.9 [+22.1, +45.6] | 0.361 [0.345, 0.377] | 0.156 [0.152, 0.161] | 0.388 [0.379, 0.397] | 0.450 [0.442, 0.458] | 0.760 |
| Baseline-VITS2 | 0 1 0 | 447.7 [441.1, 454.4] | -15.7 [-26.8, -4.4] | 0.356 [0.341, 0.371] | 0.160 [0.155, 0.164] | 0.381 [0.372, 0.389] | 0.446 [0.439, 0.453] | 0.755 |
| Baseline-F0 | 0 0 1 | 513.0 [504.9, 521.2] | +50.7 [+37.2, +64.2] | 0.316 [0.301, 0.332] | 0.168 [0.164, 0.172] | 0.432 [0.422, 0.441] | 0.485 [0.477, 0.492] | 0.738 |
| Baseline | 0 0 0 | 456.1 [449.5, 462.7] | +15.5 [+4.1, +26.7] | 0.339 [0.324, 0.354] | 0.160 [0.156, 0.164] | 0.395 [0.386, 0.404] | 0.457 [0.450, 0.464] | 0.754 |
| Historical Baseline-773 | (773) | 451.6 [444.9, 458.3] | +21.3 [+9.7, +33.0] | 0.374 [0.359, 0.389] | 0.155 [0.151, 0.159] | 0.385 [0.376, 0.394] | 0.446 [0.438, 0.453] | 0.759 |
| Historical Full-1079 | (1079) | 452.6 [445.6, 459.5] | -24.3 [-35.3, -13.0] | 0.369 [0.353, 0.384] | 0.155 [0.151, 0.160] | 0.380 [0.371, 0.390] | 0.443 [0.435, 0.452] | 0.760 |

#### Paired comparisons, DTW (on − off; Holm across the six within each metric)

| Comparison | Metric | Mean diff [95 % CI] | p (raw) | p (Holm) | n |
|---|---|---|---:|---:|---:|
| Baseline → Baseline-F0 | rmse_cents | +56.9 [+49.6, +64.6] | 2.0e-38 | 1.2e-37 | 500 |
| Baseline → Baseline-F0 | corr_logf0 | -0.023 [-0.040, -0.006] | 0.002 | 0.011 | 500 |
| Baseline → Baseline-F0 | vde | +0.008 [+0.004, +0.012] | 1.9e-05 | 1.2e-04 | 500 |
| Baseline → Baseline-F0 | gpe | +0.037 [+0.027, +0.046] | 5.0e-15 | 3.0e-14 | 500 |
| Baseline → Baseline-F0 | offset_cents | +35.2 [+24.3, +46.2] | 7.3e-10 | 7.3e-10 | 500 |
| Baseline-VITS2 → NoBigVGAN | rmse_cents | +0.3 [-5.9, +6.7] | 0.639 | 1.000 | 500 |
| Baseline-VITS2 → NoBigVGAN | corr_logf0 | +0.005 [-0.010, +0.021] | 0.217 | 0.650 | 500 |
| Baseline-VITS2 → NoBigVGAN | vde | -0.003 [-0.007, +0.001] | 0.098 | 0.488 | 500 |
| Baseline-VITS2 → NoBigVGAN | gpe | +0.007 [-0.001, +0.015] | 0.049 | 0.147 | 500 |
| Baseline-VITS2 → NoBigVGAN | offset_cents | +49.7 [+41.1, +58.3] | 1.8e-25 | 9.2e-25 | 500 |
| Baseline-BigVGAN → NoVITS2 | rmse_cents | -0.8 [-7.3, +5.6] | 0.937 | 1.000 | 500 |
| Baseline-BigVGAN → NoVITS2 | corr_logf0 | -0.012 [-0.027, +0.003] | 0.067 | 0.333 | 500 |
| Baseline-BigVGAN → NoVITS2 | vde | -0.001 [-0.005, +0.003] | 0.120 | 0.488 | 500 |
| Baseline-BigVGAN → NoVITS2 | gpe | -0.013 [-0.021, -0.005] | 0.004 | 0.017 | 500 |
| Baseline-BigVGAN → NoVITS2 | offset_cents | -74.2 [-82.7, -65.7] | 3.7e-45 | 2.2e-44 | 500 |
| NoF0 → Full | rmse_cents | -6.8 [-12.8, -0.9] | 0.021 | 0.083 | 500 |
| NoF0 → Full | corr_logf0 | -0.013 [-0.029, +0.002] | 0.075 | 0.333 | 500 |
| NoF0 → Full | vde | -0.003 [-0.007, +0.001] | 0.135 | 0.488 | 500 |
| NoF0 → Full | gpe | +0.000 [-0.008, +0.008] | 0.714 | 0.714 | 500 |
| NoF0 → Full | offset_cents | -28.7 [-36.6, -21.0] | 9.5e-12 | 1.9e-11 | 500 |
| Baseline → Full | rmse_cents | -17.0 [-23.0, -11.1] | 2.0e-08 | 9.9e-08 | 500 |
| Baseline → Full | corr_logf0 | -0.001 [-0.017, +0.016] | 0.657 | 1.000 | 500 |
| Baseline → Full | vde | -0.001 [-0.005, +0.003] | 0.501 | 1.000 | 500 |
| Baseline → Full | gpe | -0.023 [-0.030, -0.015] | 5.0e-09 | 2.5e-08 | 500 |
| Baseline → Full | offset_cents | -48.2 [-56.5, -40.0] | 8.8e-24 | 3.5e-23 | 500 |
| Historical Baseline-773 → Historical Full-1079 | rmse_cents | +1.0 [-5.4, +7.4] | 0.609 | 1.000 | 500 |
| Historical Baseline-773 → Historical Full-1079 | corr_logf0 | -0.005 [-0.020, +0.009] | 0.602 | 1.000 | 500 |
| Historical Baseline-773 → Historical Full-1079 | vde | +0.000 [-0.004, +0.004] | 0.949 | 1.000 | 500 |
| Historical Baseline-773 → Historical Full-1079 | gpe | -0.004 [-0.013, +0.004] | 0.243 | 0.487 | 500 |
| Historical Baseline-773 → Historical Full-1079 | offset_cents | -45.6 [-55.0, -36.1] | 2.0e-19 | 6.0e-19 | 500 |

**Linear-stretch sensitivity.** Full tables: `summary.json` → `models.linear`, `comparisons.linear`.
- Every model scores worse with the linear stretch: RMSE 492–567 cents, correlation 0.18–0.23, VDE 0.20–0.23.
- The significance after Holm agrees with DTW, with two exceptions:
  - **NoF0 → Full:** with the stretch, Full is better on RMSE (−29.0 [−35.5, −22.5]), correlation, VDE (−0.032) and GPE (−0.027), all p (Holm) ≤ 0.03. With DTW, none of these is significant (RMSE −6.8, p (Holm) 0.083).
    - NoF0's audio is about 40 % longer (section 8), so a linear stretch misaligns it more.
  - **The correlation drop of Baseline → Baseline-F0** is significant with DTW (p (Holm) 0.011) but not with the stretch (0.648).
- The DTW result is the primary one.

**What it shows:**
- **No model's waveform pitch closely follows the recording's contour.**
  - log-F0 correlation is 0.32–0.37 and F0 RMSE 439–513 cents, with gross pitch errors in 37–43 % of the voiced pairs.
  - These models produce their own intonation, not the recording's. This is expected without the recordings' durations and prosody. The level is similar for the 200-epoch cells and for the historical pair.
- **The F0 group does not bring the waveform pitch closer to the recording.**
  - **Alone** (Baseline → Baseline-F0), it makes every metric worse: RMSE +56.9 cents, correlation −0.023, GPE +0.037; all p (Holm) ≤ 0.011.
  - **With VITS2 or with BigVGAN**, it changes RMSE, correlation, VDE and GPE by nothing significant, except a GPE −0.013 with BigVGAN (p (Holm) 0.017).
  - **With both** (NoF0 → Full), the DTW differences on these four metrics are not significant after Holm.
- **The historical pair does not differ** on RMSE, correlation, VDE or GPE (1079 vs 773: RMSE +1.0 [−5.4, +7.4], p (Holm) 1.0).
- **Full vs Baseline (200 epochs)** is a small improvement: RMSE −17.0 cents (about 4 % of the level), GPE −0.023; correlation does not change.
- **The F0 group mainly moves the mean pitch level,** by −74 to +50 cents depending on the cell; these offset changes are significant on all four F edges. The direction is not consistent. Every model's mean offset from the recordings stays within ±51 cents.
- **Together with section 10:** the F0 branch neither steers the waveform pitch nor makes it track the recordings better in these checkpoints. Its UTMOSv2 effects (section 4) come through other properties.

**Limits:**
- **One reference realisation:** the recording is one valid intonation of each sentence. A low agreement with it does not by itself mean an unnatural pitch.
- **Alignment and extraction:** DTW on MFCC and harvest both add error.
- **Multiplicity:** the six comparisons are corrected within each metric, not across the metrics.
- **No pitch range:** pitch range (for example the spread of log-F0 per utterance) was not in the protocol and was not measured.

## Draft text for the author

Proposals only. The manuscript is not edited here. Each number has its source above.

> To attribute the improvement, we trained all eight combinations of the three components (BigVGAN-style decoder, VITS2 modules, F0 conditioning) under an identical budget of 200 epochs (156,400 generator updates, one seed per configuration) and evaluated them with the same protocol on the 500 test utterances. Relative to the baseline, the full model raises the predicted naturalness (UTMOSv2) by 0.575 (95 % CI 0.546–0.604; Wilcoxon, Holm-adjusted p < 10⁻⁷⁸), while WER (+0.54 pp, CI −0.42 to +1.52) and MCD (+0.065, CI −0.020 to +0.148) do not differ significantly. A Shapley decomposition of the UTMOSv2 gain attributes 56 % to the BigVGAN-style decoder, 21 % to the VITS2 modules and 23 % to F0 conditioning.

> The components interact. The VITS2 modules raise UTMOSv2 when the BigVGAN-style decoder is absent (+0.432 without F0, +0.184 with F0) but not when it is present (−0.306 without F0; −0.003, p = 0.60, with F0). F0 conditioning gives its largest gain when both other components are present (+0.327, p < 10⁻⁵⁶).

> Pitch control: scaling the F0 input of the decoder by ±2 semitones, with the latent, durations and noise fixed, moves the pitch of the output by only 0.008–0.023 semitones in every F0-conditioned model, including the historical full model (epoch 1079). The F0 conditioning changes the waveform but does not determine its pitch, which follows the latent representation. Claims of explicit pitch control through the F0 branch are therefore not supported by these checkpoints. Measured on the waveform against the recordings (DTW-aligned, 500 test sentences), the F0 branch also does not bring the pitch contour closer to the reference: on its own it increases the F0 RMSE by 57 cents, it gives no significant change when the other components are present, and the historical full and baseline models do not differ (F0 RMSE 453 vs 452 cents; log-F0 correlation 0.37 for both).

> Limitations: one training seed per configuration, so the reported intervals reflect variation across test sentences, not across training runs; the eight runs were trained on two GPU types; all runs were still improving at 200 epochs; checkpoints were selected by a teacher-forced mel reconstruction loss on validation, which does not reflect the duration predictor; UTMOSv2 is a predicted score, not a listening test.
