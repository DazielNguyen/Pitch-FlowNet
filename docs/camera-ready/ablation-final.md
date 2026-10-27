# Final statistics of the camera-ready comparisons

**Scope.** SOICT 2026 paper 4478, camera-ready, 2026-10-28.
- **What this page holds:** the statistics of every comparison that the paper reports or should report, in one method:
  - the main pair, PitchFlowNet (epoch 1079) against Piper (epoch 773);
  - the 2^3 ablation at 200 epochs;
  - the pitch analyses.
- **Ready-to-use tables:**
  - Table F1 for the main result;
  - Table F2 for the ablation;
  - the appendix tables below them.
- **No new score:** nothing was trained, synthesized or re-scored for this page. Every number is computed from per-sentence files that exist, by `tools/camera_ready/ablation_final.py` (commit `c89e9e8`).
- **The manuscript:** it is not edited here. Proposed wording is in the last section.

**Reading the numbers:**
- UTMOSv2 is a predicted score, not MOS.
- A non-significant difference is not equivalence.
- Significance is never read from a table of means; every claim below has its paired test.

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
- That page used percentile CIs from 10,000 resamples (seed 2026). This page uses the manuscript's BCa CIs, so a few interval bounds differ by about 0.01 in the reported units.
- Means and p-values are identical.

**Sources:**
- per-sentence files of the 8 cells: `<WSL_DEV_HOME>/outputs-pfn-cr/eval/<label>/`;
- the historical pair: `evaluation/eval_h_vs_baseline/`, tracked;
- T13 records: `tests/waveform-f0-20261025/records.jsonl`; T14 records: `tests/pitch-shift-20261025/records.jsonl`;
- RTF: `eval/ablation8-rtf.json`, `historical-pair-rtf.json`, `historical-pair-rtf-t12.json`;
- checkpoints: `final/` and the historical checkpoint files.
- Outputs: `eval/ablation-final-stats.{json,csv}`, `ablation-final-tables.md`, `ablation-final-paper.md`. Their hashes are in `manifests/camera-ready/ablation8-derived.json`.

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

### Inference parameters of the historical pair, recomputed

| Model | Elements | Millions | Manuscript (M) | FP32 bytes (MB, 10^6) |
|---|---:|---:|---:|---:|
| Piper-773 | 15,664,671 | 15.66 | 15.66 | 62.66 |
| PitchFlowNet-1079 | 17,207,008 | 17.21 | 17.21 | 68.83 |

**Notes on the appendix tables:**
- **RTF:**
  - It is the model-only RTF on the idle PC (WSL, Intel Core i7-14700). Threads are declared, pinned one per virtual core; the placement on P-cores or E-cores is not controlled.
  - It is the median over three rounds; the 8 cells used 100 test utterances.
  - No test is applied: three rounds give no sentence-level pairing.
  - The two historical values recorded in the paper (0.0559, 0.0568) have no recorded thread count. They are not in this table.
- **P13 numbers:** they use the historical pair synthesized with seeds (T13). They are not the historical evaluation.
- **Source of every number:** `eval/ablation-final-stats.json`.

## Draft text for the author

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
