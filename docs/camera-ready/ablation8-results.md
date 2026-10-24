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

## Sources

| What | Where (outside Git unless stated) |
|---|---|
| Evaluations, one folder per label | `<WSL_DEV_HOME>/outputs-pfn-cr/eval/<label>/`: `summary.json`, `protocol.json`, `*_per_sentence.csv` (WER, UTMOSv2), `mcd/mcd_gtmos.csv` (MCD, reference UTMOSv2), `reference.jsonl` |
| Final checkpoints and run records | `<WSL_DEV_HOME>/outputs-pfn-cr/final/<run_id>/` (read-only, `SHA256SUMS`) |
| Pairwise Wilcoxon tests of WER and UTMOSv2 | `eval/compare-<off>_<on>.json`, from `tools/camera_ready/evaluate_like_history.py compare` (commit `f773be2`) |
| Analysis tool | `tools/camera_ready/ablation8_analysis.py` (in Git, commit `5eb3510`), tests `tests/camera_ready/test_ablation8.py` |
| Derived files | `eval/ablation8-check.json`, `ablation8-table.{csv,md}`, `ablation8-pairs.{csv,json}`, `ablation8-effects.{csv,json}`, `ablation8-cost.{csv,json}` |
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

| Label | B | V | F | Best epoch | WER mean | WER median | UTMOSv2 mean | UTMOSv2 median | MCD mean | F0 RMSE | F0 corr | Recorded RTF | Training GPU |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `PFN-Full-200ep-best` | 1 | 1 | 1 | 188 | 0.1135 | 0.0741 | 3.5156 | 3.5204 | 25.638 | 0.5399 | 0.1861 | 0.0682 | NVIDIA GeForce RTX 5070 |
| `PFN-NoF0-200ep-best` | 1 | 1 | 0 | 192 | 0.1302 | 0.0870 | 3.1887 | 3.2026 | 25.576 | n/a | n/a | 0.0926 | NVIDIA GeForce RTX 5070 |
| `PFN-NoVITS2-200ep-best` | 1 | 0 | 1 | 188 | 0.1101 | 0.0588 | 3.5185 | 3.5452 | 25.760 | 0.5827 | 0.1992 | 0.0722 | NVIDIA GeForce RTX 4070 Ti |
| `PFN-Baseline-BigVGAN-200ep-best` | 1 | 0 | 0 | 199 | 0.1009 | 0.0667 | 3.4945 | 3.5125 | 25.744 | n/a | n/a | 0.0737 | NVIDIA GeForce RTX 4070 Ti |
| `PFN-NoBigVGAN-200ep-best` | 0 | 1 | 1 | 199 | 0.1214 | 0.0769 | 3.2442 | 3.2703 | 25.923 | 0.4813 | 0.1784 | 0.0709 | NVIDIA GeForce RTX 4070 Ti |
| `PFN-Baseline-VITS2-200ep-best` | 0 | 1 | 0 | 188 | 0.1105 | 0.0714 | 3.3723 | 3.3817 | 25.623 | n/a | n/a | 0.0775 | NVIDIA GeForce RTX 5070 |
| `PFN-Baseline-F0-200ep-best` | 0 | 0 | 1 | 195 | 0.1349 | 0.0714 | 3.0599 | 3.0956 | 26.728 | 0.6200 | 0.1923 | 0.0529 | NVIDIA GeForce RTX 5070 |
| `PFN-Baseline-200ep-best` | 0 | 0 | 0 | 194 | 0.1082 | 0.0667 | 2.9407 | 2.9585 | 25.573 | n/a | n/a | 0.0593 | NVIDIA GeForce RTX 4070 Ti |

- WER is a fraction here (0.1135 = 11.35 %).
- F0 RMSE and correlation are the F0-predictor metric (phoneme-level log-F0 against the cached reference). They exist only for F = 1.
- **The recorded RTF is not a reportable RTF:** it was measured inline, on two machines, with other work running (section 5).

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

**Inference cost.** The recorded `mean_rtf` was measured inline during evaluation, on two different machines that were not idle. It is **indicative only and not reportable as RTF.** No idle-machine RTF benchmark (T15) has been run yet.

**Pareto sets** of mean quality against cost:

- WER vs inference-path size: Baseline, Baseline-BigVGAN
- WER vs recorded RTF: Baseline, Baseline-BigVGAN, Baseline-F0
- UTMOSv2 vs inference-path size: Baseline, Baseline-BigVGAN, NoVITS2
- UTMOSv2 vs recorded RTF: Baseline-F0, Full, NoVITS2
- MCD vs inference-path size: Baseline
- MCD vs recorded RTF: Baseline, Baseline-F0

**Reading:**
- **The top UTMOSv2 group** has three cells: NoVITS2 (3.519), Full (3.516) and Baseline-BigVGAN (3.495).
  - Two of their differences were tested: NoVITS2 vs Full, −0.003 (Holm 0.60), and Baseline-BigVGAN → NoVITS2, +0.024 (Holm 0.21).
  - Neither is significant. **This does not show that they are equal.**
  - Full vs Baseline-BigVGAN is not a cube edge and was not tested.
- **Size:** Baseline-BigVGAN is the smallest of the three; it is B alone, 3,648 elements more than Baseline. NoVITS2 is 2.1 % larger than Baseline, Full 9.4 %.
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
  - NoVITS2 is therefore a smaller alternative (−6.7 % inference-path elements) with the same predicted naturalness within the test's resolution.
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
- **RTF:** only inline values exist; no idle-machine benchmark (T15).

## Draft text for the author

Proposals only. The manuscript is not edited here. Each number has its source above.

> To attribute the improvement, we trained all eight combinations of the three components (BigVGAN-style decoder, VITS2 modules, F0 conditioning) under an identical budget of 200 epochs (156,400 generator updates, one seed per configuration) and evaluated them with the same protocol on the 500 test utterances. Relative to the baseline, the full model raises the predicted naturalness (UTMOSv2) by 0.575 (95 % CI 0.546–0.604; Wilcoxon, Holm-adjusted p < 10⁻⁷⁸), while WER (+0.54 pp, CI −0.42 to +1.52) and MCD (+0.065, CI −0.020 to +0.148) do not differ significantly. A Shapley decomposition of the UTMOSv2 gain attributes 56 % to the BigVGAN-style decoder, 21 % to the VITS2 modules and 23 % to F0 conditioning.

> The components interact. The VITS2 modules raise UTMOSv2 when the BigVGAN-style decoder is absent (+0.432 without F0, +0.184 with F0) but not when it is present (−0.306 without F0; −0.003, p = 0.60, with F0). F0 conditioning gives its largest gain when both other components are present (+0.327, p < 10⁻⁵⁶).

> Limitations: one training seed per configuration, so the reported intervals reflect variation across test sentences, not across training runs; the eight runs were trained on two GPU types; all runs were still improving at 200 epochs; UTMOSv2 is a predicted score, not a listening test.
