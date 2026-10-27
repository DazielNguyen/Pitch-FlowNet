# Experiment plan: camera-ready ablation (SOICT 2026, paper 4478)

**Status.** All decisions below were taken by the author before any test-set result. None was based on validation results.

| Date | Decision |
|---|---|
| 2026-10-20 | Ablation design and protocol: 275 epochs per run, four leave-one-out runs |
| 2026-10-21 | Extended to all eight group combinations |
| 2026-10-21 | Budget lowered from 275 to **200 epochs per run**, so that all eight fit |

**Outcome (24/10).** The ablation is done: all eight cells were trained for 200 epochs and evaluated with the locked protocol (`eval-protocol.md`). They were analysed on the PC: `ablation8-results.md`, with the hashes of every source and derived file in `manifests/camera-ready/ablation8-derived.json`.

**Analysis choices made after the test results existed.** They come from the author's task of 24/10; they are not decisions of this plan:
- the 13 paired comparisons: the 12 cube edges and Full vs Baseline;
- the Holm correction within each metric;
- the paired bootstrap CIs;
- the factorial effects and the Shapley attribution.

**RTF (T15).** The CPU benchmark of the eight checkpoints ran on 24/10 (`ablation8-results.md`, section 5). Its protocol was agreed with the author on that day, after the test results existed:
- 1 and 2 threads;
- 100 test utterances;
- three rounds.

**Two parts.**
- **The paper's main result stays the historical pair:** Baseline epoch 773 and Full epoch 1079. It is evaluated as it is; nothing is retrained to replace it.
- **New: an ablation over the three groups.**
  - Leave-one-out and add-one runs, i.e. all 2^3 = 8 combinations of BigVGAN (B), VITS2 (V) and F0 (F).
  - It answers the main criticism shared by the three reviews: "only baseline and full are evaluated; it is impossible to determine which component is responsible".

## Decisions (O5 / T08)

| | Question | Decision |
|---|---|---|
| (a) | `train_shuffle` | `false`, the historical data order |
| (b) | MAS noise decay | `4e-6` per G update, the historical schedule. The noise reaches 0 after 2,500 G updates, about 3.2 epochs |
| (c) | Budget and devices | **200 epochs per run** = 200 × 782 batches = **156,400 G updates**, at batch 16 on one GPU. Was 275 epochs (215,050 G updates) until 21/10 |
| (d) | Replication seeds | not planned (no time); single seed 1234, reported as a limitation |
| (e) | Checkpoint for evaluation | the minimum `val_mel_l1_global` on the 100 validation utterances (`best-*.ckpt`), fixed before any test result. The test split is never used for selection. The best and the last checkpoint (epoch 200) of every run are kept as read-only copies (`tools/camera_ready/finalize_run.sh`) |

**What 200 epochs means against the historical runs** (1,100 epochs of 12,500 utterances, 430,100 G updates at batch 2 × 16):
- **Epochs and samples seen:** 200/1,100 = 18.2 %.
- **G updates:** 156,400/430,100 = 36.4 %.

**Why the budget does not change training before the stop.** The budget only decides when a run stops (`GUpdateBudget`):
- the LR schedule steps once per epoch with a fixed gamma (`ExponentialLR`, 0.999875);
- the MAS noise decays per G update;
- neither depends on the total budget.

So the first 200 epochs of a 275-epoch run are the same training as a 200-epoch run.

**Why the historical Full model is not the reference for the ablation.**
- It differs from any run that fits before 27/10 in budget (1,100 epochs), devices and batch (2 GPUs, 32), and source: the historical source is a candidate, not proven to be what ran.
- A difference to it would mostly measure training budget, not the removed group.
- So the ablation has its own reference, `PFN-Full`, trained like the other seven runs.

## Run matrix: all 2^3 = 8 combinations

**Recipe.** All runs use:
- train seed 1234, data seed 1234, validation seed 2026;
- strict group gating, batch 16, one GPU, bf16-mixed;
- 200 epochs.

**Checks on the configs** (`configs/PFN-<variant>/PFN-<variant>-200ep.json`):
- a script showed that they differ only in the group flags (`use_bigvgan`, `use_vits2`, `use_f0`), the run ID and the notes;
- each one differs from its 275-epoch version only in run ID, `max_g_updates`, `budget_label` and note;
- every config passes `train.py --dry-run`;
- `tests/camera_ready/test_gating.py` builds all eight combinations.

| Priority | Run ID | Config | B | V | F | Question it answers |
|---:|---|---|:-:|:-:|:-:|---|
| 1 | `PFN-Full-275ep-s1234`, **ended at epoch 200** (below) | `configs/PFN-Full/PFN-Full-275ep.json` + `--stop-after-epochs 200` | on | on | on | reference |
| 2 | `PFN-NoF0-200ep-s1234` | `configs/PFN-NoF0/PFN-NoF0-200ep.json` | on | on | off | leave-one-out: F0 |
| 2 | `PFN-NoBigVGAN-200ep-s1234` | `configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json` | off | on | on | leave-one-out: BigVGAN |
| 2 | `PFN-NoVITS2-200ep-s1234` | `configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json` | on | off | on | leave-one-out: VITS2 |
| 3 | `PFN-Baseline-BigVGAN-200ep-s1234` | `configs/PFN-Baseline-BigVGAN/PFN-Baseline-BigVGAN-200ep.json` | on | off | off | add-one: BigVGAN only (Review 1, "Piper with only a BigVGAN-style decoder") |
| 3 | `PFN-Baseline-200ep-s1234` | `configs/PFN-Baseline/PFN-Baseline-200ep.json` | off | off | off | add-one reference; new strict baseline, **not** the historical epoch-773 baseline |
| 4 | `PFN-Baseline-VITS2-200ep-s1234` | `configs/PFN-Baseline-VITS2/PFN-Baseline-VITS2-200ep.json` | off | on | off | add-one: VITS2 only (Review 1, "VITS2 itself") |
| 4 | `PFN-Baseline-F0-200ep-s1234` | `configs/PFN-Baseline-F0/PFN-Baseline-F0-200ep.json` | off | off | on | add-one: F0 only |

**The Full reference.**
- It had started on 20/10 with the 275-epoch config and was at epoch 168 when the budget changed.
- On 21/10 it was interrupted once, right after the epoch-168 checkpoint, and resumed with `--stop-after-epochs 200`. Details in `runs-MardeusPC.md`.
- Its run directory, run ID and config file keep the 275-epoch names, so its records stay consistent. Its best checkpoint is chosen among epochs 0–199, like every other run.
- By the argument above, it is equivalent to `PFN-Full-200ep`.

**What each set answers:**
- **Leave-one-out:** Full against each one-group-off run. How much each group contributes when the other two are present.
- **Add-one:** Baseline against each one-group-on run. How much each group gives on its own.
- **Interactions:** with all eight cells, the main effects and interactions of the 2^3 factorial can be estimated. One seed per cell makes the interaction estimates noisy, and the paper says so.

## Devices and schedule

**Speed so far:**
- The RTX 5070 runs about 9.5 min per epoch (`PFN-Full`, 20–21/10), against 13.2 min in the pilot (`eta.md`). So a 200-epoch run takes about 32 h, plus interruptions.
- The RTX 4070 Ti speed is measured in the first epoch of its runs.

| GPU | Runs, in order | Estimated end |
|---|---|---|
| MardeusPC, RTX 5070 | `PFN-Full` (to epoch 200) → `PFN-NoF0` → `PFN-Baseline-VITS2` → `PFN-Baseline-F0` | ≈ 21/10 10:30 → ≈ 22/10 19:00 → ≈ 24/10 03:00 → ≈ 25/10 11:00 |
| PC, RTX 4070 Ti #0 | `PFN-NoBigVGAN` → `PFN-Baseline-BigVGAN` | ≈ 22/10 22:00 → ≈ 24/10 06:00, if it starts ≈ 21/10 14:00 |
| PC, RTX 4070 Ti #1 | `PFN-NoVITS2` → `PFN-Baseline` | same |

**Actual ends** (`runs-MardeusPC.md`, `runs-Daziel.md`):
- RTX 5070: Full 21/10 10:08, NoF0 22/10 17:18, Baseline-VITS2 23/10 18:27, Baseline-F0 24/10 16:24.
- PC GPU 0: NoBigVGAN 23/10 00:12, Baseline-BigVGAN 23/10 23:09.
- PC GPU 1: NoVITS2 22/10 23:16, Baseline 23/10 20:20.
- The PC started on 22/10 01:33, not on 21/10.

**Capacity.**
- All eight runs should end by about 25/10 11:00. That leaves about 57 h for evaluation and reserve before 27/10 evening; the PC GPUs are free for evaluation from about 24/10.
- If the PC starts later, `PFN-Baseline-F0` may move to whichever GPU is free first.
- Never shorten a run to fit; every run keeps 200 epochs.

**GPU types.**
- `Full` and `NoF0`, the central F0 contrast, share the RTX 5070.
- The pair `Baseline` and `Baseline-BigVGAN` shares the RTX 4070 Ti type.
- Every other contrast crosses GPU types. Configs, batch and update count are the same; the paper states the GPU type of each run.

**Constraints:**
- Each run uses one GPU and runs in its own tmux session. One run per GPU.
- On the PC, a dedicated environment is used. Nothing of the other project is touched, and nothing starts before its GPUs are free.
- Every launch needs the author's OK of the exact command.
- The watchdog only resumes after the known CUDA error and never starts a new run (`runbook-MardeusPC.md`).

## Estimand and limits

**Estimand.** The effect of each group (B, V, F) on the evaluation metrics, at a fixed 200-epoch budget with one seed. It is measured two ways:
- by turning the group off in the full configuration (leave-one-out);
- by turning it on in the baseline (add-one).

**Not estimated:**
- the effect at the historical 1,100-epoch budget;
- interactions, unless all eight cells finish. Even then, with one seed per cell, they are imprecise.

**Limits to state in the paper:**
- **Training:** one seed per run; a shortened budget, 18.2 % of the historical epochs.
- **Hardware:** two GPU types.
- **Data and quality metrics:** single-speaker English. UTMOSv2 is a predicted score, not MOS. A non-significant difference does not establish equivalence.

## Evaluation (locked before any new test result; T12–T17)

**Plan as written on 20–21/10, before the lock:**
- test split of `split-manifest.json`, 500 utterances;
- synthesis seed fixed per utterance;
- WER macro and corpus, UTMOSv2 as a predicted score;
- waveform F0 metrics (RMSE, correlation on voiced frames, voicing error), as Review 1 asks;
- the pitch-shift test −2/0/+2 semitones;
- RTF with declared threads and batch.

**What was locked and run instead.**
- **The lock of 21/10.** The author locked the evaluation as the historical one, with no new metric (`eval-protocol.md`). It has:
  - WER, as the mean and median of the per-utterance WER (no corpus WER);
  - UTMOSv2;
  - MCD;
  - the F0-predictor metric. This is a phoneme-level log-F0 RMSE and correlation of the predictor, not a waveform F0 metric.
- **The other three items were done later, as supplementary analyses** outside the lock. Each protocol was agreed with the author before its run (`ablation8-results.md`):
  - waveform F0 against the recordings (T13, section 11). It measured RMSE and offset in cents, the log-F0 correlation, VDE, GPE and FFE, with MFCC-DTW alignment;
  - the pitch-shift test (T14, section 10);
  - the CPU RTF benchmark with declared threads and batch 1 (T15, section 5; `ablation-final.md`, Measurement conditions).
- **Which F0 numbers are which:** `ablation-final.md`, section "F0: predictor log-F0 and waveform pitch".
