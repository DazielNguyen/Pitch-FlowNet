# Experiment plan: camera-ready ablation (SOICT 2026, paper 4478)

**Status:** decided by the author on 2026-10-20 and extended to all eight group combinations on 2026-10-21. Both decisions came before any test-set result.

**Two parts.**
- **The paper's main result stays the historical pair:** Baseline epoch 773 and Full epoch 1079. It is evaluated as it is; nothing is retrained to replace it.
- **New: an ablation over the three groups.** Leave-one-out first, then add-one, up to all 2^3 = 8 combinations (extended on 21/10). It answers the shared main criticism of the three reviews: "only baseline and full are evaluated; it is impossible to determine which component is responsible".

## Decisions (O5 / T08)

| | Question | Decision |
|---|---|---|
| (a) | `train_shuffle` | `false`, the historical data order |
| (b) | MAS noise decay | `4e-6` per G update, the historical schedule. The noise reaches 0 after 2,500 G updates, about 3.2 epochs |
| (c) | Budget and devices | **275 epochs per run** = 275 × 782 batches = **215,050 G updates** at batch 16 on one GPU |
| (d) | Replication seeds | not planned (no time); single seed 1234, reported as a limitation |
| (e) | Checkpoint for evaluation | the minimum `val_mel_l1_global` on the 100 validation utterances (`best-*.ckpt`), fixed before any test result. The test split is never used for selection |

**What 275 epochs means against the historical runs:**
- **Epochs and samples seen:** 1/4. The historical runs did 1,100 epochs of 12,500 utterances.
- **G updates:** 1/2. The historical runs did 430,100, at batch 2 × 16.

**Why the historical Full model is not the reference for the ablation.**
- It differs from any run that fits before 27/10 in budget (1,100 epochs), devices and batch (2 GPUs, 32), and source: the historical source is a candidate, not proven to be what ran.
- A difference to it would mostly measure training budget, not the removed group.
- So the ablation has its own reference, `PFN-Full-275ep`, trained like the ablated runs.

## Run matrix: all 2^3 = 8 combinations (extended on 21/10)

All runs use the same recipe:
- train seed 1234, data seed 1234, validation seed 2026;
- strict group gating, batch 16, one GPU, bf16-mixed;
- 275 epochs.

The eight configs differ only in the group flags (`use_bigvgan`, `use_vits2`, `use_f0`), the run ID and the notes. This was checked by a script; every config passes `train.py --dry-run`. `tests/camera_ready/test_gating.py` builds all eight combinations.

| Priority | Run ID | Config | B | V | F | Question it answers |
|---:|---|---|:-:|:-:|:-:|---|
| 1 | `PFN-Full-275ep-s1234` | `configs/PFN-Full/PFN-Full-275ep.json` | on | on | on | reference |
| 2 | `PFN-NoF0-275ep-s1234` | `configs/PFN-NoF0/PFN-NoF0-275ep.json` | on | on | off | leave-one-out: F0 |
| 2 | `PFN-NoBigVGAN-275ep-s1234` | `configs/PFN-NoBigVGAN/PFN-NoBigVGAN-275ep.json` | off | on | on | leave-one-out: BigVGAN |
| 2 | `PFN-NoVITS2-275ep-s1234` | `configs/PFN-NoVITS2/PFN-NoVITS2-275ep.json` | on | off | on | leave-one-out: VITS2 |
| 3 | `PFN-Baseline-BigVGAN-275ep-s1234` | `configs/PFN-Baseline-BigVGAN/PFN-Baseline-BigVGAN-275ep.json` | on | off | off | add-one: BigVGAN only (Review 1, "Piper with only a BigVGAN-style decoder") |
| 3 | `PFN-Baseline-275ep-s1234` | `configs/PFN-Baseline/PFN-Baseline-275ep.json` | off | off | off | add-one reference; new strict baseline, **not** the historical epoch-773 baseline |
| 4 | `PFN-Baseline-VITS2-275ep-s1234` | `configs/PFN-Baseline-VITS2/PFN-Baseline-VITS2-275ep.json` | off | on | off | add-one: VITS2 only (Review 1, "VITS2 itself") |
| 5 | `PFN-Baseline-F0-275ep-s1234` | `configs/PFN-Baseline-F0/PFN-Baseline-F0-275ep.json` | off | off | on | add-one: F0 only |

**What each set answers:**
- **Leave-one-out (priority 1–2):** Full against each one-group-off run. How much each group contributes when the other two are present.
- **Add-one (priority 3–5):** Baseline against each one-group-on run. How much each group gives on its own.
- **Interactions:** with all eight, the main effects and interactions of the factorial design can be estimated. One seed per cell makes interaction estimates noisy, and the paper says so.

## Devices and schedule

**Speed so far:**
- The RTX 5070 runs about 9.5 min per epoch (`PFN-Full-275ep`, 20–21/10), against 13.2 min in the pilot (`eta.md`). So a run takes about 44 h, plus interruptions.
- The RTX 4070 Ti speed is measured in the first epoch of its runs.

| GPU | Runs, in order | Estimated end |
|---|---|---|
| MardeusPC, RTX 5070 | `PFN-Full` → `PFN-NoF0` → `PFN-Baseline-VITS2` | ≈ 21/10 night → ≈ 23/10 evening → ≈ 25/10 afternoon |
| PC, RTX 4070 Ti #0 | `PFN-NoBigVGAN` → `PFN-Baseline-BigVGAN` | ≈ 23/10 → ≈ 25/10, if it starts ≈ 21/10 midday |
| PC, RTX 4070 Ti #1 | `PFN-NoVITS2` → `PFN-Baseline` | same |
| first GPU free | `PFN-Baseline-F0` | only if it can end before ≈ 26/10 evening |

**Capacity.**
- Three GPUs fit about seven 44-h runs before the evaluation window (26–27/10).
- `PFN-Baseline-F0` is last and may not fit. If it does not, the paper reports seven of the eight cells and says which one is missing.
- Never shorten a run to fit; every run keeps 275 epochs.

**GPU types.**
- `Full` and `NoF0`, the central F0 contrast, share the RTX 5070.
- The two add-one pairs on the PC (`Baseline` and `Baseline-BigVGAN`) share the RTX 4070 Ti type.
- Every other contrast crosses GPU types. Configs, batch and update count are the same; the paper states the GPU type of each run.

**Constraints:**
- Each run uses one GPU and runs in its own tmux session. One run per GPU.
- On the PC, a dedicated environment is used. Nothing of the other project is touched, and nothing starts before its GPUs are free.
- Every launch needs the author's OK of the exact command. The watchdog only resumes after the known CUDA error and never starts a new run (`runbook-MardeusPC.md`).

## Estimand and limits

**Estimand.** The effect on the evaluation metrics of each group (B, V, F), at a fixed 275-epoch budget with one seed. It is measured two ways: by turning the group off in the full configuration (leave-one-out), and by turning it on in the baseline (add-one).
- **Not estimated:** the effect at the historical 1,100-epoch budget.
- **Interactions:** estimable only if all eight cells finish, and with one seed per cell they are imprecise.

**Limits to state in the paper:**
- **Training:** one seed per run; a shortened budget, 1/4 of the historical epochs.
- **Hardware:** two GPU types.
- **Data and quality metrics:** single-speaker English. UTMOSv2 is a predicted score, not MOS. A non-significant difference does not establish equivalence.

## Evaluation (locked before any new test result; T12–T17)

Same protocol for the four 275-epoch runs and for the historical pair:
- test split of `split-manifest.json`, 500 utterances;
- synthesis seed fixed per utterance;
- WER macro and corpus, UTMOSv2 as a predicted score;
- waveform F0 metrics (RMSE, correlation on voiced frames, voicing error), as Review 1 asks;
- the pitch-shift test −2/0/+2 semitones;
- RTF with declared threads and batch.

The details are fixed in `launch_eval.sh` before the first test-set run.
