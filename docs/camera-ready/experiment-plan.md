# Experiment plan: camera-ready ablation (SOICT 2026, paper 4478)

**Status:** decided by the author on 2026-10-20, before any new run and before any test-set result.

**Two parts.**
- **The paper's main result stays the historical pair:** Baseline epoch 773 and Full epoch 1079. It is evaluated as it is; nothing is retrained to replace it.
- **New: a leave-one-group-out ablation.** It answers the shared main criticism of the three reviews: "only baseline and full are evaluated; it is impossible to determine which component is responsible".

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

## Run matrix

All runs use the same recipe, except the one group that is turned off:
- train seed 1234, data seed 1234, validation seed 2026;
- strict group gating, batch 16, one GPU, bf16-mixed;
- 275 epochs.

| Run ID | Config | BigVGAN (B) | VITS2 (V) | F0 (F) | Role |
|---|---|:-:|:-:|:-:|---|
| `PFN-Full-275ep-s1234` | `configs/PFN-Full/PFN-Full-275ep.json` | on | on | on | reference |
| `PFN-NoF0-275ep-s1234` | `configs/PFN-NoF0/PFN-NoF0-275ep.json` | on | on | **off** | F0 group |
| `PFN-NoBigVGAN-275ep-s1234` | `configs/PFN-NoBigVGAN/PFN-NoBigVGAN-275ep.json` | **off** | on | on | BigVGAN group |
| `PFN-NoVITS2-275ep-s1234` | `configs/PFN-NoVITS2/PFN-NoVITS2-275ep.json` | on | **off** | on | VITS2 group |

**Optional, if GPU time is left:** `PFN-Baseline-BigVGAN`, "Piper with only a BigVGAN-style decoder", which Review 1 asks for. It gets the same budget and a new 275-epoch config.

## Devices and schedule

Estimates use the measured 13.2 min per epoch on the RTX 5070 (`eta.md`). The RTX 4070 Ti speed is measured in the first epoch of its runs.

| GPU | Runs, in order | Estimated span |
|---|---|---|
| MardeusPC, RTX 5070 | `PFN-Full-275ep`, then `PFN-NoF0-275ep` | 20/10 ≈ 02:45 → 22/10 ≈ 15:15; then → 25/10 ≈ 03:45 |
| PC (author's other machine), RTX 4070 Ti #0 | `PFN-NoBigVGAN-275ep` | from ≈ 20/10 14:30, after the other project frees the GPUs; ≈ 60 h |
| PC, RTX 4070 Ti #1 | `PFN-NoVITS2-275ep` | same |

**Why Full and NoF0 share a GPU:** they form the central comparison (F0, the paper's focus), so they run on the same GPU type. The other two ablations run on RTX 4070 Ti GPUs. Same configs, batch and update count; the GPU type differs, and the paper says so.

**Constraints:**
- Each run uses one GPU and runs in its own tmux session. One run per GPU.
- On the PC, a dedicated environment is used. Nothing of the other project is touched, and nothing starts before its GPUs are free.

**If the PC cannot start by 22/10 evening:** drop `PFN-NoVITS2` rather than lower the budget. Every run keeps the same 275 epochs.

## Estimand and limits

**Estimand.** The effect on the evaluation metrics of turning off one group (B, V or F) of the full configuration, at a fixed 275-epoch budget, with one seed.
- **Not estimated:** the effect at the historical 1,100-epoch budget.
- **Interactions:** only the Full-vs-one-group-off contrasts are measured, not every combination.

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
