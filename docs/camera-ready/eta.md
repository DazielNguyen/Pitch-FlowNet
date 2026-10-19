# ETA and budget: single-GPU queue on MardeusPC (T07)

**Basis:** pilot `PFN-Full-Pilot` on 2026-10-20, 01:32–01:49 +07.
- Records: `run_manifest.json`, `checkpoint_ledger.jsonl` and `val_diagnostics.jsonl` of the pilot, the two launch logs, and the GPU log (`nvidia-smi` every 5 s). All are outside Git.
- **The pilot is not a result and is never a warm start.**

## Pilot

| Item | Value |
|---|---|
| Config | `configs/PFN-Full/PFN-Full-Pilot.json`, commit `9e3d7ee`, clean tree |
| Attempt 1 | `--limit-train-batches 100 --stop-after-epochs 1`: 100 G updates in 3 min 30 s, including warm-up |
| Attempt 2 | `--resume`. A launcher defect (fixed afterwards in `resume_train.sh`) did not pass `--limit-train-batches` on, so epoch 1 ran all 782 batches. The pilot ended at 882 G updates instead of 200 |
| Resume on the GPU | Counters continued (100 → 882 G updates; samples 1,600 → 14,100). `last.ckpt` passed `check_ckpt.py`. The best-checkpoint state survived: best of epoch 0 was replaced by epoch 1. Bit equality is not expected on the GPU (cuDNN autotuning) |
| Validation metric | `val_mel_l1_global` 52.08 (100 updates) → 36.59 (882 updates); finite |

## Measurements

| Quantity | Value |
|---|---|
| Steady training speed | 782 batches in 12 min 05 s = **0.93 s per G update** (batch 16) |
| Validation, 100 utterances | ≈ 55 s per epoch |
| Checkpoint save | ≈ 2 s per file, 875 MB each; 2–3 files per epoch |
| **One epoch (782 G updates)** | **≈ 13.2 min ≈ 3,560 G updates per hour** |
| Peak GPU memory | 9,989 MiB of 12,227 MiB, display included: ≈ 2.2 GB headroom at batch 16 |
| Mean GPU utilisation | 37 %: likely limited by data loading or CPU. Not tuned; the configuration is kept as instructed |
| Disk per run | ≈ 2.6 GB of checkpoints (best, rolling epoch, last) |

`PFN-NoF0` has not been measured. It lacks the F0 predictor and F0 conditioning, so it is probably slightly faster; its first epoch will show this.

## Time available

| Step | Hours |
|---|---:|
| 20/10 02:00 to 27/10 20:00 | 186 |
| Evaluation of the T1 checkpoints (estimate until T12 measures it) | − 8 |
| Reserve, 25–30 % | − 45 to − 54 |
| **GPU time for the queue** | **≈ 124–133** |

## Budget options

The budget is a whole number of epochs: `max_g_updates = 782 × N`. The budget callback stops at the end of the epoch that reaches it.

| Option | Runs | Epochs per run | G updates per run | Hours per run | Pair finishes about | Samples seen per run, vs historical 13.76 M |
|---|---|---:|---:|---:|---|---:|
| **A** | T1 pair only (`PFN-Full`, `PFN-NoF0`) | 273 | 213,486 | ≈ 60 | 25/10 morning | 3.42 M ≈ 25 % |
| B | T1 pair + T2 pair (`PFN-NoVITS2`, `PFN-NoBigVGAN`) | 137 | 107,134 | ≈ 30 | T1 22/10 evening; all 25/10 | 1.71 M ≈ 12 % |
| C | T1 pair first, then T2 if time is left | 200 | 156,400 | ≈ 44 | T1 23/10 morning; T2 not at full budget | 2.50 M ≈ 18 % |

**For every option:**
- **The historical budget is out of reach.** It was 430,100 G updates at batch 2 × 16, i.e. 13.76 M samples.
- **Label and comparison.** The runs are labelled "budget-matched", and compared only with each other, never with the 1,100-epoch historical models.
- **Configs.** Each option gets new configs (`configs/PFN-<variant>/PFN-<variant>-<N>ep.json`); the placeholder configs stay unchanged.
