# Preflight: camera-ready workspace (2026-10-19)

**Scope:** SOICT 2026 paper 4478, PitchFlowNet, camera-ready experiments only.
- This file records what exists on the PC.
- **Nothing was trained** on this machine for the camera-ready work, and no GPU was used.
- Paths are written with placeholders: `<WSL_DEV_HOME>` is the home directory of the WSL account that holds the historical runs and data.

## Repository and workspace

| Item | Value |
|---|---|
| Repository | Pitch-FlowNet (evidence store), branch `camera-ready-20261019` from `main` at `9971b133786f6d0639e616aa4a1b5ec40d8fcfed` |
| Workspace folder | `camera-ready-20261019/` in this repository |
| Evidence-store folders | `code/`, `configs/`, `runs/`, `datasets/`, `evaluation/`, `history/`: unchanged |
| Source of the workspace | Byte-exact copy of `code/historical-candidates/edgetts-badf381/src/python` |
| Source copy check | 59 files, all equal to `manifests/file-manifest-core.json` (`source-provenance-pre.json`) |
| Source changes | Listed in `source-provenance-post.json` and in the Git history of this branch |
| Not copied | `extract_test_sentences.py` (machine paths; not used for training), the Cython output `core.c`, binaries |

The copied source is a **historical source candidate**, not the source that ran.

## Machine (read-only checks, 2026-10-19 15:00 +07)

| Item | Value |
|---|---|
| GPUs | 2 × NVIDIA GeForce RTX 4070 Ti, 12,282 MiB each; UUIDs `GPU-e512cdeb-2918-b5e3-e7b1-91fd010996a6`, `GPU-352ef33d-d1ff-e02d-cd44-df3f41332fca` |
| GPU use | Both busy with another project's training (Piper-SeqIR, DDP, about 10 GB each, epoch 974 of 1,500 at 15:19). An automatic evaluation and timing session follows it. **GPU_BLOCKED for this workspace** |
| CPU / RAM | Intel Core i7-14700 (24 logical CPUs in WSL); 62 GB RAM, about 52 GB available |
| Disk | WSL ext4: 593 GB free inside. **The WSL virtual disk lives on Windows C:, which has about 53 GB free; this is the real limit for new outputs** |
| Environment | conda `edgetts`, Python 3.10.20, torch 2.12.1+cu130 (CUDA 13.0), pytorch-lightning 2.6.5 (the version recorded in the historical checkpoints), numpy 2.2.6. Not modified. `pytest` is absent, so the tests use `unittest` |

## Data (read-only; SHA-256)

| Item | Location | Value |
|---|---|---|
| Processed dataset | `<WSL_DEV_HOME>/data_preprocessed/dataset.jsonl` | SHA-256 `3b93b9883e1ac98d951049c3dbd5e5130c06985baa1b70c60c75e431b29c7bdf`. 13,100 rows; longest phoneme sequence 399 (so `max_phoneme_ids` 400 drops nothing) |
| Dataset config | `<WSL_DEV_HOME>/data_preprocessed/config.json` | SHA-256 `6eeb898264813993e00f4a85d0a4fd44210bedfa64d976cf531584baa0993de7`. 22,050 Hz, espeak `en-us`, 256 symbols (phoneme map inside) |
| Feature cache | `<WSL_DEV_HOME>/data_preprocessed/cache/22050/` | About 22 GB (normalised audio, linear spectrograms, F0 tensors). Per-file hashes are in the evidence record's local `manifests/file-manifest.json` |
| Raw LJ Speech WAVs | `<WSL_DEV_HOME>/data/wavs/` | 13,100 files, about 3.6 GB. Per-file SHA-256 in `split-manifest.json` |
| Stored test entries | `<WSL_DEV_HOME>/test_entries_500.jsonl` | SHA-256 `8c97aa2b2f39d49240acaf6ef7f6e1b01f27007089b9bc26d74c61bcf1314029` |

**Stable IDs:** the stable utterance ID is the LJ Speech ID, i.e. the file stem of `audio_path`.

**`split-manifest.json`** fixes the following before any training:
- ordered IDs, rows and raw-WAV SHA-256 for train 12,500 / validation 100 / test 500;
- **IDs and raw-audio hashes are disjoint**;
- the test split equals the stored test entries in order;
- 3 transcripts occur in both test and train as different recordings; they are listed, not treated as duplicate samples.

It is a reconstruction of the historical split (`random_split`, seed 1234), not a run-time record.

## Historical checkpoints

| Run | File | Bytes | SHA-256 | Safe load | Strict state dict |
|---|---|---:|---|---|---|
| Baseline, epoch 773 | `<WSL_DEV_HOME>/02_Baseline_VanillaVITS/lightning_logs/version_0/checkpoints/best-epoch=773-val_loss_mel=20.2158.ckpt` | 845,967,744 | `54d867192b673c5f493ee937db06ff3bba6a1c0192969131407e3a795245aac1` | Yes | Yes, `group_semantics="legacy"` |
| Full, epoch 1079 | `<WSL_DEV_HOME>/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt` | 874,730,405 | `8e6eacface1105c329ada0ccbec4c7a5cce88e86ac08797092ab30c1698a4edf` | Yes | Yes; the strict H-new architecture has exactly these parameters |

**How the checkpoints load safely:**
- They load with `torch.load(weights_only=True)` and one allowlisted type, `pathlib.PosixPath` (`piper_train/camera_ready/ckpt_io.py`).
- A static pickletools scan of both `data.pkl` files found only these globals: `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch.FloatStorage`, `torch.IntStorage` and `pathlib.PosixPath`.
- Without the allowlist the load is refused (tested).

**Baseline under strict B=0:** the strict B-new architecture drops exactly the 36 ResBlock Snake tensors. So B-new is a new configuration, not the historical baseline.

## Follow-up package

`conference-followup-evidence-2026-10-12` is no longer on this PC. Its expected `SHA256SUMS` hash `96d2e412…` was therefore **not verified here**.

## Behaviours found in the historical candidate source

These come from code reading and are not run-time proof.

| Topic | Finding | Consequence for the camera-ready protocol |
|---|---|---|
| Training data order | The training DataLoader has no shuffle. Under DDP, Lightning 2.6.5 injects `DistributedSampler(shuffle=False)`, so every epoch probably used the same order | `train_shuffle` is explicit (default `false`, the historical-equivalent order); a decision is open |
| MAS noise decay | `0.01 - global_step * 2e-6`, with Lightning `global_step` counting 2 optimizer steps per batch | Expressed per generator update (`4e-6` per G update = the same schedule) |
| Validation metric | Mean of per-batch means on each rank's shard, random crop per call, MAS noise active, not synchronised across ranks; the best-checkpoint decision needs all ranks to agree | Per-utterance, seeded, no MAS noise, exact sum/count over ranks, identical on every rank |
| Preview audio | The first five **test** utterances are synthesized every validation | Five fixed **validation** IDs; the test split is never loaded in training |
| Resume | A new log directory reset the best-checkpoint state (`dirpath has changed`) | Fixed checkpoint directory and logger version per run |

## Tests run on this machine (CPU only, 2026-10-19)

| Test | Result |
|---|---|
| `test_gating` (8 B/V/F combinations, legacy semantics, F0 loss, decoder without Snake) | pass |
| `test_split` (sizes, disjoint IDs and raw hashes, stored test order, tamper detection) | pass |
| `test_distributed` (sharding without padding; gloo 2-process sum/count) | pass |
| `test_historical_ckpt` (strict loads; H-new = full model parameters; allowlist needed) | pass |
| `test_train_cpu`: budget/counters, directory guard, diagnostics | pass |
| `test_train_cpu`: resume equality, ledger best file | **fail**: CPU training is not bit-deterministic. Two fresh 1-epoch runs differ by about 1e-8 in 455 of 994 tensors, so exact equality cannot hold; see PLAN.md |
| `test_train_cpu`: DDP validation equal to one process | **error**: `ddp_spawn` cannot pickle the weight-norm tensors; needs a subprocess DDP check (PLAN.md) |
