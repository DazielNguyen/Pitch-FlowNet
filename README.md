# Pitch-FlowNet

Evidence and source record for the paper **"PitchFlowNet: An Empirical Study of Architectural
Integration in Single-Stage Text-to-Speech"**. According to the author's receipt, it was submitted to
SOICT 2026 as submission 4478. This repository does not establish acceptance or publication.

The paper compares two runs: a retrained Piper baseline and the full PitchFlowNet model. For both,
the repository catalogues:
- the source candidates;
- the launch configurations;
- training records;
- stored evaluation outputs;
- provenance, with each item classed by kind of evidence.

**It is a record of what exists. It is not a verified reproduction package.**
- Nothing was trained, synthesized, scored or re-processed to build it.
- No clean-environment setup or re-run has been attempted.

| Start here | |
|---|---|
| [docs/project-identity.md](docs/project-identity.md) | Which runs, how they were identified, and what is not this paper |
| [evidence/manuscript-crosswalk.md](evidence/manuscript-crosswalk.md) | Each manuscript statement checked against stored records |
| [docs/missing-information.md](docs/missing-information.md) | Records that do not exist or were not found |
| [docs/artifacts.md](docs/artifacts.md) | Data, checkpoints and logs kept outside Git, and how to point code at them |
| [docs/setup-status.md](docs/setup-status.md) | What would be needed to run anything again |

## Relation to EdgeTTS, Piper and BanhmiTTS

- **EdgeTTS.** The model code was developed in the EdgeTTS repository.
  - EdgeTTS is a fork of [rhasspy/piper](https://github.com/rhasspy/piper); its remote is `MardeusVN/PitchFlowNet`.
  - The project was renamed to PitchFlowNet in EdgeTTS commit `6d0522b` (2026-09-24).
  - Snapshots of it are under `code/`. This repository does not contain EdgeTTS's Git history.
- **BanhmiTTS.** BanhmiTTS is a separate project.
  - "Banhmi-TTS" was an earlier name of the full model in drafts.
  - `code/current-source/home-dev-BanhmiTTS-port` is a later (2026-07-14) port of the full configuration into the BanhmiTTS framework. It is **not** one of the evaluated runs.
  - BanhmiTTS's `.git` was not copied.
- **Third-party code.** It keeps its notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## The two runs

| | Baseline ("Piper baseline") | Full model ("PitchFlowNet") |
|---|---|---|
| Run directory, session | `02_Baseline_VanillaVITS`, `version_0` | `01_Baseline_BigVgan_VITS2_FO`, `version_2` |
| Launch hparams | [configs/baseline/](configs/baseline/) | [configs/full/](configs/full/) |
| Component flags (hparams) | `use_bigvgan`, `use_vits2`, `use_f0` false | all three true |
| History | From scratch, 2026-07-09 → 07-13; TensorBoard span 77.26 h | Resumed: a from-scratch session ended in `RuntimeError`, then two resumes from `version_0/last.ckpt`. `version_0`/`version_1` no longer exist. `version_2` events span 99.47 h (epochs 43–1099) |
| Selected checkpoint | `best-epoch=773-val_loss_mel=20.2158.ckpt` (epoch 773, step 605,268) | `best-epoch=1079-val_loss_mel=19.2943.ckpt` (epoch 1079, step 844,560) |
| Shared recipe (hparams) | 2 GPUs, DDP, bf16-mixed, batch 16 per GPU, seed 1234, 1,100 epochs, grad clip 1.0, AdamW 2e-4 | same |

**Identification evidence** (details in [docs/project-identity.md](docs/project-identity.md)):
- the hparams written at launch;
- the training logs;
- TensorBoard values at the selected epochs;
- the evaluation summary that names both checkpoints;
- a static scan of the checkpoints (pickle opcodes only, no unpickling).

**Two points are unresolved:**
- Neither selected checkpoint is the minimum of its logged rank-0 `val_loss_mel` series.
- The paper's 103.31 h for the full model cannot be reproduced from the remaining events.

**Not this paper:**
- the journal's Parallel-IR / Sequential-IR models (epochs 1386 / 1442);
- the HiFi-MobiNet baseline (epoch 1489);
- the 2026-10 `Piper_no_VITS2_cpn` run;
- the published Piper checkpoint.

## Full-model components and what checks them

| Component | Record | Status |
|---|---|---|
| HiFi-GAN ResBlock2, upsampling 8/8/4 | `resblock '2'`, upsample rates 8/8/4 in both hparams | Recorded at launch |
| Snake activations (named `Snake1d` in the code at the time; renamed `SnakeBeta` later) | 44 Snake parameters in the full model's decoder | Present in the checkpoint |
| Multi-resolution discriminator | 82 `model_d_mrd.*` parameter names | Present; the number of resolutions was not re-derived |
| Self-attention in flow coupling | 144 flow-attention parameter names | Present in the checkpoint |
| Duration discriminator | 33 `model_d_dur.*` parameter names | Present in the checkpoint |
| Noised MAS | `mas_noise_scale_decay` 2e-6 in hparams; the switch exists from commit `badf381` | Formula not verified in the source that ran |
| Phoneme-level F0 predictor and `f0_cond` | 17 `F0Predictor` and 3 `f0_cond` parameter names | Present; layer sizes not re-derived |

**What these checks show, and what they do not:**
- "Present in the checkpoint" means that the parameter names exist in the saved state. It does not verify run-time behaviour.
- **The baseline differs from the manuscript's description:**
  - its checkpoint holds 36 Snake parameters inside the decoder ResBlocks;
  - the candidate code always builds them there.

## Stored results (historical)

These values come from `evaluation/eval_h_vs_baseline/summary.sanitized.json`. A recomputation from the
per-sentence CSVs (`tools/verify_pitchflownet.py`) gives the same means.
- They are stored results.
- They were **not** re-run, and they have not been audited independently on the conference side.
- UTMOSv2 is a predicted quality score, not a listening test.

| 500 test sentences | Baseline (epoch 773) | Full model (epoch 1079) | Paired Wilcoxon p |
|---|---:|---:|---:|
| WER | 9.171% | 8.638% | 0.1146 |
| UTMOSv2 | 3.3656 | 3.8070 | 1.65e-79 |
| RTF | 0.0559 | 0.0568 | — |

**Ground truth:** WER 6.67%, UTMOSv2 3.830 (`evaluation/ground-truth/`).

**Other notes on the results:**
- Other evaluations of the same checkpoints give different values. They are kept under `evaluation/other/` and `evaluation/full/`, and are not substituted.
- The WER p-value does not establish that the two models are equivalent.
- No record of the BCa intervals or Cohen's dz in the paper was found.

## Repository layout

| Path | Tracked content | Kept local (git-ignored) |
|---|---|---|
| `code/historical-candidates/` | EdgeTTS at `6033be5` and `badf381`; launch scripts at four commits | — |
| `code/current-source/` | EdgeTTS `HEAD` `a73a897`, the WSL training directory as it is now, the BanhmiTTS port | Compiled extensions, build folders, Silero VAD ONNX |
| `configs/` | Launch hparams of both runs and of exploratory runs | Originals (machine paths) |
| `runs/` | Training logs (sanitized) | Checkpoints, TensorBoard events, originals, one 11 MB port log |
| `datasets/splits/` | Stored test entries and sentences; reconstructed split | — |
| `datasets/raw/`, `datasets/processed/` | — | LJ Speech as used; processed cache (22 GB) |
| `evaluation/` | Per-sentence CSVs, summaries, logs, figures, figure and statistics scripts | Generated WAVs |
| `evidence/` | Crosswalk; derived checkpoint and TensorBoard scans | Manuscript drafts |
| `history/` | Commit logs, working-tree status, environment lists (2026-10-12), 2026-09-23 audits | Git bundle, author notes, a manuscript draft |
| `manifests/` | Run identity, source map, dataset and checkpoint manifests, `file-manifest-core.json`, sanitization maps, check reports | Full `file-manifest.json` (34 MB), `SHA256SUMS` |
| `tools/`, `tests/` | Read-only build, scan, verify, sanitize and check scripts; tests | — |

**Evidence classes used in the manifests:**
- `CONTEMPORANEOUS_RECORD`
- `DIRECT_ARTIFACT`
- `HISTORICAL_SOURCE_CANDIDATE`
- `CURRENT_SOURCE`
- `RECONSTRUCTED`
- `AUTHOR_ACCOUNT`
- `UNKNOWN`

**Source as run versus source now.** No source snapshot was saved when either run started.
- **What ran:** the launch scripts synced an uncommitted Windows working tree into WSL.
- **The candidates:**
  - `6033be5` predates both runs but lacks the VITS2 / noised-MAS switches that the full run used;
  - `badf381` has them but was committed after both launches.
- **What the repository therefore holds:** candidates and current source, not run-time source.

## Sanitized copies and placeholders

Some tracked files are derived from originals that contain machine paths or the workstation's host name.
- Each such original stays local and unchanged.
- A copy named `<name>.sanitized.<ext>` is tracked beside it.
- `manifests/sanitization/<folder>.json` lists both SHA-256 values and the number of replacements.
- `tools/sanitize_for_git.py` regenerates the copies.

| Placeholder | Stands for |
|---|---|
| `<WSL_DEV_HOME>` | Home directory of the WSL account that ran training and evaluation |
| `<WSL_PROJECT_HOME>` | Home directory of a second WSL account on the workstation |
| `<WINDOWS_WORKSPACE>` | The Windows project workspace (`FPT-Graduation-Project`) |
| `<WINDOWS_USER_HOME>` | The Windows user profile |
| `<WINDOWS_WORKSPACE_SLUG>`, `<WINDOWS_USER_SLUG>` | The same paths flattened into one directory name |
| `<WORKSTATION>` | Host name of the training workstation (it appears in TensorBoard event file names) |

The project documents (`docs/`, `evidence/manuscript-crosswalk.md`, `manifests/*.json`) use the same placeholders.

## Environment (as recorded)

Nothing below was installed or tested for this repository.
- **The conda environment `edgetts`, captured on 2026-10-12** ([history/environment/](history/environment/)):
  - Python 3.10.20;
  - `torch` 2.12.1 (+cu130 build);
  - `pytorch-lightning` 2.6.5;
  - `numpy` 2.2.6;
  - `pyworld` 0.3.5;
  - `piper-phonemize` 1.1.0;
  - `openai-whisper` 20250625;
  - `jiwer` 4.0.0;
  - UTMOSv2 from Git commit `cc2700db`.

  This is the environment as it is now, **not a training-time record**.
- **Checkpoints:** they record Lightning 2.6.5.
- **Workstation on the same date:** WSL2, 2 × RTX 4070 Ti, Intel Core i7-14700. The run records themselves do not name the GPUs.
- **Requirements files:** they are in the source snapshots (`src/python/requirements*.txt`).
- **Cython build:** the extension must be built with `build_monotonic_align.sh`.

## Commands as recorded

These are copies of what the record contains. They are **not** verified as the exact commands of the two runs.

The baseline launch script at `badf381`
([.wsl_baseline.sanitized.sh](code/historical-candidates/launch-scripts/badf381/.wsl_baseline.sanitized.sh)):

```sh
cd <WSL_DEV_HOME>/EdgeTTS/src/python
python -m piper_train \
    --dataset-dir <WSL_DEV_HOME>/data_preprocessed \
    --default_root_dir <WSL_DEV_HOME>/02_Baseline_VanillaVITS \
    --accelerator gpu --devices 2 --strategy ddp_find_unused_parameters_true \
    --precision bf16-mixed --batch-size 16 --num-val-examples 100 --num-test-examples 500 \
    --max_epochs 1100 --checkpoint-epochs 1 --seed 1234 \
    --use-bigvgan false --use-vits2 false --use-f0 false --grad-clip 1.0
```

**The full-model script** ([.wsl_train.sanitized.sh](code/historical-candidates/launch-scripts/badf381/.wsl_train.sanitized.sh)):
- It uses `--use-bigvgan true --use-vits2 true --use-f0 true` with the same settings.
- Its committed `--resume_from_checkpoint` points to `version_1/last.ckpt`, whereas the `version_2` hparams name `version_0/last.ckpt`.

**Evaluation:**
- The recorded evaluation calls use `python -m piper_train.eval_harness` (`runs/candidates/home-dev-root/run_*.sanitized.sh`).
- The command, ASR model, device and thread settings of `eval_h_vs_baseline` itself were not recorded.

## Checks

| Check | Where it runs | Command |
|---|---|---|
| Tracked package: no artifacts, private folders, machine paths, personal e-mail addresses or tokens; sanitized copies match their maps; recorded files keep their hashes; JSON/YAML parse; Markdown links resolve | Workstation or clean clone | `python tools/check_tracked_package.py` |
| Unit tests for the sanitizing rules, plus the check above | Workstation or clean clone | `python -m unittest discover -s tests -v` |
| Full record: copy hashes, dataset rows and files, split reconstruction, CSV/summary alignment | Workstation only (needs the local data) | `python tools/verify_pitchflownet.py --output <report.json>` |

The latest reports are in `manifests/tracked-package-check.json` and `manifests/verification-report.json`.

**Cloning on Windows:**
- The longest tracked path is 122 characters.
- Under a long base folder, it can pass the 260-character limit.
- Clone with `git -c core.longpaths=true clone …` into a short folder.
- Python can read such paths only when Windows long-path support is enabled.
- The package check reports any tracked file it cannot read.

**Not checked:**
- installing the environment;
- importing or running the model code;
- loading checkpoints;
- re-running training or evaluation;
- reproducing any number.

## Licensing

| Material | Terms |
|---|---|
| Piper-derived code in `code/` | MIT, as in each snapshot's `LICENSE.md`. Further third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |
| The project's own changes, the BanhmiTTS port, tools, documents and manifests | **Not decided.** No licence is granted by this repository until the authors choose one |
| LJ Speech data and transcripts (`datasets/splits/`) | Public domain per the dataset; data files are not tracked |
| Checkpoints and generated audio | Not distributed here; terms not decided |
| Manuscript | Not included |

**No top-level `LICENSE` file has been added.**

## Status

- Last updated: 2026-10-12.
- Nothing has been pushed or published.
- The decision to push, the remote's visibility and the project licence belong to the authors.
