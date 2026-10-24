# Runs on the PC (2 × RTX 4070 Ti)

**Scope.**
- This page records each training run on this machine: launches, interruptions, resumes and final checkpoints.
- The run directories are outside Git, in `<WSL_DEV_HOME>/outputs-pfn-cr/runs/<run_id>/`.
- Machine and environment: `preflight-Daziel.md`. Instructions: `PROMPT-daziel-ablation.md`. What to run: `experiment-plan.md`.
- Validation values below are `val_mel_l1_global` on the 100 validation utterances. They are training records, not evaluation results.

## Common setup

| Item | Value |
|---|---|
| Repository | `<WSL_DEV_HOME>/Pitch-FlowNet-cr`, branch `PFN-Daziel-22102026`, `HEAD` `ee6a378` |
| Environment | `<WSL_DEV_HOME>/miniconda3/envs/pfn-cr`; settings in `<WSL_DEV_HOME>/pfn-cr.env` (`CR_OUTPUTS=$HOME/outputs-pfn-cr`, `CR_TB_PORT=6007`) |
| WSL keep-alive | `wsl_keepalive.ps1 start -Distro Ubuntu-24.04`, started 2026-10-22 01:33:02 +07 (PID 16932) |
| TensorBoard | tmux `cr-tensorboard`, http://localhost:6007, started 01:33:13 +07 |
| GPU log | tmux `cr-gpulog`, `nvidia-smi` every 5 s into `<WSL_DEV_HOME>/outputs-pfn-cr/logs/gpumon-20261021T184520Z.csv` (named `gpu-20261022.csv` until 22/10 20:25, renamed to follow the output layout), started 01:45:20 +07 |

**Commands run** (2026-10-22 01:33 +07, after the author's OK of these exact commands):

```sh
source ~/pfn-cr.env && cd ~/Pitch-FlowNet-cr
bash tools/camera_ready/tensorboard.sh start
bash tools/camera_ready/launch_train.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --gpu 0 --execute
bash tools/camera_ready/auto_resume.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --start --gpu 0
bash tools/camera_ready/launch_train.sh configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json --gpu 1 --execute
bash tools/camera_ready/auto_resume.sh configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json --start --gpu 1
```

**No GPU pilot on this machine.**
- The prompt goes from the dry run straight to the launch. The GPU pilot with interruption and resume ran on MardeusPC (T06, `eta.md`), with the same training source.
- Epoch 0 of each run served as the check of speed, VRAM, validation and checkpoint saving on the RTX 4070 Ti. Both passed (below).
- Resume has not been exercised on this machine's GPUs.

### Record defect at launch: `dirty_paths` is not empty

- `run_manifest.json` of both runs records `source.git.dirty_paths = ["?? docs/camera-ready/preflight-Daziel.md"]`, not `[]`.
- Cause: the preflight page had been written into the clone, uncommitted, before the launch.
- It is an untracked document outside `piper_train/`. `source.tree_sha256` = `1ee13cb4efe39da5657398021f3b85eb3a2a3eb5b89a1f1392dc33c4dfe90a52`, the same training source as `PFN-Full` on MardeusPC.
- The runs were not restarted. Later launches on this machine happen only from a committed, clean tree.

## PFN-NoBigVGAN-200ep-s1234 (GPU 0)

| Item | Value |
|---|---|
| Config | `configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json`; B off, V on, F on; 156,400 G updates |
| GPU | 0, RTX 4070 Ti, UUID `GPU-e512cdeb-2918-b5e3-e7b1-91fd010996a6`, no display |
| Start | 2026-10-22 01:33:15 +07 (attempt 1 at 18:33:18 UTC), tmux `cr-PFN-NoBigVGAN-200ep-s1234` |
| Launch log | `<WSL_DEV_HOME>/outputs-pfn-cr/logs/PFN-NoBigVGAN-200ep-s1234-20261021T183315Z.log` |
| Watchdog | tmux `cr-watch-PFN-NoBigVGAN-200ep-s1234`, at most 5 automatic resumes |
| `run_manifest.json` | `head` `ee6a378`, `max_g_updates` 156,400, `devices` 1, `CUDA_VISIBLE_DEVICES=0`, `tree_sha256` as above; `dirty_paths` see above |

**Speed and memory:**
- Epoch 0: start to `last.ckpt` 10.8 min (01:33:18 → 01:44:06), at about 0.84 it/s.
- Epoch 1: 6.6 min (01:44:06 → 01:50:42), at about 2.03 it/s, validation and checkpoint saving included.
- Epoch 0 was slower, probably because the feature cache was read from disk for the first time (not verified).
- Peak memory in the GPU log, epochs 0–1: 9,050 MiB of 12,282 MiB.
- `val_mel_l1_global`: epoch 0: 36.99; epoch 1: 33.62.

**Expected end** at 6.6 min per epoch: about 22 h, **≈ 2026-10-22 23:40 +07**, without interruptions.

| Interruptions and resumes | |
|---|---|
| none | one attempt, no automatic resume |

**End:**
- Training ended 2026-10-23 00:12:11 +07 with exit 0, after 200 epochs (22 h 39 min). The watchdog finalized the run at 00:13:24 and stopped.
- Over the whole run, epochs 1–199 took 6.77 min on average; peak memory in the GPU log was 9,633 MiB.
- Best checkpoint: epoch 199 (the last epoch), `val_mel_l1_global` 22.3429. `best-epoch=199…ckpt` and `last-epoch=199.ckpt` hold the same epoch but are two separate saves, with different hashes.

## PFN-NoVITS2-200ep-s1234 (GPU 1)

| Item | Value |
|---|---|
| Config | `configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json`; B on, V off, F on; 156,400 G updates |
| GPU | 1, RTX 4070 Ti, UUID `GPU-352ef33d-d1ff-e02d-cd44-df3f41332fca`; **drives the display** (about 1.45 GB) |
| Start | 2026-10-22 01:33:17 +07 (attempt 1 at 18:33:18 UTC), tmux `cr-PFN-NoVITS2-200ep-s1234` |
| Launch log | `<WSL_DEV_HOME>/outputs-pfn-cr/logs/PFN-NoVITS2-200ep-s1234-20261021T183317Z.log` |
| Watchdog | tmux `cr-watch-PFN-NoVITS2-200ep-s1234`, at most 5 automatic resumes |
| `run_manifest.json` | `head` `ee6a378`, `max_g_updates` 156,400, `devices` 1, `CUDA_VISIBLE_DEVICES=1`, `tree_sha256` as above; `dirty_paths` see above |

**Speed and memory:**
- Epoch 0: 10.3 min (01:33:18 → 01:43:38), at about 0.90 it/s.
- Epoch 1: 6.4 min (01:43:38 → 01:50:03), at about 2.12 it/s.
- Peak memory in the GPU log, epochs 0–1: 10,339 MiB of 12,282 MiB, display included.
- `val_mel_l1_global`: epoch 0: 37.99; epoch 1: 34.07.

**Expected end** at 6.4 min per epoch: about 21.6 h, **≈ 2026-10-22 23:10 +07**, without interruptions.

| Interruptions and resumes | |
|---|---|
| none | one attempt, no automatic resume |

**End:**
- Training ended 2026-10-22 23:16:13 +07 with exit 0, after 200 epochs (21 h 43 min). The watchdog finalized the run at 23:17:28 and stopped.
- Over the whole run, epochs 1–199 took 6.50 min on average; peak memory in the GPU log was 10,933 MiB, display included.
- Best checkpoint: epoch 188, `val_mel_l1_global` 21.2745.

## The two baselines (section 8)

**Commands** (2026-10-23 02:35 +07, after the author's OK of these exact commands, given before the evaluation above ended):

```sh
source ~/pfn-cr.env && cd ~/Pitch-FlowNet-cr
bash tools/camera_ready/launch_train.sh configs/PFN-Baseline-BigVGAN/PFN-Baseline-BigVGAN-200ep.json --gpu 0 --execute
bash tools/camera_ready/auto_resume.sh configs/PFN-Baseline-BigVGAN/PFN-Baseline-BigVGAN-200ep.json --start --gpu 0
bash tools/camera_ready/launch_train.sh configs/PFN-Baseline/PFN-Baseline-200ep.json --gpu 1 --execute
bash tools/camera_ready/auto_resume.sh configs/PFN-Baseline/PFN-Baseline-200ep.json --start --gpu 1
```

**Conditions met before the launch**, as the author set them:
- both evaluations above passed every check;
- `runs-Daziel.md` was committed (`10cd2cd`) and `git status --short` was empty;
- both dry runs exited 0;
- both GPUs were idle (4 MiB and 1,514 MiB used, display included).

**Run manifests,** checked 2 min after the launch:
- `head` `10cd2cd`, branch `PFN-Daziel-22102026`, **`dirty_paths: []`**;
- `tree_sha256` `1ee13cb4efe39da5657398021f3b85eb3a2a3eb5b89a1f1392dc33c4dfe90a52`;
- `max_g_updates` 156,400, `devices` 1, and the group flags of each config.

### PFN-Baseline-BigVGAN-200ep-s1234 (GPU 0)

| Item | Value |
|---|---|
| Config | `configs/PFN-Baseline-BigVGAN/PFN-Baseline-BigVGAN-200ep.json`; B on, V off, F off; 156,400 G updates |
| Start | 2026-10-23 02:35:47 +07, tmux `cr-PFN-Baseline-BigVGAN-200ep-s1234`, `CUDA_VISIBLE_DEVICES=0` |
| Launch log | `<WSL_DEV_HOME>/outputs-pfn-cr/logs/PFN-Baseline-BigVGAN-200ep-s1234-20261022T193547Z.log` |
| Speed | epoch 0: 10.0 min; epoch 1: 6.2 min; epochs 1–199: 6.15 min on average (about 5.5 min once `PFN-Baseline` had ended) |
| Peak memory (GPU log) | 9,577 MiB |
| Interruptions | none; one attempt, no automatic resume |
| End | 2026-10-23 23:09:41 +07, exit 0, after 200 epochs (20 h 34 min); finalized by the watchdog at 23:09:56 |
| Best | epoch 199 (the last epoch), `val_mel_l1_global` 21.0612 |

### PFN-Baseline-200ep-s1234 (GPU 1)

| Item | Value |
|---|---|
| Config | `configs/PFN-Baseline/PFN-Baseline-200ep.json`; B off, V off, F off; 156,400 G updates; the new strict baseline, **not** the historical epoch-773 baseline |
| Start | 2026-10-23 02:35:48 +07, tmux `cr-PFN-Baseline-200ep-s1234`, `CUDA_VISIBLE_DEVICES=1` (the display GPU) |
| Launch log | `<WSL_DEV_HOME>/outputs-pfn-cr/logs/PFN-Baseline-200ep-s1234-20261022T193548Z.log` |
| Speed | epoch 0: 9.0 min; epoch 1: 5.3 min; epochs 1–199: 5.31 min on average |
| Peak memory (GPU log) | 8,575 MiB, display included |
| Interruptions | none; one attempt, no automatic resume |
| End | 2026-10-23 20:20:53 +07, exit 0, after 200 epochs (17 h 45 min); finalized by the watchdog at 20:21:58 |
| Best | epoch 194, `val_mel_l1_global` 22.1978 |

## Final checkpoints

| Run | Final folder | best (epoch, `val_mel_l1_global`) | SHA-256 best | SHA-256 last | Status |
|---|---|---|---|---|---|
| `PFN-NoBigVGAN-200ep-s1234` | `outputs-pfn-cr/final/PFN-NoBigVGAN-200ep-s1234/` | 199, 22.3429 | `89aa7550fecbbdd4909b33ba19e87e2a49f85ce8ded7ed2536c36f0432171aa9` | `90abc7e97f0fe30cf482d8d1870edd808f85a846e0f61e464867a512dab81eb4` | final, evaluated |
| `PFN-NoVITS2-200ep-s1234` | `outputs-pfn-cr/final/PFN-NoVITS2-200ep-s1234/` | 188, 21.2745 | `9aa0730e95d7b9631f233d78c35f6110a5d413ac351d1dce2563bc90ea38e7ec` | `22e9aa78c2d8418bf36c9c334722af913faa9a6a9ae9403666391b82760d98ce` | final, evaluated |
| `PFN-Baseline-BigVGAN-200ep-s1234` | `outputs-pfn-cr/final/PFN-Baseline-BigVGAN-200ep-s1234/` | 199, 21.0612 | `10950c6e2ade4174fdabfe1c139c96882413d925d2287e5cb8acfa1a3c8a6e96` | `5e48544f6be4fb968d9ffa624849ea1735f5e4e013d8212fddb46b9b3d1a289d` | final, evaluated |
| `PFN-Baseline-200ep-s1234` | `outputs-pfn-cr/final/PFN-Baseline-200ep-s1234/` | 194, 22.1978 | `ede081d416e65a3228a5a44a33eb24fcfb87c423132b693b0aa02a218e8e5bb4` | `784852eea22384637ec11cd83cc4d3245b54408116f149897e8f90d2bf7ee480` | final, evaluated |

**Checks of all four final folders** (section 6 of the prompt):
- `best-epoch=…ckpt`, `last-epoch=199.ckpt`, `SHA256SUMS`, `final.json`, the manifest, ledger and diagnostics are present, and all are read-only;
- `sha256sum -c SHA256SUMS`: all five files OK;
- `final.json`: `epochs_completed` 200, `g_updates` 156,400, `attempts` 1, `source_tree_sha256` `1ee13cb4…`; `source_commit` `ee6a378` for the first two runs and `10cd2cd` for the two baselines.

**Environment smoke test** (section 4.5), before the test-split evaluation:
- 5 validation utterances on the best checkpoint of `PFN-NoVITS2-200ep-s1234`, into `<WSL_DEV_HOME>/outputs-pfn-cr/tests/eval-smoke-20261022/`, 2026-10-22 23:23–23:24 +07, exit 0.
- `protocol.json` had the right checkpoint hash, `devices: cpu (as historical)` and the package versions of `eval-protocol.md`.
- An environment check only; its numbers are not results.

## Evaluation on the test split (section 7)

**Protocol:** `eval-protocol.md`, unchanged.

**Command**, after the author's OK of this exact command:

```sh
source ~/pfn-cr.env && cd ~/Pitch-FlowNet-cr
B0=$(ls $CR_OUTPUTS/final/PFN-NoBigVGAN-200ep-s1234/best-epoch=*.ckpt)
B1=$(ls $CR_OUTPUTS/final/PFN-NoVITS2-200ep-s1234/best-epoch=*.ckpt)
bash tools/camera_ready/launch_eval.sh --split test --threads 4 --execute "PFN-NoBigVGAN-200ep-best=$B0" "PFN-NoVITS2-200ep-best=$B1"
```

**How it ran:**
- Both labels ran at the same time, each in its own tmux session at `nice -n10` with 4 CPU threads, from 2026-10-23 01:03:37 +07.
- They ended at 02:32:56 (NoBigVGAN) and 02:32:54 (NoVITS2), both with exit 0.
- The run tree was clean at `d09b282`. No GPU work ran at that time.
- Results outside Git: `<WSL_DEV_HOME>/outputs-pfn-cr/eval/<label>/`. Logs: `logs/eval-<label>-20261022T180337Z.log`.

**Checks, all passed for both labels:**
- `summary.json`: `num_sentences` 500, `mcd.n` 500, `mcd.n_skipped` 0;
- `protocol.json`:
  - `checkpoint_sha256` equals the best file in `final/<run_id>/SHA256SUMS`;
  - `split: test`;
  - package versions as `eval-protocol.md` (torch 2.12.1+cu130, torchaudio 2.11.0, openai-whisper 20250625, jiwer 4.0.0, transformers 5.12.1, timm 1.0.27, scipy 1.15.3, librosa 0.11.0, utmosv2 1.3.1.dev0);
- `mcd.mean_gt_utmos` 3.83079 for both, against 3.8308 on MardeusPC.

**Numbers.** These are the raw values of `summary.json`, rounded.
- WER is a fraction (0.1214 = 12.14 %).
- UTMOSv2 is a **predicted score, not MOS**.
- F0 RMSE and correlation are the F0-predictor metric against the cached reference F0.

| Label | Checkpoint (SHA-256) | WER mean | WER median | UTMOSv2 mean | UTMOSv2 median | MCD mean | MCD median | F0 RMSE | F0 corr | `eval_seconds` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PFN-NoBigVGAN-200ep-best` | epoch 199 (`89aa7550…`) | 0.1214 | 0.0769 | 3.2442 | 3.2703 | 25.923 | 25.748 | 0.4813 | 0.1784 | 2456.2 |
| `PFN-NoVITS2-200ep-best` | epoch 188 (`9aa0730e…`) | 0.1101 | 0.0588 | 3.5185 | 3.5452 | 25.760 | 25.639 | 0.5827 | 0.1992 | 2458.4 |

- **RTF is not reportable** from these runs: both evaluations ran at the same time on the same CPU (`mean_rtf` 0.0709 and 0.0722 inline).
- **No comparison and no conclusion here.** MardeusPC collects `eval/<label>/` without `wavs/` and runs `compare`.

### The two baselines

**Command**, after the author's OK of this exact command:

```sh
source ~/pfn-cr.env && cd ~/Pitch-FlowNet-cr
B0=$(ls $CR_OUTPUTS/final/PFN-Baseline-BigVGAN-200ep-s1234/best-epoch=*.ckpt)
B1=$(ls $CR_OUTPUTS/final/PFN-Baseline-200ep-s1234/best-epoch=*.ckpt)
bash tools/camera_ready/launch_eval.sh --split test --threads 4 --execute "PFN-Baseline-BigVGAN-200ep-best=$B0" "PFN-Baseline-200ep-best=$B1"
```

**How it ran:**
- Both labels ran at the same time, at `nice -n10` with 4 CPU threads, from 2026-10-23 23:22:44 +07. No GPU work ran.
- They ended at 2026-10-24 00:51:46 (Baseline-BigVGAN) and 00:51:44 (Baseline), both with exit 0.
- The repository was at `10cd2cd`. During the evaluation the untracked page `PROMPT-mardeus-export.md` was written; it is not code, and the tool does not record the repository state.
- Logs: `logs/eval-<label>-20261023T162244Z.log`.

**Checks, all passed for both labels:** the same as above. `mcd.mean_gt_utmos` is 3.83079 for both.

| Label | Checkpoint (SHA-256) | WER mean | WER median | UTMOSv2 mean | UTMOSv2 median | MCD mean | MCD median | F0 RMSE | F0 corr | `eval_seconds` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PFN-Baseline-BigVGAN-200ep-best` | epoch 199 (`10950c6e…`) | 0.1009 | 0.0667 | 3.4945 | 3.5125 | 25.744 | 25.577 | n/a | n/a | 2404.2 |
| `PFN-Baseline-200ep-best` | epoch 194 (`ede081d4…`) | 0.1082 | 0.0667 | 2.9407 | 2.9585 | 25.573 | 25.350 | n/a | n/a | 2391.7 |

- **F0 n/a:** both models have no F0 group, so the F0-predictor metric does not apply (`null` in `summary.json`).
- **RTF is not reportable**, for the same reason as above (`mean_rtf` 0.0737 and 0.0593 inline).
- **No comparison and no conclusion here.**

## Services stopped (author's instruction, 2026-10-23)

- After all four runs had ended, the author asked to train nothing more on this machine.
- TensorBoard (`cr-tensorboard`) and the GPU log (`cr-gpulog`) were stopped at 23:26 +07.
- The WSL keep-alive was stopped on 24/10 at 00:52, after both evaluations had ended.

## Import from MardeusPC: 2026-10-24 19:52 +07

**What came.** The package of `runs-MardeusPC.md`, "Export to the PC", written there with `export_four.sh`. The author copied it into the Windows copy of the repository, under `docs/pfn-transfer-20261024/`, unzipped:
- 9 tar files and `SHA256SUMS.source`;
- the zip itself was not there, so its SHA-256 (`b87a6676…`) could not be checked.

**Checks, all passed** (script log `<WSL_DEV_HOME>/outputs-pfn-cr/logs/import-mardeus-20261024.log`):
- **The hash list itself:** `SHA256SUMS.source` is identical, line for line, to the copy committed in `runs-MardeusPC.md` (9 lines). The hash list does not rest only on the package that carried it.
- **Hashes:** `sha256sum -c SHA256SUMS.source` gave 9/9 OK, once at the source and again after staging the tars in `<WSL_DEV_HOME>/pfn-import-staging-20261024/` (outside `$CR_OUTPUTS`).
- **Tar contents:** every tar was listed before extraction. There was no absolute path, no `..`, no link, and no `wavs/` or `.wav` entry. Each tar has one top folder.
- **No overwrite:** none of the targets existed before.
- **Extraction:** into `$CR_OUTPUTS/eval/<label>/` (Full, NoF0, Baseline-VITS2, Baseline-F0), `$CR_OUTPUTS/eval/compare-PFN-Full-200ep-best_PFN-NoF0-200ep-best.json` and `$CR_OUTPUTS/final/<run_id>/` (`PFN-Full-275ep-s1234`, `PFN-NoF0-200ep-s1234`, `PFN-Baseline-VITS2-200ep-s1234`, `PFN-Baseline-F0-200ep-s1234`).
- **Final folders:** `sha256sum -c SHA256SUMS` in each imported `final/<run_id>/` gave 5/5 OK each. All four folders are read-only (0 writable entries).

**Afterwards:**
- The staging folder was removed after the checks.
- On the author's instruction, `docs/pfn-transfer-20261024/` (6.45 GB) was deleted from the Windows copy, so that OneDrive does not sync it and Git cannot stage it. The copy on MardeusPC (`E:\pfn-transfer-20261024\` and the zip) remains.

**Analysis of the eight cells.** Results and their sources: `ablation8-results.md`; hashes: `manifests/camera-ready/ablation8-derived.json`.
- Tool `tools/camera_ready/ablation8_analysis.py`, commits `a2521ae` and `5eb3510`.
- The twelve missing `compare-*.json` were written here with `evaluate_like_history.py compare` (log `logs/compare-ablation8-20261024.log`).
- Nothing was trained, synthesized or re-scored.

## Proposed updates to `PRIORITY.md` and `experiment-plan.md` (for MardeusPC to apply)

- RTX 4070 Ti speed, measured in epoch 1: 6.6 min per epoch (`NoBigVGAN`), 6.4 min (`NoVITS2`), with both GPUs training at once. About 22 h per 200-epoch run, against ≈ 28–32 h on the RTX 5070.
- The PC started on 22/10 01:33 +07, not 21/10 14:00. Expected ends without interruptions:
  - GPU 0: `PFN-NoBigVGAN` ≈ 22/10 23:40 → `PFN-Baseline-BigVGAN` ≈ 23/10 22:00;
  - GPU 1: `PFN-NoVITS2` ≈ 22/10 23:10 → `PFN-Baseline` ≈ 23/10 21:30.
  - The second runs assume a launch soon after each first run ends, after the author's OK.
- Because the PC is faster, one of `PFN-Baseline-VITS2` or `PFN-Baseline-F0` could move to a PC GPU from about 24/10 if MardeusPC is behind. This is the author's decision.
- **Update 2026-10-23 02:40:**
  - Both first runs are final and evaluated: `PFN-NoVITS2` ended 22/10 23:16, `PFN-NoBigVGAN` 23/10 00:12.
  - Whole-run speed: 6.50 and 6.77 min per epoch.
  - Their `eval/<label>/` folders are ready for collection.
  - The two baselines start after this commit, on 23/10 ≈ 02:45; expected end ≈ 24/10 01:00, without interruptions.
- **Update 2026-10-24 00:55:**
  - All four PC runs are final and evaluated on the test split:
    - `PFN-NoVITS2` ended 22/10 23:16;
    - `PFN-NoBigVGAN` ended 23/10 00:12;
    - `PFN-Baseline` ended 23/10 20:20;
    - `PFN-Baseline-BigVGAN` ended 23/10 23:09.
  - No run needed a resume.
  - Nothing more trains on the PC.
  - Next, by the author's decision: the PC collects `eval/<label>/` (without `wavs/`) and `final/<run_id>/` of `PFN-Full`, `PFN-NoF0` and `PFN-Baseline-VITS2` from MardeusPC on an external drive (`PROMPT-mardeus-export.md`), and runs `compare` for the seven models. `PFN-Baseline-F0` follows when it is final.
- **Update 2026-10-24 20:10:**
  - All eight cells are imported and analysed (`ablation8-results.md`). `PRIORITY.md` and `experiment-plan.md` are updated from this machine, on the author's instruction of 24/10.
  - The author asked for one RTF benchmark of the eight best checkpoints (T15). Its protocol is still to be agreed: CPU as `eval-protocol.md` says, or one idle GPU as the task of 24/10 says. Nothing has been run.
- **Update 2026-10-24 21:20: CPU RTF benchmark (T15) done.**
  - **Protocol:** the author chose 1 and 2 threads on the CPU and agreed to the protocol, the smoke run and the full run.
  - **Commands:**
    - `wsl_keepalive.ps1 start` (20:29);
    - one smoke run of `benchmark_rtf.py run-one` (Full, 1 thread) into `tests/rtf-smoke-20261024/`, RTF 0.0880;
    - `benchmark_rtf.py plan --execute` in tmux `cr-rtf`, 20:31:09 → 21:11:57, exit 0, 48 runs;
    - `wsl_keepalive.ps1 stop` (21:13).
  - **Result:** `ablation8-results.md`, section 5. The synthesized lengths equal the evaluation's for every model.
  - **Machine state:** no other job of this project ran during the benchmark, and no other tmux session existed at its start. Activity on the Windows side was not monitored.
