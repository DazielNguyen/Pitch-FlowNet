# Preflight: the PC (2 × RTX 4070 Ti), 2026-10-22

**Scope:** SOICT 2026 paper 4478, PitchFlowNet, camera-ready ablation runs only.
- This is the PC of `preflight.md`. It trains four of the eight 200-epoch runs (`experiment-plan.md`):
  - GPU 0: `PFN-NoBigVGAN-200ep` → `PFN-Baseline-BigVGAN-200ep`;
  - GPU 1: `PFN-NoVITS2-200ep` → `PFN-Baseline-200ep`.
- Instructions: `PROMPT-daziel-ablation.md`. Run record: `runs-Daziel.md`.
- Everything on this page was read-only, except the new clone, the new environment `pfn-cr`, the MAS extension build and the test run.

## Repository (step 2), 2026-10-22 01:20 +07

| Item | Value |
|---|---|
| Clone | `<WSL_DEV_HOME>/Pitch-FlowNet-cr`, new clone of `https://github.com/DazielNguyen/Pitch-FlowNet.git` |
| `HEAD` on `main` | `ee6a37804644d69902a5274e9febbc8ab8d83ff2`, equal to `origin/main` |
| Working tree | clean (`git status --short` empty) |
| Commit hook | `core.hooksPath = tools/hooks` |
| Work branch | `PFN-Daziel-22102026`, from `ee6a378`; not pushed |
| Git-ignored originals | **Not present**: this is a clone. Run only `tools/check_tracked_package.py` here, never `tools/sanitize_for_git.py` |

**Commits checked** with `git merge-base --is-ancestor <commit> HEAD`; all are ancestors of `HEAD`:

| Commit | Content |
|---|---|
| `3ac8756` | 200-epoch configs for the eight-run ablation |
| `65b5cb3` | `finalize_run.sh` (archive of best and last checkpoint) |
| `f773be2` | evaluation tool |
| `33dca31` | evaluation protocol |
| `d008b4b` | `auto_resume.sh` passes the `CR_*` settings into the watchdog tmux session |

**Files checked** (all present):
- `configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json`, `configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json`, `configs/PFN-Baseline-BigVGAN/PFN-Baseline-BigVGAN-200ep.json`, `configs/PFN-Baseline/PFN-Baseline-200ep.json`;
- `tools/camera_ready/`: `launch_train.sh`, `resume_train.sh`, `auto_resume.sh`, `finalize_run.sh`, `check_ckpt.py`, `tensorboard.sh`, `wsl_keepalive.ps1`;
- `docs/camera-ready/`: `experiment-plan.md`, `runbook-MardeusPC.md`, `eval-protocol.md`;
- `manifests/camera-ready/split-manifest.json`, `manifests/camera-ready/data-manifest.json`.

**Config contents** (read from the JSON; all as expected):

| Config | `max_g_updates` | `devices` | `strategy` | `batch_size` | `use_bigvgan` | `use_vits2` | `use_f0` |
|---|---:|---:|---|---:|---|---|---|
| `PFN-NoBigVGAN-200ep` | 156,400 | 1 | `auto` | 16 | false | true | true |
| `PFN-NoVITS2-200ep` | 156,400 | 1 | `auto` | 16 | true | false | true |
| `PFN-Baseline-BigVGAN-200ep` | 156,400 | 1 | `auto` | 16 | true | false | false |
| `PFN-Baseline-200ep` | 156,400 | 1 | `auto` | 16 | false | false | false |

## GPUs and other work (step 3), 2026-10-22 01:21 +07

| Index | Model | UUID | PCI bus | Display | Memory used | Utilisation |
|---:|---|---|---|---|---:|---:|
| 0 | GeForce RTX 4070 Ti, 12,282 MiB | `GPU-e512cdeb-2918-b5e3-e7b1-91fd010996a6` | `01:00.0` | no | 0 MiB | 0 % |
| 1 | GeForce RTX 4070 Ti, 12,282 MiB | `GPU-352ef33d-d1ff-e02d-cd44-df3f41332fca` | `06:00.0` | **yes** | 1,453 MiB | 1 % |

- No compute process on either GPU.
- `ps -eo pid,etime,cmd | grep -iE "seqir|piper|train"`: nothing.
- `tmux ls`: no server running. The sessions `piper-ir`, `post-seqir`, `tb-seqir` of the other project do not exist; WSL had been stopped before this check.
- Both GPUs are under the limits (3,000 MiB, 10 %). **Not GPU_BLOCKED.**

**GPU 1 drives the display.**
- Its 1.45 GB are the desktop. A run there has about 10.8 GB, against 12.3 GB on GPU 0.
- On MardeusPC the display GPU had `LiveKernelEvent` 141 resets and two `CUDA error: unknown error` crashes (`runs-MardeusPC.md`). The watchdog covers this error, but it is a known risk for GPU 1, not for GPU 0.

## Disk space

| Volume | Free |
|---|---|
| WSL ext4 (`/`, 1007 GB) | 601 GB |
| Windows C: (holds the WSL `.vhdx`) | 103 GB |

Four runs need about 4 × (3 + 1.75) ≈ 19 GB. Enough.

## Data (step 3), read with `nice -n19 ionice -c3`

| Item | Check | Result |
|---|---|---|
| `<WSL_DEV_HOME>/data_preprocessed/dataset.jsonl` | SHA-256 against `data-manifest.json` | OK `3b93b988…` |
| `<WSL_DEV_HOME>/data_preprocessed/config.json` | SHA-256 | OK `6eeb8982…` |
| `<WSL_DEV_HOME>/test_entries_500.jsonl` | SHA-256 | OK `8c97aa2b…` |
| `<WSL_DEV_HOME>/data_preprocessed/cache/22050/` | file count | 39,300 files (22 GB) |
| `<WSL_DEV_HOME>/data/wavs/` | file count | 13,100 files |

## System (step 3)

| Item | Value |
|---|---|
| OS | Windows 11 Pro, build 10.0.26200.8655 |
| WSL | 2.7.8.0, kernel 6.18.33.1-1, WSL 2 |
| Distribution | Ubuntu 24.04.4 LTS (`Ubuntu-24.04`); `/etc/wsl.conf` default user `dev` |
| `%USERPROFILE%\.wslconfig` | `memory=64GB`, `processors=24`, `swap=16GB` |
| CPU | Intel Core i7-14700, 24 logical CPUs |
| RAM | 127.8 GB host; 62.8 GiB visible in WSL, 61 GiB free; swap 16 GiB, unused |
| NVIDIA driver | 596.36 (Windows 32.0.15.9636); `nvidia-smi` 595.71.01; CUDA UMD 13.2 |
| Windows power (AC) | sleep: never; display off: never (both already 0 at check time) |

## Environment (step 4), `<WSL_DEV_HOME>/miniconda3/envs/pfn-cr`

Created on 22/10 with the existing Miniconda (conda 26.3.2, channel `defaults`); `base`, `edgetts` and `banhmi-tts` were not touched.

| Item | Version |
|---|---|
| Python | 3.10.20 |
| torch | 2.12.1+cu130 (CUDA 13.0), from the PyTorch cu130 index |
| cuDNN | 9.20.0 (`92000`) |
| Architectures in the torch build | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`. The RTX 4070 Ti is `sm_89`; it runs the `sm_86` kernels |
| pytorch-lightning | 2.6.5 |
| numpy | 2.2.6 |
| Other training packages | cython 0.29.37, piper-phonemize 1.1.0, librosa 0.11.0, pyworld 0.3.5, onnxruntime 1.23.2, tensorboard 2.21.0, scipy 1.15.3, setuptools 78.1.0 |
| System tools | tmux 3.4, gcc 13.3.0, git, rsync, curl, espeak-ng |

- `torch.cuda.device_count()` = 2, both GeForce RTX 4070 Ti.
- **MAS extension:** built in `code/camera-ready-source/src/python/piper_train/vits/monotonic_align/` with `cythonize -i core.pyx`; `monotonic_align/core.cpython-310-x86_64-linux-gnu.so`. The tree stays clean (build outputs are git-ignored).
- **`<WSL_DEV_HOME>/pfn-cr.env`:** `CR_OUTPUTS=$HOME/outputs-pfn-cr`, `CR_PYTHON` and `CR_TENSORBOARD` of `pfn-cr`, `CR_TB_PORT=6007`.

## Tests (2026-10-22 01:27 +07, clone at `ee6a378`)

`$CR_PYTHON -m unittest discover -s tests/camera_ready -v`: 25 tests, **24 pass, 1 ERROR**, as expected.
- The ERROR is `ValidationDDP.test_two_processes_equal_one`: `ddp_spawn` cannot serialize a non-leaf tensor that requires grad (PLAN O2). It is not needed for 1-GPU runs.
- Log outside Git: `<WSL_DEV_HOME>/outputs-pfn-cr/tests/unittest-20261021T182743Z.log`.

## Dry runs (step 5, before any launch)

`launch_train.sh … --gpu N` without `--execute`; both printed `inputs: ok (train.py --dry-run)` and exited 0. Nothing was started.

| Run ID | GPU | GPU state at dry run |
|---|---:|---|
| `PFN-NoBigVGAN-200ep-s1234` | 0 | 0 MiB, 0 % |
| `PFN-NoVITS2-200ep-s1234` | 1 | 1,484 MiB, 0 % |

## Open before the launch

- Windows side, by the author: pause Windows Update; start `wsl_keepalive.ps1 start -Distro Ubuntu-24.04`.
- Git identity is not set in this WSL clone (`user.name`, `user.email` empty). The author sets it before the first commit here.
