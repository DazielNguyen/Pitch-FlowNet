# Preflight: MardeusPC (2026-10-19)

**Scope:** SOICT 2026 paper 4478, PitchFlowNet, camera-ready experiments only.
- This is **not** the PC described in `preflight.md` (that one has 2 × RTX 4070 Ti).
- Nothing has been trained or evaluated on this machine.
- Set up on 19/10, on the author's instruction:
  - WSL and Ubuntu;
  - the `dev` account;
  - the conda environment.
- **Training here uses the one RTX 5070.** The 1-GPU configs are in `configs/PFN-*/`, see `README.md`.

**Status: BLOCKED_MISSING_INPUT** (data and checkpoints not on this machine yet).

## Repository

| Item | Value |
|---|---|
| Branch | `camera-ready-20261019`, tracking `origin/camera-ready-20261019` |
| Start point | `58cab26` (contains `d528a90`); tree identical to `main` at `d71c20b` |
| Windows copy | Under the Windows `Documents` folder (not redirected to OneDrive) |
| Git-ignored originals | **Not present**: this is a clone. Run only `tools/check_tracked_package.py` here |

**Do not run `tools/sanitize_for_git.py` on this machine.**
- Without the originals, the tool rebuilds an empty map.
- It then deletes the tracked `*.sanitized.*` copies and `manifests/sanitization/*.json`, and empties the block in `.gitignore`.
- This happened once on 19/10. The 113 files and `.gitignore` were restored from Git before any commit.

## System

| Item | Value |
|---|---|
| OS | Windows 11 Pro, build 10.0.26300 |
| WSL | WSL 3.0.3, kernel 6.18.54.1-1, WSL 2 |
| Distribution | Ubuntu 22.04.5 LTS (`Ubuntu-22.04`) |
| Virtual disk | `E:\WSL\Ubuntu-22.04`, internal NTFS volume E: (931 GB, 719 GB free at setup); ext4 inside: 1007 GB, 955 GB free |
| `/etc/wsl.conf` | `systemd=true`, default user `dev` |
| `%USERPROFILE%\.wslconfig` | `memory=24GB`, `swap=16GB` (host RAM 31.8 GB) |
| Account | `dev`; its home directory is `<WSL_DEV_HOME>` of the data paths; passwordless sudo via `/etc/sudoers.d/dev` |
| GPU | 1 × NVIDIA GeForce RTX 5070, 12,227 MiB, UUID `GPU-4c9934ff-5acc-0bc0-0734-01cf5d66b149`, compute capability 12.0 |
| Driver / CUDA | NVIDIA driver 617.42 (Windows), CUDA UMD 13.4 |
| GPU use at check time | 1,244 MiB, 0 %; desktop applications only, no compute job. The GPU also drives the display, so it is not a dedicated device |
| CPU / RAM | Intel Core i5-14400F, 16 logical CPUs; 31.8 GB RAM (23 GB visible in WSL) |
| Windows disk free | C: 152.6 GB; E: 719.3 GB |

## Environment (`~/miniforge3/envs/pfn-cr`)

| Item | Version | Same as the PC (`preflight.md`) |
|---|---|---|
| conda | Miniforge, conda 26.7.2 (conda-forge) | PC used conda env `edgetts`; distribution not recorded |
| Python | 3.10.20 | yes |
| torch | 2.12.1+cu130 (CUDA 13.0) | yes |
| cuDNN | 9.20.0 (`92000`) | not recorded on the PC |
| pytorch-lightning | 2.6.5 | yes |
| numpy | 2.2.6 | yes |
| Other training packages | `cython<1`, `librosa<1`, `tensorboard<3`, `pyworld<1`, `onnxruntime`, `piper-phonemize~=1.1.0` (from `requirements.txt`) | not compared |
| System packages (apt) | build-essential, git 2.34.1, rsync 3.2.7, tmux 3.2a, openssh-client/server, curl, espeak-ng | — |

**GPU check:**
- `torch.cuda.is_available()` is True and reports the GeForce RTX 5070.
- The torch build lists `sm_120` among its architectures.
- This only shows that the device is visible. Kernels have not been run on it yet; the GPU smoke test (O3) is still to come.

onnxruntime prints a warning that GPU device discovery failed (`/sys/class/drm/card0/device/vendor` is absent under WSL).
It is used only for export checks, not for training.

## Data (`data-sync.md`, `manifests/camera-ready/data-manifest.json`)

| Item | Expected location in WSL | Present |
|---|---|---|
| `dataset.jsonl`, `config.json` | `<WSL_DEV_HOME>/data_preprocessed/` | No |
| Feature cache, 39,300 files | `<WSL_DEV_HOME>/data_preprocessed/cache/22050/` | No |
| Raw WAVs, 13,100 files | `<WSL_DEV_HOME>/data/wavs/` | No |
| `test_entries_500.jsonl` | `<WSL_DEV_HOME>/test_entries_500.jsonl` | No |
| Checkpoint 773 | `<WSL_DEV_HOME>/02_Baseline_VanillaVITS/lightning_logs/version_0/checkpoints/` | No |
| Checkpoint 1079 | `<WSL_DEV_HOME>/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/` | No |

**Transfer.**
- The data will come over Tailscale from the PC (author's decision).
- At setup time the PC was not visible in the tailnet: only this machine and one other Windows machine, offline for 80 days.

## Not done

- Data and checkpoint hash checks (no data).
- MAS build, PLAN O1/O2, unit tests.
- GPU smoke and pilots.
