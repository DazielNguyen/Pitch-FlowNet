# Preflight: MardeusPC (2026-10-19)

**Scope:** SOICT 2026 paper 4478, PitchFlowNet, camera-ready experiments only.
- This is **not** the PC described in `preflight.md` (that one has 2 × RTX 4070 Ti).
- Nothing has been trained or evaluated on this machine.
- Set up on 19/10, on the author's instruction:
  - WSL and Ubuntu;
  - the `dev` account;
  - the conda environment.
- **Training here uses the one RTX 5070.** The 1-GPU configs are in `configs/PFN-*/`, see `README.md`.

**Status:** data verified; resume test (O1) open before any pilot.

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

All items verified on 2026-10-20 00:36 (`IMPORT PASS`).

| Item | Location in WSL | Check | Result |
|---|---|---|---|
| `dataset.jsonl` | `<WSL_DEV_HOME>/data_preprocessed/` | SHA-256 against `data-manifest.json` | OK |
| `config.json` | `<WSL_DEV_HOME>/data_preprocessed/` | SHA-256 | OK |
| Feature cache | `<WSL_DEV_HOME>/data_preprocessed/cache/22050/` | file count (no per-file manifest on this machine); covered by the archive hash | 39,300 files |
| Raw WAVs | `<WSL_DEV_HOME>/data/wavs/` | `raw_wav_sha256` of every entry in `split-manifest.json` | 13,100 checked, 0 mismatches |
| `test_entries_500.jsonl` | `<WSL_DEV_HOME>/` | SHA-256 | OK |
| Checkpoint 773 | `<WSL_DEV_HOME>/02_Baseline_VanillaVITS/lightning_logs/version_0/checkpoints/` | SHA-256 and bytes | OK, 845,967,744 bytes |
| Checkpoint 1079 | `<WSL_DEV_HOME>/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/` | SHA-256 and bytes | OK, 874,730,405 bytes |

**Transfer route.**

1. **Tailscale.**
   - The PC's WSL joined the tailnet with a user-level Tailscale in userspace mode. Nothing was installed system-wide on the PC.
   - Only the DERP relay worked, at about 0.45 MB/s.
   - Checkpoint 773 came this way.
2. **External drive.** On the PC, four tar archives were written to an external SSD.
   - Each archive was read once, at the lowest CPU and IO priority. Its SHA-256 was computed while it was written (`SHA256SUMS.source`).
   - The PC's training sessions were not touched.
3. **On this machine.** Reading the drive in the USB port was unstable: Windows logged repeated `UASPStor` resets and retried IO at the same block addresses.
   - The author copied the archives by hand into `<WSL_DEV_HOME>/PitchFlowNet-data/`.
   - All four matched `SHA256SUMS.source`.
   - They were then unpacked and checked item by item, as in the table above.

**Scripts and logs (outside Git):**
- `<WSL_DEV_HOME>/pfn-sync/import_local.sh` and its `.log` on this machine;
- `export_to_f.sh` and its `.log` beside the archives.

## Tests (2026-10-20 00:36, WSL clone at `f5669a9`)

`python -m unittest discover -s tests/camera_ready -v`: 25 tests, 22 pass, 2 FAIL, 1 ERROR. The log is outside Git, in `<WSL_DEV_HOME>/outputs/tests/`.

| Test | Result | Open item |
|---|---|---|
| distributed (4), gating (4), historical checkpoints (3), split (7), train_cpu budget / directory guard / diagnostics (3) | pass | — |
| `TrainCPU.test_resume_reproduces_uninterrupted_run` | FAIL: 986 tensors differ | O1 (T04) |
| `TrainCPU.test_ledger_and_best_state_across_resume` | FAIL: best file `…=86.4454` vs `…=85.7259` | O1 (T04) |
| `ValidationDDP.test_two_processes_equal_one` | ERROR: `ddp_spawn` cannot serialize a non-leaf tensor that requires grad | O2 (T05) |

**The resume difference here is much larger than on the PC.**
- On the PC: 455 of 994 tensors differed, by about 1e-8.
- Here: 986 tensors differ, and the validation metric after one epoch differs by about 0.7.
- This has not been investigated yet (T04). It is not known whether it is nondeterminism or a resume defect.

## Not done

- PLAN O1 (resume), before any pilot.
- GPU smoke and pilots.
