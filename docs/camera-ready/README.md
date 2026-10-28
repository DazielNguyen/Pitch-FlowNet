# Camera-ready experiments

**Scope:** the camera-ready experiments of the SOICT 2026 paper 4478 (PitchFlowNet).
- the 2³ ablation of the three component groups at 200 epochs;
- its evaluation and statistics;
- the supplementary pitch, duration and RTF analyses.

The original records in the repository are not changed. The camera-ready files sit in their own subfolders.

## Documents

| Document | Contents |
|---|---|
| [ablation-final.md](ablation-final.md) | **Final statistics of every comparison**: main pair, ablation, factorial effects, pitch and duration analyses, cost, limits, tables for the paper |
| [ablation8-results.md](ablation8-results.md) | Detailed analysis of the eight ablation runs, with the source of every number |
| [eval-protocol.md](eval-protocol.md) | The evaluation protocol, locked on 2026-10-21 before any new test-set result |
| [experiment-plan.md](experiment-plan.md) | Design of the ablation and the decisions taken before the test results |

## Machines

| Name | GPU | CPU | Used for |
|---|---|---|---|
| Machine A | 2 × RTX 4070 Ti | Intel Core i7-14700 | four ablation runs; the analyses and the CPU RTF benchmark |
| Machine B | RTX 5070 | Intel Core i5-14400F | four ablation runs; the evaluation of the historical pair |

Both run WSL 2 on Windows. The GPU of each run is listed in `ablation-final.md`, table X4.

## Where things live

| Kind | Folder | Contents |
|---|---|---|
| Source | `code/camera-ready-source/src/python/` | The historical candidate `badf381` plus the changes listed in `manifests/camera-ready/source-provenance-post*.json`; the camera-ready training entry point is `piper_train/camera_ready/` |
| Run configs, 1 GPU | `configs/PFN-<variant>/` | `<variant>-200ep.json` (the ablation runs), `<variant>-275ep.json`, `<variant>.json` and `<variant>-Pilot.json` (200 G updates) |
| Run configs, 2 GPUs | `configs/camera-ready-2gpu/` | The `H-*`, `B-*` and `pilot-*` configs; each 1-GPU config names the one it derives from |
| Test config | `configs/camera-ready-tests/` | `cpu-smoke-H.json` (CPU tests) |
| Manifests | `manifests/camera-ready/` | `split-manifest.json`, `data-manifest.json`, `source-provenance-*.json`, `ablation8-derived.json` (hashes of every derived result) |
| Tools | `tools/camera_ready/` | Training (`launch_train.sh`, `resume_train.sh`, `auto_resume.sh`, `finalize_run.sh`, `check_ckpt.py`), evaluation (`launch_eval.sh`, `evaluate_like_history.py`), analysis (`ablation8_analysis.py`, `ablation_final.py`, `benchmark_rtf.py`, `pitch_shift.py`, `waveform_f0.py`, `duration_diag.py`) |
| Tests | `tests/camera_ready/` | `python -m unittest discover -s tests/camera_ready -v`, from the repository root |

## Variants

| Folder | Groups (BigVGAN / VITS2 / F0) | Role |
|---|---|---|
| `PFN-Full` | on / on / on | reference |
| `PFN-NoF0` | on / on / off | leave-one-out: F0 |
| `PFN-NoVITS2` | on / off / on | leave-one-out: VITS2 |
| `PFN-NoBigVGAN` | off / on / on | leave-one-out: BigVGAN |
| `PFN-Baseline` | off / off / off | strict baseline; **not** the historical baseline (epoch 773) |
| `PFN-Baseline-BigVGAN` | on / off / off | add-one: BigVGAN only |
| `PFN-Baseline-VITS2` | off / on / off | add-one: VITS2 only |
| `PFN-Baseline-F0` | off / off / on | add-one: F0 only |

**Run ID:** the run ID, and so the run directory, is the config's `run_id`, for example `PFN-Full-200ep-s1234`.

**Run IDs of `PFN-Full`:** this run was started with the 275-epoch config and stopped at epoch 200. Its run ID is therefore `PFN-Full-275ep-s1234` (see `experiment-plan.md`).

## Old paths in the source-provenance records

Until 2026-10-19 the camera-ready files sat in one folder, `camera-ready-20261019/`. They were moved with `git mv`, so `git log --follow` shows their history.

**Old paths:** `source-provenance-*.json` were written before the move and still give the old paths.

| Old path | New path |
|---|---|
| `camera-ready-20261019/src/` | `code/camera-ready-source/src/` |
| `camera-ready-20261019/configs/{H,B,pilot}-*.json` | `configs/camera-ready-2gpu/` |
| `camera-ready-20261019/configs/cpu-smoke-H.json` | `configs/camera-ready-tests/` |
| `camera-ready-20261019/*.json` | `manifests/camera-ready/` |
| `camera-ready-20261019/tests/` | `tests/camera_ready/` |
| `camera-ready-20261019/tools/` | `tools/camera_ready/` |
| `camera-ready-20261019/*.md` | `docs/camera-ready/` |

**What the move changed in the code:**
- **Training source:** none. `source_tree_sha256` in `run_manifest.json` hashes `piper_train/` only.
- **Tests and tools:** only their path constants changed.
