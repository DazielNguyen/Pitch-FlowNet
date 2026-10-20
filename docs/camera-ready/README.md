# Camera-ready work: where things live

**Scope:** camera-ready experiments for SOICT 2026 paper 4478 (PitchFlowNet).
Git branch: `camera-ready-20261019`.

The camera-ready files sit in their own subfolders of the main repository folders, one place per kind of file.
The original evidence files beside them are unchanged.

| Kind | Folder | Contents |
|---|---|---|
| Source | `code/camera-ready-source/src/python/` | Copy of the historical candidate `badf381` plus the changes in `manifests/camera-ready/source-provenance-post.json`. This is current source, not the source that ran historically |
| Run configs, 1 GPU | `configs/PFN-<variant>/` | `<variant>.json` (full run) and `<variant>-Pilot.json` (200 G updates). Used on MardeusPC |
| Run configs, 2 GPUs | `configs/camera-ready-2gpu/` | The original `H-*`, `B-*` and `pilot-*` configs; each 1-GPU config names the one it derives from |
| Test config | `configs/camera-ready-tests/` | `cpu-smoke-H.json` (CPU tests) |
| Manifests | `manifests/camera-ready/` | `split-manifest.json`, `data-manifest.json`, `source-provenance-pre.json`, `source-provenance-post.json` (19/10), `source-provenance-post-2026-10-20.json` (after the resume fix in `camera_ready/model.py`), `commit-map-2026-10-20.json` (old → new commit hashes after the message rewrite of 20/10; trees unchanged) |
| Tests | `tests/camera_ready/` | `python -m unittest discover -s tests/camera_ready -v` from the repository root |
| Tools | `tools/camera_ready/` | Split manifest, source provenance; `launch_train.sh`, `resume_train.sh`, `check_ckpt.py`, `tensorboard.sh`, `wsl_keepalive.ps1` (see `runbook-MardeusPC.md`); `resume_diagnostic.py` (PLAN O1) |
| Documents | `docs/camera-ready/` | `PRIORITY.md`, `PLAN.md`, `experiment-plan.md` (decisions), `eta.md`, `preflight*.md`, `data-sync.md`, `runbook-MardeusPC.md`, `PROMPT-training-machine.md`, `PROMPT-daziel-ablation.md` (the two ablations on the PC); later `handoff_to_mac.md` |

## Variants

| Folder | Groups (BigVGAN / VITS2 / F0) | Tier |
|---|---|---|
| `PFN-Full` | on / on / on | T1 reference |
| `PFN-NoF0` | on / on / off | T1 main contrast |
| `PFN-NoVITS2` | on / off / on | T2 |
| `PFN-NoBigVGAN` | off / on / on | T2 |
| `PFN-Baseline` | off / off / off | T3; strict gating, **not** the historical baseline (epoch 773) |
| `PFN-Baseline-BigVGAN` | on / off / off | add-one: BigVGAN only |
| `PFN-Baseline-VITS2` | off / on / off | add-one: VITS2 only |
| `PFN-Baseline-F0` | off / off / on | add-one: F0 only |

The run ID, and so the run directory, is the config's `run_id` (for example `PFN-Full-s1234`).

## Move of 19/10

Until 19/10 everything sat in one folder, `camera-ready-20261019/`. The files were moved with `git mv`, so `git log --follow` shows their history.

**Old paths remain in two records.** `preflight.md` and `source-provenance-*.json` were written before the move and still give them. They are records, so they were not rewritten.

| Old path | New path |
|---|---|
| `camera-ready-20261019/src/` | `code/camera-ready-source/src/` |
| `camera-ready-20261019/.gitignore` | `code/camera-ready-source/.gitignore` |
| `camera-ready-20261019/configs/{H,B,pilot}-*.json` | `configs/camera-ready-2gpu/` |
| `camera-ready-20261019/configs/cpu-smoke-H.json` | `configs/camera-ready-tests/` |
| `camera-ready-20261019/*.json` | `manifests/camera-ready/` |
| `camera-ready-20261019/tests/` | `tests/camera_ready/` |
| `camera-ready-20261019/tools/` | `tools/camera_ready/` |
| `camera-ready-20261019/*.md` | `docs/camera-ready/` |

**Code changes from the move:**
- **Training source:** none. `source_tree_sha256` in `run_manifest.json` hashes `piper_train/` only.
- **Tests and tools:** only their path constants changed (`tests/camera_ready/_common.py`, `test_train_cpu.py`, `tools/camera_ready/*.py`).
