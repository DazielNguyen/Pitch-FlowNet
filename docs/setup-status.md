# Setup status

Status: **record only.** Nothing here has been made runnable, and nothing has been run, except:
- the static checkpoint scan;
- the TensorBoard event scan;
- the split reconstruction;
- the record verification;
- the tracked-package checks.

No environment was installed and no model code was imported to prepare the Git repository.

| Area | State | Needed later |
|---|---|---|
| Code | Two historical candidate commits, launch scripts, the current Windows source (`a73a897`) and the current WSL working tree (`85cab10` + uncommitted) are copied unchanged (Git tracks sanitized copies of the files that hold machine paths) | Choose a source baseline and record the choice. Imports expect `src/python` on the path; the Cython MAS extension must be built (`build_monotonic_align.sh`); the Silero VAD model file path. |
| Environment | Current `edgetts` conda env: Python 3.10.20, torch 2.12.1+cu130; checkpoints record Lightning 2.6.5. `pip freeze` / `conda list` captured 2026-10-12 | Not a training-time record. Recreate from the captured lists. |
| Data paths | `dataset.jsonl` and hparams hold absolute paths (`<WSL_DEV_HOME>/data_preprocessed/…`, `<WSL_DEV_HOME>/data/wavs/…`) | Remap or symlink before any re-run. Do not edit the copied originals. |
| Checkpoints | The two selected checkpoints are copied. Others are catalogued (`manifests/checkpoint-manifest.json`) | Load only with `torch.load(weights_only=…)` after checking that the allowed globals are needed (PosixPath in hparams). |
| Evaluation | Stored outputs only. The harness (`eval_harness.py`) is in the source candidates | Whisper/UTMOSv2 scorer versions and threads are unrecorded. |
| Statistics | Means and Wilcoxon reproduce from the CSVs (`evaluation/statistics/history_archived_pair.*`) | The BCa/dz script is missing. |
| Git | Repository initialised by the author; history starts with the commits that add this record. Only sanitized, small content is tracked (`.gitignore`, `manifests/sanitization/`). The EdgeTTS history stays in the private bundle | Choose the remote's visibility and a project licence before any push. Clone, then run `tools/check_tracked_package.py` and the tests |
