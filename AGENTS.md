# Working in Pitch-FlowNet

This repository is the evidence and source record for the PitchFlowNet conference paper
(SOICT 2026, submission 4478).

- Discuss in Vietnamese with the author; write documentation and manuscript text in English.
- Treat everything under `runs/`, `datasets/`, `evaluation/`, `configs/`, `code/` and `history/` as copied originals.
  - Do not edit them.
  - Add derived files beside them and record them in `manifests/`.
- Camera-ready work lives in its own subfolders, which may be edited: `code/camera-ready-source/`, `configs/PFN-*/`,
  `configs/camera-ready-*/`, `manifests/camera-ready/`, `tests/camera_ready/`, `tools/camera_ready/`, `docs/camera-ready/`.
  See `docs/camera-ready/README.md`.
- Keep evidence classes apart: contemporaneous records, direct artifacts, historical source candidates,
  current source, reconstructions, author accounts.
  - A matching hash or file name does not prove what a process executed.
- Manuscript text, README files and notes are not evidence of results.
- Do not train, synthesize, score, export, quantize, benchmark or preprocess on your own initiative.
- Do not unpickle checkpoints unsafely.
- Never invent results, citations or experimental details.
  - UTMOSv2 is a predicted score, not MOS.
  - A non-significant p-value does not establish equivalence.

## Git

- Only sanitized content is tracked.
  - Originals that contain machine paths stay local and git-ignored.
  - Their `*.sanitized.*` copies are tracked beside them, and `manifests/sanitization/*.json` maps each pair by SHA-256.
  - After adding or changing a file, run `python tools/sanitize_for_git.py`.
  - Then run `python tools/check_tracked_package.py` before committing.
- Never track the following:
  - datasets, checkpoints, ONNX, WAVs or TensorBoard events;
  - `history/private/`, `evidence/manuscript/` or `history/notes/`;
  - credentials.
- Stage explicit paths; do not run `git add .` without reviewing what it would add.
- Do not add AI attribution to commits or pull requests: no `Co-Authored-By` trailer for an AI assistant and no "Generated with" line.
- Do not push, upload or publish without the author's request.
