# Working in Pitch-FlowNet

This repository is the evidence and source record for the PitchFlowNet conference paper
(SOICT 2026, submission 4478).

- Discuss in Vietnamese with the author; write documentation and manuscript text in English.
- Treat everything under `runs/`, `datasets/`, `evaluation/`, `configs/`, `code/` and `history/` as copied originals.
  - Do not edit them.
  - Add derived files beside them and record them in `manifests/`.
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
- Do not push, upload or publish without the author's request.

## Commits (binding rule of the author, for every assistant: Claude Code, Codex or any other)

- **No AI attribution, ever.** Do not add any of the following:
  - a `Co-Authored-By` trailer for Claude, Codex or any AI tool;
  - "Generated with/by …" text, robot emoji or links to an assistant session.
  - The repository hook `tools/hooks/commit-msg` refuses such messages. Activate it once per clone with `git config core.hooksPath tools/hooks`.
  - `.claude/settings.json` turns Claude Code attribution off.
- **Commit identity:** commit with the author's Git identity. Do not change `user.name` or `user.email`.
- **One scope per commit.** Split unrelated changes, and stage explicit paths.
- **Message format:**
  - **Subject:** `type(scope): summary`, at most 72 characters, imperative, no trailing period. Types: `feat fix docs test chore refactor perf build ci style results`.
  - **Blank line.**
  - **Body:** wrapped prose or short bullets saying what changed and why. Then what was checked and with which result, and what is still open or failing. Never claim a check that was not run.
- **Before each commit:**
  - review `git diff --cached --stat` and the staged content;
  - run `git diff --cached --check`;
  - run `python tools/check_tracked_package.py`;
  - make sure no data, checkpoint, WAV or private file is staged.
- **History:** do not amend, rebase or force-push published history.
