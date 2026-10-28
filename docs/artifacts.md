# Artifacts outside Git

Large and private material stays on the training workstation, in the same folder layout, and is git-ignored.
- Ignoring a file does not delete it.
- Hashes for every file of the record are in the local `manifests/file-manifest.json` and `SHA256SUMS`.
- Hashes for everything outside the dataset folders are also in the tracked `manifests/file-manifest-core.json`.
  - Its field `git.status` marks each entry: `tracked`, `sanitized_copy` or `local_only`.

| Group | Local path | Size | Why it is not tracked |
|---|---|---:|---|
| Processed dataset: normalised audio, linear spectrograms, F0 tensors | `datasets/processed/data_preprocessed` | about 22 GB (39,302 files) | Size; derived data |
| LJ Speech as used | `datasets/raw/home-dev-data` | about 3.6 GB (13,111 files) | Size; available from the dataset's site |
| Selected checkpoints (epochs 773 and 1079) | `runs/*/…/checkpoints/` | 1.72 GB (2 files) | Size; not distributed |
| TensorBoard events | `runs/**/events.out.tfevents.*` | about 4 GB (5 files) | Size; file names carry the host name |
| Generated evaluation WAVs | `evaluation/**.wav` | about 0.4 GB | Model outputs; release not reviewed |
| Silero VAD model | `code/**/norm_audio/models/silero_vad.onnx` | small | Third-party model file; obtain upstream |
| Git bundle of EdgeTTS (all refs) | `history/private/` | 0.24 GB | Private history; not scanned for secrets |
| Manuscript drafts, author notes, one draft copy | `evidence/manuscript/`, `history/notes/`, `history/prior-audits-2026-09-23/PitchFlowNet_Conference_Draft_2026-09-24.txt` | small | Unpublished; contact details |
| Working notes and earlier audits | `history/` (apart from `history/environment/`), `docs/camera-ready/` run notes | small | Internal working material |
| BanhmiTTS-port training log | `runs/candidates/BanhmiTTS-port/…/train.log` | 11 MB | Size; not one of the paper's runs |
| Originals of sanitized files | beside each `*.sanitized.*` copy | small | Machine paths |
| Full manifests | `manifests/file-manifest.json`, `SHA256SUMS` | 34 MB, 9 MB | Size (every dataset file is listed) |

## Pointing code at the data

The run records use absolute paths of the WSL training account:
- `dataset.jsonl`, the hparams and the launch scripts use `<WSL_DEV_HOME>/data_preprocessed/...`;
- the audio paths inside `dataset.jsonl` use `<WSL_DEV_HOME>/data/wavs/...`.

To use the local copies, choose one of these and **do not edit the copied originals**:
- Create symbolic links at the recorded locations that point to `datasets/processed/data_preprocessed` and `datasets/raw/home-dev-data`.
- Write a remapped copy of `dataset.jsonl` elsewhere and pass that copy (or its folder) to the code.

`tools/verify_pitchflownet.py` remaps the paths in memory to check that every file referenced by `dataset.jsonl` exists in the local copy.

## Checkpoints

**Load checkpoints only with care:**
- They are Lightning pickles. Their hparams contain `PosixPath` objects.
- The record itself only scanned their pickle opcodes (`tools/scan_checkpoints_static.py`). It never unpickled them.
