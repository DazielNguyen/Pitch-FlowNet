# Data rights and access

| Item | Origin | Rights | In Git |
|---|---|---|---|
| `datasets/raw/home-dev-data` (LJ Speech 1.1 WAVs, `metadata.csv`) | Keith Ito, LJ Speech Dataset (keithito.com) | Public domain per the dataset's own statement. Keep the attribution to LJ Speech and LibriVox | No (size). Obtain from the dataset's site |
| `datasets/processed/data_preprocessed` | Derived from LJ Speech by the project's preprocessing (2026-07-01) | Derived data. The underlying audio is public domain; the derived tensors carry no project licence until the authors decide | No (22 GB). It also holds absolute paths of the WSL account |
| `datasets/splits` (500 test entries, test sentences, reconstructed split) | LJ Speech transcripts and IDs; split reconstructed by `tools/reconstruct_split.py` | LJ Speech text is public domain | Yes (sanitized where paths occur) |
| Checkpoints (`*.ckpt`) | Training runs | Author decision pending | No |
| Generated WAVs in `evaluation/` | Model outputs | Author decision pending; not for public release without review | No |
| Per-sentence CSVs, summaries, figures | Evaluation runs and plotting scripts | Authors; licence open | Yes |
| Code in `code/` | EdgeTTS (fork of rhasspy/piper, MIT) with project changes; BanhmiTTS port | Piper MIT (`LICENSE.md` kept with each snapshot). Project changes belong to the authors; licence open. See `THIRD_PARTY_NOTICES.md` | Yes, without compiled outputs and the Silero VAD ONNX model |
| `history/private/EdgeTTS-PitchFlowNet-all-refs.bundle` | Complete Git history of EdgeTTS | Private. Not scanned for secrets | No |
| Training logs, hparams, TensorBoard events | Training runs | Authors | Logs and hparams: sanitized copies only. Events: no |
| Manuscript drafts (`evidence/manuscript`) and author notes (`history/notes`) | Authors | Authors. Unpublished; they carry contact details | No |

**Access:**
- **Location:** the record lives on the training workstation in the Windows project folder `FPT-Graduation-Project/Pitch-FlowNet`.
  - OneDrive syncs that folder to the author's own account.
  - A backup copy built first under WSL (`<WSL_DEV_HOME>/Pitch-FlowNet`) stays on the same machine.
- **Git:** the folder is a Git repository.
  - It tracks only sanitized, small content (`.gitignore`, `manifests/sanitization/`).
  - Nothing has been pushed, published or shared.
- **Placeholders:** machine paths and the host name in tracked files are replaced by placeholders such as `<WSL_DEV_HOME>`.
  - The originals stay local, and each pair is mapped by SHA-256.
