# Pitch-FlowNet

Evidence and source record for the paper **"PitchFlowNet: An Empirical Study of Architectural
Integration in Single-Stage Text-to-Speech"**. According to the author's receipt, it was submitted to
SOICT 2026 as submission 4478. Acceptance or publication is not established here.

The repository catalogues, for the two runs compared in the paper (a retrained Piper baseline and the
full PitchFlowNet model):
- the source candidates;
- the configurations;
- training records;
- stored evaluation outputs.

It is a record of what exists. **It is not a verified reproduction package.**

- Which runs, and why: [docs/project-identity.md](docs/project-identity.md).
- Working rules for this repository: [AGENTS.md](AGENTS.md).

## What Git holds

- **Tracked:** small, sanitized content only.
  - Originals that contain machine paths stay on the workstation and are git-ignored.
  - Their `*.sanitized.*` copies are tracked beside them.
- **Not tracked:** datasets, checkpoints, audio, TensorBoard events, manuscript drafts and private history.
  - These stay local; see [.gitignore](.gitignore).
