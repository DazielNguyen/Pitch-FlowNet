# Transfer to the Mac

## Evidence ZIP (small)

`pitchflownet-conference-evidence-2026-10-12.zip`, with a `.zip.sha256` file next to it.

**What it contains:**
- the documents and manifests;
- the configs (hparams);
- the training logs;
- the stored test entries and the reconstructed split;
- the per-sentence evaluation CSVs and summaries (without WAVs);
- the statistics recomputation;
- the figure scripts and PDFs;
- the code candidates and current sources (without binaries);
- the commit logs and environment lists;
- the derived scans;
- the tools.

**Sanitized copies:**
- Text files carry machine paths replaced by placeholders:
  - `<WSL_DEV_HOME>/` for the home directory of the WSL training account;
  - `<WSL_PROJECT_HOME>/` for the home directory of a second WSL account;
  - `<WINDOWS_WORKSPACE>/` for the Windows workspace;
  - `<WINDOWS_USER_HOME>/` for the Windows user profile.
- `zip-sanitization-map.json` lists, for every changed file, the original and sanitized SHA-256.
- The originals stay in this folder on the PC.

**Not included in the ZIP:** the private Git bundle, datasets, checkpoints, TensorBoard events and WAVs.

**Relation to the Git repository:**
- The ZIP was built on 2026-10-12, before the Git layer existed. It has its own sanitizer and map.
- Git tracks a separate set of sanitized copies (`*.sanitized.*`, mapped in `manifests/sanitization/`).
  - That set also replaces the workstation host name.
  - It leaves out the manuscript drafts and author notes, which the ZIP includes.
- For sharing, a clone of the repository is the reviewed form. The ZIP is a private hand-off.

```sh
shasum -a 256 -c pitchflownet-conference-evidence-2026-10-12.zip.sha256
unzip pitchflownet-conference-evidence-2026-10-12.zip && cd pitchflownet-conference-evidence-2026-10-12
shasum -a 256 -c SHA256SUMS
```

## Large groups (transfer separately when needed)

They stay complete in `FPT-Graduation-Project/Pitch-FlowNet` on the PC (backup under WSL at `<WSL_DEV_HOME>/Pitch-FlowNet`). Hashes for every file are in `manifests/file-manifest.json`.

| Group | Path | Size | Files |
|---|---|---:|---:|
| Processed dataset (audio, spectrogram and F0 tensors) | `datasets/processed/data_preprocessed` | about 22 GB | 39,302 |
| Raw LJ Speech as used | `datasets/raw/home-dev-data` | about 3.6 GB | 13,111 |
| Selected checkpoints | `runs/*/…/checkpoints/` | 1.72 GB | 2 |
| TensorBoard events | `runs/**/events.out.tfevents.*` | about 4 GB | 5 |
| Evaluation WAVs | `evaluation/**.wav` | about 0.4 GB | about 1,700 |
| Private Git history | `history/private/EdgeTTS-PitchFlowNet-all-refs.bundle` | 0.24 GB | 1 |

**Suggested method** (no external service): make one tar per group on the PC, then copy it over the local network or an external drive.

```sh
tar -cf pfn-checkpoints.tar runs/baseline/02_Baseline_VanillaVITS/lightning_logs/version_0/checkpoints runs/full/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints
sha256sum pfn-checkpoints.tar > pfn-checkpoints.tar.sha256
```

After extraction, compare each file with `manifests/file-manifest.json` (`destination_sha256`).

`dataset.jsonl` and the hparams contain absolute `<WSL_DEV_HOME>/...` paths. Remap them on the Mac with a symlink or a path rewrite applied to a copy, never to the original.
