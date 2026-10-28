# Missing information

| Missing | Affects | Notes / what would close it |
|---|---|---|
| Source code as executed (working-tree snapshot) for either run | Every code-level claim | Only commits before (`6033be5`) and after (`badf381`) exist. `6033be5` lacks the VITS2 switches that the full run used. No recovery is possible from the repository. |
| Exact launch command lines | Launch provenance | The committed `.wsl_*.sh` versions are candidates. The `.wsl_train.sh` resume path does not match the `version_2` hparams. `<WSL_DEV_HOME>/.bash_history` holds 96 bytes and was not copied. |
| Full-model sessions `version_0` and `version_1` (directories, events, checkpoints) | Training history, duration, initialization check | `version_2` hparams name `version_0/last.ckpt` as the resume source. The logs `train.log` and `train_h.log` remain. |
| Full-model duration 103.31 h | Training-cost claim | Not reproducible: `version_2` events cover 99.47 h only. |
| Why the selected checkpoints are not the minimum of the logged `val_loss_mel` series | Selection claim | The monitor is `val_loss_mel` (min). The logged series is rank 0 only. Possible cause: per-rank metrics under DDP. Would need the per-rank values, which were not logged. |
| BCa intervals, Cohen's dz, 20,000-resample bootstrap (script and output) | Statistics paragraph | Not found in EdgeTTS, `<WSL_DEV_HOME>`, `paper_revisions` or `Compact`. The per-sentence CSVs exist, so a recomputation is possible later. It was not done here. |
| Ground-truth evaluation script (`gt_wer_utmos_500.py`) | Ground-truth WER/UTMOSv2 values | `evaluation/ground-truth/eval_pitchflownet_500_gt_run.sanitized.log` shows that it ran from an assistant scratchpad directory on 2026-09-24. That directory is empty now, and the script is not in the record. The outputs (CSV, summary JSON) exist. |
| Logs of the `eval_h_vs_baseline` run (command, environment, threads) | Evaluation protocol, timing | Only the outputs (CSV, `summary.json`, WAVs) exist. Candidate harness: `src/python/piper_train/eval_harness.py` at the candidate commits. |
| CPU thread / batch settings for timing | RTF claim | Not recorded. The i7-14700 CPU is an author statement. |
| GPU model in run records | Hardware claim | The run records do not name GPUs. The current system record lists the workstation GPUs. |
| Run-time split file | Split claim | None was written. The split is reconstructed and corroborated (`datasets/splits/`). |
| Parameter counts and weight sizes as run-time records | Model-size table | Only a later audit of the checkpoints (2026-09-23, kept locally). The camera-ready recomputation of the inference parameters is in `docs/camera-ready/ablation-final.md`. |
| Complete matched ablations | Ablation claims | Config E is incomplete (clip 0.5, stopped at epoch 508). Config H-773 stopped after about 2 minutes. |
| Preprocessing run record beyond `preprocess.log` | Data-processing claims | The preprocessing code that ran on 2026-07-01 was not snapshotted. Candidate code is in `historical-candidates`. |
| WSL `<WSL_DEV_HOME>/BanhmiTTS/preprocessed` (22 GB) and run checkpoints | BanhmiTTS port | Catalogued only; this is not the evaluated pair. |

**References to local-only material:**
- Some rows cite `history/notes/` (author notes) or the manuscript drafts in `evidence/manuscript/`.
- These stay on the workstation and are not tracked in Git. Their hashes are listed in `manifests/file-manifest-core.json`.
