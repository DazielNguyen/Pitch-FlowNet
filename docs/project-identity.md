# Project identity

**Paper:** "PitchFlowNet: An Empirical Study of Architectural Integration in Single-Stage Text-to-Speech" (SOICT 2026, submission 4478).

**Other names in manuscript drafts and folders:** MRFlowNet (renamed to PitchFlowNet on 2026-09-23); Banhmi-TTS / banhmiTTS (an earlier name of the full model); "Config H" (the full configuration in the project's ablation plan).

The code lives in the repository **EdgeTTS** on the Windows workspace. Its remote is `MardeusVN/PitchFlowNet`, and it is a fork of rhasspy/piper. The project was renamed in commit `6d0522b` (2026-09-24).

Training ran under WSL in `<WSL_DEV_HOME>/PitchFlowNet/src/python` with the conda environment `edgetts`. The code was copied there by `rsync` from the Windows working tree (`.wsl_train.sh`).

## The two runs

| | Baseline ("Piper baseline") | Full model ("PitchFlowNet") |
|---|---|---|
| Run directory | `<WSL_DEV_HOME>/02_Baseline_VanillaVITS` | `<WSL_DEV_HOME>/01_Baseline_BigVgan_VITS2_FO`, session `version_2` |
| Flags (hparams) | `use_bigvgan`, `use_vits2`, `use_f0` false | all three true |
| Start / end | 2026-07-09 23:41 → 2026-07-13 04:57 (+07); TensorBoard span 77.26 h | Start of the from-scratch session not dated in the log; the run directory exists since 2026-07-03. Earlier attempts in `train.log` were aborted or killed. `version_2` hparams were written 2026-07-05; the session ended 2026-07-09 13:20. TensorBoard (`version_2` only) spans 99.47 h, epochs 43–1099 |
| Initialization | From scratch (`resume_from_checkpoint` null) | `train_h.log` sequence: a from-scratch session ends in `RuntimeError`, then two resumes from `version_0/last.ckpt`; the last reaches `max_epochs=1100` |
| Selected checkpoint | `best-epoch=773-val_loss_mel=20.2158.ckpt`, SHA-256 `54d86719…` | `best-epoch=1079-val_loss_mel=19.2943.ckpt`, SHA-256 `8e6eacfa…` |
| Evaluation used in the paper | `eval_h_vs_baseline` (2026-07-13), 500 test sentences | same |

## How the identification was made

- **Contemporaneous records:**
  - Lightning `hparams.yaml` of each run: component flags, clipping, epochs, seed, devices, precision, dataset path.
  - Training logs.
  - TensorBoard events: the validation value at the selected epochs equals the file name; the audio examples are identical.
  - `eval_h_vs_baseline/summary.json`, which names both checkpoints and gives every evaluation number in the paper's table.
- **Direct artifacts:** static scans of the checkpoints (no unpickling) for epoch, global step and module name groups.
- **Located through earlier assistant audits:** the 2026-09-23 audits in `history/prior-audits-2026-09-23` point to the same checkpoints. Their conclusions are not used as evidence.

**Not this paper:**
- the journal's Parallel-IR / Sequential-IR models (epochs 1386 / 1442);
- the HiFi-MobiNet baseline (epoch 1489);
- the 2026-10 `Piper_no_VITS2_cpn` run;
- the published Piper checkpoint.

## Sources that ran versus sources that exist now

**No source snapshot was saved at run time.**

- **Launch mechanism:** the launch scripts synced the Windows working tree into WSL and excluded `.git`.
- **Commits around the runs:** `6033be5` (2026-06-30) predates both runs but **lacks** the `use_vits2` / noised-MAS code that the full run's hparams require. `badf381` (2026-07-10) is the first commit containing it and was committed after both launches.
- **What actually ran:** an uncommitted working tree that is no longer available.
- **Folder `code/historical-candidates/`:** both commits, as `HISTORICAL_SOURCE_CANDIDATE`.
- **Folder `code/current-source/`:** the code as it is now, as `CURRENT_SOURCE`.
