# Runs on MardeusPC (RTX 5070)

**Scope.**
- This page is a record of each training run on this machine: launches, interruptions, resumes and final checkpoints.
- The run directories themselves are outside Git, in `<WSL_DEV_HOME>/outputs/runs/<run_id>/`.
- How to run: `runbook-MardeusPC.md`. What to run: `experiment-plan.md`.

## PFN-Full-275ep-s1234

| Item | Value |
|---|---|
| Config | `configs/PFN-Full/PFN-Full-275ep.json`, 275 epochs = 215,050 G updates |
| Start | 2026-10-20 02:29 +07, commit `f1495c0` (= `b0390ef` after the message rewrite, same tree; `manifests/camera-ready/commit-map-2026-10-20.json`), clean tree |

### Interruption 1: 2026-10-20 20:30:57 +07

**What happened.**
- The run stopped in epoch 126, at batch 458 of 782, with `torch.AcceleratorError: CUDA error: unknown error` (`cudaErrorUnknown`) in `batch.to(device)`. The process exited with status 1.
- The last complete checkpoint was epoch 125 (98,532 G updates), saved 20:24:57: `last.ckpt` and `epoch-epoch=125.ckpt`. Lost: the partial epoch 126, about 6 minutes of training.
- **Not RAM.** WSL had 23 GB free, Windows 21 GB free; swap was nearly unused; there was no OOM in the kernel log.

**Windows events around it:**
- 19:41:45: `LiveKernelEvent` 141 (Windows Error Reporting). Code 141 is a GPU engine timeout, which the driver recovered from (TDR). The run kept going.
- 19:59: the GPU log shows used memory falling from 11.6 GB to 2.2 GB for one sample, then back to 11.6 GB; training continued.
- 20:29:36: Kernel-Power 566, "the system session has transitioned from 37 to 39", reason `InputHid`. The display had been off for about 2.4 h, under the 5-minute display timeout on AC, and user input turned it back on. The CUDA error followed about 80 s later.

**Probable cause** (inferred from the timing, not proven): the RTX 5070 also drives the display. The display power transition, possibly together with the earlier TDR, invalidated the CUDA context of the WSL process.

**Actions taken:**
- The display timeout on AC was set to never (`powercfg /change monitor-timeout-ac 0`). It can be reverted with `... 5`. Sleep on AC was already off.
- CUDA was checked: a 4096 × 4096 matmul ran with a finite result.
- `last.ckpt` passed `check_ckpt.py`: epoch 125, 98,532 G updates.
- **Resumed** 2026-10-20 20:44:39 +07 with `resume_train.sh configs/PFN-Full/PFN-Full-275ep.json --execute`, after the author's OK. Same run directory; no second copy, no restart from scratch.

**Record defect found.**
- `run_manifest.json` shows the first attempt with an empty status, not "failed".
- `train.py` updates the attempt in a `finally` block that reads `best_model_score`, a CUDA tensor. With the CUDA context gone, that read raised the same error, so the failure was not written.
- Training itself is not affected. This page is the record of the interruption.
- A fix is open: convert the score without touching the device after a failure, or catch the error and still write the status.

**Fallback if the error comes back:**
- raise the Windows TDR delay (registry `TdrDelay`; needs administrator rights and a reboot);
- or move the display to another output.

### Interruption 2: 2026-10-21 01:09:14 +07

**What happened.**
- Same failure as interruption 1: `CUDA error: unknown error` in epoch 156, at batch 303 of 782; exit status 1.
- The last complete checkpoint was epoch 155 (121,992 G updates). Lost: the partial epoch 156, and the idle time until the resume (≈ 2 h 10 min).
- The display timeout was already "never", so a display wake is not the trigger this time.

**Pattern.** Windows Error Reporting logged `LiveKernelEvent` 141 (GPU engine timeout, driver reset) eight times since training started:
- 20/10 01:44 (twice), 03:56, 07:13, 11:39, 15:34, 19:41 and 23:04;
- roughly every 3–4 h.

Most resets left the run alive; two of them, or events like them, killed its CUDA context. The likely cause is the Windows TDR watchdog (2 s by default) on a GPU that drives the display and runs long compute work. Inferred, not proven.

**Resumed** 2026-10-21 03:21:03 +07 from epoch 155, after a CUDA check and `check_ckpt.py`.

**Proposed, not done:**
- (1) Raise `TdrDelay`, which needs administrator rights and a reboot.
- (2) A watchdog that resumes automatically after this specific error, within limits. It needs the author's standing OK.

### Automatic resume enabled: 2026-10-21

- The author chose the watchdog (`tools/camera_ready/auto_resume.sh`, see `runbook-MardeusPC.md`) and no `TdrDelay` change, to avoid a reboot.
- It runs in tmux `cr-watch-PFN-Full-275ep-s1234`, with at most 5 automatic resumes, each recorded in `checkpoint_ledger.jsonl`.
- Tested beforehand on a simulated run with five cases: run alive, CUDA unknown error (resume, dry run), another error (stop), normal end (stop), limit reached (stop).

### Budget change to 200 epochs: 2026-10-21 05:12–05:18 +07

The author lowered the ablation budget to 200 epochs (`experiment-plan.md`). The run was at epoch 168 of 275 and continues in the same directory, ending at epoch 200.

**Script:** `<WSL_DEV_HOME>/pfn-sync/switch_full_to_200.sh`, log `<WSL_DEV_HOME>/outputs/logs/switch-full-200ep.log`. Steps:
1. Stopped the watchdog.
2. Waited for the epoch-168 `last.ckpt`, saved 05:15:43 (132,158 G updates).
3. Sent SIGINT to the training process at 05:15:54. It had not exited after 120 s, so SIGTERM followed at 05:17:54; the session ended at 05:18:14. Nothing of epoch 169 had been saved, so nothing was lost but its first seconds.
4. Resumed at 05:18:32 with `resume_train.sh configs/PFN-Full/PFN-Full-275ep.json --stop-after-epochs 200 --execute`. `last.ckpt` passed `check_ckpt.py` (epoch 168).
5. Restarted the watchdog with `--stop-after-epochs 200`.

**Records.** `run_manifest.json` now lists four attempts. The first three have an empty status:
- attempts 1 and 2 ended with the CUDA error, as described above;
- attempt 3 was stopped by SIGTERM, which ends Python before the attempt record is written.

This page is the record of all three.

### Automatic resumes 3 and 4: 2026-10-21

The watchdog (`--stop-after-epochs 200`) handled both. Each time it waited 60 s, checked CUDA, checked `last.ckpt` and resumed; each resume is in `checkpoint_ledger.jsonl` (`"event": "auto_resume"`).

| # | CUDA error | Where | Resumed from | Resumed at |
|---:|---|---|---|---|
| 3 | 21/10 07:07:33 | epoch 181, batch 184 of 782 | epoch 180 | 07:10:02 (auto resume 1) |
| 4 | 21/10 09:44:04 | epoch 198, batch 452 of 782 | epoch 197 | 09:47:42 (auto resume 2) |

### End: 2026-10-21 10:08:36 +07 (exit 0)

**Result.**
- `Trainer.fit stopped: max_epochs=200 reached`: 200 epochs, 156,400 G updates, 2,500,000 samples.
- **Best:** epoch 188, `val_mel_l1_global` 21.3153. Last epoch (199): 21.3991.

**Attempts.** `run_manifest.json` lists six:
- attempts 1–5 have an empty status: four ended with the CUDA error, one with SIGTERM (the budget change);
- attempt 6 is `completed`.

**Interruptions in total:** four, all `CUDA error: unknown error`. Two were resumed by hand (20/10 20:44, 21/10 03:21) and two by the watchdog. Training lost is the partial epochs, plus ≈ 2 h 10 min idle before the second resume.

**Archive.** Finalized by the watchdog at 10:10:02 into `<WSL_DEV_HOME>/outputs/final/PFN-Full-275ep-s1234/`, read-only, with `final.json`. The training source tree is `1ee13cb4…`.

**Evaluation** (`eval-protocol.md`, 21/10 12:18). Best checkpoint, test split, 500 utterances; results outside Git, in `<WSL_DEV_HOME>/outputs/eval/PFN-Full-200ep-best/`:

| Metric | Value |
|---|---:|
| WER, mean / median | 11.35 % / 7.41 % |
| UTMOSv2, predicted, mean | 3.516 |
| MCD, mean | 25.64 |
| F0 predictor RMSE / corr | 0.540 / 0.186 |
| Reference UTMOSv2 | 3.831 |

The inline RTF (0.068) is not a reportable RTF: other evaluations ran beside it.

## PFN-NoF0-200ep-s1234

| Item | Value |
|---|---|
| Config | `configs/PFN-NoF0/PFN-NoF0-200ep.json`, 200 epochs = 156,400 G updates |
| Start | 2026-10-21 12:22:14 +07, after the author's OK. Commit `33dca31`, clean tree. Training source tree `1ee13cb4…`, the same as `PFN-Full-275ep-s1234` |
| Watchdog | `cr-watch-PFN-NoF0-200ep-s1234`, from the launch (limit 5 automatic resumes) |
| GPU idle before it | 21/10 10:08 → 12:22. The author chose to finish the Full evaluation first |

**Interruptions so far:** three, all `CUDA error: unknown error`, all resumed by the watchdog within about 3 minutes.

| # | CUDA error | Where | Resumed from | Resumed at |
|---:|---|---|---|---|
| 1 | 21/10 18:41:09 | epoch 44, batch 278 of 782 | epoch 43 | 18:44:11 (auto resume 1) |
| 2 | 21/10 21:06:26 | epoch 59, batch 758 of 782 | epoch 58 | 21:09:52 (auto resume 2) |
| 3 | 22/10 01:18:39 | epoch 87, batch 557 of 782 | epoch 86 | 01:21:35 (auto resume 3) |

As for the Full run, the attempts that ended with the CUDA error have an empty status in `run_manifest.json`.

### End: 2026-10-22 17:18:16 +07 (exit 0)

**Result.**
- 200 epochs, 156,400 G updates. No interruption after the third.
- **Best:** epoch 192, `val_mel_l1_global` 21.1037. Last epoch (199): 21.2832.

**Attempts.** `run_manifest.json` lists four:
- attempts 1–3 ended with the CUDA error (empty status);
- attempt 4 is `completed`, 22/10 01:21:35 → 17:18:00.

**Archive.** Finalized by the watchdog at 17:19:16 into `<WSL_DEV_HOME>/outputs/final/PFN-NoF0-200ep-s1234/`, read-only.

### Evaluation and the F0 contrast: 2026-10-22

**Chain.** The one-off chain `<WSL_DEV_HOME>/pfn-sync/after_nof0.sh` (tmux `cr-chain`, author's OK on 22/10; log `outputs/logs/chain-after-nof0.log`) ran after the archive:
- 17:20:04: updated the WSL clone to `229b9b5` (clean);
- 17:20:08: launched `PFN-Baseline-VITS2-200ep` with its watchdog;
- 17:22:47: started the evaluation of the best checkpoint (`eval-protocol.md`, label `PFN-NoF0-200ep-best`);
- 19:01:49: ran `compare`, after the evaluation finished at 18:59:51. The evaluation took 2,763 s for synthesis and scoring, alongside the new training run.

**Results.** Test split, 500 utterances. Results outside Git, in `<WSL_DEV_HOME>/outputs/eval/`.

| Metric | `PFN-Full-200ep-best` (epoch 188) | `PFN-NoF0-200ep-best` (epoch 192) | Paired Wilcoxon (two-sided, per utterance) |
|---|---:|---:|---|
| WER mean / median | 11.35 % / 7.41 % | 13.02 % / 8.70 % | p = 0.0077 |
| UTMOSv2 mean / median (predicted, not MOS) | 3.516 / 3.520 | 3.189 / 3.203 | p = 1.9 × 10⁻⁵⁸ |
| MCD mean / median | 25.64 / 25.46 | 25.58 / 25.48 | not tested (the historical comparison tests WER and UTMOSv2 only) |
| F0 predictor RMSE / corr | 0.540 / 0.186 | — (no F0 group) | |
| Reference UTMOSv2 | 3.8308 | 3.8308 | |

The comparison file is `compare-PFN-Full-200ep-best_PFN-NoF0-200ep-best.json`.

**Limits:**
- **One training seed per run.** The Wilcoxon test covers variation across utterances, not across training runs.
- **UTMOSv2 is a predicted score.**
- **Inline RTF is not reportable.** It was 0.093, measured beside a training run.
- **Multiple comparisons.** A correction across the ablation comparisons, if any, has to be fixed before the other results are seen.

## PFN-Baseline-VITS2-200ep-s1234

| Item | Value |
|---|---|
| Config | `configs/PFN-Baseline-VITS2/PFN-Baseline-VITS2-200ep.json`; B off, V on, F off; 156,400 G updates |
| Start | 2026-10-22 17:20:08 +07, by the chain above (author's OK). Commit `229b9b5`, `dirty_paths` `[]`, training source tree `1ee13cb4…` |
| Watchdog | `cr-watch-PFN-Baseline-VITS2-200ep-s1234` (limit 5 automatic resumes) |

### End: 2026-10-23 18:27:12 +07 (exit 0)

**Result.**
- 200 epochs, 156,400 G updates, in one attempt: no interruption, no automatic resume.
- **Best:** epoch 188, `val_mel_l1_global` 22.2346. Last epoch (199): 22.7031.

**Archive.** Finalized by the watchdog at 18:29:59 into `outputs/final/PFN-Baseline-VITS2-200ep-s1234/`.

**Chain.** The one-off chain `<WSL_DEV_HOME>/pfn-sync/after_baseline_vits2.sh` (tmux `cr-chain-f0`, author's OK "ok chain F0" on 23/10; log `outputs/logs/chain-after-baseline-vits2.log`):
- updated the WSL clone to `b91fff3`;
- launched `PFN-Baseline-F0-200ep` at 18:30;
- evaluated the best checkpoint, 18:32 → 20:06.

**Evaluation** (locked protocol, test split, 500 utterances, label `PFN-Baseline-VITS2-200ep-best`):

| Metric | Value |
|---|---:|
| WER mean / median | 11.05 % / 7.14 % |
| UTMOSv2 mean / median (predicted) | 3.372 / 3.382 |
| MCD mean / median | 25.62 / 25.50 |
| Reference UTMOSv2 | 3.8308 |

- No comparison was run here. Its add-one reference, `PFN-Baseline`, was trained on the PC.
- The inline RTF (0.078) is not reportable: it was measured beside a training run.

## PFN-Baseline-F0-200ep-s1234

| Item | Value |
|---|---|
| Config | `configs/PFN-Baseline-F0/PFN-Baseline-F0-200ep.json`; B off, V off, F on; 156,400 G updates |
| Start | 2026-10-23 18:30:42 +07, by the chain above. Commit `b91fff3`, `dirty_paths` `[]`, training source tree `1ee13cb4…` |
| Watchdog | `cr-watch-PFN-Baseline-F0-200ep-s1234` (limit 5 automatic resumes) |

### End: 2026-10-24 16:24:26 +07 (exit 0)

**Result.**
- 200 epochs, 156,400 G updates, in one attempt: no interruption, no automatic resume. About 6.4 min per epoch at the end.
- **Best:** epoch 195, `val_mel_l1_global` 22.5705. Last epoch (199): 23.1776.

**Archive.** Finalized by the watchdog at 16:24:58 into `outputs/final/PFN-Baseline-F0-200ep-s1234/`.

**Evaluation.** Started automatically by `<WSL_DEV_HOME>/pfn-sync/eval_after_baseline_f0.sh` (tmux `cr-eval-wait-f0`), 16:26 → 17:20. Locked protocol, test split, 500 utterances, label `PFN-Baseline-F0-200ep-best`:

| Metric | Value |
|---|---:|
| WER mean / median | 13.49 % / 7.14 % |
| UTMOSv2 mean / median (predicted) | 3.060 / 3.096 |
| MCD mean / median | 26.73 / 26.66 |
| F0 predictor RMSE / corr | 0.620 / 0.192 |
| Reference UTMOSv2 | 3.8308 |

- No comparison was run here; its reference, `PFN-Baseline`, is on the PC.
- The inline RTF (0.053) was measured with no training run beside it. It is still not the declared-thread RTF measurement (T15), so it is not reported as RTF.

**All four MardeusPC runs are trained and evaluated.** Training on this machine is complete.

## Final checkpoints of every run

When a run ends with exit 0, `tools/camera_ready/finalize_run.sh` copies its checkpoints to `<WSL_DEV_HOME>/outputs/final/<run_id>/`:
- `best-epoch=…ckpt`, the minimum `val_mel_l1_global`, which is the checkpoint used for evaluation;
- `last-epoch=199.ckpt`, the last epoch;
- `run_manifest.json`, `checkpoint_ledger.jsonl`, `val_diagnostics.jsonl`;
- `SHA256SUMS` and `final.json`.

**Safeguards:**
- The copies are real copies, not hard links: Lightning rewrites `last.ckpt` in place.
- Each copy is checked against its source by SHA-256.
- Everything is made read-only.
- The watchdog runs this step itself when the run ends with exit 0. Without the watchdog, run it by hand.

| Run | Final folder | best | last | Status |
|---|---|---|---|---|
| `PFN-Full-275ep-s1234` (to epoch 200) | `outputs/final/PFN-Full-275ep-s1234/` | `best-epoch=188-val_mel_l1_global=21.3153.ckpt`, SHA-256 `95e6d6c9…` | `last-epoch=199.ckpt`, SHA-256 `f600d86f…` | finalized 21/10 10:10 |
| `PFN-NoF0-200ep-s1234` | `outputs/final/PFN-NoF0-200ep-s1234/` | `best-epoch=192-val_mel_l1_global=21.1037.ckpt`, SHA-256 `8f8570c6…` | `last-epoch=199.ckpt`, SHA-256 `1ed42171…` | finalized 22/10 17:19 |
| `PFN-Baseline-VITS2-200ep-s1234` | `outputs/final/PFN-Baseline-VITS2-200ep-s1234/` | `best-epoch=188-val_mel_l1_global=22.2346.ckpt`, SHA-256 `2f3c28ad…` | `last-epoch=199.ckpt`, SHA-256 `c665c914…` | finalized 23/10 18:29 |
| `PFN-Baseline-F0-200ep-s1234` | `outputs/final/PFN-Baseline-F0-200ep-s1234/` | `best-epoch=195-val_mel_l1_global=22.5705.ckpt`, SHA-256 `27bd2dec…` | `last-epoch=199.ckpt`, SHA-256 `e13daf50…` | finalized 24/10 16:24 |

## Export to the PC: 2026-10-24 17:21–17:45 +07

**What and why.** This follows `PROMPT-mardeus-export.md`, extended by the author on 24/10 to all four MardeusPC runs, i.e. with `PFN-Baseline-F0`. The author was away and wanted it done without waiting.

**Where it went, and the deviation from the prompt.**
- No external drive was attached, so the package was written to the internal data drive `E:`, not to an external drive:
  - folder `E:\pfn-transfer-20261024\`;
  - one store-only zip of that folder, `E:\pfn-transfer-20261024.zip`, which the author asked for;
  - a plain report, `E:\pfn-transfer-20261024-BAOCAO.txt`.
- The author copies the folder or the zip to the external drive. The read-back from the drive (prompt section 3.4) is still to be done there: `sha256sum -c SHA256SUMS.source`.

**Scripts (outside Git):**
- `<WSL_DEV_HOME>/pfn-sync/export_four.sh`, started by `after_f0_eval_export.sh` (tmux `cr-export`) once the Baseline-F0 evaluation had ended;
- log: `<WSL_DEV_HOME>/outputs/logs/export-four-runs.log`.

**Checks of section 2, all passed:**

| Label | `final/` SHA256SUMS | `num_sentences` | MCD n / skipped | split | `checkpoint_sha256` = best file | Reference UTMOSv2 |
|---|---|---:|---|---|---|---:|
| `PFN-Full-200ep-best` | OK | 500 | 500 / 0 | test | yes (`95e6d6c9…`) | 3.8308 |
| `PFN-NoF0-200ep-best` | OK | 500 | 500 / 0 | test | yes (`8f8570c6…`) | 3.8308 |
| `PFN-Baseline-VITS2-200ep-best` | OK | 500 | 500 / 0 | test | yes (`2f3c28ad…`) | 3.8308 |
| `PFN-Baseline-F0-200ep-best` | OK | 500 | 500 / 0 | test | yes (`27bd2dec…`) | 3.8308 |

**Checks of section 3:**
- `wavs/` entries in every `eval-*.tar`: 0;
- `sha256sum -c SHA256SUMS.source` read back from `E:`: all nine files OK;
- zip: 10 entries, CRC test OK.

**Size:** folder 6.5 GB; zip 6.5 GB.

**`SHA256SUMS.source`:**

```
98b3dc762caf0cc9cb53a013815c00630ad6e26246286fe48dd309ab318cdf7e  eval-PFN-Baseline-F0-200ep-best.tar
9c33c3e07d09b8849bd99db13aabdb1685bbb11cbb6482dd3230623aaac7071c  eval-PFN-Baseline-VITS2-200ep-best.tar
f140d38ddb88ee2516dddba047bbee4b40c470b25479146bd059fa6dff1c72fe  eval-PFN-Full-200ep-best.tar
ee1e7b218bfce99249680b7ff07760530175c635142719b03dea037a5d788fe6  eval-PFN-NoF0-200ep-best.tar
e7384c4c8b1a7ed0fd9b168a7755564f1304d0b866f9228e3433000048e9c9cd  eval-compare.tar
e818b9aaa333cd87e65b64cba12c9ad05fdd483b73df8e370710fe51a26ddc9b  final-PFN-Baseline-F0-200ep-s1234.tar
e9c1370399e03068a4394ac617a20df3880d951482f0c104691e85d7c0be3b0f  final-PFN-Baseline-VITS2-200ep-s1234.tar
d9cc462cdc7dbb17a6425a703ea61db448df0564b330f5e7aaa00280d0748835  final-PFN-Full-275ep-s1234.tar
9197aab7f391535dd6d1e1f6ec42abcdad16d7285d7d1c72417b1a500389f841  final-PFN-NoF0-200ep-s1234.tar
```

**Zip:** `b87a66760315220c6f59f0eff531db121de06cf572bbbb053dbf65703c7fb9b5  pfn-transfer-20261024.zip`

**Status:** `READY_FOR_TRANSFER`. Nothing was trained or re-evaluated for the export, and nothing was sent over a network.
