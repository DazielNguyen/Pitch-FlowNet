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

**Progress on 22/10 15:47.**
- Last checkpoint: epoch 188 (147,798 G updates), at 15:40.
- `val_mel_l1_global`: 21.3212; best so far 21.2366 at epoch 187.
- About 8.4 min per epoch, so the expected end is ≈ 22/10 17:20, if no further interruption occurs.

**Next.** The one-off chain `<WSL_DEV_HOME>/pfn-sync/after_nof0.sh` (tmux `cr-chain`, author's OK on 22/10) waits for the NoF0 archive. Then it does the following, stopping at the first failed check:
1. updates the WSL clone and checks that it is clean;
2. launches `PFN-Baseline-VITS2-200ep` with its watchdog;
3. evaluates the NoF0 best checkpoint;
4. compares it with `PFN-Full-200ep-best`.

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
| `PFN-NoF0-200ep-s1234` | `outputs/final/PFN-NoF0-200ep-s1234/` | | | pending (training, epoch 188/200 on 22/10 15:47) |
