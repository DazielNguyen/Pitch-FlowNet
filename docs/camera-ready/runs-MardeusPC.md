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
