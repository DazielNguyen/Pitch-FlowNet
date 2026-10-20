# Runbook: training on MardeusPC (one RTX 5070, WSL)

**Scope.**
- This page covers how a camera-ready run is started, watched and recovered on MardeusPC.
- What to train and in which order: see `PRIORITY.md`.
- The machine itself: see `preflight-MardeusPC.md`.

## Rules

- **One training run at a time.** `launch_train.sh` refuses a second one.
- **Every launch needs the author's OK** for the exact command.
- **Never restart an interrupted run from scratch.** Use `resume_train.sh`, which continues from the run's own `last.ckpt`.
- **Train from a clean, committed tree.** `run_manifest.json` records the commit and any uncommitted paths.
- **No test-set use, and no warm start from an old checkpoint.**

## Where things are (inside WSL, user `dev`)

| What | Path or session |
|---|---|
| Repository used for training | `<WSL_DEV_HOME>/Pitch-FlowNet` (a clone of the Windows copy, branch `camera-ready-20261019-v2`; MAS extension built here) |
| Environment | `<WSL_DEV_HOME>/miniforge3/envs/pfn-cr` |
| Run directory | `<WSL_DEV_HOME>/outputs/runs/<run_id>/`: `run_manifest.json`, `checkpoint_ledger.jsonl`, `val_diagnostics.jsonl`, `checkpoints/`, `tb/` |
| Launch logs | `<WSL_DEV_HOME>/outputs/logs/<run_id>-<UTC time>.log` |
| Training session | tmux `cr-<run_id>` |
| TensorBoard | tmux `cr-tensorboard`, http://localhost:6006 in the Windows browser |
| WSL keep-alive | hidden `wsl.exe` process; state in `%LOCALAPPDATA%\pfn-cr\wsl-keepalive.json` |

## Before a launch

1. **Windows power settings.**
   - Sleep on AC power is off (`powercfg /change standby-timeout-ac 0`, set on 20/10); hibernate is off.
   - Windows Update can still restart the machine. Pause updates in *Settings → Windows Update* for the training period.
2. **Keep WSL alive:** `powershell -ExecutionPolicy Bypass -File tools\camera_ready\wsl_keepalive.ps1 start` (Windows; idempotent).
3. **Clean tree, inside WSL:** `cd ~/Pitch-FlowNet && git pull --ff-only && git status --short`. The status output must be empty.
4. **TensorBoard:** `bash tools/camera_ready/tensorboard.sh start`.

## Launch

```sh
cd ~/Pitch-FlowNet
bash tools/camera_ready/launch_train.sh configs/PFN-Full/PFN-Full-Pilot.json --limit-train-batches 100            # dry run
bash tools/camera_ready/launch_train.sh configs/PFN-Full/PFN-Full-Pilot.json --limit-train-batches 100 --execute  # real
```

## Watch (read-only)

- **Live output:** `tmux attach -t cr-<run_id>`, then detach with `Ctrl-b d`. Never press `Ctrl-c` inside the session.
- **Records:** the run's `checkpoint_ledger.jsonl`, `val_diagnostics.jsonl` and launch log.
- **Metrics:** TensorBoard.
- **GPU:** `nvidia-smi`.

## After a crash, a reboot or a power loss

Each epoch the run first writes `epoch-<N>.ckpt` and removes the previous one, then overwrites `last.ckpt`.
So at most one epoch is lost, and at least one of the two files is complete.

1. **Windows:** `wsl_keepalive.ps1 start`.
2. **WSL:** `bash tools/camera_ready/tensorboard.sh start`.
3. **WSL:** `bash tools/camera_ready/resume_train.sh configs/<variant>/<config>.json`.
   - This is a dry run.
   - It refuses while the run's tmux session or process is still alive.
   - It checks `last.ckpt` with `check_ckpt.py`: zip CRC, safe load, counters, finite values.
4. **If step 3 reports a damaged `last.ckpt`:** run it again with `--fallback`.
   - The damaged file is kept as `last.ckpt.damaged-<time>`.
   - The newest complete `epoch-*.ckpt` becomes `last.ckpt`.
   - The swap is recorded in `checkpoint_ledger.jsonl`.
5. **Resume:** after the author's OK, run the same command with `--execute`.
6. **Record:** the interruption, its time and the resume, in `handoff_to_mac.md`.

## Resume behaviour

A resumed run continues where the uninterrupted run would have been. The CPU check is `tests/camera_ready/test_train_cpu.py`:
- weights and optimizer state agree with an uninterrupted run, within the noise floor between two uninterrupted runs;
- schedulers, counters and the validation metric are equal.

The GPU pilots (T06) check this again with one interruption.

## Automatic resume after the CUDA unknown error (watchdog)

On this machine the GPU also drives the display. Windows resets it every few hours (`LiveKernelEvent` 141), and a run can then die with `CUDA error: unknown error` (`runs-MardeusPC.md`).

**Standing permission.** On 2026-10-21 the author gave a standing OK to resume automatically after this one error, without rebooting or changing the registry.

```sh
bash tools/camera_ready/auto_resume.sh configs/<variant>/<config>.json --start [--gpu N]   # tmux cr-watch-<run_id>
```

**What it does.**
- Every 2 min it checks the run's tmux session.
- It resumes only when all of these hold:
  - the session is gone;
  - the newest launch log ends with a non-zero exit;
  - the log contains `CUDA error: unknown error`.
- Before resuming it waits 60 s, checks CUDA with a small matmul, then runs `resume_train.sh --execute`, which checks `last.ckpt`.
- **At most 5 times per run.** Each automatic resume is written to `$CR_OUTPUTS/logs/watchdog-<run_id>.log` and appended to the run's `checkpoint_ledger.jsonl` (`"event": "auto_resume"`).

**It stops and leaves the run to the author when:**
- the run ends with another error;
- `last.ckpt` is damaged (it never uses `--fallback`);
- the CUDA check fails;
- the limit is reached;
- the session is gone without an exit line, e.g. after a WSL restart.

**It never starts the next run of the queue.** When the run ends with exit 0, the watchdog stops.

**After a reboot**, start it again after `resume_train.sh`, because it runs in tmux.
