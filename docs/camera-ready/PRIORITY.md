# Priority tracker: camera-ready PitchFlowNet (SOICT 2026, paper 4478)

**Deadlines:**
- final paper due **30/10/2026**;
- experiments end by **27/10 evening**;
- writing on 28–29/10.

**Training machine:** MardeusPC, **one RTX 5070 (12 GB)**, configs `configs/PFN-*/` (`README.md`).
- **One run at a time.** The GPU holds one training run, so every run below queues behind the previous one.
- **Same configuration as before,** on the author's instruction: batch 16. The effective batch is 16, against 2 × 16 = 32 historically.
- **The GPU also drives the display.** Keep desktop use light while a run trains.

**How to use this file:**
- **Updating:** change a row's status when it moves, and add the commit or file that shows it.
- **Order:** do not start a row before its dependencies are DONE.
- **Long runs:** they need the author's confirmation, marked "author OK".
- **Details:** `PLAN.md`. **Data:** `data-sync.md`. **Prompt for the training machine:** `PROMPT-training-machine.md`.

**Status values:**
- `TODO`
- `IN PROGRESS`
- `BLOCKED` (give the reason)
- `WAITING AUTHOR`
- `DONE` (give the commit or file)

## Models to train and the GPU queue

**Decided on 20/10, extended to eight runs and lowered to 200 epochs on 21/10 (`experiment-plan.md`), replacing the queue below.**
- **Design:** all 2^3 = 8 combinations of the three groups at **200 epochs per run** (156,400 G updates; was 275 until 21/10).
- **RTX 5070:** `PFN-Full` (the running `PFN-Full-275ep-s1234`, ended at epoch 200) → `PFN-NoF0-200ep` → `PFN-Baseline-VITS2-200ep` → `PFN-Baseline-F0-200ep`.
- **PC, RTX 4070 Ti #0:** `PFN-NoBigVGAN-200ep` → `PFN-Baseline-BigVGAN-200ep`, once the GPUs are free.
- **PC, RTX 4070 Ti #1:** `PFN-NoVITS2-200ep` → `PFN-Baseline-200ep`.
- **All eight** should end by ≈ 25/10 11:00. Runs are never shortened below 200 epochs.
- **Main result:** still the historical pair.

Every run uses train seed 1234 unless marked. One budget in G updates for all runs, fixed at T08.

| Queue | Run ID | Config | Tier | Needed for | Status |
|---:|---|---|---|---|---|
| 0 | `PFN-Full-Pilot`, `PFN-NoF0-Pilot` | `configs/PFN-Full/`, `configs/PFN-NoF0/` (`*-Pilot.json`) | pilot, 200 G updates each | ETA, VRAM, resume check; never a result | TODO |
| 1 | `PFN-Full-s1234` | `configs/PFN-Full/PFN-Full.json` | T1 | main contrast (reference) | TODO |
| 2 | `PFN-NoF0-s1234` | `configs/PFN-NoF0/PFN-NoF0.json` | T1 | main contrast (F0 off) | TODO |
| 3 | `PFN-NoVITS2-s1234` | `configs/PFN-NoVITS2/PFN-NoVITS2.json` | T2 | per-group ablation | only if T07 shows time |
| 4 | `PFN-NoBigVGAN-s1234` | `configs/PFN-NoBigVGAN/PFN-NoBigVGAN.json` | T2 | per-group ablation | only if T07 shows time |
| 5 | `PFN-Full-s2345`, `PFN-NoF0-s2345` | as 1–2, train seed 2345 (new configs) | replication | seed variability of T1 | only if T07 shows time |
| 6 | `PFN-Baseline-s1234`, `PFN-Baseline-BigVGAN-s1234` | `configs/PFN-Baseline/`, `configs/PFN-Baseline-BigVGAN/` | T3 | new comparators (not the historical baseline) | later |

**Order rules:**
- Rows 1 and 2 are a pair. They are compared only with each other, at the same budget.
- **A row never starts if the T1 pair would not finish by 27/10.** Rows 3–6 only use time left after the T1 pair and its evaluation.

**Time available (calendar arithmetic, not a measurement):**

| Step | Hours |
|---|---:|
| First T1 run on 20/10 evening to 27/10 20:00 | ≈ 168 |
| Minus evaluation of the T1 checkpoints | ≈ 8 (to be measured at T12) |
| Minus 25–30 % reserve | |
| **GPU time for all queued runs** | **≈ 110–120** |

**What follows from it:**
- **T1 pair alone:** at most ≈ 55–60 h per run.
- **Rows 3–4 as well:** four runs at ≈ 28–30 h each.
- **Unknown until the pilot:** how many G updates fit into those hours on the RTX 5070. The historical full run needed 99.47 h of events on 2 × RTX 4070 Ti (not a verified total). If the budget is cut, the runs are labelled "budget-matched pilot" and compared only with each other, never with the 1,100-epoch historical models.

## P0: blocks training

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T01 | Bring the branch to the training machine | author | — | 20/10 | DONE | Clone on MardeusPC at `58cab26`; layout moved in `920b604`; WSL clone at `<WSL_DEV_HOME>/Pitch-FlowNet`, branch `camera-ready-20261019-v2` |
| T02 | Move the data outside Git and verify the hashes (`data-sync.md`) | author + assistant | T01 | 20/10 | DONE: all inputs on MardeusPC verified (dataset, config, 39,300 cache files, 13,100 WAVs, test entries, checkpoints 773 and 1079) | `preflight-MardeusPC.md` (Data) |
| T03 | Dedicated env, MAS build, `preflight-MardeusPC.md` | assistant | T01 | 20/10 | DONE: env `pfn-cr`, MAS built, full test run: 22/25 pass; the 3 failures are O1 (T04) and O2 (T05) | `preflight-MardeusPC.md` (Tests) |
| T04 | Fix the resume continuity test (PLAN O1). On one GPU, resume is the only protection against interruptions | assistant | T02, T03 | 20/10 | DONE: cause found and fixed (DataLoaders drew their base seed from the global RNG); the resumed run is bit-identical to an uninterrupted one; tests 24/25 (only O2 left) | PLAN O1, `resume_diagnostic.py`, `runbook-MardeusPC.md` |
| T06 | GPU smoke and pilots, queue row 0: `PFN-Full-Pilot` then `PFN-NoF0-Pilot` (200 G updates, interrupt + resume, at most 30 min each, one after the other). Measure s/it, peak VRAM with the display attached, validation and checkpoint time | assistant, author OK | T04 | 21/10 | DONE: `PFN-Full-Pilot` with one interruption + resume (882 G updates); NoF0 pilot skipped by the author, its speed comes from its first epoch | `eta.md` |
| T07 | `eta.md` from the pilot measurements: G updates per hour, a common budget for the queue, which rows 3–6 fit | assistant | T06 | 21/10 | DONE: 0.93 s per G update, 13.2 min per epoch, peak 9,989 MiB | `eta.md` |
| T08 | Author decisions in `experiment-plan.md`: (a) `train_shuffle`; (b) MAS decay; (c) the common `max_g_updates` and how many queue rows to run (device plan is fixed: 1 GPU, batch 16); (d) replication seed 2345 or not; (e) checkpoint rule (min `val_mel_l1_global`, fixed before any test) | author | T07 | 21/10 | DONE: shuffle false, MAS 4e-6, 275 epochs (215,050 G updates), seed 1234 only, min `val_mel_l1_global`; (c) changed on 21/10 to 200 epochs (156,400 G updates) to fit eight runs | `experiment-plan.md` |
| T09 | `tools/camera_ready/launch_train.sh`: dry-run by default; refuses a busy GPU, a second training process, an existing run directory (except `--resume`), missing inputs, a placeholder budget | assistant | T03 | 21/10 | DONE: dry run checked for both pilots and the placeholder refusal; `--execute` first used at T06 | `tools/camera_ready/launch_train.sh` |
| T10 | Launch the 200-epoch ablation runs in the order of `experiment-plan.md`, each after the author's OK of the exact command. RTX 5070: `PFN-Full` (ended at epoch 200) → `PFN-NoF0` → `PFN-Baseline-VITS2` → `PFN-Baseline-F0`. PC GPU 0: `PFN-NoBigVGAN` → `PFN-Baseline-BigVGAN`. PC GPU 1: `PFN-NoVITS2` → `PFN-Baseline` (`PROMPT-daziel-ablation.md`). Configs `configs/PFN-*/PFN-*-200ep.json`. Clean tree at every launch | author OK | T08, T09 | 20–25/10 | DONE: all eight runs ended with exit 0 after 200 epochs (156,400 G updates), training tree `1ee13cb4…`; RTX 5070: Full (21/10), NoF0 (22/10), Baseline-VITS2 (23/10), Baseline-F0 (24/10); PC: NoVITS2, NoBigVGAN (22–23/10), Baseline, Baseline-BigVGAN (23/10) | `runs-MardeusPC.md`, `runs-Daziel.md`; `run_manifest.json` of each run (outside Git) |
| T11 | Monitor every run read-only (log, `val_diagnostics.jsonl`, `checkpoint_ledger.jsonl`, TensorBoard, `nvidia-smi`): NaN, rising validation, OOM, stalls. Resume an interrupted run with `resume_train.sh` (never a second copy, never a restart from scratch) and record each interruption. On the RTX 5070, start `PFN-NoF0-275ep` only after `PFN-Full-275ep` ends. At each end, `finalize_run.sh` keeps read-only copies of `best-*.ckpt` and the last checkpoint (epoch 200) with SHA-256 in `outputs/final/<run_id>/` (the watchdog does it on exit 0); record epoch and best `val_mel_l1_global` | assistant | T10 | until done (≤ 27/10) | DONE: every run finalized into `final/<run_id>/` (read-only, `SHA256SUMS`). Interruptions, all `CUDA error: unknown error` on the RTX 5070: Full 4 (2 resumed by hand, 2 by the watchdog), NoF0 3 (watchdog); no interruption on the PC. On 24/10 the four MardeusPC final folders were imported to the PC and checked | `runs-MardeusPC.md`, `runs-Daziel.md` |

## P1: needed for the camera-ready content

**The GPU is busy with training during most of this.** Develop and test the evaluation on validation with the CPU or in the gaps between runs, and do not start GPU work beside a training run.

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T12 | Evaluation harness: synthesis seeded per utterance ID, every run kept; WER macro/corpus; UTMOSv2 as predicted score; full records. Measure its duration for the time budget | assistant | T03 | 24/10 | DONE: historical evaluation reproduced and made deterministic (seeded synthesis and UTMOSv2, strict safe load), env `pfn-eval` at the historical versions; checked twice on 5 validation utterances | `eval-protocol.md`, `tools/camera_ready/evaluate_like_history.py` |
| T13 | Waveform F0 metrics (same extractor for synthesized and reference; alignment chosen on validation) | assistant | T12 | 24/10 | TODO | |
| T14 | Pitch intervention −2/0/+2 semitones, hook with tests (z, duration, noise fixed) | assistant | T12 | 25/10 | TODO | |
| T15 | RTF protocol: isolated CPU, explicit threads, warm-up, model-only vs end-to-end. **Not while a run trains** (its data loader uses the CPU): measure after the queue ends | assistant | T12, idle machine | 27/10 | TODO: the queue has ended. On 24/10 the author asked for one benchmark of the eight best checkpoints; its protocol (CPU as `eval-protocol.md`, or one idle GPU) is to be agreed before anything runs. Only inline RTF exists so far, not reportable | `ablation8-results.md` (section 5) |
| T16 | `tools/camera_ready/launch_eval.sh` (dry-run by default); evaluation protocol locked before any new test result | assistant, author OK | T12–T15 | 26/10 | DONE: locked on 21/10 as the historical evaluation (author) | `eval-protocol.md`, `tools/camera_ready/launch_eval.sh` |
| T17 | Run the locked evaluation on the T1 checkpoints (and the historical 773/1079 pair if decided) | author OK | T11, T16 | 27/10 | DONE for the eight 200-epoch runs: test split, 500 utterances each, every check passed; analysed on 24/10 (consistency, 13 paired tests with Holm, factorial effects, Shapley, cost). This entry does not cover the historical pair | `ablation8-results.md`, `manifests/camera-ready/ablation8-derived.json` |
| T18 | Author answers A1–A6 (previews, checkpoint and session choice) for the manuscript wording | author | — | 28/10 | WAITING AUTHOR | `author-answers.md` of the follow-up package |

## P2: if T07 shows time

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T20 | (merged into T10 on 20/10) `PFN-NoVITS2`, `PFN-NoBigVGAN` are now leave-one-out runs of the ablation | — | — | — | moved to T10 | `experiment-plan.md` |
| T19 | Queue row 5: replication seed 2345 for the T1 pair (no seed chosen by test results) | author OK | T08, T07 | — | TODO | |
| T21 | Human evaluation preparation: manifest, blind A/B UI, sentence selection seed. No invitations or uploads | assistant | T12 | — | TODO | |

## P3: later or optional

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T22 | (merged into T10 on 21/10) `PFN-Baseline`, `PFN-Baseline-BigVGAN`, plus the new `PFN-Baseline-VITS2`, `PFN-Baseline-F0`, are now add-one runs of the ablation | — | — | — | moved to T10 | `experiment-plan.md` |
| T05 | DDP validation equals one process, subprocess DDP (PLAN O2). Not needed for 1-GPU runs (world size 1); needed again before any 2-GPU run | assistant | T03 | — | TODO | |
| T23 | `run_manifest.schema.json` matching `train.py` output | assistant | — | — | TODO | |
| T24 | `tools/camera_ready/verify_data.py` (read-only data check) | assistant | — | — | TODO | |

## Handoff (continuous)

| ID | Task | Owner | Status | Evidence |
|---|---|---|---|---|
| T25 | `handoff_to_mac.md`: done / pending / failed, commands, checkpoint hashes, source changes | assistant | TODO | |

## Done

| Date | Item | Evidence |
|---|---|---|
| 19/10 | Workspace = byte-exact copy of the `badf381` candidate (59/59 equal to the record) | `44992bb`, `source-provenance-pre.json` |
| 19/10 | Strict group gating (ResBlock Snake under B) | `da7d148` |
| 19/10 | Split manifest by stable ID with raw-WAV hashes | `442dec1` |
| 19/10 | Camera-ready training package and run configs | `f0dfa2a` |
| 19/10 | Tests (gating, split, sharding, historical checkpoints pass; resume and DDP checks open as T04/T05) | `9ce3eb5` |
| 19/10 | Preflight, data sync plan, PLAN | `d528a90` |
| 19/10 | Prompt for the training machine | `e9e5647` |
| 19/10 | Workspace moved into the main repository folders | `920b604` |
| 19/10 | Single-GPU run configs `configs/PFN-*/` | `da54b21` |
| 19/10 | Layout page and MardeusPC preflight | `17323a3` |

## Log

One line per change of plan or blocker, newest first.

- 24/10: Ablation done. All eight 200-epoch cells are trained, evaluated with the locked protocol and analysed on the PC (`ablation8-results.md`). The four MardeusPC runs reached the PC as a hash-checked package; their hash list matches the copy in `runs-MardeusPC.md`. Analysis tool `tools/camera_ready/ablation8_analysis.py`; its Holm correction was specified after all evaluations existed. Open: the RTF benchmark (T15), whose protocol is to be agreed. This file and `experiment-plan.md` were updated from the PC on the author's instruction.

- 21/10: Author lowered the budget from 275 to 200 epochs per run (156,400 G updates; 18.2 % of the historical epochs) so that all eight combinations fit. New configs `configs/PFN-*/PFN-*-200ep.json`. The running `PFN-Full-275ep-s1234` was at epoch 168. It was interrupted right after the epoch-168 checkpoint and resumed with `--stop-after-epochs 200`. This is equivalent to a 200-epoch run: the budget only decides the stop; the LR steps per epoch and the MAS noise per G update (`experiment-plan.md`, `runs-MardeusPC.md`).

- 21/10: Author extended the ablation to all 2^3 = 8 combinations. New 275-epoch configs: `PFN-Baseline`, `PFN-Baseline-BigVGAN`, and the new folders `PFN-Baseline-VITS2`, `PFN-Baseline-F0`. Priority: the four leave-one-out runs, then `Baseline-BigVGAN` and `Baseline`, then `Baseline-VITS2`, then `Baseline-F0`, the last one only if it fits before 26/10 evening (`experiment-plan.md`).

- 20/10: On the author's request the AI co-author line was removed from all MardeusVN commit messages (history rewritten; trees, authors and dates unchanged). Old → new hashes: `manifests/camera-ready/commit-map-2026-10-20.json`; the runs started before it cite the old hashes. New commits carry no AI attribution (AGENTS.md).

- 20/10: Author decisions after the reviews (all three name the missing ablation): leave-one-group-out with a new 275-epoch reference (1/4 of the historical epochs); historical Full is not the ablation reference. Pilot measured 13.2 min per epoch on the RTX 5070. The PC joins from about 20/10 13:00.

- 20/10: O1 closed: resume diverged because the DataLoaders drew their worker base seed from the global RNG; fixed with a private generator per DataLoader (`camera_ready/model.py`). Crash recovery prepared: `check_ckpt.py`, `resume_train.sh` (with a fallback to the newest epoch checkpoint), TensorBoard in tmux, WSL keep-alive; Windows sleep on AC turned off (`runbook-MardeusPC.md`).

- 20/10: All inputs verified on MardeusPC (IMPORT PASS); the external drive read unreliably over USB, so the author copied the archives by hand and they matched their hashes. Full tests: 22/25 pass; resume (O1) differs far more than on the PC.

- 19/10: Priorities rewritten for one GPU: runs form a queue (rows 0–6 above), the T1 pair first; T2, replication and T3 only if the pilot ETA leaves time. The DDP check (T05) moves to P3, and RTF (T15) moves after the queue.
- 19/10: Data packed from the PC onto an external drive (F) after Tailscale proved too slow (relay only, ≈ 0.45 MB/s); checkpoint 773 already arrived over Tailscale.
- 19/10: Author: train on MardeusPC, 1 × RTX 5070 only, keeping the configuration otherwise unchanged; data moves over Tailscale. New 1-GPU configs named by variant: `PFN-Full`, `PFN-NoF0`, `PFN-NoVITS2`, `PFN-NoBigVGAN`, `PFN-Baseline`, `PFN-Baseline-BigVGAN` and their `-Pilot` versions, each derived from the matching `H-*`/`B-*` config (devices 1, strategy `auto`, batch 16, so effective batch 16 instead of 32); the budget stays a placeholder until O4. Run directories use the `run_id` (e.g. `PFN-Full-s1234`). MardeusPC still needs WSL (admin) and the data (`preflight-MardeusPC.md`).
- 19/10: GPUs on the PC are busy with another project; training moves to another machine, which the author operates.
