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
| T01 | Bring the branch to the training machine | author | — | 20/10 | DONE | Clone on MardeusPC at `58cab26`; layout moved in `1e4cc1a`; WSL clone at `<WSL_DEV_HOME>/Pitch-FlowNet`, branch `camera-ready-20261019-v2` |
| T02 | Move the data outside Git and verify the hashes (`data-sync.md`) | author + assistant | T01 | 20/10 | DONE: all inputs on MardeusPC verified (dataset, config, 39,300 cache files, 13,100 WAVs, test entries, checkpoints 773 and 1079) | `preflight-MardeusPC.md` (Data) |
| T03 | Dedicated env, MAS build, `preflight-MardeusPC.md` | assistant | T01 | 20/10 | DONE: env `pfn-cr`, MAS built, full test run: 22/25 pass; the 3 failures are O1 (T04) and O2 (T05) | `preflight-MardeusPC.md` (Tests) |
| T04 | Fix the resume continuity test (PLAN O1). On one GPU, resume is the only protection against interruptions | assistant | T02, T03 | 20/10 | TODO: on MardeusPC 986 tensors differ and the 1-epoch validation metric differs by ≈ 0.7 (PC: 455 tensors, ≈ 1e-8); cause not yet known | `preflight-MardeusPC.md` (Tests) |
| T06 | GPU smoke and pilots, queue row 0: `PFN-Full-Pilot` then `PFN-NoF0-Pilot` (200 G updates, interrupt + resume, at most 30 min each, one after the other). Measure s/it, peak VRAM with the display attached, validation and checkpoint time | assistant, author OK | T04 | 21/10 | TODO | |
| T07 | `eta.md` from the pilot measurements: G updates per hour, a common budget for the queue, which rows 3–6 fit | assistant | T06 | 21/10 | TODO | |
| T08 | Author decisions in `experiment-plan.md`: (a) `train_shuffle`; (b) MAS decay; (c) the common `max_g_updates` and how many queue rows to run (device plan is fixed: 1 GPU, batch 16); (d) replication seed 2345 or not; (e) checkpoint rule (min `val_mel_l1_global`, fixed before any test) | author | T07 | 21/10 | TODO | |
| T09 | `tools/camera_ready/launch_train.sh`: dry-run by default; refuses a busy GPU, a second training process, an existing run directory (except `--resume`), missing inputs, a placeholder budget | assistant | T03 | 21/10 | DONE: dry run checked for both pilots and the placeholder refusal; `--execute` first used at T06 | `tools/camera_ready/launch_train.sh` |
| T10 | Launch queue rows 1 and 2 one after the other, at the same budget | author OK | T08, T09 | 21–22/10 | TODO | |
| T11 | Monitor the queue; resume an interrupted run (never a second copy, never a restart from scratch); start the next row only after the previous one ends | assistant | T10 | until done (≤ 27/10) | TODO | |

## P1: needed for the camera-ready content

**The GPU is busy with training during most of this.** Develop and test the evaluation on validation with the CPU or in the gaps between runs, and do not start GPU work beside a training run.

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T12 | Evaluation harness: synthesis seeded per utterance ID, every run kept; WER macro/corpus; UTMOSv2 as predicted score; full records. Measure its duration for the time budget | assistant | T03 | 24/10 | TODO | |
| T13 | Waveform F0 metrics (same extractor for synthesized and reference; alignment chosen on validation) | assistant | T12 | 24/10 | TODO | |
| T14 | Pitch intervention −2/0/+2 semitones, hook with tests (z, duration, noise fixed) | assistant | T12 | 25/10 | TODO | |
| T15 | RTF protocol: isolated CPU, explicit threads, warm-up, model-only vs end-to-end. **Not while a run trains** (its data loader uses the CPU): measure after the queue ends | assistant | T12, idle machine | 27/10 | TODO | |
| T16 | `tools/camera_ready/launch_eval.sh` (dry-run by default); evaluation protocol locked before any new test result | assistant, author OK | T12–T15 | 26/10 | TODO | |
| T17 | Run the locked evaluation on the T1 checkpoints (and the historical 773/1079 pair if decided) | author OK | T11, T16 | 27/10 | TODO | |
| T18 | Author answers A1–A6 (previews, checkpoint and session choice) for the manuscript wording | author | — | 28/10 | WAITING AUTHOR | `author-answers.md` of the follow-up package |

## P2: if T07 shows time

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T20 | Queue rows 3–4: `PFN-NoVITS2`, `PFN-NoBigVGAN` | author OK | T10, T07 | — | TODO | |
| T19 | Queue row 5: replication seed 2345 for the T1 pair (no seed chosen by test results) | author OK | T08, T07 | — | TODO | |
| T21 | Human evaluation preparation: manifest, blind A/B UI, sentence selection seed. No invitations or uploads | assistant | T12 | — | TODO | |

## P3: later or optional

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T22 | Queue row 6: `PFN-Baseline`, `PFN-Baseline-BigVGAN` | author OK | GPU time | — | TODO | |
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
| 19/10 | Workspace moved into the main repository folders | `1e4cc1a` |
| 19/10 | Single-GPU run configs `configs/PFN-*/` | `dace802` |
| 19/10 | Layout page and MardeusPC preflight | `f5669a9` |

## Log

One line per change of plan or blocker, newest first.

- 20/10: All inputs verified on MardeusPC (IMPORT PASS); the external drive read unreliably over USB, so the author copied the archives by hand and they matched their hashes. Full tests: 22/25 pass; resume (O1) differs far more than on the PC.

- 19/10: Priorities rewritten for one GPU: runs form a queue (rows 0–6 above), the T1 pair first; T2, replication and T3 only if the pilot ETA leaves time. The DDP check (T05) moves to P3, and RTF (T15) moves after the queue.
- 19/10: Data packed from the PC onto an external drive (F) after Tailscale proved too slow (relay only, ≈ 0.45 MB/s); checkpoint 773 already arrived over Tailscale.
- 19/10: Author: train on MardeusPC, 1 × RTX 5070 only, keeping the configuration otherwise unchanged; data moves over Tailscale. New 1-GPU configs named by variant: `PFN-Full`, `PFN-NoF0`, `PFN-NoVITS2`, `PFN-NoBigVGAN`, `PFN-Baseline`, `PFN-Baseline-BigVGAN` and their `-Pilot` versions, each derived from the matching `H-*`/`B-*` config (devices 1, strategy `auto`, batch 16, so effective batch 16 instead of 32); the budget stays a placeholder until O4. Run directories use the `run_id` (e.g. `PFN-Full-s1234`). MardeusPC still needs WSL (admin) and the data (`preflight-MardeusPC.md`).
- 19/10: GPUs on the PC are busy with another project; training moves to another machine, which the author operates.
