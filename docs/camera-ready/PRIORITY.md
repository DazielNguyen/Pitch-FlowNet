# Priority tracker: camera-ready PitchFlowNet (SOICT 2026, paper 4478)

**Deadlines:**
- final paper due **30/10/2026**;
- experiments end by **27/10 evening**;
- writing on 28–29/10.

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

## P0: blocks training

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T01 | Bring branch `camera-ready-20261019` to the training machine (push or copy) | author | — | 20/10 | TODO | |
| T02 | Sync the data outside Git and verify the hashes (`data-sync.md`) | author + assistant | T01 | 20/10 | TODO | |
| T03 | Dedicated env, MAS build, `preflight-<machine>.md` | assistant | T01, T02 | 20/10 | TODO | |
| T04 | Fix the resume continuity test (PLAN O1) | assistant | T03 | 20/10 | TODO | |
| T05 | DDP validation equals one process, subprocess DDP (PLAN O2) | assistant | T03 | 20/10 | TODO | |
| T06 | GPU smoke and pilots: `pilot-H-new`, `pilot-H-noF0` (200 G updates, interrupt + resume, at most 30 min each) | assistant, author OK | T04, T05, free GPU | 21/10 | TODO | |
| T07 | `eta.md` from the pilot measurements, with a budget proposal | assistant | T06 | 21/10 | TODO | |
| T08 | Author decisions (a) shuffle, (b) MAS decay, (c) budget and GPU plan, (d) replication seeds, (e) checkpoint rule, recorded in `experiment-plan.md` | author | T07 | 21/10 | TODO | |
| T09 | `launch_train.sh` (dry-run by default, guards) | assistant | T03 | 21/10 | TODO | |
| T10 | Launch the T1 pair H-new and H-noF0 | author OK | T08, T09 | 21–22/10 | TODO | |
| T11 | Monitor T1; resume if interrupted (never a second copy) | assistant | T10 | until done (≤ 27/10) | TODO | |

## P1: needed for the camera-ready content

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T12 | Evaluation harness: synthesis seeded per utterance ID, every run kept; WER macro/corpus; UTMOSv2 as predicted score; full records | assistant | T03 | 24/10 | TODO | |
| T13 | Waveform F0 metrics (same extractor for synthesized and reference; alignment chosen on validation) | assistant | T12 | 24/10 | TODO | |
| T14 | Pitch intervention −2/0/+2 semitones, hook with tests (z, duration, noise fixed) | assistant | T12 | 25/10 | TODO | |
| T15 | RTF protocol: isolated CPU, explicit threads, warm-up, model-only vs end-to-end | assistant | T12, idle machine | 26/10 | TODO | |
| T16 | `launch_eval.sh` (dry-run by default); evaluation protocol locked before any new test result | assistant, author OK | T12–T15 | 26/10 | TODO | |
| T17 | Run the locked evaluation on the T1 checkpoints (and the historical 773/1079 pair if decided) | author OK | T11, T16 | 27/10 | TODO | |
| T18 | Author answers A1–A6 (previews, checkpoint and session choice) for the manuscript wording | author | — | 28/10 | WAITING AUTHOR | `author-answers.md` of the follow-up package |

## P2: if time allows

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T19 | Replication seeds 2345/3456 for the T1 pair (no seed chosen by test results) | author OK | T08, GPU time | — | TODO | |
| T20 | T2 runs: H-noVITS2, H-noBigVGAN | author OK | T10, GPU time | — | TODO | |
| T21 | Human evaluation preparation: manifest, blind A/B UI, sentence selection seed. No invitations or uploads | assistant | T12 | — | TODO | |

## P3: later or optional

| ID | Task | Owner | Depends on | Target | Status | Evidence |
|---|---|---|---|---|---|---|
| T22 | T3 comparators: B-new, B-plusBigVGAN | author OK | GPU time | — | TODO | |
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

## Log

One line per change of plan or blocker, newest first.

- 19/10: Author: train on MardeusPC, 1 × RTX 5070 only, keeping the configuration otherwise unchanged; data moves over Tailscale. New 1-GPU configs named by variant: `PFN-Full`, `PFN-NoF0`, `PFN-NoVITS2`, `PFN-NoBigVGAN`, `PFN-Baseline`, `PFN-Baseline-BigVGAN` and their `-Pilot` versions, each derived from the matching `H-*`/`B-*` config (devices 1, strategy `auto`, batch 16, so effective batch 16 instead of 32); the budget stays a placeholder until O4. Run directories use the `run_id` (e.g. `PFN-Full-s1234`). MardeusPC still needs WSL (admin) and the data (`preflight-MardeusPC.md`).

- 19/10: GPUs on the PC are busy with another project; training moves to another machine, which the author operates.
