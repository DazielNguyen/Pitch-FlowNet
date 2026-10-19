# Plan: camera-ready experiments, PitchFlowNet (SOICT 2026, paper 4478)

**Status (2026-10-19):** setup only. **GPU_BLOCKED** on this PC.
- Both GPUs run another project's training. On the author's instruction, nothing is trained on this machine.
- **Not READY_FOR_TRAINING_REVIEW yet:** the items marked OPEN below come first.
- **Final-paper deadline:** 30/10/2026; experiments should end by 27/10 at the latest.

**Reading order:**
1. `preflight.md`: what exists on the PC;
2. `data-sync.md`: how data reaches a training machine;
3. this file.

## Done in this branch

| Item | Where | State |
|---|---|---|
| Workspace source = historical candidate `badf381`, byte-exact copy | `src/python`, `source-provenance-pre.json` | done (59/59 equal to record) |
| A. Group gating: `use_bigvgan` also removes the ResBlock Snake under `group_semantics="strict"`; legacy mode keeps the historical checkpoints loadable | `vits/modules.py`, `vits/models.py`, `vits/lightning.py` | done, tested |
| B. Split by stable ID from `split-manifest.json`; split seed separated from the training seed; the test split is never loaded in training | `split-manifest.json`, `tools/make_split_manifest.py`, `camera_ready/split.py` | done, tested |
| C. Deterministic per-utterance validation, exact sum/count over ranks, previews from validation IDs, fixed checkpoint directory, ledger of saves/removals | `camera_ready/model.py`, `camera_ready/callbacks.py` | done; the DDP equality test is OPEN (O2) |
| D. G/D update and sample counters in checkpoints, G-update budget, MAS decay per G update, separate init/data/validation seeds, per-epoch RNG reseed | `camera_ready/model.py`, `callbacks.py`, `train.py` | done; the resume test is OPEN (O1) |
| Run configs | `configs/H-new`, `H-noF0` (T1); `H-noVITS2`, `H-noBigVGAN` (T2); `B-new`, `B-plusBigVGAN` (T3); `pilot-*` (200 G updates); `cpu-smoke-H` | written; budget values are placeholders (O4) |
| Data inventory and sync plan | `preflight.md`, `data-manifest.json`, `data-sync.md` | done |

**Source changes against the historical candidate:**
- `source-provenance-post.json`: 3 files changed, 7 files added;
- `git diff 44992bb -- camera-ready-20261019/src`.

## OPEN items, in order

**O1. Resume continuity test.**
- **Problem:** CPU training is not bit-deterministic here. Two fresh 1-epoch runs differ by about 1e-8 in 455/994 tensors, so the current exact-equality assertion cannot pass.
- **Fix:** run the test with `torch.use_deterministic_algorithms(True)` and one CPU thread, then compare exactly.
  - If some operation stays nondeterministic, compare "resumed vs uninterrupted" against "uninterrupted vs uninterrupted" (same tolerance).
  - Also check optimizer/scheduler state, counters, the ledger best state and the per-epoch RNG seed.
- **Do not weaken the test** to silence the failure.

**O2. DDP validation equals one process.**
- **Problem:** `ddp_spawn` cannot pickle the weight-norm tensors.
- **Fix:**
  - write a small script, `tests/ddp_validate_check.py`, that runs `Trainer.validate` with `strategy="ddp"` (subprocess launcher), CPU and gloo, on 3 validation utterances (shards 2 + 1);
  - call it from the test for `devices=1` and `devices=2`;
  - compare the per-utterance values and the global metric.

**O0. Before any test on a machine:** build the MAS Cython extension in the workspace. It is git-ignored, so each machine builds its own:
```sh
cd camera-ready-20261019/src/python/piper_train/vits/monotonic_align
mkdir -p monotonic_align && cythonize -i core.pyx && mv core*.so monotonic_align/
```
Then run `python -m unittest discover -s camera-ready-20261019/tests -v`, with the data paths of `data-sync.md`.

**O3. GPU smoke and pilot.**
- **When:** only on a GPU verified free. On this PC that means after Piper-SeqIR and its automatic evaluation/timing session (SeqIR was at epoch 974/1,500 on 19/10 15:19, about 4.4 min/epoch, so the GPUs are free about 21/10 at the earliest), or on another training machine.
- **Steps:**
  1. one forward/backward and one validation per T1 config;
  2. then `pilot-H-new` and `pilot-H-noF0` with `--limit-train-batches 100` (2 epochs, 200 G updates), with one interruption and resume;
  3. measure steady-state s/it after warm-up, peak VRAM, validation and checkpoint time;
  4. stop cleanly if a pilot exceeds 30 minutes.
- **Pilots are never results and never warm starts.**

**O4. Budget and ETA (`eta.md`, ETA_PENDING).**
- **Configs today:** they hold `max_g_updates` 430,100, the historical-equivalent budget of 1,100 epochs × 391 batches per rank.
- **Earlier wall-clock spans:**
  - historical runs: about 77 h (baseline) and 99.47 h (full, the `version_2` events only; not a verified total) on 2 GPUs;
  - a full-budget pair, run one after the other, would need about 200 h, beyond the window to 27/10.
- **After O3, decide one of these:**
  - a common reduced budget in G updates, labelled "budget-matched pilot", with 25–30 % reserve and evaluation time;
  - or report NO_FEASIBLE_TRAINING_PAIR and evaluate the existing checkpoints only.
- **Not allowed:** comparing a shortened run with the old 1,100-epoch full model as a causal ablation.

**O5. Protocol decisions for the author**, recorded in `experiment-plan.md` before any launch:
- (a) `train_shuffle`: `false` reproduces the historical order; `true` is the usual practice.
- (b) MAS decay: `4e-6` per G update reproduces the historical schedule; `2e-6` per G update is the literal VITS2 value.
- (c) Device plan: 2-GPU DDP runs one after another, or one GPU per run. The latter changes the effective batch unless the batch is doubled, which must be measured.
- (d) Replication seeds 2345 and 3456 for the pair, only if the time allows. No seed is chosen by test results.
- (e) The training machine and the data path on it (see `data-sync.md`).

**O6. Remaining deliverables of the setup brief:**
- **`experiment-plan.md`:**
  - matrix and run IDs;
  - seeds and budget;
  - device allocation and checkpoint rule (min `val_mel_l1_global`, fixed before any test);
  - estimand and limits.
- **`run_manifest.schema.json`:** matches what `train.py` writes.
- **`launch_train.sh`:**
  - dry-run by default; refuses an existing run directory, a missing input, a busy GPU, another training process, or an unresolved budget;
  - `--execute` only after the author's review.
- **`launch_eval.sh`:** dry-run by default.
- **`eta.md`** and **`handoff_to_mac.md`:** done / pending / failed, launch commands, all source changes against history.
- **`tools/verify_data.py`:** read-only data check, see `data-sync.md`.

**O7. Evaluation tooling** (prepare; do not run test-set evaluation during setup).
- **Synthesis:**
  - seeded per utterance ID, primary synthesis seed fixed in advance (for example 2026; 2027/2028 as sensitivity);
  - save WAVs, durations, ASR hypotheses/references, per-utterance scores, failures and denominators;
  - record scorer weights and commits, threads, device and argv.
- **Reporting:** WER macro and corpus; UTMOSv2 as a predicted score; paired tests at the right unit. Training seeds are not pooled as independent utterances.
- **F0 from waveforms:**
  - the same extractor and settings for synthesized and reference WAVs;
  - choose the alignment on validation;
  - RMSE in Hz or cents, correlation on voiced pairs, voicing decision error, coverage;
  - the old `collect_f0_pairs` measures the predictor, not the waveform.
- **Pitch intervention −2 / 0 / +2 semitones:**
  - applied to the decoder F0 conditioning after expansion (scale 2^(s/12)), with z, durations and noise held fixed, via a tested hook;
  - also check the `f0_cond` norm and response. A non-zero norm is not proof of controllability.
- **RTF:**
  - CPU isolated, batch 1, explicit intra/inter-op and OMP/MKL threads, fixed warm-up;
  - model-only and end-to-end reported separately; per sentence and per repeat;
  - mean, pooled, median and p95.
  - Measure only on an idle machine.
- **Human evaluation:**
  - manifest and randomised blind A/B UI, sentence selection fixed by a recorded seed before seeing scores;
  - consent, instructions, headphone check, exclusion rules.
  - No invitations or uploads without the author.

## Constraints that stay in force

**Kept separate:**
- No mixing with the journal HiFi-MobiNet work, the BanhmiTTS port or ParIR/SeqIR.
- The evidence-store folders of this repository stay unchanged.

**Training and evaluation:**
- No training on this PC without the author's go-ahead, and none while the GPUs serve other work.
- Never touch the `piper-ir`, `post-seqir` or `tb-seqir` tmux sessions.
- No test-set use for tuning, early stopping or checkpoint choice.
- No warm start from the historical full checkpoint for an ablation.

**Publishing:**
- No push, merge, upload, manuscript or ledger edits.

**Interpreting the record:**
- Author answers A1–A6 are still PENDING.
- The historical checkpoint/session choices are not explained here.
- Hypothesis 2.9 (rank agreement) is not a confirmed historical cause.
