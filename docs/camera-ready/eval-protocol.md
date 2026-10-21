# Evaluation protocol: camera-ready (T16), locked on 2026-10-21

**Decision of the author (21/10).** Evaluate exactly as the historical evaluation did, and add no new metric. This page fixes that protocol before any new test-set result.
- The historical pair (Baseline epoch 773, Full epoch 1079) and every 200-epoch ablation run are evaluated with it.
- **Tool:** `tools/camera_ready/evaluate_like_history.py`, in the separate environment `pfn-eval`.
- **Launcher:** `tools/camera_ready/launch_eval.sh`, dry run by default.

## What is measured (the historical evaluation)

The historical evaluation is recorded in `evaluation/eval_h_vs_baseline/`, `evaluation/full/eval_pitchflownet_500_run.sanitized.log` and `history/environment/edgetts-env-pip-freeze_2026-10-12.txt`. The tool calls the historical code itself: `piper_train/eval_harness.py` and `piper_train/compute_reference_metrics.py`.

| Item | Setting |
|---|---|
| Sentences | the 500 test utterances of `manifests/camera-ready/split-manifest.json`, checked line by line against `datasets/splits/test_sentences_500.txt` (the historical file) |
| Synthesis | CPU; phonemes from the text (espeak `en-us`); noise_scale 0.667, length_scale 1.0, noise_w 0.8; 22,050 Hz |
| WER | Whisper `base.en` on the CPU, audio resampled to 16 kHz; the harness's normalisation (lower case, `[^a-z0-9' ]` removed); `jiwer.wer` per utterance; mean and median over utterances |
| Naturalness | UTMOSv2 (`create_model(pretrained=True)`, CPU, `num_repetitions` 1); mean and median. **A predicted score, not MOS** |
| RTF | inference time / audio duration per utterance, inline, as historically. **Only comparable on an idle machine**; see below |
| F0 predictor metric | RMSE and correlation of the phoneme-level log-F0 of the F0 predictor against the cached reference F0 (`collect_f0_pairs`); only models with the F0 group |
| MCD and reference UTMOS | DTW on 24 MFCCs (C1–C24) against the normalised reference audio; UTMOSv2 of the reference audio (`compute_reference_metrics.py`) |
| Pairwise tests | two-sided Wilcoxon signed-rank on per-utterance WER and UTMOSv2, SciPy defaults (`evaluate_like_history.py compare`) |
| Package versions | torch 2.12.1, torchaudio 2.11.0, openai-whisper 20250625, jiwer 4.0.0, transformers 5.12.1, timm 1.0.27, scipy 1.15.3, librosa 0.11.0, UTMOSv2 commit `cc2700db` (the historical versions) |
| Checkpoint per run | the minimum `val_mel_l1_global` (`best-epoch=…ckpt` in `outputs/final/<run_id>/`), decision (e) of `experiment-plan.md`. For the historical pair, the checkpoints of the paper (773, 1079) |

## Four differences from the historical runs (reproducibility only)

Each one is recorded in every run's `protocol.json`. None changes what a metric measures.

1. **Safe, strict checkpoint load.**
   - Checkpoints are loaded with `torch.load(weights_only=True)` through `camera_ready/ckpt_io.py`, and the state dict with `strict=True`.
   - The historical harness used `load_from_checkpoint(strict=False)`, which can drop tensors silently.
2. **Seeded synthesis.**
   - Each utterance gets its own seed, `(2026 × 1,000,003 + index) mod 2^31`, set before its synthesis.
   - Historically synthesis was not seeded: two historical runs of the epoch-1079 model differ (WER 8.64 % vs 8.76 %, UTMOSv2 3.807 vs 3.813, `evaluation/eval_h_vs_baseline/` vs `f0_configH_v2/`).
3. **Seeded UTMOSv2.**
   - UTMOSv2 takes a random crop of the audio (`utmosv2/dataset/_utils.py`, `np.random.randint`). The same audio scored twice differed by up to ≈ 0.18 per utterance.
   - Numpy, `random` and torch are now seeded before each scoring call, for synthesized and reference audio alike.
   - The historical UTMOSv2 numbers include this scoring noise.
4. **Rows for the F0 predictor metric.**
   - The metric uses the dataset rows of the evaluated split, i.e. the 500 test rows.
   - The historical run passed some `--dataset-jsonl`; the log does not record which file. Its F0 numbers may have included other rows.

**Checked on 21/10.** On 5 validation utterances, two runs gave byte-identical WAVs and identical WER, UTMOSv2, F0, MCD and reference UTMOSv2. Only the timing columns (`infer_sec`, `rtf`) differ.

## RTF

The inline RTF depends on what else the machine runs. RTF numbers for the paper come from a separate measurement on an idle machine (T15): same CPU, threads declared, warm-up. Inline RTF of runs made while training or other evaluations run is recorded, but not reported as RTF.

## Order

1. The historical pair (773, 1079) and `PFN-Full-275ep-s1234` (best, epoch 188): on the CPU of MardeusPC on 21/10.
2. Each 200-epoch run when it is finalized.
3. `compare` for the main pair and for the ablation contrasts.

Results stay outside Git, in `<WSL_DEV_HOME>/outputs/eval/<label>/`. Only summaries derived from them are added to the repository later, with their hashes.
