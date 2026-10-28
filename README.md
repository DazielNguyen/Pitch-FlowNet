# PitchFlowNet

**PitchFlowNet: An Empirical Study of Architectural Integration in Single-Stage Text-to-Speech**

Trần Hoàng Tuấn Hưng · Nguyễn Văn Anh Duy · Trần Huỳnh Bảo Minh · Hồ Lê Minh Toàn

Accepted at SOICT 2026 (paper 4478).

This repository holds the source code, configurations, training records, evaluation outputs and analysis tools of the paper.

## Overview

PitchFlowNet is built on the single-stage VITS architecture as implemented in [Piper](https://github.com/rhasspy/piper). It adds three groups of components:

| Group | Components | Flag |
|---|---|---|
| **B**, BigVGAN-style decoder | Snake activations in the decoder; multi-resolution discriminator | `use_bigvgan` |
| **V**, VITS2 | Self-attention in the flow coupling layers; duration discriminator; noised monotonic alignment search | `use_vits2` |
| **F**, F0 | Phoneme-level F0 predictor; F0 conditioning of the decoder | `use_f0` |

The paper compares two systems on LJ Speech:
- **PitchFlowNet**, with all three groups (epoch 1079);
- **a retrained Piper baseline** (epoch 773), trained with the same recipe.

For the camera-ready version, the paper adds a 2³ ablation of the three groups at a matched budget of 200 epochs.

## Results

### Main comparison (500 test utterances)

| Metric | Piper | PitchFlowNet | Difference [95 % BCa CI] | p (Holm) |
|---|---:|---:|---|---:|
| UTMOSv2 ↑ | 3.366 | 3.807 | +0.441 [+0.416, +0.468] | 4.9 × 10⁻⁷⁹ |
| WER (%) ↓ | 9.17 | 8.64 | −0.53 [−1.32, +0.23] | 0.229 |
| MCD ↓ | 26.064 | 26.000 | −0.064 [−0.175, +0.046] | 0.229 |
| Inference parameters | 15.66 M | 17.21 M | | |
| CPU RTF, 1 thread ↓ | 0.0789 | 0.0888 | | |

**How these numbers were obtained:**
- **Statistics:** paired differences per utterance, 95 % BCa bootstrap CIs (20,000 resamples) and two-sided Wilcoxon signed-rank tests, Holm-adjusted across the three metrics.
- **UTMOSv2** is a predicted naturalness score, not a listening test.
- **MCD** is a DTW distance on z-normalised MFCCs (C1–C24), not in dB.
- **WER:** the difference is not significant. That does not show that the two systems are equivalent.
- **RTF:** model-only RTF on an Intel Core i7-14700 CPU, batch 1, 1 thread; median of three rounds over 100 test utterances. The values at 2 and 12 threads are in the [final statistics](docs/camera-ready/ablation-final.md).
- **Ground truth:** WER 6.67 %, UTMOSv2 3.830 ([evaluation/ground-truth/](evaluation/ground-truth/)).

### Ablation (200 epochs, one seed per configuration, 500 test utterances)

| Configuration | B | V | F | WER (%) ↓ | UTMOSv2 ↑ | MCD ↓ |
|---|:-:|:-:|:-:|---:|---:|---:|
| Full | ✓ | ✓ | ✓ | 11.35 | 3.516 | 25.638 |
| NoF0 | ✓ | ✓ | | 13.02 | 3.189 | 25.576 |
| NoVITS2 | ✓ | | ✓ | 11.01 | 3.519 | 25.760 |
| NoBigVGAN | | ✓ | ✓ | 12.14 | 3.244 | 25.923 |
| Baseline-BigVGAN | ✓ | | | 10.09 | 3.495 | 25.744 |
| Baseline-VITS2 | | ✓ | | 11.05 | 3.372 | 25.623 |
| Baseline-F0 | | | ✓ | 13.49 | 3.060 | 26.728 |
| Baseline | | | | 10.82 | 2.941 | 25.573 |

**Main findings:**
- **UTMOSv2:** Full is higher than Baseline by +0.575 [+0.546, +0.604]. The exact Shapley attribution of this gain is BigVGAN 56 %, VITS2 21 % and F0 23 %.
- **Interactions:** the three groups interact strongly. The effect of VITS2, for example, depends on whether BigVGAN is present.
- **Pitch:** the F0 conditioning is used by the decoder, but it does not steer the pitch of the output. A requested shift of ±2 semitones moves the output pitch by less than 0.03 semitones.

The full analysis is in [docs/camera-ready/ablation-final.md](docs/camera-ready/ablation-final.md). It covers the paired tests of every contrast, the factorial effects, the pitch and duration analyses, cost against quality, and the limits of the study.

## Repository layout

| Path | Contents |
|---|---|
| [`code/camera-ready-source/`](code/camera-ready-source/) | **The model code used for the camera-ready experiments** (`src/python/piper_train/`) |
| [`code/historical-candidates/`](code/historical-candidates/) | Snapshots of the code base at the commits around the original training runs, and their launch scripts |
| [`code/current-source/`](code/current-source/) | Later snapshots of the code base and of the training directory |
| [`configs/`](configs/) | Launch hyperparameters of the original runs (`baseline/`, `full/`, `exploratory/`) and the ablation configs (`PFN-<variant>/`) |
| [`datasets/splits/`](datasets/splits/) | The 500 test utterances and the reconstructed train/validation/test split |
| [`evaluation/`](evaluation/) | Per-utterance scores, evaluation summaries, figures and their scripts |
| [`runs/`](runs/) | Training logs |
| [`manifests/`](manifests/) | Run identity, dataset and checkpoint manifests, file hashes, the camera-ready split manifest |
| [`tools/`](tools/), [`tests/`](tests/) | Training, evaluation and analysis tools, with their tests |
| [`docs/`](docs/) | Run identification, data access, camera-ready protocol and results |
| [`evidence/`](evidence/) | Each statement of the manuscript checked against the stored records; derived checkpoint and TensorBoard scans |

Datasets, checkpoints, TensorBoard events and generated audio are not included; see [docs/artifacts.md](docs/artifacts.md).

## Getting started

### Requirements

- Linux or WSL 2, Python 3.10, and a CUDA GPU for training.
- The evaluation runs on the CPU.
- The package versions recorded for the original runs are in [history/environment/](history/environment/).

```sh
cd code/camera-ready-source/src/python
pip install -r requirements.txt        # training
pip install -r requirements-eval.txt   # evaluation: Whisper, jiwer, UTMOSv2 (separate environment recommended)
./build_monotonic_align.sh             # Cython extension for monotonic alignment search
```

### Data

- **Corpus:** LJ Speech, preprocessed with Piper's `piper_train.preprocess`.
- **What the preprocessed folder must contain:** `dataset.jsonl` and `config.json`.
- **Split:** read by utterance ID from [manifests/camera-ready/split-manifest.json](manifests/camera-ready/split-manifest.json). It reproduces the split of the original runs (seed 1234), checked against the stored 500 test utterances.

### Training

The tools read their paths from environment variables:
- `CR_DATASET_DIR`: the preprocessed data;
- `CR_OUTPUTS`: runs, final checkpoints and evaluations;
- `CR_PYTHON`, `CR_EVAL_PYTHON`: the interpreters of the training and evaluation environments.

Without `--execute`, every launcher only checks its inputs and prints the command.

```sh
# Train one ablation configuration on GPU 0 (dry run first, then --execute)
tools/camera_ready/launch_train.sh configs/PFN-Full/PFN-Full-200ep.json --gpu 0
tools/camera_ready/launch_train.sh configs/PFN-Full/PFN-Full-200ep.json --gpu 0 --execute

# Resume after an interruption; archive the best and last checkpoints when the run ends
tools/camera_ready/resume_train.sh configs/PFN-Full/PFN-Full-200ep.json --gpu 0 --execute
tools/camera_ready/finalize_run.sh configs/PFN-Full/PFN-Full-200ep.json --expect-epochs 200
```

### Evaluation and analysis

The evaluation follows the locked protocol of [docs/camera-ready/eval-protocol.md](docs/camera-ready/eval-protocol.md): CPU, 4 threads, batch 1, with seeded synthesis and scoring.

```sh
tools/camera_ready/launch_eval.sh --split test --threads 4 --execute LABEL=<checkpoint.ckpt>

python tools/camera_ready/ablation8_analysis.py all --eval-dir "$CR_OUTPUTS/eval" --final-dir "$CR_OUTPUTS/final"
python tools/camera_ready/ablation_final.py --eval-dir "$CR_OUTPUTS/eval" --tests-dir "$CR_OUTPUTS/tests" --final-dir "$CR_OUTPUTS/final"
```

**Further analyses**, each documented in its docstring:
- `benchmark_rtf.py`: CPU RTF;
- `pitch_shift.py`: pitch intervention;
- `waveform_f0.py`: waveform F0 against the recordings;
- `duration_diag.py`: duration diagnostics.

### Tests

```sh
python -m unittest discover -s tests/camera_ready -v   # training, evaluation and analysis tools (CPU)
python tools/check_tracked_package.py                  # repository integrity
```

## Original training runs

| | Piper baseline | PitchFlowNet |
|---|---|---|
| Launch hyperparameters | [configs/baseline/](configs/baseline/) | [configs/full/](configs/full/) |
| Component flags | `use_bigvgan`, `use_vits2`, `use_f0` false | all three true |
| Selected checkpoint | epoch 773, step 605,268 | epoch 1079, step 844,560 |
| Shared recipe | 2 GPUs, DDP, bf16-mixed, batch 16 per GPU, seed 1234, 1,100 epochs, gradient clipping 1.0, AdamW 2 × 10⁻⁴ | same |

**Further documentation:**
- [docs/project-identity.md](docs/project-identity.md) explains how the two runs were identified.
- [evidence/manuscript-crosswalk.md](evidence/manuscript-crosswalk.md) checks each statement of the manuscript against the records.
- [docs/missing-information.md](docs/missing-information.md) lists the records that were not kept.

**Machine paths in records:** some records contain machine paths. They are tracked as `*.sanitized.*` copies, with placeholders such as `<WSL_DEV_HOME>`. The map of each pair is in [manifests/sanitization/](manifests/sanitization/).

## Citation

```bibtex
@inproceedings{pitchflownet,
  title     = {PitchFlowNet: An Empirical Study of Architectural Integration in Single-Stage Text-to-Speech},
  author    = {Trần, Hoàng Tuấn Hưng and Nguyễn, Văn Anh Duy and Trần, Huỳnh Bảo Minh and Hồ, Lê Minh Toàn},
  booktitle = {Proceedings of the International Symposium on Information and Communication Technology (SOICT 2026)},
  year      = {2026},
  note      = {To appear}
}
```

See also [CITATION.cff](CITATION.cff).

## License

| Material | Terms |
|---|---|
| The authors' code, tools and documents | [MIT](LICENSE) |
| Piper-derived code in `code/` | MIT, copyright Michael Hansen; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |
| LJ Speech | Public domain; the data files are not included |
| Checkpoints and generated audio | Not distributed in this repository |

## Acknowledgements

This work builds on these projects:
- [Piper](https://github.com/rhasspy/piper);
- VITS and VITS2;
- BigVGAN;
- [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2);
- [Whisper](https://github.com/openai/whisper);
- the [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) dataset.
