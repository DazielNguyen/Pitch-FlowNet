# Third-party notices

This repository contains copies of third-party code inside the source snapshots under `code/`.
Each notice below is kept as it appears in those files. Nothing here re-licenses third-party work.

The licence of the project's own changes and of this repository as a whole has **not been decided**.
See [Licensing](README.md#licensing).

| Component | Where | Licence (as stated in the files) |
|---|---|---|
| Piper (rhasspy/piper), copyright 2022 Michael Hansen. EdgeTTS, the code base of every snapshot, is a fork of it | `code/historical-candidates/edgetts-*/`, `code/current-source/edgetts-windows-HEAD-a73a897/`, `code/current-source/wsl-home-dev-PitchFlowNet-working-tree/` | MIT. The text is in the `LICENSE.md` of each snapshot |
| JSON for Modern C++ (nlohmann/json), 2013–2022 Niels Lohmann | `code/current-source/wsl-home-dev-PitchFlowNet-working-tree/src/cpp/json.hpp` | MIT (SPDX header in the file) |
| UTF8-CPP (utfcpp), 2006 Nemanja Trifunovic | `code/current-source/wsl-home-dev-PitchFlowNet-working-tree/src/cpp/utf8.h`, `src/cpp/utf8/` | Licence text in each file header. It is the Boost Software License 1.0 wording |
| Silero VAD model (snakers4/silero-vad), referenced by `piper_train/norm_audio/vad.py` | `…/piper_train/norm_audio/models/silero_vad.onnx` in the snapshots | **Not tracked** (`*.onnx` is ignored). Obtain it from the upstream project under its own terms. The local copy's hash is in `manifests/file-manifest-core.json` |

**Files that carry no licence notice:**
- The BanhmiTTS port code (`code/current-source/home-dev-BanhmiTTS-port/`).
- The project's own scripts, tools and documents.

Their terms are decided by the authors and are still open.

**Model components named in the paper** (VITS, VITS2, HiFi-GAN, BigVGAN Snake activations, a multi-resolution discriminator):
- They are implemented inside the Piper-derived code above.
- This repository has not audited which upstream projects individual functions were adapted from, beyond the notices listed here.
