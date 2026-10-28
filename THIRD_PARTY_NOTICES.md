# Third-party notices

This repository contains copies of third-party code inside the source snapshots under `code/`.
Each notice below is kept as it appears in those files. Nothing here re-licenses third-party work.

The authors' own code, tools and documents are released under the [MIT License](LICENSE).
See [License](README.md#license).

| Component | Where | Licence (as stated in the files) |
|---|---|---|
| Piper (rhasspy/piper), copyright 2022 Michael Hansen. EdgeTTS, the code base of every snapshot, is a fork of it | `code/camera-ready-source/`, `code/historical-candidates/edgetts-*/`, `code/current-source/edgetts-windows-HEAD-a73a897/`, `code/current-source/wsl-home-dev-PitchFlowNet-working-tree/` | MIT. The text is in the `LICENSE.md` of each snapshot |
| JSON for Modern C++ (nlohmann/json), 2013–2022 Niels Lohmann | `code/current-source/wsl-home-dev-PitchFlowNet-working-tree/src/cpp/json.hpp` | MIT (SPDX header in the file) |
| UTF8-CPP (utfcpp), 2006 Nemanja Trifunovic | `code/current-source/wsl-home-dev-PitchFlowNet-working-tree/src/cpp/utf8.h`, `src/cpp/utf8/` | Licence text in each file header. It is the Boost Software License 1.0 wording |
| Silero VAD model (snakers4/silero-vad), referenced by `piper_train/norm_audio/vad.py` | `…/piper_train/norm_audio/models/silero_vad.onnx` in the snapshots | **Not tracked** (`*.onnx` is ignored). Obtain it from the upstream project under its own terms. The local copy's hash is in `manifests/file-manifest-core.json` |

**The authors' own work:** the project's changes to the Piper code, and the scripts, tools and documents, are covered by the repository's [MIT License](LICENSE).

**No licence notice:** the BanhmiTTS port (`code/current-source/home-dev-BanhmiTTS-port/`) is a separate project and carries none.

**Model components named in the paper** (VITS, VITS2, HiFi-GAN, BigVGAN Snake activations, a multi-resolution discriminator):
- They are implemented inside the Piper-derived code above.
- This repository has not audited which upstream projects individual functions were adapted from, beyond the notices listed here.
