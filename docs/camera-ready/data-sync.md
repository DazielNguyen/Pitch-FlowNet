# Data sync (outside Git)

**Git holds only code, configs, manifests and documents.**
- About 27 GB of data plus the checkpoints cannot be committed and must not be: dataset, feature cache, raw WAVs, checkpoints.
- `data-manifest.json` lists every item with its path on the PC and its SHA-256.
- `split-manifest.json` holds the SHA-256 of every raw WAV.

## What to move

| Group | Size | Needed for |
|---|---:|---|
| `data_preprocessed/` (`dataset.jsonl`, `config.json`, `cache/22050/`) | about 22 GB | training |
| `data/wavs/` | about 3.6 GB | evaluation references, waveform F0, re-making the split manifest |
| Checkpoints 773 and 1079 | 1.72 GB | evaluating the historical pair, tests |
| `test_entries_500.jsonl` | small | checks |

## How (no cloud service; nothing is uploaded)

1. **Pack on the PC**, inside WSL. Use one archive per group, read-only on the sources, with low priority so other training is not disturbed:
   ```sh
   cd <WSL_DEV_HOME>
   nice -n19 ionice -c3 tar -cf /path/on/external-or-lan/data_preprocessed.tar data_preprocessed
   nice -n19 ionice -c3 tar -cf /path/on/external-or-lan/wavs.tar data/wavs
   sha256sum /path/on/external-or-lan/*.tar > /path/on/external-or-lan/SHA256SUMS
   ```
   **Where the archives go:** the WSL virtual disk sits on Windows C:, which has about 53 GB free. Write the archives to an external drive or to `D:` (`/mnt/d`), never to C:.
2. **Move them** by external drive or over the local network (`rsync -av --progress`). Then check the archive hashes with `sha256sum -c SHA256SUMS`.
3. **Unpack on the training machine at the same absolute path.**
   - `dataset.jsonl` stores absolute paths to the cache, and `split-manifest.json` checks the exact hash of `dataset.jsonl`.
   - So keep the file byte-identical, and reproduce the directory `<WSL_DEV_HOME>/data_preprocessed` (a symlink is fine).
   - Do not edit `dataset.jsonl` to new paths. That changes its hash, and the split check then refuses to start.
4. **Verify after unpacking:**
   - `dataset.jsonl` and `config.json` against `data-manifest.json`;
   - the raw WAVs against `split-manifest.json`, for example:
     ```sh
     python - <<'PY'
     import hashlib, json, pathlib
     m = json.load(open("manifests/camera-ready/split-manifest.json"))
     wavs = pathlib.Path("<WSL_DEV_HOME>/data/wavs")
     bad = [e["id"] for s in m["splits"].values() for e in s
            if hashlib.sha256((wavs / f"{e['id']}.wav").read_bytes()).hexdigest() != e["raw_wav_sha256"]]
     print("mismatch:", len(bad), bad[:5])
     PY
     ```
   - the cache files against the evidence record's local `manifests/file-manifest.json`, if that file is copied too.
   - **Not done yet:** a small read-only `tools/camera_ready/verify_data.py` that runs all three checks; see PLAN.md.

## What not to do

- Do not commit data, checkpoints, WAVs or TensorBoard events. The repository `.gitignore` already excludes them.
- Do not put the data in OneDrive folders. OneDrive would upload tens of GB.
- Do not re-preprocess the dataset to "fix" paths. Doing so changes the data.
