"""Assemble <WSL_DEV_HOME>/Pitch-FlowNet from read-only sources (copy + hash; nothing is modified at the source)."""
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

DEST = Path("<WSL_DEV_HOME>/Pitch-FlowNet")
H = Path("<WSL_DEV_HOME>")
WS = Path("<WINDOWS_WORKSPACE>")
EDGE = WS / "EdgeTTS"
entries = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def label(p):
    s = str(p)
    for pre, lab in ((str(WS) + "/", "windows-workspace:"), ("<WSL_DEV_HOME>/", "wsl-dev:"), ("<WSL_PROJECT_HOME>/", "wsl-capstone:")):
        if s.startswith(pre):
            return lab + s[len(pre):]
    return s


def add(dest_rel, origin, cls, run, src_sha, note="", **flags):
    p = DEST / dest_rel
    entries.append({"destination": dest_rel, "origin": origin, "classification": cls, "run": run,
                    "bytes": p.stat().st_size, "source_sha256": src_sha, "destination_sha256": sha(p),
                    "note": note, **flags})


def copy_file(src, dest_rel, cls, run, note="", **flags):
    src = Path(src)
    d = DEST / dest_rel
    d.parent.mkdir(parents=True, exist_ok=True)
    if d.exists():
        raise SystemExit(f"refusing to overwrite {d}")
    s = sha(src)
    shutil.copy2(src, d)
    add(dest_rel, label(src), cls, run, s, note, **flags)


def copy_tree(src_dir, dest_rel, cls, run, note="", skip=lambda p: False):
    src_dir = Path(src_dir)
    for root, dirs, files in os.walk(src_dir):
        dirs.sort()
        for f in sorted(files):
            p = Path(root) / f
            if p.is_symlink():
                raise SystemExit(f"symlink found: {p}")
            rel = p.relative_to(src_dir)
            if skip(rel):
                continue
            copy_file(p, f"{dest_rel}/{rel.as_posix()}", cls, run, note)


def git(*args, repo=EDGE):
    return subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(repo), *args])


def git_export(commit, paths, dest_rel, cls, run, note=""):
    """Extract files of a commit from `git archive` (bytes of the committed blobs)."""
    data = git("archive", "--format=tar", commit, *paths)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            b = t.extractfile(m).read()
            d = DEST / dest_rel / m.name
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes(b)
            add(f"{dest_rel}/{m.name}", f"git:EdgeTTS@{commit}:{m.name}", cls, run, hashlib.sha256(b).hexdigest(), note)


def git_show(commit, path, dest_rel, cls, run, note=""):
    b = git("show", f"{commit}:{path}")
    d = DEST / dest_rel
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_bytes(b)
    add(dest_rel, f"git:EdgeTTS@{commit}:{path}", cls, run, hashlib.sha256(b).hexdigest(), note)


if DEST.exists():
    raise SystemExit(f"{DEST} exists; not overwriting")
DEST.mkdir()
for d in ("docs", "code/historical-candidates", "code/current-source", "code/vendor", "configs/baseline", "configs/full",
          "configs/exploratory", "datasets/raw", "datasets/processed", "datasets/splits", "datasets/features", "runs/baseline",
          "runs/full", "runs/candidates", "evaluation/baseline", "evaluation/full", "evaluation/ground-truth",
          "evaluation/statistics", "evaluation/plots", "artifacts", "history", "evidence", "manifests", "tools"):
    (DEST / d).mkdir(parents=True, exist_ok=True)

BASE = "baseline:02_Baseline_VanillaVITS"
FULL = "full:01_Baseline_BigVgan_VITS2_FO/version_2"

# ---------------------------------------------------------------- run-time records
copy_file(H / "02_Baseline_VanillaVITS/lightning_logs/version_0/hparams.yaml", "configs/baseline/02_Baseline_VanillaVITS_version_0_hparams.yaml", "CONTEMPORANEOUS_RECORD", BASE, "Lightning hparams written at launch")
copy_file(H / "01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/hparams.yaml", "configs/full/01_Baseline_BigVgan_VITS2_FO_version_2_hparams.yaml", "CONTEMPORANEOUS_RECORD", FULL, "Lightning hparams of the resumed session (version_2)")
for run, rel in (("exploratory:06_Config_E_BigVGAN_VITS2", "06_Config_E_BigVGAN_VITS2/lightning_logs/version_1/hparams.yaml"),
                 ("exploratory:09_Config_H_Full_ep773", "09_Config_H_Full_ep773/lightning_logs/version_0/hparams.yaml"),
                 ("exploratory:baseline_run", "baseline_run/lightning_logs/version_0/hparams.yaml")):
    copy_file(H / rel, "configs/exploratory/" + rel.replace("/lightning_logs/", "_").replace("/", "_"), "CONTEMPORANEOUS_RECORD", run)

copy_file(H / "02_baseline_train.log", "runs/baseline/02_baseline_train.log", "CONTEMPORANEOUS_RECORD", BASE, "stdout/stderr of the baseline run")
copy_file(H / "02_Baseline_VanillaVITS/lightning_logs/version_0/" / next(p.name for p in (H / "02_Baseline_VanillaVITS/lightning_logs/version_0").glob("events*")),
          "runs/baseline/02_Baseline_VanillaVITS/lightning_logs/version_0/" + next(p.name for p in (H / "02_Baseline_VanillaVITS/lightning_logs/version_0").glob("events*")),
          "CONTEMPORANEOUS_RECORD", BASE, "TensorBoard events (scalars, audio examples)")
copy_file(H / "02_Baseline_VanillaVITS/lightning_logs/version_0/checkpoints/best-epoch=773-val_loss_mel=20.2158.ckpt",
          "runs/baseline/02_Baseline_VanillaVITS/lightning_logs/version_0/checkpoints/best-epoch=773-val_loss_mel=20.2158.ckpt",
          "DIRECT_ARTIFACT", BASE, "selected checkpoint (manuscript epoch 773)")
copy_file(H / "01_Baseline_BigVgan_VITS2_FO/train.log", "runs/full/01_Baseline_BigVgan_VITS2_FO_train.log", "CONTEMPORANEOUS_RECORD", FULL,
          "log of the full-model sessions up to 2026-07-04 (aborted/killed attempts, resumes)")
copy_file(H / "train_h.log", "runs/full/train_h.log", "CONTEMPORANEOUS_RECORD", FULL, "log of full-model sessions to 2026-07-09 (includes a traceback and resumes)")
ev = next(p.name for p in (H / "01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2").glob("events*"))
copy_file(H / "01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2" / ev, "runs/full/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/" + ev,
          "CONTEMPORANEOUS_RECORD", FULL, "TensorBoard events of version_2 only; version_0/1 directories no longer exist")
copy_file(H / "01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt",
          "runs/full/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt",
          "DIRECT_ARTIFACT", FULL, "selected checkpoint (manuscript epoch 1079)")
# exploratory / candidate runs: small records only
for f, run in (("config_e_train.log", "exploratory:06_Config_E_BigVGAN_VITS2"), ("config_h_ep773_train.log", "exploratory:09_Config_H_Full_ep773"),
               ("baseline_train.log", "exploratory:baseline_run"), ("baseline_run.log", "exploratory:baseline_run"),
               ("bigvgan_train.log", "exploratory:bigvgan (no run directory found)"), ("preprocess.log", "dataset"),
               ("eval_baseline.log", "evaluation:eval_results/baseline_full"), ("ref_metrics.log", "evaluation:ground-truth reference metrics"),
               ("ref_metrics2.log", "evaluation:ground-truth reference metrics"), ("eval_f0_ablation_E_vs_H.log", "exploratory:F0 ablation E vs H"),
               ("eval_f0_ablation_diag.txt", "exploratory:F0 ablation E vs H"), ("run_eval100.sh", "exploratory:eval_100samples"),
               ("run_f0_ablation_eval.sh", "exploratory:F0 ablation E vs H"), ("train_bigvgan.sh", "exploratory:bigvgan"),
               ("tensorboard.log", "logs"), ("tensorboard_baseline.log", "logs"), ("tensorboard_bigvgan.log", "logs"), ("tensorboard_config_e.log", "logs")):
    copy_file(H / f, "runs/candidates/home-dev-root/" + f, "CONTEMPORANEOUS_RECORD", run)
for rel, run in (("06_Config_E_BigVGAN_VITS2/lightning_logs/version_1", "exploratory:06_Config_E_BigVGAN_VITS2"),
                 ("09_Config_H_Full_ep773/lightning_logs/version_0", "exploratory:09_Config_H_Full_ep773"),
                 ("baseline_run/lightning_logs/version_0", "exploratory:baseline_run")):
    for p in (H / rel).glob("events*"):
        copy_file(p, f"runs/candidates/{rel}/{p.name}", "CONTEMPORANEOUS_RECORD", run, "TensorBoard events (checkpoints catalogued only)")
for f in ("training_config.json", "train.log", "tensorboard-server.log"):
    copy_file(H / "BanhmiTTS/runs/full-config-h-cython-mas-batch16-20260714" / f,
              "runs/candidates/BanhmiTTS-port/full-config-h-cython-mas-batch16-20260714/" + f, "CONTEMPORANEOUS_RECORD",
              "exploratory:BanhmiTTS port of config H (not the evaluated pair)")
copy_tree(H / "BanhmiTTS/src", "code/current-source/home-dev-BanhmiTTS-port/src", "CURRENT_SOURCE", "exploratory:BanhmiTTS port",
          skip=lambda r: "__pycache__" in r.parts)
copy_tree(H / "BanhmiTTS/scripts", "code/current-source/home-dev-BanhmiTTS-port/scripts", "CURRENT_SOURCE", "exploratory:BanhmiTTS port")
copy_tree(H / "BanhmiTTS/docs", "code/current-source/home-dev-BanhmiTTS-port/docs", "CURRENT_SOURCE", "exploratory:BanhmiTTS port")

# ---------------------------------------------------------------- dataset
copy_tree(H / "data", "datasets/raw/home-dev-data", "DIRECT_ARTIFACT", "dataset:both runs", "LJ Speech 1.1 wavs + metadata.csv as used by preprocessing")
copy_tree(H / "data_preprocessed", "datasets/processed/data_preprocessed", "DIRECT_ARTIFACT", "dataset:both runs",
          "dataset.jsonl/config.json and cache/22050 (normalised audio, spectrograms, F0 tensors) read by both runs (hparams dataset path)")
copy_file(H / "test_entries_500.jsonl", "datasets/splits/test_entries_500.jsonl", "RECONSTRUCTED", "evaluation:500-sentence test set",
          "written 2026-07-10 by piper_train.extract_test_sentences (random_split seed 1234); first five entries equal the TensorBoard audio examples of both runs")
copy_file(H / "test_sentences_500.txt", "datasets/splits/test_sentences_500.txt", "RECONSTRUCTED", "evaluation:500-sentence test set")

# ---------------------------------------------------------------- evaluation outputs
copy_tree(H / "eval_h_vs_baseline", "evaluation/eval_h_vs_baseline", "CONTEMPORANEOUS_RECORD", "evaluation:baseline 773 vs full 1079 (manuscript pair)",
          "stored harness outputs incl. WAVs, per-sentence CSVs, summary.json, MCD/GT-UTMOS and F0 variants")
for rel, run in (("eval_results", "evaluation:earlier full-model evaluation 2026-07-10"),
                 ("eval_f0_ablation_E_vs_H", "exploratory:F0 ablation E vs H"),
                 ("eval_100samples", "exploratory:earlier runs epoch 861/863"),
                 ("eval_baseline_vs_piper_modern", "exploratory:earlier runs epoch 861/863")):
    copy_tree(H / rel, "evaluation/other/" + rel, "CONTEMPORANEOUS_RECORD", run)
for rel, dest, run in (("PitchFlowNet/eval_pitchflownet_500", "evaluation/full/eval_pitchflownet_500", "evaluation:later full-model re-run 2026-09-24"),
                       ("PitchFlowNet/eval_pitchflownet_500_gt", "evaluation/ground-truth/eval_pitchflownet_500_gt", "evaluation:ground truth 2026-09-24"),
                       ("PitchFlowNet/eval_results", "evaluation/other/PitchFlowNet-eval_results", "exploratory:earlier checkpoints")):
    copy_tree(H / rel, dest, "CONTEMPORANEOUS_RECORD", run)
for f, dest, run in (("PitchFlowNet/eval_pitchflownet_500_run.log", "evaluation/full/eval_pitchflownet_500_run.log", "evaluation:later full-model re-run 2026-09-24"),
                     ("PitchFlowNet/eval_pitchflownet_500_gt_run.log", "evaluation/ground-truth/eval_pitchflownet_500_gt_run.log", "evaluation:ground truth 2026-09-24"),
                     ("PitchFlowNet/measure_rtf.sh", "evaluation/full/measure_rtf.sh", "evaluation:timing script (current copy)")):
    copy_file(H / f, dest, "CONTEMPORANEOUS_RECORD" if f.endswith(".log") else "CURRENT_SOURCE", run)
copy_file(WS / "paper_revisions/thesis_ptq/history_archived_pair.py", "evaluation/statistics/history_archived_pair.py", "RECONSTRUCTED",
          "evaluation:manuscript pair", "later recomputation (thesis round) of means and Wilcoxon from the stored CSVs")
copy_file(WS / "paper_revisions/thesis_ptq/history_archived_pair.json", "evaluation/statistics/history_archived_pair.json", "RECONSTRUCTED",
          "evaluation:manuscript pair", "its output, with SHA-256 of the CSVs and summary it read")
copy_tree(EDGE / "data_processing_assets", "evaluation/plots/data_processing_assets", "CURRENT_SOURCE", "manuscript figures",
          "figure files and plotting scripts as they are now (relabelled PNGs and backups included)")

# ---------------------------------------------------------------- code
SRC = ["src/python", "configs", "LICENSE.md", "README.md", "TRAINING.md"]
for commit, note in (("6033be5", "last commit before the full-model run began (2026-07-03); working-tree changes at launch unknown"),
                     ("badf381", "first commit after the baseline launch (2026-07-10); adds .wsl_baseline.sh, eval tools, extract_test_sentences.py")):
    present = [p for p in SRC if subprocess.run(["git", "-c", "safe.directory=*", "-C", str(EDGE), "cat-file", "-e", f"{commit}:{p}"]).returncode == 0]
    git_export(commit, present, f"code/historical-candidates/edgetts-{commit}", "HISTORICAL_SOURCE_CANDIDATE", "both runs", note)
for commit, files in (("7ba86e2", [".wsl_train.sh", ".wsl_run.sh", ".wsl_fix_train.sh"]),
                      ("badf381", [".wsl_train.sh", ".wsl_baseline.sh"]),
                      ("ef20d0b", [".wsl_ablation.sh", "measure_rtf.sh"]),
                      ("6d0522b", [".wsl_train.sh", ".wsl_baseline.sh", ".wsl_ablation.sh", ".wsl_fix_train.sh", ".wsl_run.sh", "measure_rtf.sh"])):
    for f in files:
        git_show(commit, f, f"code/historical-candidates/launch-scripts/{commit}/{f}", "HISTORICAL_SOURCE_CANDIDATE", "both runs",
                 "launch script as committed; the executed command line is not recorded (hparams of version_2 differ from this file's resume path)")
present = [p for p in SRC + ["script", "docs/evaluation_methodology.md"]
           if subprocess.run(["git", "-c", "safe.directory=*", "-C", str(EDGE), "cat-file", "-e", f"HEAD:{p}"]).returncode == 0]
git_export("HEAD", present, "code/current-source/edgetts-windows-HEAD-a73a897", "CURRENT_SOURCE", "both runs", "current committed source of the PitchFlowNet repository")
copy_tree(H / "PitchFlowNet/src", "code/current-source/wsl-home-dev-PitchFlowNet-working-tree/src", "CURRENT_SOURCE", "both runs",
          "the directory training ran from (rsync target of .wsl_train.sh); state after its last sync, HEAD 85cab10 + uncommitted changes",
          skip=lambda r: "__pycache__" in r.parts or r.suffix in (".so", ".c", ".wav", ".onnx", ".ckpt", ".pt") or "build" in r.parts)
for f in ("TRAINING.md", "README.md", "LICENSE.md"):
    copy_file(H / "PitchFlowNet" / f, f"code/current-source/wsl-home-dev-PitchFlowNet-working-tree/{f}", "CURRENT_SOURCE", "both runs")
for p in (H / "PitchFlowNet").glob("*.patch"):
    copy_file(p, f"code/current-source/wsl-home-dev-PitchFlowNet-working-tree/{p.name}", "CURRENT_SOURCE", "unclear", "patch file found in the working tree")

# ---------------------------------------------------------------- history and context
hist = DEST / "history"
(hist / "private").mkdir(parents=True, exist_ok=True)
subprocess.check_call(["git", "-c", "safe.directory=*", "-C", str(EDGE), "bundle", "create", str(hist / "private/EdgeTTS-PitchFlowNet-all-refs.bundle"), "--all"],
                      stderr=subprocess.DEVNULL)
add("history/private/EdgeTTS-PitchFlowNet-all-refs.bundle", "git:EdgeTTS --all", "HISTORICAL_SOURCE_CANDIDATE", "both runs",
    None, "complete Git history (includes upstream Piper history); private, not scanned for secrets")
log = git("log", "--all", "--date=iso", "--format=%H%x09%ad%x09%an%x09%s", "--name-status")
(hist / "edgetts-commit-log-with-files.txt").write_bytes(log)
add("history/edgetts-commit-log-with-files.txt", "git:EdgeTTS log --all --name-status", "HISTORICAL_SOURCE_CANDIDATE", "both runs", hashlib.sha256(log).hexdigest())
wsl_log = git("log", "--all", "--date=iso", "--format=%H%x09%ad%x09%s", repo=H / "PitchFlowNet")
(hist / "wsl-PitchFlowNet-commit-log.txt").write_bytes(wsl_log)
add("history/wsl-PitchFlowNet-commit-log.txt", "git:<WSL_DEV_HOME>/PitchFlowNet log", "CURRENT_SOURCE", "both runs", hashlib.sha256(wsl_log).hexdigest())
st = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(H / "PitchFlowNet"), "status", "--porcelain"])
(hist / "wsl-PitchFlowNet-status-porcelain.txt").write_bytes(st)
add("history/wsl-PitchFlowNet-status-porcelain.txt", "git:<WSL_DEV_HOME>/PitchFlowNet status", "CURRENT_SOURCE", "both runs", hashlib.sha256(st).hexdigest())
for f in ("context_PitchFlowNet_2026-09-23.md", "PitchFlowNet_Conference_Draft_2026-09-24.txt", "question2_checkpoint_shapes_2026-09-23.json",
          "question2_parameter_counts_2026-09-23.json", "question2_sources_2026-09-23.json", "question3_audit_2026-09-23.json",
          "question6_audit_2026-09-23.json", "pruning_checkpoint_metadata_2026-09-23.json", "pruning_audit_windows_2026-09-23.json",
          "summarize_question2_parameters.py", "audit_question2_checkpoint.py", "audit_question2_sources.py", "audit_question3.py",
          "audit_question6.py", "audit_checkpoint_metadata.py", "audit_pruning_readonly.py"):
    cls = "AUTHOR_ACCOUNT" if f.endswith((".md", ".txt")) else "RECONSTRUCTED"
    copy_file(WS / "Compact" / f, "history/prior-audits-2026-09-23/" + f, cls, "both runs",
              "earlier assistant audit (2026-09-23); the author withdrew its unverified technical conclusions")
for f in ("Bao_cao_tien_do_PitchFlowNet_22-23_09_2026.md", "PitchFlowNet_nhat_ky_cau_hoi_truoc_khi_sua.md"):
    copy_file(WS / f, "history/notes/" + f, "AUTHOR_ACCOUNT", "manuscript review notes")
for rel in ("final/main_draft.tex", "final/README.md", "final/FINAL_REVIEW.md", "final/sn-bibliography.bib", "group_D/model_size_table.tex",
            "group_D/README_compact.md", "group_D/FINAL_REVIEW.md"):
    copy_file(WS / "paper_revisions" / rel, "evidence/manuscript/" + rel, "AUTHOR_ACCOUNT", "manuscript (not evidence)")

json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "entries": entries}, open(DEST / "manifests/file-manifest.json", "w"), indent=1)
print(len(entries), "entries")
