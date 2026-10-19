"""Camera-ready training entry point.

    python -m piper_train.camera_ready.train --config <run config .json> --dataset-dir <data_preprocessed> \
        --split-manifest <split-manifest.json> --run-dir <new directory> [--resume] [--dry-run]

A fresh run refuses an existing run directory. --resume continues the same run from
<run-dir>/checkpoints/last.ckpt (same checkpoint directory, so the best-checkpoint state is kept).
--dry-run validates every input and prints the resolved settings without creating anything.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

REQUIRED_CONFIG_KEYS = {"run_id", "model", "protocol", "trainer"}
FORBIDDEN_MODEL_KEYS = {"dataset", "split_manifest", "group_semantics", "seed", "mas_noise_scale_decay"}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def source_tree_sha(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def git_state(path: Path) -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=path, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_paths": (run("status", "--porcelain") or "").splitlines()}


def is_rank_zero() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--split-manifest", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--resume", action="store_true", help="continue this run from <run-dir>/checkpoints/last.ckpt")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-train-batches", type=int, default=None, help="smoke tests only; recorded in the run manifest")
    p.add_argument("--stop-after-epochs", type=int, default=None,
                   help="tests only: end this attempt after N epochs in total (emulates an interruption); recorded")
    return p.parse_args(argv)


def resolve(args) -> dict:
    """Validate inputs; return the resolved plan. Raises SystemExit with a reason on any problem."""
    problems = []
    cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config.is_file() else None
    if cfg is None:
        problems.append(f"config not found: {args.config}")
    else:
        missing = REQUIRED_CONFIG_KEYS - set(cfg)
        if missing:
            problems.append(f"config lacks {sorted(missing)}")
        bad = FORBIDDEN_MODEL_KEYS & set(cfg.get("model", {}))
        if bad:
            problems.append(f"config.model must not set {sorted(bad)} (they are fixed by the protocol)")
        if "max_g_updates" not in cfg.get("protocol", {}):
            problems.append("config.protocol.max_g_updates is required (the budget is in generator updates)")
    for name in ("dataset.jsonl", "config.json"):
        if not (args.dataset_dir / name).is_file():
            problems.append(f"missing {args.dataset_dir / name}")
    if not args.split_manifest.is_file():
        problems.append(f"missing split manifest {args.split_manifest}")
    last = args.run_dir / "checkpoints" / "last.ckpt"
    if args.resume:
        if not last.is_file():
            problems.append(f"--resume needs {last}")
        if not (args.run_dir / "run_manifest.json").is_file():
            problems.append("--resume needs the run_manifest.json of the run")
    elif args.run_dir.exists() and is_rank_zero():  # DDP child processes re-run this script after rank 0 created it
        problems.append(f"run directory exists: {args.run_dir} (use a new directory, or --resume for the same run)")
    if problems:
        raise SystemExit("refusing to start:\n  - " + "\n  - ".join(problems))
    return {"config": cfg, "resume_from": str(last) if args.resume else None}


def build_manifest(args, plan, src_root: Path, repo_root: Path) -> dict:
    ds = args.dataset_dir / "dataset.jsonl"
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append({"index": i, "name": torch.cuda.get_device_name(i)})
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines()
    except FileNotFoundError:
        smi = []
    import numpy
    import pytorch_lightning as pl
    return {
        "schema": "camera-ready run manifest v1",
        "run_id": plan["config"]["run_id"],
        "argv": sys.argv,
        "config_path": str(args.config), "config_sha256": sha_file(args.config), "config": plan["config"],
        "dataset_jsonl_sha256": sha_file(ds), "dataset_config_sha256": sha_file(args.dataset_dir / "config.json"),
        "split_manifest_sha256": sha_file(args.split_manifest),
        "source": {"tree_sha256": source_tree_sha(src_root / "piper_train"), "git": git_state(repo_root)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "pytorch_lightning": pl.__version__,
                        "numpy": numpy.__version__, "platform": platform.platform()},
        "devices": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": gpus, "nvidia_smi": smi},
        "determinism": {"cudnn_benchmark": True, "deterministic_algorithms": False,
                        "known_nondeterminism": "cuDNN autotuning and atomic-add kernels make GPU runs not bit-reproducible"},
        "smoke_limit_train_batches": args.limit_train_batches,
        "attempts": [],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = resolve(args)
    cfg = plan["config"]
    src_root = Path(__file__).resolve().parents[2]       # .../src/python
    repo_root = src_root.parents[1]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "run_dir": str(args.run_dir), "resume_from": plan["resume_from"],
                          "config": cfg, "inputs_ok": True}, indent=1))
        return 0

    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger

    from .callbacks import GUpdateBudget, LedgerModelCheckpoint
    from .model import MONITOR, CameraReadyVitsModel

    run_dir = args.run_dir
    manifest_path = run_dir / "run_manifest.json"
    attempt = {"kind": "resume" if args.resume else "start",
               "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "resume_from": plan["resume_from"], "stop_after_epochs": args.stop_after_epochs,
               "limit_train_batches": args.limit_train_batches}
    if is_rank_zero():
        run_dir.mkdir(parents=True, exist_ok=args.resume)
        manifest = json.loads(manifest_path.read_text()) if args.resume else build_manifest(args, plan, src_root, repo_root)
        if args.resume and manifest["config_sha256"] != sha_file(args.config):
            raise SystemExit("the config differs from the one this run started with")
        manifest["attempts"].append(attempt)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    protocol = cfg["protocol"]
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(int(protocol["train_seed"]), workers=False)   # model initialisation seed

    with open(args.dataset_dir / "config.json", encoding="utf-8") as f:
        dcfg = json.load(f)
    model = CameraReadyVitsModel(
        num_symbols=int(dcfg["num_symbols"]), num_speakers=int(dcfg["num_speakers"]),
        # A str (not a Path) keeps checkpoints loadable with weights_only=True and no allowlist.
        sample_rate=int(dcfg["audio"]["sample_rate"]), dataset=[str(args.dataset_dir / "dataset.jsonl")],
        split_manifest=str(args.split_manifest),
        train_seed=int(protocol["train_seed"]), data_seed=int(protocol["data_seed"]), val_seed=int(protocol["val_seed"]),
        train_shuffle=bool(protocol["train_shuffle"]), mas_decay_per_g_update=float(protocol["mas_decay_per_g_update"]),
        preview_every_n_epochs=int(protocol.get("preview_every_n_epochs", 1)),
        diagnostics_path=str(run_dir / "val_diagnostics.jsonl"),
        **cfg["model"],
    )
    ckpt_dir = run_dir / "checkpoints"
    ledger = str(run_dir / "checkpoint_ledger.jsonl")
    callbacks = [
        GUpdateBudget(int(protocol["max_g_updates"])),
        LedgerModelCheckpoint(dirpath=ckpt_dir, filename="best-{epoch}-{" + MONITOR + ":.4f}", monitor=MONITOR, mode="min",
                              save_top_k=1, save_last=False, ledger_path=ledger, role="best"),
        LedgerModelCheckpoint(dirpath=ckpt_dir, filename="epoch-{epoch}", every_n_epochs=1, save_top_k=1, save_last=True,
                              ledger_path=ledger, role="rolling"),
    ]
    trainer_cfg = cfg["trainer"]
    trainer = Trainer(
        accelerator=trainer_cfg["accelerator"], devices=trainer_cfg["devices"], strategy=trainer_cfg["strategy"],
        precision=trainer_cfg["precision"],
        max_epochs=min(int(trainer_cfg["max_epochs"]), args.stop_after_epochs or int(trainer_cfg["max_epochs"])),
        default_root_dir=str(run_dir), logger=TensorBoardLogger(save_dir=str(run_dir), name="tb", version="run"),
        callbacks=callbacks, use_distributed_sampler=False, log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 50)),
        num_sanity_val_steps=int(trainer_cfg.get("num_sanity_val_steps", 2)),
        limit_train_batches=args.limit_train_batches if args.limit_train_batches else 1.0,
        enable_progress_bar=bool(trainer_cfg.get("enable_progress_bar", True)),
    )
    status, error = "completed", None
    try:
        trainer.fit(model, ckpt_path=plan["resume_from"])
    except BaseException as exc:  # recorded, then re-raised
        status, error = "failed", repr(exc)
        raise
    finally:
        if is_rank_zero() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest["attempts"][-1].update({
                "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "status": status, "error": error, "g_updates": int(model.g_updates), "d_updates": int(model.d_updates),
                "samples_seen_global": int(model.samples_seen_global), "epochs_completed": int(trainer.current_epoch),
                "lightning_global_step": int(trainer.global_step),
                "best_checkpoint": Path(callbacks[1].best_model_path).name if callbacks[1].best_model_path else None,
                "best_score": float(callbacks[1].best_model_score) if callbacks[1].best_model_score is not None else None,
            })
            manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
