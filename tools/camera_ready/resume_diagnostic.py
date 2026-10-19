"""Diagnose the resume continuity check (PLAN O1) on the CPU, in deterministic mode.

    python tools/camera_ready/resume_diagnostic.py [--threads 1] [--output report.json]

Uses the CPU smoke config and the tiny split of tests/camera_ready (4 train / 3 validation utterances,
2 batches per epoch, 2 epochs). Runs, each in a new directory under CR_TEST_OUTPUT:

  A1, A2   uninterrupted 2-epoch runs              -> noise floor (should be identical when deterministic)
  H1, H2   1-epoch runs (= first half of B)        -> is epoch 0 deterministic on its own
  B        1 epoch, then --resume to 2 epochs      -> compared with A1

For each pair it reports the number of differing tensors and the largest absolute difference of the
weights and the optimizer moments, the scheduler states, the camera_ready counters, the Lightning
loop counters and the validation metric per epoch. Read-only apart from the run directories and the
report. Does not use the GPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "camera_ready"))

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--output", type=Path, default=None)
args = ap.parse_args()

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

from _common import DATASET_DIR, REPO, new_run_dir, small_manifest  # noqa: E402
from piper_train.camera_ready import train as cr_train  # noqa: E402

CONFIG = REPO / "configs" / "camera-ready-tests" / "cpu-smoke-H.json"
MANIFEST = small_manifest(n_train=4, n_val=3, n_test=2)


def run(tag, *extra):
    d = new_run_dir(tag)
    argv = ["--config", str(CONFIG), "--dataset-dir", str(DATASET_DIR), "--split-manifest", str(MANIFEST),
            "--run-dir", str(d), "--limit-train-batches", "2", *extra]
    cr_train.main(argv)
    return d


def resume(d):
    cr_train.main(["--config", str(CONFIG), "--dataset-dir", str(DATASET_DIR), "--split-manifest", str(MANIFEST),
                   "--run-dir", str(d), "--limit-train-batches", "2", "--resume"])
    return d


def load(d):
    return torch.load(d / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True)


def tensor_diff(a: dict, b: dict):
    keys = sorted(set(a) | set(b))
    differ, worst, worst_key = 0, 0.0, None
    for k in keys:
        if k not in a or k not in b:
            differ += 1
            continue
        x, y = a[k], b[k]
        if not torch.is_tensor(x):
            continue
        if not torch.equal(x, y):
            differ += 1
            m = (x.double() - y.double()).abs().max().item() if x.is_floating_point() else float("nan")
            if m > worst:
                worst, worst_key = m, k
    return {"tensors": len(keys), "differ": differ, "max_abs": worst, "max_abs_key": worst_key}


def opt_diff(a, b):
    out = []
    for i, (oa, ob) in enumerate(zip(a["optimizer_states"], b["optimizer_states"])):
        moments = {}
        for name in ("exp_avg", "exp_avg_sq"):
            moments[name] = tensor_diff({k: v[name] for k, v in oa["state"].items()},
                                        {k: v[name] for k, v in ob["state"].items()})
        steps_a = sorted({float(v["step"]) for v in oa["state"].values()})
        steps_b = sorted({float(v["step"]) for v in ob["state"].values()})
        out.append({"optimizer": i, **moments, "step_a": steps_a, "step_b": steps_b,
                    "param_groups_equal": oa["param_groups"] == ob["param_groups"]})
    return out


def val_series(d):
    p = d / "val_diagnostics.jsonl"
    return [{"epoch": r["epoch"], "g_updates": r["g_updates"], "value": r["value"]}
            for r in map(json.loads, p.read_text().splitlines())] if p.is_file() else []


def compare(name, da, db):
    a, b = load(da), load(db)
    return {
        "pair": name, "dirs": [str(da), str(db)],
        "state_dict": tensor_diff(a["state_dict"], b["state_dict"]),
        "optimizers": opt_diff(a, b),
        "lr_schedulers_equal": a["lr_schedulers"] == b["lr_schedulers"],
        "lr_schedulers": [a["lr_schedulers"], b["lr_schedulers"]],
        "camera_ready": [a["camera_ready"], b["camera_ready"]],
        "epoch_global_step": [[a["epoch"], a["global_step"]], [b["epoch"], b["global_step"]]],
        "val": [val_series(da), val_series(db)],
    }


a1, a2 = run("diag-A1"), run("diag-A2")
h1, h2 = run("diag-H1", "--stop-after-epochs", "1"), run("diag-H2", "--stop-after-epochs", "1")
b = run("diag-B", "--stop-after-epochs", "1")
b_half = load(b)
resume(b)

report = {
    "settings": {"threads": args.threads, "interop_threads": 1, "deterministic_algorithms": True,
                 "torch": torch.__version__, "config": str(CONFIG.relative_to(REPO))},
    "pairs": [compare("A1 vs A2 (noise floor, 2 epochs)", a1, a2),
              compare("H1 vs H2 (noise floor, 1 epoch)", h1, h2),
              compare("A1 vs B (uninterrupted vs resumed)", a1, b)],
    "B_after_first_attempt": {"epoch": b_half["epoch"], "global_step": b_half["global_step"],
                              "camera_ready": b_half["camera_ready"], "lr_schedulers": b_half["lr_schedulers"],
                              "loops_fit_epoch": b_half.get("loops", {}).get("fit_loop", {}).get("epoch_progress")},
}
text = json.dumps(report, indent=1, default=str)
if args.output:
    args.output.write_text(text + "\n")
for p in report["pairs"]:
    sd = p["state_dict"]
    print(f"{p['pair']}: {sd['differ']}/{sd['tensors']} tensors differ, max |diff| {sd['max_abs']:.3e} ({sd['max_abs_key']}); "
          f"schedulers equal {p['lr_schedulers_equal']}; val {[round(v['value'], 4) for v in p['val'][0]]} "
          f"vs {[round(v['value'], 4) for v in p['val'][1]]}")
print("B after first attempt:", json.dumps(report["B_after_first_attempt"], default=str))
