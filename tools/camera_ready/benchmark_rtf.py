"""RTF benchmark of the eight ablation checkpoints on the CPU with 1 and 2 threads (T15, protocol agreed on 2026-10-24).

    python tools/camera_ready/benchmark_rtf.py plan --final-dir <CR_OUTPUTS>/final --eval-dir <CR_OUTPUTS>/eval [--execute]
    python tools/camera_ready/benchmark_rtf.py historical --checkpoint Historical-Full-1079=<ckpt> \
        --checkpoint Historical-Baseline-773=<ckpt> --eval-dir <CR_OUTPUTS>/eval [--execute]
    (internal) python tools/camera_ready/benchmark_rtf.py run-one --checkpoint <ckpt> --label L --threads T --round R --out <jsonl>

Without --execute, plan only checks the inputs and the machine and prints the schedule. With --execute it runs every
(round, label, threads) cell in its own process, pinned with taskset, and writes <eval-dir>/ablation8-rtf-runs.jsonl,
ablation8-rtf.json and ablation8-rtf.csv. It stops when the machine is not idle (load average >= 1, a tmux session
other than its own cr-rtf, or GPU work), when a checkpoint does not match final/<run_id>/SHA256SUMS, or when an output file exists.

historical does the same for the paper's historical pair (Full epoch 1079, Baseline epoch 773; added 2026-10-28).
Each checkpoint must have the file name and SHA-256 recorded in manifests/checkpoint-manifest.json; the outputs are
<eval-dir>/historical-pair-rtf-runs.jsonl, historical-pair-rtf.json and historical-pair-rtf.csv. --threads T [T ...]
times other settings (up to 12, pinned like the defaults to the first vCPU of each virtual core: 0, 2, 4, ...); the
outputs are then named historical-pair-rtf-t<T>-runs.jsonl and so on. The checkpoints load
through evaluate_like_history.load_model in "legacy" group semantics, strictly, not through the historical harness's
load_from_checkpoint(strict=False).

Protocol:
  - CPU only; torch.set_num_threads(T), set_num_interop_threads(1), OMP/MKL threads T; T = 1 pinned to vCPU 0,
    T = 2 to vCPUs 0 and 2 (two virtual cores). Under WSL the vCPUs are scheduled by Hyper-V on host cores, so
    P-core or E-core placement is not controlled; three rounds show the spread.
  - The synthesis path of the evaluation harness: checkpoint loaded with weights_only=True and strict state dict
    (evaluate_like_history.load_model), eval mode, decoder weight norm removed, eval_harness.synthesize with
    noise_scale 0.667, length_scale 1.0, noise_w 0.8, seeded per test utterance as in the evaluation.
  - 10 validation utterances as warm-up (not timed in the result), then the first 100 test utterances of the split
    manifest. Model-only time is the harness's own window (model forward); end-to-end adds phonemization (espeak)
    and tensor building.
  - Three rounds; the model order is rotated each round; each round runs T = 1 then T = 2 for each model.
  - Reported per label and T: corpus RTF = sum(time) / sum(audio duration), and the mean per-utterance RTF as the
    historical harness reports, both as the median and the range over the three rounds.
"""
import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "camera-ready-source" / "src" / "python"))

THREADS = (1, 2)
MAX_THREADS = 12                   # one per virtual core: vCPUs 2k and 2k + 1 are siblings of core k


def cpuset(t):
    """The vCPUs for t threads: the first of each of the first t virtual cores (0, 2, 4, ...)."""
    if not 1 <= t <= MAX_THREADS:
        raise ValueError(f"threads must be 1..{MAX_THREADS}, not {t}")
    return ",".join(str(2 * i) for i in range(t))


CPUSET = {t: cpuset(t) for t in THREADS}
ROUNDS = 3
N_TEST, N_WARMUP = 100, 10
LANGUAGE = "en-us"
SESSION = "cr-rtf"                 # the benchmark's own tmux session, allowed by the idle check
CHECKPOINT_MANIFEST = ROOT / "manifests" / "checkpoint-manifest.json"
HISTORICAL = {                     # label -> checkpoint file name in checkpoint-manifest.json
    "Historical-Full-1079": "best-epoch=1079-val_loss_mel=19.2943.ckpt",
    "Historical-Baseline-773": "best-epoch=773-val_loss_mel=20.2158.ckpt",
}


def round_order(labels, r):
    """The labels rotated by r places, so that each label runs at a different position in each round."""
    k = r % len(labels)
    return list(labels[k:]) + list(labels[:k])


def schedule(labels, rounds=ROUNDS, threads=THREADS):
    return [(r, label, t) for r in range(rounds) for label in round_order(labels, r) for t in threads]


def summarize(records):
    """Per (label, threads): median and range over rounds of corpus RTF and mean per-utterance RTF (model only and
    end to end), plus whether the audio lengths agree across rounds and thread settings."""
    out = {}
    for rec in records:
        out.setdefault((rec["label"], rec["threads"]), []).append(rec)
    lengths = {}
    for rec in records:
        lengths.setdefault(rec["label"], set()).add(tuple(rec["samples"]))
    rows = []
    for (label, t), recs in sorted(out.items()):
        row = {"label": label, "threads": t, "rounds": len(recs)}
        for kind in ("model", "e2e"):
            corpus = [sum(r[f"{kind}_sec"]) / sum(r["audio_sec"]) for r in recs]
            per_utt = [statistics.fmean(x / d for x, d in zip(r[f"{kind}_sec"], r["audio_sec"])) for r in recs]
            row[f"{kind}_corpus_rtf_median"] = statistics.median(corpus)
            row[f"{kind}_corpus_rtf_min"], row[f"{kind}_corpus_rtf_max"] = min(corpus), max(corpus)
            row[f"{kind}_mean_utt_rtf_median"] = statistics.median(per_utt)
        row["audio_sec_total"] = sum(recs[0]["audio_sec"])
        row["audio_lengths_identical_across_runs"] = len(lengths[label]) == 1
        rows.append(row)
    return rows


# ---------------------------------------------------------------- child: one model, one thread setting, one round

def cmd_run_one(a):
    import torch
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    import evaluate_like_history as elh
    from piper_train import eval_harness as eh

    model, info = elh.load_model(a.checkpoint)
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()
    test_rows, test_ids, _ = elh.split_rows("test", N_TEST)
    warm_rows, _, _ = elh.split_rows("validation", N_WARMUP)
    seed_of = lambda i: (2026 * 1_000_003 + i) % (2 ** 31)

    for i, r in enumerate(warm_rows):
        torch.manual_seed(seed_of(10 ** 7 + i))
        eh.synthesize(model, r["text"], LANGUAGE, elh.SCALES, elh.SAMPLE_RATE)
    model_sec, e2e_sec, audio_sec, samples = [], [], [], []
    for i, r in enumerate(test_rows):
        torch.manual_seed(seed_of(i))                       # the evaluation's seed of test utterance i
        t0 = time.perf_counter()
        audio, infer_sec, dur, _ = eh.synthesize(model, r["text"], LANGUAGE, elh.SCALES, elh.SAMPLE_RATE)
        e2e_sec.append(time.perf_counter() - t0)
        model_sec.append(infer_sec)
        audio_sec.append(dur)
        samples.append(int(audio.shape[-1]))
    rec = {"label": a.label, "threads": a.threads, "round": a.round, "checkpoint": str(a.checkpoint),
           "ids_first_last": [test_ids[0], test_ids[-1]], "n": len(test_rows), "flags": info["flags"],
           "torch_threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)),
           "loadavg_before": a.loadavg, "model_sec": model_sec, "e2e_sec": e2e_sec, "audio_sec": audio_sec,
           "samples": samples, "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{a.label} T={a.threads} round {a.round}: model RTF {sum(model_sec) / sum(audio_sec):.4f}, "
          f"e2e RTF {sum(e2e_sec) / sum(audio_sec):.4f}")


# ---------------------------------------------------------------- parent: checks, schedule, aggregation

def machine_state():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    ls = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True)
    sessions = ls.stdout.split() if ls.returncode == 0 else []
    mhz = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
    return {"loadavg": os.getloadavg(), "tmux_sessions": sessions, "gpus": gpu,
            "cpu_mhz_reported": [min(mhz), max(mhz)] if mhz else None, "cpus": os.cpu_count(), "platform": platform.platform()}


def idle_problems(state):
    problems = []
    if state["loadavg"][0] >= 1.0:
        problems.append(f"load average {state['loadavg'][0]:.2f} >= 1")
    others = [s for s in state["tmux_sessions"] if s != SESSION]
    if others:
        problems.append(f"other tmux sessions run (training, evaluation or other work): {', '.join(others)}")
    for g in state["gpus"]:
        idx, util, mem = (x.strip() for x in g.split(","))
        if int(util) >= 10:
            problems.append(f"GPU {idx} at {util} % utilisation")
    return problems


def historical_cells(pairs, manifest=CHECKPOINT_MANIFEST, sha256=None):
    """(label, checkpoint) for the historical pair from LABEL=PATH arguments, and the problems found: a label other
    than the two of HISTORICAL, a missing label, a file name other than the recorded one, or a SHA-256 other than
    the one checkpoint-manifest.json records for that file."""
    if sha256 is None:
        import ablation8_analysis as A
        sha256 = A.sha256
    recorded = {e["file"]: e["sha256"] for e in json.loads(pathlib.Path(manifest).read_text())["checkpoints"]}
    problems, given = [], {}
    for item in pairs:
        label, _, path = item.partition("=")
        if label not in HISTORICAL:
            problems.append(f"unknown label {label!r}; expected {', '.join(HISTORICAL)}")
            continue
        given[label] = pathlib.Path(path)
    problems += [f"missing --checkpoint {label}=<path>" for label in HISTORICAL if label not in given]
    cells = []
    for label, ckpt in given.items():
        if ckpt.name != HISTORICAL[label]:
            problems.append(f"{label}: file {ckpt.name} is not {HISTORICAL[label]}")
        elif not ckpt.is_file():
            problems.append(f"{label}: {ckpt} does not exist")
        elif sha256(ckpt) != recorded.get(ckpt.name):
            problems.append(f"{label}: {ckpt} SHA-256 differs from {pathlib.Path(manifest).name}")
        cells.append((label, ckpt))
    return cells, problems


def cmd_historical(a):
    cells, problems = historical_cells(a.checkpoint)
    threads = tuple(a.threads)
    stem = "historical-pair-rtf" if threads == THREADS else "historical-pair-rtf-t" + "-".join(map(str, threads))
    outs = [a.eval_dir / f for f in (f"{stem}-runs.jsonl", f"{stem}.json", f"{stem}.csv")]
    recorded = {e["file"]: e["sha256"] for e in json.loads(CHECKPOINT_MANIFEST.read_text())["checkpoints"]}
    extra = {"checkpoints": {label: {"file": ckpt.name, "sha256": recorded.get(ckpt.name)} for label, ckpt in cells},
             "loading": "evaluate_like_history.load_model, legacy group semantics, strict state dict",
             "seeds": "the locked protocol's (base 2026, per test index); the historical evaluation was unseeded"}
    run_cells(a, cells, outs, problems, extra, threads)


def cmd_plan(a):
    import ablation8_analysis as A
    problems, cells = [], []
    for label in A.CELLS:
        fin = a.final_dir / A.RUNS[label]
        final = json.loads((fin / "final.json").read_text())
        ckpt = fin / final["best_checkpoint"]
        sums = {p.split(None, 1)[1].strip(): p.split(None, 1)[0] for p in (fin / "SHA256SUMS").read_text().splitlines() if p}
        if A.sha256(ckpt) != sums.get(ckpt.name):
            problems.append(f"{ckpt}: SHA-256 differs from SHA256SUMS")
        cells.append((label, ckpt))
    outs = [a.eval_dir / f for f in ("ablation8-rtf-runs.jsonl", "ablation8-rtf.json", "ablation8-rtf.csv")]
    run_cells(a, cells, outs, problems)


def run_cells(a, cells, outs, problems, extra=None, threads=THREADS):
    """Check the outputs and the machine, print the schedule, and with --execute run it and write the outputs."""
    problems = list(problems) + [f"exists: {p}" for p in outs if p.exists()]
    state = machine_state()
    problems += idle_problems(state)
    cpusets = {t: cpuset(t) for t in threads}
    plan = schedule([label for label, _ in cells], threads=threads)
    print(f"machine: {json.dumps(state)}")
    print(f"{len(plan)} runs: {ROUNDS} rounds x {len(cells)} labels x threads {tuple(threads)}; cpusets {cpusets}; "
          f"{N_WARMUP} warm-up + {N_TEST} test utterances each")
    for r, label, t in plan[:len(cells) * len(threads)]:
        print(f"  round {r}: {label} T={t}")
    print("  ... (rounds 1 and 2 rotate the order)")
    if problems:
        print("REFUSED:\n  - " + "\n  - ".join(problems))
        raise SystemExit(1)
    if not a.execute:
        print("DRY RUN: nothing started. Add --execute to run.")
        return
    ckpt_of = dict(cells)
    runs = outs[0]
    start = {"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": state}
    for r, label, t in plan:
        env = dict(os.environ, OMP_NUM_THREADS=str(t), MKL_NUM_THREADS=str(t))
        load = os.getloadavg()[0]
        subprocess.run(["taskset", "-c", cpusets[t], sys.executable, __file__, "run-one", "--checkpoint", str(ckpt_of[label]),
                        "--label", label, "--threads", str(t), "--round", str(r), "--out", str(runs),
                        "--loadavg", str(load)], env=env, check=True)
    records = [json.loads(l) for l in runs.read_text().splitlines() if l]
    rows = summarize(records)
    import csv
    with open(outs[2], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    protocol = {"device": "cpu", "threads": tuple(threads), "cpusets": cpusets, "interop_threads": 1, "rounds": ROUNDS,
                "warmup": f"{N_WARMUP} validation utterances", "test": f"first {N_TEST} test utterances",
                "scales": [0.667, 1.0, 0.8], "seeds": "as the evaluation (base 2026, per test index)",
                "model_time": "eval_harness.synthesize window (model forward)",
                "e2e_time": "phonemization + tensors + model forward + numpy conversion",
                "placement_note": "WSL vCPUs are scheduled by Hyper-V; P-core/E-core placement is not controlled"}
    protocol.update(extra or {})
    outs[1].write_text(json.dumps({
        "protocol": protocol,
        "start": start, "end": {"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": machine_state()},
        "results": rows}, indent=1) + "\n")
    print(f"wrote {outs[0]}, {outs[1]}, {outs[2]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--final-dir", type=pathlib.Path, required=True)
    p.add_argument("--eval-dir", type=pathlib.Path, required=True)
    p.add_argument("--execute", action="store_true")
    h = sub.add_parser("historical")
    h.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    h.add_argument("--eval-dir", type=pathlib.Path, required=True)
    h.add_argument("--threads", type=int, nargs="+", default=list(THREADS),
                   help="thread settings (default 1 2); other settings name the outputs historical-pair-rtf-t<T>*")
    h.add_argument("--execute", action="store_true")
    o = sub.add_parser("run-one")
    o.add_argument("--checkpoint", type=pathlib.Path, required=True)
    o.add_argument("--label", required=True)
    o.add_argument("--threads", type=int, required=True)
    o.add_argument("--round", type=int, required=True)
    o.add_argument("--out", type=pathlib.Path, required=True)
    o.add_argument("--loadavg", type=float, default=None)
    a = ap.parse_args()
    {"plan": cmd_plan, "historical": cmd_historical, "run-one": cmd_run_one}[a.cmd](a)


if __name__ == "__main__":
    main()
