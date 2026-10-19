"""Check that a camera-ready training checkpoint is complete and safe to resume from (read-only).

    python tools/camera_ready/check_ckpt.py <file.ckpt> [<file.ckpt> ...] [--json]

For each file:
  1. the zip container is intact (every member's CRC is checked);
  2. it loads with torch.load(weights_only=True) and no allowlist;
  3. it carries the camera_ready counters (g_updates, d_updates, samples_seen_global) and the
     Lightning state (epoch, global_step, optimizers, schedulers, loops);
  4. every floating-point tensor of the weights and optimizer state is finite.

Exit status 0 when every file passes, 1 otherwise.
"""
import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import torch


def check(path: Path) -> dict:
    out = {"file": str(path), "ok": False, "bytes": path.stat().st_size if path.is_file() else None, "problems": []}
    if not path.is_file():
        out["problems"].append("missing")
        return out
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
        if bad is not None:
            out["problems"].append(f"zip CRC error in member {bad}")
            return out
    except zipfile.BadZipFile as exc:
        out["problems"].append(f"not a valid zip container: {exc}")
        return out
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # reported, not hidden
        out["problems"].append(f"weights_only load failed: {exc!r}")
        return out
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "epoch", "global_step", "loops", "camera_ready"):
        if key not in ck:
            out["problems"].append(f"missing key {key}")
    cr = ck.get("camera_ready", {})
    for key in ("g_updates", "d_updates", "samples_seen_global"):
        if key not in cr:
            out["problems"].append(f"missing camera_ready.{key}")
    nonfinite = [k for k, v in ck.get("state_dict", {}).items()
                 if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
    for i, opt in enumerate(ck.get("optimizer_states", [])):
        for p, state in opt.get("state", {}).items():
            for name, v in state.items():
                if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
                    nonfinite.append(f"optimizer{i}.{p}.{name}")
    if nonfinite:
        out["problems"].append(f"{len(nonfinite)} tensors not finite, e.g. {nonfinite[:3]}")
    lrs = [g.get("lr") for opt in ck.get("optimizer_states", []) for g in opt.get("param_groups", [])]
    out.update({"epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
                "g_updates": cr.get("g_updates"), "d_updates": cr.get("d_updates"),
                "samples_seen_global": cr.get("samples_seen_global"),
                "lr": [float(x) for x in lrs if x is not None and math.isfinite(float(x))],
                "tensors": len(ck.get("state_dict", {}))})
    out["ok"] = not out["problems"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="print one JSON object per file")
    a = ap.parse_args(argv)
    ok = True
    for f in a.files:
        r = check(f)
        ok &= r["ok"]
        if a.json:
            print(json.dumps(r))
        else:
            status = "OK " if r["ok"] else "BAD"
            detail = (f"epoch {r.get('epoch')}, g_updates {r.get('g_updates')}, global_step {r.get('global_step')}, "
                      f"{r.get('bytes')} bytes") if r["ok"] else "; ".join(r["problems"])
            print(f"{status} {f}: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
