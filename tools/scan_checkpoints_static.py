"""Static scan of PitchFlowNet-candidate checkpoints (no unpickling): hash, size, epoch/step, name groups."""
import hashlib
import json
import pickletools
import re
import sys
import zipfile
from pathlib import Path

GROUPS = {
    "flow_attention": lambda n: n.startswith("model_g.flow.") and (".attn" in n or "attention" in n),
    "duration_discriminator": lambda n: n.startswith("model_d_dur."),
    "mrd": lambda n: n.startswith("model_d_mrd."),
    "f0_predictor": lambda n: n.startswith("model_g.f0_predictor") or ".f0_pred" in n,
    "f0_cond": lambda n: "f0_cond" in n,
    "snake_params_in_decoder": lambda n: n.startswith("model_g.dec.") and re.search(r"\.(alpha|beta)$", n) is not None,
    "decoder_resblocks": lambda n: n.startswith("model_g.dec.resblocks."),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def scan(p):
    z = zipfile.ZipFile(p)
    data = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
    ops = list(pickletools.genops(data))
    strings = [a for _, a, _ in ops if isinstance(a, str)]
    found = {}
    for i, (op, arg, _) in enumerate(ops):
        if arg in ("epoch", "global_step", "pytorch-lightning_version") and arg not in found:
            for op2, arg2, _ in ops[i + 1:i + 4]:
                if isinstance(arg2, (int, str)) and arg2 != arg and op2.name not in ("BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT"):
                    found[arg] = arg2
                    break
    names = sorted({s for s in strings if re.match(r"model_[a-z_]+\.", s)})
    out = {"zip_members": len(z.namelist()), **found,
           "top_level_keys": [k for k in ("state_dict", "optimizer_states", "lr_schedulers", "hyper_parameters", "callbacks", "loops") if k in strings],
           "state_name_count": len(names), "prefixes": sorted({n.split(".")[0] for n in names})}
    for g, fn in GROUPS.items():
        sel = [n for n in names if fn(n)]
        out[g] = {"count": len(sel), "examples": sel[:4]}
    return out


paths = [Path(a) for a in sys.argv[1:-1]]
res = []
for p in paths:
    res.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p), "static_scan": scan(p)})
    print(p.name, res[-1]["static_scan"].get("epoch"), res[-1]["static_scan"].get("global_step"),
          {g: res[-1]["static_scan"][g]["count"] for g in GROUPS}, flush=True)
json.dump(res, open(sys.argv[-1], "w"), indent=1)
