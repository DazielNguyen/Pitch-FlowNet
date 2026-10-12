"""Inventory TensorBoard event files (read-only): scalar series summary, wall-time span, audio/text tags."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

out = {}
for f in sys.argv[1:-1]:
    p = Path(f)
    scal, audio_tags, walls = {}, {}, []
    for e in EventFileLoader(str(p)).Load():
        if e.wall_time:
            walls.append(e.wall_time)
        for v in e.summary.value:
            if v.HasField("simple_value"):
                x = v.simple_value
            elif v.HasField("tensor") and v.tensor.dtype == 1 and not v.tensor.tensor_shape.dim:
                x = float(np.frombuffer(v.tensor.tensor_content, np.float32)[0]) if v.tensor.tensor_content else float(v.tensor.float_val[0])
            elif v.metadata.plugin_data.plugin_name == "audio" or v.HasField("audio"):
                audio_tags[v.tag] = audio_tags.get(v.tag, 0) + 1
                continue
            else:
                continue
            scal.setdefault(v.tag, {})[e.step] = (e.wall_time, x)
    series = {}
    for tag, pts in scal.items():
        steps = sorted(pts)
        vals = [pts[s][1] for s in steps]
        series[tag] = {"points": len(steps), "first_step": steps[0], "last_step": steps[-1],
                       "nonfinite": sum(1 for x in vals if not math.isfinite(x))}
    val = scal.get("val_loss_mel", {})
    vsteps = sorted(val)
    vv = [val[s][1] for s in vsteps]
    best = int(np.argmin(vv)) if vv else None
    out[str(p)] = {"bytes": p.stat().st_size, "wall_first": min(walls) if walls else None, "wall_last": max(walls) if walls else None,
                   "hours_first_to_last": (max(walls) - min(walls)) / 3600 if walls else None,
                   "series": series,
                   "val_loss_mel": {"points": len(vv), "min": min(vv) if vv else None, "argmin_index": best,
                                    "argmin_step": vsteps[best] if vv else None, "values": vv, "steps": vsteps},
                   "audio_tags": dict(sorted(audio_tags.items())[:40]), "audio_tag_count": len(audio_tags)}
    print(p.name, {k: out[str(p)][k] for k in ("hours_first_to_last",)}, out[str(p)]["val_loss_mel"]["points"],
          out[str(p)]["val_loss_mel"]["min"], out[str(p)]["val_loss_mel"]["argmin_index"], len(audio_tags), flush=True)
json.dump(out, open(sys.argv[-1], "w"), indent=1)
