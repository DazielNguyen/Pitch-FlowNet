"""Load checkpoints with torch.load(weights_only=True) only.

The historical checkpoints pickle one non-tensor type outside torch's default allowlist:
pathlib.PosixPath (the dataset_dir hparam). Their data.pkl references only collections.OrderedDict,
torch._utils._rebuild_tensor_v2, torch.FloatStorage, torch.IntStorage and pathlib.PosixPath
(static pickletools scan, 2026-10-19). Only PosixPath is added, and only inside this context.
"""
import pathlib
from typing import Any, Dict, Union

import torch

SAFE_GLOBALS = [pathlib.PosixPath]


def load_checkpoint(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    with torch.serialization.safe_globals(SAFE_GLOBALS):
        return torch.load(path, map_location="cpu", weights_only=True)
