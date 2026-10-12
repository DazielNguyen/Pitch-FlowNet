"""Read checkpoint tensor shapes only, using weights_only and FakeTensorMode."""
import collections
import json
import pathlib
import sys
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

results = []
for path in sys.argv[1:]:
    with torch.serialization.safe_globals([pathlib.PosixPath]), FakeTensorMode():
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    state = ckpt['state_dict']
    groups = collections.Counter()
    tensors = []
    for name, value in state.items():
        if not name.startswith('model_g.'):
            continue
        groups[name.split('.')[1]] += value.numel()
        tensors.append({'name': name, 'shape': list(value.shape), 'numel': value.numel()})
    results.append({'checkpoint': path, 'torch_version': torch.__version__,
                    'generator_state_elements_by_module': dict(groups), 'tensors': tensors})
print(json.dumps(results, indent=2))
