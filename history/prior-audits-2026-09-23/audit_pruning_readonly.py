"""Read-only text audit; does not load checkpoints or execute project code."""
import collections
import json
import os
import re
import sys
from pathlib import Path

skip = {'.git', '.cache', '__pycache__', '.pytest_cache', 'miniconda3', 'env',
        '.venv', 'venv', 'node_modules', '.deps', 'deps', 'site-packages',
        'data', 'data_preprocessed', 'cache', 'artifacts', '.research-work',
        'AI-Research-SKILLs', 'Research-Paper-Writing-Skills'}
extensions = {'.py', '.sh', '.md', '.txt', '.log', '.yaml', '.yml', '.json',
              '.toml', '.cfg', '.ini', '.ipynb', '.out', '.tex'}
pattern = re.compile(r'\bprun\w*|sparsity|weight_mask|weight_orig|torch_pruning|l1_unstructured|ln_structured', re.I)
result = {'roots': sys.argv[1:], 'files': [], 'matches': [], 'errors': [], 'checkpoints': []}
for root in sys.argv[1:]:
    for folder, dirs, files in os.walk(root, onerror=lambda e: result['errors'].append(str(e))):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            path = Path(folder) / name
            if path.suffix == '.ckpt':
                result['checkpoints'].append(str(path))
            if path.suffix not in extensions and not re.search(r'log\d*$|history$', name):
                continue
            try:
                result['files'].append({'path': str(path), 'bytes': path.stat().st_size})
                with path.open(encoding='utf-8', errors='replace') as handle:
                    for number, line in enumerate(handle, 1):
                        match = pattern.search(line)
                        if match:
                            start = max(0, match.start() - 100)
                            result['matches'].append({'path': str(path), 'line': number,
                                'text': line[start:start+450].strip()})
            except OSError as error:
                result['errors'].append(str(error))
result['counts_by_extension'] = dict(collections.Counter(Path(f['path']).suffix for f in result['files']))
print(json.dumps(result, ensure_ascii=True, indent=2))
