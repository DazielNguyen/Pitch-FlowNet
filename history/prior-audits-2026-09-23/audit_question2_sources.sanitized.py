"""Read-only collection of evaluation provenance and CPU/thread references."""
import hashlib
import json
import pathlib
import re
import subprocess
import sys

with open(sys.argv[1], encoding='utf-16') as f:
    inventory = json.load(f)
pattern = re.compile(r'eval_h_vs_baseline|14700|OMP_NUM_THREADS|MKL_NUM_THREADS|set_num_threads|get_num_threads|set_num_interop_threads|cpu_model', re.I)
matches = []
for entry in inventory['files']:
    path = entry['path']
    if not path.startswith('<WSL_DEV_HOME>/') or '/.claude/' in path:
        continue
    if pathlib.Path(path).suffix not in {'.py', '.sh', '.log', '.md', '.txt'}:
        continue
    with open(path, encoding='utf-8', errors='replace') as f:
        for number, line in enumerate(f, 1):
            m = pattern.search(line)
            if m:
                start = max(0, m.start()-100)
                matches.append({'path': path, 'line': number, 'text': line[start:start+600].strip()})
sources = {}
for path in [
    '<WSL_DEV_HOME>/eval_h_vs_baseline/summary.json',
    '<WSL_DEV_HOME>/EdgeTTS/src/python/piper_train/eval_harness.py',
    '<WSL_DEV_HOME>/EdgeTTS/src/python/piper_train/vits/models.py',
    '<WSL_DEV_HOME>/EdgeTTS/src/python/piper_train/vits/modules.py',
]:
    content = pathlib.Path(path).read_bytes()
    sources[path] = {'sha256': hashlib.sha256(content).hexdigest(), 'text': content.decode('utf-8')}
print(json.dumps({'matches': matches, 'sources': sources,
                  'current_lscpu': subprocess.check_output(['lscpu'], text=True)}, indent=2))
