"""Inspect pickle string literals without unpickling or loading tensor data."""
import json
import pickletools
import re
import sys
import zipfile

with open(sys.argv[1], encoding='utf-16') as handle:
    audit = json.load(handle)
results = []
for path in audit['checkpoints']:
    entry = {'path': path}
    try:
        with zipfile.ZipFile(path) as archive:
            member = next(n for n in archive.namelist() if n.endswith('/data.pkl') or n == 'data.pkl')
            literals = [arg for op, arg, pos in pickletools.genops(archive.read(member)) if isinstance(arg, str)]
            entry['pruning_literals'] = [s for s in literals if re.search(r'prun|weight_mask|weight_orig|sparsity', s, re.I)]
            entry['decoder_resblock_indices'] = sorted(set(int(m.group(1)) for s in literals
                for m in [re.search(r'(?:dec|decoder)\.resblocks\.(\d+)\.', s)] if m))
    except Exception as error:
        entry['error'] = str(error)
    results.append(entry)
print(json.dumps(results, indent=2))
