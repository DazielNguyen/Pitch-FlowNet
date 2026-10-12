"""Read existing GT evaluation outputs only; no model inference."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path('<WSL_DEV_HOME>')
entries = [json.loads(s) for s in (root/'test_entries_500.jsonl').read_text().splitlines() if s.strip()]
texts = [e['text'] for e in entries]
result = {'entries': len(entries), 'gt_audio_fields': entries[0], 'comparisons': {}, 'sources': {}, 'wer_hits': []}
paths = [root/'eval_h_vs_baseline/mcd_baseline/mcd_gtmos.csv', root/'eval_h_vs_baseline/mcd_configH/mcd_gtmos.csv',
         root/'eval_h_vs_baseline/best-epoch=773-val_loss_mel=20.2158_per_sentence.csv',
         root/'eval_h_vs_baseline/best-epoch=1079-val_loss_mel=19.2943_per_sentence.csv']
gt_rows = []
for path in paths:
    rows = list(csv.DictReader(path.open()))
    result['comparisons'][str(path)] = {'rows': len(rows), 'text_order_matches_manifest': [r['text'] for r in rows] == texts}
    if 'gt_utmos' in rows[0]:
        values = [float(r['gt_utmos']) for r in rows]
        result['comparisons'][str(path)]['mean_rounded_csv_gt_utmos'] = sum(values)/len(values)
        gt_rows.append(values)
result['gt_scores_different_rows'] = sum(a != b for a,b in zip(*gt_rows))
result['test_sentences_match_manifest'] = (root/'test_sentences_500.txt').read_text().splitlines() == texts
for path in [root/'eval_h_vs_baseline/mcd_baseline/mcd_gtmos_summary.json',
             root/'eval_h_vs_baseline/mcd_configH/mcd_gtmos_summary.json',
             root/'eval_results/baseline_full/mcd_gtmos_summary.json',
             root/'EdgeTTS/src/python/piper_train/compute_reference_metrics.py',
             root/'EdgeTTS/src/python/piper_train/gt_wer.py']:
    data = path.read_bytes()
    result['sources'][str(path)] = {'sha256': hashlib.sha256(data).hexdigest(), 'text': data.decode()}
pattern = re.compile(r'GT mean WER|GT median WER|GT audio WER|gt_wer|ground.truth.{0,60}wer|wer.{0,60}ground.truth', re.I)
with open(sys.argv[1], encoding='utf-16') as f:
    inventory = json.load(f)
for item in inventory['files']:
    path = Path(item['path'])
    if '/.claude/' in str(path) or path.suffix not in {'.py', '.sh', '.log', '.txt', '.md', '.json'}:
        continue
    with path.open(errors='replace') as f:
        for n,line in enumerate(f,1):
            m=pattern.search(line)
            if m:
                result['wer_hits'].append({'path': str(path), 'line': n, 'text':line[max(0,m.start()-60):m.end()+250].strip()})
print(json.dumps(result, indent=2))
