"""Recompute the archived paired evaluation from existing Linux CSVs only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import scipy
from scipy.stats import wilcoxon

ROOT = Path('<WSL_DEV_HOME>/eval_h_vs_baseline')
output = Path(__file__).with_suffix('.json')
summary_path = ROOT/'summary.json'
summary = json.loads(summary_path.read_text())
result = {'method': 'Read existing CSV, JSON and manifest; no inference or training. Arithmetic means per utterance. SciPy default two-sided Wilcoxon signed-rank test, zero_method=wilcox, correction=False.', 'scipy_version': scipy.__version__, 'sources': {}, 'models': {}}
def record(path):
    data = path.read_bytes()
    result['sources'][str(path)] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
record(summary_path)
texts = {}
values = {}
for name, filename in [('baseline', 'best-epoch=773-val_loss_mel=20.2158_per_sentence.csv'), ('full', 'best-epoch=1079-val_loss_mel=19.2943_per_sentence.csv')]:
    path = ROOT/filename
    record(path)
    rows = list(csv.DictReader(path.open()))
    texts[name] = [row['text'] for row in rows]
    values[name] = {key: [float(row[key]) for row in rows] for key in ('wer', 'utmos', 'rtf')}
    saved = next(item for item in summary['checkpoints'] if Path(item['checkpoint']).stem == filename.replace('_per_sentence.csv', ''))
    result['models'][name] = {
        'rows': len(rows), 'columns': list(rows[0]), 'unique_texts': len(set(texts[name])),
        'all_metrics_finite': all(math.isfinite(value) for metric in values[name].values() for value in metric),
        'means': {key: statistics.mean(vals) for key, vals in values[name].items()},
        'medians': {key: statistics.median(vals) for key, vals in values[name].items()},
        'mean_rtf_from_rounded_summary_is_same_to_1e_12': abs(statistics.mean(values[name]['rtf'])-saved['mean_rtf']) < 1e-12,
        'all_saved_summary_means_match_to_1e_12': all(abs(statistics.mean(vals)-saved['mean_'+key]) < 1e-12 for key, vals in values[name].items()),
        'rtf_identity_max_abs_error': max(abs(float(row['rtf'])-float(row['infer_sec'])/float(row['audio_duration_sec'])) for row in rows),
        'checkpoint_in_saved_summary': saved['checkpoint'],
    }
result['pairing'] = {'same_text_order': texts['baseline'] == texts['full'], 'matching_unique_texts': len(set(texts['baseline']) & set(texts['full']))}
for manifest_path in [Path('<WSL_DEV_HOME>/test_entries_500.jsonl'), Path('<WSL_DEV_HOME>/test_sentences_500.txt')]:
    if not manifest_path.exists():
        continue
    record(manifest_path)
    manifest_texts = [json.loads(line)['text'] for line in manifest_path.read_text().splitlines() if line.strip()] if manifest_path.suffix == '.jsonl' else manifest_path.read_text().splitlines()
    result['pairing'][str(manifest_path)] = {'rows': len(manifest_texts), 'matches_baseline_order': texts['baseline'] == manifest_texts, 'matches_full_order': texts['full'] == manifest_texts}
assert result['pairing']['same_text_order']
result['paired_tests_recomputed'] = {}
for key in ['wer', 'utmos']:
    test = wilcoxon(values['full'][key], values['baseline'][key], alternative='two-sided', zero_method='wilcox', correction=False)
    result['paired_tests_recomputed'][key] = {
        'statistic': float(test.statistic), 'pvalue': float(test.pvalue),
        'saved_statistic': summary['paired_tests'][0][key+'_wilcoxon_stat'],
        'saved_pvalue': summary['paired_tests'][0][key+'_wilcoxon_p'],
        'same_saved_statistic': float(test.statistic)==summary['paired_tests'][0][key+'_wilcoxon_stat'],
        'pvalue_relative_difference': (float(test.pvalue)/summary['paired_tests'][0][key+'_wilcoxon_p'])-1,
        'zero_difference_pairs': sum(a==b for a,b in zip(values['full'][key],values['baseline'][key])),
    }
result['deltas_full_minus_baseline'] = {
    'wer_percentage_points': (statistics.mean(values['full']['wer'])-statistics.mean(values['baseline']['wer']))*100,
    'utmos': statistics.mean(values['full']['utmos'])-statistics.mean(values['baseline']['utmos']),
    'rtf_absolute': statistics.mean(values['full']['rtf'])-statistics.mean(values['baseline']['rtf']),
    'rtf_relative_percent': (statistics.mean(values['full']['rtf'])/statistics.mean(values['baseline']['rtf'])-1)*100,
}
result['limits'] = ['Same text order establishes the pairing of observed outputs; training/test leakage and split provenance require separate checks.', 'No hardware thread configuration, repeat-run timing protocol, waveform fidelity, ASR model version, or UTMOS model version is inferred from CSV alone.', 'Archived per-sentence values were not rescored from audio.', 'F0 data in summary.json remain null; any separate F0 report must be linked and checked separately.']
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
