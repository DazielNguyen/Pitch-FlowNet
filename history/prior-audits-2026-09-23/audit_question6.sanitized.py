"""Read WAV headers and manifests to verify EDA outlier retention."""
import csv
import json
from pathlib import Path
import wave

root = Path('<WSL_DEV_HOME>/data')
rows = list(csv.reader((root/'metadata.csv').open(), delimiter='|', quoting=csv.QUOTE_NONE))
records = []
for row in rows:
    with wave.open(str(root/'wavs'/f'{row[0]}.wav')) as wav:
        duration = wav.getnframes()/wav.getframerate()
    records.append({'id': row[0], 'rate': len(row[1])/duration})
rates = sorted(r['rate'] for r in records)
def quantile(p):
    x = (len(rates)-1)*p
    i = int(x)
    return rates[i] + (rates[min(i+1,len(rates)-1)]-rates[i])*(x-i)
q1,q3 = quantile(.25),quantile(.75)
lower,upper = q1-1.5*(q3-q1),q3+1.5*(q3-q1)
outliers = [r for r in records if r['rate'] < lower or r['rate'] > upper]
entries = [json.loads(s) for s in Path('<WSL_DEV_HOME>/data_preprocessed/dataset.jsonl').read_text().splitlines() if s.strip()]
ids = {Path(e['audio_path']).stem for e in entries}
raw_ids = {r['id'] for r in records}
print(json.dumps({'raw_rows':len(rows), 'raw_unique_ids':len(raw_ids), 'manifest_rows':len(entries),
    'manifest_unique_ids':len(ids), 'missing_raw_ids':sorted(raw_ids-ids), 'extra_ids':sorted(ids-raw_ids),
    'q1':q1,'q3':q3,'lower':lower,'upper':upper,'outlier_count':len(outliers),
    'outliers_missing_from_manifest':[r['id'] for r in outliers if r['id'] not in ids],
    'max_phoneme_ids':max(len(e['phoneme_ids']) for e in entries),
    'outliers':outliers}, indent=2))
