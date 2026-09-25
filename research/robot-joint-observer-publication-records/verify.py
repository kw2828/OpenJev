"""Verify original publication bytes and saved row counts without model calls."""
import csv
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
PUB = ROOT/'research/robot-joint-observer-results'
ENGINEERING = ROOT/'output/robot-joint-observer-publication-engineering-v1'

def descriptor(path):
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

manifest = json.loads((PUB/'manifest.json').read_text())
receipt = json.loads((PUB/'receipt.json').read_text())
audit = json.loads((PUB/'audit.json').read_text())
assert audit['status'] == 'PASS' and audit['agreement'] is True
assert receipt['status'] == 'PASS'
assert receipt['scientific_status'] == audit['results']['result']['status']
assert receipt['manifest'] == descriptor(PUB/'manifest.json')
assert set(manifest['files']) | {'manifest.json', 'receipt.json'} == {
    str(p.relative_to(PUB)) for p in PUB.rglob('*') if p.is_file()}
for name, pin in manifest['files'].items():
    assert descriptor(PUB/name) == pin, name
for name, item in manifest['copy_sources'].items():
    pin = {key: item[key] for key in ('sha256', 'bytes')}
    assert descriptor(Path(item['path'])) == descriptor(PUB/name) == pin, name
with (PUB/'scores.csv').open() as handle:
    scores = list(csv.DictReader(handle))
with (PUB/'costs.csv').open() as handle:
    costs = list(csv.DictReader(handle))
assert len(scores) == len(audit['results']['rows']) == 584
assert len(costs) == len(audit['resources']) == 61
assert sum(r['status'] == 'FAILED' for r in scores) == audit['counts']['failed_metric_rows']
assert [r['status'] for r in costs] == [r['status'] for r in audit['resources']]
documents = ['README.md', 'research/robot-joint-observer-results.md']
links = []
for name in documents:
    path = ROOT/name
    raw = path.read_text()
    assert '\u2014' not in raw, name
    for target in re.findall(r'\]\(([^\s)]+)\)', raw):
        if target.startswith(('https://', 'http://', '#')):
            continue
        destination = (path.parent/target.split('#', 1)[0]).resolve()
        assert destination.exists(), (name, target)
        links.append({'source': name, 'target': target})
images = {}
for url in re.findall(r'!\[[^\]]*\]\((https://raw\.githubusercontent\.com/kw2828/OpenJev/main/[^)]+)\)',
                      (ROOT/'README.md').read_text()):
    images[url] = descriptor(ROOT/url.split('/main/', 1)[1])
result = {'status': 'PASS', 'created_utc': datetime.now(UTC).isoformat(),
          'scope': 'Saved opaque byte/copy and CSV/document checks; no model, array, scoring or timing replay',
          'manifest': descriptor(PUB/'manifest.json'), 'publication_files': len(manifest['files']),
          'payload_bytes': sum(r['bytes'] for r in manifest['files'].values()),
          'byte_exact_copies': len(manifest['copy_sources']), 'score_rows': len(scores), 'cost_rows': len(costs),
          'scientific_status': receipt['scientific_status'],
          'charts_visually_reviewed': ['benchmark.png', 'benchmark-detail.png'],
          'excluded_measurement_files': 4, 'new_array_decodes': 0, 'new_model_calls': 0,
          'new_scoring_or_timing_calls': 0, 'readme_images': images, 'local_links': links,
          'documents': {name: descriptor(ROOT/name) for name in [*documents, 'research/experiment-index.md']}}
with (ENGINEERING/'verification.json').open('x') as handle:
    json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')
print(json.dumps({key: result[key] for key in ('status', 'publication_files', 'payload_bytes', 'byte_exact_copies', 'score_rows', 'cost_rows')}))
