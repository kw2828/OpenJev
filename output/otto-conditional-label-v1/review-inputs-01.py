"""Source and seed metadata review; no numerical imports or empirical decoding."""
import hashlib
import json
import re
import sys
import time
from pathlib import Path
sys.path.insert(0, 'scripts')
import otto_conditional_label_common as c

def descriptor(p):
    data = p.read_bytes()
    return data, {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

previous = c.read('output/otto-cost-information-v1/seed-review-01.json')
paths = {c.ROOT / row['path'] for row in previous['files']}
for pattern in ('output/otto*/registration*.json', 'output/otto*/plan*.json',
                'output/otto*/seed-review*.json', 'output/otto*/collection-*/cases.jsonl',
                'scripts/*otto*.py', 'src/openjev/research/otto*.py', 'research/otto*.md'):
    paths.update(c.ROOT.glob(pattern))
excluded = sorted(str(p.relative_to(c.ROOT)) for p in paths
                  if 'conditional-label' in str(p) or 'conditional_label' in str(p))
paths = sorted(p for p in paths if str(p.relative_to(c.ROOT)) not in excluded)
seeds = {r[k] for r in c.roster() for k in ('seed', 'mc_seed')}
seeds.update(c.FIT_SEEDS)
seeds.add(c.CONFIG['bootstrap_seed'])
blocks = sorted({s // 1000000 for s in seeds})
pattern = re.compile(rb'(?<![0-9])(?:' + b'|'.join(str(x).encode() for x in blocks) + rb')[0-9]{6}(?![0-9])')
rows, hits, exact, changed = [], [], [], []
for path in paths:
    data, desc = descriptor(path)
    numbers = sorted({int(m.group(0)) for m in pattern.finditer(data)})
    exact_numbers = sorted(set(numbers) & seeds)
    row = {'path': str(path.relative_to(c.ROOT)), **desc,
           'block_matches': numbers, 'exact_seed_matches': exact_numbers}
    rows.append(row)
    if numbers:
        hits.append(row['path'])
    if exact_numbers:
        exact.append(row['path'])
for row in rows:
    if c.desc(row['path']) != {k: row[k] for k in ('sha256', 'bytes')}:
        changed.append(row['path'])
review = {'version': 'otto-conditional-label-seed-review-v1', 'created_unix_ns': time.time_ns(),
          'passed': not hits and not changed, 'proposed_unique_seeds': len(seeds),
          'exact_seed_hit_count': len(exact), 'block_hits': hits,
          'changed_files_during_scan': changed, 'excluded_current_study_paths': excluded,
          'seed_blocks': blocks, 'files': rows,
          'counts': {'files': len(rows), 'bytes': sum(r['bytes'] for r in rows)},
          'ranges_inclusive': {'train': [336000001,336000512], 'dev_lambda3': [337000001,337000128],
              'dev_lambda4': [338000001,338000128], 'train_mc': [339000001,339000512],
              'dev_mc': [340000001,340000256], 'fits': [343000001,343000003],
              'bootstrap': [344000001,344000001]},
          'scope': 'Prior diagnostic historical metadata roster plus current OTTO plans, seeds, cases, source and research; not global uniqueness.',
          'empirical_array_decodes': 0, 'checkpoint_decodes': 0, 'admits_execution': False}
c.write(c.OUT / 'seed-review-01.json', review)
oldpath = c.ROOT / 'output/otto-cost-information-v1/source-review-01.json'
old = c.read(oldpath)
for row in old['files'].values():
    c.require(c.desc(row['path']) == {k:row[k] for k in ('sha256', 'bytes')}, 'same installed source bytes')
source = {**old, 'version': 'otto-conditional-label-source-review-v1',
          'created_unix_ns': time.time_ns(),
          'inherited_source_review': {'path': str(oldpath.relative_to(c.ROOT)), **c.desc(oldpath)},
          'review_findings': old['review_findings'] + [
              'Endpoint-only Monte Carlo retains the same root/sensor grid and saved uint64 draws. No exact-H4 target is used.',
              'TRAIN and DEV seeds are prospective and separately allocated; label averaging changes optimizer noise only.']}
c.write(c.OUT / 'source-review-01.json', source)
print(json.dumps({'seed_review_passed':review['passed'], 'seed_count':len(seeds),
                  'scanned_files':len(rows), 'block_hits':hits,
                  'source':c.desc(c.OUT / 'source-review-01.json'),
                  'seeds':c.desc(c.OUT / 'seed-review-01.json')}))
c.require(review['passed'], 'fresh scoped seed blocks')
