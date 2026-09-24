"""Freeze one fresh belief-distillation pilot after fabricated qualification."""
import importlib.metadata
import json
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import otto_belief_distillation_common as c
engineering_dir = c.OUT / 'engineering-02'
engineering = c.read(engineering_dir / 'receipt.json')
native = c.read(engineering_dir / 'native-preflight.json')
seeds = c.read(c.OUT / 'seed-review-02.json')
c.require(native['status'] == 'passed' and not any(native['counts'].values()) and native['metadata_import_guard'], 'native metadata preflight')
c.require(engineering['status'] == 'passed' and engineering['source_before'] == engineering['source_after'], 'stable qualified sources')
c.require(all(engineering['source_after'][k] == c.desc(k) for k in c.SOURCES), 'all current sources qualified')
c.require(all(r['returncode'] == 0 and r['reaped'] and r['group_absent'] and not r['timed_out'] for r in engineering['commands']), 'closed fabricated qualification')
c.require(c.read(engineering_dir / 'capacity.json')['status'] == 'passed', 'prospective capacity')
c.require(seeds['passed'] and seeds['exact_seed_hit_count'] == 0 and seeds['proposed_unique_seeds'] == 1795 and seeds['ranges_inclusive'] == {'train_lambda3':[324000001,324001536], 'dev_lambda3':[325000001,325000128], 'dev_lambda4':[326000001,326000128], 'fit':[327000001,327000003]}, 'exact seed reservation')
native_sources = {}
for name, expected in native['native_sources'].items():
    actual = c.desc(name)
    c.require(actual['sha256'] == expected, 'unchanged original source')
    native_sources[name] = actual
inputs = {}
for record in native['native_inputs'].values():
    descriptor = {k:record[k] for k in ('sha256','bytes')}
    c.require(c.desc(record['path']) == descriptor, 'unchanged original input')
    inputs[record['path']] = descriptor
for record in native['installed_sources'].values():
    descriptor = {k:record[k] for k in ('sha256','bytes')}
    c.require(c.desc(record['path']) == descriptor, 'native installed conversion source')
    inputs[record['path']] = descriptor
for name in ['seed-review-01.json','seed-review-02.json','source-review-01.json','register-02.py','register-attempt-01.py','registration-attempt-01.json']:
    path = c.OUT / name
    inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
for directory in sorted(c.OUT.glob('engineering-*')) + sorted(c.OUT.glob('component-*')):
    for path in sorted(directory.iterdir()):
        if path.is_file():
            inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
sources = {**native_sources, **engineering['source_after']}
plan = {'version':c.VERSION, 'status':'frozen_before_collection', 'config':c.CONFIG, 'roster':c.roster(),
        'sources':sources, 'inputs':inputs, 'native_sources':native_sources, 'runtime':native['native_runtime'],
        'numerical_runtime':{'python':sys.version,'executable':sys.executable,'packages':{name:importlib.metadata.version(name) for name in ('numpy','torch','pytest')}},
        'bounds':c.CAPS, 'old_test_admitted':False, 'repeat_scientific_attempts':False,
        'scope':'one prospective full-belief distillation development pilot, no architecture or autonomous control claim'}
c.check_plan(plan)
c.write(c.OUT/'registration-01.json', plan)
print(json.dumps(c.desc(c.OUT/'registration-01.json')))
