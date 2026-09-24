"""Freeze one paired action-effect study after fabricated qualification."""
import importlib.metadata
import json
import sys
sys.path.insert(0, 'scripts')
import otto_action_effect_common as c
engineering_dir = c.OUT / 'engineering-02'
engineering = c.read(engineering_dir / 'receipt.json')
native = c.read(engineering_dir / 'native-preflight.json')
seeds = c.read(c.OUT / 'seed-review-01.json')
c.require(native['status'] == 'passed' and not any(native['counts'].values()) and native['metadata_import_guard'], 'native metadata preflight')
c.require(engineering['status'] == 'passed' and engineering['source_before'] == engineering['source_after'], 'stable qualified sources')
c.require(all(engineering['source_after'][k] == c.desc(k) for k in c.SOURCES), 'all current sources qualified')
c.require(all(r['returncode'] == 0 and r['reaped'] and r['group_absent'] and not r['timed_out'] for r in engineering['commands']), 'closed fabricated qualification')
c.require(c.read(engineering_dir / 'capacity.json')['status'] == 'passed', 'prospective capacity')
c.require(seeds['passed'] and seeds['exact_seed_hit_count'] == 0 and seeds['proposed_unique_seeds'] == 259 and seeds['ranges_inclusive'] == {'dev_lambda3':[328000001,328000128], 'dev_lambda4':[329000001,329000128], 'fit':[330000001,330000003]}, 'exact seed reservation')
reference = c.training_reference()
c.require(reference == native['training_reference'], 'same qualified parent TRAIN')
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
for name in ['seed-review-01.json','source-review-01.json','register-01.py']:
    path = c.OUT / name
    inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
for name in ['closure-01.json','collection-native-01.terminal.json','collection-native-01.launch.json',
             'collection-01/receipt.json','collection-01/summary.json','collection-01/train.npz']:
    path = c.PARENT / name
    inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
for name in ['research/otto-belief-distillation-results.md','research/otto-belief-distillation-results/summary.json']:
    inputs[name] = c.desc(name)
for directory in sorted(c.OUT.glob('engineering-*')) + sorted(c.OUT.glob('component-*')):
    for path in sorted(directory.iterdir()):
        if path.is_file():
            inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
sources = {**native_sources, **engineering['source_after']}
plan = {'version':c.VERSION, 'status':'frozen_before_collection', 'config':c.CONFIG, 'roster':c.roster(),
        'sources':sources, 'inputs':inputs, 'native_sources':native_sources, 'runtime':native['native_runtime'],
        'numerical_runtime':{'python':sys.version,'executable':sys.executable,'packages':{name:importlib.metadata.version(name) for name in ('numpy','torch','pytest')}},
        'bounds':c.CAPS, 'training_reference':reference, 'old_test_admitted':False,
        'old_dev_array_decodes':0, 'repeat_scientific_attempts':False,
        'scope':'one prospective paired action-effect development study, no architecture or autonomous control claim'}
c.check_plan(plan)
c.write(c.OUT/'registration-01.json', plan)
print(json.dumps(c.desc(c.OUT/'registration-01.json')))
