"""Freeze a fresh conditional-label pilot only after fabricated qualification."""
import importlib.metadata
import json
import sys
sys.path.insert(0, 'scripts')
import otto_conditional_label_common as c

engineering_dir = c.OUT / sys.argv[1]
engineering = c.read(engineering_dir / 'receipt.json')
for name, expected in engineering['files'].items():
    c.require(c.desc(engineering_dir / name) == expected, 'closed qualification payload: ' + name)
native = c.read(engineering_dir / 'native-preflight.json')
seeds = c.read(c.OUT / 'seed-review-02.json')
c.require(native['status'] == 'passed' and not any(native['counts'].values()) and native['metadata_import_guard'],
          'native metadata preflight')
c.require(engineering['status'] == 'passed' and engineering['source_before'] == engineering['source_after'],
          'stable qualified source closure')
c.require(all(engineering['source_after'][k] == c.desc(k) for k in c.SOURCES), 'current sources qualified')
c.require(len(engineering['commands']) == 4 and all(r['returncode'] == 0 and r['reaped'] and r['group_absent'] and not r['timed_out']
              for r in engineering['commands']), 'all qualification processes closed')
c.require(c.read(engineering_dir / 'capacity.json')['status'] == 'passed', 'prospective resource capacity')
c.require(seeds['passed'] and seeds['exact_seed_hit_count'] == 0 and seeds['proposed_unique_seeds'] == 1540
          and not seeds['block_hits'] and not seeds['changed_files_during_scan'], 'fresh scoped seed reservation')
c.require(seeds['ranges_inclusive'] == {
    'train': [336000001,336000512], 'dev_lambda3': [337000001,337000128],
    'dev_lambda4': [338000001,338000128], 'train_mc': [339000001,339000512],
    'dev_mc': [340000001,340000256], 'fits': [343000001,343000003],
    'bootstrap': [344000001,344000001]}, 'exact future seed ranges')
reference = c.parent_reference()
c.require(reference == native['parent_reference'], 'same audited headroom evidence')
native_sources = {}
for name, expected in native['native_sources'].items():
    actual = c.desc(name)
    c.require(actual['sha256'] == expected, 'unchanged original native source')
    native_sources[name] = actual
inputs = {}
for record in [*native['native_inputs'].values(), *native['installed_sources'].values()]:
    descriptor = {k: record[k] for k in ('sha256', 'bytes')}
    c.require(c.desc(record['path']) == descriptor, 'unchanged original native input')
    inputs[record['path']] = descriptor
for path in sorted(c.OUT.iterdir()):
    if path.is_file() and path.suffix in ('.py', '.json'):
        inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
for name in ['closure-01.json', 'collection-01/receipt.json']:
    path = c.PARENT / name
    inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
for name in ['research/otto-cost-information-results.md', 'research/otto-cost-information-results/summary.json',
             'research/otto-cost-information-next.md']:
    inputs[name] = c.desc(name)
for directory in sorted(c.OUT.glob('engineering-*')):
    for path in sorted(directory.iterdir()):
        if path.is_file():
            inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
plan = {'version': c.VERSION, 'status': 'frozen_before_collection', 'config': c.CONFIG, 'roster': c.roster(),
        'sources': {**native_sources, **engineering['source_after']}, 'inputs': inputs,
        'native_sources': native_sources, 'runtime': native['native_runtime'],
        'numerical_runtime': {'python': sys.version, 'executable': sys.executable,
            'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'pytest')}},
        'bounds': c.CAPS, 'parent_reference': reference, 'old_test_admitted': False,
        'parent_training_or_development_array_decodes': 0, 'parent_checkpoint_decodes': 0,
        'repeat_scientific_attempts': False, 'new_training': True,
        'scope': 'Six fresh fixed-H8 cost-only GRU fits. Shared data bank and updates; conditional-target gradient noise intervention, no architecture claim.'}
c.check_plan(plan)
c.write(c.OUT / 'registration-01.json', plan)
print(json.dumps(c.desc(c.OUT / 'registration-01.json')))
