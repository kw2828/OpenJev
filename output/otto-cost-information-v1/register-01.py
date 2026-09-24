"""Freeze the conditional-cost diagnostic after bounded fabricated qualification."""
import importlib.metadata
import json
import sys

sys.path.insert(0, 'scripts')
import otto_cost_information_common as c

engineering_dir = c.OUT / sys.argv[1]
engineering = c.read(engineering_dir / 'receipt.json')
native = c.read(engineering_dir / 'native-preflight.json')
seeds = c.read(c.OUT / 'seed-review-01.json')
c.require(native['status'] == 'passed' and not any(native['counts'].values())
          and native['metadata_import_guard'], 'native metadata preflight')
c.require(engineering['status'] == 'passed' and engineering['source_before'] == engineering['source_after'],
          'stable qualified sources')
c.require(all(engineering['source_after'][k] == c.desc(k) for k in c.SOURCES), 'current sources qualified')
c.require(all(r['returncode'] == 0 and r['reaped'] and r['group_absent'] and not r['timed_out']
              for r in engineering['commands']), 'closed fabricated qualification')
c.require(c.read(engineering_dir / 'capacity.json')['status'] == 'passed', 'prospective capacity')
c.require(seeds['passed'] and seeds['exact_seed_hit_count'] == 0 and seeds['proposed_unique_seeds'] == 193
          and not seeds['block_hits'] and not seeds['changed_files_during_scan'], 'scoped seed reservation')
c.require(seeds['ranges_inclusive'] == {
    'dev_lambda3': [331000001, 331000032], 'dev_lambda4': [332000001, 332000032],
    'selection': [333000001, 333000064], 'evaluation': [334000001, 334000064],
    'bootstrap': [335000001, 335000001]}, 'exact diagnostic seed ranges')
reference = c.parent_reference()
c.require(reference == native['parent_reference'], 'same qualified frozen checkpoints')
native_sources = {}
for name, expected in native['native_sources'].items():
    actual = c.desc(name)
    c.require(actual['sha256'] == expected, 'unchanged original source')
    native_sources[name] = actual
inputs = {}
for record in [*native['native_inputs'].values(), *native['installed_sources'].values()]:
    descriptor = {k: record[k] for k in ('sha256', 'bytes')}
    c.require(c.desc(record['path']) == descriptor, 'unchanged original input')
    inputs[record['path']] = descriptor
for name in ['seed-review-01.json', 'source-review-01.json', 'register-01.py']:
    path = c.OUT / name
    inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
for name in ['closure-01.json', 'fit-native-01.terminal.json', 'fit-native-01.launch.json',
             'fit-01/receipt.json', 'fit-01/summary.json']:
    path = c.PARENT / name
    inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
for record in reference['checkpoints'].values():
    inputs[record['path']] = {k: record[k] for k in ('sha256', 'bytes')}
for name in ['research/otto-action-effect-results.md', 'research/otto-action-effect-results/summary.json']:
    inputs[name] = c.desc(name)
directories = sorted(c.OUT.glob('engineering-*')) + [c.ROOT / 'output/otto-conditional-cost-component-v1/attempt-01']
for directory in directories:
    for path in sorted(directory.iterdir()):
        if path.is_file():
            inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
sources = {**native_sources, **engineering['source_after']}
plan = {'version': c.VERSION, 'status': 'frozen_before_collection', 'config': c.CONFIG, 'roster': c.roster(),
        'sources': sources, 'inputs': inputs, 'native_sources': native_sources, 'runtime': native['native_runtime'],
        'numerical_runtime': {'python': sys.version, 'executable': sys.executable,
                              'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'pytest')}},
        'bounds': c.CAPS, 'parent_reference': reference, 'old_test_admitted': False,
        'parent_training_or_development_array_decodes': 0, 'repeat_scientific_attempts': False,
        'new_training': False,
        'scope': 'One fresh conditional-cost diagnostic with twelve frozen policies; no model or architecture improvement claim.'}
c.check_plan(plan)
c.write(c.OUT / 'registration-01.json', plan)
print(json.dumps(c.desc(c.OUT / 'registration-01.json')))
