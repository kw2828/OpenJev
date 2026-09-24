import importlib.metadata
import json
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import otto_action_latent_common as c
native = c.read(c.OUT / 'native-preflight-01.json')
engineering = c.read(c.OUT / 'engineering-02/receipt.json')
seeds = c.read(c.OUT / 'seed-review-01.json')
c.require(native['status'] == 'passed' and native['native_calls'] == 0 and native['numerical_imports'] == [], 'native metadata preflight')
c.require(engineering['status'] == 'passed' and engineering['source_before'] == engineering['source_after'] == {k:c.desc(k) for k in c.SOURCES}, 'all current sources qualified')
c.require(all(r['returncode'] == 0 and r['reaped'] and r['group_absent'] and not r['timed_out'] for r in engineering['commands']), 'closed fabricated qualification')
c.require(c.read(c.OUT / 'engineering-02/capacity.json')['status'] == 'passed', 'prospective capacity')
c.require(seeds['passed'] and seeds['exact_seed_hit_count'] == 0 and seeds['proposed_unique_seeds'] == 291 and seeds['ranges_inclusive'] == {'train_lambda3':[320000001,320000192], 'dev_lambda3':[321000001,321000048], 'dev_lambda4':[322000001,322000048], 'fit':[323000001,323000003]}, 'exact seed reservation')
for name, descriptor in {**native['sources'], **native['native_inputs']}.items():
    c.require(c.desc(name) == descriptor, 'unchanged original source/input')
inputs = dict(native['native_inputs'])
for name in ['seed-review-01.json','native-preflight-01.json','register-01.py','engineering-01/receipt.json','engineering-02/receipt.json']:
    path = c.OUT / name
    inputs[str(path.relative_to(c.ROOT))] = c.desc(path)
for attempt in ('engineering-01','engineering-02'):
    receipt = c.read(c.OUT/attempt/'receipt.json')
    for name, descriptor in receipt['files'].items():
        path = c.OUT/attempt/name
        c.require(c.desc(path) == descriptor, 'qualification payload')
        inputs[str(path.relative_to(c.ROOT))] = descriptor
sources = {**native['sources'], **engineering['source_after']}
plan = {'version': c.VERSION,'status':'frozen_before_collection','config':c.CONFIG,'roster':c.roster(),
        'sources':sources,'inputs':inputs,'native_sources':native['sources'],'runtime':native['runtime'],
        'numerical_runtime':{'python':sys.version,'executable':sys.executable,'packages':{name:importlib.metadata.version(name) for name in ('numpy','torch','pytest')}},
        'bounds':c.CAPS,'old_test_admitted':False,'repeat_scientific_attempts':False,
        'scope':'single prospective development pilot, no architecture or autonomous control claim'}
c.check_plan(plan)
c.write(c.OUT/'registration-01.json', plan)
print(json.dumps(c.desc(c.OUT/'registration-01.json')))
