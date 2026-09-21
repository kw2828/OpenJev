"""Freeze source identities for one saved-only diagnostic; no numerical decoding."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/otto-return-consistency-v2'


def identity(name):
    path = ROOT / name
    data = path.read_bytes()
    return {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


sources = [
    'scripts/diagnose_otto_return_consistency.py',
    'tests/test_otto_return_consistency.py',
    'research/otto-return-consistency-design.md',
    'scripts/render_otto_return_value.py',
    'src/openjev/research/suspend_clock.py',
    'scripts/diagnose_otto_return_consistency_v2.py',
    'tests/test_otto_return_consistency_v2.py',
    'research/otto-return-consistency-repair.md',
]
roles = {
    'study_plan': ('output/otto-return-value-v1/plan-01.json', '92df71d5e20ca48d8485bfa0a5f50bdf0e6e7a276c8e0c097bccd8e1c095aee2'),
    'worker_receipt': ('output/otto-return-value-v1/run-01/receipt.json', '965cc3eee7c77ea64700137bda2961af1a259f1a98d15b54041f9530a7f66ac3'),
    'terminal': ('output/otto-return-value-v1/run-process-01.terminal.json', 'bbf0bdb8361134ad4d7ca87f690969a4c91a7d727754bf3adac92609e452b653'),
    'audit_receipt': ('output/otto-return-value-v1/audit-01/receipt.json', '5900092b04a8707e446471544c01d3fa0fe4d27c37ca67212a13fa57cb883666'),
    'prior_receipt': ('output/otto-symmetry-head-v1/run-01/receipt.json', 'afd3ed59c6fc907721ebebcff47c1342fc53e592ec14b190497c647cb8d26c62'),
}
inputs = {role: identity(name) for role, (name, _) in roles.items()}
for role, (_, pin) in roles.items():
    assert inputs[role]['sha256'] == pin, role

plan = {
    'version': 'otto-return-consistency-v2',
    'status': 'frozen_before_saved_read',
    'output': 'output/otto-return-consistency-v2/run-01',
    'limits': {'seconds': 180, 'rss_bytes': 2 * 1024**3, 'output_bytes': 64 * 1024**2},
    'configuration': {'fits': 9, 'prefixes_per_split': 8, 'splits': ['train', 'valid'],
                      'rows': 144, 'atol': 1e-10, 'rtol': 1e-10, 'selection_epsilon': 1e-10},
    'sources': {name: identity(name)['sha256'] for name in sources},
    'inputs': inputs,
    'python_executable': sys.executable,
    'python_version': sys.version.split()[0],
    'all_distributions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
    'scope': 'All144 existing parity rows from16 prefixes in two teacher episodes. Descriptive saved arithmetic; no inference, training, simulator or efficacy gate.',
    'assembly_source': identity(str(Path(__file__).relative_to(ROOT))),
}
assert not (OUT / 'run-01').exists()
with (OUT / 'plan-01.json').open('x') as stream:
    json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(identity('output/otto-return-consistency-v2/plan-01.json')))
