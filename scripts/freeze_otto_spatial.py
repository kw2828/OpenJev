"""Freeze an additive spatial study from authenticated closed metadata only."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_spatial_freezer_runner', ROOT/'scripts/study_otto_spatial.py')
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)
INPUTS = {
    'capacity_plan': ('output/otto-capacity-v1/plan-01.json', 'f00b138d01c8687d600a04fd37eac793387e026a1926e4094a2c0c079601e4de'),
    'capacity_receipt': ('output/otto-capacity-v1/run-01/receipt.json', 'ec7f9551c5a590f8548b14ec2a42a4456b0c65a87756185c76b67cc81986e8a9'),
    'capacity_terminal': ('output/otto-capacity-v1/run-process-01.terminal.json', '8d7ef2cc90ab986e2dc53e47bbe9dbdee7e6b7044c360ce03c27b2417e4c00d8'),
    'capacity_audit': ('output/otto-capacity-v1/audit-01/receipt.json', 'ac33b263d27a45dfd3f92f9486c4e1c6afe8da746c4bf6142851f2dc4adcc447'),
}


def freeze(args):
    output = args.output
    S.require(output.is_absolute() and not output.exists() and not any(p.is_symlink() for p in output.parents),
              'exclusive nonsymlink plan output')
    inputs = {}
    for role, (name, pin) in INPUTS.items():
        path = S.regular(name)
        S.require(S.sha(path) == pin, 'external historical identity')
        inputs[role] = {'path': name, **S.descriptor(path)}
    base = S.C.authenticate(SimpleNamespace(plan=S.regular(inputs['capacity_plan']['path']),
                                           plan_sha256=inputs['capacity_plan']['sha256']))
    qualification = {}
    for role, (name, pin) in S.QUALIFICATION.items():
        path = S.regular(name)
        S.require(S.sha(path) == pin, 'external synthetic qualification identity')
        qualification[role] = {'path': name, **S.descriptor(path)}
    qualified = S.authenticate_qualification(qualification)
    sources = dict(base['sources'])
    for name, pin in qualified['sources'].items():
        S.require(name not in sources or sources[name] == pin, 'consistent inherited source union')
        sources[name] = pin
    for name in S.NEW:
        pin = S.sha(S.regular(name))
        S.require(name not in sources or sources[name] == pin, 'no replacement of frozen source')
        sources[name] = pin
    for role in ('train_data', 'train_rows', 'valid_data', 'valid_rows'):
        inputs[role] = base['inputs'][role]
    plan = {'version': S.VERSION, 'status': 'frozen_before_execution', 'mode': 'study',
            'configuration': S.configuration(), 'limits': S.limits(), 'expected_calls': S.expected_calls(),
            'independent_audit_limits': S.AUDIT_LIMITS, 'sources': sources, 'inputs': inputs,
            'qualification': qualification, 'python_executable': base['python_executable'],
            'python_version': base['python_version'], 'all_distributions': base['all_distributions']}
    output.parent.mkdir(parents=True, exist_ok=True)
    S.write(output, plan)
    S.require(S.authenticate(SimpleNamespace(plan=output, plan_sha256=S.sha(output))) == plan,
              'assembled metadata plan validates without array decoding')
    print(json.dumps({'path': str(output), **S.descriptor(output), 'sources': len(sources),
                      'arrays_decoded': 0, 'empirical_model_calls': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    freeze(parser.parse_args())
