"""Freeze collection from completed evidence and seed metadata, without arrays."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_coverage_freezer', ROOT/'scripts/collect_otto_coverage.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
BASE = ROOT/'output/otto-capacity-v1/plan-01.json'
BASE_PIN = 'f00b138d01c8687d600a04fd37eac793387e026a1926e4094a2c0c079601e4de'
INPUTS = {
    'capacity_plan': ('output/otto-capacity-v1/plan-01.json', BASE_PIN),
    'capacity_receipt': ('output/otto-capacity-v1/run-01/receipt.json', 'ec7f9551c5a590f8548b14ec2a42a4456b0c65a87756185c76b67cc81986e8a9'),
    'capacity_terminal': ('output/otto-capacity-v1/run-process-01.terminal.json', '8d7ef2cc90ab986e2dc53e47bbe9dbdee7e6b7044c360ce03c27b2417e4c00d8'),
    'capacity_audit': ('output/otto-capacity-v1/audit-01/receipt.json', 'ac33b263d27a45dfd3f92f9486c4e1c6afe8da746c4bf6142851f2dc4adcc447'),
}


def freeze(output):
    S.require(output.is_absolute() and not output.exists() and not any(p.is_symlink() for p in output.parents), 'exclusive plan')
    base = S.C.authenticate(SimpleNamespace(plan=BASE, plan_sha256=BASE_PIN))
    sources = dict(base['sources'])
    for name in S.NEW:
        sources[name] = S.sha(S.regular(name))
    inputs = {}
    for role, (name, pin) in INPUTS.items():
        path = S.regular(name)
        S.require(S.sha(path) == pin, 'externally bound historical evidence')
        inputs[role] = {'path': name, **S.descriptor(path)}
    review = S.read(ROOT/'output/otto-coverage-v1/seed-review-01.json')
    S.require(review['passed'] is True and review['overlap'] == [], 'metadata seed nonreuse')
    for name, pin in review['files'].items():
        S.require(S.sha(S.regular(name)) == pin, 'unchanged inspected seed source')
    ledger = ROOT/'output/otto-coverage-v1/seed-ledger-review-01.json'
    ledger_review = S.read(ledger)
    S.require(ledger_review['passed'] is True and ledger_review['overlaps'] == [], 'actual recorded seed nonreuse')
    sys.path.insert(0, str(ROOT/'src'))
    from openjev.research.otto_coverage_selection import coverage_quotas
    rows = [json.loads(v) for v in S.regular(base['inputs']['train_rows']['path']).read_text().splitlines()]
    plan = {'version': S.VERSION, 'status': 'frozen_before_execution', 'configuration': S.configuration(),
            'limits': S.LIMITS, 'independent_audit_limits': S.AUDIT_LIMITS,
            'sources': sources, 'inputs': inputs, 'quota_table': coverage_quotas(rows),
            'python_executable': base['python_executable'], 'python_version': base['python_version'],
            'all_distributions': base['all_distributions'], 'seed_review': review,
            'seed_ledger_review': {'path': str(ledger.relative_to(ROOT)), **S.descriptor(ledger)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'path': str(output), 'sha256': S.sha(output), 'sources': len(sources), 'arrays_decoded': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    freeze(parser.parse_args().output)
