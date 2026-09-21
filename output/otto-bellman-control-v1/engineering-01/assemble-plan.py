"""Metadata-only plan assembly. Does not import numerical code or decode arrays."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / 'output/otto-bellman-control-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def desc(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def integers(value):
    if type(value) is int:
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from integers(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from integers(item)


def seed_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if 'seed' in key:
                yield from integers(item)
            else:
                yield from seed_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from seed_values(item)


def assemble(mode):
    spec = importlib.util.spec_from_file_location('_bellman_plan_runner', ROOT / 'scripts/study_otto_bellman_control.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    prior = read(ROOT / 'output/otto-return-value-v1/plan-01.json')
    sources = dict(prior['sources'])
    required = [*module.NEW_SOURCES, 'research/otto-bellman-control-design.md', 'research/otto-learning-direction.md',
                str(Path(__file__).resolve().relative_to(ROOT))]
    for folder in ('learning-qualification-01', 'learning-qualification-02', 'runner-qualification-01'):
        required.extend(str(p.relative_to(ROOT)) for p in (BASE / 'engineering-01' / folder).iterdir() if p.is_file())
    if mode == 'study':
        required.extend(['scripts/audit_otto_bellman_control.py', 'tests/test_audit_otto_bellman_control.py'])
    for rel in required:
        assert (ROOT / rel).is_file(), rel
        sources[rel] = sha(ROOT / rel)
    for rel, pin in prior['sources'].items():
        assert sha(ROOT / rel) == pin, rel
    assert read(BASE / 'engineering-01/runner-qualification-01/receipt.json')['status'] == 'passed'
    for rel, pin in read(BASE / 'engineering-01/runner-qualification-01/receipt.json')['sources'].items():
        assert sha(ROOT / rel) == pin, rel
    new_seeds = {first + i for first in module.FIRST.values() for i in range(48)} | set(range(module.TEMPLATE_FIRST, module.TEMPLATE_FIRST + 3))
    ledger_files, plan_files, seen = [], [], set()
    for directory in (ROOT / 'output').glob('otto*'):
        if directory == BASE:
            continue
        for path in directory.glob('*plan*.json'):
            values = set(seed_values(read(path)))
            assert not values & new_seeds, path
            plan_files.append({**desc(path), 'seed_declarations': sorted(values)})
        for path in directory.rglob('*episodes*.jsonl'):
            values = set()
            with path.open() as stream:
                for line in stream:
                    values.update(seed_values(json.loads(line)))
            assert not values & new_seeds, path
            seen.update(values)
            ledger_files.append({**desc(path), 'distinct_seed_values': len(values)})
    inputs = {role: desc(ROOT / name) for role, name in {
        'prior_plan': 'output/otto-return-value-v1/plan-01.json',
        'prior_receipt': 'output/otto-return-value-v1/run-01/receipt.json',
        'prior_terminal': 'output/otto-return-value-v1/run-process-01.terminal.json',
        'prior_audit_receipt': 'output/otto-return-value-v1/audit-01/receipt.json'}.items()}
    plan = {'version': module.VERSION, 'status': 'frozen_before_execution', 'mode': mode,
            'configuration': module.configuration(mode), 'limits': module.limits(mode), 'sources': sources, 'inputs': inputs,
            'evaluation_first_seeds': module.FIRST, 'template_first_seed': module.TEMPLATE_FIRST,
            'python_version': sys.version.split()[0], 'python_executable': sys.executable,
            'all_distributions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'seed_review': {'plan_files': plan_files, 'episode_ledgers': ledger_files,
                            'distinct_observed_seed_values': len(seen), 'proposed_seeds': sorted(new_seeds),
                            'overlap': [], 'scope': 'Exact observed episode seed nonreuse plus declared seed-start review; no performance fields used.'}}
    if mode == 'study':
        plan['qualification'] = {role: desc(BASE / name) for role, name in {
            'plan': 'qualification-plan-01.json', 'receipt': 'qualification-01/receipt.json',
            'terminal': 'qualification-process-01.terminal.json'}.items()}
        assert read(BASE / 'qualification-01/receipt.json')['status'] == 'completed'
        assert read(BASE / 'qualification-process-01.terminal.json')['status'] == 'completed'
        plan['independent_audit_limits'] = {'seconds': 5400, 'rss_bytes': 4 * 1024**3, 'output_bytes': 256 * 1024**2}
    output = BASE / ('qualification-plan-01.json' if mode == 'qualify' else 'plan-01.json')
    with output.open('x') as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'mode': mode, 'path': str(output), 'sha256': sha(output), 'sources': len(sources),
                      'prior_plans_checked': len(plan_files), 'episode_ledgers_checked': len(ledger_files), 'observed_seeds': len(seen)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['qualify', 'study'])
    assemble(parser.parse_args().mode)
