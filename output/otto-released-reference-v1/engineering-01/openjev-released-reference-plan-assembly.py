"""Prepare one exclusive OpenJev reference-study plan when the owner invokes it.

Stdlib only. This helper does not authenticate by running the study and does not
import NumPy, TensorFlow, OTTO, model weights or saved trajectory arrays.
The owner must finish source review/engineering qualification before invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

HELD = {
    'scripts/study_otto_released_reference.py': 'da2a7e418dd29819bddf190f46d1707f1163a22b01a048b33d3dedab875ad87e',
    'tests/test_otto_released_reference.py': 'e7d7f5191134a9dfef0b84648a3b3cb6bf82628797c9455007898de1569050dc',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('qualification-plan', 'qualification-receipt', 'qualification-terminal',
                 'audit-receipt', 'analytic-preflight-receipt'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--engineering', type=Path, action='append', required=True,
                        help='Repeat for each completed, preserved engineering receipt included in the source closure.')
    parser.add_argument('--analysis-source', type=Path, action='append', required=True,
                        help='Repeat for the final held independent auditor and its synthetic tests.')
    args = parser.parse_args()
    root = args.root
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'absolute real repository root')
    require(args.output.is_absolute() and args.output.is_relative_to(root), 'absolute output inside repository')

    def regular(value):
        path = value if value.is_absolute() else root / value
        require(path.is_relative_to(root) and '..' not in path.parts and path.is_file()
                and not any(p.is_symlink() for p in (path, *path.parents)), 'contained nonsymlink file')
        return path

    def document(value):
        return json.loads(regular(value).read_text())

    for name, pin in HELD.items():
        require(sha(regular(Path(name))) == pin, f'held runner/test changed: {name}')
    spec = importlib.util.spec_from_file_location('reference_plan_constants', root / 'scripts/study_otto_released_reference.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    previous = document(args.qualification_plan)
    worker = document(args.qualification_receipt)
    terminal = document(args.qualification_terminal)
    audit = document(args.audit_receipt)
    preflight = document(args.analytic_preflight_receipt)
    require(previous['status'] == 'frozen_before_run' and worker['status'] == 'completed' and worker['qualified'] is True,
            'successful prior native qualification')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['group_absent'] is True,
            'successful native parent terminal')
    require(audit['status'] == 'completed' and audit['agreement'] is True, 'successful independent native audit')
    require(preflight['status'] == 'completed' and preflight['qualified'] is True and preflight['agreement'] is True,
            'successful exact analytic qualification')
    source_pins = dict(previous['sources'])

    def merge(name, pin):
        require(name not in source_pins or source_pins[name] == pin, f'conflicting inherited source identity: {name}')
        require(sha(regular(Path(name))) == pin, f'current source differs from evidence: {name}')
        source_pins[name] = pin

    for name, pin in preflight['sources'].items():
        merge(name, pin)
    for name in study.REQUIRED:
        merge(name, sha(regular(Path(name))))
    require(source_pins[study.PREFLIGHT] == preflight['self_source']['sha256'], 'analytic checker matches completed qualification')
    for path in (*args.analysis_source, *args.engineering):
        actual = regular(path)
        merge(actual.relative_to(root).as_posix(), sha(actual))
    for name, pin in source_pins.items():
        require(sha(regular(Path(name))) == pin, f'complete source closure: {name}')
    inputs = {}
    for role in sorted(study.ROLES):
        actual = regular(getattr(args, role))
        inputs[role] = {'path': actual.relative_to(root).as_posix(), 'sha256': sha(actual), 'bytes': actual.stat().st_size}
    plan = {'version': study.VERSION, 'status': 'frozen_before_native_run',
            'configuration': study.CONFIGURATION, 'limits': study.LIMITS,
            'runtime': previous['runtime'], 'sources': dict(sorted(source_pins.items())), 'inputs': inputs,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'assembly': {'helper_sha256': sha(Path(__file__)), 'helper_scope': 'Source/pin assembly only; no scientific calls.',
                         'inherited_native_sources': len(previous['sources']), 'engineering_receipts': len(args.engineering),
                         'analysis_sources': [regular(p).relative_to(root).as_posix() for p in args.analysis_source]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'plan': str(args.output), 'sha256': sha(args.output),
                      'sources': len(source_pins), 'inputs': len(inputs), 'native_or_model_calls': 0}))


if __name__ == '__main__':
    main()
