"""Assemble the reviewed boundary-control plan only when the owner invokes it.

This helper uses stdlib source loading and file hashing only. It does not import
the numerical runtime, authenticate by starting the study, or run model/simulator
calls. Final plan authentication and launch remain separate owner actions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

HELD = {
    'scripts/study_otto_boundary_control.py': 'f00af83597e9ecd4bf8e2354fad7e4c6e25f69a9e182fcdd12f0e47050081dd8',
    'tests/test_otto_boundary_control.py': 'b22187b411f439df10b2c5a89cc7f8314dbc41aa989cf5543768936249b94766',
    'scripts/audit_otto_boundary_control.py': '9a1ab23570104a73c7700db3a43d0245fe0b3487f3233212884cd2ded34e35b2',
    'tests/test_audit_otto_boundary_control.py': '7fd1df56288c284b640eb5c2a2595ee9fbdf8fddfca6576a67214bd26fc9b92b',
    'src/openjev/research/otto_restricted_policy.py': '3e63240b3502b4d4633f9c1671b4ad7ce34466f21c74a4d470bed91d77514630',
    'tests/test_otto_restricted_policy.py': '49132b1c8cf3e7891a88cb387f40f01e10afb946952a58463c129743b275c0eb',
    'scripts/qualify_otto_restricted_policy.py': '05a93d9260dcf483d208879bc0c2c922dd0e1c1d6ed7eb227778fb76ac7431d8',
    'research/otto-boundary-control-protocol.md': '9c86c160e51307c51df8b890fffc00914ea3a89093e83330cb1aef9491f6aef5',
}
INPUTS = {
    'prior_plan': ('output/otto-released-reference-v1/plan-01.json', '85b64b07a37f40766f8e890170ed102035af01f4cd3611dfcc80cce6d7cfe1f4'),
    'prior_receipt': ('output/otto-released-reference-v1/run-01/receipt.json', 'd360c63525d72a289c7ade4ec09377769ee4156bad2402e6185a1808d337d154'),
    'prior_terminal': ('output/otto-released-reference-v1/run-process-01.terminal.json', '306ad30cb07748e30c6292b7f3cf85cc83db3d4d21e3715f3ff7179469b0abb1'),
    'prior_audit_receipt': ('output/otto-released-reference-v1/audit-01/receipt.json', '0ce85a845994a3ce9877876e1f2ebf6774131ce390580948412051708b2515aa'),
    'boundary_preflight_receipt': ('output/otto-boundary-control-v1/selection-preflight-01/receipt.json', '4fc1a60a2638fbfe4e32d51bbdb5d47df6f62fb750c61976b92e435879ede996'),
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
    parser.add_argument('--engineering', type=Path, action='append', required=True,
                        help='Repeat for files/directories; every contained file enters the source closure.')
    args = parser.parse_args()
    root = args.root
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'absolute real repository root')
    require(args.output.is_absolute() and args.output.is_relative_to(root) and '..' not in args.output.parts,
            'exclusive absolute output inside repository')
    require(not args.output.exists() and not any(p.is_symlink() for p in (args.output, *args.output.parents)),
            'new nonsymlink output')

    def regular(value):
        path = value if value.is_absolute() else root / value
        require(path.is_relative_to(root) and '..' not in path.parts and path.is_file()
                and not any(p.is_symlink() for p in (path, *path.parents)), 'contained nonsymlink input file')
        return path

    def document(name):
        return json.loads(regular(Path(name)).read_text())

    for name, pin in HELD.items():
        require(sha(regular(Path(name))) == pin, f'held reviewed source changed: {name}')
    inputs = {}
    for role, (name, pin) in INPUTS.items():
        path = regular(Path(name))
        require(sha(path) == pin, f'external completed input pin: {role}')
        inputs[role] = {'path': name, 'sha256': pin, 'bytes': path.stat().st_size}

    previous = document(INPUTS['prior_plan'][0])
    worker = document(INPUTS['prior_receipt'][0])
    terminal = document(INPUTS['prior_terminal'][0])
    audit = document(INPUTS['prior_audit_receipt'][0])
    preflight = document(INPUTS['boundary_preflight_receipt'][0])
    require(previous['status'] == 'frozen_before_native_run' and len(previous['sources']) == 71,
            'complete immutable71-source prior plan')
    require(worker['status'] == 'completed' and worker['completed_episodes'] == 576
            and worker['plan_sha256'] == INPUTS['prior_plan'][1]
            and worker['sources'] == previous['sources'] and worker['inputs'] == previous['inputs'],
            'complete prior worker with unchanged bindings')
    require(all(worker[k] is False for k in ('competent_reference', 'stronger_value_teacher', 'utility_compute_advantage')),
            'preserve all original failed overall rules')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['group_absent'] is True,
            'successful prior native parent terminal')
    require(audit['status'] == 'completed' and audit['agreement'] is True
            and audit['plan_sha256'] == INPUTS['prior_plan'][1]
            and audit['worker_sha256'] == INPUTS['prior_receipt'][1]
            and audit['terminal_sha256'] == INPUTS['prior_terminal'][1], 'independent prior audit joins')
    require(preflight['status'] == 'completed' and preflight['qualified'] is True and preflight['agreement'] is True
            and preflight['comparisons'] == preflight['fake_policy_calls'] == preflight['public_actor_constructions'] == 16
            and preflight['public_updates'] == 128
            and preflight['model_calls'] == preflight['native_calls'] == preflight['simulator_calls'] == 0,
            'successful saved-cost selection qualification')

    # This import is verified source-only; the runner top level uses stdlib and
    # two hash-authenticated namespaces of the unchanged stdlib reference runner.
    spec = importlib.util.spec_from_file_location('boundary_plan_constants', root / 'scripts/study_otto_boundary_control.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    require(study.ROLES == INPUTS.keys(), 'exact five input roles')
    source_pins = dict(previous['sources'])

    def merge(name, pin):
        require(name not in source_pins or source_pins[name] == pin, f'conflicting inherited source: {name}')
        require(sha(regular(Path(name))) == pin, f'current source differs from reviewed evidence: {name}')
        source_pins[name] = pin

    for name, pin in preflight['sources'].items():
        merge(name, pin)
    for name, pin in HELD.items():
        merge(name, pin)
    for name in study.REQUIRED:
        merge(name, HELD.get(name, sha(regular(Path(name)))))
    require(source_pins[study.PREFLIGHT] == preflight['self_source']['sha256'], 'exact completed selector checker')
    engineering = set()
    for value in args.engineering:
        path = value if value.is_absolute() else root / value
        require(path.is_relative_to(root) and '..' not in path.parts and path.exists()
                and not any(p.is_symlink() for p in (path, *path.parents)), 'contained engineering path')
        for item in ([path] if path.is_file() else sorted(path.rglob('*'))):
            require(not item.is_symlink(), 'engineering closure contains no symlinks')
            if item.is_dir():
                continue
            actual = regular(item)
            name = actual.relative_to(root).as_posix()
            merge(name, sha(actual))
            engineering.add(name)
    require(engineering, 'nonempty preserved engineering closure')
    for name, pin in source_pins.items():
        require(sha(regular(Path(name))) == pin, f'complete current source closure: {name}')
    plan = {
        'version': study.VERSION, 'status': 'frozen_before_native_run',
        'configuration': study.CONFIGURATION, 'limits': study.LIMITS,
        'runtime': previous['runtime'], 'sources': dict(sorted(source_pins.items())), 'inputs': inputs,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'assembly': {'helper_sha256': sha(Path(__file__)),
                     'scope': 'Source and pin assembly only; no model or simulator calls. Separate final authentication required.',
                     'inherited_sources': len(previous['sources']), 'engineering_files': sorted(engineering),
                     'reviewed_sources': HELD},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'plan': str(args.output), 'sha256': sha(args.output),
                      'sources': len(source_pins), 'inputs': len(inputs), 'model_or_native_calls': 0}))


if __name__ == '__main__':
    main()
