"""Assemble the reviewed return-value plan without numerical execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUTS = {
    'prior_plan': ('output/otto-symmetry-head-v1/plan-01.json', 'ae4475170f3eb8211b49c480871d1b534f201b7240cb579f099e01efee2727be'),
    'prior_receipt': ('output/otto-symmetry-head-v1/run-01/receipt.json', 'afd3ed59c6fc907721ebebcff47c1342fc53e592ec14b190497c647cb8d26c62'),
    'prior_terminal': ('output/otto-symmetry-head-v1/run-process-01.terminal.json', 'a5bd0e90b12df04d1bb8cb0a677c072c6d67f70b67c2892c66d6158cd87ccf90'),
    'prior_audit_receipt': ('output/otto-symmetry-head-v1/audit-01/receipt.json', '43ec2e8dc4016a9ec857b7ee3cf9dfb89e1107042c9fbfbfb4d1026d0d7ba6bb'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024**2), b''):
            h.update(part)
    return h.hexdigest()


def regular(name):
    rel = Path(name)
    require(not rel.is_absolute() and '..' not in rel.parts, 'relative contained source')
    path = ROOT / rel
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular contained source')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and not args.output.exists()
            and not any(p.is_symlink() for p in (args.output, *args.output.parents)), 'exclusive absolute plan path')
    engineering = Path(__file__).parent
    reviewed = json.loads((engineering / 'final-source-review.json').read_text())
    require(reviewed['status'] == 'completed' and reviewed['source_reviews_clear'] is True, 'final actual source reviews')
    for name, pin in reviewed['sources'].items():
        require(sha(regular(name)) == pin, 'held source changed: ' + name)
    inputs = {}
    for role, (name, pin) in INPUTS.items():
        path = regular(name)
        require(sha(path) == pin, 'completed input pin: ' + role)
        inputs[role] = {'path': name, 'sha256': pin, 'bytes': path.stat().st_size}
    prior = json.loads(regular(INPUTS['prior_plan'][0]).read_text())
    worker = json.loads(regular(INPUTS['prior_receipt'][0]).read_text())
    audit = json.loads(regular(INPUTS['prior_audit_receipt'][0]).read_text())
    require(len(prior['sources']) == 121 and worker['status'] == 'completed'
            and worker['completed_episodes'] == 720 and worker['completed_stage_fits'] == 12
            and worker['collection_episodes'] == 384 and worker['pending'] == [], 'complete prior study')
    require(audit['status'] == 'completed' and audit['agreement'] is True, 'complete independent prior audit')
    summary = json.loads((regular(INPUTS['prior_receipt'][0]).parent / 'summary.json').read_text())
    require(summary['pilot_continuation'] is False
            and summary['learned_architecture_advantage_established'] is False
            and summary['inherited_gate_revised'] is False, 'unchanged failed prior scientific decision')
    spec = importlib.util.spec_from_file_location('_return_plan_constants', ROOT / 'scripts/study_otto_return_value.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(not {'numpy', 'torch', 'scipy', 'tensorflow'}.intersection(sys.modules), 'source-only plan assembly')
    require(set(inputs) == module.ROLES and module.REQUIRED <= reviewed['sources'].keys() | prior['sources'].keys(),
            'complete reviewed new sources')
    sources = dict(prior['sources'])

    def merge(name, pin):
        require(name not in sources or sources[name] == pin, 'source conflict: ' + name)
        require(sha(regular(name)) == pin, 'current source identity: ' + name)
        sources[name] = pin

    for name, pin in reviewed['sources'].items():
        merge(name, pin)
    files = []
    evidence_roots = [engineering, *(ROOT / p for p in reviewed['engineering_directories'])]
    for folder in evidence_roots:
        require(folder.is_relative_to(ROOT) and not folder.is_symlink(), 'contained engineering folder')
        for path in sorted(folder.rglob('*')):
            require(not path.is_symlink(), 'no engineering symlink')
            if path.is_file():
                require('__pycache__' not in path.parts, 'no volatile bytecode')
                name = path.relative_to(ROOT).as_posix()
                merge(name, sha(path))
                files.append(name)
    for name, pin in sources.items():
        require(sha(regular(name)) == pin, 'unchanged complete source closure: ' + name)
    plan = {
        'version': module.VERSION, 'status': 'frozen_before_native_run',
        'configuration': module.CONFIGURATION, 'limits': module.LIMITS,
        'sources': dict(sorted(sources.items())), 'inputs': inputs,
        'python_version': sys.version.split()[0], 'python_executable': sys.executable,
        'runtime_versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'torch')},
        'all_distributions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'assembly': {'helper_sha256': sha(Path(__file__)), 'inherited_source_count': 121,
                     'engineering_files': sorted(set(files)),
                     'scope': 'Source and identity assembly only. Authentication and supervised execution remain separate.'},
    }
    with args.output.open('x') as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'plan': str(args.output), 'sha256': sha(args.output), 'sources': len(sources),
                      'runtime_distributions': len(plan['all_distributions']), 'new_native_or_training_calls': 0}))


if __name__ == '__main__':
    main()
