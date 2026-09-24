"""Create a new immutable engineering or study registration, without model calls."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from finite_balanced_learning_worker import (
    ROOT,
    admit_qualification,
    descriptor,
    files,
    publish,
    require,
    runtime,
)
from publish_finite_balanced_transition_qualification import REGISTRATION_SHA256 as KERNEL_SHA256
from publish_finite_balanced_transition_qualification import authenticate as authenticate_kernel
from register_finite_training_allocation import SOURCES as PREVIOUS_SOURCES

SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES,
    'src/openjev/research/finite_balanced_transition.py',
    'src/openjev/research/finite_balanced_models.py',
    'scripts/qualify_finite_balanced_transition.py',
    'scripts/publish_finite_balanced_transition_qualification.py',
    'scripts/register_finite_balanced_learning.py',
    'scripts/finite_balanced_learning_worker.py',
    'scripts/run_finite_balanced_learning.py',
    'scripts/audit_finite_balanced_learning.py',
    'tests/test_finite_balanced_transition.py',
    'tests/test_finite_balanced_models.py',
    'tests/test_finite_balanced_learning_runner.py',
    'tests/test_finite_balanced_learning_audit.py',
    'tests/test_finite_balanced_learning_worker.py',
    'research/finite-balanced-transition-qualification-protocol.md',
    'research/finite-balanced-learning-protocol.md',
    'research/finite-balanced-learning-requirements.txt',
    'src/openjev/__init__.py', 'src/openjev/research/__init__.py',
    'pyproject.toml', 'uv.lock',
)))
TESTS = ['tests/test_finite_balanced_transition.py', 'tests/test_finite_balanced_models.py',
         'tests/test_finite_training_allocation.py', 'tests/test_finite_balanced_learning_runner.py',
         'tests/test_finite_balanced_learning_audit.py', 'tests/test_finite_balanced_learning_worker.py']
LINT = [name for name in SOURCES if name.endswith('.py')]
CONFIG = {'seed_namespace': 431260924, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': [431261001, 431261002, 431261003],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'stage1_seconds': 10., 'total_seconds': 40.}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('engineering', 'study'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--engineering-attempt', type=int, default=1)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'dedicated repository only')
    require(args.engineering_attempt > 0, 'positive engineering attempt number')
    engineering_suffix = f'{args.engineering_attempt:02d}'
    folder = ROOT / 'output/finite-balanced-learning-v1'
    folder.mkdir(exist_ok=True)
    phases = {name: {'cap_seconds': cap, 'output': str(folder / output),
                     'supervision': str(folder / (prefix + '.launch.json'))}
              for name, cap, output, prefix in (
                  ('qualify', 300, 'engineering-' + engineering_suffix, 'engineering-native-' + engineering_suffix),
                  ('fit', 1200, 'run-01', 'fit-native-01'),
                  ('audit', 600, 'audit-01', 'audit-native-01'))}
    sources = {name: descriptor(ROOT / name) for name in SOURCES}
    kernel = ROOT / 'output/finite-balanced-transition-qualification-v1'
    qualified = authenticate_kernel(kernel, KERNEL_SHA256)
    kernel_plan = qualified['plan']
    require({name: sources[name] for name in kernel_plan['sources']} == kernel_plan['sources'],
            'unchanged qualified balancing sources')
    plan = {'version': 'finite-balanced-learning-v1', 'mode': args.mode,
            'kernel_qualification': {'folder': str(kernel), 'files': files(kernel)},
            'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'config': CONFIG, 'phases': phases, 'tests': TESTS, 'lint_sources': LINT,
            'rss_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2}
    if args.mode == 'study':
        require(args.qualification is not None, 'completed engineering receipt required')
        qualification = json.loads(args.qualification.read_text())
        require(qualification['status'] == 'PASS' and qualification['sources_after'] == sources,
                'qualification must pass on identical sources')
        terminal = admit_qualification(plan, args.qualification)
        plan['qualification'] = {'path': str(args.qualification.resolve()),
                                 'descriptor': descriptor(args.qualification),
                                 'terminal': descriptor(terminal)}
        require(not Path(phases['fit']['output']).exists(), 'no prior study outputs')
    publish(args.plan, plan)
    snapshot = folder / ('source-snapshot-' + args.mode + '-' + engineering_suffix)
    snapshot.mkdir(exist_ok=False)
    for name, pin in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (ROOT / name).read_bytes()
        with target.open('xb') as stream:
            stream.write(payload)
        require(descriptor(target) == pin, 'unchanged source while snapshotting')
    publish(snapshot / 'manifest.json', {'registration': {'path': str(args.plan.resolve()),
            **descriptor(args.plan)}, 'sources': sources})
    print(json.dumps({'plan': str(args.plan.resolve()), **descriptor(args.plan)}))


if __name__ == '__main__':
    main()
