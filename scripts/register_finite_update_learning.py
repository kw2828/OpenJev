"""Freeze a fresh equal-update comparison and its qualified engineering exposure."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from finite_update_learning_worker import (
    ROOT,
    admit_qualification,
    descriptor,
    files,
    publish,
    require,
    runtime,
)
from register_finite_rounded_learning import SOURCES as PREVIOUS_SOURCES
from register_finite_rounded_learning import authenticate_kernel

CONFIG = {'seed_namespace': 433260924, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': [433261001, 433261002, 433261003],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'prefix_updates': 1024, 'joint_updates': 3072, 'fit_cap_seconds': 120.}
EXPOSURE = {'config': {**CONFIG, 'seed_namespace': 942201, 'fit_seeds': [942301],
                       'dev_attempts': 8, 'prefix_updates': 32, 'joint_updates': 64,
                       'fit_cap_seconds': 30.},
            'target_counts': {'prefix_updates': 1024, 'joint_updates': 3072},
            'maximum_projected_seconds': 90.0}
SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES,
    'src/openjev/research/finite_update_allocation.py',
    'scripts/run_finite_update_learning.py', 'scripts/audit_finite_update_learning.py',
    'scripts/register_finite_update_learning.py', 'scripts/finite_update_learning_worker.py',
    'scripts/qualify_finite_update_exposure.py',
    'tests/test_finite_update_allocation.py', 'tests/test_finite_update_learning_runner.py',
    'tests/test_finite_update_learning_audit.py', 'tests/test_finite_update_learning_worker.py',
    'research/finite-update-learning-protocol.md', 'research/finite-update-learning-requirements.txt',
)))
TESTS = ['tests/test_finite_rounded_transition.py', 'tests/test_finite_rounded_models.py',
         'tests/test_finite_update_allocation.py', 'tests/test_finite_update_learning_runner.py',
         'tests/test_finite_update_learning_audit.py', 'tests/test_finite_update_learning_worker.py']
PARENT_SHA256 = '9779c1fa2db7ef6a69c195793b7ca7f0da992c55c9433c441a563d4c6d614a45'
PARENT_FILES = ('output/finite-rounded-learning-v1/study-registration.json',
    'research/finite-rounded-learning-results/summary.json',
    'research/finite-rounded-learning-results/receipt.json',
    'output/finite-rounded-learning-package-v1/manifest.json',
    'output/finite-rounded-learning-delivery-v1/delivery-verification.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('engineering', 'study'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'dedicated repository')
    folder = ROOT / 'output/finite-update-learning-v1'
    folder.mkdir(exist_ok=True)
    expected_plan = folder / ('engineering-registration-01.json' if args.mode == 'engineering'
                              else 'study-registration.json')
    require(args.plan.resolve() == expected_plan and not expected_plan.exists(), 'exclusive registered plan path')
    phases = {name: {'cap_seconds': cap, 'output': str(folder / output),
                     'supervision': str(folder / (prefix + '.launch.json'))}
              for name, cap, output, prefix in (
                  ('qualify', 300, 'engineering-01', 'engineering-native-01'),
                  ('fit', 1200, 'run-01', 'fit-native-01'),
                  ('audit', 600, 'audit-01', 'audit-native-01'))}
    sources = {name: descriptor(ROOT / name) for name in SOURCES}
    parent = json.loads((ROOT / PARENT_FILES[0]).read_text())
    require(descriptor(ROOT / PARENT_FILES[0])['sha256'] == PARENT_SHA256, 'closed parent registration')
    require(all(descriptor(ROOT / name) == pin for name, pin in parent['sources'].items()),
            'all parent sources unchanged')
    require(json.loads((ROOT / PARENT_FILES[1]).read_text())['advance']['passed'] is False,
            'previous equal-time result remains failed')
    kernel = ROOT / 'output/finite-rounded-transition-qualification-v1'
    qualified = authenticate_kernel(kernel)
    plan = {'version': 'finite-update-learning-v1', 'mode': args.mode,
            'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'config': CONFIG, 'exposure_probe': EXPOSURE, 'phases': phases,
            'tests': TESTS, 'lint_sources': [name for name in SOURCES if name.endswith('.py')],
            'rss_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2,
            'kernel_qualification': {'folder': str(kernel), 'files': files(kernel),
                'publication': str(qualified['publication']),
                'publication_files': files(qualified['publication'])},
            'parent_evidence': {name: descriptor(ROOT / name) for name in PARENT_FILES}}
    if args.mode == 'study':
        require(args.qualification is not None, 'completed engineering receipt required')
        terminal = admit_qualification(plan, args.qualification)
        plan['qualification'] = {'path': str(args.qualification.resolve()),
            'descriptor': descriptor(args.qualification), 'terminal': descriptor(terminal)}
        require(not Path(phases['fit']['output']).exists(), 'no prior scientific run')
    publish(args.plan, plan)
    snapshot = folder / ('source-snapshot-' + args.mode + '-01')
    snapshot.mkdir(exist_ok=False)
    for name, pin in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
        require(descriptor(target) == pin, 'unchanged snapshot source')
    publish(snapshot / 'manifest.json', {'registration': {'path': str(args.plan.resolve()),
            **descriptor(args.plan)}, 'sources': sources})
    print(json.dumps({'plan': str(args.plan.resolve()), **descriptor(args.plan)}))


if __name__ == '__main__':
    main()
