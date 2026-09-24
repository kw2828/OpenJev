"""Create a new immutable engineering or study registration, without model calls."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from finite_prefix_learning_worker import ROOT, closed_producer, descriptor, publish, require, runtime

SOURCES = (
    'scripts/register_finite_prefix_learning.py',
    'scripts/finite_prefix_learning_worker.py',
    'scripts/run_finite_prefix_learning.py',
    'scripts/audit_finite_prefix_learning.py',
    'scripts/run_finite_observation_learning.py',
    'scripts/audit_finite_observation_learning.py',
    'scripts/supervise_dialogue_observation_v2.py',
    'src/openjev/research/suspend_clock.py',
    'src/openjev/research/finite_factor_models.py',
    'src/openjev/research/finite_prefix_learning.py',
    'src/openjev/research/finite_shared_filter_models.py',
    'src/openjev/research/finite_observation_world.py',
    'src/openjev/research/finite_observation_models.py',
    'src/openjev/research/otto_observation_operator_model.py',
    'src/openjev/__init__.py',
    'src/openjev/research/__init__.py',
    'tests/test_finite_prefix_learning.py',
    'tests/test_finite_prefix_learning_audit.py',
    'tests/test_finite_prefix_learning_worker.py',
    'tests/test_finite_prefix_learning_runner.py',
    'research/finite-prefix-learning-protocol.md',
    'pyproject.toml',
    'uv.lock',
)
TESTS = [name for name in SOURCES if name.startswith('tests/')]
LINT = [name for name in SOURCES if name.endswith('.py')]
CONFIG = {'seed_namespace': 423260924, 'train_attempts': 512, 'dev_attempts': 128,
          'epochs': 480, 'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': [423261001, 423261002, 423261003],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8}


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
    folder = ROOT / 'output/finite-prefix-learning-v1'
    folder.mkdir(exist_ok=True)
    phases = {name: {'cap_seconds': cap, 'output': str(folder / output),
                     'supervision': str(folder / (prefix + '.launch.json'))}
              for name, cap, output, prefix in (
                  ('qualify', 300, 'engineering-' + engineering_suffix, 'engineering-native-' + engineering_suffix),
                  ('fit', 1800, 'run-01', 'fit-native-01'),
                  ('audit', 600, 'audit-01', 'audit-native-01'))}
    sources = {name: descriptor(ROOT / name) for name in SOURCES}
    plan = {'version': 'finite-prefix-learning-v1', 'mode': args.mode,
            'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'config': CONFIG, 'phases': phases, 'tests': TESTS, 'lint_sources': LINT,
            'rss_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2}
    if args.mode == 'study':
        require(args.qualification is not None, 'completed engineering receipt required')
        qualification = json.loads(args.qualification.read_text())
        require(qualification['status'] == 'PASS' and qualification['sources_after'] == sources,
                'qualification must pass on identical sources')
        terminal = closed_producer(qualification)
        plan['qualification'] = {'path': str(args.qualification.resolve()),
                                 'descriptor': descriptor(args.qualification),
                                 'terminal': descriptor(terminal)}
        require(not Path(phases['fit']['output']).exists(), 'no prior study outputs')
    publish(args.plan, plan)
    print(json.dumps({'plan': str(args.plan.resolve()), **descriptor(args.plan)}))


if __name__ == '__main__':
    main()
