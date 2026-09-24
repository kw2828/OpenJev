"""Register fabricated arithmetic and integration checks before execution."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from qualify_finite_expected_count import ROOT, VERSION, descriptor, publish, runtime

SOURCES = (
    'scripts/register_finite_expected_count_qualification.py',
    'scripts/qualify_finite_expected_count.py',
    'scripts/supervise_dialogue_observation_v2.py',
    'src/openjev/research/suspend_clock.py',
    'src/openjev/research/finite_expected_count.py',
    'src/openjev/research/finite_expected_count_bridge.py',
    'src/openjev/research/finite_factorized_dynamics_models.py',
    'src/openjev/research/finite_factor_models.py',
    'src/openjev/research/finite_cost_readout_models.py',
    'src/openjev/research/finite_shared_filter_models.py',
    'src/openjev/research/finite_observation_world.py',
    'src/openjev/research/otto_observation_operator_model.py',
    'src/openjev/__init__.py',
    'src/openjev/research/__init__.py',
    'tests/test_finite_expected_count.py',
    'tests/test_finite_expected_count_oracle.py',
    'tests/test_finite_expected_count_bridge.py',
    'research/finite-expected-count-qualification-protocol.md',
    'research/finite-expected-count-qualification-requirements.txt',
    'pyproject.toml',
    'uv.lock',
)
FIXTURES = [([], [0]), ([0], [0, 1]), ([1], [2, 4]), ([0, 1], [0, 2, 3]),
            ([1, 0], [3, 1, 4]), ([0, 1, 0], [0, 1, 2, 3]), ([1, 0, 1], [2, 0, 3, 4])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        raise ValueError('dedicated repository required')
    folder = ROOT / 'output' / VERSION
    folder.mkdir(exist_ok=True)
    if args.plan.resolve() != folder / 'registration.json':
        raise ValueError('one original engineering registration')
    plan = {'version': VERSION, 'root': str(ROOT), 'runtime': runtime(),
            'sources': {name: descriptor(ROOT / name) for name in SOURCES},
            'tests': [name for name in SOURCES if name.startswith('tests/')],
            'lint_sources': [name for name in SOURCES if name.endswith('.py')],
            'exact_fixtures': FIXTURES, 'methods': ['em', 'gradient'],
            'model_seed': 937101, 'updates': 3, 'pseudocount': .001,
            'cap_seconds': 600, 'rss_limit_bytes': 4 * 1024**3,
            'output_limit_bytes': 64 * 1024**2,
            'output': str(folder / 'run-01'), 'supervision': str(folder / 'native-01.launch.json'),
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'empirical_inputs': [], 'successor_admitted': False}
    publish(args.plan, plan)
    print({'plan': str(args.plan.resolve()), **descriptor(args.plan)})


if __name__ == '__main__':
    main()
