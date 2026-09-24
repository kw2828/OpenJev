"""Freeze the synthetic suite without importing a solver or generating arrays."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from qualify_finite_gap_solver import ROOT, VERSION, descriptor, publish, runtime

SOURCES = (
    'src/openjev/research/finite_convex_readout.py',
    'src/openjev/research/finite_gap_readout.py',
    'src/openjev/research/suspend_clock.py',
    'scripts/supervise_dialogue_observation_v2.py',
    'scripts/finite_gap_solver_fixtures.py',
    'scripts/qualify_finite_gap_solver.py',
    'scripts/register_finite_gap_solver_qualification.py',
    'tests/test_finite_convex_readout.py',
    'tests/test_finite_gap_readout.py',
    'tests/test_finite_gap_solver_fixtures.py',
    'research/finite-gap-solver-qualification-protocol.md',
    'research/finite-gap-solver-qualification-requirements.txt',
)
FIXTURES = [f'{geometry}__{optimum}__{target}'
            for geometry in ('basis', 'correlated95', 'correlated999', 'rank_deficient')
            for optimum in ('interior', 'boundary') for target in ('exact', 'misspecified')]
FIXTURES += ['zero_support', 'small_mass']


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
            'fixtures': FIXTURES, 'methods': ['slsqp', 'gap_projected'],
            'cap_seconds': 600, 'rss_limit_bytes': 4 * 1024**3,
            'output_limit_bytes': 64 * 1024**2,
            'output': str(folder / 'run-01'), 'supervision': str(folder / 'native-01.launch.json'),
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'empirical_inputs': [], 'successor_admitted': False}
    publish(args.plan, plan)
    print({'plan': str(args.plan.resolve()), **descriptor(args.plan)})


if __name__ == '__main__':
    main()
