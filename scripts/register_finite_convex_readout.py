"""Pin one bounded convex-head diagnostic without model calls or array decoding."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from finite_convex_readout_worker import ROOT, closed_producer, descriptor, files, publish, require, runtime
from render_finite_factorized_dynamics import authenticate

VERSION = 'finite-convex-readout-v1'
UPSTREAM = ROOT / 'output/finite-factorized-dynamics-v1'
OWN_SOURCES = (
    'scripts/register_finite_convex_readout.py', 'scripts/finite_convex_readout_worker.py',
    'scripts/run_finite_convex_readout.py', 'scripts/audit_finite_convex_readout.py',
    'scripts/render_finite_factorized_dynamics.py', 'src/openjev/research/finite_convex_readout.py',
    'tests/test_finite_convex_readout.py', 'tests/test_finite_convex_readout_runner.py',
    'tests/test_finite_convex_readout_worker.py', 'tests/test_finite_convex_readout_audit.py',
    'research/finite-convex-readout-protocol.md', 'research/finite-convex-readout-requirements.txt',
)
TESTS = [p for p in OWN_SOURCES if p.startswith('tests/')]
CONFIG = {'dev_seed_namespace': 427260924, 'dev_attempts': 128, 'dev_horizon': 8,
          'batch_size': 64, 'parent_seeds': [426261001, 426261002, 426261003],
          'parent_arms': ['factorized', 'matched_free', 'dense_free'],
          'train_namespace': 426260924, 'train_horizon': 2,
          'solver_maxiter': 2000, 'solver_ftol': 1e-12,
          'solver_max_objective_calls': 10000, 'solver_gap_tolerance': 1e-8,
          'solver_repair_tolerance': 1e-10, 'solver_nonincrease_tolerance': 1e-12}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('engineering', 'study'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--engineering-attempt', type=int, default=1)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.engineering_attempt > 0, 'dedicated repo and positive attempt')
    upstream_path, upstream_plan, phases, _saved, _audit = authenticate(UPSTREAM)
    sources = {name: descriptor(ROOT / name) for name in (*upstream_plan['sources'], *OWN_SOURCES)}
    folder = ROOT / 'output' / VERSION
    folder.mkdir(exist_ok=True)
    suffix = f'{args.engineering_attempt:02d}'
    phase_specs = {name: {'cap_seconds': cap, 'output': str(folder / output),
                         'supervision': str(folder / (prefix + '.launch.json'))}
                   for name, cap, output, prefix in (
                     ('qualify', 300, 'engineering-' + suffix, 'engineering-native-' + suffix),
                     ('fit', 600, 'run-01', 'fit-native-01'),
                     ('audit', 600, 'audit-01', 'audit-native-01'))}
    plan = {'version': VERSION, 'mode': args.mode, 'root': str(ROOT), 'runtime': runtime(),
            'sources': sources, 'config': CONFIG, 'phases': phase_specs, 'tests': TESTS,
            'lint_sources': [p for p in sources if p.endswith('.py')],
            'rss_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'upstream': {'folder': str(UPSTREAM), 'registration': {'path': str(upstream_path), **descriptor(upstream_path)},
                         'run': str(phases['fit']['directory']), 'files': files(UPSTREAM)}}
    if args.mode == 'study':
        require(args.qualification is not None, 'closed engineering required')
        qualification = json.loads(args.qualification.read_text())
        require(qualification['status'] == 'PASS' and qualification['sources_after'] == sources,
                'same qualified source closure')
        terminal = closed_producer(qualification)
        plan['qualification'] = {'path': str(args.qualification.resolve()),
                                 'descriptor': descriptor(args.qualification), 'terminal': descriptor(terminal)}
        require(not Path(phase_specs['fit']['output']).exists(), 'one scientific attempt')
    publish(args.plan, plan)
    print(json.dumps({'plan': str(args.plan.resolve()), **descriptor(args.plan)}))


if __name__ == '__main__':
    main()
