"""Create a new immutable engineering or study registration, without model calls."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from finite_training_allocation_worker import (
    ROOT,
    admit_qualification,
    closed_producer,
    descriptor,
    files,
    publish,
    require,
    runtime,
)
from register_finite_expected_count_learning import SOURCES as PREVIOUS_SOURCES

SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES,
    'scripts/register_finite_training_allocation.py',
    'scripts/finite_training_allocation_worker.py',
    'scripts/run_finite_training_allocation.py',
    'scripts/audit_finite_training_allocation.py',
    'src/openjev/research/finite_training_allocation.py',
    'tests/test_finite_training_allocation.py',
    'tests/test_finite_training_allocation_runner.py',
    'tests/test_finite_training_allocation_worker.py',
    'tests/test_finite_training_allocation_audit.py',
    'research/finite-training-allocation-protocol.md',
    'research/finite-training-allocation-requirements.txt',
)))
TESTS = [name for name in SOURCES if name.startswith('tests/')]
LINT = [name for name in SOURCES if name.endswith('.py')]
CONFIG = {'seed_namespace': 430260924, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': [430261001, 430261002, 430261003],
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
    folder = ROOT / 'output/finite-training-allocation-v1'
    folder.mkdir(exist_ok=True)
    phases = {name: {'cap_seconds': cap, 'output': str(folder / output),
                     'supervision': str(folder / (prefix + '.launch.json'))}
              for name, cap, output, prefix in (
                  ('qualify', 300, 'engineering-' + engineering_suffix, 'engineering-native-' + engineering_suffix),
                  ('fit', 1200, 'run-01', 'fit-native-01'),
                  ('audit', 600, 'audit-01', 'audit-native-01'))}
    sources = {name: descriptor(ROOT / name) for name in SOURCES}
    kernel = ROOT / 'output/finite-expected-count-qualification-v1'
    registration = kernel / 'registration.json'
    require(descriptor(registration)['sha256'] == '310868f8e0b66607f90440c528dcaa51825d554577e6a17f6ea3c14d3034ad91', 'exact numerical prerequisite registration')
    kernel_plan = json.loads(registration.read_text())
    kernel_receipt = json.loads((kernel / 'run-01.receipt.json').read_text())
    require(kernel_receipt['status'] == 'COMPLETED' and kernel_receipt['engineering_gate'] == 'PASS', 'numerical qualification passed')
    require(kernel_receipt['plan'] == str(registration)
            and kernel_receipt['plan_sha256'] == descriptor(registration)['sha256']
            and kernel_receipt['supervision'] == str(kernel / 'native-01.launch.json')
            and kernel_receipt['output'] == str(kernel / 'run-01')
            and kernel_receipt['launch'] == json.loads((kernel / 'native-01.launch.json').read_text()),
            'original numerical plan, phase and launch joins')
    closed_producer(kernel_receipt)
    require(files(kernel / 'run-01') == kernel_receipt['files'], 'numerical qualification payload identities')
    require({name: sources[name] for name in kernel_plan['sources']} == kernel_plan['sources'] == kernel_receipt['sources_after'], 'unchanged qualified numerical sources')
    plan = {'version': 'finite-training-allocation-v1', 'mode': args.mode,
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
    print(json.dumps({'plan': str(args.plan.resolve()), **descriptor(args.plan)}))


if __name__ == '__main__':
    main()
