"""Freeze the separate task-independent readout initialization diagnostic."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import register_finite_head_learning as parent
from finite_head_learning_worker_v2 import (
    ROOT,
    admit_qualification,
    descriptor,
    files,
    publish,
    require,
    runtime,
)

CONFIG = {'seed_namespace': 436260924, 'train_attempts': 512, 'dev_attempts': 512,
          'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': [436261001, 436261002, 436261003, 436261004, 436261005],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'prefix_updates': 1024, 'joint_updates': 3072, 'fit_cap_seconds': 120.}
EXPOSURE = {'config': {**CONFIG, 'seed_namespace': 946201, 'fit_seeds': [946301],
                       'dev_attempts': 8, 'prefix_updates': 32, 'joint_updates': 64,
                       'fit_cap_seconds': 30.},
            'target_counts': {'prefix_updates': 1024, 'joint_updates': 3072},
            'maximum_projected_seconds': 90.0}
SOURCES = tuple(dict.fromkeys((*parent.SOURCES,
    'scripts/register_finite_head_learning_v2.py',
    'scripts/finite_head_learning_worker_v2.py',
    'scripts/qualify_finite_head_exposure_v2.py',
    'tests/test_finite_head_initialization_v2.py',
    'tests/test_finite_head_learning_worker_v2.py',
    'research/finite-head-learning-protocol-v2.md',
)))
TESTS = [name.replace('test_finite_head_initialization.py', 'test_finite_head_initialization_v2.py')
             .replace('test_finite_head_learning_worker.py', 'test_finite_head_learning_worker_v2.py')
         for name in parent.TESTS]
FAILED_PLAN_SHA256 = '73be6395712de76b2fc4891161158d9e16c4d236011b8af432b9ebf36f0f34d1'
FAILED_RECEIPT_SHA256 = 'd5733d7c5f4e7bcfa53b12c5a3ded16adcecb3e8108d4c89b0cbbf4b1e204b9b'
FAILED_TERMINAL_SHA256 = '28249468dd9ba17c0b7bffdae3ae2b8ea69d6106a9ed0ae9015a2fbc431ab4fb'
FAILED_LOG_SHA256 = 'b9482f7fbbf9a324ae5397cfecd88bafc97c616ed4ca776d3959fa6e761875a8'


def phase_specs():
    folder = ROOT / 'output/finite-head-learning-v2'
    return {name: {'cap_seconds': cap, 'output': str(folder / output),
                   'supervision': str(folder / (prefix + '.launch.json'))}
            for name, cap, output, prefix in (
                ('qualify', 300, 'engineering-01', 'engineering-native-01'),
                ('fit', 1200, 'run-01', 'fit-native-01'),
                ('audit', 600, 'audit-01', 'audit-native-01'))}


def authenticate_parent():
    """Authenticate the closed failed qualification and sole test correction."""
    import finite_head_learning_worker as original
    folder = ROOT / 'output/finite-head-learning-v1'
    plan_path = folder / 'engineering-registration-01.json'
    receipt_path = folder / 'engineering-01.receipt.json'
    terminal_path = folder / 'engineering-native-01.terminal.json'
    log_path = folder / 'engineering-01/command-1.log'
    for path, expected in ((plan_path, FAILED_PLAN_SHA256), (receipt_path, FAILED_RECEIPT_SHA256),
                           (terminal_path, FAILED_TERMINAL_SHA256), (log_path, FAILED_LOG_SHA256)):
        require(descriptor(path)['sha256'] == expected, 'exact original failed qualification evidence')
    plan = json.loads(plan_path.read_text())
    original.validate_registered_plan(plan_path, plan)
    require(plan['mode'] == 'engineering' and len(plan['sources']) == 131,
            'original complete engineering registration')
    receipt = json.loads(receipt_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    require(receipt['status'] == 'FAILED' and receipt['phase'] == 'qualify'
            and receipt['plan_sha256'] == FAILED_PLAN_SHA256
            and receipt['error'] == "ValueError('qualification command 1 failed')"
            and receipt['sources_before'] == plan['sources'], 'original qualification-only failure')
    original.validate_launch_binding(plan, receipt)
    require(json.loads(Path(receipt['supervision']).read_text()) == receipt['launch'], 'original persisted launch')
    require(terminal['status'] == 'failed' and terminal['returncode'] == 1
            and terminal['timed_out'] is False and terminal['group_absent'] is True
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True
            and terminal['cleanup']['errors'] == [] and terminal['cleanup']['signals'] == []
            and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['timing_available'] is True
            and terminal['started_ns'] <= terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1e9,
            'original failed process fully closed without timeout or cleanup error')
    require(all(terminal[key] == receipt['launch'][key] for key in
                ('pid', 'pgid', 'command', 'cwd', 'cap_seconds', 'clock_backend',
                 'started_ns', 'deadline_ns', 'watchdog_sha256', 'clock_source_sha256')),
            'failed terminal joins original launch')
    require(files(folder / 'engineering-01') == receipt['files'], 'all failed qualification payloads unchanged')
    commands = receipt['commands']
    require(len(commands) == 2 and [row['command'] for row in commands]
            == original.qualification_commands(plan)[:2]
            and [row['returncode'] for row in commands] == [0, 1], 'lint passed; selected tests failed before exposure')
    for index, row in enumerate(commands):
        require(row['log'] == descriptor(folder / f'engineering-01/command-{index}.log'),
                'original failed command logs')
    log = log_path.read_text()
    require('3 failed, 372 passed, 1 warning' in log
            and log.count('TypeError: SharedFilterModel.blind_rollout() got an unexpected keyword argument') == 3,
            'only the three public oracle-keyword exception assertions failed')
    require(not (folder / 'engineering-01/exposure').exists()
            and not (folder / 'study-registration.json').exists()
            and not (folder / 'run-01').exists() and not (folder / 'audit-01').exists(),
            'no original exposure or scientific phase to reopen')
    before = (ROOT / 'tests/test_finite_head_initialization.py').read_text()
    after = (ROOT / 'tests/test_finite_head_initialization_v2.py').read_text()
    old = "with pytest.raises(ValueError, match='oracle'):"
    new = "with pytest.raises(TypeError, match='oracle_prefix'):"
    require(before.count(old) == 1 and after == before.replace(old, new),
            'one exception assertion changed; no algorithm or numerical criterion changes')
    require(CONFIG == parent.CONFIG and EXPOSURE == parent.EXPOSURE,
            'same unused science settings and original exposure geometry')
    return {'failed_qualification': {'folder': str(folder), 'files': files(folder),
                'registration': {'path': str(plan_path), **descriptor(plan_path)},
                'receipt': descriptor(receipt_path), 'terminal': descriptor(terminal_path),
                'sources': plan['sources'], 'status': 'FAILED', 'tests_passed': 372, 'tests_failed': 3,
                'seconds': terminal['wall_seconds'], 'exposure_started': False, 'science_started': False},
            'head_prerequisite': plan['head_prerequisite'],
            'repair_scope': 'One test assertion expects the actual TypeError for a rejected public keyword. Original model, training, runner, audit and numerical criteria stay byte-identical.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('engineering', 'study'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'dedicated repository')
    folder = ROOT / 'output/finite-head-learning-v2'
    folder.mkdir(exist_ok=True)
    expected_plan = folder / ('engineering-registration-01.json' if args.mode == 'engineering'
                              else 'study-registration.json')
    require(args.plan.resolve() == expected_plan and not expected_plan.exists(), 'exclusive registered plan path')
    phases = phase_specs()
    sources = {name: descriptor(ROOT / name) for name in SOURCES}
    qualified = authenticate_parent()
    plan = {'version': 'finite-head-learning-v2', 'mode': args.mode,
            'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'config': CONFIG, 'exposure_probe': EXPOSURE, 'phases': phases,
            'tests': TESTS, 'lint_sources': [name for name in SOURCES if name.endswith('.py')],
            'rss_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2,
            'head_prerequisite': qualified}
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
