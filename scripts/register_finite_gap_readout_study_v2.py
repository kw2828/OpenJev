"""Register a separate frozen-readout study after numerical qualification."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from finite_gap_readout_study_worker_v2 import (
    ROOT,
    closed_producer,
    descriptor,
    files,
    publish,
    require,
    runtime,
)
from publish_finite_convex_readout_stop import authenticate as authenticate_stopped
from render_finite_factorized_dynamics import authenticate

VERSION = 'finite-gap-readout-study-v1'
UPSTREAM = ROOT / 'output/finite-factorized-dynamics-v1'
GAP_FOLDER = ROOT / 'output/finite-gap-solver-qualification-v1'
GAP_SHA = '3513fd540b1a01af6568a4268a8ea7e0a93680ad39ef35ddb206d998cbf255eb'
PREDECESSOR = ROOT / 'output/finite-convex-readout-v1'
PREDECESSOR_SHA = '129af3b025548eda6e475cb2d69bd5a7f3bfb325b085a56dff3ee71af798bab4'
OWN_SOURCES = (
    'scripts/register_finite_gap_readout_study_v2.py', 'scripts/finite_gap_readout_study_worker_v2.py',
    'scripts/run_finite_gap_readout_study.py', 'scripts/audit_finite_gap_readout_study_v2.py',
    'scripts/publish_finite_convex_readout_stop.py', 'scripts/render_finite_factorized_dynamics.py',
    'tests/test_finite_gap_readout_study_runner.py', 'tests/test_finite_gap_readout_study_worker_v2.py',
    'tests/test_finite_gap_readout_study_audit_v2.py',
    'research/finite-gap-readout-study-protocol.md', 'research/finite-gap-readout-study-requirements.txt',
)
TESTS = [name for name in OWN_SOURCES if name.startswith('tests/')]
CONFIG = {'dev_seed_namespace': 428260924, 'dev_attempts': 128, 'dev_horizon': 8,
          'batch_size': 64, 'parent_seeds': [426261001, 426261002, 426261003],
          'parent_arms': ['factorized', 'matched_free', 'dense_free'],
          'train_namespace': 426260924, 'train_horizon': 2,
          'solver_maxiter': 20000, 'solver_certificate_every': 10,
          'solver_monotonicity_roundoff': 1e-15, 'solver_gap_tolerance': 1e-8,
          'solver_nonincrease_tolerance': 1e-12}


def method_qualification():
    path = GAP_FOLDER / 'registration.json'
    require(descriptor(path)['sha256'] == GAP_SHA, 'original synthetic registration')
    plan = json.loads(path.read_text())
    require(plan['version'] == 'finite-gap-solver-qualification-v1'
            and plan['root'] == str(ROOT) and plan['empirical_inputs'] == [], 'synthetic-only prerequisite')
    require({name: descriptor(ROOT / name) for name in plan['sources']} == plan['sources'],
            'qualified synthetic source closure')
    receipt_path = Path(plan['output'] + '.receipt.json')
    receipt = json.loads(receipt_path.read_text())
    terminal = closed_producer(receipt)
    require(receipt['status'] == 'COMPLETED' and receipt['engineering_gate'] == 'PASS'
            and receipt['plan_sha256'] == GAP_SHA and receipt['plan'] == str(path)
            and receipt['output'] == plan['output'] and receipt['supervision'] == plan['supervision']
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and receipt['qualifying_fixtures'] == {'gap_projected': 18, 'slsqp': 10},
            'original all-fixture prerequisite PASS')
    require(files(Path(plan['output'])) == receipt['files'], 'all original synthetic payload bytes')
    summary = json.loads((Path(plan['output']) / 'summary.json').read_text())
    require(summary['engineering_gate'] == 'PASS' and len(summary['results']) == 36
            and summary['qualifying_fixtures'] == receipt['qualifying_fixtures']
            and summary['counts'] == {'fixtures': 18, 'gram_builds': 18, 'solver_calls': 36,
                'empirical_array_decodes': 0, 'model_calls': 0, 'checkpoint_loads': 0,
                'development_generations': 0}, 'complete synthetic suite with no empirical data')
    return plan, {'folder': str(GAP_FOLDER), 'registration': {'path': str(path), **descriptor(path)},
                  'receipt': {'path': str(receipt_path), **descriptor(receipt_path)},
                  'terminal': {'path': str(terminal), **descriptor(terminal)}, 'files': files(GAP_FOLDER)}


def failed_integration():
    folder = ROOT / 'output' / VERSION
    path = folder / 'engineering-registration-01.json'
    expected = '3bb2b18fe3f264b3843454a974c381fdda2646507c75fd32697f0aa8e6acf795'
    require(descriptor(path)['sha256'] == expected, 'original failed engineering registration')
    plan = json.loads(path.read_text())
    require({name: descriptor(ROOT / name) for name in plan['sources']} == plan['sources'],
            'all original engineering sources remain unchanged')
    snapshot = folder / 'engineering-source-01'
    require({name: descriptor(snapshot / name) for name in plan['sources']} == plan['sources'],
            'failed engineering source snapshot matches original pins')
    spec = plan['phases']['qualify']
    receipt_path = Path(spec['output'] + '.receipt.json')
    launch_path = Path(spec['supervision'])
    terminal_path = Path(spec['supervision'].replace('.launch.json', '.terminal.json'))
    log_path = Path(spec['supervision'].replace('.launch.json', '.log'))
    receipt, launch, terminal = (json.loads(p.read_text()) for p in (receipt_path, launch_path, terminal_path))
    require(receipt['status'] == 'FAILED' and receipt['phase'] == 'qualify'
            and receipt['plan_sha256'] == expected and receipt['sources_before'] == plan['sources']
            and receipt['launch'] == launch and all(terminal[k] == v for k, v in launch.items())
            and terminal['status'] == 'failed' and terminal['returncode'] == 1
            and not terminal['timed_out'] and terminal['group_absent'] and terminal['cleanup']['reaped']
            and terminal['cleanup']['errors'] == [] and terminal['error'] is None
            and terminal['timing_available'] and terminal['started_ns'] < terminal['finished_ns'] < terminal['deadline_ns'],
            'preserved original failed test phase and clean terminal')
    require(files(Path(spec['output'])) == receipt['files']
            and [row['returncode'] for row in receipt['commands']] == [0, 1],
            'preserved lint pass and test failure with original logs')
    return plan, {'folder': str(snapshot), 'files': files(snapshot), 'status': 'FAILED',
        'registration': {'path': str(path), **descriptor(path)},
        'receipt': {'path': str(receipt_path), **descriptor(receipt_path)},
        'launch': {'path': str(launch_path), **descriptor(launch_path)},
        'terminal': {'path': str(terminal_path), **descriptor(terminal_path)},
        'log': {'path': str(log_path), **descriptor(log_path)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('engineering', 'study'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'dedicated repo required')
    upstream_path, upstream_plan, phases, _saved, _audit = authenticate(UPSTREAM)
    stopped_path, stopped_plan, _old_q, _old_phases = authenticate_stopped(PREDECESSOR, PREDECESSOR_SHA)
    gap_plan, gap = method_qualification()
    earlier_plan, earlier = failed_integration()
    names = dict.fromkeys((*upstream_plan['sources'], *stopped_plan['sources'], *gap_plan['sources'],
                          *earlier_plan['sources'], *OWN_SOURCES))
    sources = {name: descriptor(ROOT / name) for name in names}
    folder = ROOT / 'output' / VERSION
    folder.mkdir(exist_ok=True)
    specs = {name: {'cap_seconds': cap, 'output': str(folder / output),
                    'supervision': str(folder / (prefix + '.launch.json'))}
             for name, cap, output, prefix in (
                 ('qualify', 300, 'engineering-02', 'engineering-native-02'),
                 ('fit', 600, 'run-01', 'fit-native-01'),
                 ('audit', 600, 'audit-01', 'audit-native-01'))}
    plan = {'version': VERSION, 'mode': args.mode, 'root': str(ROOT), 'runtime': runtime(),
            'sources': sources, 'config': CONFIG, 'phases': specs, 'tests': TESTS,
            'lint_sources': [name for name in sources if name.endswith('.py')],
            'rss_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'upstream': {'folder': str(UPSTREAM), 'registration': {'path': str(upstream_path), **descriptor(upstream_path)},
                         'run': str(phases['fit']['directory']), 'files': files(UPSTREAM)},
            'method_qualification': gap,
            'engineering_predecessor': earlier,
            'predecessor': {'folder': str(PREDECESSOR),
                'registration': {'path': str(stopped_path), **descriptor(stopped_path)},
                'status': 'STOPPED_BEFORE_DEV', 'files': files(PREDECESSOR)}}
    if args.mode == 'study':
        require(args.qualification is not None, 'closed current integration qualification required')
        qualification = json.loads(args.qualification.read_text())
        require(qualification['status'] == 'PASS' and qualification['sources_after'] == sources,
                'same qualified successor source closure')
        terminal = closed_producer(qualification)
        plan['qualification'] = {'path': str(args.qualification.resolve()),
                                 'descriptor': descriptor(args.qualification), 'terminal': descriptor(terminal)}
        require(not Path(specs['fit']['output']).exists(), 'one prospective scientific attempt')
    publish(args.plan, plan)
    print(json.dumps({'plan': str(args.plan.resolve()), **descriptor(args.plan)}))


if __name__ == '__main__':
    main()
