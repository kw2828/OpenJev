"""One frozen retrospective decomposition, with separate original phase closures."""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from finite_head_learning_worker_v2 import (
    ROOT,
    THREADS,
    SuspendClock,
    closed_producer,
    descriptor,
    files,
    publish,
    require,
    runtime,
)

VERSION = 'finite-decision-error-v1'
FOLDER = ROOT / 'output' / VERSION
PLAN = FOLDER / 'registration.json'
PARENT = ROOT / 'research/finite-head-learning-results'
PARENT_STUDY = ROOT / 'output/finite-head-learning-v2'
PARENT_RECEIPT_SHA = 'f1ba0d9fa6af6adbe21f93273cdc12ec154edff301e213e8bf4118b5e5211ea8'
PUBLISHER_SHA = 'ec28f58620d6a57ff9a57542fbe2843ed3511b9405b3a91102108830d29e3eb6'
ARMS = ('rounded_anchor', 'rounded_random', 'matched_free_random')
SEEDS = tuple(range(436261001, 436261006))
OWN = (
    'scripts/run_finite_decision_error.py',
    'scripts/finite_decision_error_diagnostic.py',
    'scripts/audit_finite_decision_error.py',
    'tests/test_finite_decision_error_diagnostic.py',
    'tests/test_finite_decision_error_audit.py',
    'research/finite-decision-error-protocol.md',
)
TESTS = [name for name in OWN if name.startswith('tests/')]
CONFIG = {'arms': list(ARMS), 'seeds': list(SEEDS), 'horizons': [1, 2, 4, 8],
          'cases': 486, 'margin_edges': [0, .001, .01, .1],
          'fit_groups': 60, 'contrasts': 20, 'model_selection': False,
          'retrospective': True, 'new_experiment_admission': False}
RSS_LIMIT = 3 * 1024**3
OUTPUT_LIMIT = 128 * 1024**2


def read(path):
    return json.loads(Path(path).read_text())


def parent():
    """Verify published evidence and original closure before decoding any NPZ."""
    require(descriptor(PARENT / 'receipt.json')['sha256'] == PARENT_RECEIPT_SHA,
            'exact parent publication receipt')
    receipt = read(PARENT / 'receipt.json')
    require(receipt['status'] == 'PASS' and receipt['inputs_unchanged'] is True,
            'completed original parent publication')
    for name, pin in receipt['files'].items():
        require(descriptor(PARENT / name) == pin, 'unchanged published parent file')
    for name, pin in receipt['sources'].items():
        require(descriptor(ROOT / name) == pin, 'unchanged parent source before imports')
    require(descriptor(ROOT / 'scripts/publish_finite_head_learning.py')['sha256'] == PUBLISHER_SHA,
            'reviewed parent authenticator unchanged')
    from publish_finite_head_learning import authenticate
    auth = authenticate()
    require(auth['audit']['advance']['passed'] is False
            and sum(auth['audit']['advance']['conditions'].values()) == 14,
            'parent failed continuation remains failed')
    require(auth['study_files'] == receipt['study_files'], 'complete parent payload inventory')
    return {'receipt': descriptor(PARENT / 'receipt.json'),
            'summary': descriptor(PARENT / 'summary.json'),
            'payloads': {name: receipt['study_files']['run-01/' + name]
                         for name in ('base.npz', 'predictions-base.npz')},
            'sources': receipt['sources'], 'status': 'HEAD_INDEPENDENT_ADVANCE_FAIL'}


def specs():
    return {phase: {'cap_seconds': 120, 'output': str(FOLDER / (phase + '-01')),
                    'supervision': str(FOLDER / (phase + '-native-01.launch.json'))}
            for phase in ('qualify', 'diagnose', 'audit')}


def register():
    require(Path.cwd() == ROOT and not FOLDER.exists(), 'exclusive retrospective registration')
    provenance = parent()
    names = sorted(set(provenance['sources']) | set(OWN) | {'scripts/publish_finite_head_learning.py'})
    sources = {name: descriptor(ROOT / name) for name in names}
    plan = {'version': VERSION, 'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'parent': provenance, 'phases': specs(), 'tests': TESTS,
            'config': CONFIG, 'rss_limit_bytes': RSS_LIMIT, 'output_limit_bytes': OUTPUT_LIMIT,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    FOLDER.mkdir()
    publish(PLAN, plan)
    snapshot = FOLDER / 'source-snapshot'
    snapshot.mkdir()
    for name, pin in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
        require(descriptor(target) == pin, 'snapshot bytes')
    publish(snapshot / 'manifest.json', {'registration': descriptor(PLAN), 'sources': sources})
    print(json.dumps({'plan': str(PLAN), **descriptor(PLAN), 'sources': len(sources)}))


def validate(plan, sha):
    require(descriptor(PLAN)['sha256'] == sha and plan['version'] == VERSION
            and plan['root'] == str(ROOT) and Path.cwd() == ROOT
            and plan['runtime'] == runtime() and plan['phases'] == specs()
            and plan['tests'] == TESTS and plan['config'] == CONFIG
            and plan['rss_limit_bytes'] == RSS_LIMIT and plan['output_limit_bytes'] == OUTPUT_LIMIT,
            'exact frozen plan and runtime')
    provenance = parent()
    names = sorted(set(provenance['sources']) | set(OWN) | {'scripts/publish_finite_head_learning.py'})
    require(sorted(plan['sources']) == names and len(names) == 144, 'complete exact source closure')
    require({name: descriptor(ROOT / name) for name in plan['sources']} == plan['sources'],
            'all frozen sources unchanged')
    snapshot = FOLDER / 'source-snapshot'
    manifest = {'registration': descriptor(PLAN), 'sources': plan['sources']}
    require(read(snapshot / 'manifest.json') == manifest
            and files(snapshot) == {**plan['sources'], 'manifest.json': descriptor(snapshot / 'manifest.json')},
            'complete original source snapshot')
    require(provenance == plan['parent'], 'original parent evidence unchanged')


def command(plan, sha, phase):
    spec = plan['phases'][phase]
    return [plan['runtime']['executable'], str(ROOT / 'scripts/run_finite_decision_error.py'),
            '--phase', phase, '--plan-sha256', sha, '--supervision', spec['supervision']]


def validate_launch(plan, sha, phase, launch):
    spec = plan['phases'][phase]
    require(launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['command'] == command(plan, sha, phase) and launch['cwd'] == str(ROOT)
            and launch['cap_seconds'] == spec['cap_seconds']
            and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['deadline_ns'] - launch['started_ns'] == spec['cap_seconds'] * 10**9
            and launch['pid'] == launch['pgid'] and launch['pid'] != launch['parent_pid'],
            'exact original launch command and bounds')
    for key, name in (('watchdog_sha256', 'scripts/supervise_dialogue_observation_v2.py'),
                      ('clock_source_sha256', 'src/openjev/research/suspend_clock.py')):
        require(launch[key] == plan['sources'][name]['sha256'], 'original supervisor source')


def closed(plan, sha, phase):
    spec = plan['phases'][phase]
    path = Path(spec['output'] + '.receipt.json')
    receipt = read(path)
    require(receipt['status'] == 'PASS' and receipt['phase'] == phase
            and receipt['plan_sha256'] == sha and receipt['plan'] == str(PLAN)
            and receipt['supervision'] == spec['supervision'] and receipt['output'] == spec['output']
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and files(spec['output']) == receipt['files'], 'complete original phase receipt')
    require(read(spec['supervision']) == receipt['launch'], 'persisted original launch')
    validate_launch(plan, sha, phase, receipt['launch'])
    if phase == 'qualify':
        commands = receipt['commands']
        require(len(commands) == 2 and [row['command'] for row in commands] == qualification_commands()
                and all(row['returncode'] == 0 and row['log'] == descriptor(Path(spec['output']) / f'command-{index}.log')
                        for index, row in enumerate(commands)), 'exact passing qualification commands and logs')
    require(receipt['counts'] == {'npz_opens': 0 if phase == 'qualify' else 2,
                                 'array_decodes': 0 if phase == 'qualify' else 17,
                                 'model_calls': 0, 'optimizer_calls': 0, 'generator_calls': 0},
            'closed original input accounting')
    terminal = closed_producer(receipt)
    return {'receipt': descriptor(path), 'terminal': descriptor(terminal),
            'finished_ns': read(terminal)['finished_ns'], 'clock_backend': read(terminal)['clock_backend']}


def qualification_commands():
    return [[str(ROOT / '.venv/bin/ruff'), 'check', *[n for n in OWN if n.endswith('.py')]],
            [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--noconftest', *TESTS]]


def payloads(plan, counts):
    """Each numerical phase opens two archives and materializes exactly 17 arrays."""
    import numpy as np
    folder = PARENT_STUDY / 'run-01'
    for name, pin in plan['parent']['payloads'].items():
        require(descriptor(folder / name) == pin, 'exact original array archive')
    counts['npz_opens'] += 1
    with np.load(folder / 'base.npz', allow_pickle=False) as data:
        truth, ids = data['blind_costs'], data['case_ids']
        counts['array_decodes'] += 2
    require(truth.shape == (486, 8, 4) and truth.dtype == np.float64
            and ids.shape == (486,) and ids.dtype.kind == 'U'
            and len(set(ids.tolist())) == 486, 'original complete retained population')
    counts['npz_opens'] += 1
    predictions = {}
    with np.load(folder / 'predictions-base.npz', allow_pickle=False) as data:
        for arm in ARMS:
            for seed in SEEDS:
                value = data[f'{arm}__{seed}__blind_costs']
                counts['array_decodes'] += 1
                require(value.shape == truth.shape and value.dtype == np.float64, 'full original prediction field')
                predictions[arm, seed] = value
    return truth, predictions, ids.tolist()


def raw_records(records, truth, predictions, case_ids, check):
    """Independent raw-vector join before the separate scalar auditor."""
    require(len(records) == 60 * len(case_ids), 'every source case across all model/horizon groups')
    seen = set()
    for row in records:
        check()
        arm, seed, horizon, index = row['arm'], row['seed'], row['horizon'], row['case_index']
        require(arm in ARMS and seed in SEEDS and horizon in (1, 2, 4, 8)
                and type(index) is int and 0 <= index < len(case_ids), 'declared vector identity')
        key = arm, seed, horizon, index
        require(key not in seen and row['case_id'] == case_ids[index], 'unique exact original case ID')
        seen.add(key)
        require(row['true_costs'] == truth[index, horizon - 1].tolist()
                and row['predicted_costs'] == predictions[arm, seed][index, horizon - 1].tolist(),
                'saved JSON vectors exactly equal original source values')
    return len(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('register', 'qualify', 'diagnose', 'audit'), required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--supervision', type=Path)
    args = parser.parse_args()
    if args.phase == 'register':
        register()
        return
    plan = read(PLAN)
    spec = plan['phases'][args.phase]
    output = Path(spec['output'])
    receipt_path = Path(str(output) + '.receipt.json')
    require(not output.exists() and not receipt_path.exists(), 'exclusive attempt paths')
    receipt = {'version': VERSION, 'phase': args.phase, 'status': 'FAILED',
               'plan': str(PLAN), 'plan_sha256': args.plan_sha256,
               'supervision': str(args.supervision.resolve()), 'output': str(output)}
    counts = dict.fromkeys(('npz_opens', 'array_decodes', 'model_calls', 'optimizer_calls', 'generator_calls'), 0)
    try:
        validate(plan, args.plan_sha256)
        require(all(os.environ.get(key) == '1' for key in THREADS)
                and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
                and os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
                and os.environ.get('PYTEST_ADDOPTS') == os.environ.get('PYTEST_PLUGINS') == '', 'fixed environment')
        require(str(args.supervision.resolve()) == spec['supervision'], 'exact supervisor path')
        receipt['sources_before'] = plan['sources']
        for _ in range(100):
            if args.supervision.exists():
                break
            time.sleep(.01)
        launch = read(args.supervision)
        validate_launch(plan, args.plan_sha256, args.phase, launch)
        require(launch['pid'] == os.getpid() and launch['pgid'] == os.getpgrp(), 'this original child process')
        receipt['launch'] = launch
        clock = SuspendClock()
        require(clock.backend == launch['clock_backend'], 'original native clock')
        checks = 0

        def check():
            nonlocal checks
            checks += 1
            require(clock.now_ns() < launch['deadline_ns'], 'phase deadline')
            if checks == 1 or checks % 256 == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                require((rss if sys.platform == 'darwin' else rss * 1024) <= plan['rss_limit_bytes'], 'sampled worker RSS')
                require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= plan['output_limit_bytes'],
                        'sampled output limit')

        check()
        for prior in (() if args.phase == 'qualify' else ('qualify',) if args.phase == 'diagnose' else ('qualify', 'diagnose')):
            bound = closed(plan, args.plan_sha256, prior)
            require(bound['clock_backend'] == launch['clock_backend']
                    and bound['finished_ns'] <= launch['started_ns'], 'prior phase originally closed before launch')
            receipt[prior + '_closure'] = bound
        output.mkdir()
        if args.phase == 'qualify':
            commands = qualification_commands()
            receipt['commands'] = []
            for index, call in enumerate(commands):
                check()
                with (output / f'command-{index}.log').open('xb') as log:
                    result = subprocess.run(call, stdout=log, stderr=subprocess.STDOUT, check=False,
                                            timeout=max(.01, (launch['deadline_ns'] - clock.now_ns()) / 1e9))
                receipt['commands'].append({'command': call, 'returncode': result.returncode,
                                            'log': descriptor(output / f'command-{index}.log')})
                require(result.returncode == 0, 'qualification command failed')
        elif args.phase == 'diagnose':
            from finite_decision_error_diagnostic import analyze
            truth, predictions, case_ids = payloads(plan, counts)
            result = analyze(truth, predictions, case_ids, check=check)
            require(len(result['records']) == 29160, 'complete all-case result')
            publish(output / 'summary.json', result['summary'])
            with (output / 'cases.jsonl').open('x') as stream:
                for row in result['records']:
                    check()
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            receipt['result'] = {'case_records': len(result['records']), 'cases': len(case_ids)}
        else:
            from audit_finite_decision_error import audit
            source = Path(plan['phases']['diagnose']['output'])
            summary = read(source / 'summary.json')
            records = [json.loads(line) for line in (source / 'cases.jsonl').read_text().splitlines()]
            truth, predictions, case_ids = payloads(plan, counts)
            joins = raw_records(records, truth, predictions, case_ids, check)
            audited = audit(records, summary, read(PARENT / 'summary.json')['rows'], expected_case_ids=case_ids, check=check)
            publish(output / 'audit.json', audited)
            receipt['result'] = {'raw_vector_joins': joins, 'agreement': True}
            require(closed(plan, args.plan_sha256, 'diagnose') == receipt['diagnose_closure'], 'producer unchanged')
        check()
        require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= plan['output_limit_bytes'], 'final output limit')
        require(counts == {'npz_opens': 0 if args.phase == 'qualify' else 2,
                           'array_decodes': 0 if args.phase == 'qualify' else 17,
                           'model_calls': 0, 'optimizer_calls': 0, 'generator_calls': 0},
                'exact materialized scientific input accounting')
        validate(plan, args.plan_sha256)
        receipt['sources_after'] = plan['sources']
        receipt['status'] = 'PASS'
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['counts'] = counts
        receipt['files'] = files(output)
        publish(receipt_path, receipt)
        print(json.dumps({'phase': args.phase, 'status': receipt['status'], 'receipt': str(receipt_path)}), flush=True)


if __name__ == '__main__':
    main()
