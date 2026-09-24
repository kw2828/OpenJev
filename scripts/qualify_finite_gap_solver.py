"""One registered synthetic solver qualification, with no empirical inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from openjev.research.suspend_clock import SuspendClock

VERSION = 'finite-gap-solver-qualification-v1'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')
METHODS = ('slsqp', 'gap_projected')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular evidence file required')
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def files(folder):
    return {p.relative_to(folder).as_posix(): descriptor(p)
            for p in sorted(folder.rglob('*')) if p.is_file()} if folder.exists() else {}


def runtime():
    return {'python': platform.python_version(), 'executable': sys.executable,
            'system': platform.system(), 'machine': platform.machine(),
            'packages': {name: importlib.metadata.version(name)
                         for name in ('numpy', 'scipy', 'pytest', 'ruff')}}


def plain(value):
    if isinstance(value, dict):
        return {key: plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(child) for child in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def qualify(output, plan, check, clock):
    import numpy as np
    from finite_gap_solver_fixtures import check_certificate, fixtures, scalar_oracle

    from openjev.research import finite_convex_readout as original
    from openjev.research import finite_gap_readout as candidate

    require(candidate.MAX_ITERATIONS == 20000 and candidate.CERT_EVERY == 10,
            'registered candidate iteration and certificate schedule')
    start = clock.now_ns()
    cases = fixtures()
    fixture_generation_seconds = (clock.now_ns() - start) / 1e9
    require([case['name'] for case in cases] == plan['fixtures'], 'exact predeclared fixture roster')
    results, case_records = [], []
    for index, case in enumerate(cases):
        check(hard=True)
        name = case['name']
        inputs = [case[key] for key in ('xb', 'xo', 'yb', 'yo')]
        for array in [*inputs, case['initial']]:
            require(array.dtype == np.float64 and bool(np.isfinite(array).all()), 'finite float64 fixture')
        known = case['known_optimum']
        with (output / (name + '.npz')).open('xb') as stream:
            np.savez(stream, **{key: case[key] for key in ('xb', 'xo', 'yb', 'yo', 'initial')},
                     known_optimum=np.empty((0,), dtype=np.float64) if known is None else known)
        start = clock.now_ns()
        initial = scalar_oracle(*inputs, case['initial'])
        optimum = None if known is None else scalar_oracle(*inputs, known)
        if optimum is not None:
            require(optimum['passed'], 'analytically supplied optimum must independently certify')
            expected_loss = 1 / 64 if name == 'zero_support' else 0.
            require(abs(optimum['objective'] - expected_loss) <= 1e-12,
                    'known optimum matches its analytic objective')
        fixture_certificate_seconds = (clock.now_ns() - start) / 1e9
        start = clock.now_ns()
        problem = original.build_problem(*inputs)
        build_seconds = (clock.now_ns() - start) / 1e9
        case_records.append({'name': name, 'description': case['description'],
                             'fixture': descriptor(output / (name + '.npz')),
                             'initial_certificate': initial, 'known_optimum_certificate': optimum,
                             'build_seconds': build_seconds, 'gram_builds': 1,
                             'fixture_certificate_seconds': fixture_certificate_seconds})
        order = METHODS if index % 2 == 0 else METHODS[::-1]
        for method in order:
            check(hard=True)
            start = clock.now_ns()
            result = (original.solve if method == 'slsqp' else candidate.solve)(
                problem, case['initial'], check=check)
            solve_seconds = (clock.now_ns() - start) / 1e9
            # Every original result is retained before any independent check can fail.
            result_path = output / f'{name}--{method}.json'
            publish(result_path, {'fixture': name, 'method': method, 'result': result,
                                  'solve_seconds': solve_seconds})
            start = clock.now_ns()
            final = check_certificate(*inputs, result['probabilities'],
                                      result['certificate'], require_pass=False)
            # The checker may return nothing; always retain the independent scalar calculation.
            if final is None:
                final = scalar_oracle(*inputs, result['probabilities'])
            independent_seconds = (clock.now_ns() - start) / 1e9
            require(abs(result['objective_initial'] - initial['objective']) <= 1e-12,
                    'reported initial objective matches separate scalar arithmetic')
            no_increase = final['objective'] <= initial['objective'] + 1e-12
            known_loss = optimum is None or final['objective'] <= optimum['objective'] + 1e-8 + 1e-12
            admitted = bool(result['complete'] and final['passed'] and no_increase and known_loss)
            row = {'fixture': name, 'method': method, 'execution_position': len(results),
                   'solver_complete': result['complete'], 'failure_reasons': result['failure_reasons'],
                   'independent_certificate': final, 'initial_gap': initial['fw_gap'],
                   'initial_loss': initial['objective'], 'nonincrease': no_increase,
                   'known_optimum_loss_check': known_loss, 'qualifies': admitted,
                   'solve_seconds': solve_seconds, 'independent_seconds': independent_seconds,
                   'work': result['work'], 'solver': result['solver'],
                   'saved_result': {'path': result_path.name, **descriptor(result_path)}}
            results.append(row)
            with (output / 'results.jsonl').open('a') as stream:
                stream.write(json.dumps(plain(row), sort_keys=True, allow_nan=False) + '\n')
            check(hard=True)
    require(len(results) == 36, 'all eighteen fixtures and both methods required')
    counts = {method: sum(row['qualifies'] for row in results if row['method'] == method)
              for method in METHODS}
    return {'version': VERSION, 'engineering_gate': 'PASS' if counts['gap_projected'] == 18 else 'FAIL',
            'qualifying_fixtures': counts, 'fixtures': case_records, 'results': results,
            'fixture_generation_seconds': fixture_generation_seconds,
            'counts': {'fixtures': 18, 'gram_builds': 18, 'solver_calls': 36,
                       'empirical_array_decodes': 0, 'model_calls': 0, 'checkpoint_loads': 0,
                       'development_generations': 0},
            'scope': 'Synthetic numerical qualification only. No empirical comparison or architecture claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output, receipt_path = args.output.resolve(), Path(str(args.output.resolve()) + '.receipt.json')
    require(not output.exists() and not receipt_path.exists(), 'exclusive engineering attempt')
    record = {'version': VERSION, 'status': 'FAILED', 'plan': str(args.plan.resolve()),
              'plan_sha256': args.plan_sha256, 'output': str(output),
              'supervision': str(args.supervision.resolve()), 'started_unix': time.time()}
    clock, checks = None, 0
    try:
        require(descriptor(args.plan)['sha256'] == args.plan_sha256, 'registered plan identity')
        plan = json.loads(args.plan.read_text())
        require(plan['version'] == VERSION and Path.cwd() == ROOT and plan['root'] == str(ROOT),
                'registered version and dedicated working directory')
        require(plan['runtime'] == runtime() and plan['output'] == str(output)
                and plan['supervision'] == str(args.supervision.resolve()), 'registered runtime and paths')
        require(all(os.environ.get(key) == '1' for key in THREADS)
                and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'single-thread and no-bytecode environment')
        before = {name: descriptor(ROOT / name) for name in plan['sources']}
        require(before == plan['sources'], 'all frozen sources before execution')
        record['sources_before'] = before
        for _ in range(100):
            if args.supervision.exists():
                break
            time.sleep(.01)
        launch = json.loads(args.supervision.read_text())
        require(launch['command'] == [sys.executable, *sys.argv] and launch['cwd'] == str(ROOT)
                and launch['pid'] == os.getpid() and launch['pgid'] == os.getpgrp()
                and launch['cap_seconds'] == plan['cap_seconds'], 'original supervised process and bounds')
        require(launch['watchdog_sha256'] == before['scripts/supervise_dialogue_observation_v2.py']['sha256']
                and launch['clock_source_sha256'] == before['src/openjev/research/suspend_clock.py']['sha256'],
                'original pinned clock and supervisor')
        record['launch'] = launch
        clock = SuspendClock()
        require(clock.backend == launch['clock_backend'], 'same suspend-inclusive native clock')

        def check(*, hard=False):
            nonlocal checks
            checks += 1
            require(clock.now_ns() < launch['deadline_ns'], 'registered engineering deadline')
            if hard or checks % 100 == 0:
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                require((peak if sys.platform == 'darwin' else peak * 1024) <= plan['rss_limit_bytes'],
                        'registered worker RSS limit')
            if hard and output.exists():
                paths = list(output.rglob('*'))
                require(len(paths) <= 256 and sum(p.stat().st_size for p in paths if p.is_file())
                        <= plan['output_limit_bytes'], 'engineering output bounds')

        check(hard=True)
        output.mkdir()
        commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *plan['lint_sources']],
                    [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *plan['tests']]]
        record['commands'] = []
        for index, command in enumerate(commands):
            check(hard=True)
            start = clock.now_ns()
            with (output / f'command-{index}.log').open('xb') as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=max(.01, (launch['deadline_ns'] - start) / 1e9), check=False)
            record['commands'].append({'command': command, 'returncode': process.returncode,
                                       'seconds': (clock.now_ns() - start) / 1e9})
            require(process.returncode == 0, 'engineering implementation check failed')
        summary = qualify(output, plan, check, clock)
        publish(output / 'summary.json', summary)
        check(hard=True)
        after = {name: descriptor(ROOT / name) for name in before}
        record['sources_after'] = after
        require(before == after, 'frozen source closure after execution')
        record['status'] = 'COMPLETED'
        record['engineering_gate'] = summary['engineering_gate']
        record['qualifying_fixtures'] = summary['qualifying_fixtures']
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['finished_unix'] = time.time()
        record['checks'] = checks
        record['files'] = files(output)
        if clock is not None and 'launch' in record:
            try:
                record['seconds'] = (clock.now_ns() - record['launch']['started_ns']) / 1e9
            except Exception as error:  # noqa: BLE001 - preserve the original failure receipt
                record['final_clock_error'] = repr(error)
        publish(receipt_path, record)
        print(json.dumps({'status': record['status'], 'engineering_gate': record.get('engineering_gate'),
                          'receipt': str(receipt_path)}), flush=True)


if __name__ == '__main__':
    main()
