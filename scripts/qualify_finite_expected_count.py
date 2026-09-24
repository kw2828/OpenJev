"""One registered expected-count qualification, with no empirical inputs."""
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

VERSION = 'finite-expected-count-qualification-v1'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')


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
                         for name in ('numpy', 'torch', 'pytest', 'ruff')}}


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
    import torch

    sys.path.insert(0, str(ROOT / 'tests'))
    from test_finite_expected_count_bridge import public_histories
    from test_finite_expected_count_oracle import oracle, parameters

    from openjev.research.finite_expected_count import forward_backward
    from openjev.research.finite_expected_count_bridge import fit_prefix
    from openjev.research.finite_factorized_dynamics_models import make_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    witnesses = []
    t, o, h = parameters()
    for actions, observations in plan['exact_fixtures']:
        check(hard=True)
        a, y = np.array(actions, np.int64), np.array(observations, np.int64)
        actual = forward_backward(t, o, h, a, y, check=check)
        expected = oracle(t, o, h, a, y)
        errors = {'log_likelihood': abs(actual['log_likelihood'] - expected['log_likelihood'])}
        for name in ('scales', 'filtered', 'smoothed', 'transition_posteriors'):
            errors[name] = float(np.max(np.abs(actual[name] - expected[name]), initial=0))
        for name in expected['counts']:
            errors['counts_' + name] = float(np.max(np.abs(actual['counts'][name] - expected['counts'][name])))
        require(all(error <= 2e-12 for error in errors.values()), 'independent exact-path arithmetic')
        witnesses.append({'actions': actions, 'observations': observations,
            'actual': plain(actual), 'expected': {k: plain(v) for k, v in expected.items() if k != 'likelihood'},
            'rational_likelihood': str(expected['likelihood']), 'absolute_errors': errors})
    publish(output / 'exact-witnesses.json', witnesses)
    prefix, lengths = public_histories()
    with (output / 'fabricated-prefixes.npz').open('xb') as stream:
        np.savez(stream, prefix=prefix.numpy(), lengths=lengths.numpy())
    fits = []
    for method in plan['methods']:
        check(hard=True)
        model = make_model('factorized', plan['model_seed'])
        initial = {k: v.detach().numpy().copy() for k, v in model.named_parameters()}
        row = fit_prefix(model, prefix, lengths, method, plan['updates'],
                         pseudocount=plan['pseudocount'], check=check)
        final = {k: v.detach().numpy().copy() for k, v in model.named_parameters()}
        require(np.array_equal(initial['cost_logits'], final['cost_logits']), 'cost-head unchanged')
        with (output / (method + '-states.npz')).open('xb') as stream:
            np.savez(stream, **{'initial__' + k: v for k, v in initial.items()},
                     **{'final__' + k: v for k, v in final.items()})
        publish(output / (method + '-trace.json'), row)
        fits.append(row)
    return {'version': VERSION, 'engineering_gate': 'PASS', 'qualifying_fixtures': len(witnesses),
            'exact_enumeration_states': 2, 'max_actions': 3,
            'methods': fits, 'synthetic_model_seed': plan['model_seed'],
            'empirical_data_loads': 0, 'empirical_checkpoint_loads': 0,
            'scientific_train_generations': 0, 'scientific_dev_generations': 0,
            'synthetic_model_constructions_in_retained_witness': len(fits),
            'test_model_calls_instrumented': False,
            'performance_or_novelty_claim': False, 'scientific_successor_admitted': False}


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
