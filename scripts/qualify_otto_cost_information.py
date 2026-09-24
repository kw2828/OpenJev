"""Bounded fabricated qualification and metadata-only native runtime preflight."""
from __future__ import annotations

import argparse
import importlib.abc
import json
import os
import resource
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import otto_cost_information_common as c

SELF = 'scripts/qualify_otto_cost_information.py'
SOURCE_REVIEW = 'output/otto-cost-information-v1/source-review-01.json'
SOURCE_REVIEW_PIN = 'f1a5c24f9b889b34d3f038dc27cfc797433fb5be17b79cf9d418d385e7cd931f'
SEED_REVIEW = 'output/otto-cost-information-v1/seed-review-01.json'
SEED_REVIEW_PIN = '1958eafa144c8521d6874d8f7ca14396b1d7ebb7502e8e110e6b7d699eb63fa4'
FORBIDDEN = frozenset(('numpy', 'scipy', 'torch', 'tensorflow', 'keras', 'tf_keras', 'jax', 'mlx'))
ZERO_COUNTS = dict.fromkeys(('empirical_array_decodes', 'checkpoint_decodes', 'native_calls',
                            'model_calls', 'teacher_calls', 'optimizer_calls', 'random_draws'), 0)


@contextmanager
def metadata_only():
    c.require(not FORBIDDEN.intersection(sys.modules), 'no numerical runtime in metadata preflight')

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in FORBIDDEN:
                raise ImportError('metadata-only preflight rejected ' + fullname)

    guard = Guard()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)
    c.require(not FORBIDDEN.intersection(sys.modules), 'metadata preflight imported no numerical runtime')


def source_evidence():
    c.require(c.desc(SOURCE_REVIEW)['sha256'] == SOURCE_REVIEW_PIN, 'primary source review pin')
    evidence = c.read(SOURCE_REVIEW)
    c.require(evidence['version'] == 'otto-cost-information-source-review-v1'
              and evidence['numpy_version'] == '2.5.3' and evidence['law']['uniform_bits'] == 53
              and set(evidence['files']) == {'conversion', 'shipped_test', 'package_metadata'}
              and not any(evidence['counts'].values()), 'source-only conversion evidence')
    inherited = evidence['inherited_source_review']
    c.require(c.desc(inherited['path']) == {k: inherited[k] for k in ('sha256', 'bytes')}
              and c.read(inherited['path'])['files'] == evidence['files'], 'unchanged inherited conversion source review')
    c.require(c.desc(SEED_REVIEW)['sha256'] == SEED_REVIEW_PIN, 'fresh seed review pin')
    seeds = c.read(SEED_REVIEW)
    c.require(seeds['passed'] is True and seeds['proposed_unique_seeds'] == 193
              and not seeds['exact_seed_hit_count'] and not seeds['block_hits']
              and not seeds['changed_files_during_scan'] and seeds['admits_execution'] is False,
              'prospective scoped seed review')
    for record in evidence['files'].values():
        c.require(c.desc(record['path']) == {k: record[k] for k in ('sha256', 'bytes')}, 'installed source pin')
    return evidence


def native_preflight(output):
    c.require(Path.cwd() == c.ROOT and Path(sys.executable).absolute() == c.NATIVE,
              'original native interpreter and working directory')
    with metadata_only():
        evidence = source_evidence()
        prior, paths, _, _ = c.original_native()
        parent_reference = c.parent_reference()
        c.require(prior['runtime']['python_executable'] == str(c.NATIVE)
                  and prior['runtime']['versions']['numpy'] == evidence['numpy_version'], 'native runtime source version')
        c.write(output, {'status': 'passed', 'native_runtime': prior['runtime'],
            'native_sources': prior['sources'], 'native_inputs': prior['native_inputs'],
            'inherited_input_paths': {k: str(v) for k, v in paths.items()},
            'source_review': c.desc(SOURCE_REVIEW), 'installed_sources': evidence['files'],
            'parent_reference': parent_reference,
            'metadata_import_guard': True, 'counts': ZERO_COUNTS})


def resource_check():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    c.require(rss <= 4 * 1024**3, 'fabricated capacity 4GiB RSS cap')
    return rss


def capacity(path):
    """Full-roster synthetic analysis and a historical teacher planning estimate."""
    import numpy as np
    import torch

    sys.path.insert(0, str(c.ROOT / 'src'))
    import audit_otto_cost_information as audit
    import run_otto_cost_information as runner

    from openjev.research import otto_conditional_cost_tree as tree
    from openjev.research.otto_action_latent_model import make_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    n, m = 2 * c.CONFIG['attempted_per_regime'], c.CONFIG['mc_draws']
    root = np.zeros(2809, np.float64)
    root[0] = 1.
    positions = np.array([25 * 53 + 26, 26 * 53 + 26] * 4, np.int64)
    laws = np.full((8, 2809, 4), .25, np.float64)
    for h, position in enumerate(positions):
        laws[h, position] = 0.
    callback_counts = {'legacy_updates': 0, 'fake_teacher_calls': 0}

    def update(state, _position, odor):
        callback_counts['legacy_updates'] += 1
        # An explicitly fabricated state encodes its complete odor history.
        state[1] = 4 * state[1] + odor
        state[2] += 1
        return state

    def score(state, _position):
        callback_counts['fake_teacher_calls'] += 1
        code = int(state[1])
        return np.array([1. + ((code + a) % 7) for a in range(4)], np.float32)

    tick = time.perf_counter()
    exact = tree.exact_h4(root, root, laws[:4], positions[:4], update, score, check=resource_check)
    streams, integer_draws = [], []
    for stream in range(2):
        draws = (np.random.PCG64(712 + stream).random_raw(m * 9) >> np.uint64(11)).reshape(m, 9)
        integer_draws.append(draws)
        streams.append(tree.sample_h8(root, root, laws, positions, draws, exact,
                                     update, score, check=resource_check))
    tree_seconds = time.perf_counter() - tick
    c.require(callback_counts == {'legacy_updates': 340 + 2 * m * 8,
                                  'fake_teacher_calls': 256 + 2 * m}, 'maximum fabricated tree geometry')
    rng = np.random.Generator(np.random.PCG64(715))

    def repeat(value):
        return np.repeat(value[None], n, axis=0)

    data = {'prefix': rng.normal(size=(n, 9, 31)).astype(np.float32),
        'prefix_lengths': np.full(n, 9, np.int64),
        'prefix_actions': np.tile(np.array([0, 1] * 4, np.int64), (n, 1)),
        'prefix_outcomes': np.zeros((n, 8), np.int64),
        'prefix_position': np.full((n, 2), 26, np.int64),
        'actions': np.tile(np.array([0, 1] * 4, np.int64), (n, 1)),
        **{name: repeat(root) for name in ('initial_belief', 'root_strict', 'root_grid', 'legacy_root')},
        'exact_weights': repeat(exact['weights']), 'exact_costs': repeat(exact['costs']),
        'exact_alive': repeat(exact['supported']), 'mc_draws': repeat(np.stack(integer_draws)),
        'mc_source_indices': repeat(np.stack([r['source_indices'] for r in streams])),
        'mc_outcomes': repeat(np.stack([r['outcomes'] for r in streams])),
        **{f'mc_{name}': repeat(np.stack([r[name] for r in streams]))
           for name in ('alive4', 'alive8', 'costs4', 'costs8')},
        'case_ids': np.array([f'fabricated:{i}' for i in range(n)]),
        'regimes': np.array(['lambda3'] * (n // 2) + ['lambda4'] * (n // 2))}
    runner.validate_data(data, np)
    predictions, inference_seconds = {}, []
    for family in c.FAMILIES:
        for seed in c.FIT_SEEDS:
            resource_check()
            tick = time.perf_counter()
            model = make_model(runner.MODEL_KINDS[family], seed)
            model.eval()
            rows = []
            for i in range(n):
                resource_check()
                batch = [torch.from_numpy(data[k][i:i + 1].copy()) for k in runner.INPUT_KEYS]
                with torch.no_grad():
                    result = model.blind_rollout(*batch)
                rows.append(result['cost_contrasts'].numpy())
            predictions[f'{family}__{seed}__cost'] = np.concatenate(rows)
            inference_seconds.append({'family': family, 'seed': seed, 'seconds': time.perf_counter() - tick})
    tick = time.perf_counter()
    report = runner.analyze(data, predictions, np, check=resource_check)
    analysis_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    independent = audit.analyze(data, predictions, np, check=resource_check)
    audit_seconds = time.perf_counter() - tick
    audit.compare_report(report, independent)
    c.require(report['gate']['status'] == independent['gate']['status'], 'fabricated independent headroom status')
    c.require(len(report['rows']) == len(independent['rows']) == 48, 'all twelve policies and horizons/settings')
    for actual, expected in zip(report['rows'], independent['rows'], strict=True):
        for key in ('family', 'fit_seed', 'regime', 'horizon', 'cases', 'measure'):
            c.require(actual[key] == expected[key], 'fabricated independent row identity')
        for key in ('total_regret', 'information_advantage', 'approximation_regret'):
            c.require(abs(actual[key] - expected[key]) <= 1e-10 * (1 + abs(expected[key])),
                      'fabricated independent row arithmetic')
    for actual, expected in zip(report['gate']['groups'], independent['gate']['groups'], strict=True):
        c.require(actual['conditions'] == expected['conditions']
                  and actual['bootstrap_index_sha256'] == expected['bootstrap_index_sha256'],
                  'fabricated independent bootstrap and conditions')
        c.require(np.allclose(actual['bootstrap_replicates'], expected['bootstrap_replicates'], rtol=1e-10, atol=1e-11),
                  'fabricated independent bootstrap arithmetic')
    inference_total = sum(row['seconds'] for row in inference_seconds)
    tree_full_roster_seconds = n * tree_seconds
    max_teacher_calls = n * (256 + 2 * m)
    teacher_seconds_per_call = .034
    collection_projection = max_teacher_calls * teacher_seconds_per_call + 2 * tree_full_roster_seconds + 180
    prediction_projection = 2 * (inference_total + analysis_seconds) + 180
    audit_projection = 2 * (audit_seconds + tree_full_roster_seconds) + 180
    passed = (collection_projection <= c.CAPS['collect']
              and prediction_projection <= c.CAPS['predict']
              and audit_projection <= c.CAPS['audit'])
    output_bytes = len(json.dumps(report, sort_keys=True, allow_nan=False).encode())
    c.write(path, {'status': 'passed' if passed else 'failed', 'fabricated_cases': n,
        'fabricated_models': len(inference_seconds), 'fabricated_blind_rollouts': n * len(inference_seconds),
        'training_updates': 0, 'inference_seconds': inference_seconds,
        'analysis_seconds': analysis_seconds, 'independent_analysis_seconds': audit_seconds,
        'bootstrap_replicates_per_analysis': 2 * c.CONFIG['bootstrap_replicates'],
        'tree_seconds_one_full_size_case': tree_seconds, 'tree_projected_full_roster_seconds': tree_full_roster_seconds,
        'tree_callback_counts_one_case': callback_counts, 'tree_work_exact': exact['work'],
        'tree_work_streams': [row['work'] for row in streams],
        'collection_projection_seconds': collection_projection, 'prediction_projection_seconds': prediction_projection,
        'audit_projection_seconds': audit_projection, 'cap_seconds': c.CAPS,
        'teacher_max_calls': max_teacher_calls, 'historical_planning_teacher_seconds_per_call': teacher_seconds_per_call,
        'fixed_overhead_seconds_per_phase': 180, 'measured_overhead_projection_factor': 2,
        'serialized_producer_report_bytes': output_bytes, 'peak_rss_bytes': resource_check(),
        'empirical_decodes': 0, 'checkpoint_decodes': 0, 'native_calls': 0, 'teacher_calls': 0,
        'scope': 'Synthetic arrays, fake legacy/filter callbacks, random-weight frozen policies only. '
                 'No empirical data, parent checkpoint decode, real teacher or native call.',
        'collection_projection_scope': 'Historical .034 seconds per teacher call is a declared planning estimate, '
                 'not a new teacher benchmark or a guaranteed runtime bound.',
        'audit_projection_scope': 'Measured independent report/bootstrap plus twice projected full-size tree overhead; '
                 'native provenance-law reconstruction is represented by that overhead allowance, not measured directly.'})
    c.require(passed, 'prospective three-phase capacity gate')


def qualify(output):
    c.require(Path.cwd() == c.ROOT and output.is_absolute() and output.is_relative_to(c.OUT), 'contained qualification output')
    c.require(not output.exists(), 'fresh qualification directory')
    evidence = source_evidence()
    names = sorted(set(c.SOURCES) | {SOURCE_REVIEW, SEED_REVIEW})
    before = {k: c.desc(k) for k in names}
    output.mkdir()
    tests = sorted(k for k in c.SOURCES if k.startswith('tests/'))
    lint = sorted(k for k in c.SOURCES if k.endswith('.py') and not k.endswith(('suspend_clock.py', 'supervise_dialogue_observation_v2.py')))
    commands = [(180, [str(c.NUMERICAL), '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp', str(output / 'pytest-temp'), *tests]),
        (60, [str(c.ROOT / '.venv/bin/ruff'), 'check', '--no-cache', *lint]),
        (240, [str(c.NUMERICAL), str(c.ROOT / SELF), '--capacity', '--output', str(output / 'capacity.json')]),
        (60, [str(c.NATIVE), str(c.ROOT / SELF), '--native-preflight', '--output', str(output / 'native-preflight.json')])]
    c.write(output / 'started.json', {'sources': before, 'commands': commands, 'installed_sources': evidence['files'],
                                    'scope': 'fabricated qualification and native metadata only'})
    outcomes = []
    env = {**os.environ, **dict.fromkeys((*c.THREADS, 'PYTHONDONTWRITEBYTECODE', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD'), '1')}
    for i, (cap, command) in enumerate(commands):
        tick, timeout = time.monotonic(), False
        log = output / f'command-{i + 1}.log'
        with log.open('xb') as stream:
            process = subprocess.Popen(command, cwd=c.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
        try:
            os.killpg(process.pid, 0)
            absent = False
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            absent = True
        outcomes.append({'command': command, 'cap_seconds': cap, 'returncode': code, 'timed_out': timeout,
            'reaped': process.poll() is not None, 'group_absent': absent, 'seconds': time.monotonic() - tick,
            'log': log.name, **c.desc(log)})
        if code or timeout or not absent:
            break
    after = {k: c.desc(k) for k in names}
    passed = before == after and len(outcomes) == len(commands) and all(
        x['returncode'] == 0 and not x['timed_out'] and x['group_absent'] and x['reaped'] for x in outcomes)
    if passed:
        passed = source_evidence() == evidence and c.read(output / 'capacity.json')['status'] == 'passed'
        native = c.read(output / 'native-preflight.json')
        passed = passed and native['status'] == 'passed' and native['counts'] == ZERO_COUNTS
    c.write(output / 'receipt.json', {'status': 'passed' if passed else 'failed', 'source_before': before,
        'source_after': after, 'installed_sources': evidence['files'], 'commands': outcomes,
        'files': {p.name: c.desc(p) for p in output.iterdir() if p.is_file()}, 'empirical_decodes': 0})
    print({'status': 'passed' if passed else 'failed', 'output': str(output)}, flush=True)
    c.require(passed, 'complete source qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--capacity', action='store_true')
    mode.add_argument('--native-preflight', action='store_true')
    args = parser.parse_args()
    if args.native_preflight:
        native_preflight(args.output)
    elif args.capacity:
        capacity(args.output)
    else:
        qualify(args.output)
