"""Bounded fabricated qualification and metadata-only native runtime preflight."""
from __future__ import annotations

import argparse
import importlib.abc
import os
import resource
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import otto_belief_distillation_common as c

SELF = 'scripts/qualify_otto_belief_distillation.py'
SOURCE_REVIEW = 'output/otto-belief-distillation-v1/source-review-01.json'
SOURCE_REVIEW_PIN = '3cd551b28d6a2162dbe789e46f3832c659a4b396a57f884f9612420c1c116e0a'
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
    c.require(evidence['version'] == 'otto-belief-distillation-source-review-v1'
              and evidence['numpy_version'] == '2.5.3' and evidence['law']['uniform_bits'] == 53
              and set(evidence['files']) == {'conversion', 'shipped_test', 'package_metadata'}
              and not any(evidence['counts'].values()), 'source-only conversion evidence')
    for record in evidence['files'].values():
        c.require(c.desc(record['path']) == {k: record[k] for k in ('sha256', 'bytes')}, 'installed source pin')
    return evidence


def native_preflight(output):
    c.require(Path.cwd() == c.ROOT and Path(sys.executable).absolute() == c.NATIVE,
              'original native interpreter and working directory')
    with metadata_only():
        evidence = source_evidence()
        prior, paths, _, _ = c.original_native()
        c.require(prior['runtime']['python_executable'] == str(c.NATIVE)
                  and prior['runtime']['versions']['numpy'] == evidence['numpy_version'], 'native runtime source version')
        c.write(output, {'status': 'passed', 'native_runtime': prior['runtime'],
            'native_sources': prior['sources'], 'native_inputs': prior['native_inputs'],
            'inherited_input_paths': {k: str(v) for k, v in paths.items()},
            'source_review': c.desc(SOURCE_REVIEW), 'installed_sources': evidence['files'],
            'metadata_import_guard': True, 'counts': ZERO_COUNTS})


def resource_check():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    c.require(rss <= 4 * 1024**3, 'fabricated capacity 4GiB RSS cap')
    return rss


def fabricated(np, count, horizon):
    """All-alive fabricated blocks exercise maximum observation/model work."""
    rng = np.random.default_rng(202 + horizon)
    oracle = np.full((count, horizon, 5), .25, np.float64)
    oracle[..., 4] = 0.
    initial = np.zeros((count, 2809), np.float64)
    initial[:, 0] = 1.  # Fabricated source-cell belief, outside every synthetic path.
    return {'prefix': rng.normal(size=(count, 9, 31)).astype(np.float32),
        'prefix_lengths': np.full(count, 9, np.int64),
        'actions': rng.integers(0, 4, (count, horizon), dtype=np.int64),
        'continuation': rng.normal(size=(count, horizon, 31)).astype(np.float32),
        'outcomes': rng.integers(0, 4, (count, horizon), dtype=np.int64),
        'raw_costs': rng.normal(size=(count, horizon, 4)).astype(np.float32),
        'legal': np.ones((count, horizon, 4), np.bool_),
        'gap_oracle': oracle.copy(), 'normal_oracle': oracle.copy(), 'opposite_oracle': oracle.copy(),
        'initial_belief': initial,
        'prefix_actions': np.tile(np.array([0, 1] * 4, np.int64), (count, 1)),
        'prefix_outcomes': np.zeros((count, 8), np.int64),
        'prefix_position': np.full((count, 2), 26, np.int64),
        'case_ids': np.array([f'fabricated:{i}' for i in range(count)]),
        'regimes': np.array(['lambda3' if i < count // 2 else 'lambda4' for i in range(count)])}


def capacity(path):
    import numpy as np
    import torch

    sys.path.insert(0, str(c.ROOT / 'src'))
    import audit_otto_belief_distillation as audit
    import run_otto_belief_distillation as runner

    from openjev.research import otto_belief_distillation_metrics as metrics
    from openjev.research.otto_action_latent_model import make_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    n, batch_size = c.CONFIG['train_cases'], c.CONFIG['batch']
    train = fabricated(np, n, 4)
    dev = fabricated(np, 2 * c.CONFIG['dev_cases_per_regime'], 8)
    keys = ('prefix', 'prefix_lengths', 'actions', 'continuation', 'outcomes', 'raw_costs', 'gap_oracle', 'normal_oracle')
    tensors = {k: torch.from_numpy(train[k]) for k in keys}
    times, inference, scoring, independent, updates = {}, {}, {}, {}, {}
    epoch_order = np.random.Generator(np.random.PCG64(c.CONFIG['fit_seeds'][0])).permutation(n)
    for family in runner.FAMILIES:
        tick = time.perf_counter()
        model = make_model(runner.MODEL_KINDS[family], c.CONFIG['fit_seeds'][0], cost_scale=.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.CONFIG['lr'])
        updates[family] = 0
        for start in range(0, n, batch_size):
            resource_check()
            selected = torch.from_numpy(epoch_order[start:start + batch_size])
            batch = {k: v[selected] for k, v in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            gap = model.blind_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'])
            normal = model.normal_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'],
                                         batch['continuation'], found=batch['outcomes'] == 4)
            soft = family != 'recurrent_sampled'
            loss = (runner.loss_for(gap, batch, torch, .01, condition='gap', soft_targets=soft)
                    + runner.loss_for(normal, batch, torch, .01, condition='normal', soft_targets=soft)) / 2
            c.require(bool(torch.isfinite(loss)), 'finite fabricated training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.CONFIG['clip'], error_if_nonfinite=True)
            optimizer.step()
            updates[family] += 1
        times[family] = time.perf_counter() - tick
        tick = time.perf_counter()
        model.eval()
        predictions, costs = {}, {}
        # Match the producer's actual batch-one inference and input copies.
        for condition in ('gap', 'normal', 'opposite'):
            resource_check()
            outputs, cost_outputs = [], []
            for i in range(len(dev['case_ids'])):
                selected = slice(i, i + 1)
                batch = {k: torch.from_numpy(dev[k][selected].copy()) for k in runner.INPUT_KEYS}
                if condition == 'opposite':
                    batch['actions'] ^= 1
                with torch.no_grad():
                    args = [batch[k] for k in ('prefix', 'prefix_lengths', 'actions')]
                    if condition == 'normal':
                        outcomes = torch.from_numpy(dev['outcomes'][selected].copy())
                        future = torch.from_numpy(dev['continuation'][selected].copy())
                        result = model.normal_rollout(*args, future, found=outcomes == 4)
                        outcome = runner.probabilities(result['outcome_logits'], torch,
                                                       condition='normal', outcomes=outcomes)
                    else:
                        result = model.blind_rollout(*args)
                        outcome = runner.probabilities(result['outcome_logits'], torch, condition='gap')
                outputs.append(outcome.numpy())
                cost_outputs.append(result['cost_contrasts'].numpy())
            predictions[condition], costs[condition] = np.concatenate(outputs), np.concatenate(cost_outputs)
        inference[family] = time.perf_counter() - tick
        tick = time.perf_counter()
        for condition in ('gap', 'normal'):
            metrics.score(dev['outcomes'], predictions[condition], costs[condition], dev['raw_costs'], dev['legal'],
                oracle_probabilities=dev[condition + '_oracle'], case_ids=dev['case_ids'].tolist(),
                regimes=dev['regimes'].tolist(), family=family, fit_seed=c.CONFIG['fit_seeds'][0],
                condition=condition, prediction_kind='probabilities')
        metrics.action_sensitivity(dev['gap_oracle'], dev['opposite_oracle'],
            case_ids=dev['case_ids'].tolist(), regimes=dev['regimes'].tolist(), actions=dev['actions'],
            alternate_actions=dev['actions'] ^ 1, predicted_original=predictions['gap'],
            predicted_alternate=predictions['opposite'])
        scoring[family] = time.perf_counter() - tick
        tick = time.perf_counter()
        for condition in ('gap', 'normal'):
            audit.scalar_report(np, dev, predictions[condition], costs[condition], family=family,
                                fit_seed=c.CONFIG['fit_seeds'][0], condition=condition, check=resource_check)
        audit.sensitivity_report(np, dev, predictions['gap'], predictions['opposite'], check=resource_check)
        independent[family] = time.perf_counter() - tick
    table = np.full((105, 105, 4), .25, np.float64)
    table[52, 52] = 0.
    tables = {k: table.copy() for k in ('lambda3', 'lambda4', 'lambda3_raw', 'lambda4_raw')}
    tick = time.perf_counter()
    oracle_counts = audit.verify_oracles(np, dev, tables, check=resource_check)
    oracle_seconds = time.perf_counter() - tick
    seeds, epochs = len(c.CONFIG['fit_seeds']), c.CONFIG['epochs']
    fit_projection = 2 * seeds * (epochs * sum(times.values()) + sum(inference.values()) + sum(scoring.values())) + 180
    audit_projection = 2 * (seeds * sum(independent.values()) + oracle_seconds) + 180
    passed = fit_projection <= c.CAPS['fit'] and audit_projection <= c.CAPS['audit']
    c.write(path, {'status': 'passed' if passed else 'failed', 'one_epoch_seconds': times,
        'dev_inference_seconds': inference, 'producer_scoring_seconds': scoring,
        'independent_scoring_seconds': independent, 'oracle_verification_seconds': oracle_seconds,
        'oracle_verification_counts': oracle_counts, 'updates_per_family': updates,
        'pessimistic_projection_seconds': fit_projection, 'audit_projection_seconds': audit_projection,
        'cap_seconds': c.CAPS['fit'], 'audit_cap_seconds': c.CAPS['audit'],
        'projection_factor': 2, 'fixed_overhead_seconds_per_phase': 180,
        'fabricated_train_cases': n, 'fabricated_dev_cases': len(dev['outcomes']),
        'measured_epochs_per_family': 1, 'projected_fits': seeds * len(times), 'projected_epochs_per_fit': epochs,
        'projected_training_updates': seeds * epochs * sum(updates.values()),
        'projected_training_case_exposures': seeds * epochs * len(times) * n,
        'peak_rss_bytes': resource_check(), 'empirical_decodes': 0, 'native_calls': 0,
        'scope': 'All observations alive; fabricated categorical grid, features and targets. No empirical files or native runtime.'})
    c.require(passed, 'prospective capacity gate')


def qualify(output):
    c.require(Path.cwd() == c.ROOT and output.is_absolute() and output.is_relative_to(c.OUT), 'contained qualification output')
    c.require(not output.exists(), 'fresh qualification directory')
    evidence = source_evidence()
    names = sorted(set(c.SOURCES) | {SOURCE_REVIEW})
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
