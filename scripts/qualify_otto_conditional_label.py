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

import otto_conditional_label_common as c

SELF = 'scripts/qualify_otto_conditional_label.py'
SOURCE_REVIEW = 'output/otto-conditional-label-v1/source-review-01.json'
SOURCE_REVIEW_PIN = 'a4ded58e92643814e12955380946c88dd9fbab696d4ef0c45d5932422969b3b7'
SEED_REVIEW = 'output/otto-conditional-label-v1/seed-review-02.json'
SEED_REVIEW_PIN = '47f214f4a96060e4e617567078efd220ed7f742eb854028b454fd544b07fd439'
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
    c.require(evidence['version'] == 'otto-conditional-label-source-review-v1'
              and evidence['numpy_version'] == '2.5.3' and evidence['law']['uniform_bits'] == 53
              and set(evidence['files']) == {'conversion', 'shipped_test', 'package_metadata'}
              and not any(evidence['counts'].values()), 'source-only conversion evidence')
    inherited = evidence['inherited_source_review']
    c.require(c.desc(inherited['path']) == {k: inherited[k] for k in ('sha256', 'bytes')}
              and c.read(inherited['path'])['files'] == evidence['files'], 'unchanged inherited conversion source review')
    c.require(c.desc(SEED_REVIEW)['sha256'] == SEED_REVIEW_PIN, 'fresh seed review pin')
    seeds = c.read(SEED_REVIEW)
    c.require(seeds['passed'] is True and seeds['proposed_unique_seeds'] == 1540
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
    """Only fabricated arrays/random models; all empirical work is prohibited."""
    import numpy as np
    import torch

    sys.path.insert(0, str(c.ROOT / 'src'))
    import audit_otto_conditional_label as audit
    import run_otto_conditional_label as runner

    from openjev.research import otto_conditional_label as target
    from openjev.research.otto_action_latent_model import make_model
    from openjev.research.otto_conditional_label_sampling import sample_endpoint

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.Generator(np.random.PCG64(712))
    n_train = c.CONFIG['train_attempts']
    n_dev = 2 * c.CONFIG['dev_attempts_per_regime']

    def fabricate(n, m, split):
        root = np.zeros((n, 2809), np.float64)
        root[:, 0] = 1.
        return {'prefix': rng.normal(size=(n, 9, 31)).astype(np.float32),
            'prefix_lengths': np.full(n, 9, np.int64),
            'prefix_actions': np.tile(np.array([0, 1] * 4, np.int64), (n, 1)),
            'prefix_outcomes': np.zeros((n, 8), np.int64),
            'prefix_position': np.full((n, 2), 26, np.int64),
            'actions': np.tile(np.array([0, 1] * 4, np.int64), (n, 1)),
            **{k: root.copy() for k in ('initial_belief', 'root_strict', 'root_grid', 'legacy_root')},
            'mc_draws': np.zeros((n, m, 9), np.uint64),
            'mc_source_indices': np.zeros((n, m), np.int64),
            'mc_outcomes': np.zeros((n, m, 8), np.int64),
            'alive': np.ones((n, m), np.bool_),
            'costs': rng.uniform(1, 8, (n, m, 4)).astype(np.float32),
            'case_ids': np.array([f'fabricated:{split}:{i}' for i in range(n)]),
            'regimes': np.array(['lambda3'] * n if split == 'train' else
                                ['lambda3'] * (n // 2) + ['lambda4'] * (n // 2))}

    train = fabricate(n_train, c.CONFIG['train_draws'], 'train')
    dev = fabricate(n_dev, c.CONFIG['dev_draws'], 'dev')
    runner.validate_data(train, np, split='train')
    runner.validate_data(dev, np, split='dev')
    bank = target.centered_bank(torch.from_numpy(train['costs']), torch.from_numpy(train['alive']))
    scale = target.train_scale(bank, variance_floor=c.CONFIG['variance_floor'])['cost_scale']
    mean_targets = target.select_target(bank, 'mean32')
    prefix, lengths, actions = [torch.from_numpy(train[k]) for k in runner.INPUT_KEYS]
    total_epoch = total_inference = 0.
    predictions = {}
    fabricated_updates = 0
    for family in c.FAMILIES:
        for seed in c.FIT_SEEDS:
            model = make_model('action_recurrent', seed, cost_scale=scale)
            for name, parameter in model.named_parameters():
                if name.startswith(runner.UNUSED_PREFIXES):
                    parameter.requires_grad_(False)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=c.CONFIG['learning_rate'])
            tick = time.perf_counter()
            for start in range(0, n_train, c.CONFIG['batch_size']):
                resource_check()
                end = start + c.CONFIG['batch_size']
                result = model.blind_rollout(prefix[start:end], lengths[start:end], actions[start:end])
                selected = mean_targets[start:end] if family == 'mean32' else target.select_target(
                    bank[start:end], family, indices=torch.zeros(len(bank[start:end]), dtype=torch.int64))
                loss = target.horizon8_loss(result['cost_contrasts'], selected, cost_scale=scale)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), c.CONFIG['clip_norm'])
                optimizer.step()
                fabricated_updates += 1
            total_epoch += time.perf_counter() - tick
            tick = time.perf_counter()
            rows = []
            model.eval()
            for start in range(n_dev):
                inputs = [torch.from_numpy(dev[k][start:start + 1]) for k in runner.INPUT_KEYS]
                with torch.no_grad():
                    rows.append(model.blind_rollout(*inputs)['cost_contrasts'].numpy())
            predictions[f'{family}__{seed}__cost'] = np.concatenate(rows)
            total_inference += time.perf_counter() - tick
    tick = time.perf_counter()
    report = runner.analyze(dev, predictions, np, check=resource_check)
    analysis_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    independent = audit.analyze(dev, predictions, np, check=resource_check)
    audit_seconds = time.perf_counter() - tick
    audit.compare_report(report, independent)

    # Maximum callback geometry at each declared bank size, on fabricated law.
    sampler_times = {}
    for m in (c.CONFIG['train_draws'], c.CONFIG['dev_draws']):
        root = np.zeros(2809, np.float64)
        root[0] = 1.
        laws = np.full((8, 2809, 4), .25, np.float64)
        indices = np.array([25 * 53 + 26, 26 * 53 + 26] * 4, np.int64)
        for h, index in enumerate(indices):
            laws[h, index] = 0.
        draws = np.random.PCG64(719 + m).random_raw((m, 9)) >> np.uint64(11)
        counts = {'updates': 0, 'fake_teacher': 0}

        def update(state, _index, odor, counts=counts):
            counts['updates'] += 1
            state[1] += odor
            return state

        def score(state, _index, counts=counts):
            counts['fake_teacher'] += 1
            return np.array([1, 2, 3, 4], np.float32)

        tick = time.perf_counter()
        sample_endpoint(root, root, laws, indices, draws, update, score, check=resource_check)
        sampler_times[m] = time.perf_counter() - tick
        c.require(counts == {'updates': 8 * m, 'fake_teacher': m}, 'maximal fake callback geometry')
    max_teacher = n_train * c.CONFIG['train_draws'] + n_dev * c.CONFIG['dev_draws']
    sampler_total = n_train * sampler_times[c.CONFIG['train_draws']] + n_dev * sampler_times[c.CONFIG['dev_draws']]
    collection_projection = max_teacher * .03425 + 2 * sampler_total + 300
    fit_projection = 2 * (total_epoch * c.CONFIG['epochs'] + total_inference + analysis_seconds) + 180
    audit_projection = 2 * (audit_seconds + sampler_total) + 300
    # Use authenticated previous output size as a transparent conservative planning proxy.
    c.parent_reference()
    c.require(c.desc(c.PARENT / 'collection-01/receipt.json') ==
              c.read(c.PARENT / 'closure-01.json')['processes']['collection']['receipt'],
              'historical planning proxy bound to audited closure')
    previous = c.read(c.PARENT / 'collection-01/receipt.json')
    teacher_previous = previous['calls']['teacher_score']['returned']
    previous_bytes = sum(row['bytes'] for row in previous['files'].values())
    output_projection = 2 * previous_bytes * max_teacher / teacher_previous + train['prefix'].nbytes + train['costs'].nbytes
    passed = collection_projection <= c.CAPS['collect'] and fit_projection <= c.CAPS['fit'] and audit_projection <= c.CAPS['audit']
    passed = passed and output_projection < 512 * 1024**2
    c.write(path, {'status': 'passed' if passed else 'failed', 'fabricated_train_cases': n_train,
        'fabricated_dev_cases': n_dev, 'fabricated_model_updates': fabricated_updates,
        'one_epoch_all_six_models_seconds': total_epoch, 'inference_seconds': total_inference,
        'analysis_seconds': analysis_seconds, 'independent_analysis_seconds': audit_seconds,
        'sampler_seconds_by_draws': sampler_times, 'max_teacher_calls': max_teacher,
        'collection_projection_seconds': collection_projection, 'fit_projection_seconds': fit_projection,
        'audit_projection_seconds': audit_projection, 'collection_output_projection_bytes': output_projection,
        'projection_previous_receipt': c.desc(c.PARENT / 'collection-01/receipt.json'),
        'projection_previous_bytes': previous_bytes, 'projection_previous_teacher_calls': teacher_previous,
        'peak_rss_bytes': resource_check(), 'caps_seconds': c.CAPS,
        'empirical_decodes': 0, 'checkpoint_decodes': 0, 'native_calls': 0, 'teacher_calls': 0,
        'scope': 'Fabricated arrays, random models and dummy filter/teacher functions only; no performance evidence.',
        'planning_limitations': 'Historical .03425 s/teacher and twice prior bytes/teacher are planning proxies, not guarantees. Audit probability reconstruction is represented by twice sampler time plus300s overhead.'})
    c.require(passed, 'prospective capacity gate')


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
        (480, [str(c.NUMERICAL), str(c.ROOT / SELF), '--capacity', '--output', str(output / 'capacity.json')]),
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
