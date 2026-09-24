"""Independent saved-output audit of the finite observation learning diagnostic.

The caller must close and authenticate the original producer before calling
``audit``. This module does not admit a process, train a model, replay gradients,
run a simulator, or import the world/model/producer. NumPy is loaded only inside
the admitted audit call. Exact-world coefficients are derived from rational
scalar definitions, independently of the vectorized target generator.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from fractions import Fraction
from pathlib import Path

VERSION = 'finite-observation-learning-audit-v1'
NAMESPACE = 420260924
ARMS = ('tied_dense', 'untied_dense', 'tied_retentive', 'untied_retentive', 'gru')
SEEDS = (420261001, 420261002, 420261003)
HORIZONS = (1, 2, 4, 8)
FIELDS = ('blind_costs', 'blind_survival', 'observed_costs', 'observed_survival',
          'observed_probabilities', 'shuffled_blind_costs')
DATA_KEYS = {'prefix', 'lengths', 'actions', 'observations', 'blind_costs', 'blind_survival',
             'observed_costs', 'observed_survival', 'observed_probabilities', 'case_ids'}
CONFIG = {'seed_namespace': NAMESPACE, 'train_attempts': 512, 'dev_attempts': 128,
          'epochs': 48, 'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': list(SEEDS), 'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational_world(epsilon):
    """Independent small finite-world reference, all entries first rational."""
    eps = Fraction(str(epsilon))
    emission = [[1 - eps if odor == state % 4 else eps / 3
                 for state in range(8)] for odor in range(4)]
    branches = [[[[Fraction(0) for _ in range(8)] for _ in range(8)] for _ in range(4)] for _ in range(4)]
    found = [[Fraction(0) for _ in range(8)] for _ in range(4)]
    for action in range(4):
        for current in range(8):
            destination = (current ^ 1, (current + 1) % 8,
                           ((current * 2) % 8) + current // 4, current ^ 4)[action]
            for following in range(8):
                transition = Fraction(1, 400) + (Fraction(49, 50) if following == destination else 0)
                hazard = Fraction(1 + ((following // 4) ^ (action % 2)), 200)
                found[action][current] += transition * hazard
                for odor in range(4):
                    branches[action][odor][following][current] = transition * (1 - hazard) * emission[odor][following]
    marginal = [[[sum(branches[action][odor][i][j] for odor in range(4)) for j in range(8)]
                 for i in range(8)] for action in range(4)]
    costs = [[Fraction(-3, 4) if action == ((state ^ (state >> 1)) % 4) else Fraction(1, 4)
              for state in range(8)] for action in range(4)]
    return {'B': branches, 'found': found, 'A': marginal, 'costs': costs, 'emission': emission}


def scalar_world(epsilon):
    def convert(value):
        return [convert(x) for x in value] if isinstance(value, list) else float(value)
    return {key: convert(value) for key, value in rational_world(epsilon).items()}


def multiply(matrix, state):
    return [math.fsum(value * state[j] for j, value in enumerate(row)) for row in matrix]


def probabilities(world, state, action):
    branches = [multiply(matrix, state) for matrix in world['B'][action]]
    masses = [math.fsum(branch) for branch in branches]
    masses.append(math.fsum(world['found'][action][j] * state[j] for j in range(8)))
    require(all(x > 0 and math.isfinite(x) for x in masses)
            and abs(math.fsum(masses) - 1) <= 1e-12, 'positive normalized reference event law')
    return branches, masses


def validate_data(data, split, np):
    require(split in ('train', 'base', 'shift') and set(data) == DATA_KEYS, 'exact declared data schema')
    n, h = len(data['case_ids']), 2 if split == 'train' else 8
    shapes = {'prefix': (n, 9, 31), 'lengths': (n,), 'actions': (n, h), 'observations': (n, h),
        'blind_costs': (n, h, 4), 'observed_costs': (n, h, 4), 'blind_survival': (n, h),
        'observed_survival': (n, h), 'observed_probabilities': (n, h, 5), 'case_ids': (n,)}
    require(n > 0 and all(data[k].shape == shape for k, shape in shapes.items()), 'nonempty exact data shapes')
    require(data['prefix'].dtype == np.float32 and all(data[k].dtype == np.int64
            for k in ('lengths', 'actions', 'observations')), 'exact public input dtypes')
    require(all(data[k].dtype == np.float64 for k in DATA_KEYS - {'prefix', 'lengths', 'actions', 'observations', 'case_ids'}),
            'float64 reference targets')
    require(data['case_ids'].dtype.kind == 'U' and len(set(data['case_ids'])) == n, 'unique Unicode case identities')
    split_id = {'train': 0, 'base': 1, 'shift': 2}[split]
    pattern = re.compile(rf'ns{NAMESPACE}-split{split_id}-case([0-9]{{10}})')
    matches = [pattern.fullmatch(str(value)) for value in data['case_ids']]
    require(all(matches), 'registered namespace and data split identities')
    indices = [int(match.group(1)) for match in matches]
    require(indices == sorted(set(indices)) and all(index < 2**32 for index in indices),
            'ordered unique uint32 attempted cases')
    require(all(np.isfinite(value).all() for key, value in data.items() if key != 'case_ids'), 'finite public inputs and targets')
    p = data['prefix']
    require(np.all(data['lengths'] == 9) and np.all((p == 0) | (p == 1))
            and not p[:, :, 10:].any() and not p[:, 0, :4].any()
            and np.all(p[:, 1:, :4].sum(-1) == 1) and np.all(p[:, :, 4:8].sum(-1) == 1)
            and not p[:, :, 8].any() and np.all(p[:, 0, 9] == 1) and not p[:, 1:, 9].any(),
            'public one-hot history only, no oracle features or found prefixes')
    require(np.all((data['actions'] >= 0) & (data['actions'] < 4))
            and np.all((data['observations'] >= 0) & (data['observations'] <= 4)), 'legal actions and labels')
    found = data['observations'] == 4
    require(np.array_equal(found, np.maximum.accumulate(found, axis=1)), 'absorbing found suffix')
    return n, h


def reconstruct(data, split, np, *, check=lambda: None):
    n, horizon = validate_data(data, split, np)
    world = scalar_world(.30 if split == 'shift' else .12)
    reconstructed = {key: np.zeros_like(data[key]) for key in
        ('blind_costs', 'blind_survival', 'observed_costs', 'observed_survival', 'observed_probabilities')}
    uniform_costs = np.zeros_like(data['blind_costs'])
    for case in range(n):
        check()
        prefix = data['prefix'][case]
        odor = int(prefix[0, 4:8].argmax())
        state = [x / 8 for x in world['emission'][odor]]
        total = math.fsum(state)
        state = [x / total for x in state]
        for row in prefix[1:]:
            action, odor = int(row[:4].argmax()), int(row[4:8].argmax())
            branches, mass = probabilities(world, state, action)
            state = [x / mass[odor] for x in branches[odor]]
        blind, observed, uniform = state.copy(), state.copy(), [1 / 8] * 8
        absorbed = False
        for step, action in enumerate(data['actions'][case]):
            action = int(action)
            blind = multiply(world['A'][action], blind)
            uniform = multiply(world['A'][action], uniform)
            reconstructed['blind_costs'][case, step] = multiply(world['costs'], blind)
            reconstructed['blind_survival'][case, step] = math.fsum(blind)
            uniform_costs[case, step] = multiply(world['costs'], uniform)
            if absorbed:
                reconstructed['observed_probabilities'][case, step, 4] = 1
                continue
            branches, mass = probabilities(world, observed, action)
            prior = [math.fsum(branch[j] for branch in branches) for j in range(8)]
            reconstructed['observed_costs'][case, step] = multiply(world['costs'], prior)
            reconstructed['observed_survival'][case, step] = math.fsum(prior)
            reconstructed['observed_probabilities'][case, step] = mass
            odor = int(data['observations'][case, step])
            if odor == 4:
                absorbed, observed = True, [0.] * 8
            else:
                observed = [x / mass[odor] for x in branches[odor]]
    errors = {key: float(np.max(np.abs(values - data[key]))) for key, values in reconstructed.items()}
    require(all(error <= 1e-12 for error in errors.values()), 'independently reconstructed exact reference targets')
    return {'cases': n, 'horizon': horizon, 'maximum_absolute_errors': errors}, reconstructed, uniform_costs


def validate_predictions(data, predictions, np):
    expected = {f'{arm}__{seed}__{field}' for arm in ARMS for seed in SEEDS for field in FIELDS}
    require(set(predictions) == expected, 'complete fifteen-model prediction fields')
    n = len(data['case_ids'])
    for name, value in predictions.items():
        field = name.rsplit('__', 1)[-1]
        shape = (n, 8, 5) if field == 'observed_probabilities' else (n, 8) if field.endswith('survival') else (n, 8, 4)
        require(value.shape == shape and value.dtype == np.float64
                and np.isfinite(value).all(), 'finite prediction shape/dtype: ' + name)
        if field.endswith('survival') or field == 'observed_probabilities':
            require(np.all((value >= 0) & (value <= 1 + 1e-12)), 'bounded predicted probability: ' + name)
        if field == 'observed_probabilities':
            require(np.all(np.abs(value.astype(np.float64).sum(-1) - 1) <= 1e-6), 'normalized predicted event law')
            require(np.all(value[data['observed_probabilities'] > 0] > 0), 'positive predicted support wherever reference is positive')


def rows_for(data, predictions, uniform_costs, regime, np, *, check=lambda: None):
    validate_predictions(data, predictions, np)
    n = len(data['case_ids'])
    rows, baseline_rows = [], []
    for horizon in HORIZONS:
        h = horizon - 1
        truth = data['blind_costs'][:, h]
        baseline_values = uniform_costs[:, h]
        # Predeclared reference-only tie handling removes arbitrary reduction
        # rounding on symmetric true-world costs. Learned policies use argmin.
        baseline_action = (baseline_values <= baseline_values.min(-1, keepdims=True) + 1e-12).argmax(-1)
        baseline_rows.append({'regime': regime, 'horizon': horizon, 'cases': n,
            'blind_cost_mse': math.fsum(float(x) for x in (uniform_costs[:, h] - truth).ravel() ** 2) / (n * 4),
            'blind_regret': math.fsum(float(truth[i, baseline_action[i]] - min(truth[i])) for i in range(n)) / n})
    for arm in ARMS:
        for seed in SEEDS:
            check()
            p = {field: predictions[f'{arm}__{seed}__{field}'].astype(np.float64) for field in FIELDS}
            for horizon in HORIZONS:
                h = horizon - 1
                blind = data['blind_costs'][:, h]
                selected, shuffled = p['blind_costs'][:, h].argmin(-1), p['shuffled_blind_costs'][:, h].argmin(-1)
                target_p = data['observed_probabilities'][:, h]
                kl = math.fsum(float(q) * (math.log(float(q)) - math.log(float(p['observed_probabilities'][i, h, o])))
                    for i in range(n) for o, q in enumerate(target_p[i]) if q > 0) / n
                row = {'arm': arm, 'seed': seed, 'regime': regime, 'horizon': horizon, 'cases': n,
                    'blind_cost_mse': math.fsum(float(x) ** 2 for x in (p['blind_costs'][:, h] - blind).ravel()) / (n * 4),
                    'blind_regret': math.fsum(float(blind[i, selected[i]] - min(blind[i])) for i in range(n)) / n,
                    'blind_survival_mae': math.fsum(abs(float(x)) for x in p['blind_survival'][:, h] - data['blind_survival'][:, h]) / n,
                    'observed_cost_mse': math.fsum(float(x) ** 2 for x in (p['observed_costs'][:, h] - data['observed_costs'][:, h]).ravel()) / (n * 4),
                    'observed_survival_mae': math.fsum(abs(float(x)) for x in p['observed_survival'][:, h] - data['observed_survival'][:, h]) / n,
                    'observed_kl': kl,
                    'shuffled_blind_regret': math.fsum(float(blind[i, shuffled[i]] - min(blind[i])) for i in range(n)) / n}
                require(all(math.isfinite(value) for value in row.values() if type(value) is float), 'finite audit metric')
                rows.append(row)
    return rows, baseline_rows


def gates(rows, baselines, counts):
    keyed = {(row['arm'], row['seed'], row['regime'], row['horizon']): row for row in rows}
    baseline = {(row['regime'], row['horizon']): row for row in baselines}
    results = {}
    for regime in ('base', 'shift'):
        conditions = {'minimum_train': counts['train'] >= 256, 'minimum_development': counts[regime] >= 64}
        for horizon in (4, 8):
            reference = baseline[regime, horizon]
            conditions[f'h{horizon}_positive_baseline_mse'] = reference['blind_cost_mse'] > 0
            conditions[f'h{horizon}_positive_baseline_regret'] = reference['blind_regret'] > 0
            for seed in SEEDS:
                row = keyed['tied_retentive', seed, regime, horizon]
                conditions[f'{seed}_h{horizon}_half_mse'] = row['blind_cost_mse'] <= .5 * reference['blind_cost_mse']
                conditions[f'{seed}_h{horizon}_half_regret'] = row['blind_regret'] <= .5 * reference['blind_regret']
        for seed in SEEDS:
            conditions[f'{seed}_h8_survival'] = keyed['tied_retentive', seed, regime, 8]['blind_survival_mae'] <= .05
            if regime == 'base':
                for horizon in (1, 2):
                    conditions[f'{seed}_h{horizon}_observed_kl'] = keyed['tied_retentive', seed, regime, horizon]['observed_kl'] <= .1
        passed = all(conditions.values())
        success = 'BASE_TRAINABLE' if regime == 'base' else 'SHIFT_TRANSFER'
        results[regime] = {'passed': passed, 'status': success if passed else success + '_FAIL', 'conditions': conditions}
    return results


def compare(saved, independent, *, path='report'):
    if type(independent) is dict:
        require(type(saved) is dict and set(saved) == set(independent), 'exact scalar schema at ' + path)
        for key in independent:
            compare(saved[key], independent[key], path=path + '.' + key)
    elif type(independent) is list:
        require(type(saved) is list and len(saved) == len(independent), 'exact scalar roster at ' + path)
        for index, (left, right) in enumerate(zip(saved, independent, strict=True)):
            compare(left, right, path=f'{path}[{index}]')
    elif type(independent) is float:
        require(type(saved) in (int, float) and math.isfinite(saved)
                and math.isclose(saved, independent, rel_tol=1e-10, abs_tol=1e-12), 'independent scalar agreement at ' + path)
    else:
        require(type(saved) is type(independent) and saved == independent, 'independent exact agreement at ' + path)


def metadata(folder, summary, *, check=lambda: None):
    """Authenticate saved joins and opaque bytes, without decoding a checkpoint.

    These records attest historical ordering and optimizer work. Their agreement
    does not independently replay either training or the temporal barrier.
    """
    folder = Path(folder)

    def read(name, *, lines=False):
        path = folder / name
        require(path.is_file() and not path.is_symlink(), 'regular metadata file: ' + name)
        return [json.loads(line) for line in path.read_text().splitlines()] if lines else json.loads(path.read_text())

    require(summary['version'] == 'finite-observation-learning-v1', 'producer version')
    compare(summary['config'], CONFIG, path='registered config')
    compare(read('config.json'), CONFIG, path='saved config')
    expected_fits = [(arm, seed) for seed in SEEDS
                     for arm in ARMS[SEEDS.index(seed) % len(ARMS):] + ARMS[:SEEDS.index(seed) % len(ARMS)]]
    filenames = {'config.json', 'train.npz', 'base.npz', 'shift.npz', 'training-orders.jsonl',
                 'training-epochs.jsonl', 'fits.jsonl', 'checkpoint-barrier.json',
                 'prediction-times.jsonl', 'predictions-base.npz', 'predictions-shift.npz'}
    filenames |= {f'{arm}-{seed}.npz' for arm, seed in expected_fits}
    require(set(summary['files']) == filenames, 'complete original producer file roster')
    for name, descriptor in summary['files'].items():
        check()
        path = folder / name
        require(path.is_file() and not path.is_symlink(), 'regular original payload: ' + name)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        compare(descriptor, {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}, path=name)
    dataset_counts = summary['dataset_counts']
    require(set(dataset_counts) == {'train', 'base', 'shift'}, 'three dataset count records')
    retained = {}
    for split_id, split in enumerate(('train', 'base', 'shift')):
        row = dataset_counts[split]
        attempts, horizon = (512, 2) if split == 'train' else (128, 8)
        require(row['version'] == 'finite-observation-world-v1'
                and row['epsilon'] == (.30 if split == 'shift' else .12)
                and row['seed_namespace'] == NAMESPACE and row['split_id'] == split_id
                and row['attempts'] == attempts and row['horizon'] == horizon,
                'registered generation geometry')
        n, excluded = row['retained'], row['excluded_found']
        found = row['prefix_found_by_step']
        require(type(n) is int and 0 < n <= attempts and type(excluded) is int and excluded >= 0
                and n + excluded == attempts and type(found) is list and len(found) == 8
                and all(type(v) is int and v >= 0 for v in found) and sum(found) == excluded,
                'complete retained/excluded prefix accounting')
        require(row['initial_odor_draws'] == attempts and row['prefix_action_draws'] == 8 * attempts
                and row['prefix_event_draws'] == 8 * n + sum((i + 1) * v for i, v in enumerate(found))
                and row['forecast_action_draws'] == horizon * n, 'generation draw accounting')
        retained[split] = n
    n = retained['train']
    fits = summary['fits']
    require([(row['arm'], row['seed']) for row in fits] == expected_fits, 'all fifteen final fits in fixed execution order')
    compare(read('fits.jsonl', lines=True), fits, path='durable fits')
    updates = 48 * math.ceil(n / 64)
    for row in fits:
        require(row['epochs'] == 48 and row['updates'] == updates and row['training_cases'] == n
                and row['training_case_exposures'] == 48 * n, 'fixed complete fit schedule')
        require(type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                'finite fit duration')
        for key in ('initial_state_sha256', 'shared_initial_state_sha256', 'final_state_sha256', 'case_order_sha256'):
            require(type(row[key]) is str and re.fullmatch('[0-9a-f]{64}', row[key]), 'state/order digest')
        name = f"{row['arm']}-{row['seed']}.npz"
        compare(row['checkpoint'], {'path': name, **summary['files'][name]}, path='final checkpoint')
    barrier = {'checkpoints': [{'arm': row['arm'], 'seed': row['seed'], **row['checkpoint']} for row in fits],
               'dev_generation_count': 0, 'fit_count': 15}
    compare(summary['checkpoint_barrier'], barrier, path='summary barrier')
    compare(read('checkpoint-barrier.json'), barrier, path='durable barrier')
    orders, epochs = read('training-orders.jsonl', lines=True), read('training-epochs.jsonl', lines=True)
    expected_epochs = [(arm, seed, epoch) for arm, seed in expected_fits for epoch in range(48)]
    for records in (orders, epochs):
        require([(r['arm'], r['seed'], r['epoch']) for r in records] == expected_epochs,
                'complete fixed epoch journal roster')
    order_digests = {pair: hashlib.sha256() for pair in expected_fits}
    paired_orders = {}
    for row in orders:
        indices = row['indices']
        require(set(row) == {'arm', 'seed', 'epoch', 'indices', 'batch_size'} and row['batch_size'] == 64
                and type(indices) is list and len(indices) == n and all(type(i) is int for i in indices)
                and sorted(indices) == list(range(n)), 'complete once-per-epoch case permutation')
        key = row['seed'], row['epoch']
        if key in paired_orders:
            require(indices == paired_orders[key], 'same-seed paired case order across arms')
        paired_orders[key] = indices
        order_digests[row['arm'], row['seed']].update(struct.pack('<' + 'q' * n, *indices))
    for row in epochs:
        require(row['cases'] == n and row['updates'] == math.ceil(n / 64)
                and all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                        for k in ('mean_objective', 'seconds')), 'finite completed epoch work')
    fit_map = {(row['arm'], row['seed']): row for row in fits}
    for pair, digest in order_digests.items():
        require(fit_map[pair]['case_order_sha256'] == digest.hexdigest(), 'saved case-order digest')
    for seed in SEEDS:
        for init in ('dense', 'retentive'):
            require(fit_map['tied_' + init, seed]['shared_initial_state_sha256']
                    == fit_map['untied_' + init, seed]['shared_initial_state_sha256'], 'paired shared initialization')
    times = summary['prediction_times']
    require([(r['regime'], r['arm'], r['seed']) for r in times]
            == [(regime, arm, seed) for regime in ('base', 'shift') for arm in ARMS for seed in SEEDS],
            'thirty complete evaluation views')
    compare(read('prediction-times.jsonl', lines=True), times, path='durable evaluation records')
    for row in times:
        require(row['model_state_before'] == row['model_state_after']
                == fit_map[row['arm'], row['seed']]['final_state_sha256']
                and row['cases'] == retained[row['regime']] and row['batch_size'] == 64
                and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                'unchanged final checkpoint state for each evaluation')
    eval_batches = 15 * sum(math.ceil(retained[regime] / 64) for regime in ('base', 'shift'))
    expected_counts = {'train_generation_count': 1, 'dev_generation_count': 2, 'model_constructions': 15,
        'fit_count': 15, 'training_blind_rollouts': 15 * updates, 'training_observed_rollouts': 15 * updates,
        'optimizer_attempts': 15 * updates, 'optimizer_steps': 15 * updates,
        'training_case_exposures': 15 * 48 * n, 'checkpoint_writes': 15,
        'evaluation_blind_rollouts': eval_batches, 'evaluation_observed_rollouts': eval_batches,
        'evaluation_shuffled_rollouts': eval_batches, 'evaluation_case_views': 15 * (retained['base'] + retained['shift']),
        'array_decodes': 0, 'checkpoint_decodes': 0, 'external_model_calls': 0, 'native_calls': 0, 'teacher_calls': 0}
    compare(summary['counts'], expected_counts, path='complete executed counters')
    return {'fits': 15, 'checkpoint_files_hashed': 15, 'epochs': 720, 'evaluation_views': 30,
            'retained': retained, 'optimizer_steps_attested': 15 * updates,
            'training_case_exposures_attested': 15 * 48 * n,
            'historical_ordering_independently_replayed': False}


def audit(folder, *, check=lambda: None):
    """Read the externally authenticated original saved output exactly once."""
    import numpy as np

    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder,
            'absolute original saved-output directory')
    summary_path = folder / 'summary.json'
    require(summary_path.is_file() and not summary_path.is_symlink(), 'regular saved producer summary')
    summary = json.loads(summary_path.read_text())
    metadata_record = metadata(folder, summary, check=check)
    counters = {'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0,
                'optimizer_calls': 0, 'world_or_generator_calls': 0, 'native_calls': 0}

    def load(name):
        check()
        path = folder / name
        require(path.is_file() and not path.is_symlink(), 'regular admitted saved array')
        with np.load(path, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), 'unique saved array members')
            value = {key: archive[key] for key in archive.files}
        counters['array_decodes'] += 1
        return value

    datasets = {split: load(split + '.npz') for split in ('train', 'base', 'shift')}
    all_ids = [str(identity) for data in datasets.values() for identity in data['case_ids']]
    require(len(all_ids) == len(set(all_ids)), 'disjoint registered split identities')
    reconstructions, rows, baselines = {}, [], []
    counts = {split: len(data['case_ids']) for split, data in datasets.items()}
    compare(counts, metadata_record['retained'], path='saved arrays match retained counts')
    for split, data in datasets.items():
        attempts = 512 if split == 'train' else 128
        require(all(int(str(identity).rsplit('case', 1)[1]) < attempts for identity in data['case_ids']),
                'case identities within fixed attempted roster')
    for split, data in datasets.items():
        record, reconstructed, uniform = reconstruct(data, split, np, check=check)
        reconstructions[split] = record
        observations = data['observations']
        found = observations == 4
        found_cases = found.any(axis=1)
        event_draws = np.where(found_cases, found.argmax(axis=1) + 1, observations.shape[1])
        require(summary['dataset_counts'][split]['forecast_found_cases'] == int(found_cases.sum())
                and summary['dataset_counts'][split]['forecast_event_draws'] == int(event_draws.sum()),
                'forecast draw counts independently derived from saved observations')
        if split != 'train':
            predictions = load('predictions-' + split + '.npz')
            # Metrics use independent targets after exact saved-label agreement.
            independent_data = {**data, **reconstructed}
            these_rows, these_baselines = rows_for(independent_data, predictions, uniform, split, np, check=check)
            rows.extend(these_rows)
            baselines.extend(these_baselines)
    row_key = lambda row: (row['arm'], row['seed'], row['regime'], row['horizon'])
    base_key = lambda row: (row['regime'], row['horizon'])
    compare(sorted(summary['rows'], key=row_key), sorted(rows, key=row_key), path='rows')
    compare(sorted(summary['baseline_rows'], key=base_key), sorted(baselines, key=base_key), path='baseline_rows')
    return {'version': VERSION, 'agreement': True, 'technical_complete': False,
        'requires_original_supervisor_closure': True, 'counts': counters,
        'data_cases': counts, 'metadata': metadata_record, 'target_reconstruction': reconstructions, 'rows': rows,
        'baseline_rows': baselines, 'gate': gates(rows, baselines, counts),
        'architecture_claim': False, 'uniform_reference_tie_tolerance': 1e-12,
        'candidate_action_selection': 'lowest-index raw argmin', 'limitations': [
            'The caller supplies original process/source/payload authentication before these numerical reads.',
            'Predictions, trained checkpoints and optimizer updates are not re-executed; this audits saved outputs only.',
            'Checkpoint bytes, paired schedules, barrier and inference-state records agree; historical execution order remains a producer attestation.',
            'Case namespace and public target construction are checked, but RNG sampling histories are not independently replayed.',
            'Shuffled predictions are scored as saved; their producer permutation and causal input isolation remain source-reviewed.',
            'The uniform-state known-dynamics reference is history-ignorant, not an optimal history-ignorant policy.',
            'An oracle-realizable synthetic world does not establish native-environment transfer or architectural novelty.']}
