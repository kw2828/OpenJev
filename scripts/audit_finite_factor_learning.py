"""Independent saved-output audit for the fresh finite-world factor diagnostic.

The caller authenticates original sources, process closure and payloads before
``audit``. Only frozen scalar rational-world arithmetic is reused. No model,
world generator, producer or optimizer is imported or executed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

from audit_finite_observation_learning import compare, multiply, probabilities, require, scalar_world

VERSION = 'finite-factor-learning-audit-v1'
NAMESPACE = 421260924
FACTOR_ARMS = ('learned_exact', 'exact_learned', 'learned_learned')
ARMS = (*FACTOR_ARMS, 'gru')
SEEDS = (421261001, 421261002, 421261003)
HORIZONS = (1, 2, 4, 8)
TARGETS = ('blind_costs', 'blind_survival', 'observed_costs', 'observed_survival', 'observed_probabilities')
FIELDS = (*TARGETS, 'shuffled_blind_costs')
DATA_KEYS = {'prefix', 'lengths', 'actions', 'observations', 'case_ids', *TARGETS}
CONFIG = {'seed_namespace': NAMESPACE, 'train_attempts': 512, 'dev_attempts': 128,
          'epochs': 480, 'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': list(SEEDS), 'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8}


def validate_data(data, split, np):
    require(split in ('train', 'base') and set(data) == DATA_KEYS, 'exact factor-study data schema')
    n, h = len(data['case_ids']), 2 if split == 'train' else 8
    shapes = {'prefix': (n, 9, 31), 'lengths': (n,), 'actions': (n, h), 'observations': (n, h),
              'case_ids': (n,), 'blind_costs': (n, h, 4), 'observed_costs': (n, h, 4),
              'blind_survival': (n, h), 'observed_survival': (n, h), 'observed_probabilities': (n, h, 5)}
    require(n > 0 and all(data[k].shape == shape for k, shape in shapes.items()), 'nonempty exact factor-study shapes')
    require(data['prefix'].dtype == np.float32 and all(data[k].dtype == np.int64
            for k in ('lengths', 'actions', 'observations')), 'exact public input dtypes')
    require(all(data[k].dtype == np.float64 for k in TARGETS), 'float64 saved reference targets')
    require(data['case_ids'].dtype.kind == 'U' and len(set(data['case_ids'])) == n, 'unique Unicode case identities')
    split_id, attempts = (0, 512) if split == 'train' else (1, 128)
    pattern = re.compile(rf'ns{NAMESPACE}-split{split_id}-case([0-9]{{10}})')
    matches = [pattern.fullmatch(str(value)) for value in data['case_ids']]
    require(all(matches), 'fresh registered namespace and split')
    indices = [int(match.group(1)) for match in matches]
    require(indices == sorted(set(indices)) and all(index < attempts for index in indices), 'fixed attempted case roster')
    require(all(np.isfinite(value).all() for key, value in data.items() if key != 'case_ids'), 'finite public inputs and targets')
    p = data['prefix']
    require(np.all(data['lengths'] == 9) and np.all((p == 0) | (p == 1)) and not p[:, :, 10:].any()
            and not p[:, 0, :4].any() and np.all(p[:, 1:, :4].sum(-1) == 1)
            and np.all(p[:, :, 4:8].sum(-1) == 1) and not p[:, :, 8].any()
            and np.all(p[:, 0, 9] == 1) and not p[:, 1:, 9].any(), 'public history features without oracle input')
    require(np.all((data['actions'] >= 0) & (data['actions'] < 4))
            and np.all((data['observations'] >= 0) & (data['observations'] <= 4)), 'legal public actions and labels')
    found = data['observations'] == 4
    require(np.array_equal(found, np.maximum.accumulate(found, axis=1)), 'absorbing found suffix')
    return n, h


def reconstruct(data, prefix_oracle, exact_predictions, split, np, *, check=lambda: None):
    """Derive all targets and privileged prefix states from public tokens alone."""
    n, horizon = validate_data(data, split, np)
    require(set(prefix_oracle) == {'prefix_state'}, 'single separately stored oracle input')
    state_array = prefix_oracle['prefix_state']
    require(state_array.shape == (n, 8) and state_array.dtype == np.float64
            and np.isfinite(state_array).all() and np.all(state_array >= 0)
            and np.all(np.abs(state_array.sum(-1) - 1) <= 1e-12), 'normalized float64 privileged prefix state')
    require(set(exact_predictions) == set(TARGETS), 'exact/exact control has five fields and no fit seed')
    for name in TARGETS:
        value = exact_predictions[name]
        require(value.shape == data[name].shape and value.dtype == np.float64 and np.isfinite(value).all(),
                'finite float64 exact-control output: ' + name)
    world = scalar_world(.12)
    targets = {key: np.zeros_like(data[key]) for key in TARGETS}
    prefix_states = np.zeros((n, 8), np.float64)
    uniform_costs = np.zeros_like(data['blind_costs'])
    for case in range(n):
        check()
        prefix = data['prefix'][case]
        odor = int(prefix[0, 4:8].argmax())
        state = [value / 8 for value in world['emission'][odor]]
        total = math.fsum(state)
        state = [value / total for value in state]
        for row in prefix[1:]:
            action, odor = int(row[:4].argmax()), int(row[4:8].argmax())
            branches, mass = probabilities(world, state, action)
            state = [value / mass[odor] for value in branches[odor]]
        prefix_states[case] = state
        blind, observed, uniform = state.copy(), state.copy(), [1 / 8] * 8
        absorbed = False
        for step, action in enumerate(data['actions'][case]):
            action = int(action)
            blind = multiply(world['A'][action], blind)
            uniform = multiply(world['A'][action], uniform)
            targets['blind_costs'][case, step] = multiply(world['costs'], blind)
            targets['blind_survival'][case, step] = math.fsum(blind)
            uniform_costs[case, step] = multiply(world['costs'], uniform)
            if absorbed:
                targets['observed_probabilities'][case, step, 4] = 1
                continue
            branches, mass = probabilities(world, observed, action)
            prior = [math.fsum(branch[j] for branch in branches) for j in range(8)]
            targets['observed_costs'][case, step] = multiply(world['costs'], prior)
            targets['observed_survival'][case, step] = math.fsum(prior)
            targets['observed_probabilities'][case, step] = mass
            odor = int(data['observations'][case, step])
            if odor == 4:
                absorbed, observed = True, [0.] * 8
            else:
                observed = [value / mass[odor] for value in branches[odor]]
    prefix_error = float(np.max(np.abs(prefix_states - state_array)))
    target_errors = {name: float(np.max(np.abs(targets[name] - data[name]))) for name in TARGETS}
    exact_errors = {name: float(np.max(np.abs(targets[name] - exact_predictions[name]))) for name in TARGETS}
    require(prefix_error <= 1e-12 and all(error <= 1e-12 for error in target_errors.values()),
            'independent public-history posterior and reference targets')
    require(all(error <= 1e-12 for error in exact_errors.values()), 'exact/exact oracle agreement at every saved horizon')
    return {'cases': n, 'horizon': horizon, 'prefix_maximum_absolute_error': prefix_error,
            'target_maximum_absolute_errors': target_errors, 'oracle_maximum_absolute_errors': exact_errors}, targets, uniform_costs


def validate_predictions(data, predictions, np):
    expected = {f'{arm}__{seed}__{field}' for arm in ARMS for seed in SEEDS for field in FIELDS}
    require(set(predictions) == expected, 'complete twelve-model prediction fields')
    n = len(data['case_ids'])
    for name, value in predictions.items():
        field = name.rsplit('__', 1)[-1]
        shape = (n, 8, 5) if field == 'observed_probabilities' else (n, 8) if field.endswith('survival') else (n, 8, 4)
        require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(), 'finite float64 prediction: ' + name)
        if field.endswith('survival') or field == 'observed_probabilities':
            require(np.all((value >= 0) & (value <= 1 + 1e-12)), 'bounded predicted probability: ' + name)
        if field == 'observed_probabilities':
            require(np.all(np.abs(value.sum(-1) - 1) <= 1e-6), 'normalized predicted event law')
            require(np.all(value[data['observed_probabilities'] > 0] > 0), 'positive oracle support requires positive predictions')


def rows_for(data, predictions, uniform_costs, np, *, check=lambda: None):
    validate_predictions(data, predictions, np)
    n = len(data['case_ids'])
    rows, baselines = [], []
    for horizon in HORIZONS:
        h = horizon - 1
        target, reference = data['blind_costs'][:, h], uniform_costs[:, h]
        chosen = (reference <= reference.min(-1, keepdims=True) + 1e-12).argmax(-1)
        baselines.append({'regime': 'base', 'horizon': horizon, 'cases': n,
            'blind_cost_mse': math.fsum(float(x) ** 2 for x in (reference - target).ravel()) / (n * 4),
            'blind_regret': math.fsum(float(target[i, chosen[i]] - min(target[i])) for i in range(n)) / n})
    for arm in ARMS:
        for seed in SEEDS:
            check()
            p = {field: predictions[f'{arm}__{seed}__{field}'] for field in FIELDS}
            for horizon in HORIZONS:
                h = horizon - 1
                blind = data['blind_costs'][:, h]
                selected, shuffled = p['blind_costs'][:, h].argmin(-1), p['shuffled_blind_costs'][:, h].argmin(-1)
                target_p = data['observed_probabilities'][:, h]
                kl = math.fsum(float(q) * (math.log(float(q)) - math.log(float(p['observed_probabilities'][i, h, o])))
                               for i in range(n) for o, q in enumerate(target_p[i]) if q > 0) / n
                row = {'arm': arm, 'seed': seed, 'regime': 'base', 'horizon': horizon, 'cases': n,
                    'blind_cost_mse': math.fsum(float(x) ** 2 for x in (p['blind_costs'][:, h] - blind).ravel()) / (n * 4),
                    'blind_regret': math.fsum(float(blind[i, selected[i]] - min(blind[i])) for i in range(n)) / n,
                    'blind_survival_mae': math.fsum(abs(float(x)) for x in p['blind_survival'][:, h] - data['blind_survival'][:, h]) / n,
                    'observed_cost_mse': math.fsum(float(x) ** 2 for x in (p['observed_costs'][:, h] - data['observed_costs'][:, h]).ravel()) / (n * 4),
                    'observed_survival_mae': math.fsum(abs(float(x)) for x in p['observed_survival'][:, h] - data['observed_survival'][:, h]) / n,
                    'observed_kl': kl,
                    'shuffled_blind_regret': math.fsum(float(blind[i, shuffled[i]] - min(blind[i])) for i in range(n)) / n}
                require(all(math.isfinite(value) for value in row.values() if type(value) is float), 'finite independently computed metrics')
                rows.append(row)
    return rows, baselines


def gates(rows, baselines, counts):
    """Three distinct predeclared diagnostic criteria; GRU is descriptive."""
    keyed = {(row['arm'], row['seed'], row['horizon']): row for row in rows}
    reference = {row['horizon']: row for row in baselines}
    require(len(rows) == len(keyed) == 48 and set(keyed)
            == {(arm, seed, h) for arm in ARMS for seed in SEEDS for h in HORIZONS}, 'complete metric gate roster')
    require(len(baselines) == len(reference) == 4 and set(reference) == set(HORIZONS), 'complete reference gate roster')
    results = {}
    for arm in FACTOR_ARMS:
        criteria = {}
        for kind, horizons in (('SHORT_HORIZON_LEARNING', (1, 2)), ('BLIND_EXTRAPOLATION', (4, 8)),
                               ('OBSERVED_FILTERING_EXTRAPOLATION', (4, 8))):
            conditions = {'minimum_train': counts['train'] >= 256, 'minimum_development': counts['base'] >= 64}
            for horizon in horizons:
                if kind != 'OBSERVED_FILTERING_EXTRAPOLATION':
                    conditions[f'h{horizon}_positive_reference_mse'] = reference[horizon]['blind_cost_mse'] > 0
                    conditions[f'h{horizon}_positive_reference_regret'] = reference[horizon]['blind_regret'] > 0
                for seed in SEEDS:
                    row = keyed[arm, seed, horizon]
                    if kind != 'OBSERVED_FILTERING_EXTRAPOLATION':
                        conditions[f'{seed}_h{horizon}_half_mse'] = row['blind_cost_mse'] <= .5 * reference[horizon]['blind_cost_mse']
                        conditions[f'{seed}_h{horizon}_half_regret'] = row['blind_regret'] <= .5 * reference[horizon]['blind_regret']
                    if kind != 'BLIND_EXTRAPOLATION':
                        conditions[f'{seed}_h{horizon}_observed_kl'] = row['observed_kl'] <= .1
            if kind == 'BLIND_EXTRAPOLATION':
                for seed in SEEDS:
                    conditions[f'{seed}_h8_survival'] = keyed[arm, seed, 8]['blind_survival_mae'] <= .05
            passed = all(conditions.values())
            criteria[kind] = {'passed': passed, 'status': kind + ('_PASS' if passed else '_FAIL'), 'conditions': conditions}
        results[arm] = criteria
    return results


def validate_metadata(folder, summary, *, check=lambda: None):
    """Join opaque files and execution attestations without replaying training."""
    def read(name, *, lines=False):
        path = folder / name
        require(path.is_file() and not path.is_symlink(), 'regular saved factor metadata')
        return [json.loads(line) for line in path.read_text().splitlines()] if lines else json.loads(path.read_text())

    require(summary['version'] == 'finite-factor-learning-v1', 'factor producer version')
    compare(summary['config'], CONFIG, path='registered factor configuration')
    fits = summary['fits']
    expected_pairs = [(arm, seed) for index, seed in enumerate(SEEDS) for arm in ARMS[index:] + ARMS[:index]]
    require([(row['arm'], row['seed']) for row in fits] == expected_pairs, 'twelve completed factor fits')
    expected_files = {'config.json', 'train.npz', 'base.npz', 'train-oracle.npz', 'base-oracle.npz',
                      'oracle-train.npz', 'oracle-base.npz', 'predictions-base.npz', 'fits.jsonl',
                      'training-orders.jsonl', 'training-epochs.jsonl', 'checkpoint-barrier.json', 'prediction-times.jsonl',
                      'oracle-train-check.json', 'oracle-base-check.json'}
    expected_files |= {f'{arm}-{seed}.npz' for arm, seed in expected_pairs}
    require(set(summary['files']) == expected_files, 'complete factor producer payload roster')
    for name, descriptor in summary['files'].items():
        check()
        path = folder / name
        require(path.is_file() and not path.is_symlink(), 'regular original factor payload')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        compare(descriptor, {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}, path='payload ' + name)
    compare(read('config.json'), CONFIG, path='saved configuration')
    compare(read('fits.jsonl', lines=True), fits, path='durable fits')
    require(set(summary['dataset_counts']) == {'train', 'base'}, 'two exact dataset count records')
    retained = {}
    for split_id, split in enumerate(('train', 'base')):
        row = summary['dataset_counts'][split]
        attempts, horizon = (512, 2) if split == 'train' else (128, 8)
        require(row['version'] == 'finite-observation-world-v1' and row['epsilon'] == .12
                and row['seed_namespace'] == NAMESPACE and row['split_id'] == split_id
                and row['attempts'] == attempts and row['horizon'] == horizon, 'registered generation geometry')
        n, excluded, found = row['retained'], row['excluded_found'], row['prefix_found_by_step']
        require(type(n) is int and 0 < n <= attempts and type(excluded) is int and excluded >= 0
                and n + excluded == attempts and type(found) is list and len(found) == 8
                and all(type(v) is int and v >= 0 for v in found) and sum(found) == excluded,
                'complete fixed prefix allocation')
        require(row['initial_odor_draws'] == attempts and row['prefix_action_draws'] == 8 * attempts
                and row['prefix_event_draws'] == 8 * n + sum((i + 1) * v for i, v in enumerate(found))
                and row['forecast_action_draws'] == horizon * n, 'fixed draw accounting')
        retained[split] = n
        errors = summary['oracle_checks'][split]
        require(set(errors) == set(TARGETS) and all(type(v) in (int, float) and math.isfinite(v)
                and 0 <= v <= 1e-12 for v in errors.values()), 'every exact-control validation passed')
        compare(read('oracle-' + split + '-check.json'), errors, path='durable exact-control checks')
    require(summary['oracle_metadata']['parameter_count'] == 0
            and summary['oracle_metadata']['parameter_bytes'] == 0
            and summary['oracle_metadata']['parameters'] == {}, 'exact control has no trainable parameters')
    require(type(summary['oracle_state_sha256']) is str
            and re.fullmatch('[0-9a-f]{64}', summary['oracle_state_sha256']), 'exact control state digest')
    n, dev_n = retained['train'], retained['base']
    updates = 480 * math.ceil(n / 64)
    fit_map = {(row['arm'], row['seed']): row for row in fits}
    for row in fits:
        arm = row['arm']
        require(row['epochs'] == 480 and row['updates'] == updates and row['training_cases'] == n
                and row['training_case_exposures'] == 480 * n, 'complete fixed fit schedule')
        require(type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                'finite completed fit duration')
        for key in ('initial_state_sha256', 'final_state_sha256', 'case_order_sha256', 'fixed_buffers_sha256'):
            require(type(row[key]) is str and re.fullmatch('[0-9a-f]{64}', row[key]), 'fit state/work digest')
        for key, present in (('prefix_initial_sha256', arm in ('learned_exact', 'learned_learned')),
                             ('operator_initial_sha256', arm in ('exact_learned', 'learned_learned'))):
            require((type(row[key]) is str and re.fullmatch('[0-9a-f]{64}', row[key])) if present else row[key] is None,
                    'present and absent parameter-group digests')
        require(row['oracle_prefix_input'] is (arm == 'exact_learned'), 'declared privileged input restricted to exact-prefix arm')
        name = f"{arm}-{row['seed']}.npz"
        compare(row['checkpoint'], {'path': name, **summary['files'][name]}, path='final checkpoint')
    for seed in SEEDS:
        require(fit_map['learned_exact', seed]['prefix_initial_sha256']
                == fit_map['learned_learned', seed]['prefix_initial_sha256'], 'paired learned-prefix initialization')
        require(fit_map['exact_learned', seed]['operator_initial_sha256']
                == fit_map['learned_learned', seed]['operator_initial_sha256'], 'paired learned-operator initialization')
    barrier = {'checkpoints': [{'arm': row['arm'], 'seed': row['seed'], **row['checkpoint']} for row in fits],
               'dev_generation_count': 0, 'fit_count': 12, 'oracle_train_verified': True}
    compare(summary['checkpoint_barrier'], barrier, path='summary barrier')
    compare(read('checkpoint-barrier.json'), barrier, path='durable barrier')
    orders, epochs = read('training-orders.jsonl', lines=True), read('training-epochs.jsonl', lines=True)
    expected_epochs = [(arm, seed, epoch) for arm, seed in expected_pairs for epoch in range(480)]
    for records in (orders, epochs):
        require([(r['arm'], r['seed'], r['epoch']) for r in records] == expected_epochs, 'complete 5760-epoch journal roster')
    order_hashes = {pair: hashlib.sha256() for pair in expected_pairs}
    paired_orders = {}
    for row in orders:
        indices = row['indices']
        require(set(row) == {'arm', 'seed', 'epoch', 'indices', 'batch_size'} and row['batch_size'] == 64
                and type(indices) is list and len(indices) == n and all(type(i) is int for i in indices)
                and sorted(indices) == list(range(n)), 'complete once-per-epoch case permutation')
        key = row['seed'], row['epoch']
        if key in paired_orders:
            require(indices == paired_orders[key], 'same-seed paired case order')
        paired_orders[key] = indices
        order_hashes[row['arm'], row['seed']].update(struct.pack('<' + 'q' * n, *indices))
    for pair, digest in order_hashes.items():
        require(fit_map[pair]['case_order_sha256'] == digest.hexdigest(), 'saved case-order digest')
    for row in epochs:
        require(row['cases'] == n and row['updates'] == math.ceil(n / 64)
                and all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                        for k in ('mean_objective', 'seconds')), 'finite completed epoch work')
    times = summary['prediction_times']
    require([(row['arm'], row['seed']) for row in times] == [(arm, seed) for arm in ARMS for seed in SEEDS],
            'twelve complete evaluation views')
    compare(read('prediction-times.jsonl', lines=True), times, path='durable evaluation records')
    for row in times:
        require(row['regime'] == 'base' and row['cases'] == dev_n and row['batch_size'] == 64
                and row['oracle_prefix_input'] is (row['arm'] == 'exact_learned') and row['shuffle_offset'] == 1
                and row['model_state_before'] == row['model_state_after'] == fit_map[row['arm'], row['seed']]['final_state_sha256']
                and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                'unchanged final state and declared matched input routing')
    eval_batches = 12 * math.ceil(dev_n / 64)
    oracle_batches = math.ceil(n / 64) + math.ceil(dev_n / 64)
    expected_counts = {'train_generation_count': 1, 'dev_generation_count': 1, 'model_constructions': 12,
        'oracle_model_constructions': 1, 'fit_count': 12, 'checkpoint_writes': 12,
        'training_blind_rollouts': 12 * updates, 'training_observed_rollouts': 12 * updates,
        'optimizer_attempts': 12 * updates, 'optimizer_steps': 12 * updates, 'training_case_exposures': 12 * 480 * n,
        'evaluation_blind_rollouts': eval_batches, 'evaluation_observed_rollouts': eval_batches,
        'evaluation_shuffled_rollouts': eval_batches, 'evaluation_case_views': 12 * dev_n,
        'oracle_blind_rollouts': oracle_batches, 'oracle_observed_rollouts': oracle_batches,
        'array_decodes': 0, 'checkpoint_decodes': 0, 'external_model_calls': 0, 'native_calls': 0, 'teacher_calls': 0}
    compare(summary['counts'], expected_counts, path='complete declared executed counters')
    return {'fits': 12, 'checkpoint_files_hashed': 12, 'epochs': 5760, 'evaluation_views': 12,
            'retained': retained, 'optimizer_steps_attested': 12 * updates,
            'training_case_exposures_attested': 12 * 480 * n,
            'historical_ordering_independently_replayed': False}


def audit(folder, *, check=lambda: None):
    """Decode only a separately admitted, closed original producer's outputs."""
    import numpy as np

    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder, 'absolute original saved-output directory')
    summary_path = folder / 'summary.json'
    require(summary_path.is_file() and not summary_path.is_symlink(), 'regular producer summary')
    summary = json.loads(summary_path.read_text())
    metadata = validate_metadata(folder, summary, check=check)
    counters = {'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'optimizer_calls': 0,
                'world_or_generator_calls': 0, 'native_calls': 0}

    def load(name):
        check()
        with np.load(folder / name, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), 'unique saved array members')
            value = {key: archive[key] for key in archive.files}
        counters['array_decodes'] += 1
        return value

    records, datasets, targets, uniform = {}, {}, {}, {}
    for split in ('train', 'base'):
        data = load(split + '.npz')
        datasets[split] = data
        exact_predictions = load('oracle-' + split + '.npz')
        record, rebuilt, reference = reconstruct(data, load(split + '-oracle.npz'), exact_predictions, split, np, check=check)
        producer_errors = {name: float(np.max(np.abs(exact_predictions[name] - data[name]))) for name in TARGETS}
        compare(summary['oracle_checks'][split], producer_errors, path='independent exact-control error receipt')
        found = data['observations'] == 4
        found_cases = found.any(axis=1)
        event_draws = np.where(found_cases, found.argmax(axis=1) + 1, data['observations'].shape[1])
        require(summary['dataset_counts'][split]['forecast_found_cases'] == int(found_cases.sum())
                and summary['dataset_counts'][split]['forecast_event_draws'] == int(event_draws.sum()),
                'forecast draw counts independently reconstructed from labels')
        records[split], targets[split], uniform[split] = record, rebuilt, reference
    require(not set(datasets['train']['case_ids']) & set(datasets['base']['case_ids']), 'disjoint fresh TRAIN/DEV identities')
    predictions = load('predictions-base.npz')
    rows, baselines = rows_for({**datasets['base'], **targets['base']}, predictions, uniform['base'], np, check=check)
    row_key = lambda row: (row['arm'], row['seed'], row['horizon'])
    compare(sorted(summary['rows'], key=row_key), sorted(rows, key=row_key), path='independent factor metrics')
    compare(sorted(summary['baseline_rows'], key=lambda row: row['horizon']), baselines, path='independent uniform-state reference')
    counts = {split: len(data['case_ids']) for split, data in datasets.items()}
    compare(counts, metadata['retained'], path='array support matches saved metadata')
    return {'version': VERSION, 'agreement': True, 'technical_complete': False,
            'requires_original_supervisor_closure': True, 'counts': counters, 'metadata': metadata,
            'data_cases': counts, 'target_reconstruction': records, 'exact_oracle_agreement': True,
            'rows': rows, 'baseline_rows': baselines, 'gates': gates(rows, baselines, counts),
            'architecture_claim': False, 'latent_identification_claim': False,
            'limitations': [
                'Original process/source/payload admission belongs to the external caller.',
                'The exact prefix is a privileged posterior input and the exact operators are privileged known dynamics.',
                'Checkpoint bytes and saved work records do not replay training or prove historical execution order.',
                'Shuffled outputs are scored as saved; corresponding public/oracle prefix routing remains source-reviewed.',
                'The fixed cost readout identifies four decision signatures, not every coordinate of the eight-state posterior.',
                'Observed filtering has intervening observations and is separate from blind extrapolation.',
                'Uniform-state known dynamics is a history-ignorant reference, not an optimal history-ignorant policy.',
                'This synthetic diagnostic establishes neither convergence nor native-environment transfer or architectural novelty.']}
