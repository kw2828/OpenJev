"""Prospective five-arm synthetic diagnostic; admission is caller-owned.

run() creates one exclusive output directory. No DEV case is generated until
all final checkpoints and their barrier have been durably saved. Models see
only public prefix tokens, lengths, actions and observed labels. Exact target
arrays enter the loss, never the model interface. There is no model selection,
early stopping, checkpoint loading or continuation of a previous attempt.

Reported local timings use perf_counter and include the stated operations;
the caller supplies the independent resource/deadline check and process closure.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research.finite_observation_models import ARMS, make_model, metadata, objective
from openjev.research.finite_observation_world import generate_split, world

VERSION = 'finite-observation-learning-v1'
FIELDS = ('blind_costs', 'blind_survival', 'observed_costs', 'observed_survival',
          'observed_probabilities', 'shuffled_blind_costs')
HORIZONS = (1, 2, 4, 8)
DEFAULT_CONFIG = {
    'seed_namespace': 420260924, 'train_attempts': 512, 'dev_attempts': 128,
    'epochs': 48, 'batch_size': 64, 'learning_rate': .003,
    'fit_seeds': [420261001, 420261002, 420261003], 'gradient_clip': 5.,
    'train_horizon': 2, 'dev_horizon': 8,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _config(config):
    require(type(config) is dict and not (set(config) - set(DEFAULT_CONFIG)), 'declared config fields')
    result = {**DEFAULT_CONFIG, **config}
    for key in ('train_attempts', 'dev_attempts', 'epochs', 'batch_size'):
        require(type(result[key]) is int and result[key] > 0, 'positive integer ' + key)
    require(type(result['seed_namespace']) is int and 0 <= result['seed_namespace'] < 2**32,
            'uint32 data namespace')
    for key in ('learning_rate', 'gradient_clip'):
        require(type(result[key]) in (int, float) and math.isfinite(result[key]) and result[key] > 0,
                'finite positive ' + key)
    seeds = result['fit_seeds']
    require(type(seeds) in (list, tuple) and len(seeds) > 0
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
            and len(set(seeds)) == len(seeds), 'distinct uint32 fit seeds')
    require(type(result['train_horizon']) is int and result['train_horizon'] == 2
            and type(result['dev_horizon']) is int and result['dev_horizon'] == 8, 'fixed TRAIN H2 / DEV H8')
    result['fit_seeds'] = list(seeds)
    return result


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _save(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _desc(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def _state(model):
    arrays = {}
    for name, value in model.state_dict().items():
        data = value.detach().cpu().contiguous().numpy().copy()
        require(np.isfinite(data).all(), 'finite saved model state: ' + name)
        arrays[name] = data
    return arrays


def _state_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        header = json.dumps([name, value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def _tensor_data(data):
    names = ('prefix', 'lengths', 'actions', 'observations', 'blind_costs', 'blind_survival',
             'observed_costs', 'observed_survival', 'observed_probabilities')
    return {name: torch.from_numpy(data[name].copy()) for name in names}


def _train(model, arm, seed, data, config, folder, counts, check):
    n = len(data['prefix'])
    require(n > 0, 'at least one retained TRAIN case')
    initial = _state(model)
    initial_hash = _state_hash(initial)
    shared_hash = _state_hash({name: value for name, value in initial.items() if name != 'core.blind_logits'})
    target_names = ('blind_costs', 'blind_survival', 'observed_costs', 'observed_survival', 'observed_probabilities')
    order_hash = hashlib.sha256()
    started = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    updates = 0
    for epoch in range(config['epochs']):
        check()
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, 818]))).permutation(n)
        order_hash.update(np.asarray(order, dtype='<i8').tobytes())
        _append(folder / 'training-orders.jsonl', {
            'arm': arm, 'seed': seed, 'epoch': epoch, 'indices': order.tolist(),
            'batch_size': config['batch_size'],
        })
        weighted_loss = 0.
        epoch_started = time.perf_counter()
        for start in range(0, n, config['batch_size']):
            check()
            indices = torch.from_numpy(order[start:start + config['batch_size']].copy())
            batch = {name: values[indices] for name, values in data.items()}
            optimizer.zero_grad(set_to_none=True)
            counts['training_blind_rollouts'] += 1
            blind = model.blind_rollout(batch['prefix'], batch['lengths'], batch['actions'])
            counts['training_observed_rollouts'] += 1
            observed = model.observed_rollout(batch['prefix'], batch['lengths'], batch['actions'], batch['observations'])
            loss = objective(blind, observed, {name: batch[name] for name in target_names})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            counts['optimizer_attempts'] += 1
            optimizer.step()
            counts['optimizer_steps'] += 1
            counts['training_case_exposures'] += len(indices)
            updates += 1
            weighted_loss += float(loss.detach()) * len(indices)
            check()
        _append(folder / 'training-epochs.jsonl', {
            'arm': arm, 'seed': seed, 'epoch': epoch, 'cases': n,
            'updates': math.ceil(n / config['batch_size']), 'mean_objective': weighted_loss / n,
            'seconds': time.perf_counter() - epoch_started,
        })
    final = _state(model)
    checkpoint = folder / f'{arm}-{seed}.npz'
    _save(checkpoint, final)
    counts['checkpoint_writes'] += 1
    record = {
        'arm': arm, 'seed': seed, 'epochs': config['epochs'], 'updates': updates,
        'training_cases': n, 'training_case_exposures': n * config['epochs'],
        'seconds': time.perf_counter() - started,
        'timing_scope': 'Adam construction, all epochs, durable epoch/order logs, final state snapshot and checkpoint write',
        'initial_state_sha256': initial_hash, 'shared_initial_state_sha256': shared_hash,
        'final_state_sha256': _state_hash(final), 'parameter_metadata': metadata(model),
        'case_order_sha256': order_hash.hexdigest(),
        'checkpoint': {'path': checkpoint.name, **_desc(checkpoint)},
    }
    _append(folder / 'fits.jsonl', record)
    counts['fit_count'] += 1
    check()
    return record


def _prediction_arrays(model, data, config, counts, check):
    n, horizon = data['actions'].shape
    result = {name: np.empty((n, horizon, 5 if name == 'observed_probabilities' else 4), dtype=np.float64)
              if name.endswith('costs') or name == 'observed_probabilities'
              else np.empty((n, horizon), dtype=np.float64) for name in FIELDS}
    # Each case receives the next complete prefix in the same retained regime.
    # Labels, action block and targets remain those of the original case.
    shuffled_indices = (np.arange(n) + 1) % n
    public = {name: torch.from_numpy(data[name].copy()) for name in ('prefix', 'lengths', 'actions', 'observations')}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, n, config['batch_size']):
                check()
                stop = min(start + config['batch_size'], n)
                prefix, lengths, actions, observations = (public[name][start:stop]
                    for name in ('prefix', 'lengths', 'actions', 'observations'))
                counts['evaluation_blind_rollouts'] += 1
                blind = model.blind_rollout(prefix, lengths, actions)
                counts['evaluation_observed_rollouts'] += 1
                observed = model.observed_rollout(prefix, lengths, actions, observations)
                indices = torch.from_numpy(shuffled_indices[start:stop].copy())
                counts['evaluation_shuffled_rollouts'] += 1
                shuffled = model.blind_rollout(public['prefix'][indices], public['lengths'][indices], actions)
                values = {
                    'blind_costs': blind['cost_contrasts'], 'blind_survival': blind['survival_mass'],
                    'observed_costs': observed['cost_contrasts'], 'observed_survival': observed['survival_mass'],
                    'observed_probabilities': observed['probabilities'], 'shuffled_blind_costs': shuffled['cost_contrasts'],
                }
                for name, value in values.items():
                    array = value.detach().cpu().numpy()
                    require(array.dtype == np.float64 and array.shape == result[name][start:stop].shape
                            and np.isfinite(array).all(), 'finite float64 prediction: ' + name)
                    result[name][start:stop] = array
                counts['evaluation_case_views'] += stop - start
                check()
    finally:
        model.train(was_training)
    return result


def _regret(target, prediction, *, baseline=False):
    if baseline:
        chosen = np.argmax(prediction <= prediction.min(axis=1, keepdims=True) + 1e-12, axis=1)
    else:
        chosen = prediction.argmin(axis=1)
    return float(np.mean(target[np.arange(len(target)), chosen] - target.min(axis=1), dtype=np.float64))


def _kl(target, prediction):
    require(target.shape == prediction.shape and np.isfinite(target).all() and np.isfinite(prediction).all()
            and (target >= 0).all() and (prediction >= 0).all()
            and np.all(np.abs(target.sum(axis=-1) - 1) <= 1e-12)
            and np.all(np.abs(prediction.sum(axis=-1) - 1) <= 1e-6), 'finite probability rows')
    positive = target > 0
    require((prediction[positive] > 0).all(), 'positive oracle probability requires positive prediction')
    terms = np.zeros_like(target)
    terms[positive] = target[positive] * (np.log(target[positive]) - np.log(prediction[positive]))
    return float(np.mean(terms.sum(axis=-1), dtype=np.float64))


def _rows(data, predictions, arm, seed, regime, check):
    rows = []
    n = len(data['prefix'])
    require(n > 0, 'nonempty retained DEV regime')
    for horizon in HORIZONS:
        check()
        h = horizon - 1
        target, predicted = data['blind_costs'][:, h], predictions['blind_costs'][:, h]
        rows.append({
            'arm': arm, 'seed': seed, 'regime': regime, 'horizon': horizon, 'cases': n,
            'blind_cost_mse': float(np.mean((target - predicted) ** 2, dtype=np.float64)),
            'blind_regret': _regret(target, predicted),
            'blind_survival_mae': float(np.mean(np.abs(data['blind_survival'][:, h]
                                                       - predictions['blind_survival'][:, h]), dtype=np.float64)),
            'observed_cost_mse': float(np.mean((data['observed_costs'][:, h]
                                                - predictions['observed_costs'][:, h]) ** 2, dtype=np.float64)),
            'observed_survival_mae': float(np.mean(np.abs(data['observed_survival'][:, h]
                                                          - predictions['observed_survival'][:, h]), dtype=np.float64)),
            'observed_kl': _kl(data['observed_probabilities'][:, h], predictions['observed_probabilities'][:, h]),
            'shuffled_blind_regret': _regret(target, predictions['shuffled_blind_costs'][:, h]),
        })
    require(all(math.isfinite(value) for row in rows for value in row.values()
                if type(value) is float), 'finite reported metrics')
    return rows


def _baseline_rows(data, regime, check):
    operators = world(.12 if regime == 'base' else .30)
    n = len(data['prefix'])
    state = np.full((n, 8), 1 / 8, dtype=np.float64)
    rows = []
    for h in range(data['actions'].shape[1]):
        check()
        state = np.einsum('bij,bj->bi', operators['A'][data['actions'][:, h]], state)
        if h + 1 in HORIZONS:
            predicted = state @ operators['costs'].T
            target = data['blind_costs'][:, h]
            rows.append({'regime': regime, 'horizon': h + 1, 'cases': n,
                         'blind_cost_mse': float(np.mean((target - predicted) ** 2, dtype=np.float64)),
                         'blind_regret': _regret(target, predicted, baseline=True)})
    return rows


def run(folder: Path, config: dict, check: callable) -> dict:
    """Execute one caller-admitted attempt; no outputs exist before this call.

    Defaults describe the prospective 15-fit diagnostic. A caller may reduce
    counts/epochs and supply distinct engineering seeds for qualification.
    Statistical/continuation decisions belong to the independent saved-output
    audit, not this producer. Every arm, seed, regime and declared horizon is
    retained regardless of its measured performance.
    """
    config = _config(config)
    require(isinstance(folder, Path) and callable(check), 'Path output and callable boundary check')
    check()
    folder.mkdir(parents=True, exist_ok=False)
    counts = {name: 0 for name in (
        'train_generation_count', 'dev_generation_count', 'model_constructions', 'fit_count',
        'training_blind_rollouts', 'training_observed_rollouts', 'optimizer_attempts', 'optimizer_steps',
        'training_case_exposures', 'checkpoint_writes', 'evaluation_blind_rollouts',
        'evaluation_observed_rollouts', 'evaluation_shuffled_rollouts', 'evaluation_case_views',
        'array_decodes', 'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls')}
    fits, prediction_times, rows, baseline_rows = [], [], [], []
    dataset_counts, generation_seconds = {}, {}
    models = {}
    stage = 'TRAIN generation'
    started = time.perf_counter()
    try:
        _write(folder / 'config.json', config)
        check()
        counts['train_generation_count'] += 1
        generation_started = time.perf_counter()
        train = generate_split(0, config['train_attempts'], config['train_horizon'],
                               seed_namespace=config['seed_namespace'])
        generation_seconds['train'] = time.perf_counter() - generation_started
        dataset_counts['train'] = train['counts']
        _save(folder / 'train.npz', train['data'])
        training = _tensor_data(train['data'])
        require(len(training['prefix']) > 0, 'retained TRAIN support')
        check()
        for seed_index, seed in enumerate(config['fit_seeds']):
            offset = seed_index % len(ARMS)
            arm_order = ARMS[offset:] + ARMS[:offset]
            for arm in arm_order:
                stage = f'fit {arm} {seed}'
                check()
                counts['model_constructions'] += 1
                model = make_model(arm, seed)
                fits.append(_train(model, arm, seed, training, config, folder, counts, check))
                models[arm, seed] = model
        stage = 'checkpoint barrier'
        expected = len(ARMS) * len(config['fit_seeds'])
        require(len(fits) == counts['checkpoint_writes'] == expected and counts['dev_generation_count'] == 0,
                'all final fits precede any DEV generation')
        barrier = {'checkpoints': [{'arm': row['arm'], 'seed': row['seed'], **row['checkpoint']} for row in fits],
                   'dev_generation_count': 0, 'fit_count': expected}
        _write(folder / 'checkpoint-barrier.json', barrier)
        check()
        for split_id, regime in ((1, 'base'), (2, 'shift')):
            stage = regime + ' generation'
            check()
            counts['dev_generation_count'] += 1
            generation_started = time.perf_counter()
            generated = generate_split(split_id, config['dev_attempts'], config['dev_horizon'],
                                       seed_namespace=config['seed_namespace'])
            generation_seconds[regime] = time.perf_counter() - generation_started
            dataset_counts[regime] = generated['counts']
            data = generated['data']
            require(len(data['prefix']) > 0, 'retained DEV support: ' + regime)
            _save(folder / (regime + '.npz'), data)
            baseline_rows.extend(_baseline_rows(data, regime, check))
            predictions = {}
            for arm in ARMS:
                for seed in config['fit_seeds']:
                    stage = f'evaluate {regime} {arm} {seed}'
                    check()
                    model = models[arm, seed]
                    before = _state_hash(_state(model))
                    prediction_started = time.perf_counter()
                    arrays = _prediction_arrays(model, data, config, counts, check)
                    prediction_seconds = time.perf_counter() - prediction_started
                    after = _state_hash(_state(model))
                    require(before == after, 'model state unchanged by evaluation')
                    timing = {'arm': arm, 'seed': seed, 'regime': regime, 'seconds': prediction_seconds,
                              'cases': len(data['prefix']), 'batch_size': config['batch_size'],
                              'model_state_before': before, 'model_state_after': after,
                              'timing_scope': 'public tensor copies, all blind/observed/shuffled forwards, validation and output copies; excludes state hashes and scalar metrics'}
                    prediction_times.append(timing)
                    _append(folder / 'prediction-times.jsonl', timing)
                    predictions.update({f'{arm}__{seed}__{name}': value for name, value in arrays.items()})
                    rows.extend(_rows(data, arrays, arm, seed, regime, check))
            require(len(predictions) == expected * len(FIELDS), 'complete prediction fields')
            _save(folder / ('predictions-' + regime + '.npz'), predictions)
            check()
        stage = 'summary'
        summary = {
            'version': VERSION, 'config': config, 'counts': counts,
            'rows': rows, 'baseline_rows': baseline_rows, 'fits': fits,
            'prediction_times': prediction_times, 'checkpoint_barrier': barrier,
            'dataset_counts': dataset_counts, 'generation_seconds': generation_seconds,
            'seconds_before_summary_write': time.perf_counter() - started,
            'baseline_tie_rule': 'lowest index within 1e-12 of minimum; candidate and shuffled predictions use raw argmin',
            'files': {path.name: _desc(path) for path in sorted(folder.iterdir()) if path.is_file()},
        }
        _write(folder / 'summary.json', summary)
        check()
        return summary
    except BaseException as error:
        failure = {'version': VERSION, 'stage': stage, 'error_type': type(error).__name__,
                   'error': str(error), 'counts': counts, 'completed_fits': counts['fit_count'],
                   'completed_prediction_views': len(prediction_times),
                   'seconds': time.perf_counter() - started}
        try:
            _write(folder / 'failure.json', failure)
        except OSError:
            pass
        raise
