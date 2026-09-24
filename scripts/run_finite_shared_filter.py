"""Shared learned prefix filters and forecast operators; admitted by the worker."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from run_finite_observation_learning import (
    FIELDS,
    _append,
    _baseline_rows,
    _config,
    _desc,
    _rows,
    _save,
    _state,
    _state_hash,
    _tensor_data,
    _write,
    require,
)

from openjev.research.finite_factor_models import make_model as factor_model
from openjev.research.finite_observation_models import metadata, objective
from openjev.research.finite_observation_world import generate_split
from openjev.research.finite_shared_filter_models import make_model as learned_model

VERSION = 'finite-shared-filter-v1'
ARMS = ('shared_filter', 'untied_filter', 'gru_prefix')
ROUTES = ('training_blind', 'training_observed', 'evaluation_blind', 'evaluation_observed', 'evaluation_shuffled')
ORACLE_FIELDS = FIELDS[:-1]
DEFAULT_CONFIG = {'seed_namespace': 422260924, 'train_attempts': 512,
    'dev_attempts': 128, 'epochs': 480, 'batch_size': 64, 'learning_rate': .003,
    'fit_seeds': [422261001, 422261002, 422261003], 'gradient_clip': 5.,
    'train_horizon': 2, 'dev_horizon': 8}


def call(model, arm, batch, observed=False):
    kwargs = {'oracle_prefix': batch['oracle_prefix']} if arm == 'exact_exact' else {}
    args = (batch['prefix'], batch['lengths'], batch['actions'])
    if observed:
        return model.observed_rollout(*args, batch['observations'], **kwargs)
    return model.blind_rollout(*args, **kwargs)


def initial_group_hash(model, prefixes):
    arrays = {name: value.detach().cpu().numpy().copy() for name, value in model.named_parameters()
              if name.startswith(prefixes)}
    return _state_hash(arrays) if arrays else None


def fixed_buffer_hash(model):
    return _state_hash({name: value.detach().cpu().numpy().copy() for name, value in model.named_buffers()})


def record_work(totals, arm, route, result):
    work = result['work']
    require(isinstance(work, dict) and all(type(v) is int and v >= 0 for v in work.values()),
            'nonnegative integer structural work counters')
    target = totals[arm][route]
    require(not target or set(target) == set(work), 'stable structural work roster')
    for name, value in work.items():
        target[name] = target.get(name, 0) + value


def prefix_operator_hash(model, arm):
    if arm == 'gru_prefix':
        return None
    value = model.observed_logits if arm == 'shared_filter' else model.prefix_observed_logits
    return _state_hash({'observed_logits': value.detach().cpu().numpy().copy()})


def train(model, arm, seed, data, config, folder, counts, structural_work, check):
    n = len(data['prefix'])
    initial = _state_hash(_state(model))
    prefix_hash = initial_group_hash(model, ('assimilation.', 'projection.')) if arm == 'gru_prefix' else None
    operator_hash = initial_group_hash(model, ('observed_logits',))
    reset_hash = initial_group_hash(model, ('reset_logits',))
    prefix_op_hash = prefix_operator_hash(model, arm)
    fixed_hash = fixed_buffer_hash(model)
    order_hash = hashlib.sha256()
    started = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    updates = 0
    for epoch in range(config['epochs']):
        check()
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, 818]))).permutation(n)
        order_hash.update(np.asarray(order, dtype='<i8').tobytes())
        _append(folder / 'training-orders.jsonl', {'arm': arm, 'seed': seed, 'epoch': epoch,
                 'indices': order.tolist(), 'batch_size': config['batch_size']})
        loss_sum = 0.
        epoch_start = time.perf_counter()
        for start in range(0, n, config['batch_size']):
            check()
            indices = torch.from_numpy(order[start:start + config['batch_size']].copy())
            batch = {name: value[indices] for name, value in data.items()}
            optimizer.zero_grad(set_to_none=True)
            counts['training_blind_rollouts'] += 1
            blind = call(model, arm, batch)
            record_work(structural_work, arm, 'training_blind', blind)
            counts['training_observed_rollouts'] += 1
            observed = call(model, arm, batch, observed=True)
            record_work(structural_work, arm, 'training_observed', observed)
            loss = objective(blind, observed, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            counts['optimizer_attempts'] += 1
            optimizer.step()
            counts['optimizer_steps'] += 1
            counts['training_case_exposures'] += len(indices)
            updates += 1
            loss_sum += float(loss.detach()) * len(indices)
            check()
        _append(folder / 'training-epochs.jsonl', {'arm': arm, 'seed': seed, 'epoch': epoch,
            'cases': n, 'updates': math.ceil(n / config['batch_size']),
            'mean_objective': loss_sum / n, 'seconds': time.perf_counter() - epoch_start})
    require(fixed_buffer_hash(model) == fixed_hash, 'fixed readout/operator buffers unchanged by training')
    final = _state(model)
    checkpoint = folder / f'{arm}-{seed}.npz'
    _save(checkpoint, final)
    counts['checkpoint_writes'] += 1
    row = {'arm': arm, 'seed': seed, 'epochs': config['epochs'], 'updates': updates,
        'training_cases': n, 'training_case_exposures': n * config['epochs'],
        'seconds': time.perf_counter() - started,
        'initial_state_sha256': initial, 'prefix_initial_sha256': prefix_hash,
        'operator_initial_sha256': operator_hash, 'reset_initial_sha256': reset_hash,
        'prefix_operator_initial_sha256': prefix_op_hash, 'fixed_buffers_sha256': fixed_hash,
        'final_state_sha256': _state_hash(final), 'parameter_metadata': metadata(model),
        'case_order_sha256': order_hash.hexdigest(),
        'oracle_prefix_input': False,
        'timing_scope': 'Adam construction, all training, durable journals, buffer checks and final checkpoint write',
        'checkpoint': {'path': checkpoint.name, **_desc(checkpoint)}}
    _append(folder / 'fits.jsonl', row)
    counts['fit_count'] += 1
    check()
    return row


def predict(model, arm, data, oracle, config, counts, check, *, shuffled, structural_work=None):
    public = {name: torch.from_numpy(data[name].copy()) for name in ('prefix', 'lengths', 'actions', 'observations')}
    if arm == 'exact_exact':
        public['oracle_prefix'] = torch.from_numpy(oracle.copy())
    n, horizon = data['actions'].shape
    names = FIELDS if shuffled else ORACLE_FIELDS
    arrays = {name: np.empty((n, horizon, 5 if name == 'observed_probabilities' else 4), dtype=np.float64)
              if name.endswith('costs') or name == 'observed_probabilities'
              else np.empty((n, horizon), dtype=np.float64) for name in names}
    original_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, n, config['batch_size']):
                check()
                stop = min(n, start + config['batch_size'])
                batch = {name: value[start:stop] for name, value in public.items()}
                counter = 'evaluation' if shuffled else 'oracle'
                counts[counter + '_blind_rollouts'] += 1
                blind = call(model, arm, batch)
                counts[counter + '_observed_rollouts'] += 1
                observed = call(model, arm, batch, observed=True)
                outputs = {'blind_costs': blind['cost_contrasts'], 'blind_survival': blind['survival_mass'],
                    'observed_costs': observed['cost_contrasts'], 'observed_survival': observed['survival_mass'],
                    'observed_probabilities': observed['probabilities']}
                if shuffled:
                    record_work(structural_work, arm, 'evaluation_blind', blind)
                    record_work(structural_work, arm, 'evaluation_observed', observed)
                    indices = torch.from_numpy((np.arange(start, stop) + 1) % n)
                    # Shift only public history; learned models never receive oracle state.
                    moved = {**batch, **{name: public[name][indices]
                                        for name in ('prefix', 'lengths')}}
                    counts['evaluation_shuffled_rollouts'] += 1
                    shifted = call(model, arm, moved)
                    record_work(structural_work, arm, 'evaluation_shuffled', shifted)
                    outputs['shuffled_blind_costs'] = shifted['cost_contrasts']
                    counts['evaluation_case_views'] += stop - start
                for name, value in outputs.items():
                    result = value.detach().cpu().numpy()
                    require(result.dtype == np.float64 and result.shape == arrays[name][start:stop].shape
                            and np.isfinite(result).all(), 'finite exact prediction shape/dtype')
                    arrays[name][start:stop] = result
                check()
    finally:
        model.train(original_mode)
    return arrays


def run(folder: Path, config: dict, check):
    config = _config({**DEFAULT_CONFIG, **config})
    check()
    folder.mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(('train_generation_count', 'dev_generation_count', 'model_constructions',
        'oracle_model_constructions', 'fit_count', 'checkpoint_writes', 'training_blind_rollouts',
        'training_observed_rollouts', 'optimizer_attempts', 'optimizer_steps', 'training_case_exposures',
        'evaluation_blind_rollouts', 'evaluation_observed_rollouts', 'evaluation_shuffled_rollouts',
        'evaluation_case_views', 'oracle_blind_rollouts', 'oracle_observed_rollouts', 'array_decodes',
        'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls'), 0)
    fits, rows, models, prediction_times, dataset_counts, generation_seconds = [], [], {}, [], {}, {}
    structural_work = {arm: {route: {} for route in ROUTES} for arm in ARMS}
    stage = 'initialize'
    started = time.perf_counter()
    try:
        _write(folder / 'config.json', config)
        counts['oracle_model_constructions'] += 1
        reference = factor_model('exact_exact', 0)
        require(not list(reference.parameters()), 'untrained reference has no learned parameters')
        reference_hash = _state_hash(_state(reference))
        reference_checks = {}

        def generate(split, regime):
            check()
            counts['train_generation_count' if split == 0 else 'dev_generation_count'] += 1
            start = time.perf_counter()
            generated = generate_split(split, config['train_attempts' if split == 0 else 'dev_attempts'],
                config['train_horizon' if split == 0 else 'dev_horizon'],
                seed_namespace=config['seed_namespace'], include_oracle=True)
            data = generated['data']
            oracle = generated['oracle']['prefix_beliefs'][:, -1].copy()
            require(len(oracle) > 0, 'retained cases required')
            _save(folder / (regime + '.npz'), data)
            _save(folder / (regime + '-oracle.npz'), {'prefix_state': oracle})
            reference_predictions = predict(reference, 'exact_exact', data, oracle, config, counts, check, shuffled=False)
            errors = {name: float(np.max(np.abs(reference_predictions[name] - data[name]))) for name in ORACLE_FIELDS}
            require(all(error <= 1e-12 for error in errors.values()), 'exact reference reproduces every target before learning interpretation')
            require(reference_hash == _state_hash(_state(reference)), 'oracle model state unchanged')
            reference_checks[regime] = errors
            _save(folder / ('oracle-' + regime + '.npz'), reference_predictions)
            _write(folder / ('oracle-' + regime + '-check.json'), errors)
            generation_seconds[regime] = time.perf_counter() - start
            dataset_counts[regime] = generated['counts']
            check()
            return data, oracle

        stage = 'TRAIN generation and exact control'
        train_data, _train_oracle = generate(0, 'train')
        training = _tensor_data(train_data)
        require('oracle_prefix' not in training, 'no oracle state in learned training batches')
        for seed_index, seed in enumerate(config['fit_seeds']):
            offset = seed_index % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                stage = f'fit {arm} {seed}'
                check()
                counts['model_constructions'] += 1
                model = learned_model(arm, seed)
                fits.append(train(model, arm, seed, training, config, folder, counts, structural_work, check))
                models[arm, seed] = model
        expected = len(ARMS) * len(config['fit_seeds'])
        require(len(fits) == counts['fit_count'] == counts['checkpoint_writes'] == expected
                and counts['dev_generation_count'] == 0, 'all final fits precede DEV')
        barrier = {'checkpoints': [{'arm': r['arm'], 'seed': r['seed'], **r['checkpoint']} for r in fits],
            'dev_generation_count': 0, 'fit_count': expected, 'oracle_train_verified': True}
        _write(folder / 'checkpoint-barrier.json', barrier)
        stage = 'DEV generation and exact control'
        data, oracle = generate(1, 'base')
        baseline_rows = _baseline_rows(data, 'base', check)
        predictions = {}
        for arm in ARMS:
            for seed in config['fit_seeds']:
                stage = f'evaluate {arm} {seed}'
                check()
                model = models[arm, seed]
                before = _state_hash(_state(model))
                start = time.perf_counter()
                arrays = predict(model, arm, data, oracle, config, counts, check, shuffled=True, structural_work=structural_work)
                seconds = time.perf_counter() - start
                after = _state_hash(_state(model))
                require(before == after, 'model state unchanged by evaluation')
                timing = {'arm': arm, 'seed': seed, 'regime': 'base', 'seconds': seconds,
                    'cases': len(oracle), 'batch_size': config['batch_size'],
                    'model_state_before': before, 'model_state_after': after,
                    'oracle_prefix_input': False, 'shuffle_offset': 1,
                    'timing_scope': 'input copies, all three forwards, guards and prediction copies; excludes state hashes and scalar metrics'}
                prediction_times.append(timing)
                _append(folder / 'prediction-times.jsonl', timing)
                predictions.update({f'{arm}__{seed}__{name}': value for name, value in arrays.items()})
                rows.extend(_rows(data, arrays, arm, seed, 'base', check))
        _save(folder / 'predictions-base.npz', predictions)
        summary = {'version': VERSION, 'config': config, 'counts': counts, 'rows': rows,
            'baseline_rows': baseline_rows, 'fits': fits, 'prediction_times': prediction_times,
            'structural_work': structural_work,
            'structural_work_scope': 'Summed actual forward calls and rows by arm across all seeds; excludes validation FLOPs, backward operations, optimizer work and memory allocation.',
            'checkpoint_barrier': barrier, 'dataset_counts': dataset_counts,
            'generation_seconds': generation_seconds,
            'generation_timing_scope': 'world generation, data and oracle saves, exact-control forwards and validation',
            'oracle_checks': reference_checks, 'oracle_metadata': metadata(reference),
            'oracle_state_sha256': reference_hash,
            'seconds_before_summary_write': time.perf_counter() - started,
            'files': {p.name: _desc(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        _write(folder / 'summary.json', summary)
        check()
        return summary
    except BaseException as error:
        _write(folder / 'failure.json', {'version': VERSION, 'stage': stage,
            'error': repr(error), 'counts': counts, 'completed_fits': counts['fit_count'],
            'seconds': time.perf_counter() - started})
        raise
