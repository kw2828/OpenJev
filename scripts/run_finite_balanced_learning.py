"""Fresh three-arm transition comparison under an identical prefix/joint schedule.

Adapted from the frozen allocation runner without changing its update, data or
rollback rules. Only the probability construction and initial matching differ.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from run_finite_expected_count_learning import (
    ORACLE_FIELDS,
    ROUTES,
    predict,
    record_work,
)
from run_finite_observation_learning import (
    _append,
    _baseline_rows,
    _desc,
    _rows,
    _save,
    _state,
    _state_hash,
    _tensor_data,
    _write,
    require,
)
from run_finite_observation_learning import _config as _base_config

from openjev.research.finite_balanced_models import ARMS, DYNAMICS, prefix_predictions, torch_objective
from openjev.research.finite_balanced_models import make_model as learned_model
from openjev.research.finite_factor_models import make_model as factor_model
from openjev.research.finite_observation_models import metadata, objective
from openjev.research.finite_prefix_learning import generate_attempt_split
from openjev.research.finite_training_allocation import Hooks, Snapshot, run_allocation

VERSION = 'finite-balanced-learning-v1'
DEFAULT_CONFIG = {'seed_namespace': 431260924, 'train_attempts': 512, 'dev_attempts': 128,
    'batch_size': 64, 'learning_rate': .003, 'fit_seeds': [431261001, 431261002, 431261003],
    'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
    'stage1_seconds': 10., 'total_seconds': 40.}
UPDATE_WORK = ('joint_updates', 'prefix_updates', 'joint_attempt_exposures',
    'joint_case_exposures', 'joint_event_exposures', 'prefix_event_exposures',
    'joint_blind_rollouts', 'joint_observed_rollouts', 'joint_prefix_rollouts',
    'prefix_full_rollouts', 'zero_endpoint_batches', 'backward_passes', 'adam_steps')


def _config(config):
    require(type(config) is dict and set(config) == set(DEFAULT_CONFIG), 'declared allocation settings')
    values = dict(config)
    first, total = values.pop('stage1_seconds'), values.pop('total_seconds')
    require(type(first) is float and type(total) is float and math.isfinite(first)
            and math.isfinite(total) and 0 < first < total <= 60., 'ordered bounded eligibility budgets')
    checked = _base_config({**values, 'epochs': 1})
    del checked['epochs']
    return {**checked, 'stage1_seconds': first, 'total_seconds': total}


def optimizer_payload(value):
    """Lossless finite JSON encoding of an Adam state dict, without pickle."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        require(np.isfinite(array).all(), 'finite optimizer tensor')
        return {'kind': 'tensor', 'dtype': str(value.dtype), 'shape': list(value.shape),
                'values': array.tolist()}
    if type(value) is dict:
        require(len({str(k) for k in value}) == len(value), 'unambiguous optimizer keys')
        return {str(k): optimizer_payload(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return [optimizer_payload(v) for v in value]
    require(value is None or type(value) in (str, bool, int, float), 'optimizer JSON primitive')
    require(type(value) is not float or math.isfinite(value), 'finite optimizer scalar')
    return value


def optimizer_digest(state):
    payload = optimizer_payload(state)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def tensor_storage(value):
    if isinstance(value, torch.Tensor):
        return 1, value.numel() * value.element_size()
    if type(value) in (dict, list, tuple):
        parts = value.values() if type(value) is dict else value
        counts = [tensor_storage(v) for v in parts]
        return sum(x[0] for x in counts), sum(x[1] for x in counts)
    return 0, 0


def predict_prefix(model, arm, seed, data, config, counts, structural_work, check):
    n = len(data['prefix'])
    arrays = {'probabilities': np.zeros((n, 9, 5), dtype=np.float64),
              'nll': np.zeros((n, 9), dtype=np.float64)}
    original_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, n, config['batch_size']):
                check()
                stop = min(n, start + config['batch_size'])
                result = prefix_predictions(model, torch.from_numpy(data['prefix'][start:stop].copy()),
                                             torch.from_numpy(data['lengths'][start:stop].copy()))
                counts['evaluation_prefix_rollouts'] += 1
                counts['evaluation_prefix_event_views'] += int(data['event_mask'][start:stop].sum())
                record_work(structural_work, arm, 'evaluation_prefix', result)
                for name, target in arrays.items():
                    value = result[name].detach().cpu().numpy()
                    require(value.dtype == np.float64 and np.isfinite(value).all()
                            and value.shape == target[start:stop].shape, 'finite prefix prediction array')
                    target[start:stop] = value
    finally:
        model.train(original_mode)
    events = int(data['event_mask'].sum())
    row = {'arm': arm, 'seed': seed, 'attempts': n, 'valid_events': events,
           'mean_nll': float(arrays['nll'].sum() / events)}
    return arrays, row


def train(arm, seed, data, prefixes, config, folder, counts, structural_work, check):
    wrapper_start = time.perf_counter()
    context, orders = {}, {}
    n = survivors = events = batches = None

    def validate():
        nonlocal n, survivors, events, batches
        n, survivors = len(prefixes['prefix']), len(data['prefix'])
        events = int(prefixes['event_mask'].sum())
        batches = math.ceil(n / config['batch_size'])
        require(n > 0 and survivors > 0 and events >= n, 'positive fixed dataset denominators')
        require('oracle_prefix' not in data and 'oracle_prefix' not in prefixes,
                'public histories and target supervision only')
        start = time.perf_counter()
        counts['model_constructions'] += 1
        context['model'] = learned_model(arm, seed, check=check)
        context['construction_seconds'] = time.perf_counter() - start
        context['parameter_metadata'] = metadata(context['model'])
        return {'parameter_metadata': context['parameter_metadata'], 'attempts': n,
                'eligible': survivors, 'events': events, 'batch_size': config['batch_size'],
                'seed': seed, 'hidden_state_input': False}

    def optimizer(kind):
        model = context['model']
        parameters = [getattr(model, name) for name in DYNAMICS] if kind == 'prefix' else list(model.parameters())
        return torch.optim.Adam(parameters, lr=config['learning_rate'])

    def snapshot_model():
        value = _state(context['model'])
        return Snapshot(value, _state_hash(value), len(value), sum(x.nbytes for x in value.values()))

    def restore_model(value):
        context['model'].load_state_dict({name: torch.from_numpy(array.copy()) for name, array in value.items()})

    def snapshot_optimizer(opt):
        value = copy.deepcopy(opt.state_dict())
        tensors, size = tensor_storage(value)
        return Snapshot(value, optimizer_digest(value), tensors, size)

    def checkpoint(label, opt, cursor):
        state = _state(context['model'])
        model_path = folder / f'{label}-{arm}-{seed}.npz'
        opt_path = folder / f'{label}-optimizer-{arm}-{seed}.json'
        _save(model_path, state)
        _write(opt_path, optimizer_payload(opt.state_dict()))
        counts['checkpoint_writes'] += 1
        counts['optimizer_checkpoint_writes'] += 1
        return {'label': label, 'model': {'path': model_path.name, **_desc(model_path)},
                'optimizer': {'path': opt_path.name, **_desc(opt_path)},
                'model_state_sha256': _state_hash(state),
                'optimizer_state_sha256': optimizer_digest(opt.state_dict()), 'joint_cursor': cursor}

    def update(kind, opt, cursor):
        model = context['model']
        work = dict.fromkeys(UPDATE_WORK, 0)
        opt.zero_grad(set_to_none=True)
        if kind == 'prefix':
            require(cursor is None, 'prefix has no joint cursor')
            result = torch_objective(model, prefixes['prefix'], prefixes['lengths'], .001)
            loss = result['loss']
            parameters = [getattr(model, name) for name in DYNAMICS]
            record_work(structural_work, arm, 'training_prefix', result)
            work.update(prefix_updates=1, prefix_event_exposures=events, prefix_full_rollouts=1)
            diagnostics = {'kind': kind, 'valid_events': events, 'attempts': n,
                           'pseudocount': .001, 'cost_head_updated': False}
            counts['training_prefix_rollouts'] += 1
            counts['training_prefix_event_exposures'] += events
        else:
            require(kind == 'joint' and type(cursor) is int and cursor >= 0, 'accepted joint cursor')
            epoch, offset = divmod(cursor, batches)
            if epoch not in orders:
                order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, 818]))).permutation(n)
                orders[epoch] = order
                _append(folder / 'training-orders.jsonl', {'arm': arm, 'seed': seed, 'epoch': epoch,
                         'indices': order.tolist(), 'batch_size': config['batch_size']})
                counts['epoch_order_generations'] += 1
            selected = orders[epoch][offset * config['batch_size']:(offset + 1) * config['batch_size']]
            indices = torch.from_numpy(selected.copy())
            b = len(indices)
            endpoint_rows = prefixes['endpoint_rows'][indices]
            eligible = endpoint_rows[endpoint_rows >= 0]
            parameters = list(model.parameters())
            loss = sum(parameter.sum() * 0 for parameter in parameters)
            if len(eligible):
                batch = {name: value[eligible] for name, value in data.items()}
                blind = model.blind_rollout(batch['prefix'], batch['lengths'], batch['actions'])
                observed = model.observed_rollout(batch['prefix'], batch['lengths'], batch['actions'], batch['observations'])
                record_work(structural_work, arm, 'training_blind', blind)
                record_work(structural_work, arm, 'training_observed', observed)
                loss = loss + objective(blind, observed, batch) * len(eligible) / survivors
                work.update(joint_blind_rollouts=1, joint_observed_rollouts=1)
                counts['training_blind_rollouts'] += 1
                counts['training_observed_rollouts'] += 1
            else:
                work['zero_endpoint_batches'] = 1
                counts['zero_endpoint_batches'] += 1
            result = prefix_predictions(model, prefixes['prefix'][indices], prefixes['lengths'][indices])
            record_work(structural_work, arm, 'training_prefix', result)
            loss = (loss + result['nll'].sum() / events) * n / b
            batch_events = int(prefixes['event_mask'][indices].sum())
            work.update(joint_updates=1, joint_attempt_exposures=b, joint_case_exposures=len(eligible),
                        joint_event_exposures=batch_events, joint_prefix_rollouts=1)
            diagnostics = {'kind': kind, 'cursor': cursor, 'epoch': epoch, 'batch_offset': offset,
                           'indices': selected.tolist(), 'eligible': len(eligible), 'valid_events': batch_events,
                           'batch_size': b, 'order_sha256': hashlib.sha256(orders[epoch].astype('<i8').tobytes()).hexdigest()}
            counts['training_prefix_rollouts'] += 1
            counts['training_attempt_exposures'] += b
            counts['training_case_exposures'] += len(eligible)
            counts['training_prefix_event_exposures'] += batch_events
        require(bool(torch.isfinite(loss)), 'finite registered objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'], error_if_nonfinite=True)
        counts['optimizer_attempts'] += 1
        opt.step()
        counts['optimizer_steps'] += 1
        work.update(backward_passes=1, adam_steps=1)
        return {'loss': float(loss.detach()), 'diagnostics': diagnostics, 'work': work}

    hooks = Hooks(validate=validate, new_optimizer=optimizer, snapshot_model=snapshot_model,
        restore_model=restore_model, model_hash=lambda: _state_hash(_state(context['model'])),
        snapshot_optimizer=snapshot_optimizer, restore_optimizer=lambda opt, state: opt.load_state_dict(copy.deepcopy(state)),
        optimizer_hash=lambda opt: optimizer_digest(opt.state_dict()), update=update, checkpoint=checkpoint,
        final_summary=lambda opt, cursor: {'parameter_metadata': metadata(context['model']),
                                           'joint_cursor': cursor, 'construction_seconds': context['construction_seconds']})
    allocation_path = folder / f'allocation-{arm}-{seed}.json'
    try:
        allocation = run_allocation('prefix_then_joint', hooks, stage1_seconds=config['stage1_seconds'],
                                    total_seconds=config['total_seconds'], check=check)
    except BaseException as error:
        if 'model' in context:
            error.balanced_construction_work = dict(context['model'].construction_work)
        if hasattr(error, 'allocation_result'):
            _write(allocation_path, error.allocation_result)
        if hasattr(error, 'balanced_model_work'):
            _write(folder / f'failed-normalization-{arm}-{seed}.json', error.balanced_model_work)
        raise
    _write(allocation_path, allocation)
    require(allocation['status'] == 'PASS', 'both original allocation stages must pass')
    counts['accepted_optimizer_steps'] += allocation['accepted_updates']
    counts['accepted_joint_steps'] += allocation['accepted_joint_updates']
    counts['accepted_prefix_steps'] += allocation['accepted_prefix_updates']
    model = context['model']
    final = next(row for row in allocation['checkpoints'] if row['label'] == 'final')['metadata']['model']
    row = {'arm': arm, 'seed': seed, 'allocation': {'path': allocation_path.name, **_desc(allocation_path)},
           'seconds': time.perf_counter() - wrapper_start,
           'timing_scope': 'complete train wrapper through durable allocation trace; excludes final fit-row publication, included in outer producer time',
           'timed_seconds': allocation['timed_seconds'], 'final_summary_seconds': allocation['final_summary_seconds'],
           'updates': allocation['accepted_joint_updates'], 'accepted_prefix_updates': allocation['accepted_prefix_updates'],
           'attempted_updates': allocation['attempted_updates'], 'parameter_metadata': metadata(model),
           'construction_seconds': context['construction_seconds'], 'construction_work': dict(model.construction_work),
           'initial_state_sha256': allocation['initial_model_sha256'], 'final_state_sha256': allocation['final_model_sha256'],
           'oracle_prefix_input': False, 'checkpoint': final}
    _append(folder / 'fits.jsonl', row)
    counts['fit_count'] += 1
    check()
    return model, row


def run(folder: Path, config: dict, check):
    config = _config({**DEFAULT_CONFIG, **config})
    check()
    folder.mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(('train_generation_count', 'dev_generation_count', 'model_constructions',
        'oracle_model_constructions', 'fit_count', 'checkpoint_writes', 'training_blind_rollouts',
        'training_observed_rollouts', 'optimizer_attempts', 'optimizer_steps', 'training_case_exposures',
        'evaluation_blind_rollouts', 'evaluation_observed_rollouts', 'evaluation_shuffled_rollouts',
        'evaluation_case_views', 'oracle_blind_rollouts', 'oracle_observed_rollouts', 'array_decodes',
        'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls',
        'training_attempt_exposures', 'training_prefix_rollouts', 'training_prefix_event_exposures',
        'zero_endpoint_batches', 'evaluation_prefix_rollouts', 'evaluation_prefix_event_views',
        'readout_snapshot_evaluations', 'readout_snapshot_softmax_evaluations',
        'dynamics_snapshot_evaluations', 'optimizer_checkpoint_writes', 'epoch_order_generations',
        'accepted_optimizer_steps', 'accepted_joint_steps', 'accepted_prefix_steps'), 0)
    fits, rows, models, prediction_times, dataset_counts, generation_seconds = [], [], {}, [], {}, {}
    prefix_rows, prefix_times = [], []
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
            generated = generate_attempt_split(split, config['train_attempts' if split == 0 else 'dev_attempts'],
                config['train_horizon' if split == 0 else 'dev_horizon'],
                seed_namespace=config['seed_namespace'])
            data = generated['data']
            oracle = generated['oracle']['prefix_state'].copy()
            prefix_data = generated['prefix_data']
            _save(folder / (regime + '-prefix.npz'), prefix_data)
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
            return data, oracle, prefix_data

        stage = 'TRAIN generation and exact control'
        train_data, _train_oracle, train_prefixes = generate(0, 'train')
        training = _tensor_data(train_data)
        prefix_training = {name: torch.from_numpy(value.copy()) for name, value in train_prefixes.items() if name != 'case_ids'}
        require('oracle_prefix' not in training, 'no oracle state in learned training batches')
        for seed_index, seed in enumerate(config['fit_seeds']):
            offset = seed_index % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                stage = f'fit {arm} {seed}'
                check()
                model, fit = train(arm, seed, training, prefix_training, config, folder,
                                   counts, structural_work, check)
                fits.append(fit)
                models[arm, seed] = model
        # Raw initial T logits differ in matched_free; the independent audit
        # reconstructs initial probabilities from every saved boundary. The
        # constructor guards its matched-free roundtrip before fitting.
        expected = len(ARMS) * len(config['fit_seeds'])
        require(len(fits) == counts['fit_count'] == expected
                and counts['checkpoint_writes'] == counts['optimizer_checkpoint_writes'] == 3 * expected
                and counts['dev_generation_count'] == 0, 'all final fits precede DEV')
        barrier = {'checkpoints': [{'arm': r['arm'], 'seed': r['seed'], **r['checkpoint']} for r in fits],
            'dev_generation_count': 0, 'fit_count': expected, 'oracle_train_verified': True}
        _write(folder / 'checkpoint-barrier.json', barrier)
        stage = 'DEV generation and exact control'
        data, oracle, dev_prefixes = generate(1, 'base')
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
                prefix_start = time.perf_counter()
                prefix_arrays, prefix_row = predict_prefix(model, arm, seed, dev_prefixes, config, counts, structural_work, check)
                prefix_seconds = time.perf_counter() - prefix_start
                require(_state_hash(_state(model)) == before, 'model unchanged by prefix diagnostics')
                _save(folder / f'prefix-{arm}-{seed}-base.npz', prefix_arrays)
                prefix_rows.append(prefix_row)
                prefix_times.append({'arm': arm, 'seed': seed, 'seconds': prefix_seconds,
                    'attempts': len(dev_prefixes['prefix']), 'valid_events': prefix_row['valid_events'],
                    'timing_scope': 'prefix input copies, all likelihood forwards, guards, prediction copies and scalar NLL; excludes state hashes and file write'})
        _save(folder / 'predictions-base.npz', predictions)
        summary = {'version': VERSION, 'config': config, 'counts': counts, 'rows': rows,
            'counts_scope': 'Optimizer steps and rollout/exposure counters include accepted and discarded attempts. Accepted counts are separate. Checkpoint writes count 27 parameter and 27 optimizer boundary files; detailed snapshot/rollback/work counters and every attempt reside in each fit allocation file.',
            'prefix_rows': prefix_rows, 'prefix_prediction_times': prefix_times,
            'allocation_scope': 'Every arm uses the prefix_then_joint schedule. One overall eligibility clock, including model construction and boundary saves. Both stages use deadlines relative to that original start. Actual overshoot, final summaries and trace serialization are charged in fit seconds. This is not exact FLOP or actual wall-time equality.',
            'baseline_rows': baseline_rows, 'fits': fits, 'prediction_times': prediction_times,
            'structural_work': structural_work,
            'structural_work_scope': 'Summed actual forward calls and rows by arm across all seeds; excludes separately recorded construction/diagnostic work, validation FLOPs, backward operations, optimizer work and memory allocation.',
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
            'structural_work': structural_work,
            'failed_fit_construction_work': getattr(error, 'balanced_construction_work', None),
            'failed_route_work': getattr(error, 'balanced_model_work', None),
            'seconds': time.perf_counter() - started})
        raise
