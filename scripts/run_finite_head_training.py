"""Head-initialization study fit using the unchanged qualified reuse arithmetic.

Only the factory, external arm identity and saved initialization metadata differ.
The exact controller, objective, parameter order, optimizer/reset, batch schedule
and numerical update statements remain frozen. The retained separate branch is
unreachable: callers must explicitly request implementation='reuse'. Additional
head initialization is within construction_seconds; three dynamics-only and
three head-only hashes reuse already-copied checkpoint arrays inside charged
checkpoint callbacks.
No data generation, registration, evaluation or hidden-state input occurs here.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import numpy as np
import torch
from run_finite_expected_count_learning import ROUTES, record_work
from run_finite_observation_learning import _append, _desc, _save, _state, _state_hash, _write, require
from run_finite_update_learning import UPDATE_WORK, optimizer_digest, optimizer_payload, tensor_storage

from openjev.research.finite_head_initialization import ARMS, model_metadata
from openjev.research.finite_head_initialization import make_model as learned_model
from openjev.research.finite_joint_reuse import TARGET_FIELDS, WORK_ROUTES, joint_objective
from openjev.research.finite_observation_models import metadata, objective
from openjev.research.finite_rounded_models import DYNAMICS, prefix_predictions, torch_objective
from openjev.research.finite_update_allocation import Hooks, Snapshot, run_update_allocation

VERSION = 'finite-head-training-v1'
IMPLEMENTATIONS = ('reuse',)
REUSE_ROUTES = tuple('joint_reuse_' + route for route in WORK_ROUTES)
COUNT_KEYS = ('model_constructions', 'checkpoint_writes', 'optimizer_checkpoint_writes',
    'training_prefix_rollouts', 'training_prefix_event_exposures', 'epoch_order_generations',
    'training_blind_rollouts', 'training_observed_rollouts', 'zero_endpoint_batches',
    'training_attempt_exposures', 'training_case_exposures', 'optimizer_attempts', 'optimizer_steps',
    'accepted_optimizer_steps', 'accepted_joint_steps', 'accepted_prefix_steps', 'fit_count')


def new_counts():
    return dict.fromkeys(COUNT_KEYS, 0)


def new_structural_work():
    return {arm: {route: {} for route in (*ROUTES, *REUSE_ROUTES)} for arm in ARMS}


def train(arm, seed, data, prefixes, config, folder, counts, structural_work, check, *, implementation):
    wrapper_start = time.perf_counter()
    require(type(implementation) is str and implementation in IMPLEMENTATIONS, 'explicit reuse implementation')
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
        context['parameter_metadata'] = metadata(context['model'])
        context['model_metadata'] = model_metadata(context['model'], arm)
        context['head_initialization_work'] = dict(context['model'].head_initialization_work)
        context['construction_seconds'] = time.perf_counter() - start
        return {'parameter_metadata': context['parameter_metadata'],
                'model_metadata': context['model_metadata'],
                'head_initialization_work': context['head_initialization_work'], 'attempts': n,
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
                'dynamics_state_sha256': _state_hash({name: state[name] for name in DYNAMICS}),
                'head_state_sha256': _state_hash({'cost_logits': state['cost_logits']}),
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
            if implementation == 'separate':
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
            else:
                batch = {name: value[eligible] for name, value in data.items()}
                positions = torch.nonzero(endpoint_rows >= 0, as_tuple=False).flatten()
                require(torch.equal(batch['prefix'], prefixes['prefix'][indices][positions])
                        and torch.equal(batch['lengths'], prefixes['lengths'][indices][positions]),
                        'endpoint data matches surviving attempted prefixes')
                result = joint_objective(model, prefixes['prefix'][indices], prefixes['lengths'][indices],
                    positions, batch['actions'], batch['observations'],
                    {name: batch[name] for name in TARGET_FIELDS},
                    total_attempts=n, total_survivors=survivors, total_events=events)
                loss = result['loss']
                for route, block in result['work'].items():
                    record_work(structural_work, arm, 'joint_reuse_' + route, {'work': block})
                if len(eligible):
                    work.update(joint_blind_rollouts=1, joint_observed_rollouts=1)
                    counts['training_blind_rollouts'] += 1
                    counts['training_observed_rollouts'] += 1
                else:
                    work['zero_endpoint_batches'] = 1
                    counts['zero_endpoint_batches'] += 1
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
                                           'model_metadata': model_metadata(context['model'], arm),
                                           'head_initialization_work': dict(context['model'].head_initialization_work),
                                           'joint_cursor': cursor, 'construction_seconds': context['construction_seconds']})
    allocation_path = folder / f'allocation-{arm}-{seed}.json'
    try:
        allocation = run_update_allocation(hooks, prefix_updates=config['prefix_updates'],
                                           joint_updates=config['joint_updates'],
                                           max_seconds=config['fit_cap_seconds'], check=check)
    except BaseException as error:
        if 'model' in context:
            error.rounded_construction_work = dict(context['model'].construction_work)
            error.head_initialization_work = dict(context['model'].head_initialization_work)
        if hasattr(error, 'head_initialization_work'):
            _write(folder / f'failed-head-initialization-{arm}-{seed}.json', error.head_initialization_work)
        if hasattr(error, 'allocation_result'):
            _write(allocation_path, error.allocation_result)
        if hasattr(error, 'joint_reuse_work'):
            _write(folder / f'failed-joint-reuse-{arm}-{seed}.json', error.joint_reuse_work)
        if hasattr(error, 'rounded_model_work'):
            _write(folder / f'failed-normalization-{arm}-{seed}.json', error.rounded_model_work)
        raise
    _write(allocation_path, allocation)
    require(allocation['status'] == 'PASS', 'both exact-update stages must pass')
    require(allocation['accepted_prefix_updates'] == config['prefix_updates']
            and allocation['accepted_joint_updates'] == allocation['joint_cursor'] == config['joint_updates']
            and allocation['attempted_updates'] == allocation['accepted_updates']
            == config['prefix_updates'] + config['joint_updates']
            and all(row['accepted'] is True and row['rolled_back'] is False for row in allocation['trace']),
            'successful allocation completes every requested update without discarded work')
    counts['accepted_optimizer_steps'] += allocation['accepted_updates']
    counts['accepted_joint_steps'] += allocation['accepted_joint_updates']
    counts['accepted_prefix_steps'] += allocation['accepted_prefix_updates']
    model = context['model']
    final = next(row for row in allocation['checkpoints'] if row['label'] == 'final')['metadata']['model']
    row = {'arm': arm, 'seed': seed, 'implementation': implementation, 'integration_version': VERSION,
           'joint_structural_routes': list(REUSE_ROUTES) if implementation == 'reuse' else
                                      ['training_blind', 'training_observed', 'training_prefix'],
           'allocation': {'path': allocation_path.name, **_desc(allocation_path)},
           'seconds': time.perf_counter() - wrapper_start,
           'timing_scope': 'complete train wrapper through durable allocation trace; excludes final fit-row publication, included in outer producer time',
           'timed_seconds': allocation['timed_seconds'], 'final_summary_seconds': allocation['final_summary_seconds'],
           'updates': allocation['accepted_joint_updates'], 'accepted_prefix_updates': allocation['accepted_prefix_updates'],
           'attempted_updates': allocation['attempted_updates'], 'parameter_metadata': metadata(model),
           'construction_seconds': context['construction_seconds'], 'construction_work': dict(model.construction_work),
           'model_metadata': model_metadata(model, arm),
           'head_initialization_work': dict(model.head_initialization_work),
           'dynamics_boundary_sha256': {point['label']: point['metadata']['dynamics_state_sha256']
                                       for point in allocation['checkpoints']},
           'dynamics_boundary_hash_evaluations': len(allocation['checkpoints']),
           'head_boundary_sha256': {point['label']: point['metadata']['head_state_sha256']
                                   for point in allocation['checkpoints']},
           'head_boundary_hash_evaluations': len(allocation['checkpoints']),
           'initial_state_sha256': allocation['initial_model_sha256'], 'final_state_sha256': allocation['final_model_sha256'],
           'oracle_prefix_input': False, 'checkpoint': final}
    _append(folder / 'fits.jsonl', row)
    counts['fit_count'] += 1
    check()
    return model, row
