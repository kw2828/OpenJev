"""Fresh BASE-only diagnostic of anchored versus task-uninformed cost heads.

All fifteen fits use the qualified computation reuse and exact-update recipe.
The two rounded arms differ only in cost-head initialization; their identical
prefix dynamics and Adam states are checked before any BASE generation.
This does not retry or rescue the separate observation-noise-shift study.
Admission and resource supervision belong to the caller.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from run_finite_expected_count_learning import ORACLE_FIELDS, predict
from run_finite_head_training import new_structural_work, train
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
from run_finite_update_learning import _config, predict_prefix

from openjev.research.finite_head_initialization import ARMS, TRANSPORT_ARMS
from openjev.research.finite_prefix_learning import generate_attempt_split
from openjev.research.finite_regime_reference import make_reference

VERSION = 'finite-head-learning-v1'
DEFAULT_CONFIG = {'seed_namespace': 436260924, 'train_attempts': 512, 'dev_attempts': 512,
    'batch_size': 64, 'learning_rate': .003, 'fit_seeds': [436261001, 436261002, 436261003, 436261004, 436261005],
    'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
    'prefix_updates': 1024, 'joint_updates': 3072, 'fit_cap_seconds': 120.}


def verify_paired_exposure(fits, folder, config):
    """Check exact saved success and same-seed joint indices before DEV exists."""
    require(len(fits) == len(ARMS) * len(config['fit_seeds'])
            and {(row['arm'], row['seed']) for row in fits}
            == {(arm, seed) for arm in ARMS for seed in config['fit_seeds']},
            'complete exact-update fit roster')
    paired = {}
    for fit in fits:
        path = folder / fit['allocation']['path']
        require(_desc(path) == {key: fit['allocation'][key] for key in ('sha256', 'bytes')},
                'durable original allocation before DEV')
        allocation = json.loads(path.read_text())
        require(allocation['status'] == 'PASS'
                and allocation['accepted_prefix_updates'] == fit['accepted_prefix_updates'] == config['prefix_updates']
                and allocation['accepted_joint_updates'] == allocation['joint_cursor'] == fit['updates'] == config['joint_updates']
                and allocation['attempted_updates'] == allocation['accepted_updates'] == fit['attempted_updates']
                == config['prefix_updates'] + config['joint_updates'], 'exact successful saved exposure')
        trace = allocation['trace']
        require(len(trace) == config['prefix_updates'] + config['joint_updates']
                and [row['kind'] for row in trace]
                == ['prefix'] * config['prefix_updates'] + ['joint'] * config['joint_updates']
                and all(row['accepted'] is True and row['rolled_back'] is False for row in trace),
                'complete accepted prefix-then-joint trace')
        indices = [row['result']['diagnostics']['indices'] for row in trace if row['kind'] == 'joint']
        sequence = hashlib.sha256(json.dumps(indices, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        paired.setdefault(fit['seed'], sequence)
        require(sequence == paired[fit['seed']], 'all same-seed arms consume the identical ordered joint minibatches')
    return paired


def verify_prefix_pairing(fits, folder, config):
    """Bind saved hashes, then require the rounded pair to enter joint identically.

    Only allocation JSON and opaque descriptors are read. Boundary parameter
    and Adam bytes are independently checked by the saved-output audit.
    """
    saved = {}
    for fit in fits:
        path = folder / fit['allocation']['path']
        require(_desc(path) == {key: fit['allocation'][key] for key in ('sha256', 'bytes')},
                'unchanged allocation for prefix pairing')
        allocation = json.loads(path.read_text())
        checkpoints = {row['label']: row['metadata'] for row in allocation['checkpoints']}
        require(set(checkpoints) == {'initial', 'boundary', 'final'}
                and len(allocation['checkpoints']) == 3, 'all three saved model and Adam boundaries')
        dynamic_hashes = {label: row['dynamics_state_sha256'] for label, row in checkpoints.items()}
        require(fit['dynamics_boundary_sha256'] == dynamic_hashes
                and fit['dynamics_boundary_hash_evaluations'] == 3,
                'fit dynamics hashes bind the saved checkpoint metadata')
        head_hashes = {label: row['head_state_sha256'] for label, row in checkpoints.items()}
        require(fit['head_boundary_sha256'] == head_hashes
                and fit['head_boundary_hash_evaluations'] == 3,
                'fit head hashes bind the saved checkpoint metadata')
        require(head_hashes['initial'] == head_hashes['boundary'],
                'every head remains unchanged throughout prefix training')
        saved[fit['arm'], fit['seed']] = checkpoints
    records = []
    for seed in config['fit_seeds']:
        anchor = saved['rounded_anchor', seed]
        random = saved['rounded_random', seed]
        for label in ('initial', 'boundary'):
            require(anchor[label]['dynamics_state_sha256'] == random[label]['dynamics_state_sha256'],
                    'rounded head arms have identical initial and post-prefix dynamics')
            require(anchor[label]['optimizer_state_sha256'] == random[label]['optimizer_state_sha256'],
                    'rounded head arms have identical prefix Adam states')
        records.append({'seed': seed, 'arms': ['rounded_anchor', 'rounded_random'],
            'initial_dynamics_sha256': anchor['initial']['dynamics_state_sha256'],
            'boundary_dynamics_sha256': anchor['boundary']['dynamics_state_sha256'],
            'initial_optimizer_sha256': anchor['initial']['optimizer_state_sha256'],
            'boundary_optimizer_sha256': anchor['boundary']['optimizer_state_sha256'],
            'same_initial_and_prefix_dynamics': True, 'same_prefix_optimizer_states': True,
            'head_initial_boundary_sha256': {arm: saved[arm, seed]['initial']['head_state_sha256'] for arm in ARMS},
            'all_heads_unchanged_after_prefix': True,
            'scope': 'Producer hash joins before BASE; saved bytes independently audited without learning replay.'})
    return records


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
    structural_work = new_structural_work()
    stage = 'initialize'
    started = time.perf_counter()
    try:
        _write(folder / 'config.json', config)
        counts['oracle_model_constructions'] += 1
        reference = make_reference(.12)
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
                                   counts, structural_work, check, implementation='reuse')
                fits.append(fit)
                models[arm, seed] = model
        expected = len(ARMS) * len(config['fit_seeds'])
        require(len(fits) == counts['fit_count'] == expected
                and counts['checkpoint_writes'] == counts['optimizer_checkpoint_writes'] == 3 * expected
                and counts['dev_generation_count'] == 0, 'all final fits precede DEV')
        paired_batches = verify_paired_exposure(fits, folder, config)
        prefix_pair_checks = verify_prefix_pairing(fits, folder, config)
        require(counts['accepted_prefix_steps'] == expected * config['prefix_updates']
                and counts['accepted_joint_steps'] == expected * config['joint_updates']
                and counts['accepted_optimizer_steps'] == counts['optimizer_attempts'] == counts['optimizer_steps']
                == expected * (config['prefix_updates'] + config['joint_updates']),
                'exact all-arm optimizer exposure before DEV')
        barrier = {'checkpoints': [{'arm': r['arm'], 'seed': r['seed'], **r['checkpoint']} for r in fits],
            'dev_generation_count': 0, 'fit_count': expected, 'oracle_train_verified': True,
            'paired_batch_sha256': paired_batches, 'prefix_pair_checks': prefix_pair_checks}
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
                prefix_rows.append({**prefix_row, 'regime': 'base'})
                prefix_times.append({'arm': arm, 'seed': seed, 'regime': 'base', 'seconds': prefix_seconds,
                    'attempts': len(dev_prefixes['prefix']), 'valid_events': prefix_row['valid_events'],
                    'timing_scope': 'prefix input copies, all likelihood forwards, guards, prediction copies and scalar NLL; excludes state hashes and file write'})
        _save(folder / 'predictions-base.npz', predictions)
        summary = {'version': VERSION, 'implementation': 'reuse', 'config': config, 'counts': counts, 'rows': rows,
            'counts_scope': 'Successful exact-update runs contain only accepted attempts; failed runs retain attempted work and partial states separately. Accepted counts remain explicit. Checkpoint writes count three parameter and three optimizer boundary files per fit; detailed snapshot/rollback/work counters and every attempt reside in each fit allocation file.',
            'prefix_rows': prefix_rows, 'prefix_prediction_times': prefix_times,
            'allocation_scope': 'Every arm completes the exact same prefix and joint update counts. Same-seed arms consume identical joint minibatches. One safety clock includes validation, construction, boundary saves and final summary; expiry fails the allocation rather than selecting fewer updates. Durable trace serialization remains included in full fit seconds. This is not equal FLOPs or wall time.',
            'baseline_rows': baseline_rows, 'fits': fits, 'prediction_times': prediction_times,
            'structural_work': structural_work,
            'structural_work_scope': 'Actual forward work by arm: training_prefix is the prefix-only stage; five joint_reuse_* routes own disjoint shared fields, endpoint filtering, all-attempt NLL and two continuations. Legacy training_blind/training_observed routes are empty. Evaluation routes are unchanged. Logical rollout counters are not physical shared operations or backward FLOPs. Excludes separately recorded construction/diagnostic work, validation FLOPs, backward operations, optimizer work and memory allocation.',
            'checkpoint_barrier': barrier, 'dataset_counts': dataset_counts,
            'paired_batch_sha256': paired_batches, 'prefix_pair_checks': prefix_pair_checks,
            'transport_arms': dict(TRANSPORT_ARMS),
            'head_scope': 'BASE-only cost-head initialization diagnostic; no observation-noise-shift evaluation, retraining or rescue.',
            'regimes': {'train': {'split_id': 0, 'epsilon': .12}, 'base': {'split_id': 1, 'epsilon': .12}},
            'generation_seconds': generation_seconds,
            'generation_timing_scope': 'world generation, data and oracle saves, exact-control forwards and validation',
            'oracle_checks': reference_checks, 'oracle_metadata': reference.parameter_metadata(),
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
            'failed_fit_construction_work': getattr(error, 'rounded_construction_work', None),
            'failed_route_work': getattr(error, 'rounded_model_work', None),
            'failed_joint_reuse_work': getattr(error, 'joint_reuse_work', None),
            'failed_head_initialization_work': getattr(error, 'head_initialization_work', None),
            'seconds': time.perf_counter() - started})
        raise
