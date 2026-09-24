"""Independent-cohort, six-arm blind-cost-loss comparison.

Each cohort pairs six fresh fits on its own public TRAIN histories. Every final
checkpoint across every cohort precedes any BASE or SHIFT generation. The
unannounced shift changes emission noise only; no shifted training or adaptation
occurs. This producer retains all outcomes and does not select a continuation
rule. Admission, fixed resource caps and saved-output auditing are external.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from run_finite_action_range_training import new_structural_work, train
from run_finite_expected_count_learning import ORACLE_FIELDS, predict
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
from run_finite_update_learning import _config as _fit_config
from run_finite_update_learning import predict_prefix

from openjev.research.finite_action_range_loss import ARMS, HEAD_ARMS, LOSS_KINDS, TRANSPORT_ARMS
from openjev.research.finite_prefix_learning import generate_attempt_split
from openjev.research.finite_regime_reference import make_reference

VERSION = 'finite-action-range-learning-v1'
DEFAULT_CONFIG = {
    'cohorts': [{'seed_namespace': 437260924 + i, 'fit_seed': 437261001 + i} for i in range(5)],
    'train_attempts': 512, 'dev_attempts': 512, 'batch_size': 64, 'learning_rate': .003,
    'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
    'prefix_updates': 1024, 'joint_updates': 3072, 'fit_cap_seconds': 120.,
}
COUNT_KEYS = ('train_generation_count', 'dev_generation_count', 'model_constructions',
    'oracle_model_constructions', 'fit_count', 'checkpoint_writes', 'training_blind_rollouts',
    'training_observed_rollouts', 'optimizer_attempts', 'optimizer_steps', 'training_case_exposures',
    'evaluation_blind_rollouts', 'evaluation_observed_rollouts', 'evaluation_shuffled_rollouts',
    'evaluation_case_views', 'oracle_blind_rollouts', 'oracle_observed_rollouts', 'array_decodes',
    'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls',
    'training_attempt_exposures', 'training_prefix_rollouts', 'training_prefix_event_exposures',
    'zero_endpoint_batches', 'evaluation_prefix_rollouts', 'evaluation_prefix_event_views',
    'readout_snapshot_evaluations', 'readout_snapshot_softmax_evaluations',
    'dynamics_snapshot_evaluations', 'optimizer_checkpoint_writes', 'epoch_order_generations',
    'accepted_optimizer_steps', 'accepted_joint_steps', 'accepted_prefix_steps')


def cohort_config(config, index):
    identity = config['cohorts'][index]
    return _fit_config({**{key: value for key, value in config.items() if key != 'cohorts'},
                        'seed_namespace': identity['seed_namespace'], 'fit_seeds': [identity['fit_seed']]})


def _config(config):
    require(type(config) is dict and set(config) == set(DEFAULT_CONFIG), 'declared cohort settings')
    cohorts = config['cohorts']
    require(type(cohorts) is list and 0 < len(cohorts) <= 5, 'one to five declared independent cohorts')
    require(all(type(row) is dict and set(row) == {'seed_namespace', 'fit_seed'} for row in cohorts),
            'explicit namespace and fit-seed pairs')
    result = {**config, 'cohorts': [dict(row) for row in cohorts]}
    for i in range(len(cohorts)):
        cohort_config(result, i)
    require(len({row['seed_namespace'] for row in cohorts}) == len(cohorts)
            and len({row['fit_seed'] for row in cohorts}) == len(cohorts),
            'distinct data namespaces and fit seeds across cohorts')
    return result


def aggregate_counts(cohorts):
    return {name: sum(row['counts'][name] for row in cohorts) for name in COUNT_KEYS}


def aggregate_work(cohorts):
    result = new_structural_work()
    for cohort in cohorts:
        for arm, routes in cohort['structural_work'].items():
            for route, block in routes.items():
                for key, value in block.items():
                    require(type(value) is int and value >= 0, 'nonnegative actual structural work')
                    result[arm][route][key] = result[arm][route].get(key, 0) + value
    return result


def verify_paired_exposure(fits, folder, config):
    """Bind original allocation JSON and compare every joint index in a cohort."""
    seed, = config['fit_seeds']
    require(len(fits) == len(ARMS) and {(row['arm'], row['seed']) for row in fits}
            == {(arm, seed) for arm in ARMS}, 'complete six-arm cohort fit roster')
    paired = None
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
        digest = hashlib.sha256(json.dumps(indices, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if paired is None:
            paired = digest
        require(digest == paired, 'all six cohort arms consume identical ordered joint minibatches')
    return paired


def verify_prefix_pairing(fits, folder, config):
    """Check the loss intervention begins from identical architecture-local state.

    Hashes bind saved checkpoint metadata without decoding checkpoint arrays.
    The independent audit checks the actual model/Adam bytes and the initial
    cross-architecture transition-function match separately.
    """
    seed, = config['fit_seeds']
    require(len(fits) == len(ARMS) and {(row['arm'], row['seed']) for row in fits}
            == {(arm, seed) for arm in ARMS}, 'complete six-arm prefix pairing roster')
    saved = {}
    for fit in fits:
        path = folder / fit['allocation']['path']
        require(_desc(path) == {key: fit['allocation'][key] for key in ('sha256', 'bytes')},
                'unchanged allocation for prefix pairing')
        allocation = json.loads(path.read_text())
        checkpoints = {row['label']: row['metadata'] for row in allocation['checkpoints']}
        require(set(checkpoints) == {'initial', 'boundary', 'final'}
                and len(allocation['checkpoints']) == 3, 'three saved model and Adam boundaries')
        for field, metadata_key in (('dynamics', 'dynamics_state_sha256'), ('head', 'head_state_sha256')):
            hashes = {label: row[metadata_key] for label, row in checkpoints.items()}
            require(fit[field + '_boundary_sha256'] == hashes
                    and fit[field + '_boundary_hash_evaluations'] == 3,
                    'fit ' + field + ' hashes bind saved checkpoint metadata')
        require(fit['initial_state_sha256'] == checkpoints['initial']['model_state_sha256']
                and fit['final_state_sha256'] == checkpoints['final']['model_state_sha256'],
                'fit full-state hashes bind saved boundaries')
        require(checkpoints['initial']['head_state_sha256'] == checkpoints['boundary']['head_state_sha256'],
                'every head remains unchanged throughout prefix training')
        saved[fit['arm']] = checkpoints
    initial_heads = {arm: saved[arm]['initial']['head_state_sha256'] for arm in ARMS}
    require(len(set(initial_heads.values())) == 1, 'all six initial random heads are paired')
    groups = []
    for transport in ('rounded', 'matched_free'):
        arms = [arm for arm in ARMS if TRANSPORT_ARMS[arm] == transport]
        first = saved[arms[0]]
        for arm in arms[1:]:
            for label in ('initial', 'boundary'):
                for key in ('model_state_sha256', 'dynamics_state_sha256', 'optimizer_state_sha256'):
                    require(saved[arm][label][key] == first[label][key],
                            'same-architecture loss variants have identical initial and prefix model/Adam states')
        groups.append({'transport_arm': transport, 'arms': arms,
            'initial_model_sha256': first['initial']['model_state_sha256'],
            'boundary_model_sha256': first['boundary']['model_state_sha256'],
            'initial_optimizer_sha256': first['initial']['optimizer_state_sha256'],
            'boundary_optimizer_sha256': first['boundary']['optimizer_state_sha256'],
            'same_initial_and_prefix_model': True, 'same_prefix_optimizer_states': True})
    return {'seed': seed, 'groups': groups, 'head_initial_boundary_sha256': initial_heads,
            'all_initial_heads_paired': True, 'all_heads_unchanged_after_prefix': True,
            'scope': 'Producer metadata joins before any DEV; saved bytes and initial transition functions independently audited.'}


def run(folder: Path, config: dict, check):
    config = _config({**DEFAULT_CONFIG, **config})
    require(isinstance(folder, Path) and callable(check), 'Path output and callable boundary check')
    check()
    folder.mkdir(parents=True, exist_ok=False)
    cohorts, contexts, fits, rows, baseline_rows = [], [], [], [], []
    prediction_times, prefix_rows, prefix_times = [], [], []
    stage, active_cohort = 'initialize', None
    started = time.perf_counter()
    try:
        _write(folder / 'config.json', config)

        def generate(cohort, context, split, regime):
            check()
            counts, local = cohort['counts'], cohort['config']
            counts['train_generation_count' if split == 0 else 'dev_generation_count'] += 1
            start = time.perf_counter()
            output = folder / cohort['folder']
            generated = generate_attempt_split(split, local['train_attempts' if split == 0 else 'dev_attempts'],
                local['train_horizon' if split == 0 else 'dev_horizon'], seed_namespace=local['seed_namespace'])
            data, prefix_data = generated['data'], generated['prefix_data']
            oracle = generated['oracle']['prefix_state'].copy()
            _save(output / (regime + '-prefix.npz'), prefix_data)
            require(len(oracle) > 0, 'retained cases required')
            _save(output / (regime + '.npz'), data)
            _save(output / (regime + '-oracle.npz'), {'prefix_state': oracle})
            reference_key = 'shift' if split == 2 else 'base'
            reference = context['references'][reference_key]
            predictions = predict(reference, 'exact_exact', data, oracle, local, counts, check, shuffled=False)
            errors = {name: float(np.max(np.abs(predictions[name] - data[name]))) for name in ORACLE_FIELDS}
            require(all(error <= 1e-12 for error in errors.values()), 'exact reference reproduces every target')
            require(cohort['oracle_state_sha256'][reference_key] == _state_hash(_state(reference)),
                    'oracle state unchanged')
            _save(output / ('oracle-' + regime + '.npz'), predictions)
            _write(output / ('oracle-' + regime + '-check.json'), errors)
            cohort['oracle_checks'][regime] = errors
            cohort['dataset_counts'][regime] = generated['counts']
            cohort['generation_seconds'][regime] = time.perf_counter() - start
            check()
            return data, oracle, prefix_data

        # The complete training loop ends before any evaluation generation.
        for index, identity in enumerate(config['cohorts']):
            active_cohort = index
            local = cohort_config(config, index)
            output = folder / f'cohort-{index:02d}'
            output.mkdir(exist_ok=False)
            _write(output / 'config.json', local)
            cohort = {'cohort_index': index, **identity, 'folder': output.name, 'config': local,
                'counts': dict.fromkeys(COUNT_KEYS, 0), 'structural_work': new_structural_work(),
                'dataset_counts': {}, 'generation_seconds': {}, 'oracle_checks': {},
                'oracle_metadata': {}, 'oracle_state_sha256': {}}
            context = {'references': {}, 'models': {}, 'fits': []}
            cohorts.append(cohort)
            contexts.append(context)
            cohort['counts']['oracle_model_constructions'] += 1
            reference = make_reference(.12)
            require(not list(reference.parameters()), 'untrained base reference has no parameters')
            context['references']['base'] = reference
            cohort['oracle_metadata']['base'] = reference.parameter_metadata()
            cohort['oracle_state_sha256']['base'] = _state_hash(_state(reference))
            stage = f'cohort {index} TRAIN generation and exact control'
            data, _oracle, prefixes = generate(cohort, context, 0, 'train')
            training = _tensor_data(data)
            prefix_training = {name: torch.from_numpy(value.copy()) for name, value in prefixes.items() if name != 'case_ids'}
            require('oracle_prefix' not in training and 'oracle_prefix' not in prefix_training,
                    'no oracle state in learned training inputs')
            offset = index % len(ARMS)
            cohort['fit_order'] = list(ARMS[offset:] + ARMS[:offset])
            for arm in cohort['fit_order']:
                stage = f'cohort {index} fit {arm} {identity["fit_seed"]}'
                check()
                model, fit = train(arm, identity['fit_seed'], training, prefix_training, local, output,
                    cohort['counts'], cohort['structural_work'], check, implementation='reuse')
                context['fits'].append(fit)
                context['models'][arm] = model
                fits.append({**fit, 'cohort_index': index, 'seed_namespace': identity['seed_namespace'],
                             'artifact_folder': output.name})
            require(cohort['counts']['fit_count'] == len(ARMS)
                    and cohort['counts']['checkpoint_writes'] == cohort['counts']['optimizer_checkpoint_writes'] == 3 * len(ARMS),
                    'complete cohort model and Adam boundaries')
            cohort['paired_batch_sha256'] = verify_paired_exposure(context['fits'], output, local)
            cohort['prefix_pair_checks'] = verify_prefix_pairing(context['fits'], output, local)
        expected = len(ARMS) * len(cohorts)
        counts = aggregate_counts(cohorts)
        require(len(fits) == counts['fit_count'] == expected and counts['dev_generation_count'] == 0,
                'all cohort fits precede every DEV generation')
        require(counts['accepted_prefix_steps'] == expected * config['prefix_updates']
                and counts['accepted_joint_steps'] == expected * config['joint_updates']
                and counts['accepted_optimizer_steps'] == counts['optimizer_attempts'] == counts['optimizer_steps']
                == expected * (config['prefix_updates'] + config['joint_updates']),
                'exact all-cohort optimizer exposure before DEV')
        barrier = {'checkpoints': [{'cohort_index': row['cohort_index'], 'seed_namespace': row['seed_namespace'],
            'arm': row['arm'], 'seed': row['seed'], **row['checkpoint'],
            'path': row['artifact_folder'] + '/' + row['checkpoint']['path']} for row in fits],
            'dev_generation_count': 0, 'fit_count': expected, 'oracle_train_verified': True,
            'cohorts': [{'cohort_index': row['cohort_index'], 'seed_namespace': row['seed_namespace'],
                'fit_seed': row['fit_seed'], 'paired_batch_sha256': row['paired_batch_sha256'],
                'prefix_pair_checks': row['prefix_pair_checks']} for row in cohorts]}
        _write(folder / 'checkpoint-barrier.json', barrier)
        for cohort, context in zip(cohorts, contexts, strict=True):
            index, seed = cohort['cohort_index'], cohort['fit_seed']
            active_cohort = index
            local, counts = cohort['config'], cohort['counts']
            output = folder / cohort['folder']
            tags = {'cohort_index': index, 'seed_namespace': cohort['seed_namespace']}
            for split, regime in ((1, 'base'), (2, 'shift')):
                if regime == 'shift':
                    stage = f'cohort {index} shift reference construction'
                    check()
                    require(aggregate_counts(cohorts)['fit_count'] == expected
                            and (folder / 'checkpoint-barrier.json').is_file(), 'global fit barrier before shifted reference')
                    counts['oracle_model_constructions'] += 1
                    reference = make_reference(.30)
                    require(not list(reference.parameters()), 'untrained shifted reference has no parameters')
                    context['references']['shift'] = reference
                    cohort['oracle_metadata']['shift'] = reference.parameter_metadata()
                    cohort['oracle_state_sha256']['shift'] = _state_hash(_state(reference))
                stage = f'cohort {index} {regime} generation and exact control'
                data, oracle, prefixes = generate(cohort, context, split, regime)
                baseline_rows.extend({**row, **tags} for row in _baseline_rows(data, regime, check))
                predictions = {}
                for arm in ARMS:
                    stage = f'cohort {index} evaluate {arm} {regime}'
                    check()
                    model = context['models'][arm]
                    before = _state_hash(_state(model))
                    fit = next(row for row in context['fits'] if row['arm'] == arm)
                    require(before == fit['final_state_sha256'], 'same final checkpoint used in both regimes')
                    start = time.perf_counter()
                    arrays = predict(model, arm, data, oracle, local, counts, check, shuffled=True,
                                     structural_work=cohort['structural_work'])
                    seconds = time.perf_counter() - start
                    after = _state_hash(_state(model))
                    require(before == after, 'model state unchanged by evaluation')
                    timing = {**tags, 'arm': arm, 'seed': seed, 'regime': regime, 'seconds': seconds,
                        'cases': len(oracle), 'batch_size': local['batch_size'],
                        'model_state_before': before, 'model_state_after': after,
                        'oracle_prefix_input': False, 'shuffle_offset': 1,
                        'timing_scope': 'input copies, three forwards, guards and prediction copies; excludes state hashes and scalar metrics'}
                    prediction_times.append(timing)
                    _append(output / 'prediction-times.jsonl', timing)
                    predictions.update({f'{arm}__{seed}__{name}': value for name, value in arrays.items()})
                    rows.extend({**row, **tags} for row in _rows(data, arrays, arm, seed, regime, check))
                    start = time.perf_counter()
                    prefix_arrays, prefix_row = predict_prefix(model, arm, seed, prefixes, local, counts,
                                                               cohort['structural_work'], check)
                    prefix_seconds = time.perf_counter() - start
                    require(_state_hash(_state(model)) == before, 'model unchanged by public-prefix diagnostics')
                    _save(output / f'prefix-{arm}-{seed}-{regime}.npz', prefix_arrays)
                    prefix_rows.append({**prefix_row, **tags, 'regime': regime})
                    prefix_times.append({**tags, 'arm': arm, 'seed': seed, 'regime': regime,
                        'seconds': prefix_seconds, 'attempts': len(prefixes['prefix']),
                        'valid_events': prefix_row['valid_events'],
                        'timing_scope': 'input copies, likelihood forwards, guards, prediction copies and scalar NLL; excludes state hashes and file write'})
                _save(output / ('predictions-' + regime + '.npz'), predictions)
        counts = aggregate_counts(cohorts)
        summary = {'version': VERSION, 'implementation': 'reuse', 'config': config, 'counts': counts,
            'cohorts': cohorts, 'fits': fits, 'rows': rows, 'baseline_rows': baseline_rows,
            'prefix_rows': prefix_rows, 'prediction_times': prediction_times, 'prefix_prediction_times': prefix_times,
            'checkpoint_barrier': barrier, 'structural_work': aggregate_work(cohorts),
            'transport_arms': dict(TRANSPORT_ARMS), 'head_arms': dict(HEAD_ARMS), 'loss_kinds': dict(LOSS_KINDS),
            'regimes': {'train': {'split_id': 0, 'epsilon': .12}, 'base': {'split_id': 1, 'epsilon': .12},
                        'shift': {'split_id': 2, 'epsilon': .30}},
            'cohort_scope': 'Independent data namespace and fit seed per cohort; six arms share that cohort only. No pooled continuation or policy selection is computed here.',
            'shift_scope': 'Unannounced observation-noise shift .12 to .30; transitions, hazard and true costs unchanged. No shifted training, adaptation or model selection.',
            'counts_scope': 'Sum of explicit cohort counters; successful exact-update fits have only accepted attempts. Each fit saves three model and three Adam boundaries; no checkpoint decode/replay in producer.',
            'allocation_scope': 'All arms receive equal prefix/joint updates and identical ordered joint minibatches within their cohort, not equal wall time or FLOPs. All final checkpoints across all cohorts precede both DEV regimes.',
            'structural_work_scope': 'Disjoint five joint_reuse routes plus unchanged prefix/evaluation routes, aggregated from cohorts. Separate fit loss_work counts the blind-loss intervention. Logical rollout counters are not physical shared calls; backward, optimizer and validation FLOPs are excluded.',
            'generation_timing_scope': 'world generation, saves, exact-reference forwards and validation',
            'seconds_before_summary_write': time.perf_counter() - started,
            'files': {str(path.relative_to(folder)): _desc(path) for path in sorted(folder.rglob('*')) if path.is_file()}}
        _write(folder / 'summary.json', summary)
        check()
        return summary
    except BaseException as error:
        _write(folder / 'failure.json', {'version': VERSION, 'stage': stage, 'active_cohort': active_cohort,
            'error': repr(error), 'counts': aggregate_counts(cohorts), 'cohorts': cohorts,
            'completed_fits': sum(row['counts']['fit_count'] for row in cohorts),
            'structural_work': aggregate_work(cohorts),
            'failed_fit_construction_work': getattr(error, 'rounded_construction_work', None),
            'failed_route_work': getattr(error, 'rounded_model_work', None),
            'failed_joint_reuse_work': getattr(error, 'joint_reuse_work', None),
            'failed_head_initialization_work': getattr(error, 'head_initialization_work', None),
            'failed_action_range_loss_work': getattr(error, 'action_range_loss_work', None),
            'seconds': time.perf_counter() - started})
        raise
