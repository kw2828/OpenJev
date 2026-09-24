"""Fresh five-seed replication with base and observation-noise-shift evaluation.

The qualified reuse training implementation, objective, initialization and
exact-update controller remain unchanged. All fifteen final checkpoints precede
either DEV cohort. The shift changes only the synthetic emission noise from
.12 to .30; it never trains or adapts any learned model. Separate zero-parameter
references witness the correct law for each regime without global mutation.
Admission and resource supervision belong to the caller.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from run_finite_expected_count_learning import ORACLE_FIELDS, predict
from run_finite_joint_reuse_training import new_structural_work, train
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
from run_finite_update_learning import _config, predict_prefix, verify_paired_exposure

from openjev.research.finite_prefix_learning import generate_attempt_split
from openjev.research.finite_regime_reference import make_reference
from openjev.research.finite_rounded_models import ARMS

VERSION = 'finite-reuse-replication-v1'
DEFAULT_CONFIG = {'seed_namespace': 435260924, 'train_attempts': 512, 'dev_attempts': 512,
    'batch_size': 64, 'learning_rate': .003, 'fit_seeds': [435261001, 435261002, 435261003, 435261004, 435261005],
    'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
    'prefix_updates': 1024, 'joint_updates': 3072, 'fit_cap_seconds': 120.}


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
        references = {'base': reference}
        reference_hashes = {'base': _state_hash(_state(reference))}
        reference_metadata = {'base': reference.parameter_metadata()}
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
            reference_key = 'shift' if split == 2 else 'base'
            witness = references[reference_key]
            reference_predictions = predict(witness, 'exact_exact', data, oracle, config, counts, check, shuffled=False)
            errors = {name: float(np.max(np.abs(reference_predictions[name] - data[name]))) for name in ORACLE_FIELDS}
            require(all(error <= 1e-12 for error in errors.values()), 'exact reference reproduces every target before learning interpretation')
            require(reference_hashes[reference_key] == _state_hash(_state(witness)), 'oracle model state unchanged')
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
        # Raw initial T logits differ in matched_free; the independent audit
        # reconstructs initial probabilities from every saved boundary. The
        # constructor guards its matched-free roundtrip before fitting.
        expected = len(ARMS) * len(config['fit_seeds'])
        require(len(fits) == counts['fit_count'] == expected
                and counts['checkpoint_writes'] == counts['optimizer_checkpoint_writes'] == 3 * expected
                and counts['dev_generation_count'] == 0, 'all final fits precede DEV')
        verify_paired_exposure(fits, folder, config)
        require(counts['accepted_prefix_steps'] == expected * config['prefix_updates']
                and counts['accepted_joint_steps'] == expected * config['joint_updates']
                and counts['accepted_optimizer_steps'] == counts['optimizer_attempts'] == counts['optimizer_steps']
                == expected * (config['prefix_updates'] + config['joint_updates']),
                'exact all-arm optimizer exposure before DEV')
        barrier = {'checkpoints': [{'arm': r['arm'], 'seed': r['seed'], **r['checkpoint']} for r in fits],
            'dev_generation_count': 0, 'fit_count': expected, 'oracle_train_verified': True}
        _write(folder / 'checkpoint-barrier.json', barrier)
        baseline_rows = []
        for split, regime in ((1, 'base'), (2, 'shift')):
            if regime == 'shift':
                stage = 'shift exact-reference construction'
                check()
                require(counts['fit_count'] == expected and (folder / 'checkpoint-barrier.json').is_file(),
                        'all fits precede the shifted exact reference')
                counts['oracle_model_constructions'] += 1
                references['shift'] = make_reference(.30)
                require(not list(references['shift'].parameters()), 'shift reference has no learned parameters')
                reference_hashes['shift'] = _state_hash(_state(references['shift']))
                reference_metadata['shift'] = references['shift'].parameter_metadata()
            stage = regime + ' DEV generation and exact control'
            data, oracle, dev_prefixes = generate(split, regime)
            baseline_rows.extend(_baseline_rows(data, regime, check))
            predictions = {}
            for arm in ARMS:
                for seed in config['fit_seeds']:
                    stage = f'evaluate {regime} {arm} {seed}'
                    check()
                    model = models[arm, seed]
                    before = _state_hash(_state(model))
                    start = time.perf_counter()
                    arrays = predict(model, arm, data, oracle, config, counts, check, shuffled=True, structural_work=structural_work)
                    seconds = time.perf_counter() - start
                    after = _state_hash(_state(model))
                    require(before == after, 'model state unchanged by evaluation')
                    timing = {'arm': arm, 'seed': seed, 'regime': regime, 'seconds': seconds,
                        'cases': len(oracle), 'batch_size': config['batch_size'],
                        'model_state_before': before, 'model_state_after': after,
                        'oracle_prefix_input': False, 'shuffle_offset': 1,
                        'timing_scope': 'input copies, all three forwards, guards and prediction copies; excludes state hashes and scalar metrics'}
                    prediction_times.append(timing)
                    _append(folder / 'prediction-times.jsonl', timing)
                    predictions.update({f'{arm}__{seed}__{name}': value for name, value in arrays.items()})
                    rows.extend(_rows(data, arrays, arm, seed, regime, check))
                    prefix_start = time.perf_counter()
                    prefix_arrays, prefix_row = predict_prefix(model, arm, seed, dev_prefixes, config, counts, structural_work, check)
                    prefix_seconds = time.perf_counter() - prefix_start
                    require(_state_hash(_state(model)) == before, 'model unchanged by prefix diagnostics')
                    _save(folder / f'prefix-{arm}-{seed}-{regime}.npz', prefix_arrays)
                    prefix_rows.append({**prefix_row, 'regime': regime})
                    prefix_times.append({'arm': arm, 'seed': seed, 'regime': regime, 'seconds': prefix_seconds,
                        'attempts': len(dev_prefixes['prefix']), 'valid_events': prefix_row['valid_events'],
                        'timing_scope': 'prefix input copies, all likelihood forwards, guards, prediction copies and scalar NLL; excludes state hashes and file write'})
            _save(folder / f'predictions-{regime}.npz', predictions)
        summary = {'version': VERSION, 'implementation': 'reuse', 'config': config, 'counts': counts, 'rows': rows,
            'counts_scope': 'Successful exact-update runs contain only accepted attempts; failed runs retain attempted work and partial states separately. Accepted counts remain explicit. Checkpoint writes count three parameter and three optimizer boundary files per fit; detailed snapshot/rollback/work counters and every attempt reside in each fit allocation file.',
            'prefix_rows': prefix_rows, 'prefix_prediction_times': prefix_times,
            'allocation_scope': 'Every arm completes the exact same prefix and joint update counts. Same-seed arms consume identical joint minibatches. One safety clock includes validation, construction, boundary saves and final summary; expiry fails the allocation rather than selecting fewer updates. Durable trace serialization remains included in full fit seconds. This is not equal FLOPs or wall time.',
            'baseline_rows': baseline_rows, 'fits': fits, 'prediction_times': prediction_times,
            'structural_work': structural_work,
            'structural_work_scope': 'Actual forward work by arm: training_prefix is the prefix-only stage; five joint_reuse_* routes own disjoint shared fields, endpoint filtering, all-attempt NLL and two continuations. Legacy training_blind/training_observed routes are empty. Evaluation routes aggregate both base and shift; their prediction timings and rows retain regime identity. Logical rollout counters are not physical shared operations or backward FLOPs. Excludes separately recorded construction/diagnostic work, validation FLOPs, backward operations, optimizer work and memory allocation.',
            'checkpoint_barrier': barrier, 'dataset_counts': dataset_counts,
            'generation_seconds': generation_seconds,
            'generation_timing_scope': 'world generation, data and oracle saves, exact-control forwards and validation',
            'oracle_checks': reference_checks, 'oracle_metadata': reference_metadata,
            'oracle_state_sha256': reference_hashes,
            'regimes': {'train': {'split_id': 0, 'epsilon': .12},
                        'base': {'split_id': 1, 'epsilon': .12},
                        'shift': {'split_id': 2, 'epsilon': .30}},
            'shift_scope': 'Unannounced observation-noise shift only; transition, hazard and cost laws remain fixed. No shift training or adaptation.',
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
            'seconds': time.perf_counter() - started})
        raise
