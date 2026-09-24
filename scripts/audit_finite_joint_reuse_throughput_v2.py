"""Saved-evidence audit of the fixed engineering training-throughput comparison.

Original closed phases and source inventories are admitted before any NumPy
import or array decode. Reused frozen audit helpers validate state/Adam formats
and the unchanged exact-update controller; no globals are modified. This does
not replay training, generate worlds, or evaluate task performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from audit_finite_observation_learning import require
from audit_finite_update_learning import (
    PARAMETER_SHAPES,
    PREFIX_KEYS,
    ROUTES,
    UPDATE_WORK,
    WORK_KEYS,
    StudyProfile,
    boundary_hashes,
    construction_work,
    endpoint_work,
    finite_scalar,
    forward_work,
    integer_work,
    optimizer_state,
    parameter_metadata,
    prefix_work,
    validate_allocation,
    validate_data,
)

VERSION = 'finite-joint-reuse-throughput-audit-v1'
PRODUCER_VERSION = 'finite-joint-reuse-throughput-v1'
ARMS = ('original_free', 'matched_free', 'rounded')
MODES = ('separate', 'reuse')
LABELS = ('initial', 'boundary', 'final')
REUSE_ROUTES = tuple('joint_reuse_' + name for name in ('shared', 'endpoint_prefix', 'prefix_nll', 'blind', 'observed'))
ALL_ROUTES = (*ROUTES, *REUSE_ROUTES)
SPEC = StudyProfile('throughput-943201', 943201, (943301,), 512, 128, 64, 32, 64, 30.)
ATOL = RTOL = 1e-7


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, 'ordinary exact evidence path')
    payload = path.read_bytes()
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def inventory(folder):
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'no linked evidence path')
        if path.is_file():
            result[str(path.relative_to(folder))] = descriptor(path)
    return result


def schedule():
    """Independent fixed roster, including the declared warmup exclusion."""
    result = []
    for round_index in range(4):
        offset = 0 if round_index == 0 else (round_index - 1) % 3
        for index in range(3):
            arm = ARMS[(index + offset) % 3]
            modes = MODES if (round_index + ARMS.index(arm)) % 2 == 0 else MODES[::-1]
            result.extend({'run_id': f'r{round_index}-{arm}-{mode}', 'round': round_index,
                           'arm': arm, 'implementation': mode, 'seed': 943301, 'warmup': round_index == 0}
                          for mode in modes)
    return result


def fit_files(arm, seed):
    return {*(f'{label}-{arm}-{seed}.npz' for label in LABELS),
            *(f'{label}-optimizer-{arm}-{seed}.json' for label in LABELS),
            f'allocation-{arm}-{seed}.json', 'training-orders.jsonl', 'fits.jsonl', 'fit-record.json'}


def expected_files():
    names = {'config.json', 'schedule.json', 'train.npz', 'train-prefix.npz',
             'generated-oracle.npz', 'generation.json', 'summary.json'}
    for row in schedule():
        names.update(f"fits/{row['run_id']}/{name}" for name in fit_files(row['arm'], row['seed']))
    return names


def admit(study):
    # The helper imports only stdlib metadata and the qualified native clock.
    from qualify_finite_joint_reuse_throughput_v2 import PLAN_PATH, closed_phase
    require(study == PLAN_PATH.parent, 'fixed registered throughput study path')
    plan = read(PLAN_PATH)
    qualified = closed_phase(plan, 'qualify')
    producer = closed_phase(plan, 'run')
    require(qualified['terminal']['finished_ns'] <= producer['receipt']['launch']['started_ns'],
            'qualification genuinely closed before original benchmark launch')
    return plan, qualified, producer


def validate_metadata(folder, summary):
    require(summary['version'] == PRODUCER_VERSION and summary['status'] == 'PASS'
            and summary['config'] == SPEC.config and summary['schedule'] == schedule()
            and summary['scientific_admission'] is False, 'complete registered engineering benchmark')
    actual = inventory(folder)
    require(set(actual) == expected_files() and len(actual) == 247, 'exact complete 24-fit payload roster; no failure or repeat')
    require(summary['files_before_summary'] == {k: v for k, v in actual.items() if k != 'summary.json'},
            'complete pre-summary byte inventory')
    require(read(folder / 'config.json') == SPEC.config and read(folder / 'schedule.json') == schedule()
            and read(folder / 'generation.json') == summary['generation'], 'original configuration, schedule and generation joins')
    require(summary['counts'] == {'train_generation_calls': 1, 'dev_generation_calls': 0,
                                 'teacher_calls': 0, 'task_evaluation_calls': 0}, 'TRAIN-only engineering scope')
    require(finite_scalar(summary['seconds_before_summary_write']) and 0 < summary['seconds_before_summary_write'] < 90,
            'complete producer stays within its fixed cooperative bound')
    require(len(summary['runs']) == 24, 'all warmup and timed fits retained')
    for row, spec in zip(summary['runs'], schedule(), strict=True):
        require(all(row[key] == value and type(row[key]) is type(value) for key, value in spec.items()),
                'exact predeclared run order and identities')
        require(read(folder / 'fits' / spec['run_id'] / 'fit-record.json') == row, 'outer fit record byte-content join')
        fit = row['fit']
        require(lines(folder / 'fits' / spec['run_id'] / 'fits.jsonl') == [fit], 'one completed fit, no retry')
        require(fit['arm'] == row['arm'] and fit['seed'] == row['seed'] and fit['implementation'] == row['implementation']
                and fit['integration_version'] == 'finite-joint-reuse-training-v1'
                and fit['oracle_prefix_input'] is False, 'declared integrated fit and public-only inputs')
        require(fit['joint_structural_routes'] == (list(REUSE_ROUTES) if row['implementation'] == 'reuse'
                else ['training_blind', 'training_observed', 'training_prefix']), 'disjoint actual-work ownership')
        require(all(finite_scalar(value) for value in (row['call_seconds'], fit['seconds'], fit['timed_seconds'], fit['construction_seconds']))
                and 0 < fit['timed_seconds'] <= fit['seconds'] <= row['call_seconds']
                and 0 <= fit['construction_seconds'] <= fit['timed_seconds'], 'nested complete call and internal timings')
        parameter_metadata(fit['parameter_metadata'])
        require(fit['construction_work'] == construction_work(row['arm']), 'actual constructor work separately retained')
    require(math.fsum(row['call_seconds'] for row in summary['runs']) <= summary['seconds_before_summary_write'],
            'all complete fit calls are nested within producer time')
    return actual


def validate_inputs(data, prefix, oracle, generation, np):
    """Geometry and exposure audit, not another synthetic-world benchmark."""
    n, _ = validate_data(data, 'train', np, spec=SPEC)
    require(set(prefix) == PREFIX_KEYS and set(oracle) == {'prefix_state'}, 'exact saved learner and separate oracle fields')
    shapes = {'prefix': (512, 9, 31), 'lengths': (512,), 'case_ids': (512,), 'event_mask': (512, 9),
              'endpoint_eligible': (512,), 'endpoint_rows': (512,)}
    require(all(prefix[k].shape == shape for k, shape in shapes.items()), 'complete all-attempt inputs')
    require(prefix['prefix'].dtype == np.float32 and prefix['lengths'].dtype == prefix['endpoint_rows'].dtype == np.int64
            and prefix['event_mask'].dtype == prefix['endpoint_eligible'].dtype == np.bool_
            and prefix['case_ids'].dtype == np.dtype('U64'), 'exact public prefix dtypes')
    p, lengths, mask = prefix['prefix'], prefix['lengths'], prefix['event_mask']
    require(np.array_equal(prefix['case_ids'], np.array([f'ns943201-split0-case{i:010d}' for i in range(512)], dtype='U64')),
            'all fresh allocated attempts in original order')
    require(np.isfinite(p).all() and np.all((p == 0) | (p == 1)) and np.all((lengths >= 2) & (lengths <= 9))
            and np.array_equal(mask, np.arange(9)[None, :] < lengths[:, None]), 'finite binary histories and valid lengths')
    require(not p[~mask].any() and not p[:, :, 10:].any() and not p[:, 0, :4].any() and not p[:, 0, 8].any()
            and np.all(p[:, 0, 4:8].sum(-1) == 1) and np.all(p[:, 0, 9] == 1) and not p[:, 1:, 9].any()
            and np.all(p[:, 1:, :4].sum(-1)[mask[:, 1:]] == 1)
            and np.all(p[:, :, 4:9].sum(-1)[mask] == 1), 'public reset/actions/events and zero padding')
    found, eligible = p[:, :, 8] == 1, prefix['endpoint_eligible']
    require(np.array_equal(eligible, ~found.any(-1)) and np.all(lengths[eligible] == 9)
            and np.all(found.sum(-1) == (~eligible).astype(np.int64))
            and np.all(found[np.flatnonzero(~eligible), lengths[~eligible] - 1]), 'first-found inclusive termination')
    require(np.all(prefix['endpoint_rows'][~eligible] == -1)
            and np.array_equal(prefix['endpoint_rows'][eligible], np.arange(n))
            and np.array_equal(data['prefix'], p[eligible]) and np.array_equal(data['lengths'], lengths[eligible])
            and np.array_equal(data['case_ids'], prefix['case_ids'][eligible]), 'exact survivor mapping and shared endpoint inputs')
    state = oracle['prefix_state']
    require(state.dtype == np.float64 and state.shape == (n, 8) and np.isfinite(state).all()
            and np.all(state >= 0) and np.all(np.abs(state.sum(-1) - 1) <= 1e-12), 'separately retained finite oracle; not learner input')
    forecast_found = data['observations'] == 4
    event_draws = sum(int(np.argmax(row)) + 1 if row.any() else 2 for row in forecast_found)
    events = int(mask.sum())
    expected = {'version': 'finite-prefix-learning-data-v1', 'seed_namespace': 943201, 'split_id': 0,
        'epsilon': .12, 'horizon': 2, 'attempted': 512, 'retained': n, 'discarded_found': 512 - n,
        'valid_prefix_events': events, 'prefix_found_by_step': [int(found[:, i].sum()) for i in range(1, 9)],
        'initial_odor_draws': 512, 'prefix_event_draws': events - 512, 'prefix_action_draws': 4096,
        'forecast_action_draws': 2 * n, 'forecast_event_draws': event_draws,
        'forecast_found_cases': int(forecast_found.any(-1).sum())}
    require(generation['counts'] == expected and finite_scalar(generation['seconds']) and generation['seconds'] > 0,
            'all original generation counts derive from saved public histories')
    return {'attempts': 512, 'survivors': n, 'valid_events': events, 'oracle_passed_to_training': False,
            'target_predictions_recomputed': False}


def add_work(destination, route, values):
    if not destination[route]:
        destination[route] = dict.fromkeys(WORK_KEYS, 0)
    for key, value in values.items():
        destination[route][key] += value


def reuse_work(arm, prefix, indices, observations, np):
    """Independent geometry for the helper's five disjoint forward blocks."""
    values = {route: dict.fromkeys(WORK_KEYS, 0) for route in REUSE_ROUTES}
    shared = forward_work(arm, dict.fromkeys(WORK_KEYS, 0), prefix=True)
    shared['operator_marginal_sum_calls'] = int(len(observations) > 0)
    values['joint_reuse_shared'] = shared
    nll = prefix_work(prefix, indices, np)
    nll['reset_emission_softmax_calls'] = nll['prefix_operator_softmax_calls'] = 0
    values['joint_reuse_prefix_nll'] = nll
    if len(observations):
        values['joint_reuse_endpoint_prefix'].update(reset_emission_rows=len(observations),
                                                   prefix_filter_calls=8, prefix_filter_rows=8 * len(observations))
        for route, observed in (('blind', False), ('observed', True)):
            block = endpoint_work(observations, observed, np)
            for key in ('operator_observed_softmax_calls', 'operator_marginal_sum_calls', 'prefix_filter_calls',
                        'prefix_filter_rows', 'reset_emission_softmax_calls', 'reset_emission_rows', 'prefix_operator_softmax_calls'):
                block[key] = 0
            block['cost_head_softmax_calls'] = block['cost_readout_calls']
            block['cost_head_probability_rows'] = 8 * block['cost_readout_calls']
            values['joint_reuse_' + route] = block
    return values


def execution(record, allocation, orders, data, prefix, np):
    arm, seed, mode = record['arm'], record['seed'], record['implementation']
    require(len(orders) == 8, 'eight complete joint epochs, no repeated/rejected order')
    rebuilt = []
    for epoch, row in enumerate(orders):
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, 818]))).permutation(512)
        require(row == {'arm': arm, 'seed': seed, 'epoch': epoch, 'indices': order.tolist(), 'batch_size': 64},
                'original joint order matches independently reconstructed fixed permutation')
        rebuilt.append(order)
    expected = {name: {route: {} for route in ALL_ROUTES} for name in ARMS}
    total = dict.fromkeys(UPDATE_WORK, 0)
    events = int(prefix['event_mask'].sum())
    for index, row in enumerate(allocation['trace']):
        d = row['result']['diagnostics']
        work = dict.fromkeys(UPDATE_WORK, 0)
        work.update(backward_passes=1, adam_steps=1)
        if index < 32:
            require(d == {'kind': 'prefix', 'valid_events': events, 'attempts': 512,
                          'pseudocount': .001, 'cost_head_updated': False}, 'full-prefix objective and denominators')
            work.update(prefix_updates=1, prefix_event_exposures=events, prefix_full_rollouts=1)
            block = forward_work(arm, prefix_work(prefix, np.arange(512), np), prefix=True, prior=True)
            add_work(expected[arm], 'training_prefix', block)
        else:
            cursor = index - 32
            epoch, offset = divmod(cursor, 8)
            order = rebuilt[epoch]
            indices = order[offset * 64:(offset + 1) * 64]
            selected = prefix['endpoint_rows'][indices]
            selected = selected[selected >= 0]
            batch_events = int(prefix['event_mask'][indices].sum())
            require(d == {'kind': 'joint', 'cursor': cursor, 'epoch': epoch, 'batch_offset': offset,
                    'indices': indices.tolist(), 'eligible': len(selected), 'valid_events': batch_events,
                    'batch_size': 64, 'order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()},
                    'all 64 paired minibatches and true exposures')
            work.update(joint_updates=1, joint_attempt_exposures=64, joint_case_exposures=len(selected),
                        joint_event_exposures=batch_events, joint_prefix_rollouts=1,
                        joint_blind_rollouts=int(bool(len(selected))), joint_observed_rollouts=int(bool(len(selected))),
                        zero_endpoint_batches=int(not len(selected)))
            observations = data['observations'][selected]
            if mode == 'reuse':
                for route, block in reuse_work(arm, prefix, indices, observations, np).items():
                    add_work(expected[arm], route, block)
            else:
                add_work(expected[arm], 'training_prefix', forward_work(arm, prefix_work(prefix, indices, np), prefix=True))
                if len(selected):
                    for route, observed in (('training_blind', False), ('training_observed', True)):
                        add_work(expected[arm], route, forward_work(arm, endpoint_work(observations, observed, np), prefix=False))
        integer_work(row['result']['work'], UPDATE_WORK)
        require(row['result']['work'] == work, 'exact attempted work and no hidden discarded updates')
        for key, value in work.items():
            total[key] += value
    require(allocation['update_work'] == total, 'controller numerical work sums exact complete exposure')
    counts = {'model_constructions': 1, 'checkpoint_writes': 3, 'optimizer_checkpoint_writes': 3,
        'training_prefix_rollouts': 96, 'training_prefix_event_exposures': 40 * events,
        'epoch_order_generations': 8, 'training_blind_rollouts': total['joint_blind_rollouts'],
        'training_observed_rollouts': total['joint_observed_rollouts'], 'zero_endpoint_batches': total['zero_endpoint_batches'],
        'training_attempt_exposures': 4096, 'training_case_exposures': 8 * len(data['case_ids']),
        'optimizer_attempts': 96, 'optimizer_steps': 96, 'accepted_optimizer_steps': 96,
        'accepted_joint_steps': 64, 'accepted_prefix_steps': 32, 'fit_count': 1}
    integer_work(record['counts'], set(counts))
    require(record['counts'] == counts, 'complete logical update/exposure counters')
    require(type(record['structural_work']) is dict and set(record['structural_work']) == set(ARMS), 'complete work arm roster')
    for name in ARMS:
        require(set(record['structural_work'][name]) == set(ALL_ROUTES), 'complete disjoint route roster')
        for route, block in record['structural_work'][name].items():
            integer_work(block, WORK_KEYS if block else set())
    require(record['structural_work'] == expected, 'actual disjoint forward-work geometry and unused routes')
    return {'logical_counts': counts, 'update_work': total, 'structural_work': expected[arm],
            'epoch_orders': 8, 'model_updates_replayed': False}


def compare_arrays(a, b, np, *, exact=False):
    require(a.dtype == b.dtype == np.float64 and a.shape == b.shape
            and np.isfinite(a).all() and np.isfinite(b).all(), 'paired finite float64 geometry')
    delta = np.abs(a - b)
    bound = ATOL + RTOL * np.maximum(np.abs(a), np.abs(b))
    require(a.tobytes() == b.tobytes() if exact else bool(np.all(delta <= bound)),
            'exact boundary parity' if exact else 'prospective final absolute/relative parity')
    return {'maximum_absolute_error': float(delta.max()) if delta.size else 0.,
            'maximum_tolerance_fraction': float((delta / bound).max()) if delta.size else 0., 'exact': exact}


def compare_optimizer(a, b, np, *, exact):
    require(a['param_groups'] == b['param_groups'] and set(a['state']) == set(b['state']), 'paired optimizer groups and parameters')
    reports = []
    for key in a['state']:
        require(a['state'][key]['step'] == b['state'][key]['step'], 'paired Adam retained integer step')
        for name in ('exp_avg', 'exp_avg_sq'):
            x, y = a['state'][key][name], b['state'][key][name]
            require({k: v for k, v in x.items() if k != 'values'} == {k: v for k, v in y.items() if k != 'values'},
                    'paired lossless optimizer tensor metadata')
            reports.append(compare_arrays(np.asarray(x['values'], dtype=np.float64), np.asarray(y['values'], dtype=np.float64), np, exact=exact))
    if exact:
        require(a == b, 'exact initial/prefix Adam JSON')
    return {'maximum_absolute_error': max((x['maximum_absolute_error'] for x in reports), default=0.),
            'maximum_tolerance_fraction': max((x['maximum_tolerance_fraction'] for x in reports), default=0.), 'exact': exact}


def timing_gate(records):
    require(len(records) == 24 and [{key: row[key] for key in schedule()[0]} for row in records] == schedule(),
            'complete timing roster in predeclared order')
    mapped = {(row['round'], row['arm'], row['implementation']): row for row in records}
    paired = []
    for round_index in range(4):
        for arm in ARMS:
            separate, reuse = (mapped[round_index, arm, mode]['call_seconds'] for mode in MODES)
            require(finite_scalar(separate) and finite_scalar(reuse) and separate > 0 and reuse > 0, 'positive finite complete-call times')
            paired.append({'round': round_index, 'arm': arm, 'warmup': round_index == 0,
                           'separate_seconds': separate, 'reuse_seconds': reuse, 'ratio': separate / reuse})
    measured = [row for row in paired if not row['warmup']]
    medians = {arm: statistics.median(row['ratio'] for row in measured if row['arm'] == arm) for arm in ARMS}
    conditions = {f"{row['arm']}_round{row['round']}_ratio_gt1": row['ratio'] > 1 for row in measured}
    conditions.update({f'{arm}_median_gt1.05': value > 1.05 for arm, value in medians.items()})
    passed = all(conditions.values())
    return {'name': 'ENGINEERING_THROUGHPUT', 'passed': passed,
            'status': 'ENGINEERING_THROUGHPUT_PASS' if passed else 'ENGINEERING_THROUGHPUT_FAIL',
            'conditions': conditions, 'paired': paired, 'median_ratios': medians,
            'warmup_call_seconds': math.fsum(row['call_seconds'] for row in records if row['warmup']),
            'measured_call_seconds': math.fsum(row['call_seconds'] for row in records if not row['warmup'])}


def audit(study):
    study = Path(study).resolve()
    plan, qualified, producer = admit(study)
    folder = producer['directory'] / 'benchmark'
    summary = read(folder / 'summary.json')
    before = validate_metadata(folder, summary)
    # No numerical import or payload decode is reachable before original closure.
    import numpy as np

    decodes = 0
    def load(path):
        nonlocal decodes
        decodes += 1
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key].copy() for key in source.files}
    data, prefix, oracle = (load(folder / name) for name in ('train.npz', 'train-prefix.npz', 'generated-oracle.npz'))
    inputs = validate_inputs(data, prefix, oracle, summary['generation'], np)
    require(summary['generation']['files'] == {name: before[name] for name in ('train.npz', 'train-prefix.npz', 'generated-oracle.npz')},
            'generation descriptors bind the one shared saved dataset')
    checked, states, optimizers = [], {}, {}
    for row in summary['runs']:
        run_id, arm, seed = row['run_id'], row['arm'], row['seed']
        path, fit = folder / 'fits' / run_id, row['fit']
        alloc_path = path / f'allocation-{arm}-{seed}.json'
        require(fit['allocation'] == {'path': alloc_path.name, **descriptor(alloc_path)}, 'fit allocation descriptor')
        allocation = read(alloc_path)
        report = validate_allocation(allocation, arm, spec=SPEC)
        require(allocation['metadata'] == {'parameter_metadata': fit['parameter_metadata'], 'attempts': 512,
                'eligible': inputs['survivors'], 'events': inputs['valid_events'], 'batch_size': 64,
                'seed': seed, 'hidden_state_input': False}, 'fixed full-dataset denominators and no oracle input')
        require(allocation['initial_model_sha256'] == fit['initial_state_sha256']
                and allocation['final_model_sha256'] == fit['final_state_sha256']
                and allocation['timed_seconds'] == fit['timed_seconds']
                and allocation['final_summary_seconds'] == fit['final_summary_seconds']
                and (fit['updates'], fit['accepted_prefix_updates'], fit['attempted_updates']) == (64, 32, 96), 'fit joins exact controller result')
        boundaries = []
        for index, label in enumerate(LABELS):
            model_path, opt_path = path / f'{label}-{arm}-{seed}.npz', path / f'{label}-optimizer-{arm}-{seed}.json'
            arrays, opt = load(model_path), read(opt_path)
            hashes = boundary_hashes(arrays, np)
            opt_checked = optimizer_state(opt, 'joint' if label == 'final' else 'prefix', (0, 32, 64)[index])
            boundary = allocation['checkpoints'][index]
            require(boundary['metadata'] == {'label': label, 'model': {'path': model_path.name, **descriptor(model_path)},
                    'optimizer': {'path': opt_path.name, **descriptor(opt_path)}, 'model_state_sha256': hashes['full'],
                    'optimizer_state_sha256': opt_checked['sha256'], 'joint_cursor': 64 if label == 'final' else 0}
                    and boundary['model_sha256'] == hashes['full'] and boundary['optimizer_sha256'] == opt_checked['sha256'],
                    'opaque descriptors, decoded actual state hashes and controller boundary join')
            states[run_id, label], optimizers[run_id, label] = arrays, opt
            boundaries.append({'label': label, 'model': hashes, 'optimizer': opt_checked})
        require(states[run_id, 'initial']['cost_logits'].tobytes() == states[run_id, 'boundary']['cost_logits'].tobytes(),
                'prefix stage leaves actual head unchanged')
        initial_opt = optimizers[run_id, 'initial']
        fresh_joint = {'state': {}, 'param_groups': [{**initial_opt['param_groups'][0], 'params': list(range(4))}]}
        require(allocation['boundary']['optimizer_after_sha256'] == optimizer_state(fresh_joint, 'joint', 0)['sha256'], 'fresh joint Adam reset')
        require(fit['checkpoint'] == allocation['checkpoints'][2]['metadata']['model'], 'final checkpoint descriptor')
        executed = execution(row, allocation, lines(path / 'training-orders.jsonl'), data, prefix, np)
        checked.append({'run_id': run_id, 'controller': report, 'boundaries': boundaries, 'execution': executed,
                        'call_seconds': row['call_seconds'], 'fit_seconds': fit['seconds'],
                        'construction_seconds': fit['construction_seconds'], 'construction_work': fit['construction_work']})
    parity = []
    for round_index in range(4):
        for arm in ARMS:
            a, b = (f'r{round_index}-{arm}-{mode}' for mode in MODES)
            boundaries = []
            for label in LABELS:
                exact = label != 'final'
                parameters = {name: compare_arrays(states[a, label][name], states[b, label][name], np, exact=exact)
                              for name in PARAMETER_SHAPES}
                adam = compare_optimizer(optimizers[a, label], optimizers[b, label], np, exact=exact)
                boundaries.append({'label': label, 'parameters': parameters, 'optimizer': adam})
            parity.append({'round': round_index, 'arm': arm, 'boundaries': boundaries, 'passed': True})
    require(decodes == 75, 'exact three input plus72 checkpoint decodes')
    require(inventory(folder) == before, 'all audited evidence unchanged')
    # Reauthenticate original source and full native evidence after all reads.
    after_plan, _, after_producer = admit(study)
    require(after_plan == plan and after_producer['receipt'] == producer['receipt'], 'original closure and sources unchanged after audit')
    return {'version': VERSION, 'agreement': True, 'technical_complete': False,
            'requires_original_supervisor_closure': True, 'scientific_admission': False,
            'inputs': inputs, 'fits': checked, 'parity': parity, 'throughput': timing_gate(summary['runs']),
            'tolerance': {'absolute': ATOL, 'relative': RTOL, 'scale': 'max(abs(separate),abs(reuse))',
                          'initial_and_prefix_boundary': 'exact'},
            'counts': {'fits': 24, 'warmup_fits': 6, 'measured_fits': 18, 'pairs': 12, 'measured_pairs': 9,
                       'prefix_updates': 768, 'joint_updates': 1536, 'array_decodes': decodes, 'checkpoint_decodes': 72,
                       'optimizer_json_decodes': 72, 'model_calls': 0, 'optimizer_calls': 0, 'generator_calls': 0,
                       'teacher_calls': 0, 'task_evaluation_calls': 0},
            'timings': {'generation_seconds': summary['generation']['seconds'],
                        'producer_before_summary_seconds': summary['seconds_before_summary_write'],
                        'qualification_native_seconds': qualified['terminal']['wall_seconds'],
                        'producer_native_seconds': producer['terminal']['wall_seconds']},
            'sources': plan['sources'], 'limitations': [
                'Intermediate training updates, timers and input nonmutation remain original source/trace attestations; no updates are replayed.',
                'Decoded model and optimizer boundaries and saved batch exposure are independently checked.',
                'Named forward counts are not backward FLOPs; logical rollout counts are distinct from physical shared work.',
                'Warmups are retained. All nine measured ratios are used; three pairs per arm have a declared order imbalance.',
                'This is one-machine engineering throughput, not significance, task effectiveness, or scientific admission.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists()
            and args.output.parent.resolve() == args.study.resolve() / 'audit-01'
            and args.output.name == 'audit.json', 'exclusive original audit output')
    result = audit(args.study)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
