"""Independent saved-output reconciliation for paired independent-cohort losses.

Caller authenticates source pins and original producer closure before entry.
Only frozen scalar/reference, state-hash and geometry helpers are imported.
No producer, model, optimizer or data generator is invoked. Intermediate
update losses/hashes remain execution attestations, not numerical replays.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import audit_finite_head_learning as h
import audit_finite_reuse_replication as r
from audit_finite_joint_reuse_throughput_v2 import reuse_work
from audit_finite_observation_learning import compare, require

VERSION = 'finite-action-range-learning-audit-v1'
ARMS = ('rounded_mse', 'rounded_double', 'rounded_range', 'free_mse', 'free_double', 'free_range')
TRANSPORT = MappingProxyType({arm: 'rounded' if arm.startswith('rounded_') else 'matched_free' for arm in ARMS})
HEAD = MappingProxyType({arm: 'rounded_random' if arm.startswith('rounded_') else 'matched_free_random' for arm in ARMS})
LOSS = MappingProxyType({arm: arm.rsplit('_', 1)[1] for arm in ARMS})
HORIZONS, REGIMES, SPLITS = r.HORIZONS, r.REGIMES, r.SPLITS
TARGETS, FIELDS, ROUTES = r.TARGETS, r.FIELDS, r.ROUTES
ENDPOINT_WORK_KEYS, PREFIX_WORK_KEYS, UPDATE_WORK = h.ENDPOINT_WORK_KEYS, h.PREFIX_WORK_KEYS, h.UPDATE_WORK
endpoint_work, prefix_work = h.endpoint_work, h.prefix_work
integer_work, finite_scalar, digest_value = h.integer_work, h.finite_scalar, h.digest_value
LOSS_WORK_KEYS = ('wrapper_calls', 'zero_endpoint_batches', 'blind_error_entries', 'mse_square_entries',
    'mse_mean_calls', 'range_max_rows', 'range_min_rows', 'range_subtract_entries', 'range_square_entries',
    'range_scale_entries', 'range_mean_calls', 'replacement_subtractions', 'weight_scalar_divisions',
    'weight_scalar_multiplications', 'loss_adjustment_multiplications', 'loss_adjustment_additions')
SCIENCE = tuple(r.StudyProfile('science', 437260924+i, (437261001+i,), 512, 512, 64, 1024, 3072, 120.) for i in range(5))
PROFILES = MappingProxyType({'science': SCIENCE, 'smoke': (
    r.StudyProfile('smoke', 948001, (948101,), 64, 32, 16, 2, 3, 30.),)})
# The frozen prospective rule is explicit and separately reported from technical agreement.
RULE = MappingProxyType({'name': 'ACTION_RANGE_ADVANCE', 'candidate': 'rounded_range',
    'controls': ('rounded_mse', 'rounded_double'), 'regimes': ('base', 'shift'), 'horizons': (4, 8),
    'minimum_train': 256, 'minimum_development': 64, 'maximum_reference_ratio': .5,
    'maximum_observed_kl': .1, 'maximum_h8_survival_mae': .05, 'mean_reduction': .1,
    'strict_paired_improvement': True})
REGIME_METADATA = {split: {'split_id': i, 'epsilon': .30 if split == 'shift' else .12} for i, split in enumerate(SPLITS)}


def profile_config(specs):
    common = {key: value for key, value in specs[0].config.items() if key not in ('seed_namespace', 'fit_seeds')}
    return {'cohorts': [{'seed_namespace': s.namespace, 'fit_seed': s.seeds[0]} for s in specs], **common}


def read(path, *, lines=False):
    require(path.is_file() and not path.is_symlink(), 'regular original metadata')
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)


def descriptor(path):
    require(path.is_file() and not path.is_symlink(), 'regular original payload')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def strict(actual, expected, message):
    # Avoid bool-as-int and silent extra metadata keys in fixed configurations.
    require(type(actual) is type(expected), message)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), message)
        for key in expected:
            strict(actual[key], expected[key], message)
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), message)
        for left, right in zip(actual, expected, strict=True):
            strict(left, right, message)
    else:
        require(actual == expected, message)


def local_files(seed):
    names = {'config.json', 'fits.jsonl', 'training-orders.jsonl', 'prediction-times.jsonl'}
    for split in SPLITS:
        names |= {split+'.npz', split+'-prefix.npz', split+'-oracle.npz', 'oracle-'+split+'.npz', 'oracle-'+split+'-check.json'}
    names |= {'predictions-'+regime+'.npz' for regime in REGIMES}
    for arm in ARMS:
        names |= {f'allocation-{arm}-{seed}.json', *(f'prefix-{arm}-{seed}-{regime}.npz' for regime in REGIMES)}
        for label in ('initial', 'boundary', 'final'):
            names |= {f'{label}-{arm}-{seed}.npz', f'{label}-optimizer-{arm}-{seed}.json'}
    return names


def loss_work(eligible, horizon, kind):
    require(type(eligible) is int and eligible >= 0 and type(horizon) is int and horizon > 0 and kind in LOSS.values(), 'loss-work geometry')
    result = dict.fromkeys(LOSS_WORK_KEYS, 0)
    result['wrapper_calls'] = 1
    if not eligible:
        result['zero_endpoint_batches'] = 1
    elif kind != 'mse':
        rows = eligible * horizon
        result.update(blind_error_entries=4*rows, mse_square_entries=4*rows, mse_mean_calls=1,
            weight_scalar_divisions=2, weight_scalar_multiplications=1,
            loss_adjustment_multiplications=1, loss_adjustment_additions=1)
        if kind == 'range':
            result.update(range_max_rows=rows, range_min_rows=rows, range_subtract_entries=rows,
                range_square_entries=rows, range_scale_entries=rows, range_mean_calls=1, replacement_subtractions=1)
    return result


def forward_work(arm, values, *, prefix, prior=False):
    return h.forward_work(HEAD[arm], values, prefix=prefix, prior=prior)


def model_metadata(fit):
    arm = fit['arm']
    pm = fit['model_metadata']
    require(arm in ARMS and pm['arm'] == arm and pm['version'] == 'finite-action-range-loss-v1'
        and pm['head_factory_version'] == 'finite-head-initialization-v1' and pm['head_arm'] == HEAD[arm]
        and pm['loss_kind'] == fit['loss_kind'] == LOSS[arm], 'explicit external arm and blind-cost intervention')
    expressions = {'mse': 'mean(error.square())', 'double': '2*mean(error.square())',
        'range': 'mean((amax(error)-amin(error)).square()/4)'}
    require(pm['blind_cost_objective'] == expressions[LOSS[arm]]
        and pm['range_tie_gradient'] == 'torch.amax/amin equally share gradient among tied extrema', 'fixed loss definition')
    # Validation adapter only: numerical arrays, source objects and external IDs
    # remain untouched. Inherited checks require their own head-factory labels.
    adapted = {**fit, 'arm': HEAD[arm], 'model_metadata': {**pm, 'version': pm['head_factory_version'], 'arm': HEAD[arm]}}
    h.parameter_metadata(fit['parameter_metadata'])
    h.head_metadata(adapted)


def validate_fit(folder, fit, spec, retained, events):
    arm, seed = fit['arm'], fit['seed']
    require(arm in ARMS and seed == spec.seeds[0] and fit['implementation'] == 'reuse'
        and fit['integration_version'] == 'finite-action-range-training-v1'
        and fit['joint_structural_routes'] == list(h.REUSE_ROUTES) and fit['oracle_prefix_input'] is False,
        'qualified six-arm public-only trainer')
    model_metadata(fit)
    name = f'allocation-{arm}-{seed}.json'
    strict(fit['allocation'], {'path': name, **descriptor(folder/name)}, 'allocation descriptor')
    allocation = read(folder/name)
    h.validate_allocation(allocation, HEAD[arm], spec=spec)
    fixed = {'parameter_metadata': fit['parameter_metadata'], 'model_metadata': fit['model_metadata'],
        'head_initialization_work': fit['head_initialization_work'], 'loss_kind': LOSS[arm]}
    strict(allocation['metadata'], {**fixed, 'loss_work': dict.fromkeys(LOSS_WORK_KEYS, 0),
        'attempts': spec.train_attempts, 'eligible': retained, 'events': events, 'batch_size': spec.batch_size,
        'seed': seed, 'hidden_state_input': False}, 'fixed global TRAIN denominators')
    strict(allocation['final_summary'], {**fixed, 'loss_work': fit['loss_work'], 'joint_cursor': spec.joint_updates,
        'construction_seconds': fit['construction_seconds']}, 'read-only final summary')
    for key, actual in [('initial_state_sha256', allocation['initial_model_sha256']), ('final_state_sha256', allocation['final_model_sha256']),
        ('updates', spec.joint_updates), ('accepted_prefix_updates', spec.prefix_updates), ('attempted_updates', spec.prefix_updates+spec.joint_updates),
        ('timed_seconds', allocation['timed_seconds']), ('final_summary_seconds', allocation['final_summary_seconds'])]:
        strict(fit[key], actual, 'fit/controller join '+key)
    require(finite_scalar(fit['seconds']) and fit['seconds'] > 0 and fit['seconds']+1e-8 >= allocation['timed_seconds']
        and finite_scalar(fit['construction_seconds']) and fit['construction_seconds'] >= 0, 'complete positive fit timing')
    strict(fit['construction_work'], h.construction_work(HEAD[arm]), 'constructor geometry')
    for field, key in [('head', 'head_state_sha256'), ('dynamics', 'dynamics_state_sha256')]:
        require(fit[field+'_boundary_hash_evaluations'] == 3, 'three restricted boundary hashes')
        strict(fit[field+'_boundary_sha256'], {p['label']: p['metadata'][key] for p in allocation['checkpoints']}, 'restricted hashes bind all boundaries')
    for point in allocation['checkpoints']:
        label, meta = point['label'], point['metadata']
        require(set(meta) == {'label','model','optimizer','model_state_sha256','dynamics_state_sha256','head_state_sha256','optimizer_state_sha256','joint_cursor'}
            and meta['label'] == label and meta['model_state_sha256'] == point['model_sha256']
            and meta['optimizer_state_sha256'] == point['optimizer_sha256'] and meta['joint_cursor'] == point['joint_cursor'], 'exact boundary metadata')
        for kind, filename in [('model', f'{label}-{arm}-{seed}.npz'), ('optimizer', f'{label}-optimizer-{arm}-{seed}.json')]:
            strict(meta[kind], {'path': filename, **descriptor(folder/filename)}, 'complete boundary artifact')
    strict(fit['checkpoint'], allocation['checkpoints'][-1]['metadata']['model'], 'retained final checkpoint')
    return allocation


def population_metadata(cohort, spec, folder):
    require(set(cohort['dataset_counts']) == set(SPLITS) and set(cohort['oracle_checks']) == set(SPLITS)
        and set(cohort['generation_seconds']) == set(SPLITS), 'all three cohort populations')
    retained = {}
    for split in SPLITS:
        row = cohort['dataset_counts'][split]
        attempted = spec.train_attempts if split == 'train' else spec.dev_attempts
        horizon = 2 if split == 'train' else 8
        require(row['version'] == 'finite-prefix-learning-data-v1' and row['seed_namespace'] == spec.namespace
            and row['split_id'] == r.SPLIT_IDS[split] and row['epsilon'] == r.EPSILON[split]
            and row['attempted'] == attempted and row['horizon'] == horizon, 'fresh cohort-specific population')
        n, lost, found = row['retained'], row['discarded_found'], row['prefix_found_by_step']
        require(type(n) is int and 0 < n <= attempted and type(lost) is int and n+lost == attempted
            and type(found) is list and len(found) == 8 and all(type(x) is int and x >= 0 for x in found)
            and sum(found) == lost, 'all attempted prefixes accounted')
        require(row['initial_odor_draws'] == attempted and row['prefix_action_draws'] == 8*attempted
            and row['prefix_event_draws'] == 8*n+sum((i+1)*v for i,v in enumerate(found))
            and row['valid_prefix_events'] == attempted+row['prefix_event_draws'] and row['forecast_action_draws'] == horizon*n,
            'complete declared draw geometry')
        errors = cohort['oracle_checks'][split]
        require(set(errors) == set(TARGETS) and all(finite_scalar(v) and 0 <= v <= 1e-12 for v in errors.values()), 'exact control admitted')
        strict(read(folder/('oracle-'+split+'-check.json')), errors, 'saved exact reference errors')
        require(finite_scalar(cohort['generation_seconds'][split]) and cohort['generation_seconds'][split] >= 0, 'generation time')
        retained[split] = n
    r.validate_reference_metadata({**cohort, 'regimes': REGIME_METADATA})
    return retained


def validate_metadata(folder, summary, specs, check):
    strict(summary['config'], profile_config(specs), 'immutable cohort profile')
    strict(read(folder/'config.json'), summary['config'], 'durable configuration')
    require(summary['version'] == 'finite-action-range-learning-v1' and summary['implementation'] == 'reuse', 'six-arm cohort producer')
    strict(summary['transport_arms'], dict(TRANSPORT), 'transport mapping')
    strict(summary['head_arms'], dict(HEAD), 'head mapping')
    strict(summary['loss_kinds'], dict(LOSS), 'loss mapping')
    strict(summary['regimes'], REGIME_METADATA, 'observation laws')
    require(len(summary['cohorts']) == len(specs), 'every independent cohort')
    names = {'config.json', 'checkpoint-barrier.json'}
    for i,spec in enumerate(specs):
        names |= {f'cohort-{i:02d}/'+name for name in local_files(spec.seeds[0])}
    require(set(summary['files']) == names, 'exact complete cohort payload roster')
    actual = set()
    for path in folder.rglob('*'):
        require(not path.is_symlink(), 'no symbolic links in evidence')
        if path.is_file() and path != folder/'summary.json':
            actual.add(str(path.relative_to(folder)))
    require(actual == names, 'no omitted or unlisted original payloads')
    for name in names:
        check()
        strict(summary['files'][name], descriptor(folder/name), 'unchanged payload '+name)
    all_fits, allocations, orders = [], [], []
    for index, (cohort,spec) in enumerate(zip(summary['cohorts'], specs, strict=True)):
        check()
        seed, = spec.seeds
        require(cohort['cohort_index'] == index and cohort['seed_namespace'] == spec.namespace
            and cohort['fit_seed'] == seed and cohort['folder'] == f'cohort-{index:02d}', 'cohort identity')
        strict(cohort['config'], spec.config, 'cohort-local profile')
        local = folder/cohort['folder']
        strict(read(local/'config.json'), spec.config, 'cohort-local saved profile')
        population = population_metadata(cohort,spec,local)
        fits = read(local/'fits.jsonl', lines=True)
        offset = index % len(ARMS)
        roster = list(ARMS[offset:]+ARMS[:offset])
        require(cohort['fit_order'] == roster and [(x['arm'],x['seed']) for x in fits] == [(arm,seed) for arm in roster], 'complete rotated loss-arm roster')
        all_fits.extend({**fit,'cohort_index':index,'seed_namespace':spec.namespace,'artifact_folder':cohort['folder']} for fit in fits)
        local_allocations = {(fit['arm'],seed): validate_fit(local,fit,spec,population['train'],cohort['dataset_counts']['train']['valid_prefix_events']) for fit in fits}
        allocations.append(local_allocations)
        orders.append(read(local/'training-orders.jsonl',lines=True))
        expected_views = [(regime,arm,seed) for regime in REGIMES for arm in ARMS]
        timing = [x for x in summary['prediction_times'] if x['cohort_index'] == index]
        strict(read(local/'prediction-times.jsonl',lines=True),timing,'durable inference timings')
        require([(x['regime'],x['arm'],x['seed']) for x in timing] == expected_views,'each frozen final in both regimes')
        fit_map = {x['arm']:x for x in fits}
        for x in timing:
            require(x['seed_namespace'] == spec.namespace and x['cases'] == population[x['regime']]
                and x['batch_size'] == spec.batch_size and x['model_state_before'] == x['model_state_after'] == fit_map[x['arm']]['final_state_sha256']
                and x['oracle_prefix_input'] is False and x['shuffle_offset'] == 1 and finite_scalar(x['seconds']) and x['seconds'] >= 0,'unaltered public-only evaluation')
        pt = [x for x in summary['prefix_prediction_times'] if x['cohort_index'] == index]
        require([(x['regime'],x['arm'],x['seed']) for x in pt] == expected_views,'all public-prefix evaluations')
        for x in pt:
            require(x['seed_namespace'] == spec.namespace and x['attempts'] == spec.dev_attempts
                and x['valid_events'] == cohort['dataset_counts'][x['regime']]['valid_prefix_events']
                and finite_scalar(x['seconds']) and x['seconds'] >= 0,'complete prefix diagnostic timing')
    strict(summary['fits'],all_fits,'global and local original fits')
    barrier = {'checkpoints':[{'cohort_index':x['cohort_index'],'seed_namespace':x['seed_namespace'],'arm':x['arm'],'seed':x['seed'],
        **x['checkpoint'],'path':x['artifact_folder']+'/'+x['checkpoint']['path']} for x in all_fits],
        'dev_generation_count':0,'fit_count':6*len(specs),'oracle_train_verified':True,
        'cohorts':[{key:x[key] for key in ('cohort_index','seed_namespace','fit_seed','paired_batch_sha256','prefix_pair_checks')} for x in summary['cohorts']]}
    strict(summary['checkpoint_barrier'],barrier,'one global all-finals-before-any-DEV barrier')
    strict(read(folder/'checkpoint-barrier.json'),barrier,'durable global barrier')
    require(len(summary['prediction_times']) == len(summary['prefix_prediction_times']) == 12*len(specs), 'no extra inference views')
    return allocations, orders


def validate_predictions(data, predictions, np, *, spec=SCIENCE[0]):
    expected = {f'{arm}__{seed}__{field}' for arm in ARMS for seed in spec.seeds for field in FIELDS}
    require(set(predictions) == expected, 'complete all-model prediction fields')
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

def rows_for(data, predictions, uniform_costs, np, *, regime, check=lambda: None, spec=SCIENCE[0]):
    require(regime in REGIMES, 'explicit development regime')
    validate_predictions(data, predictions, np, spec=spec)
    n = len(data['case_ids'])
    rows, baselines = [], []
    for horizon in HORIZONS:
        h = horizon - 1
        target, reference = data['blind_costs'][:, h], uniform_costs[:, h]
        chosen = (reference <= reference.min(-1, keepdims=True) + 1e-12).argmax(-1)
        baselines.append({'regime': regime, 'horizon': horizon, 'cases': n,
            'blind_cost_mse': math.fsum(float(x) ** 2 for x in (reference - target).ravel()) / (n * 4),
            'blind_regret': math.fsum(float(target[i, chosen[i]] - min(target[i])) for i in range(n)) / n})
    for arm in ARMS:
        for seed in spec.seeds:
            check()
            p = {field: predictions[f'{arm}__{seed}__{field}'] for field in FIELDS}
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
                require(all(math.isfinite(value) for value in row.values() if type(value) is float), 'finite independently computed metrics')
                rows.append(row)
    return rows, baselines



def validate_execution(summary, allocations, datasets, prefixes, orders, np, *, check=lambda: None, spec=SCIENCE[0], training_only=False):
    """Reconstruct attempted batches and forward work, including discarded updates."""
    declared = summary['structural_work']
    require(set(declared) == set(ARMS) and all(set(declared[arm]) == set(ROUTES) for arm in ARMS),
            'all common structural-work routes')
    expected = {arm: {route: dict.fromkeys(PREFIX_WORK_KEYS if route.endswith('_prefix') else ENDPOINT_WORK_KEYS, 0)
                      for route in ROUTES} for arm in ARMS}
    order_map, order_sequence = {}, []
    for row in orders:
        check()
        require(type(row) is dict and set(row) == {'arm', 'seed', 'epoch', 'indices', 'batch_size'}
                and row['arm'] in ARMS and row['seed'] in spec.seeds and type(row['epoch']) is int and row['epoch'] >= 0
                and row['batch_size'] == spec.batch_size and type(row['indices']) is list and len(row['indices']) == spec.train_attempts
                and all(type(v) is int for v in row['indices']) and sorted(row['indices']) == list(range(spec.train_attempts)),
                'one complete deterministic epoch order')
        key = row['arm'], row['seed'], row['epoch']
        require(key not in order_map, 'each used epoch serialized once')
        rebuilt = np.random.Generator(np.random.PCG64(np.random.SeedSequence([row['seed'], row['epoch'], 818]))).permutation(spec.train_attempts)
        require(np.array_equal(rebuilt, np.asarray(row['indices'], np.int64)), 'registered pure cursor-to-permutation schedule')
        order_map[key] = rebuilt
        order_sequence.append(key)
    observed_orders = []
    seen_orders = set()
    aggregate = dict.fromkeys(UPDATE_WORK, 0)
    retained_work = dict.fromkeys(UPDATE_WORK, 0)
    rejected_work = dict.fromkeys(UPDATE_WORK, 0)

    def add(arm, route, values, *, prior=False):
        values = forward_work(arm, values, prefix=route.endswith('_prefix'), prior=prior)
        for key, value in values.items():
            expected[arm][route][key] += value

    losses = {}
    schedules = {}
    fit_map = {x['arm']: x for x in summary['fits']}
    for (arm, seed), allocation in allocations.items():
        losses[arm] = dict.fromkeys(LOSS_WORK_KEYS, 0)
        schedules[arm] = []
        for row in allocation['trace']:
            check()
            kind, diagnostics = row['kind'], row['result']['diagnostics']
            work = dict.fromkeys(UPDATE_WORK, 0)
            work.update(backward_passes=1, adam_steps=1)
            if kind == 'prefix':
                require(set(diagnostics) == {'kind', 'valid_events', 'attempts', 'pseudocount', 'cost_head_updated'}
                        and diagnostics['kind'] == 'prefix' and diagnostics['valid_events'] == int(prefixes['train']['event_mask'].sum())
                        and diagnostics['attempts'] == spec.train_attempts and diagnostics['pseudocount'] == .001
                        and diagnostics['cost_head_updated'] is False, 'registered full-public-prefix objective')
                work.update(prefix_updates=1, prefix_event_exposures=diagnostics['valid_events'], prefix_full_rollouts=1)
                add(arm, 'training_prefix', prefix_work(prefixes['train'], np.arange(spec.train_attempts), np), prior=True)
            else:
                cursor = row['cursor_before']
                epoch, offset = divmod(cursor, math.ceil(spec.train_attempts / spec.batch_size))
                key = arm, seed, epoch
                require(key in order_map, 'every attempted batch has its original epoch order')
                if key not in seen_orders:
                    seen_orders.add(key)
                    observed_orders.append(key)
                order = order_map[key]
                indices = order[offset * spec.batch_size:(offset + 1) * spec.batch_size]
                selected = prefixes['train']['endpoint_rows'][indices]
                selected = selected[selected >= 0]
                events = int(prefixes['train']['event_mask'][indices].sum())
                extra = loss_work(len(selected), 2, LOSS[arm])
                integer_work(diagnostics['loss_work'], set(LOSS_WORK_KEYS))
                for key, value in extra.items():
                    losses[arm][key] += value
                schedules[arm].append(indices.tolist())
                compare(diagnostics, {'loss_kind': LOSS[arm], 'loss_work': extra, 'kind': 'joint', 'cursor': cursor, 'epoch': epoch, 'batch_offset': offset,
                    'indices': indices.tolist(), 'eligible': len(selected), 'valid_events': events, 'batch_size': len(indices),
                    'order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()},
                    path='actual attempted joint batch and fixed global denominators')
                work.update(joint_updates=1, joint_attempt_exposures=len(indices), joint_case_exposures=len(selected),
                    joint_event_exposures=events, joint_prefix_rollouts=1, zero_endpoint_batches=int(not len(selected)))
                if len(selected):
                    work.update(joint_blind_rollouts=1, joint_observed_rollouts=1)
                # Five blocks already include the one shared field construction.
                # Do not apply forward_work again or add them to legacy routes.
                blocks = reuse_work(TRANSPORT[arm], prefixes['train'], indices,
                                    datasets['train']['observations'][selected], np)
                for route, values in blocks.items():
                    for key, value in values.items():
                        expected[arm][route][key] += value
            integer_work(row['result']['work'], UPDATE_WORK)
            compare(row['result']['work'], work, path='independent attempted update and exposure counts')
            destination = retained_work if row['accepted'] else rejected_work
            for key, value in work.items():
                aggregate[key] += value
                destination[key] += value
    require(order_sequence == observed_orders, 'complete first-use epoch order journal without extra generated orders')
    for arm in ARMS:
        strict(fit_map[arm]['loss_work'], losses[arm], 'per-fit loss-only work')
        strict(allocations[arm, spec.seeds[0]]['final_summary']['loss_work'], losses[arm], 'final loss-work export')
    require(all(schedules[a] == schedules[ARMS[0]] for a in ARMS), 'identical full joint case exposure across six loss arms')
    paired = hashlib.sha256(json.dumps(schedules[ARMS[0]], separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    strict(summary['paired_batch_sha256'], paired, 'paired batch digest')
    regimes = () if training_only else REGIMES
    for arm in ARMS:
        for _seed in spec.seeds:
            for regime in regimes:
                for start in range(0, len(datasets[regime]['case_ids']), spec.batch_size):
                    observations = datasets[regime]['observations'][start:start + spec.batch_size]
                    for route in ('evaluation_blind', 'evaluation_observed', 'evaluation_shuffled'):
                        add(arm, route, endpoint_work(observations, route == 'evaluation_observed', np))
                for start in range(0, spec.dev_attempts, spec.batch_size):
                    add(arm, 'evaluation_prefix', prefix_work(prefixes[regime],
                        np.arange(start, min(start + spec.batch_size, spec.dev_attempts)), np))
    for arm in ARMS:
        for route in ROUTES:
            value = declared[arm][route]
            integer_work(value)
            if value == {}:
                require(not any(expected[arm][route].values()), 'unused route alone may have no counters')
            else:
                compare(value, expected[arm][route], path='independently derived forward geometry: ' + arm + '/' + route)
    n = len(datasets['train']['case_ids'])
    dev_sizes = [len(datasets[regime]['case_ids']) for regime in regimes]
    attempted = sum(a['attempted_updates'] for a in allocations.values())
    fits = len(ARMS) * len(spec.seeds)
    eval_batches = fits * sum(math.ceil(size / spec.batch_size) for size in dev_sizes)
    oracle_batches = 0 if training_only else math.ceil(n / spec.batch_size) + sum(math.ceil(size / spec.batch_size) for size in dev_sizes)
    counts = {'train_generation_count': 1, 'dev_generation_count': len(regimes), 'model_constructions': fits,
        'oracle_model_constructions': len(regimes), 'fit_count': fits, 'checkpoint_writes': 3 * fits, 'optimizer_checkpoint_writes': 3 * fits,
        'training_blind_rollouts': aggregate['joint_blind_rollouts'], 'training_observed_rollouts': aggregate['joint_observed_rollouts'],
        'optimizer_attempts': attempted, 'optimizer_steps': attempted,
        'training_case_exposures': aggregate['joint_case_exposures'],
        'training_attempt_exposures': aggregate['joint_attempt_exposures'],
        'training_prefix_rollouts': aggregate['joint_prefix_rollouts'] + aggregate['prefix_full_rollouts'],
        'training_prefix_event_exposures': aggregate['joint_event_exposures'] + aggregate['prefix_event_exposures'],
        'zero_endpoint_batches': aggregate['zero_endpoint_batches'],
        'evaluation_blind_rollouts': eval_batches, 'evaluation_observed_rollouts': eval_batches,
        'evaluation_shuffled_rollouts': eval_batches, 'evaluation_case_views': fits * sum(dev_sizes),
        'evaluation_prefix_rollouts': len(regimes) * fits * math.ceil(spec.dev_attempts / spec.batch_size),
        'evaluation_prefix_event_views': fits * sum(int(prefixes[regime]['event_mask'].sum()) for regime in regimes),
        'oracle_blind_rollouts': oracle_batches, 'oracle_observed_rollouts': oracle_batches,
        'array_decodes': 0, 'checkpoint_decodes': 0, 'external_model_calls': 0, 'native_calls': 0, 'teacher_calls': 0,
        'readout_snapshot_evaluations': 0, 'readout_snapshot_softmax_evaluations': 0, 'dynamics_snapshot_evaluations': 0,
        'epoch_order_generations': len(orders), 'accepted_optimizer_steps': retained_work['adam_steps'],
        'accepted_joint_steps': retained_work['joint_updates'], 'accepted_prefix_steps': retained_work['prefix_updates']}
    declared_counts = summary['counts']
    wanted = {key: counts[key] for key in TRAIN_COUNT_KEYS} if training_only else counts
    integer_work(declared_counts, set(wanted))
    require(declared_counts == wanted, 'exact executed and accepted counts')
    require(not any(rejected_work.values())
            and aggregate == retained_work
            and attempted == fits * (spec.prefix_updates + spec.joint_updates),
            'successful exact-update study has no discarded work or missing exposure')
    return {'arms': 6, 'routes': 72, 'counts': counts, 'loss_work': losses, 'epoch_permutations_reconstructed': len(orders),
            'attempted_update_work': aggregate, 'accepted_update_work': retained_work, 'discarded_update_work': rejected_work,
            'structural_forward_geometry_checked': True, 'model_updates_replayed': False}


TRAIN_COUNT_KEYS = ('model_constructions', 'checkpoint_writes', 'optimizer_checkpoint_writes',
    'optimizer_attempts', 'optimizer_steps', 'epoch_order_generations', 'training_attempt_exposures',
    'training_case_exposures', 'training_prefix_event_exposures', 'training_blind_rollouts',
    'training_observed_rollouts', 'training_prefix_rollouts', 'zero_endpoint_batches',
    'accepted_optimizer_steps', 'accepted_joint_steps', 'accepted_prefix_steps', 'fit_count')
EXPOSURE = r.StudyProfile('exposure', 948201, (948301,), 512, 8, 64, 32, 64, 30.)


def verify_boundaries(allocations, arrays, optimizers, declared_pairs, np, *, spec, check=lambda: None):
    """Inspect every retained parameter/Adam boundary, never replay an update."""
    seed, = spec.seeds
    require(set(allocations) == {(arm, seed) for arm in ARMS}, 'six complete allocations')
    expected_keys = {(label, arm) for arm in ARMS for label in ('initial', 'boundary', 'final')}
    require(set(arrays) == set(optimizers) == expected_keys, 'complete boundary byte roster')
    states, checks, diagnostics = {}, [], []
    initial_head = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 436, 1]))).standard_normal((4, 8))
    for arm in ARMS:
        allocation = allocations[arm, seed]
        for point in allocation['checkpoints']:
            check()
            label = point['label']
            hashes = h.boundary_hashes(arrays[label, arm], np)
            states[label, arm] = hashes
            meta = point['metadata']
            require(hashes['full'] == point['model_sha256'] == meta['model_state_sha256']
                and hashes['dynamics_state'] == meta['dynamics_state_sha256']
                and hashes['head_state'] == meta['head_state_sha256'], 'decoded full/dynamics/head boundary joins')
            kind, steps = ('joint', spec.joint_updates) if label == 'final' else ('prefix', spec.prefix_updates if label == 'boundary' else 0)
            opt = h.optimizer_state(optimizers[label, arm], kind, steps)
            require(opt['sha256'] == point['optimizer_sha256'] == meta['optimizer_state_sha256'], 'decoded Adam boundary joins')
            raw = arrays[label, arm]['transition_logits']
            if TRANSPORT[arm] == 'rounded':
                transition, correction = h.rounded_transition_from_logits(raw, np)
            else:
                transition, correction = h.column_transition(raw, np), None
            require(np.isfinite(transition).all() and (transition > 0).all() and (transition < 1).all(), 'strict boundary transition law')
            diagnostics.append({'arm': arm, 'seed': seed, 'label': label, 'transport_arm': TRANSPORT[arm],
                'row_residual_max': float(np.max(np.abs(transition.sum(2)-1))),
                'column_residual_max': float(np.max(np.abs(transition.sum(1)-1))), 'correction': correction})
            checks.append({'arm': arm, 'seed': seed, 'label': label, 'model': hashes, 'optimizer': opt})
        require(arrays['initial', arm]['cost_logits'].tobytes() == initial_head.tobytes(), 'independent task-free random head stream')
        require(states['initial', arm]['head_state'] == states['boundary', arm]['head_state'], 'head unchanged by prefix-only objective')
        fresh = {'state': {}, 'param_groups': [{**optimizers['initial', arm]['param_groups'][0], 'params': [0, 1, 2, 3]}]}
        reset = h.optimizer_state(fresh, 'joint', 0)
        require(reset['sha256'] == allocation['boundary']['optimizer_after_sha256'], 'fresh four-parameter Adam at joint boundary')
    for arm in ARMS[1:]:
        for name in ('emission_logits', 'hazard_logits', 'cost_logits'):
            require(np.array_equal(arrays['initial', arm][name], arrays['initial', ARMS[0]][name]), 'paired initial emission, hazard and independent head')
    groups = []
    for transport in ('rounded', 'matched_free'):
        arms = [a for a in ARMS if TRANSPORT[a] == transport]
        first = arms[0]
        for arm in arms[1:]:
            for label in ('initial', 'boundary'):
                for name in h.PARAMETER_SHAPES:
                    require(np.array_equal(arrays[label, first][name], arrays[label, arm][name]), 'identical architecture-local initial and prefix state')
                require(states[label, arm]['full'] == states[label, first]['full'], 'bitwise paired full model state')
                left_opt = next(x for x in allocations[arm, seed]['checkpoints'] if x['label'] == label)
                right_opt = next(x for x in allocations[first, seed]['checkpoints'] if x['label'] == label)
                require(left_opt['optimizer_sha256'] == right_opt['optimizer_sha256'], 'bitwise paired Adam state')
                strict(optimizers[label, arm], optimizers[label, first], 'identical architecture-local prefix Adam')
        points = {p['label']: p for p in allocations[first, seed]['checkpoints']}
        groups.append({'transport_arm': transport, 'arms': arms,
            'initial_model_sha256': states['initial', first]['full'], 'boundary_model_sha256': states['boundary', first]['full'],
            'initial_optimizer_sha256': points['initial']['optimizer_sha256'], 'boundary_optimizer_sha256': points['boundary']['optimizer_sha256'],
            'same_initial_and_prefix_model': True, 'same_prefix_optimizer_states': True})
    left, _ = h.rounded_transition_from_logits(arrays['initial', 'rounded_mse']['transition_logits'], np)
    right = h.column_transition(arrays['initial', 'free_mse']['transition_logits'], np)
    error = float(np.max(np.abs(left-right)))
    require(error <= 1e-12, 'matched initial transition functions across architectures')
    paired = {'seed': seed, 'groups': groups,
        'head_initial_boundary_sha256': {arm: states['initial', arm]['head_state'] for arm in ARMS},
        'all_initial_heads_paired': True, 'all_heads_unchanged_after_prefix': True,
        'scope': 'Producer metadata joins before any DEV; saved bytes and initial transition functions independently audited.'}
    strict(declared_pairs, paired, 'pre-DEV pairing record matches independently decoded boundaries')
    return {'checks': checks, 'transition_diagnostics': diagnostics, 'initial_transition_max_abs': error,
        'initializer_reconstructions': 1, 'prefix_pair_checks': paired, 'optimizer_updates_replayed': False}


def verify_population(prefix_record, data, declared, np, *, spec, split):
    for key, reported in (('attempts', 'attempted'), ('retained', 'retained'), ('discarded_found', 'discarded_found'),
            ('valid_events', 'valid_prefix_events'), ('prefix_found_by_step', 'prefix_found_by_step')):
        strict(prefix_record[key], declared[reported], 'all-attempt public-prefix count '+key)
    attempted = spec.train_attempts if split == 'train' else spec.dev_attempts
    horizon = 2 if split == 'train' else 8
    require(declared['version'] == 'finite-prefix-learning-data-v1' and declared['seed_namespace'] == spec.namespace
        and declared['split_id'] == r.SPLIT_IDS[split] and declared['epsilon'] == r.EPSILON[split]
        and declared['horizon'] == horizon and declared['attempted'] == attempted,
        'independent cohort population identity')
    found = data['observations'] == 4
    cases = found.any(1)
    draws = np.where(cases, found.argmax(1)+1, horizon)
    n = len(data['case_ids'])
    require(declared['forecast_found_cases'] == int(cases.sum()) and declared['forecast_event_draws'] == int(draws.sum())
        and declared['forecast_action_draws'] == horizon*n and declared['initial_odor_draws'] == attempted
        and declared['prefix_action_draws'] == 8*attempted and declared['prefix_event_draws'] == prefix_record['valid_events']-attempted,
        'independent source draw geometry')


def evaluate_rule(rows, baselines, populations, *, rule=RULE):
    """Fixed candidate, equal-cohort aggregation; no rescue by another arm/regime."""
    require(type(rule) in (dict, type(RULE)), 'explicit prospective rule object')
    normalized = {key: tuple(value) if key in ('controls', 'regimes', 'horizons') and type(value) is list else value for key,value in rule.items()}
    strict(normalized, dict(RULE), 'explicit immutable prospective rule')
    rule = normalized
    cohort_ids = sorted(populations)
    require(cohort_ids == list(range(len(cohort_ids))) and cohort_ids, 'complete consecutive independent cohorts')
    indexed = {(x['cohort_index'], x['arm'], x['regime'], x['horizon']): x for x in rows}
    reference = {(x['cohort_index'], x['regime'], x['horizon']): x for x in baselines}
    require(len(indexed) == len(rows) == len(cohort_ids)*len(ARMS)*len(REGIMES)*len(HORIZONS)
        and set(indexed) == {(c,a,g,t) for c in cohort_ids for a in ARMS for g in REGIMES for t in HORIZONS}, 'every endpoint row exactly once')
    require(len(reference) == len(baselines) == len(cohort_ids)*len(REGIMES)*len(HORIZONS)
        and set(reference) == {(c,g,t) for c in cohort_ids for g in REGIMES for t in HORIZONS}, 'every independent reference exactly once')
    for row in rows + baselines:
        require(all(finite_scalar(v) for k,v in row.items() if k in ('blind_regret','blind_cost_mse','observed_kl','blind_survival_mae')), 'finite gate metrics')
    names = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
    gates = {g:{} for g in REGIMES}
    for regime in REGIMES:
        for arm in ARMS:
            per_cohort = {}
            for c in cohort_ids:
                support = populations[c]['train'] >= rule['minimum_train'] and populations[c][regime] >= rule['minimum_development']
                criteria = {name:{'minimum_support':support} for name in names}
                for horizon in HORIZONS:
                    row, base = indexed[c,arm,regime,horizon], reference[c,regime,horizon]
                    target = names[0] if horizon in (1,2) else names[1]
                    for metric in ('blind_cost_mse','blind_regret'):
                        criteria[target][f'h{horizon}_{metric}'] = base[metric] > 0 and row[metric] <= rule['maximum_reference_ratio']*base[metric]
                    target = names[0] if horizon in (1,2) else names[2]
                    criteria[target][f'h{horizon}_observed_kl'] = row['observed_kl'] <= rule['maximum_observed_kl']
                criteria[names[1]]['h8_survival'] = indexed[c,arm,regime,8]['blind_survival_mae'] <= rule['maximum_h8_survival_mae']
                per_cohort[str(c)] = {name:{'passed':all(parts.values()), 'conditions':parts} for name,parts in criteria.items()}
            gates[regime][arm] = {name:{'passed':all(per_cohort[str(c)][name]['passed'] for c in cohort_ids),
                'cohorts':{str(c):per_cohort[str(c)][name] for c in cohort_ids}} for name in names}
    candidate = rule['candidate']
    conditions = {f'{regime}_{name}':gates[regime][candidate][name]['passed'] for regime in REGIMES for name in names}
    pairs, means = [], []
    for control in rule['controls']:
        for regime in rule['regimes']:
            for horizon in rule['horizons']:
                candidate_values, control_values = [], []
                for c in cohort_ids:
                    a, b = indexed[c,candidate,regime,horizon], indexed[c,control,regime,horizon]
                    require(a['seed'] == b['seed'] and a['seed_namespace'] == b['seed_namespace'], 'within-cohort paired loss contrast')
                    delta = a['blind_regret']-b['blind_regret']
                    conditions[f'{control}_{regime}_h{horizon}_cohort{c}_strict'] = delta < 0
                    pairs.append({'cohort_index':c,'seed_namespace':a['seed_namespace'],'seed':a['seed'],
                        'candidate':candidate,'control':control,'regime':regime,'horizon':horizon,
                        'candidate_regret':a['blind_regret'],'control_regret':b['blind_regret'],'difference':delta})
                    candidate_values.append(a['blind_regret']); control_values.append(b['blind_regret'])
                a, b = math.fsum(candidate_values)/len(cohort_ids), math.fsum(control_values)/len(cohort_ids)
                conditions[f'{control}_{regime}_h{horizon}_mean_reduction'] = b > 0 and a <= (1-rule['mean_reduction'])*b
                means.append({'candidate':candidate,'control':control,'regime':regime,'horizon':horizon,'cohorts':len(cohort_ids),
                    'candidate_regret':a,'control_regret':b,'difference':a-b,'reduction_fraction':None if b <= 0 else 1-a/b})
    require(len(conditions) == 14+8*len(cohort_ids), 'complete fixed absolute/paired/mean rule roster')
    descriptive = []
    for c in cohort_ids:
        for regime in REGIMES:
            for horizon in HORIZONS:
                for kind in ('mse','double','range'):
                    a,b = indexed[c,'rounded_'+kind,regime,horizon], indexed[c,'free_'+kind,regime,horizon]
                    descriptive.append({'cohort_index':c,'seed':a['seed'],'seed_namespace':a['seed_namespace'],
                        'loss_kind':kind,'regime':regime,'horizon':horizon,
                        **{metric:a[metric]-b[metric] for metric in ('blind_regret','blind_cost_mse','observed_kl','blind_survival_mae')}})
    passed = all(conditions.values())
    return {'rule':dict(rule),'gates':gates,'paired_loss_contrasts':pairs,'equal_cohort_means':means,
        'architecture_contrasts':descriptive,'architecture_contrasts_descriptive_only':True,
        'advance':{'name':rule['name'],'passed':passed,'status':rule['name']+('_PASS' if passed else '_FAIL'),'conditions':conditions}}


def decoder(folder, np, counters, check):
    def load(name, *, checkpoint=False):
        check()
        with np.load(folder/name, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), 'unique saved array members')
            result = {key: archive[key] for key in archive.files}
        counters['array_decodes'] += 1
        counters['checkpoint_decodes'] += int(checkpoint)
        return result
    return load


def decode_boundaries(folder, load, allocations, counters):
    arrays, optimizers = {}, {}
    for arm, seed in allocations:
        for label in ('initial', 'boundary', 'final'):
            arrays[label, arm] = load(f'{label}-{arm}-{seed}.npz', checkpoint=True)
            optimizers[label, arm] = read(folder/f'{label}-optimizer-{arm}-{seed}.json')
            counters['optimizer_json_decodes'] += 1
    return arrays, optimizers


def zero_counters():
    return dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'optimizer_json_decodes',
        'initializer_reconstructions', 'model_calls', 'optimizer_calls', 'world_or_generator_calls', 'native_calls'), 0)


def aggregate_work(cohorts):
    total = {arm:{route:{} for route in ROUTES} for arm in ARMS}
    for cohort in cohorts:
        for arm in ARMS:
            for route in ROUTES:
                for key, value in cohort['structural_work'][arm][route].items():
                    total[arm][route][key] = total[arm][route].get(key,0)+value
    return total


def audit(folder, *, check=lambda: None, profile='science', rule=RULE):
    """Saved evidence only; external wrapper owns original process/source admission."""
    require(type(profile) is str and profile in PROFILES, 'explicit immutable audit profile')
    specs = PROFILES[profile]
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder, 'absolute original output directory')
    summary = read(folder/'summary.json')
    allocations, orders = validate_metadata(folder,summary,specs,check)
    # No numerical import or array decode precedes the complete opaque inventory.
    import numpy as np
    counters = zero_counters()
    rows, baselines, prefix_rows, cohort_checks, ids = [], [], [], [], []
    populations = {}
    for index,(cohort,spec) in enumerate(zip(summary['cohorts'],specs,strict=True)):
        check()
        tags = {'cohort_index':index,'seed_namespace':spec.namespace}
        local = folder/cohort['folder']
        load = decoder(local,np,counters,check)
        arrays,optimizers = decode_boundaries(local,load,allocations[index],counters)
        boundary = verify_boundaries(allocations[index],arrays,optimizers,cohort['prefix_pair_checks'],np,spec=spec,check=check)
        counters['initializer_reconstructions'] += boundary['initializer_reconstructions']
        datasets,prefixes,targets,uniform,records,prefix_records = {},{},{},{},{},{}
        for split in SPLITS:
            data,prefix = load(split+'.npz'),load(split+'-prefix.npz')
            oracle,exact = load(split+'-oracle.npz'),load('oracle-'+split+'.npz')
            record,rebuilt,reference = r.reconstruct(data,oracle,exact,split,np,check=check,spec=spec)
            prefix_record,states,_ = r.reconstruct_prefix(prefix,data,split,np,check=check,spec=spec)
            require(float(np.max(np.abs(states-oracle['prefix_state']))) <= 1e-12, 'separate reference prefix is the public-history posterior')
            verify_population(prefix_record,data,cohort['dataset_counts'][split],np,spec=spec,split=split)
            errors = {name:float(np.max(np.abs(exact[name]-data[name]))) for name in TARGETS}
            compare(cohort['oracle_checks'][split],errors,path='independent exact-control error receipt')
            datasets[split],prefixes[split],targets[split],uniform[split] = data,prefix,rebuilt,reference
            records[split],prefix_records[split] = record,prefix_record
            ids.extend(str(x) for x in prefix['case_ids'])
        for regime in REGIMES:
            predictions = load('predictions-'+regime+'.npz')
            current,base = rows_for({**datasets[regime],**targets[regime]},predictions,uniform[regime],np,regime=regime,check=check,spec=spec)
            rows.extend({**x,**tags} for x in current)
            baselines.extend({**x,**tags} for x in base)
            seed, = spec.seeds
            for arm in ARMS:
                prediction = load(f'prefix-{arm}-{seed}-{regime}.npz')
                scored = r.prefix_row(prefixes[regime],prediction,TRANSPORT[arm],seed,np,regime=regime,spec=spec)
                prefix_rows.append({**scored,**tags,'arm':arm})
        local_fits = [{k:v for k,v in x.items() if k not in ('cohort_index','seed_namespace','artifact_folder')}
            for x in summary['fits'] if x['cohort_index'] == index]
        execution = validate_execution({**cohort,'fits':local_fits},allocations[index],datasets,prefixes,orders[index],np,check=check,spec=spec)
        populations[index] = {split:len(datasets[split]['case_ids']) for split in SPLITS}
        cohort_checks.append({**tags,'fit_seed':spec.seeds[0],'data_cases':populations[index],
            'target_reconstruction':records,'prefix_reconstruction':prefix_records,'boundaries':boundary,'execution':execution})
    require(len(ids) == len(set(ids)), 'all five complete cohort populations and all splits are disjoint')
    key = lambda x:(x['cohort_index'],x['regime'],x['arm'],x['seed'],x['horizon'])
    compare(sorted(summary['rows'],key=key),sorted(rows,key=key),path='all independent endpoint metrics')
    basekey = lambda x:(x['cohort_index'],x['regime'],x['horizon'])
    compare(sorted(summary['baseline_rows'],key=basekey),sorted(baselines,key=basekey),path='all independent uniform references')
    compare(summary['prefix_rows'],prefix_rows,path='all public event-weighted prefix metrics')
    summed = {key:sum(x['counts'][key] for x in summary['cohorts']) for key in summary['cohorts'][0]['counts']}
    strict(summary['counts'],summed,'global logical work is the exact sum of independent cohorts')
    strict(summary['structural_work'],aggregate_work(summary['cohorts']),'global physical work has no additional/shared cohort work')
    require(counters['array_decodes'] == 44*len(specs) and counters['checkpoint_decodes'] == 18*len(specs)
        and counters['optimizer_json_decodes'] == 18*len(specs) and counters['initializer_reconstructions'] == len(specs), 'complete fixed audit decode and initializer roster')
    decisions = evaluate_rule(rows,baselines,populations,rule=rule)
    # Ensure the audit itself did not mutate original evidence.
    for name,before in summary['files'].items():
        check()
        strict(descriptor(folder/name),before,'unchanged original bytes after independent audit')
    return {'version':VERSION,'profile':profile,'agreement':True,'technical_complete':False,
        'requires_original_supervisor_closure':True,'exact_oracle_agreement':True,'counts':counters,'cohorts':cohort_checks,
        'rows':rows,'baseline_rows':baselines,'prefix_rows':prefix_rows,**decisions,
        'fits':summary['fits'],'structural_work':summary['structural_work'],'logical_counts':summary['counts'],
        'metadata':{'payloads':2+75*len(specs),'fits':6*len(specs),'model_boundary_files':18*len(specs),
            'optimizer_boundary_files':18*len(specs),'global_checkpoint_barrier_verified':True},
        'architecture_claim':False,'latent_identification_claim':False,
        'limitations':[
            'External admission binds original sources, runtime and closed processes. This audit does not establish historical chronology by replay.',
            'Independent public-token targets, complete saved boundary parameters/Adam states, minibatches and work geometry are reconciled. Intermediate update losses, gradients, timings and hash chains are execution attestations, not replayed updates.',
            'Each cohort has independent data and fit seeds; the six arms share only within-cohort data. Means weight cohorts equally, not by retained cases.',
            'Only the fixed rounded_range versus rounded_mse and rounded_double rule is prospective. Free-architecture contrasts are descriptive and cannot rescue any failed condition.',
            'Equal updates and paired minibatches do not equalize wall time, floating-point operations or gradients. Extra scalar loss work is separate from disjoint reused forward work.',
            'SHIFT changes observation noise only, preserving transition, hazard and cost laws. No retraining occurs; observed filtering and blind extrapolation remain distinct.',
            'All cohorts and both regimes must satisfy the fixed rule. No significance, architectural novelty, state identification or native-transfer claim follows.']}


def audit_exposure(folder, *, check=lambda: None):
    """Independently reconcile the one fixed TRAIN-only engineering probe."""
    spec = EXPOSURE
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder, 'absolute original exposure directory')
    summary = read(folder/'summary.json')
    require(summary['version'] == 'finite-action-range-exposure-v1', 'one fixed engineering exposure')
    strict(summary['config'],spec.config,'fixed TRAIN-only exposure profile')
    strict(read(folder/'config.json'),spec.config,'durable exposure profile')
    strict(summary['arms'],list(ARMS),'all six engineering arms')
    strict(summary['fit_order'],list(ARMS),'fixed exposure order')
    seed, = spec.seeds
    names = {'config.json','train.npz','train-prefix.npz','dataset.json','fits.jsonl','training-orders.jsonl'}
    for arm in ARMS:
        names.add(f'allocation-{arm}-{seed}.json')
        for label in ('initial','boundary','final'):
            names |= {f'{label}-{arm}-{seed}.npz',f'{label}-optimizer-{arm}-{seed}.json'}
    require(set(summary['files']) == names,'exact TRAIN-only payload roster with no DEV')
    observed = set()
    for path in folder.rglob('*'):
        require(not path.is_symlink(),'no exposure symbolic links')
        if path.is_file() and path != folder/'summary.json':
            observed.add(str(path.relative_to(folder)))
    require(observed == names,'no omitted exposure files')
    for name in names:
        check()
        strict(descriptor(folder/name),summary['files'][name],'original engineering evidence')
    strict(summary['input_files'],{name:summary['files'][name] for name in ('train.npz','train-prefix.npz')},'training-only input descriptors')
    strict(summary['scope_counts'],{'train_generation_count':1,'dev_generation_count':0,'array_decodes':0,
        'checkpoint_decodes':0,'evaluation_rollouts':0,'oracle_model_constructions':0},'no DEV or reference-model execution')
    strict(summary['dataset_counts'],read(folder/'dataset.json'),'durable TRAIN population metadata')
    fits = read(folder/'fits.jsonl',lines=True)
    strict(summary['fits'],fits,'complete original engineering fits')
    require([(x['arm'],x['seed']) for x in fits] == [(a,seed) for a in ARMS], 'complete exposure fit roster')
    declared = summary['dataset_counts']
    allocations = {(x['arm'],seed):validate_fit(folder,x,spec,declared['retained'],declared['valid_prefix_events']) for x in fits}
    orders = read(folder/'training-orders.jsonl',lines=True)
    import numpy as np
    counters = zero_counters()
    load = decoder(folder,np,counters,check)
    data,prefix = load('train.npz'),load('train-prefix.npz')
    prefix_record,states,_ = r.reconstruct_prefix(prefix,data,'train',np,check=check,spec=spec)
    # Reconstruct targets from public tokens. The pure helper also compares its
    # targets to the same saved target fields; there is no exported oracle model
    # in this probe and no exact-model execution is claimed here.
    record,_,_ = r.reconstruct(data,{'prefix_state':states},{name:data[name] for name in TARGETS},'train',np,check=check,spec=spec)
    record.pop('oracle_maximum_absolute_errors')
    verify_population(prefix_record,data,declared,np,spec=spec,split='train')
    arrays,optimizers = decode_boundaries(folder,load,allocations,counters)
    boundary = verify_boundaries(allocations,arrays,optimizers,summary['prefix_pair_checks'],np,spec=spec,check=check)
    counters['initializer_reconstructions'] += 1
    execution = validate_execution(summary,allocations,{'train':data},{'train':prefix},orders,np,check=check,spec=spec,training_only=True)
    per_arm = {}
    for arm in ARMS:
        trace = allocations[arm,seed]['trace']
        work = {key:sum(row['result']['work'][key] for row in trace) for key in UPDATE_WORK}
        per_arm[arm] = {'model_constructions':1,'checkpoint_writes':3,'optimizer_checkpoint_writes':3,
            'optimizer_attempts':spec.prefix_updates+spec.joint_updates,'optimizer_steps':spec.prefix_updates+spec.joint_updates,
            'training_prefix_rollouts':work['prefix_full_rollouts']+work['joint_prefix_rollouts'],
            'training_prefix_event_exposures':work['prefix_event_exposures']+work['joint_event_exposures'],
            'epoch_order_generations':sum(row['arm'] == arm for row in orders),
            'training_blind_rollouts':work['joint_blind_rollouts'],'training_observed_rollouts':work['joint_observed_rollouts'],
            'zero_endpoint_batches':work['zero_endpoint_batches'],'training_attempt_exposures':work['joint_attempt_exposures'],
            'training_case_exposures':work['joint_case_exposures'],'accepted_optimizer_steps':work['adam_steps'],
            'accepted_joint_steps':work['joint_updates'],'accepted_prefix_steps':work['prefix_updates'],'fit_count':1}
    strict(summary['per_arm_counts'],per_arm,'independent per-arm training count reconciliation')
    require(finite_scalar(summary['generation_seconds']) and summary['generation_seconds'] >= 0, 'finite generation duration')
    require(counters['array_decodes'] == 20 and counters['checkpoint_decodes'] == counters['optimizer_json_decodes'] == 18,
        'exact two TRAIN arrays plus eighteen saved model boundaries')
    for name,before in summary['files'].items():
        check()
        strict(descriptor(folder/name),before,'unchanged exposure evidence after audit')
    return {'version':VERSION,'profile':'exposure','agreement':True,'descriptive_only':True,'no_dev_generated':True,
        'technical_complete':False,'requires_original_supervisor_closure':True,'counts':counters,
        'target_reconstruction':record,'prefix_reconstruction':prefix_record,'boundaries':boundary,
        'execution':execution,'per_arm_counts':per_arm,'payloads':len(names),
        'scope':'TRAIN-only engineering reconciliation, no predictive DEV metrics, model calls, selection or scientific advancement.'}
