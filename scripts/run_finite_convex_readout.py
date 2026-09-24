"""Nine frozen parents, TRAIN-only convex heads, then one fresh DEV pool."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from run_finite_factorized_dynamics import ORACLE_FIELDS
from run_finite_factorized_dynamics import predict as exact_predict
from run_finite_observation_learning import (
    FIELDS,
    _append,
    _baseline_rows,
    _desc,
    _rows,
    _save,
    _state,
    _state_hash,
    _write,
    require,
)

from openjev.research.finite_convex_readout import build_problem, solve
from openjev.research.finite_factor_models import make_model as exact_model
from openjev.research.finite_factorized_dynamics_models import make_model
from openjev.research.finite_prefix_learning import generate_attempt_split

VERSION = 'finite-convex-readout-v1'
PARENTS = ('factorized', 'matched_free', 'dense_free')
HEADS = ('original', 'solved')
ROUTES = ('train_blind', 'train_observed', *(f'{h}_{r}' for h in HEADS for r in ('blind', 'observed', 'shuffled')))
COUNTERS = ('train_array_decodes', 'checkpoint_decodes', 'model_constructions', 'oracle_model_constructions',
            'original_head_exports', 'original_head_softmaxes', 'train_blind_calls', 'train_observed_calls',
            'train_readout_validation_products', 'train_readout_validation_rows', 'solver_calls', 'completed_solves',
            'dev_generation_count', 'oracle_blind_rollouts', 'oracle_observed_rollouts',
            'evaluation_blind_calls', 'evaluation_observed_calls', 'evaluation_shuffled_calls',
            'head_matrix_products', 'head_matrix_rows', 'evaluation_case_views',
            'checkpoint_writes', 'optimizer_steps', 'external_model_calls', 'teacher_calls', 'native_calls')


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(part) for part in value]
    return value


def raw_hash(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def same_bytes(first, second):
    return first.shape == second.shape and first.dtype == second.dtype and first.tobytes() == second.tobytes()


def decode(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def public(data, indices):
    return {name: torch.from_numpy(data[name][indices].copy()) for name in ('prefix', 'lengths', 'actions', 'observations')}


def rollout(model, batch, observed=False):
    args = (batch['prefix'], batch['lengths'], batch['actions'])
    return model.observed_rollout(*args, batch['observations']) if observed else model.blind_rollout(*args)


def array(tensor):
    result = tensor.detach().cpu().numpy().copy()
    require(result.dtype == np.float64 and np.isfinite(result).all(), 'finite float64 inference export')
    return result


def charge(work, parent, route, result):
    values = result['work']
    require(all(type(v) is int and v >= 0 for v in values.values()), 'integer forward work')
    target = work[parent][route]
    require(not target or set(target) == set(values), 'stable forward-work schema')
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def extract_train(model, parent, data, costs, config, counts, work, check):
    n = len(data['prefix'])
    states = {key: np.empty((n, 2, 8), np.float64) for key in ('blind_states', 'observed_states')}
    validation_error = 0.
    with torch.no_grad():
        for start in range(0, n, config['batch_size']):
            check()
            stop = min(n, start + config['batch_size'])
            batch = public(data, slice(start, stop))
            for observed, route in ((False, 'blind'), (True, 'observed')):
                counts[f'train_{route}_calls'] += 1
                result = rollout(model, batch, observed)
                charge(work, parent, 'train_' + route, result)
                value = array(result['prior_states'])
                states[route + '_states'][start:stop] = value
                mapped = value @ costs.T
                counts['train_readout_validation_products'] += 1
                counts['train_readout_validation_rows'] += (stop - start) * 2
                validation_error = max(validation_error, float(np.max(np.abs(mapped - array(result['cost_contrasts'])))))
    require(validation_error <= 1e-12, 'original TRAIN linear head agrees with original forwards')
    return states, validation_error


def evaluate(model, parent, head, seed, data, costs, config, counts, work, check):
    n, horizon = data['actions'].shape
    arrays = {name: np.empty((n, horizon, 5 if name == 'observed_probabilities' else 4), np.float64)
              if name.endswith('costs') or name == 'observed_probabilities'
              else np.empty((n, horizon), np.float64) for name in FIELDS}
    states = {name: np.empty((n, horizon, 8), np.float64)
              for name in ('blind_states', 'observed_states', 'shuffled_states')}
    states.update({name: np.empty((n, 8), np.float64) for name in ('prefix_states', 'shuffled_prefix_states')})
    validation_error = 0.
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, config['batch_size']):
            check()
            stop = min(n, start + config['batch_size'])
            batch = public(data, slice(start, stop))
            moved = public(data, (np.arange(start, stop) + 1) % n)
            moved['actions'], moved['observations'] = batch['actions'], batch['observations']
            prefix = None
            for route, current, observed in (('blind', batch, False), ('observed', batch, True), ('shuffled', moved, False)):
                counts[f'evaluation_{route}_calls'] += 1
                result = rollout(model, current, observed)
                charge(work, parent, head + '_' + route, result)
                value = array(result['prior_states'])
                states[route + '_states'][start:stop] = value
                mapped = value @ costs.T
                counts['head_matrix_products'] += 1
                counts['head_matrix_rows'] += (stop - start) * horizon
                cost_key = 'shuffled_blind_costs' if route == 'shuffled' else route + '_costs'
                if head == 'original':
                    original = array(result['cost_contrasts'])
                    validation_error = max(validation_error, float(np.max(np.abs(mapped - original))))
                    arrays[cost_key][start:stop] = original
                else:
                    arrays[cost_key][start:stop] = mapped
                current_prefix = array(result['prefix_state'])
                if route == 'blind':
                    prefix = current_prefix
                    states['prefix_states'][start:stop] = prefix
                elif route == 'observed':
                    require(same_bytes(prefix, current_prefix), 'same blind/observed prefix filter')
                else:
                    states['shuffled_prefix_states'][start:stop] = current_prefix
                if route != 'shuffled':
                    arrays[route + '_survival'][start:stop] = array(result['survival_mass'])
                if observed:
                    arrays['observed_probabilities'][start:stop] = array(result['probabilities'])
            counts['evaluation_case_views'] += stop - start
    require(validation_error <= 1e-12, 'original DEV head agrees with posthoc linear map')
    return arrays, states, {'arm': parent + '_' + head, 'parent': parent, 'head': head, 'seed': seed,
        'seconds': time.perf_counter() - started, 'cases': n, 'horizon': horizon,
        'original_head_max_error': validation_error,
        'state_array_sha256': {key: raw_hash(value) for key, value in states.items()},
        'prediction_array_sha256': {key: raw_hash(value) for key, value in arrays.items()},
        'timing_scope': 'input copies, all three frozen forwards including original-head work, explicit head maps, guards and exported arrays; excludes file writes, scalar metrics and state hashes'}


def run(folder, config, check, *, upstream):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(COUNTERS, 0)
    work = {parent: {route: {} for route in ROUTES} for parent in config['parent_arms']}
    rows, solves, models, probabilities, times, invariants = [], [], {}, {}, [], []
    upstream_run = Path(upstream['run'])
    stage = 'original TRAIN and checkpoint authentication'
    started = time.perf_counter()
    try:
        _write(folder / 'config.json', config)
        parent_summary = json.loads((upstream_run / 'summary.json').read_text())
        parents = {(row['arm'], row['seed']): row for row in parent_summary['fits']}
        raw_train = (upstream_run / 'train.npz').read_bytes()
        require(_desc(upstream_run / 'train.npz') == parent_summary['files']['train.npz'], 'original TRAIN bytes')
        (folder / 'train.npz').write_bytes(raw_train)
        counts['train_array_decodes'] += 1
        train = decode(folder / 'train.npz')
        require(train['actions'].shape[1] == 2 and len(train['prefix']) > 0, 'original H2 TRAIN support')
        n = len(train['prefix'])
        for index, seed in enumerate(config['parent_seeds']):
            offset = index % len(config['parent_arms'])
            order = config['parent_arms'][offset:] + config['parent_arms'][:offset]
            for parent in order:
                stage = f'head solve {parent} {seed}'
                check()
                parent_row = parents[parent, seed]
                checkpoint = upstream_run / f'{parent}-{seed}.npz'
                require(_desc(checkpoint) == {k: parent_row['checkpoint'][k] for k in ('sha256', 'bytes')}, 'original checkpoint bytes')
                load_start = time.perf_counter()
                counts['checkpoint_decodes'] += 1
                values = decode(checkpoint)
                require(_state_hash(values) == parent_row['final_state_sha256'], 'original checkpoint semantic state digest')
                counts['model_constructions'] += 1
                model = make_model(parent, seed)
                model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in values.items()}, strict=True)
                model.eval()
                before = _state_hash(_state(model))
                require(before == parent_row['final_state_sha256'], 'strict original state load')
                with torch.no_grad():
                    p0 = array(model.cost_logits.softmax(0))
                counts['original_head_exports'] += 1
                counts['original_head_softmaxes'] += 1
                c0 = .25 - p0
                require(np.max(np.abs(c0 - np.asarray(parent_row['readout_final']['matrix']))) <= 1e-12,
                        'actual parent learned cost head')
                load_seconds = time.perf_counter() - load_start
                extraction_start = time.perf_counter()
                states, train_error = extract_train(model, parent, train, c0, config, counts, work, check)
                extraction_seconds = time.perf_counter() - extraction_start
                state_path = folder / f'states-train-{parent}-{seed}.npz'
                _save(state_path, states)
                solve_start = time.perf_counter()
                problem = build_problem(states['blind_states'], states['observed_states'], train['blind_costs'], train['observed_costs'])
                counts['solver_calls'] += 1
                result = solve(problem, p0, check=check)
                solve_seconds = time.perf_counter() - solve_start
                after = _state_hash(_state(model))
                require(after == before, 'solver and extraction leave every model parameter unchanged')
                row = {'parent': parent, 'seed': seed, 'training_cases': n, 'horizon': 2,
                    'source_checkpoint': {'path': str(checkpoint), **_desc(checkpoint)},
                    'source_state_sha256': before, 'model_state_after': after,
                    'original_probabilities': p0.tolist(), 'original_cost_matrix': c0.tolist(),
                    'solver_result': json_value(result), 'load_seconds': load_seconds,
                    'extraction_seconds': extraction_seconds, 'solve_seconds': solve_seconds,
                    'train_original_head_max_error': train_error,
                    'train_state_file': {'path': state_path.name, **_desc(state_path)},
                    'train_state_array_sha256': {key: raw_hash(value) for key, value in states.items()},
                    'timing_scope': 'load includes checkpoint decode, construction, strict load, head export and guards; extraction includes forwards and original-head maps; solve includes problem building, optimizer and certificate; journals/file writes charged to phase only',
                    'frozen_weights': True, 'optimizer_weight_updates': 0}
                _write(folder / f'solve-{parent}-{seed}.json', row)
                _append(folder / 'solves.jsonl', row)
                solves.append(row)
                counts['completed_solves'] += int(result['complete'])
                models[parent, seed] = model
                probabilities[parent, seed] = (p0, np.asarray(result['probabilities'], np.float64))
        require(len(solves) == counts['completed_solves'] == 3 * len(config['parent_seeds']),
                'every TRAIN-only solve must succeed before DEV, no replacement or fallback')
        barrier = {'solves': [{'parent': r['parent'], 'seed': r['seed'],
                     'file': f"solve-{r['parent']}-{r['seed']}.json",
                     **_desc(folder / f"solve-{r['parent']}-{r['seed']}.json")} for r in solves],
                   'completed_solves': counts['completed_solves'], 'dev_generation_count': 0}
        _write(folder / 'solve-barrier.json', barrier)
        stage = 'fresh DEV and independent exact control'
        check()
        generation_start = time.perf_counter()
        counts['dev_generation_count'] += 1
        generated = generate_attempt_split(1, config['dev_attempts'], config['dev_horizon'], seed_namespace=config['dev_seed_namespace'])
        data, oracle = generated['data'], generated['oracle']['prefix_state']
        _save(folder / 'base.npz', data)
        _save(folder / 'base-prefix.npz', generated['prefix_data'])
        _save(folder / 'base-oracle.npz', {'prefix_state': oracle})
        counts['oracle_model_constructions'] += 1
        reference = exact_model('exact_exact', 0)
        exact = exact_predict(reference, 'exact_exact', data, oracle, config, counts, check, shuffled=False)
        errors = {key: float(np.max(np.abs(exact[key] - data[key]))) for key in ORACLE_FIELDS}
        require(max(errors.values()) <= 1e-12, 'exact control reproduces every fresh target')
        _save(folder / 'oracle-base.npz', exact)
        _write(folder / 'oracle-base-check.json', errors)
        generation_seconds = time.perf_counter() - generation_start
        baselines = _baseline_rows(data, 'base', check)
        predictions = {}
        for parent in config['parent_arms']:
            for seed in config['parent_seeds']:
                model = models[parent, seed]
                before = _state_hash(_state(model))
                view_predictions, view_states = {}, {}
                for index, head in enumerate(HEADS):
                    stage = f'evaluate {parent} {head} {seed}'
                    costs = .25 - probabilities[parent, seed][index]
                    values, states, timing = evaluate(model, parent, head, seed, data, costs, config, counts, work, check)
                    require(_state_hash(_state(model)) == before, 'unchanged frozen model after each head view')
                    timing.update({'model_state_before': before, 'model_state_after': before})
                    times.append(timing)
                    _append(folder / 'prediction-times.jsonl', timing)
                    arm = parent + '_' + head
                    rows.extend(_rows(data, values, arm, seed, 'base', check))
                    predictions.update({f'{arm}__{seed}__{key}': value for key, value in values.items()})
                    view_predictions[head], view_states[head] = values, states
                parity = {key: same_bytes(view_predictions['original'][key], view_predictions['solved'][key])
                          for key in ('blind_survival', 'observed_survival', 'observed_probabilities')}
                parity.update({key: same_bytes(view_states['original'][key], view_states['solved'][key]) for key in view_states['original']})
                require(all(parity.values()), 'head changes cannot change frozen event, survival or state outputs')
                invariants.append({'parent': parent, 'seed': seed, 'bitwise_equal': parity})
                _save(folder / f'states-base-{parent}-{seed}.npz',
                      {head + '_' + key: value for head, state in view_states.items() for key, value in state.items()})
        _save(folder / 'predictions-base.npz', predictions)
        summary = {'version': VERSION, 'config': config, 'upstream': upstream,
                   'original_train_descriptor': parent_summary['files']['train.npz'],
                   'original_train_counts': parent_summary['dataset_counts']['train'],
                   'dataset_counts': {'base': generated['counts']}, 'solves': solves,
                   'rows': rows, 'baseline_rows': baselines, 'prediction_times': times,
                   'frozen_invariants': invariants, 'solve_barrier': barrier, 'counts': counts,
                   'structural_work': work, 'generation_seconds': generation_seconds,
                   'oracle_checks': errors, 'seconds_before_summary_write': time.perf_counter() - started,
                   'files': {p.name: _desc(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        _write(folder / 'summary.json', summary)
        check()
        return summary
    except BaseException as error:
        _write(folder / 'failure.json', {'version': VERSION, 'stage': stage, 'error': repr(error),
            'counts': counts, 'completed_solve_records': len(solves), 'seconds': time.perf_counter() - started})
        raise
