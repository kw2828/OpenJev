"""Independent saved-output BLA audit: no author package, optimizer or producer imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REG = 'research/fsm-author-bla-registration.json'
REGISTRATION_SHA256 = '03e01bf97630d4555338fbaa63ca15f749fd96df64e268940bbc8511d302c0a4'
VERSION = 'fsm-author-bla-study-v1'
BLA = 'author_bla28'
SEEDS = (9201, 9202, 9203)
DEV = tuple(f'{a}-realization-{r}-period-{p}' for a in ('100mV', '200mV') for r in range(3, 6) for p in range(2))
FIT = tuple(f'{a}-realization-{r}-period-{p}' for a in ('100mV', '200mV') for r in range(3) for p in range(2))
LINEAR = tuple(f'varx{p}-ridge{a:g}' for p in (32, 64, 96) for a in (1e-6, .001, .1))
NEURAL = ('affine_output_only-lr0.0001', 'affine_feedback-lr0.0001', 'tanh_output_only-lr0.001', 'tanh_feedback-lr0.0003')
CANDIDATE = NEURAL[-1]
FAMILIES = (*LINEAR, 'native_varx', *NEURAL, 'folded_affine_feedback')
ORDERED = [(f, None) for f in (*LINEAR, 'native_varx')]+[(f, s) for f in (*NEURAL, 'folded_affine_feedback') for s in SEEDS]
MODEL_KEYS = ('A', 'B_u', 'C_y', 'D_yu', 'u_mean', 'u_std', 'y_mean', 'y_std', 'ts')
BANK_KEYS = ('prediction', 'target', 'starts', 'final_state', 'singular_values', 'context_residual_norm', 'y_context', 'u_context', 'future_u')
EXPERIMENT = {'order': 28, 'nq': 29, 'max_iter': 5000, 'rtol': .001, 'atol': .00001, 'frequency_weighting': False, 'context': 100, 'horizon': 128, 'stride': 256, 'outer_timeout_seconds': 1800}
REPLAY_ATOL = REPLAY_RTOL = 1e-8


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def pin(path):
    blob = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def descriptor(path):
    return {'path': str(Path(path).resolve()), **pin(path)}


def relative(value):
    require(isinstance(value, str) and '\\' not in value and not Path(value).is_absolute() and all(p not in ('', '.', '..') for p in value.split('/')), 'canonical repository-relative path')
    return Path(value)


def inventory(folder):
    require(folder.is_dir() and not folder.is_symlink(), 'regular study directory')
    result = {}
    for p in sorted(folder.rglob('*')):
        require(not p.is_symlink(), 'symlink evidence')
        if p.is_file():
            result[str(p.relative_to(folder))] = pin(p)
    return result


def close(actual, expected, *, rtol=1e-10, atol=1e-12):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'mapping schema disagreement')
        for k in expected:
            close(actual[k], expected[k], rtol=rtol, atol=atol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), 'list schema disagreement')
        for a, b in zip(actual, expected, strict=True):
            close(a, b, rtol=rtol, atol=atol)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), 'scalar disagreement')
    else:
        require(type(actual) is type(expected) and actual == expected, 'exact value disagreement')


def load_arrays(path, keys):
    with np.load(path, allow_pickle=False) as src:
        require(len(src.files) == len(set(src.files)) and set(src.files) == set(keys), 'NPZ roster')
        return {k: src[k].copy(order='K') for k in keys}


def finite(value, shape):
    require(isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(), 'finite float64 geometry')


def model(arrays):
    require(set(arrays) == set(MODEL_KEYS), 'model roster')
    for k, shape in {'A': (28, 28), 'B_u': (28, 3), 'C_y': (3, 28), 'D_yu': (3, 3), 'u_mean': (3,), 'u_std': (3,), 'y_mean': (3,), 'y_std': (3,), 'ts': ()}.items():
        finite(arrays[k], shape)
    require((arrays['u_std'] > 0).all() and (arrays['y_std'] > 0).all() and float(arrays['ts']) == 1/6400, 'normalizer/time contract')


def metrics(prediction, target, scale):
    finite(prediction, (32, 128, 3)); finite(target, (32, 128, 3)); finite(scale, (3,))
    require((scale > 0).all(), 'positive scoring scales')
    with np.errstate(over='ignore', invalid='ignore'):
        square = (prediction-target)**2
        channel = np.sqrt(square.mean(axis=(0, 1)))
        native = channel*scale
        mse = float(square.mean())
    require(np.isfinite(square).all() and np.isfinite(native).all() and math.isfinite(mse), 'metric overflow')
    return {'mse': mse, 'rmse': math.sqrt(mse), 'per_channel_rmse': channel.tolist(), 'native_output_per_channel_rmse': native.tolist(), 'requests': 32, 'horizon': 128}


def replay(arrays, y_context, u_context, future_u):
    """Independent observability solve and scalar-time output-before-update loop."""
    model(arrays)
    batch = len(y_context)
    finite(y_context, (batch, 100, 3)); finite(u_context, (batch, 99, 3)); finite(future_u, (batch, 128, 3))
    y = (y_context-arrays['y_mean'])/arrays['y_std']
    u = (u_context-arrays['u_mean'])/arrays['u_std']
    future = (future_u-arrays['u_mean'])/arrays['u_std']
    a, b, c, d = (arrays[k] for k in ('A', 'B_u', 'C_y', 'D_yu'))
    power, forced = np.eye(28), np.zeros((batch, 28))
    observability, rhs = [], []
    for t in range(99):
        observability.append(c@power)
        rhs.append(y[:, t+1]-forced@c.T-u[:, t]@d.T)
        forced = forced@a.T+u[:, t]@b.T
        power = a@power
    o = np.concatenate(observability, axis=0)
    target = np.stack(rhs, axis=1).reshape(batch, -1).T
    require(np.isfinite(o).all() and np.isfinite(target).all(), 'nonfinite context system')
    solution, _, rank, singular = np.linalg.lstsq(o, target, rcond=1e-12)
    residual = np.linalg.norm(o@solution-target, axis=0)
    state = solution.T.copy()
    for t in range(99):
        state = state@a.T+u[:, t]@b.T
    outputs = []
    for t in range(128):
        outputs.append(state@c.T+future[:, t]@d.T)
        state = state@a.T+future[:, t]@b.T
    output = np.stack(outputs, axis=1)*arrays['y_std']+arrays['y_mean']
    for value in (output, state, singular, residual):
        require(np.isfinite(value).all(), 'nonfinite independent replay')
    return output, state, singular, residual, int(rank)


def frequency_error(arrays, target):
    model(arrays)
    require(target.dtype == np.complex128 and target.shape == (3839, 3, 3) and np.isfinite(target).all(), 'frequency target')
    a, b, c, d = (arrays[k] for k in ('A', 'B_u', 'C_y', 'D_yu'))
    z = np.exp(2j*np.pi*np.arange(1, 3840)/8192)
    response = c@np.linalg.solve(z[:, None, None]*np.eye(28)-a, np.broadcast_to(b, (3839, 28, 3)))+d
    result = float(np.mean(np.abs(response-target)**2))
    require(math.isfinite(result), 'nonfinite frequency score')
    return result


def decisions(evaluations, bla_rows, reference_complete):
    require([(e['family'], e['seed']) for e in evaluations] == ORDERED, '25 fixed comparison slots')
    groups = {f: [e for e in evaluations if e['family'] == f] for f in FAMILIES}
    means, per_record, per_seed = {}, {}, {}
    for family, members in groups.items():
        if all([r['record_id'] for r in e['rows']] == list(DEV) and all(r['status'] == 'complete' and type(r.get('rmse')) in (float, int) and math.isfinite(r['rmse']) and r['rmse'] >= 0 for r in e['rows']) for e in members):
            per_record[family] = {r: float(np.mean([next(x['rmse'] for x in e['rows'] if x['record_id'] == r) for e in members])) for r in DEV}
            per_seed[family] = {str(e['seed']): float(np.mean([r['rmse'] for r in e['rows']])) for e in members}
            means[family] = float(np.mean(list(per_record[family].values())))
    if reference_complete:
        require([r['record_id'] for r in bla_rows] == list(DEV) and all(r['status'] == 'complete' and math.isfinite(r['rmse']) and r['rmse'] >= 0 for r in bla_rows), 'complete BLA rows')
        per_record[BLA] = {r['record_id']: r['rmse'] for r in bla_rows}
        means[BLA] = float(np.mean(list(per_record[BLA].values())))
        per_seed[BLA] = {'None': means[BLA]}
    controls = [f for f in means if f != CANDIDATE]
    strongest = min(controls, key=lambda f: (means[f], f)) if controls else None
    conditions = dict.fromkeys(('five_percent_below_strongest_control', 'every_seed_below_strongest_control', 'no_record_over_two_percent_strongest_control', 'both_amplitudes_below_strongest_control'), False)
    if reference_complete and strongest is not None and CANDIDATE in means:
        c, b = per_record[CANDIDATE], per_record[strongest]
        conditions['five_percent_below_strongest_control'] = means[CANDIDATE] <= .95*means[strongest]
        conditions['every_seed_below_strongest_control'] = all(per_seed[CANDIDATE][str(s)] < per_seed[strongest].get(str(s), per_seed[strongest].get('None')) for s in SEEDS)
        conditions['no_record_over_two_percent_strongest_control'] = all(c[r] <= 1.02*b[r] for r in DEV)
        conditions['both_amplitudes_below_strongest_control'] = all(np.mean([c[r] for r in DEV if r.startswith(a)]) < np.mean([b[r] for r in DEV if r.startswith(a)]) for a in ('100mV', '200mV'))
    amplitudes = {f: {a: float(np.mean([v for r, v in rec.items() if r.startswith(a)])) for a in ('100mV', '200mV')} for f, rec in per_record.items()}
    return {'status': 'CONTINUE_REFERENCE_CHECK' if all(conditions.values()) else 'DO_NOT_CONTINUE_REFERENCE_CHECK', 'reference_complete': reference_complete, 'candidate': CANDIDATE, 'strongest_control': strongest, 'conditions': conditions, 'passed': sum(conditions.values()), 'total': 4, 'family_means': means, 'per_record_means': per_record, 'per_seed_means': per_seed, 'per_amplitude_means': amplitudes, 'seed_rule': 'Paired seed for stochastic controls; common deterministic score otherwise. No new candidate selection.'}


def authenticate(study, process):
    """Opaque identity and original closure admission before scientific decoding."""
    study, process = Path(study).resolve(), Path(process).resolve()
    require(study == ROOT/'output/fsm-author-bla-study-v1', 'registered study path')
    require(isinstance(REGISTRATION_SHA256, str) and pin(ROOT/REG)['sha256'] == REGISTRATION_SHA256, 'unbound/changed registration')
    plan = read(ROOT/REG)
    require(plan['version'] == VERSION and plan['experiment'] == EXPERIMENT, 'registered experiment')
    terminal = read(process)
    expected = [str(ROOT/'research/fsm_author/.venv/bin/python'), str(ROOT/'research/fsm_author/scripts/fit_bla.py'), '--registration', REG, '--output', 'output/fsm-author-bla-study-v1']
    require(terminal['command'] == expected and terminal['registration_sha256'] == REGISTRATION_SHA256 and terminal['timeout_seconds'] == 1800 and terminal['end_identity_matches'] is True, 'original process identity')
    require(terminal['status'] in ('completed', 'failed', 'timeout') and type(terminal['observed_exit_code']) is int and math.isfinite(terminal['elapsed_seconds']) and terminal['elapsed_seconds'] > 0, 'original process must be closed')
    require((terminal['status'] == 'completed') == (terminal['observed_exit_code'] == 0), 'process terminal consistency')
    log = process.parent/'process.log'
    require(pin(log)['sha256'] == terminal['log_sha256'], 'original log pin')
    for name, sha in plan['source_sha256'].items():
        require(pin(ROOT/relative(name))['sha256'] == sha, 'source drift')
    paths = {}
    for key, value in plan['prerequisites'].items():
        path = ROOT/relative(value['path'])
        require(pin(path)['sha256'] == value['sha256'], 'prerequisite drift')
        paths[key] = path
        if key.endswith('_qualification'):
            require(read(path)['status'] == 'PASS', 'failed qualification prerequisite')
    reference = ROOT/'output/fsm-linear-controls-study-v1'
    prior = read(paths['parent_audit'])
    require(prior['status'] == 'PASS' and prior['agreement'] is True, 'parent independent audit')
    require(paths['parent_summary'] == reference/'summary.json' and paths['parent_evaluations'] == reference/'evaluations.json', 'fixed reference metadata paths')
    for key, name in (('parent_summary', 'summary.json'), ('parent_evaluations', 'evaluations.json')):
        require(prior['inputs']['files'][name] == pin(paths[key]), 'reference audited scalar pin')
    require(prior['inputs']['registration'] == pin(paths['parent_registration']) and prior['inputs']['process']['sha256'] == pin(paths['parent_process'])['sha256'], 'parent registration/process join')
    parent_process, closure = read(paths['parent_process']), read(paths['parent_closure'])
    require(parent_process['observed_exit_code'] == 0 and closure['status'] == 'completed' and parent_process['summary_sha256'] == closure['summary_sha256'] == pin(paths['parent_summary'])['sha256'], 'parent original closure')
    require(prior['inputs']['files']['closure.json'] == pin(paths['parent_closure']), 'parent audited closure pin')
    for name, value in prior['inputs']['files'].items():
        require(pin(reference/relative(name)) == value, 'reference audited payload changed')
    before = inventory(study)
    for name in ('summary.json', 'fit.json', 'failure.json', 'admission.json'):
        if name in before:
            require(terminal[name+'_sha256'] == before[name]['sha256'], 'terminal output join')
    require(read(study/'started.json')['registration_sha256'] == REGISTRATION_SHA256 and read(study/'started.json')['pid'] == terminal['pid'], 'original child identity')
    inputs = {'study': str(study), 'registration': descriptor(ROOT/REG), 'process': descriptor(process), 'process_log': descriptor(log), 'files': before, 'reference_audit': descriptor(paths['parent_audit']), 'reference_summary': descriptor(paths['parent_summary']), 'reference_evaluations': descriptor(paths['parent_evaluations']), 'reference_files': prior['inputs']['files'], 'prerequisites': {k: descriptor(p) for k, p in paths.items()}}
    return plan, inputs, paths, terminal


def audit(study, process):
    study, process = Path(study).resolve(), Path(process).resolve()
    plan, inputs, paths, terminal = authenticate(study, process)
    if terminal['status'] != 'completed':
        return {'status': 'PASS', 'agreement': True, 'study': str(study), 'inputs': inputs, 'scientific_status': 'REFERENCE_FAILED', 'results': {'reference_status': 'REFERENCE_FAILED', 'continuation': None, 'failure': read(study/'failure.json') if (study/'failure.json').exists() else None}, 'counts': {'forecast_replays': 0, 'refits': 0, 'optimizer_calls': 0}, 'scope': 'Closed failed original attempt preserved; no successful-fit claim or fallback.'}
    before = inputs['files']
    expected = {'started.json', 'identity-before.json', 'identity-after.json', 'admission.json', 'events.jsonl', 'initial.zip', 'initial.npz', 'frequency-target.npz', 'final.zip', 'final.npz', 'solver-trace.npz', 'fit.json', 'evaluation.json', 'summary.json'}
    identity = read(study/'identity-before.json')
    require(identity == read(study/'identity-after.json') and identity['status'] == 'PASS' and identity['source_sha256'] == plan['source_sha256'] and identity['prerequisites'] == plan['prerequisites'], 'source/runtime before-after agreement')
    preflight = read(paths['runtime_preflight'])
    require(preflight['status'] == 'PASS' and identity['versions'] == preflight['versions'] and identity['installed_author_sha256'] == {k: v['sha256'] for k, v in preflight['installed_source_matches'].items()}, 'qualified author runtime join')
    admission = read(study/'admission.json')
    require(admission['source_sha256'] == plan['data_sha256'] and admission['decoded_keys'] == ['u_100mV_train', 'y_100mV_train', 'u_200mV_train', 'y_200mV_train'] and admission['fit_ids'] == list(FIT) and admission['dev_ids'] == list(DEV), 'restricted data/FIT roster')
    initial, final = (load_arrays(study/(n+'.npz'), MODEL_KEYS) for n in ('initial', 'final'))
    model(initial); model(final)
    require(all(np.array_equal(initial[k], final[k]) for k in MODEL_KEYS[4:]), 'FIT normalizer/time changed during optimization')
    freq = load_arrays(study/'frequency-target.npz', ('target', 'indices'))
    require(freq['indices'].dtype == np.int64 and np.array_equal(freq['indices'], np.arange(1, 3840)), 'frozen frequency indices')
    trace = load_arrays(study/'solver-trace.npz', ('loss_history', 'iter_times'))
    fit = read(study/'fit.json')
    n = fit['iterations']
    require(type(n) is int and 1 <= n <= 5000 and type(fit['author_stop_flag']) is bool, 'optimizer stop evidence')
    for value in trace.values():
        finite(value, (n,)); require((value >= 0).all(), 'nonnegative trace evidence')
    require(fit['status'] == ('complete' if fit['author_stop_flag'] else 'iteration_cap_reached') and (fit['author_stop_flag'] or n == 5000), 'fit completion meaning')
    require(fit['final_sha256'] == before['final.npz']['sha256'] and fit['initial_sha256'] == before['initial.npz']['sha256'], 'fit checkpoint pins')
    losses = [frequency_error(v, freq['target']) for v in (initial, final)]
    close(fit['initial_frequency_mse'], losses[0]); close(fit['final_frequency_mse'], losses[1])
    require(losses[1] <= losses[0], 'frequency fit worsened')
    close(fit['spectral_radius'], float(np.max(np.abs(np.linalg.eigvals(final['A'])))))
    require(all(type(fit[k]) in (int, float) and math.isfinite(fit[k]) and fit[k] >= 0 for k in ('outer_fit_seconds', 'author_reported_seconds')), 'fit timing')
    norm = load_arrays(paths['common_normalizer'], ('u_mean', 'u_scale', 'y_mean', 'y_scale'))
    for value in norm.values():
        finite(value, (3,))
    require((norm['u_scale'] > 0).all() and (norm['y_scale'] > 0).all(), 'common positive scales')
    old = read(paths['parent_evaluations'])
    targets, old_banks = {}, 0
    reference = ROOT/'output/fsm-linear-controls-study-v1'
    require([(e['family'], e['seed']) for e in old] == ORDERED, 'parent fixed evaluation roster')
    for evaluation in old:
        require([r['record_id'] for r in evaluation['rows']] == list(DEV), 'parent record coverage')
        stem = evaluation['family']+(f'-{evaluation["seed"]}' if evaluation['seed'] is not None else '')
        for row in evaluation['rows']:
            if row['status'] != 'complete':
                require(row['status'] in ('failed', 'not_run'), 'parent failure status'); continue
            bank = load_arrays(reference/stem/'evaluation'/(row['record_id']+'.npz'), ('prediction', 'target', 'starts'))
            require(bank['starts'].dtype == np.int64 and np.array_equal(bank['starts'], np.arange(0, 7937, 256)), 'parent request starts')
            m = metrics(bank['prediction'], bank['target'], norm['y_scale'])
            close(row, {'record_id': row['record_id'], 'status': 'complete', **m})
            previous = targets.setdefault(row['record_id'], bank['target'])
            require(np.array_equal(previous, bank['target']), 'parent targets differ')
            old_banks += 1
    evaluation = read(study/'evaluation.json')
    require([r['record_id'] for r in evaluation['rows']] == list(DEV), '12 BLA record attempts')
    differences, replays = {}, 0
    for row in evaluation['rows']:
        name = 'evaluation/'+row['record_id']+'.npz'
        if row['status'] == 'failed':
            require(set(row) == {'record_id', 'status', 'error'} and isinstance(row['error'], str) and name not in before, 'failed record preservation'); continue
        require(row['status'] == 'complete', 'BLA record status'); expected.add(name)
        bank = load_arrays(study/name, BANK_KEYS)
        require(bank['starts'].dtype == np.int64 and np.array_equal(bank['starts'], np.arange(0, 7937, 256)), 'BLA request starts')
        require(np.array_equal(bank['target'], targets[row['record_id']]), 'exact parent target join')
        m = metrics(bank['prediction'], bank['target'], norm['y_scale'])
        physical, state, singular, residual, rank = replay(final, bank['y_context'], bank['u_context'], bank['future_u'])
        normalized = (physical-norm['y_mean'])/norm['y_scale']
        pairs = {'prediction': (normalized, bank['prediction']), 'state': (state, bank['final_state']), 'singular': (singular, bank['singular_values']), 'residual': (residual, bank['context_residual_norm'])}
        diffs = {}
        for k, (actual, saved) in pairs.items():
            finite(saved, actual.shape)
            require(np.allclose(actual, saved, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), 'independent causal '+k+' replay disagreement')
            diffs[k] = float(np.max(np.abs(actual-saved)))
        close(row, {'record_id': row['record_id'], 'status': 'complete', **m, 'requests': 32, 'horizon': 128, 'rank': rank, 'rank_deficient': rank < 28})
        differences[row['record_id']] = diffs; replays += 1
    costs = evaluation['request_ms']
    require(isinstance(costs, list) and len(costs) <= 24 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in costs), 'positive finite original timing samples')
    if evaluation['timing_error'] is None:
        require(len(costs) == 24, 'complete timing'); close(evaluation['median_request_ms'], float(np.median(costs)))
    else:
        require(isinstance(evaluation['timing_error'], str) and evaluation['median_request_ms'] is None, 'failed timing remains missing')
    storage = sum(a.nbytes for a in final.values())+28*8+3*8
    require(storage == 8040 and evaluation['persistent_numeric_bytes'] == storage, 'deployed numeric storage')
    complete = fit['status'] == 'complete' and replays == 12 and evaluation['median_request_ms'] is not None
    summary = read(study/'summary.json')
    require(summary['status'] == ('REFERENCE_COMPLETE' if complete else 'REFERENCE_INCOMPLETE'), 'reference completion')
    close(summary['fit'], fit); close(summary['median_request_ms'], evaluation['median_request_ms'])
    close(summary['persistent_numeric_bytes'], storage)
    mean = float(np.mean([r['rmse'] for r in evaluation['rows']])) if replays == 12 else None
    close(summary['mean_rmse'], mean)
    events = [json.loads(line) for line in (study/'events.jsonl').read_text().splitlines()]
    phases = ['admitting_exposed_fit_dev', 'subspace_started', 'refinement_started', 'fit_closed']+['record_closed']*12+['study_closed']
    require([e['phase'] for e in events] == phases and all(type(e['time_ns']) is int for e in events) and all(a['time_ns'] <= b['time_ns'] for a, b in pairwise(events)), 'chronological fit-before-evaluation events')
    close({k: events[3][k] for k in fit}, fit)
    for event, row in zip(events[4:16], evaluation['rows'], strict=True):
        close({k: event[k] for k in row}, row)
    close({k: events[-1][k] for k in summary}, summary)
    require(set(before) == expected and inventory(study) == before, 'exact unchanged complete study inventory')
    after = authenticate(study, process)[1]
    require(after == inputs, 'input evidence changed during audit')
    result = decisions(old, evaluation['rows'], complete)
    return {'status': 'PASS', 'agreement': True, 'study': str(study), 'inputs': inputs, 'scientific_status': summary['status'], 'results': {'reference_status': summary['status'], 'rows': evaluation['rows'], 'reference_mean_rmse': mean, 'median_request_ms': evaluation['median_request_ms'], 'persistent_numeric_bytes': storage, 'continuation': result}, 'counts': {'original_fit_attempts': 1, 'record_rows': 12, 'forecast_replays': replays, 'request_replays': replays*32, 'old_forecast_banks_rescored': old_banks, 'old_model_replays': 0, 'timing_samples': len(costs), 'timing_replays': 0, 'refits': 0, 'optimizer_calls': 0}, 'replay': {'rtol': REPLAY_RTOL, 'atol': REPLAY_ATOL, 'differences': differences}, 'scope': 'Independent saved-array arithmetic and causal NumPy replay. No author package, raw measurement decode, fitting, backward calls or new timing. Exposed development only; existing candidate unchanged.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--process', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive audit output')
    result = audit(args.study, args.process)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'scientific_status': result['scientific_status'], 'counts': result['counts']}))


if __name__ == '__main__':
    main()
