"""Matched development search for linear/nonlinear, output/feedback corrections.

The backbone and normalization come from an already exposed development study.
Every recipe is frozen before the first new fit; this is not confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import fsm_data, fsm_linear
from .fsm_residual import FSMResidual
from .fsm_study import (
    batch_from_arrays,
    normalized_records,
    normalized_request,
    record_metric,
    sampling_schedule,
)

ARCHITECTURES = ('affine_output_only', 'affine_feedback', 'tanh_output_only', 'tanh_feedback')
RATES = (1e-4, 3e-4, 1e-3)
SEEDS = (9201, 9202, 9203)
RECIPES = tuple(f'{arch}-lr{rate:g}' for arch in ARCHITECTURES for rate in RATES)
RECIPE = {f'{arch}-lr{rate:g}': (arch, rate) for arch in ARCHITECTURES for rate in RATES}
DEV_IDS = tuple(f'{a}-realization-{r}-period-{p}' for a in fsm_data.AMPLITUDES
                for r in fsm_data.DEV_REALIZATIONS for p in (0, 1))
SOURCE_PATHS = ('src/openjev/research/fsm_residual_study.py', 'src/openjev/research/fsm_residual.py',
                'src/openjev/research/fsm_study.py', 'src/openjev/research/fsm_data.py',
                'src/openjev/research/fsm_linear.py', 'src/openjev/research/fsm_gru.py',
                'src/openjev/research/predictive_state_correction.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def model_for(recipe, seed, coefficients, cfg):
    arch, _ = RECIPE[recipe]
    kind, mode = arch.split('_', 1)
    return FSMResidual(coefficients, order=cfg['order'], mode=mode, seed=seed,
                       hidden_width=cfg['hidden_width'], residual_kind=kind)


def predict(model, inputs):
    tensors = tuple(torch.as_tensor(np.array(a, dtype=np.float64, copy=True)) for a in inputs)
    with torch.no_grad():
        value = model.predict(*tensors)
    return value.detach().numpy().copy()


def initial_identity(model, backbone, batch, *, atol, rtol):
    inputs = tuple(t.detach().numpy() for t in batch[:3])
    expected = fsm_linear.predict(backbone, *inputs)
    actual = predict(model, inputs)
    difference = float(np.max(np.abs(actual-expected)))
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    return {'passed': passed, 'max_abs_difference': difference, 'atol': atol, 'rtol': rtol,
            'scope': 'first scheduled FIT batch, initial H128 forecast, NumPy versus float64 PyTorch'}


def fit(recipe, seed, backbone, y, u, schedule, cfg, directory):
    directory.mkdir()
    model = model_for(recipe, seed, backbone.coefficients, cfg)
    initial_spec = model.model_spec()
    original = model.state_dict()['coefficients'].clone()
    torch.save(model.state_dict(), directory/'initial.pt')
    _, rate = RECIPE[recipe]
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=0.)
    accepted, error, identity = 0, None, None
    started = time.perf_counter()
    with (directory/'training.jsonl').open('w') as trace:
        try:
            first = batch_from_arrays(y, u, schedule['record'][0], schedule['start'][0],
                                      cfg['context'], cfg['horizon'])
            identity = initial_identity(model, backbone, first, atol=cfg['identity_atol'],
                                        rtol=cfg['identity_rtol'])
            require(identity['passed'], 'initial forecast differs from frozen backbone')
            for update in range(cfg['updates']):
                if time.perf_counter()-started > cfg['fit_timeout_seconds']:
                    raise TimeoutError('frozen per-fit cap reached')
                optimizer.zero_grad(set_to_none=True)
                yc, uc, fu, target = batch_from_arrays(
                    y, u, schedule['record'][update], schedule['start'][update],
                    cfg['context'], cfg['horizon'])
                predicted = model.predict(yc, uc, fu)
                loss = (predicted-target).square().mean()
                require(bool(torch.isfinite(loss)), 'nonfinite rollout loss')
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'],
                                                         error_if_nonfinite=True)
                optimizer.step()
                require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                        'nonfinite residual after optimizer update')
                require(torch.equal(model.state_dict()['coefficients'], original), 'frozen backbone changed')
                accepted += 1
                trace.write(json.dumps({'update': accepted, 'loss': float(loss.detach()),
                                        'gradient_norm': float(gradient),
                                        'elapsed_seconds': time.perf_counter()-started}, allow_nan=False)+'\n')
                if update % 64 == 0:
                    trace.flush()
            if time.perf_counter()-started > cfg['fit_timeout_seconds']:
                raise TimeoutError('frozen fit cap exceeded by last update')
        except Exception as exc:  # noqa: BLE001 - preserve every failed fit
            error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    seconds = time.perf_counter()-started
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'accepted_updates': accepted}, directory/'final.pt')
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    receipt = {'family': recipe, 'architecture': RECIPE[recipe][0], 'learning_rate': rate,
               'seed': seed, 'status': 'complete' if error is None else 'failed',
               'accepted_updates': accepted, 'fit_seconds': seconds, 'error': error,
               'initial_identity': identity, 'backbone_unchanged': torch.equal(original, model.coefficients),
               'model_spec': initial_spec, 'initial_sha256': sha(directory/'initial.pt'),
               'final_sha256': sha(directory/'final.pt')}
    write(directory/'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('family', 'seed', 'status', 'accepted_updates', 'fit_seconds')}),
          flush=True)
    return model, receipt


def full_request(model, record, start, cfg, norm, *, linear=False):
    end = start+cfg['context']
    inputs = ((record.y[start:end]-norm.y_mean)/norm.y_scale,
              (record.u[start+1:end]-norm.u_mean)/norm.u_scale,
              (record.u[end:end+cfg['horizon']]-norm.u_mean)/norm.u_scale)
    inputs = tuple(a[None] for a in inputs)
    value = fsm_linear.predict(model, *inputs) if linear else predict(model, inputs)
    return value*norm.y_scale+norm.y_mean


def storage(model, norm, *, linear=False):
    normalizer_bytes = sum(getattr(norm, k).nbytes for k in ('u_mean', 'u_scale', 'y_mean', 'y_scale'))
    if linear:
        tensors, state, metadata = model.coefficients.nbytes, model.state_scalars*8, 24
    else:
        tensors = sum(t.numel()*t.element_size() for t in model.state_dict().values())
        state, metadata = 6*model.order*8, model.model_spec()['numeric_metadata_bytes']
    return {'tensor_bytes': tensors, 'state_bytes': state, 'normalizer_bytes': normalizer_bytes,
            'scalar_metadata_bytes': metadata, 'persistent_numeric_bytes': tensors+state+normalizer_bytes+metadata}


def evaluate(model, receipt, records, cfg, norm, directory, *, linear=False):
    directory.mkdir()
    rows, costs, timing_error = [], [], None
    starts = np.arange(0, len(records[0].u)-cfg['context']-cfg['horizon']+1, cfg['evaluation_stride'])
    before = time.perf_counter()
    for record in records:
        try:
            inputs, target = normalized_request(record, starts, cfg['context'], cfg['horizon'], norm)
            value = fsm_linear.predict(model, *inputs) if linear else predict(model, inputs)
            metrics = record_metric(value, target)
            with np.errstate(over='ignore'):
                native_rmse = np.asarray(metrics['per_channel_rmse'])*norm.y_scale
            require(np.isfinite(native_rmse).all(), 'nonfinite native-output error')
            metrics['native_output_per_channel_rmse'] = native_rmse.tolist()
            np.savez_compressed(directory/(record.record_id+'.npz'), prediction=value, target=target, starts=starts)
            rows.append({'record_id': record.record_id, 'status': 'complete', **metrics})
        except Exception as exc:  # noqa: BLE001 - preserve record-level failures
            rows.append({'record_id': record.record_id, 'status': 'failed',
                         'error': f'{type(exc).__name__}: {exc}'})
    scoring_seconds = time.perf_counter()-before
    try:
        full_request(model, records[0], int(starts[0]), cfg, norm, linear=linear)
        for record in records:
            for start in (int(starts[0]), int(starts[-1])):
                begin = time.perf_counter()
                value = full_request(model, record, start, cfg, norm, linear=linear)
                costs.append((time.perf_counter()-begin)*1000)
                require(np.isfinite(value).all(), 'nonfinite timed forecast')
    except Exception as exc:  # noqa: BLE001 - retain timing errors
        timing_error = f'{type(exc).__name__}: {exc}'
    result = {'family': receipt['family'], 'seed': receipt.get('seed'), 'rows': rows,
              'request_ms': costs, 'median_request_ms': float(np.median(costs)) if costs and timing_error is None else None,
              'timing_error': timing_error, 'scoring_seconds': scoring_seconds,
              'model_spec': model.model_spec(), **storage(model, norm, linear=linear)}
    write(directory/'receipt.json', result)
    return result


def skipped_evaluation(receipt, directory):
    directory.mkdir()
    result = {'family': receipt['family'], 'seed': receipt['seed'],
              'rows': [{'record_id': rid, 'status': 'not_run', 'error': 'fit_failed'} for rid in DEV_IDS],
              'request_ms': [], 'median_request_ms': None, 'timing_error': 'fit_failed',
              'model_spec': receipt['model_spec'], 'persistent_numeric_bytes': None}
    write(directory/'receipt.json', result)
    return result


def summarize(evaluations, fits, cfg):
    families = {}
    for family in (*RECIPES, 'native_varx'):
        members = [e for e in evaluations if e['family'] == family]
        required_seeds = {None} if family == 'native_varx' else set(SEEDS)
        roster_ok = len(members) == len(required_seeds) and {e['seed'] for e in members} == required_seeds
        matching_fits = [f for f in fits if f['family'] == family]
        fit_ok = family == 'native_varx' or (len(matching_fits) == 3 and {f['seed'] for f in matching_fits} == set(SEEDS)
                 and all(f['status'] == 'complete' and f['accepted_updates'] == cfg['updates'] for f in matching_fits))
        score_ok = fit_ok and roster_ok and all(
            len(e['rows']) == 12 and {r['record_id'] for r in e['rows']} == set(DEV_IDS)
            and all(r['status'] == 'complete' and np.isfinite(r['rmse']) and r['rmse'] >= 0 for r in e['rows'])
            for e in members)
        latency_ok = roster_ok and all(
            e['timing_error'] is None and len(e['request_ms']) == 24
            and all(np.isfinite(t) and t > 0 for t in e['request_ms'])
            and e['median_request_ms'] is not None and np.isfinite(e['median_request_ms'])
            and e['median_request_ms'] > 0 for e in members)
        storage_ok = roster_ok and all(e['persistent_numeric_bytes'] is not None
                                       and e['persistent_numeric_bytes'] > 0 for e in members)
        families[family] = {'eligible': score_ok and latency_ok and storage_ok,
                            'score_eligible': score_ok, 'latency_eligible': latency_ok,
                            'storage_eligible': storage_ok}
        if score_ok:
            families[family]['mean_rmse'] = float(np.mean([r['rmse'] for e in members for r in e['rows']]))
        if latency_ok:
            families[family]['mean_seed_median_ms'] = float(np.mean([e['median_request_ms'] for e in members]))
        if storage_ok:
            families[family]['persistent_numeric_bytes'] = max(e['persistent_numeric_bytes'] for e in members)
    selected = {}
    for arch in ARCHITECTURES:
        candidates = [f for f in RECIPES if RECIPE[f][0] == arch and families[f]['score_eligible']]
        selected[arch] = min(candidates, key=lambda f: (families[f]['mean_rmse'], RECIPE[f][1])) if candidates else None
    affine = [selected[a] for a in ('affine_output_only', 'affine_feedback') if selected[a]]
    best_affine = min(affine, key=lambda f: (families[f]['mean_rmse'], f)) if affine else None
    candidate, output = selected['tanh_feedback'], selected['tanh_output_only']
    controls = [f for f in (output, best_affine, 'native_varx') if f and families[f]['score_eligible']]
    strongest = min(controls, key=lambda f: (families[f]['mean_rmse'], f)) if controls else None
    roster = {(f, s) for f in RECIPES for s in SEEDS}
    conditions = {
        'all_36_fits_complete': len(fits) == len(roster) and {(f['family'], f['seed']) for f in fits} == roster
            and all(f['status'] == 'complete' and f['accepted_updates'] == cfg['updates'] for f in fits),
        'all_37_evaluations_complete': len(evaluations) == len(roster)+1
            and {(e['family'], e['seed']) for e in evaluations} == roster | {('native_varx', None)}
            and all(f['eligible'] for f in families.values()),
        'initial_identity_and_frozen_backbone': len(fits) == len(roster)
            and all(f.get('initial_identity') and f['initial_identity']['passed'] and f['backbone_unchanged'] for f in fits),
        'five_percent_below_strongest_control': False,
        'every_seed_below_three_selected_controls': False,
        'no_record_over_two_percent_strongest_control': False,
        'both_amplitudes_below_strongest_control': False,
        'latency_within_ten_percent_tanh_output': False,
        'storage_no_more_than_tanh_output': False,
    }
    if candidate and output and best_affine and len(controls) == 3:
        def mean(f):
            return families[f]['mean_rmse']
        conditions['five_percent_below_strongest_control'] = mean(candidate) <= .95*mean(strongest)
        def seed_mean(f, seed):
            matches = [e for e in evaluations if e['family'] == f and e['seed'] == (None if f == 'native_varx' else seed)]
            return float(np.mean([r['rmse'] for r in matches[0]['rows']]))
        conditions['every_seed_below_three_selected_controls'] = all(
            seed_mean(candidate, s) < seed_mean(control, s) for s in SEEDS for control in controls)
        def record_means(f):
            rows = [r for e in evaluations if e['family'] == f for r in e['rows']]
            return {rid: float(np.mean([r['rmse'] for r in rows if r['record_id'] == rid])) for rid in DEV_IDS}
        c, b = record_means(candidate), record_means(strongest)
        conditions['no_record_over_two_percent_strongest_control'] = all(c[rid] <= 1.02*b[rid] for rid in DEV_IDS)
        conditions['both_amplitudes_below_strongest_control'] = all(
            np.mean([v for k, v in c.items() if k.startswith(a)]) < np.mean([v for k, v in b.items() if k.startswith(a)])
            for a in fsm_data.AMPLITUDES)
        conditions['latency_within_ten_percent_tanh_output'] = (
            families[candidate]['latency_eligible'] and families[output]['latency_eligible'] and
            families[candidate]['mean_seed_median_ms'] <= 1.1*families[output]['mean_seed_median_ms'])
        conditions['storage_no_more_than_tanh_output'] = (
            families[candidate]['storage_eligible'] and families[output]['storage_eligible'] and
            families[candidate]['persistent_numeric_bytes'] <= families[output]['persistent_numeric_bytes'])
    return {'status': 'DEVELOPMENT_PASS' if all(conditions.values()) else 'DEVELOPMENT_FAIL',
            'families': families, 'selected_by_architecture': selected, 'candidate': candidate,
            'strongest_affine': best_affine, 'strongest_control': strongest,
            'conditions': conditions, 'passed': sum(bool(v) for v in conditions.values()), 'total': len(conditions),
            'selection': 'One rate per architecture selected on pooled exposed DEV accuracy; lower rate breaks ties. All recipes retained; no confirmation or convergence claim.'}


def validate_pins(protocol):
    require(set(protocol['source_sha256']) == set(SOURCE_PATHS), 'exact frozen source roster required')
    for name, expected in protocol['source_sha256'].items():
        require(sha(name) == expected, 'frozen source changed: '+name)
    for item in protocol['parent_artifacts'].values():
        require(sha(item['path']) == item['sha256'], 'parent artifact changed: '+item['path'])
    require(sha(protocol['protocol_markdown_path']) == protocol['protocol_markdown_sha256'],
            'frozen protocol text changed')


def load_parent(protocol):
    paths = protocol['parent_artifacts']
    with np.load(paths['coefficients']['path'], allow_pickle=False) as src:
        require(src.files == ['coefficients'], 'coefficient member mismatch')
        coefficients = src['coefficients'].copy()
    receipt = json.loads(Path(paths['fit_receipt']['path']).read_text())
    require(receipt['status'] == 'complete' and receipt['order'] == 32 and receipt['alpha'] == 1e-6,
            'parent recipe mismatch')
    backbone = fsm_linear.VARXModel(coefficients, 32, 1e-6, receipt['model_spec']['fit_rows'],
                                  tuple(receipt['model_spec']['fit_record_ids']))
    with np.load(paths['normalizer']['path'], allow_pickle=False) as src:
        names = ('u_mean', 'u_scale', 'y_mean', 'y_scale')
        require(set(src.files) == set(names), 'normalizer member mismatch')
        values = [src[name].copy() for name in names]
    require(all(v.dtype == np.float64 and v.shape == (3,) and np.isfinite(v).all() for v in values)
            and (values[1] > 0).all() and (values[3] > 0).all(), 'invalid frozen normalization')
    norm = fsm_data.Normalizer(*values, fsm_linear.FIT_IDS, 98304)
    return backbone, norm


def run(protocol_path, data_path, output):
    protocol = json.loads(Path(protocol_path).read_text()); cfg = protocol['experiment']
    require(tuple(cfg['architectures']) == ARCHITECTURES and tuple(cfg['learning_rates']) == RATES
            and tuple(cfg['seeds']) == SEEDS, 'fixed architecture/rate/seed roster required')
    require((cfg['order'], cfg['hidden_width'], cfg['context'], cfg['horizon'], cfg['batch'], cfg['updates'],
             cfg['evaluation_stride'], cfg['gradient_clip']) == (32, 24, 100, 128, 16, 2048, 256, 1.),
            'fixed production geometry and budget required')
    validate_pins(protocol)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write(output/'protocol.json', protocol)
    snapshot = output/'source'; snapshot.mkdir()
    for name in SOURCE_PATHS:
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(Path(name).read_bytes())
    parents = output/'parent'; parents.mkdir()
    for name, item in protocol['parent_artifacts'].items():
        (parents/(name+Path(item['path']).suffix)).write_bytes(Path(item['path']).read_bytes())
    torch.set_num_threads(1)
    backbone, norm = load_parent(protocol)
    require(sha(data_path) == protocol['data_sha256'], 'measurement archive changed before decoding')
    data = fsm_data.read_npz_estimation(data_path)
    require(data.source_sha256 == protocol['data_sha256'], 'measurement archive changed')
    write(output/'admission.json', {'source_sha256': data.source_sha256, 'decoded_keys': list(data.decoded_keys),
                                   'fit_ids': list(norm.fit_record_ids), 'fit_samples': norm.fit_samples,
                                   'dev_ids': [r.record_id for r in data.partition('dev')],
                                   'normalizer': 'unchanged parent FIT normalization; no refit',
                                   'development_exposure': 'same DEV used in previous pilot; new LR selection and score use these same exposed records'})
    train = normalized_records(data.partition('fit'), norm)
    y, u = (torch.as_tensor(np.stack([getattr(r, key) for r in train]), dtype=torch.float64) for key in ('y', 'u'))
    fits, models, evaluations = [], {}, []
    started = time.perf_counter()
    for si, seed in enumerate(SEEDS):
        schedule = sampling_schedule(seed, updates=cfg['updates'], batch=cfg['batch'], records=12,
                                     samples=8192, context=cfg['context'], horizon=cfg['horizon'])
        np.savez(output/f'schedule-{seed}.npz', **schedule)
        for recipe in RECIPES[si:]+RECIPES[:si]:
            model, receipt = fit(recipe, seed, backbone, y, u, schedule, cfg, output/f'{recipe}-{seed}')
            fits.append(receipt)
            if receipt['status'] == 'complete':
                models[(recipe, seed)] = model
            write(output/'fits.json', fits)
    validate_pins(protocol)
    write(output/'fits-closed.json', {'fits': len(fits), 'complete': sum(f['status'] == 'complete' for f in fits),
                                    'elapsed_seconds': time.perf_counter()-started,
                                    'barrier': 'all declared fits closed before DEV predictions, scoring or rate selection'})
    for receipt in fits:
        key = receipt['family'], receipt['seed']; directory = output/f'{key[0]}-{key[1]}'/'evaluation'
        e = (evaluate(models[key], receipt, data.partition('dev'), cfg, norm, directory)
             if key in models else skipped_evaluation(receipt, directory))
        evaluations.append(e); write(output/'evaluations.json', evaluations)
    reference_dir = output/'native_varx'; reference_dir.mkdir()
    evaluations.append(evaluate(backbone, {'family': 'native_varx'}, data.partition('dev'), cfg, norm,
                                reference_dir/'evaluation', linear=True))
    write(output/'evaluations.json', evaluations)
    summary = summarize(evaluations, fits, cfg)
    summary['elapsed_seconds'] = time.perf_counter()-started
    write(output/'summary.json', summary)
    validate_pins(protocol)
    write(output/'closure.json', {'status': 'completed', 'result': summary['status'],
                                 'source_sha256': protocol['source_sha256'], 'summary_sha256': sha(output/'summary.json')})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', required=True)
    parser.add_argument('--data', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); existed = Path(args.output).exists()
    try:
        run(args.protocol, args.data, args.output)
    except Exception as exc:
        if not existed and Path(args.output).is_dir():
            write(Path(args.output)/'closure.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                                                   'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
