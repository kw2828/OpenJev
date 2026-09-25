"""Frozen, development-only FSM recurrent-correction comparison.

No official-test or 300mV member is decoded. Every declared fit and failure is
retained. This pilot can reject a mechanism; it cannot establish benchmark SOTA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import fsm_data, fsm_linear
from .fsm_gru import FSMGRU
from .predictive_state_correction import HORIZONS, PredictiveStateCorrection

FAMILIES = ('dense', 'selective', 'rewired', 'fixed0', 'fixed1', 'fixed2', 'autoregressive')
SEEDS = (9101, 9102, 9103)
SOURCE_PATHS = ('src/openjev/research/fsm_study.py', 'src/openjev/research/fsm_data.py',
                'src/openjev/research/fsm_linear.py', 'src/openjev/research/fsm_gru.py',
                'src/openjev/research/predictive_state_correction.py')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_for(family, seed, config):
    args = {'latent_dim': config['latent'], 'aux_width': config['aux_width'], 'seed': seed}
    if family == 'autoregressive':
        return FSMGRU(3, 3, **args)
    if family not in FAMILIES:
        raise ValueError('undeclared model family')
    return PredictiveStateCorrection(3, 3, mode=family, **args)


def sampling_schedule(seed, *, updates, batch, records, samples, context, horizon):
    if min(updates, batch, records, context, horizon) <= 0 or samples < context+horizon:
        raise ValueError('invalid sampling dimensions')
    rng = np.random.default_rng(100000+seed)
    return {'record': rng.integers(0, records, size=(updates, batch), dtype=np.int64),
            'start': rng.integers(0, samples-context-horizon+1,
                                  size=(updates, batch), dtype=np.int64)}


def batch_from_arrays(y, u, records, starts, context, horizon):
    # u/y contain FIT only. Each index stays inside its own period.
    ri = torch.as_tensor(records, dtype=torch.long)[:, None]
    s = torch.as_tensor(starts, dtype=torch.long)[:, None]
    yc = y[ri, s+torch.arange(context)]
    uc = u[ri, s+torch.arange(1, context)]
    fu = u[ri, s+context+torch.arange(horizon)]
    target = y[ri, s+context+torch.arange(horizon)]
    return yc, uc, fu, target


def training_loss(model, yc, uc, fu, target, auxiliary_weight):
    state = model.condition(yc, uc)
    predicted, _ = model.rollout(fu, state)
    main = (predicted-target).square().mean()
    aux = torch.stack([(model.aux_decode(state, fu, h)-target[:, h-1]).square().mean()
                       for h in HORIZONS]).mean()
    return main + auxiliary_weight*aux, main, aux


def normalized_records(records, normalizer):
    return tuple(replace(r, u=(r.u-normalizer.u_mean)/normalizer.u_scale,
                         y=(r.y-normalizer.y_mean)/normalizer.y_scale) for r in records)


def record_metric(predicted, target):
    predicted, target = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError('matched [requests,horizon,3] arrays required')
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError('nonfinite predictions or targets')
    residual = predicted-target
    with np.errstate(over='ignore'):
        if not np.isfinite(residual**2).all():
            raise FloatingPointError('squared residual overflow')
    with np.errstate(over='ignore'):
        mse = float(np.mean(residual**2))
        channels = np.sqrt(np.mean(residual**2, axis=(0, 1)))
    if not np.isfinite(mse) or not np.isfinite(channels).all():
        raise FloatingPointError('nonfinite aggregate metric')
    return {'rmse': float(np.sqrt(mse)), 'per_channel_rmse': channels.tolist(),
            'mse': mse, 'requests': len(target), 'horizon': target.shape[1]}


def normalized_request(record, starts, context, horizon, norm):
    windows = [fsm_data.make_window(record, int(s), context=context, horizon=horizon,
                                   normalizer=norm) for s in starts]
    inputs = tuple(np.stack([getattr(w, name) for w in windows]) for name in
                   ('y_context', 'transition_context_u', 'future_u'))
    targets = np.stack([w.target for w in windows])
    return inputs, targets


def neural_prediction(model, inputs):
    yc, uc, fu = (torch.as_tensor(np.array(a, dtype=np.float32, copy=True)) for a in inputs)
    with torch.no_grad():
        state = model.condition(yc, uc)
        pred, _ = model.rollout(fu, state)
    return pred.numpy().astype(np.float64)


def routing_diagnostic(model, inputs):
    """Extra diagnostic replay, excluded from online timing and recorded separately."""
    yc, uc, _ = (torch.as_tensor(np.array(a, dtype=np.float32, copy=True)) for a in inputs)
    counts = np.zeros(3, dtype=np.int64)
    with torch.no_grad():
        state = torch.tanh(model.initializer(yc[:, 0]))
        prepared = model._prepare_correction()
        for t in range(1, yc.shape[1]):
            prior = model.transition(uc[:, t-1], state)
            error = yc[:, t] - model.observation(prior)
            gradient = error @ model.observation.weight
            energy = gradient.reshape(len(yc), 3, model.block_dim).square().sum(-1)
            nonzero = energy.sum(-1) > 0
            chosen = energy.argmax(-1)[nonzero]
            counts += torch.bincount(chosen, minlength=3).numpy()
            state = model._correct(prior, yc[:, t], prepared)
        expected = model.condition(yc, uc)
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
    return counts


def fit_neural(family, seed, y, u, schedule, cfg, directory):
    directory.mkdir()
    model = model_for(family, seed, cfg)
    initial_spec = model.model_spec()
    torch.save(model.state_dict(), directory/'initial.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    start = time.perf_counter()
    accepted, error = 0, None
    with (directory/'training.jsonl').open('w') as trace:
        try:
            for update in range(cfg['updates']):
                if time.perf_counter()-start > cfg['fit_timeout_seconds']:
                    raise TimeoutError('fixed per-fit wall-clock cap reached')
                optimizer.zero_grad(set_to_none=True)
                batch = batch_from_arrays(y, u, schedule['record'][update], schedule['start'][update],
                                          cfg['context'], cfg['horizon'])
                loss, main, aux = training_loss(model, *batch, cfg['auxiliary_weight'])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'],
                                                     error_if_nonfinite=True)
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError('nonfinite parameters after optimizer update')
                accepted += 1
                trace.write(json.dumps({'update': update+1, 'loss': float(loss.detach()),
                                        'forecast_loss': float(main.detach()), 'auxiliary_loss': float(aux.detach()),
                                        'gradient_norm': float(norm),
                                        'elapsed_seconds': time.perf_counter()-start}, allow_nan=False)+'\n')
                if update % 64 == 0:
                    trace.flush()
            if time.perf_counter()-start > cfg['fit_timeout_seconds']:
                raise TimeoutError('fixed per-fit wall-clock cap exceeded by final update')
        except Exception as exc:  # noqa: BLE001 - retain failed experiment attempts
            error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    fit_seconds = time.perf_counter()-start
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'accepted_updates': accepted}, directory/'final.pt')
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    receipt = {'family': family, 'seed': seed, 'status': 'complete' if error is None else 'failed',
               'accepted_updates': accepted, 'fit_seconds': fit_seconds, 'error': error,
               'model_spec': initial_spec, 'initial_sha256': digest(directory/'initial.pt'),
               'final_sha256': digest(directory/'final.pt')}
    write_json(directory/'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('family', 'seed', 'status', 'accepted_updates', 'fit_seconds')}), flush=True)
    return model, receipt


def full_request(model, record, start, cfg, norm, *, linear=False):
    # Target creation is outside this boundary: inputs are normalized directly.
    end = start+cfg['context']
    yc = (record.y[start:end]-norm.y_mean)/norm.y_scale
    uc = (record.u[start+1:end]-norm.u_mean)/norm.u_scale
    fu = (record.u[end:end+cfg['horizon']]-norm.u_mean)/norm.u_scale
    inputs = (yc[None], uc[None], fu[None])
    pred = fsm_linear.predict(model, *inputs) if linear else neural_prediction(model, inputs)
    return pred*norm.y_scale+norm.y_mean


def evaluate(model, fit_receipt, records, cfg, norm, directory, *, linear=False):
    directory.mkdir()
    rows, costs, counts = [], [], np.zeros(3, dtype=np.int64)
    starts = np.arange(0, len(records[0].u)-cfg['context']-cfg['horizon']+1, cfg['evaluation_stride'])
    started = time.perf_counter()
    for record in records:
        inputs, target = normalized_request(record, starts, cfg['context'], cfg['horizon'], norm)
        try:
            pred = fsm_linear.predict(model, *inputs) if linear else neural_prediction(model, inputs)
            metric = record_metric(pred, target)
            np.savez_compressed(directory/(record.record_id+'.npz'), prediction=pred, target=target, starts=starts)
            if fit_receipt['family'] == 'selective':
                counts += routing_diagnostic(model, inputs)
            rows.append({'record_id': record.record_id, 'status': 'complete', **metric})
        except Exception as exc:  # noqa: BLE001 - retain failed experiment attempts
            rows.append({'record_id': record.record_id, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
    evaluation_seconds = time.perf_counter()-started
    # Warm up on the first predeclared request, then two fixed requests per record.
    try:
        full_request(model, records[0], int(starts[0]), cfg, norm, linear=linear)
        for record in records:
            for start in (int(starts[0]), int(starts[-1])):
                before = time.perf_counter()
                value = full_request(model, record, start, cfg, norm, linear=linear)
                costs.append((time.perf_counter()-before)*1000)
                if not np.isfinite(value).all():
                    raise FloatingPointError('nonfinite timed output')
        timing_error = None
    except Exception as exc:  # noqa: BLE001 - retain failed experiment attempts
        timing_error = f'{type(exc).__name__}: {exc}'
    if linear:
        spec = model.model_spec()
        numeric_bytes = spec['retained_numeric_bytes'] + spec['state_scalars']*8 + 12*8
    else:
        spec = model.model_spec()
        numeric_bytes = sum(p.numel()*p.element_size() for p in model.parameters()) + spec['state_scalars']*4 + 12*8
    receipt = {'family': fit_receipt['family'], 'seed': fit_receipt.get('seed'), 'rows': rows,
               'evaluation_seconds_including_diagnostics': evaluation_seconds,
               'request_ms': costs, 'timing_error': timing_error,
               'median_request_ms': float(np.median(costs)) if costs and timing_error is None else None,
               'persistent_numeric_bytes': numeric_bytes,
               'storage_scope': 'all retained parameters, normalization and one stream state; excludes request, workspace and Python object overhead',
               'routing_nonzero_counts': counts.tolist() if fit_receipt['family'] == 'selective' else None,
               'model_spec': spec}
    write_json(directory/'receipt.json', receipt)
    return receipt


def summarize(evaluations, fits, cfg):
    families = {}
    declared_families = list(FAMILIES) + [f'varx{order}-ridge{alpha:g}' for order in cfg['linear_orders'] for alpha in cfg['linear_alphas']]
    for family in declared_families:
        members = [e for e in evaluations if e['family'] == family]
        needed = 3 if family in FAMILIES else 1
        expected_ids = {f'{a}-realization-{r}-period-{p}' for a in fsm_data.AMPLITUDES for r in fsm_data.DEV_REALIZATIONS for p in range(2)}
        good = (len(members) == needed and {e.get('seed') for e in members} == (set(SEEDS) if family in FAMILIES else {None}) and all(len(e['rows']) == 12 and {r['record_id'] for r in e['rows']} == expected_ids and
                all(r['status'] == 'complete' for r in e['rows']) and
                e['timing_error'] is None for e in members))
        families[family] = {'eligible': good}
        if good:
            families[family].update(mean_rmse=float(np.mean([r['rmse'] for e in members for r in e['rows']])),
                                   mean_seed_median_ms=float(np.mean([e['median_request_ms'] for e in members])),
                                   persistent_numeric_bytes=max(e['persistent_numeric_bytes'] for e in members))
    expected = {(family, seed) for family in FAMILIES for seed in SEEDS}
    expected |= {(f'varx{order}-ridge{alpha:g}', None) for order in cfg['linear_orders'] for alpha in cfg['linear_alphas']}
    fit_keys = [(f['family'], f.get('seed')) for f in fits]
    evaluation_keys = [(e['family'], e.get('seed')) for e in evaluations]
    controls = {k: v for k, v in families.items() if k != 'selective' and v['eligible']}
    strongest = min(controls, key=lambda k: controls[k]['mean_rmse']) if controls else None
    candidate = families.get('selective', {'eligible': False})
    dense = families.get('dense', {'eligible': False})
    conditions = {'all_declared_fits_complete': len(fits) == len(expected) and set(fit_keys) == expected and all(f['status'] == 'complete' and (f['family'] not in FAMILIES or f['accepted_updates'] == cfg['updates']) for f in fits),
                  'all_declared_evaluations_complete': len(evaluations) == len(expected) and set(evaluation_keys) == expected and all(v['eligible'] for v in families.values()),
                  'candidate_eligible': candidate['eligible'],
                  'five_percent_below_strongest_control': False,
                  'latency_within_ten_percent_dense': False,
                  'storage_within_ten_percent_dense': False,
                  'no_record_over_two_percent_dense': False,
                  'every_seed_beats_dense': False,
                  'two_blocks_at_least_five_percent_each_seed': False}
    if candidate['eligible'] and dense['eligible'] and strongest:
        conditions['five_percent_below_strongest_control'] = candidate['mean_rmse'] <= .95*controls[strongest]['mean_rmse']
        conditions['latency_within_ten_percent_dense'] = candidate['mean_seed_median_ms'] <= 1.1*dense['mean_seed_median_ms']
        conditions['storage_within_ten_percent_dense'] = candidate['persistent_numeric_bytes'] <= 1.1*dense['persistent_numeric_bytes']
        selected = [e for e in evaluations if e['family'] == 'selective']
        denselist = [e for e in evaluations if e['family'] == 'dense']
        ids = [r['record_id'] for r in selected[0]['rows']]
        candidate_records = {rid: np.mean([next(r['rmse'] for r in e['rows'] if r['record_id'] == rid) for e in selected]) for rid in ids}
        dense_records = {rid: np.mean([next(r['rmse'] for r in e['rows'] if r['record_id'] == rid) for e in denselist]) for rid in ids}
        conditions['no_record_over_two_percent_dense'] = all(candidate_records[rid] <= 1.02*dense_records[rid] for rid in ids)
        conditions['every_seed_beats_dense'] = all(np.mean([r['rmse'] for r in e['rows']]) < np.mean([r['rmse'] for r in next(d for d in denselist if d['seed']==e['seed'])['rows']]) for e in selected)
        conditions['two_blocks_at_least_five_percent_each_seed'] = all(sum(v >= .05*sum(e['routing_nonzero_counts']) for v in e['routing_nonzero_counts']) >= 2 and sum(e['routing_nonzero_counts'])>0 for e in selected)
    return {'status': 'DEVELOPMENT_PASS' if all(conditions.values()) else 'DEVELOPMENT_FAIL',
            'families': families, 'strongest_control': strongest, 'conditions': conditions,
            'passed': sum(conditions.values()), 'total': len(conditions),
            'interpretation': 'Custom reduced-training development pilot; not official benchmark score, confirmation, architectural novelty or closed-loop control evidence.'}


def run(protocol_path, data_path, output):
    protocol = json.loads(Path(protocol_path).read_text())
    cfg = protocol['experiment']
    if tuple(cfg['families']) != FAMILIES or tuple(cfg['seeds']) != SEEDS:
        raise ValueError('declared roster required')
    if set(protocol['source_sha256']) != set(SOURCE_PATHS):
        raise ValueError('complete frozen source roster required')
    for name, expected in protocol['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError('frozen source changed: '+name)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', protocol)
    source_dir = output/'source'; source_dir.mkdir()
    for name in SOURCE_PATHS:
        target = source_dir/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    torch.set_num_threads(1)
    if digest(data_path) != protocol['data_sha256']:
        raise ValueError('data archive changed before decoding')
    data = fsm_data.read_npz_estimation(data_path)
    if data.source_sha256 != protocol['data_sha256']:
        raise ValueError('data archive changed')
    norm = fsm_data.fit_normalizer(data.partition('fit'))
    np.savez(output/'normalizer.npz', u_mean=norm.u_mean, u_scale=norm.u_scale,
             y_mean=norm.y_mean, y_scale=norm.y_scale)
    write_json(output/'admission.json', {'decoded_keys': list(data.decoded_keys), 'source_sha256': data.source_sha256,
                                       'fit_ids': list(norm.fit_record_ids), 'fit_samples': norm.fit_samples,
                                       'dev_ids': [r.record_id for r in data.partition('dev')]})
    train = normalized_records(data.partition('fit'), norm)
    y = torch.as_tensor(np.stack([r.y for r in train]).astype(np.float32))
    u = torch.as_tensor(np.stack([r.u for r in train]).astype(np.float32))
    fits, models, evaluations = [], [], []
    started = time.perf_counter()
    for si, seed in enumerate(SEEDS):
        schedule = sampling_schedule(seed, updates=cfg['updates'], batch=cfg['batch'], records=len(train),
                                     samples=len(train[0].u), context=cfg['context'], horizon=cfg['horizon'])
        np.savez(output/f'schedule-{seed}.npz', **schedule)
        roster = FAMILIES[si:]+FAMILIES[:si]
        for family in roster:
            model, receipt = fit_neural(family, seed, y, u, schedule, cfg, output/f'{family}-{seed}')
            fits.append(receipt)
            if receipt['status'] == 'complete':
                models.append((model, receipt, False))
    # All neural fits finish before DEV prediction or baseline selection.
    for order in cfg['linear_orders']:
        for alpha in cfg['linear_alphas']:
            family = f'varx{order}-ridge{alpha:g}'
            directory = output/family; directory.mkdir()
            before = time.perf_counter()
            try:
                model = fsm_linear.fit_varx(train, order=order, alpha=alpha)
                np.savez(directory/'model.npz', coefficients=model.coefficients)
                receipt = {'family': family, 'status': 'complete', 'order': order, 'alpha': alpha,
                           'fit_seconds': time.perf_counter()-before, 'model_spec': model.model_spec()}
                models.append((model, receipt, True))
            except Exception as exc:  # noqa: BLE001 - retain failed experiment attempts
                receipt = {'family': family, 'status': 'failed', 'order': order, 'alpha': alpha,
                           'fit_seconds': time.perf_counter()-before, 'error': f'{type(exc).__name__}: {exc}'}
            fits.append(receipt); write_json(directory/'receipt.json', receipt)
            print(json.dumps({'family': family, 'status': receipt['status'], 'fit_seconds': receipt['fit_seconds']}), flush=True)
    for model, receipt, linear in models:
        name = receipt['family'] + (f"-{receipt['seed']}" if 'seed' in receipt else '')
        evaluations.append(evaluate(model, receipt, data.partition('dev'), cfg, norm,
                                    output/name/'evaluation', linear=linear))
    summary = summarize(evaluations, fits, cfg)
    summary['elapsed_seconds'] = time.perf_counter()-started
    write_json(output/'fits.json', fits); write_json(output/'evaluations.json', evaluations)
    write_json(output/'summary.json', summary)
    if any(digest(name) != pin for name, pin in protocol['source_sha256'].items()):
        raise ValueError('frozen source changed during run')
    write_json(output/'closure.json', {'status': 'completed', 'result': summary['status'],
                                      'source_sha256': protocol['source_sha256'],
                                      'summary_sha256': digest(output/'summary.json')})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    preexisted = Path(args.output).exists()
    try:
        run(args.protocol, args.data, args.output)
    except Exception as exc:
        if not preexisted and Path(args.output).is_dir():
            write_json(Path(args.output)/'closure.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
