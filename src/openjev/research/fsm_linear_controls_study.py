"""Frozen longer-memory linear controls and native affine-feedback deployment.

All neural checkpoints are unchanged. Only already exposed FIT/DEV members
are admitted. Shared sufficient statistics preserve the existing ridge solve.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from . import fsm_data, fsm_linear
from .fsm_affine_fold import fold_affine_feedback
from .fsm_residual import FSMResidual
from .fsm_residual_study import (
    DEV_IDS,
    evaluate,
    load_parent,
    normalized_records,
    normalized_request,
    require,
    sha,
    write,
)

ORDERS, ALPHAS, SEEDS = (32, 64, 96), (1e-6, 1e-3, .1), (9201, 9202, 9203)
SELECTED = {'affine_output_only': 'affine_output_only-lr0.0001',
            'affine_feedback': 'affine_feedback-lr0.0001',
            'tanh_output_only': 'tanh_output_only-lr0.001',
            'tanh_feedback': 'tanh_feedback-lr0.0003'}
LINEAR = {f'varx{p}-ridge{a:g}': (p, a) for p in ORDERS for a in ALPHAS}
FOLDED, NATIVE = 'folded_affine_feedback', 'native_varx'
FAMILIES = (*LINEAR, *SELECTED.values(), FOLDED, NATIVE)
ROSTER = {(f, None) for f in (*LINEAR, NATIVE)} | {
    (f, s) for f in (*SELECTED.values(), FOLDED) for s in SEEDS}
SOURCES = tuple(f'src/openjev/research/{name}.py' for name in (
    'fsm_linear_controls_study', 'fsm_affine_fold', 'fsm_residual_study',
    'fsm_residual', 'fsm_linear', 'fsm_data', 'fsm_study', 'fsm_gru',
    'predictive_state_correction'))


def sufficient_statistics(records, order):
    """Canonical within-period unregularized mean Gram/cross, FIT only."""
    fsm_linear._order(order)
    records = fsm_linear._fit_records(records, order)
    width = 6*order+4
    gram, cross = np.zeros((width, width)), np.zeros((width, 3))
    rows = 0
    with np.errstate(over='ignore', invalid='ignore'):
        for r in records:
            n = len(r.u)-order
            x = np.concatenate([r.y[j:j+n] for j in range(order)] + [r.u[order:]]
                               + [r.u[j:j+n] for j in range(order)] + [np.ones((n, 1))], axis=1)
            gram += x.T @ x
            cross += x.T @ r.y[order:]
            rows += n
        gram /= rows
        cross /= rows
    require(np.isfinite(gram).all() and np.isfinite(cross).all(), 'nonfinite sufficient statistics')
    return gram, cross, rows


def solve_statistics(gram, cross, rows, *, order, alpha):
    fsm_linear._order(order); fsm_linear._alpha(alpha)
    width = 6*order+4
    fsm_linear._array(gram, (width, width), 'Gram')
    fsm_linear._array(cross, (width, 3), 'cross')
    require(np.array_equal(gram, gram.T), 'symmetric Gram required')
    system = gram.copy()
    system[np.arange(width-1), np.arange(width-1)] += alpha
    require(np.isfinite(system).all(), 'nonfinite regularized system')
    coefficients = cho_solve(cho_factor(system, lower=True, check_finite=False), cross,
                             check_finite=False).T
    model = fsm_linear.VARXModel(coefficients, order, alpha, rows, fsm_linear.FIT_IDS)
    residual = np.linalg.norm(system @ model.coefficients.T-cross)
    denominator = np.linalg.norm(system)*np.linalg.norm(model.coefficients)+np.linalg.norm(cross)
    require(np.isfinite(residual) and np.isfinite(denominator), 'nonfinite ridge certificate norm')
    certificate = float(residual/denominator) if denominator > 0 else float(residual)
    require(np.isfinite(certificate) and certificate <= 1e-10, 'ridge backward-error certificate failed')
    return model, certificate


def fit_linear(family, statistics, cfg, directory):
    directory.mkdir()
    order, alpha = LINEAR[family]
    started, error, model, certificate = time.perf_counter(), None, None, None
    try:
        require(statistics is not None, 'sufficient statistics unavailable')
        model, certificate = solve_statistics(*statistics, order=order, alpha=alpha)
        np.savez_compressed(directory/'model.npz', coefficients=model.coefficients)
        require(time.perf_counter()-started <= cfg['solve_timeout_seconds'], 'linear solve cap exceeded')
    except Exception as exc:  # noqa: BLE001 - preserve each declared attempt
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    receipt = {'family': family, 'seed': None, 'order': order, 'alpha': alpha,
               'status': 'complete' if error is None else 'failed', 'error': error,
               'solve_seconds': time.perf_counter()-started, 'backward_error': certificate,
               'model_spec': model.model_spec() if model is not None else None,
               'model_sha256': sha(directory/'model.npz') if (directory/'model.npz').exists() else None,
               'statistics_path': f'statistics-order{order}.npz',
               'timing_scope': 'solve and checkpoint write; shared statistics charged separately'}
    write(directory/'receipt.json', receipt)
    return (model if error is None else None), receipt


def unavailable(family, seed, reason, directory):
    directory.mkdir()
    result = {'family': family, 'seed': seed,
              'rows': [{'record_id': rid, 'status': 'not_run', 'error': reason} for rid in DEV_IDS],
              'request_ms': [], 'median_request_ms': None, 'timing_error': reason,
              'model_spec': None, 'persistent_numeric_bytes': None}
    write(directory/'receipt.json', result)
    return result


def folded_parity(model, unfused_directory, records, cfg, norm, directory):
    directory.mkdir()
    rows = []
    for record in records:
        try:
            with np.load(unfused_directory/(record.record_id+'.npz'), allow_pickle=False) as source:
                expected, starts, target = source['prediction'].copy(), source['starts'].copy(), source['target'].copy()
            inputs, _ = normalized_request(record, starts, cfg['context'], cfg['horizon'], norm)
            actual = fsm_linear.predict(model, *inputs)
            np.savez_compressed(directory/(record.record_id+'.npz'), prediction=actual, target=target, starts=starts)
            require(actual.shape == expected.shape and expected.dtype == np.float64
                    and np.isfinite(expected).all(), 'invalid unfused forecast')
            rows.append({'record_id': record.record_id,
                         'passed': bool(np.allclose(actual, expected, atol=cfg['parity_atol'], rtol=cfg['parity_rtol'])),
                         'max_abs_difference': float(np.max(np.abs(actual-expected)))})
        except Exception as exc:  # noqa: BLE001 - parity failure forbids folded timing
            rows.append({'record_id': record.record_id, 'passed': False,
                         'error': f'{type(exc).__name__}: {exc}'})
    return {'passed': len(rows) == 12 and all(r['passed'] for r in rows), 'records': rows,
            'atol': cfg['parity_atol'], 'rtol': cfg['parity_rtol'],
            'scope': 'all fresh unfused DEV predictions, before folded scoring and timing'}


def summarize(evaluations, fits, folds):
    families = {}
    for family in FAMILIES:
        members = [e for e in evaluations if e['family'] == family]
        seeds = {None} if family in (*LINEAR, NATIVE) else set(SEEDS)
        roster = len(members) == len(seeds) and {e['seed'] for e in members} == seeds
        f = [r for r in fits if r['family'] == family]
        fit_ok = family not in LINEAR or (len(f) == 1 and f[0]['status'] == 'complete')
        if family == FOLDED:
            fit_ok = (len(folds) == 3 and {f['seed'] for f in folds} == set(SEEDS)
                      and all(f['status'] == 'complete' and f.get('parity', {}).get('passed') for f in folds))
        score = fit_ok and roster and all(len(e['rows']) == 12
            and {r['record_id'] for r in e['rows']} == set(DEV_IDS)
            and all(r['status'] == 'complete' and np.isfinite(r['rmse']) and r['rmse'] >= 0 for r in e['rows'])
            for e in members)
        latency = roster and all(e['timing_error'] is None and len(e['request_ms']) == 24
            and all(np.isfinite(t) and t > 0 for t in e['request_ms'])
            and e['median_request_ms'] is not None and np.isfinite(e['median_request_ms'])
            and e['median_request_ms'] > 0 for e in members)
        storage = roster and all(type(e['persistent_numeric_bytes']) is int
                                  and e['persistent_numeric_bytes'] > 0 for e in members)
        v = {'eligible': score and latency and storage, 'score_eligible': score,
             'latency_eligible': latency, 'storage_eligible': storage}
        if score: v['mean_rmse'] = float(np.mean([r['rmse'] for e in members for r in e['rows']]))
        if latency: v['mean_seed_median_ms'] = float(np.mean([e['median_request_ms'] for e in members]))
        if storage: v['persistent_numeric_bytes'] = max(e['persistent_numeric_bytes'] for e in members)
        families[family] = v
    candidate, output = SELECTED['tanh_feedback'], SELECTED['tanh_output_only']
    controls = [f for f in FAMILIES if f != candidate and families[f]['score_eligible']]
    strongest = min(controls, key=lambda f: (families[f]['mean_rmse'], f)) if controls else None
    conditions = {
        'all_9_fits_complete': len(fits) == 9 and {f['family'] for f in fits} == set(LINEAR)
            and all(f['status'] == 'complete' for f in fits),
        'all_25_evaluations_complete': len(evaluations) == 25
            and {(e['family'], e['seed']) for e in evaluations} == ROSTER
            and all(f['eligible'] for f in families.values()),
        'all_3_folds_equivalent': len(folds) == 3 and {f['seed'] for f in folds} == set(SEEDS)
            and all(f['status'] == 'complete' and f.get('parity', {}).get('passed') for f in folds),
        'five_percent_below_strongest_control': False,
        'every_seed_below_strongest_control': False,
        'no_record_over_two_percent_strongest_control': False,
        'both_amplitudes_below_strongest_control': False,
        'latency_within_ten_percent_tanh_output': False,
        'storage_no_more_than_tanh_output': False}
    if families[candidate]['score_eligible'] and strongest:
        conditions['five_percent_below_strongest_control'] = families[candidate]['mean_rmse'] <= .95*families[strongest]['mean_rmse']
        def seed_mean(f, s):
            seed = None if f in (*LINEAR, NATIVE) else s
            return float(np.mean([r['rmse'] for e in evaluations if e['family'] == f and e['seed'] == seed for r in e['rows']]))
        conditions['every_seed_below_strongest_control'] = all(seed_mean(candidate, s) < seed_mean(strongest, s) for s in SEEDS)
        def record_means(f):
            return {rid: float(np.mean([r['rmse'] for e in evaluations if e['family'] == f for r in e['rows']
                                       if r['record_id'] == rid])) for rid in DEV_IDS}
        c, b = record_means(candidate), record_means(strongest)
        conditions['no_record_over_two_percent_strongest_control'] = all(c[r] <= 1.02*b[r] for r in DEV_IDS)
        conditions['both_amplitudes_below_strongest_control'] = all(
            np.mean([v for r, v in c.items() if r.startswith(a)]) < np.mean([v for r, v in b.items() if r.startswith(a)])
            for a in fsm_data.AMPLITUDES)
    conditions['latency_within_ten_percent_tanh_output'] = (families[candidate]['latency_eligible'] and families[output]['latency_eligible']
        and families[candidate]['mean_seed_median_ms'] <= 1.1*families[output]['mean_seed_median_ms'])
    conditions['storage_no_more_than_tanh_output'] = (families[candidate]['storage_eligible'] and families[output]['storage_eligible']
        and families[candidate]['persistent_numeric_bytes'] <= families[output]['persistent_numeric_bytes'])
    return {'status': 'DEVELOPMENT_PASS' if all(conditions.values()) else 'DEVELOPMENT_FAIL',
            'families': families, 'candidate': candidate, 'strongest_control': strongest,
            'conditions': conditions, 'passed': sum(bool(v) for v in conditions.values()), 'total': len(conditions),
            'selection': 'Fixed parent candidate; globally strongest complete control by exposed DEV error, then lexical name. Timing failure never replaces it.'}


def validate_pins(protocol):
    require(set(protocol['source_sha256']) == set(SOURCES), 'exact source roster required')
    for name, expected in protocol['source_sha256'].items():
        require(sha(name) == expected, 'source drift: '+name)
    for item in protocol['parent_artifacts'].values():
        require(sha(item['path']) == item['sha256'], 'parent drift: '+item['path'])
    require(sha(protocol['protocol_markdown_path']) == protocol['protocol_markdown_sha256'], 'protocol text drift')


def parent_models(protocol, backbone):
    artifacts = protocol['parent_artifacts']
    summary = json.loads(Path(artifacts['residual_summary']['path']).read_text())
    audit = json.loads(Path(artifacts['residual_audit']['path']).read_text())
    require(summary['selected_by_architecture'] == SELECTED and summary['status'] == 'DEVELOPMENT_PASS'
            and audit['status'] == 'PASS' and audit['agreement'] is True, 'closed selected parent required')
    require(audit['inputs']['registration']['sha256'] == artifacts['residual_registration']['sha256']
            and audit['inputs']['process']['sha256'] == artifacts['residual_process']['sha256'], 'parent audit joins')
    process = json.loads(Path(artifacts['residual_process']['path']).read_text())
    closure = json.loads(Path(artifacts['residual_closure']['path']).read_text())
    require(process['observed_exit_code'] == 0 and closure['status'] == 'completed'
            and process['summary_sha256'] == closure['summary_sha256'] == artifacts['residual_summary']['sha256'],
            'parent terminal summary join')
    models, receipts = {}, []
    for arch, family in SELECTED.items():
        kind, mode = arch.split('_', 1)
        for seed in SEEDS:
            item = artifacts[f'checkpoint_{arch}_{seed}']
            relative = f'{family}-{seed}/final.pt'
            require(audit['inputs']['files'][relative]['sha256'] == item['sha256'], 'checkpoint audit join')
            payload = torch.load(item['path'], map_location='cpu', weights_only=True)
            require(payload['accepted_updates'] == 2048, 'incomplete parent checkpoint')
            model = FSMResidual(backbone.coefficients, order=32, mode=mode, residual_kind=kind, hidden_width=24, seed=seed)
            require(set(payload['model']) == set(model.state_dict()), 'parent state roster')
            for name, tensor in payload['model'].items():
                require(tensor.dtype == torch.float64 and tensor.shape == model.state_dict()[name].shape
                        and torch.isfinite(tensor).all(), 'parent tensor contract')
            require(np.array_equal(payload['model']['coefficients'].numpy(), backbone.coefficients), 'parent backbone differs')
            model.load_state_dict(payload['model'], strict=True); model.eval()
            model._validate_parameters()
            models[(family, seed)] = model
            receipts.append({'family': family, 'seed': seed, 'status': 'complete',
                             'checkpoint_sha256': item['sha256'], 'accepted_updates': 2048,
                             'model_spec': model.model_spec(), 'new_updates': 0})
    return models, receipts


def run(protocol_path, data_path, output):
    protocol = json.loads(Path(protocol_path).read_text()); cfg = protocol['experiment']
    require(cfg == {'orders': list(ORDERS), 'alphas': list(ALPHAS), 'seeds': list(SEEDS),
        'selected_parent_recipes': SELECTED, 'context': 100, 'horizon': 128, 'evaluation_stride': 256,
        'parity_atol': 1e-9, 'parity_rtol': 1e-9, 'statistics_timeout_seconds': 120.,
        'solve_timeout_seconds': 120.}, 'fixed production configuration required')
    validate_pins(protocol)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write(output/'protocol.json', protocol)
    for name in SOURCES:
        target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    (output/'parent').mkdir()
    for name, item in protocol['parent_artifacts'].items():
        (output/'parent'/(name+Path(item['path']).suffix)).write_bytes(Path(item['path']).read_bytes())
    torch.set_num_threads(1)
    backbone, norm = load_parent(protocol)
    models, parents = parent_models(protocol, backbone)
    write(output/'parents.json', parents)
    require(sha(data_path) == protocol['data_sha256'], 'data changed before decode')
    data = fsm_data.read_npz_estimation(data_path)
    require(data.source_sha256 == protocol['data_sha256'], 'data drift')
    write(output/'admission.json', {'source_sha256': data.source_sha256, 'decoded_keys': list(data.decoded_keys),
        'fit_ids': list(norm.fit_record_ids), 'dev_ids': [r.record_id for r in data.partition('dev')],
        'fit_samples': norm.fit_samples, 'normalizer': 'unchanged parent FIT normalization',
        'scope': 'same exposed FIT/DEV; no 300mV or official-test decode'})
    started = time.perf_counter(); fits, folds, evaluations = [], [], []
    train = normalized_records(data.partition('fit'), norm)
    for order in ORDERS:
        stats, error, before = None, None, time.perf_counter()
        try:
            gram, cross, rows = sufficient_statistics(train, order)
            np.savez_compressed(output/f'statistics-order{order}.npz', gram=gram, cross=cross)
            require(time.perf_counter()-before <= cfg['statistics_timeout_seconds'], 'statistics cap exceeded')
            stats = gram, cross, rows
        except Exception as exc:  # noqa: BLE001 - preserve all order/solve failures
            error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        write(output/f'statistics-order{order}.json', {'order': order, 'status': 'complete' if error is None else 'failed',
            'error': error, 'seconds': time.perf_counter()-before, 'fit_rows': stats[2] if stats else None,
            'fit_record_ids': list(fsm_linear.FIT_IDS),
            'sha256': sha(output/f'statistics-order{order}.npz') if (output/f'statistics-order{order}.npz').exists() else None})
        for family, (p, _) in LINEAR.items():
            if p != order: continue
            model, receipt = fit_linear(family, stats, cfg, output/family)
            fits.append(receipt)
            if model is not None: models[(family, None)] = model
            write(output/'fits.json', fits)
    write(output/'fits-closed.json', {'fits': len(fits), 'complete': sum(f['status'] == 'complete' for f in fits),
        'elapsed_seconds': time.perf_counter()-started, 'barrier': 'all new FIT solves close before DEV forecasting'})
    models[(NATIVE, None)] = backbone
    # Native and all unchanged neural controls are measured in this run too.
    eval_roster = [(f, None) for f in LINEAR] + [(NATIVE, None)] + [(f, s) for f in SELECTED.values() for s in SEEDS]
    for family, seed in eval_roster:
        directory = output/(family if seed is None else f'{family}-{seed}')
        directory.mkdir(exist_ok=True)
        if (family, seed) not in models:
            e = unavailable(family, seed, 'fit_failed', directory/'evaluation')
        else:
            e = evaluate(models[(family, seed)], {'family': family, 'seed': seed}, data.partition('dev'), cfg, norm,
                         directory/'evaluation', linear=family in (*LINEAR, NATIVE))
        evaluations.append(e); write(output/'evaluations.json', evaluations)
    for seed in SEEDS:
        directory = output/f'{FOLDED}-{seed}'; directory.mkdir()
        receipt = {'family': FOLDED, 'seed': seed, 'status': 'failed', 'error': None, 'parity': None}
        try:
            folded = fold_affine_feedback(models[(SELECTED['affine_feedback'], seed)], backbone)
            np.savez_compressed(directory/'model.npz', coefficients=folded.coefficients)
            receipt['model_sha256'] = sha(directory/'model.npz')
            receipt['model_spec'] = folded.model_spec()
            parity = folded_parity(folded, output/f'{SELECTED["affine_feedback"]}-{seed}'/'evaluation',
                                   data.partition('dev'), cfg, norm, directory/'parity')
            receipt['parity'] = parity; write(directory/'parity.json', parity)
            require(parity['passed'], 'folded parity failed before timing')
            receipt['status'] = 'complete'
        except Exception as exc:  # noqa: BLE001 - no fallback timing or replacement
            receipt['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
            e = unavailable(FOLDED, seed, 'fold_or_parity_failed', directory/'evaluation')
        else:
            # Numeric forecast/timing failures are recorded by evaluate. An
            # unexpected I/O/schema failure remains fatal with its original cause.
            e = evaluate(folded, receipt, data.partition('dev'), cfg, norm, directory/'evaluation', linear=True)
        folds.append(receipt); write(directory/'receipt.json', receipt); write(output/'folds.json', folds)
        evaluations.append(e); write(output/'evaluations.json', evaluations)
    summary = summarize(evaluations, fits, folds)
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
