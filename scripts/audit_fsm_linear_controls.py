"""Independent saved-output audit of stronger FSM linear controls.

No producer/folding imports, model construction, inference or fitting. The
study contract is bound before any scientific payload can be decoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import audit_fsm_correction as common
import numpy as np
import torch

require, read, pin, close, arrays = common.require, common.read, common.pin, common.close, common.arrays
DEV, FIT = common.DEV, common.FIT
SEEDS = (9201, 9202, 9203)
ORDERS, ALPHAS = (32, 64, 96), (1e-6, 1e-3, .1)
LINEAR = {f'varx{p}-ridge{a:g}': (p, a) for p in ORDERS for a in ALPHAS}
SELECTED = {'affine_output_only': 'affine_output_only-lr0.0001',
            'affine_feedback': 'affine_feedback-lr0.0001',
            'tanh_output_only': 'tanh_output_only-lr0.001',
            'tanh_feedback': 'tanh_feedback-lr0.0003'}
FOLDED, NATIVE = 'folded_affine_feedback', 'native_varx'
CANDIDATE, OUTPUT = SELECTED['tanh_feedback'], SELECTED['tanh_output_only']
FAMILIES = (*LINEAR, *SELECTED.values(), FOLDED, NATIVE)
ROSTER = {(f, None) for f in (*LINEAR, NATIVE)} | {(f, s) for f in (*SELECTED.values(), FOLDED) for s in SEEDS}
ROOT = Path(__file__).resolve().parents[1]
REG = 'research/fsm-linear-controls-registration.json'
REGISTRATION_SHA256 = '996e492b7db79754fec9ac79239563431076673fe378ca696d240ca07d53c364'
COMMAND = ['.venv/bin/python', '-u', '-m', 'openjev.research.fsm_linear_controls_study', '--protocol', REG, '--data', 'output/fsm-correction-engineering-v1/admission-01/combined_data.npz', '--output', 'output/fsm-linear-controls-study-v1']


def record_metrics(prediction, target, scale):
    """Reconstruct normalized and separate native-channel errors only."""
    require(scale.dtype == np.float64 and scale.shape == (3,) and np.isfinite(scale).all() and (scale > 0).all(), 'native output scales')
    result = common.metrics(prediction, target)
    with np.errstate(over='ignore', invalid='ignore'):
        physical = np.asarray(result['per_channel_rmse'])*scale
    require(np.isfinite(physical).all(), 'nonfinite native channel errors')
    result['native_output_per_channel_rmse'] = physical.tolist()
    return result


def folded_coefficients(state, backbone):
    """Direct algebraic certificate; no folding/model implementation is used."""
    require(isinstance(backbone, np.ndarray) and backbone.dtype == np.float64 and backbone.shape == (3, 196) and np.isfinite(backbone).all(), 'parent backbone geometry')
    shapes = {'coefficients': (3, 196), 'residual.weight': (3, 195), 'residual.bias': (3,)}
    require(set(state) == set(shapes), 'affine checkpoint tensor roster')
    for name, shape in shapes.items():
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float64 and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), 'finite CPU float64 affine tensor')
    require(np.array_equal(state['coefficients'].numpy(), backbone), 'affine frozen backbone changed')
    result = backbone.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        result[:, :-1] += state['residual.weight'].numpy()
        result[:, -1] += state['residual.bias'].numpy()
    require(np.isfinite(result).all(), 'nonfinite folded coefficients')
    return result


def prediction_parity(folded, unfused, *, rtol, atol):
    require(folded.dtype == unfused.dtype == np.float64 and folded.shape == unfused.shape == (32, 128, 3) and np.isfinite(folded).all() and np.isfinite(unfused).all(), 'finite folded/unfused forecast geometry')
    difference = np.abs(folded-unfused)
    return {'passed': bool(np.allclose(folded, unfused, rtol=rtol, atol=atol)), 'max_abs_difference': float(difference.max()), 'rtol': rtol, 'atol': atol}


def ridge_certificate(coefficients, gram, cross, order, alpha):
    """Backward error for saved sufficient statistics, not a raw FIT replay."""
    width = 6*order+4
    require(order in ORDERS and alpha in ALPHAS, 'declared ridge recipe')
    for value, shape in ((coefficients, (3, width)), (gram, (width, width)), (cross, (width, 3))):
        require(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(), 'finite ridge certificate geometry')
    require(np.array_equal(gram, gram.T), 'symmetric mean Gram')
    regularized = gram.copy()
    regularized[np.arange(width-1), np.arange(width-1)] += alpha
    solution = coefficients.T
    numerator = float(np.linalg.norm(regularized@solution-cross))
    denominator = float(np.linalg.norm(regularized)*np.linalg.norm(solution)+np.linalg.norm(cross))
    require(np.isfinite(numerator) and np.isfinite(denominator), 'finite ridge backward error')
    relative = numerator/denominator if denominator else numerator
    require(relative <= 1e-10, 'ridge normal equation disagrees')
    return {'relative_backward_error': relative, 'tolerance': 1e-10, 'numerator': numerator, 'denominator': denominator}


def family_summary(family, members, fits, folds):
    seeds = {None} if family in (*LINEAR, NATIVE) else set(SEEDS)
    roster = len(members) == len(seeds) and {e['seed'] for e in members} == seeds
    fitted = [f for f in fits if f['family'] == family]
    fit_ok = family not in LINEAR or (len(fitted) == 1 and fitted[0]['status'] == 'complete')
    if family == FOLDED:
        fit_ok = len(folds) == 3 and {f['seed'] for f in folds} == set(SEEDS) and all(f['status'] == 'complete' and f.get('parity', {}).get('passed') for f in folds)
    score = fit_ok and roster and all(len(e['rows']) == 12 and {r['record_id'] for r in e['rows']} == set(DEV) and all(r['status'] == 'complete' and type(r['rmse']) in (int, float) and np.isfinite(r['rmse']) and r['rmse'] >= 0 for r in e['rows']) for e in members)
    latency = roster and all(e['timing_error'] is None and len(e['request_ms']) == 24 and all(type(t) in (int, float) and np.isfinite(t) and t > 0 for t in e['request_ms']) and type(e['median_request_ms']) in (int, float) and np.isfinite(e['median_request_ms']) and e['median_request_ms'] > 0 for e in members)
    storage = roster and all(type(e['persistent_numeric_bytes']) is int and e['persistent_numeric_bytes'] > 0 for e in members)
    result = {'eligible': score and latency and storage, 'score_eligible': score, 'latency_eligible': latency, 'storage_eligible': storage}
    if score:
        result['mean_rmse'] = float(np.mean([r['rmse'] for e in members for r in e['rows']]))
    if latency:
        result['mean_seed_median_ms'] = float(np.mean([e['median_request_ms'] for e in members]))
    if storage:
        result['persistent_numeric_bytes'] = max(e['persistent_numeric_bytes'] for e in members)
    return result


def decisions(evaluations, fits, folds):
    groups = {f: [e for e in evaluations if e['family'] == f] for f in FAMILIES}
    families = {f: family_summary(f, group, fits, folds) for f, group in groups.items()}
    controls = [f for f in FAMILIES if f != CANDIDATE and families[f]['score_eligible']]
    strongest = min(controls, key=lambda f: (families[f]['mean_rmse'], f)) if controls else None
    conditions = {'all_9_fits_complete': len(fits) == 9 and {f['family'] for f in fits} == set(LINEAR) and all(f['status'] == 'complete' for f in fits),
                  'all_25_evaluations_complete': len(evaluations) == 25 and {(e['family'], e['seed']) for e in evaluations} == ROSTER and all(f['eligible'] for f in families.values()),
                  'all_3_folds_equivalent': len(folds) == 3 and {f['seed'] for f in folds} == set(SEEDS) and all(f['status'] == 'complete' and f.get('parity', {}).get('passed') for f in folds),
                  'five_percent_below_strongest_control': False, 'every_seed_below_strongest_control': False,
                  'no_record_over_two_percent_strongest_control': False, 'both_amplitudes_below_strongest_control': False,
                  'latency_within_ten_percent_tanh_output': False, 'storage_no_more_than_tanh_output': False}
    if families[CANDIDATE]['score_eligible'] and strongest:
        conditions['five_percent_below_strongest_control'] = families[CANDIDATE]['mean_rmse'] <= .95*families[strongest]['mean_rmse']
        def mean(family, seed=None, record=None):
            return float(np.mean([r['rmse'] for e in groups[family] if seed is None or e['seed'] == (None if family in (*LINEAR, NATIVE) else seed) for r in e['rows'] if record is None or r['record_id'] == record]))
        conditions['every_seed_below_strongest_control'] = all(mean(CANDIDATE, seed=s) < mean(strongest, seed=s) for s in SEEDS)
        c, b = ({r: mean(f, record=r) for r in DEV} for f in (CANDIDATE, strongest))
        conditions['no_record_over_two_percent_strongest_control'] = all(c[r] <= 1.02*b[r] for r in DEV)
        conditions['both_amplitudes_below_strongest_control'] = all(np.mean([v for r, v in c.items() if r.startswith(a)]) < np.mean([v for r, v in b.items() if r.startswith(a)]) for a in ('100mV', '200mV'))
    c, o = families[CANDIDATE], families[OUTPUT]
    conditions['latency_within_ten_percent_tanh_output'] = c['latency_eligible'] and o['latency_eligible'] and c['mean_seed_median_ms'] <= 1.1*o['mean_seed_median_ms']
    conditions['storage_no_more_than_tanh_output'] = c['storage_eligible'] and o['storage_eligible'] and c['persistent_numeric_bytes'] <= o['persistent_numeric_bytes']
    return {'status': 'DEVELOPMENT_PASS' if all(conditions.values()) else 'DEVELOPMENT_FAIL', 'families': families, 'candidate': CANDIDATE, 'strongest_control': strongest, 'conditions': conditions, 'passed': sum(bool(v) for v in conditions.values()), 'total': 9, 'selection': 'Fixed parent candidate; globally strongest complete control by exposed DEV error, then lexical name. Timing failure never replaces it.'}


SOURCES = {f'src/openjev/research/{name}.py' for name in (
    'fsm_linear_controls_study', 'fsm_affine_fold', 'fsm_residual_study',
    'fsm_residual', 'fsm_linear', 'fsm_data', 'fsm_study', 'fsm_gru', 'predictive_state_correction')}
PARENTS = {'coefficients', 'fit_receipt', 'normalizer', 'residual_summary', 'residual_audit',
           'residual_registration', 'residual_process', 'residual_closure'} | {
    f'checkpoint_{arch}_{seed}' for arch in SELECTED for seed in SEEDS}
COMMON_SHA256 = '2b38af454f28eb5c606bd3cadc4698dcdbcd5996e629dbb05a15371f0b440c8b'
PARENT_ROOT = Path('output/fsm-residual-study-v1')
CONFIG = {'orders': list(ORDERS), 'alphas': list(ALPHAS), 'seeds': list(SEEDS),
          'selected_parent_recipes': SELECTED, 'context': 100, 'horizon': 128, 'evaluation_stride': 256,
          'parity_atol': 1e-9, 'parity_rtol': 1e-9, 'statistics_timeout_seconds': 120., 'solve_timeout_seconds': 120.}


def canonical_relative(value):
    require(isinstance(value, str) and '\\' not in value and not Path(value).is_absolute() and all(p not in ('', '.', '..') for p in value.split('/')), 'canonical relative path required')
    return Path(value)


def parent_key(relative, recorded_root):
    path = canonical_relative(relative)
    require(path.parts[:len(PARENT_ROOT.parts)] == PARENT_ROOT.parts and len(path.parts) > len(PARENT_ROOT.parts), 'payload outside residual parent root')
    require(isinstance(recorded_root, str) and '\\' not in recorded_root and all(p not in ('.', '..') for p in recorded_root.split('/')), 'canonical parent root required')
    root = Path(recorded_root)
    require(root.is_absolute() and root.parts[-len(PARENT_ROOT.parts):] == PARENT_ROOT.parts, 'recorded parent root suffix')
    return str(path.relative_to(PARENT_ROOT))


def authenticate(study, process):
    terminal = read(process)
    require(terminal.get('observed_exit_code') == 0 and terminal.get('command') == COMMAND and type(terminal.get('tool_session_id')) is int and terminal['tool_session_id'] > 0 and terminal.get('observation') == 'original Codex exec/write_stdin completion', 'original successful process required')
    require(isinstance(REGISTRATION_SHA256, str) and len(REGISTRATION_SHA256) == 64 and pin(ROOT/REG)['sha256'] == REGISTRATION_SHA256, 'unbound or changed registration')
    plan = read(ROOT/REG)
    require(study == ROOT/'output/fsm-linear-controls-study-v1' and read(study/'protocol.json') == plan, 'study registration copy')
    require(plan['experiment'] == CONFIG and set(plan['source_sha256']) == SOURCES and set(plan['parent_artifacts']) == PARENTS, 'frozen config/source/parent roster')
    for name, sha in plan['source_sha256'].items():
        require(pin(ROOT/name)['sha256'] == sha == pin(study/'source'/name)['sha256'], 'source drift')
    require(pin(ROOT/canonical_relative(plan['protocol_markdown_path']))['sha256'] == plan['protocol_markdown_sha256'], 'protocol drift')
    paths = {}
    for key, item in plan['parent_artifacts'].items():
        original = ROOT/canonical_relative(item['path'])
        copied = study/'parent'/(key+original.suffix)
        require(pin(original)['sha256'] == item['sha256'] == pin(copied)['sha256'], 'parent payload drift')
        paths[key] = copied
    parent_audit, parent_process, parent_closure, parent_summary = (read(paths[k]) for k in ('residual_audit', 'residual_process', 'residual_closure', 'residual_summary'))
    require(parent_audit['status'] == 'PASS' and parent_audit['agreement'] is True and parent_process['observed_exit_code'] == 0 and parent_closure['status'] == 'completed', 'closed parent evidence')
    require(parent_audit['inputs']['registration'] == pin(paths['residual_registration']) and parent_audit['inputs']['process']['sha256'] == pin(paths['residual_process'])['sha256'], 'parent audit input joins')
    require(parent_process['summary_sha256'] == parent_closure['summary_sha256'] == pin(paths['residual_summary'])['sha256'], 'parent terminal summary join')
    require(parent_summary['status'] == 'DEVELOPMENT_PASS' and parent_summary['selected_by_architecture'] == SELECTED, 'fixed parent selection')
    for key in ('residual_summary', 'residual_closure', *(f'checkpoint_{a}_{s}' for a in SELECTED for s in SEEDS)):
        relative = parent_key(plan['parent_artifacts'][key]['path'], parent_audit['inputs']['study'])
        require(parent_audit['inputs']['files'][relative] == pin(paths[key]), 'parent audited payload join')
    parent_plan = read(paths['residual_registration'])
    for key in ('coefficients', 'fit_receipt', 'normalizer'):
        require(plan['parent_artifacts'][key] == parent_plan['parent_artifacts'][key] and parent_audit['inputs']['files']['parent/'+key+paths[key].suffix] == pin(paths[key]), 'original backbone/normalizer lineage')
    closure = read(study/'closure.json')
    require(closure['status'] == 'completed' and closure['source_sha256'] == plan['source_sha256'] and closure['summary_sha256'] == terminal['summary_sha256'] == pin(study/'summary.json')['sha256'], 'original closure join')
    require(pin(Path(common.__file__))['sha256'] == COMMON_SHA256, 'independent helper drift')
    return plan, paths, terminal, closure


def checkpoint_state(payload, architecture, backbone):
    kind, mode = architecture.split('_', 1)
    shapes = ({'residual.weight': (3, 195), 'residual.bias': (3,)} if kind == 'affine' else
              {'residual.0.weight': (24, 195), 'residual.0.bias': (24,), 'residual.2.weight': (3, 24), 'residual.2.bias': (3,)})
    require(set(payload) == {'model', 'optimizer', 'accepted_updates'} and payload['accepted_updates'] == 2048, 'complete frozen parent checkpoint')
    state = payload['model']
    require(set(state) == {'coefficients', *shapes}, 'frozen checkpoint tensor roster')
    for name, shape in {'coefficients': (3, 196), **shapes}.items():
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float64 and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), 'frozen tensor geometry')
    require(np.array_equal(state['coefficients'].numpy(), backbone), 'frozen parent backbone changed')
    return state, sum(int(np.prod(s)) for s in shapes.values()), mode


def validate_spec(spec, *, order, alpha, kind=None, mode=None):
    if kind is None:
        count = 3*(6*order+4)
        require(spec['order'] == order and spec['alpha'] == alpha and spec['fit_rows'] == 12*(8192-order) and spec['fit_record_ids'] == list(FIT) and spec['scalar_metadata_bytes'] == 24 and spec['retained_numeric_bytes'] == count*8+24, 'linear historical fit/spec')
        tensors, metadata = count*8, 24
    else:
        count = 588 if kind == 'affine' else 4779
        metadata = 16 if kind == 'affine' else 24
        require(spec['order'] == 32 and spec['residual_kind'] == kind and spec['mode'] == mode and spec['trainable_parameter_count'] == count and spec['buffer_bytes'] == 4704 and spec['numeric_metadata_bytes'] == metadata, 'residual spec')
        tensors = count*8+4704
    require(spec['dtype'] == 'float64' and spec['parameter_count'] == count and spec['parameter_bytes'] == count*8 and spec['state_scalars'] == 6*order, 'model parameter/state accounting')
    return {'tensor_bytes': tensors, 'state_bytes': 6*order*8, 'normalizer_bytes': 96, 'scalar_metadata_bytes': metadata, 'persistent_numeric_bytes': tensors+6*order*8+96+metadata}


def audit(study, process):
    study, process = Path(study).resolve(), Path(process).resolve()
    plan, paths, terminal, closure = authenticate(study, process)
    before = common.inventory(study)
    expected = {'protocol.json', 'parents.json', 'admission.json', 'fits.json', 'fits-closed.json', 'evaluations.json', 'folds.json', 'summary.json', 'closure.json'} | {'source/'+n for n in SOURCES} | {'parent/'+p.name for p in paths.values()}
    backbone = arrays(paths['coefficients'], ('coefficients',))['coefficients']
    require(backbone.dtype == np.float64 and backbone.shape == (3, 196) and np.isfinite(backbone).all(), 'frozen coefficient geometry')
    parent_fit = read(paths['fit_receipt'])
    require(parent_fit['status'] == 'complete' and parent_fit['order'] == 32 and parent_fit['alpha'] == 1e-6, 'backbone recipe')
    validate_spec(parent_fit['model_spec'], order=32, alpha=1e-6)
    norm = arrays(paths['normalizer'], ('u_mean', 'u_scale', 'y_mean', 'y_scale'))
    require(all(v.dtype == np.float64 and v.shape == (3,) and np.isfinite(v).all() for v in norm.values()) and all((norm[k] > 0).all() for k in ('u_scale', 'y_scale')), 'frozen FIT normalizer')
    admission = read(study/'admission.json')
    require(admission['source_sha256'] == plan['data_sha256'] and set(admission['decoded_keys']) == {'u_100mV_train', 'y_100mV_train', 'u_200mV_train', 'y_200mV_train'} and admission['fit_ids'] == list(FIT) and admission['dev_ids'] == list(DEV) and admission['fit_samples'] == 98304 and admission['normalizer'] == 'unchanged parent FIT normalization', 'data admission')
    fits, folds, parents, evaluations = (read(study/(n+'.json')) for n in ('fits', 'folds', 'parents', 'evaluations'))
    require([(f['family'], f['seed']) for f in fits] == [(f, None) for f in LINEAR], 'ordered nine FIT solves')
    require([(f['family'], f['seed']) for f in parents] == [(f, s) for f in SELECTED.values() for s in SEEDS], 'ordered twelve frozen checkpoints')
    require([(f['family'], f['seed']) for f in folds] == [(FOLDED, s) for s in SEEDS], 'three fold attempts')
    ordered = [(f, None) for f in LINEAR]+[(NATIVE, None)]+[(f, s) for f in SELECTED.values() for s in SEEDS]+[(FOLDED, s) for s in SEEDS]
    require([(e['family'], e['seed']) for e in evaluations] == ordered, 'ordered25 evaluation attempts')
    barrier = read(study/'fits-closed.json')
    require(barrier['fits'] == 9 and barrier['complete'] == sum(f['status'] == 'complete' for f in fits) and np.isfinite(barrier['elapsed_seconds']) and barrier['elapsed_seconds'] > 0 and barrier['barrier'] == 'all new FIT solves close before DEV forecasting', 'FIT barrier')
    statistics, stats_status = {}, {}
    for order in ORDERS:
        base = f'statistics-order{order}'; expected.add(base+'.json'); r = read(study/(base+'.json'))
        require(r['order'] == order and r['fit_record_ids'] == list(FIT) and r['status'] in ('complete', 'failed') and np.isfinite(r['seconds']) and r['seconds'] > 0, 'statistics receipt')
        stats_status[order] = r['status']
        if (study/(base+'.npz')).exists():
            expected.add(base+'.npz'); require(r['sha256'] == pin(study/(base+'.npz'))['sha256'], 'statistics hash')
            a = arrays(study/(base+'.npz'), ('gram', 'cross')); width = 6*order+4
            require(a['gram'].dtype == a['cross'].dtype == np.float64 and a['gram'].shape == (width, width) and a['cross'].shape == (width, 3) and all(np.isfinite(v).all() for v in a.values()) and np.array_equal(a['gram'], a['gram'].T), 'mean sufficient statistics geometry')
            statistics[order] = a
        else:
            require(r['sha256'] is None and r['status'] == 'failed', 'missing statistics retained')
        if r['status'] == 'complete':
            require(order in statistics and r['error'] is None and r['fit_rows'] == 12*(8192-order) and r['seconds'] <= CONFIG['statistics_timeout_seconds'], 'completed statistics')
        else:
            require(isinstance(r['error'], dict) and r['fit_rows'] is None, 'failed statistics')
    accounting = {(NATIVE, None): validate_spec(parent_fit['model_spec'], order=32, alpha=1e-6)}
    specs, certificates, states = {(NATIVE, None): parent_fit['model_spec']}, {}, {}
    for receipt in fits:
        family = receipt['family']; order, alpha = LINEAR[family]; folder = study/family
        expected.add(family+'/receipt.json')
        require(read(folder/'receipt.json') == receipt and receipt['order'] == order and receipt['alpha'] == alpha and receipt['statistics_path'] == f'statistics-order{order}.npz' and receipt['status'] in ('complete', 'failed') and np.isfinite(receipt['solve_seconds']) and receipt['solve_seconds'] > 0, 'linear receipt')
        if (folder/'model.npz').exists():
            expected.add(family+'/model.npz')
            require(receipt['model_sha256'] == pin(folder/'model.npz')['sha256'] and order in statistics, 'linear coefficient hash/statistics')
            coefficients = arrays(folder/'model.npz', ('coefficients',))['coefficients']
            certificate = ridge_certificate(coefficients, statistics[order]['gram'], statistics[order]['cross'], order, alpha)
            close(receipt['backward_error'], certificate['relative_backward_error']); certificates[family] = certificate
            accounting[(family, None)] = validate_spec(receipt['model_spec'], order=order, alpha=alpha); specs[(family, None)] = receipt['model_spec']
        else:
            require(receipt['model_sha256'] is None and receipt['status'] == 'failed', 'missing failed linear payload')
        if receipt['status'] == 'complete':
            require((family, None) in specs and receipt['error'] is None and stats_status[order] == 'complete' and receipt['solve_seconds'] <= CONFIG['solve_timeout_seconds'], 'qualified linear solve')
        else:
            require(isinstance(receipt['error'], dict), 'failed linear evidence')
    for receipt in parents:
        family, seed = receipt['family'], receipt['seed']; arch = next(a for a, f in SELECTED.items() if f == family)
        path = paths[f'checkpoint_{arch}_{seed}']
        payload = torch.load(path, map_location='cpu', weights_only=True)
        state, _, mode = checkpoint_state(payload, arch, backbone); states[(family, seed)] = state
        require(receipt['status'] == 'complete' and receipt['accepted_updates'] == 2048 and receipt['new_updates'] == 0 and receipt['checkpoint_sha256'] == pin(path)['sha256'], 'frozen parent load receipt')
        accounting[(family, seed)] = validate_spec(receipt['model_spec'], order=32, alpha=1e-6, kind=arch.split('_', 1)[0], mode=mode)
        specs[(family, seed)] = receipt['model_spec']
    for receipt in folds:
        seed = receipt['seed']; stem = f'{FOLDED}-{seed}'; folder = study/stem; expected.add(stem+'/receipt.json')
        require(read(folder/'receipt.json') == receipt and receipt['status'] in ('complete', 'failed'), 'fold receipt')
        if (folder/'model.npz').exists():
            expected.add(stem+'/model.npz'); require(pin(folder/'model.npz')['sha256'] == receipt['model_sha256'], 'fold coefficient pin')
            merged = arrays(folder/'model.npz', ('coefficients',))['coefficients']
            require(np.array_equal(merged, folded_coefficients(states[(SELECTED['affine_feedback'], seed)], backbone)), 'merged coefficients disagree')
            accounting[(FOLDED, seed)] = validate_spec(receipt['model_spec'], order=32, alpha=1e-6); specs[(FOLDED, seed)] = receipt['model_spec']
        else:
            require(receipt['status'] == 'failed' and 'model_sha256' not in receipt, 'missing failed fold payload')
        if receipt['status'] == 'complete':
            require(receipt['error'] is None and receipt['parity'] and receipt['parity']['passed'], 'complete fold parity')
        else:
            require(isinstance(receipt['error'], dict), 'fold failure retained')
    targets, banks, timing_samples, rows = {}, 0, 0, 0
    for evaluation in evaluations:
        family, seed = evaluation['family'], evaluation['seed']; identity = family, seed
        stem = family if seed is None else f'{family}-{seed}'; prefix = stem+'/evaluation/'; expected.add(prefix+'receipt.json')
        require(read(study/(prefix+'receipt.json')) == evaluation and [r['record_id'] for r in evaluation['rows']] == list(DEV), 'evaluation receipt/record roster')
        failed = (family in LINEAR and next(f for f in fits if f['family'] == family)['status'] == 'failed') or (family == FOLDED and next(f for f in folds if f['seed'] == seed)['status'] == 'failed')
        if failed:
            reason = 'fit_failed' if family in LINEAR else 'fold_or_parity_failed'
            close(evaluation, {'family': family, 'seed': seed, 'rows': [{'record_id': r, 'status': 'not_run', 'error': reason} for r in DEV], 'request_ms': [], 'median_request_ms': None, 'timing_error': reason, 'model_spec': None, 'persistent_numeric_bytes': None})
            rows += 12
            continue
        require(evaluation['model_spec'] == specs[identity], 'evaluated model spec')
        close({k: evaluation[k] for k in accounting[identity]}, accounting[identity])
        costs = evaluation['request_ms']; timing_samples += len(costs)
        require(len(costs) <= 24 and all(type(v) in (int, float) and np.isfinite(v) and v > 0 for v in costs) and np.isfinite(evaluation['scoring_seconds']) and evaluation['scoring_seconds'] > 0, 'timing evidence')
        if evaluation['timing_error'] is None:
            require(len(costs) == 24, 'complete timing samples'); close(evaluation['median_request_ms'], float(np.median(costs)))
        else:
            require(isinstance(evaluation['timing_error'], str) and evaluation['median_request_ms'] is None, 'failed timing retained')
        for row in evaluation['rows']:
            rows += 1; name = prefix+row['record_id']+'.npz'
            if row['status'] == 'failed':
                require(isinstance(row['error'], str) and set(row) == {'record_id', 'status', 'error'} and not (study/name).exists(), 'failed forecast record')
                continue
            require(row['status'] == 'complete', 'record status'); expected.add(name)
            bank = arrays(study/name, ('prediction', 'target', 'starts')); banks += 1
            require(bank['starts'].dtype == np.int64 and np.array_equal(bank['starts'], np.arange(0, 7937, 256)), '32 causal start indices')
            value = record_metrics(bank['prediction'], bank['target'], norm['y_scale'])
            signature = hashlib.sha256(bank['target'].tobytes()).hexdigest()
            require(targets.setdefault(row['record_id'], signature) == signature, 'targets differ between models')
            close(row, {'record_id': row['record_id'], 'status': 'complete', **value}); row.update(value)
    parity_banks, parity_compared = 0, 0
    for receipt in folds:
        stem = f'{FOLDED}-{receipt["seed"]}'; parity = receipt['parity']
        if parity is None:
            require(not (study/stem/'parity.json').exists(), 'unexpected parity evidence'); continue
        expected.add(stem+'/parity.json')
        require(read(study/stem/'parity.json') == parity and parity['atol'] == parity['rtol'] == 1e-9 and [r['record_id'] for r in parity['records']] == list(DEV), 'fold parity receipt')
        for row in parity['records']:
            name = stem+'/parity/'+row['record_id']+'.npz'
            if not (study/name).exists():
                require(row['passed'] is False and set(row) == {'record_id', 'passed', 'error'} and isinstance(row['error'], str), 'absent parity failure evidence'); continue
            expected.add(name); parity_banks += 1; bank = arrays(study/name, ('prediction', 'target', 'starts'))
            require(bank['starts'].dtype == np.int64 and np.array_equal(bank['starts'], np.arange(0, 7937, 256)), 'parity starts')
            original = arrays(study/f'{SELECTED["affine_feedback"]}-{receipt["seed"]}'/'evaluation'/(row['record_id']+'.npz'), ('prediction', 'target', 'starts'))
            require(np.array_equal(bank['target'], original['target']), 'fold parity target alignment')
            value = prediction_parity(bank['prediction'], original['prediction'], rtol=1e-9, atol=1e-9); parity_compared += 1
            close(row, {'record_id': row['record_id'], 'passed': value['passed'], 'max_abs_difference': value['max_abs_difference']})
            evaluated = study/stem/'evaluation'/(row['record_id']+'.npz')
            if evaluated.exists():
                actual = arrays(evaluated, ('prediction', 'target', 'starts'))
                require(np.array_equal(actual['prediction'], bank['prediction']) and np.array_equal(actual['target'], bank['target']), 'fold parity and evaluation disagree')
        require(type(parity['passed']) is bool and parity['passed'] == all(r['passed'] for r in parity['records']), 'parity aggregate')
    result = decisions(evaluations, fits, folds); summary = read(study/'summary.json')
    close({k: summary[k] for k in result}, result)
    require(closure['result'] == result['status'] and np.isfinite(summary['elapsed_seconds']) and summary['elapsed_seconds'] > 0, 'terminal result/time')
    require(set(before) == expected and common.inventory(study) == before and read(process) == terminal, 'complete inventory/unchanged inputs')
    authenticate(study, process)
    return {'status': 'PASS', 'agreement': True, 'scientific_status': result['status'], 'results': result,
            'source_pins': plan['source_sha256'], 'inputs': {'study': str(study), 'process': dict(path=str(process), **pin(process)), 'registration': pin(ROOT/REG), 'files': before},
            'counts': {'new_linear_fits': 9, 'frozen_checkpoints': 12, 'new_neural_updates': 0, 'fold_attempts': 3, 'evaluation_slots': 25, 'record_rows': rows, 'saved_prediction_banks': banks, 'parity_banks': parity_banks, 'parity_comparisons': parity_compared, 'timing_samples': timing_samples, 'statistics_orders': len(statistics)},
            'ridge_certificates': certificates, 'auditor': pin(__file__),
            'scope': 'Independent saved-output metrics, common targets, ridge backward-error certificates, coefficient folding, fixed selection and nine rules. No model calls, raw data decoding, refits, optimizer or timing replay. Sufficient-statistics FIT lineage and timing samples are attested by pinned sources; a certificate does not independently establish raw-data provenance. Fold parity is independently checked wherever saved finite arrays exist; numerical failures with no arrays remain explicit attested failures. This is exposed-DEV development, not confirmation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--study', required=True); parser.add_argument('--process', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); destination = Path(args.output); require(not destination.exists(), 'exclusive audit output')
    result = audit(args.study, args.process); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
