"""Saved-output residual screen audit. No producer/model imports or model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import audit_fsm_correction as common
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REG = 'research/fsm-residual-registration.json'
REGISTRATION_SHA256 = '1ac959d74756df7ae6e78c070056ec55b97955e3de0ba070490df61bf9cf90ea'
COMMON_SHA256 = '2b38af454f28eb5c606bd3cadc4698dcdbcd5996e629dbb05a15371f0b440c8b'
ARCHITECTURES = ('affine_output_only', 'affine_feedback', 'tanh_output_only', 'tanh_feedback')
RATES, SEEDS = (1e-4, 3e-4, 1e-3), (9201, 9202, 9203)
RECIPES = {f'{a}-lr{r:g}': (a, r) for a in ARCHITECTURES for r in RATES}
ROSTER = {(f, s) for f in RECIPES for s in SEEDS}
DEV, FIT = common.DEV, common.FIT
SOURCES = {f'src/openjev/research/{f}.py' for f in ('fsm_residual_study', 'fsm_residual', 'fsm_study', 'fsm_data', 'fsm_linear', 'fsm_gru', 'predictive_state_correction')}
PARENTS = {'coefficients', 'fit_receipt', 'normalizer', 'registration', 'audit', 'closure', 'summary', 'process'}
PARENT_ROOT = Path('output/fsm-correction-study-v1')
PARENT_PAYLOADS = ('coefficients', 'fit_receipt', 'normalizer', 'summary', 'closure')
COMMAND = ['.venv/bin/python', '-u', '-m', 'openjev.research.fsm_residual_study', '--protocol', REG, '--data', 'output/fsm-correction-engineering-v1/admission-01/combined_data.npz', '--output', 'output/fsm-residual-study-v1']
require, read, pin, close, arrays, metrics = common.require, common.read, common.pin, common.close, common.arrays, common.metrics


def record_metrics(prediction, target, scale):
    require(scale.dtype == np.float64 and scale.shape == (3,) and np.isfinite(scale).all() and (scale > 0).all(), 'native output scales')
    result = metrics(prediction, target)
    with np.errstate(over='ignore'):
        physical = np.asarray(result['per_channel_rmse'])*scale
    require(np.isfinite(physical).all(), 'nonfinite native channel errors')
    result['native_output_per_channel_rmse'] = physical.tolist()
    return result


def family_summary(family, members, fits, cfg):
    seeds = {None} if family == 'native_varx' else set(SEEDS)
    roster = len(members) == len(seeds) and {e['seed'] for e in members} == seeds
    matching = [f for f in fits if f['family'] == family]
    fit_ok = family == 'native_varx' or (len(matching) == 3 and {f['seed'] for f in matching} == set(SEEDS) and all(f['status'] == 'complete' and f['accepted_updates'] == cfg['updates'] for f in matching))
    score = fit_ok and roster and all(len(e['rows']) == 12 and {r['record_id'] for r in e['rows']} == set(DEV) and all(r['status'] == 'complete' and type(r['rmse']) in (int, float) and np.isfinite(r['rmse']) and r['rmse'] >= 0 for r in e['rows']) for e in members)
    latency = roster and all(e['timing_error'] is None and len(e['request_ms']) == 24 and all(np.isfinite(t) and t > 0 for t in e['request_ms']) and type(e['median_request_ms']) in (int, float) and np.isfinite(e['median_request_ms']) and e['median_request_ms'] > 0 for e in members)
    storage = roster and all(type(e['persistent_numeric_bytes']) is int and e['persistent_numeric_bytes'] > 0 for e in members)
    result = {'eligible': score and latency and storage, 'score_eligible': score, 'latency_eligible': latency, 'storage_eligible': storage}
    if score:
        result['mean_rmse'] = float(np.mean([r['rmse'] for e in members for r in e['rows']]))
    if latency:
        result['mean_seed_median_ms'] = float(np.mean([e['median_request_ms'] for e in members]))
    if storage:
        result['persistent_numeric_bytes'] = max(e['persistent_numeric_bytes'] for e in members)
    return result


def decisions(evaluations, fits, cfg):
    groups = {f: [e for e in evaluations if e['family'] == f] for f in (*RECIPES, 'native_varx')}
    families = {f: family_summary(f, group, fits, cfg) for f, group in groups.items()}
    selected = {}
    for a in ARCHITECTURES:
        options = [f for f, (arch, _) in RECIPES.items() if arch == a and families[f]['score_eligible']]
        selected[a] = min(options, key=lambda f: (families[f]['mean_rmse'], RECIPES[f][1])) if options else None
    affine = [selected[a] for a in ARCHITECTURES[:2] if selected[a]]
    best_affine = min(affine, key=lambda f: (families[f]['mean_rmse'], f)) if affine else None
    candidate, output = selected['tanh_feedback'], selected['tanh_output_only']
    controls = [f for f in (output, best_affine, 'native_varx') if f and families[f]['score_eligible']]
    strongest = min(controls, key=lambda f: (families[f]['mean_rmse'], f)) if controls else None
    conditions = {
        'all_36_fits_complete': len(fits) == 36 and {(f['family'], f['seed']) for f in fits} == ROSTER and all(f['status'] == 'complete' and f['accepted_updates'] == cfg['updates'] for f in fits),
        'all_37_evaluations_complete': len(evaluations) == 37 and {(e['family'], e['seed']) for e in evaluations} == ROSTER | {('native_varx', None)} and all(v['eligible'] for v in families.values()),
        'initial_identity_and_frozen_backbone': len(fits) == 36 and all(bool(f.get('initial_identity')) and f['initial_identity']['passed'] and f['backbone_unchanged'] for f in fits),
        'five_percent_below_strongest_control': False, 'every_seed_below_three_selected_controls': False,
        'no_record_over_two_percent_strongest_control': False, 'both_amplitudes_below_strongest_control': False,
        'latency_within_ten_percent_tanh_output': False, 'storage_no_more_than_tanh_output': False}
    if candidate and output and best_affine and len(controls) == 3:
        conditions['five_percent_below_strongest_control'] = families[candidate]['mean_rmse'] <= .95*families[strongest]['mean_rmse']
        def seed_mean(f, seed):
            return float(np.mean([r['rmse'] for e in groups[f] if e['seed'] == (None if f == 'native_varx' else seed) for r in e['rows']]))
        conditions['every_seed_below_three_selected_controls'] = all(seed_mean(candidate, s) < seed_mean(f, s) for s in SEEDS for f in controls)
        def record_means(f):
            return {r: float(np.mean([row['rmse'] for e in groups[f] for row in e['rows'] if row['record_id'] == r])) for r in DEV}
        c, b = record_means(candidate), record_means(strongest)
        conditions['no_record_over_two_percent_strongest_control'] = all(c[r] <= 1.02*b[r] for r in DEV)
        conditions['both_amplitudes_below_strongest_control'] = all(np.mean([v for r, v in c.items() if r.startswith(a)]) < np.mean([v for r, v in b.items() if r.startswith(a)]) for a in ('100mV', '200mV'))
        conditions['latency_within_ten_percent_tanh_output'] = families[candidate]['latency_eligible'] and families[output]['latency_eligible'] and families[candidate]['mean_seed_median_ms'] <= 1.1*families[output]['mean_seed_median_ms']
        conditions['storage_no_more_than_tanh_output'] = families[candidate]['storage_eligible'] and families[output]['storage_eligible'] and families[candidate]['persistent_numeric_bytes'] <= families[output]['persistent_numeric_bytes']
    return {'status': 'DEVELOPMENT_PASS' if all(conditions.values()) else 'DEVELOPMENT_FAIL', 'families': families, 'selected_by_architecture': selected, 'candidate': candidate, 'strongest_affine': best_affine, 'strongest_control': strongest, 'conditions': conditions, 'passed': sum(bool(v) for v in conditions.values()), 'total': 9}


def canonical_parent_key(relative):
    require(isinstance(relative, str) and '\\' not in relative and all(part not in ('', '.', '..') for part in relative.split('/')), 'canonical relative parent payload required')
    path = Path(relative)
    require(not path.is_absolute() and path.parts[:len(PARENT_ROOT.parts)] == PARENT_ROOT.parts and len(path.parts) > len(PARENT_ROOT.parts), 'payload outside canonical parent root')
    return str(path.relative_to(PARENT_ROOT))


def parent_payload_key(relative, recorded_root):
    require(isinstance(recorded_root, str) and '\\' not in recorded_root and all(part not in ('.', '..') for part in recorded_root.split('/')), 'canonical recorded parent root required')
    root = Path(recorded_root)
    require(root.is_absolute() and root.parts[-len(PARENT_ROOT.parts):] == PARENT_ROOT.parts, 'recorded parent root has wrong suffix')
    return canonical_parent_key(relative)


def authenticate(study, process):
    terminal = read(process)
    require(terminal['observed_exit_code'] == 0 and terminal['command'] == COMMAND and type(terminal['tool_session_id']) is int and terminal['tool_session_id'] > 0 and terminal['observation'] == 'original Codex exec/write_stdin completion', 'original successful process required')
    require(isinstance(REGISTRATION_SHA256, str) and len(REGISTRATION_SHA256) == 64 and pin(ROOT/REG)['sha256'] == REGISTRATION_SHA256, 'unbound or changed registration')
    plan = read(ROOT/REG)
    require(study == ROOT/'output/fsm-residual-study-v1' and read(study/'protocol.json') == plan, 'study registration copy')
    require(set(plan['source_sha256']) == SOURCES and set(plan['parent_artifacts']) == PARENTS, 'source/parent roster')
    for name, sha in plan['source_sha256'].items():
        require(pin(ROOT/name)['sha256'] == sha == pin(study/'source'/name)['sha256'], 'source drift')
    require(pin(ROOT/plan['protocol_markdown_path'])['sha256'] == plan['protocol_markdown_sha256'], 'protocol drift')
    paths = {}
    for key, item in plan['parent_artifacts'].items():
        if key in PARENT_PAYLOADS:
            canonical_parent_key(item['path'])
        path = ROOT/item['path']; copied = study/'parent'/(key+path.suffix); paths[key] = copied
        require(pin(path)['sha256'] == item['sha256'] == pin(copied)['sha256'], 'parent payload drift')
    parent_audit, parent_process, parent_closure = (read(paths[k]) for k in ('audit', 'process', 'closure'))
    require(parent_audit['status'] == 'PASS' and parent_audit['agreement'] is True and parent_process['observed_exit_code'] == 0 and parent_closure['status'] == 'completed', 'closed parent evidence')
    require(parent_audit['inputs']['registration'] == pin(paths['registration']) and parent_audit['inputs']['process']['sha256'] == pin(paths['process'])['sha256'], 'parent audit input joins')
    for key in PARENT_PAYLOADS:
        relative = parent_payload_key(plan['parent_artifacts'][key]['path'], parent_audit['inputs']['study'])
        require(parent_audit['inputs']['files'][relative] == pin(paths[key]), 'parent audited payload join')
    require(parent_process['summary_sha256'] == parent_closure['summary_sha256'] == pin(paths['summary'])['sha256'], 'parent terminal summary join')
    closure = read(study/'closure.json')
    require(closure['status'] == 'completed' and closure['source_sha256'] == plan['source_sha256'] and closure['summary_sha256'] == terminal['summary_sha256'] == pin(study/'summary.json')['sha256'], 'original closure join')
    require(pin(Path(common.__file__))['sha256'] == COMMON_SHA256, 'independent helper drift')
    return plan, paths, terminal, closure


def parameter_shapes(kind):
    return ({'residual.weight': (3, 195), 'residual.bias': (3,)} if kind == 'affine' else {'residual.0.weight': (24, 195), 'residual.0.bias': (24,), 'residual.2.weight': (3, 24), 'residual.2.bias': (3,)})


def validate_checkpoint(initial, final, receipt, coefficients, paired, cfg):
    architecture, rate = RECIPES[receipt['family']]; kind, mode = architecture.split('_', 1); shapes = parameter_shapes(kind)
    n, complete = receipt['accepted_updates'], receipt['status'] == 'complete'
    require(receipt['architecture'] == architecture and receipt['learning_rate'] == rate and type(n) is int and 0 <= n <= cfg['updates'], 'fit recipe/count')
    require(set(final) == {'model', 'optimizer', 'accepted_updates'} and final['accepted_updates'] == n, 'final checkpoint schema')
    for state in (initial, final['model']):
        require(set(state) == {'coefficients', *shapes}, 'checkpoint tensor roster')
        require(state['coefficients'].dtype == torch.float64 and torch.equal(state['coefficients'], torch.from_numpy(coefficients)), 'frozen backbone tensor changed')
        for key, shape in shapes.items():
            require(state[key].dtype == torch.float64 and tuple(state[key].shape) == shape and (state is not initial and not complete or bool(torch.isfinite(state[key]).all())), 'residual tensor')
    heads = ('residual.weight', 'residual.bias') if kind == 'affine' else ('residual.2.weight', 'residual.2.bias')
    require(all(torch.count_nonzero(initial[key]).item() == 0 for key in heads), 'initial residual is nonzero')
    signature = {key: hashlib.sha256(initial[key].numpy().tobytes()).hexdigest() for key in shapes}
    require(paired.setdefault((kind, receipt['seed']), signature) == signature, 'initial modes/rates not paired')
    require(receipt['backbone_unchanged'] is True, 'backbone invariant receipt')
    identity = receipt['initial_identity']
    if identity is not None:
        require(type(identity['passed']) is bool and np.isfinite(identity['max_abs_difference']) and identity['max_abs_difference'] >= 0 and identity['atol'] == cfg['identity_atol'] and identity['rtol'] == cfg['identity_rtol'], 'initial identity diagnostic')
    if complete:
        require(n == cfg['updates'] and identity and identity['passed'] and receipt['error'] is None and receipt['fit_seconds'] <= cfg['fit_timeout_seconds'], 'completed fit qualification')
    else:
        require(receipt['status'] == 'failed' and isinstance(receipt['error'], dict), 'failed fit receipt')
    spec = receipt['model_spec']; count = sum(int(np.prod(s)) for s in shapes.values())
    require(spec['mode'] == mode and spec['residual_kind'] == kind and spec['parameter_count'] == count and spec['trainable_parameter_count'] == count and spec['parameter_bytes'] == count*8 and spec['buffer_bytes'] == 4704 and spec['state_scalars'] == 192 and spec['numeric_metadata_bytes'] == (16 if kind == 'affine' else 24), 'model storage/spec')
    optimizer = final['optimizer']; require(set(optimizer) == {'state', 'param_groups'} and len(optimizer['param_groups']) == 1, 'Adam schema')
    group = optimizer['param_groups'][0]
    require(group['params'] == list(range(len(shapes))) and group['lr'] == rate and group['weight_decay'] == 0 and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8 and set(optimizer['state']) <= set(group['params']), 'Adam roster/options')
    require(not complete or len(optimizer['state']) == len(shapes), 'complete Adam coverage')
    for index, slots in optimizer['state'].items():
        require(set(slots) == {'step', 'exp_avg', 'exp_avg_sq'} and float(slots['step']) in ({n} if complete else {n, n+1}), 'Adam update count')
        for key in ('exp_avg', 'exp_avg_sq'):
            require(slots[key].dtype == torch.float64 and tuple(slots[key].shape) == list(shapes.values())[index] and (not complete or bool(torch.isfinite(slots[key]).all())), 'Adam moments')
    return count*8+4704, 16 if kind == 'affine' else 24


def audit(study, process):
    study, process = Path(study).resolve(), Path(process).resolve()
    plan, paths, terminal, closure = authenticate(study, process); cfg = plan['experiment']; before = common.inventory(study)
    require((cfg['order'], cfg['hidden_width'], cfg['context'], cfg['horizon'], cfg['batch'], cfg['updates'], cfg['evaluation_stride'], cfg['gradient_clip']) == (32, 24, 100, 128, 16, 2048, 256, 1.) and tuple(cfg['architectures']) == ARCHITECTURES and tuple(cfg['learning_rates']) == RATES and tuple(cfg['seeds']) == SEEDS, 'registered geometry')
    require(cfg['fit_timeout_seconds'] == 240 and cfg['identity_atol'] == cfg['identity_rtol'] == 1e-9, 'registered cap/identity tolerance')
    coefficients = arrays(paths['coefficients'], ('coefficients',))['coefficients']
    require(coefficients.dtype == np.float64 and coefficients.shape == (3, 196) and np.isfinite(coefficients).all(), 'frozen coefficient geometry')
    parent_fit = read(paths['fit_receipt']); require(parent_fit['status'] == 'complete' and parent_fit['order'] == 32 and parent_fit['alpha'] == 1e-6 and parent_fit['model_spec']['fit_rows'] == 97920 and parent_fit['model_spec']['fit_record_ids'] == list(FIT), 'parent FIT recipe')
    norm = arrays(paths['normalizer'], ('u_mean', 'u_scale', 'y_mean', 'y_scale'))
    require(all(v.dtype == np.float64 and v.shape == (3,) and np.isfinite(v).all() for v in norm.values()) and all((norm[k] > 0).all() for k in ('u_scale', 'y_scale')), 'frozen FIT normalizer')
    expected = {'protocol.json', 'admission.json', 'fits.json', 'fits-closed.json', 'evaluations.json', 'summary.json', 'closure.json'} | {'source/'+s for s in SOURCES} | {'parent/'+p.name for p in paths.values()}
    admission = read(study/'admission.json')
    require(admission['source_sha256'] == plan['data_sha256'] and set(admission['decoded_keys']) == {'u_100mV_train', 'y_100mV_train', 'u_200mV_train', 'y_200mV_train'} and admission['fit_ids'] == list(FIT) and admission['dev_ids'] == list(DEV) and admission['fit_samples'] == 98304, 'data admission')
    for seed in SEEDS:
        name = f'schedule-{seed}.npz'; expected.add(name); schedule = arrays(study/name, ('record', 'start')); rng = np.random.default_rng(100000+seed)
        for key, bound in (('record', 12), ('start', 7965)):
            require(schedule[key].dtype == np.int64 and np.array_equal(schedule[key], rng.integers(0, bound, (2048, 16), dtype=np.int64)), 'paired schedule')
    fits, evaluations = read(study/'fits.json'), read(study/'evaluations.json')
    recipes = tuple(RECIPES); ordered = [(f, s) for i, s in enumerate(SEEDS) for f in recipes[i:]+recipes[:i]]
    require([(f['family'], f['seed']) for f in fits] == ordered and [(e['family'], e['seed']) for e in evaluations] == [*ordered, ('native_varx', None)], 'complete ordered36/37 roster')
    barrier = read(study/'fits-closed.json'); require(barrier['fits'] == 36 and barrier['complete'] == sum(f['status'] == 'complete' for f in fits) and np.isfinite(barrier['elapsed_seconds']) and barrier['elapsed_seconds'] > 0 and barrier['barrier'] == 'all declared fits closed before DEV predictions, scoring or rate selection', 'fit barrier ledger')
    paired, accounting = {}, {'native_varx': (4704, 24)}
    for fit in fits:
        stem = f"{fit['family']}-{fit['seed']}"; folder = study/stem
        expected.update(stem+'/'+n for n in ('initial.pt', 'final.pt', 'training.jsonl', 'receipt.json'))
        require(read(folder/'receipt.json') == fit and np.isfinite(fit['fit_seconds']) and fit['fit_seconds'] > 0 and fit['initial_sha256'] == pin(folder/'initial.pt')['sha256'] and fit['final_sha256'] == pin(folder/'final.pt')['sha256'], 'fit receipt/pins')
        initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=True); final = torch.load(folder/'final.pt', map_location='cpu', weights_only=True)
        accounting[stem] = validate_checkpoint(initial, final, fit, coefficients, paired, cfg)
        trace = [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        require(len(trace) == fit['accepted_updates'] and [r['update'] for r in trace] == list(range(1, len(trace)+1)), 'accepted trace coverage')
        require(all(set(r) == {'update', 'loss', 'gradient_norm', 'elapsed_seconds'} and all(np.isfinite(r[k]) and r[k] >= 0 for k in ('loss', 'gradient_norm', 'elapsed_seconds')) for r in trace) and all(a['elapsed_seconds'] <= b['elapsed_seconds'] for a, b in pairwise(trace)) and (not trace or trace[-1]['elapsed_seconds'] <= fit['fit_seconds']), 'trace finite/schema/time')
    targets, banks, timing_samples = {}, 0, 0
    for e in evaluations:
        f, seed = e['family'], e['seed']; stem = f if seed is None else f'{f}-{seed}'; folder = study/stem/'evaluation'; expected.add(stem+'/evaluation/receipt.json')
        require(read(folder/'receipt.json') == e and len(e['rows']) == 12 and {r['record_id'] for r in e['rows']} == set(DEV), 'record attempts')
        fit = next((x for x in fits if (x['family'], x['seed']) == (f, seed)), None)
        if fit and fit['status'] == 'failed':
            require(e['model_spec'] == fit['model_spec'] and e['persistent_numeric_bytes'] is None and e['request_ms'] == [] and e['median_request_ms'] is None and e['timing_error'] == 'fit_failed' and all(r == {'record_id': r['record_id'], 'status': 'not_run', 'error': 'fit_failed'} for r in e['rows']), 'failed fit evaluation preserved'); continue
        tensors, metadata = accounting[stem]
        require(e['tensor_bytes'] == tensors and e['state_bytes'] == 1536 and e['normalizer_bytes'] == 96 and e['scalar_metadata_bytes'] == metadata and e['persistent_numeric_bytes'] == tensors+1536+96+metadata, 'complete retained storage')
        require(e['model_spec'] == (fit['model_spec'] if fit else parent_fit['model_spec']), 'evaluation model spec')
        require(np.isfinite(e['scoring_seconds']) and e['scoring_seconds'] > 0, 'scoring timing')
        costs = e['request_ms']; require(len(costs) <= 24 and all(type(v) in (float, int) and np.isfinite(v) and v > 0 for v in costs), 'finite timing observations'); timing_samples += len(costs)
        if e['timing_error'] is None:
            require(len(costs) == 24, 'complete timing coverage'); close(e['median_request_ms'], float(np.median(costs)))
        else:
            require(isinstance(e['timing_error'], str) and e['median_request_ms'] is None, 'timing failure preserved')
        for row in e['rows']:
            name = stem+'/evaluation/'+row['record_id']+'.npz'; present = (study/name).exists()
            require(row['status'] in ('complete', 'failed') and (present or row['status'] == 'failed'), 'forecast status/evidence')
            if row['status'] == 'failed':
                require(isinstance(row.get('error'), str), 'forecast failure reason')
            if not present:
                continue
            expected.add(name); bank = arrays(study/name, ('prediction', 'target', 'starts')); banks += 1
            require(bank['starts'].dtype == np.int64 and np.array_equal(bank['starts'], np.arange(0, 7937, 256)), 'aligned32starts')
            value = record_metrics(bank['prediction'], bank['target'], norm['y_scale']); target_pin = hashlib.sha256(bank['target'].tobytes()).hexdigest()
            require(targets.setdefault(row['record_id'], target_pin) == target_pin, 'common targets differ')
            if row['status'] == 'complete':
                close(row, {'record_id': row['record_id'], 'status': 'complete', **value}); row.update(value)
    summary = read(study/'summary.json'); result = decisions(evaluations, fits, cfg); close({k: summary[k] for k in result}, result)
    require(closure['result'] == result['status'] and np.isfinite(summary['elapsed_seconds']) and summary['elapsed_seconds'] >= barrier['elapsed_seconds'], 'terminal result/elapsed')
    require(set(before) == expected and common.inventory(study) == before, 'complete unchanged file inventory')
    require(authenticate(study, process)[0] == plan and read(process) == terminal, 'post-audit admission')
    return {'status': 'PASS', 'agreement': True, 'scientific_status': result['status'], 'results': result, 'source_pins': plan['source_sha256'], 'inputs': {'study': str(study), 'process': {'path': str(process), **pin(process)}, 'registration': pin(ROOT/REG), 'files': before}, 'counts': {'fits': 36, 'evaluations': 37, 'accepted_updates': sum(f['accepted_updates'] for f in fits), 'record_rows': 444, 'saved_prediction_banks': banks, 'timing_samples': timing_samples, 'initial_pair_groups': len(paired)}, 'scope': 'Saved-output metrics and grid/gate arithmetic; checkpoint tensors, frozen backbone, zero heads, Adam counters, storage and sampling. No model calls, raw measurement decode, optimizer replay or independent timing. Initial forecast parity and temporal fit barrier are qualified-source/receipt attestations, not replayed. Exposed DEV rate selection is not confirmation.', 'auditor': pin(__file__), 'independent_helper': pin(Path(common.__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--study', required=True); parser.add_argument('--process', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); destination = Path(args.output); require(not destination.exists(), 'exclusive audit output')
    result = audit(args.study, args.process); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
