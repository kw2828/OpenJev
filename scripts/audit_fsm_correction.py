"""Independent saved-output FSM audit; no model construction, inference or fitting."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REG = 'research/fsm-correction-registration.json'
PIN = '6369c3121380bd98ef109c4b6aea2c1f7c1b9c9110fdd7f0b2327a41845f1af1'
NEURAL = ('dense', 'selective', 'rewired', 'fixed0', 'fixed1', 'fixed2', 'autoregressive')
SEEDS = (9101, 9102, 9103)
LINEAR = {f'varx{p}-ridge{a:g}': (p, a) for p in (8, 16, 32) for a in (1e-6, .001, .1)}
FAMILIES = (*NEURAL, *LINEAR)
DEV = tuple(f'{a}mV-realization-{r}-period-{p}' for a in (100, 200) for r in (3, 4, 5) for p in (0, 1))
FIT = tuple(f'{a}mV-realization-{r}-period-{p}' for a in (100, 200) for r in (0, 1, 2) for p in (0, 1))
ROSTER = {(f, s) for f in NEURAL for s in SEEDS} | {(f, None) for f in LINEAR}
COMMAND = ['.venv/bin/python', '-u', '-m', 'openjev.research.fsm_study', '--protocol', REG, '--data', 'output/fsm-correction-engineering-v1/admission-01/combined_data.npz', '--output', 'output/fsm-correction-study-v1']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def pin(path):
    p = Path(path)
    return {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def inventory(folder):
    require(not any(p.is_symlink() for p in folder.rglob('*')), 'symlinks prohibited')
    return {str(p.relative_to(folder)): pin(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def close(actual, expected):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'mapping roster')
        for key in expected:
            close(actual[key], expected[key])
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'list roster')
        for a, b in zip(actual, expected, strict=True):
            close(a, b)
    elif type(expected) is float:
        require(type(actual) in (int, float) and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-12, atol=1e-12), 'scalar disagreement')
    else:
        require(type(actual) is type(expected) and actual == expected, 'identity disagreement')


def arrays(path, keys):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == set(keys), 'NPZ keys')
        return {k: data[k].copy(order='K') for k in keys}


def metrics(prediction, target):
    require(all(a.dtype == np.float64 and a.shape == (32, 128, 3) and np.isfinite(a).all() for a in (prediction, target)), 'finite float64 forecast geometry')
    with np.errstate(over='ignore', invalid='ignore'):
        squared = (prediction-target)**2
        mse, channels = float(squared.mean()), np.sqrt(squared.mean(axis=(0, 1)))
    require(np.isfinite(mse) and np.isfinite(channels).all(), 'nonfinite reconstructed metric')
    return {'rmse': float(np.sqrt(mse)), 'per_channel_rmse': channels.tolist(), 'mse': mse, 'requests': 32, 'horizon': 128}


def decisions(evaluations, fits):
    groups = {f: [e for e in evaluations if e['family'] == f] for f in FAMILIES}
    families = {}
    for f, members in groups.items():
        seeds = set(SEEDS) if f in NEURAL else {None}
        good = len(members) == len(seeds) and {e['seed'] for e in members} == seeds and all(e['timing_error'] is None and len(e['rows']) == 12 and {r['record_id'] for r in e['rows']} == set(DEV) and all(r['status'] == 'complete' for r in e['rows']) for e in members)
        families[f] = {'eligible': good}
        if good:
            families[f].update(mean_rmse=float(np.mean([r['rmse'] for e in members for r in e['rows']])), mean_seed_median_ms=float(np.mean([e['median_request_ms'] for e in members])), persistent_numeric_bytes=max(e['persistent_numeric_bytes'] for e in members))
    controls = [f for f in FAMILIES if f != 'selective' and families[f]['eligible']]
    best = min(controls, key=lambda f: families[f]['mean_rmse']) if controls else None
    conditions = {'all_declared_fits_complete': len(fits) == 30 and {(f['family'], f.get('seed')) for f in fits} == ROSTER and all(f['status'] == 'complete' and (f['family'] not in NEURAL or f['accepted_updates'] == 1024) for f in fits), 'all_declared_evaluations_complete': len(evaluations) == 30 and {(e['family'], e['seed']) for e in evaluations} == ROSTER and all(v['eligible'] for v in families.values()), 'candidate_eligible': families['selective']['eligible'], 'five_percent_below_strongest_control': False, 'latency_within_ten_percent_dense': False, 'storage_within_ten_percent_dense': False, 'no_record_over_two_percent_dense': False, 'every_seed_beats_dense': False, 'two_blocks_at_least_five_percent_each_seed': False}
    if families['selective']['eligible'] and families['dense']['eligible'] and best:
        s, d = families['selective'], families['dense']
        conditions['five_percent_below_strongest_control'] = s['mean_rmse'] <= .95*families[best]['mean_rmse']
        conditions['latency_within_ten_percent_dense'] = s['mean_seed_median_ms'] <= 1.1*d['mean_seed_median_ms']
        conditions['storage_within_ten_percent_dense'] = s['persistent_numeric_bytes'] <= 1.1*d['persistent_numeric_bytes']
        by_seed = {f: {e['seed']: e for e in groups[f]} for f in ('selective', 'dense')}
        def mean(f, record=None, seed=None):
            return float(np.mean([r['rmse'] for e in groups[f] if seed is None or e['seed'] == seed for r in e['rows'] if record is None or r['record_id'] == record]))
        conditions['no_record_over_two_percent_dense'] = all(mean('selective', r) <= 1.02*mean('dense', r) for r in DEV)
        conditions['every_seed_beats_dense'] = all(mean('selective', seed=s) < mean('dense', seed=s) for s in SEEDS)
        conditions['two_blocks_at_least_five_percent_each_seed'] = all(sum(e['routing_nonzero_counts']) > 0 and sum(v >= .05*sum(e['routing_nonzero_counts']) for v in e['routing_nonzero_counts']) >= 2 for e in by_seed['selective'].values())
    return {'status': 'DEVELOPMENT_PASS' if all(conditions.values()) else 'DEVELOPMENT_FAIL', 'families': families, 'strongest_control': best, 'conditions': conditions, 'passed': sum(conditions.values()), 'total': 9}


def checkpoint(folder, receipt, initial_by_seed):
    f, seed, n = receipt['family'], receipt['seed'], receipt['accepted_updates']
    require(type(n) is int and 0 <= n <= 1024, 'update count')
    require(receipt['initial_sha256'] == pin(folder/'initial.pt')['sha256'] and receipt['final_sha256'] == pin(folder/'final.pt')['sha256'], 'checkpoint pins')
    initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=True)
    final = torch.load(folder/'final.pt', map_location='cpu', weights_only=True)
    shapes = {'transition.weight_ih': (216, 6 if f == 'autoregressive' else 3), 'transition.weight_hh': (216, 72), 'transition.bias_ih': (216,), 'transition.bias_hh': (216,), 'observation.weight': (3, 72), 'observation.bias': (3,), 'initializer.weight': (72, 3), 'initializer.bias': (72,)}
    for h in (1, 8, 32):
        shapes.update({f'auxiliary.{h}.0.weight': (96, 24+3*h), f'auxiliary.{h}.0.bias': (96,), f'auxiliary.{h}.2.weight': (3, 96), f'auxiliary.{h}.2.bias': (3,)})
    require(set(final) == {'model', 'optimizer', 'accepted_updates'} and final['accepted_updates'] == n, 'final checkpoint schema')
    for state in (initial, final['model']):
        require(set(state) == set(shapes), 'parameter roster')
        for k, shape in shapes.items():
            require(state[k].dtype == torch.float32 and tuple(state[k].shape) == shape and (state is not initial and receipt['status'] == 'failed' or bool(torch.isfinite(state[k]).all())), 'parameter tensor')
    if f != 'autoregressive':
        signature = {k: hashlib.sha256(t.numpy().tobytes()).hexdigest() for k, t in initial.items()}
        require(initial_by_seed.setdefault(seed, signature) == signature, 'unpaired correction initialization')
    trace = [json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
    require(len(trace) == n and [r['update'] for r in trace] == list(range(1, n+1)), 'training trace coverage')
    require(all(all(np.isfinite(r[k]) and r[k] >= 0 for k in ('loss', 'forecast_loss', 'auxiliary_loss', 'gradient_norm', 'elapsed_seconds')) for r in trace), 'training trace finite')
    require(all(set(r) == {'update', 'loss', 'forecast_loss', 'auxiliary_loss', 'gradient_norm', 'elapsed_seconds'} for r in trace) and all(a['elapsed_seconds'] <= b['elapsed_seconds'] for a, b in itertools.pairwise(trace)) and (not trace or trace[-1]['elapsed_seconds'] <= receipt['fit_seconds']), 'trace schema/time')
    optimizer = final['optimizer']; group = optimizer['param_groups']
    require(len(group) == 1 and group[0]['params'] == list(range(len(shapes))) and group[0]['lr'] == .001 and set(optimizer['state']) <= set(group[0]['params']), 'Adam roster')
    if receipt['status'] == 'complete':
        require(n == 1024 and receipt['error'] is None and receipt['fit_seconds'] <= 180 and len(optimizer['state']) == len(shapes), 'completed fit budget')
    else:
        require(isinstance(receipt['error'], dict), 'failed fit evidence')
    for index, slots in optimizer['state'].items():
        require(set(slots) == {'step', 'exp_avg', 'exp_avg_sq'} and float(slots['step']) in ({n} if receipt['status'] == 'complete' else {n, n+1}), 'Adam steps')
        for k in ('exp_avg', 'exp_avg_sq'):
            require(slots[k].dtype == torch.float32 and tuple(slots[k].shape) == list(shapes.values())[index], 'Adam slot shape')
            require(receipt['status'] == 'failed' or bool(torch.isfinite(slots[k]).all()), 'completed Adam finite')
    return sum(t.numel()*t.element_size() for t in initial.values())


def audit(study, process):
    study, process = Path(study).resolve(), Path(process).resolve()
    terminal = read(process); registration = read(ROOT/REG)
    require(terminal['observed_exit_code'] == 0 and terminal['tool_session_id'] == 70537 and terminal['command'] == COMMAND and terminal['observation'] == 'original Codex exec/write_stdin completion', 'original successful process required')
    require(pin(ROOT/REG)['sha256'] == PIN and read(study/'protocol.json') == registration and study == ROOT/'output/fsm-correction-study-v1', 'frozen registration/study')
    sources = registration['source_sha256']
    for name, value in sources.items():
        require(pin(ROOT/name)['sha256'] == value == pin(study/'source'/name)['sha256'], 'source drift')
    require(pin(ROOT/'research/fsm-correction-protocol.md')['sha256'] == registration['protocol_markdown_sha256'], 'protocol drift')
    closure = read(study/'closure.json')
    require(closure['status'] == 'completed' and closure['source_sha256'] == sources and closure['summary_sha256'] == terminal['summary_sha256'] == pin(study/'summary.json')['sha256'], 'original closure join')
    before = inventory(study); expected = {'protocol.json', 'normalizer.npz', 'admission.json', 'fits.json', 'evaluations.json', 'summary.json', 'closure.json'} | {'source/'+s for s in sources}
    norm = arrays(study/'normalizer.npz', ('u_mean', 'u_scale', 'y_mean', 'y_scale'))
    require(all(x.dtype == np.float64 and x.shape == (3,) and np.isfinite(x).all() for x in norm.values()) and all((norm[k] > 0).all() for k in ('u_scale', 'y_scale')), 'normalization geometry')
    a = read(study/'admission.json')
    require(set(a['decoded_keys']) == {'u_100mV_train', 'y_100mV_train', 'u_200mV_train', 'y_200mV_train'} and a['source_sha256'] == registration['data_sha256'] and a['fit_ids'] == list(FIT) and a['dev_ids'] == list(DEV) and a['fit_samples'] == 98304, 'data admission')
    for seed in SEEDS:
        name = f'schedule-{seed}.npz'; expected.add(name); schedule = arrays(study/name, ('record', 'start')); rng = np.random.default_rng(100000+seed)
        for key, bound in (('record', 12), ('start', 7965)):
            require(schedule[key].dtype == np.int64 and np.array_equal(schedule[key], rng.integers(0, bound, (1024, 8), dtype=np.int64)), 'paired sampling schedule')
    fits, evaluations = read(study/'fits.json'), read(study/'evaluations.json')
    require(len(fits) == 30 and {(f['family'], f.get('seed')) for f in fits} == ROSTER, 'complete fit roster')
    ordered = [(f, seed) for i, seed in enumerate(SEEDS) for f in NEURAL[i:]+NEURAL[:i]]+[(f, None) for f in LINEAR]
    require([(f['family'], f.get('seed')) for f in fits] == ordered, 'original fit order')
    require(len(evaluations) == sum(f['status'] == 'complete' for f in fits) and {(e['family'], e['seed']) for e in evaluations} == {(f['family'], f.get('seed')) for f in fits if f['status'] == 'complete'}, 'evaluation attempt roster')
    require([(e['family'], e['seed']) for e in evaluations] == [(f['family'], f.get('seed')) for f in fits if f['status'] == 'complete'], 'original evaluation order')
    targets, initials, banks, rows, timing_samples = {}, {}, 0, 0, 0
    for fit in fits:
        f, seed = fit['family'], fit.get('seed'); stem = f if seed is None else f'{f}-{seed}'; folder = study/stem
        require(read(folder/'receipt.json') == fit and fit['status'] in ('complete', 'failed') and np.isfinite(fit['fit_seconds']) and fit['fit_seconds'] > 0, 'fit receipt')
        expected.add(stem+'/receipt.json')
        if seed is not None:
            expected.update(stem+'/'+s for s in ('initial.pt', 'final.pt', 'training.jsonl')); parameter_bytes = checkpoint(folder, fit, initials); state, metadata = (75 if f == 'autoregressive' else 72), 0
        else:
            order, alpha = LINEAR[f]; require((fit['order'], fit['alpha']) == (order, alpha), 'linear recipe'); state, metadata, parameter_bytes = 6*order, 24, 3*(6*order+4)*8
            if (folder/'model.npz').exists():
                expected.add(stem+'/model.npz'); coefficients = arrays(folder/'model.npz', ('coefficients',))['coefficients']
                require(coefficients.dtype == np.float64 and coefficients.shape == (3, 6*order+4) and np.isfinite(coefficients).all(), 'linear coefficient schema')
            require(fit['status'] == 'failed' or (folder/'model.npz').exists(), 'linear fit payload')
        if fit['status'] == 'failed':
            require(fit.get('error'), 'failure reason retained'); continue
        e = next(e for e in evaluations if (e['family'], e['seed']) == (f, seed)); expected.add(stem+'/evaluation/receipt.json')
        require(read(folder/'evaluation/receipt.json') == e and len(e['rows']) == 12 and {r['record_id'] for r in e['rows']} == set(DEV), 'record attempt roster')
        spec = e['model_spec']; require(spec == fit['model_spec'] and spec['parameter_bytes'] == parameter_bytes and spec['parameter_count'] == parameter_bytes//(4 if seed else 8) and spec['state_scalars'] == state, 'actual parameter/state accounting')
        if seed is None:
            require(spec['fit_record_ids'] == list(FIT) and spec['fit_rows'] == 12*(8192-order) and spec['scalar_metadata_bytes'] == 24 and spec['retained_numeric_bytes'] == parameter_bytes+24, 'linear FIT/state metadata')
        require(e['persistent_numeric_bytes'] == parameter_bytes+metadata+state*(4 if seed else 8)+96, 'full persistent storage')
        costs = e['request_ms']; require(len(costs) <= 24 and all(type(x) in (float, int) and np.isfinite(x) and x > 0 for x in costs), 'timing samples'); timing_samples += len(costs)
        if e['timing_error'] is None:
            require(len(costs) == 24, 'complete timing coverage'); close(e['median_request_ms'], float(np.median(costs)))
        else:
            require(isinstance(e['timing_error'], str) and e['median_request_ms'] is None, 'failed timing retained')
        counts = e['routing_nonzero_counts']
        require((f != 'selective' and counts is None) or (f == 'selective' and isinstance(counts, list) and len(counts) == 3 and all(type(v) is int and v >= 0 for v in counts) and sum(counts) <= 12*32*99), 'routing count geometry')
        for row in e['rows']:
            rows += 1; name = stem+'/evaluation/'+row['record_id']+'.npz'; exists = (study/name).exists()
            require(row['status'] in ('complete', 'failed') and (exists or row['status'] == 'failed'), 'forecast evidence')
            if row['status'] == 'failed':
                require(isinstance(row.get('error'), str), 'failed record reason')
            if not exists:
                continue
            expected.add(name); data = arrays(study/name, ('prediction', 'target', 'starts')); banks += 1
            require(data['starts'].dtype == np.int64 and np.array_equal(data['starts'], np.arange(0, 7937, 256)), '32 aligned starts')
            recomputed = metrics(data['prediction'], data['target']); signature = hashlib.sha256(data['target'].tobytes()).hexdigest()
            require(targets.setdefault(row['record_id'], signature) == signature, 'targets differ between models')
            if row['status'] == 'complete':
                close(row, dict(record_id=row['record_id'], status='complete', **recomputed)); row.update(recomputed)
    summary = read(study/'summary.json'); result = decisions(evaluations, fits)
    close({k: summary[k] for k in result}, result)
    require(closure['result'] == result['status'] and np.isfinite(summary['elapsed_seconds']) and summary['elapsed_seconds'] > 0, 'terminal result/time')
    require(set(before) == expected and inventory(study) == before and read(process) == terminal, 'complete inventory/unchanged inputs')
    require(all(pin(ROOT/n)['sha256'] == h for n, h in sources.items()) and pin(ROOT/REG)['sha256'] == PIN, 'post-audit source pins')
    return {'status': 'PASS', 'agreement': True, 'scientific_status': result['status'], 'results': result, 'source_pins': sources, 'inputs': {'study': str(study), 'process': dict(path=str(process), **pin(process)), 'registration': pin(ROOT/REG), 'files': before}, 'counts': {'fits': 30, 'neural_checkpoints': 42, 'accepted_updates': sum(f.get('accepted_updates', 0) for f in fits), 'saved_prediction_banks': banks, 'record_rows': rows, 'timing_samples': timing_samples, 'schedules': 3}, 'scope': 'Saved-output metrics, common targets, tensor/schema/storage, sampling and nine-condition arithmetic only. No model calls, fitting, raw-data decode or optimizer replay. Routing counts and timings are attested, not independently replayed. Failed attempts remain failures. Producer elapsed excludes startup/admission; no total process wall-time measurement.', 'auditor': pin(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--study', required=True); parser.add_argument('--process', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); destination = Path(args.output); require(not destination.exists(), 'exclusive audit output required')
    result = audit(args.study, args.process); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
