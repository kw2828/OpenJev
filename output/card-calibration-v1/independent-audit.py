"""Independent saved-array arithmetic for the completed card calibration study.

No fitter, reporter, model, tracker, environment, or RNG is imported. This checks
authenticated saved rewards and beliefs, not whether a model produced the beliefs
or whether the native simulator was correct. The frozen report separately checks
public policy replay and scalar-fit optimality. No scalar is fitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODES = ('delta', 'gated_delta', 'kalman', 'innovation_local', 'innovation_matched', 'gru')
NAMES = tuple(f'{m}-pair{i}' for m in MODES for i in range(3))
POLICIES = ('baseline', 'temperature', 'hard')
REFERENCES = ('exact', 'last32')
LAYOUT_PREFIX = b'card-layout-int64le-v1\0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def member(folder, name):
    require(type(name) is str and not Path(name).is_absolute(), 'Relative member required')
    root, path = Path(folder).resolve(), (Path(folder) / name).resolve()
    require(path.is_relative_to(root) and path != root, 'Member escapes its tree')
    return path


def same(actual, expected, label):
    """Tolerances apply only to independent numeric reductions, never gate margins."""
    if isinstance(actual, dict):
        require(isinstance(expected, dict), label + ': expected mapping')
        for key, value in actual.items():
            require(key in expected, label + ': missing ' + key)
            same(value, expected[key], label + '/' + key)
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), label + ': list coverage')
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            same(a, b, label + '/' + str(i))
    elif type(actual) is float:
        require(type(expected) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11), label + ': numeric mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact mismatch')


def payload_tree(folder, digest, names):
    folder = Path(folder)
    require(sha(folder / 'completed.json') == digest, 'External completion hash differs')
    done = read(folder / 'completed.json')
    require(done['status'] == 'complete' and set(done['files']) == names, 'Incomplete/wrong member set')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    require(actual == names | {'completed.json'}, 'Unexpected/missing files, including failure markers')
    for name, spec in done['files'].items():
        path = member(folder, name)
        require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Corrupt member: ' + name)
    return done


def evaluation_members():
    result = {'started.json', 'all-fits-ready.json'}
    for name in (*NAMES, *REFERENCES):
        for policy in POLICIES if name in NAMES else ('baseline',):
            stem = f'controllers/{name}/{policy}'
            result.add(stem + '/completed.json')
            result.update(f'{stem}/episodes/{i:03d}.{ext}' for i in range(64) for ext in ('npz', 'json'))
    return result


def score_episode(raw, observations, policy, beta):
    """Vectorized loss cells followed by explicit boundary/episode reductions."""
    require(raw.dtype == np.float64 and raw.ndim == 3 and raw.shape[1:] == (52, 13)
            and 1 <= len(raw) <= 104 and np.isfinite(raw).all()
            and np.all((raw >= 0) & (raw <= 1))
            and np.all(np.abs(raw.sum(axis=2) - 1.) <= 1e-5), 'Invalid saved probabilities')
    t = len(raw)
    require(observations.dtype == np.int64 and observations.shape == (t + 1, 52)
            and np.all((observations >= 0) & (observations <= 13))
            and np.all(observations[0] == 13), 'Invalid public frames')
    require(policy in POLICIES and type(beta) in (float, int) and .05 <= beta <= 20, 'Invalid transform')
    known, last_seen = np.full(52, -1, np.int64), np.full(52, -1, np.int64)
    targets, ages = np.full((t, 52), -1, np.int64), np.full((t, 52), -1, np.int64)
    for clock, frame in enumerate(observations):
        visible = frame < 13
        require(np.all((known[visible] == -1) | (known[visible] == frame[visible])), 'Public rank changed')
        known[visible], last_seen[visible] = frame[visible], clock
        if clock < t:
            eligible = (known >= 0) & ~visible
            targets[clock, eligible] = known[eligible]
            ages[clock, eligible] = clock - last_seen[eligible]
    logs = np.full(raw.shape, -np.inf, np.float64)
    np.log(raw, out=logs, where=raw > 0)
    logs -= np.max(logs, axis=2, keepdims=True)
    scale = beta if policy == 'temperature' else 1.
    z = logs * scale
    q = np.exp(z)
    q /= q.sum(axis=2, keepdims=True)
    if policy == 'baseline' or (policy == 'temperature' and beta == 1.):
        q = raw / raw.sum(axis=2, keepdims=True)
    if policy == 'hard':
        q = np.zeros_like(raw)
        np.put_along_axis(q, raw.argmax(axis=2)[..., None], 1., axis=2)
    clipped_target = np.maximum(targets, 0)
    target_p = np.take_along_axis(q, clipped_target[..., None], axis=2)[..., 0]
    if policy == 'hard':
        loss = np.where(target_p == 1., 0., np.inf)
    else:
        loss = np.log(np.exp(z).sum(axis=2)) - np.take_along_axis(z, clipped_target[..., None], axis=2)[..., 0]
    # Subtracting one at the true class avoids cancellation at nearly-one beliefs.
    error = q.copy()
    np.put_along_axis(error, clipped_target[..., None], target_p[..., None] - 1., axis=2)
    brier = np.square(error).sum(axis=2)
    correct = q.argmax(axis=2) == targets
    result = {}
    for group, mask in (('all', targets >= 0), ('age_gt32', ages > 32)):
        cells = int(mask.sum())
        counts = mask.sum(axis=1)
        active = counts > 0
        infinite = int(np.count_nonzero(mask & ~np.isfinite(loss)))
        def reduce(value, mask=mask, active=active, counts=counts):
            boundary = np.where(mask, value, 0.).sum(axis=1)[active] / counts[active]
            return float(math.fsum(boundary) / len(boundary)) if len(boundary) else None
        result[group] = {'query_boundaries': int(active.sum()), 'query_cards': cells,
            'nll': None if infinite else reduce(loss), 'nll_is_infinite': bool(infinite),
            'infinite_nll_queries': infinite,
            'probability_underflow_queries': int(np.count_nonzero(mask & (target_p == 0) & np.isfinite(loss))),
            'brier': reduce(brier), 'accuracy': reduce(correct.astype(np.float64))}
    return result, known


def combine_scores(rows):
    result = {'per_episode': [{'episode_index': i, **row} for i, row in enumerate(rows)]}
    for group in ('all', 'age_gt32'):
        selected = [r[group] for r in rows if r[group]['query_boundaries']]
        infinite = any(r['nll_is_infinite'] for r in selected)
        result[group] = {'eligible_episodes': len(selected),
            **{k: sum(r[k] for r in selected) for k in ('query_boundaries', 'query_cards',
               'infinite_nll_queries', 'probability_underflow_queries')}, 'nll_is_infinite': infinite,
            **{k: None if not selected or (k == 'nll' and infinite)
               else math.fsum(r[k] for r in selected) / len(selected) for k in ('nll', 'brier', 'accuracy')}}
    return result


def calculate(rows, scores):
    """Independent aggregate expressions; exact Decimal thresholds, no epsilon."""
    d = lambda x: Decimal(str(x))
    def avg(values):
        values = list(values)
        return float(sum(map(d, values)) / Decimal(len(values)))
    contrasts = {}
    for policy in ('temperature', 'hard'):
        fits = {n: float(d(rows[policy][n]['mean_return']) - d(rows['baseline'][n]['mean_return'])) for n in NAMES}
        contrasts[policy + '_minus_baseline'] = {'mean_difference': avg(fits.values()),
            'families': {m: avg(fits[f'{m}-pair{i}'] for i in range(3)) for m in MODES},
            'paired_fit_aggregates': [avg(fits[f'{m}-pair{i}'] for m in MODES) for i in range(3)],
            'per_fit': {n: {'mean_difference': fits[n],
                'per_seed_differences': [float(d(b) - d(a)) for a, b in zip(rows['baseline'][n]['per_seed_returns'],
                    rows[policy][n]['per_seed_returns'], strict=True)]} for n in NAMES}}
    transfer = {}
    for policy in POLICIES:
        values = [scores[n][policy]['all'] for n in NAMES]
        require(all(r['eligible_episodes'] == 64 and r['query_cards'] > 0 for r in values), 'Missing test targets')
        infinite = any(r['nll_is_infinite'] for r in values)
        transfer[policy] = {'nll': None if infinite else avg(r['nll'] for r in values),
            'nll_is_infinite': infinite, 'infinite_nll_queries': sum(r['infinite_nll_queries'] for r in values),
            **{k: avg(r[k] for r in values) for k in ('brier', 'accuracy')}}
    change, checks = contrasts['temperature_minus_baseline'], []
    def check(name, value, op, threshold):
        left, right = d(value), d(threshold)
        passed = {'>=': left >= right, '>': left > right, '<=': left <= right}[op]
        checks.append({'name': name, 'value': value, 'comparison': op, 'threshold': float(threshold), 'passed': passed})
    check('mean_temperature_minus_baseline_at_least_0.03', change['mean_difference'], '>=', .03)
    for i, value in enumerate(change['paired_fit_aggregates']):
        check(f'pair{i}_aggregate_positive', value, '>', 0)
    a, b = transfer['baseline'], transfer['temperature']
    if a['nll_is_infinite'] or b['nll_is_infinite'] or a['nll'] <= 0:
        checks.append({'name': 'fresh_C_prefix_NLL_at_least_5_percent_lower', 'value': b['nll'],
            'comparison': '<=', 'threshold': None, 'passed': False})
    else:
        check('fresh_C_prefix_NLL_at_least_5_percent_lower', b['nll'], '<=', d(a['nll']) * Decimal('.95'))
    check('fresh_C_prefix_Brier_nonworse', b['brier'], '<=', a['brier'])
    for mode in MODES:
        check(mode + '_native_loss_at_most_0.01', change['families'][mode], '>=', -.01)
    return contrasts, transfer, {'checks': checks, 'checks_passed': sum(c['passed'] for c in checks),
        'total_checks': 12, 'passed': all(c['passed'] for c in checks)}


def audit(*, report, expected_report_receipt_sha256, evaluation, expected_evaluation_completed_sha256,
          calibration, expected_calibration_completed_sha256, protocol, inputs, bindings, checkpoint_map, out, root=ROOT):
    start = time.monotonic()
    root, report, evaluation, calibration, out = (Path(p).resolve() for p in (root, report, evaluation, calibration, out))
    require(all(not out.is_relative_to(p) for p in (report, evaluation, calibration)), 'Audit cannot alter sealed inputs')
    out.mkdir(parents=True, exist_ok=False)
    source_sha = sha(__file__)
    inputs_identity = {'report_receipt_sha256': expected_report_receipt_sha256,
        'evaluation_completed_sha256': expected_evaluation_completed_sha256,
        'calibration_completed_sha256': expected_calibration_completed_sha256}
    stage, checked = 'start', 0
    try:
        write(out / 'started.json', {'status': 'started', 'inputs': inputs_identity, 'source_sha256': source_sha,
            'scope': 'Independent saved-array arithmetic; no fitting, inference, environment or RNG calls'})
        stage = 'authentication'
        require(sha(report / 'receipt.json') == expected_report_receipt_sha256, 'External report receipt differs')
        receipt = read(report / 'receipt.json')
        require(receipt['status'] == 'complete' and receipt['new_model_calls'] == receipt['new_native_calls'] == 0,
                'Report must be complete')
        require(set(receipt['files']) == {'summary.json', 'report.md'}, 'Report member schema differs')
        require({p.name for p in report.iterdir()} == {'summary.json', 'report.md', 'receipt.json'}, 'Unexpected report files')
        for name, spec in receipt['files'].items():
            path = report / name
            require(sha(path) == spec['sha256'] and path.stat().st_size == spec['bytes'], 'Corrupt report file')
        for key, path in {'protocol': protocol, 'inputs': inputs, 'bindings': bindings, 'checkpoint_map': checkpoint_map}.items():
            require(Path(path).resolve().is_relative_to(root) and sha(path) == receipt[key + '_sha256'], 'Bound input differs: ' + key)
        require(receipt['evaluation_completed_sha256'] == expected_evaluation_completed_sha256
                and receipt['calibration_completed_sha256'] == expected_calibration_completed_sha256, 'Report input mismatch')
        recipe, cases, sources, checkpoints = map(read, (protocol, inputs, bindings, checkpoint_map))
        require(recipe['version'] == 'card-calibration-v1' and recipe['evaluation']['episodes'] == 64
                and recipe['evaluation']['policies'] == list(POLICIES), 'Wrong recipe')
        for name, digest in sources['files'].items():
            require(sha(member(root, name)) == digest, 'Frozen source/input changed: ' + name)
        summary = read(report / 'summary.json')
        require(summary['status'] == 'complete' and len(cases['evaluation']) == 64
                and len({c['seed'] for c in cases['evaluation']}) == 64, 'Incomplete summary or test cases')
        calibration_names = {'started.json'} | {f'fits/{n}/{p}' for n in NAMES
            for p in ('fit.json', 'source-episodes.json', 'completed.json')}
        calibrated = payload_tree(calibration, expected_calibration_completed_sha256, calibration_names)
        completed = payload_tree(evaluation, expected_evaluation_completed_sha256, evaluation_members())
        require(set(calibrated['fits']) == set(checkpoints) == set(NAMES), 'All eighteen fixed fits required')
        betas = {}
        for name in NAMES:
            entry = calibrated['fits'][name]
            fitted = read(calibration / f'fits/{name}/fit.json')
            require(entry['checkpoint_sha256'] == checkpoints[name]['checkpoint_sha256']
                    and type(entry['beta']) in (float, int) and .05 <= entry['beta'] <= 20
                    and entry['beta'] == fitted['beta'], 'Scalar/checkpoint binding differs')
            betas[name] = entry['beta']
        for record in (calibrated, completed):
            for key in ('protocol', 'inputs', 'bindings', 'checkpoint_map'):
                require(record[key + '_sha256'] == receipt[key + '_sha256'], 'Completion lineage mismatch')
        stage = 'saved_arrays'
        rows, scores = {p: {} for p in POLICIES}, {}
        layouts = [np.full(52, -1, np.int64) for _ in range(64)]
        layout_hashes, native_steps = {}, 0
        for name in (*NAMES, *REFERENCES):
            score_rows = {p: [] for p in POLICIES}
            for policy in POLICIES if name in NAMES else ('baseline',):
                returns, successes, steps, seconds = [], [], 0, []
                folder = evaluation / f'controllers/{name}/{policy}'
                for i, case in enumerate(cases['evaluation']):
                    stem = folder / f'episodes/{i:03d}'
                    saved = read(stem.with_suffix('.json'))
                    require(saved['status'] == 'complete' and saved['controller'] == name and saved['policy'] == policy
                            and saved['index'] == i and saved['seed'] == case['seed']
                            and saved['beta'] == (betas[name] if policy == 'temperature' else None), 'Episode identity differs')
                    require(sha(stem.with_suffix('.npz')) == saved['npz_sha256'], 'Episode hash differs')
                    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as archive:
                        rewards, terminated, truncated, obs = (archive[k] for k in ('rewards', 'terminated', 'truncated', 'observations'))
                        n = len(rewards)
                        require(rewards.dtype == np.float64 and rewards.shape == (n,) and 1 <= n <= 104
                                and np.isfinite(rewards).all(), 'Invalid native rewards')
                        require(terminated.dtype == truncated.dtype == np.bool_ and terminated.shape == truncated.shape == (n,)
                                and not np.any(terminated[:-1] | truncated[:-1]) and (terminated[-1] or truncated[-1]), 'Invalid terminal flags')
                        total = math.fsum(rewards)
                        same(total, saved['return'], 'Episode return')
                        require(type(saved['success']) is bool and saved['success'] == bool(terminated[-1])
                                and saved['native_steps'] == n, 'Episode counts/success differ')
                        require(obs.dtype == np.int64 and obs.shape == (n + 1, 52)
                                and np.all((obs >= 0) & (obs <= 13)), 'Invalid public observations')
                        for frame in obs:
                            visible = frame < 13
                            require(np.all((layouts[i][visible] == -1) | (layouts[i][visible] == frame[visible])), 'Inconsistent paired public rank')
                            layouts[i][visible] = frame[visible]
                        known_hash = layout_hashes.setdefault(str(i), saved['layout_sha256'])
                        require(known_hash == saved['layout_sha256'], 'Paired deck digest differs')
                        if name in NAMES and policy == 'baseline':
                            raw = archive['raw_probabilities']
                            for transform in POLICIES:
                                result, _ = score_episode(raw, obs, transform, betas[name] if transform == 'temperature' else 1.)
                                score_rows[transform].append(result)
                    returns.append(total); successes.append(saved['success']); steps += n
                    require(type(saved['whole_episode_seconds']) in (float, int)
                            and math.isfinite(saved['whole_episode_seconds']) and saved['whole_episode_seconds'] >= 0, 'Invalid timing')
                    seconds.append(saved['whole_episode_seconds']); checked += 1
                row = {'mean_return': math.fsum(returns) / 64, 'per_seed_returns': returns,
                    'per_seed_successes': successes, 'native_steps': steps, 'wall_seconds': math.fsum(seconds)}
                rows[policy][name] = row
                same(row, summary['per_fit'][policy][name], 'Native row/' + name + '/' + policy)
                row_done = read(folder / 'completed.json')
                same(row['mean_return'], row_done['mean_return'], 'Completed row mean')
                require(row_done['successes'] == sum(successes) and row_done['native_steps'] == steps
                        and row_done['episodes'] == 64, 'Completed row counts')
                native_steps += steps
            if name in NAMES:
                scores[name] = {p: combine_scores(score_rows[p]) for p in POLICIES}
                for policy in POLICIES:
                    same(scores[name][policy], summary['fresh_baseline_prefix_scores'][name][policy], 'Fresh proper score/' + name + '/' + policy)
        require(checked == completed['episodes'] == 3584
                and native_steps == completed['counts']['native_returned'] <= 372736, 'Total native coverage differs')
        for i, deck in enumerate(layouts):
            require(np.all(deck >= 0) and np.array_equal(np.bincount(deck, minlength=13), np.full(13, 4)), 'Incomplete public deck reconstruction')
            require(hashlib.sha256(LAYOUT_PREFIX + deck.astype('<i8').tobytes()).hexdigest() == layout_hashes[str(i)], 'Public layout digest differs')
        stage = 'aggregates_and_gate'
        contrasts, transfer, gate = calculate(rows, scores)
        same(contrasts, summary['contrasts'], 'Paired contrasts')
        same(transfer, summary['transfer'], 'Proper-score transfer')
        same(gate, summary['continuation_gate'], 'Continuation gate')
        require(receipt['continuation_passed'] == gate['passed'] and receipt['checks_passed'] == gate['checks_passed'], 'Receipt gate differs')
        family_means = {p: {m: math.fsum(rows[p][f'{m}-pair{i}']['mean_return'] for i in range(3)) / 3 for m in MODES} for p in POLICIES}
        for policy in POLICIES:
            for mode, value in family_means[policy].items():
                same(value, summary['families'][policy][mode]['mean_return'], 'Family mean')
        result = {'status': 'passed', 'scope': 'Independent saved-array arithmetic; scientific qualification may fail',
            'inputs': inputs_identity, 'source_sha256': source_sha, 'episodes': checked, 'native_rows': 56,
            'native_actions_read': native_steps, 'fresh_baseline_prefix_episodes': 1152,
            'family_means': family_means, 'contrasts': contrasts, 'transfer': transfer, 'continuation_gate': gate,
            'references': {name: rows['baseline'][name] for name in REFERENCES},
            'new_model_calls': 0, 'new_native_calls': 0, 'new_scalar_fits': 0,
            'limits': ['No original numerical helper/report code is imported; its frozen schemas and declared Decimal aggregation convention are shared.',
                'The saved beliefs and rewards are authenticated evidence, not independently reproduced model/native outputs.',
                'Scalar optimality and C action replay remain the frozen reporter scope; this audit does not refit temperatures.',
                'Numerical score agreement permits reduction roundoff; every gate uses its declared inclusive/strict comparison with no epsilon.',
                'Earlier C test data is calibration training here; no original outcome or architectural claim is revised.'],
            'wall_seconds': time.monotonic() - start}
        require(sha(__file__) == source_sha and sha(report / 'receipt.json') == expected_report_receipt_sha256
                and sha(evaluation / 'completed.json') == expected_evaluation_completed_sha256
                and sha(calibration / 'completed.json') == expected_calibration_completed_sha256, 'Final identity changed')
        write(out / 'completed.json', result)
        return result
    except BaseException as error:
        try:
            write(out / 'failed.json', {'status': 'failed', 'stage': stage, 'episodes_checked': checked,
                'error': repr(error), 'inputs': inputs_identity, 'source_sha256': source_sha,
                'wall_seconds': time.monotonic() - start, 'automatic_retry': False})
        except BaseException as secondary:  # noqa: BLE001 - retain original validation error.
            if callable(getattr(error, 'add_note', None)):
                error.add_note(f'Failure receipt could not be written: {secondary!r}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    for name in ('report', 'evaluation', 'calibration', 'protocol', 'inputs', 'bindings', 'checkpoint_map', 'out'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    for name in ('report_receipt', 'evaluation_completed', 'calibration_completed'):
        parser.add_argument('--expected-' + name.replace('_', '-') + '-sha256', required=True)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ('status', 'episodes', 'native_rows', 'wall_seconds')}, indent=2))
