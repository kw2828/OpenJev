"""Saved-output calibration/control audit; no model, optimizer or native calls.

The original C histories are explicitly calibration training data. Proper-score
transfer is evaluated on fixed fresh baseline prefixes, independently of the
temperature/hard policies' different native trajectories.
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
import report_card_controllers as prior

MODES, REFERENCES = prior.MODES, prior.REFERENCES
NAMES = tuple(f'{mode}-pair{i}' for mode in MODES for i in range(3))
POLICIES = ('baseline', 'temperature', 'hard')
WEIGHTING = 'queries within boundary; nonempty boundaries within episode; eligible episodes equally'
require, sha, read, write = prior.require, prior.sha, prior.read, prior.write
member, finite, mean, same_number = prior.member, prior.finite, prior.mean, prior.same_number


def transform(raw, policy, beta=None):
    """Independent NumPy transformation before unchanged public C overrides."""
    require(policy in POLICIES, 'Unknown confidence policy')
    values = np.asarray(raw)
    require(values.dtype in (np.dtype('float32'), np.dtype('float64'))
            and values.shape[-2:] == (52, 13) and values.ndim >= 2
            and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)), 'Invalid raw beliefs')
    values = values.astype(np.float64, copy=True)
    require(np.all(np.abs(values.sum(axis=-1) - 1) <= 1e-5), 'Invalid raw probability mass')
    if policy == 'temperature':
        require(type(beta) in (int, float) and math.isfinite(beta) and .05 <= beta <= 20, 'Invalid fitted beta')
        if beta != 1:
            logs = np.full(values.shape, -np.inf)
            np.log(values, out=logs, where=values > 0)
            logs -= logs.max(axis=-1, keepdims=True)
            values = np.exp(beta * logs)
            values /= values.sum(axis=-1, keepdims=True)
    else:
        require(beta is None or beta == 1, 'Non-temperature policy cannot use fitted beta')
        if policy == 'hard':
            top = values.argmax(axis=-1)
            values.fill(0)
            np.put_along_axis(values, top[..., None], 1., axis=-1)
    return values


def public_targets(observations):
    """Only earlier/current public frames supply a label; current visible is excluded."""
    obs = np.asarray(observations)
    require(obs.dtype == np.int64 and obs.ndim == 2 and obs.shape[1] == 52 and 2 <= len(obs) <= 105
            and np.all((obs >= 0) & (obs <= 13)) and np.all(obs[0] == 13), 'Invalid public frames')
    labels, seen_at = np.full(52, -1, np.int64), np.full(52, -1, np.int32)
    targets = np.full((len(obs) - 1, 52), -1, np.int64)
    ages = np.full(targets.shape, -1, np.int32)
    mask = np.zeros(targets.shape, bool)
    for t, frame in enumerate(obs):
        visible = frame != 13
        require(np.all((labels[visible] == -1) | (labels[visible] == frame[visible])), 'Public rank changed')
        labels[visible], seen_at[visible] = frame[visible], t
        if t == len(obs) - 1:
            break
        eligible = (labels >= 0) & ~visible
        mask[t] = eligible
        targets[t, eligible], ages[t, eligible] = labels[eligible], t - seen_at[eligible]
    return targets, mask, ages


def proper_scores(episodes, policy, beta=None):
    """Independent equal-episode/boundary/query arithmetic, including true infinities."""
    require(bool(episodes), 'No score episodes supplied')
    rows = []
    for index, ep in enumerate(episodes):
        require(set(ep) == {'raw_probabilities', 'targets', 'target_mask', 'ages'}, 'Unexpected score episode fields')
        raw = ep['raw_probabilities']
        probs = transform(raw, policy, beta)
        require(probs.ndim == 3 and 1 <= len(probs) <= 104, 'Invalid score episode length')
        probs /= probs.sum(axis=-1, keepdims=True)
        y, mask, ages = (ep[k] for k in ('targets', 'target_mask', 'ages'))
        for value, dtype in ((y, np.int64), (mask, np.bool_), (ages, np.int32)):
            require(value.shape == probs.shape[:2] and value.dtype == dtype, 'Invalid score mask/label schema')
        require(np.all(y[~mask] == -1) and np.all(ages[~mask] == -1)
                and np.all((y[mask] >= 0) & (y[mask] < 13))
                and np.all(ages[mask] >= 1)
                and np.all(ages <= np.arange(len(probs))[:, None]), 'Noncausal score target/age')
        # Mathematical finite-beta likelihood stays finite even when exp underflows.
        logs = np.full(raw.shape, -np.inf, np.float64)
        np.log(raw.astype(np.float64), out=logs, where=raw > 0)
        logs -= logs.max(axis=-1, keepdims=True)
        scaled = logs * (beta if policy == 'temperature' else 1.)
        logmass = np.log(np.exp(scaled).sum(axis=-1))
        row = {'episode_index': index}
        for group, selected in (('all', mask), ('age_gt32', mask & (ages > 32))):
            bs, ns, acc, zeros, underflows, cards = [], [], [], 0, 0, 0
            for t in np.flatnonzero(selected.any(axis=1)):
                p, target = probs[t, selected[t]], y[t, selected[t]]
                correct = p[np.arange(len(target)), target]
                if policy == 'hard':
                    nll = np.where(correct > 0, 0., np.inf)
                else:
                    nll = logmass[t, selected[t]] - scaled[t, selected[t], target]
                zero = int(np.count_nonzero(~np.isfinite(nll)))
                zeros += zero
                underflows += int(np.count_nonzero((correct == 0) & np.isfinite(nll)))
                cards += len(target)
                ns.append(None if zero else mean(float(v) for v in nll))
                errors = p.copy(); errors[np.arange(len(target)), target] -= 1
                bs.append(mean(float(v) for v in np.square(errors).sum(axis=1)))
                acc.append(float(np.count_nonzero(p.argmax(axis=1) == target)) / len(target))
            row[group] = {'query_boundaries': len(bs), 'query_cards': cards,
                          'nll': None if zeros or not ns else mean(ns), 'nll_is_infinite': bool(zeros),
                          'infinite_nll_queries': zeros, 'probability_underflow_queries': underflows,
                          'brier': mean(bs) if bs else None, 'accuracy': mean(acc) if acc else None}
        rows.append(row)
    result = {'policy': policy, 'beta': beta, 'weighting': WEIGHTING, 'episodes': len(rows), 'per_episode': rows}
    for group in ('all', 'age_gt32'):
        active = [r[group] for r in rows if r[group]['query_boundaries']]
        infinite = any(r['nll_is_infinite'] for r in active)
        result[group] = {'eligible_episodes': len(active),
                         **{key: sum(r[key] for r in active) for key in
                            ('query_boundaries', 'query_cards', 'infinite_nll_queries', 'probability_underflow_queries')},
                         'nll_is_infinite': infinite,
                         'nll': None if infinite or not active else mean(r['nll'] for r in active),
                         **{key: mean(r[key] for r in active) if active else None for key in ('brier', 'accuracy')}}
    return result


def expected_members():
    result = {'started.json', 'all-fits-ready.json'}
    for name in (*NAMES, *REFERENCES):
        for policy in POLICIES if name in NAMES else ('baseline',):
            prefix = f'controllers/{name}/{policy}'
            result.add(prefix + '/completed.json')
            result.update(f'{prefix}/episodes/{i:03d}.{ext}' for i in range(64) for ext in ('json', 'npz'))
    return result


def authenticate_tree(folder, expected_sha, expected=None):
    folder = Path(folder)
    require(sha(folder / 'completed.json') == expected_sha, 'External completion SHA mismatch')
    done = read(folder / 'completed.json')
    require(done['status'] == 'complete', 'Incomplete source tree')
    files = done['files']
    if expected is not None:
        require(set(files) == set(expected), 'Declared source membership differs')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    require(actual == set(files) | {'completed.json'}, 'Actual source membership differs')
    for name, binding in files.items():
        path = member(folder, name)
        require(sha(path) == binding['sha256'] and path.stat().st_size == binding['bytes'], 'Source member changed: ' + name)
    return done


def aggregate(per_fit, scores):
    """All 18 fits, 64 decks each, 5 grouped prospective conditions (12 rows)."""
    require(set(per_fit) == set(POLICIES) and set(scores) == set(NAMES), 'Missing policies/fresh score fits')
    require(set(per_fit['baseline']) == set(NAMES) | set(REFERENCES), 'Missing baseline/reference rows')
    for policy in POLICIES[1:]:
        require(set(per_fit[policy]) == set(NAMES), 'Missing learned rows or replicated references')
    families, contrasts = {}, {}
    for policy in POLICIES:
        families[policy] = {}
        for family in MODES:
            rows = [per_fit[policy][f'{family}-pair{i}'] for i in range(3)]
            for row in rows:
                require(len(row['per_seed_returns']) == len(row['per_seed_successes']) == 64, 'Wrong native deck coverage')
                require(all(type(v) is bool for v in row['per_seed_successes']), 'Malformed success flag')
                same_number(row['mean_return'], mean(row['per_seed_returns']), 'Native mean differs')
            families[policy][family] = {'mean_return': mean(r['mean_return'] for r in rows),
                'fit_mean_returns': [r['mean_return'] for r in rows], 'successes': sum(sum(r['per_seed_successes']) for r in rows),
                'episodes': 192, 'native_steps': sum(r['native_steps'] for r in rows),
                'episode_seconds': math.fsum(finite(r['wall_seconds'], minimum=0) for r in rows)}
    for policy in ('temperature', 'hard'):
        fits = {}
        for name in NAMES:
            a, b = per_fit['baseline'][name], per_fit[policy][name]
            differences = [float(Decimal(str(y)) - Decimal(str(x))) for x, y in zip(a['per_seed_returns'], b['per_seed_returns'], strict=True)]
            fits[name] = {'mean_difference': float(Decimal(str(b['mean_return'])) - Decimal(str(a['mean_return']))),
                          'per_seed_differences': differences, 'positive_decks': sum(v > 0 for v in differences),
                          'nonworse_decks': sum(v >= 0 for v in differences)}
        contrasts[policy + '_minus_baseline'] = {'per_fit': fits,
            'mean_difference': float(sum(Decimal(str(fits[n]['mean_difference'])) for n in NAMES) / Decimal(18)),
            'families': {m: float(sum(Decimal(str(fits[f'{m}-pair{i}']['mean_difference'])) for i in range(3)) / Decimal(3)) for m in MODES},
            'paired_fit_aggregates': [float(sum(Decimal(str(fits[f'{m}-pair{i}']['mean_difference'])) for m in MODES) / Decimal(6)) for i in range(3)]}
    transfer = {}
    for policy in POLICIES:
        values = []
        for name in NAMES:
            require(set(scores[name]) == set(POLICIES), 'Missing same-prefix confidence scores')
            row = scores[name][policy]['all']
            require(row['eligible_episodes'] == 64 and row['query_cards'] > 0, 'Every fresh baseline episode needs targets')
            values.append(row)
        infinite = any(r['nll_is_infinite'] for r in values)
        def decimal_mean(key, values=values):
            return float(sum(Decimal(str(finite(r[key]))) for r in values) / Decimal(len(values)))
        transfer[policy] = {'nll': None if infinite else decimal_mean('nll'),
            'nll_is_infinite': infinite, 'infinite_nll_queries': sum(r['infinite_nll_queries'] for r in values),
            'brier': decimal_mean('brier'), 'accuracy': decimal_mean('accuracy'),
            'fits': 18, 'episodes_per_fit': 64, 'weighting': 'equal fits; ' + WEIGHTING}
    change = contrasts['temperature_minus_baseline']
    checks = []
    def check(name, value, comparison, threshold):
        a, b = Decimal(str(value)), Decimal(str(threshold))
        passed = a >= b if comparison == '>=' else a > b if comparison == '>' else a <= b
        checks.append({'name': name, 'value': value, 'comparison': comparison, 'threshold': float(threshold),
                       'threshold_decimal': str(b), 'passed': passed})
    check('mean_temperature_minus_baseline_at_least_0.03', change['mean_difference'], '>=', .03)
    for i, value in enumerate(change['paired_fit_aggregates']):
        check(f'pair{i}_aggregate_positive', value, '>', 0)
    baseline, temperature = transfer['baseline'], transfer['temperature']
    if baseline['nll_is_infinite'] or temperature['nll_is_infinite'] or baseline['nll'] <= 0:
        checks.append({'name': 'fresh_C_prefix_NLL_at_least_5_percent_lower', 'value': temperature['nll'],
                       'comparison': '<=', 'threshold': None, 'passed': False,
                       'reason': 'Relative NLL improvement requires a positive finite baseline and finite temperature mean'})
    else:
        check('fresh_C_prefix_NLL_at_least_5_percent_lower', temperature['nll'], '<=', Decimal(str(baseline['nll'])) * Decimal('.95'))
    check('fresh_C_prefix_Brier_nonworse', temperature['brier'], '<=', baseline['brier'])
    for m in MODES:
        check(m + '_native_loss_at_most_0.01', change['families'][m], '>=', -.01)
    return families, contrasts, transfer, {'passed': all(c['passed'] for c in checks),
        'checks_passed': sum(c['passed'] for c in checks), 'total_checks': 12, 'checks': checks,
        'grouped_requirements': {'overall_native_margin': checks[0]['passed'],
            'all_three_paired_aggregates_positive': all(c['passed'] for c in checks[1:4]),
            'fresh_prefix_NLL_improvement': checks[4]['passed'], 'fresh_prefix_Brier_nonworse': checks[5]['passed'],
            'no_family_native_loss_above_0.01': all(c['passed'] for c in checks[6:])},
        'hard_and_references_cannot_rescue_temperature': True,
        'interpretation': 'Five jointly required conditions expanded to twelve checks; not independent significance tests'}


def audit_episode(path, receipt, *, policy, controller, beta=None):
    """Replay public bookkeeping only; learned raw probabilities remain supplied."""
    from openjev.research.card_memory_picker import CardPolicyTracker
    from openjev.research.card_memory_task import PublicTeacher

    require(receipt['status'] == 'complete' and receipt['policy'] == policy
            and receipt['controller'] == controller and receipt['tracker_policy'] == 'C'
            and receipt['beta'] == beta, 'Episode identity/status/beta mismatch')
    require(sha(path) == receipt['npz_sha256'] and Path(path).stat().st_size == receipt['npz_bytes'],
            'Episode payload changed')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    n = receipt['native_steps']
    require(type(n) is int and 1 <= n <= 104, 'Invalid episode length')
    schema = {'actions': ((n,), np.int64), 'ranks': ((n,), np.int64), 'rewards': ((n,), np.float64),
              'terminated': ((n,), np.bool_), 'truncated': ((n,), np.bool_),
              'observations': ((n + 1, 52), np.int64), 'raw_probabilities': ((n, 52, 13), np.float64),
              'picker_probabilities': ((n, 52, 13), np.float64)}
    require(set(arrays) == set(schema), 'Episode array membership mismatch')
    for key, (shape, dtype) in schema.items():
        require(arrays[key].shape == shape and arrays[key].dtype == dtype and np.isfinite(arrays[key]).all(),
                'Invalid saved array: ' + key)
    require(sum(a.nbytes for a in arrays.values()) == receipt['array_bytes'], 'Uncompressed array byte count differs')
    require(len(receipt['decisions']) == n, 'Missing decision diagnostics')
    tracker, teacher = CardPolicyTracker('C'), PublicTeacher()
    tracker.reset(arrays['observations'][0])
    teacher.reset(arrays['observations'][0])
    events, layout = [], np.full(52, -1, dtype=np.int64)
    mismatch_pairs, repeated_mismatches, mismatch_count = set(), 0, 0
    counts = {'first_phase': 0, 'tie_decisions': 0, 'selected_unseen': 0, 'first_selected_unseen': 0, 'C_differs_from_B': 0,
              'tie_size_sum': 0, 'max_rowmass_abs': 0.0}
    for t in range(n):
        raw = arrays['raw_probabilities'][t]
        if controller == 'exact':
            require(np.array_equal(raw, teacher.probabilities()), 'Exact reference memory mismatch')
        elif controller == 'last32':
            expected = np.full((52, 13), 1 / 13, dtype=np.float64)
            for event in events[-32:]:
                expected[event.pos] = 0
                expected[event.pos, event.rank] = 1
            require(np.array_equal(raw, expected), 'Window reference memory mismatch')
        beliefs = transform(raw, policy, beta)
        action, picker, diagnostic = tracker.decision(beliefs)
        require(action == int(arrays['actions'][t]), 'Saved action differs from the frozen picker')
        require(np.array_equal(picker, arrays['picker_probabilities'][t]), 'Picker probabilities changed')
        require(json.dumps(diagnostic, sort_keys=True) == json.dumps(receipt['decisions'][t], sort_keys=True),
                'Picker diagnostics changed')
        pending = tracker.pending
        after = arrays['observations'][t + 1]
        event = tracker.observe(action, float(arrays['rewards'][t]), after)
        require(event.rank == int(arrays['ranks'][t]), 'Actual selected rank mismatch')
        teacher.observe(after)
        events.append(event)
        visible = after != 13
        require(np.all((layout[visible] == -1) | (layout[visible] == after[visible])), 'Public rank changed')
        layout[visible] = after[visible]
        if pending is not None and after[pending] != after[action]:
            pair = tuple(sorted((pending, action)))
            mismatch_count += 1
            repeated_mismatches += int(pair in mismatch_pairs)
            mismatch_pairs.add(pair)
        require(bool(arrays['terminated'][t]) == bool(tracker.matched.all()), 'Terminal flag mismatch')
        require(bool(arrays['truncated'][t]) == (t == 103), 'Truncation flag mismatch')
        require(bool(arrays['terminated'][t] or arrays['truncated'][t]) == (t == n - 1), 'Premature/late episode end')
        counts['first_phase'] += int(diagnostic['firstphase'])
        counts['tie_decisions'] += int(diagnostic['tie_size'] > 1)
        counts['selected_unseen'] += int(diagnostic['selected_unseen'])
        counts['first_selected_unseen'] += int(diagnostic['firstphase'] and diagnostic['selected_unseen'])
        counts['tie_size_sum'] += diagnostic['tie_size']
        counts['max_rowmass_abs'] = max(counts['max_rowmass_abs'], diagnostic['rowmass_max_abs'])
        counts['C_differs_from_B'] += int(action != diagnostic['Baction_ifC'])
    total = math.fsum(arrays['rewards'])
    same_number(total, receipt['return'], 'Episode native return mismatch')
    require(type(receipt['success']) is bool and receipt['success'] == bool(arrays['terminated'][-1]),
            'Episode success mismatch')
    require(receipt['matched_pairs'] == int(tracker.matched.sum()) // 2, 'Matched-pair count mismatch')
    work = receipt['counts']
    require(work['reset_attempted'] == work['reset_returned'] == 1, 'Native reset count mismatch')
    require(work['identity_read_attempted'] == work['identity_read_returned'] == 1, 'Deck identity count mismatch')
    require(work['native_attempted'] == work['native_returned'] == n, 'Native action count mismatch')
    require(work['decision_attempted'] == work['decision_returned'] == n, 'Picker call count mismatch')
    require(work['transform_attempted'] == work['transform_returned'] == n, 'Confidence transform count mismatch')
    require(work['init_attempted'] == work['init_returned'] == int(controller not in REFERENCES),
            'Model state initialization count mismatch')
    expected_neural = 0 if controller in REFERENCES else n
    require(all(work[key] == expected_neural for key in
                ('predict_attempted', 'predict_returned', 'write_attempted', 'write_returned')),
            'Predict/write count mismatch')
    for value in receipt['timings'].values():
        finite(value, minimum=0)
    finite(receipt['wall_seconds'], minimum=0)
    require(math.fsum(receipt['timings'].values()) <= receipt['wall_seconds'] + 1e-6,
            'Nested episode timers exceed elapsed wall time')
    return {'return': total, 'success': receipt['success'], 'native_steps': n,
            'unique_positions': len(set(arrays['actions'].tolist())), 'mismatching_pairs': mismatch_count,
            'repeat_mismatching_pairs': repeated_mismatches, 'picker_counts': counts,
            'layout': layout, 'counts': work, 'timings': receipt['timings'],
            'score_episode': dict(zip(('targets', 'target_mask', 'ages'),
                                     public_targets(arrays['observations']), strict=True),
                                  raw_probabilities=arrays['raw_probabilities'])}


def authenticate_inputs(root, paths, expected):
    """Hash/runtime/checkpoint closure, without importing the new execution driver."""
    import importlib.metadata
    import platform
    import sys

    for key, path in paths.items():
        require(Path(path).resolve().is_relative_to(root) and sha(path) == expected[key], 'External input differs: ' + key)
    recipe, inputs, bindings, checkpoints = (read(paths[k]) for k in ('protocol', 'inputs', 'bindings', 'checkpoint_map'))
    require(recipe['version'] == inputs['version'] == bindings['version'] == 'card-calibration-v1', 'Wrong calibration study')
    require(recipe['evaluation'] == {'episodes': 64, 'max_actions': 104, 'controllers': 20,
        'policies': list(POLICIES), 'wall_cap_seconds': 1800, 'output_cap_bytes': 6_000_000_000}, 'Evaluation recipe changed')
    require(recipe['calibration'] == {'fits': 18, 'episodes_per_fit': 64, 'beta_bounds': [.05, 20.],
        'bisection_iterations': 64, 'max_oracle_evaluations': 1224, 'wall_cap_seconds': 900,
        'output_cap_bytes': 200_000_000, 'new_model_calls': 0, 'new_native_calls': 0}, 'Calibration recipe changed')
    require(recipe['calibration_source_role'] == 'previous_C_evaluation_repurposed_as_training', 'Exposed split role omitted')
    require(recipe['continuation'] == {'temperature_minus_baseline_mean_return_min': .03,
        'paired_fit_aggregate_positive_all': True, 'baseline_prefix_relative_nll_reduction_min': .05,
        'baseline_prefix_brier_nonincrease': True, 'family_mean_return_difference_min': -.01,
        'all_required': True, 'hard_control_descriptive_only': True,
        'original_architecture_gate_unchanged': True}, 'Prospective continuation rule changed')
    require(member(root, recipe['bindings_file']) == Path(paths['bindings']).resolve(), 'Wrong binding file')
    runtime = {'python': platform.python_version(), 'executable': sys.executable,
        'machine': platform.machine(), 'platform': platform.platform(),
        **{key: importlib.metadata.version(key) for key in ('numpy', 'torch', 'gymnasium')}}
    require(bindings['runtime'] == runtime, 'Runtime differs')
    required_sources = {'scripts/card_calibration_common.py', 'scripts/fit_card_calibration.py',
        'scripts/evaluate_card_calibration.py', 'src/openjev/research/card_probability_calibration.py',
        'scripts/report_card_calibration.py', 'tests/test_report_card_calibration.py'}
    require(required_sources <= set(bindings['files']), 'Required execution/analysis source omitted')
    for relative, digest in bindings['files'].items():
        require(sha(member(root, relative)) == digest, 'Bound source differs: ' + relative)
    for key in ('protocol', 'inputs'):
        require(bindings['files'].get(str(Path(paths[key]).resolve().relative_to(root))) == expected[key], 'Unbound prospective input')
    keys = {'controller_protocol', 'controller_inputs', 'controller_bindings', 'controller_completed',
            'controller_report_receipt', 'controller_summary', 'checkpoint_map'}
    require(set(recipe['lineage']) == keys, 'Wrong inherited lineage members')
    lineage = {}
    for key, item in recipe['lineage'].items():
        require(set(item) == {'path', 'sha256'}, 'Malformed lineage binding')
        path = member(root, item['path'])
        require(sha(path) == item['sha256'], 'Inherited artifact changed: ' + key)
        lineage[key] = read(path)
    old_paths = {key: member(root, recipe['lineage'][oldkey]['path']) for key, oldkey in
                 (('protocol', 'controller_protocol'), ('inputs', 'controller_inputs'),
                  ('bindings', 'controller_bindings'), ('checkpoint_map', 'checkpoint_map'))}
    old_expected = {key: recipe['lineage'][oldkey]['sha256'] for key, oldkey in
                    (('protocol', 'controller_protocol'), ('inputs', 'controller_inputs'),
                     ('bindings', 'controller_bindings'), ('checkpoint_map', 'checkpoint_map'))}
    old_recipe, old_inputs, old_map, _ = prior.authenticate_inputs(root, old_paths, old_expected)
    require(old_map == checkpoints and expected['checkpoint_map'] == old_expected['checkpoint_map'], 'Inherited weights/map changed')
    source_dir = member(root, recipe['lineage']['controller_completed']['path']).parent
    old_complete = prior.authenticate_evaluation(source_dir, recipe['lineage']['controller_completed']['sha256'])
    require(old_complete['episodes'] == 3840 and old_complete['new_fits'] == old_complete['new_optimizer_steps'] == 0,
            'Incomplete original control run')
    for key, digest in old_expected.items():
        require(old_complete[key + '_sha256'] == digest, 'Original evaluation provenance differs')
    receipt, summary = lineage['controller_report_receipt'], lineage['controller_summary']
    require(receipt['status'] == summary['status'] == 'complete' and receipt['continuation_passed'] is True
            and receipt['evaluation_completed_sha256'] == recipe['lineage']['controller_completed']['sha256']
            and receipt['files']['summary.json']['sha256'] == recipe['lineage']['controller_summary']['sha256']
            and summary['original_architecture_gate'] == {'passed': False, 'checks_passed': 1, 'total_checks': 6, 'unchanged': True},
            'Prior report/gate binding differs')
    require(len(inputs['evaluation']) == 64, 'Exactly64 fresh test cases required')
    for i, case in enumerate(inputs['evaluation']):
        require(set(case) == {'seed', 'policy_order'} and type(case['seed']) is int and 0 <= case['seed'] < 2**32
                and case['policy_order'] == list(POLICIES[i % 3:] + POLICIES[:i % 3]), 'Bad paired case/order')
    older_inputs = read(member(root, old_recipe['lineage']['old_inputs']['path']))
    seeds = {c['seed'] for c in inputs['evaluation']}
    require(len(seeds) == 64 and seeds.isdisjoint(prior._seed_inventory(old_inputs)
            | prior._seed_inventory(older_inputs) | {0, 410}), 'Test seeds overlap exposed/training streams')
    return recipe, inputs, checkpoints, bindings, source_dir, old_inputs


def validate_scalar_fit(episodes, fitted):
    """Recompute the saved convex objective/derivative trace, not a new fitted model."""
    logs, targets, weights = [], [], []
    eligible = sum(bool(ep['target_mask'].any()) for ep in episodes)
    require(eligible > 0, 'No calibration labels')
    boundaries = cards = 0
    for ep in episodes:
        # Validate the full public mask schema through independent proper-score arithmetic.
        proper_scores([ep], 'baseline')
        mask = ep['target_mask']; per_boundary = mask.sum(axis=1)
        count = int(np.count_nonzero(per_boundary))
        if not count:
            continue
        values = ep['raw_probabilities'][mask].astype(np.float64)
        y = ep['targets'][mask]
        require(np.all(values[np.arange(len(y)), y] > 0), 'Zero calibration likelihood cannot fit finite beta')
        z = np.full(values.shape, -np.inf)
        np.log(values, out=z, where=values > 0); z -= z.max(axis=1, keepdims=True)
        logs.append(z); targets.append(y)
        scale = 1. / (eligible * count * np.maximum(per_boundary, 1))
        weights.append(np.broadcast_to(scale[:, None], mask.shape)[mask])
        boundaries += count; cards += len(y)
    z, y, w = np.concatenate(logs), np.concatenate(targets), np.concatenate(weights)
    selected = z[np.arange(len(y)), y]; finite_z = np.where(np.isfinite(z), z, 0.)
    trace = fitted['trace']; cursor = 0
    def oracle(beta, label):
        nonlocal cursor
        require(cursor < len(trace), 'Truncated calibration objective trace')
        row = trace[cursor]; cursor += 1
        require(set(row) == {'point', 'beta', 'nll', 'derivative'} and row['point'] == label and row['beta'] == beta,
                'Calibration search/tie trajectory differs')
        exponents = np.exp(beta * z); mass = exponents.sum(axis=1)
        p = exponents / mass[:, None]
        nll = math.fsum(w * (np.log(mass) - beta * selected))
        derivative = math.fsum(w * ((p * finite_z).sum(axis=1) - selected))
        same_number(nll, row['nll'], 'Calibration objective differs')
        same_number(derivative, row['derivative'], 'Calibration derivative differs')
        return {'beta': beta, 'nll': nll, 'derivative': derivative}
    lo, hi = .05, 20.
    left, right, identity = oracle(lo, 'lower'), oracle(hi, 'upper'), oracle(1., 'identity')
    if left['derivative'] >= 0:
        chosen = left
    elif right['derivative'] <= 0:
        chosen = right
    else:
        for i in range(64):
            mid = (lo + hi) / 2
            point = oracle(mid, f'bisection_{i:02d}')
            if point['derivative'] > 0: hi = mid
            elif point['derivative'] < 0: lo = mid
            else: lo = hi = mid
        chosen = oracle((lo + hi) / 2, 'final')
    if chosen['nll'] >= identity['nll']:
        chosen = identity
    require(cursor == len(trace) == fitted['oracle_evaluations'] <= 68, 'Calibration work trace differs')
    require(fitted['version'] == 'card-probability-calibration-v1' and fitted['status'] == 'fitted'
            and fitted['beta_bounds'] == [.05, 20.] and fitted['bisection_iterations'] == 64
            and fitted['weighting'] == WEIGHTING and fitted['new_model_calls'] == fitted['new_memory_weight_updates'] == 0,
            'Calibration semantics/work differs')
    require(fitted['counts'] == {'episodes': len(episodes), 'eligible_episodes': eligible,
                'query_boundaries': boundaries, 'query_cards': cards}, 'Calibration hierarchy coverage differs')
    require(fitted['beta'] == chosen['beta'] and fitted['temperature'] == 1. / chosen['beta']
            and fitted['at_bound'] == (chosen['beta'] in (.05, 20.)), 'Calibrated parameter/tie differs')
    for key, value in (('baseline_nll', identity['nll']), ('calibrated_nll', chosen['nll']), ('chosen_derivative', chosen['derivative'])):
        same_number(fitted[key], value, 'Calibration result differs: ' + key)
    return {key: fitted[key] for key in ('beta', 'temperature', 'baseline_nll', 'calibrated_nll',
                                       'chosen_derivative', 'at_bound', 'counts', 'oracle_evaluations')}


def public_dataset_hash(episodes):
    digest = hashlib.sha256(b'card-calibration-public-episodes-v1\0')
    for index, ep in enumerate(episodes):
        for key in sorted(ep):
            a = np.ascontiguousarray(ep[key])
            digest.update(json.dumps([index, key, str(a.dtype), a.shape], separators=(',', ':')).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


def compare_scores(actual, saved):
    """Check every episode and stratum without silently dropping infinite errors."""
    def compare(a, b):
        if isinstance(a, dict):
            for key, value in a.items():
                compare(value, b['underflowed_target_probabilities' if key == 'probability_underflow_queries' else key])
        elif isinstance(a, list):
            require(len(a) == len(b), 'Score episode coverage differs')
            for x, y in zip(a, b, strict=True): compare(x, y)
        elif type(a) is float:
            same_number(a, b, 'Saved proper score differs')
        else:
            require(type(a) is type(b) and a == b, 'Saved score count/flag differs')
    require(saved['mode'] == actual['policy'] and saved['weighting'] == WEIGHTING
            and saved['beta'] == (actual['beta'] if actual['beta'] is not None else 1.)
            and saved['episodes'] == actual['episodes'], 'Saved score semantics differ')
    for key in ('all', 'age_gt32', 'per_episode'):
        compare(actual[key], saved[key])


def audit_calibration(folder, expected_sha, recipe, expected, maps, source_dir, old_inputs):
    names = {'started.json'} | {f'fits/{n}/{p}' for n in NAMES for p in ('fit.json', 'source-episodes.json', 'completed.json')}
    done = authenticate_tree(folder, expected_sha, names)
    require(done['version'] == 'card-calibration-v1' and done['lineage'] == recipe['lineage']
            and done['new_model_calls'] == done['new_native_calls'] == done['new_neural_weight_updates'] == 0,
            'Calibration scope/lineage differs')
    started = read(folder / 'started.json')
    require(started['status'] == 'started' and started['automatic_retry'] is False
            and started['version'] == 'card-calibration-v1', 'Calibration start differs')
    for key, digest in expected.items():
        require(done[key + '_sha256'] == started[key + '_sha256'] == digest, 'Calibration provenance differs')
    require(set(done['fits']) == set(NAMES), 'All18 calibrated memories required')
    verified, oracles, queries, fit_seconds = {}, 0, 0, 0.
    for name in NAMES:
        stem = folder / 'fits' / name
        entry, fit_done, source, fitted = done['fits'][name], read(stem / 'completed.json'), read(stem / 'source-episodes.json'), read(stem / 'fit.json')
        require(set(entry) == {'beta', 'temperature', 'checkpoint_sha256', 'receipt_path', 'receipt_sha256'}
                and entry['receipt_path'] == f'fits/{name}/completed.json'
                and entry['receipt_sha256'] == sha(stem / 'completed.json')
                and entry['checkpoint_sha256'] == maps[name]['checkpoint_sha256'], 'Calibration entry binding differs')
        require(fit_done['status'] == 'complete' and fit_done['controller'] == name
                and fit_done['checkpoint_sha256'] == entry['checkpoint_sha256']
                and fit_done['fit_sha256'] == sha(stem / 'fit.json')
                and fit_done['source_episodes_sha256'] == sha(stem / 'source-episodes.json')
                and fit_done['new_model_calls'] == fit_done['new_native_calls'] == 0 and fit_done['scoring_calls'] == 3,
                'Calibration fit receipt differs')
        require(source['scope'] == 'calibration_training_only'
                and source['prior_completed_sha256'] == recipe['lineage']['controller_completed']['sha256']
                and len(source['episodes']) == 64, 'Calibration source role/coverage differs')
        episodes = []
        for index, (binding, case) in enumerate(zip(source['episodes'], old_inputs['evaluation'], strict=True)):
            relative = f'controllers/{name}/C/episodes/{index:03d}.npz'
            npz = member(source_dir, relative); receipt_path = npz.with_suffix('.json'); receipt = read(receipt_path)
            require(binding == {'index': index, 'seed': case['seed'], 'npz_path': relative,
                'npz_sha256': sha(npz), 'receipt_sha256': sha(receipt_path), 'layout_sha256': receipt['layout_sha256']},
                'Calibration episode selection/binding differs')
            require(receipt['index'] == index and receipt['seed'] == case['seed']
                    and receipt['checkpoint_sha256'] == maps[name]['checkpoint_sha256'], 'Calibration episode fit/case differs')
            prior.audit_episode(npz, receipt, policy='C', controller=name)
            with np.load(npz, allow_pickle=False) as arrays:
                ep = dict(zip(('targets', 'target_mask', 'ages'), public_targets(arrays['observations']), strict=True),
                          raw_probabilities=arrays['raw_probabilities'])
            episodes.append(ep)
        require(source['public_dataset_sha256'] == fit_done['public_dataset_sha256'] == public_dataset_hash(episodes),
                'Calibration causal tensor identity differs')
        result = validate_scalar_fit(episodes, fitted)
        require(entry['beta'] == result['beta'] and entry['temperature'] == result['temperature']
                and fit_done['oracle_evaluations'] == result['oracle_evaluations'], 'Calibration result binding differs')
        result['scores'] = {}
        require(set(fitted['calibration_scores']) == set(POLICIES), 'Missing calibration score policy')
        for policy in POLICIES:
            score = proper_scores(episodes, policy, entry['beta'] if policy == 'temperature' else None)
            compare_scores(score, fitted['calibration_scores'][policy])
            result['scores'][policy] = score
        same_number(result['baseline_nll'], result['scores']['baseline']['all']['nll'], 'Fitting/score hierarchy differs')
        same_number(result['calibrated_nll'], result['scores']['temperature']['all']['nll'], 'Fitting/score hierarchy differs')
        verified[name] = result
        oracles += result['oracle_evaluations']; queries += result['counts']['query_cards']
        fit_seconds += finite(fit_done['wall_seconds'], minimum=0)
    require(done['source_episodes'] == 1152 and done['fitted_parameters'] == 18 and done['scoring_calls'] == 54
            and done['oracle_evaluations'] == oracles <= 1224 and done['source_query_occurrences'] == queries,
            'Calibration work/coverage differs')
    require(0 < finite(done['wall_seconds']) <= 900 and fit_seconds <= done['wall_seconds'] + 1e-5, 'Calibration timing/cap differs')
    payload_bytes = sum(b['bytes'] for b in done['files'].values())
    require(done['payload_bytes'] == payload_bytes and payload_bytes + (folder / 'completed.json').stat().st_size <= 200_000_000,
            'Calibration bytes/cap differs')
    return done, verified


def report(root, protocol, expected_protocol_sha256, inputs, expected_inputs_sha256, bindings,
           expected_bindings_sha256, checkpoint_map, expected_checkpoint_map_sha256,
           calibration, expected_calibration_completed_sha256,
           evaluation, expected_evaluation_completed_sha256, out):
    begin = time.monotonic()
    root, calibration, evaluation, out = (Path(p).resolve() for p in (root, calibration, evaluation, out))
    require(not out.is_relative_to(evaluation) and not out.is_relative_to(calibration), 'Report must be outside sealed inputs')
    out.mkdir(parents=True, exist_ok=False)
    source_sha = sha(__file__)
    paths = {k: Path(v).resolve() for k, v in {'protocol': protocol, 'inputs': inputs, 'bindings': bindings,
                                             'checkpoint_map': checkpoint_map}.items()}
    expected = {'protocol': expected_protocol_sha256, 'inputs': expected_inputs_sha256,
                'bindings': expected_bindings_sha256, 'checkpoint_map': expected_checkpoint_map_sha256}
    try:
        recipe, cases, maps, source, source_dir, old_inputs = authenticate_inputs(root, paths, expected)
        calibrated, calibration_report = audit_calibration(calibration, expected_calibration_completed_sha256,
                                                           recipe, expected, maps, source_dir, old_inputs)
        complete = authenticate_tree(evaluation, expected_evaluation_completed_sha256, expected_members())
        require(complete['version'] == 'card-calibration-v1' and complete['controllers'] == 20
                and complete['episodes'] == 3584 and complete['policies'] == list(POLICIES)
                and complete['tracker_policy'] == 'C' and complete['calibration_completed_sha256'] == expected_calibration_completed_sha256
                and complete['model_restores'] == 18 and complete['new_fits'] == complete['new_optimizer_steps'] == 0
                and complete['native_replay_calls'] == 0 and complete['lineage'] == recipe['lineage'], 'Evaluation coverage/scope differs')
        started, ready = read(evaluation / 'started.json'), read(evaluation / 'all-fits-ready.json')
        require(sha(evaluation / 'started.json') == complete['started_sha256']
                and started['status'] == 'started' and started['automatic_retry'] is False
                and started['evaluation'] == recipe['evaluation'] and started['tracker_policy'] == 'C'
                and started['calibration_completed_sha256'] == expected_calibration_completed_sha256, 'Evaluation start differs')
        for key, value in expected.items():
            require(complete[key + '_sha256'] == started[key + '_sha256'] == value, 'Evaluation provenance differs')
        require(ready['status'] == 'complete' and ready['fits'] == maps
                and ready['calibration_completed_sha256'] == expected_calibration_completed_sha256
                and ready['calibration_fits'] == calibrated['fits'], 'Pre-evaluation checkpoint/calibration boundary differs')
        require(set(complete['restored_models']) == set(NAMES), 'Incomplete restored checkpoint coverage')
        for name, item in complete['restored_models'].items():
            fit = read(member(root, maps[name]['completed_path']))
            require(item['checkpoint_sha256'] == maps[name]['checkpoint_sha256']
                    and item['weights_sha256'] == fit['final_weights_sha256'], 'Restored weights differ')
        per_fit, scores = {p: {} for p in POLICIES}, {}
        layouts = [np.full(52, -1, np.int64) for _ in range(64)]
        layout_hashes, counters, shared_restores = {}, {}, {}
        array_bytes = 0
        for name in (*NAMES, *REFERENCES):
            entry = calibrated['fits'].get(name)
            baseline_prefixes = []
            for policy in POLICIES if name in NAMES else ('baseline',):
                beta = entry['beta'] if policy == 'temperature' else None
                folder = evaluation / 'controllers' / name / policy
                rows, receipts = [], []
                for index, case in enumerate(cases['evaluation']):
                    stem = folder / 'episodes' / f'{index:03d}'
                    receipt = read(stem.with_suffix('.json'))
                    require(receipt['index'] == index and receipt['seed'] == case['seed']
                            and receipt['protocol_sha256'] == expected['protocol'] and receipt['inputs_sha256'] == expected['inputs']
                            and receipt['checkpoint_sha256'] == (maps[name]['checkpoint_sha256'] if name in maps else None)
                            and receipt['calibration_completed_sha256'] == expected_calibration_completed_sha256
                            and receipt['calibration_receipt_sha256'] == (entry['receipt_sha256'] if entry else None),
                            'Episode seed/weight/calibration binding differs')
                    item = audit_episode(stem.with_suffix('.npz'), receipt, policy=policy, controller=name, beta=beta)
                    prefix = item.pop('score_episode')
                    if policy == 'baseline' and name in NAMES:
                        baseline_prefixes.append(prefix)
                    prior.merge_public_layout(layouts[index], item.pop('layout'))
                    known = layout_hashes.setdefault(str(index), receipt['layout_sha256'])
                    require(known == receipt['layout_sha256'], 'Native deck identity changed across comparisons')
                    for key, value in item['counts'].items(): counters[key] = counters.get(key, 0) + value
                    array_bytes += receipt['array_bytes']
                    component = math.fsum(finite(receipt[k], minimum=0) for k in
                                         ('wall_seconds', 'construction_seconds', 'close_seconds', 'serialization_seconds'))
                    require(component <= finite(receipt['whole_episode_seconds'], minimum=0) + 1e-5, 'Episode timer scopes overlap/exceed whole')
                    rows.append(item); receipts.append(receipt)
                row_done = read(folder / 'completed.json')
                require(row_done['status'] == 'complete' and row_done['controller'] == name and row_done['policy'] == policy
                        and row_done['episodes'] == 64 and row_done['tracker_policy'] == 'C' and row_done['beta'] == beta
                        and row_done['calibration_completed_sha256'] == expected_calibration_completed_sha256
                        and row_done['calibration_receipt_sha256'] == (entry['receipt_sha256'] if entry else None), 'Policy row completion differs')
                returns, success = [r['return'] for r in rows], [r['success'] for r in rows]
                steps = sum(r['native_steps'] for r in rows)
                seconds = math.fsum(r['whole_episode_seconds'] for r in receipts)
                same_number(row_done['mean_return'], mean(returns), 'Policy native mean differs')
                same_number(row_done['whole_episode_seconds'], seconds, 'Policy episode timing differs')
                require(row_done['successes'] == sum(success) and row_done['native_steps'] == steps, 'Policy native coverage differs')
                restore = shared_restores.setdefault(name, row_done['controller_shared_restore_seconds'])
                require(restore == row_done['controller_shared_restore_seconds'], 'Shared restore time differs')
                per_fit[policy][name] = {'mean_return': mean(returns), 'per_seed_returns': returns,
                    'per_seed_successes': success, 'native_steps': steps, 'wall_seconds': seconds,
                    'beta': beta, 'mean_positions_discovered': mean(r['unique_positions'] for r in rows),
                    'repeat_selections': sum(r['native_steps'] - r['unique_positions'] for r in rows),
                    'repeated_mismatching_pairs': sum(r['repeat_mismatching_pairs'] for r in rows),
                    'picker_counts': {k: max(r['picker_counts'][k] for r in rows) if k == 'max_rowmass_abs'
                        else sum(r['picker_counts'][k] for r in rows) for k in rows[0]['picker_counts']},
                    'component_timings': {k: math.fsum(r['timings'][k] for r in rows) for k in rows[0]['timings']}}
            if name in NAMES:
                require(len(baseline_prefixes) == 64, 'Fixed fresh baseline prefix coverage differs')
                scores[name] = {p: proper_scores(baseline_prefixes, p, entry['beta'] if p == 'temperature' else None) for p in POLICIES}
        require(counters == complete['counts'], 'Aggregate attempted/returned work differs')
        require(counters['reset_returned'] == counters['identity_read_returned'] == 3584
                and counters['init_returned'] == 3456 and counters['native_returned'] <= 372736
                and counters['predict_returned'] == counters['write_returned'] <= 359424
                and counters['transform_returned'] == counters['decision_returned'] == counters['native_returned'], 'Coverage/work cap differs')
        require(array_bytes == complete['uncompressed_array_bytes'], 'Array payload count differs')
        require(layout_hashes == complete['layout_sha256_by_case'], 'Paired layout completion differs')
        for i, layout in enumerate(layouts):
            require(prior.layout_digest(layout) == layout_hashes[str(i)], 'Public-reconstructed layout digest differs')
        require(len(set(layout_hashes.values())) == complete['distinct_layouts'], 'Distinct layout count differs')
        payload_bytes = sum(b['bytes'] for b in complete['files'].values())
        require(payload_bytes == complete['payload_bytes'] and payload_bytes + (evaluation / 'completed.json').stat().st_size <= 6_000_000_000,
                'Evaluation output cap differs')
        require(0 < finite(complete['wall_seconds']) <= 1800, 'Evaluation elapsed cap exceeded')
        episode_seconds = math.fsum(r['wall_seconds'] for rows in per_fit.values() for r in rows.values())
        restore_seconds = math.fsum(finite(v, minimum=0) for v in shared_restores.values())
        controller_seconds = math.fsum(finite(v, minimum=0) for v in complete['controller_wall_seconds'].values())
        require(set(complete['controller_wall_seconds']) == set(NAMES) | set(REFERENCES)
                and episode_seconds + restore_seconds <= controller_seconds + 1e-5
                and controller_seconds <= complete['wall_seconds'] + 1e-5, 'Sequential cost reconciliation differs')
        families, contrasts, transfer, gate = aggregate(per_fit, scores)
        summary = {'status': 'complete', 'version': 'card-calibration-report-v1', 'continuation_gate': gate,
            'families': families, 'per_fit': per_fit, 'contrasts': contrasts,
            'references': {n: per_fit['baseline'][n] for n in REFERENCES}, 'references_evaluated_once': True,
            'fresh_baseline_prefix_scores': scores, 'transfer': transfer, 'calibration_training': calibration_report,
            'original_architecture_gate': {'passed': False, 'checks_passed': 1, 'total_checks': 6, 'unchanged': True},
            'coverage': {'learned_fits': 18, 'learned_native_games': 3456, 'reference_games': 128, 'native_games': 3584,
                'native_rows': 56, 'calibration_training_histories': 1152, 'fixed_fresh_baseline_histories': 1152,
                'evaluation_payloads': 7226, 'payload_bytes': payload_bytes, 'counts': counters,
                'distinct_layouts': complete['distinct_layouts'], 'public_reconstructed_layouts': 64},
            'costs': {'calibration_wall_seconds': calibrated['wall_seconds'], 'calibration_oracle_evaluations': calibrated['oracle_evaluations'],
                'evaluation_wall_seconds': complete['wall_seconds'], 'episode_seconds_nested': episode_seconds,
                'shared_restore_seconds_counted_once': restore_seconds, 'controller_seconds_nested': controller_seconds,
                'final_hashing_seconds_nested': complete['final_hashing_seconds'],
                'scope': 'Actual instrumented wall time on a shared host; transformations, C diagnostics, native actions and serialization are included. Different trajectory lengths and transformation arithmetic preclude intrinsic equal-work speed claims. Per-policy restore metadata is repeated and counted once per controller.'},
            'bindings': {**{k + '_sha256': v for k, v in expected.items()},
                'calibration_completed_sha256': expected_calibration_completed_sha256,
                'evaluation_completed_sha256': expected_evaluation_completed_sha256,
                'source_sha256': source_sha, 'bound_source_count': len(source['files'])},
            'layout_sha256_by_case': layout_hashes, 'new_model_calls': 0, 'new_native_calls': 0,
            'limits': ['Previously exposed 64 C decks are declared calibration training, never new test evidence.',
                'Fresh proper scores use exactly baseline C histories and causal seen-hidden labels for every transform; native trajectories may diverge.',
                'Hierarchy: equal fits, equal eligible episodes, equal nonempty boundaries, equal eligible queries; not a pooled-query score.',
                'Hard wrong labels have infinite NLL, represented by null plus flags/counts; no smoothing or omitted errors.',
                'Finite-beta log-domain likelihood distinguishes true zero support from floating exponent underflow.',
                'All 18 inherited fits retained; no new neural weights or new memory architecture.',
                'Exact and last32 each run once per deck, not replicated across belief policies.',
                'Learned outputs are authenticated saved probabilities; this audit performs no checkpoint forward or native simulation.',
                'Five jointly required conditions are not twelve independent statistical tests. The original architecture gate remains failed, 1/6.']}
        write(out / 'summary.json', summary)
        write_report(summary, out / 'report.md')
        authenticate_inputs(root, paths, expected)
        authenticate_tree(calibration, expected_calibration_completed_sha256)
        authenticate_tree(evaluation, expected_evaluation_completed_sha256, expected_members())
        require(sha(__file__) == source_sha, 'Reporter changed during audit')
        result = {'status': 'complete', 'version': 'card-calibration-report-v1',
            **summary['bindings'], 'continuation_passed': gate['passed'], 'checks_passed': gate['checks_passed'],
            'total_checks': gate['total_checks'], 'original_gate_unchanged': True,
            'new_model_calls': 0, 'new_native_calls': 0, 'wall_seconds': time.monotonic() - begin,
            'files': {n: {'sha256': sha(out / n), 'bytes': (out / n).stat().st_size} for n in ('summary.json', 'report.md')}}
        write(out / 'receipt.json', result)
        return result
    except BaseException as error:
        try:
            write(out / 'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.monotonic() - begin})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original audit error.
            if callable(getattr(error, 'add_note', None)): error.add_note(f'Failure receipt failed: {secondary!r}')
        raise


def write_report(data, path):
    gate = data['continuation_gate']
    lines = ['# Card calibration transfer and native control', '',
        f"Temperature continuation: **{'PASS' if gate['passed'] else 'FAIL'}**, {gate['checks_passed']}/{gate['total_checks']} expanded checks across five jointly required conditions.", '',
        'The former 64 C evaluation decks are calibration training for this explicitly new study. Eighteen scalar temperatures are frozen before 64 fresh paired decks. Neural weights and C selection rules are unchanged. Original architecture result remains FAIL, 1/6.', '',
        '| Policy | Family | Mean native return | Successes |', '|---|---|---:|---:|']
    for policy in POLICIES:
        for family in MODES:
            row = data['families'][policy][family]
            lines.append(f"| {policy} | {family} | {row['mean_return']:.6f} | {row['successes']}/{row['episodes']} |")
    for name, row in data['references'].items():
        lines.append(f"| shared reference, evaluated once | {name} | {row['mean_return']:.6f} | {sum(row['per_seed_successes'])}/64 |")
    lines += ['', '| Fit | Beta | Temperature minus baseline | Hard minus baseline |', '|---|---:|---:|---:|']
    for name in NAMES:
        a, b = (data['contrasts'][key]['per_fit'][name]['mean_difference'] for key in ('temperature_minus_baseline', 'hard_minus_baseline'))
        lines.append(f"| {name} | {data['calibration_training'][name]['beta']:.6g} | {a:+.6f} | {b:+.6f} |")
    lines += ['', 'Proper scores below use identical saved fresh baseline prefixes, before current-visibility/unseen overrides. They average queries within a boundary, nonempty boundaries within an episode, episodes within a fit, and all 18 fits equally.', '',
        '| Belief | Hierarchical NLL | Hierarchical Brier | Top-rank accuracy | Infinite NLL queries |', '|---|---:|---:|---:|---:|']
    for policy, row in data['transfer'].items():
        nll = 'infinite' if row['nll_is_infinite'] else f"{row['nll']:.6f}"
        lines.append(f"| {policy} | {nll} | {row['brier']:.6f} | {row['accuracy']:.4%} | {row['infinite_nll_queries']} |")
    lines += ['', '| Required check | Value | Threshold | Result |', '|---|---:|---:|---|']
    for c in gate['checks']:
        lines.append(f"| {c['name']} | {c['value']} | {c['comparison']} {c['threshold']} | {'PASS' if c['passed'] else 'FAIL'} |")
    cost = data['costs']
    lines += ['', f"Scalar calibration: {cost['calibration_wall_seconds']:.3f}s, {cost['calibration_oracle_evaluations']} objective/derivative evaluations. Fresh native evaluation: {cost['evaluation_wall_seconds']:.3f}s, {data['coverage']['counts']['native_returned']:,} actions across 3,584 games. No new neural fit.",
        '', cost['scope'], '', *data['limits'], '']
    with Path(path).open('x') as handle: handle.write('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'protocol', 'inputs', 'bindings', 'checkpoint_map', 'calibration', 'evaluation', 'out'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    for name in ('protocol', 'inputs', 'bindings', 'checkpoint_map', 'calibration_completed', 'evaluation_completed'):
        parser.add_argument('--expected-' + name.replace('_', '-') + '-sha256', required=True)
    print(json.dumps(report(**vars(parser.parse_args())), indent=2))
