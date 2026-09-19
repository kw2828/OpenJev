"""Saved-output audit for three frozen card controllers; no models or native calls.

The auditor checks public observation/action consistency and picker arithmetic.
It does not independently recompute learned probabilities from checkpoints.
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

MODES = ('delta', 'gated_delta', 'kalman', 'innovation_local', 'innovation_matched', 'gru')
REFERENCES = ('exact', 'last32')
POLICIES = ('A', 'B', 'C')
NAMES = tuple(f'{mode}-pair{i}' for mode in MODES for i in range(3)) + REFERENCES
FAMILIES = (*MODES, *REFERENCES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def member(folder, relative):
    name = Path(relative)
    require(not name.is_absolute() and '..' not in name.parts, 'Unsafe artifact path')
    result = (Path(folder) / name).resolve()
    require(result.is_relative_to(Path(folder).resolve()), 'Artifact escapes its root')
    return result


def finite(value, *, minimum=None):
    require(type(value) in (float, int) and math.isfinite(value), 'Expected finite number')
    require(minimum is None or value >= minimum, 'Number below allowed minimum')
    return float(value)


def mean(values):
    values = list(values)
    require(bool(values), 'Empty mean')
    return math.fsum(finite(v) for v in values) / len(values)


def difference(left, right):
    """Canonical decimal difference; the inclusive .03 threshold is unchanged."""
    finite(left)
    finite(right)
    return Decimal(str(left)) - Decimal(str(right))


def same_number(left, right, message):
    require(math.isclose(finite(left), finite(right), rel_tol=0, abs_tol=1e-12), message)


def aggregate(per_fit):
    """All 60 controller rows, all 64 paired decks and all 18 learned fits."""
    require(set(per_fit) == set(POLICIES), 'All three policies required')
    for policy in POLICIES:
        require(set(per_fit[policy]) == set(NAMES), 'All models/references required')
    families, contrasts = {}, {}
    for policy in POLICIES:
        families[policy] = {}
        for family in FAMILIES:
            names = [f'{family}-pair{i}' for i in range(3)] if family in MODES else [family]
            for name in names:
                row = per_fit[policy][name]
                require(len(row['per_seed_returns']) == len(row['per_seed_successes']) == 64,
                        'Exactly 64 decks required')
                require(all(type(v) is bool for v in row['per_seed_successes']), 'Success flags must be bool')
                same_number(row['mean_return'], mean(row['per_seed_returns']), 'Fit mean mismatch')
            returns = [v for name in names for v in per_fit[policy][name]['per_seed_returns']]
            successes = sum(sum(per_fit[policy][name]['per_seed_successes']) for name in names)
            seconds = math.fsum(finite(per_fit[policy][name]['wall_seconds'], minimum=0) for name in names)
            steps = sum(per_fit[policy][name]['native_steps'] for name in names)
            require(type(steps) is int and steps > 0, 'Positive native work required')
            families[policy][family] = {
                'fit_names': names, 'fit_mean_returns': [per_fit[policy][n]['mean_return'] for n in names],
                'mean_return': mean(returns), 'successes': successes, 'episodes': len(returns),
                'success_rate': successes / len(returns), 'native_steps': steps,
                'whole_controller_seconds': seconds, 'whole_controller_ms_per_action': seconds * 1000 / steps,
            }
    for label, first, second in (('B_minus_A', 'A', 'B'), ('C_minus_B', 'B', 'C')):
        by_fit = {}
        for name in NAMES:
            a, b = per_fit[first][name], per_fit[second][name]
            values = [float(difference(y, x)) for x, y in zip(a['per_seed_returns'], b['per_seed_returns'], strict=True)]
            by_fit[name] = {'mean_return_difference': float(difference(b['mean_return'], a['mean_return'])),
                            'paired_deck_return_differences': values,
                            'nonworse_decks': sum(v >= 0 for v in values), 'positive_decks': sum(v > 0 for v in values)}
        by_family = {}
        for family in FAMILIES:
            names = families[first][family]['fit_names']
            values = [by_fit[n]['mean_return_difference'] for n in names]
            by_family[family] = {'mean_return_difference': float(difference(families[second][family]['mean_return'],
                                                                           families[first][family]['mean_return'])),
                                 'paired_fit_differences': values, 'positive_fits': sum(v > 0 for v in values),
                                 'nonworse_fits': sum(v >= 0 for v in values)}
        contrasts[label] = {'per_fit': by_fit, 'families': by_family,
                            'learned_mean_difference': float(difference(
                                mean(per_fit[second][n]['mean_return'] for n in NAMES if n not in REFERENCES),
                                mean(per_fit[first][n]['mean_return'] for n in NAMES if n not in REFERENCES)))}
    change = contrasts['C_minus_B']
    checks = [{'name': 'learned_mean_C_minus_B_at_least_0.03',
               'value': change['learned_mean_difference'], 'threshold': .03,
               'comparison': '>=', 'passed': Decimal(str(change['learned_mean_difference'])) >= Decimal('.03')}]
    for family in MODES:
        row = change['families'][family]
        checks.append({'name': family + '_mean_nonnegative', 'value': row['mean_return_difference'],
                       'threshold': 0, 'comparison': '>=', 'passed': row['mean_return_difference'] >= 0})
    for family in MODES:
        row = change['families'][family]
        checks.append({'name': family + '_at_least_two_positive_fits', 'value': row['positive_fits'],
                       'threshold': 2, 'comparison': '>=', 'passed': row['positive_fits'] >= 2})
    gate = {'passed': all(c['passed'] for c in checks), 'checks_passed': sum(c['passed'] for c in checks),
            'total_checks': len(checks), 'checks': checks, 'references_in_primary_gate': False,
            'B_minus_A_is_descriptive_only': True,
            'grouped_requirements': {'overall_mean_margin': checks[0]['passed'],
              'all_family_means_nonnegative': all(c['passed'] for c in checks[1:7]),
              'all_families_at_least_two_positive_fits': all(c['passed'] for c in checks[7:])},
            'interpretation': 'Three jointly required conditions expanded into thirteen check rows, not independent statistical tests'}
    return families, contrasts, gate


def audit_episode(path, receipt, *, policy, controller):
    """Replay public bookkeeping only; learned raw probabilities remain supplied."""
    from openjev.research.card_memory_picker import CardPolicyTracker
    from openjev.research.card_memory_task import PublicTeacher

    require(receipt['status'] == 'complete' and receipt['policy'] == policy
            and receipt['controller'] == controller, 'Episode identity/status mismatch')
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
    tracker, teacher = CardPolicyTracker(policy), PublicTeacher()
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
        action, picker, diagnostic = tracker.decision(raw)
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
        counts['C_differs_from_B'] += int(policy == 'C' and action != diagnostic['Baction_ifC'])
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
            'layout': layout, 'counts': work, 'timings': receipt['timings']}


def authenticate_checkpoints(root, checkpoint_map):
    require(set(checkpoint_map) == set(NAMES) - set(REFERENCES), 'All eighteen inherited checkpoints required')
    for name, entry in checkpoint_map.items():
        require(set(entry) == {'mode', 'pair', 'checkpoint_path', 'checkpoint_sha256',
                               'completed_path', 'completed_sha256'}, 'Unexpected checkpoint-map schema')
        require(entry['mode'] in MODES and type(entry['pair']) is int and entry['pair'] in range(3)
                and name == f"{entry['mode']}-pair{entry['pair']}", 'Checkpoint mode/pair identity mismatch')
        checkpoint, receipt = (member(root, entry[k]) for k in ('checkpoint_path', 'completed_path'))
        require(checkpoint.name == 'final-checkpoint.pt' and receipt.name == 'completed.json'
                and checkpoint.parent == receipt.parent, 'Unexpected fit paths')
        require(sha(checkpoint) == entry['checkpoint_sha256'] and sha(receipt) == entry['completed_sha256'],
                'Inherited checkpoint/receipt changed')
        fitted = read(receipt)
        require(fitted['status'] == 'complete' and fitted['evaluation_calls'] == 0
                and fitted['counts']['successful_updates'] == fitted['recipe']['expected_updates'] == 128
                and fitted['counts']['completed_epochs'] == fitted['recipe']['epochs'] == 16,
                'Incomplete inherited fit')
        require({p.name for p in receipt.parent.iterdir() if p.is_file()}
                == set(fitted['files']) | {'completed.json'}, 'Inherited fit membership mismatch')
        for name, binding in fitted['files'].items():
            path = member(receipt.parent, name)
            require(sha(path) == binding['sha256'] and path.stat().st_size == binding['bytes'],
                    'Inherited fit artifact changed')


def expected_members():
    names = {'started.json', 'all-fits-ready.json'}
    for name in NAMES:
        for policy in POLICIES:
            stem = f'controllers/{name}/{policy}'
            names.add(stem + '/completed.json')
            for index in range(64):
                names.update(f'{stem}/episodes/{index:03d}.{suffix}' for suffix in ('json', 'npz'))
    return names


def authenticate_evaluation(evaluation, expected_completed_sha256):
    require(sha(evaluation / 'completed.json') == expected_completed_sha256, 'External evaluation receipt mismatch')
    receipt = read(evaluation / 'completed.json')
    require(receipt['status'] == 'complete', 'Evaluation not complete')
    expected = expected_members()
    require(set(receipt['files']) == expected, 'Wrong declared evaluation member set')
    actual = {str(p.relative_to(evaluation)) for p in evaluation.rglob('*') if p.is_file()}
    require(actual == expected | {'completed.json'}, 'Unexpected/missing evaluation artifacts')
    for name, binding in receipt['files'].items():
        path = member(evaluation, name)
        require(sha(path) == binding['sha256'] and path.stat().st_size == binding['bytes'],
                'Evaluation member changed: ' + name)
    return receipt


def _seed_inventory(value):
    if isinstance(value, dict):
        return {v for k, v in value.items() if k in ('seed', 'behavior_seed') and type(v) is int} | set().union(
            *(_seed_inventory(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_seed_inventory(v) for v in value))
    return set()


def authenticate_inputs(root, paths, expected):
    import importlib.metadata
    import platform
    import sys

    for key, path in paths.items():
        require(Path(path).resolve().is_relative_to(root) and sha(path) == expected[key], 'External input hash mismatch: ' + key)
    protocol, inputs, bindings, checkpoints = (read(paths[k]) for k in ('protocol', 'inputs', 'bindings', 'checkpoint_map'))
    evaluation = {'controllers': 20, 'episodes': 64, 'max_actions': 104, 'policies': list(POLICIES),
                  'wall_cap_seconds': 1800, 'output_cap_bytes': 6_000_000_000}
    require(protocol['version'] == inputs['version'] == bindings['version'] == 'card-controllers-v1'
            and protocol['evaluation'] == evaluation and protocol['tie_tolerance'] == 1e-12,
            'Frozen evaluation settings mismatch')
    criterion = protocol['continuation']
    require(criterion['contrast'] == 'C_minus_B' and criterion['all_required'] is True
            and criterion['minimum_overall_learned_mean_return_difference'] == .03
            and criterion['every_family_mean_nonnegative'] is True
            and criterion['minimum_positive_paired_fits_per_family'] == 2
            and criterion['not_a_retest_of_original_architecture_gate'] is True,
            'Frozen continuation requirements mismatch')
    require(member(root, protocol['bindings_file']) == Path(paths['bindings']).resolve(), 'Source binding path mismatch')
    required = {'old_bindings', 'old_protocol', 'old_inputs', 'training_completed', 'training_boundary',
                'independent_review', 'checkpoint_map'}
    require(set(protocol['lineage']) == required, 'Lineage member mismatch')
    lineage = {}
    for key, item in protocol['lineage'].items():
        require(set(item) == {'path', 'sha256'}, 'Invalid lineage descriptor')
        path = member(root, item['path'])
        require(sha(path) == item['sha256'], 'Inherited lineage changed: ' + key)
        lineage[key] = read(path)
    require(protocol['lineage']['checkpoint_map']['sha256'] == expected['checkpoint_map'], 'Inherited map differs')
    actual_runtime = {'python': platform.python_version(), 'executable': sys.executable,
                      'machine': platform.machine(), 'platform': platform.platform(),
                      'numpy': importlib.metadata.version('numpy'), 'torch': importlib.metadata.version('torch'),
                      'gymnasium': importlib.metadata.version('gymnasium')}
    for source in (lineage['old_bindings'], bindings):
        require(source['runtime'] == actual_runtime, 'Runtime differs from source binding')
        for name, digest in source['files'].items():
            require(sha(member(root, name)) == digest, 'Bound source changed: ' + name)
    require(len(lineage['old_bindings']['files']) == 78, 'Original source coverage changed')
    for key in ('protocol', 'inputs'):
        require(bindings['files'].get(str(Path(paths[key]).resolve().relative_to(root))) == expected[key],
                'Protocol/inputs not source-bound')
    train, boundary = lineage['training_completed'], lineage['training_boundary']
    require(train['status'] == boundary['status'] == 'complete' and train['fits'] == 18 and train['updates'] == 2304
            and train['native_evaluation_calls'] == boundary['development_or_native_evaluations'] == 0,
            'Incomplete original training boundary')
    require(train['checkpoint_map_sha256'] == boundary['checkpoint_map_sha256'] == expected['checkpoint_map']
            and train['training_completed_sha256'] == protocol['lineage']['training_boundary']['sha256'],
            'Training/map boundary differs')
    train_path = member(root, protocol['lineage']['training_completed']['path'])
    require(sha(train_path.with_name('started.json')) == train['started_sha256'], 'Training provenance changed')
    require(len(boundary['fits']) == 18 and {f['id'] for f in boundary['fits']} == set(checkpoints), 'Original fit membership differs')
    for fitted in boundary['fits']:
        require(all(fitted.get(k) == v for k, v in checkpoints[fitted['id']].items()), 'Original fit identity differs')
    old_review = lineage['independent_review']
    require(old_review['status'] == 'complete' and old_review['continuation_passed'] is False
            and old_review['checks_passed'] == 1 and old_review['total_checks'] == 6, 'Original failed gate changed')
    for field, key in (('training_completed_sha256', 'training_completed'), ('source_bindings_sha256', 'old_bindings'),
                       ('protocol_sha256', 'old_protocol')):
        require(old_review['bindings'][field] == protocol['lineage'][key]['sha256'], 'Original review lineage differs')
    cases = inputs['evaluation']
    require(len(cases) == 64, 'Exactly 64 cases required')
    for i, case in enumerate(cases):
        require(set(case) == {'seed', 'policy_order'} and type(case['seed']) is int and 0 <= case['seed'] < 2**32
                and case['policy_order'] == list(POLICIES[i % 3:] + POLICIES[:i % 3]), 'Case pairing/order differs')
    seeds = {c['seed'] for c in cases}
    require(len(seeds) == 64 and seeds.isdisjoint(_seed_inventory(lineage['old_inputs']) | {0, 410}),
            'Fresh seed identity collision')
    authenticate_checkpoints(root, checkpoints)
    return protocol, inputs, checkpoints, bindings


def merge_public_layout(current, observation):
    require(current.shape == observation.shape == (52,), 'Invalid public layout shape')
    visible = observation >= 0
    require(np.all((current[visible] == -1) | (current[visible] == observation[visible])),
            'Paired cases reveal conflicting deck ranks')
    current[visible] = observation[visible]


def layout_digest(layout):
    require(layout.dtype == np.int64 and layout.shape == (52,) and np.all((layout >= 0) & (layout < 13))
            and np.array_equal(np.bincount(layout, minlength=13), np.full(13, 4)),
            'Incomplete/invalid reconstructed public layout')
    return hashlib.sha256(b'card-layout-int64le-v1\0' + np.asarray(layout, dtype='<i8').tobytes()).hexdigest()


def report(root, protocol, expected_protocol_sha256, inputs, expected_inputs_sha256, bindings,
           expected_bindings_sha256, checkpoint_map, expected_checkpoint_map_sha256,
           evaluation, expected_completed_sha256, out):
    begin = time.monotonic()
    root, evaluation, out = Path(root).resolve(), Path(evaluation).resolve(), Path(out).resolve()
    require(not out.is_relative_to(evaluation), 'Audit output must be outside completed evaluation')
    out.mkdir(parents=True, exist_ok=False)
    paths = {k: Path(v) for k, v in {'protocol': protocol, 'inputs': inputs, 'bindings': bindings,
                                    'checkpoint_map': checkpoint_map}.items()}
    expected = {'protocol': expected_protocol_sha256, 'inputs': expected_inputs_sha256,
                'bindings': expected_bindings_sha256, 'checkpoint_map': expected_checkpoint_map_sha256}
    try:
        recipe, cases, maps, source = authenticate_inputs(root, paths, expected)
        complete = authenticate_evaluation(evaluation, expected_completed_sha256)
        require(complete['controllers'] == 20 and complete['episodes'] == 3840 and complete['policies'] == list(POLICIES)
                and complete['model_restores'] == 18 and complete['new_fits'] == complete['new_optimizer_steps'] == 0
                and complete['native_replay_calls'] == 0 and complete['lineage'] == recipe['lineage'],
                'Execution coverage/lineage differs')
        for key, value in expected.items():
            require(complete[key + '_sha256'] == value, 'Execution input identity differs')
        require(sha(evaluation / 'started.json') == complete['started_sha256'], 'Execution start changed')
        started = read(evaluation / 'started.json')
        require(started['automatic_retry'] is False and started['evaluation'] == recipe['evaluation'],
                'Unexpected retry/cap settings')
        for key, value in expected.items():
            require(started[key + '_sha256'] == value, 'Execution start input differs')
        ready = read(evaluation / 'all-fits-ready.json')
        require(ready['status'] == 'complete' and ready['fits'] == maps, 'Restoration boundary differs')
        require(set(complete['restored_models']) == set(maps), 'Missing model restorations')
        for name, item in complete['restored_models'].items():
            fit = read(member(root, maps[name]['completed_path']))
            require(item['checkpoint_sha256'] == maps[name]['checkpoint_sha256']
                    and item['weights_sha256'] == fit['final_weights_sha256'], 'Restored weight identity differs')
        per_fit = {p: {} for p in POLICIES}
        layouts = [np.full(52, -1, dtype=np.int64) for _ in range(64)]
        layout_hashes, counters, shared_restores = {}, {}, {}
        saved_array_bytes = 0
        for name in NAMES:
            for policy in POLICIES:
                stem = evaluation / 'controllers' / name / policy
                row, receipts = [], []
                for index, case in enumerate(cases['evaluation']):
                    receipt = read(stem / 'episodes' / f'{index:03d}.json')
                    require(receipt['index'] == index and receipt['seed'] == case['seed']
                            and receipt['protocol_sha256'] == expected['protocol']
                            and receipt['inputs_sha256'] == expected['inputs']
                            and receipt['checkpoint_sha256'] == (maps[name]['checkpoint_sha256'] if name in maps else None),
                            'Episode case/checkpoint/provenance differs')
                    item = audit_episode(stem / 'episodes' / f'{index:03d}.npz', receipt, policy=policy, controller=name)
                    merge_public_layout(layouts[index], item.pop('layout'))
                    known = layout_hashes.setdefault(str(index), receipt['layout_sha256'])
                    require(known == receipt['layout_sha256'], 'Deck identity differs across paired controllers')
                    for key, value in item['counts'].items():
                        counters[key] = counters.get(key, 0) + value
                    saved_array_bytes += receipt['array_bytes']
                    component = sum(finite(receipt[k], minimum=0) for k in
                                    ('wall_seconds', 'construction_seconds', 'close_seconds', 'serialization_seconds'))
                    require(component <= finite(receipt['whole_episode_seconds'], minimum=0) + 1e-5,
                            'Nested construction/native/storage timing exceeds whole episode')
                    row.append(item)
                    receipts.append(receipt)
                result = read(stem / 'completed.json')
                require(result['status'] == 'complete' and result['controller'] == name and result['policy'] == policy
                        and result['episodes'] == 64, 'Policy completion identity differs')
                returns, successes = [r['return'] for r in row], [r['success'] for r in row]
                steps = sum(r['native_steps'] for r in row)
                same_number(mean(returns), result['mean_return'], 'Policy mean return differs')
                require(sum(successes) == result['successes'] and steps == result['native_steps'], 'Policy success/work differs')
                seconds = math.fsum(r['whole_episode_seconds'] for r in receipts)
                same_number(seconds, result['whole_episode_seconds'], 'Policy elapsed sum differs')
                restored = shared_restores.setdefault(name, result['controller_shared_restore_seconds'])
                require(restored == result['controller_shared_restore_seconds'], 'Shared restore timing differs')
                summary = {'mean_return': mean(returns), 'per_seed_returns': returns, 'per_seed_successes': successes,
                           'native_steps': steps, 'wall_seconds': seconds,
                           'mean_positions_discovered': mean(r['unique_positions'] for r in row),
                           'repeat_selections': sum(r['native_steps'] - r['unique_positions'] for r in row),
                           'repeated_mismatching_pairs': sum(r['repeat_mismatching_pairs'] for r in row),
                           'matched_pairs': sum(r['matched_pairs'] for r in receipts),
                           'picker_counts': {key: (max(r['picker_counts'][key] for r in row) if key == 'max_rowmass_abs'
                                                   else sum(r['picker_counts'][key] for r in row)) for key in row[0]['picker_counts']},
                           'component_timings': {key: math.fsum(r['timings'][key] for r in row) for key in row[0]['timings']}}
                per_fit[policy][name] = summary
        require(counters == complete['counts'], 'Aggregate attempted/returned counters differ')
        require(counters['reset_returned'] == counters['identity_read_returned'] == 3840
                and counters['init_returned'] == 3456 and counters['native_returned'] <= 399360
                and counters['predict_returned'] == counters['write_returned'] <= 359424,
                'Aggregate work exceeds/misses frozen coverage')
        require(saved_array_bytes == complete['uncompressed_array_bytes'], 'Uncompressed payload count differs')
        require(layout_hashes == complete['layout_sha256_by_case'], 'Completion layout identities differ')
        for index, layout in enumerate(layouts):
            require(layout_digest(layout) == layout_hashes[str(index)], 'Public-reconstructed layout digest differs')
        require(len(set(layout_hashes.values())) == complete['distinct_layouts'], 'Distinct deck count differs')
        payload_bytes = sum(b['bytes'] for b in complete['files'].values())
        require(payload_bytes == complete['payload_bytes'] and payload_bytes + (evaluation / 'completed.json').stat().st_size
                <= recipe['evaluation']['output_cap_bytes'], 'Output bytes differ/exceed cap')
        require(0 < finite(complete['wall_seconds']) <= recipe['evaluation']['wall_cap_seconds'], 'Execution exceeded wall cap')
        episode_seconds = math.fsum(row['wall_seconds'] for rows in per_fit.values() for row in rows.values())
        restore_seconds = math.fsum(finite(v, minimum=0) for v in shared_restores.values())
        controller_seconds = math.fsum(finite(v, minimum=0) for v in complete['controller_wall_seconds'].values())
        require(set(complete['controller_wall_seconds']) == set(NAMES)
                and episode_seconds + restore_seconds <= controller_seconds + 1e-5
                and controller_seconds <= complete['wall_seconds'] + 1e-5, 'Sequential wall-time accounting differs')
        families, contrasts, gate = aggregate(per_fit)
        summary = {'status': 'complete', 'version': 'card-controllers-report-v1', 'families': families,
                   'per_fit': per_fit, 'contrasts': contrasts, 'continuation_gate': gate,
                   'original_architecture_gate': {'passed': False, 'checks_passed': 1, 'total_checks': 6, 'unchanged': True},
                   'coverage': {'policies': 3, 'controllers_per_policy': 20, 'episodes': 3840, 'learned_fits': 18,
                                'evaluation_payloads': len(complete['files']), 'payload_bytes': payload_bytes,
                                'distinct_layouts': complete['distinct_layouts'], 'public_reconstructed_layouts': 64,
                                'native_actions': counters['native_returned'], 'counts': counters},
                   'costs': {'execution_wall_seconds': complete['wall_seconds'], 'all_controller_seconds_nested': controller_seconds,
                             'all_episode_seconds_nested': episode_seconds, 'shared_restore_seconds_counted_once': restore_seconds,
                             'final_hashing_seconds_nested': complete['final_hashing_seconds'],
                             'scope': 'Instrumented episodes include construction, reset, identity read, inference, public bookkeeping, native steps, close and NPZ I/O/hash. Shared restores and receipt writes are outer costs. A deliberately pays an additional choice pass for equivalence diagnostics; these are not intrinsic or matched-work speed comparisons.'},
                   'bindings': {**{k + '_sha256': v for k, v in expected.items()},
                                'evaluation_completed_sha256': expected_completed_sha256, 'source_sha256': sha(__file__),
                                'bound_source_count': len(source['files'])},
                   'layout_sha256_by_case': layout_hashes, 'new_native_calls': 0, 'new_model_calls': 0,
                   'claim_limits': ['Three jointly required controller conditions, not thirteen independent significance tests.',
                                    'Sixty-four paired deck draws; trajectories diverge. No claim of 3840 independent decks.',
                                    'B bundles normalization and stable ties. C adds a deterministic exploration heuristic.',
                                    'No new architecture, probability calibration, biological mechanism, or optimal information gain claim.',
                                    'Learned probabilities are authenticated saved outputs; no checkpoint forward replay was performed.',
                                    'Original failed architecture criterion remains failed; this controller test cannot rescue it.']}
        write(out / 'summary.json', summary)
        write_report(summary, out / 'report.md')
        # Bind both endpoints; do not accept mutation during numerical/public replay.
        authenticate_inputs(root, paths, expected)
        authenticate_evaluation(evaluation, expected_completed_sha256)
        receipt = {'status': 'complete', 'version': 'card-controllers-report-v1', 'source_sha256': sha(__file__),
                   'evaluation_completed_sha256': expected_completed_sha256, 'protocol_sha256': expected_protocol_sha256,
                   'continuation_passed': gate['passed'], 'original_gate_unchanged': True,
                   'wall_seconds': time.monotonic() - begin, 'new_model_calls': 0, 'new_native_calls': 0,
                   'files': {name: {'sha256': sha(out / name), 'bytes': (out / name).stat().st_size}
                             for name in ('summary.json', 'report.md')}}
        write(out / 'receipt.json', receipt)
        return receipt
    except BaseException as error:
        try:
            write(out / 'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.monotonic() - begin})
        except BaseException as secondary:  # noqa: BLE001 - original validation failure must remain visible.
            if callable(getattr(error, 'add_note', None)):
                error.add_note(f'Failure receipt also failed: {secondary!r}')
        raise


def write_report(data, path):
    gate = data['continuation_gate']
    lines = ['# Card controller follow-up', '',
             f"Controller continuation: **{'PASS' if gate['passed'] else 'FAIL'}**. Three jointly required conditions expand to {gate['checks_passed']}/{gate['total_checks']} check rows; these are not independent statistical tests.",
             '', 'A is the original picker; B normalizes rows and uses stable numerical ties; C additionally prefers an unseen endpoint within tied first-card pairs. All 18 original fitted models are frozen. The original architecture result remains FAIL, 1/6.', '',
             '| Policy | Family | Mean return | Successes | Episode ms/action |', '| --- | --- | ---: | ---: | ---: |']
    for policy in POLICIES:
        for family in FAMILIES:
            r = data['families'][policy][family]
            lines.append(f"| {policy} | {family} | {r['mean_return']:.6f} | {r['successes']}/{r['episodes']} | {r['whole_controller_ms_per_action']:.3f} |")
    lines += ['', 'All paired fit differences follow. Each fit uses all 64 shared fresh seed identities. References are descriptive and excluded from the controller gate.', '',
              '| Fit/reference | B minus A | C minus B | C minus B positive decks |', '| --- | ---: | ---: | ---: |']
    for name in NAMES:
        first, second = (data['contrasts'][k]['per_fit'][name] for k in ('B_minus_A', 'C_minus_B'))
        lines.append(f"| {name} | {first['mean_return_difference']:+.6f} | {second['mean_return_difference']:+.6f} | {second['positive_decks']}/64 |")
    lines += ['', '| Required check | Value | Threshold | Result |', '| --- | ---: | ---: | --- |']
    for c in gate['checks']:
        lines.append(f"| {c['name']} | {c['value']:.6f} | {c['comparison']} {c['threshold']} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines += ['', f"Whole evaluation: {data['costs']['execution_wall_seconds']:.3f} seconds. {data['coverage']['native_actions']:,} saved native actions checked. All 64 layout hashes were reconstructed from the union of actual public reveals; {data['coverage']['distinct_layouts']} distinct layouts, with any duplicates retained.",
              '', data['costs']['scope'], '', 'Coverage, repeat selections, picker ties, component costs and every paired deck difference are included in summary.json.',
              '', *data['claim_limits'], '']
    with Path(path).open('x') as handle:
        handle.write('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'protocol', 'inputs', 'bindings', 'checkpoint_map', 'evaluation', 'out'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    for name in ('protocol', 'inputs', 'bindings', 'checkpoint_map', 'completed'):
        parser.add_argument('--expected-' + name.replace('_', '-') + '-sha256', required=True)
    print(json.dumps(report(**vars(parser.parse_args())), indent=2))
