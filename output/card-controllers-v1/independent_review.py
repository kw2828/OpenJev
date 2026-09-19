"""Independent saved public-transition/return review, never model/native replay.

Prepared before current outcomes. Invocation requires explicit completion
authorization and the externally supplied final execution-receipt hash.
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

PROTOCOL_SHA = '7cf3b82dad7023c4c3e0ba044dc2283861d70beb60bf56bb6abb86f8cdd025ee'
INPUTS_SHA = '72f1daf2c6717ea6d0b7def3433316284590a721a71841b9134e5a9ea6105939'
BINDINGS_SHA = 'bbd609e4d9542c6fb1e4bc51dd7144347a96f951c6a091a8a509d916c4d5f49d'
MODES = ('delta', 'gated_delta', 'kalman', 'innovation_local', 'innovation_matched', 'gru')
POLICIES = ('A', 'B', 'C')
NAMES = tuple(f'{mode}-pair{i}' for mode in MODES for i in range(3)) + ('exact', 'last32')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def bound(root, name, expected):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid bound path')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and sha(path) == expected, 'Bound bytes changed: ' + str(name))
    return path


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def average(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(v) for v in values), 'Missing/nonfinite arithmetic inputs')
    return math.fsum(values) / len(values)


def delta(left, right):
    return float(Decimal(str(left)) - Decimal(str(right)))


def episode(arrays, receipt):
    """Independent native public-rule reconstruction; no picker/model import."""
    actions, observations = arrays['actions'], arrays['observations']
    n = len(actions)
    require(1 <= n <= 104 and actions.dtype == np.int64 and observations.shape == (n + 1, 52)
            and observations.dtype == np.int64 and np.all(observations[0] == 13), 'Invalid native public prefix')
    require(np.all((observations >= 0) & (observations <= 13)), 'Nonpublic rank/token')
    for key, dtype in (('ranks', np.int64), ('rewards', np.float64), ('terminated', np.bool_), ('truncated', np.bool_)):
        require(arrays[key].shape == (n,) and arrays[key].dtype == dtype and np.isfinite(arrays[key]).all(),
                'Invalid array: ' + key)
    matched, revealed, pending, rewards = np.zeros(52, bool), np.full(52, -1, np.int64), None, []
    for t, a in enumerate(actions):
        action = int(a)
        require(0 <= action < 52 and not matched[action] and action != pending, 'Shared legal-action rule violated')
        after = observations[t + 1]
        visible = matched.copy()
        visible[action] = True
        if pending is not None:
            visible[pending] = True
        require(np.array_equal(after != 13, visible), 'Native before-pair-clear observation timing changed')
        known = visible & (revealed != -1)
        require(np.array_equal(after[known], revealed[known]), 'Previously revealed rank changed')
        revealed[visible] = after[visible]
        require(arrays['ranks'][t] == after[action], 'Selected reveal does not match native frame')
        if pending is None:
            reward, pending = 0., action
        else:
            same = after[action] == after[pending]
            reward = 2 / 52 if same else -2 / 104
            if same:
                matched[[pending, action]] = True
            pending = None
        require(arrays['rewards'][t] == reward, 'Reward differs from independent native rule')
        rewards.append(reward)
        terminated, truncated = bool(matched.all()), t == 103
        require(bool(arrays['terminated'][t]) == terminated and bool(arrays['truncated'][t]) == truncated,
                'Native terminal flag differs')
        require((terminated or truncated) == (t == n - 1), 'Incomplete or overrun episode')
    total = math.fsum(rewards)
    require(total == receipt['return'] and receipt['native_steps'] == n
            and receipt['success'] == bool(matched.all()) and receipt['matched_pairs'] == int(matched.sum()) // 2,
            'Receipt outcome differs from public arrays')
    counts = receipt['counts']
    require(counts['native_attempted'] == counts['native_returned'] == n, 'Native work count differs')
    learned = receipt['controller'] not in ('exact', 'last32')
    require(all(counts[key] == (n if learned else 0) for key in
                ('predict_attempted', 'predict_returned', 'write_attempted', 'write_returned')), 'Model work count differs')
    return total, bool(matched.all()), n, revealed


def aggregate(returns):
    """Independent 18-fit/family arithmetic using the declared mean-first rule."""
    require(set(returns) == set(POLICIES), 'Three policies required')
    require(all(set(returns[p]) == set(NAMES) and all(len(returns[p][n]) == 64 for n in NAMES)
                for p in POLICIES), 'All twenty controllers and 64 paired cases required')
    means = {policy: {name: average(returns[policy][name]) for name in NAMES} for policy in POLICIES}
    family = {policy: {mode: average(value for i in range(3) for value in returns[policy][f'{mode}-pair{i}'])
                       for mode in MODES} for policy in POLICIES}
    contrasts = {}
    for label, later, earlier in (('B_minus_A', 'B', 'A'), ('C_minus_B', 'C', 'B')):
        contrasts[label] = {
            'learned_mean_difference': delta(average(means[later][n] for n in NAMES[:-2]),
                                              average(means[earlier][n] for n in NAMES[:-2])),
            'family_differences': {mode: delta(family[later][mode], family[earlier][mode]) for mode in MODES},
            'per_fit_differences': {name: delta(means[later][name], means[earlier][name]) for name in NAMES},
            'paired_deck_learned_mean_differences': [average(delta(returns[later][name][i], returns[earlier][name][i])
                                                            for name in NAMES[:-2]) for i in range(64)],
            'per_family_positive_fits': {mode: sum(means[later][f'{mode}-pair{i}'] > means[earlier][f'{mode}-pair{i}']
                                                    for i in range(3)) for mode in MODES},
        }
    primary = contrasts['C_minus_B']
    checks = {'overall_mean_C_minus_B_at_least_0.03': Decimal(str(primary['learned_mean_difference'])) >= Decimal('.03')}
    checks.update({mode + '_mean_nonnegative': primary['family_differences'][mode] >= 0 for mode in MODES})
    checks.update({mode + '_at_least_two_positive_fits': primary['per_family_positive_fits'][mode] >= 2 for mode in MODES})
    return means, family, contrasts, checks


def review(root, evaluation, expected_completed_sha256, out_prefix, *, completed_authorized=False):
    require(completed_authorized is True, 'Wait for parent completed-result authorization before any result read')
    started = time.monotonic()
    root, evaluation, prefix = Path(root).resolve(), Path(evaluation).resolve(), Path(out_prefix)
    result_path, prose_path, failed_path = (prefix.with_suffix(suffix) for suffix in ('.json', '.md', '.failed.json'))
    require(not any(p.exists() for p in (result_path, prose_path, failed_path)), 'Review output already exists')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        protocol = read(bound(root, 'evidence/card-controllers-v1/protocol.json', PROTOCOL_SHA))
        inputs = read(bound(root, 'evidence/card-controllers-v1/inputs.json', INPUTS_SHA))
        binding_path = bound(root, protocol['bindings_file'], BINDINGS_SHA)
        bindings = read(binding_path)
        for name, digest in bindings['files'].items():
            bound(root, name, digest)
        lineage = {key: read(bound(root, item['path'], item['sha256'])) for key, item in protocol['lineage'].items()}
        for name, digest in lineage['old_bindings']['files'].items():
            bound(root, name, digest)
        original = lineage['independent_review']
        require(original['continuation_passed'] is False and original['checks_passed'] == 1 and original['total_checks'] == 6,
                'Original failed architecture gate identity changed')
        completed_path = evaluation / 'completed.json'
        require(sha(completed_path) == expected_completed_sha256, 'External completion hash differs')
        completed = read(completed_path)
        require(completed['status'] == 'complete' and completed['episodes'] == 3840
                and completed['new_fits'] == completed['new_optimizer_steps'] == 0, 'Execution incomplete or retrained')
        require(completed['protocol_sha256'] == PROTOCOL_SHA and completed['inputs_sha256'] == INPUTS_SHA
                and completed['bindings_sha256'] == BINDINGS_SHA, 'Execution source/input identities differ')
        expected = {'started.json', 'all-fits-ready.json'}
        for name in NAMES:
            for policy in POLICIES:
                stem = f'controllers/{name}/{policy}'
                expected.add(stem + '/completed.json')
                expected.update(f'{stem}/episodes/{i:03d}.{ext}' for i in range(64) for ext in ('json', 'npz'))
        require(set(completed['files']) == expected and len(expected) == 7742, 'Declared scientific coverage differs')
        actual = {str(p.relative_to(evaluation)) for p in evaluation.rglob('*') if p.is_file()}
        require(actual == expected | {'completed.json'}, 'Actual closure missing/extra/failed')
        for name, item in completed['files'].items():
            path = bound(evaluation, name, item['sha256'])
            require(path.stat().st_size == item['bytes'], 'Payload byte count differs')
        checkpoints = lineage['checkpoint_map']
        require(set(checkpoints) == set(NAMES[:-2]), 'Original 18 checkpoint coverage differs')
        for name, entry in checkpoints.items():
            bound(root, entry['checkpoint_path'], entry['checkpoint_sha256'])
            bound(root, entry['completed_path'], entry['completed_sha256'])
        returns = {policy: {name: [] for name in NAMES} for policy in POLICIES}
        successes = {policy: {name: 0 for name in NAMES} for policy in POLICIES}
        steps = episodes = 0
        layouts = [np.full(52, -1, dtype=np.int64) for _ in range(64)]
        for name in NAMES:
            for policy in POLICIES:
                directory = evaluation / 'controllers' / name / policy
                policy_steps = 0
                for index, case in enumerate(inputs['evaluation']):
                    require(time.monotonic() - started < 600, 'Independent saved review exceeded 600 seconds')
                    stem = directory / 'episodes' / f'{index:03d}'
                    receipt = read(stem.with_suffix('.json'))
                    require(receipt['status'] == 'complete' and receipt['controller'] == name and receipt['policy'] == policy
                            and receipt['index'] == index and receipt['seed'] == case['seed'], 'Paired episode identity differs')
                    require(receipt['checkpoint_sha256'] == (checkpoints[name]['checkpoint_sha256'] if name in checkpoints else None)
                            and receipt['protocol_sha256'] == PROTOCOL_SHA and receipt['inputs_sha256'] == INPUTS_SHA,
                            'Episode checkpoint/input binding differs')
                    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as archive:
                        # Deliberately do not load neural prediction arrays for this independent arithmetic check.
                        arrays = {key: archive[key] for key in ('actions', 'ranks', 'rewards', 'observations', 'terminated', 'truncated')}
                    total, success, work, public = episode(arrays, receipt)
                    seen = public != -1
                    require(np.all((layouts[index][seen] == -1) | (layouts[index][seen] == public[seen])),
                            'Same seed reveals conflicting boards')
                    layouts[index][seen] = public[seen]
                    require(receipt['layout_sha256'] == completed['layout_sha256_by_case'][str(index)], 'Paired layout digest differs')
                    returns[policy][name].append(total)
                    successes[policy][name] += success
                    steps += work
                    policy_steps += work
                    episodes += 1
                row = read(directory / 'completed.json')
                require(row['episodes'] == 64 and row['mean_return'] == average(returns[policy][name])
                        and row['successes'] == successes[policy][name] and row['native_steps'] == policy_steps,
                        'Controller completion arithmetic differs')
        for index, layout in enumerate(layouts):
            require(np.all(layout >= 0) and np.array_equal(np.bincount(layout, minlength=13), np.full(13, 4)),
                    'Public union does not reconstruct a full native board')
            digest = hashlib.sha256(b'card-layout-int64le-v1\0' + np.asarray(layout, dtype='<i8').tobytes()).hexdigest()
            require(digest == completed['layout_sha256_by_case'][str(index)], 'Public board digest differs from evaluator identity')
        require(episodes == 3840 and steps == completed['counts']['native_returned'], 'Native coverage/count differs')
        means, families, contrasts, checks = aggregate(returns)
        require(sha(completed_path) == expected_completed_sha256 and sha(binding_path) == BINDINGS_SHA, 'Boundary receipts changed')
        result = {'status': 'complete', 'source_sha256': sha(__file__), 'bindings': {
            'protocol_sha256': PROTOCOL_SHA, 'inputs_sha256': INPUTS_SHA, 'bindings_sha256': BINDINGS_SHA,
            'execution_completed_sha256': expected_completed_sha256,
            'original_independent_review_sha256': protocol['lineage']['independent_review']['sha256']},
            'coverage': {'episodes': episodes, 'native_public_transitions_checked': steps, 'fits': 18,
                         'policies': 3, 'controllers_per_policy': 20, 'paired_deck_draws': 64, 'payloads': 7742,
                         'distinct_public_layouts': len(set(completed['layout_sha256_by_case'].values()))},
            'per_fit_mean_returns': means, 'per_fit_successes': successes, 'family_mean_returns': families,
            'contrasts': contrasts, 'continuation_checks': checks, 'checks_passed': sum(checks.values()),
            'total_checks': 13, 'continuation_passed': all(checks.values()),
            'original_architecture_gate': {'passed': False, 'checks_passed': 1, 'total_checks': 6, 'unchanged': True},
            'new_model_calls': 0, 'new_native_calls': 0, 'wall_seconds': time.monotonic() - started,
            'limits': ['Independent public native-rule and outcome arithmetic, not native engine replay.',
                       'No learned checkpoint forward or probability calibration verification.',
                       'Three jointly required controller conditions expanded into 13 checks, not independent significance tests.',
                       '64 paired deck draws, not 3840 independent layouts; later trajectories may diverge.',
                       'C-B tests the exploration tie preference; B-A bundles normalization and tie tolerance.',
                       'Original failed architecture gate is not rescued by this controller intervention.']}
        write(result_path, result)
        text = ['# Independent card-controller results review', '',
                f"Controller rule: **{'PASS' if result['continuation_passed'] else 'FAIL'}**, {result['checks_passed']}/13 checks.", '',
                (f'{episodes:,} complete paired episodes and {steps:,} saved native public transitions independently checked. '
                 'All 18 original checkpoints and both public references are retained.'), '',
                (f"Learned C minus B mean return: {contrasts['C_minus_B']['learned_mean_difference']:+.6f}. "
                 f"B minus A: {contrasts['B_minus_A']['learned_mean_difference']:+.6f}."), '',
                ('The original architecture result remains FAIL 1/6. This review made no model or native calls and '
                 'does not prove checkpoint forward correctness, architecture novelty or generalization beyond 64 paired draws.'), '',
                f'Execution receipt SHA-256: `{expected_completed_sha256}`.',
                f'Review JSON SHA-256: `{sha(result_path)}`.', '']
        with prose_path.open('x') as handle:
            handle.write('\n'.join(text))
        return result
    except BaseException as error:
        try:
            write(failed_path, {'status': 'failed', 'error': repr(error), 'wall_seconds': time.monotonic() - started})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure.
            if callable(getattr(error, 'add_note', None)):
                error.add_note(f'Review failure receipt also failed: {secondary!r}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('root', 'evaluation', 'out-prefix'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--expected-completed-sha256', required=True)
    parser.add_argument('--completed-authorized', action='store_true')
    review(**vars(parser.parse_args()))
