"""Fabricated cohort selection, public replay and unsupported-state accounting.

No saved metadata/arrays, native environment, model, sampler or training calls.
The complete selector fixture has 4,596 artificial rows; tiny replay fixtures
derive their posterior witnesses directly from public likelihood arithmetic.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from openjev.research import otto_teacher_cohort as cohort

ARMS = ('dense@9101', 'dense@9102', 'dense@9103', 'shared@9101', 'shared@9102', 'shared@9103')


def identities():
    return [{'episode_id': f'dagger:{regime}:{seed}:{arm}', 'stage': 'dagger', 'regime': regime,
             'seed': seed, 'initial_hit': 1 + (seed - first) % 3, 'arm': arm}
            for regime, first in (('lambda3', 950001), ('lambda4', 960001))
            for seed in range(first, first + 12) for arm in ARMS]


def packet(position=(26, 26), *, step=0, hit=1):
    return {'position': list(position), 'step': step, 'hit': hit, 'done': False,
            'valid_actions': [a for a in range(4) if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]}


def metadata(*, edges=False):
    lengths = [32] * 132 + [31] * 12
    if edges:
        lengths[:4] = [1, 2, 4, 9]
        lengths[-1] += 4596 - sum(lengths)
    rows = []
    for identity, count in zip(identities(), lengths, strict=True):
        for prefix in range(count):
            rows.append({'row_index': len(rows), **identity, 'prefix_index': prefix,
                         'public': packet(step=prefix, hit=identity['initial_hit'] if prefix == 0 else 0),
                         'posterior': {'sha256': '0' * 64, 'mass': 1.}, 'teacher_costs': [1., 2., 3., 4.]})
    assert len(rows) == 4596
    return rows, lengths


def kernel(*, subfloor=False):
    if subfloor:
        result = np.stack([np.full((107, 107), value, dtype=np.float64)
                           for value in (1e-14, .25, .25, .5 - 1e-14)])
    else:
        x, y = np.indices((107, 107), dtype=np.float64)
        result = np.stack(((x + 1) / 432, (y + 1) / 432,
                           np.full_like(x, .25), .75 - (x + y + 2) / 432))
    result[:, 53, 53] = 0
    return result


def update(prior, public, likelihood):
    x, y = public['position']
    result = prior.copy()
    result[x, y] = 0
    result *= likelihood[public['hit'], 53-x:106-x, 53-y:106-y]
    mass = np.sum(result)
    if mass > 1e-10:
        result /= mass
    return result


def witness(belief):
    return {'sha256': hashlib.sha256(belief.tobytes(order='C')).hexdigest(), 'mass': float(np.sum(belief))}


def trace(*, subfloor=False):
    identity = identities()[0]
    likelihood = kernel(subfloor=subfloor)
    positions = ((26, 26), (25, 26), (25, 27), (26, 27), (26, 26))
    hits = (1, 2, 0, 1, 1) if subfloor else (1, 0, 1, 2, 3)
    packets = [packet(position, step=step, hit=hit)
               for step, (position, hit) in enumerate(zip(positions, hits, strict=True))]
    prior = np.ones((53, 53), dtype=np.float64) / (53 * 53 - 1)
    prior[26, 26] = 0
    beliefs = []
    for public in packets:
        prior = update(prior, public, likelihood)
        beliefs.append(prior)
    records = [{'kind': 'reset', **identity, 'block': None, 'public': packets[0],
                'posterior_after': witness(beliefs[0]), 'source_evaluation_only': [3, 44]}]
    for step, action in enumerate((0, 3, 1, 2), start=1):
        records.append({'kind': 'step', 'episode_id': identity['episode_id'], 'step': step, 'action': action,
                        'allowed_actions': packets[step-1]['valid_actions'], 'public': packets[step],
                        'posterior_before': witness(beliefs[step-1]), 'posterior_after': witness(beliefs[step]),
                        'costs': [1., 2., 3., 4.], 'choose_seconds': .01, 'choose_instrumented_seconds': .015,
                        'choose_excluded_io_seconds': .005, 'update_seconds': .01,
                        'environment_seconds': .01, 'native_p_end': 0.})
    selections = tuple({'anchor_id': i, 'row_index': prefix, **identity, 'prefix_index': prefix,
                        'public': copy.deepcopy(packets[prefix]), 'posterior': witness(beliefs[prefix])}
                       for i, prefix in enumerate((0, 2, 4)))
    return selections, records, beliefs, {'lambda3': likelihood, 'lambda4': likelihood.copy()}


def encode(records):
    return [json.dumps(row, separators=(',', ':'), allow_nan=False).encode() + b'\n' for row in records]


def reconstruct(*, subfloor=False, emit=None):
    selected, records, beliefs, kernels = trace(subfloor=subfloor)
    public = cohort.iter_public_records(encode(records), selected)
    result = cohort.reconstruct_cohort(selected, public, kernels, emit=emit)
    return result, selected, beliefs, kernels


def test_complete_pool_selects_floor_even_prefixes_including_each_reset():
    rows, lengths = metadata(edges=True)
    result = cohort.select_cohort(iter(reversed(rows)))
    episodes = result['episodes']
    expected_ids = [row['episode_id'] for row in identities()]
    assert len(episodes) == 144 and [row['episode_id'] for row in episodes] == expected_ids
    for episode, count in zip(episodes, lengths, strict=True):
        assert episode['retained_rows'] == count
        assert episode['selected_rows'] == min(4, count)
    chosen = result['selections']
    assert len(chosen) == sum(min(4, count) for count in lengths)
    assert [row['anchor_id'] for row in chosen] == list(range(len(chosen)))
    assert all(row['prefix_index'] >= 0 and 'teacher_costs' not in row for row in chosen)
    assert {row['episode_id'] for row in chosen if row['prefix_index'] == 0} == set(expected_ids)
    prefixes = {episode_id: [r['prefix_index'] for r in chosen if r['episode_id'] == episode_id]
                for episode_id in expected_ids[:4]}
    assert [prefixes[key] for key in expected_ids[:4]] == [[0], [0, 1], [0, 1, 2, 3], [0, 2, 5, 8]]
    assert [(r['regime'], r['seed'], r['arm'], r['prefix_index'], r['row_index']) for r in chosen] == sorted(
        (r['regime'], r['seed'], r['arm'], r['prefix_index'], r['row_index']) for r in chosen)
    assert cohort.select_cohort(rows) == result


def test_selection_ignores_scores_and_mass_without_mutating_or_aliasing_rows():
    class Forbidden:
        def __float__(self):
            raise AssertionError('teacher score entered selection')

        def __lt__(self, _other):
            raise AssertionError('teacher score entered selection')

    rows, _ = metadata()
    baseline = cohort.select_cohort(rows)
    for row in rows:
        row['teacher_costs'] = [Forbidden()] * 4
        row['posterior']['mass'] = (0., .5, 1.)[row['row_index'] % 3]
    before = copy.deepcopy([(r['public'], r['posterior']) for r in rows])
    changed = cohort.select_cohort(rows)
    def identity(result):
        return [(r['anchor_id'], r['row_index'], r['episode_id'], r['prefix_index'])
                for r in result['selections']]

    assert identity(changed) == identity(baseline)
    assert [(r['public'], r['posterior']) for r in rows] == before
    changed['selections'][0]['public']['position'][0] = 0
    changed['selections'][0]['posterior']['mass'] = 7.
    assert [(r['public'], r['posterior']) for r in rows] == before


@pytest.mark.parametrize('defect', ['short_pool', 'missing_episode', 'duplicate_row', 'duplicate_prefix', 'valid', 'eval'])
def test_pool_integrity_failures_cannot_be_replaced_by_other_episodes(defect):
    rows, _ = metadata()
    if defect == 'short_pool':
        rows.pop()
    elif defect == 'missing_episode':
        first, second = identities()[:2]
        for row in rows:
            if row['episode_id'] == first['episode_id']:
                row.update(second)
                row['prefix_index'] += 1000
                row['public']['step'] = row['prefix_index']
    elif defect == 'duplicate_row':
        rows[1]['row_index'] = rows[0]['row_index']
    elif defect == 'duplicate_prefix':
        rows[1]['prefix_index'] = rows[0]['prefix_index']
        rows[1]['public'] = copy.deepcopy(rows[0]['public'])
    else:
        rows[0]['stage'] = defect
        rows[0]['episode_id'] = rows[0]['episode_id'].replace('dagger:', defect + ':', 1)
    with pytest.raises((ValueError, TypeError)):
        cohort.select_cohort(rows)


def test_multiple_selected_prefixes_share_one_replay_and_are_not_reassimilated(monkeypatch):
    selected, records, beliefs, kernels = trace()
    original = cohort.PublicBeliefView
    calls = {'construct': 0, 'reset': [], 'update': []}

    class CountedView(original):
        def __init__(self, observation_kernel):
            calls['construct'] += 1
            super().__init__(observation_kernel)

        def _reset(self, hit):
            calls['reset'].append(hit)
            return super()._reset(hit)

        def _observe(self, public):
            assert set(public) == {'position', 'step', 'hit', 'done', 'valid_actions'}
            calls['update'].append(public['step'])
            return super()._observe(public)

    monkeypatch.setattr(cohort, 'PublicBeliefView', CountedView)
    projected = cohort.iter_public_records(encode(records), selected)
    result = cohort.reconstruct_cohort(selected, projected, kernels)
    assert calls == {'construct': 1, 'reset': [1], 'update': [1, 2, 3, 4]}
    assert result['counts'] == {'anchors': 3, 'episodes': 1, 'public_resets': 1, 'public_updates': 4}
    assert [row['prefix_index'] for row in result['anchors']] == [0, 2, 4]
    for row, prefix in zip(result['anchors'], (0, 2, 4), strict=True):
        assert row['belief'].tobytes() == beliefs[prefix].tobytes()
        assert row['public'] == selected[row['anchor_id']]['public']
        assert row['belief'].tobytes() != update(beliefs[prefix], row['public'], kernels['lambda3']).tobytes()
        assert not row['belief'].flags.writeable
        with pytest.raises(ValueError):
            row['belief'].flags.writeable = True


def test_reset_only_selection_stops_before_any_public_update():
    selected, records, beliefs, kernels = trace()
    selected = selected[:1]
    projected = tuple(cohort.iter_public_records(encode(records), selected))
    assert len(projected) == 1 and projected[0]['kind'] == 'reset'
    result = cohort.reconstruct_cohort(selected, projected, kernels)
    assert result['counts'] == {'anchors': 1, 'episodes': 1, 'public_resets': 1, 'public_updates': 0}
    assert result['anchors'][0]['public']['step'] == 0
    assert result['anchors'][0]['belief'].tobytes() == beliefs[0].tobytes()


def test_replay_input_ownership_prevents_later_caller_mutation():
    selected, records, _, kernels = trace()
    before = copy.deepcopy((selected, records))
    kernel_before = {name: value.tobytes() for name, value in kernels.items()}
    projected = tuple(cohort.iter_public_records(encode(records), selected))
    result = cohort.reconstruct_cohort(selected, projected, kernels)
    assert (selected, records) == before
    assert {name: value.tobytes() for name, value in kernels.items()} == kernel_before
    public_before = copy.deepcopy(result['anchors'][0]['public'])
    belief_before = result['anchors'][0]['belief'].tobytes()
    selected[0]['public']['position'][0] = 0
    projected[0]['public']['position'][0] = 1
    kernels['lambda3'][:] = 0
    assert result['anchors'][0]['public'] == public_before
    assert result['anchors'][0]['belief'].tobytes() == belief_before


def test_exact_allowlist_and_prefix_limit_exclude_validation_evaluation_and_truth(monkeypatch):
    selected, records, _, _ = trace()
    expected = tuple(cohort.iter_public_records(encode(records), selected))
    excluded = [
        b'{"kind":"reset","episode_id":"valid:lambda3:930001:teacher","public":BROKEN\n',
        b'{"kind":"step","episode_id":"eval:lambda4:980001:dense@9101","public":NaN INVALID\n',
        b'{"kind":"step","episode_id":"dagger:lambda3:950001:dense@91010","step":BAD\n',
        (b'{"kind":"reset","episode_id":"valid:lambda3:930001:teacher",'
         b'"note":"dagger:lambda3:950001:dense@9101","public":BROKEN\n'),
    ]
    future = copy.deepcopy(records[-1])
    future.update(step=5, action='must not decode', public={'malformed': 'beyond selected prefix'},
                  posterior_before={'unselected': True}, posterior_after={'unselected': True})
    original, decoded = cohort._decode_record, []

    class ForbiddenNumber(cohort.original._NumberToken):
        def __float__(self):
            raise AssertionError('hidden truth was converted to a numerical input')

        def __int__(self):
            raise AssertionError('hidden truth was converted to a numerical input')

    def guarded(raw):
        as_bytes = raw.encode() if isinstance(raw, str) else raw
        assert as_bytes not in excluded, 'unrelated bytes reached the JSON decoder'
        value = original(raw)
        value.update(source_evaluation_only=[ForbiddenNumber('0.5'), ForbiddenNumber('1')],
                     reward=ForbiddenNumber('1'), draws_evaluation_only={'uniform': ForbiddenNumber('0.5')})
        decoded.append(as_bytes)
        return value

    monkeypatch.setattr(cohort, '_decode_record', guarded)
    actual = tuple(cohort.iter_public_records(excluded[:2] + encode(records + [future]) + excluded[2:], selected))
    assert actual == expected and len(actual) == 5 and len(decoded) == 6
    forbidden = {'source_evaluation_only', 'reward', 'draws_evaluation_only', 'costs', 'native_p_end'}
    assert all(not forbidden.intersection(row) for row in actual)


@pytest.mark.parametrize('defect', ['missing_step', 'wrong_hash', 'wrong_mass', 'extra_truth_in_public'])
def test_broken_public_prefix_fails_instead_of_skipping_selected_anchors(defect):
    selected, records, _, kernels = trace()
    if defect == 'missing_step':
        del records[2]
    elif defect == 'wrong_hash':
        records[2]['posterior_after']['sha256'] = '0' * 64
    elif defect == 'wrong_mass':
        records[2]['posterior_after']['mass'] = .5
    else:
        records[2]['public']['source'] = [3, 44]
    with pytest.raises((ValueError, TypeError)):
        public = cohort.iter_public_records(encode(records), selected)
        cohort.reconstruct_cohort(selected, public, kernels)


def test_supported_and_subfloor_prefixes_all_remain_and_every_snapshot_is_attempted(monkeypatch):
    result, selected, beliefs, kernels = reconstruct(subfloor=True)
    before = [row['belief'].tobytes() for row in result['anchors']]
    real_snapshot, observed = cohort.TeacherSnapshot, []

    class RecordingSnapshot(real_snapshot):
        def __init__(self, public, belief, observation_kernel):
            assert set(public) == {'position', 'step', 'hit', 'done', 'valid_actions'}
            observed.append((public['step'], belief.tobytes()))
            super().__init__(public, belief, observation_kernel)

        def choose(self):
            raise AssertionError('support assessment must not choose')

        def reset(self, *_args, **_kwargs):
            raise AssertionError('support assessment must not reset')

    monkeypatch.setattr(cohort, 'TeacherSnapshot', RecordingSnapshot)
    events = []
    assessment = cohort.assess_support(result['anchors'], kernels, emit=events.append)
    assert [row['anchor_id'] for row in assessment['assessments']] == [0, 1, 2]
    assert [row['supported'] for row in assessment['assessments']] == [True, False, False]
    assert assessment['assessments'][0]['error'] is None
    assert all(row['error']['type'] == 'ValueError' and row['error']['message']
               for row in assessment['assessments'][1:])
    assert assessment['counts'] == {'anchors': 3, 'supported': 1, 'unsupported': 2,
                                    'snapshot_attempts': 3, 'snapshot_returns': 1}
    assert observed == [(prefix, beliefs[prefix].tobytes()) for prefix in (0, 2, 4)]
    assert [row['belief'].tobytes() for row in result['anchors']] == before
    assert len(result['anchors']) == len(selected) == 3
    assert 0 < result['anchors'][1]['belief'].sum() < 1e-10
    assert 0 < result['anchors'][2]['belief'].sum() < 1e-10
    assert [(row['event'], row['anchor_id']) for row in events] == [
        ('attempt', 0), ('return', 0), ('attempt', 1), ('return', 1), ('attempt', 2), ('return', 2)]
    assert [row['snapshot_returned'] for row in events if row['event'] == 'return'] == [True, False, False]


@pytest.mark.parametrize('error', [InterruptedError('stop assessment'), ValueError('unexpected constructor defect')])
def test_unexpected_snapshot_failure_propagates_instead_of_becoming_unsupported(monkeypatch, error):
    result, _, _, kernels = reconstruct()
    calls = []

    def failed_snapshot(*_args, **_kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(cohort, 'TeacherSnapshot', failed_snapshot)
    with pytest.raises(type(error)) as captured:
        cohort.assess_support(result['anchors'], kernels)
    assert captured.value is error and calls == [1]


def test_shared_kernel_failure_is_not_reported_as_unsupported_beliefs(monkeypatch):
    result, _, _, kernels = reconstruct()
    kernels['lambda3'][0, 53, 53] = .1

    def forbidden_snapshot(*_args, **_kwargs):
        raise AssertionError('invalid shared kernel reached an anchor snapshot')

    monkeypatch.setattr(cohort, 'TeacherSnapshot', forbidden_snapshot)
    with pytest.raises(ValueError, match='kernel'):
        cohort.assess_support(result['anchors'], kernels)
