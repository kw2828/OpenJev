"""Fabricated anchor metadata and public traces only; no saved study inputs.

The fixture mirrors the original serializer's reset/step distinction. Posterior
witnesses come from direct array arithmetic, without an actor or simulator.
These checks establish extraction mechanics, not teacher-label effectiveness.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from openjev.research import otto_teacher_anchors as anchors

CELLS = (
    ('lambda3', 1, 'shared@9101'),
    ('lambda3', 2, 'dense@9102'),
    ('lambda3', 3, 'shared@9103'),
    ('lambda4', 1, 'dense@9101'),
    ('lambda4', 2, 'shared@9102'),
    ('lambda4', 3, 'dense@9103'),
)


def public(position=(26, 26), *, step=0, hit=1, done=False):
    return {'position': list(position), 'step': step, 'hit': hit, 'done': done,
            'valid_actions': [] if done else [a for a in range(4)
                if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]}


def kernel():
    x, y = np.indices((107, 107), dtype=np.float64)
    value = np.stack(((x + 1) / 432, (y + 1) / 432,
                      np.full_like(x, .25), .75 - (x + y + 2) / 432))
    value[:, 53, 53] = 0
    return value


def assimilate(prior, packet, observation_kernel):
    """Fixture oracle: explicitly crop the public likelihood and update once."""
    x, y = packet['position']
    result = prior.copy()
    result[x, y] = 0
    result *= observation_kernel[packet['hit'], 53 - x:106 - x, 53 - y:106 - y]
    mass = np.sum(result)
    if mass > 1e-10:
        result /= mass
    return result


def witness(belief):
    return {'sha256': hashlib.sha256(belief.tobytes(order='C')).hexdigest(),
            'mass': float(np.sum(belief))}


def fixture(cell=0, *, row_index=None, seed=None, prefix=2, observation_kernel=None):
    regime, hit, collector = CELLS[cell]
    if seed is None:
        seed = (950000 if regime == 'lambda3' else 960000) + hit
    identity = {'episode_id': f'dagger:{regime}:{seed}:{collector}',
                'stage': 'dagger', 'regime': regime, 'seed': seed,
                'initial_hit': hit, 'arm': collector}
    k = kernel() if observation_kernel is None else observation_kernel
    prior = np.ones((53, 53), dtype=np.float64) / (53 * 53 - 1)
    prior[26, 26] = 0
    packets = [public(hit=hit), public((25, 26), step=1, hit=0), public((25, 27), step=2, hit=1)]
    beliefs = [assimilate(prior, packets[0], k)]
    for packet in packets[1:]:
        beliefs.append(assimilate(beliefs[-1], packet, k))
    records = [{'kind': 'reset', **identity, 'block': None, 'public': packets[0],
                'posterior_after': witness(beliefs[0]), 'source_evaluation_only': [7, 41]}]
    for step, action in enumerate((0, 3), start=1):
        records.append({'kind': 'step', 'episode_id': identity['episode_id'], 'step': step,
                        'action': action, 'allowed_actions': packets[step - 1]['valid_actions'],
                        'public': packets[step], 'posterior_before': witness(beliefs[step - 1]),
                        'posterior_after': witness(beliefs[step]), 'native_p_end': .25,
                        'costs': [100, 200, 300, 400], 'choose_seconds': .01,
                        'choose_instrumented_seconds': .015, 'choose_excluded_io_seconds': .005,
                        'update_seconds': .02, 'environment_seconds': .03})
    row = {'row_index': cell if row_index is None else row_index, **identity,
           'prefix_index': prefix, 'public': copy.deepcopy(packets[prefix]),
           'posterior': witness(beliefs[prefix]), 'teacher_costs': [10, 20, 30, 40]}
    return row, records, beliefs, k


def cohort():
    values = [fixture(i) for i in range(6)]
    return [v[0] for v in values], [r for v in values for r in v[1]], values


def encoded(records):
    return [json.dumps(row, separators=(',', ':'), allow_nan=False).encode() + b'\n' for row in records]


def prepare(rows=None, records=None):
    if rows is None:
        rows, records, _ = cohort()
    selected = anchors.select_anchors(rows)
    projected = tuple(anchors.iter_public_records(encoded(records), selected))
    return selected, projected, {'lambda3': kernel(), 'lambda4': kernel()}


def test_fixed_six_cells_choose_lexicographic_metadata_independent_of_iteration_order():
    rows, expected = [], []
    for cell in range(6):
        first, _, _, _ = fixture(cell, row_index=cell * 10 + 4, prefix=1)
        seed = first['seed']
        rows.extend((fixture(cell, row_index=cell * 10 + 3, prefix=0)[0],
                     fixture(cell, row_index=cell * 10 + 2, prefix=2)[0],
                     fixture(cell, row_index=cell * 10 + 1, seed=seed + 3, prefix=1)[0], first))
        expected.append((cell, first['row_index'], first['episode_id'], 1))
    snapshots = []
    for order in (rows, list(reversed(rows)), rows[::2] + rows[1::2]):
        selected = anchors.select_anchors(iter(order))
        snapshots.append(selected)
        assert len(selected) == 6
        assert [(r['anchor_id'], r['row_index'], r['episode_id'], r['prefix_index'])
                for r in selected] == expected
        assert [(r['regime'], r['initial_hit'], r['arm']) for r in selected] == list(CELLS)
        assert all('teacher_costs' not in row for row in selected)
    assert snapshots[0] == snapshots[1] == snapshots[2]


@pytest.mark.parametrize('defect', [
    'missing_cell', 'duplicate_row_index', 'duplicate_episode_prefix', 'valid_stage',
    'identity_mismatch', 'outside_seed_range', 'wrong_seed_hit', 'boolean_index',
])
def test_selection_rejects_incomplete_duplicate_or_invalid_identity_without_replacement(defect):
    rows, _, _ = cohort()
    if defect == 'missing_cell':
        rows.pop()
    elif defect == 'duplicate_row_index':
        rows[1]['row_index'] = rows[0]['row_index']
    elif defect == 'duplicate_episode_prefix':
        duplicate = copy.deepcopy(rows[0])
        duplicate['row_index'] = 100
        rows.append(duplicate)
    elif defect == 'valid_stage':
        rows[0]['stage'] = 'valid'
    elif defect == 'identity_mismatch':
        rows[0]['episode_id'] = rows[1]['episode_id']
    elif defect == 'outside_seed_range':
        rows[0]['seed'] = 950013
        rows[0]['episode_id'] = 'dagger:lambda3:950013:shared@9101'
    elif defect == 'wrong_seed_hit':
        rows[0]['initial_hit'] = 2
    else:
        rows[0]['row_index'] = True
    with pytest.raises((ValueError, TypeError)):
        anchors.select_anchors(rows)


def test_selection_ignores_teacher_scores_and_does_not_mutate_or_alias_metadata():
    class ForbiddenScore:
        def __float__(self):
            raise AssertionError('teacher target must not participate in anchor selection')

        def __lt__(self, _other):
            raise AssertionError('teacher target must not participate in anchor selection')

    rows, _, _ = cohort()
    for row in rows:
        row['teacher_costs'] = [ForbiddenScore()] * 4
    before_public = copy.deepcopy([r['public'] for r in rows])
    selected = anchors.select_anchors(rows)
    assert [r['public'] for r in rows] == before_public
    selected[0]['public']['position'][0] = 0
    selected[0]['posterior']['mass'] = .25
    assert rows[0]['public'] == before_public[0]
    assert rows[0]['posterior']['mass'] != .25


def test_exact_episode_allowlist_precedes_decoder_even_for_malformed_validation_payload(monkeypatch):
    rows, records, _ = cohort()
    selected = anchors.select_anchors(rows)
    allowed_lines = encoded(records)
    excluded = [
        b'{"kind":"reset","episode_id":"valid:lambda3:930001:teacher","public":NaN INVALID\n',
        b'{"kind":"step","episode_id":"eval:lambda3:970001:shared@9101","public":[}\n',
        b'{"kind":"reset","episode_id":"dagger:lambda3:950004:shared@9101","posterior":broken\n',
        b'{"kind":"step","episode_id":"dagger:lambda3:950001:shared@91010","step":BAD\n',
        b'{"kind":"reset","episode_id":"train:lambda4:920001:teacher","public":null}\n',
        (b'{"kind":"step","episode_id":"valid:lambda3:930001:teacher",'
         b'"note":"dagger:lambda3:950001:shared@9101","public":BROKEN\n'),
    ]
    decode, decoded = anchors._decode_record, []

    def guarded_decode(raw, *args, **kwargs):
        as_bytes = raw.encode() if isinstance(raw, str) else raw
        assert as_bytes not in excluded, 'unselected bytes reached numerical decoding'
        decoded.append(as_bytes)
        return decode(raw, *args, **kwargs)

    monkeypatch.setattr(anchors, '_decode_record', guarded_decode)
    actual = tuple(anchors.iter_public_records(excluded[:2] + allowed_lines + excluded[2:], selected))
    assert len(actual) == 18
    assert len(decoded) == 18
    assert {r['episode_id'] for r in actual} == {r['episode_id'] for r in selected}


def test_selected_malformed_payload_is_rejected_instead_of_silently_skipped():
    rows, _, _ = cohort()
    selected = anchors.select_anchors(rows)
    raw = b'{"kind":"reset","episode_id":"dagger:lambda3:950001:shared@9101","public":BROKEN}\n'
    with pytest.raises((ValueError, TypeError)):
        tuple(anchors.iter_public_records([raw], selected))


def test_projection_discards_hidden_truth_and_costs_before_public_reconstruction():
    rows, records, _ = cohort()
    selected = anchors.select_anchors(rows)
    clean = tuple(anchors.iter_public_records(encoded(records), selected))
    for row in records:
        row.update(source_evaluation_only=[52, 0], draws_evaluation_only={'secret': [0, 1]},
                   native_posterior={'never': 'an actor input'}, reward=-999999,
                   costs=['secret'], teacher_costs=['secret'])
    decorated = tuple(anchors.iter_public_records(encoded(records), selected))
    assert decorated == clean
    forbidden = {'source_evaluation_only', 'draws_evaluation_only', 'native_posterior',
                 'reward', 'costs', 'teacher_costs', 'native_p_end', 'environment_seconds'}
    assert all(not forbidden.intersection(record) for record in decorated)


def test_reconstruction_preserves_pre_action_prefix_without_reset_or_reassimilation():
    rows, records, values = cohort()
    selected, projected, kernels = prepare(rows, records)
    restored = anchors.reconstruct_anchors(selected, projected, kernels)
    assert len(restored) == 6
    for cell, actual in enumerate(restored):
        expected = values[cell][2][2]
        assert actual['anchor_id'] == cell
        assert actual['public'] == rows[cell]['public']
        assert actual['public']['step'] == actual['prefix_index'] == 2
        assert actual['public']['position'] == [25, 27]
        assert actual['sensing_length'] == (3 if cell < 3 else 4)
        assert actual['belief'].dtype == np.float64 and actual['belief'].shape == (53, 53)
        assert actual['belief'].tobytes() == expected.tobytes()
        assert actual['belief'].tobytes() != values[cell][2][0].tobytes()
        reapplied = assimilate(expected, rows[cell]['public'], kernels[rows[cell]['regime']])
        assert actual['belief'].tobytes() != reapplied.tobytes()
        assert not actual['belief'].flags.writeable
        with pytest.raises(ValueError):
            actual['belief'].flags.writeable = True


@pytest.mark.parametrize('defect', [
    'missing_reset', 'missing_step', 'duplicate_reset', 'duplicate_step', 'step_gap',
    'public_step', 'wrong_action', 'wrong_position', 'reset_hash', 'before_hash',
    'after_hash', 'reset_mass', 'after_mass', 'metadata_hash', 'metadata_mass', 'hidden_public_field',
])
def test_reconstruction_rejects_broken_public_lineage_or_witnesses(defect):
    rows, records, _ = cohort()
    first = records[:3]
    if defect == 'missing_reset':
        del records[0]
    elif defect == 'missing_step':
        del records[1]
    elif defect == 'duplicate_reset':
        records.insert(1, copy.deepcopy(first[0]))
    elif defect == 'duplicate_step':
        records.insert(2, copy.deepcopy(first[1]))
    elif defect == 'step_gap':
        first[1]['step'] = 3
    elif defect == 'public_step':
        first[1]['public']['step'] = 0
    elif defect == 'wrong_action':
        first[1]['action'] = 1
    elif defect == 'wrong_position':
        first[1]['public']['position'] = [24, 26]
    elif defect in ('reset_hash', 'before_hash', 'after_hash'):
        record, field = {'reset_hash': (first[0], 'posterior_after'),
                         'before_hash': (first[1], 'posterior_before'),
                         'after_hash': (first[1], 'posterior_after')}[defect]
        record[field]['sha256'] = '0' * 64
    elif defect in ('reset_mass', 'after_mass'):
        first[0 if defect == 'reset_mass' else 1]['posterior_after']['mass'] = .5
    elif defect == 'metadata_hash':
        rows[0]['posterior']['sha256'] = '0' * 64
    elif defect == 'metadata_mass':
        rows[0]['posterior']['mass'] = .5
    else:
        first[1]['public']['source'] = [7, 41]
    with pytest.raises((ValueError, TypeError)):
        selected, projected, kernels = prepare(rows, records)
        anchors.reconstruct_anchors(selected, projected, kernels)


def test_unsupported_selected_anchor_fails_instead_of_substituting_later_candidate():
    rows, records, _ = cohort()
    rows[0]['posterior']['mass'] = 0.
    replacement, _, _, _ = fixture(0, row_index=50, seed=950004)
    rows.append(replacement)
    selected = anchors.select_anchors(rows)
    assert selected[0]['row_index'] == 0
    with pytest.raises((ValueError, TypeError)):
        projected = tuple(anchors.iter_public_records(encoded(records), selected))
        anchors.reconstruct_anchors(selected, projected, {'lambda3': kernel(), 'lambda4': kernel()})


def test_reconstruction_does_not_mutate_or_retain_caller_packets_witnesses_or_kernels():
    selected, projected, kernels = prepare()
    selection_before, projected_before = copy.deepcopy(selected), copy.deepcopy(projected)
    kernel_bytes = {name: value.tobytes() for name, value in kernels.items()}
    restored = anchors.reconstruct_anchors(selected, projected, kernels)
    assert selected == selection_before and projected == projected_before
    assert {name: value.tobytes() for name, value in kernels.items()} == kernel_bytes
    output_public = copy.deepcopy(restored[0]['public'])
    output_witness = copy.deepcopy(restored[0]['posterior'])
    output_belief = restored[0]['belief'].tobytes()
    selected[0]['public']['position'][0] = 0
    selected[0]['posterior']['mass'] = 0
    projected[0]['public']['position'][0] = 1
    kernels['lambda3'][:] = 0
    assert restored[0]['public'] == output_public
    assert restored[0]['posterior'] == output_witness
    assert restored[0]['belief'].tobytes() == output_belief


def test_all_six_validation_snapshots_receive_only_public_inputs_and_never_choose(monkeypatch):
    selected, projected, kernels = prepare()
    real_snapshot, calls = anchors.TeacherSnapshot, []

    class RecordingSnapshot(real_snapshot):
        def __init__(self, public, belief, observation_kernel):
            assert set(public) == {'position', 'step', 'hit', 'done', 'valid_actions'}
            assert public['step'] == 2
            calls.append((copy.deepcopy(public), belief.tobytes(), observation_kernel.tobytes()))
            super().__init__(public, belief, observation_kernel)

        def choose(self):
            raise AssertionError('anchor validation must not choose an action')

        def update(self, *_args, **_kwargs):
            raise AssertionError('anchor validation must not re-assimilate an observation')

        def reset(self, *_args, **_kwargs):
            raise AssertionError('anchor validation must not reset to a new episode')

    monkeypatch.setattr(anchors, 'TeacherSnapshot', RecordingSnapshot)
    restored = anchors.reconstruct_anchors(selected, projected, kernels)
    assert len(calls) == 6
    assert [call[1] for call in calls] == [row['belief'].tobytes() for row in restored]
    assert [call[2] for call in calls] == [kernels[row['regime']].tobytes() for row in restored]
    validated = anchors.validate_anchors(restored, kernels)
    assert len(calls) == 12
    assert [row['belief'].tobytes() for row in validated] == [row['belief'].tobytes() for row in restored]
    assert all(not row['belief'].flags.writeable for row in validated)
    validated[0]['public']['position'][0] = 0
    assert restored[0]['public']['position'] == [25, 27]


@pytest.mark.parametrize('defect', ['half_mass', 'subfloor', 'current_cell'])
def test_validation_rejects_unsupported_belief_without_repairing_caller_array(defect):
    selected, projected, kernels = prepare()
    restored = anchors.reconstruct_anchors(selected, projected, kernels)
    modified = [dict(row) for row in restored]
    candidate = restored[0]['belief'].copy()
    if defect == 'half_mass':
        candidate *= .5
    elif defect == 'subfloor':
        candidate *= 1e-12
    else:
        candidate[25, 27] = 1e-300
    modified[0]['belief'] = candidate
    modified[0]['posterior'] = witness(candidate)
    before = candidate.tobytes()
    with pytest.raises((ValueError, TypeError)):
        anchors.validate_anchors(modified, kernels)
    assert candidate.tobytes() == before
