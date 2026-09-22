"""Fabricated continuation cases only; no native environments or saved inputs.

Random-stream expectations are reconstructed from the declared namespace rather
than inferred from empirical frequencies. These tests qualify sampling and
lifecycle semantics, not learning effectiveness or native-generator parity.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from openjev.research import otto_teacher_rollouts as rollout
from openjev.research.otto_teacher_costs import ContinuationRecord
from openjev.research.otto_teacher_snapshot import TeacherSnapshot


def packet(position=(10, 10), *, step=119, hit=3, done=False):
    return {'position': list(position), 'step': step, 'hit': hit, 'done': done,
            'valid_actions': [] if done else [a for a in range(4)
                if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]}


def kernel():
    result = np.zeros((4, 107, 107), dtype=np.float64)
    result[0] = 1.
    result[:, 53, 53] = 0.
    return result


def point_belief(position=(9, 10)):
    result = np.zeros((53, 53), dtype=np.float64)
    result[position] = 1.
    return result


def sample(*, public=None, belief=None, observation_kernel=None, **kwargs):
    options = {'seed': 17, 'anchor_id': 23, 'replicate_ids': (7,), 'horizon': 1}
    options.update(kwargs)
    return rollout.sample_teacher_panel(
        packet() if public is None else public,
        point_belief() if belief is None else belief,
        kernel() if observation_kernel is None else observation_kernel,
        **options,
    )


def recording_teacher(monkeypatch, *, choose=None, update_error=None):
    teachers = []

    class RecordingTeacher(TeacherSnapshot):
        def __init__(self, public, belief, observation_kernel):
            self.anchor = copy.deepcopy(public)
            self.updates = []
            self.choices = []
            super().__init__(public, belief, observation_kernel)
            teachers.append(self)

        def choose(self):
            self.choices.append((self.public['step'], len(self.updates)))
            if choose is not None:
                action = choose(self)
                return action, np.zeros(4, dtype=np.float64)
            return super().choose()

        def update(self, action, public):
            if update_error is not None:
                raise update_error
            super().update(action, public)
            self.updates.append((action, copy.deepcopy(self.public)))

    monkeypatch.setattr(rollout, 'TeacherSnapshot', RecordingTeacher)
    return teachers


def test_forced_first_move_found_sentinel_and_public_step_with_real_teacher(monkeypatch):
    teachers = recording_teacher(monkeypatch)
    records = sample()
    assert records == tuple(ContinuationRecord(7, a, 1, a == 0) for a in range(4))
    active = [teacher for teacher in teachers if teacher.updates]
    assert len(active) == 4
    for first_action, teacher in enumerate(active):
        assert teacher.anchor['step'] == 119
        assert teacher.choices == []
        assert len(teacher.updates) == 1
        action, observed = teacher.updates[0]
        assert action == first_action and observed['step'] == 120
        assert observed['done'] is (first_action == 0)
        assert observed['hit'] == (-2 if first_action == 0 else 0)
        assert set(observed) == {'position', 'hit', 'done', 'step', 'valid_actions'}
        if first_action == 0:
            assert observed['valid_actions'] == ()
            assert teacher.belief[9, 10] == 1.


def test_real_teacher_success_exactly_at_horizon_and_final_updates(monkeypatch):
    teachers = recording_teacher(monkeypatch)
    records = sample(belief=point_belief((8, 10)), horizon=2)
    assert records[0] == ContinuationRecord(7, 0, 2, True)
    active = [teacher for teacher in teachers if teacher.updates]
    assert len(active) == 4
    for teacher in active:
        assert len(teacher.updates) == 2
        assert teacher.choices == [(120, 1)]
        assert teacher.public['step'] == 121
    assert active[0].updates[-1][1]['done'] is True
    assert active[0].updates[-1][1]['hit'] == -2
    assert all(record.steps == 2 for record in records)


def test_completed_unsuccessful_cycle_keeps_full_cap_and_final_public_packet(monkeypatch):
    teachers = recording_teacher(
        monkeypatch, choose=lambda actor: 1 if actor.public['position'][0] <= 9 else 0)
    records = sample(belief=point_belief((40, 41)), horizon=4)
    assert records == tuple(ContinuationRecord(7, a, 4, False) for a in range(4))
    active = [teacher for teacher in teachers if teacher.updates]
    assert len(active) == 4
    assert [row[1]['position'] for row in active[0].updates] == [
        (9, 10), (10, 10), (9, 10), (10, 10)]
    for teacher in active:
        assert len(teacher.updates) == 4 and len(teacher.choices) == 3
        assert teacher.public['step'] == 123
        assert teacher.public['done'] is False and teacher.public['hit'] == 0


def test_action_and_replicate_order_are_canonical_and_global_rngs_unchanged():
    legacy_state, python_state = np.random.get_state(), random.getstate()
    public, belief, observation_kernel = packet(), point_belief((40, 41)), kernel()
    original_public = copy.deepcopy(public)
    original_belief, original_kernel = belief.copy(), observation_kernel.copy()
    first_events, second_events = [], []
    expected = sample(public=public, belief=belief, observation_kernel=observation_kernel,
                      replicate_ids=(9, 2), first_actions=(3, 1, 2, 0), emit=first_events.append)
    assert expected == sample(public=public, belief=belief, observation_kernel=observation_kernel,
                              replicate_ids=(2, 9), first_actions=(0, 1, 2, 3), emit=second_events.append)
    assert first_events == second_events
    assert [(r.replicate_id, r.first_action) for r in expected] == [
        (replicate, action) for replicate in (2, 9) for action in range(4)]
    after = np.random.get_state()
    assert legacy_state[0] == after[0] and legacy_state[2:] == after[2:]
    np.testing.assert_array_equal(legacy_state[1], after[1])
    assert random.getstate() == python_state
    assert public == original_public
    np.testing.assert_array_equal(belief, original_belief)
    np.testing.assert_array_equal(observation_kernel, original_kernel)


def test_corner_requires_all_and_only_eligible_first_actions():
    records = sample(public=packet((0, 0)), belief=point_belief((1, 0)), first_actions=(3, 1))
    assert records == (ContinuationRecord(7, 1, 1, True), ContinuationRecord(7, 3, 1, False))
    with pytest.raises((TypeError, ValueError)):
        sample(public=packet((0, 0)), belief=point_belief((1, 0)), first_actions=(0, 1, 2, 3))


@pytest.mark.parametrize('options', [
    {'seed': -1}, {'seed': 2**32}, {'seed': True}, {'anchor_id': 1.5},
    {'anchor_id': -1}, {'anchor_id': 2**32}, {'replicate_ids': ()},
    {'replicate_ids': (2, 2)}, {'replicate_ids': (True,)},
    {'replicate_ids': (-1,)}, {'replicate_ids': (2**32,)},
    {'horizon': 0}, {'horizon': 2189}, {'horizon': True},
    {'first_actions': (0, 1, 2)}, {'first_actions': (0, 1, 2, 2)},
    {'first_actions': (0, 1, 2, 4)}, {'first_actions': (False, 1, 2, 3)},
])
def test_invalid_sampling_declarations_are_rejected(options):
    with pytest.raises((TypeError, ValueError)):
        sample(**options)


@pytest.mark.parametrize('defect', ['subnormalized', 'current_cell', 'terminal', 'hidden_source'])
def test_invalid_anchor_cannot_be_repaired_into_a_label(defect):
    public, belief = packet(), point_belief()
    if defect == 'subnormalized':
        belief *= .5
    elif defect == 'current_cell':
        belief = point_belief((10, 10))
    elif defect == 'terminal':
        public = packet(done=True, hit=-2)
    else:
        public['source'] = [9, 10]
    with pytest.raises((TypeError, ValueError)):
        sample(public=public, belief=belief)


@pytest.mark.parametrize(('uniform', 'expected'), [
    (0., 1), (np.nextafter(.25, 0.), 1), (.25, 3),
    (np.nextafter(1., 0.), 3),
])
def test_categorical_half_open_boundaries_skip_zero_mass(uniform, expected):
    probabilities = np.array([0., .25, 0., .75, 0.], dtype=np.float64)
    assert rollout.categorical_index(probabilities, uniform) == expected


def test_categorical_tolerated_roundoff_closes_tail_without_input_mutation():
    probabilities = np.array([0., .5, .5 - 5e-11, 0.], dtype=np.float64)
    original = probabilities.copy()
    assert rollout.categorical_index(probabilities, np.nextafter(1., 0.)) == 2
    np.testing.assert_array_equal(probabilities, original)


@pytest.mark.parametrize('uniform', [-1e-12, 1., np.nan, np.inf, True])
def test_categorical_rejects_nonuniform_values(uniform):
    with pytest.raises((TypeError, ValueError)):
        rollout.categorical_index(np.array([.25, .75], dtype=np.float64), uniform)


@pytest.mark.parametrize('probabilities', [
    [], [[.5, .5]], [-.1, 1.1], [0., 0.], [.2, .3], [np.nan, 1.], [np.inf, 0.],
])
def test_categorical_rejects_unsupported_distributions(probabilities):
    with pytest.raises((TypeError, ValueError)):
        rollout.categorical_index(np.array(probabilities, dtype=np.float64), .4)


def returned(events, operation):
    return [event for event in events
            if event['event'] == 'return' and event['operation'] == operation]


def independent_stream(seed, anchor, replicate, channel):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [0x4F54544F, seed, anchor, replicate, channel])))


def test_source_and_hit_streams_reconstruct_declared_independent_channels():
    events = []
    belief = np.zeros((53, 53), dtype=np.float64)
    belief[8, 5], belief[15, 14] = .35, .65
    records = sample(belief=belief, replicate_ids=(2, 7), emit=events.append)
    assert len(records) == 8
    sources = returned(events, 'source_draw')
    hits = returned(events, 'hit_draw')
    assert len(sources) == 2 and len(hits) == 8
    for replicate in (2, 7):
        uniform = float(independent_stream(17, 23, replicate, 0).random())
        source = (8, 5) if uniform < .35 else (15, 14)
        row, = [event for event in sources if event['replicate_id'] == replicate]
        assert row['uniform'] == uniform and tuple(row['source']) == source
        assert row['selected_index'] == source[0] * 53 + source[1]
        hit_uniform = float(independent_stream(17, 23, replicate, 1).random())
        paired = [event for event in hits if event['replicate_id'] == replicate]
        assert {event['first_action'] for event in paired} == {0, 1, 2, 3}
        assert all(event['uniform'] == hit_uniform and event['draw_index'] == 0
                   and event['selected_index'] == 0 for event in paired)


def test_coordinate_oriented_likelihood_uses_signed_source_minus_position():
    observation_kernel = kernel()
    expected = np.array([.1, .2, .3, .4], dtype=np.float64)
    mirrored = np.array([.4, .3, .2, .1], dtype=np.float64)
    # Source (8,15), queried agent (11,10): signed displacement (-3,+5).
    observation_kernel[:, 50, 58] = expected
    observation_kernel[:, 56, 48] = mirrored
    teacher = TeacherSnapshot(packet(), point_belief((8, 15)), observation_kernel)
    actual = rollout.source_hit_probabilities(teacher, (8, 15), (11, 10))
    np.testing.assert_array_equal(actual, expected)
    actual[:] = 0
    np.testing.assert_array_equal(
        rollout.source_hit_probabilities(teacher, (8, 15), (11, 10)), expected)
    with pytest.raises((TypeError, ValueError)):
        rollout.source_hit_probabilities(teacher, (8, 15), (8, 15))


def test_shared_step_uniforms_can_produce_different_action_hits_and_stay_private(monkeypatch):
    def toward_anchor(teacher):
        x, y = teacher.public['position']
        return 1 if x < 10 else 0 if x > 10 else 3 if y < 10 else 2

    teachers = recording_teacher(monkeypatch, choose=toward_anchor)
    source = (40, 41)
    observation_kernel = kernel()
    targets = ((9, 10), (11, 10), (10, 9), (10, 11))
    for action, position in enumerate(targets):
        index = (53 + source[0] - position[0], 53 + source[1] - position[1])
        observation_kernel[:, index[0], index[1]] = 0
        observation_kernel[action, index[0], index[1]] = 1
    events = []
    records = sample(belief=point_belief(source), observation_kernel=observation_kernel,
                     horizon=2, emit=events.append)
    assert records == tuple(ContinuationRecord(7, a, 2, False) for a in range(4))
    uniforms = independent_stream(17, 23, 7, 1).random(2)
    hits = returned(events, 'hit_draw')
    assert len(hits) == 8
    for action in range(4):
        rows = [event for event in hits if event['first_action'] == action]
        assert [event['draw_index'] for event in rows] == [0, 1]
        assert [event['uniform'] for event in rows] == uniforms.tolist()
        assert [event['selected_index'] for event in rows] == [action, 0]
    for teacher in teachers:
        assert set(teacher.anchor) == {'position', 'hit', 'done', 'step', 'valid_actions'}
        assert all(not hasattr(teacher, name)
                   for name in ('source', 'seed', 'anchor_id', 'replicate_id', 'rng', 'stream', 'streams'))


def test_found_branch_consumes_no_odor_draw_and_records_only_after_update():
    events = []
    sample(emit=events.append)
    assert len(returned(events, 'anchor_snapshot')) == 1
    assert len(returned(events, 'teacher_snapshot')) == 4
    attempts = [event for event in events if event['event'] == 'attempt']
    returns = [event for event in events if event['event'] == 'return']
    attempted_ids = [event['operation_id'] for event in attempts]
    returned_ids = [event['operation_id'] for event in returns]
    assert len(attempted_ids) == len(set(attempted_ids))
    assert len(returned_ids) == len(set(returned_ids))
    assert set(attempted_ids) == set(returned_ids)
    by_id = {event['operation_id']: event for event in returns}
    identity = ('operation', 'seed', 'anchor_id', 'replicate_id',
                'first_action', 'local_step', 'from_step')
    for attempt in attempts:
        result = by_id[attempt['operation_id']]
        assert {key: attempt[key] for key in identity if key in attempt} == {
            key: result[key] for key in identity if key in result}
    assert len(returned(events, 'source_draw')) == 1
    assert len(returned(events, 'hit_draw')) == 3
    assert all(event['first_action'] != 0 for event in returned(events, 'hit_draw'))
    assert len([event for event in events if event['event'] == 'record']) == 4
    for action in range(4):
        update = next(i for i, event in enumerate(events)
                      if event['event'] == 'return' and event.get('operation') == 'teacher_update'
                      and event['first_action'] == action)
        complete = next(i for i, event in enumerate(events)
                        if event['event'] == 'record' and event['first_action'] == action)
        assert update < complete
    assert events[-1]['event'] == 'panel_complete'


def assert_unfinished(events, operation):
    attempts = {event['operation_id'] for event in events
                if event['event'] == 'attempt' and event['operation'] == operation}
    completed = {event['operation_id'] for event in events if event['event'] == 'return'}
    assert attempts - completed
    assert not any(event['event'] in ('record', 'panel_complete') for event in events)


def test_teacher_update_failure_preserves_pending_attempt_not_censored_record(monkeypatch):
    events = []
    recording_teacher(monkeypatch, update_error=RuntimeError('update failed'))
    with pytest.raises(RuntimeError, match='update failed'):
        sample(emit=events.append)
    assert_unfinished(events, 'teacher_update')


def test_interrupted_teacher_choice_propagates_without_fabricated_completion(monkeypatch):
    def interrupted(_teacher):
        raise KeyboardInterrupt('interrupted choice')

    events = []
    recording_teacher(monkeypatch, choose=interrupted)
    with pytest.raises(KeyboardInterrupt, match='interrupted choice'):
        sample(belief=point_belief((40, 41)), horizon=2, emit=events.append)
    assert_unfinished(events, 'teacher_choose')


def test_event_sink_failure_aborts_the_panel_instead_of_returning_partial_labels():
    events = []

    def emit(event):
        events.append(event)
        if event['event'] == 'attempt' and event['operation'] == 'movement':
            raise OSError('event sink full')

    with pytest.raises(OSError, match='event sink full'):
        sample(emit=emit)
    assert_unfinished(events, 'movement')


def test_teacher_factory_failure_preserves_attempt_and_does_not_make_a_label(monkeypatch):
    calls, events = [], []

    def factory(public, belief, observation_kernel):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError('fresh teacher unavailable')
        return TeacherSnapshot(public, belief, observation_kernel)

    monkeypatch.setattr(rollout, 'TeacherSnapshot', factory)
    with pytest.raises(RuntimeError, match='fresh teacher unavailable'):
        sample(emit=events.append)
    assert_unfinished(events, 'teacher_snapshot')


def test_later_action_failure_preserves_completed_record_but_returns_no_partial_panel(monkeypatch):
    calls, events = [], []

    def factory(public, belief, observation_kernel):
        calls.append(None)
        # Anchor validation, then first action succeed; second action fails.
        if len(calls) == 3:
            raise RuntimeError('later teacher unavailable')
        return TeacherSnapshot(public, belief, observation_kernel)

    monkeypatch.setattr(rollout, 'TeacherSnapshot', factory)
    with pytest.raises(RuntimeError, match='later teacher unavailable'):
        sample(emit=events.append)
    records = [event for event in events if event['event'] == 'record']
    assert len(records) == 1
    assert {key: records[0][key] for key in ('replicate_id', 'first_action', 'steps', 'found')} == {
        'replicate_id': 7, 'first_action': 0, 'steps': 1, 'found': True}
    assert len(returned(events, 'anchor_snapshot')) == 1
    assert len(returned(events, 'teacher_snapshot')) == 1
    pending, = [event for event in events if event['event'] == 'attempt'
                and event['operation'] == 'teacher_snapshot' and event['first_action'] == 1]
    assert pending['operation_id'] not in {
        event['operation_id'] for event in events if event['event'] == 'return'}
    assert not any(event['event'] == 'panel_complete' for event in events)


def test_external_stop_check_propagates_and_prevents_panel_completion():
    events = []

    def check():
        if any(event.get('operation') == 'movement' for event in events):
            raise TimeoutError('collection bound reached')

    with pytest.raises(TimeoutError, match='collection bound reached'):
        sample(belief=point_belief((40, 41)), emit=events.append, check=check)
    assert len(returned(events, 'movement')) == 1
    assert not returned(events, 'teacher_update')
    assert not any(event['event'] in ('record', 'panel_complete') for event in events)
