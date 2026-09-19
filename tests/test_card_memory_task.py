"""Handwritten public frames only: no environment, model or RNG construction."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from openjev.research.card_memory_task import PublicCardTracker, PublicTeacher, Reveal


def frame(**positions):
    result = np.full(52, 13, dtype=np.int64)
    for pos, rank in positions.items():
        result[int(pos[1:])] = rank
    return result


def uniform():
    return np.full((52, 13), 1 / 13, dtype=np.float64)


def tracker():
    result = PublicCardTracker()
    result.reset(frame())
    return result


def state(value):
    return (value.observation.tobytes(), value.seen.tobytes(), value.matched.tobytes(),
            value.pending, value.step, value.terminal)


def test_reset_and_queries_have_no_rank_memory_or_rng():
    value = tracker()
    assert value.step == 0 and value.pending is None and not value.terminal
    assert not value.seen.any() and not value.matched.any()
    assert set(PublicCardTracker.__slots__) == {'_obs', '_seen', '_matched', '_pending', '_step'}
    before = state(value)
    assert value.choose(uniform()) == 0
    assert value.choose(uniform(), rng=object()) == 0  # No random path at zero mixture.
    assert state(value) == before


def test_mismatch_return_is_visible_but_next_turn_has_no_pending():
    value = tracker()
    assert value.observe(0, 0, frame(p0=2)) == Reveal(0, 2)
    assert value.pending == 0
    event = value.observe(1, -2 / 104, frame(p0=2, p1=8))
    assert event == Reveal(1, 8) and value.pending is None
    probs = value.probabilities(uniform())
    assert probs[0, 2] == probs[1, 8] == 1
    value.observe(2, 0, frame(p2=4))
    assert value.pending == 2 and value.seen[:3].all()
    assert value.observation[0] == value.observation[1] == 13


def test_match_bookkeeping_and_pending_rank_picker():
    value = tracker()
    value.observe(4, 0, frame(p4=6))
    # Every unseen prediction is overridden, even if the model is overconfident.
    raw = uniform()
    raw[10] = 0
    raw[10, 6] = 1
    assert value.choose(raw) == 0
    value.observe(8, 2 / 52, frame(p4=6, p8=6))
    assert value.matched[[4, 8]].all() and value.pending is None
    assert value.choose(uniform()) == 0
    value.observe(0, 0, frame(p0=2, p4=6, p8=6))
    assert value.pending == 0 and value.choose(uniform()) == 1


@pytest.mark.parametrize('with_pending', [False, True])
def test_native_matched_position_penalty_is_not_a_new_match(with_pending):
    value = tracker()
    value.observe(0, 0, frame(p0=5))
    value.observe(1, 2 / 52, frame(p0=5, p1=5))
    if with_pending:
        value.observe(2, 0, frame(p0=5, p1=5, p2=3))
    after = frame(p0=5, p1=5, **({'p2': 3} if with_pending else {}))
    event = value.observe(0, -(2 if with_pending else 1) / 104, after)
    assert event == Reveal(0, 5) and value.pending is None
    assert value.matched.sum() == 2


def test_same_card_twice_is_a_mismatch_even_when_ranks_equal():
    value = tracker()
    value.observe(7, 0, frame(p7=3))
    value.observe(7, -2 / 104, frame(p7=3))
    assert value.pending is None and not value.matched.any()


def test_known_hidden_model_probabilities_drive_pair_and_pending_choice():
    value = tracker()
    value.observe(0, 0, frame(p0=2))
    value.observe(1, -2 / 104, frame(p0=2, p1=3))
    value.observe(2, 0, frame(p2=2))
    raw = uniform()
    raw[0] = 0
    raw[0, 2] = 1
    raw[1] = 0
    raw[1, 3] = 1
    assert value.choose(raw) == 0
    # Complete mismatch, then exact known hidden pair outranks unseen cards.
    value.observe(3, -2 / 104, frame(p2=2, p3=4))
    assert value.choose(raw) == 0


def test_pair_ties_choose_lowest_first_then_lowest_partner():
    value = tracker()
    value.observe(5, 0, frame(p5=2))
    value.observe(7, -2 / 104, frame(p5=2, p7=3))
    assert value.choose(uniform()) == 0
    value.observe(0, 0, frame(p0=4))
    raw = uniform()
    raw[[5, 7]] = 0
    raw[[5, 7], 4] = 1
    assert value.choose(raw) == 5


class ScriptedRandom:
    def __init__(self, draw, index):
        self.draw, self.index, self.calls = draw, index, []

    def random(self):
        self.calls.append('random')
        return self.draw

    def integers(self, high):
        self.calls.append(('integers', high))
        return self.index


def test_random_mixture_uses_only_explicit_generator_and_legal_choices():
    value = tracker()
    value.observe(0, 0, frame(p0=6))
    value.observe(1, 2 / 52, frame(p0=6, p1=6))
    value.observe(2, 0, frame(p0=6, p1=6, p2=7))
    rng = ScriptedRandom(.2, 0)
    before = state(value)
    assert value.choose(uniform(), rng, .5) == 3
    assert rng.calls == ['random', ('integers', 49)]
    assert state(value) == before
    rng = ScriptedRandom(.8, 999)
    assert value.choose(uniform(), rng, .5) == 3
    assert rng.calls == ['random']
    with pytest.raises(ValueError, match='explicit'):
        value.choose(uniform(), None, .5)


@pytest.mark.parametrize('chance', [-.1, 1.1, float('nan'), True, '0'])
def test_bad_mixture_rejected_without_state_change(chance):
    value = tracker()
    before = state(value)
    with pytest.raises(ValueError):
        value.choose(uniform(), random_chance=chance)
    assert state(value) == before


def test_teacher_targets_are_hidden_seen_only_and_age_is_public_visibility():
    value, teacher = tracker(), PublicTeacher()
    teacher.reset(frame())
    labels, mask, ages = teacher.targets(frame())
    assert not mask.any() and np.all(labels == -1) and np.all(ages == -1)
    for action, reward, after in [(0, 0, frame(p0=2)),
                                  (1, -2 / 104, frame(p0=2, p1=3)),
                                  (2, 0, frame(p2=6))]:
        value.observe(action, reward, after)
        teacher.observe(after)
    labels, mask, ages = teacher.targets(value.observation)
    assert np.flatnonzero(mask).tolist() == [0, 1]
    assert labels[:3].tolist() == [2, 3, -1]
    # Position0 was written on action1, but remained publicly visible on action2.
    assert ages[:3].tolist() == [1, 1, -1]
    assert labels.dtype == np.int64 and mask.dtype == bool and ages.dtype == np.int32
    exact = teacher.probabilities()
    assert exact[0, 2] == exact[1, 3] == exact[2, 6] == 1
    np.testing.assert_array_equal(exact[3], uniform()[3])
    with pytest.raises(ValueError, match='latest'):
        teacher.targets(frame(p0=2, p1=3))


def test_teacher_rejects_changed_rank_atomically_and_reset_forgets():
    value = PublicTeacher()
    value.reset(frame())
    value.observe(frame(p0=3))
    before = value.probabilities().copy()
    with pytest.raises(ValueError, match='changed rank'):
        value.observe(frame(p0=4))
    np.testing.assert_array_equal(value.probabilities(), before)
    value.observe(frame(p1=2))
    assert value.targets(frame(p1=2))[2][0] == 1  # Failed call did not tick clock.
    value.reset(frame())
    assert not value.targets(frame())[1].any()


def test_matched_positions_are_never_recall_targets():
    teacher = PublicTeacher()
    teacher.reset(frame())
    for after in [frame(p0=3), frame(p0=3, p1=3), frame(p0=3, p1=3, p2=4)]:
        teacher.observe(after)
    assert not teacher.targets(after)[1].any()


def test_historical_ranks_do_not_leak_into_public_tracker():
    left, right = tracker(), tracker()
    for value, ranks in [(left, (1, 2)), (right, (7, 8))]:
        value.observe(0, 0, frame(p0=ranks[0]))
        value.observe(1, -2 / 104, frame(p0=ranks[0], p1=ranks[1]))
        value.observe(2, 0, frame(p2=4))
    assert state(left) == state(right)
    np.testing.assert_array_equal(left.probabilities(uniform()), right.probabilities(uniform()))
    assert left.choose(uniform()) == right.choose(uniform())


@pytest.mark.parametrize(('action', 'reward', 'after'), [
    (False, 0, frame(p0=2)), (52, 0, frame(p0=2)), (0, float('nan'), frame(p0=2)),
    (0, True, frame(p0=2)), (0, .1, frame(p0=2)), (0, 0, frame()),
    (0, 0, frame(p0=2, p1=3)), (0, 0, np.zeros(52, dtype=np.float64)),
])
def test_malformed_transition_preserves_actual_prefix(action, reward, after):
    value = tracker()
    before = state(value)
    with pytest.raises((ValueError, TypeError)):
        value.observe(action, reward, after)
    assert state(value) == before


def test_still_visible_rank_change_and_missing_matched_card_rejected():
    value = tracker()
    value.observe(0, 0, frame(p0=2))
    before = state(value)
    with pytest.raises(ValueError, match='changed rank'):
        value.observe(1, -2 / 104, frame(p0=3, p1=4))
    assert state(value) == before
    value.observe(1, 2 / 52, frame(p0=2, p1=2))
    with pytest.raises(ValueError, match='returned frame'):
        value.observe(2, 0, frame(p2=4))


@pytest.mark.parametrize('bad', [np.zeros((52, 13)), np.full((52, 13), np.nan),
                                np.ones((52, 13), dtype=np.int64), np.ones((51, 13))])
def test_bad_model_probabilities_rejected_even_when_entries_would_be_overridden(bad):
    with pytest.raises(ValueError, match='probabilities'):
        tracker().probabilities(bad)


def test_no_input_or_output_aliases_and_reveal_is_scalar_frozen():
    value, teacher = tracker(), PublicTeacher()
    teacher.reset(frame())
    after = frame(p4=3)
    event = value.observe(4, 0, after)
    teacher.observe(after)
    after[:] = 13
    assert value.observation[4] == 3 and teacher.probabilities()[4, 3] == 1
    raw = uniform()
    saved = raw.copy()
    result = value.probabilities(raw)
    np.testing.assert_array_equal(raw, saved)
    for array in [value.observation, value.seen, value.matched, result,
                  teacher.probabilities(), *teacher.targets(frame(p4=3))]:
        assert not array.flags.writeable
        array.flags.writeable = True
        array.flat[0] = 0
    assert value.observation[4] == 3 and not value.matched.any()
    with pytest.raises(FrozenInstanceError):
        event.rank = 4


def test_all_matched_and_final_budget_have_terminal_boundaries():
    value = tracker()
    visible = frame()
    for first in range(0, 52, 2):
        visible[first] = (first // 2) % 13
        value.observe(first, 0, visible.copy())
        visible[first + 1] = visible[first]
        value.observe(first + 1, 2 / 52, visible.copy())
    assert value.terminal and value.step == 52 and value.matched.all()
    with pytest.raises(ValueError, match='termination'):
        value.observe(0, -1 / 104, visible)
    with pytest.raises(ValueError, match='no decision'):
        value.choose(uniform())
    value.reset(frame())
    teacher = PublicTeacher()
    teacher.reset(frame())
    for step in range(104):
        after = frame(p0=3)
        value.observe(0, 0 if step % 2 == 0 else -2 / 104, after)
        teacher.observe(after)
    assert value.terminal and value.step == 104 and not value.matched.any()
    with pytest.raises(ValueError, match='budget'):
        teacher.observe(after)


def test_float32_native_reward_is_accepted():
    value = tracker()
    value.observe(0, 0, frame(p0=3))
    value.observe(1, np.float32(2 / 52), frame(p0=3, p1=3))
    assert value.matched.sum() == 2


@pytest.mark.parametrize('kind', [PublicCardTracker, PublicTeacher])
def test_reset_required_and_visible_reset_rejected(kind):
    value = kind()
    with pytest.raises(ValueError, match='reset'):
        value.probabilities(uniform()) if kind is PublicCardTracker else value.probabilities()
    with pytest.raises(ValueError, match='hide every'):
        value.reset(frame(p0=1))


def test_module_has_no_environment_model_or_generator_construction():
    path = Path(__file__).resolve().parents[1] / 'src/openjev/research/card_memory_task.py'
    tree = ast.parse(path.read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert set(imported) == {'dataclasses', 'math', 'numpy'}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {'default_rng', 'seed', 'RandomState'}
                   for node in ast.walk(tree))
