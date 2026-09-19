"""Pure numeric/public-prefix tests; saved pilot parity invokes no model/native."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from openjev.research.card_memory_picker import TIE_ATOL, CardPolicyTracker, _tied_indices
from openjev.research.card_memory_task import PublicCardTracker

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / 'output/card-memory-pilot-v1/evaluation-01'
COMPLETED_SHA = '21b6fc27cd87088c9bfb6a68eb339ac768fefb1aff30667d4498f36533835eb2'


def hidden():
    return np.full(52, 13, dtype=np.int64)


def uniform():
    return np.full((52, 13), 1 / 13, dtype=np.float64)


def initialized(policy):
    tracker = CardPolicyTracker(policy)
    tracker.reset(hidden())
    return tracker


def snapshot(tracker):
    return (tracker.observation.tobytes(), tracker.seen.tobytes(), tracker.matched.tobytes(),
            tracker.pending, tracker.step, tracker.policy)


def all_seen(policy):
    tracker = initialized(policy)
    # Every adjacent pair mismatches, so every position becomes seen, none matched.
    for pos in range(0, 52, 2):
        observation = hidden()
        observation[pos] = 0
        tracker.observe(pos, 0, observation)
        observation[pos + 1] = 1
        tracker.observe(pos + 1, -2 / 104, observation)
    return tracker


@pytest.mark.parametrize(('gap', 'expected'), [
    (np.nextafter(TIE_ATOL, 0), [0, 1]), (TIE_ATOL, [0, 1]),
    (np.nextafter(TIE_ATOL, np.inf), [1]),
])
def test_direct_float64_inclusive_tie_threshold(gap, expected):
    assert _tied_indices(np.array([0., gap], dtype=np.float64), TIE_ATOL).tolist() == expected


@pytest.mark.parametrize('policy', ['B', 'C'])
@pytest.mark.parametrize(('gap', 'action'), [
    (np.nextafter(TIE_ATOL, 0), 0), (TIE_ATOL, 0),
    (np.nextafter(TIE_ATOL, np.inf), 1),
])
def test_second_card_actual_probability_threshold_and_same_b_c(policy, gap, action):
    tracker = all_seen(policy)
    observation = hidden()
    observation[51] = 1
    tracker.observe(51, 0, observation)
    raw = np.zeros((52, 13), dtype=np.float64)
    raw[:, 2] = 1
    raw[1, 1], raw[1, 2] = gap, 1 - gap
    chosen, probabilities, diag = tracker.decision(raw)
    assert probabilities[1, 1] == gap  # These rows have exact computed mass one.
    assert chosen == action and not diag['firstphase']
    assert diag['unseen_tied_endpoints'] == [] and not diag['selected_unseen']
    assert diag['Baction_ifC'] == (action if policy == 'C' else None)


def test_c_can_take_higher_unseen_endpoint_of_unique_best_pair():
    trackers = [initialized(policy) for policy in ('B', 'C')]
    for tracker in trackers:
        observation = hidden()
        # Match0..49 through public records; only50 and51 remain legally eligible.
        for pos in range(0, 50, 2):
            observation[pos] = (pos // 2) % 13
            tracker.observe(pos, 0, observation.copy())
            observation[pos + 1] = observation[pos]
            tracker.observe(pos + 1, 2 / 52, observation.copy())
        observation[50] = 2
        tracker.observe(50, 0, observation.copy())
        tracker.observe(50, -2 / 104, observation.copy())
    b, p_b, d_b = trackers[0].decision(uniform())
    c, p_c, d_c = trackers[1].decision(uniform())
    assert (b, c) == (50, 51)
    np.testing.assert_array_equal(p_b, p_c)
    assert d_b['tie_size'] == d_c['tie_size'] == 1
    assert d_c['unseen_tied_endpoints'] == [51] and d_c['Baction_ifC'] == 50
    assert d_c['selected_unseen'] and not d_b['selected_unseen']


def test_no_unseen_endpoint_falls_back_to_b():
    b, c = all_seen('B'), all_seen('C')
    assert b.decision(uniform())[0] == c.decision(uniform())[0]
    assert c.decision(uniform())[2]['unseen_tied_endpoints'] == []


def test_unseen_outside_tie_set_does_not_override_a_stronger_known_match():
    tracker = initialized('C')
    for action, rank, reward, previous in [(0, 2, 0, None), (1, 3, -2 / 104, (0, 2)),
                                            (2, 4, 0, None), (3, 5, -2 / 104, (2, 4))]:
        observation = hidden()
        if previous:
            observation[previous[0]] = previous[1]
        observation[action] = rank
        tracker.observe(action, reward, observation)
    raw = uniform()
    raw[:2] = 0
    raw[:2, 6] = 1  # Hidden seen0/1 predict a certain match; public2/3 override.
    action, _, diag = tracker.decision(raw)
    assert action == 0 and diag['tie_size'] == 1
    assert diag['unseen_tied_endpoints'] == [] and not diag['selected_unseen']


@pytest.mark.parametrize('policy', ['B', 'C'])
def test_normalization_precedes_public_overrides_and_is_row_scale_invariant(policy):
    tracker = all_seen(policy)
    raw = np.arange(1, 52 * 13 + 1, dtype=np.float64).reshape(52, 13)
    row_scale = np.arange(1, 53, dtype=np.float64)[:, None]
    a, p, d = tracker.decision(raw)
    scaled_a, scaled_p, _ = tracker.decision(raw * row_scale)
    assert a == scaled_a
    np.testing.assert_allclose(p, scaled_p, rtol=0, atol=3e-17)
    np.testing.assert_array_equal(p[50], np.eye(13)[0])
    np.testing.assert_array_equal(p[51], np.eye(13)[1])
    assert d['rowmass_max_abs'] == np.max(abs(raw.sum(1) - 1))
    empty = initialized(policy)
    np.testing.assert_array_equal(empty.probabilities(raw), uniform())


def test_a_calls_frozen_original_choose_once_and_has_exact_overrides(monkeypatch):
    original, calls = PublicCardTracker.choose, []

    def spy(self, raw, rng=None, random_chance=0):
        calls.append((self, raw))
        return original(self, raw, rng, random_chance)

    monkeypatch.setattr(PublicCardTracker, 'choose', spy)
    tracker = all_seen('A')
    raw = uniform().astype(np.float32)
    action, probabilities, diagnostics = tracker.decision(raw)
    assert len(calls) == 1 and calls[0][0] is tracker and calls[0][1] is raw
    np.testing.assert_array_equal(probabilities, PublicCardTracker.probabilities(tracker, raw))
    assert action == original(tracker, raw)
    assert diagnostics['Baction_ifC'] is None


@pytest.mark.parametrize('policy', ['A', 'B', 'C'])
def test_decision_is_pure_and_outputs_are_detached_json_finite(policy):
    tracker = all_seen(policy)
    raw = uniform()
    before, copied = snapshot(tracker), raw.copy()
    action, probabilities, diagnostics = tracker.decision(raw)
    assert action == tracker.choose(raw) and snapshot(tracker) == before
    np.testing.assert_array_equal(raw, copied)
    assert not probabilities.flags.writeable and not np.shares_memory(raw, probabilities)
    probabilities.flags.writeable = True
    probabilities[:] = 0
    assert snapshot(tracker) == before
    assert set(diagnostics) == {'tie_size', 'firstphase', 'unseen_tied_endpoints', 'selected_unseen',
                                'Baction_ifC', 'rowmass_max_abs'}
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize(('rng', 'chance'), [(object(), 0), (None, .1), (None, np.nan), (None, True)])
def test_rng_and_mixing_are_rejected(rng, chance):
    with pytest.raises(ValueError, match='deterministic'):
        initialized('B').choose(uniform(), rng, chance)


@pytest.mark.parametrize('policy', ['A', 'B', 'C'])
@pytest.mark.parametrize('bad', [np.zeros((52, 13)), np.full((52, 13), np.nan),
                                np.full((52, 13), -1.), np.ones((52, 12)), np.ones((52, 13), dtype=np.int64)])
def test_invalid_raw_inputs_rejected_without_mutation(policy, bad):
    tracker = initialized(policy)
    before = snapshot(tracker)
    with pytest.raises(ValueError):
        tracker.decision(bad)
    assert snapshot(tracker) == before


def test_zero_and_overflowing_mass_rejected_even_if_unseen_override_would_hide_it():
    for raw in (np.zeros((52, 13)), np.full((52, 13), np.finfo(np.float64).max)):
        with pytest.raises(ValueError, match='mass'):
            initialized('B').probabilities(raw)


def test_unreset_and_terminal_boundaries_and_invalid_policy():
    with pytest.raises(ValueError, match='reset'):
        CardPolicyTracker('B').decision(uniform())
    with pytest.raises(ValueError, match='policy'):
        CardPolicyTracker('D')
    tracker = initialized('B')
    observation = hidden()
    observation[0] = 2
    for step in range(104):
        tracker.observe(0, 0 if step % 2 == 0 else -2 / 104, observation)
    with pytest.raises(ValueError, match='no decision'):
        tracker.decision(uniform())


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.skipif(not (EVALUATION / 'completed.json').exists(), reason='Original authenticated pilot artifacts unavailable')
def test_a_parity_on_all1280_authenticated_saved_original_public_prefixes():
    start = time.monotonic()
    completed = EVALUATION / 'completed.json'
    assert file_sha(completed) == COMPLETED_SHA
    receipt = json.loads(completed.read_text())
    assert receipt['status'] == 'complete' and receipt['episodes'] == 1280
    actual = {str(p.relative_to(EVALUATION)) for p in EVALUATION.rglob('*') if p.is_file()}
    assert actual == set(receipt['files']) | {'completed.json'}
    for relative, binding in receipt['files'].items():
        assert time.monotonic() - start < 180, 'Bounded saved-only parity check exceeded180seconds'
        path = EVALUATION / relative
        assert file_sha(path) == binding['sha256'] and path.stat().st_size == binding['bytes']
    episodes = steps = 0
    for path in sorted(EVALUATION.glob('controllers/*/episodes/*.npz')):
        tracker = initialized('A')
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ('raw_probabilities', 'picker_probabilities', 'actions',
                                                 'rewards', 'observations')}
        tracker.reset(arrays['observations'][0])
        for step, action in enumerate(arrays['actions']):
            assert time.monotonic() - start < 180, 'Bounded saved-only parity check exceeded180seconds'
            selected, probabilities, _ = tracker.decision(arrays['raw_probabilities'][step])
            assert selected == action
            np.testing.assert_array_equal(probabilities, arrays['picker_probabilities'][step])
            tracker.observe(int(action), float(arrays['rewards'][step]), arrays['observations'][step + 1])
            steps += 1
        assert tracker.terminal
        episodes += 1
    assert episodes == 1280 and 0 < steps <= 1280 * 104
    assert file_sha(completed) == COMPLETED_SHA
    print(f'authenticated saved-only A parity: {episodes}episodes/{steps}decisions, {time.monotonic()-start:.3f}s')
