"""Small fabricated exact-group and saved-score arithmetic, without neural calls."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('target_conflicts_fixture', ROOT / 'scripts/diagnose_otto_target_conflicts.py')
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def unchanged(values, _view):
    return values.copy()


def fixture():
    features = np.ones((2, 2836), dtype=np.float32)
    targets = np.array([[1., -1., 99., -99.], [-1., 1., -50., 20.]], dtype=np.float32)
    masks = np.array([[True, True, False, False]] * 2)
    weights = np.array([1., 3.], dtype=np.float32)
    return features, targets, masks, weights


def calculate(*args, **kwargs):
    return diagnostic.compute_conflicts(*args, transform_features=unchanged, transform_actions=unchanged, **kwargs)


def test_weighted_masked_conflict_has_hand_computed_floor_not_weight_normalization():
    arrays, members, groups, result = calculate(*fixture())
    assert len(members) == 16 and len(groups) == 1
    assert [r['occurrence_id'] for r in members] == list(range(16))
    assert groups[0]['mean'] == [-.5, .5, 0., 0.]
    assert groups[0]['weight_sum'] == 32
    assert groups[0]['eligible_count'] == 2
    assert groups[0]['deviation_numerator'] == 24
    assert result['denominator'] == 16 and result['B'] == 1.5
    np.testing.assert_array_equal(arrays['weighted_deviations'][0], np.full(8, 2.25))
    np.testing.assert_array_equal(arrays['weighted_deviations'][1], np.full(8, .75))
    assert result['rounded_weight_sum'] == 4  # Deliberately not N: no normalization by sum weights.


@pytest.mark.parametrize('difference', ['mask', 'signed_zero', 'one_ulp'])
def test_different_full_input_bytes_or_masks_never_merge(difference):
    features, targets, masks, weights = fixture()
    features[:, 0] = 0.
    if difference == 'mask':
        masks[1] = [True, False, True, False]
    elif difference == 'signed_zero':
        features[1, 0] = -0.
    else:
        features[1, 0] = np.nextafter(np.float32(0), np.float32(1))
    arrays, _, groups, result = calculate(features, targets, masks, weights)
    assert len(groups) == 2 and result['B'] == 0
    assert np.all(arrays['group_ids'][0] != arrays['group_ids'][1])


def test_hash_bucket_collision_is_resolved_using_complete_bytes(monkeypatch):
    features, targets, masks, weights = fixture()
    features[1, -1] = 2.
    monkeypatch.setattr(diagnostic, 'key_hash', lambda _raw: 'fabricated-collision')
    _, members, groups, result = calculate(features, targets, masks, weights)
    assert len(groups) == 2 and result['B'] == 0
    assert members[0]['key_sha256'] != members[8]['key_sha256']


def test_unequal_group_sizes_use_occurrence_denominator_not_equal_group_average():
    features = np.ones((3, 2836), dtype=np.float32)
    features[2, 0] = 2.
    targets = np.array([[1., -1., 0., 0.], [-1., 1., 0., 0.], [3., -3., 0., 0.]], dtype=np.float32)
    masks = np.array([[True, True, False, False]] * 3)
    _, _, groups, result = calculate(features, targets, masks, np.ones(3, dtype=np.float32))
    assert [g['count'] for g in groups] == [16, 8]
    assert [g['deviation_numerator'] for g in groups] == [16, 0]
    assert result['denominator'] == 24 and result['B'] == pytest.approx(2 / 3)


def test_singleton_contributions_are_exact_zero_without_multiply_divide_roundoff():
    features = np.ones((1, 2836), dtype=np.float32)
    targets = np.array([[.123, .234, .345, .456]], dtype=np.float32)
    masks, weights = np.ones((1, 4), dtype=bool), np.array([1 / 3], dtype=np.float32)

    def distinct_view(values, view):
        result = values.copy()
        result[:, 0] = view + 1
        return result
    arrays, _, groups, result = diagnostic.compute_conflicts(features, targets, masks, weights,
        transform_features=distinct_view, transform_actions=unchanged)
    assert result['singleton_groups'] == 8 and result['B'] == 0
    assert all(g['mean'] == arrays['centered_targets_float64'][0, v].tolist() for v, g in enumerate(groups))
    assert not arrays['weighted_deviations'].any()


def test_all_eight_qualified_views_retain_action_permutation_and_input_ownership():
    from openjev.research.otto_symmetry_head import transform_actions, transform_features
    features, targets, masks, weights = fixture()
    features[:] = np.arange(2836, dtype=np.float32) + 1
    originals = [v.tobytes() for v in (features, targets, masks, weights)]
    calls, events = [], []

    def observed_transform(values, view):
        calls.append((values.shape, view))
        return transform_features(values, view)
    arrays, members, _, result = diagnostic.compute_conflicts(features, targets, masks, weights,
        transform_features=observed_transform, transform_actions=transform_actions, observe=events.append)
    assert calls == [((2, 2836), v) for v in range(8)]
    assert len(events) == 16 and all(e['event'] == ('attempt' if i % 2 == 0 else 'return') for i, e in enumerate(events))
    np.testing.assert_array_equal(arrays['targets_float32'][0, 1], targets[0, [3, 2, 0, 1]])
    np.testing.assert_array_equal(arrays['masks'][0, 1], masks[0, [3, 2, 0, 1]])
    assert len(members) == 16 and result['B'] == pytest.approx(1.5)
    assert originals == [v.tobytes() for v in (features, targets, masks, weights)]


def test_saved_score_reduction_hits_free_floor_and_ignores_common_offsets_and_blocked_values():
    arrays, _, groups, result = calculate(*fixture())
    scores = np.tile(np.array([99.5, 100.5, 1e20, -1e20], dtype=np.float32), (2, 8, 1))
    compared, losses, spread = diagnostic.compare_scores(scores, arrays, groups, result['B'])
    assert compared['L'] == compared['B'] == compared['original_reduction_L'] == 1.5
    assert compared['L_minus_B'] == 0 and not spread.any()
    np.testing.assert_array_equal(losses[0], np.full(8, 2.25))
    np.testing.assert_array_equal(losses[1], np.full(8, .25))


def test_group_score_spread_is_centered_and_signed_excess_is_not_clipped():
    arrays, _, groups, result = calculate(*fixture())
    scores = np.tile(np.array([-.5, .5, 0., 0.], dtype=np.float32), (2, 8, 1))
    scores[1, :, :2] = [-.75, .75]
    compared, _, spread = diagnostic.compare_scores(scores, arrays, groups, result['B'])
    assert spread.tolist() == [.25]
    assert compared['maximum_within_group_centered_score_spread'] == .25
    assert compared['groups_with_nonzero_score_spread'] == 1
    other, _, _ = diagnostic.compare_scores(scores, arrays, groups, compared['L'] + .01)
    assert other['L_minus_B'] == pytest.approx(-.01)


@pytest.mark.parametrize('bad', ['dtype', 'nan', 'zero_weight', 'empty_mask'])
def test_invalid_arithmetic_inputs_fail_before_transform(bad):
    features, targets, masks, weights = fixture()
    if bad == 'dtype':
        weights = weights.astype(np.float64)
    elif bad == 'nan':
        targets[0, 0] = np.nan
    elif bad == 'zero_weight':
        weights[0] = 0
    else:
        masks[0] = False
    events = []
    with pytest.raises(ValueError, match='exact f32 inputs'):
        calculate(features, targets, masks, weights, observe=events.append)
    assert events == []


def test_return_fsync_failure_preserves_pending_operation(tmp_path, monkeypatch):
    run = diagnostic.Run(SimpleNamespace(output=tmp_path))
    event = {'event': 'attempt', 'operation': 'transform', 'view': 0, 'rows': 2}
    run.event(event)

    def fail(_fd):
        raise OSError('fabricated durable return failure')
    monkeypatch.setattr(diagnostic.os, 'fsync', fail)
    with pytest.raises(OSError, match='durable return'):
        run.event({**event, 'event': 'return'})
    assert run.pending == {('transform', 0): event}
    assert run.counts == {'transform_attempted': 1}


def test_return_identity_mismatch_is_not_acknowledged(tmp_path):
    run = diagnostic.Run(SimpleNamespace(output=tmp_path))
    event = {'event': 'attempt', 'operation': 'transform', 'view': 0, 'rows': 2}
    run.event(event)
    with pytest.raises(ValueError, match='matching pending'):
        run.event({**event, 'event': 'return', 'rows': 3})
    assert run.pending and run.counts == {'transform_attempted': 1}


def test_late_publication_check_demotes_completed_receipt(tmp_path, monkeypatch):
    output = tmp_path / 'exclusive'
    run = diagnostic.Run(SimpleNamespace(output=output))

    def admit():
        run.start = 1
        run.clock = SimpleNamespace(now_ns=lambda: 2)
        run.inputs = {}
        run.bound[str(ROOT / diagnostic.SELF)] = diagnostic.digest(ROOT / diagnostic.SELF)

    def compute():
        for name in diagnostic.PAYLOADS:
            diagnostic.write(output / name, {'fabricated': True})
        run.counts.update(transform_attempted=8, transform_returned=8, score_reduction_attempted=6, score_reduction_returned=6)

    def check():
        if (output / 'receipt.json').exists():
            raise RuntimeError('fabricated late limit')
    monkeypatch.setattr(run, 'admit', admit)
    monkeypatch.setattr(run, 'compute', compute)
    monkeypatch.setattr(run, 'check', check)
    monkeypatch.setattr(diagnostic.signal, 'signal', lambda *_args: None)
    with pytest.raises(RuntimeError, match='late limit'):
        run.execute()
    assert json.loads((output / 'receipt.invalid.json').read_text())['status'] == 'completed'
    assert json.loads((output / 'receipt.json').read_text())['status'] == 'failed'
