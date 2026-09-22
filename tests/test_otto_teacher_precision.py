"""Fabricated precision targets and a four-update paired training fixture.

No empirical inputs, continuation sampling, native simulator or evaluation.
Checkpoint callbacks keep in-memory exports solely for mechanics qualification.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from openjev.research import otto_cost_regression as regression
from openjev.research import otto_teacher_precision as precision


def data():
    costs = np.array([[1, 3, np.inf, np.inf], [2, 6, np.inf, np.inf]], dtype=np.float64)
    allowed = np.isfinite(costs)
    episodes = ['first', 'second']
    reference = regression.build_targets(costs, allowed, episodes, kind='continuation')
    return costs, allowed, episodes, reference


def test_r64_uses_original_r16_scale_and_fresh_immutable_arrays():
    r16, mask, episodes, reference = data()
    r64 = np.where(mask, r16 * 10, np.inf)
    before = [value.copy() for value in (r16, r64, mask)]
    bundle = precision.build_precision_targets(r16, r64, mask, episodes, reference_continuation=reference)
    assert bundle['version'] == 'otto-teacher-precision-v1'
    assert bundle['r16']['scale'] == bundle['r64']['scale'] == math.sqrt(2.5)
    assert regression.build_targets(r64, mask, episodes, kind='continuation')['scale'] != bundle['r64']['scale']
    np.testing.assert_array_equal(bundle['r64']['centered_float64'], [[-10, 10, 0, 0], [-20, 20, 0, 0]])
    expected = np.array([[-10, 10, 0, 0], [-20, 20, 0, 0]], dtype=np.float64) / math.sqrt(2.5)
    assert bundle['r64']['scaled_float32'].tobytes() == expected.astype(np.float32).tobytes()
    for key, value in reference.items():
        if isinstance(value, np.ndarray):
            assert bundle['r16'][key].tobytes() == value.tobytes()
            assert not np.shares_memory(bundle['r16'][key], value)
    for arm in precision.ARMS:
        assert bundle[arm]['kind'] == arm
        for value in bundle[arm].values():
            if isinstance(value, np.ndarray):
                with pytest.raises(ValueError):
                    value.setflags(write=True)
    for original, saved in zip((r16, r64, mask), before, strict=True):
        np.testing.assert_array_equal(original, saved)
    r16[0, 0] = 500
    r64[0, 0] = 600
    mask[0, 0] = False
    assert bundle['r16']['centered_float64'][0, 0] == -1
    assert bundle['r64']['centered_float64'][0, 0] == -10
    assert bool(bundle['r64']['allowed'][0, 0])


@pytest.mark.parametrize('field', ['centered_float64', 'scaled_float32', 'weights_float64',
                                   'weights_float32', 'allowed', 'scale'])
def test_corrupted_original_reference_is_rejected_before_training(field):
    costs, mask, episodes, reference = data()
    changed = dict(reference)
    if field == 'scale':
        changed[field] = math.nextafter(reference[field], math.inf)
    else:
        changed[field] = reference[field].copy()
        if field == 'allowed':
            changed[field][0, 0] = False
        elif field in ('centered_float64', 'scaled_float32'):
            # Signed zero is numerically equal but fails the required byte pin.
            changed[field][0, 2] = -0.0
        else:
            changed[field][0] *= 2
    with pytest.raises(ValueError, match='original R16'):
        precision.build_precision_targets(costs, costs, mask, episodes, reference_continuation=changed)


def test_training_bundle_rejects_separately_normalized_r64_even_when_cast_matches():
    costs, mask, episodes, reference = data()
    bundle = precision.build_precision_targets(costs, costs * 10, mask, episodes, reference_continuation=reference)
    changed = {**bundle, 'r64': dict(bundle['r64'])}
    changed['r64']['scale'] *= 10
    changed['r64']['scaled_float32'] = (changed['r64']['centered_float64'] / changed['r64']['scale']).astype(np.float32)
    with pytest.raises(ValueError, match='shared R16 scale'):
        precision.train_pair(np.zeros((2, precision.INPUT_DIM), dtype=np.float32), changed, 20101,
                             check=lambda: None, emit=lambda *_: None, checkpoint=lambda *_: None,
                             recipe=precision.Recipe(epochs=1, batch_size=2))


def test_identical_targets_produce_matched_fresh_training_and_named_evidence():
    costs = np.array([[0, 1, 2, np.inf], [2, 0, 1, np.inf], [1, 2, 0, np.inf]], dtype=np.float64)
    mask, episodes = np.isfinite(costs), ['a', 'a', 'b']
    reference = regression.build_targets(costs, mask, episodes, kind='continuation')
    bundle = precision.build_precision_targets(costs, costs, mask, episodes, reference_continuation=reference)
    features = np.linspace(-.03, .05, 3 * precision.INPUT_DIM, dtype=np.float32).reshape(3, -1)
    events, snapshots = [], {}

    def checkpoint(arm, seed, phase, exported):
        snapshots[arm, phase] = exported
        raw = b''.join(value.tobytes() for value in exported.values() if isinstance(value, np.ndarray))
        events.append(('publication', {'arm': arm, 'phase': phase}))
        return {'path': f'{arm}-{seed}-{phase}.npz', 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

    result = precision.train_pair(features, bundle, 20101, check=lambda: None,
                                  emit=lambda event, payload: events.append((event, payload)),
                                  checkpoint=checkpoint, recipe=precision.Recipe(epochs=1, batch_size=2))
    assert result['version'] == 'otto-teacher-precision-v1'
    assert result['progress']['completed_updates'] == 4
    assert result['progress']['completed_fits'] == 2
    assert result['progress']['pending'] == {}
    assert set(result['initial_pair']['initial_checkpoints']) == {'r16', 'r64'}
    assert {payload['arm'] for _, payload in events if 'arm' in payload} == {'r16', 'r64'}
    order = np.random.default_rng(20101 + 20000).permutation(3).astype(np.int64)
    for arm in precision.ARMS:
        recorded = [payload for event, payload in events if event == 'order' and payload['arm'] == arm]
        assert len(recorded) == 1
        assert recorded[0]['order'].tobytes() == order.tobytes()
        updates = [payload for event, payload in events if event == 'return'
                   and payload['channel'] == 'optimizer_update' and payload['arm'] == arm]
        assert [payload['rows'] for payload in updates] == [2, 1]
    for phase in ('initial', 'final'):
        for key, left in snapshots['r16', phase].items():
            if isinstance(left, np.ndarray):
                assert left.tobytes() == snapshots['r64', phase][key].tobytes()
    assert any(value.tobytes() != snapshots['r16', 'initial'][key].tobytes()
               for key, value in snapshots['r16', 'final'].items() if isinstance(value, np.ndarray))
    first_update = next(i for i, (event, payload) in enumerate(events)
                        if event == 'attempt' and payload['channel'] == 'optimizer_update')
    assert all(i < first_update for i, (event, payload) in enumerate(events)
               if event == 'publication' and payload['phase'] == 'initial')
    assert all(set(fit['optimizer_steps'].values()) == {2} and fit['parity']['passed'] for fit in result['fits'])
