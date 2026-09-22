"""Fabricated cached targets and short/long optimization mechanics only."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from openjev.research import otto_training_budget as learning


def fixture():
    centered = np.array([[-1, 0, 1, 0], [1, -1, 0, 0], [0, 1, -1, 0]], dtype=np.float64)
    weights = np.array([.75, .75, 1.5], dtype=np.float64)
    reference = {'kind': 'r64', 'scale': 2.0, 'centered_float64': centered,
                 'scaled_float32': (centered / 2).astype(np.float32),
                 'weights_float64': weights, 'weights_float32': weights.astype(np.float32),
                 'allowed': np.array([[True, True, True, False]] * 3)}
    expected = {'scale': 2.0, 'arrays': {k: hashlib.sha256(v.tobytes()).hexdigest()
                                      for k, v in reference.items() if isinstance(v, np.ndarray)}}
    features = np.linspace(-.03, .05, 3 * learning.INPUT_DIM, dtype=np.float32).reshape(3, -1)
    return features, reference, expected


def recorder():
    events, saved = [], {}

    def emit(event, payload):
        events.append((event, payload))

    def checkpoint(arm, seed, phase, exported):
        saved[arm, phase] = exported
        raw = b''.join(value.tobytes() for value in exported.values() if isinstance(value, np.ndarray))
        descriptor = {'path': f'{arm}-{seed}-{phase}.npz',
                      'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        events.append(('published', {'arm': arm, 'phase': phase, **descriptor}))
        return descriptor

    return events, saved, emit, checkpoint


def test_original_reference_bytes_are_preserved_and_corruption_is_rejected():
    _, reference, expected = fixture()
    original_bytes = {k: v.tobytes() for k, v in reference.items() if isinstance(v, np.ndarray)}
    targets = learning.build_training_targets(reference, expected=expected)
    assert targets['version'] == 'otto-training-budget-v1'
    assert targets['reference'] == expected
    for arm in learning.ARMS:
        assert targets[arm]['kind'] == arm and targets[arm]['scale'] == 2.0
        for key, raw in original_bytes.items():
            assert targets[arm][key].tobytes() == raw == reference[key].tobytes()
            assert not np.shares_memory(targets[arm][key], reference[key])
            with pytest.raises(ValueError):
                targets[arm][key].setflags(write=True)
    for key in original_bytes:
        changed = dict(reference)
        changed[key] = reference[key].copy()
        if key == 'allowed':
            changed[key][0, 0] = False
        elif key in ('centered_float64', 'scaled_float32'):
            changed[key][0, 3] = -0.0
        else:
            changed[key][0] *= 2
        with pytest.raises(ValueError, match='original R64'):
            learning.build_training_targets(changed, expected=expected)
    with pytest.raises(ValueError, match='original R64 scale'):
        learning.build_training_targets({**reference, 'scale': math.nextafter(2.0, math.inf)}, expected=expected)
    reference['centered_float64'][0, 0] = 10
    expected['arrays']['centered_float64'] = '0' * 64
    assert targets['short']['centered_float64'][0, 0] == -1
    assert targets['reference']['arrays']['centered_float64'] != '0' * 64


def test_long_prefix_matches_short_then_same_optimizer_continues():
    features, reference, expected = fixture()
    targets = learning.build_training_targets(reference, expected=expected)
    events, saved, emit, checkpoint = recorder()
    result = learning.train_pair(features, targets, 30101, check=lambda: None, emit=emit,
                                 checkpoint=checkpoint,
                                 recipe=learning.Recipe(short_epochs=1, long_epochs=2, batch_size=2))
    assert result['progress']['completed_updates'] == 6
    assert result['progress']['completed_fits'] == 2 and result['progress']['pending'] == {}
    assert result['progress']['calls']['optimizer_initialization'] == {'attempted': 2, 'returned': 2}
    for channel in ('checkpoint_export', 'checkpoint_publication'):
        assert result['progress']['calls'][channel] == {'attempted': 5, 'returned': 5}
    assert set(saved) == {('short', 'initial'), ('long', 'initial'), ('short', 'final'),
                          ('long', 'prefix'), ('long', 'final')}
    for key, value in saved['short', 'final'].items():
        if isinstance(value, np.ndarray):
            assert value.tobytes() == saved['long', 'prefix'][key].tobytes()
            assert saved['short', 'initial'][key].tobytes() == saved['long', 'initial'][key].tobytes()
    assert any(value.tobytes() != saved['long', 'prefix'][key].tobytes()
               for key, value in saved['long', 'final'].items() if isinstance(value, np.ndarray))
    prefix = result['matched_prefix']
    assert prefix['epoch'] == prefix['compared_epochs'] == 1
    assert prefix['updates'] == prefix['compared_updates'] == 2
    assert all(prefix[key] is True for key in ('identical_weights', 'identical_order_records',
                                              'identical_update_records', 'identical_epoch_records'))
    assert set(prefix['optimizer_steps'].values()) == {2}
    assert set(result['fits'][0]['optimizer_steps'].values()) == {2}
    assert set(result['fits'][1]['optimizer_steps'].values()) == {4}
    short, long = [], []
    for event, payload in events:
        if event == 'return' and payload['channel'] == 'optimizer_update' and payload['epoch'] == 1:
            (short if payload['arm'] == 'short' else long).append(
                {k: v for k, v in payload.items() if k not in ('arm', 'operation_id')})
    assert short == long and [r['rows'] for r in short] == [2, 1]
    rng = np.random.default_rng(30101 + 20000)
    orders = [rng.permutation(3).astype(np.int64) for _ in range(2)]
    for arm, n in (('short', 1), ('long', 2)):
        observed = [p for e, p in events if e == 'order' and p['arm'] == arm]
        assert len(observed) == n
        for row, expected_order in zip(observed, orders[:n], strict=True):
            assert row['order'].tobytes() == expected_order.tobytes()
    prefix_index = next(i for i, (e, _) in enumerate(events) if e == 'matched_prefix')
    continuing = next(i for i, (e, p) in enumerate(events)
                      if e == 'attempt' and p['channel'] == 'optimizer_update' and p['arm'] == 'long' and p['epoch'] == 2)
    assert prefix_index < continuing
    first_update = next(i for i, (e, p) in enumerate(events) if e == 'attempt' and p['channel'] == 'optimizer_update')
    assert all(i < first_update for i, (e, p) in enumerate(events) if e == 'published' and p['phase'] == 'initial')


def test_interrupted_prefix_publication_stops_without_long_continuation_or_resume():
    features, reference, expected = fixture()
    targets = learning.build_training_targets(reference, expected=expected)
    events, saved, emit, checkpoint = recorder()

    def interrupted(arm, seed, phase, exported):
        if phase == 'prefix':
            raise OSError('fabricated interrupted prefix publication')
        return checkpoint(arm, seed, phase, exported)

    with pytest.raises(learning.LearningFailure) as caught:
        learning.train_pair(features, targets, 30102, check=lambda: None, emit=emit, checkpoint=interrupted,
                            recipe=learning.Recipe(short_epochs=1, long_epochs=2, batch_size=2))
    assert isinstance(caught.value.original, OSError)
    progress = caught.value.progress
    assert progress['completed_updates'] == 4 and progress['completed_fits'] == 1
    assert progress['calls']['checkpoint_publication'] == {'attempted': 4, 'returned': 3}
    pending = list(progress['pending'].values())
    assert len(pending) == 1 and pending[0]['channel'] == 'checkpoint_publication'
    assert pending[0]['arm'] == 'long' and pending[0]['phase'] == 'prefix'
    assert ('long', 'final') not in saved and ('long', 'prefix') not in saved
    assert not any(e == 'matched_prefix' for e, _ in events)
    assert not any(e == 'attempt' and p['channel'] == 'optimizer_update'
                   and p['arm'] == 'long' and p['epoch'] == 2 for e, p in events)
    assert events[-1][0] == 'failure'
