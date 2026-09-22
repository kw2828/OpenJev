"""Tiny fabricated learning mechanics; no scientific data or simulation.

Explicit two/one-epoch recipes exercise the production loop and callbacks. One
fixture substitutes a differentiable constant loss to isolate row weighting.
No validation, continuation sampling or autonomous evaluation is performed.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from openjev.research import otto_cost_regression as regression
from openjev.research import otto_teacher_learning as learning


def fixture():
    features = np.linspace(-0.03, 0.05, 3 * learning.INPUT_DIM, dtype=np.float32).reshape(3, -1)
    costs = np.array([[0, 1, 2, np.inf], [2, 0, 1, np.inf], [1, 2, 0, np.inf]], dtype=np.float64)
    # A constant range makes the two globally scaled targets identical, so the
    # final paired weights must match too, despite separate optimizer instances.
    targets = regression.targets(costs, costs, np.isfinite(costs), ['a', 'a', 'b'])
    return features, targets


def recorder():
    events, snapshots, checks = [], {}, []

    def check():
        checks.append(True)

    def emit(event, payload):
        events.append((event, payload))

    def checkpoint(arm, seed, phase, exported):
        snapshots[arm, phase] = exported
        arrays = [value for value in exported.values() if isinstance(value, np.ndarray)]
        digest = hashlib.sha256(b''.join(array.tobytes() for array in arrays)).hexdigest()
        desc = {'path': f'{phase}-{arm}-{seed}.npz', 'sha256': digest,
                'bytes': sum(array.nbytes for array in arrays)}
        events.append(('published', {'arm': arm, 'phase': phase, **desc}))
        return desc

    return events, snapshots, checks, check, emit, checkpoint


def test_paired_initialization_orders_partial_batches_final_exports_and_parity():
    features, targets = fixture()
    before = features.copy()
    events, saved, checks, check, emit, checkpoint = recorder()
    result = learning.train_pair(features, targets, 10101, check=check, emit=emit,
                                 checkpoint=checkpoint, recipe=learning.Recipe(epochs=2, batch_size=2))
    np.testing.assert_array_equal(features, before)
    assert result['initial_pair']['identical_weights'] is True
    assert result['progress']['completed_updates'] == 8
    assert result['progress']['completed_fits'] == 2
    assert result['progress']['pending'] == {}
    counts = result['progress']['calls']
    expected = {'model_initialization': 2, 'checkpoint_export': 4, 'checkpoint_publication': 4,
                'optimizer_initialization': 2, 'optimizer_update': 8, 'inference_setup': 2,
                'parity_numpy': 2, 'parity_torch': 2}
    assert counts == {name: {'attempted': n, 'returned': n} for name, n in expected.items()}
    assert len(checks) == 2 * sum(expected.values())
    for phase in ('initial', 'final'):
        for key, value in saved['analytic', phase].items():
            if isinstance(value, np.ndarray):
                other = saved['continuation', phase][key]
                assert value.tobytes() == other.tobytes()
                with pytest.raises(ValueError):
                    value.setflags(write=True)
    assert any(not np.array_equal(saved['analytic', 'initial'][key], value)
               for key, value in saved['analytic', 'final'].items() if isinstance(value, np.ndarray))
    rng = np.random.default_rng(10101 + 20000)
    orders = [rng.permutation(3).astype(np.int64) for _ in range(2)]
    for arm in learning.ARMS:
        recorded = [p for event, p in events if event == 'order' and p['arm'] == arm]
        for actual, expected_order in zip(recorded, orders, strict=True):
            np.testing.assert_array_equal(actual['order'], expected_order)
            assert actual['sha256'] == hashlib.sha256(expected_order.tobytes()).hexdigest()
        updates = [p for event, p in events if event == 'return'
                   and p['channel'] == 'optimizer_update' and p['arm'] == arm]
        assert [p['rows'] for p in updates] == [2, 1, 2, 1]
        final_saved = next(i for i, (e, p) in enumerate(events)
                           if e == 'published' and p['arm'] == arm and p['phase'] == 'final')
        first_parity = next(i for i, (e, p) in enumerate(events)
                            if e == 'attempt' and p['arm'] == arm and p['channel'] == 'parity_numpy')
        assert final_saved < first_parity
    first_update = next(i for i, (e, p) in enumerate(events) if e == 'attempt' and p['channel'] == 'optimizer_update')
    assert all(i < first_update for i, (e, p) in enumerate(events) if e == 'published' and p['phase'] == 'initial')
    for fit in result['fits']:
        assert set(fit['optimizer_steps'].values()) == {4}
        assert fit['parity']['passed'] is True
        assert fit['parity']['rows'] == 3
        assert fit['parity']['absolute_tolerance'] == fit['parity']['relative_tolerance'] == 2e-5
        np.testing.assert_allclose(fit['parity']['numpy'], fit['parity']['torch'], atol=2e-5, rtol=2e-5)


def test_weighted_row_mean_uses_actual_partial_batch_denominator(monkeypatch):
    import torch

    features, targets = fixture()
    features[:, 0] = [1, 2, 3]
    events, _saved, _checks, check, emit, checkpoint = recorder()

    def constant_losses(model, x, _targets, _mask):
        # Give every parameter a present zero gradient while isolating weighting
        # from the already-qualified eight-view numerical loss.
        return x[:, 0] + sum((p.sum() * 0 for p in model.parameters()), torch.zeros(()))

    monkeypatch.setattr(regression, 'training_losses', constant_losses)
    result = learning.train_pair(features, targets, 10102, check=check, emit=emit,
                                 checkpoint=checkpoint, recipe=learning.Recipe(epochs=1, batch_size=2))
    weights = np.array([0.75, 0.75, 1.5])
    for arm in learning.ARMS:
        order = next(p['order'] for e, p in events if e == 'order' and p['arm'] == arm)
        updates = [p for e, p in events if e == 'return' and p['channel'] == 'optimizer_update' and p['arm'] == arm]
        for i, record in enumerate(updates):
            indices = order[2 * i:2 * i + 2]
            expected = sum((index + 1) * weights[index] for index in indices) / len(indices)
            assert record['result']['loss'] == expected
        fit = next(fit for fit in result['fits'] if fit['arm'] == arm)
        assert fit['epochs'][0]['weighted_mse'] == 2.25
        assert set(fit['optimizer_steps'].values()) == {2}


def test_failed_initial_publication_prevents_optimizer_and_parity():
    features, targets = fixture()
    events, _saved, _checks, check, emit, checkpoint = recorder()

    def fail_second(arm, seed, phase, exported):
        if arm == 'continuation':
            raise OSError('fabricated durable checkpoint failure')
        return checkpoint(arm, seed, phase, exported)

    with pytest.raises(learning.LearningFailure) as caught:
        learning.train_pair(features, targets, 10101, check=check, emit=emit,
                            checkpoint=fail_second, recipe=learning.Recipe(epochs=1, batch_size=2))
    progress = caught.value.progress
    assert isinstance(caught.value.original, OSError)
    assert progress['calls']['checkpoint_publication'] == {'attempted': 2, 'returned': 1}
    assert len(progress['pending']) == 1
    for channel in ('optimizer_initialization', 'optimizer_update', 'parity_numpy', 'parity_torch'):
        assert progress['calls'][channel] == {'attempted': 0, 'returned': 0}
    assert events[-1][0] == 'failure'


def test_failed_return_journal_keeps_prior_acknowledgments_and_pending_attempt_without_retry():
    features, targets = fixture()
    events, _saved, _checks, check, emit, checkpoint = recorder()
    attempts = 0

    def failing_emit(event, payload):
        nonlocal attempts
        emit(event, payload)
        if event == 'return' and payload['channel'] == 'optimizer_update':
            attempts += 1
            if attempts == 2:
                raise RuntimeError('fabricated second-update journal failure')

    with pytest.raises(learning.LearningFailure) as caught:
        learning.train_pair(features, targets, 10103, check=check, emit=failing_emit,
                            checkpoint=checkpoint, recipe=learning.Recipe(epochs=1, batch_size=2))
    progress = caught.value.progress
    assert attempts == 2
    assert progress['completed_updates'] == 1 and progress['completed_fits'] == 0
    assert progress['calls']['optimizer_update'] == {'attempted': 2, 'returned': 1}
    pending = list(progress['pending'].values())
    assert len(pending) == 1 and pending[0]['channel'] == 'optimizer_update' and pending[0]['batch'] == 1
    assert progress['calls']['parity_numpy'] == {'attempted': 0, 'returned': 0}
    assert events[-1][0] == 'failure'
