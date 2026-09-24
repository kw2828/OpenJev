"""Fabricated causal checks only; never open the real sensor CSV."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('sensor_screen', Path(__file__).parents[1]/'scripts/sensor_screen.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic():
    rng = np.random.default_rng(7)
    t = np.arange(-120, 90, dtype=np.int64)
    x = rng.normal(size=(len(t), 7))
    y = 2 + .7*x[:, 1] + .5*x[:, 1]**2 + rng.normal(size=len(t))*.1
    return t, x, y


def test_future_label_poison_cannot_change_earlier_predictions():
    t, x, y = synthetic()
    p1, _, _ = m.run_arrays(t, x, y, start=0, end=80)
    poisoned = y.copy()
    poisoned[t >= 20] = 1e9
    p2, _, _ = m.run_arrays(t, x, poisoned, start=0, end=80)
    for method in m.METHODS:
        np.testing.assert_array_equal(p1[method][t < 44], p2[method][t < 44])
    assert p1['persistence'][t == 44] != p2['persistence'][t == 44]


def test_future_input_cannot_change_fit_or_prefix():
    t, x, y = synthetic()
    p1, s1, _ = m.run_arrays(t, x, y, start=0, end=80)
    changed = x.copy()
    changed[t >= 20] *= 100
    p2, s2, _ = m.run_arrays(t, changed, y, start=0, end=80)
    np.testing.assert_array_equal(s1['mean'], s2['mean'])
    np.testing.assert_array_equal(s1['scale'], s2['scale'])
    for method in m.METHODS:
        np.testing.assert_array_equal(p1[method][t < 20], p2[method][t < 20])


def test_strict_fit_boundary_assimilates_due_label_once():
    t, x, y = synthetic()
    _, states, _ = m.run_arrays(t, x, y, start=0, end=1)
    assert not states['fit_mask'][t == -24]
    assert states['assimilated_indices'].tolist() == [int(np.flatnonzero(t == -24)[0])]
    a = states['initial-linear7-A']
    phi = m.feature_map(x[t == -24], states['mean'], states['scale'], 'linear7')[0]
    np.testing.assert_allclose(states['final-rls-linear7-lambda1-A'], a+np.outer(phi, phi))
    np.testing.assert_allclose(states['final-rls-linear7-lambda.995-A'], .995*a+np.outer(phi, phi))


def test_clock_gap_still_decays_and_does_not_collapse_delay():
    t, x, y = synthetic()
    keep = ~np.isin(t, [-24, -23, -22, 0, 1])
    _, states, _ = m.run_arrays(t[keep], x[keep], y[keep], start=0, end=3)
    assert states['assimilated_indices'].size == 0
    np.testing.assert_allclose(states['final-rls-linear7-lambda.995-A'],
                               states['initial-linear7-A']*.995**3)
    assert states['final-rls-linear7-lambda.995-clock'] == 3
    assert np.isnan(states['final-rls-linear7-lambda.995-queue'][:2]).all()


def test_persistence_reveals_valid_target_even_if_input_missing():
    t, x, y = synthetic()
    x[t == -24, 2] = np.nan
    y[t == -24] = 1234.
    p, states, _ = m.run_arrays(t, x, y, start=0, end=1)
    assert p['persistence'][t == 0] == 1234.
    assert states['assimilated_indices'].size == 0
    assert states['revealed_indices'].size == 1


def test_unscored_current_missing_target_does_not_suppress_prediction():
    t, x, y = synthetic()
    y[t == 0] = np.nan
    p, _, _ = m.run_arrays(t, x, y, start=0, end=1)
    assert all(np.isfinite(p[name][t == 0]).all() for name in m.METHODS)


def test_each_adaptive_owns_normalization_and_queue():
    mean, scale = np.zeros(7), np.ones(7)
    queue = np.zeros((24, 7))
    a, b = np.eye(8), np.zeros(8)
    left = m.Adaptive('linear7', 1, a, b, mean, scale, queue, 0)
    right = m.Adaptive('linear7', 1, a, b, mean, scale, queue, 0)
    left.queue[:] = 99
    left.mean[:] = 17
    assert not right.queue.any()
    assert not right.mean.any()
    assert not queue.any()
    assert left.logical_bytes() == 2049


def test_feed_cannot_request_future_or_repeat_steps():
    t, x, y = synthetic()
    feed = m.DelayedFeed(t, x, y, 0)
    with pytest.raises(ValueError, match='future hour'):
        feed.step(2)
    feed.step(0)
    with pytest.raises(ValueError, match='future hour'):
        feed.step(0)


def test_quadratic_basis_and_s2_index():
    x = np.arange(7, dtype=np.float64)[None]
    np.testing.assert_array_equal(m.feature_map(x, np.zeros(7), np.ones(7), 's2quadratic'), [[1, 1, 1]])
    phi = m.feature_map(x, np.zeros(7), np.ones(7), 'quadratic7')
    assert phi.shape == (1, 36)
    assert phi[0, -1] == 36
    assert phi[0, 8:15].tolist() == [0]*7


def test_missing_input_masks_every_method():
    t, x, y = synthetic()
    x[t == 2, 4] = np.nan
    p, _, _ = m.run_arrays(t, x, y, start=0, end=5)
    assert all(np.isnan(p[name][t == 2]).all() for name in m.METHODS)


def test_gate_requires_same_rls_in_both_months():
    t = np.concatenate((np.arange(m.START, m.START+300), np.arange(m.JUNE, m.JUNE+300))).astype(np.int64)
    x, y = np.ones((600, 7)), np.zeros(600)
    p = {name: np.full(600, 1.) for name in m.METHODS}
    p[m.RLS[0]][:300] = .8
    p[m.RLS[1]][300:] = .8
    states = {'fit_mask': np.ones(512, dtype=bool), 'fit_target_std': 1.}
    result = m.summarize(t, x, y, p, states)
    assert result['passed'] == 5
    assert result['outcome'] == 'REJECT_BENZENE_MEMORY_BENCHMARK'
    p[m.RLS[0]][300:] = .89
    assert m.summarize(t, x, y, p, states)['outcome'] == 'QUALIFIES_LEARNED_MEMORY_SCREEN'


def test_static_near_solution_rejects_even_with_rls_improvement():
    t = np.concatenate((np.arange(m.START, m.START+300), np.arange(m.JUNE, m.JUNE+300))).astype(np.int64)
    x, y = np.ones((600, 7)), np.zeros(600)
    p = {name: np.full(600, .009) for name in m.METHODS}
    p[m.RLS[0]][:] = 0.
    result = m.summarize(t, x, y, p, {'fit_mask': np.ones(512, dtype=bool), 'fit_target_std': 1.})
    assert result['passed'] == 4
    assert result['outcome'] == 'REJECT_BENZENE_MEMORY_BENCHMARK'


def test_feed_cannot_reinitialize_using_newly_released_labels():
    t, x, y = synthetic()
    feed = m.DelayedFeed(t, x, y, 0)
    feed.initialization()
    with pytest.raises(ValueError, match='only once'):
        feed.initialization()
    feed.step(0)
    with pytest.raises(ValueError, match='only once'):
        feed.initialization()


def test_qualification_receipt_rejects_stale_sources_and_wrong_command(tmp_path, monkeypatch):
    import json
    for source in m.SOURCES:
        path = tmp_path/source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fabricated')
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    log = tmp_path/'qualification.log'
    log.write_text('fabricated passing qualification')
    receipt = {'state': 'EXITED', 'returncode': 0, 'argv': m.QUALIFICATION_ARGV,
               'thread_env': {k: '1' for k in m.THREADS},
               'log_path': log.name, 'log': m.pin(log.read_bytes()),
               'sources_before': {s: m.pin((tmp_path/s).read_bytes()) for s in m.SOURCES},
               'sources_after': {s: m.pin((tmp_path/s).read_bytes()) for s in m.SOURCES}}
    path = tmp_path/'qualification.json'
    path.write_text(json.dumps(receipt))
    m.authenticate_qualification(path)
    receipt['argv'] = ['true']
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='exact qualification'):
        m.authenticate_qualification(path)
    receipt['argv'] = m.QUALIFICATION_ARGV
    path.write_text(json.dumps(receipt))
    (tmp_path/m.SOURCES[0]).write_text('changed source')
    with pytest.raises(ValueError, match='unchanged registered sources'):
        m.authenticate_qualification(path)
