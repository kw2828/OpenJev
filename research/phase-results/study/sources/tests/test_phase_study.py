"""Fabricated runner checks; the hardware dataset is never opened here."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('phase_study', Path(__file__).parents[1]/'scripts/phase_study.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_input_lag_order_and_zero_padding():
    np.testing.assert_array_equal(m.lag_design(np.array([2., 3., 4.]), 3),
                                  [[1, 2, 0, 0], [1, 3, 2, 0], [1, 4, 3, 2]])


def test_normalization_uses_supplied_fit_only():
    n = m.normalizers(np.array([0., 2.]), np.array([-1., 3.]))
    assert n == {'u_mean': 1., 'u_std': 1., 'y_mean': 1., 'y_std': 2.}
    with pytest.raises(ValueError):
        m.normalizers(np.ones(3), np.arange(3.))


@pytest.mark.parametrize('arm', m.ARMS)
def test_future_inputs_do_not_change_causal_prefix(arm):
    model = m.model_for(arm, 73, np.array([.7, -.2, .1, .03, 0., -.001, 0.]))
    rng = torch.Generator().manual_seed(34)
    u = torch.randn(2, 40, 1, generator=rng)*.1
    future = u.clone()
    future[:, 21:] = 4
    with torch.no_grad():
        a, _ = model(u)
        b, _ = model(future)
    torch.testing.assert_close(a[:, :21], b[:, :21], rtol=0, atol=0)


def test_cubic_ar2_delays_and_no_measured_target_argument():
    model = m.CubicAR2(np.array([.5, .2, 1., .1, 0., 0., 0.]))
    with torch.no_grad():
        pred, _ = model(torch.tensor([[[1.], [0.], [0.]]]))
    np.testing.assert_allclose(pred.numpy(), [[1., .6, .5]], rtol=1e-6)


def test_same_seed_gru_does_not_change_global_random_stream():
    torch.manual_seed(123)
    expected = torch.rand(2)
    torch.manual_seed(123)
    a, b = m.GRUModel(75), m.GRUModel(75)
    torch.testing.assert_close(torch.rand(2), expected, rtol=0, atol=0)
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('arm', m.ARMS)
def test_training_trace_and_adam_slots_for_all_models(arm):
    rng = np.random.default_rng(8)
    u, y = rng.normal(size=(2, 100))*.1
    windows = np.array([[0, 10], [20, 30]], dtype=np.int64)
    cfg = dict(m.config(), sequence_length=16, train_burn=4, fit_cap_seconds=30)
    model = m.model_for(arm, 7301, np.array([.7, -.2, .1, 0., 0., -.001, 0.]))
    initial = m.weights(model)
    trace, optimizer, seconds = m.train_one(model, u, y, windows, cfg=cfg, arm=arm)
    assert trace['loss'].shape == trace['gradnorm'].shape == (2,)
    assert np.isfinite(trace['loss']).all() and seconds > 0
    assert any(not np.array_equal(initial[k], m.weights(model)[k]) for k in initial)
    for name, parameter in model.named_parameters():
        assert optimizer[name+'.step'] == 2
        assert optimizer[name+'.exp_avg'].shape == tuple(parameter.shape)
        assert (optimizer[name+'.exp_avg_sq'] >= 0).all()


def scores_fixture():
    scores = {}
    for split in ('dev_a', 'dev_b'):
        row = {f'{arm}/{seed}': {'rmse_mv': 10., 'mae_mv': 7., 'scored_rows': 7680}
               for arm in m.ARMS for seed in m.SEEDS}
        row.update({r: {'rmse_mv': 10., 'mae_mv': 7., 'scored_rows': 7680} for r in m.REFERENCES})
        for seed in m.SEEDS:
            row[f'energy_phase/{seed}']['rmse_mv'] = 8.
        scores[split] = row
    return scores


def test_rule_all21_conditions_then_classical_control_blocks():
    scores = scores_fixture()
    r = m.evaluate_rule(scores, .1)
    assert r['passed'] == r['total'] == 21
    assert r['outcome'] == 'ADVANCE_PHASE_MECHANISM'
    scores['dev_b']['fir512']['rmse_mv'] = 1.
    r = m.evaluate_rule(scores, .1)
    assert r['passed'] == 20 and r['outcome'] == 'DO_NOT_ADVANCE_PHASE_MECHANISM'


def test_one_paired_seed_cannot_hide_behind_family_mean():
    scores = scores_fixture()
    scores['dev_a']['energy_phase/7301']['rmse_mv'] = 10.1
    scores['dev_a']['energy_phase/7302']['rmse_mv'] = 1.
    result = m.evaluate_rule(scores, .1)
    assert result['outcome'] == 'DO_NOT_ADVANCE_PHASE_MECHANISM'
    assert sum(not c['passed'] for c in result['conditions']) == 2


def test_metric_excludes_only_common_burn_and_uses_mv():
    pred, target = np.zeros(8192), np.zeros(8192)
    pred[:512], pred[512:] = 1000, 2.
    assert m.metrics(pred, target, .5) == {'rmse_mv': 1000., 'mae_mv': 1000., 'scored_rows': 7680}
    pred[0] = np.nan
    with pytest.raises(ValueError, match='finite full rollout'):
        m.metrics(pred, target, .5)


def test_failure_keeps_original_failed_model_and_partial_trace(tmp_path):
    model = m.CubicAR2(np.array([2., 0., 1., 0., 0., 1., 0.]))
    cfg = dict(m.config(), sequence_length=20, train_burn=4, fit_cap_seconds=30)
    with pytest.raises(ValueError, match='nonfinite simulation loss'):
        m.train_one(model, np.ones(30), np.zeros(30), np.array([[0]], dtype=np.int64),
                    cfg=cfg, arm='cubic_ar2', failure_folder=tmp_path)
    assert (tmp_path/'failed.npz').is_file()
    assert (tmp_path/'training-partial.npz').is_file()
    assert (tmp_path/'failure.json').is_file()
