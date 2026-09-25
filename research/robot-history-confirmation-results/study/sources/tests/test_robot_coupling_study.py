"""Fabricated runner checks; never open an industrial-robot measurement file."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_coupling_study as study


def quadratic_scalar(row):
    return [1.] + list(row) + [row[i] * row[j] for i in range(24) for j in range(i, 24)]


def gru_scalar(x, h, weight_ih, weight_hh, bias_ih, bias_hh):
    """PyTorch reset-after-affine GRU equations via scalar arithmetic."""
    hidden = len(h)
    input_terms = [sum(float(w) * float(v) for w, v in zip(row, x, strict=True)) + float(b)
                   for row, b in zip(weight_ih, bias_ih, strict=True)]
    hidden_terms = [sum(float(w) * float(v) for w, v in zip(row, h, strict=True)) + float(b)
                    for row, b in zip(weight_hh, bias_hh, strict=True)]
    output = []
    for j in range(hidden):
        reset = 1 / (1 + math.exp(-(input_terms[j] + hidden_terms[j])))
        update = 1 / (1 + math.exp(-(input_terms[hidden + j] + hidden_terms[hidden + j])))
        candidate = math.tanh(input_terms[2 * hidden + j] + reset * hidden_terms[2 * hidden + j])
        output.append((1 - update) * candidate + update * float(h[j]))
    return np.asarray(output)


def test_quadratic_features_have_intercept_linear_and_lexicographic_upper_triangle():
    x = np.stack((np.arange(24, dtype=np.float64) / 7 - 1,
                  np.arange(24, dtype=np.float64)[::-1] / 9 - .3))
    before = x.copy()
    expected = np.asarray([quadratic_scalar(row) for row in x])
    actual = study.quadratic_features(x)
    assert actual.shape == (2, 325)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, before)


def test_quadratic_torch_values_and_cross_term_gradients():
    x = torch.linspace(-.7, .9, 24, dtype=torch.float64).reshape(1, 24).requires_grad_()
    features = study.quadratic_features(x)
    expected = np.asarray([quadratic_scalar(x.detach().numpy()[0])])
    np.testing.assert_array_equal(features.detach().numpy(), expected)
    weights = torch.linspace(.1, 1.3, 325, dtype=torch.float64)
    (features * weights).sum().backward()
    derivative = weights[1:25].numpy().copy()
    index = 25
    values = x.detach().numpy()[0]
    for i in range(24):
        for j in range(i, 24):
            derivative[i] += float(weights[index]) * values[j]
            derivative[j] += float(weights[index]) * values[i]
            index += 1
    np.testing.assert_allclose(x.grad.numpy()[0], derivative, rtol=1e-14, atol=1e-14)


def records(lengths=(256, 287)):
    result = []
    for index, length in enumerate(lengths):
        time = np.arange(length, dtype=np.float64)[:, None]
        joint = np.arange(6, dtype=np.float64)[None, :]
        result.append({'name': f'fabricated-{index}', 'q': 10000 * index + 10 * time + joint,
                       'u': -20000 * index - 20 * time - joint - .5})
    return result


def stable_coefficients():
    linear = np.zeros((6, 25))
    for j in range(6):
        linear[j, j], linear[j, 6 + j] = .4, .1
        linear[j, 12 + j], linear[j, 18 + j] = .07, -.03
        linear[j, -1] = .001 * (j + 1)
    quadratic = np.zeros((6, 325))
    quadratic[:, 0] = linear[:, -1]
    quadratic[:, 1:25] = linear[:, :24]
    return linear, quadratic


def wave(length, batch=2):
    time = np.arange(length)[None, :, None]
    joint = np.arange(6)[None, None, :]
    offset = np.arange(batch)[:, None, None]
    return np.sin(.13 * time + .23 * joint + .1 * offset).astype(np.float32)


def test_window_alignment_stays_inside_each_source_recording():
    data = records()
    choice = {'record': np.array([1, 0], dtype=np.int64), 'start': np.array([71, 64], dtype=np.int64)}
    batch = study.window_batch(data, choice, 32, 128)
    for b, (r, s) in enumerate(zip(choice['record'], choice['start'], strict=True)):
        np.testing.assert_array_equal(batch['q_context'][b], data[r]['q'][s:s + 32])
        np.testing.assert_array_equal(batch['u_context'][b], data[r]['u'][s:s + 32])
        np.testing.assert_array_equal(batch['future_u'][b], data[r]['u'][s + 31:s + 159])
        np.testing.assert_array_equal(batch['target'][b], data[r]['q'][s + 32:s + 160])
    with pytest.raises(ValueError, match='bounds'):
        study.window_batch(data, {'record': np.array([0]), 'start': np.array([100])}, 32, 128)
    with pytest.raises(ValueError, match='recording'):
        study.window_batch(data, {'record': np.array([2]), 'start': np.array([0])}, 32, 128)


def test_batch_orders_are_local_seeded_complete_and_reusable():
    cfg = dict(study.config(), updates=7, batch_size=5)
    lengths = [173, 219, 287]
    before = np.random.get_state()
    first = study.make_batches(lengths, 8101, cfg)
    second = study.make_batches(lengths, 8101, cfg)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    for key in ('record', 'start'):
        assert first[key].shape == (7, 5) and first[key].dtype == np.int64
        np.testing.assert_array_equal(first[key], second[key])
    for r, s in zip(first['record'].ravel(), first['start'].ravel(), strict=True):
        assert 0 <= r < 3 and 64 <= s <= lengths[r] - 32 - 64
    changed = study.make_batches(lengths, 8102, cfg)
    assert any(not np.array_equal(first[key], changed[key]) for key in first)
    with pytest.raises(ValueError, match='windows'):
        study.make_batches([159], 8101, cfg)


def test_dev_windows_are_complete_nonoverlapping_and_have_fixed_first_index():
    starts = study.dev_windows(3636, study.config())
    np.testing.assert_array_equal(starts, 64 + 160 * np.arange(22))
    assert starts[-1] + 32 + 128 <= 3636


def test_direct_history_has_no_duplicated_boundary_torque_or_target_input():
    batch = study.window_batch(records((256,)), {'record': np.array([0]), 'start': np.array([64])}, 32, 128)
    got = study.direct_features(batch)
    expected = np.concatenate((batch['q_context'][0, -16:].ravel(),
                               batch['u_context'][0, -17:-1].ravel(),
                               batch['future_u'][0].ravel(), [1.]))
    assert got.shape == (1, 961)
    np.testing.assert_array_equal(got[0], expected)
    poisoned = dict(batch, target=np.full_like(batch['target'], np.nan))
    np.testing.assert_array_equal(study.direct_features(poisoned), got)


def test_ridge_penalizes_intercept_and_all_multivariate_coefficients():
    phi = np.column_stack((np.ones(5), [-2., -1, 0, 1, 2], [2., -1, -2, -1, 2]))
    truth = np.array([[1., -.5], [.2, .7], [-.3, .9]])
    expected = np.array([5 / 6, 10 / 11, 14 / 15])[:, None] * truth
    np.testing.assert_allclose(study.solve_ridge(phi, phi @ truth), expected, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError):
        study.solve_ridge(phi, phi @ truth, penalty=0)


def test_normalization_uses_only_supplied_fit_rows_after_skip():
    data = records((80, 91))
    norm = study.normalizers(data, 64)
    for key in ('q', 'u'):
        explicit = np.asarray([row for recording in data for row in recording[key][64:]])
        np.testing.assert_array_equal(norm[key + '_mean'], explicit.mean(axis=0))
        np.testing.assert_array_equal(norm[key + '_std'], explicit.std(axis=0))
    poisoned = copy.deepcopy(data)
    for recording in poisoned:
        recording['q'][:64] = 1e50
        recording['u'][:64] = -1e50
    for key, value in study.normalizers(poisoned, 64).items():
        np.testing.assert_array_equal(value, norm[key])


@pytest.mark.parametrize('arm', study.ARMS)
def test_inference_never_reads_future_positions(arm):
    linear, quadratic = stable_coefficients()
    model = study.model_for(arm, 8101, linear, quadratic)
    batch = {'q_context': wave(32), 'u_context': wave(32) * .2,
             'future_u': wave(12) * -.3, 'target': np.full((2, 12, 6), np.nan)}
    with torch.no_grad():
        first = study.infer(model, batch)
        batch['target'] = object()
        second = study.infer(model, batch)
        del batch['target']
        third = study.infer(model, batch)
    assert torch.equal(first, second) and torch.equal(first, third)
    assert first.shape == (2, 12, 6)


def test_gru_prefix_and_rollout_match_independent_scalar_oracle_and_chunking():
    linear, _ = stable_coefficients()
    model = study.GRUResidual(8101, linear).double()
    with torch.no_grad():
        model.head.weight.copy_(torch.linspace(-.08, .12, 60).reshape(6, 10))
        model.head.bias.copy_(torch.linspace(-.03, .02, 6))
        model.gru.bias_hh.copy_(torch.linspace(-.2, .4, 30))
    q = torch.from_numpy(wave(7, 1)).double()
    u = q * .3
    future = torch.from_numpy(wave(11, 1)).double() * -.2
    fields = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    h = np.zeros(10)
    def update(x, hidden):
        return gru_scalar(x, hidden, fields['gru.weight_ih'], fields['gru.weight_hh'],
                          fields['gru.bias_ih'], fields['gru.bias_hh'])
    for t in range(1, 6):
        h = update(np.concatenate((q[0, t], q[0, t-1], u[0, t], u[0, t-1])), h)
    conditioned = model.condition(q, u)
    np.testing.assert_allclose(conditioned.hidden.detach().numpy()[0], h, rtol=1e-13, atol=1e-14)
    current, previous, old_u = q[0, -1].numpy(), q[0, -2].numpy(), u[0, -2].numpy()
    expected = []
    for torque in future[0].numpy():
        x = np.concatenate((current, previous, torque, old_u))
        h = update(x, h)
        prediction = np.append(x, 1.) @ fields['base_weight'].T + fields['head.weight'] @ h + fields['head.bias']
        expected.append(prediction)
        current, previous, old_u = prediction, current, torque
    whole, final = model(future, conditioned)
    left, mid = model(future[:, :4], conditioned)
    right, split_final = model(future[:, 4:], mid)
    np.testing.assert_allclose(whole.detach().numpy()[0], expected, rtol=1e-13, atol=1e-14)
    assert torch.equal(whole, torch.cat((left, right), dim=1))
    assert all(torch.equal(a, b) for a, b in zip(final, split_final, strict=True))
    changed = u.clone(); changed[:, -1] += 99
    assert all(torch.equal(a, b) for a, b in zip(conditioned, model.condition(q, changed), strict=True))


def test_quadratic_autoregression_uses_own_predictions_and_public_lags():
    coefficient = np.zeros((6, 325))
    # For each output, q0 squared plus current u0. Only first predicted q feeds later steps.
    coefficient[:, 1 + 12] = .5
    coefficient[:, 25] = .1
    model = study.QuadraticAR2(coefficient).double()
    q = torch.zeros(1, 3, 6, dtype=torch.float64); q[:, -1, 0] = 2
    u = torch.zeros_like(q)
    future = torch.ones(1, 3, 6, dtype=torch.float64)
    got, _ = model(future, model.condition(q, u))
    values = []
    current = 2.
    for _ in range(3):
        current = float(model.coefficient[0, 25].detach()) * current**2 + .5
        values.append([current] * 6)
    np.testing.assert_allclose(got.detach().numpy()[0], values, rtol=1e-14, atol=1e-14)


def test_metrics_weight_joint_scales_and_full_prediction_finiteness():
    target = np.zeros((2, 128, 6))
    prediction = np.ones_like(target)
    prediction[:, 64:] = 3
    scales = np.arange(1., 7.)
    first = study.metrics(prediction, target, scales, 64)
    full = study.metrics(prediction, target, scales, 128)
    assert first['standardized_rmse'] == 1 and first['scalars'] == 2 * 64 * 6
    assert full['standardized_rmse'] == math.sqrt(5)
    np.testing.assert_array_equal(first['per_joint_rmse_deg'], scales)
    np.testing.assert_allclose(full['per_joint_rmse_deg'], math.sqrt(5) * scales)
    prediction[:, -1] = np.nan
    with pytest.raises(ValueError):
        study.metrics(prediction, target, scales, 64)


def metric_value(value, horizon=128):
    return {'standardized_rmse': value, 'standardized_sse': value**2 * 6 * horizon,
            'scalars': 6 * horizon, 'physical_rmse_deg': value,
            'per_joint_rmse_deg': [value] * 6, 'windows': 1, 'horizon': horizon}


def complete_rows(cfg=None):
    cfg = cfg or study.config()
    rows = []
    for recording in cfg['partitions']['dev']:
        for arm in cfg['arms']:
            for rate in cfg['learning_rates']:
                for seed in cfg['seeds']:
                    for horizon in (64, 128):
                        value = 1. if arm == 'chain_memory' else 2.
                        rows.append({'recording': recording, 'arm': arm, 'learning_rate': rate, 'seed': seed,
                                     'horizon': horizon, 'status': 'PASS', 'error': None,
                                     'metrics': metric_value(value, horizon)})
        for arm in study.REFERENCES:
            for horizon in (64, 128):
                value = 1.1 if arm.startswith('direct_') else 2.
                rows.append({'recording': recording, 'arm': arm, 'learning_rate': None, 'seed': None,
                             'horizon': horizon, 'status': 'PASS', 'error': None,
                             'metrics': metric_value(value, horizon)})
    return rows


def test_complete_selection_and_exact_gate_roster():
    cfg, rows = study.config(), complete_rows()
    selected = study.select_recipes(rows, cfg)
    assert selected['selected_rates'] == dict.fromkeys(study.ARMS, study.RATES[0])
    assert selected['selected_direct'] == 'direct_ridge_1'
    result = study.evaluate_rule(rows, selected, cfg)
    assert result['total'] == result['passed'] == 55
    assert len({item['name'] for item in result['conditions']}) == 55
    assert result['status'] == 'ADVANCE_TO_SEPARATELY_AUTHORIZED_CONFIRMATION'


@pytest.mark.parametrize('corruption', ['failed', 'missing', 'duplicate', 'nan', 'infinite', 'wrong_joint_count'])
def test_recipe_requires_every_seed_and_recording_with_finite_metrics(corruption):
    cfg, rows = study.config(), complete_rows()
    for rate in study.RATES:
        row = next(r for r in rows if r['arm'] == 'gru_residual' and r['learning_rate'] == rate
                   and r['horizon'] == 128)
        if corruption == 'failed':
            row.update(status='FAILED', metrics=None)
        elif corruption == 'missing':
            rows.remove(row)
        elif corruption == 'duplicate':
            rows.append(copy.deepcopy(row))
        elif corruption in ('nan', 'infinite'):
            row['metrics']['standardized_rmse'] = float('nan' if corruption == 'nan' else 'inf')
        else:
            row['metrics']['per_joint_rmse_deg'] = [1.] * 5
    selected = study.select_recipes(rows, cfg)
    assert selected['selected_rates']['gru_residual'] is None
    result = study.evaluate_rule(rows, selected, cfg)
    assert result['status'] == 'DO_NOT_ADVANCE_COUPLING'
    assert result['conditions'][0]['passed'] is False


def test_selection_uses_pooled_full_horizon_not_short_horizon_or_best_seed():
    rows, cfg = complete_rows(), study.config()
    for row in rows:
        if row['arm'] == 'chain_memory' and row['learning_rate'] == study.RATES[0]:
            row['metrics'] = metric_value(.01 if row['horizon'] == 64 else 3., row['horizon'])
    selected = study.select_recipes(rows, cfg)
    assert selected['selected_rates']['chain_memory'] == study.RATES[1]
    # One incomplete otherwise excellent rate must not replace a complete rate.
    chosen = next(r for r in rows if r['arm'] == 'chain_memory' and r['learning_rate'] == study.RATES[1]
                  and r['horizon'] == 128)
    chosen.update(status='FAILED', metrics=None)
    assert study.select_recipes(rows, cfg)['selected_rates']['chain_memory'] == study.RATES[0]


def test_gate_rejects_missing_fixed_reference_even_when_duplicate_preserves_count():
    rows, cfg = complete_rows(), study.config()
    fixed = [r for r in rows if r['arm'] == 'linear_frozen' and r['horizon'] == 128]
    rows.remove(fixed[1]); rows.append(copy.deepcopy(fixed[0]))
    result = study.evaluate_rule(rows, study.select_recipes(rows, cfg), cfg)
    assert result['conditions'][0]['passed'] is False
    assert result['status'] == 'DO_NOT_ADVANCE_COUPLING'


def test_paired_seed_or_joint_failure_cannot_be_rescued_by_good_mean():
    rows, cfg = complete_rows(), study.config()
    for row in rows:
        if row['arm'] == 'chain_memory' and row['horizon'] == 128:
            row['metrics'] = metric_value(2.1 if row['seed'] == 8101 else .1)
    result = study.evaluate_rule(rows, study.select_recipes(rows, cfg), cfg)
    assert all(c['passed'] for c in result['conditions'] if '/mean_10pct/' in c['name'])
    assert not any(c['passed'] for c in result['conditions'] if '/seed8101_5pct/' in c['name'])
    assert result['status'] == 'DO_NOT_ADVANCE_COUPLING'
    rows = complete_rows()
    for row in rows:
        if row['arm'] == 'chain_memory':
            row['metrics']['per_joint_rmse_deg'][4] = 2.3
    result = study.evaluate_rule(rows, study.select_recipes(rows, cfg), cfg)
    assert not any(c['passed'] for c in result['conditions'] if '/joint4_' in c['name'])


def test_scoring_preserves_nonfinite_and_overflow_as_failed_rows():
    cfg = study.config()
    target = np.zeros((1, 128, 6))
    for value in (np.inf, 1e300):
        prediction = np.full_like(target, value)
        rows = study.scored_rows({'arm': 'fabricated'}, prediction, target, np.ones(6), cfg)
        assert len(rows) == 2 and all(r['status'] == 'FAILED' and r['metrics'] is None for r in rows)


def pin(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def test_raw_sha_and_closed_roster_checked_before_loader(tmp_path, monkeypatch):
    name = study.PARTITIONS['fit'][0]
    path = tmp_path / name
    path.write_bytes(b'fabricated opaque bytes, never a real MAT')
    plan = {'raw_recordings': {name: {'path': str(path), **pin(path)}}}
    calls = []
    def loader(blob, recording):
        calls.append((blob, recording))
        return SimpleNamespace(pins={'source_sha256': hashlib.sha256(blob).hexdigest(),
                                    'source_bytes': len(blob), 'preprocessing_sha256': study.PREPROCESSING_SHA256})
    monkeypatch.setattr(study, 'load_recording', loader)
    study.load_registered(plan, name)
    assert len(calls) == 1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='pin before decode'):
        study.load_registered(plan, name)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='closed partition'):
        study.load_registered(plan, study.PARTITIONS['confirm'][0])
    assert len(calls) == 1


def test_source_pin_change_rejected_before_any_raw_decode(tmp_path, monkeypatch):
    root = tmp_path / 'root'; root.mkdir()
    sources = {}
    for name in study.SOURCES:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'fabricated source')
        sources[name] = pin(path)
    metadata = tmp_path / 'metadata.json'; metadata.write_text('{}')
    plan = {'version': study.VERSION, 'config': study.config(), 'sources': sources,
            'raw_recordings': {name: {'path': str(tmp_path / name), 'sha256': '0' * 64, 'bytes': 7}
                               for part in ('fit', 'dev') for name in study.PARTITIONS[part]},
            'metadata_contract': {'path': str(metadata), **pin(metadata)}}
    registration = tmp_path / 'registration.json'; registration.write_text(json.dumps(plan))
    monkeypatch.setattr(study, 'ROOT', root)
    for name in study.THREADS:
        monkeypatch.setenv(name, '1')
    study.authenticate(registration)
    (root / study.SOURCES[0]).write_bytes(b'changed source')
    monkeypatch.setattr(study, 'load_recording', lambda *args: pytest.fail('must not decode'))
    with pytest.raises(ValueError, match='source pin changed'):
        study.run(registration, tmp_path / 'forbidden-output')
    assert not (tmp_path / 'forbidden-output').exists()


def test_training_reuses_exact_supplied_batches_and_saves_all_updates(tmp_path, monkeypatch):
    cfg = dict(study.config(), updates=2, batch_size=2, context=3, train_horizon=2, skip=0)
    data = [{'name': 'fabricated', 'q': wave(14, 1)[0].astype(np.float64),
             'u': .2 * wave(14, 1)[0].astype(np.float64)}]
    batches = {'record': np.zeros((2, 2), dtype=np.int64), 'start': np.array([[1, 4], [2, 6]], dtype=np.int64)}
    saved = {key: value.copy() for key, value in batches.items()}
    original = study.infer
    seen = []
    def checked(model, batch):
        seen.append({key: value.copy() for key, value in batch.items()})
        return original(model, batch)
    monkeypatch.setattr(study, 'infer', checked)
    linear, quadratic = stable_coefficients()
    for rate_index, rate in enumerate(study.RATES):
        receipt = study.train_one(study.model_for('gru_residual', 8101, linear, quadratic), data, batches,
                                 cfg=cfg, lr=rate, folder=tmp_path / str(rate_index))
        assert receipt['status'] == 'PASS' and receipt['completed_updates'] == 2
        assert len(json.loads((tmp_path / str(rate_index) / 'trace.json').read_text())) == 2
    assert len(seen) == 4
    for update in range(2):
        for key in seen[update]:
            np.testing.assert_array_equal(seen[update][key], seen[update + 2][key])
    for key, value in batches.items():
        np.testing.assert_array_equal(value, saved[key])


@pytest.mark.parametrize('failure', ['numerical', 'schema', 'deadline'])
def test_partial_failure_is_preserved_and_only_numerical_failure_is_recoverable(tmp_path, monkeypatch, failure):
    cfg = dict(study.config(), updates=2, batch_size=1, context=3, train_horizon=2, skip=0)
    data = [{'name': 'fake', 'q': wave(10, 1)[0].astype(np.float64), 'u': wave(10, 1)[0].astype(np.float64)}]
    batches = {'record': np.zeros((2, 1), dtype=np.int64), 'start': np.zeros((2, 1), dtype=np.int64)}
    linear, quadratic = stable_coefficients()
    model = study.model_for('gru_residual', 8101, linear, quadratic)
    if failure == 'numerical':
        monkeypatch.setattr(study, 'infer', lambda *_: torch.full((1, 2, 6), float('nan')))
    else:
        def fail(*_):
            if failure == 'schema':
                raise ValueError('fabricated schema defect')
            raise study.WholeStudyTimeout('fabricated whole-run deadline')
        monkeypatch.setattr(study, 'infer', fail)
    folder = tmp_path / 'fit'
    if failure == 'numerical':
        receipt = study.train_one(model, data, batches, cfg=cfg, lr=.001, folder=folder)
        assert receipt['status'] == 'FAILED'
    else:
        with pytest.raises(ValueError if failure == 'schema' else study.WholeStudyTimeout):
            study.train_one(model, data, batches, cfg=cfg, lr=.001, folder=folder)
        receipt = json.loads((folder / 'fit-receipt.json').read_text())
        assert receipt['status'] == 'FATAL'
    assert receipt['completed_updates'] == 0 and receipt['requested_updates'] == 2
    assert set(receipt['files']) == {'initial.npz', 'final.npz', 'optimizer.npz', 'trace.json'}
    for name, expected in receipt['files'].items():
        assert pin(folder / name) == expected


def test_fabricated_thirty_attempt_campaign_preserves_barrier_pairing_and_evidence(tmp_path, monkeypatch):
    """Real one-update fits/evaluation; fabricated inputs, admission and reference initialization.

    Large reference solves and wall-time probes have separate algebra/source
    tests and are stubbed here. No official source file is opened by this test.
    """
    cfg = dict(study.config(), context=17, train_horizon=2, updates=1, batch_size=1,
               timing_warmups=0, timing_repeats=1)
    output = tmp_path / 'campaign'
    plan = {'version': study.VERSION, 'config': cfg,
            'sources': {name: pin(study.ROOT / name) for name in study.SOURCES}}
    monkeypatch.setattr(study, 'config', lambda: copy.deepcopy(cfg))
    admissions = []
    def authenticate(path):
        admissions.append(str(path))
        return plan, 'a' * 64
    monkeypatch.setattr(study, 'authenticate', authenticate)
    loads, completed, paired = [], [], {}
    def load_registered(unused_plan, name):
        assert unused_plan is plan
        assert name in (*study.PARTITIONS['fit'], *study.PARTITIONS['dev'])
        if name in study.PARTITIONS['dev']:
            assert len(completed) == 30
            barrier = json.loads((output / 'checkpoint-barrier.json').read_text())
            assert barrier['fit_attempts'] == 30 and barrier['dev_decodes'] == 0
            assert len(barrier['final_checkpoints']) == 30
        loads.append(name)
        index = [*study.PARTITIONS['fit'], *study.PARTITIONS['dev']].index(name)
        t = np.arange(256)[:, None]; j = np.arange(6)[None]
        q = np.sin(.09 * t + .2 * j + .07 * index)
        u = np.cos(.11 * t - .13 * j + .05 * index)
        return SimpleNamespace(q=q, torque=u, raw_indices=25 * np.arange(256, dtype=np.int64))
    monkeypatch.setattr(study, 'load_registered', load_registered)
    linear, quadratic = stable_coefficients()
    reference_calls = []
    def references(data, local_cfg):
        assert len(data) == 7 and not any(n in loads for n in study.PARTITIONS['dev'])
        reference_calls.append(tuple(record['name'] for record in data))
        return {'linear_frozen': linear, 'quadratic_frozen': quadratic,
                'direct_ridge_1': np.zeros((961, 768)), 'direct_ridge_100': np.zeros((961, 768))}
    monkeypatch.setattr(study, 'fit_references', references)
    original_train = study.train_one
    def training(model, data, batches, **kwargs):
        name = Path(kwargs['folder']).name
        seed = next(seed for seed in study.SEEDS if f'-{seed}-' in name)
        encoding = {key: value.tobytes() for key, value in batches.items()}
        if seed in paired:
            assert encoding == paired[seed]
        else:
            paired[seed] = encoding
        assert not any(n in loads for n in study.PARTITIONS['dev'])
        result = original_train(model, data, batches, **kwargs)
        completed.append(name)
        return result
    monkeypatch.setattr(study, 'train_one', training)
    timing_calls = []
    def timing(arm, model, coefficients, batch, norm, config, check):
        assert len(completed) == 30 and len(loads) == 9
        check(); timing_calls.append(arm)
        return {'seconds': [.001], 'median_seconds': .001, 'p95_seconds': .001,
                'scope': 'fabricated timing stub, no measured latency'}
    monkeypatch.setattr(study, 'timed_request', timing)
    result = study.run(tmp_path / 'fabricated-registration.json', output)
    assert result['fits'] == 30 and result['rows'] == 144 and len(completed) == len(set(completed)) == 30
    assert loads == list(study.PARTITIONS['fit']) + list(study.PARTITIONS['dev'])
    assert len(reference_calls) == 1 and len(admissions) == 2 and set(paired) == set(study.SEEDS)
    fits = json.loads((output / 'fits.json').read_text())
    assert all(row['fit']['status'] == 'PASS' and row['fit']['completed_updates'] == 1 for row in fits)
    assert len(timing_calls) == 20
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['confirmation_decodes'] == receipt['official_test_decodes'] == 0
    assert receipt['fit_decodes'] == 7 and receipt['dev_decodes'] == 2
    manifest = json.loads((output / 'manifest.json').read_text())['files']
    assert set(manifest) == {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} - {'manifest.json', 'receipt.json'}
    for name, expected in manifest.items():
        assert pin(output / name) == expected
    assert len(list(output.glob('prediction-*.npz'))) == 72
    assert result['result']['total'] == 55 and result['result']['status'] == 'DO_NOT_ADVANCE_COUPLING'
