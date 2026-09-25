# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated FIT axes, physical requests and score arithmetic only."""
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from openjev_fsm_author import benchmark  # noqa: E402


def records():
    result = []
    for amplitude_index, amplitude in enumerate(('100mV', '200mV')):
        for realization in range(3):
            for period in range(2):
                value = (amplitude_index*10000+realization*1000+period*100
                         +10*np.arange(7)[:, None]+np.arange(3)[None, :]).astype(np.float64)
                result.append(SimpleNamespace(u=value, y=-2*value+3, amplitude=amplitude,
                    realization=realization, period=period, partition='fit', fs_hz=6400.,
                    record_id=f'{amplitude}-realization-{realization}-period-{period}'))
    return result


def test_exact_fit_axes_and_canonical_order_independent_of_caller_order():
    source = records()
    u, y = benchmark.assemble_fit(source[::-1])
    assert u.shape == y.shape == (7, 3, 6, 2)
    assert u.dtype == y.dtype == np.float64
    for a in range(2):
        for r in range(3):
            for p in range(2):
                for t in range(7):
                    for c in range(3):
                        expected = a*10000+r*1000+p*100+t*10+c
                        assert u[t, c, a*3+r, p] == expected
                        assert y[t, c, a*3+r, p] == -2*expected+3
    expected_ids = tuple(f'{a}-realization-{r}-period-{p}' for a in ('100mV', '200mV')
                         for r in range(3) for p in range(2))
    assert benchmark.FIT_IDS == expected_ids
    for record in source:
        assert not np.shares_memory(u, record.u)
        assert not np.shares_memory(y, record.y)
    before = source[0].u.copy()
    u.fill(-999.)
    np.testing.assert_array_equal(source[0].u, before)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'dev', 'partition', 'amplitude',
                                     'realization', 'period', 'frequency', 'length', 'dtype',
                                     'channel', 'nan', 'constant'])
def test_fit_roster_and_data_boundary_rejection(mutation):
    source = records()
    first = source[0]
    if mutation == 'missing':
        source.pop()
    elif mutation == 'duplicate':
        source[1] = first
    elif mutation == 'dev':
        first.record_id, first.realization, first.partition = '100mV-realization-3-period-0', 3, 'dev'
    elif mutation == 'partition':
        first.partition = 'dev'
    elif mutation == 'amplitude':
        first.amplitude = '300mV'
    elif mutation == 'realization':
        first.realization = 3
    elif mutation == 'period':
        first.period = 1
    elif mutation == 'frequency':
        first.fs_hz = 1.
    elif mutation == 'length':
        first.u = first.u[:-1]
    elif mutation == 'dtype':
        first.y = first.y.astype(np.float32)
    elif mutation == 'channel':
        first.y = first.y[:, :2]
    elif mutation == 'nan':
        first.y[0, 0] = np.nan
    else:
        for r in source:
            r.u = np.ones_like(r.u)
    with pytest.raises(ValueError):
        benchmark.assemble_fit(source)


def model_arrays():
    return {'A': np.diag([.3, .5, .8]), 'B_u': np.diag([.2, .3, .4]),
            'C_y': np.array([[1., .2, 0.], [0., 2., .3], [.1, 0., .7]]),
            'D_yu': np.diag([.4, .5, .6]), 'u_mean': np.array([10., -20., 30.]),
            'u_std': np.array([2., 3., 5.]), 'y_mean': np.array([-2., 4., 8.]),
            'y_std': np.array([.25, 2., 4.]), 'ts': np.array(1/6400, dtype=np.float64)}


def physical_fixture(context=6, horizon=11, batch=2):
    arrays = model_arrays()
    rng = np.random.default_rng(314)
    inputs = rng.normal(size=(batch, context-1+horizon, 3))
    state = rng.normal(size=(batch, 3))
    outputs = np.empty_like(inputs)
    for b in range(batch):
        for t in range(inputs.shape[1]):
            outputs[b, t] = arrays['C_y']@state[b]+arrays['D_yu']@inputs[b, t]
            state[b] = arrays['A']@state[b]+arrays['B_u']@inputs[b, t]
    physical_y = outputs*arrays['y_std']+arrays['y_mean']
    physical_u = inputs*arrays['u_std']+arrays['u_mean']
    yc = np.concatenate((np.full((batch, 1, 3), 123.), physical_y[:, :context-1]), axis=1)
    return arrays, yc, physical_u[:, :context-1], physical_u[:, context-1:], physical_y[:, context-1:], state


@pytest.mark.parametrize('context,horizon', [(2, 3), (6, 11), (100, 128)])
def test_physical_request_nonzero_means_scales_and_direct_term(context, horizon):
    arrays, yc, uc, fu, expected, state = physical_fixture(context, horizon)
    original = [v.copy() for v in (*arrays.values(), yc, uc, fu)]
    actual, final, diagnostic = benchmark.physical_request(arrays, yc, uc, fu)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(final, state, rtol=1e-11, atol=1e-12)
    assert diagnostic['rank'] == 3 and diagnostic['context_pairs'] == context-1
    assert actual.dtype == final.dtype == np.float64
    for before, after in zip(original, (*arrays.values(), yc, uc, fu), strict=True):
        np.testing.assert_array_equal(before, after)
        assert not np.shares_memory(actual, after) and not np.shares_memory(final, after)


def test_future_suffix_causality_and_unused_first_context_output():
    arrays, yc, uc, fu, _, _ = physical_fixture()
    original = benchmark.physical_request(arrays, yc, uc, fu)[0]
    altered_y, altered_u = yc.copy(), fu.copy()
    altered_y[:, 0] = -999.
    altered_u[:, 4:] += 20.
    revised = benchmark.physical_request(arrays, altered_y, uc, altered_u)[0]
    np.testing.assert_array_equal(original[:, :4], revised[:, :4])
    assert not np.array_equal(original[:, 4], revised[:, 4])


def test_request_window_inputs_and_native32start_geometry():
    index = np.arange(8192, dtype=np.float64)[:, None]
    record = SimpleNamespace(u=index+np.array([10000., 20000., 30000.]),
                             y=index*10+np.array([1., 2., 3.]))
    starts = np.arange(0, 7937, 256, dtype=np.int64)
    yc, uc, fu = benchmark.requests(record, starts)
    assert (yc.shape, uc.shape, fu.shape) == ((32, 100, 3), (32, 99, 3), (32, 128, 3))
    for i, start in enumerate(starts):
        np.testing.assert_array_equal(yc[i], record.y[start:start+100])
        np.testing.assert_array_equal(uc[i], record.u[start+1:start+100])
        np.testing.assert_array_equal(fu[i], record.u[start+100:start+228])
    one = np.array([17], dtype=np.int64)
    original = benchmark.requests(record, one)
    record.y[117:] = np.nan  # All unavailable future outputs, including the target.
    revised = benchmark.requests(record, one)
    for before, after in zip(original, revised, strict=True):
        np.testing.assert_array_equal(before, after)
    with pytest.raises(TypeError):
        benchmark.physical_request(model_arrays(), *original, target=np.zeros((1, 128, 3)))


@pytest.mark.parametrize('starts', [np.array([], dtype=np.int64), np.array([-1], dtype=np.int64),
                                  np.array([7965], dtype=np.int64), np.array([0.]),
                                  np.array([[0]], dtype=np.int64)])
def test_window_boundaries_fail(starts):
    record = SimpleNamespace(u=np.zeros((8192, 3)), y=np.zeros((8192, 3)))
    with pytest.raises(ValueError):
        benchmark.requests(record, starts)


def test_score_matches_parent_normalized_formula_and_native_channel_units():
    scale, mean = np.array([2., 4., 8.]), np.array([17., -23., 41.])
    normalized_errors = np.array([[[1., 2., 3.], [-1., -2., -3.]],
                                  [[3., 2., 1.], [-3., -2., -1.]]])
    target = np.broadcast_to(mean, normalized_errors.shape).copy()
    prediction = target+normalized_errors*scale
    result = benchmark.score(prediction, target, scale)
    parent_error = (prediction-mean)/scale-(target-mean)/scale
    mse = float(np.mean(parent_error**2))
    channel = np.sqrt(np.mean(parent_error**2, axis=(0, 1)))
    assert result == {'rmse': float(np.sqrt(mse)), 'mse': mse,
                      'per_channel_rmse': channel.tolist(),
                      'native_output_per_channel_rmse': (channel*scale).tolist(),
                      'requests': 2, 'horizon': 2}
    assert mse == 14/3
    np.testing.assert_allclose(channel, [np.sqrt(5.), 2., np.sqrt(5.)], atol=1e-14)


@pytest.mark.parametrize('mutation', ['prediction32', 'target32', 'scale32', 'nan', 'shape',
                                     'empty', 'zero_scale', 'negative_scale', 'scale_shape', 'overflow'])
def test_bad_scores_fail(mutation):
    prediction, target, scale = np.ones((2, 3, 3)), np.zeros((2, 3, 3)), np.ones(3)
    if mutation == 'prediction32':
        prediction = prediction.astype(np.float32)
    elif mutation == 'target32':
        target = target.astype(np.float32)
    elif mutation == 'scale32':
        scale = scale.astype(np.float32)
    elif mutation == 'nan':
        prediction[0, 0, 0] = np.nan
    elif mutation == 'shape':
        target = target[:, :-1]
    elif mutation == 'empty':
        prediction, target = prediction[:, :0], target[:, :0]
    elif mutation == 'zero_scale':
        scale[0] = 0.
    elif mutation == 'negative_scale':
        scale[0] = -1.
    elif mutation == 'scale_shape':
        scale = scale[:2]
    else:
        prediction.fill(1e308)
        target.fill(-1e308)
    with pytest.raises(ValueError):
        benchmark.score(prediction, target, scale)


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('mutation', ['float32', 'nan', 'batch', 'channels'])
def test_physical_boundary_checks_happen_before_promotion(index, mutation):
    arrays, yc, uc, fu, _, _ = physical_fixture()
    args = [yc.copy(), uc.copy(), fu.copy()]
    if mutation == 'float32':
        args[index] = args[index].astype(np.float32)
    elif mutation == 'nan':
        args[index].flat[0] = np.inf
    elif mutation == 'batch':
        args[index] = args[index][:1]
    else:
        args[index] = args[index][:, :, :2]
    with pytest.raises(ValueError):
        benchmark.physical_request(arrays, *args)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'A_list', 'A_nonsquare', 'dtype',
                                     'nan', 'zero_std', 'negative_ts'])
def test_model_export_contract(mutation):
    arrays = model_arrays()
    if mutation == 'missing':
        arrays.pop('ts')
    elif mutation == 'extra':
        arrays['cache'] = np.zeros(1)
    elif mutation == 'A_list':
        arrays['A'] = arrays['A'].tolist()
    elif mutation == 'A_nonsquare':
        arrays['A'] = arrays['A'][:, :2]
    elif mutation == 'dtype':
        arrays['B_u'] = arrays['B_u'].astype(np.float32)
    elif mutation == 'nan':
        arrays['C_y'][0, 0] = np.nan
    elif mutation == 'zero_std':
        arrays['u_std'][0] = 0.
    else:
        arrays['ts'] = np.array(-1.)
    with pytest.raises(ValueError):
        benchmark.validate_model(arrays)


def test_export_owns_all_arrays_and_retains_exact_scalar_time():
    arrays = model_arrays()
    model = SimpleNamespace(**{k: v for k, v in arrays.items() if k in ('A', 'B_u', 'C_y', 'D_yu', 'ts')})
    model.norm = SimpleNamespace(**{k: arrays[k] for k in ('u_mean', 'u_std', 'y_mean', 'y_std')})
    exported = benchmark.export_model(model)
    for key in arrays:
        np.testing.assert_array_equal(exported[key], arrays[key])
        assert not np.shares_memory(exported[key], arrays[key])
    assert exported['ts'].shape == () and exported['ts'] == 1/6400


def test_frozen_reader_native_shape_whitelist_in_this_numpy_runtime(tmp_path, monkeypatch):
    """Only four fabricated estimation members may be decoded, in production geometry."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
    from openjev.research import fsm_data

    shape = (8192, 3, 6, 2)
    value = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    mapping = {key: value + index for index, key in enumerate(fsm_data.ESTIMATION_VARIABLES)}
    mapping['u_300mV_test'] = np.array([{'forbidden': True}], dtype=object)
    path = tmp_path/'fabricated-native.npz'
    np.savez(path, **mapping)
    blob = path.read_bytes()
    original_load = np.load
    with original_load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    opened, decoded = [], []

    class SpyArchive(archive_type):
        def __getitem__(self, key):
            decoded.append(key)
            return super().__getitem__(key)

    def load(source, *, allow_pickle):
        assert allow_pickle is False
        opened.append(source)
        return SpyArchive(source, allow_pickle=False)

    monkeypatch.setattr(fsm_data.np, 'load', load)
    data = fsm_data.read_npz_estimation(path)
    assert len(opened) == 1
    assert decoded == list(fsm_data.ESTIMATION_VARIABLES)
    assert data.native_shape == shape and len(data.records) == 24
    assert len(data.partition('fit')) == len(data.partition('dev')) == 12
    assert data.source_sha256 == hashlib.sha256(blob).hexdigest()
    assert data.source_bytes == len(blob)
    for record in data.records:
        source = mapping[f'u_{record.amplitude}_train']
        np.testing.assert_array_equal(record.u, source[:, :, record.realization, record.period])
        assert record.u.dtype == np.float64 and not record.u.flags.writeable
    assert path.read_bytes() == blob
