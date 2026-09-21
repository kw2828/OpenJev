"""Synthetic conditioning algebra only; no scientific caches or simulator."""
import copy
import dataclasses
import io

import numpy as np
import pytest

from openjev.research import otto_capacity_value as capacity
from openjev.research import otto_conditioned_value as m
from openjev.research import otto_return_value as old


def inputs():
    belief = np.zeros((6, 105, 105), np.float64)
    belief[0, 40, 61], belief[0, 30, 15] = .25, .75
    belief[1, :53, :53] = 1 / 2809
    belief[2, 20, 20] = .5
    belief[3, 20, 20] = 1e-12
    belief[4, 20, 20] = float(np.nextafter(np.float32(0), np.float32(1)))
    positions = np.asarray([[0, 52], [26, 26], [52, 0], [4, 17], [10, 20], [0, 0]], np.int64)
    return belief, positions


def hand_export(kind):
    result = {'version': m.VERSION, 'kind': kind, 'input_dim': 11028, 'spatial_gain': m.GAINS[kind],
              'c0': np.array(.25, np.float32)}
    for name, shape in m.SHAPES.items():
        result[name] = np.zeros(shape, np.float32)
    result['weight_0'][0, 0] = 2.
    result['weight_0'][0, -3:] = [.5, -.25, 1.]
    result['bias_0'][0] = .125
    result['weight_1'][0, 0] = -2.
    result['bias_1'][0] = -.5
    return result


@pytest.mark.parametrize('seed', [10101, 10102, 10103])
def test_exact_capacity_gain1_and_paired_inverse_initialization(seed):
    import torch

    before = torch.random.get_rng_state().clone()
    original = capacity.make_head('mlp8', seed, .37)
    raw, scaled = (m.make_head(kind, seed, .37) for kind in m.KINDS)
    assert torch.equal(torch.random.get_rng_state(), before)
    original_arrays = capacity.export_head(original)
    exports = {head.kind: m.export_head(head) for head in (raw, scaled)}
    for name in ('c0', *m.SHAPES):
        np.testing.assert_array_equal(exports['gain1'][name], original_arrays[name])
        if name != 'weight_0':
            np.testing.assert_array_equal(exports['gain53'][name], original_arrays[name])
    np.testing.assert_array_equal(exports['gain53']['weight_0'][:, :11025],
                                  original_arrays['weight_0'][:, :11025] / np.float32(53))
    np.testing.assert_array_equal(exports['gain53']['weight_0'][:, 11025:], original_arrays['weight_0'][:, 11025:])
    x = torch.from_numpy(m.value_features(*inputs(), 3., dtype='float32'))
    saved = x.clone()
    with torch.no_grad():
        a, b, c = original(x), raw(x), scaled(x)
    assert torch.equal(a, b)
    torch.testing.assert_close(b, c, atol=2e-7, rtol=2e-6)
    assert torch.equal(x, saved)
    for head in (raw, scaled):
        assert m.parameter_count(head.kind) == 88241 == sum(p.numel() for p in head.parameters())
        assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in head.parameters())
        assert not head.c0.requires_grad


@pytest.mark.parametrize('kind', m.KINDS)
def test_raw_baseline_context_signed_zero_and_subnormal_scalar_oracle(kind):
    head = m.FrozenValue(hand_export(kind), sensing_length=4.)
    x = m.value_features(*inputs(), 4.)
    x[0, 0], x[0, 40 * 105 + 61] = .25, 0.
    before = x.copy()
    expected = []
    for row in x:
        hidden = max(2 * m.GAINS[kind] * row[0] + .5 * row[-3] - .25 * row[-2] + row[-1] + .125, 0)
        expected.append(.25 * row[:11025].sum() - 2 * hidden - .5)
    np.testing.assert_allclose(head.normalized(x), expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(x, before)
    assert head.normalized(x)[-1] == -.75  # Preserve biased zero-input value, no repair.
    baseline = hand_export(kind)
    for name in m.SHAPES:
        baseline[name].fill(0)
    np.testing.assert_array_equal(m.FrozenValue(baseline).normalized(x), .25 * x[:, :11025].sum(axis=1))


@pytest.mark.parametrize('kind', m.KINDS)
def test_torch_numpy_parity_for_raw_cached_and_physical_views(kind):
    import torch

    assert m.value_features is old.value_features
    model = m.make_head(kind, 10102, .37)
    head = m.FrozenValue(m.export_head(model), 4.)
    double = copy.deepcopy(model).double()
    for raw in (m.value_features(*inputs(), 4.), m.value_features(*inputs(), 4., dtype='float32').astype(np.float64)):
        with torch.no_grad():
            expected = double(torch.from_numpy(raw)).numpy()
            single = model(torch.from_numpy(raw.astype(np.float32))).numpy()
        np.testing.assert_allclose(head.normalized(raw), expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(head.normalized(raw), single, atol=2e-6, rtol=2e-6)
    with torch.no_grad():
        physical = 64 * double(torch.from_numpy(m.value_features(*inputs(), 4.))).numpy()
    np.testing.assert_allclose(head(*inputs(), object()), physical, atol=1e-10, rtol=1e-10)


def gradient_model(kind):
    import torch

    model = m.make_head(kind, 10101, 0.).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.weight_0[0, 0] = 1 / m.GAINS[kind]
        model.weight_0[0, -1] = .5
        model.bias_0[0] = .125
        model.weight_1[0, 0] = 1.
    return model


def test_spatial_gradient_changes_coordinates_context_gradient_does_not():
    import torch

    raw, scaled = (gradient_model(kind) for kind in m.KINDS)
    x = torch.zeros((1, 11028), dtype=torch.float64)
    x[0, 0], x[0, -1] = .5, .25
    for model in (raw, scaled):
        model(x).square().mean().backward()
    torch.testing.assert_close(raw(x), scaled(x), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(scaled.weight_0.grad[:, :11025], 53 * raw.weight_0.grad[:, :11025],
                               atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(scaled.weight_0.grad[:, 11025:], raw.weight_0.grad[:, 11025:],
                               atol=1e-14, rtol=1e-14)
    for name in ('bias_0', 'weight_1', 'bias_1'):
        torch.testing.assert_close(getattr(raw, name).grad, getattr(scaled, name).grad, atol=1e-14, rtol=1e-14)
    assert raw.c0.grad is None and scaled.c0.grad is None


def test_same_adam_and_clip_change_effective_spatial_update():
    import torch

    x = torch.zeros((1, 11028), dtype=torch.float64)
    x[0, 0], x[0, -1] = .5, .25
    changes = []
    for kind in m.KINDS:
        model = gradient_model(kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        old_weight = model.weight_0[0, 0].item() * m.GAINS[kind]
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        changes.append(old_weight - model.weight_0[0, 0].item() * m.GAINS[kind])
        assert {int(v['step'].item()) for v in optimizer.state.values()} == {1}
    assert 50 < changes[1] / changes[0] < 54


@pytest.mark.parametrize('kind', m.KINDS)
def test_artificial_backward_has_finite_nonzero_gradients_for_every_parameter(kind):
    import torch

    model = m.make_head(kind, 10101, .25)
    x = torch.from_numpy(m.value_features(*inputs(), 3., dtype='float32'))
    (model(x) - 2).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert model.c0.grad is None


def test_owned_immutable_arrays_export_roundtrip_and_no_repeated_validation(monkeypatch):
    exported = hand_export('gain53')
    stream = io.BytesIO()
    np.savez(stream, **exported)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        restored = {key: archive[key] for key in archive.files}
    head = m.FrozenValue(restored)
    x = m.value_features(*inputs(), 3.)
    expected = head.normalized(x)
    for item in restored.values():
        if item.dtype.kind == 'f':
            item.fill(999)
    monkeypatch.setattr(m, 'validate_head', lambda _: (_ for _ in ()).throw(AssertionError('revalidated')))
    np.testing.assert_array_equal(head.normalized(x), expected)
    for item in (head.c0, *head.weights, *head.biases):
        assert item.dtype == np.float64 and not item.flags.writeable
        with pytest.raises(ValueError):
            item.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        head.spatial_gain = 1
    assert head.storage_bytes()['parameter_array_bytes'] == 88241 * 8
    assert head.storage_bytes()['baseline_array_bytes'] == 8
    assert head.storage_bytes()['mutable_array_bytes'] == 0


@pytest.mark.parametrize('defect', ['missing', 'extra', 'version', 'kind', 'dim', 'bool_dim',
                                  'gain', 'bool_gain', 'float_gain', 'gain_shape', 'dtype', 'shape', 'nan', 'inf'])
def test_invalid_checkpoint_fails_closed(defect):
    head = hand_export('gain53')
    if defect == 'missing':
        del head['bias_1']
    elif defect == 'extra':
        head['optimizer'] = {}
    elif defect in ('version', 'kind'):
        head[defect] = 'unknown'
    elif defect in ('dim', 'bool_dim'):
        head['input_dim'] = 11025 if defect == 'dim' else True
    elif defect in ('gain', 'bool_gain', 'float_gain', 'gain_shape'):
        head['spatial_gain'] = {'gain': 1, 'bool_gain': True, 'float_gain': 53., 'gain_shape': np.array([53])}[defect]
    elif defect == 'dtype':
        head['weight_0'] = head['weight_0'].astype(np.float64)
    elif defect == 'shape':
        head['bias_1'] = np.array(0, np.float32)
    else:
        head['weight_0'][0, 0] = float(defect)
    with pytest.raises(ValueError):
        m.FrozenValue(head)


@pytest.mark.parametrize('defect', ['empty', 'oversized', 'f32', 'shape', 'nan', 'overflow'])
def test_invalid_inference_inputs_fail_before_return(defect):
    head = m.FrozenValue(hand_export('gain53'))
    x = np.zeros((1, 11028), np.float64)
    if defect == 'empty':
        x = x[:0]
    elif defect == 'oversized':
        x = np.broadcast_to(x, (1025, 11028))
    elif defect == 'f32':
        x = x.astype(np.float32)
    elif defect == 'shape':
        x = x[:, :-1]
    else:
        x[0, 0] = np.nan if defect == 'nan' else np.finfo(np.float64).max
    with pytest.raises((ValueError, FloatingPointError)):
        head.normalized(x)


@pytest.mark.parametrize('value', [True, 0, -1, float('inf'), float('nan')])
def test_invalid_bound_sensing_length(value):
    with pytest.raises(ValueError):
        m.FrozenValue(hand_export('gain1'), value)


def test_branch_callback_requires_sensing_and_float64_exports_are_rejected():
    with pytest.raises(ValueError, match='bound sensing'):
        m.FrozenValue(hand_export('gain1'))(*inputs(), object())
    with pytest.raises(ValueError, match='CPU float32'):
        m.export_head(m.make_head('gain1', 10101, .25).double())
