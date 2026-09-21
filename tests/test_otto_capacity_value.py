"""Artificial model/array tests only; no corpus, checkpoint or native simulator."""
import copy
import dataclasses
import io

import numpy as np
import pytest

from openjev.research import otto_capacity_value as m
from openjev.research import otto_return_value as old


def public_features():
    belief = np.zeros((5, 105, 105), dtype=np.float64)
    belief[0, 17, 91], belief[0, 81, 4] = .25, .75
    belief[1, 21, 17] = 1e-12
    belief[2] = np.arange(1, 11026).reshape(105, 105)
    belief[2] /= belief[2].sum()
    belief[3, 32, 52] = .5
    positions = np.array([[0, 52], [52, 0], [26, 26], [4, 13], [10, 20]], dtype=np.int64)
    return belief, positions


def hand_export(kind):
    dims = (11028, *((8,) if kind == 'mlp8' else (128,) if kind == 'mlp128' else (128, 128, 128)), 1)
    result = {'version': m.VERSION, 'kind': kind, 'input_dim': 11028, 'c0': np.array(.25, np.float32)}
    for i, (left, right) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        result[f'weight_{i}'] = np.zeros((right, left), np.float32)
        result[f'bias_{i}'] = np.zeros(right, np.float32)
    result['weight_0'][0, 0], result['weight_0'][1, 1] = 2., -3.
    result['bias_0'][:2] = [-.5, .25]
    for i in range(1, len(dims)-2):
        result[f'weight_{i}'][0, 0], result[f'weight_{i}'][1, 1] = 1.5, .5
        result[f'bias_{i}'][:2] = [-.125, .25]
    result[f'weight_{len(dims)-2}'][0, :2] = [-2., .5]
    result[f'bias_{len(dims)-2}'][0] = -.25
    return result


def test_qualified_feature_function_is_reused_without_a_new_transform():
    assert m.value_features is old.value_features
    b, p = public_features()
    np.testing.assert_array_equal(m.value_features(b, p, 4), old.value_features(b, p, 4))
    assert not m.value_features(b, p, 4)[-1].any()


@pytest.mark.parametrize(('kind', 'count'), [('mlp8', 88241), ('mlp128', 1411841), ('deep128', 1444865)])
def test_parameter_counts_shapes_cpu_dtype_zero_bias_and_exact_baseline(kind, count):
    import torch

    model = m.make_head(kind, 10101, .37)
    exported = m.export_head(model)
    assert m.parameter_count(kind) == count == sum(p.numel() for p in model.parameters())
    assert model.c0.item() == float(np.float32(.37)) and not model.c0.requires_grad
    assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
    assert all(not np.any(v) for k, v in exported.items() if k.startswith('bias_'))
    assert m.validate_head(exported) == kind
    assert set(dict(model.named_parameters())) == {k for k in exported if k.startswith(('weight_', 'bias_'))}


def test_initialization_pairs_first_rows_without_changing_global_rng():
    import torch

    before = torch.random.get_rng_state().clone()
    models = {kind: m.export_head(m.make_head(kind, 10102, .25)) for kind in m.KINDS}
    assert torch.equal(torch.random.get_rng_state(), before)
    for kind in m.KINDS:
        np.testing.assert_array_equal(models[kind]['weight_0'][:8], models['mlp8']['weight_0'])
    np.testing.assert_array_equal(models['mlp128']['weight_0'], models['deep128']['weight_0'])
    again = m.export_head(m.make_head('deep128', 10102, .25))
    assert all(np.array_equal(v, again[k]) for k, v in models['deep128'].items())
    different = m.export_head(m.make_head('mlp8', 10103, .25))
    assert not np.array_equal(different['weight_0'], models['mlp8']['weight_0'])


@pytest.mark.parametrize('kind', m.KINDS)
def test_sparse_scalar_oracle_signed_output_zero_and_subnormalized_inputs(kind):
    head = m.FrozenValue(hand_export(kind), 3)
    x = np.zeros((4, 11028), np.float64)
    x[:, :2] = [[.25, .5], [.75, 0], [5e-12, 0], [0, 0]]
    expected = []
    for row in x:
        a, b = max(2*row[0]-.5, 0), max(.25-3*row[1], 0)
        if kind == 'deep128':
            for _ in range(2):
                a, b = max(1.5*a-.125, 0), max(.5*b+.25, 0)
        expected.append(.25*(row[0]+row[1])-2*a+.5*b-.25)
    np.testing.assert_array_equal(head.normalized(x), expected)
    assert np.any(head.normalized(x) < 0) and head.normalized(x)[-1] != 0
    belief = x[:, :11025].reshape(4, 105, 105)
    position = np.zeros((4, 2), np.int64)
    np.testing.assert_array_equal(head(belief, position, object()), 64*np.asarray(expected))


@pytest.mark.parametrize('kind', m.KINDS)
def test_torch_double_and_float32_parity_on_cached_and_raw_public_views(kind):
    import torch

    model = m.make_head(kind, 10103, .37)
    exported = m.export_head(model)
    head = m.FrozenValue(exported, 4)
    raw = m.value_features(*public_features(), 4)
    cached = raw.astype(np.float32)
    with torch.no_grad():
        single = model(torch.from_numpy(cached)).numpy()
        double_model = copy.deepcopy(model).double()
        double_cached = double_model(torch.from_numpy(cached.astype(np.float64))).numpy()
        double_raw = double_model(torch.from_numpy(raw)).numpy()
    np.testing.assert_allclose(head.normalized(cached.astype(np.float64)), double_cached, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(head.normalized(raw), double_raw, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(head.normalized(cached.astype(np.float64)), single, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(head(*public_features(), object()), 64*double_raw, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('kind', m.KINDS)
def test_artificial_backward_reaches_every_layer_with_finite_gradients(kind):
    import torch

    model = m.make_head(kind, 10101, .25)
    x = torch.from_numpy(m.value_features(*public_features(), 3, dtype='float32'))
    loss = ((model(x)-2.)**2).mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert model.c0.grad is None


def test_immutable_storage_owns_no_mutable_export_and_validation_paid_once(monkeypatch):
    exported = hand_export('deep128')
    head = m.FrozenValue(exported)
    x = np.zeros((1, 11028), np.float64)
    before = head.normalized(x)
    for value in exported.values():
        if isinstance(value, np.ndarray):
            value.fill(99)
    monkeypatch.setattr(m, 'validate_head', lambda _: (_ for _ in ()).throw(AssertionError('repeated validation')))
    np.testing.assert_array_equal(head.normalized(x), before)
    for value in (head.c0, *head.weights, *head.biases):
        assert value.dtype == np.float64 and not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        head.kind = 'mlp8'
    stored = head.storage_bytes()
    assert stored['parameter_array_bytes'] == 1444865*8 and stored['baseline_array_bytes'] == 8
    assert stored['mutable_array_bytes'] == 0


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'version', 'kind', 'dim', 'bool_dim', 'f64', 'shape', 'nan', 'inf', 'c0_shape', 'object'])
def test_invalid_checkpoint_rejected_before_inference(mutation):
    h = hand_export('mlp8')
    if mutation == 'missing':
        del h['bias_0']
    elif mutation == 'extra':
        h['optimizer'] = {}
    elif mutation in ('version', 'kind'):
        h[mutation] = 'unsupported'
    elif mutation in ('dim', 'bool_dim'):
        h['input_dim'] = 11025 if mutation == 'dim' else True
    elif mutation == 'f64':
        h['weight_0'] = h['weight_0'].astype(np.float64)
    elif mutation == 'shape':
        h['bias_1'] = np.array(0, np.float32)
    elif mutation in ('nan', 'inf'):
        h['weight_0'][0, 0] = float(mutation)
    elif mutation == 'c0_shape':
        h['c0'] = np.array([.25], np.float32)
    else:
        h['kind'] = np.array('mlp8', dtype=object)
    with pytest.raises(ValueError):
        m.FrozenValue(h)


@pytest.mark.parametrize('mutation', ['dtype', 'width', 'empty', 'large_batch', 'nan'])
def test_bad_feature_arrays_fail_closed(mutation):
    head = m.FrozenValue(hand_export('mlp8'))
    x = np.zeros((1, 11028), np.float64)
    if mutation == 'dtype':
        x = x.astype(np.float32)
    elif mutation == 'width':
        x = x[:, :-1]
    elif mutation == 'empty':
        x = x[:0]
    elif mutation == 'large_batch':
        x = np.broadcast_to(x, (1025, 11028))
    else:
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        head.normalized(x)


@pytest.mark.parametrize('value', [None, True, 0, -1, np.inf, np.nan])
def test_missing_or_invalid_sensing_cannot_call_branch_model(value):
    if value is None:
        model = m.FrozenValue(hand_export('mlp8'))
        with pytest.raises(ValueError):
            model(*public_features(), object())
    else:
        with pytest.raises(ValueError):
            m.FrozenValue(hand_export('mlp8'), value)


@pytest.mark.parametrize('seed', [True, -1, 2**63, .5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        m.make_head('mlp8', seed, .25)


def test_npz_roundtrip_is_pickle_free_and_export_is_independent():
    import torch

    model = m.make_head('mlp8', 10101, .25)
    exported = m.export_head(model)
    stream = io.BytesIO()
    np.savez_compressed(stream, **exported)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        head = m.FrozenValue(dict(archive))
    x = np.zeros((2, 11028), np.float64)
    np.testing.assert_array_equal(head.normalized(x), m.FrozenValue(exported).normalized(x))
    with torch.no_grad():
        model.weight_0.fill_(float('nan'))
    assert np.isfinite(exported['weight_0']).all()
    with pytest.raises(ValueError):
        m.export_head(model)
