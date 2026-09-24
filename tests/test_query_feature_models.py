"""Fabricated Gaussian algebra and public-interface witnesses; no study data."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch

from openjev.research.query_feature_models import (
    NOISE_VARIANCE,
    NYSTROM_JITTER,
    NystromFeatureModel,
    QueryFeatureModel,
    gaussian_feature_mixture,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def feature_fixture():
    # One scalar feature and two blocks: all exact inputs are small integers.
    return tensor([[[[1.]], [[2.]]]]), tensor([[[1.], [0.]]]), \
        tensor([[[1.], [2.]]]), tensor([[1., -1.]]), tensor([[1.]])


def public_fixture():
    bx = tensor([[[[-1., 0.], [0., 1.], [1., -.5]],
                  [[-.5, 1.], [1., 0.], [0., -1.]]]])
    by = tensor([[[.5, -.25, 1.], [-.5, .75, .25]]])
    fx, fy, qx = tensor([[[.25, -.5], [.5, .25]]]), tensor([[.25, -.75]]), tensor([[.5, -.25]])
    return bx, by, fx, fy, qx


def direct_gaussian(block, labels, few, few_labels, query, noise):
    """Independent data-space conditioning, never precision-space updates."""
    block_cov = block @ block.T + noise * np.eye(len(block))
    cross = few @ block.T
    fm = cross @ np.linalg.solve(block_cov, labels)
    fc = few @ few.T + noise * np.eye(len(few)) - cross @ np.linalg.solve(block_cov, cross.T)
    err = few_labels - fm
    ll = -.5 * (len(few) * math.log(2 * math.pi) + np.linalg.slogdet(fc)[1]
                + err @ np.linalg.solve(fc, err))
    context, targets = np.concatenate((block, few)), np.concatenate((labels, few_labels))
    cov = context @ context.T + noise * np.eye(len(context))
    qc = query @ context.T
    mean = qc @ np.linalg.solve(cov, targets)
    variance = noise + query @ query - qc @ np.linalg.solve(cov, qc)
    return mean, variance, ll, fm, fc


def test_rational_components_joint_evidence_and_mixture_density():
    result = gaussian_feature_mixture(*feature_fixture(), noise_variance=1.)
    torch.testing.assert_close(result["component_mean"], tensor([[0., -.1]]), atol=2e-15, rtol=0)
    torch.testing.assert_close(result["component_variance"], tensor([[8 / 7, 11 / 10]]),
                               atol=2e-15, rtol=0)
    ll = tensor([[-math.log(2 * math.pi) - .5 * math.log(3.5) - 1.25,
                  -math.log(2 * math.pi) - .5 * math.log(2.) - .95]])
    expected_weights = ll - torch.logsumexp(ll, dim=1, keepdim=True)
    torch.testing.assert_close(result["log_weights"], expected_weights, atol=2e-15, rtol=0)
    means, variances = [0., -.1], [8 / 7, 11 / 10]
    weights = expected_weights.exp().numpy()[0]
    positive = sum(w * .5 * (1 + math.erf(m / math.sqrt(2 * v)))
                   for w, m, v in zip(weights, means, variances, strict=True))
    assert result["prob_positive"].item() == pytest.approx(positive, abs=2e-15)
    target = .3
    density = sum(w * math.exp(-(target - m)**2 / (2 * v)) / math.sqrt(2 * math.pi * v)
                  for w, m, v in zip(weights, means, variances, strict=True))
    assert result.log_prob(tensor([target])).item() == pytest.approx(math.log(density), abs=2e-15)
    assert set(result) == {"component_mean", "component_variance", "log_weights", "prob_positive"}


def test_direct_joint_gaussian_conditioning_and_wrong_independence_witness():
    args = feature_fixture()
    result = gaussian_feature_mixture(*args, noise_variance=1.)
    expected, independent = [], []
    for k in range(2):
        row = direct_gaussian(args[0][0, k].numpy(), args[1][0, k].numpy(),
                              args[2][0].numpy(), args[3][0].numpy(), args[4][0].numpy(), 1.)
        expected.append(row[:3])
        err = args[3][0].numpy() - row[3]
        independent.append(-.5 * np.sum(np.log(2 * np.pi * np.diag(row[4]))
                                        + err**2 / np.diag(row[4])))
    expected = np.asarray(expected)
    np.testing.assert_allclose(result["component_mean"].numpy()[0], expected[:, 0], atol=2e-15)
    np.testing.assert_allclose(result["component_variance"].numpy()[0], expected[:, 1], atol=2e-15)
    joint_weights = np.exp(expected[:, 2] - np.max(expected[:, 2]))
    joint_weights /= joint_weights.sum()
    wrong_weights = np.exp(independent - np.max(independent))
    wrong_weights /= wrong_weights.sum()
    np.testing.assert_allclose(result["log_weights"].exp().numpy()[0], joint_weights, atol=2e-15)
    # Independent diagonal approximation has determinants 9/2 and54/25,
    # and quadratic forms3/2 and25/18. Its odds differ algebraically.
    assert math.log(wrong_weights[0] / wrong_weights[1]) == pytest.approx(
        .5 * math.log(12 / 25) - 1 / 18, abs=2e-15)
    assert math.log(joint_weights[0] / joint_weights[1]) == pytest.approx(
        .5 * math.log(4 / 7) - .3, abs=2e-15)
    assert not np.allclose(joint_weights, wrong_weights, atol=1e-12, rtol=1e-12)


def test_replay_is_fresh_recomputation_and_unique_fewshots_assimilated_once():
    args = list(feature_fixture())
    args[0], args[1] = args[0][:, :1], args[1][:, :1]
    saved = [x.clone() for x in args]
    first = gaussian_feature_mixture(*args, noise_variance=1.)
    for _ in range(3):
        replay = gaussian_feature_mixture(*args, noise_variance=1.)
        for name in first:
            torch.testing.assert_close(replay[name], first[name], atol=0, rtol=0)
    # A duplicated acquisition IS an extra observation under the IID model;
    # it must not be confused with revisiting the same computation.
    duplicated = args.copy()
    duplicated[2], duplicated[3] = args[2].repeat(1, 2, 1), args[3].repeat(1, 2)
    extra_observations = gaussian_feature_mixture(*duplicated, noise_variance=1.)
    assert extra_observations["component_variance"].item() < first["component_variance"].item()
    for old, current in zip(saved, args, strict=True):
        torch.testing.assert_close(current, old, atol=0, rtol=0)


def test_rank_deficient_and_zero_features_remain_spd_without_repair():
    b = torch.zeros((2, 3, 4, 5), dtype=torch.float64)
    result = gaussian_feature_mixture(b, torch.ones((2, 3, 4), dtype=torch.float64),
                                       torch.zeros((2, 2, 5), dtype=torch.float64),
                                       tensor([[1., -1.], [.5, .25]]),
                                       torch.zeros((2, 5), dtype=torch.float64))
    torch.testing.assert_close(result["component_mean"], torch.zeros((2, 3), dtype=torch.float64))
    torch.testing.assert_close(result["component_variance"], torch.full((2, 3), NOISE_VARIANCE,
                                                                       dtype=torch.float64))
    torch.testing.assert_close(result["log_weights"], torch.full((2, 3), -math.log(3), dtype=torch.float64))
    torch.testing.assert_close(result["prob_positive"], tensor([.5, .5]))


def test_helper_gradients_match_finite_differences():
    bp = tensor([[[[.3, -.5], [.2, .1]], [[-.2, .4], [.5, .3]]]]).requires_grad_()
    by = tensor([[[.2, -.1], [.4, -.3]]])
    fp = tensor([[[.25, -.1], [-.2, .3]]]).requires_grad_()
    fy = tensor([[.15, -.25]])
    qp = tensor([[.2, .4]]).requires_grad_()

    def function(x, f, q):
        value = gaussian_feature_mixture(x, by, f, fy, q, noise_variance=.4)
        return torch.cat((value["component_mean"].flatten(), value["component_variance"].flatten(),
                          value["log_weights"].flatten(), value["prob_positive"]))

    assert torch.autograd.gradcheck(function, (bp, fp, qp), eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("constructor", [QueryFeatureModel, NystromFeatureModel])
@pytest.mark.parametrize("centered", [False, True])
def test_models_public_shapes_gradients_ownership_and_update(constructor, centered):
    model = constructor(centered=centered)
    args = public_fixture()
    before_inputs = [x.clone() for x in args]
    before_parameters = [p.detach().clone() for p in model.parameters()]
    result = model(*args)
    for name in ("component_mean", "component_variance", "log_weights"):
        assert result[name].shape == (1, 2) and result[name].dtype == torch.float64
    assert result["prob_positive"].shape == (1,)
    assert bool((result["component_variance"] >= NOISE_VARIANCE).all())
    torch.testing.assert_close(result["log_weights"].exp().sum(1), tensor([1.]))
    loss = -result.log_prob(tensor([.6])).mean()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert bool((parameter.grad != 0).any())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before_parameters, model.parameters(), strict=True))
    for old, current in zip(before_inputs, args, strict=True):
        torch.testing.assert_close(current, old, atol=0, rtol=0)
    assert tuple(inspect.signature(model.forward).parameters) == ("bx", "by", "fx", "fy", "qx")
    with pytest.raises(TypeError, match="target"):
        model(*args, target=tensor([.6]))
    with pytest.raises(TypeError, match="true_index"):
        model(*args, true_index=0)


@pytest.mark.parametrize("constructor", [QueryFeatureModel, NystromFeatureModel])
def test_centered_translation_invariance_and_static_noninvariance(constructor):
    args = public_fixture()
    shifted = [x.clone() for x in args]
    for index in (0, 2, 4):
        shifted[index] = shifted[index] + tensor([2., -4.])
    centered, static = constructor(centered=True), constructor(centered=False)
    left, right = centered(*args), centered(*shifted)
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=2e-12, rtol=2e-12)
    assert not torch.allclose(static(*args)["component_mean"], static(*shifted)["component_mean"],
                              atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("constructor", [QueryFeatureModel, NystromFeatureModel])
def test_block_order_and_batch_partition_equivariance(constructor):
    model = constructor(centered=True)
    args = public_fixture()
    batch = [torch.cat((x, x * .5), dim=0) for x in args]
    together = model(*batch)
    for row in range(2):
        separate = model(*(x[row:row + 1] for x in batch))
        for name in separate:
            torch.testing.assert_close(together[name][row:row + 1], separate[name], atol=2e-12, rtol=2e-12)
    permuted = batch.copy()
    permuted[0], permuted[1] = batch[0][:, [1, 0]], batch[1][:, [1, 0]]
    swapped = model(*permuted)
    for name in ("component_mean", "component_variance", "log_weights"):
        torch.testing.assert_close(swapped[name], together[name][:, [1, 0]], atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(swapped["prob_positive"], together["prob_positive"], atol=2e-12, rtol=2e-12)


def test_local_initialization_pairing_wider_control_and_state_roundtrip():
    global_state = torch.random.get_rng_state().clone()
    static = QueryFeatureModel(seed=951101)
    centered = QueryFeatureModel(seed=951101, centered=True)
    wider = QueryFeatureModel(rank=32, seed=951101)
    NystromFeatureModel()
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert static.parameter_count == 624 and wider.parameter_count == 1152
    for key, value in static.state_dict().items():
        assert torch.equal(value, centered.state_dict()[key])
    assert torch.equal(static.encoder[0].weight, wider.encoder[0].weight)
    assert torch.equal(static.encoder[0].bias, wider.encoder[0].bias)
    clone = QueryFeatureModel(seed=951102)
    clone.load_state_dict(static.state_dict())
    for key, value in static(*public_fixture()).items():
        torch.testing.assert_close(value, clone(*public_fixture())[key], atol=0, rtol=0)
    assert wider(*public_fixture())["component_mean"].shape == (1, 2)


def test_nystrom_matches_independent_kernel_factorization_and_parameter_gradients():
    model = NystromFeatureModel(centered=True)
    assert model.parameter_count == 2
    grid = np.array([(a, b) for a in np.linspace(-2, 2, 4) for b in np.linspace(-2, 2, 4)])
    np.testing.assert_allclose(model.inducing_grid.numpy(), grid, atol=5e-16, rtol=0)
    inputs = tensor([[.1, .3], [-.25, -.75]])
    kzz = np.exp(-.5 * np.sum((grid[:, None] - grid[None])**2, axis=-1)) + NYSTROM_JITTER * np.eye(16)
    kxz = np.exp(-.5 * np.sum((inputs.numpy()[:, None] - grid[None])**2, axis=-1))
    expected = np.linalg.solve(np.linalg.cholesky(kzz), kxz.T).T
    np.testing.assert_allclose(model.features(inputs).detach().numpy(), expected, atol=2e-15, rtol=2e-15)
    objective = -model(*public_fixture()).log_prob(tensor([.4])).sum()
    objective.backward()
    for parameter in (model.log_length, model.log_amplitude):
        analytic = parameter.grad.item()
        original = parameter.item()
        values = []
        for delta in (1e-5, -1e-5):
            with torch.no_grad():
                parameter.fill_(original + delta)
            values.append(-model(*public_fixture()).log_prob(tensor([.4])).item())
        with torch.no_grad():
            parameter.fill_(original)
        assert analytic == pytest.approx((values[0] - values[1]) / 2e-5, rel=2e-6, abs=2e-7)


@pytest.mark.parametrize("constructor", [QueryFeatureModel, NystromFeatureModel])
def test_static_cache_matches_recomputation_gradients_bytes_and_rejects_stale_model(constructor):
    model = constructor()
    args = public_fixture()
    full = model(*args)
    full_grad = torch.autograd.grad(-full.log_prob(tensor([.4])).sum(), tuple(model.parameters()))
    cache = model.encode_archive(*args[:2])
    cached = model.predict_cached(cache, *args[2:])
    cache_grad = torch.autograd.grad(-cached.log_prob(tensor([.4])).sum(), tuple(model.parameters()))
    for name in full:
        torch.testing.assert_close(full[name], cached[name], atol=2e-12, rtol=2e-12)
    for left, right in zip(full_grad, cache_grad, strict=True):
        torch.testing.assert_close(left, right, atol=2e-11, rtol=2e-11)
    assert cache.tensor_bytes == 8 * 1 * 2 * (3 * 16 + 2 * 16**2 + 2 * 16)
    saved = [x.clone() for x in (cache.block_features, cache.precision, cache.eta,
                                 cache.cholesky, cache.weight_mean)]
    model.predict_cached(cache, *args[2:])
    for old, current in zip(saved, (cache.block_features, cache.precision, cache.eta,
                                    cache.cholesky, cache.weight_mean), strict=True):
        torch.testing.assert_close(old, current, atol=0, rtol=0)
    with pytest.raises(ValueError, match="owner"):
        constructor().predict_cached(cache, *args[2:])
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    with pytest.raises(ValueError, match="stale"):
        model.predict_cached(cache, *args[2:])
    with pytest.raises(ValueError, match="query-centered"):
        constructor(centered=True).encode_archive(*args[:2])
    with pytest.raises(ValueError, match="query-centered"):
        constructor(centered=True).predict_cached(cache, *args[2:])


@pytest.mark.parametrize("field", range(5))
def test_nonfinite_public_inputs_rejected(field):
    args = list(public_fixture())
    args[field] = args[field].clone()
    args[field].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        QueryFeatureModel()(*args)


@pytest.mark.parametrize("bad", [0., -1., float("inf"), float("nan"), True])
def test_invalid_noise_rejected(bad):
    with pytest.raises(ValueError, match="noise_variance"):
        gaussian_feature_mixture(*feature_fixture(), noise_variance=bad)


def test_shapes_dtypes_empty_support_and_extreme_parameters_fail_explicitly():
    args = list(public_fixture())
    args[2] = args[2].float()
    with pytest.raises(ValueError, match="float64"):
        QueryFeatureModel()(*args)
    args = list(public_fixture())
    args[4] = tensor([[.1]])
    with pytest.raises(ValueError, match="shapes"):
        QueryFeatureModel()(*args)
    features = list(feature_fixture())
    features[2], features[3] = features[2][:, :0], features[3][:, :0]
    with pytest.raises(ValueError, match="positive"):
        gaussian_feature_mixture(*features)
    features = list(feature_fixture())
    features[0] = features[0] * 1e200
    with pytest.raises(ValueError, match="sufficient statistics"):
        gaussian_feature_mixture(*features)
    with pytest.raises(ValueError, match="Nyström rank"):
        NystromFeatureModel(rank=32)
    with pytest.raises(ValueError, match="rank"):
        QueryFeatureModel(rank=True)
    with pytest.raises(ValueError, match="seed"):
        QueryFeatureModel(seed=-1)
    with pytest.raises(ValueError, match="Boolean"):
        QueryFeatureModel(centered=1)
    model = NystromFeatureModel()
    with torch.no_grad():
        model.log_amplitude.fill_(1000.)
    with pytest.raises(ValueError, match="RBF"):
        model(*public_fixture())
    result = gaussian_feature_mixture(*feature_fixture())
    with pytest.raises(ValueError, match="target"):
        result.log_prob(tensor([0., 1.]))
    with pytest.raises(ValueError, match="finite"):
        result.log_prob(tensor([float("inf")]))
