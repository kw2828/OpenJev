"""Synthetic public states and random tiny heads only; no environment or weights."""
import copy

import numpy as np
import pytest

from openjev.research import otto_symmetry_head as m


def kernel():
    axis = np.arange(107, dtype=np.float64) - 53
    radius = axis[:, None] ** 2 + axis[None, :] ** 2
    values = np.stack([np.exp(-radius / scale) / 4 for scale in (3, 30, 300, 3000)])
    values[:, 53, 53] = 0
    return values


def packet(position=(9, 17), step=7):
    actions = [a for a in range(4) if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]
    return {"position": list(position), "hit": 2, "step": step, "done": False, "valid_actions": actions}


def belief():
    values = np.random.default_rng(34).random((53, 53))
    values[values < 0.2] = 0
    return values / values.sum()


def grid_transform(value, group):
    return np.rot90(value[:, ::-1] if group >= 4 else value, group % 4)


def point_transform(position, group):
    # Independent labeled-grid definition of the transformed coordinate.
    grid = np.zeros((53, 53), dtype=np.int8)
    grid[tuple(position)] = 1
    return tuple(int(v) for v in np.argwhere(grid_transform(grid, group))[0])


def action_oracle(group):
    origin = (26, 26)
    neighbors = ((25, 26), (27, 26), (26, 25), (26, 27))
    transformed_origin = point_transform(origin, group)
    assert transformed_origin == origin
    return tuple(neighbors.index(point_transform(neighbor, group)) for neighbor in neighbors)


def reference_features(p, public, k, sensing):
    position = public["position"]
    result = list((53 * np.sqrt(p)).reshape(-1))
    result.extend(v / 26 - 1 for v in position)
    result.extend(int(a in public["valid_actions"]) for a in range(4))
    result.append(sensing / 5)
    for action in range(4):
        pos = list(position)
        pos[action // 2] = max(0, min(52, pos[action // 2] + 2 * (action % 2) - 1))
        result.append(p[tuple(pos)])
        for hit in range(4):
            total = sum(float(p[i, j]) * float(k[hit, 53 + i - pos[0], 53 + j - pos[1]])
                        for i in range(53) for j in range(53))
            result.append(total)
    return np.array(result, dtype=np.float32)


@pytest.mark.parametrize("position", [(9, 17), (0, 0), (52, 26)])
def test_features_match_independent_joint_mass_and_blocked_stay_oracle(position):
    p, k, public = belief(), kernel(), packet(position)
    before = p.copy()
    feature_map = m.PublicFeatureMap(k, 4)
    actual = feature_map.features(p, public)
    np.testing.assert_allclose(actual, reference_features(p, public, k, 4), atol=2e-8, rtol=1e-6)
    np.testing.assert_array_equal(p, before)
    assert actual.dtype == np.float32 and actual.shape == (2836,)
    assert actual[m.SENSING] == np.float32(0.8)


@pytest.mark.parametrize("group", range(8))
def test_d4_features_covary_with_physical_symmetric_kernel(group):
    p, k, public = belief(), kernel(), packet((0, 17))
    feature_map = m.PublicFeatureMap(k, 5)
    x = feature_map.features(p, public)
    moved_packet = packet(point_transform(public["position"], group))
    rebuilt = feature_map.features(grid_transform(p, group).copy(), moved_packet)
    np.testing.assert_allclose(rebuilt, m.transform_features(x, group), rtol=1e-6, atol=1e-8)
    assert m.action_permutation(group) == action_oracle(group)
    values = np.arange(4)
    transformed = m.transform_actions(values, group)
    np.testing.assert_array_equal(transformed[list(action_oracle(group))], values)


def test_feature_map_does_not_normalize_floor_or_use_hit_step_history():
    feature_map = m.PublicFeatureMap(kernel(), 3)
    p = belief() * 1e-14
    x = feature_map.features(p, packet())
    np.testing.assert_array_equal(x[:2809], (53 * np.sqrt(p)).astype(np.float32).reshape(-1))
    public = packet()
    public.update(hit=0, step=123456)
    np.testing.assert_array_equal(x, feature_map.features(p, public))
    zero = feature_map.features(np.zeros((53, 53)), public)
    assert not zero[:2809].any() and not zero[m.FORECAST:].any()
    assert set(m.PublicFeatureMap.__slots__) == {"kernel", "sensing_length"}


def test_kernel_is_owned_immutable_and_storage_complete():
    k = kernel()
    feature_map = m.PublicFeatureMap(k, 3)
    saved = feature_map.kernel.copy()
    k.fill(0)
    np.testing.assert_array_equal(feature_map.kernel, saved)
    with pytest.raises(ValueError):
        feature_map.kernel.setflags(write=True)
    assert feature_map.storage_bytes()["immutable_array_bytes"] == 4 * 107 * 107 * 8


@pytest.mark.parametrize("mutation", ["negative", "nan", "shape", "dtype", "asymmetric", "origin"])
def test_reject_bad_kernel(mutation):
    k = kernel()
    if mutation == "negative":
        k[0, 0, 0] = -1
    elif mutation == "nan":
        k[0, 0, 0] = np.nan
    elif mutation == "shape":
        k = k[:, :-1]
    elif mutation == "dtype":
        k = k.astype(np.float32)
    elif mutation == "asymmetric":
        k[0, 0, 1] = 0.123
    else:
        k[:, 53, 53] = 0.1
    with pytest.raises(ValueError):
        m.PublicFeatureMap(k, 3)


@pytest.mark.parametrize("mutation", ["hidden", "terminal", "actions", "bool_hit", "initial", "negative_p", "nan_p", "f32_p"])
def test_reject_bad_public_inputs(mutation):
    p, public = belief(), packet()
    if mutation == "hidden":
        public["source"] = [1, 2]
    elif mutation == "terminal":
        public.update(done=True, hit=-2)
    elif mutation == "actions":
        public["valid_actions"] = [1, 2, 3]
    elif mutation == "bool_hit":
        public["hit"] = True
    elif mutation == "initial":
        public["step"] = 0
    elif mutation == "negative_p":
        p[0, 0] = -1
    elif mutation == "nan_p":
        p[0, 0] = np.nan
    else:
        p = p.astype(np.float32)
    with pytest.raises((ValueError, TypeError)):
        m.PublicFeatureMap(kernel(), 3).features(p, public)


@pytest.mark.parametrize("kind", m.KINDS)
def test_parameter_counts_seed_locality_and_numpy_torch_parity(kind):
    import torch

    before = torch.random.get_rng_state().clone()
    model = m.make_head(kind, 24)
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == m.parameter_count(kind)
    assert m.parameter_count(kind) == (91329 if kind == "d4_shared" else 91380)
    x = np.random.default_rng(22).normal(size=(3, m.INPUT_DIM)).astype(np.float32)
    exported = m.export_head(model)
    frozen = m.FrozenHead(exported)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(frozen.scores(x), expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(frozen.scores(x[0]), expected[0], rtol=2e-5, atol=2e-6)
    same = m.export_head(m.make_head(kind, 24))
    for key in m.WEIGHTS:
        np.testing.assert_array_equal(same[key], exported[key])
    assert frozen.scores(x[0]).dtype == np.float32


@pytest.mark.parametrize("route", ["d4_shared", "dense_ensemble"])
def test_exact_group_score_equivariance_and_unique_action_mapping(route):
    kind = "dense_augmented" if route == "dense_ensemble" else route
    frozen = m.FrozenHead(m.export_head(m.make_head(kind, 11)))
    x = m.PublicFeatureMap(kernel(), 4).features(belief(), packet((0, 17)))
    scores = frozen.scores(x, kind=route)
    assert np.unique(scores).size == 4
    for group in range(8):
        transformed = frozen.scores(m.transform_features(x, group), kind=route)
        np.testing.assert_allclose(transformed, m.transform_actions(scores, group), rtol=1e-5, atol=2e-6)
        assert int(transformed.argmin()) == m.action_permutation(group)[int(scores.argmin())]


def test_shared_symmetric_tie_set_not_impossible_equivariant_tie_argmin():
    p = np.ones((53, 53), dtype=np.float64)
    p[26, 26] = 0
    p /= p.sum()
    x = m.PublicFeatureMap(kernel(), 3).features(p, packet((26, 26), 0))
    frozen = m.FrozenHead(m.export_head(m.make_head("d4_shared", 15)))
    scores = frozen.scores(x)
    tied = set(np.flatnonzero(np.abs(scores - scores.min()) <= 2e-6))
    assert tied == set(range(4))
    for group in range(8):
        assert {m.action_permutation(group)[a] for a in tied} == tied


def test_dense_ensemble_reuses_weights_and_averages_costs_not_probabilities():
    frozen = m.FrozenHead(m.export_head(m.make_head("dense_augmented", 8)))
    x = np.random.default_rng(23).normal(size=m.INPUT_DIM).astype(np.float32)
    expected = np.stack([frozen.scores(m.transform_features(x, g))[list(action_oracle(g))] for g in range(8)]).mean(axis=0)
    np.testing.assert_allclose(frozen.scores(x, kind="dense_ensemble"), expected, rtol=1e-5, atol=2e-6)
    assert frozen.storage_bytes()["parameter_array_bytes"] == 91380 * 4
    with pytest.raises(ValueError):
        frozen.scores(x, kind="d4_shared")


def test_frozen_export_has_no_mutable_weight_alias_or_history_and_bounds_batches():
    export = m.export_head(m.make_head("dense_augmented", 2))
    frozen = m.FrozenHead(export)
    x = np.zeros(m.INPUT_DIM, dtype=np.float32)
    expected = frozen.scores(x)
    export["weight0"].fill(77)
    np.testing.assert_array_equal(frozen.scores(x), expected)
    for value in frozen.arrays:
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert set(m.FrozenHead.__slots__) == {"kind", "arrays"}
    with pytest.raises(ValueError):
        frozen.scores(np.zeros((m.MAX_BATCH + 1, m.INPUT_DIM), dtype=np.float32))
    with pytest.raises(ValueError):
        frozen.scores(x.astype(np.float64))


@pytest.mark.parametrize("mutation", ["metadata", "shape", "dtype", "nan", "extra"])
def test_corrupt_exports_rejected(mutation):
    value = m.export_head(m.make_head("d4_shared", 4))
    if mutation == "metadata":
        value["input_dim"] = True
    elif mutation == "shape":
        value["weight2"] = np.zeros((4, 16), dtype=np.float32)
    elif mutation == "dtype":
        value["weight0"] = value["weight0"].astype(np.float64)
    elif mutation == "nan":
        value["bias1"][0] = np.nan
    else:
        value["mean"] = np.zeros(m.INPUT_DIM)
    with pytest.raises((ValueError, TypeError)):
        m.FrozenHead(value)


@pytest.mark.parametrize("kind", m.KINDS)
def test_training_losses_and_gradients_match_independent_frame_oracle(kind):
    import torch

    model = m.make_head(kind, 17)
    other = copy.deepcopy(model)
    x = np.random.default_rng(31).normal(size=(2, m.INPUT_DIM)).astype(np.float32)
    targets = np.array([[0.1, 0.4, 0.2, 0.3], [0, 0.25, 0, 0.75]], dtype=np.float32)
    allowed = np.array([[True] * 4, [False, True, False, True]])
    actual = m.training_losses(model, torch.from_numpy(x), torch.from_numpy(targets), torch.from_numpy(allowed))

    def ce(scores, target, mask):
        logp = torch.log_softmax((-scores).masked_fill(~mask, -torch.inf), dim=-1)
        return -(target * logp.masked_fill(~mask, 0)).sum(-1)

    if kind == "dense_augmented":
        terms = []
        for group in range(8):
            costs = other.core(torch.from_numpy(m.transform_features(x, group)))
            target = torch.from_numpy(m.transform_actions(targets, group))
            mask = torch.from_numpy(m.transform_actions(allowed, group))
            terms.append(ce(costs, target, mask))
        expected = torch.stack(terms).mean(0)
    else:
        per_frame = [other.core(torch.from_numpy(m.transform_features(x, g))).squeeze(-1) for g in range(8)]
        costs = torch.stack([sum(per_frame[g] for g in range(8) if action_oracle(g)[a] == 0) / 2
                             for a in range(4)], dim=-1)
        expected = ce(costs, torch.from_numpy(targets), torch.from_numpy(allowed))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    # Unequal episode weights are outside the helper and retain their scale.
    weights = torch.tensor([0.25, 1.75])
    (actual * weights).mean().backward()
    (expected * weights).mean().backward()
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        assert left.grad is not None and torch.isfinite(left.grad).all()
        torch.testing.assert_close(left.grad, right.grad, rtol=3e-4, atol=3e-6)


def test_training_rejects_disallowed_target_or_empty_mask_without_repair():
    import torch

    model = m.make_head("d4_shared", 1)
    x = torch.zeros((1, m.INPUT_DIM))
    target = torch.tensor([[1.0, 0, 0, 0]])
    for mask in (torch.tensor([[False, True, True, True]]), torch.zeros((1, 4), dtype=torch.bool)):
        with pytest.raises(ValueError):
            m.training_losses(model, x, target, mask)
