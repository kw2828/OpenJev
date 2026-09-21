"""Artificial actor arrays and CPU MLP fixtures; no simulator or task data."""
from __future__ import annotations

import copy
from collections import namedtuple
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from openjev.research import otto_action_head as M


class PublicActor:
    def __init__(self, arm, *, step=0, position=(26, 26), hit=2):
        self.position, self.step, self.initial_hit, self.done = position, step, 2, False
        self.excluded = np.zeros((53, 53), dtype=bool)
        self.excluded[position] = True
        self.q, self.extension, self.mode = 16, arm.removeprefix("dct16_"), arm
        self.coefficients = np.zeros((16, 16), dtype=np.float64)
        self.probabilities = (~self.excluded).astype(np.float64)
        self.probabilities /= self.probabilities.sum()
        self.count, self.history = step, np.zeros((32, 3), dtype=np.int64)
        for index in range(max(0, step - 32), step):
            self.history[index % 32] = (index % 53, (index * 2) % 53, index % 4)
        if step:
            self.history[(step - 1) % 32] = (*position, hit)

    def decode(self):
        raise AssertionError("candidate decoding must not be called")

    def belief(self):
        raise AssertionError("dense belief reconstruction must not be called")

    @property
    def source(self):
        raise AssertionError("hidden source must not be accessed")

    @property
    def kernel(self):
        raise AssertionError("kernel/planning must not be accessed")


def packet(step=0, position=(26, 26), hit=2):
    return {"position": list(position), "hit": hit, "done": False, "step": step,
            "valid_actions": [a for a in range(4)
                              if 0 <= position[a // 2] + (-1 if a % 2 == 0 else 1) < 53]}


@pytest.mark.parametrize("arm", M.ARMS)
def test_shapes_public_context_mask_and_no_decode(arm):
    actor = PublicActor(arm, step=4, position=(0, 52), hit=0)
    before = {k: v.copy() for k, v in vars(actor).items() if isinstance(v, np.ndarray)}
    x = M.features(arm, actor, packet(4, (0, 52), 0), 2, sensing_length=3)
    assert x.dtype == np.float32 and x.shape == (5633 if arm == "full_bayes" else 3080,)
    np.testing.assert_array_equal(x[-2824:-15], actor.excluded.reshape(-1))
    np.testing.assert_array_equal(x[-15:-6], [-1, 1, 0, 1, 0, 1, 0, 0, 0])
    assert x[-6] == pytest.approx(np.log1p(4) / np.log1p(2188))
    np.testing.assert_array_equal(x[-5:-1], [0, 1, 1, 0])
    for k, a in before.items():
        np.testing.assert_array_equal(getattr(actor, k), a)


def test_signed_log_and_full_sqrt_are_exact_feature_definitions():
    actor = PublicActor("dct16_neutral")
    actor.coefficients.flat[:5] = [-100, -1, 0, 1, 100]
    x = M.features("dct16_neutral", actor, packet(), 2, sensing_length=3)
    np.testing.assert_array_equal(x[:5], np.array([-np.log(101), -np.log(2), 0, np.log(2), np.log(101)], dtype=np.float32))
    full = PublicActor("full_bayes")
    full.probabilities[:] = 0
    full.probabilities[0, :2] = [.25, .75]
    x = M.features("full_bayes", full, packet(), 2, sensing_length=3)
    np.testing.assert_array_equal(x[:2], np.array([26.5, 53 * np.sqrt(.75)], dtype=np.float32))
    assert np.count_nonzero(x[:2809]) == 2


@pytest.mark.parametrize("step", [0, 1, 3, 32, 33, 65])
def test_recent_chronology_padding_zero_hits_and_expiry(step):
    actor = PublicActor("recent32_hard", step=step, hit=0 if step else 2)
    x = M.features("recent32_hard", actor, packet(step=step, hit=0 if step else 2), 2, sensing_length=3)
    slots = x[:256].reshape(32, 8)
    n = min(step, 32)
    np.testing.assert_array_equal(slots[:32 - n], np.zeros((32 - n, 8)))
    if n:
        np.testing.assert_array_equal(slots[32 - n:, 6], np.ones(n))
        np.testing.assert_array_equal(slots[32 - n:, 7], np.arange(n - 1, -1, -1) / 32)
        np.testing.assert_array_equal(slots[-1], [0, 0, 1, 0, 0, 0, 1, 0])
    if n > 1:
        first = step - n
        assert slots[32 - n, 0] == pytest.approx((first % 53) / 26 - 1)
        assert slots[32 - n, 2 + first % 4] == 1
    assert slots[:, 2:6].sum() == n  # zero is a category, not padding


def test_namedtuple_public_packet_supported_without_new_fields():
    Public = namedtuple("Public", "position hit done step valid_actions")
    p = packet()
    actor = PublicActor("dct16_nearest")
    np.testing.assert_array_equal(M.features("dct16_nearest", actor, Public(**p), 2, sensing_length=3),
                                  M.features("dct16_nearest", actor, p, 2, sensing_length=3))


def test_sensing_context_required_public_and_fixed_scale():
    actor = PublicActor("dct16_neutral")
    with pytest.raises(TypeError, match="sensing_length"):
        M.features("dct16_neutral", actor, packet(), 2)
    base = M.features("dct16_neutral", actor, packet(), 2, sensing_length=3)
    shift = M.features("dct16_neutral", actor, packet(), 2, sensing_length=4)
    np.testing.assert_array_equal(base[:-1], shift[:-1])
    assert base[-1] == .75 and shift[-1] == 1
    full = PublicActor("full_bayes")
    transformed = M.features("full_bayes", full, packet(), 2, sensing_length=3)
    # The invertible scale has exact RMS one before the required float32 cast.
    assert np.mean(transformed[:2809].astype(np.float64)**2) == pytest.approx(1, abs=3e-7)


@pytest.mark.parametrize("value", [True, 0, -1, 5, np.nan, np.inf, "3", None])
def test_invalid_supplied_sensing_scalar(value):
    with pytest.raises(ValueError, match="sensing_length"):
        M.features("dct16_neutral", PublicActor("dct16_neutral"), packet(), 2, sensing_length=value)


@pytest.mark.parametrize("field", ["source", "seed", "p_source", "target", "lambda_over_dx"])
def test_nonpublic_packet_fields_rejected(field):
    p = packet()
    p[field] = object()
    with pytest.raises(ValueError, match="five public fields"):
        M.features("dct16_neutral", PublicActor("dct16_neutral"), p, 2, sensing_length=3)


@pytest.mark.parametrize("defect", ["terminal", "step", "position", "hit", "initial", "legal", "mask", "nan", "identity"])
def test_invalid_public_or_actor_state_is_rejected(defect):
    p, a = packet(), PublicActor("dct16_neutral")
    if defect == "terminal":
        p.update(done=True, hit=-2, valid_actions=[])
    elif defect == "step":
        p["step"] = 1
    elif defect == "position":
        p["position"] = [53, 0]
    elif defect == "hit":
        p["hit"] = 4
    elif defect == "initial":
        p["hit"] = 1
    elif defect == "legal":
        p["valid_actions"] = [0, 2, 1, 3]
    elif defect == "mask":
        a.excluded[:] = True
    elif defect == "nan":
        a.coefficients[0, 0] = np.nan
    else:
        a.extension = "nearest"
    with pytest.raises(ValueError):
        M.features("dct16_neutral", a, p, 2, sensing_length=3)


def test_recent_latest_reading_must_match_current_packet():
    a = PublicActor("recent32_hard", step=33)
    a.history[0, 2] = 3
    with pytest.raises(ValueError, match="latest retained"):
        M.features("recent32_hard", a, packet(33), 2, sensing_length=3)


def test_full_belief_must_preserve_exact_excluded_support():
    a = PublicActor("full_bayes")
    a.probabilities[:] = 1 / 2809
    with pytest.raises(ValueError, match="exact exclusions"):
        M.features("full_bayes", a, packet(), 2, sensing_length=3)


def test_standardizer_population_moments_floor_and_no_inference_mutation():
    train = np.array([[0, 0, -4], [0, 1, 0], [0, 0, 4]], dtype=np.float32)
    mean, scale = M.fit_standardizer(train)
    np.testing.assert_allclose(mean, [0, 1 / 3, 0], rtol=1e-7)
    np.testing.assert_allclose(scale, [1, 1, np.sqrt(32 / 3)], rtol=1e-7)
    before = mean.copy(), scale.copy()
    shifted = M.standardize(np.array([1, 1, 8]), mean, scale)
    assert shifted[0] == 1 and shifted[1] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(mean, before[0])
    np.testing.assert_array_equal(scale, before[1])


@pytest.mark.parametrize("bad", [np.empty((0, 4)), np.array([1., 2.]), np.array([[np.nan]])])
def test_bad_training_standardizer_rejected(bad):
    with pytest.raises(ValueError):
        M.fit_standardizer(bad)


def test_seed_pairing_global_rng_and_honest_parameter_counts(monkeypatch):
    def forbid_accelerator_seeding(*_args, **_kwargs):
        raise AssertionError("CPU head construction must not seed accelerator generators")

    monkeypatch.setattr(torch, "manual_seed", forbid_accelerator_seeding)
    state = torch.random.get_rng_state().clone()
    models = [M.make_head(3080, 7101) for _ in range(2)]
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True))
    assert M.parameter_count(models[0]) == M.parameter_count(3080) == 199396
    assert M.parameter_count(M.make_head(5633, 7101)) == M.parameter_count(5633) == 362788
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in models[0].parameters())


@pytest.mark.parametrize("dimension", [3080, 5633])
def test_exported_numpy_torch_parity_and_independent_export(dimension):
    model = M.make_head(dimension, 47).eval()
    rng = np.random.default_rng(42)
    train = rng.normal(0, 2, (5, dimension)).astype(np.float32)
    mean, scale = M.fit_standardizer(train)
    export = M.export_head(model, mean, scale)
    for x in train:
        with torch.inference_mode():
            expected = model(torch.from_numpy(M.standardize(x, mean, scale))).numpy()
        actual = M.predict(export, x, (0, 1, 2, 3))
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    storage = M.storage_bytes(export)
    assert storage["parameter_count"] == M.parameter_count(model)
    assert storage["total_array_bytes"] == 4 * M.parameter_count(model) + 8 * dimension
    old = export["weight0"].copy()
    with torch.no_grad():
        model[0].weight.zero_()
    mean[:] = 123
    np.testing.assert_array_equal(export["weight0"], old)
    assert not np.all(export["mean"] == 123)


def test_closed_form_outputs_legal_mask_and_lower_is_better():
    model = M.make_head(3, 4)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model[4].bias.copy_(torch.tensor([-2., 3., -5., 0.]))
    head = M.export_head(model, np.zeros(3), np.ones(3))
    assert M.predict(head, np.array([10, 20, 30]), [0, 1, 3]) == [-2., 3., None, 0.]
    assert M.predict(head, np.array([0, 0, 0]), [2, 3]) == [None, None, -5., 0.]


@pytest.mark.parametrize("defect", ["shape", "nan", "scale", "dtype", "extra", "input", "legal", "terminal"])
def test_inference_rejects_corrupted_head_or_input(defect):
    head = M.export_head(M.make_head(3, 1), np.zeros(3), np.ones(3))
    x, actions = np.zeros(3), [0, 1, 2, 3]
    if defect == "shape":
        head["weight0"] = head["weight0"][:, :2]
    elif defect == "nan":
        head["bias2"][0] = np.inf
    elif defect == "scale":
        head["scale"][0] = .5
    elif defect == "dtype":
        head["weight1"] = head["weight1"].astype(np.float64)
    elif defect == "extra":
        head["policy_oracle"] = True
    elif defect == "input":
        x[0] = np.nan
    elif defect == "legal":
        actions = [0, 0, 1]
    else:
        actions = []
    with pytest.raises(ValueError):
        M.predict(head, x, actions)


def test_export_rejects_changed_activation_dtype_and_scale():
    model = M.make_head(3, 1)
    changed = copy.deepcopy(model)
    changed[1] = torch.nn.ReLU()
    with pytest.raises(ValueError, match="Tanh"):
        M.export_head(changed, np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="float32"):
        M.export_head(model.double(), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="scale"):
        M.export_head(M.make_head(3, 1), np.zeros(3), np.zeros(3))


@pytest.mark.parametrize("dimension", [3080, 5633])
def test_frozen_head_exact_strict_predict_parity_and_storage(dimension):
    head = M.export_head(M.make_head(dimension, 83), np.zeros(dimension), np.ones(dimension))
    frozen = M.FrozenHead(head)
    x = np.linspace(-2, 2, dimension, dtype=np.float32)
    assert frozen.predict(x, (0, 2, 3)) == M.predict(head, x, (0, 2, 3))
    assert frozen.storage_bytes() == M.storage_bytes(head)


def test_frozen_head_owns_immutable_bytes_not_external_mutable_arrays():
    head = M.export_head(M.make_head(3, 73), np.zeros(3), np.ones(3))
    frozen = M.FrozenHead(head)
    before = frozen.predict([1, 2, 3], [0, 1, 2, 3])
    for original, array in zip((head[k] for k in (*M.WEIGHTS, "mean", "scale")), frozen._arrays, strict=True):
        assert not np.shares_memory(original, array) and not array.flags.writeable
        backing = array
        while isinstance(backing, np.ndarray):
            backing = backing.base
        assert isinstance(backing, bytes)
        with pytest.raises(ValueError):
            array.setflags(write=True)
        original[:] = np.nan
    head.clear()
    assert frozen.predict([1, 2, 3], [0, 1, 2, 3]) == before
    assert not hasattr(frozen, "__dict__")
    with pytest.raises(FrozenInstanceError):
        frozen._dimension = 4


def test_frozen_head_validates_once_and_keeps_per_input_guards(monkeypatch):
    head = M.export_head(M.make_head(3, 83), np.zeros(3), np.ones(3))
    original, calls = M.validate_head, []

    def validate_once(value):
        calls.append(value)
        assert len(calls) == 1, "weights were scanned after construction"
        return original(value)

    monkeypatch.setattr(M, "validate_head", validate_once)
    frozen = M.FrozenHead(head)
    for _ in range(3):
        assert len(frozen.predict([0, 1, 2], [0, 1, 2, 3])) == 4
        assert frozen.storage_bytes()["parameter_count"] == M.parameter_count(3)
    for x, actions in [([0, 1], [0]), ([0, np.nan, 2], [0]), ([0, 1, 2], []), ([0, 1, 2], [4])]:
        with pytest.raises(ValueError):
            frozen.predict(x, actions)
    assert len(calls) == 1


@pytest.mark.parametrize("defect", ["weight", "scale", "shape", "schema"])
def test_frozen_head_rejects_invalid_export_at_construction(defect):
    head = M.export_head(M.make_head(3, 83), np.zeros(3), np.ones(3))
    if defect == "weight":
        head["weight0"][0, 0] = np.inf
    elif defect == "scale":
        head["scale"][0] = .5
    elif defect == "shape":
        head["bias2"] = np.zeros(3, dtype=np.float32)
    else:
        head["unused"] = 1
    with pytest.raises(ValueError):
        M.FrozenHead(head)
