"""Synthetic only, isolated engineering seed410; no datasets or native calls."""
import importlib
import json
from pathlib import Path

import pytest
import torch

from openjev.research import pose_probe_blend as module
from openjev.research.pose_coordination import blend
from openjev.research.pose_support import PoseSupport
from openjev.research.pose_transport import PoseTransport
from openjev.research.rigid_motion import geodesic_angle, so3_exp, so3_log


@pytest.fixture(autouse=True)
def engineering_scope():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield
    torch.set_num_threads(threads)


def fixture(dtype=torch.float32, batch=2):
    scales = torch.tensor([.1, .05, .01, .005], dtype=dtype)
    fast = PoseSupport(scales)
    slow = PoseTransport("body", scales)
    with torch.no_grad():
        fast.prior.copy_(torch.randn_like(fast.prior) * .002)
        slow.readout.weight.copy_(torch.randn_like(slow.readout.weight) * .02)
        slow.readout.bias.copy_(torch.tensor([.1, -.05, .02, -.04, .08, .05], dtype=dtype))
    p = torch.randn(batch, 32, 3, dtype=dtype).cumsum(1) * .003
    r = so3_exp(torch.randn(batch, 32, 3, dtype=dtype) * .003)
    past = torch.randn(batch, 31, 40, dtype=dtype) * .1
    future = torch.randn(batch, 25, 40, dtype=dtype) * .1
    fallback = torch.tensor([.25, .75], dtype=dtype)
    return fast, slow, p, r, past, future, fallback


def execute(values, variant, **kwargs):
    fast, slow, p, r, past, future, fallback = values
    return module.forecast(fast, slow, variant, p, r, past, future,
                           fallback_alpha=fallback, **kwargs)


@pytest.mark.parametrize("dtype,batch", [(torch.float32, 2), (torch.float64, 2), (torch.float32, 160)])
def test_full_original_expert_and_half_forecasts_are_bitwise_equal(dtype, batch, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    original = importlib.import_module("train_pose_adaptation")
    values = fixture(dtype, batch)
    fast, slow, p, r, past, future, _ = values
    with torch.no_grad():
        fp, fr = fast.rollout(fast.fit_context(p, r, past, "decay_huber3"), future)
        sp, sr = original.forecast(slow, p, r, past, future, "gru")
        hp, hr = blend(fp, fr, sp, sr, p.new_full((batch, 2), .5))
        for variant, expected in (("fast", (fp, fr)), ("slow", (sp, sr)), ("half", (hp, hr)),
                                  ("probe_half", (hp, hr))):
            actual = execute(values, variant, return_probe=True)
            for x, y in zip(actual[:2], expected, strict=True):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            assert len(actual) == 5
            assert (actual[4] is None) == (not variant.startswith("probe_"))
    assert torch.count_nonzero(slow.readout.weight) and torch.count_nonzero(fast.prior)


def test_probe_forecasts_ignore_all_heldout_poses_but_real_root_does_not():
    fast, slow, p, r, past, _, _ = fixture()
    first = module.probe_context(fast, slow, p, r, past)
    changed_p, changed_r = p.clone(), r.clone()
    changed_p[:, 27:] += .7
    changed_r[:, 27:] = so3_exp(r.new_tensor([.2, -.1, .3]))
    second = module.probe_context(fast, slow, changed_p, changed_r, past)
    for key in ("fast_p", "fast_R", "slow_p", "slow_R"):
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    assert not torch.equal(first["slow_state"]["p"], second["slow_state"]["p"])
    assert not torch.equal(first["slow_state"]["h"], second["slow_state"]["h"])


@pytest.mark.parametrize("variant", ["probe_half", "probe_inverse", "probe_fit"])
def test_alpha_is_independent_of_future_actions(variant):
    values = fixture()
    first = execute(values, variant, return_probe=True)
    changed = (*values[:5], values[5] + 2, values[6])
    second = execute(changed, variant, return_probe=True)
    torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)
    for key in first[4]:
        torch.testing.assert_close(first[4][key], second[4][key], atol=0, rtol=0)
    assert not torch.equal(first[0], second[0])


def test_prefix_target_alignment_ages_and_three_dense_irls_solves(monkeypatch):
    fast, _, p, r, a, _, _ = fixture(torch.float64)
    calls = []
    original = fast._solve

    def capture(*values):
        calls.append(tuple(v.detach().clone() for v in values))
        return original(*values)

    monkeypatch.setattr(fast, "_solve", capture)
    fitted = module._prefix_fit(fast, p[:, :27], r[:, :27], a[:, :26])
    assert len(calls) == 3 and calls[0][0].shape == (2, 25, 13)
    x, y, prior, _weights = calls[0]
    for t in (1, 9, 25):
        dp0, dp1 = p[:, t] - p[:, t - 1], p[:, t + 1] - p[:, t]
        w0 = so3_log(r[:, t] @ r[:, t - 1].transpose(-1, -2))
        w1 = so3_log(r[:, t + 1] @ r[:, t].transpose(-1, -2))
        inv = r[:, t].transpose(-1, -2)
        expected = torch.cat(((inv @ (dp1 - dp0)[..., None])[..., 0] / fast.scales[2],
                              (inv @ (w1 - w0)[..., None])[..., 0] / fast.scales[3]), -1)
        torch.testing.assert_close(y[:, t - 1], expected, atol=2e-14, rtol=2e-14)
        torch.testing.assert_close(x[:, t - 1], fast._features(r[:, t], dp0, w0, a[:, t]),
                                   atol=2e-14, rtol=2e-14)
    posterior = prior[None].expand(2, -1, -1)
    ages = torch.arange(24, -1, -1, dtype=torch.float64)
    for _, _, _, actual_weights in calls:
        expected_weights = torch.exp2(-ages / 5)[None, :, None] * (1.5 / (y - x @ posterior).abs().clamp_min(1.5))
        torch.testing.assert_close(actual_weights, expected_weights, atol=2e-14, rtol=2e-14)
        rows = []
        for b in range(2):
            columns = []
            for o in range(6):
                wx = expected_weights[b, :, o, None] * x[b]
                columns.append(torch.linalg.solve(x[b].T @ wx + torch.eye(13),
                    x[b].T @ (expected_weights[b, :, o] * y[b, :, o]) + prior[:, o]))
            rows.append(torch.stack(columns, -1))
        posterior = torch.stack(rows)
    torch.testing.assert_close(fitted["weights"], posterior, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(fitted["p"], p[:, 26], atol=0, rtol=0)


def scoring_fixture(dtype=torch.float64, batch=2):
    sp = torch.zeros(batch, 5, 3, dtype=dtype)
    fp = sp.clone(); fp[..., 0] = 2
    sr = torch.eye(3, dtype=dtype).expand(batch, 5, 3, 3).clone()
    fr = so3_exp(torch.tensor([0., 0., .5], dtype=dtype)).expand_as(sr).clone()
    tp, tr = blend(fp, fr, sp, sr, torch.tensor([.25, .75], dtype=dtype).expand(batch, 2))
    fallback = torch.tensor([.125, .875], dtype=dtype)
    return fp, fr, sp, sr, tp, tr, fallback


def choose(values, variant="probe_fit"):
    return module.choose_alpha(*values[:6], variant=variant, fallback_alpha=values[6])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_known_analytic_coefficient_and_exact_grid_selection(dtype):
    values = scoring_fixture(dtype)
    alpha, info = choose(values)
    torch.testing.assert_close(alpha, alpha.new_tensor([[.25, .75], [.25, .75]]), atol=0, rtol=0)
    assert torch.equal(info["position_numerator"], torch.ones(2, dtype=torch.float64))
    assert torch.equal(info["position_denominator"], torch.full((2,), 4., dtype=torch.float64))
    assert torch.equal(info["grid_index"], torch.full((2,), 12))
    fp, fr, sp, sr, _, tr, _ = values
    for k in range(17):
        _, expected = blend(fp, fr, sp, sr, fp.new_full((2, 2), k / 16))
        torch.testing.assert_close(info["grid_rotations"][:, k], expected, atol=0, rtol=0)
        loss = geodesic_angle(expected.double(), tr.double()).square().mean(1)
        # Changing the float64 metric batch shape can change reduction by an ULP.
        torch.testing.assert_close(info["grid_mse"][:, k], loss, atol=1e-16, rtol=4e-15)
    torch.testing.assert_close(info["grid_index"], info["grid_mse"].argmin(1), atol=0, rtol=0)


@pytest.mark.parametrize("target", ["fast", "slow", "same"])
def test_grid_endpoints_and_exact_ties_use_smallest_alpha(target):
    values = list(scoring_fixture(torch.float32))
    if target == "same":
        values[1] = values[3].clone()
    values[5] = values[1 if target == "fast" else 3].clone()
    alpha, info = choose(values)
    expected = 16 if target == "fast" else 0
    assert info["grid_index"].eq(expected).all()
    assert alpha[:, 1].eq(expected / 16).all()


@pytest.mark.parametrize("offset,expected", [(-1., 0.), (3., 1.)])
def test_position_clip_outside_segment(offset, expected):
    values = list(scoring_fixture())
    values[4][..., 0] = offset
    assert choose(values)[0][:, 0].eq(expected).all()


def test_tiny_denominator_fallback_and_translation_invariance():
    values = list(scoring_fixture(torch.float32))
    values[0].zero_(); values[0][..., 0] = 1e-8
    values[4][..., 0] = 1.
    alpha, info = choose(values)
    assert info["position_fallback"].all() and alpha[:, 0].eq(values[6][0]).all()
    torch.testing.assert_close(info["position_threshold"],
        torch.finfo(torch.float32).eps ** 2 * info["errors"][:, :, 0].amax(1), rtol=0, atol=0)
    values[0][..., 0] = 1e-4
    assert not choose(values)[1]["position_fallback"].any()
    v64 = list(scoring_fixture())
    a, evidence = choose(v64)
    for index in (0, 2, 4):
        v64[index] += 1024
    b, shifted = choose(v64)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(evidence["position_threshold"], shifted["position_threshold"], atol=0, rtol=0)


def test_zero_denominator_and_inverse_both_zero_use_supplied_fallback():
    values = list(scoring_fixture())
    values[0] = values[2].clone(); values[4] = values[2].clone()
    values[1] = values[3].clone(); values[5] = values[3].clone()
    fit, info = choose(values)
    assert fit[:, 0].eq(values[6][0]).all() and fit[:, 1].eq(0).all()
    assert info["position_threshold"].eq(torch.finfo(torch.float64).tiny).all()
    inverse, info = choose(values, "probe_inverse")
    torch.testing.assert_close(inverse, values[6].expand(2, 2), atol=0, rtol=0)
    assert info["inverse_zero_fallback"].all() and "grid_mse" not in info


@pytest.mark.parametrize("direction,expected", [(-1, True), (0, True), (1, False)])
def test_position_threshold_is_inclusive_at_representation_precision(direction, expected):
    values = list(scoring_fixture(torch.float32))
    epsilon = torch.tensor(torch.finfo(torch.float32).eps)
    if direction:
        epsilon = torch.nextafter(epsilon, torch.tensor(0. if direction < 0 else torch.inf))
    values[0].zero_(); values[0][..., 0] = epsilon
    values[4].zero_(); values[4][..., 0] = 1
    _, info = choose(values)
    assert info["position_fallback"].eq(expected).all()


def test_real_gru_root_matches_context_without_the_private_probe():
    fast, slow, p, r, past, _, _ = fixture()
    probed = module.probe_context(fast, slow, p, r, past)
    state = slow.assimilate(slow.initial(p[:, 0], r[:, 0]), p[:, 0], r[:, 0])
    for t in range(31):
        state = slow.assimilate(slow.advance(state, past[:, t]), p[:, t + 1], r[:, t + 1])
    for key, value in state.items():
        torch.testing.assert_close(value, probed["slow_state"][key], rtol=0, atol=0)


def test_inverse_is_opposite_error_ratio_and_half_discards_fitted_coefficients():
    values = scoring_fixture()
    inverse, info = choose(values, "probe_inverse")
    torch.testing.assert_close(inverse, info["errors"][:, 1] / info["errors"].sum(1), atol=1e-16, rtol=2e-15)
    fit, fitted = choose(values)
    half, ignored = choose(values, "probe_half")
    assert half.eq(.5).all()
    torch.testing.assert_close(ignored["fitted_alpha"], fit, atol=0, rtol=0)
    for key in fitted.keys() - {"alpha"}:
        torch.testing.assert_close(fitted[key], ignored[key], atol=0, rtol=0)


def test_inverse_rescaling_handles_large_finite_and_subnormal_errors():
    errors = torch.tensor([[[1e308, 1e308], [1e308, 5e307]],
                           [[1e-320, 0.], [1e-320, 0.]]], dtype=torch.float64)
    fallback = torch.tensor([[.125, .875], [.25, .75]], dtype=torch.float64)
    alpha, zero = module._inverse_alpha(errors, fallback)
    torch.testing.assert_close(alpha, torch.tensor([[.5, 1 / 3], [.5, .75]], dtype=torch.float64),
                               atol=0, rtol=0)
    assert torch.equal(zero, torch.tensor([[False, False], [False, True]]))
    assert torch.isfinite(alpha).all()


@pytest.mark.parametrize("variant", module.VARIANTS)
def test_successful_work_counts_match_actual_module_hooks(variant, monkeypatch):
    values = fixture()
    fast, slow = values[:2]
    counts = {"features": 0, "observation": 0, "transition": 0, "readout": 0, "solves": 0, "factors": 0}
    handles = []
    for layer, key in ((fast.encoder, "features"), (slow.observation_update, "observation"),
                       (slow.transition, "transition"), (slow.readout, "readout")):
        def hook(_layer, args, _result, key=key):
            counts[key] += args[0].numel() // args[0].shape[-1]
        handles.append(layer.register_forward_hook(hook))
    original = torch.linalg.cholesky

    def counted(matrix):
        counts["solves"] += 1
        counts["factors"] += matrix.numel() // (matrix.shape[-1] ** 2)
        return original(matrix)

    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    try:
        _, _, _, diag = execute(values, variant)
    finally:
        for handle in handles:
            handle.remove()
    work = diag["work"]
    assert counts["features"] == work["fast_feature_samples"]
    assert counts["observation"] == work["slow_observation_samples"]
    assert counts["transition"] == counts["readout"] == sum(work[key] for key in
        ("slow_context_advance_samples", "slow_probe_advance_samples", "slow_forecast_advance_samples"))
    assert counts["solves"] == work["fast_batched_solve_calls"]
    assert counts["factors"] == work["fast_cholesky_factorizations"]
    assert work["grid_rotation_samples"] == (170 if variant in ("probe_fit", "probe_half") else 0)
    json.dumps(diag, allow_nan=False)


def test_callback_precedes_scoring_and_cannot_mutate_probe_predictions(monkeypatch):
    values = fixture()
    expected = execute(values, "probe_fit", return_probe=True)
    original = module.choose_alpha
    order = []

    def callback(arrays):
        assert set(arrays) == {"fast_p", "fast_R", "slow_p", "slow_R"}
        order.append("saved")
        for key, value in arrays.items():
            torch.testing.assert_close(value, expected[4][key], atol=0, rtol=0)
            value.zero_()

    def checked(*args, **kwargs):
        assert order == ["saved"]
        order.append("scored")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "choose_alpha", checked)
    actual = execute(values, "probe_fit", return_probe=True, on_probe=callback)
    assert order == ["saved", "scored"]
    for x, y in zip(actual[:3], expected[:3], strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert actual[3]["work"]["probe_callback_calls"] == 1


def test_callback_error_is_original_and_prevents_scoring(monkeypatch):
    error = OSError("synthetic evidence write failed")

    def callback(_arrays):
        raise error

    def forbidden(*args, **kwargs):
        raise AssertionError("scoring occurred after failed evidence write")

    monkeypatch.setattr(module, "choose_alpha", forbidden)
    with pytest.raises(OSError) as caught:
        execute(fixture(), "probe_fit", on_probe=callback)
    assert caught.value is error


def test_inputs_parameters_and_private_root_survive_other_rollouts():
    values = fixture(torch.float64)
    fast, slow, p, r, past, future, fallback = values
    inputs = [v.clone() for v in values[2:]]
    weights = [{k: v.clone() for k, v in model.state_dict().items()} for model in (fast, slow)]
    result = module.probe_context(fast, slow, p, r, past)
    root = {k: v.clone() for k, v in result["slow_state"].items()}
    module._slow_rollout(slow, result["slow_state"], future)
    execute(values, "probe_fit")
    for key, value in root.items():
        torch.testing.assert_close(value, result["slow_state"][key], atol=0, rtol=0)
    for v, before in zip((p, r, past, future, fallback), inputs, strict=True):
        torch.testing.assert_close(v, before, atol=0, rtol=0)
    for model, before in zip((fast, slow), weights, strict=True):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)


@pytest.mark.parametrize("mutation", ["future_pose", "short_past", "short_future", "nan_context", "nan_future",
                                      "dtype", "fallback_nan", "fallback_range", "fallback_shape", "variant"])
def test_invalid_inputs_reject_before_any_model_work(mutation, monkeypatch):
    values = list(fixture())
    if mutation == "future_pose":
        values[2] = torch.cat((values[2], values[2][:, :1]), 1)
    elif mutation == "short_past":
        values[4] = values[4][:, :-1]
    elif mutation == "short_future":
        values[5] = values[5][:, :-1]
    elif mutation == "nan_context":
        values[2][0, 30, 0] = torch.nan
    elif mutation == "nan_future":
        values[5][0, 0, 0] = torch.inf
    elif mutation == "dtype":
        values[3] = values[3].double()
    elif mutation == "fallback_nan":
        values[6][0] = torch.nan
    elif mutation == "fallback_range":
        values[6][0] = 1.1
    elif mutation == "fallback_shape":
        values[6] = values[6][:1]

    def forbidden(*args, **kwargs):
        raise AssertionError("model work before validation")

    monkeypatch.setattr(values[0], "_features", forbidden)
    monkeypatch.setattr(values[1], "initial", forbidden)
    with pytest.raises(ValueError):
        execute(values, "unknown" if mutation == "variant" else "probe_fit")
