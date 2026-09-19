"""Tiny synthetic support-weighting checks, isolated engineering seed410 only."""
import json

import pytest
import torch

from openjev.research.pose_adaptation import PoseAdaptation
from openjev.research.pose_support import VARIANTS, PoseSupport
from openjev.research.rigid_motion import so3_exp, so3_log


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(dtype=torch.float64, batch=2):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        model = PoseSupport(torch.tensor([.1, .05, .02, .01], dtype=dtype))
        p = torch.randn(batch, 32, 3, dtype=dtype).cumsum(1) * .001
        rotation = so3_exp(torch.randn(batch, 32, 3, dtype=dtype) * .002)
        actions = torch.randn(batch, 31, 40, dtype=dtype) * .2
        future = torch.randn(batch, 3, 40, dtype=dtype) * .2
    with torch.no_grad():
        model.prior.copy_(torch.arange(78, dtype=dtype).reshape(13, 6) * .0001)
    return model, p, rotation, actions, future


def dense(x, y, prior, weights):
    """Independent scalar-output normal equations with unpenalized data RHS."""
    rows = []
    for b in range(len(x)):
        cols = []
        for o in range(6):
            w = torch.diag(weights[b, :, o])
            cols.append(torch.linalg.solve(x[b].T @ w @ x[b] + torch.eye(x.shape[-1]),
                                            x[b].T @ w @ y[b, :, o] + prior[:, o]))
        rows.append(torch.stack(cols, -1))
    return torch.stack(rows)


@pytest.mark.parametrize("variant", ["prior", "full"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_original_checkpoint_schema_and_exact_delegation(variant, dtype):
    model, p, r, a, future = fixture(dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        original = PoseAdaptation("learned", model.scales)
    original.load_state_dict(model.state_dict(), strict=True)
    assert list(model.state_dict()) == list(original.state_dict())
    assert sum(x.numel() for x in model.parameters()) == 3066
    expected = original.fit_context(p, r, a, adapt=variant == "full")
    actual, diag = model.fit_context(p, r, a, variant, return_diagnostics=True)
    for k in actual:
        torch.testing.assert_close(actual[k], expected[k], atol=0, rtol=0)
    for x, y in zip(model.rollout(actual, future), original.rollout(expected, future), strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert diag["batched_solve_calls"] == int(variant == "full")
    assert diag["cholesky_factorizations"] == len(p) * int(variant == "full")
    for field in ("weight_sum", "effective_n"):
        assert torch.tensor(diag["actual"][field]).eq(30 if variant == "full" else 0).all()
    assert torch.tensor(diag["actual"]["fraction_weight_lt_one"]).eq(0).all()


@pytest.mark.parametrize("variant", VARIANTS[2:])
def test_weighted_fit_matches_independent_dense_equations_and_counts(variant, monkeypatch):
    model, p, r, a, _ = fixture()
    phi, y, _, _ = model._context_design(p, r, a)
    calls = []
    original = torch.linalg.cholesky

    def counted(matrix):
        calls.append(tuple(matrix.shape))
        return original(matrix)

    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    state, diag = model.fit_context(p, r, a, variant, return_diagnostics=True)
    expected_calls = (3 if "huber" in variant else 1) + int(variant.endswith("_mass"))
    assert calls == [(2, 6, 13, 13)] * expected_calls
    assert diag["batched_solve_calls"] == expected_calls
    assert diag["cholesky_factorizations"] == 12 * expected_calls
    # Derive weights independently with scalar normal-equation solves.
    age = torch.arange(29, -1, -1, dtype=torch.float64)
    base = variant.removesuffix("_mass")
    base_w = (age < 5).double() if base == "recent5" else torch.exp2(-age / 5) if "decay" in base else torch.ones(30)
    base_w = base_w[None, :, None].expand(2, -1, 6)
    expected = model.prior.unsqueeze(0).expand(2, -1, -1)
    for _ in range(3 if "huber" in base else 1):
        weights = base_w
        if "huber" in base:
            residual = (y - phi @ expected).abs()
            weights = base_w * torch.where(residual <= 1.5, torch.ones_like(residual), 1.5 / residual)
        expected = dense(phi, y, model.prior, weights)
    if variant.endswith("_mass"):
        weights = weights.mean(1, keepdim=True).expand(-1, 30, -1)
        expected = dense(phi, y, model.prior, weights)
    torch.testing.assert_close(state["weights"], expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(torch.tensor(diag["actual"]["weight_sum"]), weights.sum(1).float())
    json.dumps(diag, allow_nan=False)
    assert "support_weights" not in diag


@pytest.mark.parametrize("base", ["recent5", "decay5", "huber3", "decay_huber3"])
def test_mass_has_identical_derivation_but_uniform_actual_weights(base):
    model, p, r, a, _ = fixture()
    phi, y, _, _ = model._context_design(p, r, a)
    y = y.clone(); y[0, 0, 0] += 100
    fitted, bw, _, _, _ = model._weighted_fit(phi, y, base)
    mass, actual, derived, discarded, _ = model._weighted_fit(phi, y, base + "_mass")
    torch.testing.assert_close(derived, bw, rtol=0, atol=0)
    torch.testing.assert_close(discarded, fitted, rtol=0, atol=0)
    torch.testing.assert_close(actual.sum(1), derived.sum(1), rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(actual, derived.mean(1, keepdim=True).expand_as(actual), rtol=0, atol=0)
    assert not torch.equal(mass, fitted)
    _, diag = model.fit_context(p, r, a, base + "_mass", return_diagnostics=True)
    torch.testing.assert_close(torch.tensor(diag["actual"]["effective_n"]), torch.full((2, 6), 30.))
    if base == "recent5":
        assert diag["derivation"]["weight_sum"] == [[5.] * 6] * 2
        assert diag["derivation"]["effective_n"] == [[5.] * 6] * 2
        assert diag["actual"]["fraction_weight_lt_one"] == [[1.] * 6] * 2


@pytest.mark.parametrize("variant", ["recent5", "decay_huber3_mass"])
def test_float32_posterior_cast_only_after_float64_solves(variant, monkeypatch):
    model, p, r, a, _ = fixture(torch.float32)
    solve = model._solve
    dtypes = []

    def checked(*values):
        dtypes.append([v.dtype for v in values])
        return solve(*values)

    monkeypatch.setattr(model, "_solve", checked)
    state, diag = model.fit_context(p, r, a, variant, return_diagnostics=True)
    assert state["weights"].dtype == torch.float32
    assert all(row == [torch.float64] * 4 for row in dtypes)
    base, base_diag = model.fit_context(p, r, a, variant.removesuffix("_mass"), return_diagnostics=True)
    assert diag["derivation_correction_frobenius_norm"] == base_diag["correction_frobenius_norm"]
    assert base["weights"].dtype == torch.float32


def test_recent_indices_and_half_life_are_not_reversed():
    model, p, r, a, _ = fixture()
    phi, y, _, _ = model._context_design(p, r, a)
    _, recent, *_ = model._weighted_fit(phi, y, "recent5")
    assert torch.count_nonzero(recent[:, :25]) == 0
    assert recent[:, 25:].eq(1).all()
    _, decay, *_ = model._weighted_fit(phi, y, "decay5")
    assert decay[:, -1].eq(1).all() and decay[:, -6].eq(.5).all()
    assert (decay[:, 1:] > decay[:, :-1]).all()


def test_huber_per_output_bounds_large_innovation_and_keeps_third_weights():
    model, *_ = fixture(batch=1)
    with torch.no_grad():
        model.prior.zero_()
    x = torch.zeros(1, 30, 13, dtype=torch.float64); x[..., -1] = 1
    y = torch.zeros(1, 30, 6, dtype=torch.float64); y[0, 0, 0] = 1000
    actual, weights, _, _, calls = model._weighted_fit(x, y, "huber3")
    assert calls == 3 and 0 < actual[0, -1, 0] < .1
    assert weights[0, 0, 0] < .002 and weights[..., 1:].eq(1).all()
    assert actual[0, :, 1:].eq(0).all()
    old = model.prior.unsqueeze(0)
    for _ in range(3):
        residual = (y - x @ old).abs()
        last = 1.5 / residual.clamp_min(1.5)
        old = dense(x, y, model.prior, last)
    torch.testing.assert_close(weights, last, rtol=1e-14, atol=1e-14)
    fourth = 1.5 / (y - x @ old).abs().clamp_min(1.5)
    assert not torch.equal(weights, fourth)


def test_support_targets_keep_current_body_frame_and_completed_action_alignment(monkeypatch):
    model, p, r, a, _ = fixture()
    calls = []
    design = model._context_design

    def capture(*args):
        values = design(*args)
        calls.append(values)
        return values

    monkeypatch.setattr(model, "_context_design", capture)
    state = model.fit_context(p, r, a, "recent5")
    assert len(calls) == 1
    phi, y, dp, w = calls[0]
    for t in (1, 26, 30):
        back = p[:, t] - p[:, t - 1]; ahead = p[:, t + 1] - p[:, t]
        old_w = so3_log(r[:, t] @ r[:, t - 1].transpose(-1, -2))
        next_w = so3_log(r[:, t + 1] @ r[:, t].transpose(-1, -2))
        inv = r[:, t].transpose(-1, -2)
        expected = torch.cat(((inv @ (ahead - back)[..., None])[..., 0] / model.scales[2],
                              (inv @ (next_w - old_w)[..., None])[..., 0] / model.scales[3]), -1)
        torch.testing.assert_close(y[:, t - 1], expected)
        torch.testing.assert_close(phi[:, t - 1], model._features(r[:, t], back, old_w, a[:, t]))
    torch.testing.assert_close(state["dp"], dp)
    torch.testing.assert_close(state["w"], w)
    altered = a.clone(); altered[:, 0] = 100
    torch.testing.assert_close(model.fit_context(p, r, altered, "recent5")["weights"], state["weights"], rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_reset_batch_isolation_owned_state_and_private_rollout(variant):
    model, p, r, a, future = fixture()
    inputs = [x.clone() for x in (p, r, a, future)]
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    state = model.fit_context(p, r, a, variant)
    before = {k: v.clone() for k, v in state.items()}
    predicted = model.rollout(state, future)
    single = model.fit_context(p[:1], r[:1], a[:1], variant)
    for k in state:
        torch.testing.assert_close(single[k], state[k][:1], rtol=2e-12, atol=2e-12)
    altered = future.clone(); altered[:, 2] += 7
    branched = model.rollout(state, altered)
    for x, y in zip(predicted, branched, strict=True):
        torch.testing.assert_close(x[:, :2], y[:, :2], rtol=0, atol=0)
    for k in state:
        torch.testing.assert_close(state[k], before[k], rtol=0, atol=0)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, weights[k], rtol=0, atol=0)
    for x, y in zip((p, r, a, future), inputs, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    with torch.no_grad():
        state["p"].add_(99); state["R"].zero_(); state["weights"].zero_()
    again = model.fit_context(p, r, a, variant)
    for k in again:
        torch.testing.assert_close(again[k], before[k], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["recent5", "huber3", "decay_huber3_mass"])
def test_gradients_remain_attached_without_updating_parameters(variant):
    model, p, r, a, future = fixture()
    p.requires_grad_(); a.requires_grad_()
    state = model.fit_context(p, r, a, variant)
    pp, rr = model.rollout(state, future)
    (pp.square().mean() + rr.square().mean()).backward()
    for g in (p.grad, a.grad, model.prior.grad, model.encoder[0].weight.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


@pytest.mark.parametrize("bad", ["future_pose", "future_action", "nan", "dtype", "variant", "flag"])
def test_rejects_future_or_malformed_context(bad):
    model, p, r, a, _ = fixture()
    with pytest.raises(ValueError):
        if bad == "future_pose":
            model.fit_context(torch.cat((p, p[:, -1:]), 1), r, a)
        elif bad == "future_action":
            model.fit_context(p, r, torch.cat((a, a[:, -1:]), 1))
        elif bad == "nan":
            a[0, 0, 0] = float("nan"); model.fit_context(p, r, a)
        elif bad == "dtype":
            model.fit_context(p.float(), r, a)
        elif bad == "variant":
            model.fit_context(p, r, a, "unknown")
        else:
            model.fit_context(p, r, a, return_diagnostics=1)
