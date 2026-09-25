"""Fabricated VARX qualification only; no measurement reader or file access."""
import inspect
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from openjev.research import fsm_linear as linear
from openjev.research.fsm_data import FSMRecord


def records(length=80):
    rng = np.random.default_rng(81104)
    return tuple(FSMRecord(rng.normal(size=(length, 3)), rng.normal(size=(length, 3)),
                           amplitude, realization, period, "fit",
                           f"{amplitude}-realization-{realization}-period-{period}")
                 for amplitude in ("100mV", "200mV")
                 for realization in (0, 1, 2) for period in (0, 1))


def scalar_design(items, order):
    rows, targets = [], []
    for record in items:
        for t in range(order, len(record.u)):
            row = [record.y[s, c] for s in range(t-order, t) for c in range(3)]
            row += [record.u[t, c] for c in range(3)]
            row += [record.u[s, c] for s in range(t-order, t) for c in range(3)]
            rows.append([*row, 1.])
            targets.append(record.y[t])
    return np.array(rows), np.array(targets)


def model_with(coefficients, order=1):
    return linear.VARXModel(coefficients, order, .001, 12, linear.FIT_IDS)


def test_ridge_against_independent_centered_augmented_least_squares():
    items, order, alpha = records(24), 2, .13
    design, target = scalar_design(items, order)
    features = design[:, :-1]
    xmean, ymean = features.mean(axis=0), target.mean(axis=0)
    # Centering eliminates the unpenalized intercept. Augmented least squares
    # avoids the production normal-equation/Cholesky implementation.
    augmented_x = np.vstack((features-xmean, np.sqrt(len(features)*alpha)*np.eye(features.shape[1])))
    augmented_y = np.vstack((target-ymean, np.zeros((features.shape[1], 3))))
    slopes = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    expected = np.vstack((slopes, ymean-xmean @ slopes)).T
    fitted = linear.fit_varx(items, order=order, alpha=alpha)
    np.testing.assert_allclose(fitted.coefficients, expected, rtol=1e-11, atol=1e-12)
    assert fitted.fit_rows == 12*(24-order)
    assert fitted.fit_record_ids == linear.FIT_IDS


def test_intercept_unpenalized_and_mean_gram_invariant_to_duplicate_period_rows():
    items = tuple(replace(record, u=np.zeros_like(record.u), y=np.full_like(record.y, 7.))
                  for record in records(12))
    small = linear.fit_varx(items, order=1, alpha=.1)
    larger = linear.fit_varx(tuple(replace(r, u=np.zeros((23, 3)), y=np.full((23, 3), 7.)) for r in items),
                             order=1, alpha=.1)
    expected = np.zeros((3, 10))
    expected[:, -1] = 7.
    np.testing.assert_allclose(small.coefficients, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(larger.coefficients, expected, rtol=1e-10, atol=1e-10)


def test_known_coupled_lti_recovery_and_free_run():
    a = np.array([[.35, .07, 0.], [-.04, .3, .02], [.01, -.03, .2]])
    b = np.array([[.8, .1, 0.], [0., .6, -.1], [.2, 0., .7]])
    e = np.diag([.1, -.08, .05])
    bias = np.array([.2, -.1, .05])
    generated = []
    for record in records(128):
        y = np.zeros_like(record.y)
        y[0] = record.y[0]
        for t in range(1, len(y)):
            y[t] = a @ y[t-1] + b @ record.u[t] + e @ record.u[t-1] + bias
        generated.append(replace(record, y=y))
    fitted = linear.fit_varx(generated, order=1, alpha=1e-10)
    expected = np.column_stack((a, b, e, bias))
    np.testing.assert_allclose(fitted.coefficients, expected, rtol=1e-7, atol=1e-7)
    r = generated[0]
    result = linear.predict(fitted, r.y[None, :100], r.u[None, 1:100], r.u[None, 100:128])
    np.testing.assert_allclose(result[0], r.y[100:128], rtol=1e-7, atol=1e-7)


def test_current_input_alignment_and_recursive_scalar_oracle():
    coefficients = np.arange(3*16, dtype=np.float64).reshape(3, 16)/100
    model = model_with(coefficients, order=2)
    y = np.arange(2*5*3, dtype=np.float64).reshape(2, 5, 3)/10
    u = np.arange(2*4*3, dtype=np.float64).reshape(2, 4, 3)/20
    future = np.arange(2*7*3, dtype=np.float64).reshape(2, 7, 3)/30
    actual = linear.predict(model, y, u, future)
    expected = np.empty_like(actual)
    for batch in range(2):
        yh, uh = list(y[batch]), list(u[batch])
        for t in range(7):
            row = [v for entry in yh[-2:] for v in entry]
            row += list(future[batch, t])
            row += [v for entry in uh[-2:] for v in entry]
            row += [1.]
            current = np.array([sum(w*x for w, x in zip(weights, row, strict=True)) for weights in coefficients])
            expected[batch, t] = current
            yh.append(current)
            uh.append(future[batch, t])
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_future_suffix_causality_old_context_independence_and_chunking():
    coefficients = np.zeros((3, 16))
    coefficients[:, 3:6] = .4*np.eye(3)
    coefficients[:, 6:9] = 2*np.eye(3)
    coefficients[:, 12:15] = .1*np.eye(3)
    model = model_with(coefficients, order=2)
    rng = np.random.default_rng(815)
    y, u, future = rng.normal(size=(2, 100, 3)), rng.normal(size=(2, 99, 3)), rng.normal(size=(2, 10, 3))
    before = linear.predict(model, y, u, future)
    changed = future.copy()
    changed[:, 4:] += 30
    np.testing.assert_array_equal(linear.predict(model, y, u, changed)[:, :4], before[:, :4])
    assert not np.array_equal(linear.predict(model, y, u, changed)[:, 4:], before[:, 4:])
    y2, u2 = y.copy(), u.copy()
    y2[:, :-2] += 99
    u2[:, :-2] -= 99
    np.testing.assert_array_equal(linear.predict(model, y2, u2, future), before)
    first = linear.predict(model, y, u, future[:, :4])
    second = linear.predict(model, np.concatenate((y, first), axis=1),
                            np.concatenate((u, future[:, :4]), axis=1), future[:, 4:])
    np.testing.assert_array_equal(np.concatenate((first, second), axis=1), before)


@pytest.mark.parametrize("order,count", [(8, 156), (16, 300), (32, 588)])
@pytest.mark.parametrize("alpha", linear.ALPHAS)
def test_declared_nine_controls_fit_and_exact_storage(order, count, alpha):
    fitted = linear.fit_varx(records(40), order=order, alpha=alpha)
    spec = fitted.model_spec()
    assert fitted.parameter_count() == count
    assert spec["parameter_bytes"] == 8*count
    assert spec["retained_numeric_bytes"] == 8*count+24
    assert spec["state_scalars"] == 6*order
    assert spec["state_bytes_per_stream"] == 48*order
    assert spec["intercept_penalized"] is False
    assert spec["training_rows_retained"] is False
    assert spec["persistent_cache"] is False


def test_ownership_no_mutation_rng_and_canonical_record_order():
    items = records(20)
    snapshots = [(r.u.copy(), r.y.copy()) for r in items]
    global_rng = np.random.get_state()
    first = linear.fit_varx(items, order=2, alpha=.1)
    second = linear.fit_varx(reversed(items), order=2, alpha=.1)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    for r, (u, y) in zip(items, snapshots, strict=True):
        np.testing.assert_array_equal(r.u, u)
        np.testing.assert_array_equal(r.y, y)
    after = np.random.get_state()
    assert global_rng[0] == after[0] and global_rng[2:] == after[2:]
    np.testing.assert_array_equal(global_rng[1], after[1])
    copied = first.coefficients.copy()
    model = model_with(copied, order=2)
    copied[:] = 99
    np.testing.assert_array_equal(first.coefficients, model.coefficients)
    with pytest.raises(ValueError):
        model.coefficients.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        model.alpha = 3.
    y, u, future = items[0].y[None, :5], items[0].u[None, 1:5], items[0].u[None, 5:8]
    before = [v.copy() for v in (y, u, future)]
    out = linear.predict(model, y, u, future)
    assert all(not np.shares_memory(out, v) for v in (y, u, future, model.coefficients))
    out[:] = 99
    for actual, expected in zip((y, u, future), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert linear.predict(model, y, u, future[:, :0]).shape == (1, 0, 3)
    assert tuple(inspect.signature(linear.predict).parameters) == ("model", "y_context", "u_context", "future_u")


@pytest.mark.parametrize("kind", ["missing", "duplicate", "dev", "identity", "unequal", "dtype", "nan", "short", "frequency"])
def test_fit_roster_or_measurement_schema_failure(kind):
    items = list(records(12))
    if kind == "missing":
        items.pop()
    elif kind == "duplicate":
        items[-1] = items[0]
    elif kind == "dev":
        items[0] = replace(items[0], partition="dev")
    elif kind == "identity":
        items[0] = replace(items[0], realization=1)
    elif kind == "unequal":
        items[0] = replace(items[0], u=items[0].u[:-1], y=items[0].y[:-1])
    elif kind == "dtype":
        items[0] = replace(items[0], y=items[0].y.astype(np.float32))
    elif kind == "nan":
        items[0] = replace(items[0], y=np.full((12, 3), np.nan))
    elif kind == "short":
        items = [replace(r, u=r.u[:2], y=r.y[:2]) for r in items]
    else:
        items[0] = replace(items[0], fs_hz=6401.)
    with pytest.raises(ValueError):
        linear.fit_varx(items, order=2, alpha=.1)


@pytest.mark.parametrize("order", [0, 100, True, 1., "8"])
def test_invalid_order(order):
    with pytest.raises(ValueError, match="order"):
        linear.fit_varx(records(12), order=order, alpha=.1)


@pytest.mark.parametrize("alpha", [0, -1., np.inf, np.nan, True, ".1"])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        linear.fit_varx(records(12), order=2, alpha=alpha)


@pytest.mark.parametrize("kind", ["model", "empty", "short", "dtype", "u_shape", "u_nan", "future_shape", "future_inf"])
def test_predict_guards(kind):
    model = model_with(np.zeros((3, 10)))
    y, u, future = np.zeros((2, 5, 3)), np.zeros((2, 4, 3)), np.zeros((2, 7, 3))
    if kind == "model":
        model = {}
    elif kind == "empty":
        y, u, future = y[:0], u[:0], future[:0]
    elif kind == "short":
        y, u = y[:, :1], u[:, :0]
    elif kind == "dtype":
        y = y.astype(np.float32)
    elif kind == "u_shape":
        u = u[:, :-1]
    elif kind == "u_nan":
        u[0, 0, 0] = np.nan
    elif kind == "future_shape":
        future = future[:, :, :2]
    else:
        future[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        linear.predict(model, y, u, future)


def test_failures_are_not_repaired(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise np.linalg.LinAlgError("fabricated")

    with monkeypatch.context() as patch:
        patch.setattr(linear, "cho_factor", fail)
        with pytest.raises(ValueError, match="no jitter or alpha change"):
            linear.fit_varx(records(12), order=2, alpha=.1)
    assert calls == [1]
    huge = tuple(replace(r, y=np.full_like(r.y, 1e300)) for r in records(12))
    with pytest.raises(ValueError, match="sufficient statistics"):
        linear.fit_varx(huge, order=2, alpha=.1)
    coefficients = np.zeros((3, 10))
    coefficients[:, 3:6] = np.eye(3)*1e308
    model = model_with(coefficients)
    with pytest.raises(ValueError, match="nonfinite VARX prediction"):
        linear.predict(model, np.zeros((1, 3, 3)), np.zeros((1, 2, 3)), np.full((1, 2, 3), 10.))
