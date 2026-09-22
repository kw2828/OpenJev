import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openjev.research import otto_query_gate_models as gates


@pytest.mark.parametrize("kind,count,width", [("gru32", 6273, 32), ("mlp190", 6271, 0)])
def test_constant_initial_function_parameter_count_and_rng(kind, count, width):
    state = torch.random.get_rng_state().clone()
    model = gates.make_gate(kind, 40101)
    assert torch.equal(state, torch.random.get_rng_state())
    assert gates.parameter_count(kind) == sum(p.numel() for p in model.parameters()) == count
    frozen = gates.FrozenGate(gates.export_gate(model))
    assert frozen.state_size == width
    h = frozen.initial_state()
    for x in (np.zeros(31, np.float32), np.linspace(-3, 3, 31, dtype=np.float32)):
        query, h, logit, probability = frozen.step(x, h)
        assert query and logit == float(np.float32(math.log(19)))
        assert probability == pytest.approx(.95, abs=1e-7)


@pytest.mark.parametrize("kind", gates.KINDS)
def test_trained_sequence_export_parity_reset_and_defensive_ownership(kind, tmp_path):
    model = gates.make_gate(kind, 40102)
    x = torch.linspace(-2, 2, 2 * 7 * 31).reshape(2, 7, 31)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    logits, _ = model(x, torch.zeros(2, model.state_size))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    optimizer.step()
    export = gates.export_gate(model)
    path = tmp_path / "head.npz"
    np.savez(path, **export)
    with np.load(path, allow_pickle=False) as archive:
        frozen = gates.FrozenGate({name: archive[name] for name in archive.files})
    for name in gates.SHAPES[kind]:
        assert frozen.weights[name].tobytes() == export[name].tobytes()
        with pytest.raises(ValueError):
            frozen.weights[name].setflags(write=True)
    with pytest.raises(TypeError):
        frozen.weights["output.bias"] = np.zeros(1, np.float32)
    for episode in x:
        h = frozen.initial_state()
        th = torch.zeros(1, model.state_size)
        for row in episode:
            with torch.no_grad():
                logits, th = model(row.reshape(1, 1, 31), th)
            query, h, logit, probability = frozen.step(row.numpy(), h)
            assert abs(logit - logits.item()) <= gates.PARITY_ATOL
            np.testing.assert_allclose(h, th.numpy()[0], atol=gates.PARITY_ATOL, rtol=0)
            assert query == (float(torch.sigmoid(logits).item()) >= .05)
            assert probability == pytest.approx(float(torch.sigmoid(logits).item()), abs=gates.PARITY_ATOL)
    assert frozen.initial_state().tobytes() == np.zeros(model.state_size, np.float32).tobytes()


def test_gru_reset_after_algebra_and_scalar_threshold():
    export = {k: v.copy() for k, v in gates.export_gate(gates.make_gate("gru32", 40101)).items()}
    for name in gates.SHAPES["gru32"]:
        export[name].fill(0)
    export["recurrent.bias_ih_l0"][64:] = .2
    export["recurrent.bias_hh_l0"][64:] = .8
    export["output.weight"][0, 0] = 1
    frozen = gates.FrozenGate(export)
    query, state, logit, probability = frozen.step(np.zeros(31, np.float32), np.ones(32, np.float32))
    expected = .5 * math.tanh(.2 + .5 * .8) + .5
    np.testing.assert_allclose(state, expected, atol=1e-7, rtol=0)
    assert logit == state[0] and query and probability > .5
    export["output.weight"].fill(0)
    for bias in (-3.0, -2.9):
        export["output.bias"].fill(bias)
        current = gates.FrozenGate(export)
        q, _, l, p = current.step(np.zeros(31, np.float32), current.initial_state())
        assert q == (p >= .05) == (bias > -3)
        assert l == float(np.float32(bias))


@pytest.mark.parametrize("change", ["extra", "dtype", "nan", "seed", "threshold", "shape"])
def test_strict_export_refuses_corruption(change):
    value = copy.deepcopy(gates.export_gate(gates.make_gate("mlp190", 40103)))
    if change == "extra":
        value["untrusted"] = np.zeros(1)
    elif change == "dtype":
        value["output.bias"] = np.zeros(1, np.float64)
    elif change == "nan":
        value["output.bias"] = np.asarray([np.nan], np.float32)
    elif change == "seed":
        value["seed"] = np.asarray(True)
    elif change == "threshold":
        value["threshold"] = np.asarray(.5)
    else:
        value["hidden.bias"] = np.zeros(189, np.float32)
    with pytest.raises(ValueError):
        gates.FrozenGate(value)


def test_immutable_caller_state_and_strict_inputs():
    frozen = gates.FrozenGate(gates.export_gate(gates.make_gate("gru32", 40101)))
    x, h = np.ones(31, np.float32), np.ones(32, np.float32)
    xb, hb = x.tobytes(), h.tobytes()
    _, new, _, _ = frozen.step(x, h)
    assert x.tobytes() == xb and h.tobytes() == hb
    with pytest.raises(ValueError):
        new.setflags(write=True)
    for bad_x, bad_h in ((x.astype(np.float64), h), (x, h[:31]), (x * np.inf, h)):
        with pytest.raises(ValueError):
            frozen.step(bad_x, bad_h)
