"""Engineering parity and cache-isolation tests, not model-quality evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research.shared_prefix import (
    METHODS,
    SharedPrefixExperiment,
    common_prefix_length,
    right_pad,
)


def test_common_prefix_is_token_based_and_reserves_a_suffix():
    assert common_prefix_length([[1, 2, 3], [1, 2, 4, 5]]) == 2
    assert common_prefix_length([[1, 2, 3]]) == 2
    assert common_prefix_length([[1], [1, 2]]) == 0
    assert common_prefix_length([[1, 2], [3, 2]]) == 0
    for invalid in ([], [[]], [[1], []]):
        with pytest.raises(ValueError):
            common_prefix_length(invalid)


def test_padding_retains_real_token_positions():
    rows, lengths = right_pad([[1, 2, 3], [4]], pad_id=17)
    np.testing.assert_array_equal(rows, [[1, 2, 3], [4, 17, 17]])
    np.testing.assert_array_equal(lengths, [3, 1])


@pytest.fixture
def experiment():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.qwen3 import Model, ModelArgs
    old_device = mx.default_device()
    # Strict mathematical tests use CPU float32. The separate measured pilot
    # tests the pinned quantized model on GPU without relaxing its parity gate.
    mx.set_default_device(mx.cpu)
    mx.random.seed(719)
    model = Model(ModelArgs(
        model_type="qwen3", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=64, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=1000000., head_dim=8, tie_word_embeddings=False,
    ))
    model.eval()
    mx.eval(model.parameters())
    try:
        yield SharedPrefixExperiment(SimpleNamespace(mx=mx, model=model))
    finally:
        mx.set_default_device(old_device)


@pytest.mark.parametrize("rows", [
    [[1, 2, 3, 4]],
    [[1, 2, 3, 4], [1, 2, 5]],
    [[1, 2, 3, 4], [1, 2, 5], [1, 2, 3, 8, 9, 10], [1, 2, 17]],
    [[1, 2], [2, 3, 4]],
    [[1], [1, 2, 3]],
    [[1, 2, 3], [1, 2, 3]],
])
def test_all_methods_match_causal_reference(experiment, rows):
    expected, _ = experiment.logits(rows, "serial")
    for method in METHODS:
        actual, work = experiment.logits(rows, method)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        assert work.calls == work.prefill_calls + work.branch_calls


def test_cache_branches_requests_and_order_are_isolated(experiment):
    rows = [[1, 2, 3, 4], [1, 2, 5, 6, 7, 8]]
    expected, _ = experiment.logits(rows, "serial")
    for method in ("shared_serial", "shared_batch"):
        actual, _ = experiment.logits(rows, method)
        experiment.logits([[13, 14, 15], [13, 16]], method)
        reversed_out, _ = experiment.logits(rows[::-1], method)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(reversed_out[::-1], expected, rtol=3e-5, atol=3e-5)


def test_work_counts_include_prefix_padding_and_each_branch(experiment):
    rows = [[1, 2, 3, 4], [1, 2, 5]]
    expected = {"serial": (2, 7, 0), "batch": (1, 8, 1),
                "shared_serial": (3, 5, 0), "shared_batch": (2, 6, 1)}
    for method, (calls, slots, padding) in expected.items():
        _, work = experiment.logits(rows, method)
        assert (work.calls, work.input_token_slots, work.padding_token_slots) == (calls, slots, padding)


def test_unknown_method_fails_before_model_access():
    experiment = SharedPrefixExperiment(SimpleNamespace(mx=None, model=None))
    with pytest.raises(ValueError, match="Unknown method"):
        experiment.logits([[1]], "unrecognized")
