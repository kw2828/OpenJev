"""Fabricated analytic, causal, ownership and differentiation contracts only."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from openjev.research.otto_query_memory import MODES, Config, QueryMemory, detach_carry


def packet(lengths=(7, 4), dimension=3):
    batch, span = len(lengths), max(lengths)
    active = torch.arange(span)[None, :] < torch.tensor(lengths)[:, None]
    query = torch.zeros(batch, span, dtype=torch.bool)
    base = torch.zeros(batch, span, 4, dtype=torch.float32)
    scores = torch.full_like(base, float("nan"))
    shadow = torch.full_like(base, float("nan"))
    keys = torch.full((batch, span, dimension), float("nan"), dtype=torch.float32)
    for row, length in enumerate(lengths):
        for step in range(length):
            base[row, step] = torch.tensor([-.0, .25, -.5, 1.]) + step / 64
            if step in (0, 2, 5):
                query[row, step] = True
                value = torch.tensor([1., -1., .5, -.5]) * (row + 1) * (step + 1) * 64
                if step == 0:
                    value[0] = -.0
                base[row, step] = scores[row, step] = value
                if step:
                    shadow[row, step] = 0
            if step:
                keys[row, step] = torch.arange(dimension, dtype=torch.float32) + row + step + 1
    base[:, :, 0][~active] = -.0
    key_mask = active & (torch.arange(span)[None, :] > 0)
    return {"base_action": base, "shadow_prior": shadow, "keys": keys,
            "query_scores": scores, "query_mask": query, "prior_mask": query & key_mask,
            "key_mask": key_mask, "lengths": torch.tensor(lengths, dtype=torch.int64),
            "episode_ends": torch.ones(batch, dtype=torch.bool)}


def copied(values):
    return {name: value.clone() for name, value in values.items()}


def bits(value):
    return value.detach().contiguous().numpy().tobytes()


def piece(values, start, stop):
    result = {name: value[:, start:stop].clone() for name, value in values.items()
              if name not in ("lengths", "episode_ends")}
    result["lengths"] = (values["lengths"] - start).clamp(0, stop - start)
    result["episode_ends"] = values["episode_ends"] & (values["lengths"] <= stop)
    return result


@pytest.mark.parametrize("mode", MODES)
def test_first_query_has_no_error_write_and_returns_exact_signed_zero_bits(mode):
    values = packet((1,), 2)
    result = QueryMemory(Config(mode, 2))(**values)
    assert bits(result.action_prediction) == bits(values["query_scores"])
    assert not result.prewrite_correction.any()
    assert not result.carry.matrix.any() and not result.carry.trace.any() and not result.carry.last_error.any()
    assert result.work["query_steps"] == 1 and result.work["key_steps"] == 0
    assert result.work["innovation_calculations"] == result.work["matrix_writes"] == result.work["last_error_writes"] == 0
    assert result.carry.absolute_step.tolist() == [1] and result.carry.ended.tolist() == [True]


def test_normalized_delta_is_exact_local_loss_update_and_contracts_same_cue_error():
    cfg = Config("instant_delta", 2, step_size=.4)
    model = QueryMemory(cfg)
    carry = model.initial_carry(1)
    matrix = torch.tensor([[[.1, .2], [-.1, -.2], [.3, -.1], [-.3, .1]]])
    carry = replace(carry, matrix=matrix, absolute_step=torch.tensor([1]), has_query=torch.tensor([True]))
    target = torch.tensor([[[64., -64., 32., -32.]]])
    result = model(base_action=target, shadow_prior=torch.zeros_like(target), keys=torch.tensor([[[3., 4.]]]),
        query_scores=target, query_mask=torch.ones(1, 1, dtype=torch.bool), prior_mask=torch.ones(1, 1, dtype=torch.bool),
        key_mask=torch.ones(1, 1, dtype=torch.bool), lengths=torch.tensor([1]), episode_ends=torch.tensor([False]), carry=carry)
    z = torch.tensor([.6, .8], dtype=torch.float64)
    old = matrix[0].double()
    r = target[0, 0].double() / 64
    u = old @ z
    u -= u.mean()
    error = r - u
    expected = old + .4 * error[:, None] * z[None, :] / (1e-6 + z.square().sum())
    torch.testing.assert_close(result.carry.matrix[0].double(), expected, rtol=2e-6, atol=1e-7)
    prediction = result.carry.matrix[0].double() @ z
    prediction -= prediction.mean()
    factor = 1 - .4 * z.square().sum() / (1e-6 + z.square().sum())
    torch.testing.assert_close(r - prediction, factor * error, rtol=2e-6, atol=1e-7)
    assert (r - prediction).square().sum() < error.square().sum()
    assert result.work["matrix_reads"] == result.work["matrix_writes"] == 1
    assert bits(result.action_prediction) == bits(target)


@pytest.mark.parametrize("mode", MODES[1:])
def test_single_query_change_affects_only_that_query_and_the_future(mode):
    cfg = Config(mode, 3)
    values = packet((7,))
    modified = copied(values)
    modified["query_scores"][0, 2, 0] += 64
    modified["base_action"][0, 2] = modified["query_scores"][0, 2]
    before, after = (QueryMemory(cfg)(**p) for p in (values, modified))
    assert bits(before.action_prediction[:, :2]) == bits(after.action_prediction[:, :2])
    assert bits(before.prewrite_correction[:, :3]) == bits(after.prewrite_correction[:, :3])
    assert bits(after.action_prediction[:, 2]) == bits(modified["query_scores"][:, 2])
    assert not torch.equal(before.action_prediction[:, 3], after.action_prediction[:, 3])
    assert not torch.equal(before.prewrite_correction[:, 3], after.prewrite_correction[:, 3])
    late = copied(values)
    late["query_scores"][0, 5, 1] -= 128
    late["base_action"][0, 5] = late["query_scores"][0, 5]
    changed = QueryMemory(cfg)(**late)
    assert bits(before.action_prediction[:, :5]) == bits(changed.action_prediction[:, :5])
    assert bits(before.prewrite_correction[:, :6]) == bits(changed.prewrite_correction[:, :6])


@pytest.mark.parametrize("mode", MODES)
def test_inputs_padding_query_and_carry_ownership_are_preserved(mode):
    values = packet()
    model = QueryMemory(Config(mode, 3))
    carry = model.initial_carry(2)
    initial = {name: bits(value) for name, value in values.items()}
    state = {name: bits(getattr(carry, name)) for name in
             ("matrix", "trace", "last_error", "absolute_step", "has_query", "ended")}
    result = model(**values, carry=carry)
    assert all(bits(values[name]) == expected for name, expected in initial.items())
    assert all(bits(getattr(carry, name)) == expected for name, expected in state.items())
    active = torch.arange(7)[None, :] < values["lengths"][:, None]
    assert bits(result.action_prediction[~active]) == bits(values["base_action"][~active])
    assert not result.prewrite_correction[~values["key_mask"]].any()
    assert bits(result.action_prediction[values["query_mask"]]) == bits(values["query_scores"][values["query_mask"]])
    for name in state:
        assert getattr(result.carry, name).data_ptr() != getattr(carry, name).data_ptr()


@pytest.mark.parametrize("mode", MODES)
def test_no_write_from_zero_matches_baseline_bits_with_normal_reads_and_trace_costs(mode):
    values = packet()
    values["base_action"][0, 1, 0] = -.0
    values["shadow_prior"][:] = float("nan")  # Never consumed without a write.
    result = QueryMemory(Config(mode, 3))(**values, no_write=True)
    assert bits(result.action_prediction) == bits(values["base_action"])
    assert not result.prewrite_correction.any()
    assert result.work["innovation_calculations"] == result.work["matrix_writes"] == result.work["last_error_writes"] == 0
    assert not result.carry.matrix.any() and not result.carry.last_error.any()
    if mode.startswith("trace_") or mode == "instant_delta":
        assert result.work["matrix_reads"] == result.work["trace_updates"] == result.work["key_steps"]
        assert result.carry.trace.any()
    elif mode == "none":
        assert not any(result.work_units.values())


@pytest.mark.parametrize("mode", MODES)
def test_chunking_and_episode_reset_reproduce_whole_call_and_additive_work_counts(mode):
    values = packet()
    model = QueryMemory(Config(mode, 3, memory_decay=.93))
    whole = model(**values)
    first = model(**piece(values, 0, 3))
    second = model(**piece(values, 3, 7), carry=detach_carry(first.carry))
    assert bits(torch.cat((first.action_prediction, second.action_prediction), 1)) == bits(whole.action_prediction)
    assert bits(torch.cat((first.prewrite_correction, second.prewrite_correction), 1)) == bits(whole.prewrite_correction)
    for name in ("matrix", "trace", "last_error", "absolute_step", "has_query", "ended"):
        assert bits(getattr(second.carry, name)) == bits(getattr(whole.carry, name))
    assert all(first.work[name] + second.work[name] == value for name, value in whole.work.items())
    assert all(first.work_units[name] + second.work_units[name] == value for name, value in whole.work_units.items())
    reset = model(**values, carry=model.initial_carry(2))
    assert bits(reset.action_prediction) == bits(whole.action_prediction)
    with pytest.raises(ValueError, match="ended lanes"):
        model(**values, carry=detach_carry(whole.carry))


@pytest.mark.parametrize("mode", ("instant_delta", "trace_delta", "trace_additive", "trace_scrambled"))
def test_gradients_follow_keys_and_past_writes_but_never_base_shadow_targets_or_future(mode):
    values = packet((7,))
    for name in ("base_action", "shadow_prior", "query_scores", "keys"):
        values[name].requires_grad_(True)
    result = QueryMemory(Config(mode, 3))(**values)
    result.action_prediction[0, 3, 0].backward()
    assert all(values[name].grad is None for name in ("base_action", "shadow_prior", "query_scores"))
    gradient = values["keys"].grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient[0, 2].abs().sum() > 0 and gradient[0, 3].abs().sum() > 0
    assert not gradient[0, 0].any() and not gradient[0, 4:].any()
    assert result.carry.matrix.grad_fn is not None
    with pytest.raises(ValueError, match="explicitly detached"):
        QueryMemory(Config(mode, 3))(**piece(values, 3, 7), carry=result.carry)
    detached = detach_carry(result.carry)
    assert detached.matrix.grad_fn is None and not detached.matrix.requires_grad
    assert detached.matrix.data_ptr() != result.carry.matrix.data_ptr()


def test_zero_valued_write_and_read_keep_legitimate_key_gradient():
    values = packet((4,), 2)
    values["keys"][0, 2] = 0
    values["keys"][0, 3] = torch.tensor([1., 0.])
    values["keys"].requires_grad_(True)
    result = QueryMemory(Config("instant_delta", 2))(**values)
    assert not result.carry.matrix.any()
    assert torch.equal(result.action_prediction[0, 3], values["base_action"][0, 3])
    result.action_prediction[0, 3, 0].backward()
    assert values["keys"].grad[0, 2, 0] != 0
    assert torch.isfinite(values["keys"].grad).all()


@pytest.mark.parametrize("mode", ("none", "last_error"))
def test_unused_keys_are_poison_inert_and_none_has_no_parameters_or_memory_work(mode):
    values = packet()
    values["keys"][:] = float("nan")
    if mode == "none":
        values["shadow_prior"][:] = float("nan")
    model = QueryMemory(Config(mode, 3))
    result = model(**values)
    assert list(model.parameters()) == [] and model.state_dict() == {}
    if mode == "none":
        assert bits(result.action_prediction) == bits(values["base_action"])
        assert not any(result.work_units.values())
    else:
        assert result.work["key_normalizations"] == result.work["matrix_reads"] == result.work["trace_updates"] == 0
        assert result.work["last_error_writes"] == int(values["prior_mask"].sum())


def test_masked_poison_is_inert_but_consumed_poison_fails():
    values = packet()
    model = QueryMemory(Config("trace_delta", 3))
    result = model(**values)
    clean = copied(values)
    clean["keys"][~clean["key_mask"]] = 3
    clean["shadow_prior"][~clean["prior_mask"]] = -900
    clean["query_scores"][~clean["query_mask"]] = 700
    assert bits(model(**clean).action_prediction) == bits(result.action_prediction)
    bad = copied(values)
    bad["keys"][0, 1] = float("nan")
    with pytest.raises(ValueError, match="consumed keys"):
        model(**bad)
    bad = copied(values)
    bad["shadow_prior"][0, 2] = float("nan")
    with pytest.raises(ValueError, match="consumed shadow"):
        model(**bad)
    bad = copied(values)
    bad["base_action"][1, 6] = float("nan")
    with pytest.raises(ValueError, match="zero base padding"):
        model(**bad)


@pytest.mark.parametrize("dimension", (2, 3, 5))
def test_scrambled_past_rotation_is_causal_meaningful_and_keeps_chronological_trace(dimension):
    values = packet((7,), dimension)
    values["keys"][values["key_mask"]] = torch.tensor([1.] + [0.] * (dimension - 1))
    normal = QueryMemory(Config("trace_delta", dimension))(**values)
    scrambled = QueryMemory(Config("trace_scrambled", dimension))(**values)
    assert torch.equal(normal.carry.trace, scrambled.carry.trace)
    assert not torch.equal(normal.action_prediction[:, 3], scrambled.action_prediction[:, 3])
    assert scrambled.work["past_trace_rotations"] == scrambled.work["key_steps"]
    assert normal.work["matrix_writes"] == scrambled.work["matrix_writes"]
    altered = copied(values)
    altered["keys"][0, 5:] = torch.arange(dimension, dtype=torch.float32) + 4
    later = QueryMemory(Config("trace_scrambled", dimension))(**altered)
    assert bits(later.action_prediction[:, :5]) == bits(scrambled.action_prediction[:, :5])


def test_delta_subtracts_current_read_while_additive_accumulates_target_again():
    results = {}
    for mode in ("trace_delta", "trace_additive"):
        model = QueryMemory(Config(mode, 2, step_size=.5, trace_decay=0))
        carry = replace(model.initial_carry(1), absolute_step=torch.tensor([1]), has_query=torch.tensor([True]))
        for _ in range(4):
            score = torch.tensor([[[64., -64., 0., 0.]]])
            result = model(base_action=score, shadow_prior=torch.zeros_like(score), keys=torch.tensor([[[1., 0.]]]),
                query_scores=score, query_mask=torch.tensor([[True]]), prior_mask=torch.tensor([[True]]),
                key_mask=torch.tensor([[True]]), lengths=torch.tensor([1]), episode_ends=torch.tensor([False]), carry=carry)
            carry = detach_carry(result.carry)
        results[mode] = float(carry.matrix[0, 0, 0])
    assert results["trace_delta"] == pytest.approx(1 - (1 - .5 / (1 + 1e-6))**4, abs=2e-7)
    assert results["trace_additive"] == pytest.approx(2 / (1 + 1e-6), abs=2e-7)


def test_matrix_decay_and_no_write_retain_existing_state_reads():
    model = QueryMemory(Config("instant_delta", 2, memory_decay=.5))
    carry = replace(model.initial_carry(1), absolute_step=torch.tensor([1]), has_query=torch.tensor([True]),
                    matrix=torch.tensor([[[1., 0.], [-1., 0.], [0., 0.], [0., 0.]]]))
    base = torch.zeros(1, 1, 4)
    result = model(base_action=base, shadow_prior=torch.full_like(base, float("nan")), keys=torch.tensor([[[1., 0.]]]),
        query_scores=torch.full_like(base, float("nan")), query_mask=torch.tensor([[False]]), prior_mask=torch.tensor([[False]]),
        key_mask=torch.tensor([[True]]), lengths=torch.tensor([1]), episode_ends=torch.tensor([True]), carry=carry, no_write=True)
    assert torch.equal(result.action_prediction, torch.tensor([[[32., -32., 0., 0.]]]))
    assert torch.equal(result.carry.matrix, carry.matrix * .5)
    assert result.work["matrix_decays"] == result.work["matrix_reads"] == 1 and result.work["matrix_writes"] == 0
    assert result.work_units["matrix_read_terms"] == result.work_units["matrix_decayed_coordinates"] == 8


def test_zero_step_size_counts_executed_write_not_nonzero_change():
    result = QueryMemory(Config("trace_delta", 3, step_size=0))(**packet((4,)))
    assert not result.carry.matrix.any()
    assert result.work["matrix_writes"] == result.work["innovation_calculations"] == 1


def test_last_error_read_precedes_overwrite_and_decays_at_every_key_step():
    values = packet((7,))
    result = QueryMemory(Config("last_error", 3))(**values)
    vector = torch.tensor([1., -1., .5, -.5])
    assert not result.prewrite_correction[0, 2].any()
    torch.testing.assert_close(result.prewrite_correction[0, 3], 3 * .75 * vector, rtol=0, atol=0)
    torch.testing.assert_close(result.prewrite_correction[0, 5], 3 * .75**3 * vector, rtol=0, atol=0)
    torch.testing.assert_close(result.prewrite_correction[0, 6], 6 * .75 * vector, rtol=0, atol=0)
    assert result.work["last_error_reads"] == result.work["last_error_decays"] == 6
    assert result.work["last_error_writes"] == result.work["innovation_calculations"] == 2


def test_near_zero_key_uses_clamped_norm_and_nonfinite_intermediate_fails():
    values = packet((2,), 2)
    values["keys"][0, 1] = torch.tensor([1e-9, 0.])
    result = QueryMemory(Config("instant_delta", 2, trace_decay=0))(**values)
    torch.testing.assert_close(result.carry.trace, torch.tensor([[1e-3, 0.]]), rtol=1e-6, atol=0)
    assert not result.carry.matrix.any()
    values["keys"][0, 1] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="finite key/cue norm"):
        QueryMemory(Config("instant_delta", 2))(**values)


@pytest.mark.parametrize("field,value", (("key_dim", 1), ("key_dim", True), ("step_size", float("nan")),
    ("step_size", 1.01), ("trace_decay", -.1), ("memory_decay", float("inf")),
    ("last_error_decay", True), ("epsilon", 0), ("epsilon", -.1), ("mode", "eligibility_gradient")))
def test_configuration_rejects_invalid_or_ambiguous_values(field, value):
    with pytest.raises(ValueError):
        Config(**{field: value})


@pytest.mark.parametrize("defect", ("first_query", "first_key", "later_key", "prior", "padding_query", "query_bits", "dtype"))
def test_schedule_query_copy_and_dtype_contracts_fail_closed(defect):
    values = packet()
    if defect == "first_query":
        values["query_mask"][0, 0] = False
    elif defect == "first_key":
        values["key_mask"][0, 0] = True
    elif defect == "later_key":
        values["key_mask"][0, 1] = False
    elif defect == "prior":
        values["prior_mask"][0, 2] = False
    elif defect == "padding_query":
        values["query_mask"][1, 6] = True
    elif defect == "query_bits":
        values["base_action"][0, 0, 0] = +.0
    else:
        values["keys"] = values["keys"].double()
    with pytest.raises(ValueError):
        QueryMemory(Config("trace_delta", 3))(**values)


def test_config_identity_pristine_carry_and_all_padding_continuation():
    values = packet()
    model = QueryMemory(Config("trace_delta", 3))
    wrong = QueryMemory(Config("trace_delta", 3, step_size=.1)).initial_carry(2)
    with pytest.raises(ValueError, match="configuration"):
        model(**values, carry=wrong)
    dirty = replace(model.initial_carry(2), trace=torch.ones(2, 3))
    with pytest.raises(ValueError, match="pristine"):
        model(**values, carry=dirty)
    ended = detach_carry(model(**values).carry)
    empty = piece(values, 0, 1)
    empty["lengths"].zero_()
    empty["base_action"].zero_()
    for name in ("query_mask", "prior_mask", "key_mask"):
        empty[name].zero_()
    result = model(**empty, carry=ended)
    assert not any(result.work.values())
    assert bits(result.carry.matrix) == bits(ended.matrix)
