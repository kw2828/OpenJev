"""Pure synthetic causal/mathematical checks; no encoders or datasets."""

import pytest
import torch

from openjev.research.dialogue_carry import LearnedCarry, literal_mention_carry, work_counts


@pytest.fixture
def case():
    # Existing engineering literal only, with restored ambient RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        model = LearnedCarry(5, 4, 6).double()
        turns = torch.randn(2, 5, 5, dtype=torch.float64)
        query = torch.randn(2, 3, 5, dtype=torch.float64)
        candidates = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    valid = torch.tensor([[True, False, True, True, True], [True, True, True, False, True]])
    times = torch.tensor([[0, 3, 4], [-1, 1, 4]])
    mask = torch.tensor([[[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                         [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]], dtype=torch.bool)
    return model, {"turns": turns, "valid": valid, "query": query, "candidates": candidates,
                   "query_times": times, "candidate_mask": mask}


def test_uniform_operations_have_handcomputed_carry_distribution(case):
    model, data = case
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    out = model(**data).exp()
    # Two candidates plus carry: initial [1,0], then [2/3,1/3].
    torch.testing.assert_close(out[0, 0], torch.tensor([2 / 3, 1 / 3, 0, 0], dtype=out.dtype))
    # Second row Q1 includes two actual updates, not a gold state reset.
    torch.testing.assert_close(out[1, 1], torch.tensor([5 / 9, 4 / 9, 0, 0], dtype=out.dtype))
    torch.testing.assert_close(out[1, 0], torch.tensor([1, 0, 0, 0], dtype=out.dtype))
    torch.testing.assert_close(out.sum(-1), torch.ones(2, 3, dtype=out.dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalized_and_masked_finite_gradient(dtype, case):
    model, data = case
    model = model.to(dtype)
    data = {k: v.to(dtype) if v.is_floating_point() else v for k, v in data.items()}
    out = model(**data)
    assert torch.isneginf(out[~data["candidate_mask"]]).all()
    torch.testing.assert_close(out.exp().sum(-1), torch.ones(2, 3, dtype=dtype))
    # Only eligible scored records; initial non-NONE targets are impossible.
    loss = -out[0, :, 1].mean() - out[1, 1:, 1].mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.carry_score[-1].weight.grad.abs().sum() > 0
    assert model.value_score[0].weight.grad.abs().sum() > 0


def test_candidate_permutation_with_explicit_none_index(case):
    model, data = case
    original = model(**data)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(data, candidates=data["candidates"][:, :, permutation],
                   candidate_mask=data["candidate_mask"][:, :, permutation],
                   none_index=torch.ones(2, 3, dtype=torch.long))
    torch.testing.assert_close(model(**changed), original[:, :, permutation])


def test_padding_values_and_extra_candidates_do_not_affect_results(case):
    model, data = case
    original = model(**data)
    candidates = data["candidates"].clone()
    candidates[~data["candidate_mask"]] = torch.nan
    candidates = torch.cat((candidates, torch.full((2, 3, 2, 5), torch.nan, dtype=candidates.dtype)), 2)
    mask = torch.cat((data["candidate_mask"], torch.zeros(2, 3, 2, dtype=torch.bool)), 2)
    changed = model(**dict(data, candidates=candidates, candidate_mask=mask))
    torch.testing.assert_close(changed[:, :, :4], original)
    assert torch.isneginf(changed[:, :, 4:]).all()


def test_future_turns_cannot_affect_earlier_query_or_gradient(case):
    model, data = case
    data = dict(data, turns=data["turns"].clone().requires_grad_(),
                query_times=torch.ones(2, 3, dtype=torch.long))
    original = model(**data)
    changed_turns = data["turns"].detach().clone()
    changed_turns[:, 2:] = torch.nan  # Not even consumed by finite validation.
    torch.testing.assert_close(model(**dict(data, turns=changed_turns)), original)
    loss = -original[:, :, 1].sum()
    loss.backward()
    assert torch.count_nonzero(data["turns"].grad[:, 2:]) == 0


def test_masked_turns_and_holes_do_not_update(case):
    model, data = case
    original = model(**data)
    turns = data["turns"].clone()
    turns[~data["valid"]] = torch.nan
    torch.testing.assert_close(model(**dict(data, turns=turns)), original)


def test_prefix_replay_equals_continuous_streaming_no_label_resets(case):
    model, data = case
    state = model.initial(data["candidate_mask"])
    for t in range(5):
        active = data["valid"][:, t, None] & (data["query_times"] >= t)
        state = model.step(state, data["turns"][:, t, None, :].expand(2, 3, 5),
                           data["query"], data["candidates"], data["candidate_mask"], active)
    torch.testing.assert_close(state, model(**data))
    # A single query replayed to t=4 sees every valid earlier turn, even though
    # no intermediate state was requested/scored.
    one = {"turns": data["turns"][:1], "valid": data["valid"][:1], "query": data["query"][:1, 2:3],
           "candidates": data["candidates"][:1, 2:3], "query_times": torch.tensor([[4]]),
           "candidate_mask": data["candidate_mask"][:1, 2:3]}
    torch.testing.assert_close(model(**one)[0, 0], state[0, 2])


def test_functional_ownership_and_batch_isolation(case):
    model, data = case
    copies = {k: v.clone() for k, v in data.items()}
    out = model(**data)
    for k, value in data.items():
        assert torch.equal(value, copies[k])
    split = torch.cat([model(**{k: v[i:i + 1] for k, v in data.items()}) for i in range(2)])
    torch.testing.assert_close(out, split)
    state = model.initial(data["candidate_mask"])
    new = model.step(state, data["turns"][:, 0, None, :].expand(2, 3, 5), data["query"],
                     data["candidates"], data["candidate_mask"], torch.zeros(2, 3, dtype=torch.bool))
    assert torch.equal(new, state) and new.data_ptr() != state.data_ptr()
    new[0, 0, 0] = 10
    assert state[0, 0, 0] == 0


def test_empty_prefix_and_one_candidate(case):
    model, data = case
    one = dict(data, turns=data["turns"][:, :0], valid=data["valid"][:, :0],
               query_times=torch.full((2, 3), -1), candidates=data["candidates"][:, :, :1],
               candidate_mask=data["candidate_mask"][:, :, :1])
    assert torch.equal(model(**one), torch.zeros(2, 3, 1, dtype=torch.float64))
    one.update(turns=data["turns"], valid=data["valid"], query_times=data["query_times"])
    torch.testing.assert_close(model(**one), torch.zeros(2, 3, 1, dtype=torch.float64))


def test_work_counts_match_actual_linear_hooks(case):
    model, data = case
    counts = work_counts(data["valid"], data["query_times"], data["candidate_mask"])
    actual = {}
    handles = []
    modules = {"query_projection_rows": model.query_projection,
               "candidate_projection_rows": model.candidate_projection,
               "turn_projection_rows": model.turn_projection,
               "carry_scorer_rows": model.carry_score[0], "value_scorer_rows": model.value_score[0]}
    for name, module in modules.items():
        def hook(_module, inputs, _output, name=name):
            actual[name] = actual.get(name, 0) + inputs[0].numel() // inputs[0].shape[-1]
        handles.append(module.register_forward_hook(hook))
    try:
        model(**data)
    finally:
        for handle in handles:
            handle.remove()
    assert actual == {key: counts[key] for key in actual}
    assert counts["recurrent_query_updates"] == 14
    assert counts["valid_candidate_scores"] + counts["padded_candidate_scores"] == 56


@pytest.mark.parametrize("field,change", [
    ("query_times", lambda x: x.float()), ("query_times", lambda x: x + 10),
    ("query_times", lambda x: torch.full_like(x, -2)), ("valid", lambda x: x.long()),
    ("query", lambda x: x.float()), ("query", lambda x: x * torch.nan),
    ("candidate_mask", lambda x: torch.zeros_like(x)), ("candidates", lambda x: x * torch.nan),
    ("turns", lambda x: x * torch.nan), ("query", lambda x: x[:, :, :4]),
])
def test_invalid_inputs_rejected(case, field, change):
    model, data = case
    with pytest.raises(ValueError):
        model(**dict(data, **{field: change(data[field])}))


def test_none_index_and_state_validation(case):
    model, data = case
    with pytest.raises(ValueError, match="NOT_MENTIONED"):
        model(**dict(data, none_index=torch.full((2, 3), 3)))
    state = torch.zeros(2, 3, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="normalized"):
        model.step(state, data["turns"][:, 0, None, :].expand(2, 3, 5), data["query"],
                   data["candidates"], data["candidate_mask"], torch.ones(2, 3, dtype=torch.bool))


def test_default_parameter_count_without_leaking_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        model = LearnedCarry()
    assert model.configuration()["parameter_count"] == 166434


def test_literal_longest_unique_unicode_boundaries_and_carry():
    candidates = ["NOT_MENTIONED", "York", "New York", "Rome", "Café"]
    turns = ["Yorkshire", "New York please", "nothing more", "Rome or York", "CAFÉ please", "caféteria"]
    assert literal_mention_carry(turns, candidates) == [0, 2, 2, 2, 4, 4]


def test_literal_candidate_permutation_none_and_exclusion():
    assert literal_mention_carry(["yes", "B", "A"], ["A", "NONE", "B"], none_index=1,
                                 excluded_indices=(2,)) == [1, 1, 0]
    assert literal_mention_carry(["a"], ["NONE", "A", "a"]) == [0]
    # Deliberately lexical, no unsupported negation understanding.
    assert literal_mention_carry(["not Italian"], ["NONE", "Italian"]) == [1]


@pytest.mark.parametrize("turns,candidates,kwargs", [
    ("hello", ["NONE"], {}), (["hello"], [], {}), ([1], ["NONE"], {}),
    (["hello"], ["NONE", " "], {}), (["hello"], ["NONE"], {"none_index": True}),
    (["hello"], ["NONE"], {"excluded_indices": [2]}),
])
def test_literal_invalid(turns, candidates, kwargs):
    with pytest.raises(ValueError):
        literal_mention_carry(turns, candidates, **kwargs)
