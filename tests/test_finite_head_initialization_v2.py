"""Fabricated initialization and differentiation checks, no generator or data files."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from openjev.research import finite_head_initialization as current
from openjev.research import finite_rounded_models as frozen

SEED = 946101


def inputs():
    prefix = torch.zeros((2, 9, 31), dtype=torch.float32)
    prefix[:, 0, 5] = prefix[:, 0, 9] = 1
    for step in range(1, 9):
        prefix[:, step, step % 4] = 1
        prefix[:, step, 4 + (step + 1) % 4] = 1
    lengths = torch.tensor([9, 9], dtype=torch.int64)
    actions = torch.tensor([[0, 1, 2, 3, 2, 0, 1, 3], [3, 2, 1, 0, 1, 2, 3, 0]], dtype=torch.int64)
    observations = torch.tensor([[1, 0, 2, 4, 4, 4, 4, 4], [0, 1, 2, 3, 1, 0, 3, 2]], dtype=torch.int64)
    return prefix, lengths, actions, observations


def compare_outputs(a, b, *, exact):
    assert a.keys() == b.keys()
    for key in a:
        if key == 'work':
            if exact:
                assert a[key] == b[key]
        elif a[key] is None:
            assert b[key] is None
        else:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0 if exact else 1e-12, msg=key)


def test_anchor_is_bitwise_parent_with_identical_parameter_order_and_all_outputs():
    actual, expected = current.make_model('rounded_anchor', SEED), frozen.make_model('rounded', SEED)
    assert type(actual) is type(expected) is frozen.RoundedDynamicsModel
    assert list(actual.state_dict()) == list(expected.state_dict()) == [
        'transition_logits', 'emission_logits', 'hazard_logits', 'cost_logits']
    for name, value in actual.state_dict().items():
        assert torch.equal(value, expected.state_dict()[name])
    prefix, lengths, actions, observations = inputs()
    compare_outputs(actual.blind_rollout(prefix, lengths, actions),
                    expected.blind_rollout(prefix, lengths, actions), exact=True)
    compare_outputs(actual.observed_rollout(prefix, lengths, actions, observations),
                    expected.observed_rollout(prefix, lengths, actions, observations), exact=True)
    compare_outputs(frozen.prefix_predictions(actual, prefix, lengths),
                    frozen.prefix_predictions(expected, prefix, lengths), exact=True)
    left = frozen.torch_objective(actual, prefix, lengths, .001)
    right = frozen.torch_objective(expected, prefix, lengths, .001)
    assert torch.equal(left['loss'], right['loss']) and left['work'] == right['work']
    left['loss'].backward()
    right['loss'].backward()
    for name in frozen.DYNAMICS:
        assert torch.equal(getattr(actual, name).grad, getattr(expected, name).grad)
    assert actual.cost_logits.grad is expected.cost_logits.grad is None


def test_random_head_exact_local_stream_and_same_values_in_both_transport_arms():
    first = current.make_model('rounded_random', SEED)
    second = current.make_model('matched_free_random', SEED)
    expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, 436, 1]))).standard_normal((4, 8))
    assert first.cost_logits.dtype == torch.float64 and first.cost_logits.device.type == 'cpu'
    np.testing.assert_array_equal(first.cost_logits.detach().numpy(), expected)
    assert torch.equal(first.cost_logits, second.cost_logits)
    assert first.cost_logits.data_ptr() != second.cost_logits.data_ptr()
    assert not torch.equal(first.cost_logits, current.make_model('rounded_anchor', SEED).cost_logits)
    assert not torch.equal(first.cost_logits, current.make_model('rounded_random', SEED + 1).cost_logits)
    assert not np.array_equal(expected, np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([SEED, 436, 2]))).standard_normal((4, 8)))


@pytest.mark.parametrize('arm', current.ARMS)
def test_construction_preserves_global_rng_and_all_nonhead_parameters(arm):
    before_torch = torch.random.get_rng_state().clone()
    before_numpy = copy.deepcopy(np.random.get_state())
    model = current.make_model(arm, SEED)
    parent = frozen.make_model(current.TRANSPORT_ARMS[arm], SEED)
    assert torch.equal(before_torch, torch.random.get_rng_state())
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and before_numpy[2:] == after_numpy[2:]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    for name in frozen.DYNAMICS:
        assert torch.equal(getattr(model, name), getattr(parent, name))
    assert model.construction_work == parent.construction_work
    assert type(model) is frozen.RoundedDynamicsModel
    assert not list(model.buffers()) and not list(model.children())
    assert sum(p.numel() for p in model.parameters()) == 352
    assert sum(p.numel() * p.element_size() for p in model.parameters()) == 2816


def test_random_transports_match_initial_functions_including_found_and_padding():
    rounded = current.make_model('rounded_random', SEED)
    free = current.make_model('matched_free_random', SEED)
    rf, ff = rounded.probability_fields(), free.probability_fields()
    torch.testing.assert_close(rf['transition'], ff['transition'], rtol=0, atol=1e-12)
    for field in ('emission', 'hazard'):
        assert torch.equal(rf[field], ff[field])
    prefix, lengths, actions, observations = inputs()
    compare_outputs(rounded.blind_rollout(prefix, lengths, actions),
                    free.blind_rollout(prefix, lengths, actions), exact=False)
    compare_outputs(rounded.observed_rollout(prefix, lengths, actions, observations),
                    free.observed_rollout(prefix, lengths, actions, observations), exact=False)
    prefix[0, 2, 4:9] = 0
    prefix[0, 2, 8] = 1
    prefix[0, 3:] = 0
    lengths[0] = 3
    first = frozen.prefix_predictions(rounded, prefix, lengths)
    second = frozen.prefix_predictions(free, prefix, lengths)
    compare_outputs(first, second, exact=False)
    assert not first['probabilities'][0, 3:].any() and not first['nll'][0, 3:].any()
    assert first['nll'][0, 2] > 0


@pytest.mark.parametrize('arm', current.ARMS)
def test_effective_metadata_is_owned_static_and_does_not_claim_random_head_privilege(arm):
    model = current.make_model(arm, SEED)
    meta = current.model_metadata(model, arm)
    anchor = arm == 'rounded_anchor'
    assert meta['version'] == current.VERSION and meta['transport_model_version'] == frozen.VERSION
    assert meta['arm'] == arm and meta['transport_arm'] == current.TRANSPORT_ARMS[arm]
    assert meta['privileged_readout_initialization'] is anchor
    assert meta['readout_delta'] == (.1 if anchor else None)
    assert meta['count'] == meta['trainable_count'] == 352
    assert meta['parameter_bytes'] == 2816 and meta['buffer_bytes'] == 0
    assert meta['head_initialization']['seed_sequence_entropy'] == (None if anchor else [SEED, 436, 1])
    assert meta['head_initialization_work'] == current.expected_head_work(arm)
    assert all(type(v) is int for v in meta['head_initialization_work'].values())
    saved = copy.deepcopy(meta)
    meta['head_initialization_work']['parameter_copy_entries'] = 100
    meta['parameters']['cost_logits']['count'] = 100
    assert current.model_metadata(model, arm) == saved
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    other = current.make_model(arm, SEED)
    other.load_state_dict(state)
    assert current.model_metadata(other, arm) == saved
    assert all(torch.equal(value, other.state_dict()[name]) for name, value in state.items())


@pytest.mark.parametrize('arm', current.ARMS)
def test_joint_objective_has_real_dynamic_and_head_update_path(arm):
    model = current.make_model(arm, SEED)
    prefix, lengths, actions, observations = inputs()
    blind = model.blind_rollout(prefix, lengths, actions[:, :2])
    observed = model.observed_rollout(prefix, lengths, actions[:, :2], observations[:, :2])
    target = torch.tensor([.21, -.11, .03, -.13], dtype=torch.float64)
    loss = ((blind['cost_contrasts'] - target).square().mean()
            + (observed['cost_contrasts'] + target).square().mean()
            - observed['probabilities'][0, 0, 1].log()
            + frozen.prefix_predictions(model, prefix, lengths)['nll'].mean())
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    optimizer.step()
    assert all(not torch.equal(before[name], p) for name, p in model.named_parameters())


@pytest.mark.parametrize('arm', current.ARMS)
def test_prefix_objective_excludes_head_and_refuses_oracle(arm):
    model = current.make_model(arm, SEED)
    prefix, lengths, actions, _ = inputs()
    head = model.cost_logits.detach().clone()
    result = frozen.torch_objective(model, prefix, lengths, .001)
    result['loss'].backward()
    assert model.cost_logits.grad is None
    assert result['work']['cost_head_softmax_calls'] == 0
    optimizer = torch.optim.Adam([getattr(model, name) for name in frozen.DYNAMICS], lr=.003)
    optimizer.step()
    assert torch.equal(head, model.cost_logits)
    with pytest.raises(TypeError, match='oracle_prefix'):
        model.blind_rollout(prefix, lengths, actions, oracle_prefix=torch.full((2, 8), 1 / 8, dtype=torch.float64))


@pytest.mark.parametrize('arm,seed', [('rounded', SEED), ('', SEED), (True, SEED),
                                    ('rounded_random', True), ('rounded_random', -1)])
def test_invalid_factory_inputs_fail_before_parent_constructor(monkeypatch, arm, seed):
    def forbidden(*args, **kwargs):
        raise AssertionError('parent constructor must not run')
    monkeypatch.setattr(current, 'parent_model', forbidden)
    with pytest.raises(ValueError):
        current.make_model(arm, seed)


def test_callback_failure_preserves_zero_additional_head_work():
    marker = TimeoutError('fabricated external deadline')
    def fail():
        raise marker
    with pytest.raises(TimeoutError) as caught:
        current.make_model('rounded_random', SEED, check=fail)
    assert caught.value is marker
    assert marker.head_initialization_work == dict.fromkeys(current.HEAD_WORK_KEYS, 0)
