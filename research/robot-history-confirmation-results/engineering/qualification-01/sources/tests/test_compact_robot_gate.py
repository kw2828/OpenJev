"""Fabricated gate algebra only; no measured inputs, fits or benchmark timing."""
import inspect

import pytest
import torch
from torch import nn

from openjev.research.bounded_robot_transition import BoundedLPV
from openjev.research.compact_robot_gate import CompactResetGate


def inputs(dtype=torch.float64, batch=5):
    return torch.sin(torch.arange(batch*12, dtype=dtype).reshape(batch, 12)/7)


def prior(dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(97101)
        gru = nn.GRUCell(12, 8, dtype=dtype)
        head = nn.Linear(8, 2, dtype=dtype)
    return gru, head


def reference(gru, head, x):
    shape = (8,) if x.ndim == 1 else (len(x), 8)
    return torch.softmax(head(gru(x, x.new_zeros(shape))), dim=-1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('unbatched', [False, True])
def test_nonzero_prior_inference_equivalence(dtype, unbatched):
    gru, head = prior(dtype)
    compact = CompactResetGate.from_prior(gru, head)
    x = inputs(dtype)[0] if unbatched else inputs(dtype)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(compact(x), reference(gru, head, x), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(compact(x).sum(-1), torch.ones(x.shape[:-1], dtype=dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('seed', [0, 8101, 97103])
def test_constructor_exact_parameter_pairing_to_prior_instant(dtype, seed):
    old = BoundedLPV('instant', seed, dtype=dtype)
    gate = CompactResetGate(seed, dtype=dtype)
    assert torch.equal(gate.input_weight, old.scheduler.weight_ih)
    assert torch.equal(gate.reset_update_bias, old.scheduler.bias_ih[:16]+old.scheduler.bias_hh[:16])
    assert torch.equal(gate.candidate_input_bias, old.scheduler.bias_ih[16:])
    assert torch.equal(gate.candidate_hidden_bias, old.scheduler.bias_hh[16:])
    assert torch.equal(gate.head_weight, old.gate.weight) and torch.equal(gate.head_bias, old.gate.bias)
    torch.testing.assert_close(gate(inputs(dtype)), reference(old.scheduler, old.gate, inputs(dtype)), atol=1e-6, rtol=1e-6)
    assert gate.spec()['parameter_count'] == 338
    assert gate.spec()['parameter_bytes'] == 338*torch.empty((), dtype=dtype).element_size()
    assert sum(p.numel() for p in old.scheduler.parameters())+sum(p.numel() for p in old.gate.parameters()) == 546


def test_gradients_map_to_real_active_prior_parameters():
    gru, head = prior()
    compact = CompactResetGate.from_prior(gru, head)
    x_old, x_new = inputs().requires_grad_(), inputs().requires_grad_()
    weight = torch.arange(10, dtype=torch.float64).reshape(5, 2)/3
    (reference(gru, head, x_old)*weight).sum().backward()
    (compact(x_new)*weight).sum().backward()
    mapping = [(compact.input_weight.grad, gru.weight_ih.grad),
               (compact.reset_update_bias.grad, gru.bias_ih.grad[:16]),
               (compact.reset_update_bias.grad, gru.bias_hh.grad[:16]),
               (compact.candidate_input_bias.grad, gru.bias_ih.grad[16:]),
               (compact.candidate_hidden_bias.grad, gru.bias_hh.grad[16:]),
               (compact.head_weight.grad, head.weight.grad), (compact.head_bias.grad, head.bias.grad),
               (x_new.grad, x_old.grad)]
    for actual, expected in mapping:
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    assert torch.count_nonzero(gru.weight_hh.grad) == 0
    for value in compact.parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) == value.numel()


def test_candidate_hidden_bias_is_reset_weighted_not_folded():
    gru, head = prior()
    with torch.no_grad():
        gru.weight_ih.zero_(); gru.weight_hh.fill_(999)
        gru.bias_ih.zero_(); gru.bias_hh.zero_()
        gru.bias_hh[16:] = 2
        head.weight[0].fill_(1); head.weight[1].zero_(); head.bias.zero_()
    gate = CompactResetGate.from_prior(gru, head)
    x = torch.zeros(1, 12, dtype=torch.float64)
    # r=z=.5; n=tanh(1), h=.5*tanh(1). Folding candidate biases gives tanh(2), wrong.
    expected_logit = 4*torch.tanh(torch.tensor(1., dtype=torch.float64))
    expected = torch.softmax(torch.stack((expected_logit, expected_logit*0)), 0)[None]
    torch.testing.assert_close(gate(x), expected, rtol=1e-13, atol=1e-13)
    wrong_logit = 4*torch.tanh(torch.tensor(2., dtype=torch.float64))
    assert not torch.allclose(gate(x), torch.softmax(torch.stack((wrong_logit, wrong_logit*0)), 0)[None])


def test_inactive_hidden_weights_do_not_affect_conversion():
    gru, head = prior()
    first = CompactResetGate.from_prior(gru, head)
    with torch.no_grad():
        gru.weight_hh.add_(123)
    second = CompactResetGate.from_prior(gru, head)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    assert torch.equal(first(inputs()), second(inputs()))


def test_initial_zero_head_blocks_encoder_gradients_without_inactivity_claim():
    gate = CompactResetGate(17, dtype=torch.float64)
    gate(inputs())[:, 0].sum().backward()
    for name, value in gate.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        if not name.startswith('head_'):
            assert torch.count_nonzero(value.grad) == 0
    assert torch.count_nonzero(gate.head_weight.grad) > 0
    assert gate.spec()['structurally_inactive_parameters'] == 0
    assert 'optimization coordinates' in gate.spec()['training_scope']


def test_local_rng_and_owned_parameters_outputs():
    before = torch.random.get_rng_state().clone()
    CompactResetGate(13)
    assert torch.equal(before, torch.random.get_rng_state())
    gru, head = prior()
    gate = CompactResetGate.from_prior(gru, head)
    x = inputs(); initial = gate(x).detach().clone()
    with torch.no_grad():
        gru.weight_ih.add_(5); head.weight.zero_()
    assert torch.equal(gate(x), initial)
    result = gate(x); result.detach().fill_(9)
    assert torch.equal(gate(x), initial)
    assert not dict(gate.named_buffers()) and len(list(gate.children())) == 0


def test_parameter_state_roundtrip_and_chunking():
    gru, head = prior()
    gate = CompactResetGate.from_prior(gru, head)
    other = CompactResetGate(0, dtype=torch.float64)
    other.load_state_dict(gate.state_dict(), strict=True)
    x = inputs()
    assert torch.equal(gate(x), other(x))
    torch.testing.assert_close(gate(x), torch.cat([gate(row[None]) for row in x]), atol=1e-12, rtol=1e-12)
    assert tuple(inspect.signature(gate.forward).parameters) == ('x',)


@pytest.mark.parametrize('seed,dtype', [(True, torch.float32), (-1, torch.float32), (2**63, torch.float64), (0, torch.float16)])
def test_bad_constructor(seed, dtype):
    with pytest.raises(ValueError):
        CompactResetGate(seed, dtype=dtype)


@pytest.mark.parametrize('bad', [torch.ones(11), torch.ones(0, 12), torch.ones(1, 1, 12),
                                  torch.full((2, 12), float('nan')), torch.ones(2, 12, dtype=torch.float16)])
def test_bad_inputs(bad):
    with pytest.raises(ValueError):
        CompactResetGate()(bad)


def test_bad_prior_and_nonfinite_parameters():
    gru, head = prior()
    for wrong in (nn.GRUCell(13, 8), nn.GRUCell(12, 8, bias=False), object()):
        with pytest.raises(ValueError):
            CompactResetGate.from_prior(wrong, head)
    with pytest.raises(ValueError):
        CompactResetGate.from_prior(gru, nn.Linear(8, 3))
    with torch.no_grad():
        head.weight[0, 0] = float('inf')
    with pytest.raises(ValueError):
        CompactResetGate.from_prior(gru, head)
    gate = CompactResetGate()
    with torch.no_grad():
        gate.input_weight[0, 0] = float('nan')
    with pytest.raises(ValueError):
        gate(torch.ones(1, 12))


def test_validated_public_and_private_tensor_kernel_match_values_and_gradients():
    gru, head = prior()
    public = CompactResetGate.from_prior(gru, head)
    private = CompactResetGate.from_prior(gru, head)
    left, right = inputs().requires_grad_(), inputs().requires_grad_()
    public_value = public(left)
    private_value = private._forward_unchecked(right)
    assert torch.equal(public_value, private_value)
    weight = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    (public_value*weight).sum().backward()
    (private_value*weight).sum().backward()
    assert torch.equal(left.grad, right.grad)
    for (left_name, left_parameter), (right_name, right_parameter) in zip(
            public.named_parameters(), private.named_parameters(), strict=True):
        assert left_name == right_name
        assert torch.equal(left_parameter.grad, right_parameter.grad)


def test_public_forward_retains_boundary_validation(monkeypatch):
    gate = CompactResetGate(dtype=torch.float64)
    checked = gate._validate
    calls = []
    def validate():
        calls.append('parameters')
        return checked()
    monkeypatch.setattr(gate, '_validate', validate)
    gate(inputs())
    assert calls == ['parameters']
    gate._forward_unchecked(inputs())
    assert calls == ['parameters']
    with pytest.raises(ValueError, match='input'):
        gate(torch.full((1, 12), float('nan'), dtype=torch.float64))
