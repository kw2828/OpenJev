"""Synthetic graphs, random fixture weights and invented targets only."""

import copy
import math
import os
import random

import chess
import numpy as np
import pytest
import torch

from openjev.research import chess_connectome_interface as interface
from openjev.research.chess_candidate import CandidateChess, encode_batch
from openjev.research.chess_connectome_adapter import ConnectomeChessAdapter
from openjev.research.chess_connectome_interface import (
    ARCHITECTURE,
    HardSquareConnectomeAdapter,
    evaluate_square_swap,
)
from openjev.research.connectome_graph import SignedGraph


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def graph(nodes=67):
    sources = np.arange(nodes)
    return SignedGraph(node_ids=100 + 7 * sources, groups=sources % 4,
                       sources=sources, destinations=(sources + 1) % nodes,
                       signs=np.where(sources % 3 == 0, -1, 1),
                       provenance={'fixture': 'synthetic ring, no external connectome'})


def model(*, active=False, channels=16):
    result = HardSquareConnectomeAdapter(CandidateChess('direct', seed=91, width=4), graph(),
                                        seed=73, mapping_seed=41, slot_channels=channels)
    if active:
        activate(result)
    return result


def activate(value):
    """Fixed arithmetic perturbations, never an optimizer step."""
    with torch.no_grad():
        weight = value.output_projection.weight
        weight.copy_(torch.linspace(-.2, .3, weight.numel()).reshape(weight.shape))
        value.output_projection.bias.fill_(.01)
        value.node_bias.copy_(torch.linspace(-.1, .2, value.nodes))
        value.edge_log_gain.copy_(torch.linspace(.1, .8, value.edges))


def batch(dtype=torch.float32):
    result, _ = encode_batch([chess.Board(), chess.Board('7k/8/8/8/8/8/p7/7K b - - 7 19')], 'direct')
    result['observations'] = result['observations'].to(dtype)
    return result


def training_batch():
    result = batch()
    return {**result, 'targets': torch.zeros(len(result['observations']), dtype=torch.long),
            'values': torch.zeros(len(result['observations']))}


def snapshot(value):
    return {key: tensor.detach().clone() for key, tensor in value.state_dict().items()}


def assert_same(value, old):
    assert value.state_dict().keys() == old.keys()
    for key, expected in old.items():
        assert torch.equal(value.state_dict()[key], expected), key


def test_identity_is_exact_old_adapter_parity_with_active_residual_and_independent_state():
    backbone = CandidateChess('direct', seed=91, width=4)
    old = ConnectomeChessAdapter(backbone, graph(), seed=73, mapping_seed=41, slot_channels=16)
    new = HardSquareConnectomeAdapter(backbone, graph(), seed=73, mapping_seed=41, slot_channels=16)
    activate(old)
    activate(new)
    for actual, expected in zip(new(**batch()), old(**batch()), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert new.parameter_counts() == old.parameter_counts()
    assert new.original_node_slots.data_ptr() != new.node_slots.data_ptr()
    assert new.original_slot_occupancy.data_ptr() != new.slot_occupancy.data_ptr()
    for key, value in new.backbone.named_parameters():
        assert value.data_ptr() != dict(backbone.named_parameters())[key].data_ptr()
    assert new.config['architecture'] == ARCHITECTURE != old.config['architecture']
    assert new.config['mapping_sha256'] == old.config['mapping_sha256']
    assert new.provenance['interface_permutation_learnable']


def test_square_permutation_is_absolute_tied_and_preserves_channel_occupancy_counts():
    value = model(active=True)
    original = snapshot(value)
    permutation = torch.arange(64).roll(7)
    value.apply_permutation(permutation)
    expected = original['node_slots'] // 64 * 64 + permutation[original['node_slots'] % 64]
    assert torch.equal(value.node_slots, expected)
    assert torch.equal(value.node_slots // 64, original['node_slots'] // 64)
    assert torch.equal(value.slot_occupancy.sort().values, original['slot_occupancy'].sort().values)
    assert torch.equal(value.slot_occupancy.reshape(16, 64).sum(1),
                       original['slot_occupancy'].reshape(16, 64).sum(1))
    assert torch.equal(value.slot_occupancy.reshape(16, 64)[:, permutation],
                       original['slot_occupancy'].reshape(16, 64))
    # The same inverse index is used for both assignment and occupancy-mean readout.
    node_delta = torch.linspace(-1, 1, value.nodes).unsqueeze(0)
    pooled = torch.zeros(1, value.slot_count)
    pooled.scatter_add_(1, value.node_slots.unsqueeze(0), node_delta)
    pooled /= value.slot_occupancy.clamp_min(1)
    for index in range(value.slot_count):
        selected = node_delta[0, value.node_slots == index]
        assert pooled[0, index] == pytest.approx(float(selected.mean()) if len(selected) else 0.)
    permutation.zero_()  # Caller mutations cannot alter an applied assignment.
    assert torch.equal(value.square_permutation, torch.arange(64).roll(7))
    current = value.square_permutation.clone()
    inverse = current.argsort()
    value.apply_permutation(current[inverse])
    assert_same(value, original)
    value.apply_permutation(torch.arange(64).flip(0))
    value.restore_permutation()
    assert_same(value, original)


def test_synthetic_1409_node_assignment_keeps_all_1024_slots_and_exact_collision_counts():
    value = HardSquareConnectomeAdapter(CandidateChess('direct', seed=91, width=4), graph(1409),
                                       seed=73, mapping_seed=41, slot_channels=16)
    old = value.slot_occupancy.clone()
    assert int((old == 2).sum()) == 385 and int((old == 1).sum()) == 639
    permutation = torch.arange(64).roll(23)
    value.apply_permutation(permutation)
    assert torch.equal(value.slot_occupancy.reshape(16, 64)[:, permutation], old.reshape(16, 64))
    assert int((value.slot_occupancy == 2).sum()) == 385
    assert int((value.slot_occupancy == 1).sum()) == 639
    assert torch.equal(value.original_slot_occupancy, old)


@pytest.mark.parametrize('permutation', [
    torch.arange(64, dtype=torch.float32), torch.arange(64, dtype=torch.int32),
    torch.arange(63), torch.arange(64).reshape(8, 8), torch.zeros(64, dtype=torch.long),
    torch.arange(1, 65), torch.arange(-1, 63), list(range(64)),
])
def test_invalid_bijections_leave_every_tensor_unchanged(permutation):
    value = model()
    old = snapshot(value)
    with pytest.raises(ValueError):
        value.apply_permutation(permutation)
    assert_same(value, old)


def test_original_slots_are_guarded_and_current_slot_tampering_is_detected():
    value = model()
    value.node_slots[0] = (value.node_slots[0] + 1) % value.slot_count
    with pytest.raises(ValueError, match='tied'):
        value(**batch())
    value.restore_permutation()
    value.original_node_slots[0] = (value.original_node_slots[0] + 1) % value.slot_count
    with pytest.raises(ValueError, match='Original'):
        value.restore_permutation()


def test_constructor_permutation_and_proposals_preserve_global_rng():
    backbone = CandidateChess('direct', seed=91, width=4)
    source, inputs = graph(), training_batch()
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    value = HardSquareConnectomeAdapter(backbone, source, seed=73, mapping_seed=41)
    value.apply_permutation(torch.arange(64).roll(3))
    evaluate_square_swap(value, inputs, 0, 1, policy='fixed')
    assert random.getstate() == python_state
    actual = np.random.get_state()
    assert actual[0] == numpy_state[0] and np.array_equal(actual[1], numpy_state[1])
    assert actual[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_global_channel_permutation_is_exactly_absorbable_by_existing_projections():
    old = ConnectomeChessAdapter(CandidateChess('direct', seed=91, width=4), graph(),
                                seed=73, mapping_seed=41, slot_channels=16).double()
    activate(old)
    equivalent = copy.deepcopy(old)
    permutation = torch.arange(16).roll(5)
    with torch.no_grad():
        equivalent.node_slots.copy_(permutation[old.node_slots // 64] * 64 + old.node_slots % 64)
        equivalent.slot_occupancy.copy_(torch.bincount(equivalent.node_slots, minlength=1024))
        equivalent.input_projection.weight[permutation] = old.input_projection.weight.clone()
        equivalent.input_projection.bias[permutation] = old.input_projection.bias.clone()
        equivalent.output_projection.weight[:, permutation] = old.output_projection.weight.clone()
    for actual, expected in zip(equivalent(**batch(torch.float64)), old(**batch(torch.float64)), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_simultaneous_graph_node_relabeling_is_a_gauge_not_a_topology_result():
    source = graph()
    order = np.roll(np.arange(len(source.node_ids)), 13)
    inverse = np.argsort(order)
    relabeled = SignedGraph(node_ids=source.node_ids[order], groups=source.groups[order],
                            sources=inverse[source.sources], destinations=inverse[source.destinations],
                            signs=source.signs, provenance={'fixture': 'same synthetic graph, new indexing'})
    backbone = CandidateChess('direct', seed=91, width=4)
    old = HardSquareConnectomeAdapter(backbone, source, seed=73, mapping_seed=41).double()
    equivalent = HardSquareConnectomeAdapter(backbone, relabeled, seed=73, mapping_seed=41).double()
    activate(old)
    activate(equivalent)
    with torch.no_grad():
        equivalent.node_bias.copy_(old.node_bias[order])
    permutation = torch.arange(64).roll(7)
    old.apply_permutation(permutation)
    equivalent.apply_permutation(permutation)
    assert torch.equal(equivalent.node_slots, old.node_slots[order])
    assert torch.equal(equivalent._message_matrix(), old._message_matrix()[order][:, order])
    for actual, expected in zip(equivalent(**batch(torch.float64)), old(**batch(torch.float64)), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_square_permutation_cannot_generally_be_absorbed_by_shared_one_by_one_affine_map():
    # Two one-channel, two-square inputs: [1,0] and [0,0]. Swapping squares
    # demands outputs [0,1] and [0,0]. Identical local zeros would need both
    # outputs0 and1, impossible even with a learned shared scalar and bias.
    inputs = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
    target = inputs.flip(-1).flatten().unsqueeze(-1)
    affine_design = torch.stack((inputs.flatten(), torch.ones(4, dtype=torch.float64)), dim=1)
    assert torch.linalg.matrix_rank(affine_design) == 2
    assert torch.linalg.matrix_rank(torch.cat((affine_design, target), dim=1)) == 3


def controlled_forward(value, monkeypatch, *, error_call=None, nonfinite=False, mutate=False, tie=False):
    calls = []
    initial = value.square_permutation.clone()

    def forward(observations, candidates, legal_mask):
        assert not torch.is_grad_enabled()
        calls.append((observations.data_ptr(), candidates.data_ptr(), legal_mask.data_ptr()))
        if error_call == len(calls):
            raise RuntimeError('Synthetic forward failure')
        logits = observations.new_zeros(legal_mask.shape).masked_fill(~legal_mask, -torch.inf)
        if not tie and value.square_permutation[0] != initial[0]:
            logits[:, 0] = 2.
        if nonfinite and len(calls) == 2:
            logits[:, 0] = float('nan')
        if mutate and len(calls) == 2:
            value.node_bias.add_(1.)
        return logits, observations.new_zeros(len(observations)), observations

    monkeypatch.setattr(value, 'forward', forward)
    return calls


@pytest.mark.parametrize('policy', ['learned', 'fixed'])
def test_proposal_acceptance_or_rejection_uses_identical_fixed_weight_work(monkeypatch, policy):
    value, inputs = model(active=True), training_batch()
    old, caller = snapshot(value), {key: tensor.clone() for key, tensor in inputs.items()}
    calls = controlled_forward(value, monkeypatch)
    result = evaluate_square_swap(value, inputs, 0, 1, policy=policy)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0][0] != inputs['observations'].data_ptr()
    assert result['strictly_improved'] and result['accepted'] == (policy == 'learned')
    assert result['forward_calls'] == 2 and result['optimizer_updates'] == 0
    assert result['evaluated_examples'] == 4
    assert math.isfinite(result['wall_seconds']) and result['wall_seconds'] >= 0
    assert 'batch copies' in result['timing_scope'] and 'parameter snapshots' in result['timing_scope']
    assert result['neural_device'] == 'cpu'
    assert value.training and all(parameter.grad is None for parameter in value.parameters())
    for key, parameter in value.named_parameters():
        assert torch.equal(parameter, old[key])
    assert all(torch.equal(tensor, caller[key]) for key, tensor in inputs.items())
    if policy == 'fixed':
        assert_same(value, old)
    else:
        assert value.square_permutation[0] == 1 and value.square_permutation[1] == 0
        value.restore_permutation()
        assert_same(value, old)


def test_exact_loss_tie_rejected_and_initial_zero_output_delays_mapping_signal():
    value = model()
    old = snapshot(value)
    result = evaluate_square_swap(value, training_batch(), 0, 1)
    assert result['current'] == result['proposed']
    assert not result['accepted'] and not result['strictly_improved']
    assert_same(value, old)


def test_mutated_training_batch_is_rejected_and_caller_batch_stays_unchanged(monkeypatch):
    value, inputs = model(), training_batch()
    old = snapshot(value)
    original_inputs = {key: tensor.clone() for key, tensor in inputs.items()}

    def corrupt(observations, candidates, legal_mask):
        observations.add_(1.)
        logits = observations.new_zeros(legal_mask.shape).masked_fill(~legal_mask, -torch.inf)
        return logits, observations.new_zeros(len(observations)), observations

    monkeypatch.setattr(value, 'forward', corrupt)
    with pytest.raises(ValueError, match='fixed training batch'):
        evaluate_square_swap(value, inputs, 0, 1)
    assert_same(value, old)
    assert all(torch.equal(tensor, original_inputs[key]) for key, tensor in inputs.items())


@pytest.mark.parametrize('policy', ['learned', 'fixed'])
@pytest.mark.parametrize('fault', ['first', 'second', 'nan', 'weight'])
def test_any_proposal_error_restores_nonidentity_mapping_weights_and_mode(monkeypatch, policy, fault):
    value = model(active=True)
    value.apply_permutation(torch.arange(64).roll(7))
    value.eval()
    old = snapshot(value)
    calls = controlled_forward(value, monkeypatch, error_call={'first': 1, 'second': 2}.get(fault),
                               nonfinite=fault == 'nan', mutate=fault == 'weight')
    with pytest.raises((RuntimeError, ValueError)):
        evaluate_square_swap(value, training_batch(), 0, 1, policy=policy)
    assert len(calls) == (1 if fault == 'first' else 2)
    assert_same(value, old)
    assert not value.training and all(parameter.grad is None for parameter in value.parameters())


@pytest.mark.parametrize('fault', ['nan_input', 'padding_target', 'target_dtype', 'value_nan', 'value_range',
                                 'same_square', 'bool_square', 'bad_policy', 'weight_nan', 'existing_grad'])
def test_invalid_batch_or_search_inputs_do_not_change_state(fault):
    value, inputs = model(), training_batch()
    old = snapshot(value)
    kwargs, first, second = {}, 0, 1
    if fault == 'nan_input':
        inputs['observations'][0, 0, 0, 0] = float('nan')
    elif fault == 'padding_target':
        inputs['targets'][1] = inputs['legal_mask'].shape[1] - 1
    elif fault == 'target_dtype':
        inputs['targets'] = inputs['targets'].float()
    elif fault == 'value_nan':
        inputs['values'][0] = float('nan')
    elif fault == 'value_range':
        inputs['values'][0] = 2
    elif fault == 'same_square':
        second = first
    elif fault == 'bool_square':
        first = False
    elif fault == 'bad_policy':
        kwargs['policy'] = 'greedy'
    elif fault == 'weight_nan':
        kwargs['value_weight'] = float('nan')
    else:
        value.node_bias.grad = torch.zeros_like(value.node_bias)
    with pytest.raises(ValueError):
        evaluate_square_swap(value, inputs, first, second, **kwargs)
    assert_same(value, old)


def test_active_interface_preserves_padded_candidate_semantics():
    value, inputs = model(active=True), batch()
    value.apply_permutation(torch.arange(64).roll(7))
    logits, _, _ = value(**inputs)
    assert torch.isfinite(logits[inputs['legal_mask']]).all()
    assert torch.isneginf(logits[~inputs['legal_mask']]).all()
    order = torch.arange(inputs['candidates'].shape[1]).flip(0)
    changed = {**inputs, 'candidates': inputs['candidates'][:, order],
               'legal_mask': inputs['legal_mask'][:, order]}
    reordered, _, _ = value(**changed)
    torch.testing.assert_close(reordered, logits[:, order], rtol=0, atol=0)


def test_choose_identifies_new_interface_without_changing_inherited_decision():
    value = model(active=True)
    old = ConnectomeChessAdapter(CandidateChess('direct', seed=91, width=4), graph(),
                                seed=73, mapping_seed=41, slot_channels=16)
    activate(old)
    reference = old.choose(chess.Board())
    result = value.choose(chess.Board())
    assert result['choice'] == reference['choice']
    assert result['probabilities'] == reference['probabilities']
    assert result['interface_architecture'] == ARCHITECTURE
    assert result['node_mapping_sha256'] == value.config['mapping_sha256']
    original_permutation_hash = result['square_permutation_sha256']
    value.apply_permutation(torch.arange(64).roll(3))
    moved = value.choose(chess.Board())
    assert moved['square_permutation_sha256'] != original_permutation_hash
    assert moved['node_mapping_sha256'] != result['node_mapping_sha256']
    assert math.isfinite(moved['latency_ms']) and moved['latency_ms'] > 0


def test_helper_wall_time_wraps_setup_both_forwards_and_restoration(monkeypatch):
    value, inputs = model(), training_batch()
    elapsed = [0.]
    monkeypatch.setattr(interface.time, 'perf_counter', lambda: elapsed[0])
    monkeypatch.setattr(interface, '_sync', lambda device: elapsed.__setitem__(0, elapsed[0] + 2.))
    real_batch = interface._batch

    def charged_batch(*args):
        elapsed[0] += 3.
        return real_batch(*args)

    monkeypatch.setattr(interface, '_batch', charged_batch)
    real_forward, real_apply = value.forward, value.apply_permutation

    def charged_forward(*args):
        elapsed[0] += 5.
        return real_forward(*args)

    def charged_apply(*args):
        elapsed[0] += 7.
        return real_apply(*args)

    monkeypatch.setattr(value, 'forward', charged_forward)
    monkeypatch.setattr(value, 'apply_permutation', charged_apply)
    result = evaluate_square_swap(value, inputs, 0, 1, policy='fixed')
    assert result['wall_seconds'] == 2 + 3 + 5 + 7 + 5 + 7 + 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_synthetic_proposal_has_explicit_cpu_mapping_admin_but_no_neural_fallback(monkeypatch):
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '1', 'Test requires fallback disabled'
    cpu = model(active=True).eval()
    value = copy.deepcopy(cpu).to('mps')
    inputs = {key: tensor.to('mps') for key, tensor in training_batch().items()}
    seen = []
    handle = value.backbone.encoder.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].device.type))
    original_bincount = torch.bincount

    def cpu_bincount(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu', 'Mapping bincount must be explicit CPU administration'
        return original_bincount(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, 'bincount', cpu_bincount)
    try:
        result = evaluate_square_swap(value, inputs, 0, 1, policy='fixed')
        actual = value(**{key: inputs[key] for key in ('observations', 'candidates', 'legal_mask')})
    finally:
        handle.remove()
    assert seen == ['mps', 'mps', 'mps']
    assert result['neural_device'] == 'mps:0' or result['neural_device'] == 'mps'
    assert result['mapping_administration'].startswith('Explicit CPU')
    assert math.isfinite(result['wall_seconds']) and result['wall_seconds'] > 0
    assert all(parameter.device.type == 'mps' for parameter in value.parameters())
    assert all(buffer.device.type == 'mps' for buffer in value.buffers())
    assert all(parameter.grad is None for parameter in value.parameters())
    for observed, expected in zip(actual, cpu(**batch()), strict=True):
        torch.testing.assert_close(observed.cpu(), expected, rtol=1e-5, atol=1e-5)
