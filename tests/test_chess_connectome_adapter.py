"""Synthetic adapter contracts, not a gameplay or real-connectome evaluation."""

import copy
import math
import random

import chess
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from openjev.research.chess_candidate import CandidateChess, encode_batch
from openjev.research.chess_connectome_adapter import MODES, ConnectomeChessAdapter
from openjev.research.connectome_graph import SignedGraph, graph_sha256


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_graph(nodes=67, *, reverse=False):
    """Small ring with unequal signed edges and more nodes than 64 slots."""
    sources = np.arange(nodes, dtype=np.int64)
    destinations = (sources + 1) % nodes
    if reverse:
        sources, destinations = destinations, sources
    return SignedGraph(
        node_ids=100 + 7 * np.arange(nodes),
        groups=np.arange(nodes) % 4,
        sources=sources,
        destinations=destinations,
        signs=np.where(np.arange(nodes) % 3 == 0, -1, 1),
        anatomical_counts=1 + np.arange(nodes),
        provenance={"fixture": "synthetic-only"},
    )


def boards():
    return [chess.Board(), chess.Board("7k/8/8/8/8/8/p7/7K b - - 7 19")]


def adapter(*, mode="sparse", graph=None, **kwargs):
    return ConnectomeChessAdapter(
        CandidateChess("direct", seed=19, width=4),
        tiny_graph() if graph is None else graph,
        mode=mode,
        slot_channels=1,
        **kwargs,
    )


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_state_equal(module, expected):
    actual = module.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def activate_residual(model):
    """A fixed synthetic perturbation, not a fitted model or evaluation label."""
    with torch.no_grad():
        model.output_projection.weight.fill_(0.125)
        model.output_projection.bias.fill_(0.01)


@pytest.mark.parametrize("mode", MODES)
def test_zero_residual_exactly_matches_full_direct_root_and_copies_backbone(mode):
    parent = CandidateChess("direct", seed=91, width=4)
    parent.train()
    before = snapshot(parent)
    wrapped = ConnectomeChessAdapter(parent, tiny_graph(), mode=mode, slot_channels=1)
    batch, _ = encode_batch(boards(), "direct")
    expected = parent(**batch, depth=4)
    actual = wrapped(**batch)
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert parent.training
    assert not wrapped.backbone.training
    assert_state_equal(parent, before)
    assert_state_equal(wrapped.backbone, before)
    originals = dict(parent.named_parameters())
    for name, value in wrapped.backbone.named_parameters():
        assert not value.requires_grad
        assert value.data_ptr() != originals[name].data_ptr(), name
    wrapped.train()
    assert wrapped.training and not wrapped.backbone.training
    wrapped.eval()
    assert not wrapped.training and not wrapped.backbone.training


@pytest.mark.parametrize("mode", MODES)
def test_residual_receives_gradients_then_graph_learns_without_changing_backbone(mode):
    parent = CandidateChess("direct", seed=91, width=4)
    original = snapshot(parent)
    model = ConnectomeChessAdapter(parent, tiny_graph(), mode=mode, slot_channels=1)
    frozen = snapshot(model.backbone)
    batch, _ = encode_batch(boards(), "direct")
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.05)

    def loss():
        logits, value, _ = model(**batch)
        # Synthetic arithmetic objective: no real training targets are accessed.
        return logits[batch["legal_mask"]].square().mean() + (value - 0.5).square().mean()

    loss().backward()
    assert torch.count_nonzero(model.output_projection.weight.grad)
    assert torch.count_nonzero(model.input_projection.weight.grad) == 0
    assert torch.count_nonzero(model.edge_log_gain.grad) == 0
    assert torch.count_nonzero(model.node_bias.grad) == 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss().backward()
    for parameter in (model.input_projection.weight, model.edge_log_gain, model.node_bias):
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad)
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert_state_equal(parent, original)
    assert_state_equal(model.backbone, frozen)


def test_slot_mapping_is_balanced_deterministic_and_shared_by_topology_controls():
    original = tiny_graph(131)
    reversed_graph = tiny_graph(131, reverse=True)
    baseline = adapter(graph=original, seed=43, mapping_seed=7)
    for mode, graph, seed in (("sparse", reversed_graph, 99), ("dense", original, 5),
                              ("node_local", original, 2), ("sparse", original, 43)):
        model = adapter(graph=graph, mode=mode, seed=seed, mapping_seed=7)
        assert torch.equal(model.node_slots, baseline.node_slots)
        assert torch.equal(model.slot_occupancy, baseline.slot_occupancy)
    occupancy = torch.bincount(baseline.node_slots, minlength=64)
    assert baseline.node_slots.shape == (131,)
    assert baseline.node_slots.dtype == torch.long
    assert int(occupancy.sum()) == 131
    assert int(occupancy.max() - occupancy.min()) <= 1
    torch.testing.assert_close(baseline.slot_occupancy, occupancy.to(baseline.slot_occupancy.dtype))
    changed = adapter(graph=original, seed=43, mapping_seed=8)
    assert not torch.equal(changed.node_slots, baseline.node_slots)


def test_constructor_restores_global_rng_and_repeats_seeded_initialization():
    parent = CandidateChess("direct", seed=19, width=4)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    first = ConnectomeChessAdapter(parent, tiny_graph(), seed=31, mapping_seed=59, slot_channels=1)
    second = ConnectomeChessAdapter(parent, tiny_graph(), seed=31, mapping_seed=59, slot_channels=1)
    assert_state_equal(second, snapshot(first))
    assert torch.equal(torch_rng, torch.random.get_rng_state())
    assert python_rng == random.getstate()
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and numpy_rng[2:] == after[2:]
    np.testing.assert_array_equal(numpy_rng[1], after[1])
    assert torch.count_nonzero(first.input_projection.weight)
    assert torch.count_nonzero(first.output_projection.weight) == 0
    assert torch.count_nonzero(first.output_projection.bias) == 0


@pytest.mark.parametrize("mode", MODES)
def test_state_resets_and_candidate_order_padding_do_not_change_predictions(mode):
    model = adapter(mode=mode)
    activate_residual(model)
    batch, _ = encode_batch(boards(), "direct")
    before = snapshot(model)
    reference = model(**batch)
    alternate, _ = encode_batch([chess.Board("7k/8/8/8/8/8/R7/7K w - - 0 1")], "direct")
    model(**alternate)
    repeated = model(**batch)
    for observed, expected in zip(repeated, reference, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert_state_equal(model, before)

    permutation = torch.arange(batch["candidates"].shape[1] - 1, -1, -1)
    reordered = dict(batch)
    reordered["candidates"] = batch["candidates"][:, permutation]
    reordered["legal_mask"] = batch["legal_mask"][:, permutation]
    output = model(**reordered)
    torch.testing.assert_close(output[0][:, permutation], reference[0], rtol=0, atol=0)
    for observed, expected in zip(output[1:], reference[1:], strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    padded = dict(batch)
    padded["candidates"] = torch.cat((batch["candidates"], batch["candidates"][:, :3]), dim=1)
    padded["legal_mask"] = F.pad(batch["legal_mask"], (0, 3), value=False)
    output = model(**padded)
    torch.testing.assert_close(output[0][:, :-3], reference[0], rtol=0, atol=0)
    assert torch.isneginf(output[0][:, -3:]).all()
    for observed, expected in zip(output[1:], reference[1:], strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_choose_is_full_legal_menu_with_no_board_or_history_mutation(mode):
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8", "e2e4"):
        board.push_uci(move)
    before = copy.deepcopy(board)
    model = adapter(mode=mode)
    activate_residual(model)
    result = model.choose(board)
    legal = sorted(move.uci() for move in board.legal_moves)
    assert result["choice"] in legal
    assert list(result["probabilities"]) == legal
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in result["probabilities"].values())
    assert board.fen(en_passant="fen") == before.fen(en_passant="fen")
    assert board.move_stack == before.move_stack
    assert board.root().fen(en_passant="fen") == before.root().fen(en_passant="fen")


def irregular_graph():
    return SignedGraph(
        node_ids=[91, 13, 54, 20, 8],
        groups=[0, 1, 2, 3, 0],
        sources=[0, 0, 1, 0, 2, 3, 4],
        destinations=[1, 2, 2, 2, 3, 3, 4],
        signs=[1, -1, 1, 1, -1, 1, -1],
        anatomical_counts=[200, 19, 1, 4, 70, 8, 12],
    )


def test_signed_matrix_uses_source_to_destination_and_fixed_in_degree_independent_math():
    graph = irregular_graph()
    model = adapter(graph=graph)
    gains = torch.tensor([-1.5, -0.75, 0.25, 0.5, 1.0, 1.75, 2.25])
    with torch.no_grad():
        model.edge_log_gain.copy_(gains)
    degree = [sum(int(target) == node for target in graph.destinations) for node in range(5)]
    expected = np.zeros((5, 5), dtype=np.float64)
    for source, target, sign, gain in zip(
        graph.sources, graph.destinations, graph.signs, gains.tolist(), strict=True
    ):
        expected[target, source] += int(sign) * math.log1p(math.exp(gain)) / degree[target]
    matrix = model._message_matrix()
    torch.testing.assert_close(matrix, torch.tensor(expected, dtype=matrix.dtype), rtol=1e-6, atol=1e-7)
    # A source impulse must arrive at its outgoing destinations; not the reverse.
    hidden = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 2.0, 0.25, 3.0, -2.0]])
    expected_messages = np.array([
        [sum(expected[target, source] * row[source] for source in range(5)) for target in range(5)]
        for row in hidden.tolist()
    ])
    torch.testing.assert_close(
        model._messages(hidden), torch.tensor(expected_messages, dtype=hidden.dtype), rtol=1e-6, atol=1e-7
    )
    assert torch.count_nonzero(matrix[0]) == 0  # Zero incoming degree stays finite and disconnected.
    assert torch.isfinite(matrix).all()
    np.testing.assert_array_equal(model.sources.cpu().numpy(), graph.sources)
    np.testing.assert_array_equal(model.destinations.cpu().numpy(), graph.destinations)
    np.testing.assert_array_equal(model.signs.cpu().numpy(), graph.signs)
    assert model.edge_log_gain.numel() == len(graph.sources)


def test_anatomical_counts_do_not_become_synaptic_weights_and_signs_cannot_be_trained_away():
    graph = irregular_graph()
    plain = SignedGraph(graph.node_ids, graph.groups, graph.sources, graph.destinations, graph.signs)
    first, second = adapter(graph=graph, seed=91), adapter(graph=plain, seed=91)
    torch.testing.assert_close(first._message_matrix(), second._message_matrix(), rtol=0, atol=0)
    assert graph_sha256(graph) != graph_sha256(plain)
    original_signs = first.signs.clone()
    # Every learnable sparse edge is constrained to a strictly positive magnitude.
    optimizer = torch.optim.SGD([first.edge_log_gain], lr=0.25)
    first._message_matrix().square().sum().backward()
    optimizer.step()
    assert torch.equal(first.signs, original_signs)
    assert (F.softplus(first.edge_log_gain) > 0).all()
    trainable = {name for name, value in first.named_parameters() if value.requires_grad}
    assert trainable == {
        "input_projection.weight", "input_projection.bias",
        "output_projection.weight", "output_projection.bias", "edge_log_gain", "node_bias",
    }
    supported = torch.zeros(5, 5, dtype=torch.bool)
    supported[first.destinations, first.sources] = True
    assert torch.count_nonzero(first._message_matrix()[~supported]) == 0


def test_full_hidden_residual_matches_independent_recurrence_and_occupancy_pooling():
    graph = irregular_graph()
    model = ConnectomeChessAdapter(
        CandidateChess("direct", seed=41, width=4), graph,
        slot_channels=2, seed=17, mapping_seed=3, steps=3, alpha=0.3,
    ).double()
    activate_residual(model)
    with torch.no_grad():
        model.node_bias.copy_(torch.linspace(-0.2, 0.3, 5, dtype=torch.float64))
    batch, _ = encode_batch(boards(), "direct")
    batch["observations"] = batch["observations"].double()
    _, _, root = model.backbone(**batch, depth=4)
    slots = F.conv2d(root, model.input_projection.weight, model.input_projection.bias).flatten(1)
    drive = slots[:, model.node_slots]
    initial = torch.tanh(drive)
    state = initial.clone()
    edge_magnitudes = [math.log1p(math.exp(value)) for value in model.edge_log_gain.detach().tolist()]
    incoming = [sum(int(target) == node for target in graph.destinations) for node in range(5)]
    for _ in range(3):
        messages = torch.zeros_like(state)
        for source, target, sign, magnitude in zip(
            graph.sources, graph.destinations, graph.signs, edge_magnitudes, strict=True
        ):
            messages[:, target] += int(sign) * magnitude * state[:, source] / incoming[target]
        state = 0.7 * state + 0.3 * torch.tanh(drive + messages + model.node_bias)
    pooled = torch.zeros((2, 128), dtype=torch.float64)
    for slot in range(128):
        nodes = [node for node, assigned in enumerate(model.node_slots.tolist()) if assigned == slot]
        if nodes:
            pooled[:, slot] = (state[:, nodes] - initial[:, nodes]).mean(dim=1)
    residual = F.conv2d(
        pooled.reshape(2, 2, 8, 8), model.output_projection.weight, model.output_projection.bias
    )
    actual = model(**batch)[2]
    torch.testing.assert_close(actual, root + residual, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(actual - root)


def test_edgeless_graph_has_finite_messages_without_invented_connections():
    graph = SignedGraph(
        node_ids=[5, 11, 17], groups=[0, 1, 2],
        sources=np.array([], dtype=int), destinations=np.array([], dtype=int),
        signs=np.array([], dtype=int),
    )
    model = adapter(graph=graph)
    assert model.edge_log_gain.numel() == 0
    assert torch.count_nonzero(model._message_matrix()) == 0
    assert torch.count_nonzero(model._messages(torch.ones(2, 3))) == 0
    batch, _ = encode_batch(boards(), "direct")
    logits, value, _ = model(**batch)
    assert torch.isfinite(logits[batch["legal_mask"]]).all() and torch.isfinite(value).all()


@pytest.mark.parametrize("mode,edges", [("sparse", 7), ("dense", 25), ("node_local", 5)])
def test_parameters_and_dense_compute_are_disclosed_without_claiming_sparse_speed(mode, edges):
    model = adapter(mode=mode, graph=irregular_graph())
    counts = model.parameter_counts()
    stored = sum(value.numel() for value in model.parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    frozen = sum(value.numel() for value in model.backbone.parameters())
    assert counts["stored"] == stored
    assert counts["trainable"] == trainable
    assert counts["frozen_backbone"] == frozen
    assert counts["active_frozen_backbone"] == model.backbone.parameter_counts()["active"]
    assert counts["inactive_frozen_backbone"] == frozen - counts["active_frozen_backbone"]
    assert counts["input_projection"] == 5  # 4x1 weight plus one bias.
    assert counts["output_projection"] == 8  # 1x4 weight plus four biases.
    assert counts["edge_magnitudes"] == edges
    assert counts["node_bias"] == 5
    assert trainable == 5 + 8 + edges + 5
    cost = model.computation_counts(batch_size=2, legal_candidates=23)
    assert cost["frozen_root_iterations"] == 2 * 4
    assert cost["input_projection_macs"] == 2 * 64 * 4
    assert cost["output_projection_macs"] == 2 * 64 * 4
    assert cost["graph_dense_matmul_macs"] == 2 * 4 * 25
    assert cost["graph_structural_message_products"] == 2 * 4 * edges
    assert cost["graph_matrix_entries_materialized"] == 25
    assert cost["graph_edge_gain_evaluations"] == edges
    assert cost["node_state_updates"] == 2 * 4 * 5
    assert cost["slot_gather_values"] == 2 * 5
    assert cost["slot_pool_accumulations"] == 2 * 5
    assert cost["candidate_score_slots"] == 2 * 23
    components = ("frozen_root_conv_macs", "input_projection_macs", "output_projection_macs",
                  "graph_dense_matmul_macs", "policy_head_macs", "value_head_macs")
    assert cost["accounted_dense_mac_total"] == sum(cost[name] for name in components)
    assert "not total FLOPs or measured latency" in cost["scope"]

    # Count executed modules independently rather than duplicating the receipt's formulas.
    observed_macs = {}
    active_parameters = {}

    def observe(name):
        def hook(module, inputs, output):
            for parameter in module.parameters(recurse=False):
                if name.startswith("backbone."):
                    active_parameters[id(parameter)] = parameter.numel()
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                observed_macs[name] = observed_macs.get(name, 0) + output.numel() * module.weight[0].numel()
        return hook

    handles = [
        module.register_forward_hook(observe(name))
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding))
    ]
    try:
        batch, _ = encode_batch(boards(), "direct")
        model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    actual_cost = model.computation_counts(batch_size=2, legal_candidates=batch["candidates"].shape[1])
    assert sum(active_parameters.values()) == counts["active_frozen_backbone"]
    assert sum(observed_macs.values()) == (
        actual_cost["accounted_dense_mac_total"] - actual_cost["graph_dense_matmul_macs"]
    )
    assert sum(value for name, value in observed_macs.items()
               if name.startswith(("backbone.encoder", "backbone.core"))) == actual_cost["frozen_root_conv_macs"]
    assert sum(value for name, value in observed_macs.items()
               if name.startswith("backbone.policy_head")) == actual_cost["policy_head_macs"]
    assert sum(value for name, value in observed_macs.items()
               if name.startswith("backbone.value_head")) == actual_cost["value_head_macs"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_respects_floating_dtype_and_keeps_index_buffers_integral(dtype):
    model = adapter().to(dtype=dtype)
    activate_residual(model)
    batch, _ = encode_batch(boards(), "direct")
    batch["observations"] = batch["observations"].to(dtype)
    logits, value, hidden = model(**batch)
    assert logits.dtype == value.dtype == hidden.dtype == dtype
    assert logits.device.type == value.device.type == hidden.device.type == "cpu"
    assert model.node_slots.dtype == model.sources.dtype == model.destinations.dtype == torch.long
    assert torch.isfinite(logits[batch["legal_mask"]]).all()
    assert torch.isfinite(value).all() and torch.isfinite(hidden).all()


def test_mismatched_input_dtype_or_device_is_rejected_before_computation():
    model = adapter()
    batch, _ = encode_batch(boards(), "direct")
    wrong_dtype = dict(batch, observations=batch["observations"].double())
    with pytest.raises((ValueError, TypeError)):
        model(**wrong_dtype)
    wrong_device = dict(batch, observations=batch["observations"].to("meta"))
    with pytest.raises((ValueError, TypeError)):
        model(**wrong_device)


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"slot_channels": 0}, {"slot_channels": True},
    {"steps": 0}, {"steps": True}, {"alpha": 0}, {"alpha": 1.1},
    {"alpha": float("nan")}, {"alpha": float("inf")},
    {"seed": True}, {"mapping_seed": True},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ConnectomeChessAdapter(CandidateChess("direct", width=4), tiny_graph(), **kwargs)


@pytest.mark.parametrize("arm", ["action_only", "delta", "full_afterstate"])
def test_non_direct_backbone_is_rejected(arm):
    with pytest.raises((ValueError, TypeError)):
        ConnectomeChessAdapter(CandidateChess(arm, width=4), tiny_graph(), slot_channels=1)


def test_wrong_depth_or_type_and_oversized_graph_are_rejected():
    with pytest.raises((ValueError, TypeError)):
        ConnectomeChessAdapter(CandidateChess("direct", width=4, root_depth=2), tiny_graph())
    with pytest.raises((ValueError, TypeError)):
        ConnectomeChessAdapter(torch.nn.Linear(2, 2), tiny_graph())
    with pytest.raises((ValueError, TypeError)):
        ConnectomeChessAdapter(CandidateChess("direct", width=4), "graph")
    with pytest.raises((ValueError, TypeError)):
        ConnectomeChessAdapter(CandidateChess("direct", width=4), tiny_graph(4097))


@pytest.mark.parametrize("case", ["observation_shape", "nonfinite", "integer_observation",
                                  "candidate_dtype", "candidate_range", "empty_menu",
                                  "mask_dtype", "batch_mismatch"])
def test_invalid_forward_inputs_are_rejected(case):
    model = adapter()
    batch, _ = encode_batch(boards(), "direct")
    if case == "observation_shape":
        batch["observations"] = batch["observations"][:, :18]
    elif case == "nonfinite":
        batch["observations"][0, 0, 0, 0] = float("nan")
    elif case == "integer_observation":
        batch["observations"] = batch["observations"].long()
    elif case == "candidate_dtype":
        batch["candidates"] = batch["candidates"].float()
    elif case == "candidate_range":
        batch["candidates"][0, 0, 0] = 64
    elif case == "empty_menu":
        batch["legal_mask"][0].zero_()
    elif case == "mask_dtype":
        batch["legal_mask"] = batch["legal_mask"].long()
    elif case == "batch_mismatch":
        batch["legal_mask"] = batch["legal_mask"][:1]
    with pytest.raises((ValueError, TypeError)):
        model(**batch)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": True},
                                    {"legal_candidates": 0}, {"legal_candidates": True}])
def test_computation_receipt_rejects_ambiguous_counts(kwargs):
    with pytest.raises((ValueError, TypeError)):
        adapter().computation_counts(**kwargs)
