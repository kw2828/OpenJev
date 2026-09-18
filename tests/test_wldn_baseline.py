# SPDX-License-Identifier: GPL-3.0-only
"""Selected pinned WLDN function parity through a small Torch compatibility shim.

The original TensorFlow runtime, initialization and training pipeline are not
reproduced. Unmodified selected method bodies execute using supplied identical
weights. Only their mathematical forward/gradient calculation is compared.
"""
import ast
from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openjev.research.wldn_baseline import ChessWLDNHead, WLEncoder, WLStep, native_edge_list
from test_chess_graph_contrast import fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'evidence/chess-wldn-reference-v1/sources/models.py'
SOURCE_SHA = 'efc40b127ee851bc88ca3add8ee28484ae692194c382545a90370b1f99bd2577'


def graph_fixture():
    generator = torch.Generator().manual_seed(47)
    nodes = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64)
    # Directed edges, an isolated receiving node, unequal degrees, two graphs.
    triples = [[0, 0, 1], [0, 0, 3], [0, 1, 2], [0, 3, 1],
               [1, 0, 2], [1, 1, 4], [1, 1, 0], [1, 1, 3], [1, 3, 1]]
    index = torch.tensor(triples).T.contiguous()
    attributes = torch.randn(len(triples), 4, generator=generator, dtype=torch.float64)
    counts = torch.zeros(2, 5, dtype=torch.long)
    neighbors = torch.zeros(2, 5, 3, 2, dtype=torch.long)
    bonds = torch.zeros_like(neighbors)
    all_bonds = torch.randn(2, len(triples), 4, generator=generator, dtype=torch.float64)
    for k, (owner, receiving, neighbor) in enumerate(triples):
        position = int(counts[owner, receiving]); counts[owner, receiving] += 1
        neighbors[owner, receiving, position] = torch.tensor([owner, neighbor])
        bonds[owner, receiving, position] = torch.tensor([owner, k])
        all_bonds[owner, k] = attributes[k]
    return nodes, index, attributes, (nodes, all_bonds, neighbors, bonds, counts)


def pinned_reference(module, monkeypatch):
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    tree = ast.parse(SOURCE.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ('rcnn_wl_only', 'wl_diff_net')]
    assert len(selected) == 2

    def shape_check(value, expected):
        assert len(expected) == value.ndim
        assert all(want is None or want == got for want, got in zip(expected, value.shape))
    monkeypatch.setattr(torch.Tensor, 'set_shape', shape_check, raising=False)

    def linear(value, width, name, init_bias=0.):
        if name == 'atom_embedding':
            layer = module.embedding; assert init_bias is None and layer.bias is None
        else:
            step = module.step if isinstance(module, WLEncoder) else module
            layer = {'label_U2': step.message, 'label_U1': step.update}[name]
        assert layer.out_features == width
        return layer(value)

    compat = SimpleNamespace(
        nn=SimpleNamespace(relu=torch.relu), float32=torch.float64,
        variable_scope=lambda *args, **kwargs: nullcontext(),
        gather_nd=lambda value, indices: value[tuple(indices.unbind(-1))],
        sequence_mask=lambda lengths, maximum, dtype: (torch.arange(maximum)[None] < lengths[:, None]).to(dtype),
        shape=lambda value: torch.tensor(value.shape),
        reshape=lambda value, shape: value.reshape(tuple(int(x) for x in shape)),
        concat=lambda values, dim: torch.cat([torch.as_tensor(v) for v in values], dim),
        reduce_sum=lambda value, dim: value.sum(dim))
    namespace = {'tf': compat, 'linearND': linear, 'max_nb': 3}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('depth', (1, 3))
@pytest.mark.parametrize('mode', ('encoder', 'difference'))
def test_selected_official_functions_match_values_and_gradients(mode, depth, monkeypatch):
    torch.manual_seed(53)
    nodes, index, features, original = graph_fixture()
    if mode == 'encoder':
        model = WLEncoder(7, 4, 6, depth).double()
        reference = pinned_reference(model, monkeypatch)
        nodes.requires_grad_()
        actual = model(nodes, index, features)
        expected = reference['rcnn_wl_only']((nodes, *original[1:]), 6, depth)
    else:
        model = WLStep(7, 4).double()
        reference = pinned_reference(model, monkeypatch)
        nodes.requires_grad_(); actual = nodes
        for _ in range(depth): actual = model(actual, index, features)
        actual = actual.sum(1)
        expected = reference['wl_diff_net'](original, nodes, 7, depth)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    variables = list(model.parameters())+[nodes]
    left = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    right = torch.autograd.grad(expected.square().sum(), variables)
    for a, b in zip(left, right): torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_native_flags_preserve_both_directions_and_multiple_relations():
    raw = torch.zeros(1, 2, 64, 64, dtype=torch.uint8)
    raw[0, 0, 2, 7] = 1; raw[0, 1, 7, 2] = 1
    index, features = native_edge_list(raw, torch.float64)
    assert index.tolist() == [[0, 0], [2, 7], [7, 2]]
    assert features.tolist() == [[1., 0., 0., 1.], [0., 1., 1., 0.]]
    empty = native_edge_list(torch.zeros_like(raw), torch.float64)
    assert empty[0].shape == (3, 0) and empty[1].shape == (0, 4)
    raw[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match='self edges'): native_edge_list(raw, torch.float64)


def test_edgeless_nodes_retain_self_update_without_invented_neighbors():
    model = WLStep(3, 4).double()
    nodes = torch.randn(2, 5, 3, dtype=torch.float64)
    actual = model(nodes, torch.empty(3, 0, dtype=torch.long), torch.empty(0, 4, dtype=torch.float64))
    expected = model.update(torch.cat((nodes, torch.zeros_like(nodes)), -1)).relu()
    assert torch.equal(actual, expected)


def test_graph_permutation_equivariance_and_edge_order():
    torch.manual_seed(61); model = WLEncoder(7, 4, 6, 3).double()
    nodes, index, features, _ = graph_fixture(); expected = model(nodes, index, features)
    permutation = torch.tensor([3, 0, 4, 1, 2]); inverse = permutation.argsort()
    changed = index.clone(); changed[1:] = inverse[index[1:]]
    actual = model(nodes[:, permutation], changed, features)
    torch.testing.assert_close(actual, expected[:, permutation], atol=1e-12, rtol=1e-12)
    order = torch.arange(index.shape[1]-1, -1, -1)
    torch.testing.assert_close(model(nodes, index[:, order], features[order]), expected, atol=1e-12, rtol=1e-12)


def test_head_initial_policy_parameter_count_and_compact_children():
    args = list(fixture()); head = ChessWLDNHead(seed=67).double()
    assert sum(p.numel() for p in head.parameters()) == 16638
    assert torch.equal(head(*args), args[4])
    with torch.no_grad(): head.output.weight.fill_(.1)
    padded = head(*args); args[-1] = args[-1][args[3]]
    assert torch.equal(head(*args), padded)
    assert torch.isneginf(padded[~args[3]]).all()


def test_shared_root_head_matches_separate_one_candidate_calls_and_gradients():
    torch.manual_seed(71); args = list(fixture()); head = ChessWLDNHead(seed=71).double()
    with torch.no_grad(): head.output.weight.fill_(.1)
    args[0].requires_grad_(); args[1].requires_grad_()
    actual = head(*args); independent = []
    for b, m in args[3].nonzero().tolist():
        single = [args[0][b:b+1], args[1][b:b+1, m:m+1], args[2][b:b+1, m:m+1],
                  torch.ones(1, 1, dtype=torch.bool), args[4][b:b+1, m:m+1],
                  args[5][b:b+1], args[6][b:b+1, m:m+1]]
        independent.append(head(*single).flatten())
    expected = torch.cat(independent)
    torch.testing.assert_close(actual[args[3]], expected, atol=1e-12, rtol=1e-12)
    variables = list(head.parameters())+args[:2]
    a = torch.autograd.grad(actual[args[3]].square().sum(), variables, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), variables)
    for left, right in zip(a, b): torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)


def test_difference_encoder_can_score_zero_node_difference_due_to_edges_and_biases():
    step = WLStep(2, 1).double()
    with torch.no_grad():
        step.message.weight.fill_(1); step.message.bias.zero_()
        step.update.weight.fill_(1); step.update.bias.zero_()
    delta = torch.zeros(1, 2, 2, dtype=torch.float64)
    output = step(delta, torch.tensor([[0], [0], [1]]), torch.ones(1, 1, dtype=torch.float64))
    assert output[0, 0].min() > 0 and not output[0, 1].any()


def test_invalid_graph_indices_and_nonfinite_features_are_rejected():
    model = WLEncoder(7, 4, 6, 3).double(); nodes, index, features, _ = graph_fixture()
    bad = index.clone(); bad[0, 0] = 2
    with pytest.raises(ValueError, match='out of range'): model(nodes, bad, features)
    bad = index.clone(); bad[1, 0] = -1
    with pytest.raises(ValueError, match='out of range'): model(nodes, bad, features)
    bad = features.clone(); bad[0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'): model(nodes, index, bad)
