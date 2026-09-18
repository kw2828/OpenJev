# SPDX-License-Identifier: GPL-3.0-only
import pytest
import torch

from openjev.research.wldn_baseline import ChessWLDNHead
from openjev.research.wldn_interventions import MODES, compact_permutation, hooked_reference, scores
from test_chess_graph_contrast import fixture


def inputs():
    return list(fixture()), torch.tensor([[1, 0, 2], [1, 2, 0]])


def head():
    h = ChessWLDNHead(seed=1109).double()
    with torch.no_grad(): h.output.weight.copy_(torch.linspace(-.2, .2, 59)[None])
    return h


@pytest.mark.parametrize('mode', MODES)
def test_shared_primary_matches_original_head_hooks_and_restores_state(mode):
    args, permutation = inputs(); h = head(); before = {k: v.clone() for k, v in h.state_dict().items()}
    native = h(*args).detach(); primary = scores(h, args, permutation)
    reference = hooked_reference(h, args, permutation, mode)
    torch.testing.assert_close(primary[mode], reference, atol=1e-12, rtol=1e-12)
    assert torch.equal(h(*args), native)
    assert not h.difference._forward_pre_hooks and not h.readout._forward_pre_hooks
    assert all(torch.equal(before[k], v) for k, v in h.state_dict().items())


def test_fresh_zero_projection_keeps_all_interventions_at_base_and_identity_shuffle_is_native():
    args, permutation = inputs(); h = ChessWLDNHead(seed=1109).double()
    assert all(torch.equal(v, args[4]) for v in scores(h, args, permutation).values())
    identity = torch.arange(3).expand(2, -1).clone(); result = scores(head(), args, identity)
    assert torch.equal(result['native'], result['permuted_delta'])


def test_zero_node_difference_retains_edge_path_but_zero_pool_removes_it():
    args, permutation = inputs(); h = head()
    with torch.no_grad():
        h.difference.message.weight.fill_(.05); h.difference.message.bias.zero_()
        h.difference.update.weight.fill_(.01); h.difference.update.bias.zero_()
        h.readout.weight.zero_(); h.readout.weight[:, :32].fill_(.01); h.readout.bias.zero_()
        h.output.weight.fill_(.1)
    result = scores(h, args, permutation)
    assert torch.equal(result['zero_pooled'], args[4])
    assert (result['zero_delta'][args[3]] > args[4][args[3]]).all()


def test_compact_permutation_rejects_cross_menu_or_nonbijective_assignment():
    args, permutation = inputs()
    assert compact_permutation(args[3], permutation).tolist() == [1, 0, 3, 4, 2]
    for changed in (torch.zeros_like(permutation), permutation.float()):
        with pytest.raises(ValueError): compact_permutation(args[3], changed)
    changed = permutation.clone(); changed[0, 0] = 2
    with pytest.raises(ValueError): compact_permutation(args[3], changed)


def test_temporary_hook_is_removed_when_original_forward_raises():
    args, permutation = inputs(); h = head()
    def fail(module, x): raise RuntimeError('injected readout failure')
    failure = h.readout.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError): hooked_reference(h, args, permutation, 'zero_delta')
    failure.remove()
    assert not h.difference._forward_pre_hooks and not h.readout._forward_pre_hooks
