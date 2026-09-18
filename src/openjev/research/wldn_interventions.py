# SPDX-License-Identifier: GPL-3.0-only
"""Post-training interventions on the frozen WLDN adapter, not new policies.

The primary path shares encoder work across interventions. The reference uses
temporary forward pre-hooks on the original unmodified head and always removes
them. Comparing these paths checks where each intervention is applied.
"""
import torch

from openjev.research.wldn_baseline import native_edge_list

MODES = ('native', 'zero_delta', 'root_diff_graph', 'zero_edge_flags', 'zero_pooled', 'permuted_delta')
VERSION = 'wldn-fixed-weight-pathway-interventions-v1'


def compact_permutation(mask, permutation):
    if (mask.ndim != 2 or mask.dtype != torch.bool or not mask.any(-1).all()
            or permutation.shape != mask.shape or permutation.dtype != torch.long
            or permutation.device != mask.device
            or ((permutation < 0) | (permutation >= mask.shape[1])).any()):
        raise ValueError('Invalid legal-candidate permutation')
    for b in range(len(mask)):
        legal = mask[b].nonzero().flatten()
        if not torch.equal(permutation[b, legal].sort().values, legal):
            raise ValueError('Permutation must be a bijection within each root legal menu')
    owner, slot = mask.nonzero(as_tuple=True)
    lookup = torch.full_like(permutation, -1); lookup[owner, slot] = torch.arange(len(owner), device=mask.device)
    return lookup[owner, permutation[owner, slot]]


def validate(head, args):
    hidden, features, candidates, mask, base, root, child = args
    if hidden.ndim != 3 or hidden.shape[0] < 1 or hidden.shape[1:] != (64, 32):
        raise ValueError('Expected frozen width32 node features')
    batch = len(hidden)
    if candidates.ndim != 3 or candidates.shape[0] != batch or candidates.shape[-1] != 5:
        raise ValueError('Invalid candidates')
    moves = candidates.shape[1]
    if (candidates.dtype != torch.long or ((candidates[..., :2] < 0) | (candidates[..., :2] >= 64)).any()
            or features.shape != (batch, moves, 120) or mask.shape != (batch, moves)
            or mask.dtype != torch.bool or not mask.any(-1).all()
            or base.shape != (batch, moves) or root.shape != (batch, 2, 64, 64)):
        raise ValueError('Invalid feature shapes or legal menu')
    owner, slot = mask.nonzero(as_tuple=True)
    if child.shape == (batch, moves, 2, 64, 64): child = child[owner, slot]
    if child.shape != (len(owner), 2, 64, 64): raise ValueError('Invalid compact child graphs')
    parameter = next(head.parameters())
    if any(x.device != parameter.device for x in args): raise ValueError('Device mismatch')
    if any(x.dtype != parameter.dtype for x in (hidden, features, base)): raise ValueError('Dtype mismatch')
    if any(not torch.isfinite(x).all() for x in (hidden, features, base[mask])):
        raise ValueError('Nonfinite features')
    return owner, slot, child


@torch.no_grad()
def scores(head, args, permutation):
    owner, slot, child_edges = validate(head, args)
    hidden, actions, candidates, mask, base, root_edges, _ = args
    reordered = compact_permutation(mask, permutation)
    root_index, root_flags = native_edge_list(root_edges, hidden.dtype)
    child_index, child_flags = native_edge_list(child_edges, hidden.dtype)
    expanded_index, expanded_flags = native_edge_list(root_edges[owner], hidden.dtype)
    root = head.encoder(hidden, root_index, root_flags)
    child = head.encoder(hidden[owner], child_index, child_flags)
    delta = child-root[owner]
    pooled = {
        'native': head.difference(delta, child_index, child_flags).sum(1),
        'zero_delta': head.difference(torch.zeros_like(delta), child_index, child_flags).sum(1),
        'root_diff_graph': head.difference(delta, expanded_index, expanded_flags).sum(1),
        'zero_edge_flags': head.difference(delta, child_index, torch.zeros_like(child_flags)).sum(1),
        'zero_pooled': delta.new_zeros(len(delta), 32),
        'permuted_delta': head.difference(delta[reordered], child_index, child_flags).sum(1)}
    result = {}; batch, moves = mask.shape
    for mode in MODES:
        selected = torch.cat((pooled[mode], actions[owner, slot]), -1)
        correction = head.output(head.readout(selected).relu()).flatten()
        dense = base.new_zeros(batch*moves).scatter(0, owner*moves+slot, correction).reshape(batch, moves)
        result[mode] = (base+dense).masked_fill(~mask, -torch.inf)
    return result


@torch.no_grad()
def hooked_reference(head, args, permutation, mode):
    if mode not in MODES: raise ValueError('Unknown intervention')
    owner, _, _ = validate(head, args)
    reordered = compact_permutation(args[3], permutation)
    handle = None
    if mode == 'zero_delta':
        handle = head.difference.register_forward_pre_hook(lambda module, x: (torch.zeros_like(x[0]), x[1], x[2]))
    elif mode == 'root_diff_graph':
        index, flags = native_edge_list(args[5][owner], args[0].dtype)
        handle = head.difference.register_forward_pre_hook(lambda module, x: (x[0], index, flags))
    elif mode == 'zero_edge_flags':
        handle = head.difference.register_forward_pre_hook(lambda module, x: (x[0], x[1], torch.zeros_like(x[2])))
    elif mode == 'zero_pooled':
        handle = head.readout.register_forward_pre_hook(lambda module, x: (torch.cat((torch.zeros_like(x[0][:, :32]), x[0][:, 32:]), -1),))
    elif mode == 'permuted_delta':
        handle = head.difference.register_forward_pre_hook(lambda module, x: (x[0][reordered], x[1], x[2]))
    try:
        return head(*args)
    finally:
        if handle is not None: handle.remove()
