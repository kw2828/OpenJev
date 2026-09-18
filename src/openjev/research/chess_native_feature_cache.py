"""Single-position frozen features for future studies; old caches unchanged.

This fixes a feature-generation execution contract, not model architecture or
precision. Each board uses the same float32 call and unpadded menu as native
inference. Action features are cached too, before node-layout changes. Compact
candidate storage avoids padding all120 action channels across the dataset.
"""
import chess
import torch

from openjev.research.chess_spatial import encode_board, encode_candidates
from openjev.research.chess_transport import candidate_features

VERSION = 'single-position-native-float32-features-v1'


@torch.no_grad()
def build(model, boards):
    parameter = next(model.parameters())
    if (model.arm != 'direct' or model.width != 32 or model.training
            or parameter.device.type != 'cpu' or parameter.dtype != torch.float32
            or any(p.requires_grad for p in model.parameters())):
        raise ValueError('Expected a frozen eval-mode width32 direct CPU float32 backbone')
    boards = list(boards)
    if not boards: raise ValueError('Expected nonempty board input')
    nodes, actions, base, candidates, menus, fens = [], [], [], [], [], []
    offsets = [0]
    for board in boards:
        if type(board) is not chess.Board or board.chess960 or not board.is_valid() or board.is_game_over(claim_draw=False):
            raise ValueError('Expected valid nonterminal standard boards')
        names, encoded = encode_candidates(board)
        candidate = torch.from_numpy(encoded)[None]
        mask = torch.ones(1, len(names), dtype=torch.bool)
        scores, _, hidden = model(torch.from_numpy(encode_board(board))[None], candidate, mask)
        native_nodes = hidden.flatten(2).transpose(1, 2)
        native_actions = candidate_features(model, native_nodes, candidate)
        if not all(torch.isfinite(t).all() for t in (native_nodes, native_actions, scores)):
            raise ValueError('Nonfinite native features')
        nodes.append(native_nodes[0].clone()); actions.append(native_actions[0].clone())
        base.append(scores[0].clone()); candidates.append(candidate[0].clone())
        menus.append(tuple(names)); fens.append(board.fen(en_passant='fen'))
        offsets.append(offsets[-1]+len(names))
    return {'version': VERSION, 'nodes': torch.stack(nodes), 'action_features': torch.cat(actions),
            'base_logits': torch.cat(base), 'candidates': torch.cat(candidates),
            'offsets': torch.tensor(offsets, dtype=torch.long), 'menus': tuple(menus), 'fens': tuple(fens)}


def arguments(cache, indices, graph):
    """Pack selected complete menus, retaining their aligned compact children.

This cache is for frozen features, so no neural computation occurs here. The
graph caller must provide complete native menus/FENs/masks for these roots.
"""
    if (cache['version'] != VERSION or indices.ndim != 1 or indices.dtype != torch.long
            or indices.device.type != 'cpu' or not len(indices)
            or (indices < 0).any() or (indices >= len(cache['menus'])).any()):
        raise ValueError('Invalid native-cache selection')
    ids = indices.tolist(); menus = [cache['menus'][i] for i in ids]
    fens = [chess.Board(fen).fen(en_passant='fen') for fen in graph['fens']]
    if menus != list(graph['menus']) or fens != [cache['fens'][i] for i in ids]:
        raise ValueError('Cached features and native graph identity differ')
    lengths = [len(menu) for menu in menus]; width = max(lengths); batch = len(ids)
    actions = torch.zeros(batch, width, 120); candidates = torch.zeros(batch, width, 5, dtype=torch.long)
    mask = torch.arange(width)[None] < torch.tensor(lengths)[:, None]
    base = torch.full((batch, width), -torch.inf)
    for slot, index in enumerate(ids):
        lo, hi = cache['offsets'][index:index+2].tolist()
        if hi-lo != lengths[slot]: raise ValueError('Cache offsets and menu differ')
        actions[slot, :hi-lo] = cache['action_features'][lo:hi]
        candidates[slot, :hi-lo] = cache['candidates'][lo:hi]
        base[slot, :hi-lo] = cache['base_logits'][lo:hi]
    if (not torch.equal(mask, graph['mask']) or graph['root'].shape != (batch, 2, 64, 64)
            or graph['children'].shape != (sum(lengths), 2, 64, 64)):
        raise ValueError('Native graph packing differs')
    return [cache['nodes'][indices], actions, candidates, mask, base, graph['root'], graph['children']]
