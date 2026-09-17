"""Board-aware candidate policies with tied or untied spatial computation.

All modes score the supplied legal menu without search or cross-move memory.
``reconstruct`` and ``predict`` share the recurrent architecture: their training
targets, supplied by the caller, distinguish them. Future-piece supervision is
an auxiliary representation objective, not an implemented planning algorithm.
"""

import copy
import math
import time
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

MODES = ('cnn', 'recurrent', 'reconstruct', 'predict')
WIDTH = 32
DEPTH = 4
INPUT_CHANNELS = 19
PIECE_CLASSES = 13
ENCODING_VERSION = 'side-relative-rank-mirror-19-v1'
CHECKPOINT_VERSION = 1


def _perspective(board, perspective):
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise ValueError('Expected a valid standard chess position')
    result = board.turn if perspective is None else perspective
    if type(result) is not bool:
        raise ValueError('perspective must be chess.WHITE or chess.BLACK')
    return result


def _square(square, perspective):
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def piece_targets(board, perspective):
    """Return [rank,file] classes: empty=0, own P..K=1..6, opponent=7..12.

Pass the PRE-MOVE side explicitly when encoding a successor. Otherwise changing
turn would reverse both the square orientation and the ownership labels.
    """
    perspective = _perspective(board, perspective)
    values = np.zeros((8, 8), dtype=np.int64)
    for square, piece in board.piece_map().items():
        canonical = _square(square, perspective)
        values[chess.square_rank(canonical), chess.square_file(canonical)] = (
            piece.piece_type + (0 if piece.color == perspective else 6))
    return values


def encode_board(board, perspective=None):
    """Return float32 [19,8,8] using own-side rank orientation (files unchanged).

Channels 0..11: own P/N/B/R/Q/K then opponent P/N/B/R/Q/K; 12..15:
own kingside/queenside and opponent kingside/queenside castling rights; 16:
recorded en-passant square; 17: halfmove/150; 18: log1p(fullmove)/10.
No move history, SAN hints, labels, or engine scores enter this encoding.
    """
    perspective = _perspective(board, perspective)
    values = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        canonical = _square(square, perspective)
        channel = piece.piece_type-1+(0 if piece.color == perspective else 6)
        values[channel, chess.square_rank(canonical), chess.square_file(canonical)] = 1.
    rights = (board.has_kingside_castling_rights(perspective),
              board.has_queenside_castling_rights(perspective),
              board.has_kingside_castling_rights(not perspective),
              board.has_queenside_castling_rights(not perspective))
    for channel, right in enumerate(rights, start=12):
        values[channel].fill(float(right))
    if board.ep_square is not None:
        canonical = _square(board.ep_square, perspective)
        values[16, chess.square_rank(canonical), chess.square_file(canonical)] = 1.
    values[17].fill(board.halfmove_clock/150.)
    values[18].fill(math.log1p(board.fullmove_number)/10.)
    return values


def encode_candidates(board):
    """Return original sorted UCI IDs and int64 [L,5] canonical move features.

Columns: source square, target square, promotion (0 or piece type 2..5),
file displacement+7, rank displacement+7. Empty menus have shape (0,5).
    """
    perspective = _perspective(board, None)
    moves = sorted(board.legal_moves, key=lambda move: move.uci())
    values = np.zeros((len(moves), 5), dtype=np.int64)
    for index, move in enumerate(moves):
        source, target = _square(move.from_square, perspective), _square(move.to_square, perspective)
        values[index] = (source, target, move.promotion or 0,
                         chess.square_file(target)-chess.square_file(source)+7,
                         chess.square_rank(target)-chess.square_rank(source)+7)
    return tuple(move.uci() for move in moves), values


def _check_candidates(candidates):
    if candidates.dtype != torch.long or candidates.shape[-1] != 5:
        raise ValueError('Candidate features must be int64 with five columns')
    if ((candidates[..., :2] < 0) | (candidates[..., :2] >= 64)).any():
        raise ValueError('Candidate square indices must be in 0..63')
    promotion = candidates[..., 2]
    if ((promotion != 0) & ((promotion < 2) | (promotion > 5))).any():
        raise ValueError('Promotion must be zero or piece type 2..5')
    if ((candidates[..., 3:] < 0) | (candidates[..., 3:] >= 15)).any():
        raise ValueError('Displacement indices must be in 0..14')


def action_planes(candidates, *, dtype=torch.float32):
    """Map canonical [B,5] actions to [B,3,8,8] without reading a successor.

The planes mark source=1, destination=1, and promotion/5 at destination.
The last plane is zero for a non-promoting move. Castling/EP side effects must
be learned; this function supplies neither the resulting board nor its pieces.
    """
    if candidates.ndim != 2:
        raise ValueError('Actions must have shape [batch,5]')
    _check_candidates(candidates)
    values = torch.zeros((len(candidates), 3, 64), device=candidates.device, dtype=dtype)
    values[:, 0].scatter_(1, candidates[:, 0:1], 1.)
    values[:, 1].scatter_(1, candidates[:, 1:2], 1.)
    values[:, 2].scatter_(1, candidates[:, 1:2], candidates[:, 2:3].to(dtype)/5.)
    return values.reshape(-1, 3, 8, 8)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, hidden):
        update = self.conv2(self.activation(self.conv1(hidden)))
        return self.activation(hidden+update)


class SpatialChess(nn.Module):
    def __init__(self, mode='recurrent', seed=17, width=WIDTH, depth=DEPTH):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown spatial chess mode')
        if type(width) is not int or width < 1 or type(depth) is not int or depth < 1:
            raise ValueError('width and depth must be positive integers')
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        self.mode, self.seed, self.width, self.depth = mode, seed, width, depth
        # Shared modules precede the core. Deep-copying consumes no random draws,
        # so all modes start with identical predictions and common parameters.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, width, 3, padding=1), nn.ReLU())
            self.promotion_embedding = nn.Embedding(6, 8)
            self.dx_embedding = nn.Embedding(15, 8)
            self.dy_embedding = nn.Embedding(15, 8)
            self.policy_head = nn.Sequential(nn.Linear(3*width+24, 2*width), nn.ReLU(),
                                             nn.Linear(2*width, 1, bias=False))
            self.value_head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
            self.aux_head = nn.Sequential(nn.Conv2d(width+3, width, 3, padding=1), nn.ReLU(),
                                          nn.Conv2d(width, PIECE_CLASSES, 1))
            core = ResidualBlock(width)
            if mode == 'cnn':
                self.blocks = nn.ModuleList([copy.deepcopy(core) for _ in range(depth)])
            else:
                self.core = core

    def _depth(self, depth):
        result = self.depth if depth is None else depth
        if type(result) is not int or result < 1:
            raise ValueError('depth must be a positive integer')
        if self.mode == 'cnn' and result > self.depth:
            raise ValueError('CNN depth exceeds its number of untied blocks')
        return result

    def forward(self, observations, candidates, legal_mask, depth=None):
        """Return masked [B,L] logits, bounded [B] value, and [B,C,8,8] hidden.

Pad candidate rows with zeros and mark them false in the boolean legal mask.
At least one candidate per input must remain legal. Recurrence starts anew on
every call, and reordering the candidate menu only reorders the policy output.
        """
        depth = self._depth(depth)
        if observations.ndim != 4 or observations.shape[1:] != (INPUT_CHANNELS, 8, 8):
            raise ValueError('Observations must have shape [batch,19,8,8]')
        if candidates.ndim != 3 or candidates.shape[0] != observations.shape[0]:
            raise ValueError('Candidate features must have shape [batch,moves,5]')
        _check_candidates(candidates)
        if (legal_mask.dtype != torch.bool or legal_mask.shape != candidates.shape[:2]
                or not legal_mask.any(dim=-1).all()):
            raise ValueError('Every input requires a nonempty boolean legal-move mask')
        if observations.device != candidates.device or legal_mask.device != candidates.device:
            raise ValueError('Observations, candidates and mask must use the same device')
        hidden = self.encoder(observations)
        for index in range(depth):
            hidden = self.blocks[index](hidden) if self.mode == 'cnn' else self.core(hidden)
        squares = hidden.flatten(2).transpose(1, 2)
        batch = torch.arange(len(hidden), device=hidden.device).unsqueeze(1)
        pooled = hidden.mean(dim=(2, 3))
        features = torch.cat((squares[batch, candidates[..., 0]], squares[batch, candidates[..., 1]],
                              pooled.unsqueeze(1).expand(-1, candidates.shape[1], -1),
                              self.promotion_embedding(candidates[..., 2]),
                              self.dx_embedding(candidates[..., 3]),
                              self.dy_embedding(candidates[..., 4])), dim=-1)
        logits = self.policy_head(features).squeeze(-1).masked_fill(~legal_mask, -torch.inf)
        value = torch.tanh(self.value_head(pooled)).squeeze(-1)
        return logits, value, hidden

    action_planes = staticmethod(action_planes)

    def predict_board(self, hidden, candidates):
        """Decode square classes conditioned on the current hidden and one action.

The caller selects current-board or successor-board labels, and must keep the
successor in the PRE-MOVE perspective. This function receives no target board.
        """
        if hidden.ndim != 4 or hidden.shape[1:] != (self.width, 8, 8):
            raise ValueError('Hidden state must have shape [batch,width,8,8]')
        if candidates.ndim != 2 or candidates.shape[0] != hidden.shape[0]:
            raise ValueError('Actions must have shape [batch,5] and match hidden batch')
        if hidden.device != candidates.device:
            raise ValueError('Hidden state and actions must use the same device')
        planes = action_planes(candidates, dtype=hidden.dtype)
        return self.aux_head(torch.cat((hidden, planes), dim=1))

    @torch.no_grad()
    def choose(self, board, depth=None):
        self.eval()
        start = time.perf_counter()
        depth = self._depth(depth)
        ids, candidates = encode_candidates(board)
        if not ids:
            raise ValueError('Cannot choose from an empty legal-move menu')
        parameter = next(self.parameters())
        observations = torch.from_numpy(encode_board(board)).unsqueeze(0).to(parameter)
        candidates = torch.from_numpy(candidates).unsqueeze(0).to(parameter.device)
        mask = torch.ones((1, len(ids)), device=parameter.device, dtype=torch.bool)
        logits, value, _ = self(observations, candidates, mask, depth)
        probabilities = logits.softmax(-1).cpu()[0]
        return {'choice': ids[int(probabilities.argmax())],
                'probabilities': {move: float(probabilities[index]) for index, move in enumerate(ids)},
                'value': float(value.cpu()[0]), 'mode': self.mode, 'seed': self.seed,
                'depth': depth, 'width': self.width, 'latency_ms': (time.perf_counter()-start)*1000,
                'probability_semantics': 'uncalibrated legal-move softmax',
                'model_kind': 'spatial candidate policy; no search or cross-move state'}

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def save(self, path, *, plan_sha256):
        """Write a new checkpoint binding architecture/encoding to its frozen plan."""
        _check_plan_hash(plan_sha256)
        payload = {'format_version': CHECKPOINT_VERSION, 'encoding': ENCODING_VERSION,
                   'mode': self.mode, 'seed': self.seed, 'width': self.width, 'depth': self.depth,
                   'plan_sha256': plan_sha256, 'state_dict': self.state_dict()}
        with Path(path).open('xb') as handle:
            torch.save(payload, handle)

    @classmethod
    def load(cls, path, *, expected_plan_sha256=None):
        """Read tensor-only local weights and reject plan/encoding mismatches."""
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload['format_version'] != CHECKPOINT_VERSION or payload['encoding'] != ENCODING_VERSION):
            raise ValueError('Checkpoint encoding or format mismatch')
        _check_plan_hash(payload['plan_sha256'])
        if expected_plan_sha256 is not None and payload['plan_sha256'] != expected_plan_sha256:
            raise ValueError('Checkpoint plan hash mismatch')
        model = cls(payload['mode'], payload['seed'], payload['width'], payload['depth'])
        model.load_state_dict(payload['state_dict'], strict=True)
        return model.eval()


def _check_plan_hash(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError('plan_sha256 must be a lowercase SHA-256 digest')
