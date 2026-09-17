"""Compact chess students: GRU and explicitly synthetic signed sparse circuits.

The circuit layout is inspired by sensory/inter/command/motor organization. It
does not import a biological connectome. Recurrence refines a single FEN position;
there is no persistent state across moves and no environment world-model claim.
"""

import hashlib
import math
import random
import time

import chess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('gru', 'circuit', 'rewired')
WIDTH = 64
DEPTH = 4
INPUT_SIZE = 12*64+2+4+65+2
LAYERS = {'sensory': (0, 16), 'inter': (16, 40), 'command': (40, 56), 'motor': (56, 64)}


def move_vocabulary():
    moves = set()
    for origin in chess.SQUARES:
        x, y = chess.square_file(origin), chess.square_rank(origin)
        for target in chess.SQUARES:
            dx, dy = abs(chess.square_file(target)-x), abs(chess.square_rank(target)-y)
            if origin != target and (dx == 0 or dy == 0 or dx == dy or sorted((dx, dy)) == [1, 2]):
                moves.add(chess.square_name(origin)+chess.square_name(target))
        if y in (1, 6):
            last = 0 if y == 1 else 7
            for target_x in range(max(0, x-1), min(7, x+1)+1):
                base = chess.square_name(origin)+chess.square_name(chess.square(target_x, last))
                moves.update(base+suffix for suffix in 'bnqr')
    result = tuple(sorted(moves))
    if len(result) != 1968:
        raise ValueError('Unexpected geometric UCI vocabulary')
    return result


UCI_MOVES = move_vocabulary()
MOVE_TO_INDEX = {move: index for index, move in enumerate(UCI_MOVES)}
MOVE_VOCAB_SHA256 = hashlib.sha256(('\n'.join(UCI_MOVES)+'\n').encode()).hexdigest()


def state_key(board):
    """Exact piece placement, turn, castling and EP; deliberately ignore clocks."""
    return ' '.join(board.fen(en_passant='fen').split()[:4])


def encode_board(board):
    """Numeric FEN-visible state only; no teacher labels or repetition history.

    Piece channels: white P/N/B/R/Q/K then black P/N/B/R/Q/K, square a1..h8.
    Append side one-hot (white,black), WK/WQ/BK/BQ rights, EP square one-hot with
    an explicit no-EP bin, halfmove/150 and log1p(fullmove)/10 counters.
    """
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise ValueError('Expected a valid standard chess position')
    values = np.zeros(INPUT_SIZE, dtype=np.float32)
    for square, piece in board.piece_map().items():
        channel = piece.piece_type-1+(0 if piece.color else 6)
        values[channel*64+square] = 1.
    offset = 768
    values[offset+(0 if board.turn else 1)] = 1.
    offset += 2
    values[offset:offset+4] = [board.has_kingside_castling_rights(chess.WHITE),
                              board.has_queenside_castling_rights(chess.WHITE),
                              board.has_kingside_castling_rights(chess.BLACK),
                              board.has_queenside_castling_rights(chess.BLACK)]
    offset += 4
    values[offset+(64 if board.ep_square is None else board.ep_square)] = 1.
    values[-2:] = [board.halfmove_clock/150., math.log1p(board.fullmove_number)/10.]
    return values


def legal_mask(board):
    result = np.zeros(len(UCI_MOVES), dtype=bool)
    for move in board.legal_moves:
        result[MOVE_TO_INDEX[move.uci()]] = True
    return result


def circuit_topology(rewire=False):
    """Return signed adjacency [destination,source], with fixed topology seeds.

    Structured edges: each sensory ->8 inter, each inter ->4 command, each
    command ->4 other command and ->4 motor. Every fourth source is inhibitory.
    Rewiring swaps destinations of same-sign edge pairs, preserving every node's
    signed in/out degrees, total edges, no self-edges and no duplicate edges.
    """
    rng = random.Random(90217)
    edges = set()
    groups = [(range(16), range(16, 40), 8), (range(16, 40), range(40, 56), 4),
              (range(40, 56), range(40, 56), 4), (range(40, 56), range(56, 64), 4)]
    for sources, targets, count in groups:
        for source in sources:
            for target in rng.sample([v for v in targets if v != source], count):
                edges.add((source, target, -1 if source % 4 == 0 else 1))
    ordered = sorted(edges)
    swaps = 0
    if rewire:
        rng = random.Random(90329)
        # Fixed proposal budget; no tuning to downstream performance.
        for _ in range(20000):
            i, j = rng.sample(range(len(ordered)), 2)
            a, b, sign = ordered[i]
            c, d, other_sign = ordered[j]
            if sign != other_sign or a == c or b == d or a == d or c == b:
                continue
            first, second = (a, d, sign), (c, b, sign)
            if first in edges or second in edges:
                continue
            edges.remove(ordered[i])
            edges.remove(ordered[j])
            edges.update((first, second))
            ordered[i], ordered[j] = first, second
            swaps += 1
        if swaps == 0:
            raise ValueError('Rewiring did not alter the topology')
    adjacency = torch.zeros(WIDTH, WIDTH, dtype=torch.int8)
    for source, target, sign in edges:
        adjacency[target, source] = sign
    return adjacency


class SignedCircuit(nn.Module):
    def __init__(self, rewire=False):
        super().__init__()
        adjacency = circuit_topology(rewire)
        target, source = torch.where(adjacency != 0)
        self.register_buffer('source', source)
        self.register_buffer('target', target)
        self.register_buffer('sign', adjacency[target, source].float())
        indegree = (adjacency != 0).sum(1).clamp_min(1).float().sqrt()
        self.register_buffer('normalizer', indegree[target])
        drive_mask = torch.zeros(WIDTH)
        drive_mask[:16] = 1.
        self.register_buffer('drive_mask', drive_mask)
        self.magnitude = nn.Parameter(torch.empty(len(source)).normal_(-1., .1))
        self.bias = nn.Parameter(torch.zeros(WIDTH))

    def forward(self, encoded, hidden):
        weights = self.sign*F.softplus(self.magnitude)/self.normalizer
        matrix = encoded.new_zeros(WIDTH, WIDTH).index_put((self.target, self.source), weights)
        return torch.tanh(F.linear(hidden, matrix, self.bias)+encoded*self.drive_mask)


class ChessStudent(nn.Module):
    def __init__(self, mode='gru', seed=17):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown chess student mode')
        self.mode, self.seed = mode, seed
        # Construct shared modules before mode-specific modules so their initial
        # parameters are exactly matched within each seed across all arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Linear(INPUT_SIZE, WIDTH), nn.Tanh())
            self.policy_head = nn.Linear(WIDTH, len(UCI_MOVES))
            self.value_head = nn.Linear(WIDTH, 1)
            self.core = nn.GRUCell(WIDTH, WIDTH) if mode == 'gru' else SignedCircuit(mode == 'rewired')

    def forward(self, observations, mask=None, depth=DEPTH):
        if type(depth) is not int or depth < 1:
            raise ValueError('depth must be a positive integer')
        encoded = self.encoder(observations)
        hidden = torch.zeros_like(encoded)
        for _ in range(depth):
            hidden = self.core(encoded, hidden)
        logits = self.policy_head(hidden)
        if mask is not None:
            if mask.shape != logits.shape or not mask.any(dim=-1).all():
                raise ValueError('Every input requires a nonempty legal-move mask')
            logits = logits.masked_fill(~mask.bool(), -torch.inf)
        return logits, torch.tanh(self.value_head(hidden)).squeeze(-1)

    @torch.no_grad()
    def choose(self, board):
        self.eval()
        start = time.perf_counter()
        device = next(self.parameters()).device
        observations = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
        mask = torch.from_numpy(legal_mask(board)).unsqueeze(0).to(device)
        logits, value = self(observations, mask)
        probabilities = logits.softmax(-1).cpu()[0]
        candidates = sorted(move.uci() for move in board.legal_moves)
        return {'choice': UCI_MOVES[int(probabilities.argmax())],
                'probabilities': {move: float(probabilities[MOVE_TO_INDEX[move]]) for move in candidates},
                'value': float(value.cpu()[0]), 'mode': self.mode, 'seed': self.seed,
                'depth': DEPTH, 'latency_ms': (time.perf_counter()-start)*1000,
                'probability_semantics': 'uncalibrated legal-move softmax',
                'model_kind': 'learned compact chess student; synthetic circuit, not imported connectome'}

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @classmethod
    def load(cls, path, *, expected_plan_sha256=None):
        """Load a trusted local checkpoint without unpickling arbitrary objects."""
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['vocabulary_sha256'] != MOVE_VOCAB_SHA256:
            raise ValueError('Checkpoint move vocabulary mismatch')
        if expected_plan_sha256 is not None and payload['plan_sha256'] != expected_plan_sha256:
            raise ValueError('Checkpoint plan hash mismatch')
        model = cls(payload['mode'], payload['seed'])
        model.load_state_dict(payload['state_dict'], strict=True)
        return model.eval()
