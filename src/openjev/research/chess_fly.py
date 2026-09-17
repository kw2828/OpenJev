"""CPU inference adapter for the published ChessFly neural policy.

This independent implementation follows the published model's mathematical
architecture and binary format. No upstream JavaScript is redistributed. Source:
https://huggingface.co/spaces/mlabonne/chessfly/tree/375374bf58f1c828ad7d33905b64f1fa6ac4906f
Model card:
https://huggingface.co/mlabonne/chessfly/tree/ed6744ec984ba401f06d6b0a966a2661b1458f53

The upstream Space declares GPL-3.0; its model card declares license "other" and
states that FlyWire's non-commercial terms apply to graph derivatives. External
graph files and trained weights retain those terms and must stay outside this
project's MIT distribution. This adapter does not download or redistribute them.

This is the neural policy only, without the browser's depth-three search. Its
five recurrent settling steps restart at zero for each position; they are not
memory carried between chess moves. CPU float32 reduction order differs from
the browser JS/WebGPU implementations, so bitwise parity is not claimed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

SPACE_REVISION = '375374bf58f1c828ad7d33905b64f1fa6ac4906f'
MODEL_REVISION = 'ed6744ec984ba401f06d6b0a966a2661b1458f53'
META_SHA256 = '4405c948ffaa490eeb5ce4577e6c2ce9fc06113c014732f3491a62248eb3b693'
WORKER_SHA256 = '2766a2769cab622f4a65298e3cd1aa34d6920b416019bac318f362bb3e47c281'
# Filled from the pinned Hub artifact, never from a user-supplied receipt.
WEIGHTS_SHA256 = 'e05a675c35726b6784c563f6c29ab3fe6afac249e0b143560286a81eab9380d5'
FEATURES = 780
VALUE_BINS = 64


def action_vocabulary() -> tuple[str, ...]:
    """All queen-ray/knight moves plus explicit promotions, sorted by UCI."""
    moves: set[str] = set()
    for source in chess.SQUARES:
        for target in chess.SQUARES:
            dr = abs(chess.square_rank(source)-chess.square_rank(target))
            df = abs(chess.square_file(source)-chess.square_file(target))
            if source != target and (dr == df or dr == 0 or df == 0 or (dr, df) in {(1, 2), (2, 1)}):
                moves.add(chess.square_name(source)+chess.square_name(target))
    for source_rank, target_rank in ((6, 7), (1, 0)):
        for source_file in range(8):
            for target_file in range(max(0, source_file-1), min(8, source_file+2)):
                stem = chess.square_name(chess.square(source_file, source_rank))
                stem += chess.square_name(chess.square(target_file, target_rank))
                moves.update(stem+piece for piece in 'qrbn')
    return tuple(sorted(moves))


ACTIONS = action_vocabulary()
ACTION_INDEX = {move: index for index, move in enumerate(ACTIONS)}


def mirror_uci(move: str) -> str:
    """Reflect ranks and retain files and promotion, matching color reversal."""
    parsed = chess.Move.from_uci(move)
    if not parsed or parsed.drop is not None:
        raise ValueError('Expected a non-null standard chess move')
    return chess.Move(chess.square_mirror(parsed.from_square),
                      chess.square_mirror(parsed.to_square), parsed.promotion).uci()


def encode_board(board: chess.Board) -> torch.Tensor:
    """780 float32 features; black is reflected to the white-to-move frame.

    EP uses the board's recorded FEN target, including an uncapturable target,
    exactly as the upstream parser of a standard FEN. History/clocks are absent.
    """
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise ValueError('Expected a valid standard-chess Board')
    canonical = board if board.turn else board.mirror()
    features = torch.zeros(FEATURES, dtype=torch.float32)
    for square, piece in canonical.piece_map().items():
        channel = piece.piece_type-1 + (0 if piece.color else 6)
        features[square*12+channel] = 1
    for index, right in enumerate('KQkq'):
        features[768+index] = float(right in canonical.castling_xfen())
    if canonical.ep_square is not None:
        features[772+chess.square_file(canonical.ep_square)] = 1
    return features


def _sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def parse_connectome(raw: bytes, log_gain: torch.Tensor) -> torch.Tensor:
    """Transpose outgoing binary CSR into weighted incoming CPU CSR.

    An edge's count supplies only its sign; its learned magnitude is exp(gain).
    Stable destination sorting retains original source/edge order, including
    duplicate edges. It never replaces learned gains with anatomical counts.
    """
    if len(raw) < 12:
        raise ValueError('Truncated connectome header')
    _, neurons, edges = struct.unpack_from('<III', raw)
    if raw[:4] != b'CFLY':
        raise ValueError('Invalid connectome magic')
    if neurons < 1 or len(raw) != 12+4*(neurons+1)+6*edges:
        raise ValueError('Connectome byte size does not match its header')
    rows = np.frombuffer(raw, dtype='<u4', count=neurons+1, offset=12)
    destinations = np.frombuffer(raw, dtype='<u4', count=edges, offset=12+4*(neurons+1))
    counts = np.frombuffer(raw, dtype='<i2', count=edges, offset=12+4*(neurons+1)+4*edges)
    if rows[0] != 0 or rows[-1] != edges or np.any(rows[1:] < rows[:-1]):
        raise ValueError('Invalid connectome row offsets')
    if np.any(destinations >= neurons):
        raise ValueError('Connectome destination outside neuron range')
    if log_gain.shape != (edges,) or not torch.isfinite(log_gain).all():
        raise ValueError('Invalid learned edge gains')
    source = np.repeat(np.arange(neurons, dtype=np.int32), np.diff(rows).astype(np.int64))
    order = np.argsort(destinations, kind='stable')
    incoming_rows = np.concatenate(([0], np.cumsum(np.bincount(destinations, minlength=neurons))))
    # JS Math.exp evaluates in double before assigning to a Float32Array.
    with np.errstate(over='ignore'):
        magnitude = np.exp(log_gain.detach().cpu().numpy().astype(np.float64))
    weights = (np.where(counts < 0, -1., 1.)*magnitude).astype(np.float32)
    if not np.isfinite(weights).all():
        raise ValueError('Learned edge magnitudes overflow float32')
    return torch.sparse_csr_tensor(
        torch.from_numpy(incoming_rows.astype(np.int64)),
        torch.from_numpy(source[order].astype(np.int64)),
        torch.from_numpy(weights[order]), size=(neurons, neurons), dtype=torch.float32,
        check_invariants=False,  # Indices checked above; valid multiedges are retained.
    )


class ChessFlyNetwork:
    """Inference-only settling network; accepts small fixtures for validation."""

    def __init__(self, graph: torch.Tensor, groups: np.ndarray,
                 tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]):
        if graph.layout != torch.sparse_csr or graph.device.type != 'cpu':
            raise ValueError('ChessFly requires a CPU sparse CSR graph')
        if graph.dtype != torch.float32 or graph.shape[0] != graph.shape[1]:
            raise ValueError('Graph must be square and float32')
        self.neurons = graph.shape[0]
        if groups.shape != (self.neurons,) or not np.isin(groups, [0, 1, 2, 3]).all():
            raise ValueError('Invalid neuron group labels')
        self.inputs = torch.from_numpy(np.flatnonzero(groups == 2))
        self.readout = torch.from_numpy(np.flatnonzero((groups != 1) & (groups != 2)))
        self.steps = int(metadata['steps'])
        self.hidden = int(metadata['hidden'])
        self.alpha = float(metadata['alpha'])
        if self.steps < 1 or self.hidden < 1 or not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError('Invalid model recurrence metadata')
        expected = {
            'scale': (self.steps, self.neurons), 'shift': (self.steps, self.neurons),
            'encoder.weight': (len(self.inputs), FEATURES), 'encoder.bias': (len(self.inputs),),
            'decoder.weight': (self.hidden, len(self.readout)), 'decoder.bias': (self.hidden,),
            'policy.weight': (len(ACTIONS), self.hidden), 'policy.bias': (len(ACTIONS),),
            'value.weight': (VALUE_BINS, self.hidden), 'value.bias': (VALUE_BINS,),
        }
        self.weights: dict[str, torch.Tensor] = {}
        for name, shape in expected.items():
            tensor = tensors[name].detach().to(device='cpu', dtype=torch.float32).contiguous()
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise ValueError(f'Invalid weight tensor: {name}; expected {shape}')
            self.weights[name] = tensor
        if not torch.isfinite(graph.values()).all():
            raise ValueError('Nonfinite graph weights')
        self.graph = graph

    @torch.inference_mode()
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [batch,1968] move logits and [batch,64] value-bin logits."""
        if (features.ndim != 2 or features.shape[1] != FEATURES or features.shape[0] < 1
                or features.device.type != 'cpu' or features.dtype != torch.float32
                or not torch.isfinite(features).all()):
            raise ValueError('Expected a nonempty finite CPU float32 [batch,780] input')
        weights = self.weights
        encoded = F.linear(features, weights['encoder.weight'], weights['encoder.bias'])
        drive = torch.zeros((self.neurons, features.shape[0]), dtype=torch.float32)
        drive[self.inputs] = encoded.T
        state = torch.zeros_like(drive)
        for step in range(self.steps):
            pre = (torch.sparse.mm(self.graph, state)+drive)*weights['scale'][step, :, None]
            pre += weights['shift'][step, :, None]
            state = (1-self.alpha)*state+self.alpha*torch.relu(pre)
        decoded = F.linear(state[self.readout].T, weights['decoder.weight'], weights['decoder.bias'])
        hidden = F.gelu(decoded, approximate='tanh')
        logits = F.linear(hidden, weights['policy.weight'], weights['policy.bias'])
        value = F.linear(hidden, weights['value.weight'], weights['value.bias'])
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise ValueError('ChessFly inference produced nonfinite outputs')
        return logits, value


def _checked_raw(asset_dir: Path, meta: dict[str, Any], filename: str) -> bytes:
    path = asset_dir/filename
    expected = meta['files'][filename]
    if path.stat().st_size != expected['bytes']:
        raise ValueError(f'Compressed asset size mismatch: {filename}')
    with gzip.open(path, 'rb') as handle:
        raw = handle.read(expected['rawBytes']+1)
    if len(raw) != expected['rawBytes'] or hashlib.sha256(raw).hexdigest() != expected['rawSha256']:
        raise ValueError(f'Asset checksum mismatch: {filename}')
    return raw


class ChessFlyPolicy:
    """Pinned published neural policy, deterministic legal argmax, no search.

    ``asset_dir`` must contain meta.json, flynet.safetensors, connectome.bin.gz,
    and neurons.bin.gz from the pinned Space revision above. Loading is offline.
    ``threads`` explicitly configures PyTorch CPU threads for the host process.
    """

    def __init__(self, asset_dir: str | Path, *, threads: int = 2):
        if type(threads) is not int or threads < 1:
            raise ValueError('threads must be a positive integer')
        torch.set_num_threads(threads)
        asset_dir = Path(asset_dir)
        if _sha256(asset_dir/'meta.json') != META_SHA256:
            raise ValueError('Metadata does not match the pinned ChessFly Space')
        weights_hash = _sha256(asset_dir/'flynet.safetensors')
        if not WEIGHTS_SHA256 or weights_hash != WEIGHTS_SHA256:
            raise ValueError('Weights do not match the pinned ChessFly Space')
        meta = json.loads((asset_dir/'meta.json').read_text())
        with safe_open(asset_dir/'flynet.safetensors', framework='pt', device='cpu') as handle:
            model_meta = handle.metadata()
            # SafeTensor handles expose keys(), not dictionary iteration.
            tensors = {key: handle.get_tensor(key).float() for key in handle.keys()}  # noqa: SIM118
        raw_graph = _checked_raw(asset_dir, meta, 'connectome.bin.gz')
        raw_neurons = _checked_raw(asset_dir, meta, 'neurons.bin.gz')
        neurons = int(meta['neurons'])
        if len(raw_neurons) != 13*neurons:
            raise ValueError('Neuron file byte size mismatch')
        groups = np.frombuffer(raw_neurons, dtype=np.uint8, count=neurons, offset=12*neurons)
        graph = parse_connectome(raw_graph, tensors['log_gain'])
        if graph.shape != (neurons, neurons) or graph._nnz() != meta['edges']:
            raise ValueError('Graph dimensions disagree with pinned metadata')
        self.network = ChessFlyNetwork(graph, groups, tensors, model_meta)
        if self.network.steps != 5 or len(self.network.inputs) != meta['inputs']:
            raise ValueError('Model dimensions disagree with published architecture')
        self.metadata = {
            'policy_kind': 'published_chessfly_neural_policy_no_search',
            'model': 'mlabonne/chessfly', 'model_revision': MODEL_REVISION,
            'asset_space_revision': SPACE_REVISION, 'weights_sha256': weights_hash,
            'meta_sha256': META_SHA256, 'audited_worker_sha256': WORKER_SHA256,
            'graph_sha256': meta['files']['connectome.bin.gz']['rawSha256'],
            'neuron_groups_sha256': meta['files']['neurons.bin.gz']['rawSha256'],
            'external_license': 'FlyWire non-commercial terms; model card license: other',
            'device': 'cpu', 'dtype': 'float32', 'threads': threads,
            'neurons': neurons, 'edges': graph._nnz(), 'inputs': len(self.network.inputs),
            'readout_neurons': len(self.network.readout), 'features': FEATURES,
            'actions': len(ACTIONS), 'recurrent_steps': self.network.steps,
            'alpha': self.network.alpha, 'hidden': self.network.hidden,
            'search_depth': 0, 'carry_state_between_moves': False,
            'model_metadata': model_meta,
            'parity_scope': 'same mathematical recurrence; float32 reduction order differs',
        }

    def evaluate_boards(self, boards: Sequence[chess.Board]) -> list[dict[str, Any]]:
        """Score every legal move in each board, normalizing only over legal IDs."""
        if not boards:
            raise ValueError('At least one board is required')
        features = torch.stack([encode_board(board) for board in boards])
        legal_sets = [sorted(move.uci() for move in board.legal_moves) for board in boards]
        if any(not legal for legal in legal_sets):
            raise ValueError('ChessFly policy called on a board without legal moves')
        policy_logits, value_logits = self.network.forward(features)
        results = []
        bin_centers = (torch.arange(VALUE_BINS, dtype=torch.float64)+.5)/VALUE_BINS
        for index, (board, legal) in enumerate(zip(boards, legal_sets, strict=True)):
            indices = [ACTION_INDEX[move if board.turn else mirror_uci(move)] for move in legal]
            probabilities = torch.softmax(policy_logits[index, indices].double(), dim=0).tolist()
            value_probabilities = torch.softmax(value_logits[index].double(), dim=0)
            distribution = dict(zip(legal, probabilities, strict=True))
            results.append({
                'choice': max(distribution, key=distribution.get),
                'probabilities': distribution,
                'win_probability': float((value_probabilities*bin_centers).sum()),
                'win_probability_semantics': 'uncalibrated expected value-bin center for side to move',
                'policy_kind': 'published_chessfly_neural_policy_no_search',
                'recurrent_steps': self.network.steps, 'search_depth': 0,
            })
        return results

    def __call__(self, board: chess.Board) -> dict[str, Any]:
        return self.evaluate_boards([board])[0]

    def close(self) -> None:
        """No external engine or process is opened by this CPU policy."""
