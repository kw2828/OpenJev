"""Synthetic resource preflight only: backward passes, no parameter updates."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import chess
import torch
from torch.nn import functional as F

from openjev.research.chess_candidate import ARMS, CandidateChess, encode_batch

ROOT = Path(__file__).resolve().parents[1]
MICROBATCH = 16
PREFIX = ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    if not torch.backends.mps.is_available():
        raise RuntimeError('This prospective run requires MPS')
    torch.set_num_threads(2)
    positions = []
    board = chess.Board()
    for ply in range(MICROBATCH):
        positions.append(board.copy())
        if ply % len(PREFIX) == len(PREFIX)-1:
            board = chess.Board()
        else:
            board.push_uci(PREFIX[ply % len(PREFIX)])
    result = {'status': 'completed', 'scope': 'Synthetic memory/timing preflight, no optimizer or benchmark labels.',
              'microbatch_positions': MICROBATCH, 'candidate_chunk_size': 128, 'warmups': 2, 'samples': 3,
              'platform': platform.platform(), 'torch': torch.__version__, 'python': platform.python_version(),
              'sources': {p: sha(ROOT/p) for p in ('scripts/preflight_chess_candidate.py',
                          'src/openjev/research/chess_candidate.py', 'src/openjev/research/chess_anchor.py',
                          'src/openjev/research/chess_spatial.py')},
              'positions': [b.fen(en_passant='fen') for b in positions], 'arms': {},
              'memory_scope': 'Allocated bytes sampled after forward and backward, not a measured peak.'}
    for arm in ARMS:
        model = CandidateChess(arm, seed=97).to('mps').train()
        before = state_hash(model)
        started = time.perf_counter()
        batch, menus = encode_batch(positions, arm, device='mps')
        torch.mps.synchronize()
        preparation = time.perf_counter()-started
        targets = torch.zeros(MICROBATCH, dtype=torch.long, device='mps')
        values = torch.zeros(MICROBATCH, device='mps')
        timings, allocations = [], []
        for iteration in range(5):
            model.zero_grad(set_to_none=True)
            torch.mps.synchronize()
            started = time.perf_counter()
            logits, predicted, _ = model(**batch)
            loss = F.cross_entropy(logits, targets) + .5*F.mse_loss(predicted, values)
            forward_bytes = torch.mps.current_allocated_memory()
            loss.backward()
            torch.mps.synchronize()
            elapsed = time.perf_counter()-started
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                             for p in model.parameters()):
                raise ValueError('Nonfinite synthetic forward/backward')
            if iteration >= 2:
                timings.append(elapsed)
                allocations.append({'after_forward': forward_bytes,
                                    'after_backward': torch.mps.current_allocated_memory()})
        if state_hash(model) != before:
            raise ValueError('Synthetic preflight mutated model weights')
        result['arms'][arm] = {'seconds': timings, 'median_seconds': statistics.median(timings),
                               'preparation_seconds': preparation, 'sampled_allocations': allocations,
                               'positions': MICROBATCH, 'legal_candidates': sum(map(len, menus)),
                               'parameters': model.parameter_counts(), 'weights_unchanged': True}
        del model, batch, logits, predicted, loss
        torch.mps.empty_cache()
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({arm: {key: value for key, value in row.items() if key != 'sampled_allocations'}
                      for arm, row in result['arms'].items()}, indent=2))
