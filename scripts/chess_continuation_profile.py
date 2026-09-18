"""Engineering-only MPS update timing on a repeated synthetic start position."""
import argparse
import json
import statistics
import time
from pathlib import Path

import chess
import torch

from openjev.research.chess_continuation_data import tensorize
from openjev.research.chess_continuation_model import ContinuationChess, training_objective


def profile(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if not torch.backends.mps.is_available():
        raise ValueError('MPS is required for this profile')
    torch.set_num_threads(2)
    rows = [{'id': f'synthetic-{i}', 'game_id': 'synthetic', 'fen': chess.STARTING_FEN,
             'target_uci': 'e2e4', 'behavior_uci': 'd2d4', 'target_value': .1,
             'continuation_mask': True, 'continuation_value': -.15} for i in range(128)]
    tensors, _ = tensorize(rows)
    # Allocate a large padded menu so the easy starting board does not hide
    # candidate-head allocation. This is not representative gameplay timing.
    pad = 128-tensors['mask'].shape[1]
    tensors['candidates'] = torch.nn.functional.pad(tensors['candidates'], (0, 0, 0, pad))
    tensors['mask'] = torch.nn.functional.pad(tensors['mask'], (0, pad), value=False)
    results = []
    for objective in ('policy', 'best_value', 'continuation'):
        model = ContinuationChess(objective, 193).to('mps')
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        times = []
        for _ in range(12):
            torch.mps.synchronize()
            begin = time.perf_counter()
            batch = {k: v.to('mps') for k, v in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            result = training_objective(model, batch['observations'], batch['candidates'], batch['mask'],
                                        batch['targets'], batch['values'], batch['behavior_indices'],
                                        batch['continuation_values'], batch['continuation_mask'])
            result['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            # The study logs these scalars, so include their synchronization cost.
            for value in result.values():
                if isinstance(value, torch.Tensor):
                    float(value.detach().cpu())
            torch.mps.synchronize()
            times.append(time.perf_counter()-begin)
        results.append({'objective': objective, 'warmup_seconds': times[:4], 'update_seconds': times[4:],
                        'median_seconds': statistics.median(times[4:]), 'mean_seconds': statistics.mean(times[4:])})
        del model, optimizer
        torch.mps.empty_cache()
    result = {'status': 'completed', 'synthetic_only': True, 'real_training_rows': 0,
              'checkpoints_saved': 0, 'updates_per_objective': 12, 'timed_updates_per_objective': 8,
              'scope': 'Repeated artificial start-board targets with128 padded candidates. Shared-host engineering timing only; no predictive performance, gate result or empirical training estimate.',
              'results': results,
              'linear_update_projection_seconds': 5528*3*sum(r['mean_seconds'] for r in results),
              'projection_excludes': 'Dataset preparation, epochs/checkpoint overhead, actual menu distribution, evaluation, engine and games.'}
    with (out/'profile.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(profile(parser.parse_args().out), allow_nan=False), flush=True)
