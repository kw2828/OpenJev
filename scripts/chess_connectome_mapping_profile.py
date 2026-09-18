"""Bounded MPS engineering profile with synthetic graph, boards and targets only."""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import chess
import numpy as np
import torch
from torch.nn import functional as F

from openjev.research.chess_candidate import CandidateChess, encode_batch
from openjev.research.chess_connectome_interface import HardSquareConnectomeAdapter, evaluate_square_swap
from openjev.research.connectome_graph import SignedGraph, graph_sha256

ROOT = Path(__file__).resolve().parents[1]
WARMUP = 8
TIMED = 8
SOURCES = (
    'scripts/chess_connectome_mapping_profile.py', 'tests/test_chess_connectome_mapping_profile.py',
    'src/openjev/research/chess_connectome_interface.py',
    'src/openjev/research/chess_connectome_adapter.py',
    'src/openjev/research/chess_candidate.py', 'src/openjev/research/connectome_graph.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def synthetic_graph():
    nodes, edges = 1409, 44090
    index = np.arange(edges)
    sources = index % nodes
    return SignedGraph(node_ids=100+7*np.arange(nodes), groups=np.arange(nodes) % 4,
                       sources=sources, destinations=(sources+1+index//nodes) % nodes,
                       signs=np.where(index % 3 == 0, -1, 1),
                       provenance={'fixture': 'Synthetic directed offset graph; no biological asset'})


def synthetic_batch():
    inputs, _ = encode_batch([chess.Board() for _ in range(128)], 'direct')
    # Deliberate padded-menu stress case, not a sampled training distribution.
    padding = 128-inputs['legal_mask'].shape[1]
    inputs['candidates'] = F.pad(inputs['candidates'], (0, 0, 0, padding))
    inputs['legal_mask'] = F.pad(inputs['legal_mask'], (0, padding), value=False)
    return {**inputs, 'targets': torch.zeros(128, dtype=torch.long), 'values': torch.zeros(128)}


def projected_seconds(records):
    """Fixed prospective workload only; never an empirical runtime guarantee."""
    lookup = {(r['mode'], r['policy']): r['mean_seconds'] for r in records}
    expected = {(mode, policy) for mode in ('sparse', 'node_local')
                for policy in ('ordinary', 'fixed', 'learned')}
    if set(lookup) != expected or len(records) != len(expected) or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in lookup.values()):
        raise ValueError('Exactly six finite positive case means are required')
    total = 0.
    for mode, topology_count in (('sparse', 4), ('node_local', 1)):
        for policy in ('fixed', 'learned'):
            total += topology_count*3*(1216*lookup[mode, 'ordinary']+320*lookup[mode, policy])
    return total


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    write(out/'started.json', {'status': 'started', 'synthetic_only': True, 'started_unix': time.time()})
    old_threads = torch.get_num_threads()
    try:
        if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None, '0'):
            raise ValueError('MPS is required and neural CPU fallback must be disabled')
        torch.set_num_threads(2)
        graph = synthetic_graph()
        batch = {key: value.to('mps') for key, value in synthetic_batch().items()}
        results = []
        for mode in ('sparse', 'node_local'):
            for policy in ('ordinary', 'fixed', 'learned'):
                model = HardSquareConnectomeAdapter(CandidateChess('direct', seed=97, width=32), graph,
                    mode=mode, seed=97, mapping_seed=97, slot_channels=16, steps=4, alpha=.5).to('mps')
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(parameters, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
                generator = random.Random(11600041+97)
                elapsed, proposal_times = [], []
                for _ in range(WARMUP+TIMED):
                    torch.mps.synchronize()
                    started = time.perf_counter()
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    if policy != 'ordinary':
                        a, b = generator.sample(range(64), 2)
                        proposal = evaluate_square_swap(model, batch, a, b, policy=policy)
                        proposal_times.append(proposal['wall_seconds'])
                    logits, values, _ = model(batch['observations'], batch['candidates'], batch['legal_mask'])
                    loss = F.cross_entropy(logits, batch['targets'])+.5*F.mse_loss(values, batch['values'])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite synthetic loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    torch.mps.synchronize()
                    elapsed.append(time.perf_counter()-started)
                results.append({'mode': mode, 'policy': policy, 'warmup_seconds': elapsed[:WARMUP],
                    'update_seconds': elapsed[WARMUP:], 'proposal_wall_seconds_including_warmups': proposal_times,
                    'mean_seconds': statistics.mean(elapsed[WARMUP:]),
                    'median_seconds': statistics.median(elapsed[WARMUP:]),
                    'updates': WARMUP+TIMED, 'proposal_forward_calls': 0 if policy == 'ordinary' else 2*(WARMUP+TIMED)})
                del model, optimizer, parameters
                torch.mps.empty_cache()
        result = {'status': 'completed', 'synthetic_only': True, 'real_training_rows': 0,
                  'biological_asset_reads': 0, 'teacher_calls': 0, 'checkpoints_saved': 0,
                  'graph': {'nodes': 1409, 'edges': 44090, 'sha256': graph_sha256(graph),
                            'source': 'Deterministic synthetic offsets, no biological topology'},
                  'fixture': 'Repeated starting board with invented first-legal-move/zero-value targets;128 examples and128 padded candidate slots.',
                  'results': results, 'projected_update_loop_seconds': projected_seconds(results),
                  'projection_workload': '30fits,1536updates/fit,320proposals/fit;24sparse-mode and6node-local fits.',
                  'timing_scope': 'Update includes clearing gradients, optional complete proposal helper, forward/backward, clipping, Adam and device sync. Neural work on MPS; discrete mapping checks/copies on CPU.',
                  'limits': 'Shared-host synthetic engineering profile, not a measured real-study runtime, topology benefit or performance result. Excludes cache creation, model/optimizer setup, journals, epoch audits, checkpoints, evaluation and engine calls. Real menu distribution differs.',
                  'wall_seconds': time.perf_counter()-begun}
        write(out/'profile.json', result)
        write(out/'receipt.json', {'status': 'completed', 'profile_sha256': sha(out/'profile.json'),
                                  'sources': {name: sha(ROOT/name) for name in SOURCES}})
        return {'status': 'completed', 'cases': len(results),
                'synthetic_update_projection_seconds': result['projected_update_loop_seconds']}
    except BaseException as exc:
        write(out/'failed.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        torch.set_num_threads(old_threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().out)))
