"""Inspect all completed connectome fits without inference or new training.

Parameter movement can rule out an unchanged module. It cannot establish that
the module affects decisions or explain a failed topology comparison.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research import chess_connectome_study as study

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = 'evidence/chess-connectome-v1/results/summary.json'
SUMMARY_SHA = '7c96b00490b7af01e630c647cb63a1d7a99f084d326edd4809d4e89ee983eae0'
PLAN = 'evidence/chess-connectome-v1/protocol/plan.json'
PLAN_SHA = '5543877d88d2c00b7085c93f2f01602f2d0b3439486b05661d1a8511af7f81db'
EXECUTION = 'runs/chess-connectome-v1/execution'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def stats(value):
    require(isinstance(value, torch.Tensor) and value.numel() > 0 and torch.isfinite(value).all(),
            'Statistics require finite nonempty tensors')
    value = value.detach().cpu().double().flatten()
    return {'count': value.numel(), 'mean': float(value.mean()),
            'rms': float(value.square().mean().sqrt()), 'min': float(value.min()),
            'p01': float(torch.quantile(value, .01)), 'median': float(value.median()),
            'p99': float(torch.quantile(value, .99)), 'max': float(value.max())}


def parameter_summary(state, seed):
    weight, bias = state['input_projection.weight'], state['input_projection.bias']
    output = state['output_projection.weight']
    require(weight.ndim == 4 and weight.shape[2:] == (1, 1) and weight.dtype == torch.float32
            and bias.shape == weight.shape[:1] and output.shape == (weight.shape[1], weight.shape[0], 1, 1),
            'Projection dimensions differ')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    initial_weight = torch.empty_like(weight)
    nn.init.kaiming_uniform_(initial_weight, a=math.sqrt(5), generator=generator)
    initial_bias = torch.empty_like(bias)
    bound = 1/math.sqrt(weight.shape[1])
    nn.init.uniform_(initial_bias, -bound, bound, generator=generator)
    gains = F.softplus(state['edge_log_gain'])
    slots = state['node_slots']
    count = 64*weight.shape[0]
    require(slots.dtype == torch.long and slots.ndim == 1 and ((slots >= 0) & (slots < count)).all(),
            'Invalid slot assignment')
    occupancy = torch.bincount(slots, minlength=count)
    require(torch.equal(occupancy.to(state['slot_occupancy'].dtype), state['slot_occupancy']),
            'Slot occupancy mismatch')
    result = {'edge_magnitude': stats(gains), 'edge_magnitude_change_from_one': stats(gains-1),
              'fraction_edge_magnitudes_changed_over_1e_6': float(((gains-1).abs() > 1e-6).double().mean()),
              'input_weight_change': stats(weight-initial_weight), 'input_bias_change': stats(bias-initial_bias),
              'output_weight': stats(output), 'output_bias': stats(state['output_projection.bias']),
              'node_bias': stats(state['node_bias']), 'input_singular_values': torch.linalg.svdvals(weight[:, :, 0, 0].double()).tolist(),
              'output_singular_values': torch.linalg.svdvals(output[:, :, 0, 0].double()).tolist(),
              'slot_occupancy': stats(occupancy), 'uncovered_slots': int((occupancy == 0).sum()),
              'output_projection_is_zero': bool((output == 0).all() and (state['output_projection.bias'] == 0).all())}
    return result


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    write(out/'started.json', {'status': 'started', 'created_unix': time.time(), 'saved_weights_only': True})
    try:
        torch.set_num_threads(2)
        require(sha(ROOT/SUMMARY) == SUMMARY_SHA and sha(ROOT/PLAN) == PLAN_SHA, 'Original result or plan changed')
        summary, plan = read(ROOT/SUMMARY), read(ROOT/PLAN)
        receipt_path = ROOT/EXECUTION/'completed.json'
        require(sha(receipt_path) == summary['execution_receipt_sha256'], 'Original execution receipt changed')
        receipt = read(receipt_path)
        require(receipt['status'] == 'completed' and receipt['plan_sha256'] == PLAN_SHA,
                'Original execution is incomplete or rebound')
        expected = {c['name']: c for c in study.configurations()}
        require(len(expected) == 18 and {c['name'] for c in plan['configurations']} == set(expected), 'Original fit panel differs')
        records, bindings = [], {SUMMARY: SUMMARY_SHA, PLAN: PLAN_SHA,
                                 f'{EXECUTION}/completed.json': sha(receipt_path)}
        for name, config in expected.items():
            prefix = f'fits/{name}'
            for file in ('training.json', 'weights.pt'):
                relative = f'{prefix}/{file}'
                require(sha(ROOT/EXECUTION/relative) == receipt['files'][relative], 'Bound fit file changed')
                bindings[f'{EXECUTION}/{relative}'] = receipt['files'][relative]
            training = read(ROOT/EXECUTION/prefix/'training.json')
            payload = torch.load(ROOT/EXECUTION/prefix/'weights.pt', map_location='cpu', weights_only=True)
            require(training['status'] == 'completed' and training['configuration'] == config
                    and payload['configuration'] == config and payload['plan_sha256'] == PLAN_SHA
                    and training['plan_sha256'] == PLAN_SHA and training['backbone_unchanged'] is True
                    and payload['source_sha256'] == sha(study.__file__)
                    and study._state_sha(payload['state_dict']) == training['state_sha256'] == payload['state_sha256'],
                    'Saved checkpoint identity or tensor hash differs')
            records.append({'configuration': name, 'variant': config['variant'], 'seed': config['seed'],
                            'parameters': training['parameter_counts'], **parameter_summary(payload['state_dict'], config['seed'])})
        result = {'status': 'completed', 'fits': records, 'bound_inputs': bindings,
                  'original_topology_result_unchanged': True, 'new_model_calls': 0, 'new_engine_calls': 0,
                  'new_training_updates': 0,
                  'scope': 'Posthoc static parameter audit of every original fit. Nonzero parameters do not prove causal decision use, beneficial representations or a mapping bottleneck. Norms are not comparable measures of functional importance across differently sized models.'}
        require(all(sha(ROOT/p) == h for p, h in bindings.items()), 'Inputs changed during audit')
        write(out/'summary.json', result)
        write(out/'receipt.json', {'status': 'completed', 'completed_unix': time.time(),
              'wall_seconds': time.perf_counter()-begun, 'summary_sha256': sha(out/'summary.json'),
              'sources': {p: sha(ROOT/p) for p in ('scripts/chess_connectome_parameter_audit.py',
                         'tests/test_chess_connectome_parameter_audit.py', 'src/openjev/research/chess_connectome_study.py')}})
        return {'status': 'completed', 'fits': len(records), 'output_zero': sum(r['output_projection_is_zero'] for r in records),
                'wall_seconds': time.perf_counter()-begun}
    except BaseException as exc:
        write(out/'failed.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().out), allow_nan=False))
