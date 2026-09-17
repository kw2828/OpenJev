"""One bounded resource/implementation preflight on full-size graph, synthetic inputs.

No pretrained model, real board, benchmark label, inference choice, or graph/weight
serialization is used. Six fresh subprocess cases perform at most three optimizer
updates each. Any failure stops the remaining cases; there is no retry or device
fallback. Observed timings include audit overhead and possible concurrent work.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from openjev.research.chess_candidate import CandidateChess
from openjev.research.chess_connectome_adapter import INITIAL_MAGNITUDE, ConnectomeChessAdapter
from openjev.research.connectome_graph import graph_sha256, induced_subgraph, load_pinned_graph

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'openjev-connectome-adapter-synthetic-resource-preflight-v1'
UPDATES = 3
CASE_TIMEOUT_SECONDS = 180
BACKBONE_SEED, ADAPTER_SEED, MAPPING_SEED, INPUT_SEED = 211, 223, 227, 229
CASES = tuple(
    {'id': f'{device}-{mode}', 'device': device, 'mode': mode, 'batch_size': batch}
    for device, batch in (('cpu', 8), ('mps', 128))
    for mode in ('sparse', 'dense', 'node_local')
)
SOURCES = (
    'scripts/preflight_chess_connectome_adapter.py',
    'src/openjev/research/chess_connectome_adapter.py',
    'src/openjev/research/connectome_graph.py',
    'src/openjev/research/chess_candidate.py',
    'src/openjev/research/chess_anchor.py',
    'src/openjev/research/chess_spatial.py', 'pyproject.toml', 'uv.lock',
)
ASSETS = ('meta.json', 'connectome.bin.gz', 'neurons.bin.gz')
AUDIT_COMMAND = (
    '.venv/bin/python scripts/chessbench_transfer_study.py report '
    '--plan evidence/chessbench-transfer-v1/protocol/plan.json '
    '--execution runs/chessbench-transfer-v1/execution '
    '--out runs/chessbench-transfer-v1/report'
)


def now():
    return datetime.now(UTC).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def journal(path, value):
    with Path(path).open('a') as handle:
        handle.write(canonical({'utc': now(), **value}).decode()+'\n')
        handle.flush()
        os.fsync(handle.fileno())


def source_hashes():
    return {name: sha(ROOT/name) for name in SOURCES}


def asset_hashes(asset_dir):
    return {name: {'bytes': (asset_dir/name).stat().st_size, 'sha256': sha(asset_dir/name)}
            for name in ASSETS}


def configure_cpu():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


def environment():
    names = ('PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_HIGH_WATERMARK_RATIO',
             'PYTORCH_MPS_LOW_WATERMARK_RATIO', 'PYTORCH_MPS_FAST_MATH',
             'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    return {
        'python': sys.version, 'executable': sys.executable,
        'system': platform.system(), 'release': platform.release(), 'machine': platform.machine(),
        'torch': torch.__version__, 'numpy': np.__version__, 'logical_cpus': os.cpu_count(),
        'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'default_dtype': str(torch.get_default_dtype()),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available(),
        'environment_flags': {name: os.environ.get(name) for name in names},
    }


def process_note(pid):
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    except PermissionError:
        alive = None
    return {
        'reported_concurrent_audit_pid': pid, 'pid_exists_at_check': alive,
        'command_supplied_by_parent': AUDIT_COMMAND,
        'other_work_supplied_by_parent': 'Report audit completed before launch; archive packaging may overlap this preflight',
        'timing_limit': 'Concurrent report-audit and other host work can interfere; timings are not isolated benchmarks',
    }


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    header = {'shape': list(value.shape), 'dtype': str(value.dtype)}
    return hashlib.sha256(canonical(header)+b'\0'+value.numpy().tobytes()).hexdigest()


def state_hash(model):
    return digest({name: tensor_hash(value) for name, value in sorted(model.state_dict().items())})


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def memory(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'process_peak_rss_bytes': int(rss if platform.system() == 'Darwin' else rss*1024)}
    if device == 'mps':
        result.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    return result


def graph_only(asset_dir):
    complete = load_pinned_graph(asset_dir)
    graph = induced_subgraph(complete, complete.node_ids[complete.groups == 3])
    if len(graph.node_ids) != 1409 or len(graph.sources) != 44090:
        raise ValueError('Exact descending-group graph does not match audited dimensions')
    receipt = {
        'selection': 'All and only packed neurons with coarse group 3 (descending)',
        'nodes': len(graph.node_ids), 'edges': len(graph.sources),
        'positive_edges': int(np.count_nonzero(graph.signs > 0)),
        'negative_edges': int(np.count_nonzero(graph.signs < 0)),
        'graph_sha256': graph_sha256(graph),
        'selected_node_ids_sha256': hashlib.sha256(graph.node_ids.astype('<i8').tobytes()).hexdigest(),
        'provenance': dict(graph.provenance),
        'graph_assets_or_weights_saved': False,
    }
    del complete
    gc.collect()
    return graph, receipt


def synthetic_inputs(batch_size):
    generator = torch.Generator(device='cpu').manual_seed(INPUT_SEED)
    observations = torch.randn((batch_size, 19, 8, 8), generator=generator)*0.1
    candidate_row = torch.tensor([
        [0, 8, 0, 7, 8], [1, 18, 0, 8, 9], [6, 21, 0, 6, 9], [7, 15, 0, 7, 8],
        [8, 16, 0, 7, 8], [9, 17, 0, 7, 8], [10, 18, 0, 7, 8], [11, 19, 0, 7, 8],
    ], dtype=torch.long)
    candidates = candidate_row.unsqueeze(0).expand(batch_size, -1, -1).clone()
    values = {
        'observations': observations, 'candidates': candidates,
        'legal_mask': torch.ones((batch_size, 8), dtype=torch.bool),
        'policy_targets': torch.arange(batch_size) % 8,
        'value_targets': torch.zeros(batch_size),
    }
    return values, {key: tensor_hash(value) for key, value in values.items()}


def gradient_receipt(model):
    grouped = {}
    for name, parameter in model.named_parameters():
        if name.startswith('backbone.'):
            if parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError('Frozen backbone received a gradient or became trainable')
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f'Missing or nonfinite trainable gradient: {name}')
        group = name.split('.')[0]
        grouped[group] = grouped.get(group, 0.0)+float(parameter.grad.detach().square().sum().cpu())
    result = {group: value**0.5 for group, value in grouped.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError('Nonfinite gradient norm')
    return result


def run_case(protocol_path, case_index):
    configure_cpu()
    protocol = json.loads(protocol_path.read_text())
    case = protocol['cases'][case_index]
    out = Path(protocol['output_directory'])/'cases'/case['id']
    progress = out/'events.jsonl'
    started = time.perf_counter()
    receipt = {
        'version': VERSION, 'case': case, 'status': 'running', 'started_utc': now(),
        'protocol_sha256': sha(protocol_path), 'optimizer_updates_started': 0,
        'optimizer_updates_completed': 0, 'updates': [],
        'environment': environment(), 'source_sha256': source_hashes(),
    }
    receipt['environment_sha256'] = digest(receipt['environment'])
    stage = 'bindings'
    try:
        if receipt['source_sha256'] != protocol['source_sha256']:
            raise ValueError('Source hashes changed before case start')
        if asset_hashes(Path(protocol['asset_directory'])) != protocol['assets']:
            raise ValueError('Graph asset hashes changed before case start')
        journal(progress, {'stage': stage, 'status': 'passed'})
        device = case['device']
        stage = 'device'
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no CPU fallback is permitted')
        if device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None, '0'):
            raise RuntimeError('MPS fallback is enabled or ambiguous; refusing this case without altering the environment')
        stage = 'graph_only_load'
        stamp = time.perf_counter()
        graph, receipt['graph'] = graph_only(Path(protocol['asset_directory']))
        receipt['graph_load_validation_seconds'] = time.perf_counter()-stamp
        if receipt['graph'] != protocol['graph']:
            raise ValueError('Induced graph differs from bound graph receipt')
        stage = 'random_model_and_synthetic_inputs'
        stamp = time.perf_counter()
        backbone = CandidateChess('direct', seed=BACKBONE_SEED, width=32, root_depth=4).eval()
        original_hash = state_hash(backbone)
        model = ConnectomeChessAdapter(
            backbone, graph, mode=case['mode'], seed=ADAPTER_SEED,
            mapping_seed=MAPPING_SEED, slot_channels=16, steps=4, alpha=0.5,
        ).to(device)
        before = state_hash(model.backbone)
        if original_hash != before:
            raise RuntimeError('Adapter construction changed copied backbone weights')
        values, receipt['synthetic_input_tensor_sha256'] = synthetic_inputs(case['batch_size'])
        values = {key: value.to(device) for key, value in values.items()}
        inputs = {key: values[key] for key in ('observations', 'candidates', 'legal_mask')}
        receipt.update(config=model.config, parameter_counts=model.parameter_counts(),
                       forward_computation=model.computation_counts(case['batch_size'], 8),
                       backbone_sha256_before=before,
                       backbone_matches_fresh_random_source=True)
        sync(device)
        receipt['random_model_and_inputs_seconds'] = time.perf_counter()-stamp
        del backbone, graph
        stage = 'exact_zero_initialization'
        with torch.no_grad():
            baseline = model.backbone(**inputs)
            adapted = model(**inputs)
        receipt['zero_init_exact_equal'] = {
            name: bool(torch.equal(left, right))
            for name, left, right in zip(('logits', 'value', 'hidden'), baseline, adapted, strict=True)
        }
        if not all(receipt['zero_init_exact_equal'].values()):
            raise RuntimeError('Zero-initialized adapter differs from unchanged direct model')
        del baseline, adapted
        model.train()
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
        )
        journal(progress, {'stage': stage, 'status': 'passed', 'memory': memory(device)})
        stage = 'synthetic_optimizer_updates'
        for index in range(UPDATES):
            sync(device)
            stamp = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits, value, _ = model(**inputs)
            loss = F.cross_entropy(logits, values['policy_targets'])
            loss = loss+0.5*F.mse_loss(value, values['value_targets'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite synthetic loss')
            loss.backward()
            norms = gradient_receipt(model)
            if index == 0 and norms.get('output_projection', 0) == 0:
                raise RuntimeError('Output projection has no initial learning signal')
            receipt['optimizer_updates_started'] += 1
            journal(progress, {'stage': stage, 'update': index+1, 'event': 'optimizer_step_started'})
            optimizer.step()
            sync(device)
            receipt['optimizer_updates_completed'] += 1
            journal(progress, {'stage': stage, 'update': index+1, 'event': 'optimizer_step_completed'})
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise RuntimeError('Nonfinite parameters after optimizer update')
            update = {'update': index+1, 'synthetic_loss_before_update': float(loss.detach().cpu()),
                      'gradient_l2': norms, 'observed_update_audit_seconds': time.perf_counter()-stamp,
                      'memory': memory(device)}
            receipt['updates'].append(update)
            journal(progress, {'stage': stage, **update})
            print(json.dumps({'case': case['id'], 'update': index+1, 'status': 'passed'}), flush=True)
        if not all(any(update['gradient_l2'].get(group, 0) > 0 for update in receipt['updates'][1:])
                   for group in ('input_projection', 'edge_log_gain', 'node_bias')):
            raise RuntimeError('An upstream adapter component has no post-initialization gradient')
        stage = 'final_immutability_and_bindings'
        after = state_hash(model.backbone)
        receipt.update(backbone_sha256_after=after, backbone_hash_unchanged=before == after,
                       backbone_stayed_eval=not model.backbone.training,
                       final_memory=memory(device), adapter_forward_calls=UPDATES+1,
                       direct_baseline_forward_calls=1)
        if before != after or model.backbone.training:
            raise RuntimeError('Frozen backbone changed')
        if source_hashes() != protocol['source_sha256']:
            raise ValueError('Source hashes changed during the case')
        receipt['status'] = 'completed'
    except Exception as error:  # noqa: BLE001 - preserve terminal failure receipts, never retry
        receipt.update(status='failed', failure_stage=stage,
                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    receipt.update(ended_utc=now(), observed_case_seconds=time.perf_counter()-started)
    write_new(out/'receipt.json', receipt)
    return 0 if receipt['status'] == 'completed' else 1


def read_progress(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--asset-dir', type=Path, default=ROOT/'runs/chessfly-source-audit/assets')
    parser.add_argument('--audit-pid', type=int, default=4472)
    parser.add_argument('--case-index', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--protocol', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case_index is not None:
        if args.protocol is None or not 0 <= args.case_index < len(CASES):
            parser.error('Internal case requires bound protocol and valid index')
        return run_case(args.protocol.resolve(), args.case_index)
    if args.out is None or args.protocol is not None:
        parser.error('--out is required for a new preflight')
    configure_cpu()
    out, asset_dir = args.out.resolve(), args.asset_dir.resolve()
    out.mkdir(parents=False, exist_ok=False)
    (out/'cases').mkdir()
    started = time.perf_counter()
    write_new(out/'started.json', {'version': VERSION, 'started_utc': now(), 'pid': os.getpid()})
    env = environment()
    result = {'version': VERSION, 'status': 'running', 'cases': [], 'started_utc': now(),
              'interpretation': 'Synthetic resource/implementation feasibility only; no chess efficacy or clean speed comparison',
              'concurrent_work': process_note(args.audit_pid)}
    try:
        graph, graph_receipt = graph_only(asset_dir)
        del graph
        gc.collect()
        protocol = {
            'version': VERSION, 'output_directory': str(out), 'asset_directory': str(asset_dir),
            'source_sha256': source_hashes(), 'assets': asset_hashes(asset_dir), 'graph': graph_receipt,
            'environment': env, 'environment_sha256': digest(env), 'cases': CASES,
            'seeds': {'backbone': BACKBONE_SEED, 'adapter': ADAPTER_SEED,
                      'mapping': MAPPING_SEED, 'synthetic_inputs': INPUT_SEED},
            'updates_per_case': UPDATES, 'maximum_total_updates': len(CASES)*UPDATES,
            'case_timeout_seconds': CASE_TIMEOUT_SECONDS, 'initial_magnitude': INITIAL_MAGNITUDE,
            'training_inputs': 'Gaussian synthetic observations; fixed shaped candidate features and arbitrary targets; no boards or labels',
            'optimizer': {'name': 'Adam', 'lr': 0.001, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0},
            'no_retry_or_fallback': True, 'no_graph_or_weight_persistence': True,
            'concurrent_work': result['concurrent_work'],
        }
        protocol_path = out/'protocol.json'
        write_new(protocol_path, protocol)
        result['protocol_sha256'] = sha(protocol_path)
        stopped = False
        for index, case in enumerate(CASES):
            case_out = out/'cases'/case['id']
            case_out.mkdir()
            if stopped:
                receipt = {'status': 'not_run', 'case': case, 'reason': 'Stopped after prior failure; no retries',
                           'optimizer_updates_started': 0, 'optimizer_updates_completed': 0}
                write_new(case_out/'receipt.json', receipt)
            else:
                command = [sys.executable, str(Path(__file__).resolve()), '--protocol', str(protocol_path),
                           '--case-index', str(index)]
                launch = {'case': case, 'command': command, 'started_utc': now(), 'timeout_seconds': CASE_TIMEOUT_SECONDS}
                write_new(case_out/'launch.json', launch)
                timed_out = False
                with (case_out/'console.log').open('x') as log:
                    try:
                        completed = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT,
                                                   timeout=CASE_TIMEOUT_SECONDS, check=False,
                                                   start_new_session=True)
                        returncode = completed.returncode
                    except subprocess.TimeoutExpired:
                        timed_out, returncode = True, None
                if (case_out/'receipt.json').exists():
                    receipt = json.loads((case_out/'receipt.json').read_text())
                else:
                    events = read_progress(case_out/'events.jsonl')
                    receipt = {
                        'status': 'failed', 'case': case, 'failure_stage': 'child_process',
                        'timed_out': timed_out, 'returncode': returncode,
                        'error': 'Child exceeded the fixed wall limit or exited without a final receipt',
                        'optimizer_updates_started': sum(event.get('event') == 'optimizer_step_started' for event in events),
                        'optimizer_updates_completed': sum(event.get('event') == 'optimizer_step_completed' for event in events),
                    }
                    write_new(case_out/'receipt.json', receipt)
                stopped = receipt['status'] != 'completed' or returncode != 0 or timed_out
            result['cases'].append({
                'case': case, 'status': receipt['status'],
                'receipt': str((case_out/'receipt.json').relative_to(out)),
                'receipt_sha256': sha(case_out/'receipt.json'),
                'optimizer_updates_started': receipt['optimizer_updates_started'],
                'optimizer_updates_completed': receipt['optimizer_updates_completed'],
            })
            print(json.dumps(result['cases'][-1]), flush=True)
        result['status'] = 'completed' if all(case['status'] == 'completed' for case in result['cases']) else 'failed'
        if source_hashes() != protocol['source_sha256'] or asset_hashes(asset_dir) != protocol['assets']:
            raise ValueError('Final source/asset hashes differ from bound protocol')
    except Exception as error:  # noqa: BLE001 - preserve terminal failure receipts, never retry
        result.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    recorded = {entry['case']['id'] for entry in result['cases']}
    for case in CASES:
        if case['id'] in recorded:
            continue
        case_out = out/'cases'/case['id']
        case_out.mkdir(exist_ok=True)
        path = case_out/'receipt.json'
        if path.exists():
            receipt = json.loads(path.read_text())
        else:
            receipt = {'status': 'not_run', 'case': case, 'reason': 'Preflight stopped before this case; no retries',
                       'optimizer_updates_started': 0, 'optimizer_updates_completed': 0}
            write_new(path, receipt)
        result['cases'].append({
            'case': case, 'status': receipt['status'], 'receipt': str(path.relative_to(out)),
            'receipt_sha256': sha(path),
            'optimizer_updates_started': receipt['optimizer_updates_started'],
            'optimizer_updates_completed': receipt['optimizer_updates_completed'],
        })
    result.update(ended_utc=now(), observed_total_seconds=time.perf_counter()-started,
                  total_optimizer_updates_started=sum(case['optimizer_updates_started'] for case in result['cases']),
                  total_optimizer_updates_completed=sum(case['optimizer_updates_completed'] for case in result['cases']),
                  concurrent_work_at_end=process_note(args.audit_pid))
    write_new(out/'summary.json', result)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
