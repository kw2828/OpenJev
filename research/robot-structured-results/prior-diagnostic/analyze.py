"""One frozen, post hoc gate intervention on closed robot development evidence."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STUDY = ROOT / 'output/robot-transition-study-v1'
ENG = ROOT / 'output/robot-transition-engineering-v1'
AUDIT = ROOT / 'output/robot-transition-audit-v1/audit.json'


def require(ok, label):
    if not ok:
        raise ValueError(label)


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular file: ' + str(path))
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def read(path):
    return json.loads(Path(path).read_text())


def authenticate():
    process = read(ENG / 'audit-process-01.json')
    require(process['returncode'] == 0 and process['source_unchanged'] is True,
            'closed original audit process')
    require(process['command'] == ['.venv/bin/python', 'scripts/audit_robot_transition.py',
            '--study', 'output/robot-transition-study-v1', '--run-receipt',
            'output/robot-transition-engineering-v1/run-process-01.json', '--output',
            'output/robot-transition-audit-v1/audit.json'], 'original audit command')
    require(pin(ROOT / 'scripts/audit_robot_transition.py')['sha256'] == process['auditor_sha256'], 'auditor unchanged')
    require(pin(AUDIT) == process['audit_output'] and pin(ENG / 'audit-process-01.log') == process['log'], 'audit output/log pins')
    sys.path.insert(0, str(ROOT / 'scripts'))
    import audit_robot_transition as admission
    plan, original_inputs = admission.authenticate(STUDY, ENG / 'run-process-01.json')
    result = read(AUDIT)
    require(result['status'] == 'PASS' and result['agreement'] is True
            and result['registration_sha256'] == admission.PLAN_SHA
            and result['study'] == str(STUDY) and result['source_pins'] == plan['sources'], 'original audit joins')
    require(read(AUDIT.parent / 'manifest.json') == {'files': {'audit.json': pin(AUDIT)}}, 'audit manifest')
    require({k: result['auditor'][k] for k in ('sha256', 'bytes')} == pin(ROOT / 'scripts/audit_robot_transition.py'), 'original auditor source')
    inputs = {}

    def add(path, expected=None):
        path = Path(path).resolve()
        actual = pin(path)
        require(expected is None or actual == expected, 'changed input: ' + str(path))
        inputs[str(path)] = actual

    def descriptors(value):
        if isinstance(value, dict):
            if {'path', 'sha256', 'bytes'} <= set(value):
                add(value['path'], {k: value[k] for k in ('sha256', 'bytes')})
            else:
                for item in value.values():
                    descriptors(item)
        elif isinstance(value, list):
            for item in value:
                descriptors(item)

    descriptors(original_inputs)
    descriptors(result['inputs'])
    descriptors(result['auditor'])
    for name, value in plan['sources'].items():
        add(ROOT / name, value)
        add(STUDY / 'sources' / name, value)
    for name, value in read(STUDY / 'manifest.json')['files'].items():
        add(STUDY / name, value)
    for row in read(plan['qualification']['path'])['commands']:
        add(row['log'])
    for path in (AUDIT, AUDIT.parent / 'manifest.json', ENG / 'audit-process-01.json',
                 ENG / 'audit-process-01.log', Path(__file__), OUT / 'definition.json'):
        add(path)
    return plan, inputs


def main():
    started = time.perf_counter()
    require(not (OUT / 'diagnostic.json').exists() and not (OUT / 'receipt.json').exists(), 'exclusive analysis')
    plan, pins = authenticate()
    definition = read(OUT / 'definition.json')
    require(definition['registration_sha256'] == pin(ROOT / 'research/robot-transition-registration.json')['sha256'], 'definition plan pin')
    import numpy as np
    import robot_transition_study as replay
    import torch
    torch.set_num_threads(1)
    counts = {'npz_opens': 0, 'array_decodes': 0, 'baseline_replays': 0,
              'intervention_rollouts': 0, 'training_calls': 0, 'raw_mat_reads': 0,
              'confirmation_reads': 0, 'official_test_reads': 0}
    reads = []

    def load(path):
        path = Path(path).resolve()
        require(str(path) in pins and pin(path) == pins[str(path)], 'admitted numeric input')
        with np.load(path, allow_pickle=False) as data:
            result = {key: data[key].copy(order='K') for key in data.files}
        counts['npz_opens'] += 1
        counts['array_decodes'] += len(result)
        reads.append({'path': str(path), **pins[str(path)], 'arrays': list(result)})
        return result

    def summary(values):
        a = np.asarray(values, dtype=np.float64)
        require(np.isfinite(a).all(), 'finite diagnostic statistic')
        return {'mean': a.mean(axis=0).tolist(), 'std': a.std(axis=0).tolist(),
                'min': a.min(axis=0).tolist(), 'max': a.max(axis=0).tolist()}

    norm = load(STUDY / 'normalizers.npz')
    cfg = plan['config']
    original = read(STUDY / 'results.json')
    require(original['selection']['selected_rates']['lpv_instant'] == .001
            and cfg['seeds'] == definition['seeds'] and cfg['partitions']['dev'] == definition['dev_recordings'], 'fixed selection/roster')
    batches = {}
    for name in definition['dev_recordings']:
        data = load(plan['data'][f'dev-data-{name}.npz']['path'])
        record = replay.old.normalized_record({'name': name, **data}, norm)
        starts = replay.old.dev_windows(len(record['q']), cfg)
        require(np.array_equal(starts, 64 + 160 * np.arange(22)), 'same22 windows')
        batch = replay.old.window_batch([record], {'record': np.zeros(22, dtype=np.int64), 'start': starts}, 32, 128)
        saved = load(STUDY / f'dev-windows-{name}.npz')
        require(np.array_equal(saved['target'], batch['target']) and np.array_equal(saved['starts'], starts), 'exact target/window joins')
        batches[name] = batch
    models, baseline, matrices, checks = {}, {}, [], []
    # All six exact baseline replays must pass before intervention code runs.
    for seed in definition['seeds']:
        key = f'lpv_instant-{seed}-lr0'
        state = load(STUDY / key / 'final.npz')
        model = replay.LPVAdapter('instant', seed)
        model.load_state_dict({k: torch.from_numpy(v.copy(order='K')) for k, v in state.items()}, strict=True)
        model.eval().requires_grad_(False)
        models[seed] = model
        with torch.no_grad():
            op = model.cell.operators()
            effective = op.matrix / op.scale[None, :, None] * op.scale[None, None, :]
            matrices.append({'seed': seed, 'raw_spectral_norms': op.raw_spectral_norm.tolist(),
                             'bounded_coordinate_spectral_norms': torch.linalg.matrix_norm(op.matrix, ord=2).tolist(),
                             'physical_coordinate_spectral_norms': torch.linalg.matrix_norm(effective, ord=2).tolist(),
                             'matrix_difference_frobenius': float(torch.linalg.matrix_norm(op.matrix[0] - op.matrix[1])),
                             'matrix_difference_spectral': float(torch.linalg.matrix_norm(op.matrix[0] - op.matrix[1], ord=2)),
                             'input_matrix_difference_frobenius': float(torch.linalg.matrix_norm(model.cell.input_matrix[0] - model.cell.input_matrix[1])),
                             'bias_difference_l2': float(torch.linalg.vector_norm(model.cell.expert_bias[0] - model.cell.expert_bias[1])),
                             'scale': op.scale.tolist()})
            for name, batch in batches.items():
                pred = replay.old.infer(model, batch).numpy().astype(np.float64)
                expected = load(STUDY / f'prediction-{name}-{key}.npz')['prediction']
                require(pred.dtype == expected.dtype and np.array_equal(pred, expected), 'exact baseline replay: ' + key + '/' + name)
                baseline[seed, name] = pred
                checks.append({'seed': seed, 'recording': name, 'exact_prediction_equality': True, 'scalars': int(pred.size)})
                counts['baseline_replays'] += 1
    require(counts['baseline_replays'] == 6, 'all exact replays first')

    def intervention(cell, batch, method):
        q, u, future = [torch.from_numpy(np.asarray(batch[key], dtype=np.float32))
                        for key in ('q_context', 'u_context', 'future_u')]
        state, op = cell.condition(q, u), cell.operators()
        x, outputs, gates = state.x, [], []
        for t in range(128):
            current = future[:, t]
            hidden = cell.scheduler(torch.cat((x[:, :6], current), -1), current.new_zeros(22, 8))
            dynamic = torch.softmax(cell.gate(hidden), -1)
            uniform = torch.full_like(dynamic, .5)
            matrix_mix = uniform if method in ('fixed_matrix', 'all_fixed') else dynamic
            force_mix = uniform if method in ('fixed_forcing', 'all_fixed') else dynamic
            state_expert = torch.einsum('eij,bj->bei', op.matrix, x * op.scale)
            forcing = torch.einsum('eij,bj->bei', cell.input_matrix, current) + cell.expert_bias
            if method == 'full_gate':
                # Exact original operation order, also checked below.
                expert = state_expert + torch.einsum('eij,bj->bei', cell.input_matrix, current) + cell.expert_bias
                x = (dynamic[:, :, None] * expert).sum(1) / op.scale
            else:
                x = ((matrix_mix[:, :, None] * state_expert).sum(1)
                     + (force_mix[:, :, None] * forcing).sum(1)) / op.scale
            require(bool(torch.isfinite(x).all()), 'nonfinite intervention, no repair')
            outputs.append(x[:, :6].clone())
            gates.append(dynamic)
        return torch.stack(outputs, 1).numpy().astype(np.float64), torch.stack(gates, 1).numpy()

    rows, gate_rows = [], []
    with torch.no_grad():
        for seed, model in models.items():
            for name, batch in batches.items():
                for method in definition['interventions']:
                    pred, gates = intervention(model.cell, batch, method)
                    if method == 'full_gate':
                        require(np.array_equal(pred, baseline[seed, name]), 'exact split-kernel baseline')
                    else:
                        counts['intervention_rollouts'] += 1
                    gate_rows.append({'seed': seed, 'recording': name, 'intervention': method,
                                      'initial_gate_statistics': summary(gates[:, 0]),
                                      'initial_gate_values': gates[:, 0].tolist(),
                                      'rollout_gate_statistics': summary(gates.reshape(-1, 2))})
                    for horizon in definition['horizons']:
                        error = pred[:, :horizon] - batch['target'][:, :horizon]
                        physical = error * norm['q_std']
                        rows.append({'seed': seed, 'recording': name, 'intervention': method,
                                     'horizon': horizon, 'windows': 22,
                                     'standardized_rmse': float(np.sqrt(np.mean(error**2))),
                                     'physical_rmse_deg': float(np.sqrt(np.mean(physical**2))),
                                     'per_joint_standardized_rmse': np.sqrt(np.mean(error**2, axis=(0, 1))).tolist(),
                                     'per_joint_rmse_deg': np.sqrt(np.mean(physical**2, axis=(0, 1))).tolist()})
    family = []
    for name in definition['dev_recordings']:
        for method in definition['interventions']:
            for horizon in definition['horizons']:
                selected = [r for r in rows if (r['recording'], r['intervention'], r['horizon']) == (name, method, horizon)]
                require(len(selected) == 3, 'allseed family means')
                family.append({'recording': name, 'intervention': method, 'horizon': horizon,
                               'mean_standardized_rmse': statistics.mean(r['standardized_rmse'] for r in selected),
                               'mean_physical_rmse_deg': statistics.mean(r['physical_rmse_deg'] for r in selected),
                               'mean_per_joint_standardized_rmse': np.mean([r['per_joint_standardized_rmse'] for r in selected], axis=0).tolist(),
                               'mean_per_joint_rmse_deg': np.mean([r['per_joint_rmse_deg'] for r in selected], axis=0).tolist()})
    require(authenticate() == (plan, pins), 'all inputs unchanged after intervention')
    output = {'version': definition['version'], 'definition': definition, 'counts': counts,
              'exact_replays': checks, 'trained_experts': matrices, 'gate_statistics': gate_rows,
              'rows': rows, 'family_means': family, 'numeric_reads': reads,
              'scope': definition['scope'], 'seconds': time.perf_counter() - started}
    (OUT / 'diagnostic.json').write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (OUT / 'receipt.json').write_text(json.dumps({'status': 'PASS', 'inputs': pins, 'definition': pin(OUT / 'definition.json'),
          'script': pin(__file__), 'output': pin(OUT / 'diagnostic.json'), 'counts': counts,
          'seconds': time.perf_counter() - started}, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'counts': counts, 'rows': len(rows), 'seconds': time.perf_counter() - started}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        (OUT / 'failure.json').write_text(json.dumps({'status': 'FAILED', 'type': type(exc).__name__, 'message': str(exc)}) + '\n')
        raise
