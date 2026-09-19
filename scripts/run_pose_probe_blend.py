"""Evaluate completed-probe blending with frozen pose experts and full cost."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import audit_pose_capacity as capacity_audit
import numpy as np
import run_pose_capacity as capacity
import torch
import train_pose_coordination as coordination
from train_pose_adaptation import load_data, measure

from openjev.research.pose_probe_blend import forecast

SEEDS = (1101, 1202, 1303)
PANELS = ('test_sin', 'test_zigzag')
VARIANTS = ('fast', 'slow', 'half', 'probe_half', 'probe_inverse', 'probe_fit')
CAPACITY_PROTOCOL = '843c3bc1744e8927b536cd85d3f1d54b0f2a4a5ed854fb52815a4109fbdfc237'
CAPACITY_COMPLETED = '975cf6434c6cc406ec6fc3ce7d1dc9e7e093fc1ac92fb660d0c7339936bb1198'
CAPACITY_SUMMARY = 'b85c4537118b72bec031a925f8a433a34629cdeea25d4e1ba883017263b7fb3a'
CAPACITY_RECEIPT = 'a8f072f3afa6f301df3bf5e324fa02e96f62cfe1be1e96859d4bcf821ba60714'
GATE = ('probe_fit RMSE <=0.90*each positive control on both physical endpoints/panels; '
        'all3 paired MSE nonworse; at least8/10 parents nonworse; all10 leave-one-parent-out '
        'MSE strictly lower; median full-window latency<=1.5*new slow. All17 groups required.')
sha, write_json = capacity.sha, capacity.write_json


def sources():
    return capacity.sources() + [Path(p) for p in (
        'src/openjev/research/pose_probe_blend.py', 'tests/test_pose_probe_blend.py',
        'scripts/run_pose_probe_blend.py', 'scripts/audit_pose_probe_blend.py',
        'tests/test_audit_pose_probe_blend.py', 'research/pose-probe-blend-protocol.md')]


def validate_parent(experiment, report):
    _, parent = capacity_audit.validate_inputs(experiment, CAPACITY_PROTOCOL)
    run = experiment / 'run-01'
    actual = capacity_audit.members(run)
    done = capacity_audit.read_json(run / 'completed.json')
    if (set(actual) != capacity_audit.expected_members()
            or sha(run / 'completed.json') != CAPACITY_COMPLETED
            or done['status'] != 'completed' or done['protocol_sha256'] != CAPACITY_PROTOCOL
            or done['files'] != {k: v for k, v in actual.items() if k != 'completed.json'}):
        raise ValueError('capacity execution changed')
    receipt = capacity_audit.read_json(report / 'receipt.json')
    summary = capacity_audit.read_json(report / 'summary.json')
    members = capacity_audit.members(report)
    if (set(members) != {'summary.json', 'receipt.json', 'window-errors.npz'}
            or sha(report / 'summary.json') != CAPACITY_SUMMARY
            or sha(report / 'receipt.json') != CAPACITY_RECEIPT
            or receipt['status'] != summary['status'] or summary['status'] != 'completed'
            or receipt['execution_members'] != actual
            or receipt['files'] != {k: v for k, v in members.items() if k != 'receipt.json'}
            or summary['execution_completed_sha256'] != CAPACITY_COMPLETED):
        raise ValueError('capacity report changed')
    return parent


def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'platform': platform.platform(), 'threads': 1, 'deterministic_algorithms': True}


def freeze(experiment, report, out):
    parent = validate_parent(experiment, report)
    inherited = parent['context']['protocol']
    fallback = {}
    for seed in SEEDS:
        path = parent['run'] / f'full_constant-{seed}-fit.json'
        fallback[str(seed)] = {'path': str(path.resolve()), 'sha256': sha(path),
                              'alpha': capacity_audit.read_json(path)['optimization']['alpha']}
    protocol = {
        'study': 'pose-probe-blend-v1', 'scope': 'exposed-data completed-past probe blending screen',
        'capacity_experiment': str(experiment.resolve()), 'capacity_report': str(report.resolve()),
        'capacity_protocol_sha256': CAPACITY_PROTOCOL, 'capacity_completed_sha256': CAPACITY_COMPLETED,
        'capacity_summary_sha256': CAPACITY_SUMMARY, 'capacity_receipt_sha256': CAPACITY_RECEIPT,
        'data': inherited['data'], 'data_hashes': inherited['data_hashes'],
        'checkpoints': inherited['checkpoints'], 'fallback': fallback,
        'sources': {str(p): sha(p) for p in sources()}, 'runtime': runtime(),
        'seeds': SEEDS, 'panels': PANELS, 'variants': VARIANTS, 'primary': 'probe_fit',
        'context': 32, 'horizon': 25, 'windows_per_panel': 160, 'parents_per_panel': 10,
        'probe_origin': 26, 'probe_poses': 27, 'probe_past_actions': 26,
        'probe_support_indices': [1, 25], 'probe_action_indices': [26, 30], 'probe_target_indices': [27, 31],
        'probe_age_half_life': 5, 'probe_huber_delta': 1.5, 'probe_huber_iterations': 3,
        'probe_ridge_precision': 1., 'probe_solve_dtype': 'float64',
        'position_fit': 'Mean dot(fast-slow,target-slow) divided by mean squared separation, clipped to[0,1], computed in float64.',
        'position_fallback': 'D <= max(float64.tiny, eps(input_dtype)^2 * max(fast_position_MSE,slow_position_MSE)); use saved training-only full_constant position alpha.',
        'inverse_fit': 'Opposite expert MSE divided by sum, after overflow-safe common rescaling; both zero uses saved training-only full_constant alpha.',
        'rotation_grid': [i / 16 for i in range(17)],
        'rotation_fit': 'Actual float32 production blend on the17 ascending grid coefficients, scored in float64 squared geodesic radians over5 completed leads; exact ties choose first index.',
        'probe_half': 'Compute the same position fit and rotation grid as probe_fit, then discard coefficients and use exact half.',
        'probe_inverse': 'Compute probe forecasts and endpoint errors, then inverse-error weights without the fitted coefficient or rotation grid.',
        'causality': 'At root31, retrospective probe uses only poses0..31 and recorded applied actions0..30. These actions were not necessarily known at historical origin26. No future observed pose enters any forecast or coefficient.',
        'deployment': 'Coefficients remain fixed across25 future steps. Experts roll privately; slow actual-context state continues through31 without probe feedback and preserves CV1.',
        'future_actions': 'Both full forecasts condition on supplied recorded applied actions31..55, as in all parent studies; no planned-command or closed-loop claim.',
        'evaluation_rows': 36, 'execution_files': 94, 'canonical_configurations': 36,
        'distinct_combined_rows': 192, 'controls': 35, 'gate': GATE,
        'gate_groups': 17, 'gate_comparisons': 2101,
        'warmups_per_row': 3, 'timed_windows_per_row': 20,
        'warmup_indices': [0, 1, 2], 'timed_indices': list(range(3, 23)),
        'forecast_calls': 864, 'batch_forecast_calls': 36, 'single_forecast_calls': 828,
        'new_neural_fits': 0, 'new_optimizer_updates': 0, 'new_native_calls': 0,
        'timing_scope': 'Full single-window forecast including input validation, probe fitting/rollout/scoring when used, both deployment experts when used, blending and returned diagnostics. Loading, saved probe-array packaging, metric calculation and artifact I/O excluded.',
        'blend_tolerance': {'rtol': 2e-6, 'atol': 2e-6, 'hard_endpoints': 'exact'},
        'expert_replay_tolerance': {'rtol': 1e-6, 'atol': 2e-7},
        'coefficient_arithmetic_tolerance': {'rtol': 1e-10, 'atol': 1e-11},
        'no_selection': 'One complete fixed run, all cases and seeds. No retries, sweeps, exclusions, grid refinement, threshold tuning or gate changes. Keep partial returned artifacts on failure.',
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'protocol.json', protocol)
    for p in sources():
        dest = out / 'source-snapshot' / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    print(json.dumps({'frozen': str(out), 'sha256': sha(out / 'protocol.json')}))


def validate_protocol(out, expected_sha):
    if sha(out / 'protocol.json') != expected_sha:
        raise ValueError('external protocol digest')
    p = capacity_audit.read_json(out / 'protocol.json')
    snapshots = {k: v['sha256'] for k, v in capacity_audit.members(out / 'source-snapshot').items()}
    if (p['sources'] != {str(x): sha(x) for x in sources()}
            or p['sources'] != snapshots or p['runtime'] != runtime()):
        raise ValueError('source snapshot or runtime changed')
    validate_parent(Path(p['capacity_experiment']), Path(p['capacity_report']))
    for item in (*p['checkpoints'].values(), *p['fallback'].values()):
        if sha(item['path']) != item['sha256']:
            raise ValueError('expert or fallback changed')
    return p


def save_arrays(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **{k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                                     for k, v in arrays.items()})


def run(out, expected_sha):
    protocol = validate_protocol(out, expected_sha)
    dest = out / 'run-01'
    dest.mkdir(exist_ok=False)
    start = time.perf_counter()
    rows, calls, active = [], 0, None
    try:
        write_json(dest / 'started.json', {'protocol_sha256': expected_sha})
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for panel in PANELS:
            p, rotation, actions, ids = load_data(Path(protocol['data']), panel)
            save_arrays(dest / f'{panel}-inputs.npz', p=p, R=rotation, actions=actions, ids=ids)
            for variant in VARIANTS:
                for seed in SEEDS:
                    tick = time.perf_counter()
                    prefix = f'{panel}-{variant}-{seed}'
                    fast, slow = coordination.experts(protocol, seed)
                    fallback = p.new_tensor(protocol['fallback'][str(seed)]['alpha'])
                    active = {'panel': panel, 'variant': variant, 'seed': seed, 'returned_calls': 0,
                              'warmup_ms': [], 'latency_ms': [], 'timing_diagnostics': []}
                    with torch.no_grad():
                        batch_start = time.perf_counter()
                        pp, rr, alpha, diagnostic, probe = forecast(fast, slow, variant, p[:, :32], rotation[:, :32],
                            actions[:, :31], actions[:, 31:], fallback_alpha=fallback, return_probe=True)
                        batch_seconds = time.perf_counter() - batch_start
                        calls += 1
                        active['returned_calls'] += 1
                        predictions_path = dest / f'{prefix}-predictions.npz'
                        save_arrays(predictions_path, p=pp, R=rr, alpha=alpha)
                        probe_path = dest / f'{prefix}-probe.npz'
                        if probe is not None:
                            save_arrays(probe_path, **probe)
                        active.update(prediction_sha256=sha(predictions_path),
                                      probe_sha256=sha(probe_path) if probe is not None else None,
                                      batch_diagnostics=diagnostic, batch_forecast_seconds=batch_seconds)
                        metrics = measure(pp, rr, p[:, 32:], rotation[:, 32:])
                        for i in range(23):
                            single_start = time.perf_counter_ns()
                            _, _, _, single_diagnostic = forecast(fast, slow, variant,
                                p[i:i+1, :32], rotation[i:i+1, :32], actions[i:i+1, :31], actions[i:i+1, 31:],
                                fallback_alpha=fallback, return_probe=False)
                            elapsed = (time.perf_counter_ns() - single_start) / 1e6
                            calls += 1
                            active['returned_calls'] += 1
                            active['warmup_ms' if i < 3 else 'latency_ms'].append(elapsed)
                            active['timing_diagnostics'].append(single_diagnostic)
                    row = {k: v for k, v in active.items() if k != 'returned_calls'}
                    row.update(expert_sha256={n: protocol['checkpoints'][f'{n}-{seed}']['sha256'] for n in ('meta', 'gru')},
                               fallback_alpha=fallback.tolist(), metrics=metrics, row_wall_seconds=time.perf_counter() - tick)
                    write_json(dest / f'{prefix}-evaluation.json', row)
                    rows.append(row)
                    active = None
                    print(json.dumps({'row': prefix, 'metrics': metrics}), flush=True)
        validate_protocol(out, expected_sha)
        if calls != 864 or len(rows) != 36:
            raise ValueError('incomplete fixed execution')
        manifest = capacity_audit.members(dest)
        write_json(dest / 'completed.json', {'status': 'completed', 'protocol_sha256': expected_sha, 'rows': rows,
            'forecast_calls': calls, 'batch_forecast_calls': 36, 'single_forecast_calls': 828,
            'new_optimizer_updates': 0, 'wall_seconds': time.perf_counter() - start,
            'files': manifest})
        print(json.dumps({'completed': str(dest), 'wall_seconds': time.perf_counter() - start}), flush=True)
    except BaseException as error:
        try:
            completion = dest / 'completed.json'
            if completion.exists():
                retained = dest / 'completion-before-error.json'
                if retained.exists():
                    raise FileExistsError(retained)
                completion.rename(retained)
        except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
            error.add_note(f'Could not demote premature completion: {secondary!r}')
        try:
            write_json(dest / 'failed.json', {'status': 'failed', 'protocol_sha256': expected_sha,
                'error': repr(error), 'returned_forecast_calls': calls, 'completed_rows': len(rows),
                'active_row': active, 'wall_seconds': time.perf_counter() - start})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
            error.add_note(f'Could not retain failure receipt: {secondary!r}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if args.mode == 'freeze':
        if args.parent is None or args.report is None:
            parser.error('freeze requires --parent and --report')
        freeze(args.parent, args.report, args.out)
    else:
        if args.protocol_sha256 is None:
            parser.error('run requires --protocol-sha256')
        run(args.out, args.protocol_sha256)
