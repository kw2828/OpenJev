"""Hindsight capacity of fixed expert outputs, with no model training or inference."""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import time
from pathlib import Path

import audit_pose_crossfit as parent_audit
import numpy as np
import torch
import train_pose_crossfit as parent

from openjev.research.pose_constant import fit_constant

SEEDS = (1101, 1202, 1303)
PANELS = ('test_sin', 'test_zigzag')
PARENT_PROTOCOL = 'b9e0d4c9a43527cd14d1931242cd130fd224eb948cf6f7d8662e43cb9d2b5143'
PARENT_COMPLETED = '3f57c8727e574a01e078ffea2e3cce5179619a72c8753e1b928d4f3115759be2'
PARENT_SUMMARY = 'b1d8b3ab1bc379919102b7116784a974787697f6df1abbe52a66803897f4288b'
PARENT_RECEIPT = '1bf635372997ec0349857edb038ffaaf4f961989b4e94868b3efece78c644a24'
sha, write_json = parent.sha, parent.write_json


def sources():
    return parent.sources() + [Path(p) for p in (
        'scripts/audit_pose_crossfit_repair.py', 'tests/test_audit_pose_crossfit_repair.py',
        'scripts/run_pose_capacity.py', 'scripts/audit_pose_capacity.py',
        'tests/test_audit_pose_capacity.py', 'research/pose-capacity-protocol.md')]


def validate_parent(experiment, report):
    parent_audit.validate_inputs(experiment, PARENT_PROTOCOL)
    run = experiment / 'run-01'
    done = parent_audit.read_json(run / 'completed.json')
    actual = parent_audit.members(run)
    if (sha(run / 'completed.json') != PARENT_COMPLETED or done['status'] != 'completed'
            or done['files'] != {k: v for k, v in actual.items() if k != 'completed.json'}):
        raise ValueError('parent execution seal')
    if sha(report / 'summary.json') != PARENT_SUMMARY or sha(report / 'receipt.json') != PARENT_RECEIPT:
        raise ValueError('parent report seal')
    summary = parent_audit.read_json(report / 'summary.json')
    receipt = parent_audit.read_json(report / 'receipt.json')
    report_members = parent_audit.members(report)
    if (set(report_members) != {'summary.json', 'window-errors.npz', 'receipt.json'}
            or receipt['files'] != {k: v for k, v in report_members.items() if k != 'receipt.json'}
            or receipt['execution_members'] != actual
            or summary['execution_completed_sha256'] != PARENT_COMPLETED):
        raise ValueError('parent numerical evidence seal')
    return summary


def freeze(experiment, report, out):
    summary = validate_parent(experiment, report)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        'study': 'pose-capacity-v1', 'scope': 'post-outcome hindsight capacity diagnosis; evaluation targets inform oracle choices',
        'parent_experiment': str(experiment.resolve()), 'parent_report': str(report.resolve()),
        'parent_protocol_sha256': PARENT_PROTOCOL, 'parent_completed_sha256': PARENT_COMPLETED,
        'parent_summary_sha256': PARENT_SUMMARY, 'parent_receipt_sha256': PARENT_RECEIPT,
        'sources': {str(p): sha(p) for p in sources()},
        'seeds': SEEDS, 'panels': PANELS, 'windows_per_row': 160, 'horizon': 25, 'rows': 6,
        'constant_searches': 960, 'constant_tolerance_rad_squared': 1e-7, 'constant_max_evaluations': 4095,
        'position_class': 'One scalar alpha in[0,1] per window, fixed across25 steps; exact clipped least squares.',
        'rotation_class': 'One scalar alpha in[0,1] per window, fixed across25 steps; geodesic interpolation of float64 SO3-projected experts and projected targets.',
        'hard_window': 'For each endpoint and window, min of the two expert mean squared errors over25 steps.',
        'hard_step': 'For each endpoint and window, mean of the per-step minimum squared errors; separate relaxed hard-routing diagnostic, not a bound for continuous mixing.',
        'comparison': 'Pool MSE across all480 seed-window pairs before square root. Reference threshold is0.81 times each reported control MSE. Equality meets the necessary10percent margin.',
        'controls': sorted([*summary['families']['test_sin'], *summary['references']['test_sin']]),
        'rotation_qualification': 'Projected bounds do not uniformly bound original float32 production interpolation. Report projected lower/upper comparisons to reported thresholds explicitly; no original-production exclusion claim.',
        'no_success_claim': 'Oracle labels use future targets; favorable capacity is not a causal selector, trained architecture, latency result, fresh confirmation or full continuation pass.',
        'no_selection': 'All960 cases, complete traces, fixed tolerance and cap. No retries, replacements, exclusions or budget extensions; capped searches stay visible.',
        'execution_files': 20, 'model_training_calls': 0, 'model_inference_calls': 0,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                    'platform': platform.platform()},
    }
    write_json(out / 'protocol.json', protocol)
    for p in sources():
        dest = out / 'source-snapshot' / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    print(json.dumps({'frozen': str(out), 'sha256': sha(out / 'protocol.json')}))


def validate_protocol(out, expected_sha):
    if sha(out / 'protocol.json') != expected_sha:
        raise ValueError('external frozen protocol digest')
    p = parent_audit.read_json(out / 'protocol.json')
    if (p['sources'] != {str(x): sha(x) for x in sources()}
            or p['sources'] != {k: v['sha256'] for k, v in parent_audit.members(out / 'source-snapshot').items()}):
        raise ValueError('frozen source changed')
    if p['runtime'] != {'python': platform.python_version(), 'numpy': np.__version__,
                        'torch': torch.__version__, 'platform': platform.platform()}:
        raise ValueError('frozen runtime changed')
    validate_parent(Path(p['parent_experiment']), Path(p['parent_report']))
    return p


def run(out, expected_sha):
    protocol = validate_protocol(out, expected_sha)
    dest = out / 'run-01'
    dest.mkdir(exist_ok=False)
    start = time.perf_counter()
    records, active_row = [], None
    try:
        write_json(dest / 'started.json', {'protocol_sha256': expected_sha})
        torch.set_num_threads(1)
        source = Path(protocol['parent_experiment']) / 'run-01'
        rows = []
        for panel in PANELS:
            with np.load(source / (panel + '-targets.npz'), allow_pickle=False) as data:
                tp, tr, ids = data['p'], data['R'], data['ids']
            for seed in SEEDS:
                tick = time.perf_counter()
                records, active_row = [], (panel, seed)
                fp, fr, _ = parent_audit._previous.load_prediction(source / f'{panel}-fast-{seed}-predictions.npz', alpha=True)
                sp, sr, _ = parent_audit._previous.load_prediction(source / f'{panel}-slow-{seed}-predictions.npz', alpha=True)
                fast_error = parent_audit.error_arrays(fp, fr, tp, tr)
                slow_error = parent_audit.error_arrays(sp, sr, tp, tr)
                hard_window = np.stack([np.minimum(fast_error[e].mean(1), slow_error[e].mean(1))
                                        for e in ('position', 'rotation')], -1)
                hard_step = np.stack([np.minimum(fast_error[e], slow_error[e]).mean(1)
                                      for e in ('position', 'rotation')], -1)
                for window in range(160):
                    fit_start = time.perf_counter()
                    record = fit_constant(*(x[window:window+1] for x in (fp, fr, sp, sr, tp, tr)),
                                          tolerance=1e-7, max_evaluations=4095)
                    records.append({'window': window, 'source_id': int(ids[window, 0]),
                        'window_start': int(ids[window, 1]), 'optimization': record,
                        'seconds': time.perf_counter()-fit_start})
                    if (window+1) % 40 == 0:
                        print(json.dumps({'panel': panel, 'seed': seed, 'windows_completed': window+1}), flush=True)
                prefix = f'{panel}-{seed}'
                trace_path = dest / (prefix + '-traces.json.gz')
                with trace_path.open('xb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as zipped:
                    zipped.write(json.dumps({'panel': panel, 'seed': seed, 'records': records},
                                           sort_keys=True, allow_nan=False, separators=(',', ':')).encode())
                fits = [r['optimization'] for r in records]
                arrays = {'ids': ids, 'hard_window': hard_window, 'hard_step': hard_step,
                    'continuous_position_mse': np.asarray([r['position']['objective_m2'] for r in fits]),
                    'continuous_rotation_lower_mse': np.asarray([r['rotation']['lower_bound'] for r in fits]),
                    'continuous_rotation_upper_mse': np.asarray([r['rotation']['upper_bound'] for r in fits]),
                    'alpha': np.asarray([r['alpha'] for r in fits]),
                    'certified': np.asarray([r['rotation']['certified'] for r in fits], dtype=np.bool_),
                    'calls': np.asarray([r['rotation']['call_count'] for r in fits], dtype=np.int64)}
                bounds_path = dest / (prefix + '-bounds.npz')
                np.savez_compressed(bounds_path, **arrays)
                row = {'panel': panel, 'seed': seed, 'trace_sha256': sha(trace_path), 'bounds_sha256': sha(bounds_path),
                    'constant_searches': len(records), 'objective_evaluations': int(arrays['calls'].sum()),
                    'certified_searches': int(arrays['certified'].sum()),
                    'search_seconds': sum(x['seconds'] for x in records), 'wall_seconds': time.perf_counter()-tick}
                write_json(dest / (prefix + '-row.json'), row)
                rows.append(row)
                print(json.dumps(row), flush=True)
                records, active_row = [], None
        validate_protocol(out, expected_sha)
        write_json(dest / 'completed.json', {'status': 'completed', 'protocol_sha256': expected_sha,
            'rows': rows, 'constant_searches': 960, 'objective_evaluations': sum(r['objective_evaluations'] for r in rows),
            'certified_searches': sum(r['certified_searches'] for r in rows), 'wall_seconds': time.perf_counter()-start,
            'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}})
        print(json.dumps({'completed': str(dest), 'wall_seconds': time.perf_counter()-start}), flush=True)
    except BaseException as error:
        partial = None
        if active_row is not None and records:
            panel, seed = active_row
            path = dest / f'partial-{panel}-{seed}-traces.json.gz'
            try:
                with path.open('xb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as zipped:
                    zipped.write(json.dumps({'panel': panel, 'seed': seed, 'records': records},
                                            sort_keys=True, allow_nan=False, separators=(',', ':')).encode())
                partial = {'path': path.name, 'sha256': sha(path), 'returned_searches': len(records)}
            except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
                error.add_note(f'Could not retain partial traces: {secondary!r}')
        try:
            write_json(dest / 'failed.json', {'status': 'failed', 'error': repr(error),
                'protocol_sha256': expected_sha, 'partial_trace': partial, 'wall_seconds': time.perf_counter()-start})
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
            parser.error('freeze requires parent and report')
        freeze(args.parent, args.report, args.out)
    else:
        if args.protocol_sha256 is None:
            parser.error('run requires --protocol-sha256')
        run(args.out, args.protocol_sha256)
