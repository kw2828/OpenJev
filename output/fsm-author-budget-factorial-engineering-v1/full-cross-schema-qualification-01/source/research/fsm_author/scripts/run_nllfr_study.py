# SPDX-License-Identifier: GPL-3.0-or-later
"""Supervise each registered NL-LFR fit/evaluation once, retaining native closure."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from fit_nllfr import (
    ENV,
    ROOT,
    metadata_admission,
    pin,
    read,
    relative,
    require,
    sha,
    write,
)


def phase_settings(cfg, phase):
    if phase == 'fit':
        return (relative(cfg['output']), relative(cfg['process_directory']),
                cfg['experiment']['outer_timeout_seconds'], cfg['experiment']['rss_cap_bytes'])
    require(phase == 'evaluate', 'unsupported phase')
    evaluation = cfg['evaluation']
    require(evaluation['outer_timeout_seconds'] == 3600
            and evaluation['rss_cap_bytes'] == 32*1024**3, 'evaluation resource policy changed')
    return (relative(cfg['evaluation_output']), relative(cfg['evaluation_process_directory']),
            evaluation['outer_timeout_seconds'], evaluation['rss_cap_bytes'])


def check_fit_closed(cfg, registration):
    folder = relative(cfg['output'])
    receipt_path = relative(cfg['process_directory'])/'process.json'
    receipt = read(receipt_path)
    require(receipt['phase'] == 'fit' and receipt['status'] == 'completed'
            and receipt['observed_exit_code'] == 0 and receipt['end_identity_matches'],
            'evaluation requires original fit exit zero and identity closure')
    require(receipt['registration_sha256'] == sha(registration), 'fit registration mismatch')
    require(receipt['log_sha256'] == sha(receipt_path.with_name('process.log')), 'fit log changed')
    expected_command = [str(ROOT/'research/fsm_author/.venv/bin/python'),
                        str(ROOT/'research/fsm_author/scripts/fit_nllfr.py'),
                        '--registration', str(registration.resolve()), '--output', str(folder.resolve())]
    require(receipt['command'] == expected_command and receipt['environment'] == ENV, 'fit launch identity drift')
    require(receipt['producer'] == pin(ROOT/'research/fsm_author/scripts/fit_nllfr.py')
            and receipt['supervisor'] == pin(__file__), 'fit executable source drift')
    require(receipt['timeout_seconds'] == cfg['experiment']['outer_timeout_seconds']
            and receipt['rss_cap_bytes'] == cfg['experiment']['rss_cap_bytes'], 'fit cap identity drift')
    require(receipt['artifacts'] == inventory(folder), 'complete fit inventory changed')
    require({'final.zip', 'final.npz', 'fit.json', 'summary.json'} <= set(receipt['artifacts']),
            'required finite fit artifacts missing')
    require(read(folder/'fit.json')['status'] in ('complete', 'iteration_cap_reached'), 'no finite fit')
    return pin(receipt_path)


def overlap(a, b):
    a, b = a.resolve(), b.resolve()
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def inventory(folder):
    result = {}
    if folder.exists():
        require(folder.is_dir() and not folder.is_symlink(), 'regular output directory required')
        for path in sorted(folder.rglob('*')):
            require(not path.is_symlink(), 'symlink output evidence')
            if path.is_file():
                result[str(path.relative_to(folder))] = {k: pin(path)[k] for k in ('sha256', 'bytes')}
    return result


def supervise(registration, phase='fit', requested_output=None, requested_receipt=None):
    require(Path.cwd().resolve() == ROOT, 'run from OpenJev root')
    cfg = metadata_admission(registration)
    output, receipt_folder, cap, rss_cap = phase_settings(cfg, phase)
    require(not overlap(output, receipt_folder), 'study/process directories overlap')
    if phase == 'evaluate':
        for destination in (output, receipt_folder):
            for original in (relative(cfg['output']), relative(cfg['process_directory'])):
                require(not overlap(destination, original), 'evaluation overlaps original fit evidence')
    if requested_output is not None:
        require(requested_output.resolve() == output.resolve(), 'unregistered output')
    if requested_receipt is not None:
        require(requested_receipt.resolve() == receipt_folder.resolve(), 'unregistered receipt directory')
    for path in (registration, relative(cfg['data_path']),
                 *(relative(k) for k in cfg['source_sha256']),
                 *(relative(p['path']) for p in cfg['prerequisites'].values())):
        require(not overlap(output, path) and not overlap(receipt_folder, path), 'output overwrites input evidence')
    require(not output.exists() and not receipt_folder.exists(), 'original attempt directory already exists')
    parent_fit = check_fit_closed(cfg, registration) if phase == 'evaluate' else None
    receipt_folder.mkdir(parents=True, exist_ok=False)
    producer = Path(__file__).with_name('fit_nllfr.py' if phase == 'fit' else 'evaluate_nllfr.py')
    require(str(producer.relative_to(ROOT)) in cfg['source_sha256'], 'phase source not registered')
    command = [str(ROOT/'research/fsm_author/.venv/bin/python'), str(producer.resolve()),
               '--registration', str(registration.resolve()), '--output', str(output.resolve())]
    registration_hash = sha(registration)
    receipt = {'phase': phase, 'command': command, 'status': 'running',
               'registration_sha256': registration_hash, 'started_time_ns': time.time_ns(),
               'timeout_seconds': cap, 'rss_cap_bytes': rss_cap, 'environment': ENV,
               'supervisor': pin(__file__), 'producer': pin(producer), 'parent_fit_process': parent_fit}
    process_path = receipt_folder/'process.json'
    write(receipt_folder/'launch.json', receipt)
    process_path.write_text(json.dumps(receipt, indent=2)+'\n')
    start, peak, child, error, code = time.monotonic(), 0, None, None, None
    wall_start = time.time()
    try:
        with (receipt_folder/'process.log').open('xb') as log:
            child = subprocess.Popen(command, env={**os.environ, **ENV}, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt['pid'] = child.pid
            process_path.write_text(json.dumps(receipt, indent=2)+'\n')
            while child.poll() is None:
                if max(time.monotonic()-start, time.time()-wall_start) >= cap:
                    receipt['status'] = 'timeout'
                    break
                rss_result = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(child.pid)],
                                            capture_output=True, text=True, timeout=2, check=False)
                value = rss_result.stdout.strip()
                rss = int(value)*1024 if value else 0
                peak = max(peak, rss)
                if rss > rss_cap:
                    receipt['status'] = 'memory_limit'
                    break
                time.sleep(.5)
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = child.wait()
            if receipt['status'] == 'running':
                receipt['status'] = 'completed' if code == 0 else 'failed'
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        receipt['status'] = 'supervisor_failed'
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = child.wait()
        receipt.update(observed_exit_code=code, elapsed_seconds=time.monotonic()-start,
                       wall_elapsed_seconds=time.time()-wall_start,
                       peak_polled_child_rss_bytes=peak, error=error,
                       memory_scope='Individual child RSS sampled every0.5s via ps KiB; no swap/workspace estimate',
                       deadline_scope='Maximum of monotonic and realtime elapsed; sleep and forward clock jumps count')
        identity_error = None
        try:
            metadata_admission(registration)
            require(sha(registration) == registration_hash, 'registration changed during process')
            if phase == 'evaluate':
                require(check_fit_closed(cfg, registration) == parent_fit, 'fit closure changed during evaluation')
        except Exception as exc:
            identity_error = f'{type(exc).__name__}: {exc}'
        receipt['end_identity_matches'] = identity_error is None
        receipt['identity_error'] = identity_error
        if identity_error is not None:
            receipt['status'] = 'identity_failed'
        receipt['log_sha256'] = sha(receipt_folder/'process.log') if (receipt_folder/'process.log').exists() else None
        try:
            receipt['artifacts'] = inventory(output)
        except Exception as exc:
            receipt['artifacts'] = {}
            receipt['inventory_error'] = f'{type(exc).__name__}: {exc}'
            receipt['status'] = 'evidence_failed'
        receipt['outcome'] = 'ORIGINAL_PROCESS_COMPLETE' if receipt['status'] == 'completed' else 'INCOMPLETE'
        receipt['scope'] = 'Original single process; no restart/rescue. Iteration-capped finite fit remains FIT_INCOMPLETE.'
        process_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': receipt['status'], 'phase': phase, 'observed_exit_code': code,
                      'elapsed_seconds': receipt['elapsed_seconds'], 'peak_polled_child_rss_bytes': peak,
                      'process': pin(process_path)}, indent=2), flush=True)
    return 0 if receipt['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--phase', choices=('fit', 'evaluate'), default='fit')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--receipt-directory', type=Path)
    args = parser.parse_args()
    return supervise(args.registration, args.phase, args.output, args.receipt_directory)


if __name__ == '__main__':
    raise SystemExit(main())
