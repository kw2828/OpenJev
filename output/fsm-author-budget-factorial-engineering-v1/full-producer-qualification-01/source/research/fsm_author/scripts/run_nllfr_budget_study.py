# SPDX-License-Identifier: GPL-3.0-or-later
"""Supervise one registered larger-budget FIT-only attempt; no restart or evaluation."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from fit_nllfr_budget import (
    ENV,
    ROOT,
    metadata_admission,
    pin,
    relative,
    require,
    sha,
    write,
)


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


def supervise(registration, requested_output=None, requested_receipt=None):
    require(Path.cwd().resolve() == ROOT, 'run from OpenJev root')
    cfg = metadata_admission(registration)
    output, receipt_folder = relative(cfg['output']), relative(cfg['process_directory'])
    cap, rss_cap = cfg['experiment']['outer_timeout_seconds'], cfg['experiment']['rss_cap_bytes']
    require(not overlap(output, receipt_folder), 'study/process directories overlap')
    if requested_output is not None:
        require(requested_output.resolve() == output.resolve(), 'unregistered output')
    if requested_receipt is not None:
        require(requested_receipt.resolve() == receipt_folder.resolve(), 'unregistered receipt directory')
    for path in (registration,
                 *(relative(k) for k in cfg['source_sha256']),
                 *(relative(p['path']) for p in cfg['prerequisites'].values())):
        require(not overlap(output, path) and not overlap(receipt_folder, path), 'output overwrites input evidence')
    require(not output.exists() and not receipt_folder.exists(), 'original attempt directory already exists')
    parent_fit = pin(relative(cfg['prerequisites']['parent_process']['path']))
    receipt_folder.mkdir(parents=True, exist_ok=False)
    producer = Path(__file__).with_name('fit_nllfr_budget.py')
    require(str(producer.relative_to(ROOT)) in cfg['source_sha256'], 'phase source not registered')
    command = [str(ROOT/'research/fsm_author/.venv/bin/python'), str(producer.resolve()),
               '--registration', str(registration.resolve()), '--output', str(output.resolve())]
    registration_hash = sha(registration)
    receipt = {'phase': 'fit', 'command': command, 'status': 'running',
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
            require(pin(relative(cfg['prerequisites']['parent_process']['path'])) == parent_fit, 'parent fit closure changed')
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
        receipt['scope'] = 'Original single process; no restart/rescue. Iteration cap or prefix mismatch remains FIT_ONLY_INCOMPLETE.'
        process_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': receipt['status'], 'phase': 'fit', 'observed_exit_code': code,
                      'elapsed_seconds': receipt['elapsed_seconds'], 'peak_polled_child_rss_bytes': peak,
                      'process': pin(process_path)}, indent=2), flush=True)
    return 0 if receipt['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--receipt-directory', type=Path)
    args = parser.parse_args()
    return supervise(args.registration, args.output, args.receipt_directory)


if __name__ == '__main__':
    raise SystemExit(main())
