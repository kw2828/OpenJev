# SPDX-License-Identifier: GPL-3.0-or-later
"""Retain the original empirical child process and enforce its frozen time cap."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt-directory', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if Path.cwd() != root:
        raise ValueError('run from the OpenJev root')
    cfg = json.loads(args.registration.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    args.receipt_directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).with_name('fit_bla.py')),
               '--registration', str(args.registration), '--output', str(args.output)]
    registration_hash = sha(args.registration)
    receipt = {'command': command, 'registration_sha256': registration_hash,
               'timeout_seconds': cfg['experiment']['outer_timeout_seconds'], 'status': 'running',
               'started_time_ns': time.time_ns()}
    path = args.receipt_directory/'process.json'
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    before = time.perf_counter()
    with (args.receipt_directory/'process.log').open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['pid'] = child.pid
        path.write_text(json.dumps(receipt, indent=2)+'\n')
        try:
            code = child.wait(timeout=receipt['timeout_seconds'])
            receipt['status'] = 'completed' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
            receipt['status'] = 'timeout'
    receipt.update(observed_exit_code=code, elapsed_seconds=time.perf_counter()-before,
                   log_sha256=sha(args.receipt_directory/'process.log'))
    receipt['end_identity_matches'] = (sha(args.registration) == registration_hash
        and all(sha(root/path) == digest for path, digest in cfg['source_sha256'].items())
        and all(sha(root/item['path']) == item['sha256'] for item in cfg['prerequisites'].values()))
    if not receipt['end_identity_matches']:
        receipt['status'] = 'identity_failed'
    for name in ('summary.json', 'fit.json', 'failure.json', 'admission.json'):
        if (args.output/name).exists():
            receipt[name+'_sha256'] = sha(args.output/name)
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
