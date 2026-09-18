"""Retained whole-pipeline engineering rehearsal; never a scored experiment."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'scripts'), str(ROOT / 'src')]
from reacher_memory_fixture import audit_fixture, prepare_fixture, run_fixture


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    write(folder / 'supervision-started.json', {
        'scope': 'engineering_only_no_efficacy',
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'utc_unix_time': time.time(), 'execution_cap_seconds': 600, 'audit_cap_seconds': 600,
        'retry_policy': 'No overwrite; every engineering attempt retained separately.'})
    try:
        case = prepare_fixture(folder / 'fixture', cap_seconds=600, audit_cap_seconds=600)
        print(json.dumps({'phase': 'prepared', 'plan_sha256': case.digest}), flush=True)
        with (folder / 'execution.log').open('x') as log, contextlib.redirect_stdout(log):
            execution = run_fixture(case)
        print(json.dumps({'phase': 'execution_completed', **execution}), flush=True)
        with (folder / 'audit.log').open('x') as log, contextlib.redirect_stdout(log):
            audit = audit_fixture(case, folder / 'audit')
        result = {'status': 'completed', 'scope': 'engineering_only_no_efficacy',
            'plan_sha256': case.digest, 'execution': execution,
            'audit': {'status': audit['status'], 'engineering': audit['engineering'],
                'native_transitions_checked': audit['native_transitions_checked'],
                'native_max_abs_error': audit['native_max_abs_error'],
                'new_model_calls': audit['new_model_calls'], 'coverage': audit['coverage']},
            'wall_seconds': time.monotonic()-begin}
        write(folder / 'supervision-completed.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write(folder / 'supervision-failed.json', {'status': 'failed', 'error': repr(error),
            'scope': 'engineering_only_no_efficacy', 'wall_seconds': time.monotonic()-begin})
        raise


if __name__ == '__main__':
    main()
