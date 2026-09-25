"""One original independent NL-LFR audit, with retained terminal evidence."""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINEERING = ROOT / 'output/fsm-author-engineering-v1'
RECEIPT = ENGINEERING / 'nllfr-audit-process.json'
LOG = ENGINEERING / 'nllfr-audit-process.log'
OUTPUT = ROOT / 'output/fsm-author-nllfr-audit-v1/audit.json'
TIMEOUT_SECONDS = 3600
ENV = {**dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                       'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1'),
       'PYTHONDONTWRITEBYTECODE': '1'}
COMMAND = ['.venv/bin/python', '-u', 'scripts/audit_fsm_author_nllfr.py',
           '--study', 'output/fsm-author-nllfr-study-v1',
           '--process', 'output/fsm-author-engineering-v1/nllfr-original-process-01/process.json',
           '--evaluation-process', 'output/fsm-author-engineering-v1/nllfr-evaluation-process-01/process.json',
           '--output', 'output/fsm-author-nllfr-audit-v1/audit.json']
SOURCES = {
    'scripts/audit_fsm_author_nllfr.py': '43c4787e316bc5c0a89653fece0c0694cd9f88041628b798c02382feb09dfa32',
    'tests/test_audit_fsm_author_nllfr.py': '2c824eed78bb21f4d1f709864bc31cdf9042955dfee4e5754c9c2eb54a4e5951',
}
REGISTRATION = 'research/fsm-author-nllfr-registration.json'
QUALIFICATION = 'output/fsm-author-engineering-v1/nllfr-audit-qualification-01/receipt.json'
PREFLIGHT = 'output/fsm-author-engineering-v1/nllfr-audit-qualification-01/preflight.json'
FREEZE = 'output/fsm-author-engineering-v1/nllfr-prefit-freeze.json'
PROCESSES = {
    'fit': 'output/fsm-author-engineering-v1/nllfr-original-process-01/process.json',
    'evaluate': 'output/fsm-author-engineering-v1/nllfr-evaluation-process-01/process.json',
}
HELD = {
    **SOURCES,
    REGISTRATION: 'b1202af6803c20e0f689eaadb35a1b3c93bf91a70ee11cf33a41806c4599190c',
    QUALIFICATION: '6f8e7b77b8f04f8fc10034bb422059655016c90087f712f56dd1d60fae9b6c49',
    PREFLIGHT: '5ef1bfaa01ffb1c00616c0b6c931f9b0d1f0249b18ca5adcd06058eb0c4db07d',
    FREEZE: '11228e77c51f191da9edec03195679a99106044a51f2db607e826e3cd9839c01',
    PROCESSES['fit']: '58d2641a49b33e57b0ffa3687ae34fc1ac44044c6c8cba5c206bba295b4e18f5',
    PROCESSES['evaluate']: '9947cbd421c236cb75a2755528499ce7dab9ffe826ab9aec46bab27b645a1f5b',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular evidence file: ' + str(path))
    blob = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def admit():
    """Read only metadata and opaque hashes; never import the auditor or models."""
    pins = {name: pin(ROOT / name) for name in HELD}
    require(all(pins[k]['sha256'] == v for k, v in HELD.items()), 'held source/input drift')
    qualification, preflight = read(ROOT / QUALIFICATION), read(ROOT / PREFLIGHT)
    require(qualification['status'] == 'PASS' and qualification['sources_unchanged'] is True,
            'passing original fabricated qualification required')
    source_pins = {k: pins[k] for k in SOURCES}
    require(source_pins == qualification['sources_before'] == qualification['sources_after']
            == preflight['sources'], 'qualified source roster')
    require(qualification['preflight'] == pins[PREFLIGHT], 'original preflight join')
    expected_commands = [
        ['.venv/bin/ruff', 'check', *SOURCES],
        ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_audit_fsm_author_nllfr.py'],
    ]
    require(preflight['commands'] == expected_commands and preflight['environment'] == ENV,
            'qualified commands/environment')
    for name, expected in preflight['snapshots'].items():
        value = pin(expected['path'])
        require(value == expected and value['sha256'] == SOURCES[name], 'qualified source snapshot')
        pins['snapshot:' + name] = value
    require(len(qualification['commands']) == 2, 'original qualification command count')
    for index, (row, expected) in enumerate(zip(qualification['commands'], expected_commands, strict=True)):
        value = pin(row['log']['path'])
        require(row['command'] == expected and row['returncode'] == 0 and value == row['log'],
                'original qualification log/command')
        pins[f'qualification_log:{index + 1}'] = value
    require(pin(preflight['wrapper']['path']) == preflight['wrapper'], 'original qualification wrapper')
    pins['qualification_wrapper'] = preflight['wrapper']
    require(preflight['executable'] == str(ROOT / '.venv/bin/python'), 'qualified audit interpreter')
    plan = read(ROOT / REGISTRATION)
    freeze = read(ROOT / FREEZE)
    require(freeze['status'] == 'FROZEN_BEFORE_MEASURED_LAUNCH'
            and freeze['registration_sha256'] == HELD[REGISTRATION]
            and freeze['prefit_commit'] == '4f0258928b21b5e34e6385567deff56558f8fc1a'
            and freeze['push_observed_exit_code'] == 0, 'original prefit freeze')
    require(plan['output'] == 'output/fsm-author-nllfr-study-v1'
            and plan['evaluation_output'] == 'output/fsm-author-nllfr-evaluation-v1'
            and plan['audit_output'] == 'output/fsm-author-nllfr-audit-v1', 'registered output paths')
    for name, expected in plan['source_sha256'].items():
        value = pin(ROOT / name)
        require(value['sha256'] == expected, 'registered source drift: ' + name)
        pins['registered_source:' + name] = value
    for phase, name in PROCESSES.items():
        path = ROOT / name
        terminal, launch = read(path), read(path.parent / 'launch.json')
        require(terminal['phase'] == phase and terminal['status'] in
                ('completed', 'failed', 'timeout', 'memory_limit', 'supervisor_failed'),
                'original phase must be terminal: ' + phase)
        require(terminal['registration_sha256'] == HELD[REGISTRATION]
                and terminal['end_identity_matches'] is True and terminal['identity_error'] is None,
                'original process identity closure')
        require(launch['status'] == 'running'
                and all(terminal[k] == v for k, v in launch.items() if k != 'status'), 'original launch join')
        require(type(terminal['elapsed_seconds']) in (int, float)
                and math.isfinite(terminal['elapsed_seconds']) and terminal['elapsed_seconds'] > 0,
                'original terminal duration')
        require(terminal['status'] != 'completed' or terminal['observed_exit_code'] == 0,
                'completed original exit')
        pins[phase + '_launch'] = pin(path.parent / 'launch.json')
        value = pin(path.parent / 'process.log')
        require(value['sha256'] == terminal['log_sha256'], 'original process log')
        pins[phase + '_log'] = value
    pins['helper'] = pin(__file__)
    return pins


def save(record, *, exclusive=False):
    with RECEIPT.open('x' if exclusive else 'w') as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False) + '\n')


def stop(child):
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    require(not RECEIPT.exists() and not LOG.exists() and not OUTPUT.parent.exists(),
            'original audit already attempted or output directory exists')
    before = admit()
    record = {'command': COMMAND, 'state': 'running', 'started_time_ns': time.time_ns(),
              'timeout_seconds': TIMEOUT_SECONDS, 'environment': ENV,
              'sources_before': {k: before[k] for k in SOURCES}, 'inputs_before': before,
              'helper': before['helper'], 'qualification': before[QUALIFICATION],
              'registration': before[REGISTRATION], 'preflight': before[PREFLIGHT],
              'freeze': before[FREEZE], 'fit_process': before[PROCESSES['fit']],
              'evaluation_process': before[PROCESSES['evaluate']],
              'observed_exit_code': None, 'error': None,
              'scope': 'One original independent audit; no retry. Engineering agreement does not promote an incomplete fit.'}
    save(record, exclusive=True)
    start, wall_start, child = time.monotonic(), time.time(), None
    try:
        with LOG.open('xb') as handle:
            child = subprocess.Popen(COMMAND, cwd=ROOT, env={**os.environ, **ENV},
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            save(record)
            while child.poll() is None:
                if max(time.monotonic() - start, time.time() - wall_start) >= TIMEOUT_SECONDS:
                    record['state'] = 'TIMEOUT'
                    stop(child)
                    break
                time.sleep(.25)
            record['observed_exit_code'] = child.wait()
            if record['state'] == 'running':
                record['state'] = 'EXITED'
    except BaseException as exc:  # noqa: BLE001 - retain a terminal receipt even on interruption.
        record['state'] = 'LAUNCH_FAILED' if child is None else 'WRAPPER_FAILED'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc),
                           'traceback': traceback.format_exc()}
    finally:
        stop(child)
        if child is not None:
            record['observed_exit_code'] = child.wait()
        record['elapsed_seconds'] = time.monotonic() - start
        record['wall_elapsed_seconds'] = time.time() - wall_start
        record['finished_time_ns'] = time.time_ns()
        record['elapsed_scope'] = 'Original audit child launch, wait and termination; excludes subsequent metadata closure checks.'
        record['deadline_scope'] = 'Maximum of monotonic and realtime elapsed; no retry after timeout.'
        record['sources_after'], record['inputs_after'] = {}, {}
        record['sources_unchanged'] = record['inputs_unchanged'] = False
        record['closure_error'] = None
        try:
            after = {k: pin(v['path']) for k, v in before.items()}
            record['inputs_after'] = after
            record['sources_after'] = {k: after[k] for k in SOURCES}
            record['sources_unchanged'] = record['sources_before'] == record['sources_after']
            record['inputs_unchanged'] = before == after
            require(record['sources_unchanged'] and record['inputs_unchanged'], 'post-audit source/input drift')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record['closure_error'] = f'{type(exc).__name__}: {exc}'
        record['evidence_errors'] = {}
        for key, path in (('log', LOG), ('audit_output', OUTPUT)):
            record[key] = None
            try:
                if path.exists():
                    record[key] = pin(path)
            except (OSError, ValueError) as exc:
                record['evidence_errors'][key] = f'{type(exc).__name__}: {exc}'
        record['audit_status'] = record['scientific_status'] = record['agreement'] = None
        if OUTPUT.exists():
            try:
                result = read(OUTPUT)
                record.update(audit_status=result['status'], agreement=result['agreement'],
                              scientific_status=result['scientific_status'])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record['audit_read_error'] = f'{type(exc).__name__}: {exc}'
        record['success'] = (record['state'] == 'EXITED' and record['observed_exit_code'] == 0
                             and record['sources_unchanged'] and record['inputs_unchanged']
                             and record['audit_status'] == 'PASS' and record['agreement'] is True
                             and record['log'] is not None and record['audit_output'] is not None
                             and not record['evidence_errors'])
        record['wrapper_elapsed_seconds'] = time.monotonic() - start
        save(record)
    print(json.dumps({k: record[k] for k in ('state', 'observed_exit_code', 'success',
                                            'elapsed_seconds', 'scientific_status', 'audit_output')}, indent=2))
    return 0 if record['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
