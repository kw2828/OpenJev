"""One independent factorial audit after the original evaluator closes.

Admission reads metadata and opaque hashes only. The sole child performs the
qualified replay; this wrapper never launches fitting, evaluation or a retry.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
ENGINEERING = ROOT/'output/fsm-author-budget-factorial-engineering-v1'
STUDY = ROOT/'output/fsm-author-nllfr-factorial-study-v1'
EVALUATION_PROCESS = ENGINEERING/'original-process-01/process.json'
REGISTRATION = ROOT/'research/fsm-author-nllfr-factorial-registration.json'
FREEZE = ENGINEERING/'auditor-freeze.json'
QUALIFICATION = ENGINEERING/'full-auditor-qualification-02/receipt.json'
RECEIPT = ENGINEERING/'audit-process.json'
LOG = ENGINEERING/'audit-process.log'
OUTPUT = ROOT/'output/fsm-author-nllfr-factorial-audit-v1/audit.json'
TIMEOUT_SECONDS = 3600
RSS_CAP_BYTES = 32*1024**3
ENV = {**dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                       'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1'),
       'PYTHONDONTWRITEBYTECODE': '1'}
COMMAND = ['.venv/bin/python', '-u', 'scripts/audit_fsm_author_factorial.py',
           '--study', 'output/fsm-author-nllfr-factorial-study-v1',
           '--process', 'output/fsm-author-budget-factorial-engineering-v1/original-process-01/process.json',
           '--freeze', 'output/fsm-author-budget-factorial-engineering-v1/auditor-freeze.json',
           '--output', 'output/fsm-author-nllfr-factorial-audit-v1/audit.json']
SOURCES = {
    'scripts/audit_fsm_author_factorial.py': '450aeb0112d8931bd92527ae1d40eb55161b3fe2f0b65a20dda628fa49da248b',
    'tests/test_audit_fsm_author_factorial.py': '15923277d7853df840f65ee08d95f94f6c01a4c41583472aefeafbe829f0087a',
    'scripts/fsm_nllfr_budget_audit_math.py': 'd4454df6f8854d1895d7edfee75dd1e36a58ec616fb12537a7424359a538f8c6',
    'scripts/audit_fsm_author_nllfr.py': '43c4787e316bc5c0a89653fece0c0694cd9f88041628b798c02382feb09dfa32',
}
HELD = {
    **SOURCES,
    'output/fsm-author-budget-factorial-engineering-v1/full-auditor-qualification-02/receipt.json': '00a520dfacdf9eb99f0741a611243dd431df5dce2d79f67b3706d8ee0d22195a',
    'output/fsm-author-budget-factorial-engineering-v1/full-producer-qualification-01/receipt.json': '8a3571cf0b1be9e92cee6578dca09dcc1c488f12a378dfcd753adb5e3c4d9bab',
    'output/fsm-author-budget-factorial-engineering-v1/full-producer-source-review.json': '1ce776803f21c892046fc7eb13a09d5a377c51b55fb8c0d92becb93b9ea299e2',
    'output/fsm-author-budget-factorial-engineering-v1/full-cross-schema-qualification-01/receipt.json': '093be0ee27cd7838349d75b67bd0a85d7396bc08d025e5d1d807090b87ab9db3',
    'output/fsm-author-budget-factorial-engineering-v1/full-harness-qualification-closure.json': '33936ccbd505304c491f14b47caab7cf37cef2fefbe1e2c3024d10bbd96dca15',
}
QUALIFIED_ROLES = {
    'producer_qualification': ENGINEERING/'full-producer-qualification-01/receipt.json',
    'source_review': ENGINEERING/'full-producer-source-review.json',
    'full_harness_qualification': ENGINEERING/'full-harness-qualification-closure.json',
    'cross_schema_qualification': ENGINEERING/'full-cross-schema-qualification-01/receipt.json',
    'independent_auditor_qualification': QUALIFICATION,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular evidence file: '+str(path))
    blob = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def closed_evaluation():
    """Reject absent/live evidence before imports; authenticate terminal failures later."""
    pins = {name: pin(path) for name, path in
            (('registration', REGISTRATION), ('freeze', FREEZE), ('process', EVALUATION_PROCESS))}
    terminal = read(EVALUATION_PROCESS)
    require(terminal.get('status') in ('completed', 'failed', 'timeout', 'memory_limit', 'supervisor_failed'),
            'original factorial evaluator has not closed')
    return pins


def admit():
    """Authenticate metadata/opaque hashes only, never call numerical audit()."""
    ready = closed_evaluation()
    pins = {name: pin(ROOT/name) for name in HELD}
    require(all(pins[k]['sha256'] == v for k, v in HELD.items()), 'held source/input drift')
    held = read(FREEZE)
    require(held['registration'] == ready['registration'] and held['qualification'] == pin(QUALIFICATION),
            'canonical registration and qualified independent freeze')
    spec = importlib.util.spec_from_file_location('held_factorial_audit_admission',
                                                ROOT/'scripts/audit_fsm_author_factorial.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    plan, admitted, paths, _terminal, _parents, _prior_paths = audit.authenticate(
        STUDY, EVALUATION_PROCESS, FREEZE)
    require(all(admitted[name] == value for name, value in ready.items())
            and closed_evaluation() == ready, 'original evaluator admission drift')
    require(all(paths.get(name) == path for name, path in QUALIFIED_ROLES.items()),
            'held producer and independent qualification roles')
    for name, digest in plan['source_sha256'].items():
        value = pin(ROOT/name)
        require(value['sha256'] == digest, 'registered source drift')
        pins['registered_source:'+name] = value
    for name, path in paths.items():
        pins['prerequisite:'+name] = pin(path)
    for name, expected in admitted['files'].items():
        value = pin(STUDY/name)
        require({k: value[k] for k in ('bytes', 'sha256')} == expected, 'evaluation payload drift')
        pins['evaluation_artifact:'+name] = value
    for name in ('registration', 'process', 'launch', 'freeze', 'qualification'):
        value = admitted[name]
        require(pin(value['path']) == value, 'audit input role drift')
        pins['admitted:'+name] = value
    if admitted['process_log'] is not None:
        require(pin(admitted['process_log']['path']) == admitted['process_log'], 'original evaluator log drift')
        pins['admitted:process_log'] = admitted['process_log']
    qualification = read(QUALIFICATION)
    preflight = read(qualification['preflight']['path'])
    for name, value in [('qualification_preflight', qualification['preflight']),
                        ('qualification_wrapper', preflight['wrapper']),
                        *[(f'qualification_log:{i}', row['log']) for i, row in enumerate(qualification['commands'])],
                        *[('qualification_snapshot:'+name, value) for name, value in preflight['snapshots'].items()]]:
        require(pin(value['path']) == value, 'qualification evidence drift')
        pins[name] = value
    require(preflight['executable'] == str(ROOT/'.venv/bin/python'), 'qualified audit interpreter')
    pins['helper'] = pin(__file__)
    return pins, admitted


def save(record, *, exclusive=False):
    with RECEIPT.open('x' if exclusive else 'w') as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False)+'\n')


def stop(child):
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    require(Path.cwd().resolve() == ROOT, 'run from repository root')
    require(not any(p.exists() or p.is_symlink() for p in (RECEIPT, LOG, OUTPUT.parent)),
            'original audit already attempted or output directory exists')
    before, admitted_before = admit()
    record = {'command': COMMAND, 'state': 'running', 'started_time_ns': time.time_ns(),
              'timeout_seconds': TIMEOUT_SECONDS, 'rss_cap_bytes': RSS_CAP_BYTES, 'environment': ENV,
              'sources_before': {k: before[k] for k in SOURCES}, 'inputs_before': before,
              'admission_before': admitted_before, 'helper': before['helper'],
              'qualification': before['admitted:qualification'], 'registration': before['admitted:registration'],
              'freeze': before['admitted:freeze'], 'evaluation_process': before['admitted:process'],
              'observed_exit_code': None, 'error': None,
              'scope': 'One original independent four-cell audit; no fitting, evaluation launch, retiming or retry. '
                       'Audit agreement does not promote incomplete training or an incomplete matrix.'}
    save(record, exclusive=True)
    start, wall_start, child, peak = time.monotonic(), time.time(), None, 0
    try:
        with LOG.open('xb') as handle:
            child = subprocess.Popen(COMMAND, cwd=ROOT, env={**os.environ, **ENV},
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            save(record)
            while child.poll() is None:
                if max(time.monotonic()-start, time.time()-wall_start) >= TIMEOUT_SECONDS:
                    record['state'] = 'TIMEOUT'
                    stop(child)
                    break
                probe = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(child.pid)],
                                       capture_output=True, text=True, timeout=2, check=False)
                rss = int(probe.stdout.strip())*1024 if probe.stdout.strip() else 0
                peak = max(peak, rss)
                if rss > RSS_CAP_BYTES:
                    record['state'] = 'MEMORY_LIMIT'
                    stop(child)
                    break
                time.sleep(.5)
            record['observed_exit_code'] = child.wait()
            if record['state'] == 'running':
                record['state'] = 'EXITED'
    except BaseException as exc:  # noqa: BLE001 - record interruptions and reap the sole child.
        # Preserve launch/interruption evidence and always reap any child.
        record['state'] = 'LAUNCH_FAILED' if child is None else 'WRAPPER_FAILED'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        stop(child)
        if child is not None:
            record['observed_exit_code'] = child.wait()
        record.update(elapsed_seconds=time.monotonic()-start, wall_elapsed_seconds=time.time()-wall_start,
                      finished_time_ns=time.time_ns(), peak_polled_child_rss_bytes=peak,
                      elapsed_scope='Original audit child and cleanup; metadata closure follows.',
                      deadline_scope='Maximum monotonic/realtime elapsed; no retry.',
                      memory_scope='Child RSS sampled every 0.5 seconds via ps KiB.')
        record['sources_after'], record['inputs_after'], record['admission_after'] = {}, {}, {}
        record['sources_unchanged'] = record['inputs_unchanged'] = False
        record['closure_error'] = None
        try:
            after, admitted_after = admit()
            record['inputs_after'], record['admission_after'] = after, admitted_after
            record['sources_after'] = {k: after[k] for k in SOURCES}
            record['sources_unchanged'] = record['sources_before'] == record['sources_after']
            record['inputs_unchanged'] = before == after and admitted_before == admitted_after
            require(record['sources_unchanged'] and record['inputs_unchanged'], 'post-audit identity drift')
        except Exception as exc:  # noqa: BLE001 - closure failure invalidates the terminal receipt.
            record['closure_error'] = f'{type(exc).__name__}: {exc}'
        record['evidence_errors'] = {}
        for key, path in (('log', LOG), ('audit_output', OUTPUT)):
            record[key] = None
            try:
                if path.exists():
                    record[key] = pin(path)
            except Exception as exc:  # noqa: BLE001 - unreadable evidence remains a failure.
                record['evidence_errors'][key] = f'{type(exc).__name__}: {exc}'
        record['audit_status'] = record['scientific_status'] = record['agreement'] = None
        if OUTPUT.exists():
            try:
                result = read(OUTPUT)
                require(result['study'] == str(STUDY) and result['inputs'] == admitted_before,
                        'audit output admission identity')
                record.update(audit_status=result['status'], agreement=result['agreement'],
                              scientific_status=result['scientific_status'])
            except Exception as exc:  # noqa: BLE001 - malformed audit output preserves failed closure.
                record['evidence_errors']['audit_read'] = f'{type(exc).__name__}: {exc}'
        record['success'] = (record['state'] == 'EXITED' and record['observed_exit_code'] == 0
                             and record['sources_unchanged'] and record['inputs_unchanged']
                             and record['audit_status'] == 'PASS' and record['agreement'] is True
                             and record['log'] is not None and record['audit_output'] is not None
                             and not record['evidence_errors'] and record['error'] is None
                             and record['closure_error'] is None)
        record['wrapper_elapsed_seconds'] = time.monotonic()-start
        save(record)
    print(json.dumps({k: record[k] for k in ('state', 'observed_exit_code', 'success',
                                            'elapsed_seconds', 'scientific_status', 'audit_output')}, indent=2))
    return 0 if record['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
