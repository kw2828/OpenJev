"""One original FIT-only budget audit, after its supervised fit has closed."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINEERING = ROOT/'output/fsm-author-nllfr-budget-engineering-v1'
STUDY = ROOT/'output/fsm-author-nllfr-budget-study-v1'
FIT_PROCESS = ENGINEERING/'original-process-01/process.json'
REGISTRATION = ROOT/'research/fsm-author-nllfr-budget-registration.json'
FREEZE = ENGINEERING/'auditor-freeze.json'
RECEIPT = ENGINEERING/'audit-process.json'
LOG = ENGINEERING/'audit-process.log'
OUTPUT = ROOT/'output/fsm-author-nllfr-budget-audit-v1/audit.json'
TIMEOUT_SECONDS = 3600
RSS_CAP_BYTES = 32*1024**3
ENV = {**dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                       'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1'),
       'PYTHONDONTWRITEBYTECODE': '1'}
COMMAND = ['.venv/bin/python', '-u', 'scripts/audit_fsm_author_nllfr_budget.py',
           '--study', 'output/fsm-author-nllfr-budget-study-v1',
           '--process', 'output/fsm-author-nllfr-budget-engineering-v1/original-process-01/process.json',
           '--freeze', 'output/fsm-author-nllfr-budget-engineering-v1/auditor-freeze.json',
           '--output', 'output/fsm-author-nllfr-budget-audit-v1/audit.json']
SOURCES = {
    'scripts/audit_fsm_author_nllfr_budget.py': '1cc2d01bc4210ed40cc2bc5e2f128c834d383324ffb7c7b23271660c08526a49',
    'tests/test_audit_fsm_author_nllfr_budget.py': '5e0051746326b26fb49fd88e06e3318a882d343c3e005f2de71defb43d2b94bd',
    'scripts/audit_fsm_author_nllfr.py': '43c4787e316bc5c0a89653fece0c0694cd9f88041628b798c02382feb09dfa32',
}
HELD = {
    **SOURCES,
    'research/fsm-author-nllfr-budget-registration.json': '0797356f741ccdea572183cc06e8bedb36ba996b419af1394f27fa8e291934eb',
    'output/fsm-author-nllfr-budget-engineering-v1/auditor-freeze.json': '231331f702aa858a91b017f6ec35a7e8472fe700636aa3e05cbf5a83c0b30757',
    'output/fsm-author-nllfr-budget-engineering-v1/audit-qualification-01/receipt.json': 'cfaa35a1f35d6f0bddeda908702b72878663c486efca7f3cb1553dd8961fb76a',
    'output/fsm-author-nllfr-budget-engineering-v1/prefit-freeze.json': '7b3cc99aed41ab0dbe60c5dc8f899eecd042b93624ada8f4eb7c358a71abd1f0',
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


def admit():
    """Authenticate only metadata/opaque hashes; never call numeric audit/replay."""
    pins = {name: pin(ROOT/name) for name in HELD}
    require(all(pins[k]['sha256'] == v for k, v in HELD.items()), 'held source/input drift')
    spec = importlib.util.spec_from_file_location('held_budget_audit_admission',
                                                ROOT/'scripts/audit_fsm_author_nllfr_budget.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    plan, admitted, paths, _terminal, _parent, _parent_fit = audit.authenticate(STUDY, FIT_PROCESS, FREEZE)
    for name, digest in plan['source_sha256'].items():
        value = pin(ROOT/name)
        require(value['sha256'] == digest, 'registered source drift')
        pins['registered_source:'+name] = value
    for name, path in paths.items():
        pins['prerequisite:'+name] = pin(path)
    for name, expected in admitted['files'].items():
        value = pin(STUDY/name)
        require({k: value[k] for k in ('bytes', 'sha256')} == expected, 'fit payload drift')
        pins['fit_artifact:'+name] = value
    for name in ('registration', 'process', 'launch', 'freeze', 'qualification'):
        value = admitted[name]
        require(pin(value['path']) == value, 'audit input role drift')
        pins['admitted:'+name] = value
    if admitted['process_log'] is not None:
        pins['admitted:process_log'] = admitted['process_log']
    held = read(FREEZE)
    qualification = read(held['qualification']['path'])
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
    require(not RECEIPT.exists() and not LOG.exists() and not OUTPUT.parent.exists(),
            'original audit already attempted or output directory exists')
    before, admitted_before = admit()
    record = {'command': COMMAND, 'state': 'running', 'started_time_ns': time.time_ns(),
              'timeout_seconds': TIMEOUT_SECONDS, 'rss_cap_bytes': RSS_CAP_BYTES, 'environment': ENV,
              'sources_before': {k: before[k] for k in SOURCES}, 'inputs_before': before,
              'admission_before': admitted_before, 'helper': before['helper'],
              'qualification': before['admitted:qualification'], 'registration': before['admitted:registration'],
              'freeze': before['admitted:freeze'], 'fit_process': before['admitted:process'],
              'observed_exit_code': None, 'error': None,
              'scope': 'One original independent FIT-only audit; no retry or DEV evaluation. '
                       'Agreement does not promote incomplete training.'}
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
    except BaseException as exc:
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
                      memory_scope='Child RSS sampled every0.5s via ps KiB.')
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
        except Exception as exc:
            record['closure_error'] = f'{type(exc).__name__}: {exc}'
        record['evidence_errors'] = {}
        for key, path in (('log', LOG), ('audit_output', OUTPUT)):
            record[key] = None
            try:
                if path.exists():
                    record[key] = pin(path)
            except Exception as exc:
                record['evidence_errors'][key] = f'{type(exc).__name__}: {exc}'
        record['audit_status'] = record['scientific_status'] = record['agreement'] = None
        if OUTPUT.exists():
            try:
                result = read(OUTPUT)
                record.update(audit_status=result['status'], agreement=result['agreement'],
                              scientific_status=result['scientific_status'])
            except Exception as exc:
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
