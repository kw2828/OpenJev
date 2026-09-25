"""Bind closed audited checkpoints for one future factorial evaluation.

Metadata and opaque hashes only. No fitting, array decoding, evaluation, audit
replay, publication or child-process launch. Refuse an unclosed parent before
imports or file creation. Preserve a partial failed assembly without retry.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
ENGINEERING = ROOT/'output/fsm-author-budget-factorial-engineering-v1'
BUDGET = ROOT/'output/fsm-author-nllfr-budget-engineering-v1'
REGISTRATION = ROOT/'research/fsm-author-nllfr-factorial-registration.json'
CANDIDATE = ENGINEERING/'registration-candidate.json'
FREEZE = ENGINEERING/'auditor-freeze.json'
RECEIPT = ENGINEERING/'registration-assembly.json'
HELD = {
    'research/fsm_author/scripts/evaluate_nllfr_factorial.py': '1ff7b7e8e675ab9be475e7aca1bf8e54c0f348b77fbe8036a5d4e20196d13a60',
    'scripts/audit_fsm_author_factorial.py': '450aeb0112d8931bd92527ae1d40eb55161b3fe2f0b65a20dda628fa49da248b',
    'tests/test_audit_fsm_author_factorial.py': '15923277d7853df840f65ee08d95f94f6c01a4c41583472aefeafbe829f0087a',
    'output/fsm-author-budget-factorial-engineering-v1/full-producer-qualification-01/receipt.json': '8a3571cf0b1be9e92cee6578dca09dcc1c488f12a378dfcd753adb5e3c4d9bab',
    'output/fsm-author-budget-factorial-engineering-v1/full-producer-source-review.json': '1ce776803f21c892046fc7eb13a09d5a377c51b55fb8c0d92becb93b9ea299e2',
    'output/fsm-author-budget-factorial-engineering-v1/full-auditor-qualification-02/receipt.json': '00a520dfacdf9eb99f0741a611243dd431df5dce2d79f67b3706d8ee0d22195a',
    'output/fsm-author-budget-factorial-engineering-v1/full-cross-schema-qualification-01/receipt.json': '093be0ee27cd7838349d75b67bd0a85d7396bc08d025e5d1d807090b87ab9db3',
    'output/fsm-author-budget-factorial-engineering-v1/full-harness-qualification-closure.json': '33936ccbd505304c491f14b47caab7cf37cef2fefbe1e2c3024d10bbd96dca15',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular evidence file required: '+str(path))
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def bound(value):
    require(pin(value['path']) == value, 'evidence descriptor drift')
    return Path(value['path'])


def parent_ready(fit_path, audit_path):
    fit = read(fit_path)
    require(fit['status'] == 'completed' and fit['observed_exit_code'] == 0
            and fit['end_identity_matches'] is True, 'original fit has not closed successfully')
    require(audit_path.is_file(), 'original FIT audit has not closed successfully')
    audit = read(audit_path)
    require(audit['state'] == 'EXITED' and audit['observed_exit_code'] == 0
            and audit['success'] is True and audit['agreement'] is True
            and audit['audit_status'] == 'PASS', 'original FIT audit has not closed successfully')
    return {'fit': pin(fit_path), 'audit': pin(audit_path)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualified_definition(receipt):
    q = read(receipt)
    definition_path = bound(q['definition'])
    definition = read(definition_path)
    require(q['status'] == 'PASS' and q['sources_before'] == q['sources_after'] == definition['sources'],
            'successful immutable qualification required')
    for name, value in definition['sources'].items():
        require(bound(value) == ROOT/name, 'current qualified source role')
        snapshot = pin(definition_path.parent/'source'/name)
        require(all(snapshot[k] == value[k] for k in ('bytes', 'sha256')), 'qualified snapshot drift')
    require([r['command'] for r in q['commands']] == definition['commands'], 'qualified command roster')
    for row in q['commands']:
        require(row['returncode'] == 0, 'successful original qualification command')
        bound(row['log'])
    return definition


def save(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare():
    require(Path.cwd().resolve() == ROOT, 'run from repository root')
    parents = parent_ready(BUDGET/'original-process-01/process.json', BUDGET/'audit-process.json')
    require(not any(p.exists() for p in (REGISTRATION, CANDIDATE, FREEZE, RECEIPT)),
            'registration assembly already attempted; preserve original evidence')
    for name, digest in HELD.items():
        require(pin(ROOT/name)['sha256'] == digest, 'held preparation input drift: '+name)
    producer_definition = qualified_definition(ENGINEERING/'full-producer-qualification-01/receipt.json')
    qualified_definition(ENGINEERING/'full-cross-schema-qualification-01/receipt.json')
    closure = read(ENGINEERING/'full-harness-qualification-closure.json')
    require(closure['status'] == 'QUALIFIED_BEFORE_EMPIRICAL_EVALUATION', 'full harness qualification')
    for value in closure['sources'].values():
        bound(value)
    for row in closure['qualifications']:
        require(read(bound(row['receipt']))['status'] == 'PASS', 'qualification closure')
        bound(row['pytest_log'])
    sys.path[:0] = [str(ROOT/'research/fsm_author/scripts'), str(ROOT/'research/fsm_author/src')]
    producer = load_module('factorial_registration_producer', ROOT/'research/fsm_author/scripts/evaluate_nllfr_factorial.py')
    auditor = load_module('factorial_registration_auditor', ROOT/'scripts/audit_fsm_author_factorial.py')
    original = read(ROOT/'research/fsm-author-nllfr-registration.json')
    paths = {
        'runtime_preflight': ROOT/'output/fsm-author-engineering-v1/runtime-preflight-01/receipt.json',
        'old_registration': ROOT/'research/fsm-author-nllfr-registration.json',
        'old_process': ROOT/'output/fsm-author-engineering-v1/nllfr-original-process-01/process.json',
        'old_evaluation_process': ROOT/'output/fsm-author-engineering-v1/nllfr-evaluation-process-01/process.json',
        'old_audit': ROOT/'output/fsm-author-nllfr-audit-v1/audit.json',
        'old_audit_process': ROOT/'output/fsm-author-engineering-v1/nllfr-audit-process.json',
        'new_registration': ROOT/'research/fsm-author-nllfr-budget-registration.json',
        'new_process': BUDGET/'original-process-01/process.json',
        'new_audit': ROOT/'output/fsm-author-nllfr-budget-audit-v1/audit.json',
        'new_audit_process': BUDGET/'audit-process.json',
        'new_audit_freeze': BUDGET/'auditor-freeze.json',
        'old_final': ROOT/'output/fsm-author-nllfr-study-v1/final.npz',
        'new_final': ROOT/'output/fsm-author-nllfr-budget-study-v1/final.npz',
        'common_normalizer': ROOT/original['prerequisites']['common_normalizer']['path'],
        'producer_qualification': ENGINEERING/'full-producer-qualification-01/receipt.json',
        'source_review': ENGINEERING/'full-producer-source-review.json',
        'full_harness_qualification': ENGINEERING/'full-harness-qualification-closure.json',
        'cross_schema_qualification': ENGINEERING/'full-cross-schema-qualification-01/receipt.json',
        'independent_auditor_qualification': ENGINEERING/'full-auditor-qualification-02/receipt.json',
    }
    cfg = {'version': producer.VERSION, 'experiment': producer.EXPERIMENT,
           'source_sha256': {n: v['sha256'] for n, v in producer_definition['sources'].items()},
           'prerequisites': {n: {'path': str(p.relative_to(ROOT)), 'sha256': pin(p)['sha256']} for n, p in paths.items()},
           'output': 'output/fsm-author-nllfr-factorial-study-v1',
           'process_directory': 'output/fsm-author-budget-factorial-engineering-v1/original-process-01'}
    require(not any((ROOT/cfg[k]).exists() for k in ('output', 'process_directory')), 'original evaluation path exists')
    producer.qualified(cfg, paths)
    admission = producer.parent_admission(paths)
    require(parent_ready(paths['new_process'], paths['new_audit_process']) == parents, 'parent closure drift')
    written = []
    try:
        save(CANDIDATE, cfg)
        written.append(pin(CANDIDATE))
        checked, _, actual = producer.metadata_admission(CANDIDATE)
        require(checked == cfg and actual == admission, 'candidate admission drift')
        with REGISTRATION.open('xb') as handle:
            handle.write(CANDIDATE.read_bytes())
        written.append(pin(REGISTRATION))
        checked, _, actual = producer.metadata_admission(REGISTRATION)
        require(checked == cfg and actual == admission, 'canonical admission drift')
        frozen = {'status': 'FROZEN_BEFORE_EMPIRICAL_AUDIT', 'created_time_ns': time.time_ns(),
                  'sources': {n: pin(ROOT/n) for n in (*auditor.SOURCES, auditor.MATH, auditor.HELPER)},
                  'registration': pin(REGISTRATION),
                  'qualification': pin(paths['independent_auditor_qualification'])}
        save(FREEZE, frozen)
        written.append(pin(FREEZE))
        held, registered = auditor.source_freeze(FREEZE)
        require(held == frozen and registered == REGISTRATION, 'independent freeze drift')
        require(parent_ready(paths['new_process'], paths['new_audit_process']) == parents, 'final parent drift')
        save(RECEIPT, {'status': 'PASS', 'helper': pin(__file__), 'parents': parents,
                       'registration': pin(REGISTRATION), 'freeze': pin(FREEZE), 'candidate': pin(CANDIDATE),
                       'source_count': len(cfg['source_sha256']), 'prerequisite_count': len(paths),
                       'new_fit_status': admission['new_fit_status'],
                       'new_fit_iterations': admission['new_fit_iterations'],
                       'scope': 'Metadata admission only. No array decode, model call, evaluation, audit replay or publication.'})
    except BaseException as exc:  # Retain partial assembly; never delete or silently retry.
        if not RECEIPT.exists():
            save(RECEIPT, {'status': 'FAIL', 'helper': pin(__file__), 'written': written,
                           'error': {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()},
                           'scope': 'Partial metadata assembly retained. No evaluation was launched.'})
        raise
    return pin(RECEIPT)


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
