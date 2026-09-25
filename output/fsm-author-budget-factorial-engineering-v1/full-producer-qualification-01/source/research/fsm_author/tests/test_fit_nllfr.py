# SPDX-License-Identifier: GPL-3.0-or-later
"""Opaque metadata and fabricated process tests. No author/model/data imports."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('fit_nllfr', SCRIPTS/'fit_nllfr.py')
fit = importlib.util.module_from_spec(spec)
sys.modules['fit_nllfr'] = fit
spec.loader.exec_module(fit)
spec = importlib.util.spec_from_file_location('test_nllfr_supervisor_module', SCRIPTS/'run_nllfr_study.py')
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def build(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, 'ROOT', tmp_path)
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    for name in fit.MIN_SOURCES:
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('opaque source '+name)
    sources = {name: fit.sha(tmp_path/name) for name in fit.MIN_SOURCES}
    extra_eval = 'research/fsm_author/scripts/evaluate_nllfr.py'
    (tmp_path/extra_eval).write_text('opaque evaluator')
    sources[extra_eval] = fit.sha(tmp_path/extra_eval)
    monkeypatch.setattr(supervisor, '__file__', str(tmp_path/'research/fsm_author/scripts/run_nllfr_study.py'))
    inputs = tmp_path/'evidence'
    inputs.mkdir()
    values = {}

    def put(key, value, filename=None):
        p = dump(inputs/(filename or key+'.json'), value)
        values[key] = {'path': str(p.relative_to(tmp_path)), 'sha256': fit.sha(p)}
        return p

    for key in ('runtime_preflight', 'producer_qualification', 'source_review'):
        put(key, {'status': 'PASS'})
    init = put('nllfr_runtime_initialization', {'opaque': 'fixed synthetic initialization'})
    definition = put('nllfr_runtime_definition', {'fixture': {'trainable_scalars': 7473},
        'sources': {name: fit.pin(tmp_path/name) for name in sources}})
    receipt = put('nllfr_runtime_receipt', {'status': 'PASS', 'definition': fit.pin(definition),
        'details': {'fresh_process_initialization_equal': True}})
    process = put('nllfr_runtime_process', {'status': 'PASS', 'definition': fit.pin(definition),
        'sources_unchanged': True, 'stages': [{'returncode': 0}, {'returncode': 0}],
        'files': {'fresh-initialization.json': fit.pin(init)}})
    put('nllfr_runtime_observation', {'observed_exit_code': 0, 'process': fit.pin(process),
                                     'execution_receipt': fit.pin(receipt)})
    bla = inputs/'bla'
    bla.mkdir()
    for key, filename in (('bla_final_zip', 'final.zip'), ('bla_final_npz', 'final.npz')):
        p = bla/filename
        p.write_bytes(b'opaque deliberately invalid numeric/archive bytes')
        values[key] = {'path': str(p.relative_to(tmp_path)), 'sha256': fit.sha(p)}
    for key, filename, status in (('bla_fit', 'fit.json', 'complete'),
                                   ('bla_summary', 'summary.json', 'REFERENCE_COMPLETE')):
        p = dump(bla/filename, {'status': status})
        values[key] = {'path': str(p.relative_to(tmp_path)), 'sha256': fit.sha(p)}
    registration = put('bla_registration', {'version': 'fabricated predecessor'})
    process = put('bla_process', {'status': 'completed', 'observed_exit_code': 0,
        'end_identity_matches': True, 'registration_sha256': fit.sha(registration),
        'fit.json_sha256': fit.sha(bla/'fit.json'), 'summary.json_sha256': fit.sha(bla/'summary.json')})
    audit = put('bla_audit', {'status': 'PASS', 'agreement': True, 'study': str(bla),
        'results': {'reference_status': 'REFERENCE_COMPLETE'},
        'inputs': {'process': fit.pin(process), 'registration': fit.pin(registration),
                   'files': {name: {k: fit.pin(bla/name)[k] for k in ('sha256', 'bytes')}
                             for name in ('final.zip', 'final.npz', 'fit.json', 'summary.json')}}})
    put('bla_audit_process', {'observed_exit_code': 0, 'audit_sha256': fit.sha(audit)})
    data = inputs/'combined_data.npz'
    data.write_bytes(b'invalid NPZ, never decoded by metadata admission')
    cfg = {'version': fit.VERSION, 'experiment': dict(fit.EXPERIMENT), 'source_sha256': sources,
           'prerequisites': values, 'output': 'output/fit', 'process_directory': 'output/fit-process',
           'data_path': str(data.relative_to(tmp_path)), 'data_sha256': fit.sha(data),
           'evaluation_output': 'output/evaluation', 'evaluation_process_directory': 'output/eval-process',
           'evaluation': {'outer_timeout_seconds': 3600, 'rss_cap_bytes': 32*1024**3}}
    reg = dump(tmp_path/'registration.json', cfg)
    return cfg, reg


def repin(cfg, reg, key, change):
    p = fit.prerequisite(cfg, key)
    value = fit.read(p)
    change(value)
    dump(p, value)
    cfg['prerequisites'][key]['sha256'] = fit.sha(p)
    dump(reg, cfg)


def test_opaque_admission_accepts_exact_closed_sources_without_decoding(tmp_path, monkeypatch):
    cfg, reg = build(tmp_path, monkeypatch)
    assert fit.metadata_admission(reg) == cfg
    assert fit.metadata_admission(reg, tmp_path/cfg['output']) == cfg


@pytest.mark.parametrize('key,change', [
    ('producer_qualification', lambda x: x.update(status='FAIL')),
    ('source_review', lambda x: x.update(status='FAIL')),
    ('nllfr_runtime_observation', lambda x: x.update(observed_exit_code=1)),
    ('nllfr_runtime_process', lambda x: x.update(sources_unchanged=False)),
    ('nllfr_runtime_receipt', lambda x: x['details'].update(fresh_process_initialization_equal=False)),
    ('bla_process', lambda x: x.update(status='running')),
    ('bla_process', lambda x: x.update(end_identity_matches=False)),
    ('bla_audit', lambda x: x.update(agreement=False)),
    ('bla_audit', lambda x: x['results'].update(reference_status='COMPLETE')),
    ('bla_audit_process', lambda x: x.update(observed_exit_code=1)),
])
def test_republished_bad_admission_still_rejected(tmp_path, monkeypatch, key, change):
    cfg, reg = build(tmp_path, monkeypatch)
    repin(cfg, reg, key, change)
    with pytest.raises(ValueError):
        fit.metadata_admission(reg)


@pytest.mark.parametrize('kind', ['source', 'data', 'missing_source', 'missing_prerequisite', 'recipe', 'output'])
def test_source_data_recipe_and_scope_tampering(tmp_path, monkeypatch, kind):
    cfg, reg = build(tmp_path, monkeypatch)
    if kind == 'source':
        (tmp_path/next(iter(cfg['source_sha256']))).write_text('mutated')
    elif kind == 'data':
        (tmp_path/cfg['data_path']).write_bytes(b'changed')
    elif kind == 'missing_source':
        cfg['source_sha256'].pop('src/openjev/research/fsm_data.py')
    elif kind == 'missing_prerequisite':
        cfg['prerequisites'].pop('bla_final_npz')
    elif kind == 'recipe':
        cfg['experiment']['max_iter'] = 20
    dump(reg, cfg)
    with pytest.raises(ValueError):
        fit.metadata_admission(reg, tmp_path/'wrong' if kind == 'output' else None)


@pytest.mark.parametrize('path', ['../outside', '/absolute', 'a//b', 'a/./b', 'a\\b'])
def test_noncanonical_paths_rejected(path):
    with pytest.raises(ValueError):
        fit.relative(path)


def closed_fit(tmp_path, monkeypatch):
    cfg, reg = build(tmp_path, monkeypatch)
    folder = tmp_path/cfg['output']
    folder.mkdir(parents=True)
    for name in ('final.zip', 'final.npz', 'trace.npz'):
        (folder/name).write_bytes(b'opaque original')
    dump(folder/'fit.json', {'status': 'iteration_cap_reached'})
    dump(folder/'summary.json', {'status': 'FIT_INCOMPLETE'})
    process_folder = tmp_path/cfg['process_directory']
    process_folder.mkdir()
    (process_folder/'process.log').write_text('original log')
    process = {'phase': 'fit', 'status': 'completed', 'observed_exit_code': 0, 'end_identity_matches': True,
        'registration_sha256': fit.sha(reg), 'log_sha256': fit.sha(process_folder/'process.log'),
        'command': [str(tmp_path/'research/fsm_author/.venv/bin/python'),
                    str(tmp_path/'research/fsm_author/scripts/fit_nllfr.py'),
                    '--registration', str(reg), '--output', str(folder)],
        'environment': fit.ENV, 'producer': fit.pin(tmp_path/'research/fsm_author/scripts/fit_nllfr.py'),
        'supervisor': fit.pin(supervisor.__file__), 'timeout_seconds': 18000, 'rss_cap_bytes': 32*1024**3,
        'artifacts': supervisor.inventory(folder)}
    path = dump(process_folder/'process.json', process)
    return cfg, reg, process, path


def test_finite_iteration_cap_may_only_enter_separate_diagnostic_evaluation(tmp_path, monkeypatch):
    cfg, reg, _, path = closed_fit(tmp_path, monkeypatch)
    assert supervisor.check_fit_closed(cfg, reg) == fit.pin(path)
    assert fit.read(tmp_path/cfg['output']/'summary.json')['status'] == 'FIT_INCOMPLETE'


@pytest.mark.parametrize('kind', ['trace_change', 'extra_file', 'missing_file', 'symlink',
                                  'command', 'environment', 'producer', 'supervisor', 'cap', 'phase'])
def test_full_original_closure_rejects_tampering(tmp_path, monkeypatch, kind):
    cfg, reg, process, path = closed_fit(tmp_path, monkeypatch)
    folder = tmp_path/cfg['output']
    if kind == 'trace_change':
        (folder/'trace.npz').write_bytes(b'changed')
    elif kind == 'extra_file':
        (folder/'extra').write_text('extra')
    elif kind == 'missing_file':
        (folder/'trace.npz').unlink()
    elif kind == 'symlink':
        (folder/'unexpected').symlink_to(folder/'trace.npz')
    elif kind == 'command':
        process['command'][1] = 'other_script.py'
    elif kind == 'environment':
        process['environment'] = {**fit.ENV, 'PYTHONHASHSEED': '1'}
    elif kind in ('producer', 'supervisor'):
        process[kind]['sha256'] = '0'*64
    elif kind == 'cap':
        process['timeout_seconds'] += 1
    else:
        process['phase'] = 'evaluate'
    dump(path, process)
    with pytest.raises(ValueError):
        supervisor.check_fit_closed(cfg, reg)


@pytest.mark.parametrize('destination', ['evaluation_output', 'evaluation_process_directory'])
def test_evaluation_cannot_nest_in_original_evidence(tmp_path, monkeypatch, destination):
    cfg, reg, _, _ = closed_fit(tmp_path, monkeypatch)
    cfg[destination] = cfg['output']+'/bad'
    monkeypatch.setattr(supervisor, 'metadata_admission', lambda *a: cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='overlaps'):
        supervisor.supervise(reg, 'evaluate')


@pytest.mark.parametrize('outcome', ['completed', 'failed', 'timeout', 'memory_limit'])
def test_original_process_lifecycle_preserves_terminal_failure(tmp_path, monkeypatch, outcome):
    cfg, reg = build(tmp_path, monkeypatch)
    if outcome == 'timeout':
        cfg['experiment']['outer_timeout_seconds'] = 0
    if outcome == 'memory_limit':
        cfg['experiment']['rss_cap_bytes'] = 1
    monkeypatch.setattr(supervisor, 'metadata_admission', lambda *a: cfg)
    monkeypatch.chdir(tmp_path)
    killed, calls = [], []

    class Child:
        pid = 123456
        checks = 0
        code = None

        def poll(self):
            self.checks += 1
            if outcome in ('completed', 'failed') and self.checks >= 3:
                self.code = 0 if outcome == 'completed' else 7
            return self.code

        def wait(self):
            return self.code

    child = Child()

    def launch(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['PYTHONHASHSEED'] == '0'
        assert kwargs['start_new_session'] is True
        output = tmp_path/cfg['output']
        output.mkdir(parents=True)
        (output/'initial.npz').write_bytes(b'opaque partial initial evidence')
        return child

    def kill(pid, sig):
        killed.append(pid)
        child.code = -9

    monkeypatch.setattr(supervisor.subprocess, 'Popen', launch)
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='1024'))
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    monkeypatch.setattr(supervisor.os, 'killpg', kill)
    status = supervisor.supervise(reg)
    receipt = fit.read(tmp_path/cfg['process_directory']/'process.json')
    assert receipt['status'] == outcome and status == (0 if outcome == 'completed' else 1)
    assert len(calls) == 1
    assert receipt['artifacts']['initial.npz']['sha256'] == fit.sha(tmp_path/cfg['output']/'initial.npz')
    assert receipt['end_identity_matches'] is True
    assert bool(killed) == (outcome in ('timeout', 'memory_limit'))
    assert receipt['outcome'] == ('ORIGINAL_PROCESS_COMPLETE' if outcome == 'completed' else 'INCOMPLETE')
    with pytest.raises(ValueError, match='already exists'):
        supervisor.supervise(reg)
    assert len(calls) == 1
