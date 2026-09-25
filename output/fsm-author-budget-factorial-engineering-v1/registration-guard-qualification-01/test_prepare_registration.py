"""Metadata-only guard checks; no empirical inputs or model imports."""
import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).with_name('prepare-registration.py')
    spec = importlib.util.spec_from_file_location('registration_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('case', ['running_fit', 'missing_audit', 'failed_audit'])
def test_premature_assembly_creates_nothing(tmp_path, monkeypatch, case):
    helper = load()
    monkeypatch.setattr(helper, 'ROOT', tmp_path)
    monkeypatch.setattr(helper, 'BUDGET', tmp_path/'budget')
    monkeypatch.chdir(tmp_path)
    fit = {'status': 'completed', 'observed_exit_code': 0, 'end_identity_matches': True}
    if case == 'running_fit':
        fit = {'status': 'running'}
    write(helper.BUDGET/'original-process-01/process.json', fit)
    if case == 'failed_audit':
        write(helper.BUDGET/'audit-process.json', {'state': 'EXITED', 'observed_exit_code': 1})
    outputs = [tmp_path/name for name in ('registration.json', 'candidate.json', 'freeze.json', 'assembly.json')]
    for name, value in zip(('REGISTRATION', 'CANDIDATE', 'FREEZE', 'RECEIPT'), outputs, strict=True):
        monkeypatch.setattr(helper, name, value)

    def unexpected(*args):
        raise AssertionError('Premature admission reached an import or write')

    monkeypatch.setattr(helper, 'load_module', unexpected)
    monkeypatch.setattr(helper, 'save', unexpected)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='has not closed successfully'):
        helper.prepare()
    after = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after
    assert not any(path.exists() for path in outputs)


def test_closed_parent_check_only_returns_bound_metadata(tmp_path):
    helper = load()
    fit, audit = tmp_path/'fit.json', tmp_path/'audit.json'
    write(fit, {'status': 'completed', 'observed_exit_code': 0, 'end_identity_matches': True})
    write(audit, {'state': 'EXITED', 'observed_exit_code': 0, 'success': True,
                  'agreement': True, 'audit_status': 'PASS'})
    assert helper.parent_ready(fit, audit) == {'fit': helper.pin(fit), 'audit': helper.pin(audit)}
    assert set(tmp_path.iterdir()) == {fit, audit}
