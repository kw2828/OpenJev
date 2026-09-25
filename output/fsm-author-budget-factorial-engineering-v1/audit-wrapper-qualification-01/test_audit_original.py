"""Fabricated launcher checks; metadata admission and numerical audit are stubbed."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load(tmp_path, monkeypatch):
    source = Path(__file__).with_name('audit-original.py')
    spec = importlib.util.spec_from_file_location('factorial_audit_launcher', source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(helper, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    for name, relative in {
        'REGISTRATION': 'registration.json', 'FREEZE': 'freeze.json',
        'EVALUATION_PROCESS': 'process.json', 'RECEIPT': 'audit-process.json',
        'LOG': 'audit-process.log', 'OUTPUT': 'audit/audit.json', 'STUDY': 'study',
    }.items():
        monkeypatch.setattr(helper, name, tmp_path/relative)
    return helper


@pytest.mark.parametrize('missing', ['REGISTRATION', 'FREEZE', 'EVALUATION_PROCESS', None])
def test_missing_or_live_parent_never_imports_or_launches(tmp_path, monkeypatch, missing):
    helper = load(tmp_path, monkeypatch)
    for name in ('REGISTRATION', 'FREEZE', 'EVALUATION_PROCESS'):
        if name != missing:
            getattr(helper, name).write_text(json.dumps({'status': 'running'}))

    def forbidden(*args, **kwargs):
        raise AssertionError('Unready parent reached model import or child launch')

    monkeypatch.setattr(helper.importlib.util, 'spec_from_file_location', forbidden)
    monkeypatch.setattr(helper.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError):
        helper.main()
    assert not helper.RECEIPT.exists()
    assert not helper.LOG.exists()
    assert not helper.OUTPUT.parent.exists()


@pytest.mark.parametrize('terminal', ['completed', 'failed', 'timeout', 'memory_limit', 'supervisor_failed'])
def test_terminal_outcomes_reach_only_the_later_admission_stage(tmp_path, monkeypatch, terminal):
    helper = load(tmp_path, monkeypatch)
    helper.REGISTRATION.write_text('{}')
    helper.FREEZE.write_text('{}')
    helper.EVALUATION_PROCESS.write_text(json.dumps({'status': terminal}))
    assert helper.closed_evaluation()['process'] == helper.pin(helper.EVALUATION_PROCESS)
    assert not helper.RECEIPT.exists()


@pytest.mark.parametrize('mode', ['incomplete', 'nonzero', 'timeout', 'wrong_output', 'drift'])
def test_single_stub_child_preserves_outcomes_and_blocks_retry(tmp_path, monkeypatch, mode):
    helper = load(tmp_path, monkeypatch)
    source = tmp_path/'source.py'
    source.write_text('# fabricated evidence\n')
    descriptor = helper.pin(source)
    monkeypatch.setattr(helper, 'SOURCES', {'source.py': descriptor['sha256']})
    pins = dict.fromkeys(('source.py', 'helper', 'admitted:qualification', 'admitted:registration',
                          'admitted:freeze', 'admitted:process'), descriptor)
    admitted = {'fabricated': 'metadata'}
    calls = []

    def admit():
        calls.append(True)
        if mode == 'drift' and len(calls) > 1:
            return {**pins, 'changed': descriptor}, admitted
        return pins, admitted

    monkeypatch.setattr(helper, 'admit', admit)
    child = tmp_path/'stub_child.py'
    child.write_text(
        'import json, sys, time\n'
        'from pathlib import Path\n'
        'mode, output, study = sys.argv[1:]\n'
        'print("fabricated child only", flush=True)\n'
        'if mode == "timeout": time.sleep(10)\n'
        'if mode == "nonzero": sys.exit(2)\n'
        'path = Path(output)\n'
        'path.parent.mkdir()\n'
        'value = {"status": "PASS", "agreement": True, "scientific_status": "REFERENCE_INCOMPLETE",\n'
        '         "study": study, "inputs": {"fabricated": "metadata"}}\n'
        'if mode == "wrong_output": value["inputs"] = {}\n'
        'path.write_text(json.dumps(value))\n'
    )
    monkeypatch.setattr(helper, 'COMMAND', [sys.executable, str(child), mode, str(helper.OUTPUT), str(helper.STUDY)])
    monkeypatch.setattr(helper, 'TIMEOUT_SECONDS', .05 if mode == 'timeout' else 10)
    code = helper.main()
    receipt = helper.read(helper.RECEIPT)
    assert code == (0 if mode == 'incomplete' else 1)
    assert receipt['success'] is (mode == 'incomplete')
    assert len(calls) == 2
    assert receipt['log'] == helper.pin(helper.LOG)
    if mode == 'incomplete':
        assert receipt['scientific_status'] == 'REFERENCE_INCOMPLETE'
        assert receipt['agreement'] is True
    elif mode == 'timeout':
        assert receipt['state'] == 'TIMEOUT'
        assert receipt['observed_exit_code'] != 0
    elif mode == 'nonzero':
        assert receipt['observed_exit_code'] == 2
    elif mode == 'wrong_output':
        assert 'audit_read' in receipt['evidence_errors']
    else:
        assert receipt['inputs_unchanged'] is False
        assert receipt['closure_error']
    before = helper.RECEIPT.read_bytes()
    with pytest.raises(ValueError, match='already attempted'):
        helper.main()
    assert helper.RECEIPT.read_bytes() == before
    assert len(calls) == 2
