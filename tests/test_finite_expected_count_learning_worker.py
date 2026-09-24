"""Fabricated admission checks, without scientific or child-process execution."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/finite_expected_count_learning_worker.py'
SPEC = importlib.util.spec_from_file_location('finite_admission_fixture', PATH)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def closed_fixture(tmp_path):
    launch = {'pid': 123, 'pgid': 123, 'command': ['python', 'fixture.py'], 'cwd': '/fixture',
              'cap_seconds': 30, 'clock_backend': 'mach_continuous_time', 'started_ns': 100,
              'deadline_ns': 300, 'watchdog_sha256': 'a' * 64, 'clock_source_sha256': 'b' * 64}
    terminal = dict(launch, status='completed', returncode=0, group_absent=True,
                    timed_out=False, cleanup={'reaped': True, 'errors': []},
                    timing_available=True, error=None, finished_ns=200)
    path = tmp_path / 'original.terminal.json'
    path.write_text(json.dumps(terminal))
    receipt = {'supervision': str(tmp_path / 'original.launch.json'), 'launch': launch}
    return receipt, terminal, path


def test_accepts_actual_completed_supervisor_schema(tmp_path):
    receipt, _, path = closed_fixture(tmp_path)
    assert WORKER.closed_producer(receipt) == path


@pytest.mark.parametrize('field,value', [
    ('status', 'PASS'), ('returncode', 1), ('group_absent', False),
    ('timed_out', True), ('timing_available', False), ('error', 'failed'),
    ('finished_ns', 300), ('finished_ns', 99),
    ('cleanup', {'reaped': False, 'errors': []}),
    ('cleanup', {'reaped': True, 'errors': ['unreaped child']}),
])
def test_rejects_incomplete_or_out_of_budget_terminal(tmp_path, field, value):
    receipt, terminal, path = closed_fixture(tmp_path)
    terminal[field] = value
    path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError):
        WORKER.closed_producer(receipt)


@pytest.mark.parametrize('field', ['pid', 'pgid', 'command', 'cwd', 'cap_seconds',
                                 'clock_backend', 'started_ns', 'deadline_ns',
                                 'watchdog_sha256', 'clock_source_sha256'])
def test_rejects_substituted_original_launch(tmp_path, field):
    receipt, _, _ = closed_fixture(tmp_path)
    changed = copy.deepcopy(receipt)
    changed['launch'][field] = 'a different launch'
    with pytest.raises(ValueError):
        WORKER.closed_producer(changed)


def qualification_fixture(tmp_path):
    receipt, _, _ = closed_fixture(tmp_path)
    folder = tmp_path / 'qualification'
    folder.mkdir()
    (folder / 'checks.log').write_text('PASS\n')
    Path(receipt['supervision']).write_text(json.dumps(receipt['launch']))
    spec = {'output': str(folder), 'supervision': receipt['supervision'], 'cap_seconds': 30}
    plan = {'version': 'finite-expected-count-learning-v1', 'mode': 'study',
            'root': str(tmp_path), 'runtime': {}, 'sources': {},
            'config': {'epochs': 480}, 'phases': {'qualify': spec},
            'tests': ['fixture_test.py'], 'lint_sources': ['fixture.py'],
            'rss_limit_bytes': 100, 'output_limit_bytes': 100,
            'kernel_qualification': {'folder': 'fixture', 'files': {}}}
    engineering = tmp_path / 'engineering.json'
    engineering.write_text(json.dumps({**plan, 'mode': 'engineering'}))
    receipt.update({'phase': 'qualify', 'status': 'PASS', 'output': str(folder),
                    'plan': str(engineering), 'plan_sha256': WORKER.descriptor(engineering)['sha256'],
                    'sources_before': {}, 'sources_after': {}, 'files': WORKER.files(folder)})
    path = Path(str(folder) + '.receipt.json')
    path.write_text(json.dumps(receipt))
    return plan, path


def test_qualification_joins_actual_engineering_plan_payloads_and_phase(tmp_path):
    plan, path = qualification_fixture(tmp_path)
    assert WORKER.admit_qualification(plan, path) == tmp_path / 'original.terminal.json'


@pytest.mark.parametrize('changed', ['config', 'tests', 'path', 'plan_hash', 'phase', 'payload'])
def test_qualification_rejects_foreign_configuration_or_receipt(tmp_path, changed):
    plan, path = qualification_fixture(tmp_path)
    if changed == 'config':
        plan['config']['epochs'] = 481
    elif changed == 'tests':
        plan['tests'] = []
    elif changed == 'path':
        plan['phases']['qualify']['output'] += '-other'
    elif changed == 'payload':
        (Path(plan['phases']['qualify']['output']) / 'checks.log').write_text('CHANGED')
    else:
        receipt = json.loads(path.read_text())
        receipt['plan_sha256' if changed == 'plan_hash' else 'phase'] = 'wrong'
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        WORKER.admit_qualification(plan, path)
