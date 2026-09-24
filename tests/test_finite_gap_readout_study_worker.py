"""Fabricated admission checks, without scientific or child-process execution."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/finite_gap_readout_study_worker.py'
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
