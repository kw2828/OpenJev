"""Fabricated metadata admission and fixed projection checks only."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


W = load('finite_reuse_learning_worker')
P = load('qualify_finite_reuse_exposure')


def closed_fixture(tmp_path):
    launch = {'version': 'dialogue-observation-supervision-v2',
              'pid': 123, 'pgid': 123, 'parent_pid': 122, 'command': ['python', 'fixture.py'],
              'cwd': '/fixture', 'cap_seconds': 30, 'clock_backend': 'mach_continuous_time',
              'started_ns': 100, 'deadline_ns': 30000000100,
              'watchdog_sha256': 'a' * 64, 'clock_source_sha256': 'b' * 64}
    terminal = dict(launch, status='completed', returncode=0, group_absent=True,
                    timed_out=False, cleanup={'reaped': True, 'group_absent': True, 'errors': [], 'signals': []},
                    timing_available=True, error=None, clock_error=None, finished_ns=200,
                    elapsed_ns=100, wall_seconds=1e-7)
    path = tmp_path / 'original.terminal.json'
    path.write_text(json.dumps(terminal))
    receipt = {'supervision': str(tmp_path / 'original.launch.json'), 'launch': launch}
    return receipt, terminal, path


def test_original_successful_closure(tmp_path):
    receipt, _, path = closed_fixture(tmp_path)
    assert W.closed_producer(receipt) == path


@pytest.mark.parametrize('field,value', [
    ('status', 'PASS'), ('returncode', 1), ('group_absent', False), ('timed_out', True),
    ('timing_available', False), ('error', 'failure'), ('clock_error', 'unavailable'),
    ('finished_ns', 30000000100), ('finished_ns', 99), ('elapsed_ns', 99), ('wall_seconds', 1.),
    ('cleanup', {'reaped': False, 'group_absent': True, 'errors': [], 'signals': []}),
    ('cleanup', {'reaped': True, 'group_absent': True, 'errors': [], 'signals': ['SIGTERM']}),
])
def test_failure_or_misreported_terminal_cannot_admit(tmp_path, field, value):
    receipt, terminal, path = closed_fixture(tmp_path)
    terminal[field] = value
    path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError):
        W.closed_producer(receipt)


@pytest.mark.parametrize('field', ['pid', 'pgid', 'command', 'cwd', 'cap_seconds',
                                 'clock_backend', 'started_ns', 'deadline_ns',
                                 'watchdog_sha256', 'clock_source_sha256'])
def test_original_launch_substitution_rejected(tmp_path, field):
    receipt, _, _ = closed_fixture(tmp_path)
    changed = copy.deepcopy(receipt)
    changed['launch'][field] = 'foreign'
    with pytest.raises(ValueError):
        W.closed_producer(changed)


def snapshot_fixture(tmp_path):
    source = b'example source bytes\n'
    path = tmp_path / 'source.py'
    path.write_bytes(source)
    plan = {'mode': 'engineering', 'sources': {'source.py': W.descriptor(path)}}
    plan_path = tmp_path / 'engineering-registration-01.json'
    plan_path.write_text(json.dumps(plan))
    snapshot = tmp_path / 'source-snapshot-engineering-01'
    snapshot.mkdir()
    (snapshot / 'source.py').write_bytes(source)
    W.publish(snapshot / 'manifest.json', {
        'registration': {'path': str(plan_path), **W.descriptor(plan_path)}, 'sources': plan['sources']})
    return plan_path, plan, snapshot


def test_snapshot_joins_registration_and_complete_source_roster(tmp_path):
    path, plan, _ = snapshot_fixture(tmp_path)
    W.validate_snapshot(path, plan)


@pytest.mark.parametrize('change', ['source', 'extra', 'registration', 'manifest'])
def test_snapshot_rejects_changed_or_incomplete_proof(tmp_path, change):
    path, plan, snapshot = snapshot_fixture(tmp_path)
    if change == 'source':
        (snapshot / 'source.py').write_text('changed')
    elif change == 'extra':
        (snapshot / 'extra.py').write_text('extra')
    elif change == 'registration':
        path.write_text(path.read_text() + '\n')
    else:
        manifest = json.loads((snapshot / 'manifest.json').read_text())
        manifest['registration']['sha256'] = 'f' * 64
        (snapshot / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        W.validate_snapshot(path, plan)


def test_projection_charges_scaled_stages_and_unscaled_nonstage_work():
    allocations, fits = {}, []
    for arm in ('original_free', 'matched_free', 'rounded'):
        allocations[arm] = {'stages': [{'start_elapsed': .2, 'stopped_elapsed': .7},
                                      {'start_elapsed': .8, 'stopped_elapsed': 1.8}]}
        fits.append({'arm': arm, 'seed': 944301, 'seconds': 2.})
    rows = P.projections(allocations, fits)
    assert len(rows) == 3
    for row in rows:
        assert row['stage_seconds'] == pytest.approx([.5, 1.])
        assert row['nonstage_seconds'] == pytest.approx(.5)
        assert row['projected_seconds'] == pytest.approx(2 * (.5 * 32 + 1 * 48) + .5)
        assert row['safety_multiplier'] == 2.
    assert P.EXPOSURE['maximum_projected_seconds'] == 90.
    assert P.EXPOSURE['target_counts'] == {'prefix_updates': 1024, 'joint_updates': 3072}


@pytest.mark.parametrize('change', ['negative', 'nonfinite', 'missing_stage', 'negative_overhead'])
def test_projection_cannot_hide_unusable_timing(change):
    allocations = {arm: {'stages': [{'start_elapsed': 0., 'stopped_elapsed': 1.},
                                   {'start_elapsed': 1., 'stopped_elapsed': 2.}]}
                   for arm in ('original_free', 'matched_free', 'rounded')}
    fits = [{'arm': arm, 'seed': 944301, 'seconds': 3.} for arm in allocations]
    if change == 'missing_stage':
        allocations['rounded']['stages'].pop()
    elif change == 'negative_overhead':
        fits[0]['seconds'] = 1.
    else:
        allocations['rounded']['stages'][0]['stopped_elapsed'] = -1. if change == 'negative' else float('nan')
    with pytest.raises(ValueError):
        P.projections(allocations, fits)


def test_qualification_selects_new_probe_and_tests():
    plan = {'root': str(ROOT), 'runtime': {'executable': '/fixture/python'},
            'lint_sources': ['fixture.py'], 'tests': ['fixture_test.py'],
            'phases': {'qualify': {'output': '/fixture/qualification'}}}
    commands = W.qualification_commands(plan)
    assert commands[0] == [str(ROOT / '.venv/bin/ruff'), 'check', 'fixture.py']
    assert commands[1][-2:] == ['--noconftest', 'fixture_test.py']
    assert commands[2] == ['/fixture/python', str(ROOT / 'scripts/qualify_finite_reuse_exposure.py'),
                           '--output', '/fixture/qualification/exposure']
