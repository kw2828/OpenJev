"""Fabricated admission checks, with no scientific data or model execution."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import finite_action_range_study as s


def launch_fixture():
    plan = {'mode': 'study', 'phases': s.phase_specs(), 'sources': {
        'scripts/supervise_dialogue_observation_v2.py': {'sha256': 'watchdog'},
        'src/openjev/research/suspend_clock.py': {'sha256': 'clock'}}}
    spec = plan['phases']['fit']
    launch = {'command': [sys.executable, str(s.ROOT / 'scripts/finite_action_range_study.py'),
        '--phase', 'fit', '--plan-sha256', 'sha', '--supervision', spec['supervision']],
        'cwd': str(s.ROOT), 'cap_seconds': 1800,
        'version': 'dialogue-observation-supervision-v2', 'clock_backend': 'mach_continuous_time',
        'pid': 23, 'pgid': 23, 'parent_pid': 22, 'started_ns': 1000,
        'deadline_ns': 1000 + 1800 * 10**9, 'watchdog_sha256': 'watchdog', 'clock_source_sha256': 'clock'}
    receipt = {'phase': 'fit', 'plan': str(s.plan_path('study')), 'plan_sha256': 'sha',
               'output': spec['output'], 'supervision': spec['supervision'], 'launch': launch}
    return plan, receipt


def test_exact_prospective_counts_and_paths():
    assert s.CONFIG['cohorts'] == [{'seed_namespace': 437260924 + i, 'fit_seed': 437261001 + i}
                                   for i in range(5)]
    assert s.CONFIG['prefix_updates'] == 1024 and s.CONFIG['joint_updates'] == 3072
    assert s.SMOKE['cohorts'] == [{'seed_namespace': 948001, 'fit_seed': 948101}]
    assert s.SMOKE['prefix_updates'] == 2 and s.SMOKE['joint_updates'] == 3
    assert [row['cap_seconds'] for row in s.phase_specs().values()] == [600, 1800, 600]
    assert len(s.OWN) == len(set(s.OWN)) == 9 and len(s.TESTS) == 4
    assert s.commands()[1][-4:] == s.TESTS
    assert '--noconftest' in s.commands()[1]
    assert s.RSS_LIMIT == 4 * 1024**3 and s.OUTPUT_LIMIT == 1024**3
    with pytest.raises(ValueError):
        s.plan_path('unregistered')


def test_original_launch_accepted():
    plan, receipt = launch_fixture()
    s.launch_binding(plan, 'sha', receipt)


def test_settings_survives_json_roundtrip_with_immutable_rule(monkeypatch):
    rule = MappingProxyType({'controls': ('a', 'b'), 'horizons': (4, 8), 'fraction': .1})
    monkeypatch.setitem(sys.modules, 'audit_finite_action_range_learning', SimpleNamespace(RULE=rule))
    monkeypatch.setitem(sys.modules, 'qualify_finite_action_range_exposure', SimpleNamespace(EXPOSURE={'n': 3}))
    monkeypatch.setattr(s, 'descriptor', lambda path: {'sha256': 'fake', 'bytes': 1})
    parent = {'sources': {f'fabricated-source-{i}.py': {} for i in range(152)}}
    result = s.settings(parent)
    assert result == json.loads(json.dumps(result))
    assert result['rule']['controls'] == ['a', 'b'] and result['rule']['horizons'] == [4, 8]
    assert len(result['sources']) == 161


@pytest.mark.parametrize('field,value', [
    ('command', ['fabricated']), ('cwd', '/wrong'), ('cap_seconds', 1801),
    ('deadline_ns', 1800 * 10**9), ('pgid', 24), ('parent_pid', 23),
    ('clock_backend', 'time.monotonic'), ('watchdog_sha256', 'wrong'),
    ('clock_source_sha256', 'wrong'), ('version', 'wrong')])
def test_launch_tampering_rejected(field, value):
    plan, receipt = launch_fixture()
    receipt['launch'][field] = value
    with pytest.raises(ValueError):
        s.launch_binding(plan, 'sha', receipt)


@pytest.mark.parametrize('field,value', [('plan', '/wrong'), ('plan_sha256', 'wrong'),
                                       ('output', '/wrong'), ('supervision', '/wrong')])
def test_receipt_binding_tampering_rejected(field, value):
    plan, receipt = launch_fixture()
    receipt[field] = value
    with pytest.raises(ValueError):
        s.launch_binding(plan, 'sha', receipt)


def test_snapshot_source_and_registration_are_all_authenticated(tmp_path, monkeypatch):
    folder = tmp_path / 'study'
    folder.mkdir()
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    monkeypatch.setattr(s, 'FOLDER', folder)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'source.py'
    source.write_text('fabricated = True\n')
    fixed = {'version': s.VERSION, 'sources': {'source.py': s.descriptor(source)}, 'config': {'n': 3}}
    monkeypatch.setattr(s, 'prerequisite', dict)
    monkeypatch.setattr(s, 'settings', lambda parent: copy.deepcopy(fixed))
    plan = {**fixed, 'mode': 'engineering'}
    s.publish(s.plan_path('engineering'), plan)
    pin = s.descriptor(s.plan_path('engineering'))
    snapshot = folder / 'source-snapshot-engineering'
    snapshot.mkdir()
    (snapshot / 'source.py').write_bytes(source.read_bytes())
    s.publish(snapshot / 'manifest.json', {'registration': pin, 'sources': fixed['sources']})
    assert s.validate(plan, pin['sha256']) == s.plan_path('engineering')
    (snapshot / 'source.py').write_text('tampered\n')
    with pytest.raises(ValueError, match='snapshot'):
        s.validate(plan, pin['sha256'])
    with pytest.raises(ValueError, match='registration'):
        s.validate(plan, hashlib.sha256(b'wrong').hexdigest())
    assert json.loads(s.plan_path('engineering').read_text()) == plan
