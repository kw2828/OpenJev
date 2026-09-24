"""Fabricated admission checks, without scientific or child-process execution."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/finite_update_learning_worker.py'
SPEC = importlib.util.spec_from_file_location('balanced_admission_fixture', PATH)
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
    receipt, terminal, terminal_path = closed_fixture(tmp_path)
    folder = tmp_path / 'qualification'
    folder.mkdir()
    (folder / 'checks.log').write_text('PASS\n')
    spec = {'output': str(folder), 'supervision': receipt['supervision'], 'cap_seconds': 30}
    sources = {'scripts/supervise_dialogue_observation_v2.py': {'sha256': 'a' * 64},
               'src/openjev/research/suspend_clock.py': {'sha256': 'b' * 64}}
    plan = {'version': 'finite-update-learning-v1', 'mode': 'study',
            'root': str(tmp_path), 'runtime': {'executable': '/fixture/python'}, 'sources': sources,
            'config': {'epochs': 480}, 'phases': {'qualify': spec},
            'tests': ['fixture_test.py'], 'lint_sources': ['fixture.py'],
            'rss_limit_bytes': 100, 'output_limit_bytes': 100,
            'kernel_qualification': {'folder': 'fixture', 'files': {}},
            'parent_evidence': {},
            'exposure_probe': {'config': {'fixture': True}, 'target_counts': {'prefix_updates': 3, 'joint_updates': 4},
                               'maximum_projected_seconds': 90.0}}
    probe = folder / 'exposure'; probe.mkdir()
    (probe / 'audit.json').write_text(json.dumps({'version': 'finite-update-learning-audit-v1',
        'profile': 'engineering-942201', 'agreement': True, 'exact_oracle_agreement': True}))
    (probe / 'receipt.json').write_text(json.dumps({'status': 'PASS', **plan['exposure_probe'],
        'audit': WORKER.descriptor(probe / 'audit.json'),
        'projections': [{'arm': arm, 'seed': 942301, 'projected_seconds': 1.0}
                       for arm in ('original_free', 'matched_free', 'rounded')]}))
    engineering = tmp_path / 'engineering.json'
    engineering.write_text(json.dumps({**plan, 'mode': 'engineering'}))
    receipt.update({'phase': 'qualify', 'status': 'PASS', 'output': str(folder),
                    'plan': str(engineering), 'plan_sha256': WORKER.descriptor(engineering)['sha256'],
                    'sources_before': sources, 'sources_after': sources, 'files': WORKER.files(folder),
                    'commands': [{'command': command, 'returncode': 0, 'seconds': .1}
                                 for command in WORKER.qualification_commands(plan)]})
    receipt['launch'].update(cwd=str(tmp_path), command=[
        '/fixture/python', str(tmp_path / 'scripts/finite_update_learning_worker.py'),
        '--plan', str(engineering), '--plan-sha256', receipt['plan_sha256'],
        '--phase', 'qualify', '--supervision', spec['supervision'], '--output', spec['output']])
    terminal.update(receipt['launch'])
    terminal_path.write_text(json.dumps(terminal))
    Path(receipt['supervision']).write_text(json.dumps(receipt['launch']))
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


@pytest.mark.parametrize('change', ['missing', 'extra', 'wrong_command', 'failed', 'nan_time', 'negative_time'])
def test_qualification_rejects_forged_success_command_roster(tmp_path, change):
    plan, path = qualification_fixture(tmp_path)
    receipt = json.loads(path.read_text())
    if change == 'missing':
        receipt['commands'].pop()
    elif change == 'extra':
        receipt['commands'].append(receipt['commands'][0])
    elif change == 'wrong_command':
        receipt['commands'][1]['command'][-1] = 'other_test.py'
    elif change == 'failed':
        receipt['commands'][1]['returncode'] = 1
    elif change == 'nan_time':
        receipt['commands'][1]['seconds'] = float('nan')
    else:
        receipt['commands'][1]['seconds'] = -.1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        WORKER.admit_qualification(plan, path)


@pytest.mark.parametrize('field,value', [('command', ['python', 'foreign.py']),
    ('cwd', '/different'), ('cap_seconds', 31), ('watchdog_sha256', 'c' * 64),
    ('clock_source_sha256', 'd' * 64)])
def test_qualification_rejects_self_consistent_foreign_launch(tmp_path, field, value):
    plan, path = qualification_fixture(tmp_path)
    receipt = json.loads(path.read_text())
    receipt['launch'][field] = value
    Path(receipt['supervision']).write_text(json.dumps(receipt['launch']))
    terminal_path = Path(receipt['supervision'].replace('.launch.json', '.terminal.json'))
    terminal = json.loads(terminal_path.read_text())
    terminal[field] = value
    terminal_path.write_text(json.dumps(terminal))
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        WORKER.admit_qualification(plan, path)


@pytest.mark.parametrize('change', ['failed', 'counts', 'configuration', 'missing_arm', 'duplicate_arm',
                                  'too_slow', 'negative', 'nonfinite', 'audit_disagrees', 'audit_hash'])
def test_qualification_requires_original_fixed_exposure_admission(tmp_path, change):
    plan, path = qualification_fixture(tmp_path)
    probe_path = Path(plan['phases']['qualify']['output']) / 'exposure/receipt.json'
    probe = json.loads(probe_path.read_text())
    if change == 'failed':
        probe['status'] = 'FAILED'
    elif change == 'counts':
        probe['target_counts']['prefix_updates'] += 1
    elif change == 'configuration':
        probe['config']['fixture'] = False
    elif change == 'missing_arm':
        probe['projections'].pop()
    elif change == 'duplicate_arm':
        probe['projections'][0] = probe['projections'][1]
    elif change == 'audit_hash':
        probe['audit']['sha256'] = 'f' * 64
    elif change == 'audit_disagrees':
        audit_path = probe_path.parent / 'audit.json'
        audit = json.loads(audit_path.read_text())
        audit['agreement'] = False
        audit_path.write_text(json.dumps(audit))
        probe['audit'] = WORKER.descriptor(audit_path)
    else:
        probe['projections'][0]['projected_seconds'] = {'too_slow': 90., 'negative': -1.,
                                                       'nonfinite': float('nan')}[change]
    probe_path.write_text(json.dumps(probe))
    # Rebind the enclosing payload inventory so the exposure admission itself
    # must reject this fabricated claim, rather than only its changed bytes.
    receipt = json.loads(path.read_text())
    receipt['files'] = WORKER.files(Path(plan['phases']['qualify']['output']))
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        WORKER.admit_qualification(plan, path)
