"""Hand-sized metric/classification and fake original-process admission tests."""
from __future__ import annotations

import copy
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
s = importlib.import_module('schedule_headroom_study')


def centered(gap):
    return np.array([3*gap/4,-gap/4,-gap/4,-gap/4],np.float64)


def metric_fixture():
    """Four equal-weight episodes: 25,0,2,0 alive late boundaries."""
    observations = np.zeros((4,33),np.int64)
    lengths = np.array([33,9,11,2],np.int64)
    for i,first in ((1,8),(2,10),(3,1)):
        observations[i,first:] = 4
    probabilities = np.tile([.125,.125,.125,.125,.5],(4,33,1))
    probabilities[:,0] = [.25,.25,.25,.25,0]
    for i,length in enumerate(lengths):
        probabilities[i,length:] = [0,0,0,0,1]
    fork = np.zeros((4,33,8,4),np.float64)
    post = np.zeros((4,33,4),np.float64)
    fork[0,8:,3] = centered(2); fork[0,8:,7] = centered(6)
    fork[2,8:10,3] = centered(4); fork[2,8:10,7] = centered(8)
    post[0,8:] = centered(1); post[2,8:10] = centered(3)
    data = {'public':{'observations':observations,'lengths':lengths},
        'targets':{'fork_costs':fork,'post_costs':post},'audit':{'family':np.arange(4,dtype=np.int64)}}
    prediction = {'probabilities':probabilities,'fork_costs':np.zeros((4,33,2,4),np.float64),
                  'post_costs':np.zeros((4,33,4),np.float64)}
    return data,prediction


def test_fixed_divisor_keeps_early_found_episodes_and_all_fifty_slots():
    data,prediction = metric_fixture()
    row = s.metrics(data,prediction)
    assert row['per_episode_regret'] == pytest.approx([200/50,0,24/50,0])
    assert row['primary_regret'] == pytest.approx(224/200)
    assert row['h4_regret'] == pytest.approx(58/100)
    assert row['h8_regret'] == pytest.approx(166/100)
    assert row['post_regret'] == pytest.approx(31/100)
    assert row['event_count'] == 55 and row['found_episodes'] == 3
    assert row['event_nll'] == pytest.approx((4*math.log(4)+48*math.log(8)+3*math.log(2))/55)
    assert row['family_regret'] == [
        {'family':f,'episodes':1,'primary_regret':value} for f,value in enumerate((4.,0.,.48,0.))]


def test_family_mean_is_conditional_but_population_is_episode_weighted():
    data,prediction = metric_fixture()
    # Duplicate the high-regret family0 episode, leaving all other families.
    indices = np.array([0,0,1,2,3])
    for group in data.values():
        for name,value in group.items():
            group[name] = value[indices].copy()
    prediction = {name:value[indices].copy() for name,value in prediction.items()}
    row = s.metrics(data,prediction)
    assert row['primary_regret'] == pytest.approx(8.48/5)
    assert row['family_regret'][0] == {'family':0,'episodes':2,'primary_regret':4.}
    assert row['primary_regret'] != pytest.approx(sum(r['primary_regret'] for r in row['family_regret'])/4)


def test_ties_use_first_action_and_terminal_padding_has_no_decision_loss():
    data,prediction = metric_fixture()
    original = s.metrics(data,prediction)
    prediction['fork_costs'][...,0] = 1
    prediction['post_costs'][...,0] = 1
    tied = s.metrics(data,prediction)
    assert tied['primary_regret'] == tied['post_regret'] == 0
    data,prediction = metric_fixture()
    for i,length in enumerate(data['public']['lengths']):
        prediction['fork_costs'][i,length-1 if i else 33:,:,0] = -999
        prediction['post_costs'][i,length-1 if i else 33:,0] = -999
    assert s.metrics(data,prediction) == original


def test_nll_counts_first_found_but_never_absorbing_suffix():
    data,prediction = metric_fixture()
    original = s.metrics(data,prediction)['event_nll']
    prediction['probabilities'][1,9:] = [.125,.125,.125,.125,.5]
    assert s.metrics(data,prediction)['event_nll'] == original
    prediction['probabilities'][1,8] = [.1875,.1875,.1875,.1875,.25]
    assert s.metrics(data,prediction)['event_nll'] == pytest.approx(original+math.log(2)/55)
    prediction['probabilities'][1,8,4] = 0
    with pytest.raises(ValueError,match='actual-event'):
        s.metrics(data,prediction)


def rows_fixture():
    return [{'cohort':c,'arm':arm,'primary_regret':.5 if arm in ('learned_exact','true_exact') else 1.,
             'family_regret':[{'family':f,'episodes':1,'primary_regret':.5} for f in range(4)]}
            for c in range(5) for arm in s.ARMS]


@pytest.mark.parametrize('expected',('LEARNED_MODEL_HEADROOM','MODEL_MISMATCH_HEADROOM',
    'TRUE_MODEL_ADVANTAGE_WITHOUT_SCHEDULE_MEMORY','NO_REGISTERED_HEADROOM'))
def test_all_four_fixed_classification_paths(expected):
    rows = rows_fixture()
    if expected != 'LEARNED_MODEL_HEADROOM':
        for row in rows:
            if row['arm'] == 'learned_exact':
                row['primary_regret'] = 1.
    if expected == 'TRUE_MODEL_ADVANTAGE_WITHOUT_SCHEDULE_MEMORY':
        for row in rows:
            if row['arm'] == 'true_static2':
                row['primary_regret'] = .5
    elif expected == 'NO_REGISTERED_HEADROOM':
        for row in rows:
            if row['arm'] == 'true_exact':
                row['primary_regret'] = 1.
    result = s.classify(rows)
    assert result['classification'] == expected
    assert len(result['means']) == 10 and len(result['true_history_conditions']) == 2
    for candidate in result['candidates'].values():
        assert len(candidate['conditions']) == 15 and len(candidate['comparisons']) == 7
        assert {r['control'] for r in candidate['comparisons']} == set(s.CONTROLS)


@pytest.mark.parametrize('control',s.CONTROLS)
def test_every_one_of_seven_controls_is_required(control):
    rows = rows_fixture()
    for row in rows:
        if row['arm'] == control:
            row['primary_regret'] = .5
    candidate = s.classify(rows)['candidates']['learned_exact']
    assert candidate['passed'] is False
    assert not candidate['conditions'][control+'/mean_gain10pct']
    assert not candidate['conditions'][control+'/paired_wins4of5']


def test_base_unchanged_preservation_alone_blocks_candidate():
    rows = rows_fixture()
    for row in rows:
        if row['arm'] == 'learned_exact':
            row['family_regret'][0]['primary_regret'] = 1.
    candidate = s.classify(rows)['candidates']['learned_exact']
    assert candidate['passed'] is False
    assert {k for k,v in candidate['conditions'].items() if not v} == {'base/unchanged/noninferiority'}


def test_base_absolute_tolerance_includes_boundary():
    rows = rows_fixture()
    for row in rows:
        if row['arm'] == 'unchanged':
            row['family_regret'][0]['primary_regret'] = 0.
        elif row['arm'] == 'learned_exact':
            row['family_regret'][0]['primary_regret'] = 1e-6
    assert s.classify(rows)['candidates']['learned_exact']['passed']
    for row in rows:
        if row['arm'] == 'learned_exact':
            row['family_regret'][0]['primary_regret'] += 1e-9
    assert not s.classify(rows)['candidates']['learned_exact']['passed']


@pytest.mark.parametrize('wins',(3,4))
def test_three_vs_four_strict_wins_and_history_gate(wins):
    rows = rows_fixture()
    for row in rows:
        if row['arm'] in ('learned_exact','true_exact'):
            row['primary_regret'] = .1 if row['cohort'] < wins else 1.
    result = s.classify(rows)
    for candidate in result['candidates'].values():
        assert all(candidate['conditions'][c+'/mean_gain10pct'] for c in s.CONTROLS)
        assert candidate['passed'] == (wins == 4)
        assert all(r['paired_wins'] == wins for r in candidate['comparisons'])
    assert result['true_history_conditions']['mean_gain10pct']
    assert result['true_history_conditions']['paired_wins4of5'] == (wins == 4)


def test_zero_control_regret_does_not_create_success():
    rows = rows_fixture()
    for row in rows:
        row['primary_regret'] = 0.
    result = s.classify(rows)
    assert result['classification'] == 'NO_REGISTERED_HEADROOM'
    assert not any(result['true_history_conditions'].values())
    assert all(not candidate['passed'] for candidate in result['candidates'].values())


@pytest.mark.parametrize('bad',('missing','duplicate','arm','cohort'))
def test_incomplete_or_duplicate_arm_grid_rejected(bad):
    rows = rows_fixture()
    if bad == 'missing': rows.pop()
    elif bad == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    elif bad == 'arm': rows[-1]['arm'] = 'selected_control'
    else: rows[-1]['cohort'] = 5
    with pytest.raises(ValueError):
        s.classify(rows)


def closed_fixture(tmp_path):
    phase, sha = 'qualify', 'a' * 64
    folder = tmp_path / 'study'; folder.mkdir()
    output = folder / phase; output.mkdir()
    (output / 'witness.txt').write_text('fabricated only\n')
    launch = {'version': 'dialogue-observation-supervision-v2',
        'command': s.phase_command(folder, phase, sha), 'cwd': str(s.ROOT),
        'cap_seconds': s.CAPS[phase], 'clock_backend': 'mach_continuous_time',
        'clock_source_sha256': s.desc(s.suspend_clock.__file__)['sha256'],
        'watchdog_sha256': s.desc(s.ROOT / 'scripts/supervise_dialogue_observation_v2.py')['sha256'],
        'started_ns': 1_000_000_000, 'deadline_ns': 1_000_000_000 + s.CAPS[phase] * 10**9,
        'pid': 987, 'pgid': 987, 'parent_pid': 123}
    terminal = {**launch, 'status': 'completed', 'returncode': 0, 'timed_out': False,
        'error': None, 'clock_error': None, 'group_absent': True,
        'cleanup': {'reaped': True, 'errors': [], 'signals': []},
        'finished_ns': 3_000_000_000, 'elapsed_ns': 2_000_000_000, 'wall_seconds': 2.}
    receipt = {'status': 'PASS', 'phase': phase, 'plan_sha256': sha,
               'files': s.inventory(output), 'command': launch['command']}
    return folder, phase, sha, launch, terminal, receipt


def publish_closed_fixture(fixture):
    folder, phase, _sha, launch, terminal, receipt = fixture
    s.write(folder / f'{phase}-native-01.launch.json', launch)
    s.write(folder / f'{phase}-native-01.terminal.json', terminal)
    s.write(folder / f'{phase}.receipt.json', receipt)


def test_original_launch_closure_and_exact_cap_fixture(tmp_path):
    fixture = closed_fixture(tmp_path)
    publish_closed_fixture(fixture)
    folder, phase, sha, launch, terminal, receipt = fixture
    assert s.bind_launch(folder, phase, sha) == launch
    assert s.closed(folder, phase, sha) == (receipt, terminal)


@pytest.mark.parametrize('bad', ('argv', 'cwd', 'source', 'cap', 'deadline', 'pgid',
                                 'timeout', 'cleanup', 'terminal_join', 'elapsed', 'late', 'inventory'))
def test_original_admission_rejects_fabricated_corruption(tmp_path, bad):
    fixture = closed_fixture(tmp_path)
    folder, phase, sha, launch, terminal, receipt = fixture
    if bad == 'argv':
        launch['command'] = [*launch['command'], '--retry']
    elif bad == 'cwd':
        launch['cwd'] = str(tmp_path)
    elif bad == 'source':
        launch['clock_source_sha256'] = '0' * 64
    elif bad == 'cap':
        launch['cap_seconds'] += 1
    elif bad == 'deadline':
        launch['deadline_ns'] += 1
    elif bad == 'pgid':
        launch['pgid'] += 1
    elif bad == 'timeout':
        terminal['timed_out'] = True
    elif bad == 'cleanup':
        terminal['cleanup']['signals'] = ['SIGTERM']
    elif bad == 'terminal_join':
        terminal['pid'] += 1
    elif bad == 'elapsed':
        terminal['elapsed_ns'] += 1
    elif bad == 'late':
        terminal['finished_ns'] = terminal['deadline_ns']
        terminal['elapsed_ns'] = terminal['finished_ns'] - terminal['started_ns']
        terminal['wall_seconds'] = terminal['elapsed_ns'] / 1e9
    else:
        receipt['files'] = {}
    publish_closed_fixture(fixture)
    with pytest.raises(ValueError):
        s.closed(folder, phase, sha)


def test_live_launch_uses_actual_worker_identity_and_clock(tmp_path, monkeypatch):
    fixture = closed_fixture(tmp_path); publish_closed_fixture(fixture)
    folder, phase, sha, launch, _terminal, _receipt = fixture

    class Clock:
        backend = launch['clock_backend']
        now = launch['started_ns'] + 1

        def now_ns(self):
            return self.now

    monkeypatch.setattr(s.suspend_clock, 'SuspendClock', Clock)
    monkeypatch.setattr(s.os, 'getpid', lambda: launch['pid'])
    monkeypatch.setattr(s.os, 'getpgrp', lambda: launch['pgid'])
    monkeypatch.setattr(s.os, 'getppid', lambda: launch['parent_pid'])
    assert s.bind_launch(folder, phase, sha, live=True) == launch
    Clock.now = launch['deadline_ns']
    with pytest.raises(ValueError):
        s.bind_launch(folder, phase, sha, live=True)


@pytest.mark.parametrize('bad', (None, 'original', 'snapshot', 'configuration', 'runtime', 'parent'))
def test_registration_checks_sources_snapshot_and_frozen_settings(tmp_path, monkeypatch, bad):
    root = tmp_path / 'repo'; root.mkdir()
    folder = root / 'study'; folder.mkdir()
    snapshot = folder / 'sources'; snapshot.mkdir()
    (root / 'witness.py').write_text('VALUE = 1\n')
    (snapshot / 'witness.py').write_text('VALUE = 1\n')
    parent = {'receipt': {'sha256': 'b' * 64, 'bytes': 1}}
    monkeypatch.setattr(s, 'ROOT', root)
    monkeypatch.setattr(s, 'parent_check', lambda: parent)
    monkeypatch.chdir(root)
    plan = {'version': 'schedule-headroom-v1', 'config': copy.deepcopy(s.CONFIG), 'caps': copy.deepcopy(s.CAPS),
            'runtime': s.runtime(), 'output': str(folder), 'sources': {'witness.py': s.desc(root / 'witness.py')},
            'parent': copy.deepcopy(parent), 'arms': list(s.ARMS)}
    if bad == 'configuration':
        plan['config']['episodes'] += 1
    elif bad == 'runtime':
        plan['runtime']['torch'] = 'changed'
    s.write(folder / 'registration.json', plan)
    sha = s.desc(folder / 'registration.json')['sha256']
    if bad == 'original':
        (root / 'witness.py').write_text('VALUE = 2\n')
    elif bad == 'snapshot':
        (snapshot / 'witness.py').write_text('VALUE = 2\n')
    elif bad == 'parent':
        parent['receipt']['bytes'] += 1
    if bad is None:
        assert s.validate(folder, sha) == plan
        with pytest.raises(ValueError):
            s.validate(folder, '0' * 64)
    else:
        with pytest.raises(ValueError):
            s.validate(folder, sha)


@pytest.mark.parametrize('bad',('late_predecessor','clock_backend'))
def test_main_rejects_unjoined_predecessor_before_any_run_and_preserves_failure(tmp_path,monkeypatch,bad):
    folder = tmp_path/'study'; folder.mkdir()
    sha = 'a'*64
    argv = s.phase_command(folder,'run',sha)
    monkeypatch.setattr(s.sys,'argv',argv[1:])
    monkeypatch.setattr(s,'validate',lambda *_:{})
    launch = {'command':argv,'started_ns':10,'clock_backend':'mach_continuous_time'}
    monkeypatch.setattr(s,'bind_launch',lambda *_,**__:launch)
    terminal = {'finished_ns':11 if bad == 'late_predecessor' else 9,
                'clock_backend':'CLOCK_BOOTTIME' if bad == 'clock_backend' else launch['clock_backend']}
    monkeypatch.setattr(s,'closed',lambda *_:({},terminal))
    def forbidden(*_):
        raise AssertionError('scientific work entered before predecessor admission')
    monkeypatch.setattr(s,'run',forbidden)
    with pytest.raises(ValueError,match='predecessor'):
        s.main()
    receipt = s.read(folder/'run.receipt.json')
    assert receipt['status'] == 'FAILED' and receipt['result'] is None
    assert 'predecessor' in receipt['error'] and receipt['files'] == {}
