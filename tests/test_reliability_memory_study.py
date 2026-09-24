"""Fabricated metric, schedule and admission checks; no study execution."""
from __future__ import annotations

import copy
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
s = importlib.import_module('reliability_memory_study')


def metric_fixture():
    """256 supported episodes with unequal numbers of alive late boundaries.

    The first128 contribute three late boundaries; the other128 contribute one
    and then found. A boundary-weighted mean is deliberately not the answer.
    """
    n, length = 256, 11
    labels = np.zeros((n, length), np.int64)
    labels[128:, 9:] = 4
    lengths = np.full(n, length, np.int64); lengths[128:] = 10
    probabilities = np.tile(np.array([.25, .25, .25, .25, 0.]), (n, length, 1))
    probabilities[128:, 9] = [.125, .125, .125, .125, .5]
    probabilities[128:, 10] = [0, 0, 0, 0, 1]
    fork_costs = np.zeros((n, length, 8, 4), np.float64)
    post_costs = np.zeros((n, length, 4), np.float64)

    def centered_gap(value):
        return np.array([-.75, .25, .25, .25]) * value / 16

    for boundary, h4, h8, immediate in ((8, 1, 2, 2), (9, 3, 6, 4), (10, 5, 10, 6)):
        fork_costs[:128, boundary, 3] = centered_gap(h4)
        fork_costs[:128, boundary, 7] = centered_gap(h8)
        post_costs[:128, boundary] = centered_gap(immediate)
    fork_costs[128:, 8, 3] = centered_gap(9)
    fork_costs[128:, 8, 7] = centered_gap(12)
    post_costs[128:, 8] = centered_gap(10)
    predicted_forks = np.zeros((n, length, 2, 4), np.float64); predicted_forks[..., 1] = -1
    predicted_post = np.zeros_like(post_costs); predicted_post[..., 1] = -1
    data = {'public': {'observations': labels, 'lengths': lengths},
            'targets': {'fork_costs': fork_costs, 'post_costs': post_costs},
            'audit': {'switch_indices': np.full(n, 8, np.int64)}}
    predictions = {'probabilities': probabilities, 'fork_costs': predicted_forks, 'post_costs': predicted_post}
    return data, predictions


def test_metric_population_episode_denominators_and_first_found():
    data, predictions = metric_fixture()
    row = s.metrics(data, predictions)
    assert row['primary_regret'] == pytest.approx(7.5 / 16)
    assert row['h4_regret'] == pytest.approx(6 / 16)
    assert row['h8_regret'] == pytest.approx(9 / 16)
    assert row['post_regret'] == pytest.approx(7 / 16)
    assert row['event_log_loss'] == pytest.approx((20 * math.log(4) + math.log(2)) / 21)
    assert row['event_count'] == 128 * 21
    assert row['late_episode_support'] == 256 and row['late_unsupported'] == 0
    assert row['late_boundaries'] == 128 * 4
    assert row['switch_delays'] == [
        {'delay': 1, 'episodes': 256, 'regret': 6 / 16},
        {'delay': 2, 'episodes': 128, 'regret': 4.5 / 16},
        {'delay': 4, 'episodes': 0, 'regret': None},
        {'delay': 8, 'episodes': 0, 'regret': None}]


def test_first_index_ties_and_post_found_costs_do_not_add_decisions():
    data, predictions = metric_fixture()
    predictions['fork_costs'].fill(0); predictions['post_costs'].fill(0)
    first = s.metrics(data, predictions)
    assert first['primary_regret'] == first['post_regret'] == 0
    # Predictions remain finite but intentionally awful after found. Those
    # boundaries cannot affect a decision average or its support.
    predictions['fork_costs'][128:, 9:, :, 0] = 999
    predictions['post_costs'][128:, 9:, 0] = 999
    assert s.metrics(data, predictions) == first


def test_event_nll_excludes_padding_but_includes_first_found():
    data, predictions = metric_fixture()
    original = s.metrics(data, predictions)
    predictions['probabilities'][128:, 10] = [.125, .125, .125, .125, .5]
    assert s.metrics(data, predictions) == original
    predictions['probabilities'][128:, 9] = [.1875, .1875, .1875, .1875, .25]
    changed = s.metrics(data, predictions)
    assert changed['event_log_loss'] - original['event_log_loss'] == pytest.approx(math.log(2) / 21)
    assert changed['event_count'] == original['event_count']


@pytest.mark.parametrize('bad', ('support', 'mass', 'negative', 'zero_evidence', 'nonfinite_probability', 'nonfinite_cost', 'cost_shape'))
def test_metrics_fail_on_unsupported_or_invalid_predictions(bad):
    data, predictions = metric_fixture()
    if bad == 'support':
        data['public']['lengths'][0] = 8
    elif bad == 'mass':
        predictions['probabilities'][0, 0, 0] += .01
    elif bad == 'negative':
        predictions['probabilities'][0, 0] = [-.1, .35, .35, .4, 0]
    elif bad == 'zero_evidence':
        predictions['probabilities'][0, 0] = [0, .25, .25, .5, 0]
    elif bad == 'nonfinite_probability':
        predictions['probabilities'][0, 0, 0] = np.nan
    elif bad == 'nonfinite_cost':
        predictions['post_costs'][0, 0, 0] = np.nan
    else:
        predictions['fork_costs'] = predictions['fork_costs'][..., :1, :]
    with pytest.raises(ValueError):
        s.metrics(data, predictions)


def test_batches_are_paired_complete_epoch_permutations_and_prefix_stable():
    before = np.random.get_state()
    batches = s.batch_order(8, 6, 2, 949501)
    larger = s.batch_order(8, 10, 2, 949501)
    after = np.random.get_state()
    assert batches.shape == (6, 2) and batches.dtype == np.int64
    np.testing.assert_array_equal(batches, larger[:6])
    np.testing.assert_array_equal(batches, s.batch_order(8, 6, 2, 949501))
    assert sorted(batches[:4].ravel()) == list(range(8))
    assert len(set(batches[4:].ravel())) == 4
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    with pytest.raises(ValueError):
        s.batch_order(7, 6, 2, 949501)


def test_metrics_do_not_mutate_inputs():
    data, predictions = metric_fixture()
    original = copy.deepcopy((data, predictions))
    s.metrics(data, predictions)
    for group in data:
        for key in data[group]:
            np.testing.assert_array_equal(data[group][key], original[0][group][key])
    for key in predictions:
        np.testing.assert_array_equal(predictions[key], original[1][key])


def flat(data):
    return {group + '/' + key: value for group, arrays in data.items() for key, value in arrays.items()}


def test_independent_auditor_matches_hand_metric_fixture():
    auditor = importlib.import_module('audit_reliability_memory')
    data, predictions = metric_fixture()
    actual = auditor.metrics(flat(data), predictions)
    expected = s.metrics(data, predictions)
    assert actual['primary_regret'] == pytest.approx(7.5 / 16)
    assert actual['event_log_loss'] == pytest.approx((20 * math.log(4) + math.log(2)) / 21)
    auditor.compare(actual, expected)


def gate_rows():
    rows = []
    for cohort in range(5):
        for arm in s.ARMS:
            for split in s.STRATA:
                rows.append({'cohort': cohort, 'arm': arm, 'split': split,
                    'primary_regret': .8 if arm == 'recurrent_bank' and split in ('shift', 'switch') else 1.,
                    'event_log_loss': 1., 'inference_seconds': 1.})
    return rows


def test_fixed_rule_all_conditions_and_stress_scope():
    auditor = importlib.import_module('audit_reliability_memory')
    rows = gate_rows()
    result = auditor.continuation(rows)
    assert result['passed'] is True and result['status'] == 'PASS'
    assert len(result['conditions']) == 13 and len(result['comparisons']) == 4
    # STRESS decisions remain reported but cannot turn the prospective gate.
    for row in rows:
        if row['split'] == 'stress' and row['arm'] == 'recurrent_bank':
            row['primary_regret'] = 999.
    assert auditor.continuation(rows)['conditions'] == result['conditions']
    # Inference timing includes STRESS calls, as the declared rule specifies.
    for row in rows:
        if row['split'] == 'stress' and row['arm'] == 'recurrent_bank':
            row['inference_seconds'] = 6.
    changed = auditor.continuation(rows)
    assert not changed['conditions']['all_strata/inference_time2x']


def test_paired_win_condition_is_strict_and_cannot_be_rescued_by_mean():
    auditor = importlib.import_module('audit_reliability_memory')
    rows = gate_rows()
    for row in rows:
        if row['arm'] == 'recurrent_bank' and row['split'] == 'shift':
            row['primary_regret'] = .01 if row['cohort'] < 3 else 1.
    result = auditor.continuation(rows)
    assert result['conditions']['shift/markov_bank/mean_gain10pct']
    assert not result['conditions']['shift/markov_bank/paired_wins4of5']
    assert not result['passed']


def test_base_harm_against_unchanged_alone_blocks_continuation():
    auditor = importlib.import_module('audit_reliability_memory')
    rows = gate_rows()
    for row in rows:
        if row['split'] == 'base' and row['arm'] == 'unchanged':
            row['primary_regret'] = .5
    result = auditor.continuation(rows)
    assert result['status'] == 'FAIL' and result['passed'] is False
    assert {name for name, passed in result['conditions'].items() if not passed} == {
        'base/unchanged/noninferiority'}
    assert sum(result['conditions'].values()) == 12


def test_base_absolute_tolerance_and_nll_boundary():
    auditor = importlib.import_module('audit_reliability_memory')
    rows = gate_rows()
    for row in rows:
        if row['split'] == 'base':
            row['primary_regret'] = 1e-6 if row['arm'] == 'recurrent_bank' else 0.
            row['event_log_loss'] = 1.01 if row['arm'] == 'recurrent_bank' else 1.
    result = auditor.continuation(rows)
    assert result['conditions']['base/markov_bank/noninferiority']
    assert result['conditions']['base/reset_bank/noninferiority']
    assert result['conditions']['base/unchanged/noninferiority']
    assert result['conditions']['base/event_nll_plus001']
    for row in rows:
        if row['split'] == 'base' and row['arm'] == 'recurrent_bank':
            row['primary_regret'] += 1e-9
            row['event_log_loss'] += 1e-9
    result = auditor.continuation(rows)
    assert not result['conditions']['base/markov_bank/noninferiority']
    assert not result['conditions']['base/reset_bank/noninferiority']
    assert not result['conditions']['base/unchanged/noninferiority']
    assert not result['conditions']['base/event_nll_plus001']


@pytest.mark.parametrize('bad', ('missing', 'duplicate', 'cohort', 'arm'))
def test_gate_requires_entire_fixed_roster(bad):
    auditor = importlib.import_module('audit_reliability_memory')
    rows = gate_rows()
    if bad == 'missing':
        rows.pop()
    elif bad == 'duplicate':
        rows[-1] = dict(rows[0])
    elif bad == 'cohort':
        rows[-1]['cohort'] = 5
    else:
        rows[-1]['arm'] = 'selected_winner'
    with pytest.raises(ValueError):
        auditor.continuation(rows)


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
    plan = {'version': 'reliability-memory-v1', 'config': copy.deepcopy(s.CONFIG), 'caps': copy.deepcopy(s.CAPS),
            'runtime': s.runtime(), 'output': str(folder), 'sources': {'witness.py': s.desc(root / 'witness.py')},
            'parent': copy.deepcopy(parent)}
    if bad == 'configuration':
        plan['config']['updates'] += 1
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


def test_tiny_engineering_world_model_and_saved_audit_interfaces():
    """One four-episode engineering integration, with no scientific namespace."""
    auditor = importlib.import_module('audit_reliability_memory')
    world = importlib.import_module('openjev.research.finite_reliability_world')
    filter_module = importlib.import_module('openjev.research.finite_reliability_filter')
    torch = importlib.import_module('torch')
    generated = world.generate(949001, 'train', 4, steps=24)
    data = flat({key: generated[key] for key in ('public', 'targets', 'audit')})
    reference = auditor.validate_data(data, generated['counts'], 949001, 'train', episodes=4, steps=24)
    assert set(reference) == {'probabilities', 'post_states', 'post_costs', 'fork_costs'}
    # A simple fabricated backbone is deliberately not the true environment.
    # Audit prediction consistency must use this actual learned-field stand-in.
    fields = {'transition': np.tile(np.eye(8), (4, 1, 1)), 'emission': np.full((4, 8), .25),
              'hazard': np.full((4, 8), .125), 'costs': np.full((4, 8), .25)}
    for state in range(8):
        fields['costs'][state % 4, state] = -.75
    tensor_fields = {name: torch.from_numpy(value.copy()) for name, value in fields.items()}
    actions = torch.from_numpy(data['public/actions'])
    observations = torch.from_numpy(data['public/observations'])
    fork_actions = torch.from_numpy(data['targets/fork_actions'])
    for arm in s.ARMS:
        model = filter_module.ReliabilityFilter(tensor_fields, arm, 949101)
        with torch.no_grad():
            output = model(actions, observations)
            forks = model.blind_forks(output['post_states'], fork_actions)
        predictions = {name: value.numpy().copy() for name, value in output.items()}
        predictions['fork_costs'] = forks['costs'][:, :, (3, 7), :].numpy().copy()
        predictions['fork_survival'] = forks['survival'][:, :, (3, 7)].numpy().copy()
        auditor.validate_predictions(data, predictions, fields, arm)
        predictions['fork_costs'][0, 0, 0, 0] += .01
        with pytest.raises(ValueError):
            auditor.validate_predictions(data, predictions, fields, arm)
    corrupted = {name: value.copy() for name, value in data.items()}
    corrupted['targets/post_costs'][0, 0, 0] += .01
    with pytest.raises(ValueError):
        auditor.validate_data(corrupted, generated['counts'], 949001, 'train', episodes=4, steps=24)
