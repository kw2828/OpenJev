"""Synthetic saved-record fixtures only; no scientific evidence or model reads."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('return_consistency_test_subject', ROOT / 'scripts/diagnose_otto_return_consistency.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def public(t, *, done=False):
    return {'position': [26 + t % 2, 26], 'hit': -2 if done else 1 if t == 0 else t % 4,
            'done': done, 'step': t, 'valid_actions': [0, 1, 2, 3]}


def state(t):
    return {'sha256': f'{t:064x}', 'mass': 1.}


def metadata():
    result = {}
    for split, (episode, seed, total) in d.SELECTED.items():
        for index in range(8):
            result[split, index] = {'row_index': index, 'episode_id': episode, 'stage': split,
                'regime': 'lambda3', 'seed': seed, 'initial_hit': 1, 'prefix_index': index,
                'total_steps': total, 'public': public(index), 'posterior': state(index),
                'target': (total - index) / 64}
    return result


def events():
    for split, (episode, seed, total) in d.SELECTED.items():
        yield {'kind': 'reset', 'episode_id': episode, 'stage': split, 'regime': 'lambda3',
               'seed': seed, 'initial_hit': 1, 'arm': 'teacher', 'public': public(0), 'posterior_after': state(0)}
        for step in range(1, total + 1):
            yield {'kind': 'step', 'episode_id': episode, 'step': step, 'action': step % 2,
                   'allowed_actions': [0, 1, 2, 3], 'public': public(step, done=step == total),
                   'posterior_before': state(step - 1), 'posterior_after': state(step)}


def parity():
    records = []
    meta = metadata()
    for fit, split, index in d.expected_order():
        m = meta[split, index]
        values = [2. * action + hit for action in range(4) for hit in range(4)]
        costs = [1. + sum(values[4 * a:4 * a + 4]) / 4 for a in range(4)]
        records.append({'fit_id': fit, 'split': split, 'row_index': index,
            'episode_id': m['episode_id'], 'prefix_index': index, 'posterior_sha256': state(index)['sha256'],
            'scalar_numpy': 20. / 64, 'scalar_torch': 20. / 64,
            'raw_masses': [[.25] * 4 for _ in range(4)], 'weights': [[.25] * 4 for _ in range(4)],
            'values_numpy': values, 'values_torch': list(values), 'costs_numpy': costs, 'costs_torch': list(costs),
            'action_numpy': 0, 'action_torch': 0, 'allowed_actions': [0, 1, 2, 3], 'passed': True})
    return records


def analyzed():
    m = metadata()
    return d.diagnose(parity(), m, d.teacher_join(events(), m))


def test_full144_records_physical_units_and_teacher_offset():
    rows = analyzed()
    assert len(rows) == 144 and sum(len(r['branches']) for r in rows) == 2304
    first, second = rows[:2]
    assert first['teacher_transition']['step'] == 1 and first['teacher_action'] == 1
    assert second['teacher_transition']['step'] == 2 and second['teacher_action'] == 0
    assert first['current_physical_value'] == 20 and first['teacher_realized_return'] == 79
    assert first['current_return_error'] == -59
    assert first['teacher_action_residual'] == 4.5 - 20
    assert first['greedy_action_residual'] == 2.5 - 20
    assert first['predicted_switching_advantage'] == 2
    assert first['switching_contributions_by_hit'] == [.5] * 4
    assert first['branches'][5]['physical_value'] == 3  # Never multiply a saved branch value by64.


def test_teacher_stream_stops_before_unneeded_stages():
    m = metadata()
    def guarded():
        for event in events():
            yield event
            if event['episode_id'] == d.SELECTED['valid'][0] and event.get('step') == 8:
                raise AssertionError('consumer decoded beyond final required record')
    assert len(d.teacher_join(guarded(), m)) == 16


@pytest.mark.parametrize('mutation', ['offset', 'posterior', 'eligible', 'move', 'missing', 'duplicate', 'stage'])
def test_teacher_join_rejects_invalid_stream(mutation):
    data = list(events())
    if mutation == 'offset':
        data[1]['step'] = 0
    elif mutation == 'posterior':
        data[1]['posterior_before'] = state(2)
    elif mutation == 'eligible':
        data[1]['allowed_actions'] = [0, 2, 3]
    elif mutation == 'move':
        data[1]['public']['position'] = [25, 26]
    elif mutation == 'missing':
        data = data[:3]
    elif mutation == 'duplicate':
        data.insert(2, copy.deepcopy(data[1]))
    else:
        data[0]['stage'] = 'dagger'
    with pytest.raises(ValueError):
        d.teacher_join(data, metadata())


def test_strict_tie_no_tolerance_repair_and_eligible_minimum():
    assert d.select([1e-10, 0., -100., 3.], [0, 1, 3]) == {
        'action': 1, 'minimum': 0., 'tie_ids': [1], 'chosen_minus_minimum': 0.}
    below = math.nextafter(1e-10, 0)
    chosen = d.select([below, 0., -100., 3.], [0, 1, 3])
    assert chosen['action'] == 0 and chosen['tie_ids'] == [0, 1]
    assert chosen['chosen_minus_minimum'] == below


@pytest.mark.parametrize('actions', [[], [1, 0], [0, 0], [0, True], [4], [-1], (0, 1)])
def test_bad_eligibility_rejected(actions):
    with pytest.raises(ValueError):
        d.select([0., 1., 2., 3.], actions)


def test_signed_neartie_advantage_is_preserved():
    row = parity()[0]
    costs = [5e-11, 0., 2., 3.]
    values = [cost - 1 for cost in costs for _ in range(4)]
    row.update(costs_numpy=costs, costs_torch=list(costs), values_numpy=values, values_torch=list(values))
    m = metadata()
    result = d.analyze_row(row, m['train', 0], d.teacher_join(events(), m)[d.SELECTED['train'][0], 1])
    assert result['selection']['action'] == 0
    assert result['teacher_action'] == 1 and result['predicted_switching_advantage'] == -5e-11


def test_zero_mass_biased_branch_retained_and_not_renormalized():
    row = parity()[0]
    row.update(raw_masses=[[0.] * 4 for _ in range(4)], weights=[[1e-10] * 4 for _ in range(4)],
               values_numpy=[1.] * 16, values_torch=[1.] * 16,
               costs_numpy=[1 + 4e-10] * 4, costs_torch=[1 + 4e-10] * 4)
    m = metadata()
    result = d.analyze_row(row, m['train', 0], d.teacher_join(events(), m)[d.SELECTED['train'][0], 1])
    assert result['reconstructed_costs'] == [1 + 4e-10] * 4
    assert all(b['weighted_value'] == 1e-10 and b['raw_mass'] == 0 for b in result['branches'])


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'order', 'prefix', 'sha', 'shape', 'negative', 'floor',
                                     'nan', 'inf', 'bool', 'cost', 'action', 'target', 'metadata', 'passed'])
def test_bad_saved_rows_fail_without_repair(mutation):
    data, m = parity(), metadata()
    if mutation == 'missing':
        data.pop()
    elif mutation == 'duplicate':
        data[-1] = copy.deepcopy(data[0])
    elif mutation == 'order':
        data[0], data[1] = data[1], data[0]
    elif mutation == 'prefix':
        data[0]['prefix_index'] = 1
    elif mutation == 'sha':
        data[0]['posterior_sha256'] = 'f' * 64
    elif mutation == 'shape':
        data[0]['values_numpy'].pop()
    elif mutation == 'negative':
        data[0]['raw_masses'][0][0] = -.1
    elif mutation == 'floor':
        data[0]['weights'][0][0] = .3
    elif mutation in ('nan', 'inf', 'bool'):
        data[0]['scalar_numpy'] = {'nan': float('nan'), 'inf': float('inf'), 'bool': True}[mutation]
    elif mutation == 'cost':
        data[0]['costs_numpy'][0] += .01
        data[0]['costs_torch'][0] += .01
    elif mutation == 'action':
        data[0]['action_numpy'] = data[0]['action_torch'] = 1
    elif mutation == 'target':
        m['train', 0]['target'] += 1 / 64
    elif mutation == 'metadata':
        m.pop(('train', 0))
    else:
        data[0]['passed'] = False
    teachers = d.teacher_join(events(), m)
    with pytest.raises((ValueError, KeyError)):
        d.diagnose(data, m, teachers)


def test_summary_seed_means_and_denominators_are_not_pooled_rms():
    rows = analyzed()
    for row in rows:
        row['current_return_error'] = float(d.SEEDS.index(int(row['fit_id'].split('@')[1])))
    result = d.summarize(rows)
    assert result['unique_prefixes'] == 16 and result['unique_episodes'] == 2
    assert result['fit_split']['min8@10101']['train']['teacher_disagreement_count'] == 4
    mean = result['family_means']['min8']['train']['statistics']['current_return_error']
    assert mean == {'signed_mean': 1., 'mean_absolute': 1., 'rms': 1.}
    assert mean['rms'] != math.sqrt(5 / 3)
    assert result['new_efficacy_gate'] is None and not result['original_decisions_changed']


@pytest.mark.parametrize('text', ['{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'])
def test_strict_json(text):
    with pytest.raises(ValueError):
        d.parse(text)


def test_payload_inventory_keeps_training_and_weights_without_decoding():
    assert len(d.prior_names()) == 41
    assert {'collection-transitions.jsonl', 'dagger-data.npz', 'final-dense-9103.npz'} <= d.prior_names()
    assert d.LIMITS['seconds'] == 180  # Separate from the authenticated native study's5400 seconds.


def test_bad_external_plan_stops_before_helper_import(tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    diagnostic = d.Diagnostic(SimpleNamespace(plan=plan, plan_sha256='0' * 64, output=tmp_path / 'out'))
    monkeypatch.setattr(diagnostic, 'check', lambda: None)
    monkeypatch.setattr(d, 'load', lambda *_: pytest.fail('helper imported before external pin'))
    with pytest.raises(ValueError, match='external diagnostic plan pin'):
        diagnostic.authenticate()


@pytest.mark.parametrize('failure', ['clock', 'arithmetic', 'post_receipt'])
def test_failure_preservation_and_late_completion_demoted(tmp_path, monkeypatch, failure):
    out = tmp_path / 'exclusive'
    task = d.Diagnostic(SimpleNamespace(plan=tmp_path / 'plan.json', plan_sha256='0' * 64, output=out))
    original_digest = d.digest
    monkeypatch.setattr(d, 'digest', lambda path, check=lambda: None:
                        {'sha256': d.CLOCK_PIN, 'bytes': 1} if path == d.ROOT / d.CLOCK else original_digest(path, check))
    class FakeClock:
        backend = 'injected'

        def __init__(self):
            if failure == 'clock':
                raise RuntimeError('clock unavailable')

        def now_ns(self):
            return 100
    monkeypatch.setattr(d, 'load', lambda *_: SimpleNamespace(SuspendClock=FakeClock))
    monkeypatch.setattr(task, 'authenticate', dict)
    def compute(_paths):
        if failure == 'arithmetic':
            raise ValueError('saved arithmetic failed')
        d.write(out / 'summary.json', {})
        (out / 'rows.jsonl').write_text('{}\n')
    monkeypatch.setattr(task, 'compute', compute)
    def check():
        task.finished = 100
        if failure == 'post_receipt' and (out / 'receipt.json').exists():
            raise TimeoutError('late expiry')
    monkeypatch.setattr(task, 'check', check)
    assert task.execute() == 1
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed'
    if failure == 'clock':
        assert receipt['wall_seconds'] is None and not receipt['timing_available']
    if failure == 'post_receipt':
        assert json.loads((out / 'invalid-completed-receipt.json').read_text())['status'] == 'completed'
    with pytest.raises(FileExistsError):
        task.execute()
