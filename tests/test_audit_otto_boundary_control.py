"""Artificial saved packets only; no TF, policy, native environment or weights."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def m():
    return module('boundary_audit_fake_test', 'scripts/audit_otto_boundary_control.py')


@pytest.fixture
def old():
    return module('reference_audit_test_fixtures', 'tests/test_audit_otto_released_reference.py')


def rows(m):
    result = []
    for identity in m.identities():
        candidate = identity['arm'] == 'released_inbounds'
        steps = 90 if candidate else 100
        row = dict.fromkeys(m.B.METRICS, 0.)
        row.update(identity, steps=steps, capped_time=steps, found=True,
                   choose_seconds=2. if candidate else 1., choose_calls=steps, update_calls=steps,
                   model_forward_calls=steps if identity['arm'] in m.NEURAL else 0,
                   state_array_bytes=22472, stuck_steps=0, blocked_steps=0,
                   controller_instrumented_seconds=2. if candidate else 1., controller_excluded_io_seconds=0.)
        row['controller_seconds'] = row['choose_seconds']
        result.append(row)
    return result


WEIGHTS = {'base': {1: .5, 2: .25, 3: .25}, 'shift': {1: .25, 2: .5, 3: .25}}


def test_768_four_arm_aggregate_and_all_rule_families(m):
    producer = module('boundary_schema_test', 'scripts/study_otto_boundary_control.py')
    records = rows(m)
    expected = json.loads(json.dumps(producer.summarize(records, WEIGHTS)))
    result, paired = m.aggregate(records, WEIGHTS)
    assert all(value == expected[key] for key, value in result.items())
    assert result['restriction_benefit'] and result['competent_reference'] and result['stronger_value_teacher']
    assert not result['utility_compute_advantage'] and len(paired) == 192
    assert len(result['restriction_benefit_checks']) == 5
    assert [sum(len(c[k]) for c in result['cohorts'].values()) for k in
            ('competence_checks', 'stronger_teacher_checks', 'utility_compute_checks')] == [6, 12, 16]
    assert all(len(c['blocks']) == 8 and len(c['means']) == 4 for c in result['cohorts'].values())


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'order'])
def test_complete_membership_cannot_be_selected(m, bad):
    records = rows(m)
    if bad == 'missing':
        records.pop()
    elif bad == 'duplicate':
        records[-1] = records[0]
    else:
        records[0], records[1] = records[1], records[0]
    with pytest.raises(ValueError, match='768'):
        m.aggregate(records, WEIGHTS)


def test_unequal_mixture_and_exact_inclusive_vs_strict_rules(m):
    records = rows(m)
    for row in records:
        if row['arm'] == 'released_inbounds':
            row['capped_time'] = {1: 80., 2: 100., 3: 120.}[row['initial_hit']]
    result, _ = m.aggregate(records, WEIGHTS)
    assert result['cohorts']['base']['means']['released_inbounds']['capped_time'] == 95.
    assert result['cohorts']['shift']['means']['released_inbounds']['capped_time'] == 100.
    assert result['restriction_benefit'] and not result['stronger_value_teacher']
    for value, teacher, competent in [(95., True, True), (math.nextafter(95., math.inf), False, True),
                                      (105., False, True), (math.nextafter(105., math.inf), False, False)]:
        for row in records:
            if row['arm'] == 'released_inbounds':
                row['capped_time'] = value
        result, _ = m.aggregate(records, WEIGHTS)
        assert result['stronger_value_teacher'] == teacher and result['competent_reference'] == competent
    for row in records:
        row['capped_time'] = 100.
    result, _ = m.aggregate(records, WEIGHTS)
    assert not result['restriction_benefit'] and all(c['passes'] for c in result['restriction_benefit_checks'][:4])


def test_raw_costs_retained_and_only_eligible_subset_tied(m):
    costs = np.asarray([0., 3., 0., 2.], np.float32).tolist()
    assert m.choose_from_raw(costs, [0, 1, 2, 3], 'released_tf', np) == 0
    assert m.choose_from_raw(costs, [1, 3], 'released_inbounds', np) == 3
    assert m.choose_from_raw([None, 3., None, 2.], [1, 3], 'analytic_inbounds', np) == 3
    with pytest.raises(ValueError, match='finite raw'):
        m.choose_from_raw([None, 3., None, 2.], [1, 3], 'released_inbounds', np)
    epsilon = np.float32(1e-10)
    assert m.choose_from_raw([2., float(epsilon), 2., 0.], [1, 3], 'released_inbounds', np) == 3
    assert m.choose_from_raw([2., float(np.nextafter(epsilon, np.float32(0))), 2., 0.], [1, 3], 'released_inbounds', np) == 1


def prefix(m, action=1):
    before = m.B.packet([52, 0], 0, False, 260)
    after = m.B.packet(m.B.move(before['position'], action), 0, False, 261)
    return {'before': before, 'after': after, 'state': 'saved-hash', 'costs': b'four-f32-costs', 'action': action}


def test_first_divergence_stops_comparison_not_after_different_paths(m):
    original, restricted = [prefix(m)], [prefix(m, 0)]
    original.append({'unrelated': 'later different history'})
    result = m.paired_prefix(original, restricted)
    assert result['first_divergent_action_step'] == 261 and result['common_prefix_decisions_checked'] == 1
    assert m.paired_prefix([prefix(m, 0)], [prefix(m, 0)])['first_divergent_action_step'] is None


@pytest.mark.parametrize('bad', ['cost', 'state', 'public', 'legal_original', 'post_public', 'missing_final'])
def test_common_prefix_corruption_rejected(m, bad):
    left, right = [prefix(m, 0)], [prefix(m, 0)]
    if bad == 'cost':
        right[0]['costs'] = b'different'
    elif bad == 'state':
        right[0]['state'] = 'different'
    elif bad == 'public':
        right[0]['before']['hit'] = 1
    elif bad == 'legal_original':
        right = [prefix(m, 3)]
    elif bad == 'post_public':
        right[0]['after']['hit'] = 1
    else:
        left.append(prefix(m, 0))
    with pytest.raises(ValueError):
        m.paired_prefix(left, right)


def episode(m, old, identity):
    _, row, trans, forwards, events, setup = old.episode_fixture(m.B)
    neural = identity['arm'] in m.NEURAL
    row.update(identity)
    row['initial_public']['hit'] = identity['initial_hit']
    setup.update(model_setup_seconds=384., model_setup_instrumented_seconds=384.)
    for record in trans:
        record.update(identity)
    for event in events:
        event['context'].update(identity)
    forwards[0]['context'].update(identity)
    if not neural:
        events = [event for event in events if event['channel'] != 'tensorflow_value']
        for event in events:
            if event['channel'] == 'actor_choose' and event['event'] == 'return':
                event.update(instrumented_seconds=.25, excluded_io_seconds=0.)
        row.update(model_setup_allocation_seconds=0., controller_seconds=.75, controller_instrumented_seconds=.75,
                   controller_excluded_io_seconds=0., model_forward_seconds=0., model_forward_calls=0)
        row['instrumented_component_seconds'].update(choose_seconds=.25, model_setup_allocation_seconds=0.)
        row['storage']['immutable_array_bytes'] += 107*107*8
        trans[-1].update(choose_instrumented_seconds=.25, choose_excluded_io_seconds=0., model_forward_seconds=0., model_forward_calls=0)
        forwards = []
        surviving = sorted({event['call_id'] for event in events})
        remap = {old_id: new_id for new_id, old_id in enumerate(surviving, 1)}
        for event in events:
            event['call_id'] = remap[event['call_id']]
            if event['parent_call_id'] is not None:
                event['parent_call_id'] = remap[event['parent_call_id']]
    trans[-1].update(selection_mask=[True]*4,
                    raw_costs_sha256=hashlib.sha256(np.asarray([1.]*4, np.float32 if neural else np.float64).tobytes()).hexdigest())
    return row, trans, forwards, events, setup


@pytest.mark.parametrize('arm', ['released_tf', 'released_inbounds', 'analytic_all4', 'analytic_inbounds'])
def test_all_four_saved_episode_routes(m, old, arm):
    identity = {**next(m.identities()), 'arm': arm}
    row, trans, forwards, events, setup = episode(m, old, identity)
    c, history = m.B.Comparisons(), []
    assert m.verify_episode(row, identity, iter(trans), iter(forwards), m.B.Journal(iter(events), c),
                            old.kernel(), setup, np, c, {}, history) == 0.
    assert len(history) == int(arm in m.NEURAL)


@pytest.mark.parametrize('bad', ['mask', 'rawhash', 'allocation192', 'mass', 'missing_return'])
def test_new_selection_and_cost_corruptions(m, old, bad):
    identity = {**next(m.identities()), 'arm': 'released_inbounds'}
    row, trans, forwards, events, setup = episode(m, old, identity)
    if bad == 'mask':
        trans[-1]['selection_mask'][0] = False
    elif bad == 'rawhash':
        trans[-1]['raw_costs_sha256'] = 'bad'
    elif bad == 'allocation192':
        setup.update(model_setup_seconds=192., model_setup_instrumented_seconds=192.)
    elif bad == 'mass':
        forwards[0]['branch_masses'][0][0] = .1
    else:
        events.pop()
    c = m.B.Comparisons()
    with pytest.raises((ValueError, StopIteration)):
        m.verify_episode(row, identity, iter(trans), iter(forwards), m.B.Journal(iter(events), c), old.kernel(), setup, np, c, {})


def test_auth_before_import_or_evidence_decode(m, tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    args = SimpleNamespace(plan=plan, plan_sha256='wrong', run=tmp_path/'run', output=tmp_path/'audit',
                           terminal=tmp_path/'terminal', receipt_sha256='wrong', terminal_sha256='wrong')
    audit = m.Audit(args)
    monkeypatch.setattr(audit, 'check', lambda: None)
    monkeypatch.setattr(m, 'load', lambda *args: pytest.fail('untrusted source import'))
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('array read before authentication'))
    with pytest.raises(ValueError, match='external'):
        audit.authenticate()


def test_failure_preserved_and_exclusive_output(m, tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path/'audit')
    def denied(self):
        raise ValueError('synthetic incomplete run')
    monkeypatch.setattr(m.Audit, 'authenticate', denied)
    monkeypatch.setattr(m.Audit, 'compute', lambda *args: pytest.fail('quality decoded'))
    with pytest.raises(ValueError, match='incomplete'):
        m.Audit(args).execute()
    assert json.loads((args.output/'failed.json').read_text())['agreement'] is False
    assert not (args.output/'receipt.json').exists()
    with pytest.raises(FileExistsError):
        m.Audit(args).execute()


def test_complete_artificial_saved_compute_all768_episodes(m, old, tmp_path, monkeypatch):
    producer = module('synthetic_producer_for_saved_join', 'scripts/study_otto_boundary_control.py')
    run = tmp_path / 'run'
    run.mkdir()
    paths, recordings = {}, {n: [] for n in ('work', 'transitions', 'forwards', 'episodes', 'weights', 'paired-prefixes')}
    count = {ch: {'attempted': 0, 'returned': 0, 'seconds': 0., 'instrumented_seconds': 0., 'excluded_io_seconds': 0.} for ch in m.B.CHANNELS}
    sequence, stack = 0, []

    def emit_events(events):
        nonlocal sequence
        for item in events:
            ch = item['channel']
            if item['event'] == 'attempt':
                sequence += 1
                count[ch]['attempted'] += 1
                identity = {'call_id': sequence, 'channel': ch, 'ordinal': count[ch]['attempted'],
                            'parent_call_id': stack[-1]['call_id'] if stack else None, 'context': item['context']}
                stack.append(identity)
                recordings['work'].append({'event': 'attempt', **identity})
            else:
                identity = stack.pop()
                count[ch]['returned'] += 1
                durations = {k: item[k] for k in ('seconds', 'instrumented_seconds', 'excluded_io_seconds')}
                for k, v in durations.items():
                    count[ch][k] += v
                recordings['work'].append({'event': 'return', **identity, **durations})
    for ch in ('tensorflow_construction', 'tensorflow_build', 'tensorflow_load'):
        emit_events([{'event': 'attempt', 'channel': ch, 'context': {'phase': 'model_setup'}},
                     {'event': 'return', 'channel': ch, 'seconds': .25, 'instrumented_seconds': .25, 'excluded_io_seconds': 0.}])
    setup = {'model_setup_seconds': 384., 'model_setup_instrumented_seconds': 384., 'model_setup_excluded_io_seconds': 0.,
             'allocated_over_released_episodes': 384, 'model_parameters': 13390849, 'model_tensor_bytes': 53563396}
    for identity in m.identities():
        row, trans, forwards, events, _ = episode(m, old, identity)
        neural = identity['arm'] in m.NEURAL
        emit_events(events)
        if neural:
            forwards[0]['ordinal'] = count['tensorflow_value']['returned']
            recordings['forwards'].extend(forwards)
        recordings['transitions'].extend(trans)
        recordings['episodes'].append(row)
    for regime, w in WEIGHTS.items():
        paths[regime + '_kernel'] = tmp_path / (regime + '.npz')
        np.savez(paths[regime + '_kernel'], likelihood=old.kernel(), initial_hit_weights=np.array([0, w[1], w[2], w[3]]))
    metadata = {'datasets': {f'{kind}_{i}': {'shape': [1], 'sha256_c_order': str(i)*64} for i in range(4) for kind in ('kernel', 'bias')}}
    paths['tensor_metadata'] = tmp_path / 'metadata.json'
    paths['tensor_metadata'].write_text(json.dumps(metadata))
    recordings['weights'] = [{'id': k, 'shape': v['shape'], 'sha256': v['sha256_c_order'], 'exact': True} for k, v in metadata['datasets'].items()]
    recordings['paired-prefixes'] = [{'cohort': regime, 'seed': first + i, 'common_prefix_decisions_checked': 1,
            'first_divergent_action_step': None, 'raw_cost_and_posterior_parity': True,
            'first_divergence_requires_original_blocked': True} for regime, first in m.REGIMES.items() for i in range(96)]
    for name, records in recordings.items():
        (run / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in records))
    (run / 'setup.json').write_text(json.dumps(setup))
    summary = producer.summarize(recordings['episodes'], WEIGHTS)
    summary.update(paired_neural_prefixes=recordings['paired-prefixes'], setup=setup, paired_source_cases=192, paired_uniform_checks=192, all_public_beliefs_exact=True,
                   actual_work={ch: d['returned'] for ch, d in count.items()}, artifact_io_seconds=96.)
    for field, key in (('sum_controller_seconds', 'controller_seconds'), ('sum_controller_instrumented_seconds', 'controller_instrumented_seconds'), ('sum_excluded_controller_io_seconds', 'controller_excluded_io_seconds')):
        summary[field] = math.fsum(r[key] for r in recordings['episodes'])
    (run / 'summary.json').write_text(json.dumps(summary))
    worker = {'wall_seconds': 1600., 'work': {'calls': count, 'pending': [], 'sequence': sequence,
              'context': recordings['work'][-1]['context'], 'artifact_io_seconds': 96.},
              **{k: summary[k] for k in ('restriction_benefit', 'competent_reference', 'stronger_value_teacher', 'utility_compute_advantage')}}
    a = m.Audit(SimpleNamespace(run=run, output=tmp_path / 'out'))
    monkeypatch.setattr(a, 'check', lambda: None)
    result = a.compute(paths, worker)
    assert result['agreement'] and result['episodes'] == 768 and len(result['paired_cases']) == 192
    assert result['actual_work']['tensorflow_value'] == 384 and result['public_updates'] == 768
