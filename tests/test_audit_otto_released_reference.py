"""Small synthetic saved-evidence checks; no simulator, TensorFlow or weights."""
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def m():
    return module('synthetic_reference_audit', 'scripts/audit_otto_released_reference.py')


def kernel():
    k = np.full((4, 107, 107), .25, dtype=np.float64)
    k[:, 53, 53] = 0
    return k


def test_closed_form_public_filter_threshold_terminal_and_branch_masses(m):
    p = np.zeros((53, 53), dtype=np.float64)
    p[25, 26], p[27, 26] = .25, .75
    k = kernel()
    after = m.posterior(p, m.packet([25, 26], 0, False, 1), k, np)
    assert after[27, 26] == 1 and np.count_nonzero(after) == 1
    masses = m.branch_masses(p, [26, 26], k, np)
    np.testing.assert_array_equal(masses, np.asarray([[.1875]*4, [.0625]*4, [.25]*4, [.25]*4], np.float32))
    found = m.posterior(p, m.packet([25, 26], -2, True, 1), k, np)
    assert found[25, 26] == 1 and found.sum() == 1
    tiny = p * 4e-10
    unnormalized = m.posterior(tiny, m.packet([26, 26], 0, False, 1), k, np)
    np.testing.assert_array_equal(unnormalized, p * 1e-10)


def test_float32_ties_are_not_promoted_to_float64_or_relaxed(m):
    costs = np.asarray([86.52845, 86.56498, 86.52844, 86.56498], np.float32).tolist()
    assert m.selected_action(costs, list(range(4)), np, True) == 2
    assert m.selected_action([None, 1., None, 1.], [1, 3], np) == 1
    with pytest.raises(ValueError, match='null'):
        m.selected_action([0., 1., None, 1.], [1, 3], np)


def test_neural_cost_roundoff_bound_is_separate_from_exact_action(m):
    masses = np.full((4, 4), .25, np.float32)
    record = {'branch_masses': masses.tolist(), 'values': [3.] * 16}
    assert m.neural_costs(record, [4.] * 4, masses, np) == 0
    with pytest.raises(ValueError, match='bound'):
        m.neural_costs(record, [4.001] * 4, masses, np)
    record['branch_masses'][0][0] = .125
    with pytest.raises(ValueError, match='mass'):
        m.neural_costs(record, [4.] * 4, masses, np)


def synthetic_rows(m):
    rows = []
    for identity in m.identities():
        steps = 90 if identity['arm'] == 'released_tf' else 100
        row = {key: 0. for key in m.METRICS}
        row.update(identity, steps=steps, capped_time=steps, found=True,
                   stuck_steps=0, blocked_steps=0,
                   choose_seconds=2. if identity['arm'] == 'released_tf' else 1.,
                   choose_calls=steps, update_calls=steps, model_forward_calls=steps if identity['arm'] == 'released_tf' else 0,
                   state_array_bytes=22472, controller_instrumented_seconds=0., controller_excluded_io_seconds=0.)
        row['controller_seconds'] = row['controller_instrumented_seconds'] = row['choose_seconds']
        rows.append(row)
    return rows


def weights():
    return {'base': {1: .5, 2: .25, 3: .25}, 'shift': {1: .25, 2: .5, 3: .25}}


def test_complete_576_cohort_independent_aggregate_matches_producer_schema(m):
    # Producer use is test-only and does not supply auditor arithmetic.
    producer = module('synthetic_reference_producer', 'scripts/study_otto_released_reference.py')
    rows = synthetic_rows(m)
    expected = json.loads(json.dumps(producer.summarize(rows, weights())))
    result, paired = m.aggregate(rows, weights())
    for key, value in result.items():
        assert value == expected[key]
    assert result['competent_reference'] and result['stronger_value_teacher'] and not result['utility_compute_advantage']
    assert len(paired) == 192 and sum(len(c['competence_checks']) for c in result['cohorts'].values()) == 6
    assert sum(len(c['stronger_teacher_checks']) for c in result['cohorts'].values()) == 12
    assert sum(len(c['utility_compute_checks']) for c in result['cohorts'].values()) == 16


@pytest.mark.parametrize('bad', ['missing', 'order', 'duplicate'])
def test_membership_cannot_be_selected_or_replaced(m, bad):
    rows = synthetic_rows(m)
    if bad == 'missing':
        rows.pop()
    elif bad == 'order':
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[-1] = rows[0]
    with pytest.raises(ValueError, match='576'):
        m.aggregate(rows, weights())


def test_mixture_not_raw_fraction_and_all_eight_blocks(m):
    rows = synthetic_rows(m)
    for r in rows:
        if r['arm'] == 'released_tf':
            r['capped_time'] = r['steps'] = {1: 80, 2: 100, 3: 120}[r['initial_hit']]
    result, _ = m.aggregate(rows, weights())
    assert result['cohorts']['base']['means']['released_tf']['capped_time'] == 95
    assert result['cohorts']['shift']['means']['released_tf']['capped_time'] == 100
    assert not result['stronger_value_teacher']
    check = result['cohorts']['shift']['stronger_teacher_checks'][2]
    assert check['value'] == 0


def test_exact_inclusive_margin_and_strict_block_win(m):
    rows = synthetic_rows(m)
    for r in rows:
        if r['arm'] == 'released_tf':
            r['capped_time'] = 95.
    assert m.aggregate(rows, weights())[0]['stronger_value_teacher']
    for r in rows:
        if r['arm'] == 'released_tf':
            r['capped_time'] = math.nextafter(95., math.inf)
    assert not m.aggregate(rows, weights())[0]['stronger_value_teacher']
    for r in rows:
        if r['arm'] == 'released_tf':
            r['capped_time'] = 105.
    assert m.aggregate(rows, weights())[0]['competent_reference']
    for r in rows:
        if r['arm'] == 'released_tf':
            r['capped_time'] = math.nextafter(105., math.inf)
    assert not m.aggregate(rows, weights())[0]['competent_reference']


def episode_fixture(m):
    """One source one move north: closed-form terminal and paid nested call."""
    identity = next(m.identities())
    public = m.packet([26, 26], 1, False, 0)
    final = m.packet([25, 26], -2, True, 1)
    p = np.ones((53, 53), np.float64) / 2808
    p[26, 26] = 0
    # Explicit fixture arithmetic in the producer's operation order.
    p *= .25
    p /= p.sum()
    terminal = np.zeros_like(p)
    terminal[25, 26] = 1
    state, after = m.witness(p), m.witness(terminal)
    reset_context = {'phase': 'episode_reset', **identity}
    decision = {'phase': 'decision', **identity, 'step': 1}
    events, sequence, counts = [], [0], dict.fromkeys(m.CHANNELS, 0)

    def call(ch, context, child=False, parent=None):
        sequence[0] += 1
        counts[ch] += 1
        record = {'call_id': sequence[0], 'channel': ch, 'ordinal': counts[ch], 'parent_call_id': parent, 'context': context}
        events.append({'event': 'attempt', **record})
        if child:
            call('tensorflow_value', context, parent=record['call_id'])
        events.append({'event': 'return', **record, 'seconds': .25,
                       'instrumented_seconds': .5 if child else .25, 'excluded_io_seconds': .25 if child else 0.})
    call('native_reset', reset_context)
    call('actor_construction', reset_context)
    call('actor_choose', decision, child=True)
    call('native_step', decision)
    call('actor_update', decision)
    transitions = [{'kind': 'reset', **identity, 'public': public, 'posterior_after': state, 'source_evaluation_only': [25, 26]},
                   {'kind': 'step', **identity, 'step': 1, 'action': 0, 'costs': [1.] * 4, 'allowed_actions': [0, 1, 2, 3],
                    'public': final, 'posterior_before': state, 'posterior_after': after, 'native_p_end': 1.,
                    'blocked': False, 'stuck': False, 'choose_seconds': .25, 'choose_instrumented_seconds': .5,
                    'choose_excluded_io_seconds': .25, 'update_seconds': .25, 'update_instrumented_seconds': .25,
                    'environment_seconds': .25, 'model_forward_seconds': .25, 'model_forward_calls': 1}]
    masses = []
    for a in range(4):
        target = m.move([26, 26], a)
        joint = p.copy() * .25
        joint[tuple(target)] = 0
        masses.append([float(np.float32(joint.sum()))] * 4)
    forwards = [{'context': decision, 'ordinal': 1, 'symmetry_average': True, 'input_shape': [16, 105, 105],
                 'branch_masses': masses, 'values': [0.] * 16, 'seconds': .25}]
    setup = {'model_setup_seconds': 192., 'model_setup_instrumented_seconds': 192.}
    row = {**identity, 'steps': 1, 'capped_time': 1, 'found': True, 'stuck_steps': 0, 'blocked_steps': 0,
           'actor_initialization_seconds': .25, 'choose_seconds': .25, 'update_seconds': .25, 'model_setup_allocation_seconds': 1.,
           'controller_seconds': 1.75, 'controller_instrumented_seconds': 2., 'controller_excluded_io_seconds': .25,
           'instrumented_component_seconds': {'actor_initialization_seconds': .25, 'choose_seconds': .5, 'update_seconds': .25, 'model_setup_allocation_seconds': 1.},
           'model_forward_seconds': .25, 'environment_initialization_seconds': .25, 'environment_seconds': .25,
           'episode_seconds': 2., 'state_array_bytes': 22472, 'storage': {'mutable_array_bytes': 22472, 'immutable_array_bytes': 366368},
           'choose_calls': 1, 'update_calls': 1, 'model_forward_calls': 1, 'initial_public': public,
           'source_evaluation_only': [25, 26], 'draws_evaluation_only': [{'channel': 'source', 'index': 0, 'uniform': .5,
                   'selected_index': 25*53+26, 'cdf_mass': 1.}], 'final_update_assimilated': True}
    return identity, row, transitions, forwards, events, setup


@pytest.mark.parametrize('corruption', [None, 'action', 'posterior', 'cost', 'mass', 'found', 'ledger', 'stuck', 'update', 'extra_draw'])
def test_saved_episode_join_and_deliberate_corruptions(m, corruption):
    identity, row, trans, forwards, events, setup = episode_fixture(m)
    if corruption == 'action':
        trans[-1]['action'] = 1
    elif corruption == 'posterior':
        trans[-1]['posterior_after']['sha256'] = '0'*64
    elif corruption == 'cost':
        trans[-1]['costs'] = [2.]*4
    elif corruption == 'mass':
        forwards[0]['branch_masses'][0][0] = .25
    elif corruption == 'found':
        trans[-1]['public']['hit'] = 0
    elif corruption == 'ledger':
        events[3]['call_id'] = 123
    elif corruption == 'stuck':
        trans[-1]['stuck'] = True
    elif corruption == 'update':
        row['final_update_assimilated'] = False
    elif corruption == 'extra_draw':
        row['draws_evaluation_only'].append(copy.deepcopy(row['draws_evaluation_only'][0]))
    def execute():
        c = m.Comparisons()
        return m.verify_episode(row, identity, iter(trans), iter(forwards), m.Journal(iter(events), c), kernel(), setup, np, c, {})
    if corruption:
        with pytest.raises(ValueError):
            execute()
    else:
        assert execute() == 0.


def test_incomplete_journal_keeps_attempt_as_rejection(m):
    _, _, _, _, events, _ = episode_fixture(m)
    events.pop()
    j = m.Journal(iter(events), m.Comparisons())
    with pytest.raises((ValueError, StopIteration)):
        while True:
            j.call('native_reset', events[0]['context'])


def test_auth_rejects_bad_plan_before_producer_import_or_array_reads(m, tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    out = tmp_path / 'out'
    out.mkdir()
    args = SimpleNamespace(plan=plan, plan_sha256='bad', receipt_sha256='bad', terminal_sha256='bad',
                           run=tmp_path / 'run', terminal=tmp_path / 'terminal', output=out)
    a = m.Audit(args)
    monkeypatch.setattr(a, 'check', lambda: None)
    monkeypatch.setattr(m, 'load', lambda *args: pytest.fail('untrusted code imported'))
    with pytest.raises(ValueError, match='external'):
        a.authenticate()


def test_failure_receipt_and_exclusive_output_without_any_evidence_read(m, tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / 'audit')
    def denied(self):
        raise ValueError('synthetic missing complete run')
    monkeypatch.setattr(m.Audit, 'authenticate', denied)
    monkeypatch.setattr(m.Audit, 'compute', lambda *args: pytest.fail('quality decode'))
    with pytest.raises(ValueError, match='missing complete'):
        m.Audit(args).execute()
    assert json.loads((args.output / 'failed.json').read_text())['status'] == 'failed'
    assert not (args.output / 'receipt.json').exists()
    with pytest.raises(FileExistsError):
        m.Audit(args).execute()


def test_upstream_stuck_requires_nine_consecutive_two_step_returns(m):
    repeated = 0
    flags = []
    for _ in range(10):
        repeated, stuck = m.stuck_counter(repeated, [1, 1], [1, 1])
        flags.append(stuck)
    assert flags == [False]*8 + [True, True]
    assert m.stuck_counter(repeated, [1, 1], [2, 1]) == (0, False)


def test_complete_artificial_saved_compute_all576_episodes(m, tmp_path, monkeypatch):
    producer = module('synthetic_producer_for_saved_join', 'scripts/study_otto_released_reference.py')
    run = tmp_path / 'run'
    run.mkdir()
    paths, recordings = {}, {n: [] for n in ('work', 'transitions', 'forwards', 'episodes', 'weights')}
    count = {ch: {'attempted': 0, 'returned': 0, 'seconds': 0., 'instrumented_seconds': 0., 'excluded_io_seconds': 0.} for ch in m.CHANNELS}
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
    setup = {'model_setup_seconds': 192., 'model_setup_instrumented_seconds': 192., 'model_setup_excluded_io_seconds': 0.,
             'allocated_over_released_episodes': 192, 'model_parameters': 13390849, 'model_tensor_bytes': 53563396}
    for identity in m.identities():
        _, row, trans, forwards, events, _ = episode_fixture(m)
        neural = identity['arm'] == 'released_tf'
        row.update(identity)
        row['initial_public']['hit'] = identity['initial_hit']
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
        emit_events(events)
        if neural:
            forwards[0]['ordinal'] = count['tensorflow_value']['returned']
            recordings['forwards'].extend(forwards)
        recordings['transitions'].extend(trans)
        recordings['episodes'].append(row)
    for regime, w in weights().items():
        paths[regime + '_kernel'] = tmp_path / (regime + '.npz')
        np.savez(paths[regime + '_kernel'], likelihood=kernel(), initial_hit_weights=np.array([0, w[1], w[2], w[3]]))
    metadata = {'datasets': {f'{kind}_{i}': {'shape': [1], 'sha256_c_order': str(i)*64} for i in range(4) for kind in ('kernel', 'bias')}}
    paths['tensor_metadata'] = tmp_path / 'metadata.json'
    paths['tensor_metadata'].write_text(json.dumps(metadata))
    recordings['weights'] = [{'id': k, 'shape': v['shape'], 'sha256': v['sha256_c_order'], 'exact': True} for k, v in metadata['datasets'].items()]
    for name, records in recordings.items():
        (run / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in records))
    (run / 'setup.json').write_text(json.dumps(setup))
    summary = producer.summarize(recordings['episodes'], weights())
    summary.update(setup=setup, paired_source_cases=192, paired_uniform_checks=192, all_public_beliefs_exact=True,
                   actual_work={ch: d['returned'] for ch, d in count.items()}, artifact_io_seconds=48.)
    for field, key in (('sum_controller_seconds', 'controller_seconds'), ('sum_controller_instrumented_seconds', 'controller_instrumented_seconds'), ('sum_excluded_controller_io_seconds', 'controller_excluded_io_seconds')):
        summary[field] = math.fsum(r[key] for r in recordings['episodes'])
    (run / 'summary.json').write_text(json.dumps(summary))
    worker = {'wall_seconds': 1000., 'work': {'calls': count, 'pending': [], 'sequence': sequence,
              'context': recordings['work'][-1]['context'], 'artifact_io_seconds': 48.},
              **{k: summary[k] for k in ('competent_reference', 'stronger_value_teacher', 'utility_compute_advantage')}}
    a = m.Audit(SimpleNamespace(run=run, output=tmp_path / 'out'))
    monkeypatch.setattr(a, 'check', lambda: None)
    result = a.compute(paths, worker)
    assert result['agreement'] and result['episodes'] == 576 and len(result['paired_cases']) == 192
    assert result['actual_work']['tensorflow_value'] == 192 and result['public_updates'] == 576
