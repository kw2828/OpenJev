"""Synthetic collection metadata and public-array fixtures only."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_coverage_audit_test', ROOT/'scripts/audit_otto_coverage_collection.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def test_integer_largest_remainder_preserves_original_cells():
    counts = dict(zip(A.CELLS, (8, 5, 3, 7, 6, 2), strict=True))
    result = A.largest_remainders(counts, 15)
    assert list(result.values()) == [4, 2, 2, 3, 3, 1]
    assert sum(result.values()) == 15
    assert sum(counts[k]-result[k] for k in counts) == 16


def test_quota_ties_use_fixed_cell_and_collector_order():
    counts = dict.fromkeys(A.CELLS, 1)
    assert list(A.largest_remainders(counts, 3).values()) == [1, 1, 1, 0, 0, 0]
    assert A.collector_quotas(5) == {10101: 2, 10102: 2, 10103: 1}


@pytest.mark.parametrize(('counts', 'total'), [({}, 0), ({'x': 0}, 0), ({'x': True}, 1), ({'x': 2}, 3), ({'x': -1}, 0)])
def test_invalid_quota_contract_rejected(counts, total):
    with pytest.raises(ValueError):
        A.largest_remainders(counts, total)


def test_exact72_collection_schedule_pairs_each_case_three_ways():
    rows = list(A.collection_order())
    assert len(rows) == len(set(rows)) == 72
    assert rows[:3] == [('lambda3', 13100001, 1, 10101), ('lambda3', 13100001, 1, 10102), ('lambda3', 13100001, 1, 10103)]
    assert [r[-1] for r in rows[3:6]] == [10102, 10103, 10101]
    for cell in A.CELLS:
        for seed in A.SEEDS:
            assert sum((r[0], r[2]) == cell and r[3] == seed for r in rows) == 4


def test_censored_episode_keeps_final_update_without_found_label():
    identity = next(A.collection_order())
    row = dict(zip(('regime', 'seed', 'initial_hit', 'collector_seed'), identity, strict=True))
    row.update(steps=2188, found=False, updates=2188, final_update_assimilated=True)
    assert A.recorded_episode(row, identity) == 2188
    row['steps'] = row['updates'] = 10
    with pytest.raises(ValueError, match='censored'):
        A.recorded_episode(row, identity)


def test_hash_selection_uses_public_row_ids_not_order_or_success():
    import hashlib
    ids = ['teacher:10', 'teacher:2', 'student:lambda3:13100001:10101:0', 'student:lambda3:13100001:10101:17']
    expected = [item[1] for item in sorted((hashlib.sha256(('otto-coverage-v1|'+s).encode()).hexdigest(), s) for s in ids)[:2]]
    assert A.ranked_selection(ids, 2) == expected == A.ranked_selection(ids[::-1], 2)
    assert ids[0] == 'teacher:10'


def test_selection_rejects_duplicate_ids_and_insufficient_fixed_pool():
    with pytest.raises(ValueError, match='unique'):
        A.ranked_selection(['same', 'same'], 1)
    with pytest.raises(ValueError, match='insufficient'):
        A.ranked_selection(['only'], 2)


def test_exact5589_mixture_preserves_every_original_stratum():
    rows = []
    for cell, count in zip(A.CELLS, [1500, 200, 99, 2800, 700, 290], strict=True):
        for _ in range(count):
            rows.append({'row_index': len(rows), 'regime': cell[0], 'initial_hit': cell[1]})
    records = A.quota_records(rows)
    assert sum(r['student_rows'] for r in records) == 2790
    assert sum(r['teacher_rows'] for r in records) == 2799
    for row in records:
        assert row['student_rows']+row['teacher_rows'] == row['original_rows']
        assert sum(row['collector_rows'].values()) == row['student_rows']
        assert max(row['collector_rows'].values())-min(row['collector_rows'].values()) <= 1


@pytest.fixture
def helper():
    return A.load(ROOT/A.HELPER, '_coverage_audit_independent_test_math')


def synthetic_event(helper, *, found=False):
    import numpy as np
    kernel = np.full((4, 107, 107), .25, np.float64)
    kernel[:, 53, 53] = 0
    probability = np.zeros((53, 53), np.float64)
    probability[25, 26] = probability[26, 27] = .5
    public = helper.B.packet([26, 26], 1, False, 0)
    source = [25, 26] if found else [30, 30]
    after = helper.B.packet([25, 26], -2 if found else 0, found, 1)
    updated = helper.B.posterior(probability, after, kernel, np)
    event = {'step': 1, 'action': 0, 'allowed_actions': [0, 1, 2, 3], 'public': after,
             'posterior_before': helper.A.posterior_record(probability),
             'posterior_after': helper.A.posterior_record(updated), 'native_p_end': float(found)}
    return probability, public, event, source, kernel, np


@pytest.mark.parametrize('found', [False, True])
def test_public_replay_keeps_zero_hits_and_terminal_empty_actions(helper, found):
    probability, public, event, source, kernel, np = synthetic_event(helper, found=found)
    before = probability.tobytes()
    result, after = A.public_update(probability, public, event, source, kernel, helper, np)
    assert probability.tobytes() == before
    assert after['done'] is found
    if found:
        assert after['valid_actions'] == [] and result[25, 26] == 1
    else:
        assert after['hit'] == 0 and result[26, 27] == 1


@pytest.mark.parametrize('defect', ['offset', 'terminal_nonempty', 'posterior', 'mask', 'sentinel'])
def test_public_replay_rejects_chronology_and_terminal_contract_defects(helper, defect):
    probability, public, event, source, kernel, np = synthetic_event(helper, found=True)
    if defect == 'offset':
        event['step'] = 2
    elif defect == 'terminal_nonempty':
        event['public']['valid_actions'] = [0, 1, 2, 3]
    elif defect == 'posterior':
        event['posterior_after']['sha256'] = 'bad'
    elif defect == 'mask':
        event['allowed_actions'] = [1, 2, 3]
    else:
        event['public']['hit'] = 0
    with pytest.raises(ValueError):
        A.public_update(probability, public, event, source, kernel, helper, np)


def branch_fixture(helper):
    import numpy as np
    raw = np.zeros((4, 4), np.float64)
    weights = np.full((4, 4), 1e-10, np.float64)
    values = np.ones(16, np.float64)*64
    costs = helper.costs(values, weights, np)
    event = {'raw_masses': raw.tolist(), 'weights': weights.tolist(), 'values': values.tolist(),
             'costs': costs.tolist(), 'allowed_actions': [1, 3], 'action': 1}
    return event, values, raw, weights, costs, np


def test_branch_replay_preserves_biased_zero_values_and_physical_scale(helper):
    event, values, raw, weights, costs, np = branch_fixture(helper)
    assert costs[0] > 1  # Biased C(0)=64 is not overwritten on floored zero branches.
    assert A.branch_witness(event, values, raw, weights, costs, helper, np) == 0


@pytest.mark.parametrize('defect', ['floor', 'negative_mass', 'missing_value', 'nonfinite', 'masked_action'])
def test_branch_witness_rejects_tiny_structural_and_selection_defects(helper, defect):
    event, values, raw, weights, costs, np = branch_fixture(helper)
    if defect == 'floor':
        event['weights'][0][0] = 2e-10
    elif defect == 'negative_mass':
        event['raw_masses'][0][0] = -1e-12
    elif defect == 'missing_value':
        event['values'].pop()
    elif defect == 'nonfinite':
        event['values'][0] = float('nan')
    else:
        event['action'] = 0
    with pytest.raises(ValueError):
        A.branch_witness(event, values, raw, weights, costs, helper, np)


def selected_fixture(helper, tmp_path):
    import argparse
    import hashlib

    import numpy as np
    audit = A.Audit(argparse.Namespace(output=tmp_path/'unused'))
    audit.np, audit.E = np, helper
    probability, public, _, _, _, _ = synthetic_event(helper)
    metadata = {'regime': 'lambda3', 'public': public, 'source': 'student', 'prefix_index': 0,
                'posterior': helper.A.posterior_record(probability)}
    identity = 'train:lambda3:13100001:mlp8@10101:0'
    recorded = {'mixture_index': 0, **metadata, 'identity': identity,
                'key': hashlib.sha256((A.SALT+'|'+identity).encode()).hexdigest()}
    audit.selected = {identity: (0, recorded)}
    audit.verified_selected = set()
    audit.mixture = {'beliefs': probability[None].copy(),
                     'features': helper.state_features(probability, [26, 26], 3., np)[None],
                     'positions': np.asarray([[26, 26]], np.int64), 'sensing_length': np.asarray([3.])}
    return audit, identity, probability, metadata


def test_selected_state_verifies_preaction_identity_and_exact_public_arrays(helper, tmp_path):
    audit, identity, probability, metadata = selected_fixture(helper, tmp_path)
    audit.selected_state(identity, probability, metadata)
    assert audit.verified_selected == {identity}
    with pytest.raises(ValueError, match='exactly once'):
        audit.selected_state(identity, probability, metadata)


@pytest.mark.parametrize('defect', ['offset', 'feature', 'belief', 'position', 'kernel', 'hidden_label'])
def test_selected_state_rejects_changed_state_or_extra_label(helper, tmp_path, defect):
    audit, identity, probability, metadata = selected_fixture(helper, tmp_path)
    if defect == 'offset':
        audit.selected[identity][1]['prefix_index'] = 1
    elif defect == 'feature':
        audit.mixture['features'][0, 0] = 1
    elif defect == 'belief':
        audit.mixture['beliefs'][0, 0, 0] = 1
    elif defect == 'position':
        audit.mixture['positions'][0, 0] = 25
    elif defect == 'kernel':
        audit.mixture['sensing_length'][0] = 4.
    else:
        audit.selected[identity][1]['target'] = 1
    with pytest.raises(ValueError):
        audit.selected_state(identity, probability, metadata)
    assert not audit.verified_selected


def episode_fixture(helper, tmp_path):
    import argparse
    from collections import defaultdict

    import numpy as np
    audit = A.Audit(argparse.Namespace(output=tmp_path/'unused'))
    audit.np, audit.E, audit.check = np, helper, lambda: None
    audit.sources, audit.uniforms = {}, {}
    audit.collection_reset_seconds = 0.
    audit.candidates = defaultdict(list)
    audit.selected, audit.verified_selected = {}, set()
    kernel = np.full((4, 107, 107), .25)
    kernel[:, 53, 53] = 0
    audit.kernels = {'lambda3': kernel}
    weights = {'c0': np.asarray(0.), 'first_weight': np.zeros((8, 11028)),
               'final_weight': np.zeros(8), 'hidden_bias': np.zeros(8), 'output_bias': np.asarray(1.)}
    audit.heads = {10101: weights}
    identity = next(A.collection_order())
    canonical = {'episode_id': 'train:lambda3:13100001:mlp8@10101', 'stage': 'train_collection',
                 'regime': 'lambda3', 'seed': 13100001, 'initial_hit': 1, 'collector_seed': 10101, 'case': 0}
    public = helper.B.packet([26, 26], 1, False, 0)
    p = helper.B.posterior(np.ones((53, 53))/2808, public, kernel, np)
    reset = {'kind': 'reset', **canonical, 'public': public, 'posterior_after': helper.A.posterior_record(p),
             'source_evaluation_only': [25, 26]}
    after = helper.B.packet([25, 26], -2, True, 1)
    updated = helper.B.posterior(p, after, kernel, np)
    x, raw, floor = helper.branches(p, [26, 26], kernel, 3., np)
    values = 64*helper.predict(x, weights, 'mlp8', np)
    event = {'kind': 'step', 'episode_id': canonical['episode_id'], 'step': 1, 'action': 0,
             'costs': helper.costs(values, floor, np).tolist(), 'allowed_actions': [0, 1, 2, 3],
             'raw_masses': raw.tolist(), 'weights': floor.tolist(), 'values': values.tolist(),
             'public': after, 'posterior_before': helper.A.posterior_record(p),
             'posterior_after': helper.A.posterior_record(updated), 'native_p_end': 1.,
             'choose_seconds': .3, 'choose_instrumented_seconds': .4, 'choose_excluded_io_seconds': .1,
             'update_seconds': .2, 'environment_seconds': .1}
    row = {**canonical, 'steps': 1, 'found': True, 'updates': 1, 'blocked_steps': 0,
           'init_seconds': .1, 'controller_seconds': .6, 'sampling_seconds': .01,
           'source_evaluation_only': [25, 26], 'draws_evaluation_only': [
               {'channel': 'source', 'index': 0, 'uniform': .5, 'selected_index': 25*53+26, 'cdf_mass': 1.}],
           'final_public': after, 'final_posterior_mass': 1., 'final_update_assimilated': True,
           'training_labels_generated': False,
           **{key: event[key] for key in ('choose_seconds', 'choose_instrumented_seconds',
                                         'choose_excluded_io_seconds', 'update_seconds', 'environment_seconds')}}
    context = {'phase': 'collection', 'episode': canonical['episode_id']}
    records = [[1, 0, 'native_reset', 0, 0], [1, 1, 'native_reset', .1, .1, 0.],
               [2, 0, 'value_forward', 0, 1], [2, 1, 'value_forward', .1, .1, 0.],
               [3, 0, 'native_step', 0, 1], [3, 1, 'native_step', .1, .1, 0.]]
    audit.work = helper.Work(iter(records), [context], helper.B.Comparisons())
    return audit, row, identity, [reset, event]


def test_full_saved_episode_replay_has_one16row_readout_and_final_found_update(helper, tmp_path):
    audit, row, identity, events = episode_fixture(helper, tmp_path)
    assert audit.episode(row, identity, iter(events)) is row
    assert audit.readout_attempts == audit.readout_returns == 1 and audit.readout_rows == 16
    assert audit.candidates[('lambda3', 1, 10101)] == [row['episode_id']+':0']
    assert audit.collection_reset_seconds == .1
    assert next(audit.work.iterator, None) is None


@pytest.mark.parametrize('defect', ['branch', 'controller_cost', 'next_state_selection', 'extra_draw', 'model_context'])
def test_complete_episode_integration_rejects_saved_join_defects(helper, tmp_path, defect):
    audit, row, identity, events = episode_fixture(helper, tmp_path)
    if defect == 'branch':
        events[1]['values'][0] += 1
    elif defect == 'controller_cost':
        row['controller_seconds'] += .01
    elif defect == 'next_state_selection':
        events[1]['posterior_before'] = events[1]['posterior_after']
    elif defect == 'extra_draw':
        row['draws_evaluation_only'].append(dict(row['draws_evaluation_only'][0]))
    else:
        audit.work.contexts[0]['episode'] = 'wrong'
    with pytest.raises(ValueError):
        audit.episode(row, identity, iter(events))


def test_closed_payload_manifest_rejects_unbound_file_before_json_decode(tmp_path):
    root = tmp_path/'evidence'
    root.mkdir()
    for name in A.PAYLOADS:
        (root/name).write_bytes(b'fake bytes, deliberately not scientific inputs')
    A.write(root/'receipt.json', {'status': 'completed', 'files': {name: A.descriptor(root/name) for name in A.PAYLOADS}})
    pin = A.descriptor(root/'receipt.json')['sha256']
    assert A.closed(root, pin, A.PAYLOADS, lambda: None)['status'] == 'completed'
    (root/'unbound').write_text('extra')
    with pytest.raises(ValueError, match='closure'):
        A.closed(root, pin, A.PAYLOADS, lambda: None)


def test_clock_construction_failure_preserves_failure_artifact(tmp_path, monkeypatch):
    import argparse
    from types import SimpleNamespace
    def fail_clock():
        raise RuntimeError('synthetic clock unavailable')
    monkeypatch.setattr(A, 'load', lambda *_: SimpleNamespace(SuspendClock=fail_clock))
    audit = A.Audit(argparse.Namespace(output=tmp_path/'audit'))
    with pytest.raises(RuntimeError, match='clock unavailable'):
        audit.execute()
    failed = A.read(audit.out/'failed.json')
    assert failed['started_ns'] is None and failed['wall_seconds'] is None
    assert failed['saved_readout_attempts'] == failed['saved_readout_returns'] == 0


def test_late_failure_demotes_completed_receipt_and_preserves_original(tmp_path, monkeypatch):
    import argparse
    from types import SimpleNamespace
    class Clock:
        backend = 'synthetic'
        @staticmethod
        def now_ns():
            return 1000
    monkeypatch.setattr(A, 'load', lambda *_: SimpleNamespace(SuspendClock=Clock))
    args = argparse.Namespace(output=tmp_path/'audit', plan_sha256='p', receipt_sha256='w', terminal_sha256='t')
    audit = A.Audit(args)
    audit.plan, audit.rss = {'sources': {A.RUNNER: 'r'}}, 100
    audit.authenticate, audit.compute = lambda: None, lambda: {'agreement': True}
    def check():
        if (audit.out/'receipt.json').exists():
            raise RuntimeError('synthetic late output limit')
    audit.check = check
    with pytest.raises(RuntimeError, match='late output limit'):
        audit.execute()
    assert not (audit.out/'receipt.json').exists()
    assert A.read(audit.out/'invalid-completed-receipt.json')['status'] == 'completed'
    assert 'late output limit' in A.read(audit.out/'failed.json')['error']


def test_historical_relative_input_descriptors_resolve_under_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = 'output/previous/receipt.json'
    assert A.bound_path(relative) == A.ROOT/relative
    assert A.bound_path(str(tmp_path/'receipt.json')) == tmp_path/'receipt.json'
    with pytest.raises(ValueError, match='canonical'):
        A.bound_path('../outside/receipt.json')
