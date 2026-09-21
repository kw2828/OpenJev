"""Artificial scalar/branch, cohort, authentication and failure fixtures only."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit():
    spec = importlib.util.spec_from_file_location('fake_return_audit', ROOT/'scripts/audit_otto_return_value.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_archive(kind, baseline=.5):
    value = {'version': np.asarray('otto-return-value-v1'), 'kind': np.asarray(kind),
             'input_dim': np.asarray(11028), 'c0': np.asarray(baseline, np.float32),
             'first_weight': np.zeros((8, 11028), np.float32)}
    if kind != 'min8':
        value['final_weight'] = np.ones(8, np.float32)
    if kind == 'mlp8':
        value.update(hidden_bias=np.zeros(8, np.float32), output_bias=np.asarray(0., np.float32))
    return value


def sensor(value=.25):
    kernel = np.full((4, 107, 107), value, np.float64)
    kernel[:, 53, 53] = 0
    return kernel


def test_all_prefix_targets_uniform_and_pre_action_index(audit):
    np.testing.assert_array_equal(audit.targets(3, np), [3/64, 2/64, 1/64])
    long = audit.targets(129, np)
    assert len(long) == 129 and long[0] == 129/64 and long[-1] == 1/64
    combined = np.concatenate((audit.targets(1, np), audit.targets(3, np)))
    assert np.float32(combined.mean(dtype=np.float64)) == np.float32(7/256)
    assert combined.mean() != np.mean([1/64, 2/64])  # No future-duration episode weighting.
    for invalid in (True, 0, 2189):
        with pytest.raises(ValueError):
            audit.targets(invalid, np)


def test_centered_features_preserve_mass_and_position_at_corner(audit):
    belief = np.zeros((53, 53), np.float64)
    belief[52, 52] = .25
    x = audit.state_features(belief, [0, 52], 4., np)
    assert x.shape == (11028,) and x.dtype == np.float32
    assert x[104*105+52] == .25 and np.count_nonzero(x[:-3]) == 1
    np.testing.assert_array_equal(x[-3:], np.asarray([0, .25, .2], np.float32))
    np.testing.assert_array_equal(audit.state_features(belief*0, [0, 52], 4., np), np.zeros(11028, np.float32))


@pytest.mark.parametrize('kind', ['min8', 'mlp8', 'homogeneous8'])
def test_independent_scalar_network_closed_form_signed_and_not_clipped(audit, kind):
    archive = tensor_archive(kind)
    archive['first_weight'][:, 0] = np.arange(8, dtype=np.float32)-3
    if kind != 'min8':
        archive['final_weight'][:] = -1
    if kind == 'mlp8':
        archive['output_bias'][...] = 2
    weights = audit.checkpoint(archive, kind, .5, np)
    x = np.zeros((2, 11028), np.float64)
    x[0, 0] = .25
    result = audit.predict(x, weights, kind, np)
    residual = -.75 if kind == 'min8' else -2.5 + (2 if kind == 'mlp8' else 0)
    np.testing.assert_array_equal(result, [.125+residual, 2 if kind == 'mlp8' else 0])
    assert result[0] < 0
    assert all(v.dtype == np.float64 for v in weights.values())


@pytest.mark.parametrize('fault', ['extra', 'dtype', 'shape', 'nonfinite', 'c0', 'kind'])
def test_checkpoint_contract_rejects_faults(audit, fault):
    archive = tensor_archive('mlp8')
    if fault == 'extra':
        archive['optimizer'] = np.zeros(1)
    elif fault == 'dtype':
        archive['first_weight'] = archive['first_weight'].astype(np.float64)
    elif fault == 'shape':
        archive['hidden_bias'] = np.zeros(7, np.float32)
    elif fault == 'nonfinite':
        archive['output_bias'][...] = np.nan
    elif fault == 'c0':
        archive['c0'][...] = .6
    else:
        archive['kind'] = np.asarray('min8')
    with pytest.raises(ValueError):
        audit.checkpoint(archive, 'mlp8', .5, np)


def test_sixteen_branches_mass_before_padding_and_found_exclusion(audit):
    belief = np.zeros((53, 53), np.float64)
    belief[1, 0], belief[0, 1] = .25, .75
    x, raw, weight = audit.branches(belief, [0, 0], sensor(), 3., np)
    np.testing.assert_array_equal(raw[:, 0], [.25, .1875, .25, .0625])
    np.testing.assert_array_equal(raw, np.repeat(raw[:, :1], 4, axis=1))
    np.testing.assert_array_equal(weight, raw)
    np.testing.assert_array_equal(x[:, :11025].sum(axis=1), np.ones(16))
    np.testing.assert_array_equal(x[:4, -3:], np.tile([0, 0, .6], (4, 1)))
    assert x[4, -3] == 1/52 and x[12, -2] == 1/52
    # Constant remaining cost gives 1+nonfound mass*cost, including blocked stays.
    np.testing.assert_array_equal(audit.costs(np.full(16, 8., np.float64), weight, np), [9, 7, 9, 3])


def test_subfloor_zero_branch_and_biased_zero_input_not_suppressed(audit):
    belief = np.zeros((53, 53), np.float64)
    belief[52, 52] = 1e-20
    x, raw, weights = audit.branches(belief, [0, 0], sensor(), 5., np)
    np.testing.assert_array_equal(weights, np.full((4, 4), 1e-10))
    np.testing.assert_allclose(x[:, :11025].sum(axis=1), raw.ravel()/1e-10, rtol=1e-15)
    zero = np.zeros((53, 53), np.float64)
    x, raw, weights = audit.branches(zero, [0, 0], sensor(), 5., np)
    archive = tensor_archive('mlp8', 0.)
    archive['output_bias'][...] = 2
    v = 64*audit.predict(x, audit.checkpoint(archive, 'mlp8', 0., np), 'mlp8', np)
    np.testing.assert_array_equal(v, np.full(16, 128.))
    assert not raw.any() and (audit.costs(v, weights, np) > 1).all()


def test_float64_ties_and_eligible_subset_no_float32_cast(audit):
    costs = [1+2e-8, 1., -100., 3.]
    assert audit.choice(costs, [0, 1, 3], True, np) == 1
    assert audit.choice([1+5e-11, 1., 4., 4.], [0, 1], True, np) == 0
    assert audit.choice([None, 1., 1., None], [1, 2], False, np) == 1
    with pytest.raises(ValueError):
        audit.choice([0., 1., 2., 3.], [1, 0], True, np)


def rows(audit):
    result = []
    for regime, seed, hit, arm, block in audit.evaluation_order():
        family = arm.split('@')[0]
        steps = {'min8': 9, 'mlp8': 12, 'homogeneous8': 11, 'analytic_inbounds': 10}[family]+hit
        seconds = .5 if family == 'min8' else 1.
        result.append({'regime': regime, 'seed': seed, 'initial_hit': hit, 'arm': arm, 'block': block,
                       'steps': steps, 'found': True, 'init_seconds': 0., 'choose_seconds': seconds,
                       'update_seconds': 0., 'setup_allocation_seconds': 0., 'controller_seconds': seconds,
                       'environment_seconds': .1, 'state_bytes': 22472})
    return result


def test_complete_720_54_rules_and_mixture_not_row_micro(audit):
    data = rows(audit)
    mixtures = {regime: {1: .6, 2: .3, 3: .1} for regime in audit.REGIMES}
    result = audit.aggregate(data, mixtures)
    assert result['pilot_continuation'] is True
    assert [len(result[k]) for k in ('competence_checks', 'compression_checks', 'architecture_checks')] == [18, 12, 24]
    assert result['regimes']['lambda3']['means']['min8@10101']['steps'] == 10.5
    assert len({(r['regime'], r['seed']) for r in data}) == 72
    data[0]['found'] = False
    assert audit.aggregate(data, mixtures)['pilot_continuation'] is False
    with pytest.raises(ValueError, match='identities'):
        audit.aggregate(data[:-1], mixtures)


def test_inclusive_moves_but_strict_per_fit_cost(audit):
    data = rows(audit)
    mixtures = {regime: {1: .6, 2: .3, 3: .1} for regime in audit.REGIMES}
    for row in data:
        row['steps'] = 21 if row['arm'].startswith('min8') else 20
        row['controller_seconds'] = 1.
    result = audit.aggregate(data, mixtures)
    assert all(c['passes'] for c in result['competence_checks'])
    assert not any(c['passes'] for c in result['compression_checks'] if c['name'].endswith('every_cost'))


def test_bad_plan_pin_stops_before_producer_import_or_arrays(audit, tmp_path, monkeypatch):
    plan = tmp_path/'plan.json'
    plan.write_text('{}')
    args = SimpleNamespace(plan=plan, run=tmp_path/'run', terminal=tmp_path/'terminal', output=tmp_path/'out',
                           plan_sha256='bad', receipt_sha256='bad', terminal_sha256='bad')
    checker = audit.Audit(args)
    monkeypatch.setattr(checker, 'check', lambda: None)
    monkeypatch.setattr(audit.B, 'load', lambda *a: pytest.fail('untrusted producer import'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('untrusted prediction load'))
    with pytest.raises(ValueError, match='identity'):
        checker.authenticate()


def test_failed_auth_preserves_started_and_failure_without_compute(audit, tmp_path, monkeypatch):
    checker = audit.Audit(SimpleNamespace(output=tmp_path/'out'))
    monkeypatch.setattr(checker, 'authenticate', lambda: (_ for _ in ()).throw(ValueError('synthetic auth rejected')))
    monkeypatch.setattr(checker, 'compute', lambda *a: pytest.fail('compute after failed auth'), raising=False)
    with pytest.raises(ValueError, match='synthetic auth rejected'):
        checker.execute()
    assert (checker.out/'started.json').is_file()
    failed = json.loads((checker.out/'failed.json').read_text())
    assert failed['status'] == 'failed' and failed['agreement'] is False
    assert failed['saved_checkpoint_readout_calls'] == 0
    assert not (checker.out/'receipt.json').exists()


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind', ['min8', 'mlp8', 'homogeneous8'])
def test_synthetic_export_and_raw_feature_agreement(audit, kind):
    model = load_source('fake_scalar_model', 'src/openjev/research/otto_return_value.py')
    archive = tensor_archive(kind)
    rng = np.random.default_rng(818)
    for key in audit.tensor_names(kind):
        archive[key] = rng.uniform(-.1, .1, archive[key].shape).astype(np.float32)
    fields = np.zeros((4, 105, 105), np.float64)
    fields[:, 52, 51] = [.8, 1e-20, 0, .25]
    positions = np.asarray([[0, 52], [52, 0], [26, 26], [1, 49]], np.int64)
    x = audit.features(fields, positions, 4., np)
    np.testing.assert_array_equal(x, model.value_features(fields, positions, 4.))
    actual = audit.predict(x, audit.checkpoint(archive, kind, .5, np), kind, np)
    expected = model.FrozenValue(archive).normalized(x)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_source_declared_closure_and_summary_schema_match_independent(audit):
    producer = load_source('fake_return_runner', 'scripts/study_otto_return_value.py')
    assert audit.payload_names() == producer.payload_names()
    assert list(audit.evaluation_order()) == [(r, s, h, a, b) for r, s, b, h, a in producer.evaluation_order()]
    data = rows(audit)
    for row in data:
        row.update(updates=row['steps'], blocked_steps=0)
    mixtures = {regime: {1: .7, 2: .2, 3: .1} for regime in audit.REGIMES}
    expected = json.loads(json.dumps(producer.summary(data, mixtures)))
    for key, value in audit.aggregate(data, mixtures).items():
        audit.B.Comparisons().tree(expected[key], value, 'synthetic independent summary')


def fake_one_step(audit, tmp_path):
    checker = audit.Audit(SimpleNamespace(output=tmp_path/'out'))
    checker.np, checker.maximum_prediction_error = np, 0.
    checker.check = lambda: None
    checker.sources, checker.uniforms = {}, {}
    checker.kernels = {'lambda3': sensor()}
    arm, seed, episode = 'min8@10101', 11100001, 'eval:lambda3:11100001:min8@10101'
    checker.heads = {arm: audit.checkpoint(tensor_archive('min8', 0.), 'min8', 0., np)}
    public = audit.B.packet([26, 26], 1, False, 0)
    probability = audit.B.posterior(np.ones((53, 53), np.float64)/2808, public, sensor(), np)
    before = audit.A.posterior_record(probability)
    after = audit.B.packet([25, 26], -2, True, 1)
    final = np.zeros((53, 53), np.float64)
    final[25, 26] = 1
    _, raw, weight = audit.branches(probability, [26, 26], sensor(), 3., np)
    identity = {'episode_id': episode, 'stage': 'eval', 'regime': 'lambda3', 'seed': seed,
                'initial_hit': 1, 'arm': arm, 'block': 0}
    storage = {'immutable_array_bytes': 457960, 'mutable_array_bytes': 22472,
               'immutable_arrays': {'observation_kernel': 366368, 'manhattan_distance_table': 91592}}
    row = {**identity, 'steps': 1, 'found': True, 'updates': 1, 'blocked_steps': 0,
           'init_seconds': .05, 'choose_seconds': .1, 'update_seconds': .02, 'setup_allocation_seconds': 0.,
           'controller_seconds': .17, 'choose_instrumented_seconds': .11, 'choose_excluded_io_seconds': .01,
           'environment_seconds': .2, 'state_bytes': 22472,
           'storage': {'public_actor': storage, 'head': audit.head_storage('min8'),
                       'branch_workspace': '16*105*105 float64 centered u and z plus finite branch/features workspace, transient'},
           'zero_mass_decisions': 0, 'final_posterior_mass': 1., 'last256_lag2_matches': 0,
           'last256_lag2_pairs': 0, 'distinct_preaction_positions': 1,
           'source_evaluation_only': [25, 26], 'draws_evaluation_only': [
               {'channel': 'source', 'index': 0, 'selected_index': 25*53+26, 'uniform': .2, 'cdf_mass': 1.}],
           'final_public': after, 'final_update_assimilated': True}
    transition = {'kind': 'step', 'episode_id': episode, 'step': 1, 'action': 0, 'costs': [1.]*4,
                  'allowed_actions': [0, 1, 2, 3], 'raw_masses': raw.tolist(), 'weights': weight.tolist(),
                  'values': [0.]*16, 'public': after, 'posterior_before': before, 'posterior_after': audit.A.posterior_record(final),
                  'choose_seconds': .1, 'choose_instrumented_seconds': .11, 'choose_excluded_io_seconds': .01,
                  'update_seconds': .02, 'environment_seconds': .2, 'native_p_end': 1.}
    events = [{'kind': 'reset', **identity, 'public': public, 'posterior_after': before, 'source_evaluation_only': [25, 26]}, transition]
    context = {'phase': 'eval', 'episode': episode}
    work = [[1, 0, 'native_reset', 0, 0], [1, 1, 'native_reset', .01, .01, 0.],
            [2, 0, 'value_forward', 0, 1], [2, 1, 'value_forward', .05, .06, .01],
            [3, 0, 'native_step', 0, 1], [3, 1, 'native_step', .2, .2, 0.]]
    checker.work = audit.Work(iter(work), [context], checker.c)
    return checker, row, ('lambda3', seed, 1, arm, 0), events


def test_complete_artificial_episode_counts_filter_branch_and_cost(audit, tmp_path):
    checker, row, identity, events = fake_one_step(audit, tmp_path)
    assert checker.episode(row, identity, iter(events)) == row
    assert checker.receipt['saved_checkpoint_readout_calls'] == 1
    assert checker.receipt['saved_network_rows'] == 16
    assert checker.work.sequence == 3


@pytest.mark.parametrize('fault', ['mass', 'value', 'cost', 'action', 'posterior', 'timing', 'terminal'])
def test_artificial_episode_corruption_rejected(audit, tmp_path, fault):
    checker, row, identity, events = fake_one_step(audit, tmp_path)
    event = events[1]
    if fault == 'mass':
        event['raw_masses'][0][0] += .1
    elif fault == 'value':
        event['values'][3] = 1.
    elif fault == 'cost':
        event['costs'][2] = 5.
    elif fault == 'action':
        event['action'] = 1
    elif fault == 'posterior':
        event['posterior_after']['mass'] = .99
    elif fault == 'timing':
        event['choose_seconds'] = .001
    else:
        row['found'] = False
    with pytest.raises(ValueError):
        checker.episode(row, identity, iter(events))
