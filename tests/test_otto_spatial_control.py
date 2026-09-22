"""Fabricated spatial-control contracts; no scientific heads or environments."""

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spatial_control_test", ROOT/"scripts/study_otto_spatial_control.py")
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)
KINDS = ("spatial", "neighbor_free", "cnn", "dense128", "statistics")
SEEDS = (10101, 10102, 10103)
ARMS = tuple(f"{kind}@{seed}" for seed in SEEDS for kind in KINDS) + ("analytic_inbounds",)
FIRST = {"lambda3": 16100001, "lambda4": 16200001, "lambda5": 16300001}
WEIGHTS = {regime: {1: .125, 2: .25, 3: .625} for regime in FIRST}


def identities():
    """Independent case-first, rotated-arm cohort oracle."""
    result = []
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(24):
            rotation = (ri*24+case) % 16
            for arm in ARMS[rotation:]+ARMS[:rotation]:
                result.append((regime, first+case, case//3, 1+case % 3, arm))
    return result


def rows(candidate=8, controls=10, analytic=10):
    result = []
    for regime, seed, block, hit, arm in identities():
        steps = candidate if arm.startswith("spatial@") else analytic if arm == "analytic_inbounds" else controls
        result.append({"regime": regime, "seed": seed, "block": block, "initial_hit": hit, "arm": arm,
                       "steps": steps, "found": True, "updates": steps, "blocked_steps": 0,
                       "final_update_assimilated": True, "init_seconds": .125,
                       "choose_seconds": steps/128, "update_seconds": steps/512,
                       "setup_allocation_seconds": .0625, "controller_seconds": .1875+steps/128+steps/512,
                       "environment_seconds": steps/256, "state_bytes": 22472})
    return result


def summarized(values):
    return S.aggregate(values, WEIGHTS)


def test_independent1152_cohort_has_all16_arms_and_eight_complete_stratified_blocks():
    actual = list(S.cohort())
    assert actual == identities() and len(actual) == 1152
    assert actual[0] == ("lambda3", 16100001, 0, 1, "spatial@10101")
    assert [r[-1] for r in actual[384:400]] == list(ARMS[8:]+ARMS[:8])
    for regime in FIRST:
        for arm in ARMS:
            for block in range(8):
                assert sorted(r[3] for r in actual if r[0] == regime and r[2] == block and r[4] == arm) == [1, 2, 3]


def test_all66_rules_cannot_admit_relative_gains_without_absolute_competence():
    result = summarized(rows())
    assert len(result["competence_checks"]) == 18
    assert len(result["improvement_checks"]) == 48
    assert len(result["control_competence_checks"]) == 72
    assert result["pilot_continuation"] is True
    weak = summarized(rows(candidate=100, controls=200, analytic=10))
    assert all(c["passes"] for c in weak["improvement_checks"])
    assert sum(c["passes"] for c in weak["competence_checks"]) == 9
    assert weak["pilot_continuation"] is False


@pytest.mark.parametrize("family", KINDS[1:])
def test_each_ordinary_control_can_independently_block_candidate_admission(family):
    values = rows()
    for row in values:
        if row["regime"] == "lambda5" and row["arm"].startswith(family+"@"):
            row["steps"] = row["updates"] = 7
    result = summarized(values)
    assert all(c["passes"] for c in result["competence_checks"])
    assert result["pilot_continuation"] is False
    assert sum(not c["passes"] for c in result["improvement_checks"]) == 2


def test_other_families_competence_is_published_without_becoming_candidate_gate():
    result = summarized(rows(candidate=8, controls=20, analytic=10))
    assert result["pilot_continuation"] is True
    assert len(result["control_competence_checks"]) == 72
    assert sum(c["passes"] for c in result["control_competence_checks"]) == 36
    assert result["learned_architecture_advantage_established"] is False


def test_relative_move_threshold_is_inclusive_and_cost_has_no_comparison_epsilon():
    assert summarized(rows(candidate=95, controls=100, analytic=100))["pilot_continuation"] is True
    result = summarized(rows(candidate=96, controls=100, analytic=100))
    assert sum(not c["passes"] for c in result["improvement_checks"]) == 12
    values = rows(candidate=95, controls=100, analytic=100)
    for row in values:
        cost = math.nextafter(1., math.inf) if row["arm"].startswith("spatial@") else 1.
        row.update(init_seconds=0., update_seconds=0., setup_allocation_seconds=0., choose_seconds=cost,
                   controller_seconds=cost)
    result = summarized(values)
    assert sum(not c["passes"] for c in result["improvement_checks"]) == 12
    assert result["pilot_continuation"] is False


@pytest.mark.parametrize(("positive", "passes"), [(6, True), (5, False)])
def test_block_gains_are_strict_and_require_six_of_eight(positive, passes):
    values = rows()
    for row in values:
        if row["arm"].startswith("spatial@"):
            row["steps"] = row["updates"] = 8 if row["block"] < positive else 10
    result = summarized(values)
    checks = [c for c in result["improvement_checks"] if c["name"].endswith("positive_blocks")]
    assert len(checks) == 12 and all(c["value"] == positive and c["passes"] is passes for c in checks)
    assert result["pilot_continuation"] is passes


def test_hit_weighting_then_equal_fit_seed_average_and_full_censor_denominator():
    values = rows()
    for row in values:
        if row["arm"] == "neighbor_free@10101":
            row["steps"] = row["updates"] = {1: 10, 2: 20, 3: 30}[row["initial_hit"]]
    result = summarized(values)
    for regime in FIRST:
        panel = result["regimes"][regime]
        assert panel["means"]["neighbor_free@10101"]["steps"] == 25.
        assert panel["family_means"]["neighbor_free"]["steps"] == 15.
        assert panel["raw_counts"]["neighbor_free@10101"] == {"found": 24, "episodes": 24}
    values = rows()
    row = next(r for r in values if r["regime"] == "lambda5" and r["initial_hit"] == 3
               and r["arm"] == "spatial@10103")
    row.update(found=False, steps=2188, updates=2188)
    result = summarized(values)
    assert result["regimes"]["lambda5"]["raw_counts"]["spatial@10103"]["found"] == 23
    assert result["regimes"]["lambda5"]["means"]["spatial@10103"]["found"] == 1-.625/8
    assert result["pilot_continuation"] is False


@pytest.mark.parametrize("defect", ["missing", "duplicate", "order", "nonfinite", "negative_cost", "unpaid_cost",
                                   "long", "bool_steps", "short_censor", "missing_update", "last_update", "blocked"])
def test_incomplete_or_invalid_episode_is_rejected(defect):
    values = rows()
    if defect == "missing":
        values.pop()
    elif defect == "duplicate":
        values[-1] = values[0]
    elif defect == "order":
        values[0], values[1] = values[1], values[0]
    else:
        key, value = {"nonfinite": ("controller_seconds", math.nan), "negative_cost": ("environment_seconds", -1.),
                      "unpaid_cost": ("controller_seconds", 99.), "long": ("steps", 2189), "bool_steps": ("steps", True),
                      "short_censor": ("found", False), "missing_update": ("updates", 0),
                      "last_update": ("final_update_assimilated", False), "blocked": ("blocked_steps", 1)}[defect]
        values[0][key] = value
    with pytest.raises(ValueError):
        summarized(values)


def test_untrusted_plan_is_rejected_before_json_decode(tmp_path, monkeypatch):
    path = tmp_path/"plan.json"
    path.write_text("not JSON")
    monkeypatch.setattr(S, "read", lambda *_: pytest.fail("unauthenticated plan decoded"))
    with pytest.raises(ValueError, match="plan"):
        S.authenticate(SimpleNamespace(plan=path, plan_sha256="0"*64))


def test_fixed_work_counts_limits_and_exact_payload_schemas():
    assert S.expected_calls('qualify') == {'model_load': 15, 'parity_restore': 15,
        'parity_numpy_forward': 780, 'parity_torch_forward': 780}
    assert S.expected_calls('study') == {'model_load': 15, 'native_reset': 1155,
        'native_step': 2520576, 'value_forward': 2363040, 'analytic_choose': 157536}
    common = {'started.json', 'runtime.json', 'inference-setup.json', 'work-contexts.jsonl',
              'work.jsonl', 'summary.json', 'serialization-projection.json'}
    assert S.payload_names('qualify') == common | {'preparation.json', 'qualification-arrays.npz',
        'qualification-tuples.jsonl', 'parity.jsonl'}
    assert S.payload_names('study') == common | {'native-setup.json', 'eval-transitions.jsonl',
        'eval-episodes.jsonl', 'evaluation.jsonl'}
    assert len(S.payload_names('qualify')) == len(S.payload_names('study')) == 11
    for mode, seconds, rss, output in [('qualify', 600, 4*1024**3, 256*1024**2),
                                      ('study', 21600, 8*1024**3, 12*1024**3)]:
        cap = S.limits(mode)
        assert (cap['native_seconds'], cap['rss_bytes'], cap['output_bytes']) == (seconds, rss, output)
        assert cap['optimizer_updates'] == cap['target_readouts'] == 0
        assert S.configuration(mode)['arms'] == list(ARMS)
        assert S.configuration(mode)['parity_atol'] == 1e-8
        assert S.configuration(mode)['parity_rtol'] == 1e-10
    assert S.audit_limits('study') == {'native_seconds': 21600, 'rss_bytes': 4*1024**3,
                                      'output_bytes': 1024**3}


def test_exact52_qualification_tuple_order_and_independent_dense_geometry():
    ids = S.qualification_ids()
    assert ids[:16] == [{'tuple_id': f'{split}:{i}', 'source': 'cache', 'split': split, 'row_index': i}
                        for split in ('train', 'valid') for i in range(8)]
    expected = [(regime, label, q, kind) for regime in FIRST
                for label, q in [('center', (26, 26)), ('lower', (0, 0)), ('upper', (52, 52))]
                for kind in ('asymmetric', 'point_successor', 'zero', 'subfloor')]
    assert len(ids) == 52 and len({r['tuple_id'] for r in ids}) == 52
    for row, (regime, label, q, kind) in zip(ids[16:], expected, strict=True):
        assert row == {'tuple_id': f'{regime}:{label}:{kind}', 'source': 'synthetic',
                       'regime': regime, 'position': list(q), 'belief_kind': kind}
        belief, position, sensing, legal = S.synthetic_tuple(regime, q, kind, np)
        assert belief.dtype == np.float64 and position.dtype == np.int64
        np.testing.assert_array_equal(position, q)
        assert sensing == float(regime[-1])
        allowed = [a for a, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)])
                   if 0 <= q[0]+dx < 53 and 0 <= q[1]+dy < 53]
        assert legal == allowed
        oracle = np.asarray([[0 if (x, y) == q else 1+(17*x+29*y+7*x*y) % 97
                              for y in range(53)] for x in range(53)], dtype=np.float64)
        oracle /= oracle.sum(dtype=np.float64)
        if kind == 'subfloor':
            oracle *= np.float64(1e-12)
            assert 0 < belief.sum() < 1e-10
        elif kind in ('zero', 'point_successor'):
            oracle.fill(0.)
            if kind == 'point_successor':
                successor = {'center': (25, 26), 'lower': (1, 0), 'upper': (51, 52)}[label]
                oracle[successor] = 1.
        np.testing.assert_array_equal(belief, oracle)
        assert belief[q] == 0.
    assert 2*52*16*105*105*8 == 146764800  # Shared branch tensors, not duplicated per head.


@pytest.mark.parametrize('channel', ['model_load', 'value_forward', 'parity_restore', 'parity_numpy_forward',
                                    'parity_torch_forward', 'native_step', 'native_reset', 'optimizer_update', 'target_readout'])
def test_all_capped_operations_stop_before_invocation_even_without_resource_check(tmp_path, channel):
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.plan = {'mode': 'qualify', 'limits': S.limits('qualify')}
    mapping = {'native_step': 'native_steps', 'native_reset': 'native_resets',
               'optimizer_update': 'optimizer_updates', 'target_readout': 'target_readouts'}
    cap = runner.plan['limits'][mapping.get(channel, channel)]
    runner.calls[channel] = {'attempted': cap, 'returned': cap, 'seconds': 0.}
    with pytest.raises(ValueError, match='before invocation|declared operation only'):
        runner.call(channel, lambda: pytest.fail('operation escaped allocation'), check=False)
    assert runner.calls[channel]['attempted'] == cap and runner.sequence == 0 and runner.pending == []


def test_spatial_adapter_preserves_branch_shape_successor_positions_sensing_and_signed_zero(tmp_path):
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.np = np
    z = np.zeros((16, 105, 105), np.float64)
    z[1, 52, 52] = 1e-12
    q = np.asarray([[0, 0], [52, 52], [26, 26], [25, 26]]*4, np.int64)
    snapshots = []
    class FakeHead:
        def normalized(self, centered, positions, sensing):
            snapshots.append((centered.copy(), positions.copy(), sensing.copy()))
            return np.arange(16, dtype=np.float64)-8.5
    for lam in (3., 4., 5.):
        result = runner.physical(FakeHead(), z, q, lam)
        np.testing.assert_array_equal(result, 64*(np.arange(16)-8.5))
        np.testing.assert_array_equal(snapshots[-1][0], z)
        np.testing.assert_array_equal(snapshots[-1][1], q)
        np.testing.assert_array_equal(snapshots[-1][2], np.full(16, lam, np.float64))
        assert result[0] == -544.  # Generic biased zero branch must not be forced to zero.


def test_amortization_pays_all15_fits_prep_fifteenths_and_load_once(tmp_path, monkeypatch):
    result = summarized(rows())
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    receipt = tmp_path/'prior'/'receipt.json'
    runner.plan = {'inputs': {'spatial_receipt': {'path': str(receipt)}}}
    runner.model_module_setup_seconds = 15.
    fit_costs = {arm: 12.+i for i, arm in enumerate(ARMS[:-1])}
    prior = {'training_costs': fit_costs, 'diagnostic_costs': {a: 999. for a in ARMS[:-1]},
             'preparation_seconds': 30.}
    monkeypatch.setattr(S, 'regular', lambda p: Path(p))
    monkeypatch.setattr(S, 'read', lambda p: {'wall_seconds': 200.} if Path(p).name == 'receipt.json' else prior)
    setup = {arm: {'seconds': 2.} for arm in ARMS[:-1]}
    result = runner.amortization(result, setup)
    for panel in result['scenarios'].values():
        assert set(panel) == set(ARMS)
        for arm, row in panel.items():
            if arm == 'analytic_inbounds':
                assert row['fit_seconds'] == row['preparation_share_seconds'] == row['deployment_setup_seconds'] == 0.
            else:
                assert row['fit_seconds'] == fit_costs[arm]
                assert row['preparation_share_seconds'] == 2. and row['deployment_setup_seconds'] == 3.
            assert row['seconds_per_search'] == {str(h): row['controller_without_deployment_seconds']+
                (row['fit_seconds']+row['preparation_share_seconds']+row['deployment_setup_seconds'])/h
                for h in (1, 100, 10000)}
    assert result['training_reference']['diagnostic_costs'] == prior['diagnostic_costs']


def test_cleanup_error_preserves_original_failure_and_still_publishes_receipt(tmp_path):
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.bind = lambda: None
    runner.plan = {'serialization_projection': {'scope': 'synthetic'}}
    class BadFlush:
        closed = False
        def flush(self):
            raise OSError('flush failed')
        def close(self):
            self.closed = True
    bad = BadFlush()
    runner.handles['fake'] = bad
    runner.pending = [{'id': 1, 'channel': 'parity_numpy_forward'}]
    def fail():
        raise ValueError('primary failure')
    runner.setup = fail
    with pytest.raises(ValueError, match='primary failure') as caught:
        runner.execute()
    assert any('flush failed' in note for note in caught.value.__notes__)
    failure = json.loads((runner.out/'failed.json').read_text())
    assert failure['status'] == 'failed' and failure['pending'] == runner.pending
    assert 'primary failure' in failure['error'] and bad.closed


def test_failed_readout_remains_attempted_pending_and_unreturned(tmp_path):
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.plan = {'mode': 'qualify', 'limits': S.limits('qualify')}
    journal = []
    runner.emit = lambda name, value: journal.append((name, value))
    runner.check = lambda: None
    def fail():
        raise ValueError('synthetic readout failed')
    with pytest.raises(ValueError, match='synthetic readout failed'):
        runner.call('parity_numpy_forward', fail)
    assert runner.calls['parity_numpy_forward'] == {'attempted': 1, 'returned': 0, 'seconds': 0.}
    assert len(runner.pending) == 1 and runner.pending[0]['channel'] == 'parity_numpy_forward'
    assert [value[1] for name, value in journal if name == 'work.jsonl'] == [0]


@pytest.mark.parametrize('found', [True, False])
@pytest.mark.parametrize('learned', [True, False])
def test_fake_two_step_episode_retains_terminal_and_censored_update_once(tmp_path, monkeypatch, found, learned):
    monkeypatch.setattr(S.P, 'HORIZON', 2)
    events, records = [], []
    class Environment:
        position = (26, 26)
        source = np.asarray([40, 40])
        draw_log = ()
        def step(self, action, quiet):
            assert action == 1 and quiet is True
            self.position = (self.position[0]+1, self.position[1])
            done = found and self.position[0] == 28
            return -2 if done else 0, 0., done
    env = Environment()
    class Actor:
        belief = np.ones((1, 1), np.float64)
        def __init__(self, packet, kernel, allow_stay):
            assert allow_stay is False
            self.public = packet
            self._policy = SimpleNamespace(_value_policy=lambda: (1, np.asarray([4., 1., 3., 2.])))
        def update(self, action, packet):
            events.append(('update', packet['step'], packet['done']))
            self.public = packet
        def storage_bytes(self):
            return {'mutable_array_bytes': 22472}
    branch = SimpleNamespace(raw_masses=np.zeros((4, 4)), weights=np.full((4, 4), 1e-10), eligible_actions=(0, 1, 2, 3))
    def scores(branch, callback, arithmetic):
        assert arithmetic == 'float64'
        callback(np.zeros((16, 1, 1)), np.zeros((16, 2), np.int64), object())
        return np.asarray([4., 1., 3., 2.])
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.np, runner.actor_class, runner.kernels = np, Actor, {'lambda3': object()}
    runner.environment = lambda *_: env
    runner.public = lambda env, step: {'position': env.position, 'hit': -2 if found and step == 2 else 0,
                                      'done': found and step == 2, 'step': step,
                                      'valid_actions': [] if found and step == 2 else [0, 1, 2, 3]}
    def witness(actor, env, packet):
        assert actor.public == packet
        return {'sha256': 'a'*64, 'mass': 1.}
    runner.witness = witness
    runner.branches = SimpleNamespace(rl_branches=lambda *_: branch, explicit_scores=scores,
                                      select_action=lambda costs, allowed: 1)
    runner.physical = lambda *_: np.ones(16, np.float64)
    def call(channel, operation, **_):
        events.append(('call', channel))
        result = operation()
        runner.last[channel] = {'seconds': 0.}
        return result
    runner.call, runner.check, runner.sync = call, lambda: None, lambda: None
    runner.emit = lambda name, row: records.append((name, copy.deepcopy(row)))
    head = SimpleNamespace(storage_bytes=lambda: {'parameter_array_bytes': 8}) if learned else None
    arm = 'spatial@10101' if learned else 'analytic_inbounds'
    result = runner.episode('lambda3', 16100001, 1, arm, 0, head)
    assert result['steps'] == result['updates'] == 2 and result['found'] is found
    assert result['final_update_assimilated'] is True and result['final_public']['step'] == 2
    expected_channel = 'value_forward' if learned else 'analytic_choose'
    assert [event for event in events if event[0] == 'call'] == [('call', expected_channel), ('call', 'native_step')]*2
    assert [event for event in events if event[0] == 'update'] == [('update', 1, False), ('update', 2, found)]
    assert [row['kind'] for name, row in records if name == 'eval-transitions.jsonl'] == ['reset', 'step', 'step']
    for name, row in records:
        if name == 'eval-transitions.jsonl' and row['kind'] == 'step':
            assert (row['values'] is None) is (not learned)
            if learned:
                assert len(row['values']) == 16 and np.asarray(row['raw_masses']).shape == (4, 4)


def test_fake_all_head_qualification_uses52_shared_tuples_and_one16row_call_per_route(tmp_path):
    """Exercise orchestration with fake readouts, never a learned model."""
    from contextlib import nullcontext

    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.out.mkdir()
    runner.np = np
    runner.plan = {'inputs': {'head_'+a.replace('@', '_'): {'sha256': 'a'*64} for a in ARMS[:-1]}}
    centered = np.broadcast_to(np.zeros((1, 1, 105, 105)), (52, 16, 105, 105))
    successors = np.broadcast_to(np.asarray([[25, 26]], np.int64), (52, 16, 2))
    sensing = np.asarray([3.]*16+[float(r[-1]) for r in FIRST for _ in range(12)], np.float64)
    weights = np.full((52, 4, 4), 1e-10, np.float64)
    prepared = {'centered_z': centered, 'successors': successors, 'sensing_length': sensing, 'weights': weights}
    records = [{**item, 'allowed_actions': [1, 3] if i % 2 else [0, 1, 2, 3],
                'array_sha256': {'centered_z': 'b'*64}} for i, item in enumerate(S.qualification_ids())]
    def outputs(z, q, lam):
        assert z.shape == (16, 105, 105) and q.shape == (16, 2) and lam.shape == (16,)
        return np.asarray([-4., -3., -2., -1.]*4, np.float64)
    class FakeReference:
        def __call__(self, z, q, lam):
            return SimpleNamespace(numpy=lambda: outputs(z, q, lam))
    runner.torch = SimpleNamespace(no_grad=nullcontext, from_numpy=lambda a: a)
    runner.restore_reference = lambda _: FakeReference()
    runner.heads = {a: SimpleNamespace(normalized=outputs) for a in ARMS[:-1]}
    def select(costs, allowed):
        minimum = min(costs[a] for a in allowed)
        return next(a for a in allowed if costs[a]-minimum < 1e-10)
    runner.branches = SimpleNamespace(select_action=select)
    calls, saved = [], []
    def call(channel, operation, **_):
        calls.append((channel, dict(runner.context)))
        return operation()
    runner.call, runner.sync = call, lambda: None
    runner.emit = lambda filename, row: saved.append((filename, copy.deepcopy(row)))
    runner.qualify((prepared, records))
    assert {channel: sum(c[0] == channel for c in calls) for channel in {c[0] for c in calls}} == {
        'parity_restore': 15, 'parity_numpy_forward': 780, 'parity_torch_forward': 780}
    assert len(saved) == 780 and runner.receipt['parity_passed'] is True
    assert [(r['fit_id'], r['tuple_index']) for _, r in saved] == [(a, i) for a in ARMS[:-1] for i in range(52)]
    for name, row in saved:
        assert name == 'parity.jsonl' and row['passed'] is True
        assert len(row['values_numpy']) == len(row['values_torch']) == 16
        assert row['values_numpy'][0] == -256. and row['action_numpy'] == row['allowed_actions'][0]
        assert row['costs_numpy'] == [1.-640.e-10]*4
        assert not {'centered_u', 'centered_z', 'beliefs', 'successors'} & row.keys()
        assert row['array_sha256'] == {'centered_z': 'b'*64}
    summary = json.loads((runner.out/'summary.json').read_text())
    assert summary['parity_records'] == 780 and summary['shared_tuples'] == 52


def test_complete_evaluate_allocates_module_over1080_and_each_head_over72(tmp_path, monkeypatch):
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.out.mkdir()
    runner.heads = {a: object() for a in ARMS[:-1]}
    runner.mixtures, runner.model_module_setup_seconds, runner.shared_setup_seconds = WEIGHTS, 1080., 99.
    by_id = {(r['regime'], r['seed'], r['block'], r['initial_hit'], r['arm']): r for r in rows()}
    def episode(regime, seed, hit, arm, block, head):
        assert (head is None) is (arm == 'analytic_inbounds')
        row = copy.deepcopy(by_id[regime, seed, block, hit, arm])
        row['controller_seconds'] -= row['setup_allocation_seconds']
        row['setup_allocation_seconds'] = 0.
        row.update(source_evaluation_only=[12, 23], draws_evaluation_only=[{'channel': 'hit', 'index': 0, 'uniform': .5}])
        return row
    runner.episode = episode
    runner.calls = {'native_step': {'returned': 0}}
    emitted = []
    runner.emit = lambda name, row: emitted.append((name, copy.deepcopy(row)))
    runner.amortization = lambda *_: {'scope': 'synthetic'}
    setup = {a: {'seconds': 72.} for a in ARMS[:-1]}
    result = runner.evaluate(setup)
    assert len(emitted) == 1152 and result['paired_source_cases'] == 72 and result['paired_uniforms'] == 72
    assert runner.receipt['completed_episodes'] == 1152
    for name, row in emitted:
        assert name == 'evaluation.jsonl'
        allocation = 0. if row['arm'] == 'analytic_inbounds' else 2.
        assert row['setup_allocation_seconds'] == allocation
        assert row['controller_seconds'] == row['init_seconds']+row['choose_seconds']+row['update_seconds']+allocation


def test_prospective_serialized_envelope_counts_actual_writer_bytes_and_duplicate_draws(tmp_path):
    fixtures = S.serialization_fixture()
    runner = S.Run(SimpleNamespace(output=tmp_path/'records'))
    runner.out.mkdir()
    # Use the real inherited compact writer, no logical operation or numeric inference.
    for name, record in fixtures.items():
        runner.emit(name+'.jsonl', record)
    for stream in runner.handles.values():
        stream.close()
    widths = {name: (runner.out/(name+'.jsonl')).stat().st_size for name in fixtures}
    projection = S.serialization_projection()
    assert projection['serialized_fixture_bytes'] == widths
    step_fields = {'kind', 'episode_id', 'step', 'action', 'costs', 'allowed_actions', 'raw_masses', 'weights',
        'values', 'public', 'posterior_before', 'posterior_after', 'choose_seconds', 'choose_instrumented_seconds',
        'choose_excluded_io_seconds', 'update_seconds', 'environment_seconds', 'native_p_end'}
    assert set(fixtures['step']) == step_fields
    assert len(fixtures['step']['values']) == 16 and np.asarray(fixtures['step']['raw_masses']).size == 16
    assert np.asarray(fixtures['step']['weights']).size == 16 and len(fixtures['step']['costs']) == 4
    maximal_episode = copy.deepcopy(fixtures['episode'])
    maximal_episode['draws_evaluation_only'] = [fixtures['draw']]*(2188+2)
    episode_bytes = len((json.dumps(maximal_episode, separators=(',', ':'), allow_nan=False)+'\n').encode())
    # Independent complete projection explicitly duplicates raw and allocated episode records.
    calls = 2*2520576+1155+15
    total = 2520576*widths['step']+1152*widths['reset']+calls*(widths['attempt']+widths['return'])
    total += (1152+15+3)*widths['context']+2*1152*episode_bytes+64*1024**2
    assert projection['projected_bytes'] >= total
    assert projection['projected_bytes'] == sum(projection['components'].values())
    assert projection['within_worker_cap'] is True and projection['projected_bytes'] < 12*1024**3
    assert projection['qualification_shared_array_bytes'] == sum([
        52*53*53*8, 52*2*8, 52*8, 52*4, 2*52*16*105*105*8, 2*52*4*4*8, 52*16*2*8])


def test_study_analytic_cap_is_enforced_before_an_extra_choice(tmp_path):
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.plan = {'mode': 'study', 'limits': S.limits('study')}
    runner.calls['analytic_choose'] = {'attempted': 157536, 'returned': 157536, 'seconds': 0.}
    with pytest.raises(ValueError, match='before invocation'):
        runner.call('analytic_choose', lambda: pytest.fail('extra analytic choice'), check=False)
    assert runner.pending == [] and runner.sequence == 0


def test_late_output_failure_demotes_completed_receipt_without_losing_it(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path/'run', supervision=tmp_path/'launch.json')
    runner = S.Run(args)
    runner.plan = {'mode': 'qualify', 'serialization_projection': {'scope': 'synthetic'}}
    runner.calls = {k: {'attempted': v, 'returned': v, 'seconds': 0.} for k, v in S.expected_calls('qualify').items()}
    runner.clock = SimpleNamespace(now_ns=lambda: 1_000_000_000, backend='synthetic')
    runner.start = 0
    runner.receipt['supervision_sha256'] = 'pinned'
    runner.bind = runner.setup = runner.sync = lambda: None
    runner.qualification_inputs = lambda: None
    runner.load_heads = dict
    runner.qualify = lambda _: None
    monkeypatch.setattr(S, 'payload_names', lambda _: {'serialization-projection.json'})
    monkeypatch.setattr(S, 'authenticate', lambda _: runner.plan)
    monkeypatch.setattr(S, 'sha', lambda _: 'pinned')
    def check():
        if (runner.out/'receipt.json').exists():
            raise ValueError('late output limit')
    runner.check = check
    with pytest.raises(ValueError, match='late output limit'):
        runner.execute()
    invalid = json.loads((runner.out/'invalid-completed-receipt.json').read_text())
    failed = json.loads((runner.out/'failed.json').read_text())
    assert invalid['status'] == 'completed' and failed['status'] == 'failed'
    assert not (runner.out/'receipt.json').exists() and 'late output limit' in failed['error']
