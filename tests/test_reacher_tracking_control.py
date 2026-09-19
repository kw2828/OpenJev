"""Small H2/steps3 planning fixtures, explicit engineering RNG seed410 only."""

import copy
import time

import mujoco
import numpy as np
import pytest
import torch

from openjev.research import reacher_geometry_physics as physics
from openjev.research import reacher_tracking_control as control
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_tracking_dynamics import nominal_model


@pytest.fixture
def nominal():
    return nominal_model()


def roots():
    qpos = np.array([[.1, -.2, .12, -.04]], np.float64)
    qvel = np.array([[.02, -.03, 0., 0.]], np.float64)
    return qpos, qvel, qpos[:, 2:].astype(np.float32)


def draws():
    rng = np.random.default_rng(410)
    arrays = [rng.normal(size=(1, n, 2, 2)) for n in (64, 192, 64, 64, 63)]
    return SearchInputs(arrays[0], arrays[1], tuple(arrays[2:]))


def bank(nominal, **kwargs):
    settings = {'allowed_gains': np.array([.7, 1., 1.3]), 'steps': 3,
                'noise_std': .05, 'planning_horizon': 2, 'action_block': 1}
    return control.GainPlanningBank(nominal, **{**settings, **kwargs})


def do_plan(planner, gain=1., step=0):
    return planner.plan(*roots(), gain, draws(), step=step, deadline=float('inf'))


@pytest.mark.parametrize('gain', [.7, 1., 1.3])
def test_exact_unchanged_adapter_parity_including_all_banks_and_selected_root(nominal, gain):
    planner = bank(nominal)
    specific = copy.copy(nominal)
    specific.actuator_gear[:, 0] *= gain
    reference = physics.PhysicsGeometryCEM(specific, steps=3, planning_horizon=2,
                                          action_block=1, noise_std=.05)
    qpos, qvel, target = roots()
    inputs = draws()
    got, journal, snapshot = planner.plan(qpos, qvel, target, gain, inputs,
                                         step=0, deadline=float('inf'))
    expected, native = reference.plan(qpos, qvel, target, inputs, step=0)
    for field in ('sequences', 'scores', 'selected_ids', 'selected_actions', 'selected_sequences'):
        np.testing.assert_array_equal(getattr(got, field), getattr(expected, field))
    assert got.candidate_evaluations == 256 and got.imagined_transitions == 512
    assert got.stages[-1].mean_candidate_id == 255
    assert journal['configuration'] == native['configuration']
    for actual, wanted in zip([*journal['banks'], journal['selected']],
                              [*native['banks'], native['selected']], strict=True):
        assert set(actual['arrays']) == set(wanted['arrays'])
        for key in actual['arrays']:
            np.testing.assert_array_equal(actual['arrays'][key], wanted['arrays'][key])
    np.testing.assert_array_equal(journal['selected']['arrays']['qpos'][0, 0, 0], qpos[0])
    selected_id = int(got.selected_ids[0])
    candidate = journal['banks'][selected_id // 64]['arrays']['qpos']
    np.testing.assert_array_equal(journal['selected']['arrays']['qpos'][0, 0, 1],
                                  candidate[0, selected_id % 64, 1])
    assert snapshot['last_operation']['gain'] == gain
    assert snapshot['costs']['plan_calls_completed'] == 1
    assert 'last_native_journal' not in snapshot
    totals = snapshot['costs']['aggregate_adapter_lifetime']
    assert totals['native_transitions_completed'] == 513
    assert totals['native_substeps_completed'] == 1026
    assert totals['geometry_calls_completed'] == 5


def test_gain_instances_are_reused_without_mutation_and_costs_aggregate(nominal):
    planner = bank(nominal)
    initial = planner.configuration()
    expected_original = nominal.actuator_gear.copy()
    old_result, _, _ = do_plan(planner, .7)
    do_plan(planner, 1.)
    do_plan(planner, 1.3)
    again, _, snapshot = do_plan(planner, .7)
    np.testing.assert_array_equal(old_result.scores, again.scores)
    np.testing.assert_array_equal(nominal.actuator_gear, expected_original)
    assert planner.configuration() == initial
    assert len(set(initial['member_model_binary_sha256'])) == 3
    assert [m['lifetime']['operations_completed'] for m in snapshot['members']] == [2, 1, 1]
    costs = snapshot['costs']
    assert costs['aggregate_adapter_lifetime']['operations_completed'] == 4
    assert costs['aggregate_adapter_lifetime']['native_transitions_completed'] == 4 * 513
    assert costs['temporary_gain_model_copies'] == costs['retained_adapter_model_copies'] == 3
    assert costs['native_data_instances'] == 3
    assert costs['setup_wall_seconds'] >= costs['nested_adapter_setup_seconds'] > 0
    assert costs['plan_call_wall_seconds'] >= costs['aggregate_adapter_lifetime']['operation_wall_seconds'] > 0


def test_terminal_shortening_keeps_paid_mean_and_selected_advance(nominal):
    planner = bank(nominal)
    result, journal, snapshot = do_plan(planner, step=2)
    assert result.horizon == 1 and result.imagined_transitions == 256
    assert result.candidate_evaluations == 256
    assert result.stages[-1].mean_candidate_id == 255
    assert journal['selected']['metadata']['native_transitions_completed'] == 1
    assert snapshot['costs']['aggregate_adapter_lifetime']['native_transitions_completed'] == 257
    with pytest.raises(ValueError, match='terminal'):
        do_plan(planner, step=3)
    assert not planner.snapshot()['failed']


def test_only_selected_member_snapshot_is_copied_during_plan(nominal, monkeypatch):
    planner = bank(nominal)
    for index in (0, 2):
        monkeypatch.setattr(planner._planners[index], 'snapshot',
                            lambda: pytest.fail('copied inactive member journal'))
    _, _, snapshot = do_plan(planner, 1.)
    assert snapshot['members'][1]['lifetime']['operations_completed'] == 1


def test_caller_inputs_original_model_and_snapshot_arrays_never_alias(nominal):
    gains = np.array([.7, 1., 1.3])
    planner = bank(nominal, allowed_gains=gains)
    gains[:] = 8
    qpos, qvel, target = roots()
    inputs = draws()
    before = [v.copy() for v in (qpos, qvel, target)]
    result, journal, snapshot = planner.plan(qpos, qvel, target, .7, inputs,
                                             step=0, deadline=float('inf'))
    for actual, original in zip((qpos, qvel, target), before, strict=True):
        np.testing.assert_array_equal(actual, original)
    assert planner.configuration()['allowed_gains'] == [.7, 1., 1.3]
    with pytest.raises(ValueError):
        planner._gains.flags.writeable = True
    journal['roots']['qpos'][:] = 999
    snapshot['members'][0]['lifetime']['operations_completed'] = 999
    other = planner.snapshot()
    assert not np.any(other['last_native_journal']['roots']['qpos'] == 999)
    assert other['members'][0]['lifetime']['operations_completed'] == 1
    other['last_native_journal']['selected']['arrays']['qpos'][:] = 888
    assert not np.any(planner.snapshot()['last_native_journal']['selected']['arrays']['qpos'] == 888)
    nominal.actuator_gear[:] = 7  # Caller model is independent after eager copies.
    again, _, _ = planner.plan(qpos, qvel, target, .7, inputs, step=0, deadline=float('inf'))
    np.testing.assert_array_equal(result.scores, again.scores)


def test_current_target_is_explicit_and_no_observer_state_is_carried(nominal):
    planner = bank(nominal)
    qpos, qvel, target = roots()
    original_q, original_v = qpos.copy(), qvel.copy()
    planner.plan(qpos, qvel, target, 1., draws(), step=0, deadline=float('inf'))
    qpos[:, 2:] = [-.08, .13]
    target = qpos[:, 2:].astype(np.float32)
    _, journal, _ = planner.plan(qpos, qvel, target, 1., draws(), step=1, deadline=float('inf'))
    np.testing.assert_array_equal(qpos[:, :2], original_q[:, :2])
    np.testing.assert_array_equal(qvel, original_v)
    np.testing.assert_array_equal(journal['roots']['public_target'], target)
    for stage in [*journal['banks'], journal['selected']]:
        np.testing.assert_array_equal(stage['arrays']['qpos'][0, 0, 0], qpos[0])
    assert planner.configuration()['estimator_updates'] == 0


@pytest.mark.parametrize('gain', [.8, np.nextafter(.7, 1), np.float32(.7), True, 0., np.nan, np.inf])
def test_gain_lookup_never_rounds_or_interpolates(nominal, gain):
    planner = bank(nominal)
    with pytest.raises(ValueError):
        do_plan(planner, gain)
    snapshot = planner.snapshot()
    assert snapshot['costs']['input_rejections'] == 1
    assert snapshot['costs']['aggregate_adapter_lifetime']['native_transitions_attempted'] == 0
    assert not snapshot['failed']


@pytest.mark.parametrize('kind', ['q_dtype', 'q_batch', 'v_shape', 'v_nonfinite',
                                  'target_dtype', 'target_mismatch', 'target_velocity', 'inputs'])
def test_bad_roots_reject_before_native_work(nominal, kind):
    planner = bank(nominal)
    q, v, target = roots()
    innovations = draws()
    if kind == 'q_dtype':
        q = q.astype(np.float32)
    elif kind == 'q_batch':
        q = np.repeat(q, 2, axis=0)
    elif kind == 'v_shape':
        v = v[0]
    elif kind == 'v_nonfinite':
        v[0, 0] = np.nan
    elif kind == 'target_dtype':
        target = target.astype(np.float64)
    elif kind == 'target_mismatch':
        target[0, 0] += .01
    elif kind == 'target_velocity':
        v[0, 2] = .1
    else:
        innovations = None
    with pytest.raises(ValueError):
        planner.plan(q, v, target, 1., innovations, step=0, deadline=float('inf'))
    assert not planner.snapshot()['failed']
    assert planner.snapshot()['costs']['aggregate_adapter_lifetime']['native_transitions_attempted'] == 0


@pytest.mark.parametrize('gains', [[], [[1.]], [1., 1.], [1., .7], [-1.], [np.nan], [np.inf], [True]])
def test_invalid_gain_banks(nominal, gains):
    with pytest.raises(ValueError):
        bank(nominal, allowed_gains=gains)


def test_refuses_live_environment_data_or_altered_nominal_model(nominal):
    for supplied in (object(), mujoco.MjData(nominal)):
        with pytest.raises(TypeError):
            bank(supplied)
    specific = copy.copy(nominal)
    specific.actuator_gear[:, 0] *= .7
    with pytest.raises(ValueError, match='nominal'):
        bank(specific)
    specific = copy.copy(nominal)
    specific.dof_damping[0] = 2
    with pytest.raises(ValueError, match='nominal'):
        bank(specific)


def test_expired_deadline_is_terminal_and_preserves_zero_native_failure(nominal):
    planner = bank(nominal)
    with pytest.raises(TimeoutError):
        planner.plan(*roots(), 1., draws(), step=0, deadline=time.monotonic() - 1)
    snapshot = planner.snapshot()
    assert snapshot['failed'] and snapshot['costs']['plan_calls_failed'] == 1
    assert snapshot['last_native_journal']['status'] == 'failed'
    assert snapshot['costs']['aggregate_adapter_lifetime']['native_transitions_attempted'] == 0
    with pytest.raises(RuntimeError, match='terminal'):
        do_plan(planner, .7)


def test_partial_native_failure_retains_exact_masks_and_cannot_switch_gain(nominal, monkeypatch):
    planner = bank(nominal)
    original_step = physics.mujoco.mj_step
    original_error = KeyboardInterrupt('synthetic fourth native substep failure')
    calls = 0

    def fail(model, data):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise original_error
        original_step(model, data)

    monkeypatch.setattr(physics.mujoco, 'mj_step', fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        do_plan(planner, .7)
    assert caught.value is original_error
    snapshot = planner.snapshot()
    totals = snapshot['costs']['aggregate_adapter_lifetime']
    assert totals['native_substeps_attempted'] == 4 and totals['native_substeps_completed'] == 3
    assert totals['native_transitions_completed'] == 1
    stage = snapshot['last_native_journal']['banks'][0]
    np.testing.assert_array_equal(stage['arrays']['native_substeps_completed'][0, 0], [2, 1])
    assert snapshot['last_operation']['gain'] == .7
    assert snapshot['last_operation']['error']['type'] == 'KeyboardInterrupt'
    assert snapshot['costs']['plan_calls_completed'] == 0
    with pytest.raises(RuntimeError, match='terminal'):
        do_plan(planner, 1.3)


def test_post_copy_deadline_charged_without_successful_bank_call(nominal, monkeypatch):
    planner = bank(nominal)

    def late(deadline):
        raise TimeoutError('synthetic outer copy deadline')

    monkeypatch.setattr(control, '_deadline', late)
    with pytest.raises(TimeoutError, match='outer copy'):
        do_plan(planner)
    snapshot = planner.snapshot()
    assert snapshot['costs']['plan_calls_completed'] == 0
    assert snapshot['costs']['plan_calls_failed'] == 1
    assert snapshot['costs']['aggregate_adapter_lifetime']['operations_completed'] == 1
    assert snapshot['last_native_journal']['status'] == 'completed'
    assert snapshot['last_operation']['status'] == 'failed'


def test_no_new_rng_or_global_rng_mutation(nominal, monkeypatch):
    innovations = draws()
    before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    monkeypatch.setattr(np.random, 'default_rng', lambda *a, **kw: pytest.fail('new RNG'))
    planner = bank(nominal)
    planner.plan(*roots(), 1., innovations, step=0, deadline=float('inf'))
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    after = np.random.get_state()
    np.testing.assert_array_equal(numpy_before[1], after[1])
    assert numpy_before[2:] == after[2:]
