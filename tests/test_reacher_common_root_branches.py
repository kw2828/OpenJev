"""Handwritten arrays/native states only; no RNG or scientific artifact reads."""

import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from openjev.research import reacher_common_root_branches as branches
from openjev.research.reacher_tracking_dynamics import nominal_model


def union_inputs():
    initial = np.zeros((64, 24, 2), np.float32)
    initial[:, :, 0] = np.arange(64, dtype=np.float32)[:, None] / 100
    selected = [initial[4, :12].copy(), initial[7].copy(), np.full((12, 2), -.2, np.float32),
                np.full((24, 2), -.2, np.float32), np.full((24, 2), -.0, np.float32),
                np.full((12, 2), .9, np.float32)]
    selected[-1][-1] = [.7, -.4]
    return initial, selected


def test_union_keeps_all_slots_first_occurrence_and_exact_short_extension():
    initial, selected = union_inputs()
    result = branches.build_union(initial, tuple(selected))
    assert set(result) == {'slots', 'unique', 'slot_to_unique', 'first_slot'}
    assert result['slots'].shape == (70, 24, 2) and result['unique'].shape == (66, 24, 2)
    assert result['slot_to_unique'].dtype == result['first_slot'].dtype == np.int64
    assert result['slots'].dtype == result['unique'].dtype == np.float32
    np.testing.assert_array_equal(result['slot_to_unique'][-6:], [4, 7, 64, 64, 0, 65])
    np.testing.assert_array_equal(result['first_slot'], [*range(64), 66, 69])
    np.testing.assert_array_equal(result['slots'][69, :12], selected[-1])
    np.testing.assert_array_equal(result['slots'][69, 12:], np.tile(selected[-1][-1], (12, 1)))
    for slot in range(70):
        np.testing.assert_array_equal(result['slots'][slot], result['unique'][result['slot_to_unique'][slot]])
    assert not np.signbit(result['slots'][68]).any()
    assert np.signbit(selected[4]).all()  # Canonicalization does not mutate inputs.


def test_union_only_exact_deduplication_and_no_mutable_aliases():
    initial = np.zeros((64, 24, 2), np.float32)
    initial[1, 0, 0] = np.nextafter(np.float32(0), np.float32(1))
    chosen = [initial[0].copy() for _ in range(6)]
    result = branches.build_union(initial, chosen)
    assert len(result['unique']) == 2
    initial[:] = 1
    chosen[0][:] = 1
    assert result['slots'][0, 0, 0] == 0
    for value in result.values():
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True


@pytest.mark.parametrize('bad', ['initial_shape', 'initial_dtype', 'initial_nan', 'initial_bound',
                               'selected_count', 'selected_container', 'selected_shape', 'selected_dtype',
                               'selected_nan', 'selected_bound'])
def test_union_rejects_malformed_or_unnormalized_members(bad):
    initial, selected = union_inputs()
    if bad == 'initial_shape': initial = initial[:63]
    elif bad == 'initial_dtype': initial = initial.astype(np.float64)
    elif bad == 'initial_nan': initial[0, 0, 0] = np.nan
    elif bad == 'initial_bound': initial[0, 0, 0] = 1.01
    elif bad == 'selected_count': selected.pop()
    elif bad == 'selected_container': selected = (value for value in selected)
    elif bad == 'selected_shape': selected[0] = selected[0][:-1]
    elif bad == 'selected_dtype': selected[0] = selected[0].astype(np.float64)
    elif bad == 'selected_nan': selected[0][0, 0] = np.nan
    else: selected[0][0, 0] = -1.01
    with pytest.raises(ValueError):
        branches.build_union(initial, selected)


def state(model, data):
    value = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, value, mujoco.mjtState.mjSTATE_INTEGRATION)
    return value


@pytest.fixture
def native(tmp_path):
    model = nominal_model()  # Construction only. No reset, random draw or hidden plant.
    data = mujoco.MjData(model)
    data.qpos[:] = [.21, -.32, .12, -.08]
    data.qvel[:] = [.13, -.17, 0., 0.]
    data.time = 1.0000000000000007
    data.ctrl[:] = [.08, -.09]
    data.qfrc_applied[:] = [.12, -.09, 0., 0.]
    data.qacc_warmstart[:] = [.2, -.3, 0., 0.]
    mujoco.mj_forward(model, data)
    root = {'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(),
            'integration_state': state(model, data), 'time': float(data.time)}
    sequences = np.array([[[.03, -.04], [.04, .02], [-.02, .01]],
                          [[0., 0.], [.02, -.01], [.01, .03]]], np.float32)
    noise = np.array([[[.01, -.02], [0., .03], [-.04, 0.]],
                      [[-.03, .01], [.02, 0.], [0., -.01]]], np.float64)
    return model, {'root': root, 'true_gain': .7, 'sequences': sequences, 'noise': noise,
                   'out': tmp_path/'branches', 'deadline': float('inf')}


def load_data(folder, name='data.npz'):
    with np.load(folder/name, allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def manual(model, root, gain, commands, noise):
    model = copy.copy(model)
    model.actuator_gear[:, 0] = 200*gain
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, root['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    records = [(data.qpos.copy(), data.qvel.copy(), state(model, data), float(data.time))]
    costs = []
    for command, disturbance in zip(commands, noise, strict=True):
        applied = np.clip(command.astype(np.float64)+disturbance, -1, 1)
        data.ctrl[:] = applied
        mujoco.mj_step(model, data)
        mujoco.mj_step(model, data)
        mujoco.mj_rnePostConstraint(model, data)
        delta = data.body('fingertip').xpos-data.body('target').xpos
        distance, effort = float(np.linalg.norm(delta)), float(np.sum(applied**2))
        costs.append((distance, effort, -distance-effort))
        records.append((data.qpos.copy(), data.qvel.copy(), state(model, data), float(data.time)))
    return records, costs


def test_all_branches_restore_full_state_match_manual_native_and_bind_every_file(native):
    model, args = native
    model_before = branches._model_hash(model)
    root_before = copy.deepcopy(args['root'])
    command_before, noise_before = args['sequences'].copy(), args['noise'].copy()
    receipt = branches.run_branches(model, **args)
    saved = load_data(args['out'])
    assert set(saved) == {'commands', 'noise', 'applied', 'qpos', 'qvel', 'integration_states', 'time',
                         'distance', 'effort', 'rewards', 'initialized', 'native_completed', 'recorded',
                         'substeps_completed'}
    assert saved['qpos'].shape == (2, 2, 4, 4)
    assert saved['initialized'].all() and saved['native_completed'].all() and saved['recorded'].all()
    assert (saved['substeps_completed'] == 2).all()
    for b in range(2):
        for k in range(2):
            records, costs = manual(model, args['root'], .7, args['sequences'][k], args['noise'][b])
            for t, (q, v, integration, clock) in enumerate(records):
                np.testing.assert_array_equal(saved['qpos'][b, k, t], q)
                np.testing.assert_array_equal(saved['qvel'][b, k, t], v)
                np.testing.assert_array_equal(saved['integration_states'][b, k, t], integration)
                assert saved['time'][b, k, t] == clock
            np.testing.assert_array_equal(np.stack([saved[key][b, k] for key in ('distance', 'effort', 'rewards')], axis=1), costs)
    assert np.all(saved['time'][:, :, 0] == root_before['time'])
    assert saved['time'][0, 0, 0] != 50*.02
    assert branches._model_hash(model) == model_before
    np.testing.assert_array_equal(args['sequences'], command_before)
    np.testing.assert_array_equal(args['noise'], noise_before)
    for name, expected in root_before.items(): np.testing.assert_array_equal(args['root'][name], expected)
    counts = receipt['counters']
    assert counts['roots_initialized'] == counts['restore_calls_completed'] == 4
    assert counts['native_transitions_completed'] == counts['transitions_recorded'] == 12
    assert counts['native_substeps_completed'] == 24
    assert set(receipt['files']) == {'started.json', 'data.npz'}
    for name, digest in receipt['files'].items():
        assert hashlib.sha256((args['out']/name).read_bytes()).hexdigest() == digest
    assert json.loads((args['out']/'completed.json').read_text()) == receipt


def test_branch_candidate_reordering_is_independent_and_noise_shared(native):
    model, args = native
    branches.run_branches(model, **args)
    first = load_data(args['out'])
    args = {**args, 'out': args['out'].with_name('reversed'),
            'sequences': args['sequences'][::-1].copy(), 'noise': args['noise'][::-1].copy()}
    branches.run_branches(model, **args)
    second = load_data(args['out'])
    for key in ('applied', 'qpos', 'qvel', 'integration_states', 'time', 'distance', 'effort', 'rewards'):
        np.testing.assert_array_equal(first[key], second[key][::-1, ::-1])
    np.testing.assert_array_equal(first['applied'], np.clip(
        first['commands'][None].astype(np.float64)+first['noise'][:, None], -1, 1))


def test_gain_changes_physics_but_effort_is_applied_action_once(native):
    model, args = native
    args.update(sequences=np.array([[[.9, -.9]]], np.float32), noise=np.array([[[.4, -.4]]]))
    branches.run_branches(model, **args)
    low = load_data(args['out'])
    branches.run_branches(model, **{**args, 'true_gain': 1.3, 'out': args['out'].with_name('high')})
    high = load_data(args['out'].with_name('high'))
    np.testing.assert_array_equal(low['applied'], [[[[1., -1.]]]])
    assert low['effort'][0, 0, 0] == high['effort'][0, 0, 0] == 2.
    assert low['rewards'][0, 0, 0] == -low['distance'][0, 0, 0]-2.
    assert not np.array_equal(low['qvel'], high['qvel'])


def test_json_root_is_accepted_and_ignored_native_forces_would_change_continuation(native):
    model, args = native
    root = json.loads(json.dumps(branches._json(args['root'])))
    args.update(root=root, sequences=args['sequences'][:1, :1], noise=args['noise'][:1, :1])
    branches.run_branches(model, **args)
    actual = load_data(args['out'])
    # Qpos/qvel-only initialization drops the deliberately nonzero applied force.
    private = copy.copy(model)
    private.actuator_gear[:, 0] = 140.
    data = mujoco.MjData(private)
    data.qpos[:], data.qvel[:], data.time = root['qpos'], root['qvel'], root['time']
    mujoco.mj_forward(private, data)
    data.ctrl[:] = actual['applied'][0, 0, 0]
    mujoco.mj_step(private, data, nstep=2)
    assert not np.array_equal(data.qvel, actual['qvel'][0, 0, 1])


@pytest.mark.parametrize('field', ['qpos', 'qvel', 'time'])
def test_root_state_disagreement_is_terminal_before_any_native_transition(native, field):
    model, args = native
    if field == 'time': args['root'][field] += .01
    else: args['root'][field][0] += .01
    with pytest.raises(ValueError, match='Restored'):
        branches.run_branches(model, **args)
    partial = load_data(args['out'], 'partial.npz')
    assert not partial['initialized'].any() and not partial['native_completed'].any()
    failed = json.loads((args['out']/'failed.json').read_text())
    assert failed['phase'] == 'restore' and failed['counters']['native_substeps_attempted'] == 0
    assert not (args['out']/'completed.json').exists()


@pytest.mark.parametrize('bad', ['commands_dtype', 'commands_bound', 'commands_nan', 'commands_empty',
                               'noise_dtype', 'noise_shape', 'noise_nan', 'root_nan', 'root_keys',
                               'target_velocity', 'gain_zero', 'gain_bool', 'model_gain'])
def test_invalid_inputs_fail_before_attempt(native, bad):
    model, args = native
    if bad == 'commands_dtype': args['sequences'] = args['sequences'].astype(np.float64)
    elif bad == 'commands_bound': args['sequences'][0, 0, 0] = 1.1
    elif bad == 'commands_nan': args['sequences'][0, 0, 0] = np.nan
    elif bad == 'commands_empty': args['sequences'] = args['sequences'][:0]
    elif bad == 'noise_dtype': args['noise'] = args['noise'].astype(np.float32)
    elif bad == 'noise_shape': args['noise'] = args['noise'][:, :-1]
    elif bad == 'noise_nan': args['noise'][0, 0, 0] = np.nan
    elif bad == 'root_nan': args['root']['qpos'][0] = np.nan
    elif bad == 'root_keys': args['root']['future_gain'] = .7
    elif bad == 'target_velocity': args['root']['qvel'][2] = .1
    elif bad == 'gain_zero': args['true_gain'] = 0
    elif bad == 'gain_bool': args['true_gain'] = True
    else: model.actuator_gear[0, 0] = 140.
    with pytest.raises((ValueError, TypeError)):
        branches.run_branches(model, **args)
    assert not args['out'].exists()


def test_failed_second_substep_retains_actual_prefix_and_no_retry(native, monkeypatch):
    model, args = native
    original = KeyboardInterrupt('synthetic second substep')
    real_step = branches.mujoco.mj_step
    calls = [0]

    def fail(model, data):
        calls[0] += 1
        if calls[0] == 2: raise original
        return real_step(model, data)

    monkeypatch.setattr(branches.mujoco, 'mj_step', fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        branches.run_branches(model, **args)
    assert caught.value is original
    saved = load_data(args['out'], 'partial.npz')
    assert saved['initialized'][0, 0] and saved['substeps_completed'][0, 0, 0] == 1
    assert not saved['native_completed'].any() and not saved['recorded'].any()
    assert saved['failure_time'] == args['root']['time']+.01
    failed = json.loads((args['out']/'failed.json').read_text())
    assert failed['counters']['native_substeps_attempted'] == 2
    assert failed['counters']['native_substeps_completed'] == 1
    original_bytes = (args['out']/'failed.json').read_bytes()
    with pytest.raises(FileExistsError): branches.run_branches(model, **args)
    assert (args['out']/'failed.json').read_bytes() == original_bytes


def test_failure_after_native_return_separates_native_and_recorded_work(native, monkeypatch):
    model, args = native
    original = ArithmeticError('recording failed')
    real_record = branches._record

    def fail(arrays, model, data, b, k, t):
        if t: raise original
        return real_record(arrays, model, data, b, k, t)

    monkeypatch.setattr(branches, '_record', fail)
    with pytest.raises(ArithmeticError) as caught:
        branches.run_branches(model, **args)
    assert caught.value is original
    saved = load_data(args['out'], 'partial.npz')
    assert saved['native_completed'].sum() == 1 and saved['recorded'].sum() == 0
    failed = json.loads((args['out']/'failed.json').read_text())
    assert failed['counters']['native_transitions_completed'] == 1
    assert failed['counters']['transitions_recorded'] == 0


def test_cap_after_first_substep_preserves_work(native, monkeypatch):
    model, args = native
    clock, real_step = [0.], branches.mujoco.mj_step
    monkeypatch.setattr(branches.time, 'monotonic', lambda: clock[0])

    def slow(model, data):
        real_step(model, data)
        clock[0] = 2.

    monkeypatch.setattr(branches.mujoco, 'mj_step', slow)
    with pytest.raises(TimeoutError):
        branches.run_branches(model, **{**args, 'deadline': 1.})
    partial = load_data(args['out'], 'partial.npz')
    assert partial['substeps_completed'].sum() == 1
    assert not partial['native_completed'].any()


def test_late_completion_demotes_success_and_keeps_original_even_if_failure_write_fails(native, monkeypatch):
    model, args = native
    args.update(sequences=args['sequences'][:1, :1], noise=args['noise'][:1, :1])
    clock, real_write = [0.], branches._write
    monkeypatch.setattr(branches.time, 'monotonic', lambda: clock[0])

    def late(path, value):
        if Path(path).name == 'failed.json': raise OSError('secondary disk failure')
        result = real_write(path, value)
        if Path(path).name == 'completed.json': clock[0] = 2.
        return result

    monkeypatch.setattr(branches, '_write', late)
    with pytest.raises(TimeoutError) as caught:
        branches.run_branches(model, **{**args, 'deadline': 1.})
    assert not (args['out']/'completed.json').exists()
    assert (args['out']/'partial-completed.json').exists() and (args['out']/'partial.npz').exists()
    assert any('secondary disk failure' in note for note in caught.value.__notes__)


def test_expired_entry_has_no_attempt_or_native_steps(native, monkeypatch):
    model, args = native
    monkeypatch.setattr(branches.mujoco, 'mj_step', lambda *_: pytest.fail('unexpected native step'))
    with pytest.raises(TimeoutError):
        branches.run_branches(model, **{**args, 'deadline': -1.})
    assert not args['out'].exists()
