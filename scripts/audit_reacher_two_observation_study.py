"""Independent enclosing saved-output audit for the two-observation study.

Training is checked as saved tensor/log arithmetic, never numerical retraining.
Public-history buffers, geometry/CEM scores and native executed/nominal work are
independently reconstructed. No runner, model, trainer or optimizer is invoked.
Authentication of the shared experiment envelope supplies all stream authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from contextvars import ContextVar
from pathlib import Path

import audit_reacher_geometry_memory_study as memory
import numpy as np
import torch

from openjev.research import reacher_two_observation_audit as history_audit
from openjev.research import reacher_two_observation_protocol as protocol
from openjev.research import reacher_two_observation_results as results
from openjev.research import reacher_two_observation_streams as streams
from openjev.research import reacher_two_observation_training_audit as training_audit

inherited, previous = memory.inherited, memory.previous
base, search_audit = memory.base, memory.search_audit
require, read, write, sha = base.require, base.read, base.write, base.sha
array, finite, integer = memory.array, memory.finite, memory.integer
ROOT = Path(__file__).resolve().parents[1]
VERSION = protocol.STUDY
PANELS, PAIRS = protocol.PANELS, protocol.PAIRS
PHYSICS_REFERENCES, REFERENCES = memory.PHYSICS_REFERENCES, memory.REFERENCES
GEOMETRY_FIELDS, PHYSICS_COUNTS = memory.GEOMETRY_FIELDS, memory.PHYSICS_COUNTS
np_state_hash = previous.np_state_hash
tensor_hash, state_hash = training_audit.tensor_hash, training_audit.state_hash
audit_physics_trace, physics_configuration = memory.audit_physics_trace, memory.physics_configuration
audit_kinematic_observer, reference_timing = memory.audit_kinematic_observer, memory.reference_timing
_DEADLINE = ContextVar('two_observation_enclosing_audit_deadline', default=float('inf'))


def check_budget():
    if time.monotonic() > _DEADLINE.get():
        raise TimeoutError('Frozen two-observation audit cap exceeded')


def checked(path, digest):
    check_budget()
    value = search_audit.checked(Path(path), digest)
    check_budget()
    return value


def runtime():
    import gymnasium.envs.mujoco.reacher_v5 as native
    import mujoco
    folder = Path(native.__file__).parent
    return {'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'gymnasium', 'mujoco')},
        'native_source_sha256': sha(native.__file__), 'native_xml_sha256': sha(folder / 'assets/reacher.xml'),
        'mujoco_init_sha256': sha(mujoco.__file__)}


def bound_seed(plan, role):
    return streams.seed(plan['_settings'], role, contract=plan['_streams'])


def bound_schedule(plan, index, panel):
    return streams.schedule(plan['_settings'], index, panel, contract=plan['_streams'])


def geometry_configuration(noise):
    return {'version': 'reacher-geometry-reward-v1', 'input_order': ['cos_q0', 'cos_q1', 'sin_q0', 'sin_q1'],
        'projection': 'atan2(sin_q, cos_q), independently for each joint', 'minimum_pair_norm': 1e-6,
        'norm_boundary': 'accept >= threshold; reject below, no fallback', 'link_offsets_meters': [.10, .11],
        'geometry_source': 'reacher.xml body1.pos.x=0.10; fingertip.pos.x=0.11',
        'score': '-planar_fingertip_distance - expected_clipped_action_cost',
        'actuator_cost_function': 'openjev.research.reacher_reward_residual.expected_clipped_action_cost',
        'noise_std': float(noise), 'distance_weight': 1., 'control_weight': 1., 'issued_command_range': [-1., 1.],
        'joint_limit_penalty': None, 'reward_clipping': None,
        'tensor_contract': 'matching leading shapes, CPU, common float32 or float64 dtype', 'native_reward_exact': False,
        'limitation': 'Final-angle FK approximates RK4 cached-body distance; projected distance is not expected distance.',
        'run_status_authority': 'enclosing protocol and execution receipts'}


def kernel_plan(settings, bound):
    # A local view for frozen numerical kernels, never a mutation of a registry
    # or the exact settings mapping sealed by BoundStreams.
    return {**settings, '_settings': settings, '_streams': bound,
        'score_contract': {'geometry': geometry_configuration(settings['noise_std'])}}


def audit_innovations(plan, stem, step):
    values = base.load_npz(stem.with_suffix('.npz'))
    meta = read(stem.with_suffix('.json'))
    expected = streams.draw_control_inputs(plan['_settings'], step, contract=plan['_streams'])
    names = ('initial', 'random_extra', 'cem/1', 'cem/2', 'cem/3')
    require(set(values) == {name.replace('/', '_') for name in names}, 'All saved innovation arrays')
    shapes = {}
    for name, actual in zip(names, (expected.initial, expected.random_extra, *expected.cem), strict=True):
        saved = array(values[name.replace('/', '_')], actual.shape, np.float64, 'Innovation shape/dtype')
        require(np.array_equal(saved, actual), 'Saved innovations differ from authenticated named streams')
        shapes[name] = list(actual.shape)
    require(meta == {'prefix': f'planner/control/{step}', 'input_identities': dict(expected.identities()),
        'shapes': shapes, 'unused_anchor_draws': 7, 'full_horizon_innovations': True}, 'Exact innovation identity metadata')
    return expected


def audit_cohort(plan, records, panel):
    require(len(records) == plan["control_episodes"], "All fresh control cases")
    total, maximum = 0, 0.
    for index, record in enumerate(records):
        check_budget()
        meta = record["metadata"]
        require(meta["seed"] == bound_seed(plan, f"control/reset/{index}")
                and meta["noise_seed"] == bound_seed(plan, f"control/actuator_noise/{index}")
                and meta["noise_std"] == plan["noise_std"] and meta["policy_keys"] == ["packets", "commands"]
                and meta["sensor_schedule"] == bound_schedule(plan, index, panel).tolist(), "Fresh control reset/noise/public sensor binding")
        replay = search_audit.native.native_replay(record)
        require(replay["transitions"] == plan["steps"] and replay["new_policy_calls"] == 0 and replay["saved_output_only"] is True,
                "Native replay scope and terminal boundary")
        total += replay["transitions"]
        maximum = max(maximum, replay["max_abs_error"])
    return {"episodes": len(records), "transitions": total, "max_abs_error": maximum}




def audit_particle_observer(plan, records, estimates, saved):
    """Rebuild the public filter from named child streams, not private state.

    The native simulator is explicitly supplied prior knowledge. Resampling
    and disturbance draws replay old recorded estimator streams; no new policy
    or environment evaluation cohort is chosen by the auditor.
    """
    native, count = search_audit.native, plan['particles']
    mj, env = native.mujoco, native.make_env()
    transitions, maximum = 0, 0.
    try:
        model = copy.copy(env.unwrapped.model)
        require(float(model.opt.timestep) * 2 == plan['dt'], 'Particle nominal timestep')
        for case, record in enumerate(records):
            packets = np.asarray(record['policy']['packets'], np.float64)
            commands = record['policy']['commands']
            streams = np.random.SeedSequence(bound_seed(plan, f'planner/particle_filter/{case}')).spawn(3)
            initial_rng, process_rng, resample_rng = [np.random.default_rng(value) for value in streams]
            target, age = packets[0, 4:6].copy(), 0.
            particles = [mj.MjData(model) for _ in range(count)]
            for data in particles:
                data.qpos[:] = np.r_[np.arctan2(packets[0, 2:4], packets[0, :2]), target]
                data.qvel[:2] = initial_rng.uniform(-.005, .005, 2)
                data.qvel[2:] = 0.
                mj.mj_forward(model, data)
            for step in range(plan['steps']):
                check_budget()
                packet = packets[step]
                require(np.array_equal(packet[4:6], target), 'Static public particle target')
                if step:
                    require(np.isclose(packet[7], 0. if packet[6] else age + plan['dt'], atol=1e-6, rtol=1e-6), 'Public particle age')
                    disturbances = process_rng.normal(0., plan['noise_std'], (count, 2))
                    for data, noise in zip(particles, disturbances, strict=True):
                        check_budget()
                        data.ctrl[:] = np.clip(commands[step - 1].astype(float) + noise, -1., 1.)
                        mj.mj_step(model, data, nstep=2)
                        mj.mj_rnePostConstraint(model, data)
                        transitions += 1
                    if packet[6]:
                        measured = np.arctan2(packet[2:4], packet[:2])
                        residual = (np.stack([data.qpos[:2] for data in particles]) - measured + np.pi) % (2 * np.pi) - np.pi
                        squared = np.sum(residual ** 2, axis=1)
                        with np.errstate(over='ignore', under='ignore'):
                            weights = np.exp(-.5 * ((squared - squared.min()) / plan['filter_bandwidth']) / plan['filter_bandwidth'])
                        weights /= weights.sum()
                        points = (resample_rng.random() + np.arange(count)) / count
                        cumulative = np.cumsum(weights)
                        cumulative[-1] = 1.
                        indices = np.searchsorted(cumulative, points, side='right')
                        spec = mj.mjtState.mjSTATE_INTEGRATION
                        states = np.empty((count, mj.mj_stateSize(model, spec)))
                        for index, data in enumerate(particles):
                            mj.mj_getState(model, data, states[index], spec)
                        for data, index in zip(particles, indices, strict=True):
                            mj.mj_setState(model, data, states[index], spec)
                            data.qpos[:2] += (measured - data.qpos[:2] + np.pi) % (2 * np.pi) - np.pi
                            data.qpos[2:], data.qvel[2:] = target, 0.
                            mj.mj_forward(model, data)
                    age = float(packet[7])
                angles = np.stack([data.qpos[:2] for data in particles])
                position = angles.mean(axis=0)
                position[0] = np.arctan2(np.sin(angles[:, 0]).mean(), np.cos(angles[:, 0]).mean())
                velocity = np.mean([data.qvel[:2] for data in particles], axis=0)
                expected = np.r_[position, target, velocity, [0., 0.]]
                error = float(np.max(np.abs(estimates[case, step] - expected)))
                require(math.isfinite(error) and error <= 1e-10, 'Public-only particle state reconstruction')
                maximum = max(maximum, error)
        updates = len(records) * (plan['steps'] - 1)
        expected_counts = {'setup_forward_calls': len(records) * count,
            'nominal_transition_calls': updates * count, 'native_substeps': updates * count * 2,
            'measurement_reanchor_forward_calls': sum(int(np.asarray(record['metadata']['sensor_schedule'])[1:50].sum())
                                                      for record in records) * count}
        require(set(saved) == {'reference', 'public_inputs_only', 'updates', 'particles', 'estimates', 'counts', 'scope'}
                and saved['reference'] == 'particle' and saved['public_inputs_only'] is True
                and saved['updates'] == updates and saved['particles'] == count and saved['counts'] == expected_counts
                and np.array_equal(np.asarray(saved['estimates']), estimates[:, -1])
                and saved['scope'] == 'Detached estimate after final decision assimilation, before terminal packet; not a resumable particle/RNG state.',
                'Complete public particle observer receipt')
    finally:
        env.close()
    return {'observer_native_transitions_replayed': transitions, 'max_abs_error': maximum,
            'final_observer_root': plan['steps'] - 1, 'new_planner_calls': 0,
            'scope': 'Named estimator children and actual public packets/actions only; no realized environment disturbances.'}




def audit_reference_control(plan, folder, row, records, inputs):
    arm, n, steps = row['reference'], plan['control_episodes'], plan['steps']
    require(arm in REFERENCES, 'Declared matched reference')
    planned = arm in PHYSICS_REFERENCES
    times = reference_timing(plan, folder / 'timings.json', arm)
    values = base.load_npz(folder / 'planning.npz')
    require(set(values) == {'candidate_scores', 'planner_used'}
            | ({'root_estimates', 'public_packets', 'previous_commands'} if planned else set()), 'Reference planning artifact schema')
    scores = array(values['candidate_scores'], (n, steps, 256), np.float64, 'Reference CEM scores')
    used = array(values['planner_used'], (), np.bool_, 'Reference used flag')
    require(bool(used) is planned, 'Reference planned flag')
    commands, packets = base.stack(records, 'policy', 'commands'), base.stack(records, 'policy', 'packets')
    costs = -base.stack(records, 'audit', 'rewards').sum(1)
    observer, work, clipped, predictions, searches = None, [], 0, 0, []
    if planned:
        roots = array(values['root_estimates'], (n, steps, 8), np.float64, 'Reference original supplied roots')
        require(np.array_equal(array(values['public_packets'], (n, steps, 8), np.float32, 'Reference public packets'), packets[:, :steps]),
                'Only actual public packets enter observers/planner')
        prior_actions = np.concatenate((np.zeros((n, 1, 2), np.float32), commands[:, :-1]), 1)
        require(np.array_equal(array(values['previous_commands'], (n, steps, 2), np.float32, 'Previous issued commands'), prior_actions),
                'Public observer causal action alignment')
        if arm == 'known_state':
            native = np.concatenate((base.stack(records, 'audit', 'qpos')[:, :steps], base.stack(records, 'audit', 'qvel')[:, :steps]), -1)
            require(np.array_equal(roots, native), 'Privileged current root only')
        elif arm == 'public_kinematic':
            observer = audit_kinematic_observer(plan, records, roots, read(folder / 'observer-final.json'))
        else:
            observer = audit_particle_observer(plan, records, roots, read(folder / 'observer-final.json'))
        env = search_audit.native.make_env()
        try:
            model = copy.copy(env.unwrapped.model)
            final = read(folder / 'physics-final.json')
            require(set(final) == {'configuration', 'setup_seconds', 'failed', 'lifetime'}
                    and final['configuration'] == physics_configuration(plan, model) and final['failed'] is False,
                    'Complete successful nominal physics lifetime')
            require(finite(final['setup_seconds'], 'Physics setup', positive=True) <= times['setup_seconds'], 'Physics setup charged')
            for step in range(steps):
                check_budget()
                trace = search_audit.audit_trace({**plan, 'planners': ['cem256']}, inputs[step], folder / 'decisions' / f'{step:03d}', step=step)
                require(np.array_equal(trace['result'].selected_actions, commands[:, step])
                        and np.array_equal(trace['result'].scores, scores[:, step]), 'Reference actual action and all paid scores')
                work.append(audit_physics_trace(plan, model, folder / 'physics' / f'{step:03d}', trace,
                                               roots[:, step], packets[:, step, 4:6], step=step))
                require(trace['search_seconds'] <= times['decision_seconds'][step] + 1e-6, 'Nominal work charged to decision')
                searches.append(trace['search_seconds'])
                clipped += trace['clipped_predictions']
                predictions += trace['reward_predictions']
            lifetime = final['lifetime']
            require(set(lifetime) == set(PHYSICS_COUNTS) | {'operations_completed', 'operations_failed', 'operation_wall_seconds'},
                    'Complete nominal lifetime counters')
            for key in PHYSICS_COUNTS:
                integer(lifetime[key], sum(item['counts'][key] for item in work), 'Total nominal ' + key)
            integer(lifetime['operations_completed'], steps, 'All nominal operations')
            integer(lifetime['operations_failed'], 0, 'No failed operation hidden')
            require(math.isclose(lifetime['operation_wall_seconds'], sum(searches), rel_tol=1e-12, abs_tol=1e-9),
                    'All nominal operation time accounted')
        finally:
            env.close()
    else:
        require(np.all(scores == 0), 'Floor scores are placeholders, not predictions')
        uniform = np.random.default_rng(bound_seed(plan, 'floor/uniform/0')) if arm == 'uniform' else None
        for step in range(steps):
            wanted = np.zeros((n, 2), np.float32) if arm == 'zero' else uniform.uniform(-1, 1, (n, 2)).astype(np.float32)
            require(np.array_equal(commands[:, step], wanted), 'Exact floor command/no hidden search')
    decisions = np.asarray(times['decision_seconds'], float)
    return {'episode_costs': costs.tolist(), 'mean_cost': float(costs.mean()),
        'setup_seconds': times['setup_seconds'], 'decision_wall_seconds': float(decisions.sum()),
        'native_step_seconds': float(sum(times['native_step_seconds'])), 'row_wall_seconds': times['row_wall_seconds'],
        'decision_seconds': times['decision_seconds'], 'search_seconds': searches,
        'batch_latency_seconds': {'mean': float(decisions.mean()), 'p50': float(np.quantile(decisions, .5)),
            'p95': float(np.quantile(decisions, .95)), 'max': float(decisions.max())},
        'per_case_amortized_seconds': float(decisions.mean() / n), 'planner_used': planned,
        'candidate_evaluations': n * steps * 256 if planned else 0,
        'imagined_transitions': sum(item['candidate_transitions'] for item in work),
        'clipping_fraction': clipped / predictions if predictions else None,
        'public_observer': observer, 'physics_work': {
            'candidate_native_transitions_replayed': sum(item['candidate_transitions'] for item in work),
            'selected_native_transitions_replayed': sum(item['selected_transitions'] for item in work),
            'max_abs_error': max((item['max_abs_error'] for item in work), default=0.),
            'native_seconds': sum(item['native_seconds'] for item in work),
            'geometry_seconds': sum(item['geometry_seconds'] for item in work),
            'operation_wall_seconds': sum(item['operation_wall_seconds'] for item in work),
            'counts': {key: sum(item['counts'][key] for item in work) for key in PHYSICS_COUNTS}},
        'reference_limit': 'Native nominal supplied dynamics; candidate-budget matched, not wall-time matched or stochastic-optimal.'}


def history_identity(plan, fit):
    count = sum(math.prod(shape) for shape in training_audit.parameter_shapes(plan['hidden_size']).values())
    return {'version': 'reacher-two-observation-control-v1', 'kind': 'two_observation_gru',
        'model_class': 'TwoObservationHistoryGRUWorldModel', 'configuration': training_audit.model_configuration(plan),
        'width': plan['hidden_size'], 'weight_tensor_sha256': fit['student_tensor_sha256'],
        'parameter_count': count, 'parameter_tensor_bytes': 4 * count}


def history_scoring_configuration(plan):
    return {**previous.scoring_configuration(plan, 'geometry'), 'version': 'reacher-two-observation-control-v1',
        'scoring_kernel_version': 'reacher-geometry-control-v1', 'permitted_model_kinds': ['two_observation_gru'],
        'real_history': 'last two actual observations and intervening issued commands',
        'timing_scope': 'Controller validation, hooks, reconstruction, search, selected advance and snapshots; excludes caller native steps and artifact I/O.',
        'cost_limit': 'Equal parameters/search proposals do not match total compute; buffer validation/copying and instrumentation remain charged.'}


def history_work(plan, a, d):
    h = plan['hidden_size']
    names = {key: math.prod(shape) for key, shape in training_audit.parameter_shapes(h).items()}
    u, replay, transition = 12 * a, 11 * a, 11 * a + d
    return {'assimilate_samples': a, 'advance_samples': d, 'operations': {
        'configuration': training_audit.model_configuration(plan), 'trainable_parameters': sum(names.values()),
        'named_parameters': names, 'public_assimilate_samples': a, 'advance_samples': d,
        'replayed_observation_update_samples': u, 'replayed_transition_samples': replay,
        'gru_cell_sample_calls': u + transition, 'linear_layer_sample_calls': 4 * transition,
        'analytic_reward_sample_calls': transition,
        'dense_affine_macs': u * 3 * h * (h + 8) + transition * (5 * h * h + 25 * h),
        'startup_masked_work_is_counted': True, 'compute_matched': False,
        'counts_are_not_total_flops_or_measured_wall_time': True}, 'compute_matched': False}


def neural_counts(a, d):
    counts = {'observation_update': 12 * a, 'transition': 11 * a + d}
    counts.update({name: 11 * a + d for name in ('observation_head.0', 'observation_head.2', 'reward_head.0', 'reward_head.2')})
    return {name: {'attempted_sample_calls': value, 'completed_sample_calls': value} for name, value in counts.items()}


def history_state_shapes(plan):
    return {'hidden': (plan['hidden_size'],), 'packet': (8,), 'pending_action': (2,), 'imagined_depth': (1,),
        'real_packets': (12, 8), 'real_actions': (11, 2), 'real_present': (12,), 'real_indices': (12,),
        'real_index': (1,), 'real_target': (2,)}


def history_state_dtype(key):
    return np.bool_ if key == 'real_present' else (np.int64 if key in ('real_indices', 'real_index', 'imagined_depth') else np.float32)


def audit_history_scoring(plan, stem, root, carried, fit, trace, step, executed_angles, executed_rewards):
    values, meta = base.load_npz(stem.with_suffix('.npz')), read(stem.with_suffix('.json'))
    result, raw = trace['result'], trace['raw_rewards']
    commands = result.sequences
    n, k, h, _ = commands.shape
    bank_keys = {'commands', 'root_target', 'predicted_angles', 'learned_rewards', 'selected_rewards'} | {'geometry_' + key for key in GEOMETRY_FIELDS}
    selected_keys = {'selected_angles', 'selected_learned_reward', 'selected_reward'} | {'selected_geometry_' + key for key in GEOMETRY_FIELDS}
    require(k == 256 and set(values) == bank_keys | selected_keys | {'selected_ids', 'selected_actions'}, 'Complete history scoring arrays')
    counts = previous.audit_bank_arrays(plan, 'geometry', {key: values[key] for key in bank_keys}, commands, root['packet'][:, 4:6])
    require(np.array_equal(values['selected_rewards'], raw)
            and np.array_equal(result.scores, search_audit.clipped_returns(raw)), 'Exact history score/CEM binding')
    require(np.array_equal(array(values['selected_ids'], (n,), np.int64, 'Global selected IDs'), result.selected_ids)
            and np.array_equal(array(values['selected_actions'], (n, 2), np.float32, 'Selected first actions'), result.selected_actions),
            'History global earliest best selection')
    require(set(meta) == {'version', 'model', 'step', 'score_mode', 'configuration', 'root_sha256', 'callbacks', 'work', 'search_seconds', 'selected_advance'}
            and meta['version'] == 'reacher-two-observation-control-v1' and meta['model'] == history_identity(plan, fit)
            and meta['step'] == step and meta['score_mode'] == 'geometry'
            and meta['configuration'] == history_scoring_configuration(plan)
            and meta['root_sha256'] == np_state_hash(root) and len(meta['callbacks']) == 4
            and finite(meta['search_seconds'], 'Inner search time', positive=True) <= trace['search_seconds'] + 1e-6, 'History scorer source/class/temporal identity')
    callbacks = []
    for index, callback in enumerate(meta['callbacks']):
        require(callback['candidate_start'] == 64 * index and callback['candidate_stop'] == 64 * (index + 1), 'All paid callback intervals')
        part = {key: value if key == 'root_target' else value[:, index * 64:(index + 1) * 64]
                for key, value in values.items() if key in bank_keys}
        callbacks.append(previous.audit_bank_metadata(plan, 'geometry',
            {key: value for key, value in callback.items() if key not in ('candidate_start', 'candidate_stop')}, part, root))
    work = meta['work']
    expected_fields = {'candidate_evaluations', 'imagined_transitions', 'root_tensor_bytes', 'max_single_candidate_state_tensor_bytes',
        'model_work', 'neural_kernels', 'history_work', 'geometry_samples', 'geometry_seconds', 'model_advance_seconds',
        'diagnostic_array_bytes', 'tensor_bytes_are_not_peak_process_memory'}
    require(set(work) == expected_fields, 'Complete history work accounting')
    size, real_size, transitions = sum(x.nbytes for x in root.values()), sum(root[key].nbytes for key in history_audit.REAL_KEYS), n * k * h
    for key, value in {'candidate_evaluations': n * k, 'imagined_transitions': transitions,
        'root_tensor_bytes': size, 'max_single_candidate_state_tensor_bytes': 64 * size,
        'geometry_samples': transitions, 'diagnostic_array_bytes': sum(v.nbytes for key, v in values.items() if key not in selected_keys)}.items():
        integer(work[key], value, 'History search ' + key)
    require(work['model_work'] == history_work(plan, 0, transitions) and work['neural_kernels'] == neural_counts(0, transitions)
            and work['tensor_bytes_are_not_peak_process_memory'] is True, 'Full paid history neural operations')
    require(work['history_work'] == {'advance_state_schema_attempted_samples': transitions,
        'completed_advance_history_clone_tensor_bytes': transitions * (real_size // n),
        'completed_pending_action_clone_tensor_bytes': transitions * 8,
        'candidate_root_repeat_scheduled_tensor_bytes': size * 256,
        'limits': 'Payload accounting only. Failed-tail copies, temporaries, tensor objects and validators are not fully enumerated; all remain in measured wall time.'},
        'Candidate histories are paid private copies')
    for key in ('geometry_seconds', 'model_advance_seconds'):
        require(math.isclose(finite(work[key], key), sum(x[key] for x in callbacks), rel_tol=1e-12, abs_tol=1e-12), 'Summed history search ' + key)
    require(sum(x['bank_wall_seconds'] for x in meta['callbacks']) <= meta['search_seconds'] + 1e-6, 'Every callback charged')
    previous.audit_selected(plan, 'geometry', {key: values[key] for key in selected_keys}, meta['selected_advance'],
                            root, carried, trace, executed_angles, executed_rewards)
    selected = (np.arange(n), result.selected_ids, np.zeros(n, np.int64))
    for a, b in (('selected_angles', 'predicted_angles'), ('selected_learned_reward', 'learned_rewards'), ('selected_reward', 'selected_rewards')):
        require(np.allclose(values[a], values[b][selected], rtol=2e-5, atol=2e-6), 'Selected independent re-advance agrees with candidate prefix')
    return values, meta, counts


def audit_controller_metadata(plan, meta, scoring, root, carried, fit, step):
    n, imagined = plan['control_episodes'], scoring['work']['imagined_transitions']
    fields = {'version', 'step', 'model', 'root_sha256', 'carried_sha256', 'reconstruction_neural_kernels',
        'selected_neural_kernels', 'reconstruction_attempted_samples', 'reconstruction_completed_samples', 'model_work',
        'search', 'selected_advance', 'reconstruction_analytic_reward', 'root_snapshot_tensor_bytes', 'carried_snapshot_tensor_bytes',
        'committed_state_clone_tensor_bytes', 'real_boundary_schema_attempted_samples', 'selected_history_work', 'limits',
        'reconstruction_seconds', 'search_seconds', 'selected_advance_seconds', 'controller_seconds'}
    require(set(meta) == fields and meta['version'] == 'reacher-two-observation-control-v1'
            and meta['model'] == history_identity(plan, fit) and meta['step'] == step
            and meta['root_sha256'] == np_state_hash(root) and meta['carried_sha256'] == np_state_hash(carried), 'Complete controller decision identity')
    require(meta['reconstruction_neural_kernels'] == neural_counts(n, 0)
            and meta['selected_neural_kernels'] == neural_counts(0, n)
            and meta['model_work'] == history_work(plan, n, imagined + n)
            and meta['search'] == scoring['work'] and meta['selected_advance'] == scoring['selected_advance'], 'Full reconstruction12/11 and selected work')
    for key in ('reconstruction_attempted_samples', 'reconstruction_completed_samples', 'real_boundary_schema_attempted_samples'):
        integer(meta[key], n, key)
    for key, state in (('root_snapshot_tensor_bytes', root), ('carried_snapshot_tensor_bytes', carried), ('committed_state_clone_tensor_bytes', carried)):
        integer(meta[key], sum(x.nbytes for x in state.values()), key)
    require(meta['reconstruction_analytic_reward'] == {'completed_samples_lower_bound': 11 * n,
        'completed_samples_upper_bound': 11 * n, 'exact': True}, 'All padded reward cost work retained')
    require(meta['selected_history_work'] == {'advance_state_schema_attempted_samples': n,
        'completed_history_clone_tensor_bytes': sum(root[key].nbytes for key in history_audit.REAL_KEYS),
        'completed_pending_action_clone_tensor_bytes': 8 * n}, 'Selected real-history copies charged')
    require(meta['limits'] == history_scoring_configuration(plan)['timing_scope'], 'Controller timing scope')
    for key in ('reconstruction_seconds', 'search_seconds', 'selected_advance_seconds', 'controller_seconds'):
        finite(meta[key], key, positive=True)
    require(meta['search_seconds'] >= scoring['search_seconds'] and sum(meta[key] for key in
        ('reconstruction_seconds', 'search_seconds', 'selected_advance_seconds')) <= meta['controller_seconds'] + 1e-6, 'All controller phases charged')
    require(sum(scoring['selected_advance'][key] for key in ('model_advance_seconds', 'geometry_seconds'))
            <= meta['selected_advance_seconds'] + 1e-6, 'Selected native heads and geometry charged')


def history_row_members():
    names = {'started.json', 'episodes.npz', 'episodes.json', 'states.npz', 'state-work.json',
        'executed_predictions.npz', 'inputs.json', 'timings.json'}
    names |= {f'{folder}/{step:03d}.{suffix}' for folder in ('decisions', 'scoring') for step in range(50) for suffix in ('npz', 'json')}
    names |= {f'controller-decisions/{step:03d}.json' for step in range(50)}
    return names


def history_timing(plan, folder, panel):
    n, steps = plan['control_episodes'], plan['steps']
    done, times = read(folder / 'completed.json'), read(folder / 'timings.json')
    require(set(done) == {'version', 'status', 'panel', 'cases', 'model_decisions', 'native_steps_per_case',
        'public_packets_per_case', 'row_wall_seconds', 'wall_excludes', 'files'}
        and done['version'] == 'reacher-two-observation-episode-v1' and done['status'] == 'completed'
        and done['panel'] == panel and done['cases'] == n and done['model_decisions'] == steps
        and done['native_steps_per_case'] == [steps] * n and done['public_packets_per_case'] == steps + 1
        and done['wall_excludes'] == 'Only the final completed.json receipt write; all payload, cleanup and hashing included.'
        and set(done['files']) == history_row_members(), 'Complete terminal50/51 history row')
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} == history_row_members() | {'completed.json'}, 'Exact history row file boundary')
    for name, digest in done['files'].items():
        checked(folder / name, digest)
    fields = {'version', 'setup_seconds', 'decision_seconds', 'controller_seconds', 'input_load_hash_seconds',
        'decision_trace_write_seconds', 'native_step_and_record_seconds', 'native_step_seconds', 'native_cases_per_step',
        'finalize_seconds', 'cleanup_seconds', 'row_payload_wall_seconds', 'observation_assimilations', 'executed_action_advances', 'scope'}
    require(set(times) == fields and times['version'] == 'reacher-two-observation-episode-v1'
        and times['native_cases_per_step'] == [n] * steps and times['native_step_seconds'] == times['native_step_and_record_seconds']
        and times['observation_assimilations'] == times['executed_action_advances'] == n * steps
        and times['scope'] == 'Decision includes input loading, controller and trace writes; native includes record copying/validation; sub-times overlap, do not add them again.', 'History timing schema')
    lists = ('decision_seconds', 'controller_seconds', 'input_load_hash_seconds', 'decision_trace_write_seconds', 'native_step_seconds')
    for key in lists:
        require(isinstance(times[key], list) and len(times[key]) == steps, 'All history per-step costs')
        for value in times[key]:
            finite(value, key, positive=True)
    for step in range(steps):
        require(sum(times[key][step] for key in ('controller_seconds', 'input_load_hash_seconds', 'decision_trace_write_seconds'))
            <= times['decision_seconds'][step] + 1e-6, 'Input authentication/controller/storage charged')
    for key in ('setup_seconds', 'finalize_seconds', 'cleanup_seconds', 'row_payload_wall_seconds'):
        finite(times[key], key, positive=True)
    require(sum(times[key] for key in ('setup_seconds', 'finalize_seconds', 'cleanup_seconds'))
        + sum(times['decision_seconds']) + sum(times['native_step_seconds']) <= times['row_payload_wall_seconds'] + 1e-6
        and times['row_payload_wall_seconds'] <= finite(done['row_wall_seconds'], 'Whole history row', positive=True), 'Payload, closure and hashes charged')
    return {**times, 'row_wall_seconds': done['row_wall_seconds']}


def audit_history_control(plan, execution, folder, row, records, inputs, fit):
    n, steps = plan['control_episodes'], plan['steps']
    times = history_timing(plan, folder, row['panel'])
    identity = history_identity(plan, fit)
    states = base.load_npz(folder / 'states.npz')
    schema = history_state_shapes(plan)
    require(set(states) == {phase + '__' + key for phase in ('root', 'carried') for key in schema}, 'Complete root/carried history state arrays')
    for phase in ('root', 'carried'):
        for key, shape in schema.items():
            array(states[phase + '__' + key], (n, steps, *shape), history_state_dtype(key), 'History state ' + key)
    packets, commands = base.stack(records, 'policy', 'packets'), base.stack(records, 'policy', 'commands')
    native_rewards = base.stack(records, 'audit', 'rewards')
    predictions = base.load_npz(folder / 'executed_predictions.npz', {'angles', 'rewards'})
    angle = array(predictions['angles'], (n, steps, 4), np.float32, 'Executed history native-head angles')
    reward = array(predictions['rewards'], (n, steps), np.float32, 'Executed original learned rewards')
    state_work, input_receipt, started = read(folder / 'state-work.json'), read(folder / 'inputs.json'), read(folder / 'started.json')
    require(set(state_work) == {'version', 'model', 'steps', 'state_arrays_bytes', 'aggregate_model_work',
        'auxiliary_teacher_calls', 'reset_calls', 'neural_verification'}
        and state_work['version'] == 'reacher-two-observation-episode-v1' and state_work['model'] == identity
        and len(state_work['steps']) == steps and state_work['state_arrays_bytes'] == sum(v.nbytes for v in states.values())
        and state_work['auxiliary_teacher_calls'] == state_work['reset_calls'] == 0
        and state_work['neural_verification'] == 'Recorded source-bound outputs, not independent neural replay', 'History state/work receipt')
    require(set(input_receipt) == {'version', 'steps', 'scope'} and input_receipt['version'] == state_work['version']
        and len(input_receipt['steps']) == steps
        and input_receipt['scope'] == 'Exact external input files must accompany any independently auditable release.', 'Complete external innovation receipt')
    require(set(started) == {'version', 'status', 'panel', 'model', 'cases', 'steps', 'input_stems', 'case_bindings', 'seed_allocation'}
        and started['version'] == state_work['version'] and started['status'] == 'started' and started['panel'] == row['panel']
        and started['model'] == identity and started['cases'] == n and started['steps'] == steps
        and started['seed_allocation'] == 'none; explicit caller-supplied cases'
        and len(started['input_stems']) == steps
        and started['case_bindings'] == [{'reset_seed': bound_seed(plan, f'control/reset/{i}'),
            'noise_seed': bound_seed(plan, f'control/actuator_noise/{i}'),
            'sensor_schedule': bound_schedule(plan, i, row['panel']).tolist()} for i in range(n)], 'History row explicit cases/public scope')
    work = {key: 0 for key in ('candidate_evaluations', 'imagined_transitions', 'geometry_samples', 'selected_geometry_samples')}
    work.update({key: 0. for key in ('geometry_seconds', 'model_advance_seconds', 'selected_geometry_seconds', 'selected_model_advance_seconds')})
    searches, history_checks, clipped, count = [], [], 0, 0
    for step in range(steps):
        check_budget()
        stem = execution / 'innovations' / 'control' / f'{step:03d}'
        saved_input = input_receipt['steps'][step]
        require(set(saved_input) == {'step', 'stem', 'files_sha256', 'input_identities'}
            and saved_input['step'] == step and saved_input['stem'] == started['input_stems'][step]
            and Path(saved_input['stem']).name == f'{step:03d}'
            and saved_input['files_sha256'] == {suffix: sha(stem.with_suffix(suffix)) for suffix in ('.npz', '.json')}
            and saved_input['input_identities'] == dict(inputs[step].identities()), 'Byte-bound relocated input identity')
        trace = search_audit.audit_trace({**plan, 'planners': ['cem256']}, inputs[step], folder / 'decisions' / f'{step:03d}', step=step)
        require(np.array_equal(commands[:, step], trace['result'].selected_actions), 'History selected command actually issued')
        root = {key: states['root__' + key][:, step] for key in schema}
        carried = {key: states['carried__' + key][:, step] for key in schema}
        history_checks.append(history_audit.audit_decision(public_packets=packets[:, :step+1].copy(),
            issued_commands=commands[:, :step].copy(), step=step, root_state=root, carried_state=carried,
            selected_issued_command=commands[:, step].copy(), recorded_identity=state_work['model'], expected_identity=identity))
        _values, meta, counts = audit_history_scoring(plan, folder / 'scoring' / f'{step:03d}', root, carried, fit,
            trace, step, angle[:, step], reward[:, step])
        controller = read(folder / 'controller-decisions' / f'{step:03d}.json')
        require(controller == state_work['steps'][step] and controller['controller_seconds'] == times['controller_seconds'][step]
            and controller['search_seconds'] == trace['search_seconds'], 'Controller state/timing cross-file identity')
        audit_controller_metadata(plan, controller, meta, root, carried, fit, step)
        searches.append(trace['search_seconds'])
        for key in ('candidate_evaluations', 'imagined_transitions', 'geometry_samples'):
            work[key] += counts[key]
        for key in ('geometry_seconds', 'model_advance_seconds'):
            work[key] += meta['work'][key]
        for key in ('geometry_samples', 'geometry_seconds', 'model_advance_seconds'):
            work['selected_' + key] += meta['selected_advance'][key]
        clipped += trace['clipped_predictions']
        count += trace['reward_predictions']
    require(state_work['aggregate_model_work'] == history_work(plan, n * steps, n * steps + work['imagined_transitions']), 'All real reconstruction and deployment work')
    costs, decisions = -native_rewards.sum(1), np.asarray(times['decision_seconds'], float)
    return {'episode_costs': costs.tolist(), 'mean_cost': float(costs.mean()), 'setup_seconds': times['setup_seconds'],
        'decision_wall_seconds': float(decisions.sum()), 'native_step_seconds': float(sum(times['native_step_seconds'])),
        'row_wall_seconds': times['row_wall_seconds'], 'decision_seconds': times['decision_seconds'], 'search_seconds': searches,
        'per_case_amortized_seconds': float(decisions.mean() / n), 'candidate_evaluations': work['candidate_evaluations'],
        'imagined_transitions': work['imagined_transitions'], 'clipping_fraction': clipped / count, 'planner_used': True,
        'state_and_work': state_work, 'history_audit': {'decisions': len(history_checks), 'neural_outputs_verified': False},
        'scoring_work': work}


def new_training_configuration(settings):
    return {**inherited.training_configuration(settings), 'adapter_version': 'reacher-two-observation-training-v1',
        'model_class': 'TwoObservationHistoryGRUWorldModel',
        'mlp_width_usage': 'unused; retained original paired settings metadata'}


def fit_provenance(plan, expected, row):
    source = plan['lineage']['cache']
    paths = [row['initialization_path'], row['order_path'], 'train.npz', 'train.json']
    return {'study': VERSION, 'plan_sha256': expected, 'name': row['name'], 'pair': row['pair'],
        'cache_plan_sha256': source['plan_sha256'], 'cache_audit_receipt_sha256': source['audit_receipt_sha256'],
        'original_members': {name: source['members'][name] for name in paths},
        'initial_state': 'original paired GRU tensors, not inherited fitted weights'}


def audit_training_inputs(plan, context, execution):
    parent, settings = context.cache_plan, context.settings
    original_settings = inherited.training_configuration(parent)
    require(original_settings == inherited.training_configuration(settings), 'New arm uses the complete original training recipe')
    records = base.load_records(execution / 'inherited' / 'train', parent['train_episodes'])
    public = inherited.public_learning_tensors(records)
    digest, initials, orders, bindings = tensor_hash(public), {}, {}, {}
    original_seeds = parent['random_stream_contract']['torch_registry']
    for pair in PAIRS:
        check_budget()
        for folder, values in (('initializations', initials), ('orders', orders)):
            path = execution / 'inherited' / folder / f'{pair}.pt'
            payload = inherited.load_tensors(path)
            inherited.unseal(payload)
            require(read(path.with_suffix('.json')) == {'pair': pair, 'file_sha256': sha(path),
                'integrity_sha256': payload['integrity_sha256']}, 'Original paired sidecar')
            values[pair] = payload
        initial, order = initials[pair], orders[pair]
        roles = {family: f'fit/{family}/{pair}' for family in ('gru', 'mlp')}
        require(initial['version'] == inherited.TRAINING_VERSION and initial['settings'] == original_settings
            and initial['roles'] == roles and initial['seeds'] == {key: original_seeds[value] for key, value in roles.items()}
            and set(initial['states']) == set(initial['tensor_hashes']) == {'gru', 'mlp'}, 'Paired original initialization semantics')
        for family, arm in (('gru', 'residual_gru'), ('mlp', 'cached_mlp')):
            inherited.tensors(initial['states'][family], inherited.shapes(parent, arm), 'Original ' + family + ' tensor schema')
            require(tensor_hash(initial['states'][family]) == initial['tensor_hashes'][family], 'Original tensor identity')
        require(order['version'] == inherited.TRAINING_VERSION and order['role'] == f'fit/minibatch/{pair}'
            and order['seed'] == original_seeds[order['role']], 'Original order role and seed')
        # Historical replay authenticates every supplied permutation and both
        # endpoint generator states, without creating any scientific stream.
        training_audit.replay_orders(order, new_training_configuration(settings), deadline_check=check_budget)
        bindings[pair] = {'initial_file_sha256': sha(execution / 'inherited' / 'initializations' / f'{pair}.pt'),
            'order_file_sha256': sha(execution / 'inherited' / 'orders' / f'{pair}.pt'),
            'initial_tensor_sha256': initial['tensor_hashes']['gru'], 'initialization_integrity_sha256': initial['integrity_sha256'],
            'orders_integrity_sha256': order['integrity_sha256'], 'orders_sha256': state_hash(order)}
    receipt = read(execution / 'training-inputs.json')
    require(set(receipt) == {'data_sha256', 'pairs', 'settings', 'new_initializations', 'new_orders', 'wall_seconds', 'scope'}
        and receipt['data_sha256'] == digest and receipt['pairs'] == bindings and receipt['settings'] == original_settings
        and receipt['new_initializations'] == receipt['new_orders'] == 0
        and receipt['scope'] == 'Authenticated copied original tensors/orders; no initialization or permutation generation.', 'Complete external training input receipt')
    finite(receipt['wall_seconds'], 'Training input authentication', positive=True)
    return public, initials, orders, receipt


def audit_new_fit(plan, expected, execution, row, public, initial, order):
    folder = execution / 'fits' / row['name']
    done = read(folder / 'completed.json')
    fields = {'version', 'status', 'name', 'pair', 'arm', 'model_configuration', 'settings', 'provenance', 'source_sha256', 'runtime',
        'updates', 'optimizer_steps', 'flushed_updates', 'cursor', 'initialization_sha256', 'orders_sha256', 'orders_hash_scope',
        'data_sha256', 'student_tensor_sha256', 'checkpoint_integrity_sha256', 'log_chain_sha256', 'parameters',
        'constructor_seconds', 'trainer_setup_seconds', 'training_wall_seconds', 'update_call_seconds', 'log_validation_flush_seconds',
        'payload_export_write_seconds', 'file_hash_seconds', 'wall_seconds', 'files', 'timing_scope', 'scope'}
    require(set(done) == fields and done['version'] == 'reacher-two-observation-fitting-v1' and done['status'] == 'completed'
        and all(done[key] == row[key] for key in ('name', 'pair', 'arm')), 'Complete fresh fit receipt schema')
    require(set(done['files']) == {'initial-weights.pt', 'training.jsonl', 'weights.pt', 'checkpoint.pt'}, 'All new fit payloads')
    for name, digest in done['files'].items():
        checked(folder / name, digest)
    cfg = new_training_configuration(plan['settings'])
    checkpoint = inherited.load_tensors(folder / 'checkpoint.pt')
    actual_initial, weights = inherited.load_tensors(folder / 'initial-weights.pt'), inherited.load_tensors(folder / 'weights.pt')
    initial_hash, order_hash, data_hash = tensor_hash(initial['states']['gru']), state_hash(order), tensor_hash(public)
    require(tensor_hash(actual_initial) == initial_hash, 'Fresh actual initial tensors equal original pair, not fitted parent')
    logs = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
    provenance = fit_provenance(plan, expected, row)
    summary = training_audit.audit_training(checkpoint, logs, initial_weights=initial['states']['gru'], orders=order,
        public_data=public, settings=cfg, expected_initial_sha256=initial_hash, expected_orders_sha256=order_hash,
        expected_data_sha256=data_hash, expected_checkpoint_sha256=done['checkpoint_integrity_sha256'],
        provenance=provenance, source_sha256=plan['sources'], runtime=plan['runtime'], final_weights=weights, deadline_check=check_budget)
    count = summary['updates']
    require(done['settings'] == cfg and done['model_configuration'] == training_audit.model_configuration(cfg)
        and done['provenance'] == provenance and done['source_sha256'] == plan['sources'] and done['runtime'] == plan['runtime']
        and done['updates'] == done['optimizer_steps'] == done['flushed_updates'] == count
        and done['cursor'] == {'epoch': cfg['epochs'], 'batch': 0}
        and done['initialization_sha256'] == initial_hash and done['orders_sha256'] == order_hash
        and done['data_sha256'] == data_hash and done['student_tensor_sha256'] == tensor_hash(weights)
        and done['log_chain_sha256'] == summary['log_chain_sha256'] and done['parameters'] == summary['parameters'], 'Fresh fitting/training arithmetic binding')
    require(done['orders_hash_scope'] == 'canonical_state_hash of entire sealed payload including integrity field'
        and done['timing_scope'] == 'Full fit through file hashes, excluding only completed.json write; nested timing scopes are not additive. Cap also checked after terminal write.'
        and done['scope'] == 'One externally authenticated fresh fit; not a whole-study audit or authorization.', 'New fit identity/timing scope')
    for key in ('constructor_seconds', 'trainer_setup_seconds', 'training_wall_seconds', 'update_call_seconds',
        'log_validation_flush_seconds', 'payload_export_write_seconds', 'file_hash_seconds', 'wall_seconds'):
        finite(done[key], key)
    require(done['trainer_setup_seconds'] == summary['setup_wall_seconds']
        and done['training_wall_seconds'] == summary['training_wall_seconds']
        and done['trainer_setup_seconds'] <= done['constructor_seconds'] + 1e-6
        and done['training_wall_seconds'] <= done['update_call_seconds'] + 1e-6
        and sum(done[key] for key in ('constructor_seconds', 'update_call_seconds', 'log_validation_flush_seconds',
            'payload_export_write_seconds', 'file_hash_seconds')) <= done['wall_seconds'] + 1e-6, 'All fitting costs charged once in whole fit wall')
    return {'arm': row['arm'], 'pair': row['pair'], 'parameters': summary['parameters'],
        'student_tensor_sha256': tensor_hash(weights), 'model_configuration': done['model_configuration'],
        'new_optimizer_updates': count, 'inherited_optimizer_updates': 0, 'wall_seconds': done['wall_seconds'],
        'saved_training_audit': summary, 'fitting_receipt': done}


def audit_fits_and_snapshots(plan, expected, context, execution):
    public, initials, orders, input_receipt = audit_training_inputs(plan, context, execution)
    settings, parent = context.settings, context.cache_plan
    fits = {}
    for row in protocol.fit_manifest(settings):
        check_budget()
        if row['new_fit']:
            fit = audit_new_fit(plan, expected, execution, row, public, initials[row['pair']], orders[row['pair']])
        else:
            fit = inherited.audit_fit(parent, execution / 'inherited', {**row, 'initialization_family': 'gru'},
                initials[row['pair']], orders[row['pair']], tensor_hash(public), public['packets'])
            fit.update(new_optimizer_updates=0, inherited_optimizer_updates=parent['epochs'] * math.ceil(parent['train_episodes'] / parent['batch_size']))
        fits[row['name']] = fit
    final = read(execution / 'final-models.json')
    require(set(final) == {'student_tensor_sha256', 'files', 'evaluation_optimizer_updates', 'all_models_restored_sha256'}
        and final['evaluation_optimizer_updates'] == 0 and set(final['student_tensor_sha256']) == set(fits)
        and set(final['files']) == {f'model-states/{name}-after.pt' for name in fits}
        and final['all_models_restored_sha256'] == sha(execution / 'all-models-restored.json'), 'All nine unchanged final deployment snapshots')
    restored = {}
    for stage, receipt_name in (('prefit', 'inherited-models-restored.json'), ('before', 'all-models-restored.json')):
        receipt = read(execution / receipt_name)
        rows = [row for row in protocol.fit_manifest(settings) if stage == 'before' or not row['new_fit']]
        require(set(receipt) == {'plan_sha256', 'stage', 'models', 'optimizer_constructed', 'restoration_optimizer_updates',
            'constructor_rng_isolated_and_weights_overwritten', 'constructor_seeds', 'wall_seconds', 'unix_time'}
            and receipt['plan_sha256'] == expected and receipt['stage'] == stage and receipt['optimizer_constructed'] is False
            and receipt['restoration_optimizer_updates'] == 0 and receipt['constructor_rng_isolated_and_weights_overwritten'] is True
            and receipt['constructor_seeds'] == {'inherited': 0, 'history': 410}
            and set(receipt['models']) == {row['name'] for row in rows}, 'Complete phase-bound actual-class restoration')
        finite(receipt['wall_seconds'], 'Restoration wall', positive=True)
        finite(receipt['unix_time'], 'Restoration timestamp', positive=True)
        for row in rows:
            name, fit = row['name'], fits[row['name']]
            folder = execution / ('fits' if row['new_fit'] else 'inherited/fits') / name
            weights = inherited.load_tensors(folder / 'weights.pt')
            schema = training_audit.parameter_shapes(settings['hidden_size'])
            path = execution / 'model-states' / f'{name}-{stage}.pt'
            snapshot = inherited.load_tensors(path)
            inherited.tensors(snapshot, schema, 'Actual class tensor schema')
            require(tensor_hash(snapshot) == fit['student_tensor_sha256']
                and all(torch.equal(snapshot[key], weights[key]) for key in weights), 'Exact restored deployment tensor values')
            wanted = {'kind': row['arm'], 'model_class': row['model_class'], 'configuration': fit['model_configuration'],
                'checkpoint_path': (folder / 'checkpoint.pt').relative_to(execution).as_posix(),
                'checkpoint_sha256': sha(folder / 'checkpoint.pt'), 'weights_sha256': sha(folder / 'weights.pt'),
                'student_tensor_sha256': fit['student_tensor_sha256'],
                'successful_updates': fit['new_optimizer_updates'] + fit['inherited_optimizer_updates'],
                'snapshot_path': path.relative_to(execution).as_posix(), 'snapshot_sha256': sha(path)}
            require(receipt['models'][name] == wanted, 'Exact restored receipt member')
            if stage == 'before':
                after_path = execution / 'model-states' / f'{name}-after.pt'
                checked(after_path, final['files'][after_path.relative_to(execution).as_posix()])
                after = inherited.load_tensors(after_path)
                inherited.tensors(after, schema, 'Final observed tensor schema')
                require(final['student_tensor_sha256'][name] == tensor_hash(after) == fit['student_tensor_sha256']
                    and all(torch.equal(after[key], snapshot[key]) for key in snapshot), 'Before/after deployment tensors unchanged')
        restored[stage] = receipt
    return fits, restored, input_receipt


def expected_members(settings):
    names = {'started.json', 'random-streams.json', 'inheritance.json', 'training-inputs.json', 'inherited-models-restored.json',
        'training-started.json', 'all-fits-completed.json', 'all-models-restored.json', 'evaluation-started.json',
        'control-completed.json', 'final-models.json', 'costs.json', 'results.json'}
    original = {'train.npz', 'train.json'}
    for pair in PAIRS:
        original |= {f'{folder}/{pair}.{suffix}' for folder in ('initializations', 'orders') for suffix in ('pt', 'json')}
        original |= {f'fits/{arm}-{pair}/{name}' for arm in ('residual_gru', 'cached_gru')
                     for name in ('initial-weights.pt', 'training.jsonl', 'weights.pt', 'checkpoint.pt', 'completed.json')}
    require(len(original) == 44, 'Exactly44 inherited payloads')
    names |= {'inherited/' + name for name in original}
    names |= {f'inherited/lineage/{kind}-{suffix}.json' for kind in ('cache', 'prerequisite')
        for suffix in (('plan', 'audit', 'completed', 'summary') if kind == 'cache' else ('plan', 'audit', 'completed', 'summary', 'terminal'))}
    for pair in PAIRS:
        for arm in ('residual_gru', 'cached_gru', 'two_observation_gru'):
            name = f'{arm}-{pair}'
            names |= {f'model-states/{name}-{phase}.pt' for phase in ('before', 'after')}
            if arm != 'two_observation_gru':
                names.add(f'model-states/{name}-prefit.pt')
            else:
                names |= {f'fits/{name}/{member}' for member in ('initial-weights.pt', 'training.jsonl', 'weights.pt', 'checkpoint.pt', 'completed.json')}
    names |= {f'innovations/control/{step:03d}.{suffix}' for step in range(50) for suffix in ('npz', 'json')}
    for panel in PANELS:
        for label in [f'{arm}-{pair}' for pair in PAIRS for arm in ('residual_gru', 'two_observation_gru', 'cached_gru')] + list(REFERENCES):
            path = f'control/{panel}/{label}'
            if label.startswith('two_observation_gru-'):
                local = history_row_members() | {'completed.json'}
            else:
                local = {'episodes.npz', 'episodes.json', 'timings.json'}
                if label in REFERENCES:
                    local.add('planning.npz')
                    folders = ('decisions', 'physics') if label in PHYSICS_REFERENCES else ()
                    if label in PHYSICS_REFERENCES:
                        local.add('physics-final.json')
                    if label in ('particle', 'public_kinematic'):
                        local.add('observer-final.json')
                else:
                    local |= {'states.npz', 'state-work.json', 'executed_predictions.npz'}
                    folders = ('decisions', 'scoring')
                local |= {f'{folder}/{step:03d}.{suffix}' for folder in folders for step in range(50) for suffix in ('npz', 'json')}
            names |= {f'{path}/{name}' for name in local}
    return names


def validate_members(plan, expected, execution):
    done = read(execution / 'completed.json')
    fields = {'status', 'version', 'study', 'plan_sha256', 'engineering', 'new_fits', 'inherited_fits', 'restored_models',
        'prefit_restored_models', 'control_rows', 'diagnostic_roots', 'astra_calls', 'new_optimizer_steps', 'wall_seconds',
        'evaluation_started_elapsed_seconds', 'cumulative_attempt_wall_seconds', 'files'}
    require(set(done) == fields and done['status'] == 'completed' and done['version'] == 'reacher-two-observation-runner-v1'
        and done['study'] == VERSION and done['plan_sha256'] == expected and done['engineering'] is plan['engineering'], 'Completed whole-study identity')
    for key, count in {'new_fits': 3, 'inherited_fits': 6, 'restored_models': 9, 'prefit_restored_models': 6,
        'control_rows': 42, 'diagnostic_roots': 0, 'astra_calls': 0,
        'new_optimizer_steps': protocol.coverage(plan['settings'])['new_optimizer_updates']}.items():
        integer(done[key], count, 'Whole-study ' + key)
    require(finite(done['wall_seconds'], 'Execution wall', positive=True) <= plan['cap_seconds'], 'Execution within frozen cap')
    members = expected_members(plan['settings'])
    paths = list(execution.rglob('*'))
    require(not execution.is_symlink() and not any(path.is_symlink() for path in paths), 'No artifact symlinks')
    require(set(done['files']) == members and {p.relative_to(execution).as_posix() for p in paths if p.is_file()} == members | {'completed.json'},
            'Exact whole execution membership; no partial/failed/extra artifacts')
    for name, digest in done['files'].items():
        checked(execution / name, digest)
    return done


def audit_copied_lineage(plan, context, execution):
    require(read(execution / 'inheritance.json') == {'cache': context.cache_source, 'prerequisite': context.prerequisite_source}, 'Exact dual lineage receipt')
    for kind, source in (('cache', context.cache_source), ('prerequisite', context.prerequisite_source)):
        for name, digest in source['members'].items():
            checked(execution / 'inherited' / name, digest)
        fields = {'plan': 'plan_sha256', 'audit': 'audit_receipt_sha256', 'completed': 'completed_sha256', 'summary': 'summary_sha256'}
        if kind == 'prerequisite':
            fields['terminal'] = 'terminal_sha256'
        for suffix, field in fields.items():
            checked(execution / 'inherited' / 'lineage' / f'{kind}-{suffix}.json', source[field])
    require(read(execution / 'random-streams.json') == plan['random_stream_contract'], 'Exact authenticated stream receipt')


def audit_boundaries(plan, expected, execution, done, restored, fits):
    settings = plan['settings']
    started, training, fitted, evaluation, control = [read(execution / name) for name in (
        'started.json', 'training-started.json', 'all-fits-completed.json', 'evaluation-started.json', 'control-completed.json')]
    require(set(started) == {'version', 'study', 'plan_sha256', 'engineering', 'unix_time', 'elapsed_seconds'}
        and started['version'] == 'reacher-two-observation-runner-v1' and started['study'] == VERSION
        and started['engineering'] is plan['engineering'], 'Study start receipt')
    order = [row['name'] for row in protocol.training_manifest(settings)]
    require(set(training) == {'plan_sha256', 'unix_time', 'elapsed_seconds', 'training_inputs_sha256', 'inherited_models_restored_sha256', 'fit_order'}
        and training['training_inputs_sha256'] == sha(execution / 'training-inputs.json')
        and training['inherited_models_restored_sha256'] == sha(execution / 'inherited-models-restored.json')
        and training['fit_order'] == order, 'All six inherited restore before three fresh paired fits')
    require(set(fitted) == {'plan_sha256', 'new_fits', 'fit_order', 'files', 'optimizer_steps', 'training_started_sha256', 'unix_time', 'elapsed_seconds'}
        and fitted['new_fits'] == 3 and fitted['fit_order'] == order
        and fitted['files'] == {f'fits/{name}/completed.json': sha(execution / 'fits' / name / 'completed.json') for name in order}
        and fitted['optimizer_steps'] == sum(fit['new_optimizer_updates'] for fit in fits.values())
        and fitted['training_started_sha256'] == sha(execution / 'training-started.json'), 'All fresh fits hash-bound before evaluation')
    require(set(evaluation) == {'plan_sha256', 'unix_time', 'elapsed_seconds', 'all_fits_completed_sha256', 'all_models_restored_sha256', 'random_streams_sha256'}
        and evaluation['all_fits_completed_sha256'] == sha(execution / 'all-fits-completed.json')
        and evaluation['all_models_restored_sha256'] == sha(execution / 'all-models-restored.json')
        and evaluation['random_streams_sha256'] == sha(execution / 'random-streams.json')
        and evaluation['elapsed_seconds'] == done['evaluation_started_elapsed_seconds'], 'All nine final actual classes before fresh draws')
    require(set(control) == {'plan_sha256', 'rows', 'row_order', 'unix_time', 'elapsed_seconds', 'evaluation_started_sha256'}
        and control['rows'] == 42 and control['row_order'] == [row['path'] for row in protocol.execution_order(settings)]
        and control['evaluation_started_sha256'] == sha(execution / 'evaluation-started.json'), 'Exactly42 completed control rows')
    markers = (started, training, fitted, evaluation, control)
    for marker in markers:
        require(marker['plan_sha256'] == expected, 'Phase plan binding')
        finite(marker['unix_time'], 'Phase timestamp', positive=True)
        finite(marker['elapsed_seconds'], 'Phase elapsed')
    require([marker['elapsed_seconds'] for marker in markers] == sorted(marker['elapsed_seconds'] for marker in markers)
        and control['elapsed_seconds'] <= done['wall_seconds'], 'Monotonic whole execution phases')
    times = [started['unix_time'], restored['prefit']['unix_time'], training['unix_time'], fitted['unix_time'],
             restored['before']['unix_time'], evaluation['unix_time'], control['unix_time']]
    require(times == sorted(times), 'Restore/training/evaluation receipt ordering')
    return {'prefit_restored_models': 6, 'fresh_fits_before_evaluation': 3, 'final_restored_models': 9,
        'control_rows': 42, 'ordered_phase_receipts': True, 'saved_ordering_is_not_external_process_attestation': True}


def audit_costs(plan, context, execution, done, restored, input_receipt, fits, controls):
    settings, saved = plan['settings'], read(execution / 'costs.json')
    fields = {'inheritance_copy_wall_seconds', 'training_input_validation_wall_seconds', 'prefit_restore_wall_seconds', 'final_restore_wall_seconds',
        'new_fit_wall_seconds', 'fit_wall_seconds', 'new_fits', 'new_optimizer_steps', 'inherited_optimizer_steps', 'evaluation_optimizer_steps',
        'new_prediction_episodes', 'diagnostic_roots', 'astra_calls', 'innovation_generation_and_storage_seconds', 'control_row_wall_seconds',
        'control_setup_seconds', 'control_decision_seconds', 'control_native_step_seconds', 'control_row_times', 'coverage',
        'cache_parent_costs', 'prerequisite_costs', 'compute_matched', 'accounting'}
    require(set(saved) == fields, 'Complete charged new-fit and control cost schema')
    rows = [controls[row['panel']][row['label']] for row in protocol.execution_order(settings)]
    fresh = {name: fit['wall_seconds'] for name, fit in fits.items() if fit['arm'] == 'two_observation_gru'}
    sums = {'training_input_validation_wall_seconds': input_receipt['wall_seconds'], 'prefit_restore_wall_seconds': restored['prefit']['wall_seconds'],
        'final_restore_wall_seconds': restored['before']['wall_seconds'], 'new_fit_wall_seconds': sum(fresh.values()),
        'control_row_wall_seconds': sum(row['row_wall_seconds'] for row in rows), 'control_setup_seconds': sum(row['setup_seconds'] for row in rows),
        'control_decision_seconds': sum(row['decision_wall_seconds'] for row in rows), 'control_native_step_seconds': sum(row['native_step_seconds'] for row in rows)}
    require(saved['fit_wall_seconds'] == fresh, 'Every fit time retained')
    for key, value in sums.items():
        require(math.isclose(finite(saved[key], key, positive=True), value, rel_tol=1e-12, abs_tol=1e-8), 'Reconciled phase cost ' + key)
    for key, value in {'new_fits': 3, 'new_optimizer_steps': protocol.coverage(settings)['new_optimizer_updates'],
        'inherited_optimizer_steps': 0, 'evaluation_optimizer_steps': 0, 'new_prediction_episodes': 0, 'diagnostic_roots': 0, 'astra_calls': 0}.items():
        integer(saved[key], value, 'Cost scope ' + key)
    require(isinstance(saved['control_row_times'], list) and len(saved['control_row_times']) == 42, 'All row timing receipts retained')
    for index, row in enumerate(protocol.execution_order(settings)):
        actual = read(execution / row['path'] / 'timings.json')
        if row.get('arm') == 'two_observation_gru':
            actual['row_wall_seconds'] = read(execution / row['path'] / 'completed.json')['row_wall_seconds']
        require(saved['control_row_times'][index] == {'path': row['path'], **actual}, 'Exact row timing copy/order')
    require(saved['cache_parent_costs'] == context.cache_source['prior_costs']
        and saved['prerequisite_costs'] == context.prerequisite_source['prior_costs'] and saved['compute_matched'] is False
        and saved['coverage'] == protocol.coverage(settings) and isinstance(saved['accounting'], str) and saved['accounting'], 'Nested historical costs and unmatched compute scope')
    nonoverlap = ('inheritance_copy_wall_seconds', 'training_input_validation_wall_seconds', 'prefit_restore_wall_seconds',
        'final_restore_wall_seconds', 'new_fit_wall_seconds', 'innovation_generation_and_storage_seconds', 'control_row_wall_seconds')
    require(sum(finite(saved[key], key, positive=True) for key in nonoverlap) <= done['wall_seconds'] + 1e-6, 'Whole wall includes every disjoint phase')
    cumulative = context.prerequisite_source['prior_costs']['cumulative_attempt_wall_seconds'] + done['wall_seconds']
    require(math.isclose(done['cumulative_attempt_wall_seconds'], cumulative, rel_tol=1e-12, abs_tol=1e-8), 'Prior cumulative ancestry counted once')
    learned = {key: sum(row['scoring_work'][key] for row in rows if 'scoring_work' in row) for key in (
        'candidate_evaluations', 'imagined_transitions', 'geometry_samples', 'geometry_seconds', 'model_advance_seconds',
        'selected_geometry_samples', 'selected_geometry_seconds', 'selected_model_advance_seconds')}
    physics = {key: sum(row['physics_work'][key] for row in rows if 'physics_work' in row) for key in (
        'candidate_native_transitions_replayed', 'selected_native_transitions_replayed', 'native_seconds', 'geometry_seconds', 'operation_wall_seconds')}
    counters = {key: sum(row['physics_work']['counts'][key] for row in rows if 'physics_work' in row) for key in PHYSICS_COUNTS}
    coverage = protocol.coverage(settings)
    for key, want in (('candidate_evaluations', 'learned_candidate_evaluations'), ('imagined_transitions', 'learned_imagined_transitions'),
        ('geometry_samples', 'learned_imagined_transitions'), ('selected_geometry_samples', 'learned_selected_advances')):
        integer(learned[key], coverage[want], 'Complete learned ' + key)
    for key, want in (('candidate_native_transitions_replayed', 'physics_nominal_transitions'), ('selected_native_transitions_replayed', 'physics_selected_advances')):
        integer(physics[key], coverage[want], 'Complete physics ' + key)
    integer(counters['native_substeps_completed'], coverage['physics_candidate_native_substeps'] + coverage['physics_selected_native_substeps'], 'All candidate and selected native substeps')
    integer(learned['geometry_samples'] + learned['selected_geometry_samples'] + counters['geometry_samples_completed'],
        coverage['geometry_candidate_samples'] + coverage['geometry_selected_samples'], 'All geometry evaluations including selected')
    training_totals = {key: sum(fit['saved_training_audit']['work'][key] for fit in fits.values() if fit['arm'] == 'two_observation_gru')
        for key in ('public_assimilate_samples', 'advance_samples', 'replayed_observation_update_samples', 'replayed_transition_samples',
            'gru_cell_sample_calls', 'linear_layer_sample_calls', 'analytic_reward_sample_calls', 'dense_affine_macs')}
    links = {'public_assimilate_samples': 'new_training_public_assimilate_samples', 'advance_samples': 'new_training_prefix_and_rollout_advance_samples',
        'replayed_observation_update_samples': 'new_training_replayed_observation_update_samples', 'replayed_transition_samples': 'new_training_replayed_transition_samples',
        'gru_cell_sample_calls': 'new_training_gru_cell_samples', 'linear_layer_sample_calls': 'new_training_linear_layer_samples',
        'analytic_reward_sample_calls': 'new_training_analytic_reward_samples'}
    for key, wanted in links.items():
        integer(training_totals[key], coverage[wanted], 'All training padded work ' + key)
    return {**saved, 'execution_wall_seconds': done['wall_seconds'], 'cumulative_attempt_wall_seconds': cumulative,
        'new_training_work_checked': training_totals, 'learned_control_scoring': learned,
        'physics_control_scoring': {**physics, 'counts': counters},
        'limits': 'Shared-host measured wall times include storage and authentication. Nested phases are not additive; equal parameters and search proposals are not equal compute.'}


def audit_saved(plan, expected_plan_sha256, execution, out, *, engineering=False, root=ROOT,
                plan_path=None, plan_bytes=None, freeze_ref=None):
    """Audit one byte-bound completed execution, including failed scientific gates.

    Neither a failed criterion nor weak utility makes the audit incomplete.
    Missing/corrupt artifacts, cap failures or mismatched bindings do. No retry,
    replacement seed, selected fit or learned replay is performed here.
    """
    from openjev.research import reacher_two_observation_experiment as experiment

    execution, out, root = Path(execution), Path(out), Path(root)
    require(not out.exists(), 'Exclusive new audit destination required')
    begin, tokens = time.monotonic(), []
    try:
        require(type(plan['audit_cap_seconds']) is int and plan['audit_cap_seconds'] > 0, 'Explicit positive audit cap')
        deadline = begin + plan['audit_cap_seconds']
        for contextvar in (_DEADLINE, memory._DEADLINE, inherited._DEADLINE, previous._DEADLINE):
            tokens.append((contextvar, contextvar.set(deadline)))
        if plan_path is not None:
            plan_path = Path(plan_path)
            if plan_path.is_absolute():
                plan_path = plan_path.relative_to(root)
            plan_path = plan_path.as_posix()

        def authenticate():
            check_budget()
            context = experiment.validate_plan(plan, expected_plan_sha256, root=root, runtime=experiment.runtime(),
                engineering=engineering, plan_path=plan_path, plan_bytes=plan_bytes, freeze_ref=freeze_ref)
            require(context.execution_authorized is True and context.plan_sha256 == expected_plan_sha256
                and context.engineering is engineering, 'Explicit validated experiment context')
            require(expected_members(context.settings) == set(experiment.expected_members(context.settings)), 'Independent complete artifact contract agreement')
            check_budget()
            return context

        context = authenticate()
        settings = context.settings
        completed = validate_members(plan, expected_plan_sha256, execution)
        completion_hash = sha(execution / 'completed.json')
        audit_copied_lineage(plan, context, execution)
        fits, restored, input_receipt = audit_fits_and_snapshots(plan, expected_plan_sha256, context, execution)
        numerical = kernel_plan(settings, context.streams)
        inputs = [audit_innovations(numerical, execution / 'innovations' / 'control' / f'{step:03d}', step) for step in range(50)]
        controls, cohorts, reward_rows = {panel: {} for panel in PANELS}, {}, []
        initial_states = disturbances = None
        for row in protocol.execution_order(settings):
            check_budget()
            folder = execution / row['path']
            records = base.load_records(folder / 'episodes', settings['control_episodes'])
            cohorts[row['path']] = audit_cohort(numerical, records, row['panel'])
            initial, noise = base.stack(records, 'audit', 'integration_state')[:, 0], base.stack(records, 'audit', 'actuator_noise')
            if initial_states is None:
                initial_states, disturbances = initial.copy(), noise.copy()
            require(np.array_equal(initial, initial_states) and np.array_equal(noise, disturbances), 'All42 rows share exact initial states and disturbance innovations')
            if row.get('arm') == 'two_observation_gru':
                value = audit_history_control(numerical, execution, folder, row, records, inputs, fits[row['fit']])
            elif 'fit' in row:
                value = memory.audit_learned_control(numerical, folder, {**row, 'score_mode': 'geometry'}, records, inputs, fits[row['fit']])
            else:
                value = audit_reference_control(numerical, folder, row, records, inputs)
            controls[row['panel']][row['label']] = value
            reward_rows.append({'panel': row['panel'], 'label': row['label'],
                'case_ids': [f'control/{index}' for index in range(settings['control_episodes'])],
                'native_rewards': base.stack(records, 'audit', 'rewards')})
        arithmetic = results.evaluate_rows(settings, reward_rows)
        require(read(execution / 'results.json') == arithmetic, 'All saved results and25 criteria rederived from authenticated native rewards')
        for row in protocol.execution_order(settings):
            actual, derived = controls[row['panel']][row['label']], arithmetic['controls'][row['panel']][row['label']]
            require(actual['episode_costs'] == derived['episode_costs'] and actual['mean_cost'] == derived['mean_cost'], 'Independent all-case row arithmetic agrees')
        gate = arithmetic['continuation_gate']
        require(len(gate['checks']) == 25 and gate['passed'] is all(check['left'] <= check['right'] for check in gate['checks'])
            and gate['checks_passed'] == sum(check['left'] <= check['right'] for check in gate['checks'])
            and gate['total_checks'] == 25 and gate['secondary_cannot_rescue_primary'] is True, 'Exact inclusive all-check gate retained')
        boundary = audit_boundaries(plan, expected_plan_sha256, execution, completed, restored, fits)
        costs = audit_costs(plan, context, execution, completed, restored, input_receipt, fits, controls)
        require(validate_members(plan, expected_plan_sha256, execution) == completed and sha(execution / 'completed.json') == completion_hash,
                'Exact execution membership and all bytes unchanged at audit exit')
        authenticate()
        physics = [controls[panel][name]['physics_work'] for panel in PANELS for name in PHYSICS_REFERENCES]
        observers = [controls[panel][name]['public_observer'] for panel in PANELS for name in ('particle', 'public_kinematic')]
        native_count = sum(item['transitions'] for item in cohorts.values())
        integer(native_count, protocol.coverage(settings)['native_control_transitions'], 'All42 native rows replayed')
        summary = {'status': 'completed', 'version': VERSION, 'engineering': engineering,
            'plan_sha256': expected_plan_sha256, 'execution_completed_sha256': completion_hash,
            'saved_output_only': True, 'new_model_calls': 0, 'new_policy_calls': 0, 'new_fits': 0, 'new_optimizer_steps': 0,
            'execution_new_fits': 3, 'execution_new_optimizer_steps': completed['new_optimizer_steps'],
            'native_control_transitions_checked': native_count,
            'native_control_max_abs_error': max(item['max_abs_error'] for item in cohorts.values()),
            'native_nominal_candidate_transitions_checked': sum(item['candidate_native_transitions_replayed'] for item in physics),
            'native_nominal_selected_transitions_checked': sum(item['selected_native_transitions_replayed'] for item in physics),
            'native_nominal_max_abs_error': max(item['max_abs_error'] for item in physics),
            'public_observer_transitions_checked': sum(item['observer_native_transitions_replayed'] for item in observers),
            'public_observer_max_abs_error': max(item['max_abs_error'] for item in observers),
            'coverage': protocol.coverage(settings), 'fits': fits, 'phase_boundary': boundary,
            'control': controls, 'families': arithmetic['families'], 'row_order': arithmetic['row_order'],
            'continuation_gate': gate, 'cohorts': cohorts,
            'random_streams': {'authenticated': True, 'manifest_sha256': sha(execution / 'random-streams.json'),
                'historical_order_replay_only': True, 'new_scientific_streams': 0},
            'costs': {**costs, 'audit_validation_wall_seconds': time.monotonic() - begin},
            'limits': [
                'All3 fresh history fits,6 inherited controls and42 rows retained. No posthoc fit or gate selection.',
                'Saved training tensors, Adam schemas/moments/steps, losses, mask counts and12/11 forward-work witnesses are checked; gradients and numerical optimizer transitions are not replayed.',
                'Original full orders are replayed using isolated historical generators. This validates supplied order identities, not new scientific seed allocation.',
                'Public buffers, actual issued actions, clocks and geometry/CEM arithmetic are independently checked. Saved learned hidden states and predictions are not neural re-evaluations.',
                'Before/after tensor equality and receipt ordering are source-bound evidence, not external attestation of every process action.',
                'Inherited training data and initializations are authenticated to the completed parent audit; inherited native training episodes are not redundantly replayed here.',
                'Native executed trajectories, every nominal physics candidate/selected action and causal public observers are separately replayed and counted.',
                'Two-observation reconstruction includes padded12/11 kernels, heads, residual cost, history copies and selected advance. Equal parameters/search budgets do not match total compute.',
                'Geometry uses projected final angles and approximates native RK4 cached-body reward; projected distance is not expected distance.',
                'Matched CEM256 supplied nominal physics is not an optimal stochastic control oracle. Shared-host batched wall time is not isolated latency or total FLOPs.',
                'The25 comparisons apply to these three paired fits and cases. A failed gate is not equivalence; a pass is not biological novelty or cross-environment generalization.',
            ]}
        summary['native_transitions_checked'] = native_count + summary['native_nominal_candidate_transitions_checked'] + summary['native_nominal_selected_transitions_checked']
        summary['native_max_abs_error'] = max(summary['native_control_max_abs_error'], summary['native_nominal_max_abs_error'])
        check_budget()
        out.mkdir(parents=True, exist_ok=False)
        write(out / 'summary.json', summary)
        (out / 'README.md').write_text('# Two-observation study audit\n\n'
            f"Continuation: **{'PASS' if gate['passed'] else 'FAIL'}** ({gate['checks_passed']}/25).\n\n"
            'All nine models and42 rows retained. The audit performs no learned inference or optimization. '
            'Saved training arithmetic, public-history buffers, score/search evidence and native executed/nominal transitions are checked. '
            'A completed audit can retain a failed scientific continuation gate. See summary.json for all costs and limitations.\n')
        check_budget()
        receipt = {'status': 'completed', 'version': VERSION, 'engineering': engineering,
            'plan_sha256': expected_plan_sha256, 'source_sha256': plan['sources'], 'runtime': plan['runtime'],
            'execution_completed_sha256': completion_hash, 'execution_members': completed['files'],
            'saved_output_only': True, 'costs': {**summary['costs'], 'audit_validation_wall_seconds': time.monotonic() - begin},
            'files': {name: sha(out / name) for name in ('summary.json', 'README.md')}}
        write(out / 'receipt.json', receipt)
        check_budget()
        return summary
    except BaseException as error:
        try:
            out.mkdir(parents=True, exist_ok=True)
            if (out / 'receipt.json').exists():
                (out / 'receipt.json').rename(out / 'invalid-receipt.json')
            write(out / 'failed.json', {'status': 'failed', 'version': VERSION, 'error': repr(error),
                'exception_type': type(error).__name__, 'plan_sha256': expected_plan_sha256,
                'wall_seconds': time.monotonic() - begin, 'saved_output_only': True})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f'Audit failure receipt could not be preserved: {preservation_error!r}')
        raise
    finally:
        for contextvar, token in reversed(tokens):
            contextvar.reset(token)


def audit_plan(path, expected_sha256, execution, out, *, engineering=False, freeze_ref=None, root=ROOT):
    require(not Path(out).exists(), 'Exclusive audit destination')
    try:
        payload = Path(path).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == expected_sha256, 'External plan byte SHA256')
        plan = json.loads(payload)
    except BaseException as error:
        try:
            Path(out).mkdir(parents=True, exist_ok=False)
            write(Path(out) / 'failed.json', {'status': 'failed', 'phase': 'authenticate-plan', 'error': repr(error),
                'plan_sha256': expected_sha256, 'saved_output_only': True})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f'Audit authentication failure could not be preserved: {preservation_error!r}')
        raise
    return audit_saved(plan, expected_sha256, execution, out, engineering=engineering, root=root,
                       plan_bytes=payload, freeze_ref=freeze_ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--freeze-path')
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    require((args.freeze_path is None) == (args.freeze_sha256 is None), 'Both external freeze fields required together')
    freeze = None if args.freeze_path is None else {'path': args.freeze_path, 'sha256': args.freeze_sha256}
    value = audit_plan(args.plan, args.expected_plan_sha256, args.execution, args.out, engineering=args.engineering, freeze_ref=freeze)
    print(json.dumps({'status': value['status'], 'continuation_gate_passed': value['continuation_gate']['passed'],
        'checks_passed': value['continuation_gate']['checks_passed'], 'native_transitions_checked': value['native_transitions_checked']}))


if __name__ == '__main__':
    main()
