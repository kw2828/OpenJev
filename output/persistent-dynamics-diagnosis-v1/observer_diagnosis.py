"""Saved-output-only observer diagnosis. No project, Torch, native or RNG imports."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/persistent-dynamics-qualification-v1/control-engineering-01'
OUT = Path(__file__).resolve().parent
EXPECTED = '064a0e7157088a6625bfc53afd1022d6d16232dc85dbf6cbde3238c1314bc050'
ARMS = ('nominal', 'adaptive', 'frozen', 'public_gain', 'true_state', 'zero')
INTERVALS = {'all': (0, 200), '0:50': (0, 50), '50:80': (50, 80),
             '80:100': (80, 100), '100:150': (100, 150), '150:200': (150, 200)}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, expected=None):
    content = Path(path).read_bytes()
    if expected is not None:
        require(hashlib.sha256(content).hexdigest() == expected, f'hash mismatch: {path}')
    return json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def verify_sources(completed):
    result = {}
    for name, key, count in (
        ('evidence/reacher-two-observation-study-v1/protocol/plan.json', 'sources', 116),
        ('evidence/reacher-innovation-pilot-v1/protocol/plan.json', 'source_sha256', 136)):
        mapping = read(ROOT / name)[key]
        require(len(mapping) == count, 'protected source count')
        require(all(sha(ROOT / path) == digest for path, digest in mapping.items()), 'protected source changed')
        result[name] = {'sha256': sha(ROOT / name), 'sources_verified': count}
    require(len(completed['source_sha256']) == 14, 'engineering source count')
    for name, digest in completed['source_sha256'].items():
        require(sha(ROOT / name) == digest and sha(RUN / 'source-snapshot' / name) == digest,
                f'engineering source/snapshot changed: {name}')
    result['engineering_sources'] = completed['source_sha256']
    return result


def rmse(value):
    return float(np.sqrt(np.mean(np.square(value))))


def stats(value):
    value = np.asarray(value)
    if not value.size:
        return None
    require(np.isfinite(value).all(), 'nonfinite statistic')
    return {'mean': float(value.mean()), 'median': float(np.median(value)),
            'min': float(value.min()), 'max': float(value.max())}


def wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def reconstruct(packets, dt):
    theta = np.arctan2(packets[:, 2:4], packets[:, :2])
    delta = wrap(np.diff(theta, axis=0))
    q = np.empty_like(theta)
    q[0] = theta[0]
    velocities = {name: np.zeros_like(theta) for name in
                  ('backward1', 'endpoint3_current', 'endpoint4', 'secant3_intervals')}
    for t in range(1, len(q)):
        q[t] = q[t-1] + delta[t-1]
        velocities['backward1'][t] = delta[t-1] / dt
        velocities['endpoint3_current'][t] = (delta[t-1] / dt if t == 1
                                              else (3*delta[t-1] - delta[t-2]) / (2*dt))
        velocities['endpoint4'][t] = (velocities['endpoint3_current'][t] if t < 3
                                      else (11*delta[t-1] - 7*delta[t-2] + 2*delta[t-3]) / (6*dt))
        width = min(t, 3)
        velocities['secant3_intervals'][t] = np.sum(delta[t-width:t], axis=0) / (width*dt)
    return q, delta, velocities


def gain_metrics(records, start, stop, grid, commands, noise):
    selected = [record for record in records if start <= record['step_index'] < stop]
    if not selected:
        return None
    indices = np.array([r['step_index'] for r in selected])
    post = np.array([r['gain_after'] for r in selected])
    truth = np.where(indices < 80, .7, 1.3)
    sensitivity, residual_spreads, score_spreads, margins = [], [], [], []
    truth_excess, window_energy = [], []
    flat_current = 0
    for record, t, true in zip(selected, indices, truth, strict=True):
        predictions = np.asarray(record['candidate_predictions'])
        residuals = np.asarray(record['candidate_residuals'])
        scores = np.asarray(record['window_residual_sums'])
        require(np.array_equal(residuals, np.mean((predictions - np.asarray(record['next_packet'])[:4])**2, axis=1)),
                'saved residual arithmetic')
        sensitivity.append(float(np.mean(((predictions[-1]-predictions[0])/(grid[-1]-grid[0]))**2)))
        residual_spreads.append(float(np.ptp(residuals)))
        flat_current += int(np.all(residuals == residuals[0]))
        score_spreads.append(float(np.ptp(scores)))
        order = np.sort(scores)
        margins.append(float(order[1] - order[0]))
        true_index = int(np.flatnonzero(grid == true)[0])
        truth_excess.append(float(scores[true_index] - order[0]))
        window_energy.append(float(np.sum(commands[max(0, t-19):t+1]**2)))
    return {'updates': len(selected), 'after_gain_mean': float(post.mean()),
            'after_gain_mae_experienced_gain': float(np.mean(np.abs(post-truth))),
            'lower_endpoint_count': int(np.sum(post == grid[0])),
            'upper_endpoint_count': int(np.sum(post == grid[-1])),
            'exact_flat_current_residual_count': flat_current,
            'exact_flat_rolling_bank_count': sum(r['exactly_flat_bank'] for r in selected),
            'current_sensitivity_mean_square': stats(sensitivity),
            'current_residual_range': stats(residual_spreads), 'rolling_score_range': stats(score_spreads),
            'rolling_second_minus_best': stats(margins), 'rolling_true_gain_minus_best': stats(truth_excess),
            'command_norm': stats(np.linalg.norm(commands[indices], axis=1)),
            'noise_norm_privileged_diagnostic_only': stats(np.linalg.norm(noise[indices], axis=1)),
            'last20_command_squared_energy': stats(window_energy),
            'zero_command_count': int(np.sum(np.all(commands[indices] == 0., axis=1)))}


def row_analysis(arm, row_binding, audit_binding):
    folder = RUN / 'rows' / arm
    receipt = read(folder / 'completed.json', row_binding['completed_sha256'])
    require(receipt['status'] == 'completed' and receipt['engineering'] and receipt['steps'] == 200, 'row completion')
    expected = {'started.json', 'initial-controller.json', 'controller.json', 'episode.json'}
    expected |= {f'decisions/{t:03d}.{suffix}' for t in range(200) for suffix in ('json', 'npz')}
    expected |= {f'observations/{t:03d}.json' for t in range(1, 201)}
    require(set(receipt['files']) == expected, 'complete row membership')
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} == expected | {'completed.json'},
            'exact row disk membership')
    for name, digest in receipt['files'].items():
        require(not (folder/name).is_symlink() and sha(folder/name) == digest, f'row member: {name}')
    audit = read(RUN / f'audit-{arm}.json', audit_binding['sha256'])
    require(audit['status'] == 'completed' and audit['row_completed_sha256'] == row_binding['completed_sha256'], 'audit linkage')
    episode = read(folder / 'episode.json', receipt['files']['episode.json'])
    require(episode['metadata']['status'] == 'completed' and episode['metadata']['dt'] == .02, 'episode scope')
    packets = np.asarray(episode['policy']['packets'], np.float64)
    commands = np.asarray(episode['policy']['commands'], np.float64)
    q = np.array([state['qpos'][:2] for state in episode['audit']['decision_states']])
    v = np.array([state['qvel'][:2] for state in episode['audit']['decision_states']])
    noise = np.array([state['actuator_noise'] for state in episode['audit']['transitions']])
    applied = np.array([state['applied_action'] for state in episode['audit']['transitions']])
    actual_gains = np.array([state['gear_multiplier'] for state in episode['audit']['transitions']])
    require(packets.shape == (201, 8) and commands.shape == (200, 2) and q.shape == v.shape == (201, 2), 'arrays')
    require(np.isfinite(packets).all() and np.all(packets[:, 6] == 1) and np.all(packets[:, 7] == 0), 'visible packets')
    require(np.array_equal(actual_gains, np.r_[np.full(80, .7), np.full(120, 1.3)]), 'gain history')
    public_q, public_v, root_v, planning_gains = [], [], [], []
    for t in range(200):
        with np.load(folder/f'decisions/{t:03d}.npz', allow_pickle=False) as saved:
            public_q.append(saved['public_qpos'][:2].copy())
            public_v.append(saved['public_qvel'][:2].copy())
            root_v.append(saved['root_qvel'][:2].copy())
            planning_gains.append(float(saved['planning_gain']))
            require(np.array_equal(saved['public_packet'], packets[t]) and np.array_equal(saved['action'], commands[t]), 'public alignment')
    public_q, public_v, root_v = map(np.array, (public_q, public_v, root_v))
    planning_gains = np.array(planning_gains)
    reconstructed_q, delta, methods = reconstruct(packets, .02)
    require(np.array_equal(public_q, reconstructed_q[:200]) and np.array_equal(public_v, methods['endpoint3_current'][:200]),
            'observer formula or root alignment mismatch')
    observations = [read(folder/f'observations/{t:03d}.json') for t in range(1, 201)]
    for t, observation in enumerate(observations, 1):
        require(np.array_equal(observation['public_qpos'][:2], reconstructed_q[t])
                and np.array_equal(observation['public_qvel'][:2], methods['endpoint3_current'][t]), 'observation alignment')
    ids = [row['identifier_trace'] for row in observations if row['identifier_updated']]
    grid = np.asarray(read(folder/'started.json')['gain_grid'])
    if ids:
        residual_history, previous = [], 1.
        for record in ids:
            residual_history.append(np.asarray(record['candidate_residuals']))
            scores = np.sum(np.stack(residual_history[-20:]), axis=0, dtype=np.float64)
            require(np.array_equal(scores, record['window_residual_sums']), 'rolling window arithmetic')
            flat = bool(np.all(scores == scores[0]))
            after = previous if flat else float(grid[int(np.argmin(scores))])
            require(record['gain_before'] == previous and record['gain_after'] == after
                    and record['exactly_flat_bank'] == flat, 'gain selection arithmetic')
            previous = after
    exact_packets = packets.copy()
    exact_packets[:, :4] = np.c_[np.cos(q), np.sin(q)]
    exact_bdf2 = reconstruct(exact_packets, .02)[2]['endpoint3_current']
    intervals = {}
    costs = -np.array([state['reward'] for state in episode['audit']['transitions']])
    for name, (start, stop) in INTERVALS.items():
        selected = slice(start, stop)
        intervals[name] = {'decision_roots': stop-start, 'native_cost_sum_context_only': float(costs[selected].sum()),
            'public_qpos_rmse_rad': rmse(public_q[selected]-q[selected]),
            'public_qpos_max_abs_rad': float(np.max(np.abs(public_q[selected]-q[selected]))),
            'public_qvel_rmse_rad_s': rmse(public_v[selected]-v[selected]),
            'public_qvel_mae_rad_s': float(np.mean(np.abs(public_v[selected]-v[selected]))),
            'public_qvel_max_abs_rad_s': float(np.max(np.abs(public_v[selected]-v[selected]))),
            'actual_planning_root_qvel_rmse_rad_s': rmse(root_v[selected]-v[selected]),
            'velocity_methods_rmse_rad_s': {method: rmse(value[selected]-v[selected]) for method, value in methods.items()},
            'public_quantization_contribution_to_bdf2_rmse_rad_s': rmse(methods['endpoint3_current'][selected]-exact_bdf2[selected]),
            'native_velocity_rms_rad_s': rmse(v[selected]),
            'elbow_soft_limit_exceeded_root_count': int(np.sum(np.abs(q[selected, 1]) > 3.)),
            'command_norm': stats(np.linalg.norm(commands[selected], axis=1)),
            'noise_norm_privileged_diagnostic_only': stats(np.linalg.norm(noise[selected], axis=1)),
            'mean_command_squared_energy': float(np.mean(np.sum(commands[selected]**2, axis=1))),
            'mean_noise_squared_energy': float(np.mean(np.sum(noise[selected]**2, axis=1))),
            'zero_command_count': int(np.sum(np.all(commands[selected] == 0., axis=1))),
            'planning_gain_mean': float(planning_gains[selected].mean()),
            'planning_gain_mae': float(np.mean(np.abs(planning_gains[selected]-actual_gains[selected]))),
            'planning_gain_lower_endpoint_count': int(np.sum(planning_gains[selected] == grid[0])),
            'planning_gain_upper_endpoint_count': int(np.sum(planning_gains[selected] == grid[-1])),
            'gain_identifier': gain_metrics(ids, start, stop, grid, commands, noise)}
    preswitch = planning_gains[:80]
    upper = np.flatnonzero(preswitch == grid[-1])
    lower = np.flatnonzero(preswitch == grid[0])
    return {'bindings': {'completed_sha256': row_binding['completed_sha256'], 'audit_sha256': audit_binding['sha256'],
                         'payload_files_verified': len(expected),
                         'payload_bytes_verified': sum((folder/name).stat().st_size for name in expected),
                         'episode_sha256': receipt['files']['episode.json'], 'payload_sha256': receipt['files']},
            'consistency': {'observer_formula_max_abs_error': float(np.max(np.abs(public_v-methods['endpoint3_current'][:200]))),
                            'max_actual_angle_increment_rad': float(np.max(np.abs(np.diff(q, axis=0)))),
                            'max_public_wrapped_increment_rad': float(np.max(np.abs(delta))),
                            'aliasing_transition_joint_count': int(np.sum(np.abs(np.diff(q, axis=0)-delta) > np.pi)),
                            'endpoint_packet_cos_sin_max_error': float(np.max(np.abs(packets[:, :4]-exact_packets[:, :4]))),
                            'applied_action_clipping_count': int(np.sum(np.abs(applied) == 1.)),
                            'identifier_updates': len(ids)},
            'preswitch': {'planning_gain_at_roots_0_20_39_40_50_60_79': planning_gains[[0,20,39,40,50,60,79]].tolist(),
                          'first_upper_endpoint_decision': int(upper[0]) if len(upper) else None,
                          'first_lower_endpoint_decision': int(lower[0]) if len(lower) else None,
                          'identifier': gain_metrics(ids, 0, 80, grid, commands, noise)},
            'intervals': intervals}


def main():
    for name in ('observer-results.json', 'observer-results.md'):
        require(not (OUT/name).exists(), 'exclusive diagnostic outputs')
    completed = read(RUN/'completed.json', EXPECTED)
    require(completed['status'] == 'completed' and completed['engineering']
            and completed['scientific_gate'] is None and not completed['scientific_data_allocated'], 'engineering terminal')
    execution = read(RUN/'execution-completed.json', completed['execution_completed_sha256'])
    require(execution['status'] == 'completed', 'execution terminal')
    source_binding = verify_sources(completed)
    require(tuple(row['arm'] for row in completed['rows']) == ARMS
            and tuple(row['arm'] for row in completed['audits']) == ARMS, 'all six roles')
    rows = {arm: row_analysis(arm, row, audit)
            for arm, row, audit in zip(ARMS, completed['rows'], completed['audits'], strict=True)}
    result = {'status': 'completed', 'scope': 'posthoc saved-output engineering diagnosis only',
              'run_completed_sha256': EXPECTED, 'script_sha256': sha(__file__),
              'execution_completed_sha256': completed['execution_completed_sha256'],
              'source_bindings': source_binding, 'native_calls': 0, 'model_calls': 0, 'rng_draws': 0,
              'interval_semantics': 'Decision t uses packet/state t, then command t; windows are half-open decision indices. Returned observation t+1 identifies experienced gain t.',
              'method_definitions': {'backward1': 'delta_t/dt, the preceding interval average',
                  'endpoint3_current': '(3*delta_t-delta_(t-1))/(2*dt); startup zero then backward1',
                  'endpoint4': '(11*delta_t-7*delta_(t-1)+2*delta_(t-2))/(6*dt); insufficient-prefix fallback to current method',
                  'secant3_intervals': 'mean of up to three completed angle increments divided by dt; causal interval average with lag'},
              'sensitivity_definition': 'Mean coordinate squared endpoint-grid prediction difference divided by squared gain range; descriptive local finite-difference sensitivity, not Fisher information/confidence.',
              'rows': rows,
              'limitations': ['One engineering initial condition and one .7-to-1.3 trajectory per controller; row histories differ because their selected actions differ.',
                  'No alternative observer was used to choose any action or refit any gain. Saved-bank residuals were reused only for arithmetic.',
                  'Finite-difference velocity alternatives are descriptive and not tuned; endpoint velocity is not identical to an interval average.',
                  'Noise comparisons and exact-angle differentiation use privileged audit data for diagnosis only.',
                  'Low excitation, state error and gain drift may coincide; these records do not isolate their causal effects on control cost.']}
    verify_sources(completed)
    require(sha(RUN/'completed.json') == EXPECTED, 'terminal changed')
    with (OUT/'observer-results.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'rows': len(rows), 'output_sha256': sha(OUT/'observer-results.json')}, indent=2))


if __name__ == '__main__':
    main()
