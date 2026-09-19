"""Fixed-checkpoint belief calibration with the unchanged C native picker. No work runs at import.

The only privileged read is an evaluator-only post-reset deck digest. Neither
the deck nor its digest is an input to the public tracker or learned memory.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import deque
from pathlib import Path

import evaluate_card_memory as old
import numpy as np
import torch
from card_calibration_common import (
    CONTROLLERS,
    EVALUATION,
    POLICIES,
    REFERENCES,
    VERSION,
    authenticate,
    authenticate_calibration,
)
from card_memory_common import preserve_failure

from openjev.research.card_memory_picker import CardPolicyTracker
from openjev.research.card_memory_task import PublicTeacher
from openjev.research.card_probability_calibration import BETA_BOUNDS, transform_probabilities

LAYOUT_PREFIX = b'card-layout-int64le-v1\0'
COUNTERS = ('reset_attempted', 'reset_returned', 'identity_read_attempted', 'identity_read_returned',
            'init_attempted', 'init_returned', 'predict_attempted', 'predict_returned',
            'transform_attempted', 'transform_returned', 'decision_attempted', 'decision_returned', 'native_attempted', 'native_returned',
            'write_attempted', 'write_returned')
_require, _read, _write, _sha, _path, _check = old._require, old._read, old._write, old._sha, old._path, old._check


def layout_identity(env, *, counts=None):
    """Evaluator identity only. Return a digest, never hidden rank values."""
    native_state = env.get_state()
    if counts is not None:
        counts['identity_read_returned'] += 1
    deck = native_state[0]
    _require(isinstance(deck, np.ndarray) and deck.dtype == np.int64 and deck.shape == (52,)
             and np.all((deck >= 0) & (deck < 13))
             and np.array_equal(np.bincount(deck, minlength=13), np.full(13, 4)),
             'Native layout must be 52 int64 ranks, four copies of each rank')
    digest = hashlib.sha256(LAYOUT_PREFIX + np.asarray(deck, dtype='<i8').tobytes()).hexdigest()
    del deck, native_state
    return digest


def evaluate_episode(env, *, controller, policy, seed, beta=None, model=None, deadline=None):
    """One public episode; caller owns close and serialized whole-episode costs.

Failure attaches partial_card_episode/partial_card_receipt, including every
returned native transition and attempted/completed model call separately.
"""
    begin = time.monotonic()
    _require(controller in CONTROLLERS and policy in POLICIES, 'Unknown controller/policy')
    _require(type(seed) is int and 0 <= seed < 2**32, 'Explicit uint32 reset seed required')
    _require((controller in REFERENCES) == (model is None), 'Learned/reference model mismatch')
    _require(controller not in REFERENCES or policy == 'baseline', 'References are baseline-only')
    if policy == 'temperature':
        _require(type(beta) in (int, float) and math.isfinite(beta)
                 and BETA_BOUNDS[0] <= beta <= BETA_BOUNDS[1], 'Temperature beta must be finite in frozen bounds')
        beta = float(beta)
    else:
        _require(beta is None, 'Baseline/hard/reference beta must be None')
    saved = {name: [] for name in ('observations', 'actions', 'ranks', 'rewards', 'terminated',
                                  'truncated', 'raw_probabilities', 'picker_probabilities')}
    counts = dict.fromkeys(COUNTERS, 0)
    timings = dict.fromkeys(('native_seconds', 'identity_seconds', 'init_seconds', 'predict_seconds',
                            'write_seconds', 'public_seconds', 'transform_seconds', 'decision_seconds'), 0.0)
    diagnostics, layout_sha256, tracker_ready = [], None, False
    tracker = CardPolicyTracker('C')
    teacher = PublicTeacher() if controller == 'exact' else None
    events = deque(maxlen=32)
    phase = 'reset'

    def arrays():
        result = {}
        for name, values in saved.items():
            dtype = (np.bool_ if name in ('terminated', 'truncated') else
                     np.int64 if name in ('observations', 'actions', 'ranks') else np.float64)
            tail = (52,) if name == 'observations' else (52, 13) if name.endswith('probabilities') else ()
            result[name] = np.asarray(values, dtype=dtype).reshape((-1, *tail))
        return result

    def receipt(status):
        return {'status': status, 'controller': controller, 'policy': policy, 'seed': seed,
                'phase': phase, 'layout_sha256': layout_sha256, 'tracker_policy': 'C', 'beta': beta,
                'layout_identity_scope': 'evaluator-only post-reset hidden deck digest; never actor input',
                'counts': dict(counts), 'timings': dict(timings), 'decisions': list(diagnostics),
                'wall_seconds': time.monotonic() - begin, 'native_steps': len(saved['rewards']),
                'return': math.fsum(saved['rewards']),
                'matched_pairs': int(tracker.matched.sum()) // 2 if tracker_ready else 0,
                'success': bool(saved['terminated'] and saved['terminated'][-1]),
                'reference_scope': 'last32 actual selected reveals' if controller == 'last32' else controller}

    try:
        _check(deadline)
        counts['reset_attempted'] += 1
        tick = time.monotonic()
        try:
            obs, info = env.reset(seed=seed)
            counts['reset_returned'] += 1
        finally:
            timings['native_seconds'] += time.monotonic() - tick
        saved['observations'].append(np.asarray(obs).copy())
        _require(info == {}, 'Native reset info must be empty')
        phase = 'layout_identity'
        counts['identity_read_attempted'] += 1
        tick = time.monotonic()
        try:
            layout_sha256 = layout_identity(env, counts=counts)
        finally:
            timings['identity_seconds'] += time.monotonic() - tick
        tick = time.monotonic()
        tracker.reset(obs)
        tracker_ready = True
        if teacher is not None:
            teacher.reset(obs)
        timings['public_seconds'] += time.monotonic() - tick
        state, queries = None, None
        if model is not None:
            phase = 'initialize_memory'
            counts['init_attempted'] += 1
            tick = time.monotonic()
            try:
                state = model.init_state(1, 'cpu')
                counts['init_returned'] += 1
                old._finite(state)
                queries = torch.arange(52, dtype=torch.long)[None]
            finally:
                timings['init_seconds'] += time.monotonic() - tick
        for step in range(104):
            _check(deadline)
            phase = 'predict'
            tick = time.monotonic()
            if model is None:
                probabilities = old._reference_probs(controller, teacher, events)
                timings['public_seconds'] += time.monotonic() - tick
            else:
                counts['predict_attempted'] += 1
                try:
                    with torch.no_grad():
                        logits = model.predict(state, queries)
                        counts['predict_returned'] += 1
                        _require(logits.dtype == torch.float32 and logits.shape == (1, 52, 13)
                                 and torch.isfinite(logits).all(), 'Invalid CPU float32 rank logits')
                        _require(logits.device.type == 'cpu', 'CPU rank logits required')
                        probabilities = logits.softmax(-1)[0].detach().cpu().numpy().astype(np.float64)
                finally:
                    timings['predict_seconds'] += time.monotonic() - tick
            # Preserve the returned softmax values even if a later pure transform
            # or choice fails. Failure payloads may have a longer raw prefix.
            saved['raw_probabilities'].append(probabilities.copy())
            phase = 'transform'
            counts['transform_attempted'] += 1
            tick = time.monotonic()
            try:
                beliefs = transform_probabilities(probabilities, mode=policy, beta=beta if beta is not None else 1.0)
                counts['transform_returned'] += 1
            finally:
                timings['transform_seconds'] += time.monotonic() - tick
            phase = 'decision'
            counts['decision_attempted'] += 1
            tick = time.monotonic()
            try:
                action, picker, decision = tracker.decision(beliefs)
                counts['decision_returned'] += 1
            finally:
                timings['decision_seconds'] += time.monotonic() - tick
            saved['picker_probabilities'].append(picker.copy())
            diagnostics.append(decision)
            phase = 'native_step'
            _check(deadline)
            counts['native_attempted'] += 1
            tick = time.monotonic()
            try:
                after, reward, terminated, truncated, info = env.step(action)
                counts['native_returned'] += 1
            finally:
                timings['native_seconds'] += time.monotonic() - tick
            saved['actions'].append(action)
            saved['observations'].append(np.asarray(after).copy())
            saved['rewards'].append(float(reward))
            saved['terminated'].append(bool(terminated))
            saved['truncated'].append(bool(truncated))
            saved['ranks'].append(int(after[action]))
            phase = 'public_observe'
            tick = time.monotonic()
            try:
                event = tracker.observe(action, reward, after)
                _require(info == {} and type(terminated) in (bool, np.bool_)
                         and type(truncated) in (bool, np.bool_), 'Invalid native flags/info')
                _require(bool(terminated) == bool(tracker.matched.all())
                         and bool(truncated) == (step == 103), 'Native terminal flags disagree with public progress')
                if teacher is not None:
                    teacher.observe(after)
                events.append(event)
            finally:
                timings['public_seconds'] += time.monotonic() - tick
            if model is not None:
                phase = 'write'
                _check(deadline)
                counts['write_attempted'] += 1
                tick = time.monotonic()
                try:
                    with torch.no_grad():
                        state, work = model.write(state, torch.tensor([event.pos]), torch.tensor([event.rank]),
                                                  torch.tensor([True]))
                    counts['write_returned'] += 1
                    old._finite(state)
                    old._finite(work)
                finally:
                    timings['write_seconds'] += time.monotonic() - tick
            _check(deadline)
            if terminated or truncated:
                phase = 'complete'
                return arrays(), receipt('complete')
        raise ValueError('Episode failed to terminate within 104 actions')
    except BaseException as error:
        preserve_failure(error, [lambda e=error: setattr(e, 'partial_card_episode', arrays()),
                                lambda e=error: setattr(e, 'partial_card_receipt', receipt('failed'))])
        raise


def evaluate(protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256,
             checkpoint_map_path, expected_checkpoint_map_sha256, out, *, bindings_path, expected_bindings_sha256,
             calibration_dir, expected_calibration_completed_sha256, root, env_factory=None, model_loader=None):
    """Once-only 3584 episodes; no memory training, retry, scalar fitting, or self gate."""
    begin = time.monotonic()
    deadline = begin + EVALUATION['wall_cap_seconds']
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    phase, current, active_arrays, active_receipt = 'authentication', None, None, None
    counts, episodes, array_bytes, output_bytes = dict.fromkeys(COUNTERS, 0), 0, 0, 0
    restore_count, controller_times, layout_hashes, restored = 0, {}, {}, {}
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    env_factory, model_loader = env_factory or old.native_env, model_loader or old.load_model
    auth_args = (root, protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256,
                 bindings_path, expected_bindings_sha256, checkpoint_map_path, expected_checkpoint_map_sha256)
    calibration_args = (calibration_dir, expected_calibration_completed_sha256,
                        expected_protocol_sha256, expected_inputs_sha256,
                        expected_bindings_sha256, expected_checkpoint_map_sha256)

    def authenticate_all():
        guard()
        authenticated = authenticate(*auth_args)
        guard()
        calibration = authenticate_calibration(*calibration_args)
        guard()
        checkpoints = authenticated[2]
        _require(set(calibration['fits']) == set(checkpoints), 'Calibration and checkpoint membership differ')
        for name, entry in calibration['fits'].items():
            _require(entry['checkpoint_sha256'] == checkpoints[name]['checkpoint_sha256'],
                     'Calibration fitted-checkpoint identity differs')
            _require(type(entry['beta']) in (int, float) and math.isfinite(entry['beta'])
                     and BETA_BOUNDS[0] <= entry['beta'] <= BETA_BOUNDS[1], 'Invalid authenticated calibration beta')
        return (*authenticated, calibration)

    def guard():
        _check(deadline)
        _require(output_bytes <= EVALUATION['output_cap_bytes'], 'Full-output storage cap exceeded')

    def write(path, value):
        nonlocal output_bytes
        _write(path, value)
        output_bytes += path.stat().st_size
        guard()

    def restore_runtime():
        torch.set_num_threads(prior_threads)
        torch.use_deterministic_algorithms(prior_deterministic)

    try:
        write(out / 'started.json', {'status': 'started', 'automatic_retry': False, 'version': VERSION,
              'protocol_sha256': expected_protocol_sha256, 'inputs_sha256': expected_inputs_sha256,
              'checkpoint_map_sha256': expected_checkpoint_map_sha256,
              'bindings_sha256': expected_bindings_sha256, 'evaluation': EVALUATION,
              'calibration_completed_sha256': expected_calibration_completed_sha256, 'tracker_policy': 'C'})
        protocol, inputs, checkpoints, fits, calibration = authenticate_all()
        write(out / 'all-fits-ready.json', {'status': 'complete', 'fits': checkpoints,
              'elapsed_seconds': time.monotonic() - begin,
              'calibration_completed_sha256': expected_calibration_completed_sha256,
              'calibration_fits': calibration['fits'],
              'scope': 'all eighteen final fit trees and scalar calibration artifacts authenticated before first model restore/reset'})
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for controller in CONTROLLERS:
            guard()
            phase = 'controller_restore'
            tick = time.monotonic()
            model = None
            if controller not in REFERENCES:
                model = model_loader(fits[controller]['entry'], fits[controller]['receipt'], root)
                restore_count += 1
                restored[controller] = {'checkpoint_sha256': checkpoints[controller]['checkpoint_sha256'],
                                        'weights_sha256': old._weights_hash(model.state_dict())}
                _require(restored[controller]['weights_sha256'] == fits[controller]['receipt']['final_weights_sha256'],
                         'Restored checkpoint tensor identity mismatch')
            restore_seconds = time.monotonic() - tick
            selected_policies = ('baseline',) if controller in REFERENCES else POLICIES
            entry = None if controller in REFERENCES else calibration['fits'][controller]
            summaries = {policy: [] for policy in selected_policies}
            for policy in selected_policies:
                (out / 'controllers' / controller / policy / 'episodes').mkdir(parents=True)
            for index, case in enumerate(inputs['evaluation']):
                policy_order = ('baseline',) if controller in REFERENCES else case['policy_order']
                for policy in policy_order:
                    beta = entry['beta'] if policy == 'temperature' else None
                    guard()
                    phase, current = 'episode', {'controller': controller, 'policy': policy,
                                                'case': index, 'seed': case['seed'], 'beta': beta}
                    active_arrays, active_receipt = None, None
                    episode_start = time.monotonic()
                    env, error = None, None
                    construction_start = time.monotonic()
                    try:
                        env = env_factory()
                        construction_seconds = time.monotonic() - construction_start
                        arrays, receipt = evaluate_episode(env, controller=controller, policy=policy,
                                                          seed=case['seed'], beta=beta, model=model, deadline=deadline)
                        active_arrays, active_receipt = arrays, receipt
                    except BaseException as caught:
                        error = caught
                        raise
                    finally:
                        closing = time.monotonic()
                        if env is not None:
                            try:
                                env.close()
                            except BaseException as close_error:
                                if error is None:
                                    raise
                                old._note(error, f'Environment cleanup failed: {close_error!r}')
                        close_seconds = time.monotonic() - closing
                    known = layout_hashes.setdefault(index, receipt['layout_sha256'])
                    _require(known == receipt['layout_sha256'], 'Paired seed produced a different native layout')
                    storing = time.monotonic()
                    folder = out / 'controllers' / controller / policy / 'episodes'
                    npz = folder / f'{index:03d}.npz'
                    with npz.open('xb') as handle:
                        np.savez_compressed(handle, **arrays)
                    output_bytes += npz.stat().st_size
                    payload_bytes = sum(value.nbytes for value in arrays.values())
                    array_bytes += payload_bytes
                    npz_sha = _sha(npz)
                    receipt.update(index=index, npz_sha256=npz_sha, npz_bytes=npz.stat().st_size,
                                   array_bytes=payload_bytes, construction_seconds=construction_seconds,
                                   close_seconds=close_seconds, serialization_seconds=time.monotonic() - storing,
                                   whole_episode_seconds=time.monotonic() - episode_start,
                                   whole_episode_scope='construction through close, NPZ write/hash; JSON receipt write is outer work',
                                   checkpoint_sha256=None if model is None else checkpoints[controller]['checkpoint_sha256'],
                                   protocol_sha256=expected_protocol_sha256, inputs_sha256=expected_inputs_sha256,
                                   calibration_completed_sha256=expected_calibration_completed_sha256,
                                   calibration_receipt_sha256=None if entry is None else entry['receipt_sha256'])
                    write(folder / f'{index:03d}.json', receipt)
                    for key in COUNTERS:
                        counts[key] += receipt['counts'][key]
                    episodes += 1
                    summaries[policy].append(receipt)
                    # This episode is now charged in completed totals. A later
                    # cap/finalization failure must not charge it again as partial.
                    active_arrays, active_receipt = None, None
                    _require(counts['native_attempted'] <= 372736 and counts['predict_attempted'] <= 359424
                             and counts['write_attempted'] <= 359424, 'Frozen action/model call cap exceeded')
                    guard()
            if model is not None:
                _require(old._weights_hash(model.state_dict()) == restored[controller]['weights_sha256'],
                         'Frozen model weights changed during evaluation')
            controller_times[controller] = time.monotonic() - tick
            for policy, rows in summaries.items():
                write(out / 'controllers' / controller / policy / 'completed.json',
                      {'status': 'complete', 'controller': controller, 'policy': policy, 'episodes': len(rows),
                       'whole_episode_seconds': math.fsum(row['whole_episode_seconds'] for row in rows),
                       'native_steps': sum(row['native_steps'] for row in rows),
                       'mean_return': math.fsum(row['return'] for row in rows) / 64,
                       'successes': sum(row['success'] for row in rows),
                       'tracker_policy': 'C', 'beta': entry['beta'] if policy == 'temperature' else None,
                       'calibration_completed_sha256': expected_calibration_completed_sha256,
                       'calibration_receipt_sha256': None if entry is None else entry['receipt_sha256'],
                       'controller_shared_restore_seconds': restore_seconds,
                       'restore_cost_scope': 'one shared restore per learned controller, repeated metadata; do not sum over policies'})
        phase = 'final_binding'
        authenticate_all()
        _require(episodes == 3584 and restore_count == 18 and counts['identity_read_returned'] == 3584,
                 'Incomplete episode/restore/layout coverage')
        phase = 'runtime_cleanup'
        restore_runtime()
        phase = 'final_hashing'
        files = {}
        hash_begin = time.monotonic()
        for path in sorted(out.rglob('*')):
            if path.is_file():
                guard()
                files[str(path.relative_to(out))] = {'sha256': _sha(path), 'bytes': path.stat().st_size}
        output_bytes = sum(item['bytes'] for item in files.values())
        guard()
        result = {'status': 'complete', 'version': VERSION, 'controllers': 20, 'policies': list(POLICIES),
                  'episodes': episodes, 'started_sha256': _sha(out / 'started.json'),
                  'protocol_sha256': expected_protocol_sha256, 'inputs_sha256': expected_inputs_sha256,
                  'bindings_sha256': expected_bindings_sha256, 'checkpoint_map_sha256': expected_checkpoint_map_sha256,
                  'calibration_completed_sha256': expected_calibration_completed_sha256, 'tracker_policy': 'C',
                  'lineage': protocol['lineage'], 'counts': counts, 'model_restores': restore_count,
                  'restored_models': restored, 'new_fits': 0, 'new_optimizer_steps': 0, 'native_replay_calls': 0,
                  'layout_sha256_by_case': layout_hashes, 'distinct_layouts': len(set(layout_hashes.values())),
                  'controller_wall_seconds': controller_times, 'uncompressed_array_bytes': array_bytes,
                  'payload_bytes': output_bytes, 'final_hashing_seconds': time.monotonic() - hash_begin,
                  'wall_seconds': time.monotonic() - begin, 'files': files,
                  'scope': 'fresh fixed-checkpoint probability intervention with unchanged C; no self-qualification or original-result revision'}
        write(out / 'completed.json', result)
        guard()
        return result
    except BaseException as error:
        if not hasattr(error, 'partial_card_episode') and active_arrays is not None:
            error.partial_card_episode = active_arrays
            error.partial_card_receipt = active_receipt

        def save_partial():
            if hasattr(error, 'partial_card_episode'):
                with (out / 'partial-episode.npz').open('xb') as handle:
                    np.savez_compressed(handle, **error.partial_card_episode)

        preserve_failure(error, [restore_runtime,
            lambda: (out / 'completed.json').rename(out / 'invalid-completion.json')
            if (out / 'completed.json').exists() else None, save_partial,
            lambda: _write(out / 'failed.json', {'status': 'failed', 'phase': phase, 'current': current,
                'error': repr(error), 'partial_receipt': getattr(error, 'partial_card_receipt', None),
                'completed_episode_counts': counts, 'completed_episodes': episodes,
                'model_restores': restore_count, 'uncompressed_array_bytes': array_bytes,
                'tracked_output_bytes': output_bytes, 'wall_seconds': time.monotonic() - begin,
                'automatic_retry': False})])
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('protocol', 'inputs', 'checkpoint-map', 'bindings', 'calibration-dir', 'root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('protocol', 'inputs', 'checkpoint-map', 'bindings'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    parser.add_argument('--expected-calibration-completed-sha256', required=True)
    args = parser.parse_args()
    evaluate(args.protocol, args.expected_protocol_sha256, args.inputs, args.expected_inputs_sha256,
             args.checkpoint_map, args.expected_checkpoint_map_sha256, args.out,
             bindings_path=args.bindings, expected_bindings_sha256=args.expected_bindings_sha256,
             calibration_dir=args.calibration_dir, expected_calibration_completed_sha256=args.expected_calibration_completed_sha256,
             root=args.root)
