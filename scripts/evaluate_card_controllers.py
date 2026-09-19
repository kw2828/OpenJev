"""Fixed-checkpoint, three-picker native evaluation. No work runs at import.

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
from card_memory_common import preserve_failure, runtime

from openjev.research.card_memory_picker import CardPolicyTracker
from openjev.research.card_memory_task import PublicTeacher

VERSION = 'card-controllers-v1'
POLICIES = ('A', 'B', 'C')
MODES, REFERENCES = old.MODES, old.REFERENCES
CONTROLLERS = tuple(f'{mode}-pair{pair}' for mode in MODES for pair in range(3)) + REFERENCES
EVALUATION = {'episodes': 64, 'max_actions': 104, 'controllers': 20,
              'policies': list(POLICIES), 'wall_cap_seconds': 1800,
              'output_cap_bytes': 6_000_000_000}
LAYOUT_PREFIX = b'card-layout-int64le-v1\0'
LINEAGE = {'old_bindings', 'old_protocol', 'old_inputs', 'training_completed',
           'training_boundary', 'independent_review', 'checkpoint_map'}
COUNTERS = ('reset_attempted', 'reset_returned', 'identity_read_attempted', 'identity_read_returned',
            'init_attempted', 'init_returned', 'predict_attempted', 'predict_returned',
            'decision_attempted', 'decision_returned', 'native_attempted', 'native_returned',
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


def evaluate_episode(env, *, controller, policy, seed, model=None, deadline=None):
    """One public episode; caller owns close and serialized whole-episode costs.

Failure attaches partial_card_episode/partial_card_receipt, including every
returned native transition and attempted/completed model call separately.
"""
    begin = time.monotonic()
    _require(controller in CONTROLLERS and policy in POLICIES, 'Unknown controller/policy')
    _require(type(seed) is int and 0 <= seed < 2**32, 'Explicit uint32 reset seed required')
    _require((controller in REFERENCES) == (model is None), 'Learned/reference model mismatch')
    saved = {name: [] for name in ('observations', 'actions', 'ranks', 'rewards', 'terminated',
                                  'truncated', 'raw_probabilities', 'picker_probabilities')}
    counts = dict.fromkeys(COUNTERS, 0)
    timings = dict.fromkeys(('native_seconds', 'identity_seconds', 'init_seconds', 'predict_seconds',
                            'write_seconds', 'public_seconds', 'decision_seconds'), 0.0)
    diagnostics, layout_sha256, tracker_ready = [], None, False
    tracker = CardPolicyTracker(policy)
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
                'phase': phase, 'layout_sha256': layout_sha256,
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
            phase = 'decision'
            counts['decision_attempted'] += 1
            tick = time.monotonic()
            try:
                action, picker, decision = tracker.decision(probabilities)
                counts['decision_returned'] += 1
            finally:
                timings['decision_seconds'] += time.monotonic() - tick
            saved['raw_probabilities'].append(probabilities.copy())
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


def _binding(root, descriptor):
    _require(set(descriptor) == {'path', 'sha256'}, 'Exact external lineage descriptor required')
    path = _path(root, descriptor['path'])
    _require(_sha(path) == descriptor['sha256'], 'External lineage hash mismatch')
    return path, _read(path)


def _seed_inventory(value):
    """Read literal old seeds only; never derive or sample a stream."""
    if isinstance(value, dict):
        result = {v for k, v in value.items() if k in ('seed', 'behavior_seed') and type(v) is int}
        return result | set().union(*(_seed_inventory(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_seed_inventory(v) for v in value))
    return set()


def authenticate(protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256,
                 checkpoint_map_path, expected_checkpoint_map_sha256, *, expected_bindings_sha256,
                 root, deadline=None):
    """Full entry/exit authentication. No model construction, environment, or RNG."""
    root = Path(root).resolve()
    for path, expected in ((protocol_path, expected_protocol_sha256), (inputs_path, expected_inputs_sha256),
                           (checkpoint_map_path, expected_checkpoint_map_sha256)):
        _check(deadline)
        _require(Path(path).resolve().is_relative_to(root) and _sha(path) == expected,
                 'External protocol/input/checkpoint-map hash mismatch')
    protocol, inputs, checkpoints = map(_read, (protocol_path, inputs_path, checkpoint_map_path))
    _require(protocol['version'] == inputs['version'] == VERSION and protocol['evaluation'] == EVALUATION,
             'Frozen controller evaluation scope mismatch')
    binding_path = _path(root, protocol['bindings_file'])
    _require(_sha(binding_path) == expected_bindings_sha256, 'External source binding hash mismatch')
    bindings = _read(binding_path)
    _require(bindings['version'] == VERSION and bindings['runtime'] == runtime(), 'Runtime/binding version mismatch')
    _require({'scripts/evaluate_card_controllers.py', 'src/openjev/research/card_memory_picker.py'}
             <= set(bindings['files']), 'New driver and picker must be source-bound')
    for path in (protocol_path, inputs_path):
        _require(bindings['files'].get(str(Path(path).resolve().relative_to(root))) == _sha(path),
                 'New protocol/input bytes must be source-bound')
    _require(set(protocol['lineage']) == LINEAGE, 'Exact seven lineage descriptors required')
    lineage = {key: _binding(root, value) for key, value in protocol['lineage'].items()}
    _require(lineage['checkpoint_map'][0] == Path(checkpoint_map_path).resolve()
             and protocol['lineage']['checkpoint_map']['sha256'] == expected_checkpoint_map_sha256,
             'Lineage checkpoint-map mismatch')
    old_bindings = lineage['old_bindings'][1]
    _require(old_bindings['version'] == 'card-memory-pilot-v1' and len(old_bindings['files']) == 78
             and old_bindings['runtime'] == runtime(), 'Original 78-source/runtime boundary mismatch')
    for key in ('old_protocol', 'old_inputs'):
        path, _ = lineage[key]
        _require(old_bindings['files'].get(str(path.relative_to(root))) == _sha(path), 'Old metadata is not source-bound')
    for source in (old_bindings, bindings):
        for relative, expected in source['files'].items():
            _check(deadline)
            _require(_sha(_path(root, relative)) == expected, f'Bound source changed: {relative}')
    train_path, train = lineage['training_completed']
    boundary_path, boundary = lineage['training_boundary']
    _require(train['status'] == boundary['status'] == 'complete' and train['fits'] == 18
             and train['updates'] == 2304 and train['native_evaluation_calls'] == 0
             and boundary['development_or_native_evaluations'] == 0,
             'Original eighteen-fit training boundary incomplete')
    _require(train['checkpoint_map_sha256'] == boundary['checkpoint_map_sha256'] == expected_checkpoint_map_sha256
             and train['training_completed_sha256'] == _sha(boundary_path)
             and train['started_sha256'] == _sha(train_path.with_name('started.json')),
             'Original training provenance mismatch')
    _require(len(boundary['fits']) == 18 and {row['id'] for row in boundary['fits']} == set(checkpoints),
             'Original paired fit membership mismatch')
    for row in boundary['fits']:
        _require(all(row.get(key) == value for key, value in checkpoints[row['id']].items()),
                 'Training boundary checkpoint identity mismatch')
    review = lineage['independent_review'][1]
    _require(review['status'] == 'complete' and review['bindings']['source_count'] == 78,
             'Original independent review incomplete')
    for field, key in (('training_completed_sha256', 'training_completed'),
                       ('source_bindings_sha256', 'old_bindings'), ('protocol_sha256', 'old_protocol')):
        _require(review['bindings'][field] == protocol['lineage'][key]['sha256'], 'Original review lineage mismatch')
    cases = inputs['evaluation']
    _require(len(cases) == 64, 'Exact 64 fresh cases required')
    for index, case in enumerate(cases):
        rotation = index % 3
        _require(set(case) == {'seed', 'policy_order'} and type(case['seed']) is int
                 and 0 <= case['seed'] < 2**32
                 and case['policy_order'] == list(POLICIES[rotation:] + POLICIES[:rotation]),
                 'Invalid seed or balanced policy schedule')
    seeds = {case['seed'] for case in cases}
    _require(len(seeds) == 64 and seeds.isdisjoint(_seed_inventory(lineage['old_inputs'][1]) | {0, 410}),
             'Fresh seeds collide with old declared inputs/constructor engineering streams')
    fits = old.authenticate_fits(checkpoints, root)
    _check(deadline)
    return protocol, inputs, checkpoints, fits


def evaluate(protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256,
             checkpoint_map_path, expected_checkpoint_map_sha256, out, *, expected_bindings_sha256,
             root, env_factory=None, model_loader=None):
    """Once-only 3x20x64 evaluation; no training, automatic retry, or self gate."""
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
    paths = (protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256,
             checkpoint_map_path, expected_checkpoint_map_sha256)
    auth_options = {'expected_bindings_sha256': expected_bindings_sha256, 'root': root, 'deadline': deadline}

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
              'bindings_sha256': expected_bindings_sha256, 'evaluation': EVALUATION})
        protocol, inputs, checkpoints, fits = authenticate(*paths, **auth_options)
        write(out / 'all-fits-ready.json', {'status': 'complete', 'fits': checkpoints,
              'elapsed_seconds': time.monotonic() - begin,
              'scope': 'all eighteen final fit trees authenticated before first model restore/reset'})
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
            summaries = {policy: [] for policy in POLICIES}
            for policy in POLICIES:
                (out / 'controllers' / controller / policy / 'episodes').mkdir(parents=True)
            for index, case in enumerate(inputs['evaluation']):
                for policy in case['policy_order']:
                    guard()
                    phase, current = 'episode', {'controller': controller, 'policy': policy,
                                                'case': index, 'seed': case['seed']}
                    active_arrays, active_receipt = None, None
                    episode_start = time.monotonic()
                    env, error = None, None
                    construction_start = time.monotonic()
                    try:
                        env = env_factory()
                        construction_seconds = time.monotonic() - construction_start
                        arrays, receipt = evaluate_episode(env, controller=controller, policy=policy,
                                                          seed=case['seed'], model=model, deadline=deadline)
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
                                   protocol_sha256=expected_protocol_sha256, inputs_sha256=expected_inputs_sha256)
                    write(folder / f'{index:03d}.json', receipt)
                    for key in COUNTERS:
                        counts[key] += receipt['counts'][key]
                    episodes += 1
                    summaries[policy].append(receipt)
                    # This episode is now charged in completed totals. A later
                    # cap/finalization failure must not charge it again as partial.
                    active_arrays, active_receipt = None, None
                    _require(counts['native_attempted'] <= 399360 and counts['predict_attempted'] <= 359424
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
                       'controller_shared_restore_seconds': restore_seconds,
                       'restore_cost_scope': 'one shared restore per learned controller, repeated metadata; do not sum over policies'})
        phase = 'final_binding'
        authenticate(*paths, **auth_options)
        _require(episodes == 3840 and restore_count == 18 and counts['identity_read_returned'] == 3840,
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
                  'lineage': protocol['lineage'], 'counts': counts, 'model_restores': restore_count,
                  'restored_models': restored, 'new_fits': 0, 'new_optimizer_steps': 0, 'native_replay_calls': 0,
                  'layout_sha256_by_case': layout_hashes, 'distinct_layouts': len(set(layout_hashes.values())),
                  'controller_wall_seconds': controller_times, 'uncompressed_array_bytes': array_bytes,
                  'payload_bytes': output_bytes, 'final_hashing_seconds': time.monotonic() - hash_begin,
                  'wall_seconds': time.monotonic() - begin, 'files': files,
                  'scope': 'fresh fixed-checkpoint controller evaluation; no self-qualification or original-result revision'}
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
    for name in ('protocol', 'inputs', 'checkpoint-map', 'root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('protocol', 'inputs', 'checkpoint-map', 'bindings'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    args = parser.parse_args()
    evaluate(args.protocol, args.expected_protocol_sha256, args.inputs, args.expected_inputs_sha256,
             args.checkpoint_map, args.expected_checkpoint_map_sha256, args.out,
             expected_bindings_sha256=args.expected_bindings_sha256, root=args.root)
