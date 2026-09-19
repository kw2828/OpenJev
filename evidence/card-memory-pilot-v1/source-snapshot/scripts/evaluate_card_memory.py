"""Once-only, public-input native ConcentrationHard evaluation.

No evaluation runs at import. The caller supplies externally authenticated
protocol, source bindings, inputs and all eighteen completed fit identities.
No optimizer is restored and no random stream is allocated here. Model
construction is isolated and overwritten by the authenticated final tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from card_memory_common import runtime

from openjev.research.card_memory_task import PublicCardTracker, PublicTeacher

MODES = ('delta', 'gated_delta', 'kalman', 'innovation_local', 'innovation_matched', 'gru')
REFERENCES = ('exact', 'last32')
EVALUATION = {'episodes': 64, 'max_actions': 104, 'controllers': 20, 'wall_cap_seconds': 3600}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _note(error, message):
    add = getattr(error, 'add_note', None)
    if callable(add):
        add(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def _check(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError('Card evaluation wall deadline exceeded')


def _path(root, relative):
    _require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
             and '..' not in Path(relative).parts, 'Expected a repository-relative path')
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), 'Path escapes repository')
    return path


def _weights_hash(mapping):
    digest = hashlib.sha256()
    for name, value in sorted(mapping.items()):
        _require(isinstance(value, torch.Tensor) and torch.isfinite(value).all(), 'Invalid weight tensor')
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _finite(value):
    if isinstance(value, torch.Tensor):
        _require(torch.isfinite(value).all(), 'Nonfinite model state/diagnostic')
    elif isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif value is not None:
        raise ValueError('Unsupported model state/diagnostic field')


def _reference_probs(controller, teacher, events):
    if controller == 'exact':
        return teacher.probabilities()
    probabilities = np.full((52, 13), 1 / 13, dtype=np.float64)
    for event in events:  # Later selected events overwrite earlier entries.
        probabilities[event.pos] = 0
        probabilities[event.pos, event.rank] = 1
    return probabilities


def evaluate_episode(env, *, controller, seed, model=None, deadline=None):
    """Evaluate one episode; caller owns env cleanup and artifact serialization.

    Only env.reset/step are used. Failure carries partial_card_episode and
    partial_card_receipt, including returned native transitions even when later
    public validation/model write fails. No hidden state or get_state is read.
    """
    begin = time.monotonic()
    _require(type(seed) is int and 0 <= seed < 2**32, 'Explicit uint32 reset seed required')
    _require((controller in REFERENCES) == (model is None), 'Learned/reference model mismatch')
    names = ('actions', 'ranks', 'rewards', 'terminated', 'truncated', 'observations',
             'raw_probabilities', 'picker_probabilities')
    saved = {name: [] for name in names}
    counts = {key: 0 for key in ('reset_attempted', 'reset_returned', 'native_attempted', 'native_returned',
                                'predict_attempted', 'predict_returned', 'write_attempted', 'write_returned')}
    timings = {key: 0.0 for key in ('native_seconds', 'predict_seconds', 'write_seconds', 'public_seconds')}
    tracker = PublicCardTracker()
    teacher = PublicTeacher() if controller == 'exact' else None
    events = deque(maxlen=32)
    phase = 'reset'

    def arrays():
        result = {}
        for name, values in saved.items():
            dtype = (np.bool_ if name in ('terminated', 'truncated') else
                     np.int64 if name in ('actions', 'ranks', 'observations') else np.float64)
            shape = (52,) if name == 'observations' else (52, 13) if name.endswith('probabilities') else ()
            result[name] = np.asarray(values, dtype=dtype).reshape((-1, *shape))
        return result

    def receipt(status):
        return {'status': status, 'controller': controller, 'seed': seed, 'phase': phase,
                'counts': dict(counts), 'timings': dict(timings),
                'wall_seconds': time.monotonic() - begin,
                'native_steps': len(saved['rewards']), 'return': math.fsum(saved['rewards']),
                'matched_pairs': int(tracker.matched.sum()) // 2 if saved['observations'] else 0,
                'success': bool(saved['terminated'] and saved['terminated'][-1]),
                'tracker_scope': 'full public seen/matched/current-frame metadata for every controller',
                'reference_scope': 'last32 counts selected reveals, not all visible cards' if controller == 'last32' else controller}

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
        tracker.reset(obs)
        if teacher is not None:
            teacher.reset(obs)
        state = None if model is None else model.init_state(1, 'cpu')
        if model is not None:
            _finite(state)
        queries = torch.arange(52, dtype=torch.long)[None] if model is not None else None
        for step in range(104):
            _check(deadline)
            phase = 'predict'
            tick = time.monotonic()
            if model is None:
                probabilities = _reference_probs(controller, teacher, events)
                timings['public_seconds'] += time.monotonic() - tick
            else:
                counts['predict_attempted'] += 1
                try:
                    with torch.no_grad():
                        logits = model.predict(state, queries)
                        counts['predict_returned'] += 1
                        _require(logits.shape == (1, 52, 13) and torch.isfinite(logits).all(), 'Invalid rank logits')
                        probabilities = logits.softmax(-1)[0].detach().cpu().numpy().astype(np.float64)
                finally:
                    timings['predict_seconds'] += time.monotonic() - tick
            tick = time.monotonic()
            picker = tracker.probabilities(probabilities)
            action = tracker.choose(probabilities)
            timings['public_seconds'] += time.monotonic() - tick
            saved['raw_probabilities'].append(probabilities.copy())
            saved['picker_probabilities'].append(picker)
            phase = 'native_step'
            _check(deadline)
            counts['native_attempted'] += 1
            tick = time.monotonic()
            try:
                after, reward, terminated, truncated, info = env.step(action)
                counts['native_returned'] += 1
            finally:
                timings['native_seconds'] += time.monotonic() - tick
            # Preserve raw public return before tracker/model validation.
            saved['actions'].append(action)
            saved['observations'].append(np.asarray(after).copy())
            saved['rewards'].append(float(reward))
            saved['terminated'].append(bool(terminated))
            saved['truncated'].append(bool(truncated))
            saved['ranks'].append(int(after[action]))
            phase = 'public_observe'
            tick = time.monotonic()
            event = tracker.observe(action, reward, after)
            _require(info == {} and type(terminated) in (bool, np.bool_) and type(truncated) in (bool, np.bool_),
                     'Invalid native flags/info')
            _require(bool(terminated) == bool(tracker.matched.all())
                     and bool(truncated) == (step == 103), 'Native terminal flags disagree with public progress')
            if teacher is not None:
                teacher.observe(after)
            events.append(event)
            timings['public_seconds'] += time.monotonic() - tick
            phase = 'write'
            if model is not None:
                _check(deadline)
                counts['write_attempted'] += 1
                tick = time.monotonic()
                try:
                    with torch.no_grad():
                        state, diagnostics = model.write(state, torch.tensor([event.pos]), torch.tensor([event.rank]),
                                                         torch.tensor([True]))
                    counts['write_returned'] += 1
                    _finite(state)
                    _finite(diagnostics)
                finally:
                    timings['write_seconds'] += time.monotonic() - tick
            obs = after
            _check(deadline)
            if terminated or truncated:
                phase = 'complete'
                return arrays(), receipt('complete')
        raise ValueError('Native episode failed to terminate within 104 actions')
    except BaseException as error:
        for name, producer in (('partial_card_episode', arrays), ('partial_card_receipt', lambda: receipt('failed'))):
            try:
                setattr(error, name, producer())
            except BaseException as secondary:  # noqa: BLE001 - preserve the original failure.
                _note(error, f'Partial {name} preservation failed: {secondary!r}')
        raise


def authenticate_fits(checkpoint_map, root):
    expected = {f'{mode}-pair{pair}' for mode in MODES for pair in range(3)}
    _require(set(checkpoint_map) == expected, 'Exact eighteen fit identities required before evaluation')
    result = {}
    for name, entry in checkpoint_map.items():
        _require(set(entry) == {'mode', 'pair', 'checkpoint_path', 'checkpoint_sha256', 'completed_path', 'completed_sha256'},
                 'Exact checkpoint-map fields required')
        _require(entry['mode'] in MODES and type(entry['pair']) is int and 0 <= entry['pair'] < 3
                 and name == f"{entry['mode']}-pair{entry['pair']}", 'Fit mode/pair identity mismatch')
        checkpoint, completed = (_path(root, entry[k]) for k in ('checkpoint_path', 'completed_path'))
        _require(checkpoint.name == 'final-checkpoint.pt' and completed.name == 'completed.json'
                 and checkpoint.parent == completed.parent, 'Fit artifact path mismatch')
        _require(_sha(checkpoint) == entry['checkpoint_sha256'] and _sha(completed) == entry['completed_sha256'],
                 'External final fit hash mismatch')
        receipt = _read(completed)
        _require(receipt['status'] == 'complete' and receipt['evaluation_calls'] == 0, 'Incomplete/evaluated fit')
        _require(receipt['counts']['successful_updates'] == receipt['recipe']['expected_updates']
                 and receipt['counts']['completed_epochs'] == receipt['recipe']['epochs'], 'Incomplete fit update schedule')
        actual = {p.name for p in completed.parent.iterdir() if p.is_file()}
        _require(actual == set(receipt['files']) | {'completed.json'}
                 and 'final-checkpoint.pt' in receipt['files'], 'Fit membership/failure marker mismatch')
        for filename, binding in receipt['files'].items():
            _require(Path(filename).name == filename, 'Fit member path must be a basename')
            file = completed.parent / filename
            _require(_sha(file) == binding['sha256'] and file.stat().st_size == binding['bytes'], 'Fit member changed')
        result[name] = {'entry': dict(entry), 'receipt': receipt}
    return result


def load_model(entry, receipt, root):
    from openjev.research.card_associative_memory import CardAssociativeMemory
    checkpoint = torch.load(_path(root, entry['checkpoint_path']), map_location='cpu', weights_only=True)
    _require(checkpoint['model_class'] == 'CardAssociativeMemory' and checkpoint['resume_authorized'] is False,
             'Unexpected deployment checkpoint')
    _require(checkpoint['configuration']['mode'] == entry['mode'], 'Checkpoint mode mismatch')
    for key in ('identity', 'recipe', 'counts'):
        _require(json.dumps(checkpoint[key], sort_keys=True) == json.dumps(receipt[key], sort_keys=True),
                 f'Checkpoint/fit receipt {key} mismatch')
    _require(_weights_hash(checkpoint['weights']) == receipt['final_weights_sha256'], 'Final tensor hash mismatch')
    _require(all(value.dtype == torch.float32 and value.device.type == 'cpu'
                 for value in checkpoint['weights'].values()), 'Deployment weights must be CPU float32')
    # Construction randomness is unused, isolated, and overwritten, not a seed allocation.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = CardAssociativeMemory(entry['mode'])
    _require(model.configuration() == checkpoint['configuration'], 'Actual class configuration mismatch')
    model.load_state_dict(checkpoint['weights'], strict=True)
    _require(_weights_hash(model.state_dict()) == receipt['final_weights_sha256'], 'Restored tensors differ')
    model.eval()
    return model


def native_env():
    from popgym.envs.concentration import ConcentrationHard
    return ConcentrationHard()


def evaluate(protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256,
             checkpoint_map, out, *, expected_bindings_sha256, root, deadline=None,
             env_factory=None, model_loader=None):
    """Authenticate all fits, restore each once, then evaluate paired 20x64 games."""
    begin = time.monotonic()
    deadline = min(begin + 3600, deadline) if deadline is not None else begin + 3600
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    phase, current, controller_times = 'authentication', None, {}
    active_arrays, active_receipt = None, None
    env_factory, model_loader = env_factory or native_env, model_loader or load_model
    previous_threads, previous_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        _write(out / 'started.json', {'status': 'started', 'automatic_retry': False,
               'protocol_sha256': expected_protocol_sha256, 'inputs_sha256': expected_inputs_sha256,
               'bindings_sha256': expected_bindings_sha256, 'wall_cap_seconds': 3600})
        _check(deadline)
        _require(_sha(protocol_path) == expected_protocol_sha256 and _sha(inputs_path) == expected_inputs_sha256,
                 'Protocol/input external hash mismatch')
        protocol, inputs = _read(protocol_path), _read(inputs_path)
        _require(protocol['evaluation'] == EVALUATION, 'Frozen evaluation scope mismatch')
        bindings_path = _path(root, protocol['bindings_file'])
        _require(_sha(bindings_path) == expected_bindings_sha256, 'External source binding hash mismatch')
        bindings = _read(bindings_path)
        _require(bindings['runtime'] == runtime(), 'Runtime differs from frozen source bindings')
        for path in (Path(protocol_path).resolve(), Path(inputs_path).resolve()):
            relative = str(path.relative_to(root))
            _require(bindings['files'].get(relative) == _sha(path), 'Protocol/inputs must be source-bound')
        for path, expected in bindings['files'].items():
            _check(deadline)
            _require(_sha(_path(root, path)) == expected, f'Bound source changed: {path}')
        cases = inputs['evaluation']
        _require(len(cases) == 64 and all(set(case) == {'seed'} and type(case['seed']) is int
                 and 0 <= case['seed'] < 2**32 for case in cases)
                 and len({case['seed'] for case in cases}) == 64, 'Exact 64 distinct explicit evaluation seeds required')
        fits = authenticate_fits(checkpoint_map, root)
        _check(deadline)
        _write(out / 'all-fits-ready.json', {'status': 'complete', 'fits': checkpoint_map,
               'unix_time': time.time(), 'elapsed_seconds': time.monotonic() - begin,
               'scope': 'all eighteen final fit files authenticated before any evaluation model restore/reset'})
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for name in [f'{mode}-pair{pair}' for mode in MODES for pair in range(3)] + list(REFERENCES):
            _check(deadline)
            phase = 'controller'
            tick = time.monotonic()
            directory = out / 'controllers' / name
            (directory / 'episodes').mkdir(parents=True)
            restore_begin = time.monotonic()
            model = None if name in REFERENCES else model_loader(fits[name]['entry'], fits[name]['receipt'], root)
            restore_seconds = time.monotonic() - restore_begin
            episode_receipts, io_seconds, close_seconds = [], 0.0, 0.0
            for index, case in enumerate(cases):
                _check(deadline)
                current = {'controller': name, 'case': index, 'seed': case['seed']}
                active_arrays, active_receipt = None, None
                folder = directory / 'episodes'
                env = env_factory()
                error = None
                try:
                    arrays, receipt = evaluate_episode(env, controller=name, seed=case['seed'], model=model, deadline=deadline)
                    active_arrays, active_receipt = arrays, receipt
                except BaseException as caught:
                    error = caught
                    raise
                finally:
                    closing = time.monotonic()
                    try:
                        env.close()
                    except BaseException as close_error:
                        if error is None:
                            raise
                        _note(error, f'Environment cleanup failed: {close_error!r}')
                    finally:
                        close_seconds += time.monotonic() - closing
                storing = time.monotonic()
                npz = folder / f'{index:03d}.npz'
                with npz.open('xb') as handle:
                    np.savez_compressed(handle, **arrays)
                receipt.update(index=index, npz_sha256=_sha(npz), npz_bytes=npz.stat().st_size,
                               checkpoint_sha256=None if model is None else checkpoint_map[name]['checkpoint_sha256'],
                               protocol_sha256=expected_protocol_sha256, inputs_sha256=expected_inputs_sha256)
                _write(folder / f'{index:03d}.json', receipt)
                io_seconds += time.monotonic() - storing
                episode_receipts.append(receipt)
                _check(deadline)
            controller_times[name] = time.monotonic() - tick
            _write(directory / 'completed.json', {'status': 'complete', 'controller': name, 'episodes': 64,
                   'wall_seconds': controller_times[name], 'restore_seconds': restore_seconds,
                   'episode_io_seconds': io_seconds, 'close_seconds': close_seconds,
                   'episode_seconds': math.fsum(r['wall_seconds'] for r in episode_receipts),
                   'native_steps': sum(r['native_steps'] for r in episode_receipts),
                   'mean_return': math.fsum(r['return'] for r in episode_receipts) / 64,
                   'successes': sum(r['success'] for r in episode_receipts),
                   'timing_scope': 'whole controller includes restore, env construction/reset/step, public logic, inference, episode I/O and close; receipt write is outer work'})
            _check(deadline)
        phase = 'final_binding'
        authenticate_fits(checkpoint_map, root)
        for path, expected in bindings['files'].items():
            _check(deadline)
            _require(_sha(_path(root, path)) == expected, 'Source/input changed during evaluation')
        _require(_sha(bindings_path) == expected_bindings_sha256, 'Source bindings changed during evaluation')
        _require(bindings['runtime'] == runtime(), 'Runtime changed during evaluation')
        files = {str(p.relative_to(out)): {'sha256': _sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(out.rglob('*')) if p.is_file()}
        _check(deadline)
        result = {'status': 'complete', 'controllers': 20, 'episodes': 1280,
                  'protocol_sha256': expected_protocol_sha256, 'inputs_sha256': expected_inputs_sha256,
                  'bindings_sha256': expected_bindings_sha256, 'controller_wall_seconds': controller_times,
                  'wall_seconds': time.monotonic() - begin, 'files': files,
                  'new_fits': 0, 'native_replay_calls': 0, 'scope': 'native control evaluation; no probability calibration claim'}
        _write(out / 'completed.json', result)
        _check(deadline)
        return result
    except BaseException as error:
        if not hasattr(error, 'partial_card_episode') and active_arrays is not None:
            error.partial_card_episode = active_arrays
            error.partial_card_receipt = active_receipt
        actions = [lambda: (out / 'completed.json').rename(out / 'invalid-completion.json')
                   if (out / 'completed.json').exists() else None]
        if hasattr(error, 'partial_card_episode'):
            def save_partial():
                with (out / 'partial-episode.npz').open('xb') as handle:
                    np.savez_compressed(handle, **error.partial_card_episode)
            actions.append(save_partial)
        actions.append(lambda: _write(out / 'failed.json', {'status': 'failed', 'phase': phase, 'current': current,
                       'error': repr(error), 'partial_receipt': getattr(error, 'partial_card_receipt', None),
                       'wall_seconds': time.monotonic() - begin, 'automatic_retry': False}))
        for action in actions:
            try:
                action()
            except BaseException as secondary:  # noqa: BLE001 - preserve the original failure.
                _note(error, f'Failure preservation also failed: {secondary!r}')
        raise
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('protocol', 'inputs', 'checkpoint-map', 'out', 'root'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('protocol', 'inputs', 'bindings'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    args = parser.parse_args()
    evaluate(args.protocol, args.expected_protocol_sha256, args.inputs, args.expected_inputs_sha256,
             _read(args.checkpoint_map), args.out, root=args.root,
             expected_bindings_sha256=args.expected_bindings_sha256)
