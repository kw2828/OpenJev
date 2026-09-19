"""Collect and replay public-only native ConcentrationHard trajectories."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from card_memory_common import authenticate, check, deadline_check, preserve_failure, sha, write_json

from openjev.research.card_memory_task import PublicCardTracker, PublicTeacher


def native_factory():
    from popgym.envs.concentration import ConcentrationHard
    return ConcentrationHard()


def collect_episode(seed, behavior_seed, *, deadline, work, env_factory=native_factory):
    """Only returned public frames supply labels, behavior and selected writes."""
    arrays = {name: [] for name in ('positions', 'ranks', 'targets', 'target_mask', 'ages',
                                    'rewards', 'terminated', 'truncated')}
    observations, env, original = [], None, None
    started = time.monotonic()
    try:
        deadline_check(deadline)
        env = env_factory()
        work['native_resets_attempted'] += 1
        obs, info = env.reset(seed=seed)
        work['native_resets_returned'] += 1
        observations.append(obs.copy())
        check(info == {} and np.array_equal(obs, np.full(52, 13)), 'Unexpected initial observation')
        tracker, teacher = PublicCardTracker(), PublicTeacher()
        tracker.reset(obs)
        teacher.reset(obs)
        rng = np.random.Generator(np.random.PCG64(behavior_seed))
        for _ in range(104):
            deadline_check(deadline)
            targets, mask, ages = teacher.targets(obs)
            action = tracker.choose(teacher.probabilities(), rng=rng, random_chance=.5)
            work['native_steps_attempted'] += 1
            after, reward, terminated, truncated, info = env.step(action)
            work['native_steps_returned'] += 1
            # Record returned data before adapter validation.
            observations.append(after.copy())
            for name, value in {'positions': action, 'rewards': reward,
                                'terminated': terminated, 'truncated': truncated, 'targets': targets,
                                'target_mask': mask, 'ages': ages}.items():
                arrays[name].append(value)
            arrays['ranks'].append(int(after[action]))
            reveal = tracker.observe(action, reward, after)
            check((reveal.pos, reveal.rank) == (action, int(after[action])), 'Reveal identity mismatch')
            teacher.observe(after)
            obs = after
            check(info == {}, 'Unexpected native information')
            deadline_check(deadline)
            if terminated or truncated:
                break
        check(bool(terminated or truncated), 'Native horizon did not terminate')
        dtypes = {'positions': np.int64, 'ranks': np.int64, 'targets': np.int64, 'ages': np.int32,
                  'target_mask': bool, 'rewards': np.float64, 'terminated': bool, 'truncated': bool}
        result = {name: np.asarray(values, dtype=dtypes[name]) for name, values in arrays.items()}
        result['observations'] = np.asarray(observations, dtype=np.int64)
    except BaseException as error:
        original = error
        error.partial_card_episode = {'observations': observations, **arrays}
        raise
    finally:
        if env is not None:
            if original is None:
                try:
                    env.close()
                except BaseException as error:
                    error.partial_card_episode = {'observations': observations, **arrays}
                    raise
            else:
                preserve_failure(original, [env.close])
    try:
        deadline_check(deadline)
    except BaseException as error:
        error.partial_card_episode = {'observations': observations, **arrays}
        raise
    return result, {'seed': seed, 'behavior_seed': behavior_seed,
                    'steps': len(result['positions']), 'return': float(result['rewards'].sum()),
                    'elapsed_seconds': time.monotonic() - started}


def replay_episode(seed, behavior_seed, arrays, *, deadline, work, env_factory=native_factory):
    """Replay native frames, public labels/ages and declared behavior RNG."""
    env, original = None, None
    try:
        deadline_check(deadline)
        env = env_factory()
        work['replay_resets_attempted'] += 1
        obs, info = env.reset(seed=seed)
        work['replay_resets_returned'] += 1
        teacher, tracker = PublicTeacher(), PublicCardTracker()
        teacher.reset(obs)
        tracker.reset(obs)
        rng = np.random.Generator(np.random.PCG64(behavior_seed))
        check(info == {} and np.array_equal(obs, arrays['observations'][0]), 'Replay reset mismatch')
        for t, action in enumerate(arrays['positions']):
            deadline_check(deadline)
            targets, mask, ages = teacher.targets(obs)
            for name, actual in (('targets', targets), ('target_mask', mask), ('ages', ages)):
                check(np.array_equal(actual, arrays[name][t]), f'Replay {name} mismatch')
            chosen = tracker.choose(teacher.probabilities(), rng=rng, random_chance=.5)
            check(chosen == action, 'Behavior policy replay mismatch')
            work['replay_steps_attempted'] += 1
            obs, reward, terminated, truncated, info = env.step(int(action))
            work['replay_steps_returned'] += 1
            check(np.array_equal(obs, arrays['observations'][t + 1]), 'Native replay frame mismatch')
            check(int(obs[action]) == arrays['ranks'][t], 'Native replay rank mismatch')
            check(reward == arrays['rewards'][t], 'Native replay reward mismatch')
            check(terminated == arrays['terminated'][t] and truncated == arrays['truncated'][t], 'Replay end mismatch')
            check(bool(terminated or truncated) == (t == len(arrays['positions']) - 1), 'Replay horizon mismatch')
            check(info == {}, 'Replay information mismatch')
            tracker.observe(int(action), reward, obs)
            teacher.observe(obs)
            deadline_check(deadline)
    except BaseException as error:
        original = error
        raise
    finally:
        if env is not None:
            if original is None:
                env.close()
            else:
                preserve_failure(original, [env.close])
    deadline_check(deadline)


def pack(episodes):
    check(len(episodes) > 0, 'Empty episode set')
    n = len(episodes)
    result = {'positions': np.zeros((n, 104), dtype=np.int64), 'ranks': np.zeros((n, 104), dtype=np.int64),
              'valid': np.zeros((n, 104), dtype=bool), 'targets': np.full((n, 104, 52), -1, dtype=np.int64),
              'target_mask': np.zeros((n, 104, 52), dtype=bool), 'ages': np.full((n, 104, 52), -1, dtype=np.int32)}
    for i, episode in enumerate(episodes):
        length = len(episode['positions'])
        check(0 < length <= 104, 'Invalid episode length')
        result['valid'][i, :length] = True
        for name in result.keys() - {'valid'}:
            result[name][i, :length] = episode[name]
    return result


def prepare(out, inputs, protocol, bindings, root, *, preflight=False):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    work = {f'{phase}_{kind}_{state}': 0 for phase in ('native', 'replay')
            for kind in ('resets', 'steps') for state in ('attempted', 'returned')}
    episodes_done, ledger, current = 0, [], None
    try:
        write_json(out / 'started.json', {'status': 'started', 'preflight': preflight,
                   'inputs_sha256': sha(inputs), 'protocol_sha256': sha(protocol), 'bindings_sha256': sha(bindings)})
        recipe, packet, _ = authenticate(root, protocol, inputs, bindings)
        deadline = started + recipe['data']['wall_cap_seconds']
        parts = packet['preflight'] if preflight else packet['data']
        expected = {'engineering': 4} if preflight else {'train': 128, 'dev': 32}
        check({part: len(cases) for part, cases in parts.items()} == expected, 'Unapproved data sizes')
        seeds = [case['seed'] for cases in parts.values() for case in cases]
        check(len(set(seeds)) == len(seeds), 'Repeated environment seeds')
        for part, cases in parts.items():
            check(part in expected, 'Unknown data partition')
            (out / part).mkdir()
            saved = []
            for index, case in enumerate(cases):
                current = {'part': part, 'index': index, **case}
                deadline_check(deadline)
                arrays, receipt = collect_episode(case['seed'], case['behavior_seed'], deadline=deadline, work=work)
                path = out / part / f'{index:03d}.npz'
                with path.open('xb') as handle:
                    np.savez_compressed(handle, **arrays)
                replay_episode(case['seed'], case['behavior_seed'], arrays, deadline=deadline, work=work)
                receipt.update(part=part, index=index, npz_path=str(path.relative_to(out)), npz_sha256=sha(path), replay='passed')
                write_json(out / part / f'{index:03d}.json', receipt)
                ledger.append(receipt)
                saved.append(arrays)
                episodes_done += 1
                with (out / 'ledger.jsonl').open('a') as handle:
                    handle.write(json.dumps(receipt, allow_nan=False) + '\n')
            with (out / (part + '.npz')).open('xb') as handle:
                np.savez_compressed(handle, **pack(saved))
            print(json.dumps({'part': part, 'episodes': len(cases), 'work': work}), flush=True)
        write_json(out / 'episodes.json', ledger)
        authenticate(root, protocol, inputs, bindings)
        deadline_check(deadline)
        write_json(out / 'completed.json', {'status': 'complete', 'episodes': episodes_done, 'work': work,
                   'elapsed_seconds': time.monotonic() - started, 'episodes_sha256': sha(out / 'episodes.json'),
                   'started_sha256': sha(out / 'started.json'),
                   'parts': {part: sha(out / (part + '.npz')) for part in parts}})
        deadline_check(deadline)
    except BaseException as error:
        partial = getattr(error, 'partial_card_episode', None)
        actions = []
        if partial is not None:
            actions.append(lambda: np.savez_compressed(out / 'partial-episode.npz', **{k: np.asarray(v) for k, v in partial.items()}))
        if (out / 'completed.json').exists():
            actions.append(lambda: (out / 'completed.json').rename(out / 'invalid-completion.json'))
        failure = {'status': 'failed', 'error': traceback.format_exc(), 'episodes': episodes_done,
                   'work': work, 'current': current, 'elapsed_seconds': time.monotonic() - started, 'automatic_retry': False}
        actions.append(lambda: write_json(out / 'failed.json', failure))
        preserve_failure(error, actions)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('out', 'inputs', 'protocol', 'bindings', 'root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    prepare(args.out, args.inputs, args.protocol, args.bindings, args.root, preflight=args.preflight)
