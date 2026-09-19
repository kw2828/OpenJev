"""Bind exposed roots and prospective innovations without native evaluation."""
import json
from pathlib import Path

import numpy as np

from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_search_protocol import load_inputs, named_seed, save_inputs, save_npz
from openjev.research.reacher_tracking_rollout import sha, write

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT/'output/reacher-proposal-memory-engineering-v1/attempt-01'
OUT = BASE/'inputs-v1'
NAMES = ('initial', 'random_extra', 'cem/1', 'cem/2', 'cem/3')
COUNTS = (64, 192, 64, 64, 63)


def protected_sources():
    results = []
    for name, key in [('evidence/reacher-two-observation-study-v1/protocol/plan.json', 'sources'),
                      ('evidence/reacher-innovation-pilot-v1/protocol/plan.json', 'source_sha256')]:
        mapping = json.loads((ROOT/name).read_text())[key]
        assert all(sha(ROOT/p) == digest for p, digest in mapping.items())
        results.append({'plan': name, 'sha256': sha(ROOT/name), 'count': len(mapping)})
    return results


def prior_records(protocol):
    assert sha(PRIOR/'completed.json') == protocol['prior_completed_sha256']
    prior = json.loads((PRIOR/'completed.json').read_text())
    assert prior['status'] == 'completed' and not (PRIOR/'failed.json').exists()
    for path, digest in prior['source_sha256'].items():
        assert sha(ROOT/path) == digest, path
    records = {}
    for role in protocol['roles']:
        folder = PRIOR/'rows'/f'{role}--cold'
        row = next(r for r in prior['rows'] if r['name'] == f'{role}--cold')
        assert sha(folder/'completed.json') == row['completed_sha256']
        completed = json.loads((folder/'completed.json').read_text())
        for name in ('episode.json', 'started.json'):
            assert sha(folder/name) == completed['files'][name]
        records[role] = {'episode': json.loads((folder/'episode.json').read_text()),
                         'started': json.loads((folder/'started.json').read_text()),
                         'episode_sha256': completed['files']['episode.json'],
                         'completed_sha256': row['completed_sha256']}
    return prior, records


def root_record(role, step, record):
    episode = record['episode']; meta = episode['metadata']
    target = np.asarray(meta['target_path'])[step]
    gain = float(meta['gear_multiplier'][step])
    assert np.array_equal(np.asarray(meta['target_path'])[step:step+25], np.tile(target, (25, 1)))
    assert np.all(np.asarray(meta['gear_multiplier'])[step:step+24] == gain)
    native = episode['audit']['decision_states'][step]
    assert np.array_equal(native['qpos'][2:], target) and native['qvel'][2:] == [0., 0.]
    return {'source_role': role, 'step': step, 'root': native, 'true_gain': gain,
            'public_packet': episode['policy']['packets'][step],
            'source_episode_sha256': record['episode_sha256'],
            'source_completed_sha256': record['completed_sha256']}


def main():
    OUT.mkdir(exist_ok=False); streams = {}; roots = {}; original = {}
    protocol = None
    def draw(role, shape, scale=1.):
        seed = named_seed(protocol['rng_namespace'], role)
        generator = np.random.Generator(np.random.PCG64(seed))
        before = generator.bit_generator.state
        values = generator.normal(size=shape)*scale
        streams[role] = {'seed': seed, 'shape': list(shape), 'scale': scale,
                         'initial_state': before, 'final_state': generator.bit_generator.state}
        return values
    try:
        protocol = json.loads((BASE/'protocol.json').read_text())
        write(OUT/'started.json', {'status': 'started', 'protocol_sha256': sha(BASE/'protocol.json'),
                                 'preparer_sha256': sha(Path(__file__))})
        protected = protected_sources(); prior, records = prior_records(protocol)
        write(OUT/'protected-sources.json', protected)
        for step in protocol['root_steps']:
            binding = records['nominal']['started']['inputs_by_step'][step]
            stem = Path(binding['stem'])
            for suffix in ('npz', 'json'):
                assert sha(stem.with_suffix('.'+suffix)) == binding[suffix+'_sha256']
            original[str(step)] = binding
            old = load_inputs(stem); arrays = (old.initial, old.random_extra, *old.cem)
            values_a = [np.concatenate((v, draw(f'{step}/A_tail/{n}', (1, k, 4, 2))), axis=2)
                        for n, k, v in zip(NAMES, COUNTS, arrays, strict=True)]
            values_b = [draw(f'{step}/B/{n}', (1, k, 8, 2)) for n, k in zip(NAMES, COUNTS, strict=True)]
            for label, values in [('A', values_a), ('B', values_b)]:
                save_inputs(OUT/f'{step:03d}-{label}', SearchInputs(values[0], values[1], tuple(values[2:])), f'{step}/{label}')
            save_npz(OUT/f'{step:03d}-noise.npz', noise=draw(f'{step}/branch_noise', (4, 24, 2), .05))
        for role in protocol['roles']:
            for step in protocol['root_steps']:
                roots[f'{role}--{step:03d}'] = root_record(role, step, records[role])
        assert len({v['seed'] for v in streams.values()}) == len(streams)
        write(OUT/'streams.json', streams); write(OUT/'roots.json', roots)
        protected_sources(); prior_records(protocol)
        write(OUT/'completed.json', {'status': 'completed', 'native_calls': 0, 'model_calls': 0,
            'original_inputs': original, 'prior_sources': prior['source_sha256'],
            'files': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}})
        print(json.dumps({'status': 'prepared', 'roots': len(roots), 'streams': len(streams),
                          'completed_sha256': sha(OUT/'completed.json')}))
    except BaseException as error:
        if (OUT/'completed.json').exists():
            try:
                (OUT/'completed.json').rename(OUT/'partial-completed.json')
            except BaseException as secondary:  # noqa: BLE001 - preserve original failure.
                error.add_note('Completion demotion failed: '+repr(secondary))
        try:
            write(OUT/'failed.json', {'status': 'failed', 'error': repr(error), 'automatic_retry': False})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure.
            error.add_note('Failure receipt failed: '+repr(secondary))
        raise


if __name__ == '__main__':
    main()
