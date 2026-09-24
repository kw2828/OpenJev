"""One fixed engineering comparison of complete exact-update training calls.

This does not evaluate task performance or admit a scientific comparison.
All fits, including warmups, are fresh and retained. No candidate is chosen
inside this producer; the independent saved-output audit owns interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'finite-joint-reuse-throughput-v1'
ARMS = ('original_free', 'matched_free', 'rounded')
MODES = ('separate', 'reuse')
CONFIG = {'seed_namespace': 943201, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003, 'fit_seeds': [943301],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'prefix_updates': 32, 'joint_updates': 64, 'fit_cap_seconds': 30.}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    payload = Path(path).read_bytes()
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def inventory(folder):
    result = {}
    for path in sorted(Path(folder).rglob('*')):
        require(not path.is_symlink(), 'ordinary retained output paths')
        if path.is_file():
            result[str(path.relative_to(folder))] = descriptor(path)
    return result


def make_schedule():
    """One warmup and three timed pairs per arm; no runtime-selected ordering."""
    result = []
    for round_index in range(4):
        offset = max(0, round_index - 1) % len(ARMS)
        for arm in ARMS[offset:] + ARMS[:offset]:
            modes = MODES if (round_index + ARMS.index(arm)) % 2 == 0 else MODES[::-1]
            for mode in modes:
                result.append({'run_id': f'r{round_index}-{arm}-{mode}', 'round': round_index,
                               'arm': arm, 'implementation': mode, 'seed': 943301,
                               'warmup': round_index == 0})
    return result


def run(output):
    require(not output.exists(), 'exclusive benchmark directory; no repeat')
    output.mkdir()
    started = time.perf_counter()
    records = []
    stage = 'imports'
    counts = {'train_generation_calls': 0, 'dev_generation_calls': 0,
              'teacher_calls': 0, 'task_evaluation_calls': 0}
    try:
        sys.path.insert(0, str(ROOT / 'src'))
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from run_finite_joint_reuse_training import new_counts, new_structural_work, train
        from run_finite_observation_learning import _save, _tensor_data, _write
        from run_finite_update_learning import _config

        from openjev.research.finite_prefix_learning import generate_attempt_split

        checks = 0

        def check():
            nonlocal checks
            checks += 1
            require(time.perf_counter() - started < 90., 'fixed90-second producer bound')
            if checks == 1 or checks % 256 == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
                require(rss_bytes <= 4 * 1024**3, '4GiB sampled process peak RSS bound')
                paths = list(output.rglob('*'))
                require(len(paths) <= 2048 and sum(p.stat().st_size for p in paths if p.is_file())
                        <= 512 * 1024**2, 'bounded retained benchmark output')

        check()
        config = _config(CONFIG)
        _write(output / 'config.json', config)
        _write(output / 'schedule.json', make_schedule())
        stage = 'single TRAIN generation'
        generation_started = time.perf_counter()
        counts['train_generation_calls'] += 1
        generated = generate_attempt_split(0, 512, 2, seed_namespace=943201)
        _save(output / 'train.npz', generated['data'])
        _save(output / 'train-prefix.npz', generated['prefix_data'])
        # Retain privileged generator output for provenance, never feed it to train.
        _save(output / 'generated-oracle.npz', generated['oracle'])
        data = _tensor_data(generated['data'])
        prefixes = {name: torch.from_numpy(value.copy())
                    for name, value in generated['prefix_data'].items() if name != 'case_ids'}
        require('oracle_prefix' not in data and 'oracle_prefix' not in prefixes,
                'public histories and supervised targets only')
        before = {name: descriptor(output / name)
                  for name in ('train.npz', 'train-prefix.npz', 'generated-oracle.npz')}
        original_data = {name: value.detach().clone() for name, value in data.items()}
        original_prefixes = {name: value.detach().clone() for name, value in prefixes.items()}
        generation = {'counts': generated['counts'], 'files': before,
                      'seconds': time.perf_counter() - generation_started,
                      'scope': 'generation, complete array saves, tensor copies and input snapshots'}
        _write(output / 'generation.json', generation)
        (output / 'fits').mkdir()
        for spec in make_schedule():
            stage = spec['run_id']
            check()
            folder = output / 'fits' / spec['run_id']
            folder.mkdir()
            fit_counts, structural_work = new_counts(), new_structural_work()
            start = time.perf_counter()
            model, fit = train(spec['arm'], spec['seed'], data, prefixes, config, folder,
                               fit_counts, structural_work, check,
                               implementation=spec['implementation'])
            seconds = time.perf_counter() - start
            record = {**spec, 'call_seconds': seconds, 'fit': fit,
                      'counts': fit_counts, 'structural_work': structural_work,
                      'timing_scope': 'complete train call including model and optimizer construction, '
                          'all updates, safeguards, three model and optimizer checkpoints, '
                          'allocation serialization, fit-row write and final check; excludes '
                          'directory creation, input snapshots, and this outer record write'}
            _write(folder / 'fit-record.json', record)
            records.append(record)
            del model
            require(all(torch.equal(data[k], v) for k, v in original_data.items())
                    and all(torch.equal(prefixes[k], v) for k, v in original_prefixes.items()),
                    'training leaves shared input tensors unchanged')
            require(all(np.isfinite(value.detach().numpy()).all() for value in data.values()),
                    'finite retained public inputs')
            print(json.dumps({'completed': spec['run_id'], 'call_seconds': seconds}), flush=True)
        require(before == {name: descriptor(output / name) for name in before},
                'serialized input arrays unchanged')
        check()
        summary = {'version': VERSION, 'status': 'PASS', 'config': config,
                   'schedule': make_schedule(), 'generation': generation, 'counts': counts,
                   'runs': records, 'seconds_before_summary_write': time.perf_counter() - started,
                   'files_before_summary': inventory(output), 'scientific_admission': False,
                   'scope': 'Engineering training throughput only; no task-effectiveness evaluation.'}
        _write(output / 'summary.json', summary)
        check()
        return summary
    except BaseException as error:
        failure = {'version': VERSION, 'status': 'FAILED', 'stage': stage,
                   'error': repr(error), 'completed_runs': [r['run_id'] for r in records],
                   'counts': counts, 'seconds': time.perf_counter() - started,
                   'files_before_failure': inventory(output)}
        with (output / 'failure.json').open('x') as stream:
            json.dump(failure, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == '__main__':
    main()
