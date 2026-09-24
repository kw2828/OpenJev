"""One train-only engineering feasibility probe; no DEV or decision metrics.

Six fits use the exact qualified trainer. All saved training evidence and the
independent saved-output audit precede the fixed time-projection decision.
Failure is preserved and propagated, never retried or replaced. The projection
is a conservative admission calculation, not a runtime guarantee or speed claim.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from finite_head_learning_worker_v2 import SuspendClock, descriptor, files, publish, require

VERSION = 'finite-action-range-exposure-v1'
ARMS = ('rounded_mse', 'rounded_double', 'rounded_range', 'free_mse', 'free_double', 'free_range')
CONFIG = {'seed_namespace': 948201, 'fit_seeds': [948301], 'train_attempts': 512, 'dev_attempts': 8,
          'batch_size': 64, 'learning_rate': .003, 'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'prefix_updates': 32, 'joint_updates': 64, 'fit_cap_seconds': 30.}
EXPOSURE = {'config': CONFIG, 'target_counts': {'prefix_updates': 1024, 'joint_updates': 3072},
            'maximum_projected_seconds': 90., 'projection_comparison': '<=', 'internal_cap_seconds': 240.}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def final_clock_read(clock, started):
    """One final native read, retaining its exception for post-receipt handling."""
    if started is None:
        return None, None
    try:
        elapsed_ns = clock.now_ns() - started
        require(elapsed_ns >= 0, 'nondecreasing final native exposure clock')
        return elapsed_ns / 10**9, None
    except BaseException as error:  # noqa: BLE001 - persist the original failure receipt before propagating.
        return None, error


def projections(allocations, fits):
    """Twice the scaled accepted-stage durations plus measured non-stage work."""
    require(type(allocations) is dict and set(allocations) == set(ARMS)
            and type(fits) is list and len(fits) == len(ARMS)
            and {(fit['arm'], fit['seed']) for fit in fits} == {(arm, 948301) for arm in ARMS},
            'complete six-arm engineering fit roster')
    indexed = {fit['arm']: fit for fit in fits}
    rows = []
    for arm in ARMS:
        fit, allocation = indexed[arm], allocations[arm]
        require(allocation['status'] == 'PASS' and allocation['termination'] == 'completed_updates'
                and allocation['prefix_updates'] == allocation['accepted_prefix_updates'] == 32
                and allocation['joint_updates'] == allocation['accepted_joint_updates'] == allocation['joint_cursor'] == 64
                and allocation['attempted_updates'] == allocation['accepted_updates'] == 96
                and allocation['max_seconds'] == 30., 'complete accepted fixed engineering allocation')
        stages = allocation['stages']
        require(type(stages) is list and len(stages) == 2, 'both fixed engineering stages complete')
        require(finite(fit['seconds']) and fit['seconds'] >= 0, 'finite nonnegative full-fit time')
        stage_seconds, previous_stop = [], 0.
        for index, (stage, kind, target) in enumerate(zip(stages, ('prefix', 'joint'), (32, 64), strict=True)):
            require(stage['stage'] == index + 1 and stage['kind'] == kind and stage['status'] == 'PASS'
                    and stage['termination'] == 'completed_updates'
                    and stage['target_updates'] == stage['accepted_updates'] == stage['attempted_updates'] == target
                    and stage['deadline_seconds'] == 30. and stage['overrun_seconds'] == 0,
                    'successful stage with every requested update accepted')
            start, stop = stage['start_elapsed'], stage['stopped_elapsed']
            require(finite(start) and finite(stop) and previous_stop <= start < stop <= 30.,
                    'positive stage duration on one ordered safety clock')
            stage_seconds.append(stop - start)
            previous_stop = stop
        require(fit['seconds'] >= previous_stop, 'full fitting cost includes all stage elapsed time')
        overhead = fit['seconds'] - sum(stage_seconds)
        require(finite(overhead) and overhead >= 0, 'nonnegative complete fitting overhead')
        projected = 2 * (stage_seconds[0] * 32 + stage_seconds[1] * 48) + overhead
        require(finite(projected), 'finite fixed exposure projection')
        rows.append({'arm': arm, 'seed': 948301, 'stage_seconds': stage_seconds,
            'accepted_updates': [32, 64], 'target_updates': [1024, 3072], 'scale_factors': [32., 48.],
            'nonstage_seconds': overhead, 'safety_multiplier': 2., 'projected_seconds': projected})
    return rows


def feasible(rows):
    require(type(rows) is list and len(rows) == len(ARMS)
            and [(row['arm'], row['seed']) for row in rows] == [(arm, 948301) for arm in ARMS]
            and all(finite(row['projected_seconds']) and row['projected_seconds'] >= 0 for row in rows),
            'complete finite canonical projection roster')
    return all(row['projected_seconds'] <= 90. for row in rows)


def run(output):
    output = Path(output).resolve()
    require(not output.exists(), 'exclusive engineering exposure directory')
    output.mkdir()
    started = time.perf_counter()
    record = {'version': VERSION, 'status': 'FAILED', **EXPOSURE,
        'purpose': 'Feasibility only; no DEV generation, scientific effects, loss selection or count tuning.'}
    checks = 0
    stage = 'imports'
    counts, work, per_arm, fits = {}, {}, {}, []
    clock, native_start = None, None
    active_arm, before = None, {}
    run_folder = output / 'run'
    try:
        clock = SuspendClock()
        native_start = clock.now_ns()
        record['internal_clock_backend'] = clock.backend
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from audit_finite_action_range_learning import audit_exposure
        from run_finite_action_range_learning import verify_paired_exposure, verify_prefix_pairing
        from run_finite_action_range_training import new_counts, new_structural_work, train
        from run_finite_observation_learning import _save, _tensor_data

        from openjev.research.finite_prefix_learning import generate_attempt_split

        def check(*, hard=False):
            nonlocal checks
            checks += 1
            elapsed_ns = clock.now_ns() - native_start
            require(elapsed_ns >= 0, 'nondecreasing native exposure clock')
            if elapsed_ns >= 240 * 10**9:
                raise TimeoutError('fixed engineering exposure cap')
            if hard or checks == 1 or checks % 256 == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
                require(rss_bytes <= 4 * 1024**3, 'engineering exposure child RSS bound')
                paths = list(output.rglob('*'))
                require(len(paths) <= 1024 and sum(path.stat().st_size for path in paths if path.is_file())
                        <= 1024**3, 'engineering exposure output bounds')

        check(hard=True)
        run_folder.mkdir()
        publish(run_folder / 'config.json', CONFIG)
        counts, work = new_counts(), new_structural_work()
        stage = 'TRAIN generation'
        generation_start = time.perf_counter()
        generated = generate_attempt_split(0, 512, 2, seed_namespace=948201)
        data, prefixes = generated['data'], generated['prefix_data']
        require(len(data['prefix']) > 0, 'positive retained TRAIN support')
        _save(run_folder / 'train.npz', data)
        _save(run_folder / 'train-prefix.npz', prefixes)
        publish(run_folder / 'dataset.json', generated['counts'])
        training = _tensor_data(data)
        prefix_training = {name: torch.from_numpy(value.copy()) for name, value in prefixes.items() if name != 'case_ids'}
        require('oracle_prefix' not in training and 'oracle_prefix' not in prefix_training, 'no hidden-state inputs')
        generation_seconds = time.perf_counter() - generation_start
        check(hard=True)
        for arm in ARMS:
            stage = 'fit ' + arm
            active_arm = arm
            before = dict(counts)
            _model, fit = train(arm, 948301, training, prefix_training, CONFIG, run_folder, counts, work,
                                check, implementation='reuse')
            fits.append(fit)
            per_arm[arm] = {key: counts[key] - before[key] for key in counts}
            del _model
            check(hard=True)
        paired = verify_paired_exposure(fits, run_folder, CONFIG)
        pairing = verify_prefix_pairing(fits, run_folder, CONFIG)
        require(counts['fit_count'] == 6 and counts['checkpoint_writes'] == counts['optimizer_checkpoint_writes'] == 18
                and counts['optimizer_steps'] == counts['accepted_optimizer_steps'] == 576
                and counts['accepted_prefix_steps'] == 192 and counts['accepted_joint_steps'] == 384,
                'complete six-fit fixed exposure accounting')
        summary = {'version': VERSION, 'config': CONFIG, 'arms': list(ARMS), 'fit_order': list(ARMS),
            'fits': fits, 'counts': counts, 'per_arm_counts': per_arm, 'structural_work': work,
            'dataset_counts': generated['counts'], 'generation_seconds': generation_seconds,
            'paired_batch_sha256': paired, 'prefix_pair_checks': pairing,
            'scope_counts': {'train_generation_count': 1, 'dev_generation_count': 0, 'array_decodes': 0,
                             'checkpoint_decodes': 0, 'evaluation_rollouts': 0, 'oracle_model_constructions': 0},
            'input_files': {name: descriptor(run_folder / name) for name in ('train.npz', 'train-prefix.npz')},
            'files': files(run_folder)}
        publish(run_folder / 'summary.json', summary)
        original_files = files(run_folder)
        record['run_files'] = original_files
        stage = 'saved training audit'
        verified = audit_exposure(run_folder, check=check)
        publish(output / 'audit.json', verified)
        require(files(run_folder) == original_files, 'saved training evidence unchanged by audit')
        require(verified['agreement'] is True and verified['descriptive_only'] is True
                and verified['no_dev_generated'] is True, 'independent train-only engineering audit agrees')
        record['audit'] = descriptor(output / 'audit.json')
        stage = 'fixed feasibility projection'
        allocations = {row['arm']: json.loads((run_folder / row['allocation']['path']).read_text()) for row in fits}
        record['projections'] = projections(allocations, fits)
        publish(output / 'projections.json', record['projections'])
        record['feasible'] = feasible(record['projections'])
        require(record['feasible'], 'all six preselected exposures must project to at most90 seconds')
        check(hard=True)
        record['status'] = 'PASS'
        return record
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        if active_arm is not None and active_arm not in per_arm:
            per_arm[active_arm] = {key: counts[key] - before[key] for key in counts}
        record['stage'] = stage
        record['active_arm'] = active_arm
        record['checks'] = checks
        record['seconds'] = time.perf_counter() - started
        record['seconds_scope'] = 'Descriptive perf_counter duration; the240-second safety bound uses the native suspend-aware clock.'
        record['internal_wall_seconds'], final_error = final_clock_read(clock, native_start)
        sole_final_error = final_error is not None and 'error' not in record
        if final_error is not None:
            record['final_clock_error'] = repr(final_error)
            record['status'] = 'FAILED'
            if sole_final_error:
                record['error'] = repr(final_error)
        record['completed_fit_count'] = len(fits)
        record['counts'] = counts
        record['per_arm_counts'] = per_arm
        record['structural_work'] = work
        record['files_before_receipt'] = files(output)
        publish(output / 'receipt.json', record)
        print(json.dumps({'status': record['status'], 'seconds': record['seconds']}))
        if sole_final_error:
            raise final_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
