"""Two-stage atomic training allocation, independent of numerical backends.

One monotonic stopwatch starts before validation and setup. The two deadlines
are absolute offsets from that start, not fresh budgets. Snapshotting, update
diagnostics, hashes and checks precede each acceptance timestamp. A late attempt
restores BOTH model and optimizer; the accepted joint-batch cursor stays put.
Checkpoint and rollback work is charged even after a deadline. Only the final
read-only summary has a separate, outside-eligibility timer.

The runner supplies qualified numerical hooks. Joint batches must be pure
functions of the accepted cursor, including their seed/epoch permutation. Any
mutable randomness or extra numerical state must instead belong to snapshots.
The joint hook implements the fixed attempted-case/eligible-case/event-scaled
objective. The prefix hook uses full-batch likelihood plus the 0.001 log-prior,
updates only dynamics and guards cost-head bytes. This controller does not
implement a different objective, Adam, a sampler, or model construction.

Snapshots must own their values and losslessly include optimizer moments,
steps and parameter-group configuration. Snapshot/hash hooks and checkpoint /
summary callbacks must not mutate numerical state. Hooks must report actual
work, including validation/diagnostics; controller counters are not FLOPs.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

VERSION = 'finite-training-allocation-v1'
ARMS = ('joint_continuous', 'joint_restart', 'prefix_then_joint')
MAX_ATTEMPTS = 100000
WORK_KEYS = ('validation_calls', 'optimizer_constructions', 'model_snapshot_calls',
    'model_snapshot_tensors', 'model_snapshot_bytes', 'optimizer_snapshot_calls',
    'optimizer_snapshot_tensors', 'optimizer_snapshot_bytes', 'model_restore_calls',
    'model_restore_tensors', 'model_restore_bytes', 'optimizer_restore_calls',
    'optimizer_restore_tensors', 'optimizer_restore_bytes', 'model_hash_calls',
    'optimizer_hash_calls', 'joint_update_calls', 'prefix_update_calls',
    'checkpoint_calls', 'check_calls', 'clock_calls')


@dataclass(frozen=True)
class Snapshot:
    state: Any
    sha256: str
    tensors: int
    bytes: int


@dataclass(frozen=True)
class Hooks:
    validate: Callable[[], dict]
    new_optimizer: Callable[[str], Any]
    snapshot_model: Callable[[], Snapshot]
    restore_model: Callable[[Any], None]
    model_hash: Callable[[], str]
    snapshot_optimizer: Callable[[Any], Snapshot]
    restore_optimizer: Callable[[Any, Any], None]
    optimizer_hash: Callable[[Any], str]
    update: Callable[[str, Any, int | None], dict]
    checkpoint: Callable[[str, Any, int], dict]
    final_summary: Callable[[Any, int], dict]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    """Copy only finite JSON primitives; no implicit tensor/NumPy conversion."""
    def validate(item):
        if type(item) is dict:
            require(all(type(k) is str for k in item), 'JSON string keys')
            for child in item.values():
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        else:
            require(item is None or type(item) in (str, bool, int, float), 'JSON primitive metadata')
            require(type(item) is not float or math.isfinite(item), 'finite JSON scalar')
    validate(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'canonical SHA256 state digest')
    return value


def run_allocation(arm, hooks, *, stage1_seconds=10., total_seconds=40.,
                   check=lambda: None, clock=time.perf_counter):
    """Return complete traces, including failed allocation and discarded work.

    A successful stage requires at least one accepted update. The secondary
    cap applies to total attempts across both stages. Exceptions from hooks or
    the enclosing resource check propagate with ``allocation_result`` attached;
    an in-flight numerical attempt is first rolled back. Failed allocations
    never start a later stage. Initial/boundary/final checkpoint callbacks also
    run for ordinary zero-update/cap failures, preserving their state.

    Adam continuity is object continuity in joint_continuous. Other arms obtain
    a fresh joint optimizer AFTER the boundary checkpoint. Reset and initial
    optimizer hashes are explicit. Prefix-only work never advances the joint
    cursor. Matching time eligibility is not matching executed FLOPs, accepted
    updates, or an isolated optimizer-reset intervention.
    """
    work, update_work = dict.fromkeys(WORK_KEYS, 0), {}
    trace, stages, checkpoints = [], [], []
    last_time = None
    cursor = 0
    status, termination = 'PASS', 'completed_stages'
    boundary, metadata = None, None

    def checked():
        work['check_calls'] += 1
        check()

    def stamp():
        nonlocal last_time
        work['clock_calls'] += 1
        value = clock()
        require(type(value) in (int, float) and math.isfinite(value)
                and (last_time is None or value >= last_time), 'finite nondecreasing clock')
        last_time = float(value)
        return last_time

    started = stamp()

    def hashes(optimizer):
        work['model_hash_calls'] += 1
        model_hash = _digest(hooks.model_hash())
        work['optimizer_hash_calls'] += 1
        return model_hash, _digest(hooks.optimizer_hash(optimizer))

    def snapshot(optimizer):
        values = []
        for kind, callback in (('model', hooks.snapshot_model),
                               ('optimizer', lambda: hooks.snapshot_optimizer(optimizer))):
            work[kind + '_snapshot_calls'] += 1
            value = callback()
            require(type(value) is Snapshot, 'owned Snapshot record')
            _digest(value.sha256)
            require(type(value.tensors) is int and value.tensors >= 0
                    and type(value.bytes) is int and value.bytes >= 0, 'snapshot storage counters')
            work[kind + '_snapshot_tensors'] += value.tensors
            work[kind + '_snapshot_bytes'] += value.bytes
            values.append(value)
        require(hashes(optimizer) == tuple(v.sha256 for v in values), 'snapshots match current model and optimizer')
        return values

    def restore(optimizer, values):
        for kind, value, callback in (
            ('model', values[0], lambda state: hooks.restore_model(state)),
            ('optimizer', values[1], lambda state: hooks.restore_optimizer(optimizer, state)),
        ):
            work[kind + '_restore_calls'] += 1
            callback(value.state)
            work[kind + '_restore_tensors'] += value.tensors
            work[kind + '_restore_bytes'] += value.bytes
        require(hashes(optimizer) == tuple(v.sha256 for v in values), 'exact model and optimizer rollback')

    def checkpoint(label, optimizer):
        began = stamp() - started
        before = hashes(optimizer)
        checked()
        work['checkpoint_calls'] += 1
        record = hooks.checkpoint(label, optimizer, cursor)
        require(type(record) is dict, 'checkpoint metadata mapping')
        record = _json(record)
        require(hashes(optimizer) == before, 'checkpoint callback leaves state unchanged')
        checked()
        completed = stamp() - started
        checkpoints.append({'label': label, 'start_elapsed': began, 'completed_elapsed': completed,
            'seconds': completed - began, 'joint_cursor': cursor,
            'model_sha256': before[0], 'optimizer_sha256': before[1], 'metadata': record})

    def new_optimizer(kind):
        work['optimizer_constructions'] += 1
        return hooks.new_optimizer(kind)

    def one_stage(index, kind, optimizer, deadline):
        nonlocal cursor
        start_index = len(trace)
        start_cursor = cursor
        model_start, optimizer_start = hashes(optimizer)
        stage_start = stamp() - started
        accepted = 0
        stage_status, reason = 'PASS', 'deadline_reached'
        while True:
            checked()
            began = stamp() - started
            if began >= deadline:
                break
            if len(trace) >= MAX_ATTEMPTS:
                stage_status, reason = 'FAILED_ATTEMPT_CAP', 'attempt_cap_before_deadline'
                break
            previous_work = work.copy()
            values = snapshot(optimizer)
            before_cursor = cursor
            result, attempted, completed, error_text = None, None, None, None
            try:
                work[kind + '_update_calls'] += 1
                raw_result = hooks.update(kind, optimizer, cursor if kind == 'joint' else None)
                require(type(raw_result) is dict and set(raw_result) == {'loss', 'diagnostics', 'work'}, 'exact update result')
                require(type(raw_result['loss']) in (int, float) and math.isfinite(raw_result['loss'])
                        and type(raw_result['diagnostics']) is dict and type(raw_result['work']) is dict
                        and all(type(k) is str and type(v) is int and v >= 0 for k, v in raw_result['work'].items()),
                        'finite objective, diagnostics and actual update-work counters')
                result = _json(raw_result)
                for key, value in result['work'].items():
                    update_work[key] = update_work.get(key, 0) + value
                attempted = hashes(optimizer)
                checked()
                completed = stamp() - started
            except BaseException as error:
                restore(optimizer, values)
                error_text = repr(error)
                trace.append({'attempt': len(trace) + 1, 'stage': index, 'kind': kind,
                    'deadline_seconds': deadline, 'start_elapsed': began, 'completed_elapsed': completed,
                    'accepted': False, 'rolled_back': True, 'error': error_text,
                    'cursor_before': before_cursor, 'cursor_attempted': before_cursor + int(kind == 'joint'),
                    'cursor_retained': before_cursor, 'model_before_sha256': values[0].sha256,
                    'optimizer_before_sha256': values[1].sha256,
                    'model_attempted_sha256': attempted[0] if attempted else None,
                    'optimizer_attempted_sha256': attempted[1] if attempted else None,
                    'model_retained_sha256': values[0].sha256, 'optimizer_retained_sha256': values[1].sha256,
                    'result': result, 'work_delta': {k: work[k] - previous_work[k] for k in WORK_KEYS}})
                raise
            eligible = completed <= deadline
            if eligible:
                accepted += 1
                cursor += int(kind == 'joint')
                retained = attempted
            else:
                restore(optimizer, values)
                retained = (values[0].sha256, values[1].sha256)
                reason = 'late_update_rolled_back'
            trace.append({'attempt': len(trace) + 1, 'stage': index, 'kind': kind,
                'deadline_seconds': deadline, 'start_elapsed': began, 'completed_elapsed': completed,
                'accepted': eligible, 'rolled_back': not eligible, 'error': None,
                'cursor_before': before_cursor, 'cursor_attempted': before_cursor + int(kind == 'joint'),
                'cursor_retained': cursor, 'model_before_sha256': values[0].sha256,
                'optimizer_before_sha256': values[1].sha256, 'model_attempted_sha256': attempted[0],
                'optimizer_attempted_sha256': attempted[1], 'model_retained_sha256': retained[0],
                'optimizer_retained_sha256': retained[1], 'result': result,
                'work_delta': {k: work[k] - previous_work[k] for k in WORK_KEYS}})
            if not eligible or completed == deadline:
                break
            if len(trace) >= MAX_ATTEMPTS:
                stage_status, reason = 'FAILED_ATTEMPT_CAP', 'attempt_cap_before_deadline'
                break
        if not accepted and stage_status == 'PASS':
            stage_status = 'FAILED_ZERO_ACCEPTED'
        final_hashes = hashes(optimizer)
        checked()
        stopped = stamp() - started
        row = {'stage': index, 'kind': kind, 'deadline_seconds': deadline, 'start_elapsed': stage_start,
            'stopped_elapsed': stopped, 'status': stage_status, 'termination': reason,
            'trace_start': start_index, 'trace_stop': len(trace), 'attempted_updates': len(trace) - start_index,
            'accepted_updates': accepted, 'joint_cursor_start': start_cursor, 'joint_cursor_end': cursor,
            'model_start_sha256': model_start, 'optimizer_start_sha256': optimizer_start,
            'model_end_sha256': final_hashes[0], 'optimizer_end_sha256': final_hashes[1],
            'overrun_seconds': max(0., stopped - deadline)}
        stages.append(row)
        return row

    try:
        require(arm in ARMS and type(hooks) is Hooks and callable(check) and callable(clock), 'arm and allocation hooks')
        require(all(callable(getattr(hooks, name)) for name in hooks.__dataclass_fields__), 'callable numerical hooks')
        require(type(stage1_seconds) in (int, float) and type(total_seconds) in (int, float)
                and math.isfinite(stage1_seconds) and math.isfinite(total_seconds)
                and 0 < stage1_seconds < total_seconds, 'ordered positive global eligibility deadlines')
        require(type(MAX_ATTEMPTS) is int and MAX_ATTEMPTS > 0, 'positive secondary attempt cap')
        checked()
        work['validation_calls'] += 1
        metadata = hooks.validate()
        require(type(metadata) is dict, 'validation metadata mapping')
        metadata = _json(metadata)
        kind = 'prefix' if arm == 'prefix_then_joint' else 'joint'
        optimizer = new_optimizer(kind)
        initial_hashes = hashes(optimizer)
        checkpoint('initial', optimizer)
        first = one_stage(1, kind, optimizer, float(stage1_seconds))
        checkpoint('boundary', optimizer)
        boundary = {'joint_cursor': cursor, 'model_sha256': first['model_end_sha256'],
            'stage1_optimizer_kind': kind, 'optimizer_before_sha256': first['optimizer_end_sha256'],
            'optimizer_after_sha256': None, 'stage2_optimizer_kind': 'joint', 'optimizer_reset': False}
        if first['status'] == 'PASS':
            if arm != 'joint_continuous':
                optimizer = new_optimizer('joint')
                boundary['optimizer_reset'] = True
            model_hash, opt_hash = hashes(optimizer)
            require(model_hash == boundary['model_sha256'], 'optimizer transition does not change model')
            if arm == 'joint_continuous':
                require(opt_hash == boundary['optimizer_before_sha256'], 'continuous optimizer retains moments')
            boundary['optimizer_after_sha256'] = opt_hash
            second = one_stage(2, 'joint', optimizer, float(total_seconds))
            status = second['status']
            if status != 'PASS':
                termination = 'stage2_' + second['termination']
        else:
            status, termination = first['status'], 'stage1_' + first['termination']
        checkpoint('final', optimizer)
        final_hashes = hashes(optimizer)
        checked()
        stopped = stamp()
        timed_work = work.copy()
        result = {'version': VERSION, 'arm': arm, 'status': status, 'termination': termination,
            'stage1_seconds': float(stage1_seconds), 'total_seconds': float(total_seconds),
            'max_attempts': MAX_ATTEMPTS, 'metadata': metadata, 'initial_model_sha256': initial_hashes[0],
            'initial_optimizer_sha256': initial_hashes[1], 'final_model_sha256': final_hashes[0],
            'final_optimizer_sha256': final_hashes[1], 'joint_cursor': cursor,
            'attempted_updates': len(trace), 'accepted_updates': sum(r['accepted'] for r in trace),
            'accepted_joint_updates': sum(r['accepted'] and r['kind'] == 'joint' for r in trace),
            'accepted_prefix_updates': sum(r['accepted'] and r['kind'] == 'prefix' for r in trace),
            'trace': trace, 'stages': stages, 'boundary': boundary, 'checkpoints': checkpoints,
            'work': timed_work, 'update_work': update_work, 'timed_seconds': stopped - started,
            'overrun_seconds': max(0., stopped - started - total_seconds), 'compute_matched': False,
            'timing_scope': 'One global clock including validation, setup, snapshots, diagnostics, attempts, rollback and all three checkpoint callbacks; summary is separate.'}
        summary = hooks.final_summary(optimizer, cursor)
        require(type(summary) is dict, 'final summary mapping')
        result['final_summary'] = _json(summary)
        require(hashes(optimizer) == final_hashes, 'final summary leaves retained state unchanged')
        checked()
        result['final_summary_seconds'] = stamp() - stopped
        result['final_summary_work'] = {k: work[k] - timed_work[k] for k in WORK_KEYS}
        return result
    except BaseException as error:
        partial = {'version': VERSION, 'arm': arm, 'status': 'FAILED_EXCEPTION', 'error': repr(error),
            'metadata': metadata, 'trace': trace, 'stages': stages, 'boundary': boundary,
            'checkpoints': checkpoints, 'work': work, 'update_work': update_work, 'joint_cursor': cursor,
            'last_elapsed': None if last_time is None else last_time - started}
        error.allocation_result = partial
        raise
