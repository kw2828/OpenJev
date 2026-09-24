"""Exact prefix and joint updates under one fail-closed safety stopwatch.

This controller reuses the frozen numerical Hooks/Snapshot contract, not its
time-eligibility stopping rule. Successful runs perform every requested update
once. Prefix updates never advance the joint cursor; a fresh joint optimizer
starts at cursor zero after the boundary checkpoint. The clock starts before
validation and setup and covers all three checkpoints and the final summary.

Every update, checkpoint callback and summary callback is protected by owned
model/optimizer snapshots. Exceptions and late completions restore both states
before propagating. Failed runs carry allocation_result and cannot count as a
completed stage schedule. Synchronous hooks and rollback can overrun the safety
deadline before returning control; their work is retained, never accepted as
partial success. There is no retry, best-state selection or deadline extension.

Hooks must keep any mutable randomness in their snapshots, and joint batches
must be pure functions of the accepted cursor. Snapshot/hash callbacks must be
read-only. Returned work includes all controller operations through the final
summary; final_summary_work is a subset, not an additional charge. Numerical
work is reported by completed update hooks, including a late completed update.
If an update raises before returning, its internal partial work is not inferred.
Result persistence outside these callbacks belongs to the enclosing runner.
"""
from __future__ import annotations

import math
import time

from openjev.research.finite_training_allocation import WORK_KEYS as ORIGINAL_WORK_KEYS
from openjev.research.finite_training_allocation import Hooks, Snapshot, _digest, _json, require

VERSION = 'finite-update-allocation-v1'
MAX_UPDATES = 100000
WORK_KEYS = (*ORIGINAL_WORK_KEYS, 'final_summary_calls')


def run_update_allocation(hooks, *, prefix_updates, joint_updates, max_seconds=120.,
                          check=lambda: None, clock=time.perf_counter):
    """Execute one exact prefix-then-joint schedule or raise with its evidence.

Counts are positive Python integers, with a total at most 100000. The safety
limit is a finite positive number, not an eligibility or adaptive training
budget. A completion timestamp equal to the deadline is timely, but no further
operation starts at that instant. Returned PASS also requires final checkpoint
and summary completion by that original deadline.

Snapshot and hook types are the exact frozen dataclasses re-exported here.
Successful trace rows use the old allocation layout. Each stage additionally
reports target_updates and shares max_seconds as its deadline. Checkpoint rows
add accepted/rolled_back/error fields. Failures retain partial rows and counts;
rollback_error is explicit if restoration itself fails.
"""
    work, update_work = dict.fromkeys(WORK_KEYS, 0), {}
    trace, stages, checkpoints = [], [], []
    cursor, started, last_time = 0, None, None
    metadata = boundary = initial_hashes = final_hashes = None
    final_summary = final_summary_record = None
    summary_started = None
    final_summary_work, final_summary_seconds = {}, 0.

    def finite_number(value):
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    def checked():
        work['check_calls'] += 1
        check()

    def stamp():
        nonlocal last_time
        work['clock_calls'] += 1
        value = clock()
        require(finite_number(value)
                and (last_time is None or value >= last_time), 'finite nondecreasing clock')
        last_time = float(value)
        return last_time

    def timely(elapsed, where, *, before=False):
        if elapsed > max_seconds or (before and elapsed == max_seconds):
            raise TimeoutError('exact-update safety deadline: ' + where)

    def guard(where):
        checked()
        elapsed = stamp() - started
        timely(elapsed, where, before=True)
        return elapsed

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
        require(hashes(optimizer) == tuple(value.sha256 for value in values),
                'snapshots match current model and optimizer')
        return values

    def restore(optimizer, values):
        for kind, value, callback in (
                ('model', values[0], hooks.restore_model),
                ('optimizer', values[1], lambda state: hooks.restore_optimizer(optimizer, state))):
            work[kind + '_restore_calls'] += 1
            callback(value.state)
            work[kind + '_restore_tensors'] += value.tensors
            work[kind + '_restore_bytes'] += value.bytes
        require(hashes(optimizer) == tuple(value.sha256 for value in values),
                'exact model and optimizer rollback')

    def rollback(error, optimizer, values):
        restored = False
        try:
            restore(optimizer, values)
            restored = True
        except BaseException as rollback_error:  # noqa: BLE001 - retain failure evidence, then re-raise the original error
            error.rollback_error = repr(rollback_error)
        try:
            error.rollback_completed_elapsed = stamp() - started
        except BaseException as timing_error:  # noqa: BLE001 - a broken clock must not mask the original failure
            error.failure_timing_error = repr(timing_error)
        return restored

    def read_only(label, optimizer, *, summary=False):
        """Persist/describe a boundary without admitting state or deadline changes."""
        previous_work = work.copy()
        began = guard('before ' + label)
        values = snapshot(optimizer)
        before = tuple(value.sha256 for value in values)
        record, completed = None, None
        try:
            checked()
            work['final_summary_calls' if summary else 'checkpoint_calls'] += 1
            raw = hooks.final_summary(optimizer, cursor) if summary else hooks.checkpoint(label, optimizer, cursor)
            require(type(raw) is dict, 'read-only callback metadata mapping')
            record = _json(raw)
            require(hashes(optimizer) == before, 'read-only callback leaves model and optimizer unchanged')
            checked()
            completed = stamp() - started
            timely(completed, label + ' completion')
        except BaseException as error:
            restored = rollback(error, optimizer, values)
            partial = {'label': label, 'start_elapsed': began, 'completed_elapsed': completed,
                'seconds': None if completed is None else completed - began, 'joint_cursor': cursor,
                'model_sha256': before[0], 'optimizer_sha256': before[1], 'metadata': record,
                'accepted': False, 'rolled_back': restored, 'error': repr(error)}
            partial['rollback_completed_elapsed'] = getattr(error, 'rollback_completed_elapsed', None)
            if not restored:
                partial['rollback_error'] = error.rollback_error
            if summary:
                error.final_summary_record = partial
                error.final_summary_work = {key: work[key] - previous_work[key] for key in WORK_KEYS}
            else:
                checkpoints.append(partial)
            raise
        row = {'label': label, 'start_elapsed': began, 'completed_elapsed': completed,
            'seconds': completed - began, 'joint_cursor': cursor,
            'model_sha256': before[0], 'optimizer_sha256': before[1], 'metadata': record,
            'accepted': True, 'rolled_back': False, 'error': None}
        if not summary:
            checkpoints.append(row)
        return row, {key: work[key] - previous_work[key] for key in WORK_KEYS}

    def new_optimizer(kind):
        work['optimizer_constructions'] += 1
        return hooks.new_optimizer(kind)

    def one_stage(index, kind, optimizer, target):
        nonlocal cursor
        start_index, start_cursor = len(trace), cursor
        model_start, optimizer_start = hashes(optimizer)
        stage_start = guard('stage start')
        accepted = 0
        retained_hashes = (model_start, optimizer_start)
        try:
            for _ in range(target):
                began = guard('before update')
                previous_work = work.copy()
                values = snapshot(optimizer)
                before_cursor = cursor
                result = attempted = completed = None
                try:
                    work[kind + '_update_calls'] += 1
                    raw = hooks.update(kind, optimizer, cursor if kind == 'joint' else None)
                    require(type(raw) is dict and set(raw) == {'loss', 'diagnostics', 'work'}, 'exact update result')
                    require(type(raw['loss']) in (int, float) and math.isfinite(raw['loss'])
                            and type(raw['diagnostics']) is dict and type(raw['work']) is dict
                            and all(type(k) is str and type(v) is int and v >= 0 for k, v in raw['work'].items()),
                            'finite objective, diagnostics and actual update-work counters')
                    result = _json(raw)
                    for key, value in result['work'].items():
                        update_work[key] = update_work.get(key, 0) + value
                    attempted = hashes(optimizer)
                    checked()
                    completed = stamp() - started
                    timely(completed, 'update completion')
                except BaseException as error:
                    restored = rollback(error, optimizer, values)
                    retained_hashes = tuple(value.sha256 for value in values) if restored else (None, None)
                    row = {'attempt': len(trace) + 1, 'stage': index, 'kind': kind,
                        'deadline_seconds': float(max_seconds), 'start_elapsed': began, 'completed_elapsed': completed,
                        'accepted': False, 'rolled_back': restored, 'error': repr(error),
                        'cursor_before': before_cursor, 'cursor_attempted': before_cursor + int(kind == 'joint'),
                        'cursor_retained': before_cursor, 'model_before_sha256': values[0].sha256,
                        'optimizer_before_sha256': values[1].sha256,
                        'model_attempted_sha256': attempted[0] if attempted else None,
                        'optimizer_attempted_sha256': attempted[1] if attempted else None,
                        'model_retained_sha256': retained_hashes[0], 'optimizer_retained_sha256': retained_hashes[1],
                        'result': result, 'work_delta': {key: work[key] - previous_work[key] for key in WORK_KEYS}}
                    row['rollback_completed_elapsed'] = getattr(error, 'rollback_completed_elapsed', None)
                    if not restored:
                        row['rollback_error'] = error.rollback_error
                    trace.append(row)
                    raise
                accepted += 1
                cursor += int(kind == 'joint')
                retained_hashes = attempted
                trace.append({'attempt': len(trace) + 1, 'stage': index, 'kind': kind,
                    'deadline_seconds': float(max_seconds), 'start_elapsed': began, 'completed_elapsed': completed,
                    'accepted': True, 'rolled_back': False, 'error': None,
                    'cursor_before': before_cursor, 'cursor_attempted': before_cursor + int(kind == 'joint'),
                    'cursor_retained': cursor, 'model_before_sha256': values[0].sha256,
                    'optimizer_before_sha256': values[1].sha256, 'model_attempted_sha256': attempted[0],
                    'optimizer_attempted_sha256': attempted[1], 'model_retained_sha256': attempted[0],
                    'optimizer_retained_sha256': attempted[1], 'result': result,
                    'work_delta': {key: work[key] - previous_work[key] for key in WORK_KEYS}})
            retained_hashes = hashes(optimizer)
            checked()
            stopped = stamp() - started
            timely(stopped, 'stage completion')
        except BaseException as error:
            stages.append({'stage': index, 'kind': kind, 'target_updates': target,
                'deadline_seconds': float(max_seconds), 'start_elapsed': stage_start,
                'stopped_elapsed': last_time - started, 'status': 'FAILED', 'termination': repr(error),
                'trace_start': start_index, 'trace_stop': len(trace), 'attempted_updates': len(trace) - start_index,
                'accepted_updates': accepted, 'joint_cursor_start': start_cursor, 'joint_cursor_end': cursor,
                'model_start_sha256': model_start, 'optimizer_start_sha256': optimizer_start,
                'model_end_sha256': retained_hashes[0], 'optimizer_end_sha256': retained_hashes[1],
                'overrun_seconds': max(0., last_time - started - max_seconds)})
            raise
        row = {'stage': index, 'kind': kind, 'target_updates': target,
            'deadline_seconds': float(max_seconds), 'start_elapsed': stage_start,
            'stopped_elapsed': stopped, 'status': 'PASS', 'termination': 'completed_updates',
            'trace_start': start_index, 'trace_stop': len(trace), 'attempted_updates': len(trace) - start_index,
            'accepted_updates': accepted, 'joint_cursor_start': start_cursor, 'joint_cursor_end': cursor,
            'model_start_sha256': model_start, 'optimizer_start_sha256': optimizer_start,
            'model_end_sha256': retained_hashes[0], 'optimizer_end_sha256': retained_hashes[1],
            'overrun_seconds': 0.}
        stages.append(row)
        return row

    try:
        started = stamp()
        require(type(hooks) is Hooks and callable(check) and callable(clock), 'exact allocation hooks and callable clock/check')
        require(all(callable(getattr(hooks, name)) for name in hooks.__dataclass_fields__), 'callable numerical hooks')
        require(type(prefix_updates) is int and prefix_updates > 0
                and type(joint_updates) is int and joint_updates > 0
                and prefix_updates + joint_updates <= MAX_UPDATES, 'positive integer update counts with total at most100000')
        require(finite_number(max_seconds) and max_seconds > 0,
                'positive finite safety deadline')
        guard('before validation')
        work['validation_calls'] += 1
        metadata = hooks.validate()
        require(type(metadata) is dict, 'validation metadata mapping')
        metadata = _json(metadata)
        guard('validation completion')
        optimizer = new_optimizer('prefix')
        initial_hashes = hashes(optimizer)
        guard('initial optimizer setup')
        read_only('initial', optimizer)
        first = one_stage(1, 'prefix', optimizer, prefix_updates)
        read_only('boundary', optimizer)
        boundary = {'joint_cursor': cursor, 'model_sha256': first['model_end_sha256'],
            'stage1_optimizer_kind': 'prefix', 'optimizer_before_sha256': first['optimizer_end_sha256'],
            'optimizer_after_sha256': None, 'stage2_optimizer_kind': 'joint', 'optimizer_reset': False}
        optimizer = new_optimizer('joint')
        boundary['optimizer_reset'] = True
        model_hash, optimizer_hash = hashes(optimizer)
        require(model_hash == boundary['model_sha256'] and cursor == 0, 'fresh joint optimizer preserves model and starts cursor zero')
        boundary['optimizer_after_sha256'] = optimizer_hash
        guard('joint optimizer setup')
        one_stage(2, 'joint', optimizer, joint_updates)
        read_only('final', optimizer)
        final_hashes = hashes(optimizer)
        summary_before = work.copy()
        summary_started = last_time - started
        final_summary_record, _delta = read_only('summary', optimizer, summary=True)
        final_summary = final_summary_record['metadata']
        final_summary_seconds = final_summary_record['completed_elapsed'] - summary_started
        final_summary_work = {key: work[key] - summary_before[key] for key in WORK_KEYS}
        require(len(trace) == prefix_updates + joint_updates and all(row['accepted'] for row in trace)
                and cursor == joint_updates, 'complete exact update schedule')
        return {'version': VERSION, 'arm': 'prefix_then_joint', 'status': 'PASS', 'termination': 'completed_updates',
            'prefix_updates': prefix_updates, 'joint_updates': joint_updates, 'max_seconds': float(max_seconds),
            'max_attempts': MAX_UPDATES, 'metadata': metadata, 'initial_model_sha256': initial_hashes[0],
            'initial_optimizer_sha256': initial_hashes[1], 'final_model_sha256': final_hashes[0],
            'final_optimizer_sha256': final_hashes[1], 'joint_cursor': cursor,
            'attempted_updates': len(trace), 'accepted_updates': len(trace),
            'accepted_joint_updates': joint_updates, 'accepted_prefix_updates': prefix_updates,
            'trace': trace, 'stages': stages, 'boundary': boundary, 'checkpoints': checkpoints,
            'work': work, 'update_work': update_work, 'timed_seconds': last_time - started,
            'overrun_seconds': 0., 'final_summary': final_summary, 'final_summary_record': final_summary_record,
            'final_summary_seconds': final_summary_seconds, 'final_summary_work': final_summary_work,
            'compute_matched': False, 'work_scope': 'work includes final_summary_work; that field is a subset, not additive.',
            'timing_scope': 'One safety clock through validation, setup, snapshots, exact updates, all checkpoints and final summary; no time-based partial success.'}
    except BaseException as error:
        # Preserve the original failure while measuring completed rollback and
        # exception processing. A defective clock never becomes a new success.
        if started is not None:
            try:
                stamp()
            except BaseException as timing_error:  # noqa: BLE001 - retain clock failure while re-raising the original error
                error.failure_timing_error = repr(timing_error)
        elapsed = None if last_time is None or started is None else last_time - started
        partial = {'version': VERSION, 'arm': 'prefix_then_joint',
            'status': 'FAILED_TIMEOUT' if isinstance(error, TimeoutError) else 'FAILED_EXCEPTION',
            'termination': 'safety_deadline' if isinstance(error, TimeoutError) else 'exception', 'error': repr(error),
            'prefix_updates': prefix_updates if type(prefix_updates) is int else None,
            'joint_updates': joint_updates if type(joint_updates) is int else None,
            'max_seconds': float(max_seconds) if finite_number(max_seconds) else None,
            'metadata': metadata, 'trace': trace, 'stages': stages, 'boundary': boundary,
            'checkpoints': checkpoints, 'work': work, 'update_work': update_work, 'joint_cursor': cursor,
            'attempted_updates': len(trace), 'accepted_updates': sum(row['accepted'] for row in trace),
            'accepted_joint_updates': sum(row['accepted'] and row['kind'] == 'joint' for row in trace),
            'accepted_prefix_updates': sum(row['accepted'] and row['kind'] == 'prefix' for row in trace),
            'last_elapsed': elapsed, 'timed_seconds': elapsed,
            'overrun_seconds': max(0., elapsed - max_seconds) if elapsed is not None and finite_number(max_seconds) else None,
            'final_summary_record': getattr(error, 'final_summary_record', final_summary_record),
            'final_summary_work': getattr(error, 'final_summary_work', final_summary_work),
            'final_summary_seconds': None if summary_started is None or elapsed is None else elapsed - summary_started,
            'compute_matched': False}
        if hasattr(error, 'rollback_error'):
            partial['rollback_error'] = error.rollback_error
        if hasattr(error, 'failure_timing_error'):
            partial['failure_timing_error'] = error.failure_timing_error
        error.allocation_result = partial
        raise
