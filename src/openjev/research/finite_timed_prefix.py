"""Timed prefix pretraining with atomic acceptance and charged late work.

The budget determines update eligibility, not identical executed compute.
Validation, copies, optimizer creation and before/after objective checks are
timed. An update completing exactly at the budget is eligible. A late update
is restored to the previous accepted dynamic parameter bytes; all its work
and rollback time remain charged. Adam is persistent across attempted steps
but is discarded when this call ends, including after a late step. Final
diagnostics/summary construction have their own timer outside eligibility.

This module neither selects by likelihood nor retries a stopped run. EM checks
the same penalized likelihood before/after each full update; gradient updates
need not be monotone. There is no world generation, hidden-state input, cost
head optimization, checkpoint IO or automatic scientific admission.
"""
from __future__ import annotations

import hashlib
import math
import time

import torch

from openjev.research.finite_expected_count import expected_counts, map_update, tokens_from_prefix
from openjev.research.finite_expected_count_bridge import (
    DYNAMICS,
    import_probabilities,
    numpy_objective,
    probabilities,
    require,
    torch_objective,
    validate_model,
)

VERSION = 'finite-timed-prefix-v1'
MAX_UPDATES = 100000
PSEUDOCOUNT = .001
LEARNING_RATE = .003
GRADIENT_CLIP = 5.
WORK_KEYS = ('full_gradient_passes', 'torch_diagnostic_passes', 'numpy_expectation_passes',
             'map_updates', 'adam_updates', 'probability_imports', 'probability_checks',
             'gradient_event_exposures', 'expectation_event_exposures',
             'parameter_snapshot_tensors', 'parameter_snapshot_bytes',
             'parameter_restore_tensors', 'parameter_restore_bytes', 'parameter_hash_calls',
             'check_calls', 'clock_calls')


def _hash(model, *, head=False):
    digest = hashlib.sha256()
    for name in ('cost_logits',) if head else DYNAMICS:
        if not head:
            digest.update(name.encode() + b'\0')
        digest.update(getattr(model, name).detach().numpy().astype('<f8', copy=False).tobytes(order='C'))
    return digest.hexdigest()


def timed_fit(model, prefix, lengths, method, budget_seconds, check=lambda: None, clock=time.perf_counter):
    """Return JSON-ready full trace; cap exhaustion is a retained failure.

    ``check`` owns the enclosing process/resource deadline and exceptions
    propagate. ``clock`` must supply finite nondecreasing seconds. The default
    clock is monotonic, not suspend-inclusive; the outer supervisor owns the
    stronger phase bound. Only methods em/gradient are accepted; no-arm bypass
    belongs to the caller. Both use the same full-batch prior/objective.
    """
    work = dict.fromkeys(WORK_KEYS, 0)
    last_clock = None

    def stamp():
        nonlocal last_clock
        work['clock_calls'] += 1
        value = clock()
        require(type(value) in (int, float) and math.isfinite(value)
                and (last_clock is None or value >= last_clock), 'finite nondecreasing clock')
        last_clock = float(value)
        return last_clock

    def checked():
        work['check_calls'] += 1
        check()

    start = stamp()
    require(callable(check) and callable(clock), 'callable bound and monotonic clock')
    require(method in ('em', 'gradient') and type(budget_seconds) in (int, float)
            and math.isfinite(budget_seconds) and budget_seconds > 0, 'method and positive eligibility budget')
    checked()
    validate_model(model)
    with torch.no_grad():
        original = torch_objective(model, prefix, lengths, PSEUDOCOUNT)
    work['torch_diagnostic_passes'] += 1
    tokens = tokens_from_prefix(prefix.detach().numpy(), lengths.detach().numpy())
    events = original['valid_events']
    head = model.cost_logits.detach().clone()
    initial_hash, initial_head_hash = _hash(model), _hash(model, head=True)
    work['parameter_hash_calls'] += 2
    parameters = [getattr(model, name) for name in DYNAMICS]
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE) if method == 'gradient' else None
    trace, numpy_work = [], {}
    accepted = 0
    status, termination = 'PASS', 'budget_reached'

    def expectation():
        checked()
        values = probabilities(model)
        work['probability_checks'] += 1
        result = expected_counts(*values, tokens['actions'], tokens['observations'], tokens['lengths'], check=checked)
        work['numpy_expectation_passes'] += 1
        work['expectation_event_exposures'] += events
        for key, value in result['work'].items():
            numpy_work[key] = numpy_work.get(key, 0) + value
        return values, result

    def restore(snapshot):
        with torch.no_grad():
            for parameter, previous in zip(parameters, snapshot, strict=True):
                parameter.copy_(previous)
                work['parameter_restore_tensors'] += 1
                work['parameter_restore_bytes'] += previous.numel() * previous.element_size()

    while True:
        checked()
        began = stamp() - start
        if began >= budget_seconds:
            break
        if len(trace) >= MAX_UPDATES:
            status, termination = 'FAILED_UPDATE_CAP', 'update_cap_before_deadline'
            break
        prior_work = work.copy()
        snapshot = [parameter.detach().clone() for parameter in parameters]
        work['parameter_snapshot_tensors'] += len(snapshot)
        work['parameter_snapshot_bytes'] += sum(value.numel() * value.element_size() for value in snapshot)
        before_hash = _hash(model)
        work['parameter_hash_calls'] += 1
        roundtrip = 0.
        try:
            if method == 'em':
                values, counts = expectation()
                before = numpy_objective(*values, counts, PSEUDOCOUNT)
                update = map_update(*values, counts['counts'], pseudocount=PSEUDOCOUNT)
                work['map_updates'] += 1
                roundtrip = import_probabilities(model, update['transition'], update['emission'], update['hazard'])
                work['probability_imports'] += 1
                actual, after_counts = expectation()
                after = numpy_objective(*actual, after_counts, PSEUDOCOUNT)
                require(after >= before - 1e-9, 'penalized EM likelihood decreased beyond roundoff')
            else:
                optimizer.zero_grad(set_to_none=True)
                objective = torch_objective(model, prefix, lengths, PSEUDOCOUNT)
                before = float((objective['log_likelihood'] + objective['log_prior']).detach())
                objective['loss'].backward()
                torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP, error_if_nonfinite=True)
                optimizer.step()
                work['full_gradient_passes'] += 1
                work['adam_updates'] += 1
                work['gradient_event_exposures'] += events
                with torch.no_grad():
                    diagnostic = torch_objective(model, prefix, lengths, PSEUDOCOUNT)
                work['torch_diagnostic_passes'] += 1
                after = float(diagnostic['log_likelihood'] + diagnostic['log_prior'])
            require(torch.equal(head, model.cost_logits), 'unchanged cost head for every attempted update')
            probabilities(model)
            work['probability_checks'] += 1
            attempted_hash = _hash(model)
            work['parameter_hash_calls'] += 1
            checked()
            completed = stamp() - start
        except BaseException:
            restore(snapshot)
            raise
        eligible = completed <= budget_seconds
        if eligible:
            accepted += 1
        else:
            restore(snapshot)
            require(_hash(model) == before_hash, 'late update restored exact previous parameter bytes')
            work['parameter_hash_calls'] += 1
            termination = 'late_update_rolled_back'
        trace.append({'update': len(trace) + 1, 'start_elapsed': began, 'completed_elapsed': completed,
                      'seconds': completed - began, 'accepted': eligible, 'rolled_back': not eligible,
                      'penalized_log_likelihood_before': before, 'penalized_log_likelihood_after': after,
                      'roundtrip_max_abs': roundtrip, 'dynamics_before_sha256': before_hash,
                      'dynamics_attempted_sha256': attempted_hash,
                      'dynamics_retained_sha256': attempted_hash if eligible else before_hash,
                      'work_delta': {key: work[key] - prior_work[key] for key in WORK_KEYS}})
        if not eligible or completed == budget_seconds:
            break
        if len(trace) >= MAX_UPDATES:
            status, termination = 'FAILED_UPDATE_CAP', 'update_cap_before_deadline'
            break
    checked()
    stopped = stamp()
    timed_work = work.copy()
    # All operations below are outside update eligibility and timed separately.
    with torch.no_grad():
        final = torch_objective(model, prefix, lengths, PSEUDOCOUNT)
    final_hash, final_head_hash = _hash(model), _hash(model, head=True)
    require(final_head_hash == initial_head_hash and torch.equal(head, model.cost_logits), 'unchanged final cost head')
    expected_hash = trace[-1]['dynamics_retained_sha256'] if trace else initial_hash
    require(final_hash == expected_hash, 'final dynamic bytes equal last accepted state')
    check()
    result = {'version': VERSION, 'method': method, 'status': status, 'termination': termination,
              'budget_seconds': float(budget_seconds), 'max_updates': MAX_UPDATES, 'pseudocount': PSEUDOCOUNT,
              'learning_rate': LEARNING_RATE, 'gradient_clip': GRADIENT_CLIP,
              'accepted_updates': accepted, 'attempted_updates': len(trace), 'trace': trace,
              'valid_events': events, 'attempts': len(prefix),
              'initial_penalized_log_likelihood': float(original['log_likelihood'] + original['log_prior']),
              'final_penalized_log_likelihood': float(final['log_likelihood'] + final['log_prior']),
              'initial_dynamics_sha256': initial_hash, 'final_dynamics_sha256': final_hash,
              'initial_head_sha256': initial_head_hash, 'final_head_sha256': final_head_hash,
              'head_unchanged': True, 'work': timed_work, 'numpy_work': numpy_work,
              'timed_seconds': stopped - start, 'overrun_seconds': max(0., stopped - start - budget_seconds),
              'final_summary_work': {'torch_diagnostic_passes': 1, 'parameter_hash_calls': 2,
                                     'check_calls': 1, 'clock_calls': 1},
              'compute_matched': False, 'hidden_state_input': False,
              'optimizer_state_reused_across_steps': method == 'gradient',
              'late_optimizer_state_discarded': method == 'gradient' and termination == 'late_update_rolled_back'}
    result['final_summary_seconds'] = stamp() - stopped
    return result
