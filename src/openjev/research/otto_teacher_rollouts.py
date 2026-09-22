"""Paired, capped analytic-teacher continuations from one public anchor.

This is a sampler component, not empirical-study admission. The caller owns a
frozen allocation, source/input authentication, budget enforcement and durable
failure evidence. No simulator, learned model, file or external service is used.
Source and observation draws are evaluator-only; the fixed teacher receives
only its copied public belief, packet, kernel and subsequent public packets.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real

import numpy as np

from openjev.research.otto_released_policy import EPSILON, N, _integer, _move, _position
from openjev.research.otto_teacher_costs import ContinuationRecord
from openjev.research.otto_teacher_snapshot import TeacherSnapshot

VERSION = "otto-teacher-rollouts-v1"
NAMESPACE = 0x4F54544F
SOURCE_CHANNEL, HIT_CHANNEL = 0, 1
UINT32_MAX = 2**32 - 1
MAX_HORIZON = 2188


def _categorical(probabilities):
    """Copy probabilities and normalize their CDF only for summation roundoff."""
    values = np.asarray(probabilities)
    if values.dtype.kind not in "fiu" or values.ndim != 1 or not values.size:
        raise ValueError("categorical probabilities must be a nonempty real vector")
    probability = np.array(values, dtype=np.float64, copy=True)
    mass = float(np.sum(probability, dtype=np.float64))
    if (not np.isfinite(probability).all() or np.any(probability < 0)
            or not np.isfinite(mass) or abs(mass - 1.0) > EPSILON):
        raise ValueError("categorical probabilities must be finite, nonnegative and normalized")
    cumulative = np.cumsum(probability, dtype=np.float64)
    cdf_mass = float(cumulative[-1])
    if not np.isfinite(cdf_mass) or cdf_mass <= 0:
        raise ValueError("categorical CDF mass must be positive and finite")
    cumulative /= cdf_mass
    return probability, cumulative, cdf_mass


def _select(probability, cumulative, uniform):
    if isinstance(uniform, (bool, np.bool_)) or not isinstance(uniform, Real):
        raise TypeError("uniform must be a real scalar")
    uniform = float(uniform)
    if not np.isfinite(uniform) or not 0 <= uniform < 1:
        raise ValueError("uniform must be finite in [0, 1)")
    index = int(np.searchsorted(cumulative, uniform, side="right"))
    if index >= len(probability) or probability[index] <= 0:
        raise FloatingPointError("categorical selection left the positive support")
    return index


def categorical_index(probabilities, uniform):
    """Inverse-CDF selection, matching the seeded adapter's right-edge rule.

    Total mass must be within 1e-10 of one. Only the cumulative sum is divided
    by its own final mass; no entries are floored, clipped or otherwise repaired.
    A zero-probability entry cannot be selected, including when uniform is zero.
    """
    probability, cumulative, _ = _categorical(probabilities)
    return _select(probability, cumulative, uniform)


def source_hit_probabilities(snapshot, source, position):
    """Return the four source-conditioned probabilities at a nonfound position.

    The snapshot supplies its owned immutable kernel, whose dtype, shape,
    finiteness, range and zero origin were validated during construction. This
    evaluator helper passes no source coordinate to a teacher method or state.
    The returned array is a copy. No kernel probabilities are repaired.
    """
    if not isinstance(snapshot, TeacherSnapshot):
        raise TypeError("a validated TeacherSnapshot is required")
    source, position = _position(source), _position(position)
    if source == position:
        raise ValueError("a found source has no odor distribution or draw")
    x, y = N + source[0] - position[0], N + source[1] - position[1]
    probability = snapshot._view.p_Poisson[:, x, y].copy()
    _categorical(probability)
    return probability


class _Events:
    """Scalar/public evidence only; never retain a trajectory or callback log."""

    def __init__(self, seed, anchor_id, check, emit):
        self.base = {"version": VERSION, "seed": seed, "anchor_id": anchor_id}
        self.check, self.emit, self.operations = check, emit, 0

    def guard(self):
        if self.check is not None:
            self.check()

    def write(self, event, **fields):
        if self.emit is not None:
            # A logger cannot mutate live packets, contexts or probability lists.
            self.emit(deepcopy({**self.base, "event": event, **fields}))

    def call(self, operation, function, *args, context=None, describe=None):
        self.guard()
        self.operations += 1
        identity = {**(context or {}), "operation_id": self.operations, "operation": operation}
        self.write("attempt", **identity)
        result = function(*args)
        fields = {} if describe is None else describe(result)
        # No budget check between a returned operation and its return witness.
        # If emission itself fails, the caller retains an uncertain attempt.
        self.write("return", **identity, **fields)
        return result


def _generator(seed, anchor_id, replicate_id, channel):
    entropy = [NAMESPACE, seed, anchor_id, replicate_id, channel]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _draw(generator, probabilities):
    probability, cumulative, cdf_mass = _categorical(probabilities)
    uniform = float(generator.random())
    return {"uniform": uniform, "cdf_mass": cdf_mass,
            "selected_index": _select(probability, cumulative, uniform)}


def _source_draw(generator, belief):
    result = _draw(generator, belief.reshape(-1, order="C"))
    result["source"] = list(divmod(result["selected_index"], N))
    return result


def _hit_draw(generator, snapshot, source, position, local_step):
    probability = source_hit_probabilities(snapshot, source, position)
    result = _draw(generator, probability)
    result.update(probabilities=probability.tolist(), draw_index=local_step - 1)
    return result


def _identity(value):
    return value


def _snapshot_witness(snapshot):
    return {"public": snapshot.public}


def _choice_witness(choice):
    action, scores = choice
    return {"action": action, "scores": [float(x) if np.isfinite(x) else None for x in scores]}


def _movement(position, action, source):
    moved, possible = _move(action, position)
    if not possible:
        raise ValueError("teacher continuations permit in-bounds moves only")
    return {"position": moved, "found": tuple(moved) == source}


def _update(snapshot, action, packet):
    snapshot.update(action, packet)
    return {"public": snapshot.public}


def _continuation(events, public, belief, kernel, source, seed, anchor_id, replica, first, horizon):
    context = {"replicate_id": replica, "first_action": first}
    teacher = events.call("teacher_snapshot", TeacherSnapshot, public, belief, kernel,
                          context=context, describe=_snapshot_witness)
    hits = events.call("hit_generator", _generator, seed, anchor_id, replica, HIT_CHANNEL,
                       context=context)
    # Each action restarts the same hit stream, so local nonfound step t uses
    # the same U_t. A found path terminates without consuming that step's U_t.
    for local_step in range(1, horizon + 1):
        current = teacher.public
        step_context = {**context, "local_step": local_step, "from_step": current["step"]}
        if local_step == 1:
            action = first
            events.guard()
            events.write("forced_action", **step_context, action=action)
        else:
            action, _ = events.call("teacher_choose", teacher.choose,
                                    context=step_context, describe=_choice_witness)
        moved = events.call("movement", _movement, current["position"], action, source,
                            context=step_context, describe=_identity)
        found, position = moved["found"], tuple(moved["position"])
        if found:
            hit = -2
        else:
            draw = events.call("hit_draw", _hit_draw, hits, teacher, source, position, local_step,
                               context=step_context, describe=_identity)
            hit = draw["selected_index"]
        packet = {"position": position, "hit": hit, "done": found, "step": current["step"] + 1,
                  "valid_actions": () if found else tuple(a for a in range(4) if _move(a, position)[1])}
        events.call("teacher_update", _update, teacher, action, packet,
                    context=step_context, describe=_identity)
        if found or local_step == horizon:
            record = ContinuationRecord(replica, first, local_step, found)
            events.write("record", **context, steps=record.steps, found=record.found)
            return record
    raise AssertionError("validated positive horizon must complete or propagate an error")


def sample_teacher_panel(public, belief, kernel, *, seed, anchor_id, replicate_ids,
                         horizon, first_actions=None, check=None, emit=None):
    """Return every declared replicate/action continuation, or propagate failure.

    Seed, anchor and replicate IDs are distinct fixed-width uint32 entropy
    components, not concatenated strings. Replicate IDs must be nonempty and
    unique. Every geometrically eligible first action is required exactly once;
    both declarations are processed in ascending numeric order. The horizon is
    task-specific, 1..2188 moves, including the forced first action.

    Exactly one source uniform is drawn per replicate from the normalized public
    anchor. Source channel0 and hit channel1 use PCG64(SeedSequence([0x4F54544F,
    seed, anchor_id, replicate_id, channel])). A fresh fixed analytic teacher
    and same-seeded hit generator serve each action. No odor is drawn at source
    discovery. Absolute public steps advance from the supplied anchor; even the
    final unsuccessful horizon observation is assimilated. There is no loop stop.

    An explicitly journaled validation snapshot precedes per-action snapshots.
    Optional check() runs before operations, and optional emit(dict) streams
    detached evaluator-only attempts, returns, draws and completed records.
    Neither callback is passed to the actor. A caller must make emission durable
    and retain primary failures; this function neither catches interruptions nor
    returns a partial panel or converts an exception into a censored result.
    Memory holds the fixed anchor, one continuation and completed cost records,
    not full trajectories. These mechanics still require independent sampler
    qualification and a frozen empirical protocol before collecting real labels.
    """
    seed = _integer(seed, "seed", 0, UINT32_MAX)
    anchor_id = _integer(anchor_id, "anchor_id", 0, UINT32_MAX)
    horizon = _integer(horizon, "horizon", 1, MAX_HORIZON)
    replicas = tuple(_integer(x, "replicate ID", 0, UINT32_MAX) for x in replicate_ids)
    if not replicas or len(set(replicas)) != len(replicas):
        raise ValueError("replicate IDs must be nonempty and distinct")
    if check is not None and not callable(check) or emit is not None and not callable(emit):
        raise TypeError("check and emit must be callable or None")
    events = _Events(seed, anchor_id, check, emit)
    anchor = events.call("anchor_snapshot", TeacherSnapshot, public, belief, kernel,
                         describe=_snapshot_witness)
    public, belief, kernel = anchor.public, anchor.belief, anchor._view.p_Poisson
    eligible = public["valid_actions"]
    actions = eligible if first_actions is None else tuple(
        _integer(x, "first action", 0, 3) for x in first_actions)
    if len(actions) != len(eligible) or set(actions) != set(eligible):
        raise ValueError("first_actions must contain every eligible action exactly once")
    actions, replicas = tuple(sorted(actions)), tuple(sorted(replicas))
    del anchor
    records = []
    for replica in replicas:
        context = {"replicate_id": replica}
        source_rng = events.call("source_generator", _generator, seed, anchor_id, replica, SOURCE_CHANNEL,
                                 context=context)
        draw = events.call("source_draw", _source_draw, source_rng, belief,
                           context=context, describe=_identity)
        source = tuple(draw["source"])
        for first in actions:
            records.append(_continuation(events, public, belief, kernel, source, seed, anchor_id,
                                         replica, first, horizon))
    events.guard()
    events.write("panel_complete", record_count=len(records), operation_count=events.operations)
    return tuple(records)
