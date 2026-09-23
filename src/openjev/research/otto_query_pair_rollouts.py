"""Two endpoint actions with unchanged public analytic continuation mechanics.

This additive adapter changes only the declared first-action set. One source is
shared per replicate, each distinct action restarts the same hit stream, and
identical endpoints reference one physical continuation. No policy admission,
neural scoring, native environment or filesystem operations occur here.
"""
from __future__ import annotations

from openjev.research import otto_teacher_rollouts as inherited
from openjev.research.otto_released_policy import _integer

VERSION = "otto-query-pair-rollouts-v1"


def sample_query_pair(public, belief, kernel, *, analytic_action, neural_action,
                      seed, anchor_id, replicate_ids, horizon, check=None, emit=None):
    """Return all unique action/replicate records, with no partial-panel fallback.

    The sign/meaning of endpoint differences belongs to the reducer. Record
    order is ascending replica then action, never endpoint preference. The
    supplied public posterior is already assimilated; no hidden source or
    actual-trajectory random generator is accepted. Horizon includes move one.
    """
    seed = _integer(seed, "seed", 0, inherited.UINT32_MAX)
    anchor_id = _integer(anchor_id, "anchor_id", 0, inherited.UINT32_MAX)
    horizon = _integer(horizon, "horizon", 1, inherited.MAX_HORIZON)
    analytic_action = _integer(analytic_action, "analytic action", 0, 3)
    neural_action = _integer(neural_action, "neural action", 0, 3)
    replicas = tuple(_integer(r, "replicate ID", 0, inherited.UINT32_MAX) for r in replicate_ids)
    if not replicas or len(set(replicas)) != len(replicas):
        raise ValueError("replicate IDs must be nonempty and distinct")
    if check is not None and not callable(check) or emit is not None and not callable(emit):
        raise TypeError("check and emit must be callable or None")
    events = inherited._Events(seed, anchor_id, check, emit)
    anchor = events.call("anchor_snapshot", inherited.TeacherSnapshot, public, belief, kernel,
                         describe=inherited._snapshot_witness)
    actions = tuple(sorted({analytic_action, neural_action}))
    if not set(actions) <= set(anchor.public["valid_actions"]):
        raise ValueError("both endpoint actions must be geometrically eligible")
    public, belief, kernel = anchor.public, anchor.belief, anchor._view.p_Poisson
    del anchor
    events.write("pair_declaration", adapter_version=VERSION, analytic_action=analytic_action,
                 neural_action=neural_action, distinct_actions=list(actions),
                 replicate_ids=sorted(replicas), horizon=horizon)
    records = []
    for replica in sorted(replicas):
        context = {"replicate_id": replica}
        generator = events.call("source_generator", inherited._generator, seed, anchor_id,
                                replica, inherited.SOURCE_CHANNEL, context=context)
        draw = events.call("source_draw", inherited._source_draw, generator, belief,
                           context=context, describe=inherited._identity)
        for action in actions:
            records.append(inherited._continuation(events, public, belief, kernel,
                tuple(draw["source"]), seed, anchor_id, replica, action, horizon))
    events.guard()
    events.write("panel_complete", record_count=len(records), operation_count=events.operations,
                 adapter_version=VERSION)
    return tuple(records)
