"""Select and reconstruct a fixed learner-TRAIN cohort without drawing labels.

The caller authenticates the original metadata/public journals and kernels,
freezes the selected identities, and owns budgets and durable event handling.
Selection uses only episode identities and retained-prefix order. Reconstruction
preserves every selected posterior, including unsupported beliefs. Separate
support assessment reports every anchor without repair, replacement or dropping.
No file, simulator, learned model, source coordinate or random stream is consumed.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from openjev.research import otto_teacher_anchors as original
from openjev.research.otto_released_policy import PublicBeliefView, _move, _packet
from openjev.research.otto_teacher_snapshot import TeacherSnapshot

VERSION = "otto-teacher-cohort-v1"
EXPECTED_ROWS, EXPECTED_EPISODES, PREFIXES_PER_EPISODE = 4596, 144, 4
MAX_ANCHORS = EXPECTED_EPISODES * PREFIXES_PER_EPISODE
EPISODES = tuple(
    (regime, seed, arm, f"dagger:{regime}:{seed}:{arm}")
    for regime in sorted(original.FIRST)
    for seed in range(original.FIRST[regime], original.FIRST[regime] + 12)
    for arm in sorted(original.COLLECTORS)
)
BELIEF_DOMAIN_ERRORS = frozenset({
    "anchor belief must be float64[53,53]",
    "anchor belief must be finite and nonnegative",
    "anchor belief must already be normalized; no repair",
    "nonterminal anchor must have exactly zero current-cell mass",
})
_decode_record = original._decode_record


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _order(row):
    return (row["regime"], row["seed"], row["arm"], row["prefix_index"], row["row_index"])


def select_cohort(metadata_iter):
    """Freeze at most four floor-spaced retained prefixes per original episode.

    All 4,596 retained rows and all 144 original episode identities are required.
    Prefix zero is included so initial actions are represented. An episode with
    fewer than four retained rows contributes all its rows, without replacement.
    Duplicate row IDs or episode/prefix pairs are provenance failures. Original
    analytic scores and posterior mass do not participate in the selection key.
    """
    grouped = {episode: [] for _, _, _, episode in EPISODES}
    seen_rows, seen_prefixes = set(), set()
    for item in metadata_iter:
        row = original._metadata(item)
        pair = row["episode_id"], row["prefix_index"]
        _require(row["row_index"] not in seen_rows and pair not in seen_prefixes,
                 "duplicate retained row or episode prefix")
        seen_rows.add(row["row_index"])
        seen_prefixes.add(pair)
        grouped[row["episode_id"]].append(row)
    _require(seen_rows == set(range(EXPECTED_ROWS)), "exact complete retained TRAIN row indices required")
    _require(all(grouped.values()), "each declared original episode must have retained metadata")
    selected, diagnostics = [], []
    for regime, seed, arm, episode in EPISODES:
        retained = sorted(grouped[episode], key=lambda r: (r["prefix_index"], r["row_index"]))
        count = min(PREFIXES_PER_EPISODE, len(retained))
        indices = ([0] if count == 1 else
                   [j * (len(retained) - 1) // (count - 1) for j in range(count)])
        chosen = [retained[i] for i in indices]
        selected.extend(chosen)
        diagnostics.append({"episode_id": episode, "regime": regime, "seed": seed, "arm": arm,
                            "retained_rows": len(retained), "selected_rows": count,
                            "selected_retained_indices": indices, "fewer_than_four": count < 4})
    selections = tuple({"anchor_id": i, **row} for i, row in enumerate(sorted(selected, key=_order)))
    return {"selections": selections, "episodes": tuple(diagnostics),
            "counts": {"metadata_rows": EXPECTED_ROWS, "episodes": EXPECTED_EPISODES,
                       "selected_anchors": len(selections),
                       "fewer_than_four_episodes": sum(d["fewer_than_four"] for d in diagnostics)}}


def _selections(selections):
    """Validate an already frozen cohort, including small caller-owned fixtures."""
    _require(isinstance(selections, (tuple, list)) and len(selections) <= MAX_ANCHORS,
             "bounded ordered selection required")
    clean = []
    for index, row in enumerate(selections):
        _require(isinstance(row, Mapping) and set(row) == original.SELECTION,
                 "exact frozen selection fields required")
        _require(type(row["anchor_id"]) is int and row["anchor_id"] == index,
                 "contiguous canonical anchor IDs required")
        metadata = original._metadata({k: v for k, v in row.items() if k != "anchor_id"}
                                      | {"teacher_costs": None})
        _require(metadata["row_index"] < EXPECTED_ROWS, "selected prefixes must be original retained rows")
        clean.append({"anchor_id": index, **metadata})
    _require(clean == sorted(clean, key=_order), "canonical selected-anchor order required")
    _require(len({r["row_index"] for r in clean}) == len(clean)
             and len({(r["episode_id"], r["prefix_index"]) for r in clean}) == len(clean),
             "selected provenance must be unique")
    episodes = {}
    for row in clean:
        episodes[row["episode_id"]] = episodes.get(row["episode_id"], 0) + 1
    _require(all(n <= PREFIXES_PER_EPISODE for n in episodes.values()), "at most four selections per episode")
    return tuple(clean)


def iter_public_records(raw_lines, selections):
    """Project only allowlisted public history through each last selected prefix.

    The authenticated producer header is checked before JSON decoding. For an
    allowed episode, numerical tokens in truth, costs and timing remain strings
    and are discarded; public values after its selected limit are not decoded.
    This generalizes the frozen six-anchor projector without changing that file.
    """
    limits = {}
    for row in _selections(selections):
        limits[row["episode_id"]] = max(limits.get(row["episode_id"], 0), row["prefix_index"])
    for raw in raw_lines:
        _require(isinstance(raw, (bytes, str)), "raw public JSONL text required")
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        match = original.HEADER.match(text)
        _require(match is not None, "original compact public header required")
        kind, episode = match.groups()
        if episode not in limits:
            continue
        record = _decode_record(text)
        _require(isinstance(record, dict) and record.get("kind") == kind
                 and record.get("episode_id") == episode, "public header identity mismatch")
        required = original.RESET if kind == "reset" else original.STEP
        _require(required <= set(record), "missing original transition fields")
        if kind == "step":
            step = original._integer(original._numbers(record["step"]), "transition step", 1, 2188)
            if step > limits[episode]:
                continue
        fields = original.PUBLIC_RESET if kind == "reset" else original.PUBLIC_STEP
        projected = {key: original._numbers(record[key]) for key in fields}
        step = 0 if kind == "reset" else projected["step"]
        projected["public"] = original._public(projected["public"], step)
        projected["posterior_after"] = original._witness(projected["posterior_after"])
        if kind == "reset":
            original._identity(projected)
            _require(projected["block"] is None, "learner TRAIN has no evaluation block")
        else:
            projected["action"] = original._integer(projected["action"], "action", 0, 3)
            _require(isinstance(projected["allowed_actions"], list), "ordered allowed actions required")
            projected["allowed_actions"] = [original._integer(a, "action", 0, 3)
                                            for a in projected["allowed_actions"]]
            projected["posterior_before"] = original._witness(projected["posterior_before"])
        yield projected


class _Events:
    def __init__(self, check, emit):
        _require(check is None or callable(check), "check must be callable")
        _require(emit is None or callable(emit), "emit must be callable")
        self.check, self.emit, self.count = check, emit, 0

    def begin(self, operation, **context):
        if self.check is not None:
            self.check()
        event = {"event": "attempt", "operation_id": self.count, "operation": operation, **context}
        self.count += 1
        if self.emit is not None:
            self.emit(copy.deepcopy(event))
        return event

    def end(self, event, **result):
        if self.emit is not None:
            self.emit(copy.deepcopy({**event, "event": "return", **result}))


def _kernel_mapping(kernels):
    _require(isinstance(kernels, Mapping) and set(kernels) == set(original.FIRST),
             "exactly the two authenticated public kernels required")


def reconstruct_cohort(selections, public_records, kernels, *, check=None, emit=None):
    """Replay once per selected episode, preserving every selected belief.

    This performs no support acceptance or TeacherSnapshot construction. A
    selected reset prefix is captured after its original initial hit exactly
    once. The caller retains all 144 episode diagnostics, including short
    episodes contributing fewer than four rows. Empty selections are supported
    for standalone callers; select_cohort always represents every episode.
    """
    selected = _selections(selections)
    _kernel_mapping(kernels)
    wanted = {}
    for row in selected:
        wanted.setdefault(row["episode_id"], {})[row["prefix_index"]] = row
    states, captured, events = {}, {}, _Events(check, emit)
    resets = updates = 0
    for record in public_records:
        _require(isinstance(record, Mapping) and record.get("episode_id") in wanted,
                 "nonallowlisted projected record")
        episode, kind = record["episode_id"], record.get("kind")
        expected = original.PUBLIC_RESET if kind == "reset" else original.PUBLIC_STEP
        _require(kind in ("reset", "step") and set(record) == expected, "exact public projection required")
        rows = wanted[episode]
        identity = next(iter(rows.values()))
        if kind == "reset":
            _require(episode not in states and record["block"] is None, "duplicate or invalid reset")
            _require(all(record[k] == identity[k] for k in original.IDENTITY), "reset identity mismatch")
            packet = original._public(record["public"], 0)
            _require(packet["position"] == [26, 26] and packet["hit"] == identity["initial_hit"]
                     and not packet["done"], "original nonterminal center reset required")
            event = events.begin("public_reset", episode_id=episode, step=0)
            view = PublicBeliefView(kernels[identity["regime"]])
            view._reset(packet["hit"])
            resets += 1
        else:
            _require(episode in states, "step before reset")
            view, before, previous = states[episode]
            step = original._integer(record["step"], "step", 1, max(rows))
            _require(step == before["step"] + 1 and not before["done"], "public chronology mismatch")
            action = original._integer(record["action"], "action", 0, 3)
            _require(record["allowed_actions"] == before["valid_actions"]
                     and action in before["valid_actions"], "saved action must be inbounds")
            _require(original._witness(record["posterior_before"]) == previous, "before-posterior mismatch")
            packet = original._public(record["public"], step)
            _require(packet["position"] == _move(action, before["position"])[0], "action-position mismatch")
            event = events.begin("public_update", episode_id=episode, step=step)
            view._observe(_packet(packet, step))
            updates += 1
        belief = view.p_source
        posterior = original._actual(belief)
        _require(posterior == original._witness(record["posterior_after"])
                 and view.agent == packet["position"], "exact public posterior witness required")
        states[episode] = (view, packet, posterior)
        capture = rows.get(packet["step"])
        if capture is not None:
            _require(packet == capture["public"] and posterior == capture["posterior"],
                     "selected metadata witness mismatch")
            captured[capture["anchor_id"]] = {**capture, "belief": original._immutable(belief),
                                              "sensing_length": float(identity["regime"][-1])}
        events.end(event, public=packet, posterior=posterior,
                   captured_anchor_ids=[] if capture is None else [capture["anchor_id"]])
    _require(len(captured) == len(selected) and set(states) == set(wanted), "missing selected public prefixes")
    _require(all(states[e][1]["step"] == max(rows) for e, rows in wanted.items()), "incomplete selected public path")
    return {"anchors": tuple(captured[i] for i in range(len(selected))),
            "counts": {"anchors": len(selected), "episodes": len(wanted),
                       "public_resets": resets, "public_updates": updates}}


def assess_support(anchors, kernels, *, check=None, emit=None):
    """Visit every selected belief and retain expected domain rejection details.

    Only the four frozen TeacherSnapshot belief-domain ValueErrors are handled.
    Wrong provenance/public packets, invalid shared kernels, callback failures,
    interrupts and unexpected exceptions propagate. A returned support assessment
    can contain a rejected constructor; snapshot_returns counts only actual
    successful constructors. No policy choice or sampler admission follows.
    """
    _require(isinstance(anchors, (tuple, list)) and len(anchors) <= MAX_ANCHORS,
             "bounded ordered captured anchors required")
    _require(all(isinstance(row, Mapping) and set(row) == original.ANCHOR for row in anchors),
             "exact captured anchor fields required")
    selected = _selections([{k: row[k] for k in original.SELECTION} for row in anchors])
    _kernel_mapping(kernels)
    # Shared kernel errors are provenance failures, never unsupported-belief labels.
    for kernel in kernels.values():
        _require(isinstance(kernel, np.ndarray) and kernel.dtype == np.float64
                 and kernel.shape == (4, 107, 107) and np.isfinite(kernel).all()
                 and (kernel >= 0).all() and (kernel <= 1).all()
                 and np.all(kernel[:, 53, 53] == 0), "invalid authenticated public kernel")
    result, events, returned = [], _Events(check, emit), 0
    for row, metadata in zip(anchors, selected, strict=True):
        _require(type(row["sensing_length"]) in (int, float)
                 and row["sensing_length"] == float(metadata["regime"][-1]), "sensing-length provenance mismatch")
        event = events.begin("anchor_support", anchor_id=metadata["anchor_id"],
                             episode_id=metadata["episode_id"], step=metadata["prefix_index"])
        error = None
        try:
            snapshot = TeacherSnapshot(metadata["public"], row["belief"], kernels[metadata["regime"]])
        except ValueError as failure:
            if str(failure) not in BELIEF_DOMAIN_ERRORS:
                raise
            error = {"type": type(failure).__name__, "message": str(failure)}
        else:
            returned += 1
            _require(original._actual(snapshot.belief) == metadata["posterior"], "accepted belief provenance mismatch")
        if (isinstance(row["belief"], np.ndarray) and row["belief"].dtype == np.float64
                and row["belief"].shape == (53, 53) and np.isfinite(row["belief"]).all()):
            _require(original._actual(row["belief"]) == metadata["posterior"],
                     "captured belief provenance mismatch, including unsupported anchors")
        assessment = {"anchor_id": metadata["anchor_id"], "episode_id": metadata["episode_id"],
                      "prefix_index": metadata["prefix_index"], "supported": error is None, "error": error}
        result.append(assessment)
        events.end(event, supported=error is None, error=error, snapshot_returned=error is None)
    return {"assessments": tuple(result), "counts": {"anchors": len(anchors), "supported": returned,
            "unsupported": len(anchors)-returned, "snapshot_attempts": len(anchors), "snapshot_returns": returned}}
