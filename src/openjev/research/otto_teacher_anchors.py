"""Extract six fixed TRAIN anchors using public records only.

The caller authenticates complete source manifests, freezes the selected IDs,
and owns I/O, budgets and durable events. This module reads no files and draws
no samples. It never imports a simulator or learned model. Reconstruction uses
the qualified public filter unchanged; a selected unsupported state fails with
no replacement or numerical repair.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping

import numpy as np

from openjev.research.otto_released_policy import PublicBeliefView, _move, _packet
from openjev.research.otto_teacher_snapshot import TeacherSnapshot

VERSION = "otto-teacher-anchors-v1"
CELLS = (("lambda3", 1, "shared@9101"), ("lambda3", 2, "dense@9102"),
         ("lambda3", 3, "shared@9103"), ("lambda4", 1, "dense@9101"),
         ("lambda4", 2, "shared@9102"), ("lambda4", 3, "dense@9103"))
FIRST = {"lambda3": 950001, "lambda4": 960001}
COLLECTORS = frozenset(f"{kind}@{seed}" for kind in ("shared", "dense")
                       for seed in (9101, 9102, 9103))
IDENTITY = frozenset({"episode_id", "stage", "regime", "seed", "initial_hit", "arm"})
METADATA = IDENTITY | {"row_index", "prefix_index", "public", "posterior", "teacher_costs"}
SELECTION = METADATA - {"teacher_costs"} | {"anchor_id"}
ANCHOR = SELECTION | {"belief", "sensing_length"}
RESET = IDENTITY | {"kind", "block", "public", "posterior_after", "source_evaluation_only"}
STEP = frozenset({"kind", "episode_id", "step", "action", "costs", "allowed_actions",
                  "public", "posterior_before", "posterior_after", "choose_seconds",
                  "choose_instrumented_seconds", "choose_excluded_io_seconds",
                  "update_seconds", "environment_seconds", "native_p_end"})
PUBLIC_STEP = frozenset({"kind", "episode_id", "step", "action", "allowed_actions",
                         "public", "posterior_before", "posterior_after"})
PUBLIC_RESET = RESET - {"source_evaluation_only"}
HEADER = re.compile(r'^\{"kind":"(reset|step)","episode_id":"([^"\\]+)",')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low=0, high=None):
    _require(type(value) is int and value >= low and (high is None or value <= high),
             f"invalid {name}")
    return value


def _public(value, step):
    packet = _packet(value, step)
    return {**packet, "position": list(packet["position"]),
            "valid_actions": list(packet["valid_actions"])}


def _witness(value):
    _require(isinstance(value, Mapping) and set(value) == {"sha256", "mass"},
             "exact posterior witness required")
    digest, mass = value["sha256"], value["mass"]
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             "invalid posterior hash")
    _require(type(mass) in (int, float) and math.isfinite(mass) and mass >= 0,
             "invalid posterior mass")
    return {"sha256": digest, "mass": float(mass)}


def _identity(row):
    _require(row["stage"] == "dagger" and row["regime"] in FIRST
             and row["arm"] in COLLECTORS, "only original learner TRAIN identities allowed")
    first = FIRST[row["regime"]]
    seed = _integer(row["seed"], "episode seed", first, first + 11)
    hit = _integer(row["initial_hit"], "initial hit", 1, 3)
    _require(hit == 1 + (seed - first) % 3, "seed and initial hit disagree")
    _require(row["episode_id"] == f'dagger:{row["regime"]}:{seed}:{row["arm"]}',
             "episode identity disagrees with public provenance")


def _metadata(row):
    _require(isinstance(row, Mapping) and set(row) == METADATA, "exact TRAIN metadata required")
    _identity(row)
    index = _integer(row["row_index"], "row index")
    prefix = _integer(row["prefix_index"], "prefix index", 0, 2187)
    packet = _public(row["public"], prefix)
    _require(not packet["done"], "retained pre-action metadata must be nonterminal")
    return {**{k: row[k] for k in IDENTITY}, "row_index": index, "prefix_index": prefix,
            "public": packet, "posterior": _witness(row["posterior"])}


def select_anchors(metadata_iter):
    """Choose min(seed, prefix, row) per fixed cell, without using labels/mass.

    Metadata iteration order is irrelevant. Prefix zero is retained metadata but
    ineligible. Full pool authentication/counts belong to the caller; every
    supplied row must have the original DAGGER schema and a unique identity.
    """
    best, seen_rows, seen_prefixes = {}, set(), set()
    for original in metadata_iter:
        row = _metadata(original)
        pair = (row["episode_id"], row["prefix_index"])
        _require(row["row_index"] not in seen_rows and pair not in seen_prefixes,
                 "duplicate retained TRAIN row or episode prefix")
        seen_rows.add(row["row_index"])
        seen_prefixes.add(pair)
        cell = (row["regime"], row["initial_hit"], row["arm"])
        if row["prefix_index"] == 0 or cell not in CELLS:
            continue
        key = (row["seed"], row["prefix_index"], row["row_index"])
        if cell not in best or key < best[cell][0]:
            best[cell] = (key, row)
    _require(set(best) == set(CELLS), "a fixed anchor cell has no nonreset prefix")
    return tuple({"anchor_id": i, **best[cell][1]} for i, cell in enumerate(CELLS))


def _selections(selections):
    _require(isinstance(selections, (tuple, list)) and len(selections) == 6,
             "exactly six fixed selections required")
    result = []
    for i, row in enumerate(selections):
        _require(isinstance(row, Mapping) and set(row) == SELECTION, "exact selection fields required")
        _require(_integer(row["anchor_id"], "anchor ID") == i, "fixed anchor order required")
        clean = _metadata({k: v for k, v in row.items() if k != "anchor_id"} | {"teacher_costs": None})
        _require(clean["prefix_index"] > 0 and (clean["regime"], clean["initial_hit"], clean["arm"]) == CELLS[i],
                 "selection is not the fixed nonreset anchor cell")
        result.append({"anchor_id": i, **clean})
    _require(len({row["row_index"] for row in result}) == 6, "duplicate selected row index")
    return tuple(result)


class _NumberToken(str):
    """A JSON number not yet converted to a numerical input."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _decode_record(raw):
    return json.loads(raw, parse_int=_NumberToken, parse_float=_NumberToken,
                      parse_constant=_constant, object_pairs_hook=_pairs)


def _numbers(value):
    if isinstance(value, _NumberToken):
        return float(value) if any(c in value for c in ".eE") else int(value)
    if isinstance(value, list):
        return [_numbers(v) for v in value]
    if isinstance(value, dict):
        return {k: _numbers(v) for k, v in value.items()}
    return value


def iter_public_records(raw_lines, selections):
    """Filter exact raw episode headers before JSON/public numerical decoding.

    Original mandatory record fields and exact public sub-schemas are required.
    Additional evaluator fields are ignored, as are the original hidden source,
    scores and timings. Numbers initially remain lexical tokens; only projected
    public fields become numbers. Steps after the frozen prefix are discarded
    before converting their public fields. Input must use the authenticated
    producer's compact kind/episode header order, not arbitrary JSON formatting.
    """
    allowed = {r["episode_id"]: r for r in _selections(selections)}
    for raw in raw_lines:
        _require(isinstance(raw, (str, bytes)), "raw JSONL text required")
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        match = HEADER.match(text)
        _require(match is not None, "original compact public record header required")
        kind, episode_id = match.groups()
        if episode_id not in allowed:
            continue
        record = _decode_record(text)
        _require(isinstance(record, dict) and record.get("kind") == kind
                 and record.get("episode_id") == episode_id, "decoded header identity mismatch")
        required = RESET if kind == "reset" else STEP
        _require(required <= set(record), "missing original public record fields")
        if kind == "step":
            step = _integer(_numbers(record["step"]), "transition step", 1, 2188)
            if step > allowed[episode_id]["prefix_index"]:
                continue
        fields = PUBLIC_RESET if kind == "reset" else PUBLIC_STEP
        public = {key: _numbers(record[key]) for key in fields}
        expected_step = 0 if kind == "reset" else public["step"]
        public["public"] = _public(public["public"], expected_step)
        public["posterior_after"] = _witness(public["posterior_after"])
        if kind == "reset":
            _identity(public)
            _require(public["block"] is None, "DAGGER records have no evaluation block")
        else:
            public["action"] = _integer(public["action"], "action", 0, 3)
            _require(isinstance(public["allowed_actions"], list), "ordered allowed actions required")
            public["allowed_actions"] = [_integer(a, "allowed action", 0, 3) for a in public["allowed_actions"]]
            public["posterior_before"] = _witness(public["posterior_before"])
        yield public


class _Events:
    def __init__(self, check, emit):
        _require(check is None or callable(check), "check must be callable")
        _require(emit is None or callable(emit), "emit must be callable")
        self.check, self.emit, self.count = check, emit, 0

    def begin(self, operation, row, step):
        if self.check is not None:
            self.check()
        event = {"operation_id": self.count, "event": "attempt", "operation": operation,
                 "anchor_id": row["anchor_id"], "step": step}
        self.count += 1
        if self.emit is not None:
            self.emit(copy.deepcopy(event))
        return event

    def end(self, event, public, posterior):
        if self.emit is not None:
            self.emit(copy.deepcopy({**event, "event": "return", "public": public,
                                     "posterior": posterior}))


def _actual(belief):
    return {"sha256": hashlib.sha256(belief.tobytes(order="C")).hexdigest(),
            "mass": float(np.sum(belief))}


def _immutable(belief):
    return np.frombuffer(belief.tobytes(order="C"), dtype=np.float64).reshape(53, 53)


def _kernels(kernels):
    _require(isinstance(kernels, Mapping) and set(kernels) == set(FIRST),
             "only the two declared public kernels required")


def reconstruct_anchors(selections, public_records, kernels, *, check=None, emit=None):
    """Replay selected public prefixes and validate all six before returning.

    ``emit`` receives detached attempt/return public witnesses. A failed filter,
    witness check or snapshot leaves its attempt without a fabricated return.
    The six final TeacherSnapshot constructions are included here exactly once;
    callers must not repeat validate_anchors and miscount their work.
    """
    selected = _selections(selections)
    _kernels(kernels)
    by_id = {r["episode_id"]: r for r in selected}
    states, events = {}, _Events(check, emit)
    for record in public_records:
        _require(isinstance(record, Mapping) and record.get("episode_id") in by_id,
                 "nonallowlisted projected record")
        row = by_id[record["episode_id"]]
        episode_id, kind = row["episode_id"], record.get("kind")
        _require(kind in ("reset", "step") and set(record) == (PUBLIC_RESET if kind == "reset" else PUBLIC_STEP),
                 "exact public projection required")
        if kind == "reset":
            _require(episode_id not in states and record["block"] is None, "duplicate or invalid reset")
            _require(all(record[k] == row[k] for k in IDENTITY), "reset identity mismatch")
            packet = _public(record["public"], 0)
            _require(not packet["done"] and packet["position"] == [26, 26]
                     and packet["hit"] == row["initial_hit"], "original center reset required")
            event = events.begin("public_reset", row, 0)
            view = PublicBeliefView(kernels[row["regime"]])
            view._reset(packet["hit"])
        else:
            _require(episode_id in states, "step before reset")
            view, before, previous = states[episode_id]
            step = _integer(record["step"], "transition step", 1, row["prefix_index"])
            _require(step == before["step"] + 1 and not before["done"], "public step discontinuity")
            action = _integer(record["action"], "action", 0, 3)
            _require(record["allowed_actions"] == before["valid_actions"]
                     and action in before["valid_actions"], "saved action must be eligible")
            _require(_witness(record["posterior_before"]) == previous, "pre-action witness mismatch")
            packet = _public(record["public"], step)
            _require(packet["position"] == _move(action, before["position"])[0], "action position mismatch")
            event = events.begin("public_update", row, step)
            view._observe(_packet(packet, step))
        posterior = _actual(view.p_source)
        _require(posterior == _witness(record["posterior_after"]), "public posterior witness mismatch")
        _require(view.agent == packet["position"], "public filter position mismatch")
        states[episode_id] = (view, packet, posterior)
        events.end(event, packet, posterior)
    anchors = []
    for row in selected:
        _require(row["episode_id"] in states, "selected episode missing")
        view, packet, posterior = states[row["episode_id"]]
        _require(packet == row["public"] and posterior == row["posterior"],
                 "selected prefix does not match frozen metadata")
        anchors.append({**row, "belief": _immutable(view.p_source),
                        "sensing_length": float(row["regime"][-1])})
    return _validate(anchors, kernels, events)


def _validate(anchors, kernels, events):
    _kernels(kernels)
    _require(isinstance(anchors, (tuple, list)) and len(anchors) == 6, "six reconstructed anchors required")
    _require(all(isinstance(r, Mapping) and set(r) == ANCHOR for r in anchors), "exact anchor fields required")
    selected = _selections([{k: row[k] for k in SELECTION} for row in anchors])
    result = []
    for row, metadata in zip(anchors, selected, strict=True):
        _require(type(row["sensing_length"]) in (int, float)
                 and row["sensing_length"] == float(row["regime"][-1]), "sensing length mismatch")
        belief = row["belief"]
        _require(isinstance(belief, np.ndarray) and belief.dtype == np.float64
                 and belief.shape == (53, 53), "anchor requires float64[53,53]")
        event = events.begin("anchor_validation", metadata, metadata["prefix_index"])
        snapshot = TeacherSnapshot(metadata["public"], belief, kernels[metadata["regime"]])
        owned = snapshot.belief
        _require(_actual(owned) == metadata["posterior"], "validated anchor witness mismatch")
        result.append({**metadata, "belief": _immutable(owned), "sensing_length": float(row["sensing_length"])})
        events.end(event, metadata["public"], metadata["posterior"])
    return tuple(result)


def validate_anchors(anchors, kernels, *, check=None, emit=None):
    """Validate six already-reconstructed anchors with six snapshot constructors.

    This is an alternative standalone entrypoint, not an additional operation
    needed after reconstruct_anchors. No choices, sources or samples are drawn.
    """
    return _validate(anchors, kernels, _Events(check, emit))
