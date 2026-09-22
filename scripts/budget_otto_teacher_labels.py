"""Fabricated serialization envelope for the fixed six-anchor label pilot.

No numerical imports, empirical inputs, samplers, simulators or models. The
collector must enforce the declared per-event and total output bounds itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLER = "src/openjev/research/otto_teacher_rollouts.py"
SAMPLER_SHA = "0fb4f0e4e45e0bed0c40c90c3bff5b19f58b888a838b1f155ab4ff6f0a777612"
EVENT_CAP = 512
OTHER_BYTES = 256 * 1024**2
OUTPUT_CAP = 4 * 1024**3
FLOAT_CHAR_CAP = 32
HORIZON, CONTINUATIONS, MOVES = 2188, 384, 840192


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def float_count(value):
    if isinstance(value, dict):
        return sum(float_count(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(float_count(x) for x in value)
    return int(type(value) is float)


def projection():
    # Inflated integer widths dominate every permitted panel identity/step.
    base = {"version": "otto-teacher-rollouts-v1", "seed": 19000001, "anchor_id": 5}
    replica = {"replicate_id": 15}
    action = {**replica, "first_action": 3}
    step = {**action, "local_step": HORIZON, "from_step": 4375}
    public = {"position": [52, 52], "hit": -2, "done": False, "step": 4376,
              "valid_actions": [0, 1, 2, 3]}
    # This is an oversized serialization fixture, not a legal public state.
    # Float placeholders are charged at 32 characters, including sign/exponent.
    contexts = {
        "anchor_snapshot": {}, "source_generator": replica, "source_draw": replica,
        "teacher_snapshot": action, "hit_generator": action, "teacher_choose": step,
        "movement": step, "hit_draw": step, "teacher_update": step,
    }
    descriptions = {
        "anchor_snapshot": {"public": public}, "source_generator": {},
        "source_draw": {"uniform": 0., "cdf_mass": 0., "selected_index": 2808, "source": [52, 52]},
        "teacher_snapshot": {"public": public}, "hit_generator": {},
        "teacher_choose": {"action": 3, "scores": [0.] * 4},
        "movement": {"position": [52, 52], "found": False},
        "hit_draw": {"uniform": 0., "cdf_mass": 0., "selected_index": 3,
                     "probabilities": [0.] * 4, "draw_index": 2187},
        "teacher_update": {"public": public},
    }
    operation_counts = {
        "anchor_snapshot": 6, "source_generator": 96, "source_draw": 96,
        "teacher_snapshot": CONTINUATIONS, "hit_generator": CONTINUATIONS,
        "teacher_choose": MOVES - CONTINUATIONS, "movement": MOVES,
        "hit_draw": MOVES, "teacher_update": MOVES,
    }
    rows = []

    def add(name, value, count):
        raw = encode(value)
        floats = float_count(value)
        upper = len(raw) + floats * (FLOAT_CHAR_CAP - len("0.0"))
        assert upper <= EVENT_CAP, (name, upper)
        rows.append({"event_kind": name, "count": count, "example": value,
                     "serialized_example_bytes": len(raw), "float_fields": floats,
                     "upper_bytes_each": upper, "upper_bytes_total": upper * count})

    for operation, count in operation_counts.items():
        identity = {**base, **contexts[operation], "operation_id": 999999, "operation": operation}
        add(f"{operation}.attempt", {**identity, "event": "attempt"}, count)
        add(f"{operation}.return", {**identity, "event": "return", **descriptions[operation]}, count)
    add("forced_action", {**base, **step, "event": "forced_action", "action": 3}, CONTINUATIONS)
    add("record", {**base, **action, "event": "record", "steps": HORIZON, "found": False}, CONTINUATIONS)
    add("panel_complete", {**base, "event": "panel_complete", "record_count": 64,
                           "operation_count": 999999}, 6)
    event_count = sum(row["count"] for row in rows)
    assert event_count == 402 + 4 * CONTINUATIONS + 8 * MOVES == 6723474
    loose = event_count * EVENT_CAP + OTHER_BYTES
    assert loose < OUTPUT_CAP
    return {
        "version": "otto-teacher-label-serialization-v1", "scope": "Fabricated byte projection only",
        "serializer": "json.dumps(sort_keys=True,allow_nan=False,separators=(',',':')) + newline; UTF-8",
        "source": {"path": SAMPLER, "sha256": SAMPLER_SHA},
        "assumptions": {"anchors": 6, "replicates_each": 16, "horizon": HORIZON,
                        "maximum_continuations": CONTINUATIONS, "maximum_moves": MOVES,
                        "maximum_absolute_public_step": 4376, "maximum_operation_id": 560225,
                        "float_character_allowance": FLOAT_CHAR_CAP,
                        "failure_records_never_convert_to_censored_labels": True},
        "event_count_maximum": event_count, "operation_pairs_maximum": sum(operation_counts.values()),
        "per_kind_weighted_upper_bytes": sum(row["upper_bytes_total"] for row in rows),
        "largest_padded_event_bytes": max(row["upper_bytes_each"] for row in rows),
        "enforced_sampler_event_cap_bytes": EVENT_CAP, "non_sampler_output_reserve_bytes": OTHER_BYTES,
        "conservative_total_upper_bytes": loose, "output_cap_bytes": OUTPUT_CAP,
        "output_margin_bytes": OUTPUT_CAP - loose, "fits": True, "event_classes": rows,
        "enforcement": "Reject an oversized event without truncation and fail the panel; preserve primary error. "
                       "Charge actual sampler and other output bytes; enforce the separate reserve and total cap. "
                       "Store draw evidence once. No wall-time guarantee follows from this byte projection.",
        "empirical_rows_read": 0, "sampler_calls": 0, "native_calls": 0, "model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256((ROOT / SAMPLER).read_bytes()).hexdigest() == SAMPLER_SHA
    result = projection()
    result["program_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("event_classes", "enforcement", "assumptions")}))


if __name__ == "__main__":
    main()
