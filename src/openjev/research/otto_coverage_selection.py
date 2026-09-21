"""Outcome-independent quotas and bounded salted selection for coverage TRAIN.

Inputs must already be authenticated. Only row order, stage, regime and initial
hit determine quotas; no outcomes, targets, numerical beliefs or model values
are inspected. Duplicate numerical beliefs remain separate sampling units.
"""
from bisect import bisect_left
from collections.abc import Mapping
from hashlib import sha256

VERSION = "otto-coverage-selection-v1"
SALT = "otto-coverage-v1"
STRATA = tuple((regime, hit) for regime in ("lambda3", "lambda4") for hit in (1, 2, 3))
COLLECTORS = (10101, 10102, 10103)
ORIGINAL_ROWS, STUDENT_ROWS = 5589, 2790


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(type(value) is int and value >= 0, f"{name} must be a nonnegative Python integer")
    return value


def apportion(counts, total):
    """Integer largest remainders; descending remainder, lexicographic cell tie."""
    _require(isinstance(counts, Mapping) and set(counts) == set(STRATA), "exact six stratum keys required")
    for key, count in counts.items():
        _require(type(key[1]) is int, "stratum hit must be an integer, not boolean")
        _integer(count, "cell count")
    total = _integer(total, "allocation total")
    denominator = sum(counts.values())
    _require(denominator > 0 and total <= denominator, "positive source population and feasible allocation required")
    allocated = {key: total*counts[key]//denominator for key in STRATA}
    ranked = sorted(STRATA, key=lambda key: (-(total*counts[key] % denominator), key))
    for key in ranked[:total-sum(allocated.values())]:
        allocated[key] += 1
    return allocated


def split_collectors(quota):
    """Even allocation with remainders assigned to ascending fixed seed IDs."""
    base, remainder = divmod(_integer(quota, "student cell quota"), len(COLLECTORS))
    return {seed: base+int(index < remainder) for index, seed in enumerate(COLLECTORS)}


def coverage_quotas(original_rows):
    """Consume all 5589 canonical TRAIN rows and return six JSON-friendly records.

collector_rows has integer seed keys in Python (JSON encodes them as strings).
Additional authenticated metadata fields are allowed and never inspected.
"""
    counts = dict.fromkeys(STRATA, 0)
    seen = 0
    for index, row in enumerate(original_rows):
        _require(index < ORIGINAL_ROWS and isinstance(row, Mapping), "bounded original metadata mapping required")
        _require(type(row["row_index"]) is int and row["row_index"] == index and row["stage"] == "train",
                 "complete original TRAIN row order required")
        cell = row["regime"], row["initial_hit"]
        _require(type(cell[1]) is int and cell in counts, "known regime and initial hit required")
        counts[cell] += 1
        seen += 1
    _require(seen == ORIGINAL_ROWS, "exactly 5589 original TRAIN rows required")
    student = apportion(counts, STUDENT_ROWS)
    return [{"regime": regime, "initial_hit": hit, "original_rows": counts[regime, hit],
             "student_rows": student[regime, hit], "teacher_rows": counts[regime, hit]-student[regime, hit],
             "collector_rows": split_collectors(student[regime, hit])} for regime, hit in STRATA]


def teacher_identity(row_index):
    _require(_integer(row_index, "original row index") < ORIGINAL_ROWS, "original row index out of range")
    return "teacher:"+str(row_index)


def student_identity(episode_id, preaction_index):
    _require(isinstance(episode_id, str) and bool(episode_id), "nonempty episode identity required")
    return episode_id+":"+str(_integer(preaction_index, "pre-action index"))


def selection_key(identity):
    """Canonical digest and exact identity tiebreak; there is no configurable salt."""
    _require(isinstance(identity, str) and bool(identity), "nonempty string identity required")
    return sha256((SALT+"|"+identity).encode("utf-8")).hexdigest(), identity


class SaltedSelector:
    """Keep the smallest (digest,identity) entries using O(quota) storage.

Create one instance per teacher stratum or student (stratum,collector) cell.
offer() returns whether this item is retained on arrival; later items can evict
it. payload_factory() runs only after its key qualifies, before mutating the
reservoir. The caller must return an owned snapshot, not a live actor reference.
At most quota payloads persist; one incoming snapshot is transiently additional.

The caller must enforce globally unique episode/step identities. Only currently
retained duplicate IDs can be detected without remembering the entire stream.
Rejected or evicted duplicate IDs are outside this helper's validation scope.
finish() never fills deficits from another cell, collector, or rejected pool.
"""

    __slots__ = ("_identities", "_keys", "_payloads", "_quota")

    def __init__(self, quota):
        self._quota = _integer(quota, "quota")
        self._keys, self._payloads, self._identities = [], [], set()

    def __len__(self):
        return len(self._keys)

    def offer(self, identity, payload_factory=None):
        key = selection_key(identity)
        _require(payload_factory is None or callable(payload_factory), "optional payload factory must be callable")
        _require(identity not in self._identities, "duplicate retained identity")
        index = bisect_left(self._keys, key)
        if self._quota == 0 or (len(self) == self._quota and index == self._quota):
            return False
        payload = None if payload_factory is None else payload_factory()
        if len(self) == self._quota:
            self._identities.remove(self._keys.pop()[1])
            self._payloads.pop()
        self._keys.insert(index, key)
        self._payloads.insert(index, payload)
        self._identities.add(identity)
        return True

    def finish(self):
        _require(len(self) == self._quota, "insufficient unique candidates for fixed cell quota")
        return [{"key": key, "identity": identity, "payload": payload}
                for (key, identity), payload in zip(self._keys, self._payloads, strict=True)]
