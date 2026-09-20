"""Additive split-qualified dialogue identity correction to frozen dataset v1.

Only cohort aggregation changes: TRAIN and DEV may reuse a bare dialogue ID.
The identity key is (split, dialogue_id); duplicates of that pair still fail.
Global feature IDs must still identify identical content-token sequences.
Actor/loss construction, lexical parity, order generation and work arithmetic
are inherited unchanged. No corpus, tokenizer, feature or model loading occurs.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

from openjev.research.dialogue_finetune_dataset import (
    ACTOR_KEYS,
    BINS,
    SPLITS,
    STRATA,
    assert_lexical_parity,
    build_actor_payload,
    build_loss_rows,
    paired_epoch_orders,
)
from openjev.research.dialogue_finetune_inputs import _require, workload_profile

VERSION = "dialogue-finetune-dataset-v2"
__all__ = ["ACTOR_KEYS", "BINS", "SPLITS", "STRATA", "VERSION", "aggregate_work_profiles",
           "assert_lexical_parity", "build_actor_payload", "build_loss_rows", "paired_epoch_orders"]


def aggregate_work_profiles(payloads, *, chunk_tokens=254, chunk_batch_size=32):
    """Original work sums/maxima with explicit split-qualified identity.

    Unique-text totals still sum per-dialogue deduplication, not global text
    deduplication. No supplied identities, token IDs, or input order are changed.
    """
    _require(isinstance(payloads, Iterable) and not isinstance(payloads, (str, bytes, Mapping)),
             "Cohort payload iterable")
    profiles, seen, feature_tokens = [], set(), {}
    for payload in payloads:
        split, did = payload["split"], payload["dialogue_id"]
        _require(type(split) is str and split in SPLITS and type(did) is str and did,
                 "Cohort split/dialogue identity")
        _require((split, did) not in seen, "Duplicate cohort dialogue")
        seen.add((split, did))
        work = workload_profile(payload, chunk_tokens=chunk_tokens, chunk_batch_size=chunk_batch_size)
        for feature, tokens in zip(payload["original_feature_ids"], payload["tokens"], strict=True):
            digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
            _require(feature not in feature_tokens or feature_tokens[feature] == digest,
                     "Cohort feature ID has inconsistent tokenization")
            feature_tokens[feature] = digest
        work["max_content_tokens_per_text"] = max(map(len, payload["tokens"]))
        profiles.append({"split": split, "dialogue_id": did, "work": work})
    _require(profiles, "Nonempty cohort payloads")
    settings = {"chunk_tokens", "chunk_batch_size"}
    summed = [key for key in profiles[0]["work"] if key not in settings and not key.startswith("max_")]
    maximum = [key for key in profiles[0]["work"] if key not in settings]

    def reduce_group(group):
        return {"dialogues": len(group),
                "totals": {key: sum(p["work"][key] for p in group) for key in summed},
                "maxima": {key: max(p["work"][key] for p in group) for key in maximum}}

    return {"version": VERSION,
            "scope": ("One separate encoding per complete dialogue; no measured costs; "
                      "dialogue identity is (split, dialogue_id)"),
            "chunk_tokens": chunk_tokens, "chunk_batch_size": chunk_batch_size,
            "profiles": profiles, "all": reduce_group(profiles),
            "splits": {split: reduce_group([p for p in profiles if p["split"] == split])
                       for split in SPLITS if any(p["split"] == split for p in profiles)}}
