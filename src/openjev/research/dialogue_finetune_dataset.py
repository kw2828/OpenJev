"""Explicit TRAIN/DEV inputs for full-stream observation learning.

Pure preparation functions: no files, model imports, feature loading or corpus
selection. Callers authenticate supplied metadata. Actor construction reads
only public text and the fixed query inventory; annotations have a separate API.
The frozen pilot's public validators and work arithmetic are reused unchanged.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping

import numpy as np

from openjev.research.dialogue_copy_features import DONTCARE, NONE, lexical_stream
from openjev.research.dialogue_finetune_inputs import (
    _catalog,
    _integer,
    _public_pairs,
    _require,
    _sequence,
    workload_profile,
)

VERSION = "dialogue-finetune-dataset-v1"
SPLITS = ("train", "dev")
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
BINS = (*STRATA[:2], "first_assignment", "revision", "clear")
ACTOR_KEYS = ("tokens", "turn_text_ids", "query_text_ids", "candidate_text_ids",
              "candidate_ids", "lexical")


def build_actor_payload(packet_dialogue, public_dialogue, catalog_queries,
                        packet_queries, lexical_layout, tokenize: Callable, *, split):
    """Return the pilot actor contract plus the explicit original split.

    The exact original strings, deduplication order, feature IDs and lexical
    observations match the immutable pilot. Unlike its TRAIN-only builder this
    validates DEV directly, without changing any supplied metadata. Every USER
    turn is present even when no endpoint is scored there. Tokenization accepts
    exact strings and returns nonempty untruncated content-token lists.
    """
    _require(type(split) is str and split in SPLITS, "Explicit train/dev split")
    _require(callable(tokenize), "Content tokenizer callable")
    systems, users, user_indices = _public_pairs(public_dialogue)
    did = public_dialogue["dialogue_id"]
    _require(packet_dialogue["id"] == did == lexical_layout["id"], "Dialogue identity join")
    turn_features = packet_dialogue["turns"]
    _require(_sequence(turn_features) and len(turn_features) == len(users)
             and all(_integer(x) for x in turn_features), "Complete original turn-feature map")
    _require(_sequence(packet_queries) and _sequence(packet_dialogue["queries"]), "Packet query inventory")
    query_ids = []
    for row in packet_dialogue["queries"]:
        qi = row["query"]
        _require(_integer(qi) and qi < len(packet_queries), "Supplied query index")
        query_ids.append(qi)
    query_ids = sorted(set(query_ids))
    _require(query_ids, "At least one supplied query")
    layout_ids = lexical_layout["query_ids"]
    _require(_sequence(layout_ids) and all(_integer(x) for x in layout_ids)
             and list(layout_ids) == query_ids, "Sorted independent supplied query layout")
    # _catalog validates public descriptions/identities, not split membership.
    catalog = _catalog(catalog_queries)
    selected = []
    for qi in query_ids:
        entry = packet_queries[qi]
        _require(entry["split"] == split and entry["id"] in catalog, "Supplied schema split")
        q = catalog[entry["id"]]
        _require((entry["service"], entry["slot"]) == (q["service"], q["slot"]), "Schema service/slot join")
        _require(entry["candidate_ids"] == [c["id"] for c in q["candidates"]]
                 and entry["candidate_values"] == [c["value"] for c in q["candidates"]],
                 "Exact candidate identity/order")
        _require(_integer(entry["text"]) and _sequence(entry["candidates"])
                 and len(entry["candidates"]) == len(q["candidates"])
                 and all(_integer(x) for x in entry["candidates"]), "Schema feature-index map")
        selected.append((entry, q))
    cmax = max(len(q["candidates"]) for _, q in selected)
    shape = lexical_layout["shape"]
    _require(_sequence(shape) and all(_integer(x) for x in shape)
             and list(shape) == [len(users), len(selected), cmax, 10], "Full public lexical shape")
    _require(_integer(lexical_layout["offset"]), "Original lexical offset")

    tokens, original_features, text_ids, feature_text = [], [], {}, {}

    def add(text, original_feature):
        _require(original_feature not in feature_text or feature_text[original_feature] == text,
                 "Original feature ID maps to inconsistent public text")
        feature_text[original_feature] = text
        if text in text_ids:
            index = text_ids[text]
            _require(original_features[index] == original_feature,
                     "Duplicate text has inconsistent original feature ID")
            return index
        ids = tokenize(text)
        _require(type(ids) is list and ids and all(_integer(x) for x in ids), "Nonempty content token IDs")
        index = len(tokens)
        text_ids[text] = index
        tokens.append(list(ids))
        original_features.append(original_feature)
        return index

    turn_ids = [add("System: " + system + "\nUser: " + user, feature)
                for system, user, feature in zip(systems, users, turn_features, strict=True)]
    query_text_ids, candidate_text_ids, candidate_ids = [], [], []
    lexical = np.zeros((len(users), len(selected), cmax, 10), dtype=np.float32)
    for position, (entry, q) in enumerate(selected):
        query_text_ids.append(add(q["query_text"], entry["text"]))
        candidate_text_ids.append([add(c["text"], feature)
                                   for c, feature in zip(q["candidates"], entry["candidates"], strict=True)])
        ids = list(entry["candidate_ids"])
        candidate_ids.append(ids)
        values, _ = lexical_stream(systems, users, ids, entry["candidate_values"])
        lexical[:, position, :len(ids)] = values
    return {"split": split, "dialogue_id": did, "query_ids": query_ids,
            "user_turn_indices": user_indices, "tokens": tokens,
            "original_feature_ids": original_features, "turn_text_ids": turn_ids,
            "query_text_ids": query_text_ids, "candidate_text_ids": candidate_text_ids,
            "candidate_ids": candidate_ids, "lexical": lexical}


def assert_lexical_parity(payload, original_slice):
    """Require exact original observations, including zero candidate padding.

    The caller supplies one authenticated [T,Q,C,10] float32 slice. No original
    cache is opened here. Apply this before constructing any new lexical arm.
    """
    expected = payload["lexical"]
    _require(isinstance(original_slice, np.ndarray) and original_slice.dtype == np.float32
             and original_slice.shape == expected.shape and np.isfinite(original_slice).all(),
             "Original lexical slice dtype/shape/finite")
    _require(np.array_equal(expected, original_slice), "Original lexical observations differ")
    return {"matched": True, "positions": int(expected.size), "bytes": int(expected.nbytes)}


def _transition(previous, current):
    if current == previous:
        return "unmentioned_retention" if current == NONE else "assigned_retention"
    if previous == NONE:
        return "first_assignment"
    return "clear" if current == NONE else "revision"


def build_loss_rows(packet_dialogue, payload, packet_queries):
    """Validate annotations separately; preserve original scored-row order.

    Times address USER positions in the full stream, never compressed scored
    positions. Bins are checked against the previous annotated state per query,
    including gaps, starting at NONE. Unseen flags retain parser semantics; full
    training-service schema membership is not rederived here.
    """
    split, did = payload["split"], payload["dialogue_id"]
    _require(split in SPLITS and packet_dialogue["id"] == did, "Evaluator identity join")
    positions = {qi: j for j, qi in enumerate(payload["query_ids"])}
    _require(len(positions) == len(payload["query_ids"]), "Distinct actor query inventory")
    rows, seen, panels = [], set(), {}
    for index, row in enumerate(packet_dialogue["queries"]):
        qi, ti, label = row["query"], row["time"], row["label"]
        _require(_integer(qi) and qi in positions and qi < len(packet_queries), "Evaluator query index")
        _require(_integer(ti) and ti < len(payload["turn_text_ids"]), "Evaluator USER time")
        _require((ti, qi) not in seen, "Duplicate scored endpoint")
        seen.add((ti, qi))
        j, entry = positions[qi], packet_queries[qi]
        ids = payload["candidate_ids"][j]
        _require(entry["split"] == split and entry["candidate_ids"] == ids, "Evaluator schema identity")
        _require(_integer(label) and label < len(ids), "Evaluator label support")
        cid, bin_name = ids[label], row["bin"]
        _require(type(bin_name) is str and bin_name in BINS, "Evaluator transition bin")
        unseen, dontcare = row["unseen"], row["dontcare"]
        _require(type(unseen) is bool and (split != "train" or not unseen), "Evaluator unseen flag")
        _require(type(dontcare) is bool and dontcare == (cid == DONTCARE), "Evaluator DONTCARE flag")
        service = entry["service"]
        _require(service not in panels or panels[service] == unseen, "Inconsistent service unseen flag")
        panels[service] = unseen
        stratum = bin_name if bin_name in STRATA[:2] else "changed"
        rows.append({"split": split, "dialogue_id": did, "source_row_index": index,
                     "time": ti, "turn_index": payload["user_turn_indices"][ti],
                     "query_index": qi, "query_position": j, "query_id": entry["id"],
                     "service": service, "slot": entry["slot"], "label_index": label,
                     "label_id": cid, "bin": bin_name, "stratum": stratum,
                     "stratum_index": STRATA.index(stratum), "unseen": unseen, "dontcare": dontcare})
    _require(rows and {r["query_index"] for r in rows} == set(positions), "Complete evaluator query inventory")
    previous = {}
    for row in sorted(rows, key=lambda r: (r["time"], r["query_index"])):
        qi, current = row["query_index"], row["label_id"]
        _require(row["bin"] == _transition(previous.get(qi, NONE), current), "Annotated transition bin differs")
        previous[qi] = current
    return rows


def aggregate_work_profiles(payloads, *, chunk_tokens=254, chunk_batch_size=32):
    """Per-dialogue geometry, sums for one cohort visit, and explicit maxima.

    Unique-text totals sum per-dialogue deduplication, not cohort deduplication.
    Costs assume each complete dialogue is encoded separately with the declared
    chunk batch size. They are neither timing nor a multi-dialogue batch model.
    No labels or evaluator rows are accessed.
    """
    _require(isinstance(payloads, Iterable) and not isinstance(payloads, (str, bytes, Mapping)),
             "Cohort payload iterable")
    profiles, seen, feature_tokens, panels = [], set(), {}, {}
    for payload in payloads:
        split, did = payload["split"], payload["dialogue_id"]
        _require(type(split) is str and split in SPLITS and type(did) is str and did,
                 "Cohort split/dialogue identity")
        _require((split, did) not in seen, "Duplicate cohort dialogue")
        _require(did not in panels or panels[did] == split, "Dialogue shared across splits")
        seen.add((split, did))
        panels[did] = split
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

    return {"version": VERSION, "scope": "One separate encoding per complete dialogue; no measured costs",
            "chunk_tokens": chunk_tokens, "chunk_batch_size": chunk_batch_size,
            "profiles": profiles, "all": reduce_group(profiles),
            "splits": {split: reduce_group([p for p in profiles if p["split"] == split])
                       for split in SPLITS if any(p["split"] == split for p in profiles)}}


def paired_epoch_orders(dialogue_ids, seeds, epochs):
    """Fresh PCG64 permutations shared by every arm, independent of labels.

    Caller membership/order is retained exactly. Seeds reset only between fits,
    not epochs. Returned JSON-compatible integer positions address dialogue_ids.
    No global RNG is read or changed and no model initialization occurs here.
    """
    _require(_sequence(dialogue_ids) and dialogue_ids
             and all(type(did) is str and did for did in dialogue_ids)
             and len(set(dialogue_ids)) == len(dialogue_ids), "Distinct ordered dialogue IDs")
    _require(_sequence(seeds) and seeds and all(_integer(seed) for seed in seeds)
             and len(set(seeds)) == len(seeds), "Distinct nonnegative seeds")
    _require(type(epochs) is int and epochs > 0, "Positive epoch count")
    orders = {}
    for seed in seeds:
        rng = np.random.Generator(np.random.PCG64(seed))
        orders[str(seed)] = [rng.permutation(len(dialogue_ids)).tolist() for _ in range(epochs)]
    return {"version": VERSION, "algorithm": "numpy.PCG64 consecutive epoch permutations",
            "dialogue_ids": list(dialogue_ids), "seeds": list(seeds), "epochs": epochs, "orders": orders}
