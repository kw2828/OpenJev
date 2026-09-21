"""Pure preparation of a dialogue-disjoint TRAIN calibration complement.

The caller authenticates all supplied public records, endpoint metadata, fitted
membership and evaluated DEV membership. No files, model imports or inference.
Selection reads endpoint identity/query fields only, never targets or scores.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from openjev.research.dialogue_finetune_dataset import build_actor_payload, build_loss_rows
from openjev.research.dialogue_finetune_inputs import _catalog, _public_pairs
from openjev.research.dialogue_number_lexical import lexical_observations
from openjev.research.dialogue_state_data import text_identity

VERSION = "dialogue-calibration-data-v1"
SAMPLE_SIZE = 512
SALT = "openjev-calibration-v1:"
SOURCE_SPLIT = "train"
ANALYSIS_ROLE = "calibration"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rank(dialogue_id):
    require(type(dialogue_id) is str and dialogue_id, "Nonempty dialogue ID")
    return hashlib.sha256((SALT + dialogue_id).encode()).hexdigest(), dialogue_id


def _sequence(records, name):
    require(isinstance(records, Sequence) and not isinstance(records, (str, bytes)), name)


def _public_index(records, name):
    _sequence(records, "Public record sequence")
    result = {}
    for record in records:
        require(isinstance(record, Mapping), "Public dialogue mapping")
        require(set(record) == {"dialogue_id", "turns", "user_turns"}, "Public-only dialogue fields")
        require(all(set(turn) == {"speaker", "utterance"} for turn in record["turns"]),
                "Public-only turn fields")
        _public_pairs(record)
        did = record["dialogue_id"]
        require(did not in result, "Duplicate dialogue ID within " + name)
        result[did] = record
    return result


def _validated_catalog(queries):
    catalog = _catalog(queries)
    require(all(3 <= len(q["candidates"]) <= 12 for q in catalog.values()),
            "Original three-to-twelve candidate support")
    return catalog


def _endpoint_inventory(records, public, catalog):
    """Only these three identity fields are read, including on labeled records."""
    _sequence(records, "Endpoint metadata sequence")
    inventory, seen = defaultdict(set), set()
    user_turns = {did: {t["turn_index"] for t in p["user_turns"]} for did, p in public.items()}
    for row in records:
        require(isinstance(row, Mapping), "Endpoint metadata mapping")
        did, query_id, turn = row["dialogue_id"], row["query_id"], row["turn_index"]
        require(did in public and query_id in catalog, "Endpoint public dialogue and categorical schema")
        require(type(turn) is int and turn in user_turns[did], "Endpoint addresses a public USER turn")
        identity = (did, turn, query_id)
        require(identity not in seen, "Duplicate categorical endpoint")
        seen.add(identity)
        inventory[did].add(query_id)
    return dict(inventory)


def select_calibration_dialogues(train_public, endpoint_metadata, train_catalog,
                                 fitted_ids, evaluated_dev_public):
    """Select exactly 512 eligible, unique text groups by a fixed salted ID hash.

    One lowest-ranked eligible ID represents each surviving text group. Any
    group containing a fitted TRAIN ID or evaluated DEV text is excluded first.
    Empty categorical inventories are ineligible; no target value is inspected.
    """
    public = _public_index(train_public, "TRAIN")
    dev = _public_index(evaluated_dev_public, "evaluated DEV")
    catalog = _validated_catalog(train_catalog)
    require(not isinstance(fitted_ids, (str, bytes)), "Fitted ID collection")
    fitted = set(fitted_ids)
    require(fitted and all(type(did) is str and did for did in fitted) and fitted <= set(public),
            "Complete fitted IDs belong to supplied TRAIN")
    require(dev, "Evaluated DEV membership is required")
    inventory = _endpoint_inventory(endpoint_metadata, public, catalog)
    identities = {did: text_identity(p) for did, p in public.items()}
    fitted_groups = {identities[did] for did in fitted}
    dev_groups = {text_identity(p) for p in dev.values()}
    eligible = set(inventory)
    fitted_collisions = {did for did in eligible if identities[did] in fitted_groups}
    dev_collisions = {did for did in eligible if identities[did] in dev_groups}
    excluded = fitted_collisions | dev_collisions
    survivors = eligible - excluded
    grouped = defaultdict(list)
    for did in survivors:
        grouped[identities[did]].append(did)
    representatives = sorted((min(group, key=rank) for group in grouped.values()), key=rank)
    require(len(representatives) >= SAMPLE_SIZE, "Insufficient eligible unique calibration dialogue groups")
    selected = representatives[:SAMPLE_SIZE]
    selected_rows = [{"dialogue_id": did, "text_sha256": identities[did], "salted_id_sha256": rank(did)[0]}
                     for did in selected]
    require(len({row["text_sha256"] for row in selected_rows}) == SAMPLE_SIZE, "Unique selected text groups")
    return {
        "version": VERSION, "source_split": SOURCE_SPLIT, "analysis_role": ANALYSIS_ROLE,
        "sample_size": SAMPLE_SIZE, "salt": SALT,
        "selection_rule": "Exclude fitted/evaluated-DEV text groups; lowest salted eligible ID per group; global lowest 512",
        "text_identity_rule": "Ordered roles with NFKC/casefold/whitespace-normalized complete public text",
        "selection_fields": ["dialogue_id", "query_id", "turn_index"],
        "selected_ids": selected, "selected_groups": selected_rows,
        "counts": {
            "train_public_dialogues": len(public), "fitted_ids": len(fitted),
            "evaluated_dev_dialogues": len(dev), "eligible_categorical_dialogues": len(eligible),
            "ineligible_without_categorical_endpoints": len(public) - len(eligible),
            "excluded_fitted_group_eligible_dialogues": len(fitted_collisions),
            "excluded_dev_group_eligible_dialogues": len(dev_collisions),
            "excluded_either_group_eligible_dialogues": len(excluded),
            "excluded_both_groups_eligible_dialogues": len(fitted_collisions & dev_collisions),
            "excluded_fitted_eligible_groups": len({identities[did] for did in fitted_collisions}),
            "excluded_dev_eligible_groups": len({identities[did] for did in dev_collisions}),
            "surviving_eligible_dialogues": len(survivors), "surviving_unique_groups": len(grouped),
            "duplicate_members_not_represented": len(survivors) - len(grouped),
            "selected_dialogues": SAMPLE_SIZE, "selected_unique_groups": SAMPLE_SIZE,
            "unselected_unique_groups": len(grouped) - SAMPLE_SIZE,
        },
    }


def _selection_ids(selection, public):
    require(selection["version"] == VERSION and selection["source_split"] == SOURCE_SPLIT
            and selection["analysis_role"] == ANALYSIS_ROLE and selection["sample_size"] == SAMPLE_SIZE
            and selection["salt"] == SALT, "Frozen calibration selection contract")
    ids = selection["selected_ids"]
    require(type(ids) is list and len(ids) == SAMPLE_SIZE and len(set(ids)) == SAMPLE_SIZE
            and set(ids) <= set(public) and ids == sorted(ids, key=rank), "Exact ranked calibration membership")
    expected = [{"dialogue_id": did, "text_sha256": text_identity(public[did]), "salted_id_sha256": rank(did)[0]}
                for did in ids]
    require(selection["selected_groups"] == expected and len({r["text_sha256"] for r in expected}) == SAMPLE_SIZE,
            "Selected public-text identities and group uniqueness")
    return ids


def build_calibration_inputs(selection, train_public, label_rows, train_catalog, tokenize):
    """Create fresh public actor layouts and a separate canonical evaluator side.

    The supplied selection must already be authenticated by the caller. Labels
    define the same supplied query inventory and scored endpoints as the original
    packet builder; values never enter actor observations or public-turn masks.
    Global query/text indices are new and carry no trained embedding identity.
    """
    require(callable(tokenize), "Supplied content tokenizer")
    public = _public_index(train_public, "TRAIN")
    catalog = _validated_catalog(train_catalog)
    selected = _selection_ids(selection, public)
    selected_set = set(selected)
    inventory = _endpoint_inventory(label_rows, public, catalog)
    require(selected_set <= set(inventory), "Every selected dialogue has categorical endpoints")
    by_dialogue = defaultdict(list)
    for row in label_rows:
        if row["dialogue_id"] in selected_set:
            by_dialogue[row["dialogue_id"]].append(row)

    texts, text_lookup, queries, query_lookup = [], {}, [], {}

    def text_index(text):
        if text not in text_lookup:
            text_lookup[text] = len(texts)
            texts.append(text)
        return text_lookup[text]

    for query_id, q in catalog.items():
        query_lookup[query_id] = len(queries)
        queries.append({"id": query_id, "split": SOURCE_SPLIT, "service": q["service"], "slot": q["slot"],
                        "text": text_index(q["query_text"]),
                        "candidates": [text_index(c["text"]) for c in q["candidates"]],
                        "candidate_ids": [c["id"] for c in q["candidates"]],
                        "candidate_values": [c["value"] for c in q["candidates"]]})

    actors, targets, layouts, packet_cohort, original_arrays, number_arrays = [], [], [], [], [], []
    offset = 0
    for did in selected:
        systems, users, user_indices = _public_pairs(public[did])
        time_by_turn = {turn: time for time, turn in enumerate(user_indices)}
        packet = {"id": did, "turns": [text_index("System: " + system + "\nUser: " + user)
                                        for system, user in zip(systems, users, strict=True)], "queries": []}
        for row in by_dialogue[did]:
            qi = query_lookup[row["query_id"]]
            entry = queries[qi]
            label = row["label_index"]
            require(type(label) is int and 0 <= label < len(entry["candidate_ids"])
                    and entry["candidate_ids"][label] == row["label_id"], "Original label index/ID binding")
            require((row["service"], row["slot"]) == (entry["service"], entry["slot"]), "Endpoint schema binding")
            require(row["unseen_service"] is False, "TRAIN calibration preserves original seen-service semantics")
            packet["queries"].append({"query": qi, "time": time_by_turn[row["turn_index"]], "label": label,
                                      "bin": row["bin"], "unseen": False, "dontcare": row["is_dontcare"]})
        qids = sorted({row["query"] for row in packet["queries"]})
        cmax = max(len(queries[qi]["candidate_ids"]) for qi in qids)
        layout = {"id": did, "query_ids": qids, "offset": offset, "shape": [len(users), len(qids), cmax, 10]}
        payload = build_actor_payload(packet, public[did], train_catalog, queries, layout, tokenize, split=SOURCE_SPLIT)
        original = payload["lexical"]
        numbers = np.zeros_like(original)
        registers = {"original": [], "numbers": []}
        for j, qi in enumerate(qids):
            ids = payload["candidate_ids"][j]
            values, literal = lexical_observations(public[did], catalog[queries[qi]["id"]], ids)
            require(values.shape == (len(users), len(ids), 10), "Number lexical geometry")
            numbers[:, j, :len(ids)] = values
            original_register = original[:, j, :len(ids), 4]
            require(np.all(original_register.sum(-1) == 1), "One original literal state")
            registers["original"].append(original_register.argmax(-1).tolist())
            registers["numbers"].append(literal.tolist())
        require(np.array_equal(numbers[..., 6:], original[..., 6:]), "Reserved and Boolean observations unchanged")
        rows = build_loss_rows(packet, payload, queries)
        actor = {k: value for k, value in payload.items() if k != "lexical"}
        actor.update({"source_split": SOURCE_SPLIT, "analysis_role": ANALYSIS_ROLE,
                      "lexical_offset": offset, "lexical_shape": layout["shape"]})
        actors.append(actor)
        targets.append({"split": SOURCE_SPLIT, "source_split": SOURCE_SPLIT, "analysis_role": ANALYSIS_ROLE,
                        "dialogue_id": did, "rows": rows, "literal_registers": registers})
        layouts.append(layout)
        packet_cohort.append({"id": did, "turns": packet["turns"], "query_ids": qids})
        original_arrays.append(original.reshape(-1))
        number_arrays.append(numbers.reshape(-1))
        offset += original.size
    return {
        "version": VERSION, "source_split": SOURCE_SPLIT, "analysis_role": ANALYSIS_ROLE,
        "selection": selection, "texts": texts,
        "packet": {"queries": queries, "cohort": packet_cohort, "scope": "Public indices only; targets separate"},
        "actors": actors, "targets": targets, "layouts": layouts,
        "lexical": {"original": np.concatenate(original_arrays), "numbers": np.concatenate(number_arrays)},
        "counts": {"dialogues": len(actors), "scored_endpoints": sum(len(record["rows"]) for record in targets),
                   "public_user_turns": sum(len(actor["user_turn_indices"]) for actor in actors),
                   "lexical_positions": int(offset), "unique_texts": len(texts)},
    }
