"""Literal-evidence age proxies on full public prefixes, not semantic support.

Pure stdlib: no file IO, actor construction, learned models or numerical arrays.
Ordinary-value matching intentionally duplicates dialogue_copy_features' exact
strip/casefold and Unicode whole-word regex semantics. Boolean literals do not
receive yes/no synonyms. Labels select evaluator metadata only.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence

NONE = "reserved:NOT_MENTIONED"
DONTCARE = "reserved:DONTCARE"
BINS = ("first_assignment", "revision", "clear", "unmentioned_retention", "assigned_retention")
STRATA = ("all", "changed", "retained", "unmentioned_retention", "assigned_retention")
TYPES = ("none", "dontcare", "true", "false", "other")
AGES = ("current", "recent", "distant", "never", "nonliteral")
ADJACENCY = ("adjacent", "first_annotation", "gapped")
PROXIES = ("delayed_system_only", "changed_no_recent_combined_literal",
           "changed_distant_combined_literal", "changed_never_combined_literal", "other_distant_combined_literal")
SCOPE = ("Literal occurrence proxies only. A mention does not establish acceptance, relevance, "
         "negation scope, causal dependence or memory necessity. Ages count every public USER endpoint, "
         "including unscored turns. Reserved states have no literal evidence age. Primary proxy flags "
         "are restricted to changed OTHER values; other_distant_combined_literal also describes retained OTHER "
         "values. Boolean literal ages are descriptive only. Normalized candidate collisions remain ambiguous "
         "matches, with no chosen winner or repaired identity.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def value_type(candidate_id):
    if candidate_id in (NONE, DONTCARE):
        return "none" if candidate_id == NONE else "dontcare"
    return {"true": "true", "false": "false"}.get(candidate_id[6:].casefold(), "other")


def transition(previous, current):
    if previous == current:
        return "unmentioned_retention" if current == NONE else "assigned_retention"
    return "first_assignment" if previous == NONE else "clear" if current == NONE else "revision"


def age_category(age, *, literal):
    if not literal:
        return "nonliteral"
    if age is None:
        return "never"
    return "current" if age == 0 else "recent" if age <= 3 else "distant"


def public_user_indices(dialogue):
    require(isinstance(dialogue, Mapping) and type(dialogue.get("dialogue_id")) is str
            and dialogue["dialogue_id"], "Public dialogue identity")
    turns = dialogue["turns"]
    require(sequence(turns) and turns, "Nonempty public turn stream")
    users = []
    expected = []
    for index, turn in enumerate(turns):
        require(isinstance(turn, Mapping) and turn.get("speaker") in ("USER", "SYSTEM")
                and type(turn.get("utterance")) is str, "Public role/text schema")
        if turn["speaker"] == "USER":
            users.append(index)
            expected.append({"turn_index": index,
                "previous_system_turn_index": index-1 if index and turns[index-1]["speaker"] == "SYSTEM" else None})
    if "user_turns" in dialogue:
        observed = dialogue["user_turns"]
        require(sequence(observed) and all(isinstance(v, Mapping) and type(v.get("turn_index")) is int
                and (v.get("previous_system_turn_index") is None or type(v["previous_system_turn_index"]) is int)
                for v in observed), "Strict public USER metadata indices")
        require(list(observed) == expected, "Public USER chronology disagrees with role stream")
    return users


def query_patterns(query):
    require(isinstance(query, Mapping) and all(type(query.get(k)) is str and query[k]
            for k in ("query_id", "service", "slot")), "Supplied query identity")
    candidates = query["candidates"]
    require(sequence(candidates) and candidates, "Candidate sequence")
    ids, patterns = [], {}
    for candidate in candidates:
        require(isinstance(candidate, Mapping) and type(candidate.get("id")) is str
                and "value" in candidate, "Candidate identity/value schema")
        cid, value = candidate["id"], candidate["value"]
        require(cid not in ids, "Duplicate candidate identity")
        ids.append(cid)
        if cid in (NONE, DONTCARE):
            require(value is None, "Reserved states have no literal ontology value")
        else:
            require(type(value) is str and value.strip() and cid == "value:"+value,
                    "Exact nonempty ontology identity/value binding")
            patterns[cid] = re.compile(r"(?<!\w)"+re.escape(value.strip().casefold())+r"(?!\w)")
    require(NONE in ids and DONTCARE in ids, "Both reserved candidates required")
    return ids, patterns


def audit_query(dialogue, query, labels):
    """Return one metadata row per strictly ordered annotated USER endpoint.

    ``turn_index`` addresses the raw public turn list; returned ``user_index``
    counts USER endpoints from zero. Each SYSTEM turn belongs to the following
    USER endpoint for age arithmetic, including consecutive SYSTEM turns.
    No subsequent turn contributes to an earlier endpoint. The first annotation
    compares with initial NONE but has no prior annotated endpoint.
    """
    user_turns = public_user_indices(dialogue)
    ids, patterns = query_patterns(query)
    normalized = {c["id"]: c["value"].strip().casefold() for c in query["candidates"] if c["value"] is not None}
    collision_counts = Counter(normalized.values())
    collision_groups = sum(n > 1 for n in collision_counts.values())
    require(sequence(labels), "Ordered evaluator label sequence")
    ordinals = {turn: i for i, turn in enumerate(user_turns)}
    by_turn, previous_id, previous_turn = {}, NONE, None
    for label in labels:
        require(isinstance(label, Mapping), "Evaluator label mapping")
        for key, expected in (("dialogue_id", dialogue["dialogue_id"]), ("query_id", query["query_id"]),
                              ("service", query["service"]), ("slot", query["slot"])):
            require(key not in label or label[key] == expected, "Evaluator/public identity disagreement")
        turn, cid, index = label.get("turn_index"), label.get("label_id"), label.get("label_index")
        require(type(turn) is int and turn in ordinals and (previous_turn is None or turn > previous_turn),
                "Duplicate, unordered or non-USER annotated endpoint")
        require(type(index) is int and 0 <= index < len(ids) and type(cid) is str and ids[index] == cid,
                "Unsupported or mismatched evaluator candidate")
        derived = transition(previous_id, cid)
        require(label.get("bin") == derived, "Annotated transition bin disagreement")
        by_turn[turn] = {"label_id": cid, "label_index": index, "bin": derived,
                         "previous_label_id": previous_id, "previous_label_index": ids.index(previous_id),
                         "previous_annotated_turn_index": previous_turn}
        previous_id, previous_turn = cid, turn
    last = {"USER": {}, "SYSTEM": {}}
    results, user_index = [], 0
    final_turn = max(by_turn, default=-1)
    for turn_index, turn in enumerate(dialogue["turns"]):
        if turn_index > final_turn:
            break
        speaker, text = turn["speaker"], turn["utterance"].casefold()
        for cid, pattern in patterns.items():
            if pattern.search(text):
                last[speaker][cid] = (user_index, turn_index)
        if speaker != "USER":
            continue
        if turn_index in by_turn:
            row = by_turn[turn_index]
            cid, literal = row["label_id"], row["label_id"] in patterns
            changed = row["label_id"] != row["previous_label_id"]
            prior_turn = row["previous_annotated_turn_index"]
            prior_user = ordinals[prior_turn] if prior_turn is not None else None
            result = {"dialogue_id": dialogue["dialogue_id"], "query_id": query["query_id"],
                "service": query["service"], "slot": query["slot"], "turn_index": turn_index,
                "user_index": user_index, "candidate_count": len(ids), **row,
                "previous_annotated_user_index": prior_user,
                "annotation_adjacent": prior_user is not None and prior_user == user_index-1,
                "annotation_adjacency": "first_annotation" if prior_user is None else "adjacent" if prior_user == user_index-1 else "gapped",
                "target_type": value_type(cid), "literal_target": literal,
                "ordinary_target": value_type(cid) == "other", "changed": changed,
                "query_literal_collision_groups": collision_groups,
                "target_literal_match_count": collision_counts[normalized[cid]] if literal else 0,
                "stratum": "changed" if changed else row["bin"]}
            for role in ("USER", "SYSTEM"):
                observed = last[role].get(cid) if literal else None
                age = user_index-observed[0] if observed is not None else None
                name = role.lower()
                result.update({name+"_age": age, name+"_age_category": age_category(age, literal=literal),
                    name+"_last_mention_user_index": observed[0] if observed else None,
                    name+"_last_mention_turn_index": observed[1] if observed else None})
            observed_ages = [result[k+"_age"] for k in ("user", "system") if result[k+"_age"] is not None]
            combined = min(observed_ages, default=None)
            result["combined_age"] = combined
            result["combined_age_category"] = age_category(combined, literal=literal)
            eligible = changed and result["target_type"] == "other"
            no_recent = eligible and (combined is None or combined >= 4)
            result["proxies"] = {
                "delayed_system_only": eligible and result["user_age"] is None
                    and result["system_age"] is not None and result["system_age"] >= 4,
                "changed_no_recent_combined_literal": no_recent,
                "changed_distant_combined_literal": no_recent and combined is not None,
                "changed_never_combined_literal": eligible and combined is None,
                "other_distant_combined_literal": result["target_type"] == "other"
                    and combined is not None and combined >= 4}
            results.append(result)
        user_index += 1
    return results


def aggregate(rows):
    """Count every endpoint, including never/nonliteral and nonadjacent labels.

    No rates, model correctness or semantic-acceptance claims are inferred.
    ``changed_no_recent_combined_literal`` includes never observed targets;
    its distant and never components are separately reported.
    """
    require(sequence(rows), "Endpoint metadata sequence")
    identities = set()
    for row in rows:
        identity = (row["dialogue_id"], row["query_id"], row["turn_index"])
        require(identity not in identities, "Duplicate aggregate endpoint")
        identities.add(identity)
        require(row["bin"] in BINS and row["target_type"] in TYPES
                and all(row[k+"_age_category"] in AGES for k in ("user", "system", "combined"))
                and row["annotation_adjacency"] in ADJACENCY and type(row["changed"]) is bool
                and set(row["proxies"]) == set(PROXIES) and all(type(v) is bool for v in row["proxies"].values()),
                "Known metadata categories and boolean proxies")
    def count(selected):
        cross = Counter((r["user_age_category"], r["system_age_category"]) for r in selected)
        return {"rows": len(selected), "bins": {b: sum(r["bin"] == b for r in selected) for b in BINS},
            "target_types": {t: sum(r["target_type"] == t for r in selected) for t in TYPES},
            "user_system_age": {u: {s: cross[u, s] for s in AGES} for u in AGES},
            "combined_age": {a: sum(r["combined_age_category"] == a for r in selected) for a in AGES},
            "annotation_adjacency": {a: sum(r["annotation_adjacency"] == a for r in selected) for a in ADJACENCY},
            "proxies": {p: sum(r["proxies"][p] for r in selected) for p in PROXIES}}
    def support(selected):
        services = sorted({r["service"] for r in selected})
        return {"rows": len(selected), "unique_dialogues": len({r["dialogue_id"] for r in selected}),
                "services": services, "service_count": len(services)}
    proxy_support = {}
    for proxy in PROXIES:
        selected = [r for r in rows if r["proxies"][proxy]]
        proxy_support[proxy] = {**support(selected),
            "by_bin": {b: support([r for r in selected if r["bin"] == b]) for b in BINS},
            "by_stratum": {"changed": support([r for r in selected if r["changed"]]),
                           "retained": support([r for r in selected if not r["changed"]])}}
    strata = {"all": list(rows), "changed": [r for r in rows if r["changed"]],
              "retained": [r for r in rows if not r["changed"]],
              **{b: [r for r in rows if r["bin"] == b] for b in ("unmentioned_retention", "assigned_retention")}}
    return {"scope": SCOPE, "rows": len(rows), "proxy_support": proxy_support,
        "normalized_literal_collisions": {
            "rows_with_ambiguous_target": sum(r["target_literal_match_count"] > 1 for r in rows),
            "distinct_queries_with_collisions": len({r["query_id"] for r in rows if r["query_literal_collision_groups"]}),
            "dialogue_queries_with_collisions": len({(r["dialogue_id"], r["query_id"]) for r in rows if r["query_literal_collision_groups"]})},
        "by_bin": {b: count([r for r in rows if r["bin"] == b]) for b in BINS},
        "by_stratum": {s: count(strata[s]) for s in STRATA},
        "by_service": {s: {**count([r for r in rows if r["service"] == s]),
            "by_bin": {b: count([r for r in rows if r["service"] == s and r["bin"] == b]) for b in BINS}}
            for s in sorted({r["service"] for r in rows})},
        "by_type": {t: count([r for r in rows if r["target_type"] == t]) for t in TYPES}}
