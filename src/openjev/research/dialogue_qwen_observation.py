"""Public-text packets for a frozen one-step semantic observation comparison.

Previous truth is deliberately supplied in both arms. Current truth and score
strata are never accepted by the prompt builder. Prefix-derived lexical flags
are inherited public observations, not calibrated evidence or gold state.
"""
from __future__ import annotations

import json

from openjev.decisions import Candidate, DecisionRequest, Question

ARMS = ("current", "history4")
FEATURES = ("user_match", "system_match", "unique_longest_user", "unique_longest_system",
            "literal_current", "literal_previous", "is_none", "is_dontcare",
            "affirmative_cue_for_true", "negative_cue_for_false")
TASK = (
    "What is the user's currently committed value for this slot after the final USER turn? "
    "Use the supplied previous value as the state immediately before that turn. "
    "Update it only when the dialogue supports a change; a SYSTEM proposal alone is not "
    "a user commitment. NOT_MENTIONED means no constraint has been stated; DONTCARE "
    "means the user explicitly has no preference. Literal ontology values, including "
    "a value named None, remain distinct from these reserved states. "
    "Lexical flags are noisy public string-match observations, not answers. "
    "The literal register carries the last unique longest USER mention across the "
    "public prefix; it does not understand negation or relevance. "
)


def candidate_type(cid):
    if cid == "reserved:NOT_MENTIONED":
        return "NOT_MENTIONED"
    if cid == "reserved:DONTCARE":
        return "DONTCARE"
    if cid.startswith("value:") and cid[6:].casefold() in ("true", "false"):
        return cid[6:].upper()
    return "OTHER"


def public_context(dialogue, time_index, arm):
    if arm not in ARMS:
        raise ValueError("Unknown text arm")
    # Derive chronology directly from public roles, never from scored frames.
    turns = dialogue["turns"]
    user_indices = [i for i, t in enumerate(turns) if t["speaker"] == "USER"]
    if type(time_index) is not int or not 0 <= time_index < len(user_indices):
        raise ValueError("Invalid public USER ordinal")
    first = time_index if arm == "current" else max(0, time_index - 3)
    pairs = []
    for ordinal in range(first, time_index + 1):
        i = user_indices[ordinal]
        system = turns[i - 1]["utterance"] if i and turns[i - 1]["speaker"] == "SYSTEM" else ""
        user = turns[i]["utterance"]
        if not isinstance(system, str) or not isinstance(user, str):
            raise TypeError("Public text must be strings")
        pairs.append({"SYSTEM": system, "USER": user})
    return json.dumps({"exchanges_oldest_first": pairs}, ensure_ascii=False, sort_keys=True)


def make_question(*, row_index, query, previous_candidate_id, flags):
    candidates = query["candidates"]
    ids = [c["id"] for c in candidates]
    if len(ids) != len(set(ids)) or previous_candidate_id not in ids:
        raise ValueError("Invalid candidate identity or previous value")
    if len(flags) != len(candidates) or any(len(f) != len(FEATURES) for f in flags):
        raise ValueError("Lexical dimensions changed")
    if any(type(x) not in (int, float) or x not in (0, 1) for f in flags for x in f):
        raise ValueError("Expected finite binary inherited flags")
    mapping = {f"c{i:02d}": cid for i, cid in enumerate(ids)}
    descriptions = [c["text"] + "\nCandidate type: " + candidate_type(c["id"])
                    + "\nPublic lexical flags: " + ",".join(str(int(x)) for x in f)
                    for c, f in zip(candidates, flags, strict=True)]
    previous = candidates[ids.index(previous_candidate_id)]["text"]
    question = Question(
        id=f"r{row_index}",
        question=TASK + "\n" + query["query_text"] + "\nPrevious committed value:\n" + previous
        + "\nLexical flag order: " + ",".join(FEATURES),
        candidates=[Candidate(id=cid, description=text)
                    for cid, text in zip(mapping, descriptions, strict=True)],
    )
    return question, mapping


def make_request(context, questions):
    return DecisionRequest(context=context, questions=questions)


def question_from_metadata(row, query, flags):
    """Project a mixed evaluator row onto the three allowed identity/state fields."""
    return make_question(row_index=row["row_index"], query=query,
                         previous_candidate_id=row["previous_candidate_id"], flags=flags)


def pilot_ids(requests):
    """Choose coverage using identities and prompt length, never labels/scores."""
    groups = sorted({(r["dialogue_id"], r["time"]) for r in requests})
    selected = set(groups[:12])
    # Include each arm's longest full prompt and every chunk of its turn group.
    for arm in ARMS:
        rows = [r for r in requests if r["arm"] == arm]
        longest = min(rows, key=lambda r: (-max(map(len, r["tokens"])), r["dialogue_id"],
                                          r["time"], r["request_id"]))
        selected.add((longest["dialogue_id"], longest["time"]))
    return [r["request_id"] for r in requests if (r["dialogue_id"], r["time"]) in selected]
