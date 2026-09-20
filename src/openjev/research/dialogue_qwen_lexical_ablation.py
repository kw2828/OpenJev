"""Exact lexical-only prompt subtraction; no tokenizer, model or label input."""
from __future__ import annotations

import copy
import re

from openjev.research.dialogue_qwen_observation import FEATURES, TASK, candidate_type

EXPLANATION = (
    "Lexical flags are noisy public string-match observations, not answers. "
    "The literal register carries the last unique longest USER mention across the "
    "public prefix; it does not understand negation or relevance. "
)
LEGEND = "\nLexical flag order: " + ",".join(FEATURES)
FLAG_SUFFIX = re.compile(r"(.*)\nPublic lexical flags: ([01](?:,[01]){9})", re.DOTALL)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strip_question(question, mapping):
    """Remove only the three prescribed blocks, preserving all other characters."""
    require(TASK.endswith(EXPLANATION), "Frozen lexical explanation changed")
    text = question["question"]
    require(text.startswith(TASK + "\n") and text.endswith(LEGEND), "Exact lexical task/legend boundary")
    result = copy.deepcopy(question)
    result["question"] = TASK[:-len(EXPLANATION)] + text[len(TASK):-len(LEGEND)]
    ids = [candidate["id"] for candidate in question["candidates"]]
    require(len(ids) == len(set(ids)) and set(ids) == set(mapping)
            and len(set(mapping.values())) == len(ids), "Candidate map bijection")
    for candidate in result["candidates"]:
        match = FLAG_SUFFIX.fullmatch(candidate["description"])
        require(match is not None, "Exact trailing ten binary lexical flags")
        description = match.group(1)
        require(description.endswith("\nCandidate type: " + candidate_type(mapping[candidate["id"]])),
                "Preserved candidate type")
        candidate["description"] = description
    return result


def transform_request(original):
    """Keep routing, candidate labels and old tokens until separately retokenized."""
    transformed = copy.deepcopy(original)
    questions, maps, orders = (original["request"]["questions"], original["canonical_id_maps"],
                              original["ordered_candidate_ids"])
    require(len(questions) == len(maps) == len(orders) == len(original["row_indices"]), "Aligned question metadata")
    transformed["request"]["questions"] = [strip_question(q, m) for q, m in zip(questions, maps, strict=True)]
    for old, new, mapping, order in zip(questions, transformed["request"]["questions"], maps, orders, strict=True):
        require(order == [mapping[c["id"]] for c in sorted(old["candidates"], key=lambda c: c["description"])]
                == [mapping[c["id"]] for c in sorted(new["candidates"], key=lambda c: c["description"])],
                "Lexical subtraction changed prompt-label order")
    return transformed


def verify_transform(original, transformed):
    """Reusable paired-request check; token contents are verified by tokenization separately."""
    expected = transform_request(original)
    require(set(transformed) == set(expected), "Request field identity")
    require({k: v for k, v in transformed.items() if k != "tokens"}
            == {k: v for k, v in expected.items() if k != "tokens"}, "Only prescribed lexical prompt changes")
    require(isinstance(transformed["tokens"], list)
            and len(transformed["tokens"]) == len(original["tokens"]), "Retokenized question coverage")
    return True
