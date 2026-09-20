"""Causal lexical observations for a supplied-schema categorical decision.

Only public SYSTEM/USER strings and the caller's candidates are accepted.
Literal storage does not interpret negation, relevance, or affirmative intent.
The last two features are deliberately noisy boolean word cues, not labels.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

FEATURES = ("user_match", "system_match", "unique_longest_user", "unique_longest_system",
            "literal_current", "literal_previous", "is_none", "is_dontcare",
            "affirmative_cue_for_true", "negative_cue_for_false")
NONE = "reserved:NOT_MENTIONED"
DONTCARE = "reserved:DONTCARE"
AFFIRMATIVE = re.compile(r"(?<!\w)(?:yes|yeah|yep|true)(?!\w)")
NEGATIVE = re.compile(r"(?<!\w)(?:no|nope|false)(?!\w)")


def lexical_stream(system_turns: Sequence[str], user_turns: Sequence[str],
                   candidate_ids: Sequence[str], candidate_values: Sequence[str | None]):
    """Return float32 [T,C,10] observations and int64 [T] literal decisions.

    Each question independently consumes every public turn, whether scored or
    not. Values and IDs move together under a candidate permutation. Ties carry
    the last literal value. SYSTEM mentions never write the literal register.
    """
    for seq in (system_turns, user_turns, candidate_ids, candidate_values):
        if isinstance(seq, (str, bytes)) or not isinstance(seq, Sequence):
            raise TypeError("Expected public string/candidate sequences")
    if (len(system_turns) != len(user_turns)
            or any(type(s) is not str for s in (*system_turns, *user_turns))):
        raise ValueError("One public SYSTEM string per USER string required")
    if (not candidate_ids or len(candidate_ids) != len(candidate_values)
            or any(type(s) is not str for s in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or candidate_ids.count(NONE) != 1 or candidate_ids.count(DONTCARE) != 1):
        raise ValueError("Distinct candidate identities and both reserved entries required")
    for cid, value in zip(candidate_ids, candidate_values, strict=True):
        if cid in (NONE, DONTCARE):
            if value is not None:
                raise ValueError("Reserved identities have no literal ontology value")
        elif type(value) is not str or not value.strip() or cid != "value:" + value:
            raise ValueError("Ontology identity must bind its exact nonempty value")
    t, c = len(user_turns), len(candidate_ids)
    output = np.zeros((t, c, len(FEATURES)), np.float32)
    literal = np.empty(t, np.int64)
    none_index = candidate_ids.index(NONE)
    dontcare_index = candidate_ids.index(DONTCARE)
    output[:, none_index, 6] = 1
    output[:, dontcare_index, 7] = 1
    values = {i: value.strip().casefold() for i, value in enumerate(candidate_values)
              if value is not None}
    patterns = {i: re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)") for i, v in values.items()}

    def matches(text):
        hits = [i for i, pattern in patterns.items() if pattern.search(text)]
        if not hits:
            return hits, None
        longest = max(len(values[i]) for i in hits)
        winners = [i for i in hits if len(values[i]) == longest]
        return hits, winners[0] if len(winners) == 1 else None

    current = none_index
    for step, (system, user) in enumerate(zip(system_turns, user_turns, strict=True)):
        user, system = user.casefold(), system.casefold()
        uh, uw = matches(user)
        sh, sw = matches(system)
        output[step, uh, 0] = 1
        output[step, sh, 1] = 1
        if uw is not None:
            output[step, uw, 2] = 1
        if sw is not None:
            output[step, sw, 3] = 1
        output[step, current, 5] = 1
        if uw is not None:
            current = uw
        output[step, current, 4] = 1
        literal[step] = current
        positive, negative = bool(AFFIRMATIVE.search(user)), bool(NEGATIVE.search(user))
        for index, value in values.items():
            output[step, index, 8] = value == "true" and positive
            output[step, index, 9] = value == "false" and negative
    return output, literal
