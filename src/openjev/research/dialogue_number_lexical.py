"""Number-normalized matching views for the existing ten lexical observations.

This is a noisy literal control, not a number/intent parser. Only canonical
decimal 0..99 and English cardinals in that range are equivalent. Hyphenated or
space-separated tens-plus-unit forms are supported. Maximal adjacent numeric
phrases containing ordinals, magnitudes, leading zeros, or an unsupported
composition are left intact, including ``one hundred and one``. Homophones,
ordinals and arbitrary numeric IDs are not converted. Sign/decimal semantics are
not interpreted; the original matcher's word-boundary rules remain unchanged.
Original public text, encoder strings, canonical IDs, and Boolean cues are never
rewritten. Normalized aliases retain separate identities and tie ambiguously.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from openjev.research.dialogue_copy_features import (
    AFFIRMATIVE,
    DONTCARE,
    FEATURES,
    NEGATIVE,
    NONE,
)
from openjev.research.dialogue_finetune_inputs import _public_pairs

_SMALL = dict(zip(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
     "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
     "seventeen", "eighteen", "nineteen"],
    range(20), strict=True,
))
_TENS = dict(zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"],
                 range(20, 100, 10), strict=True))
_ORDINALS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
             "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
             "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
             "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth"]
_MAGNITUDES = ["hundred", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion",
               "hundredth", "thousandth", "millionth", "billionth", "trillionth"]
_WORDS = sorted(set(_SMALL) | set(_TENS) | set(_ORDINALS) | set(_MAGNITUDES),
                key=lambda x: (-len(x), x))
_ATOM = r"(?:" + "|".join(_WORDS) + r"|[0-9]+(?:st|nd|rd|th)?)(?!\w)"
_SEP = r"(?:[ \t]*-[ \t]*|[ \t]+)"
_NUMBER_RUN = re.compile(r"(?<!\w)" + _ATOM + r"(?:" + _SEP
                         + r"(?:and" + _SEP + r")?" + _ATOM + r")*", re.IGNORECASE)
_SPLIT = re.compile(_SEP)
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]?)\Z")


def number_matching_view(text: str) -> str:
    """Return a matching-only view; leave unsupported numeric phrases intact.

    A phrase such as ``one and two`` is conservatively unsupported rather than
    interpreted as a list. ``one or two`` contains two independently recognized
    cardinals. Matching length below is the normalized spelling's length.
    """
    if type(text) is not str:
        raise TypeError("Matching view requires public text")

    def convert(match):
        raw = match.group(0)
        parts = _SPLIT.split(raw.casefold())
        if len(parts) == 1:
            word = parts[0]
            if word in _SMALL:
                return str(_SMALL[word])
            if word in _TENS:
                return str(_TENS[word])
            if _DECIMAL.fullmatch(word):
                return word
        elif len(parts) == 2 and parts[0] in _TENS and 1 <= _SMALL.get(parts[1], 0) <= 9:
            return str(_TENS[parts[0]] + _SMALL[parts[1]])
        return raw

    return _NUMBER_RUN.sub(convert, text)


def lexical_stream(system_turns: Sequence[str], user_turns: Sequence[str],
                   candidate_ids: Sequence[str], candidate_values: Sequence[str | None]):
    """Return float32 [T,C,10] and int64 [T], preserving the original feature order.

    Every public step is consumed. Only a unique longest USER match writes the
    register; SYSTEM matches and ties carry. Boolean cues use original text.
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
    output = np.zeros((len(user_turns), len(candidate_ids), len(FEATURES)), np.float32)
    literal = np.empty(len(user_turns), np.int64)
    none_index, dontcare_index = candidate_ids.index(NONE), candidate_ids.index(DONTCARE)
    output[:, none_index, 6] = 1
    output[:, dontcare_index, 7] = 1
    original = {i: v.strip().casefold() for i, v in enumerate(candidate_values) if v is not None}
    values = {i: number_matching_view(v) for i, v in original.items()}
    patterns = {i: re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)") for i, v in values.items()}

    def matches(text):
        view = number_matching_view(text.casefold())
        hits = [i for i, pattern in patterns.items() if pattern.search(view)]
        if not hits:
            return hits, None
        longest = max(len(values[i]) for i in hits)
        winners = [i for i in hits if len(values[i]) == longest]
        return hits, winners[0] if len(winners) == 1 else None

    current = none_index
    for step, (system, user) in enumerate(zip(system_turns, user_turns, strict=True)):
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
        positive = bool(AFFIRMATIVE.search(user.casefold()))
        negative = bool(NEGATIVE.search(user.casefold()))
        for index, value in original.items():
            output[step, index, 8] = value == "true" and positive
            output[step, index, 9] = value == "false" and negative
    return output, literal


def lexical_observations(public_dialogue, catalog_query, candidate_ids=None):
    """Consume complete public chronology for one supplied catalog query.

    Optional candidate IDs must permute the entire catalog support. No labels,
    scored endpoint masks, query text, or candidate encoder strings are read.
    """
    systems, users, _ = _public_pairs(public_dialogue)
    if not isinstance(catalog_query, Mapping):
        raise TypeError("Supplied public catalog query required")
    candidates = catalog_query['candidates']
    if (isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence)
            or not candidates or any(not isinstance(c, Mapping) for c in candidates)):
        raise ValueError("Public candidate catalog required")
    original_ids = [c['id'] for c in candidates]
    values = [c['value'] for c in candidates]
    if any(type(cid) is not str for cid in original_ids) or len(set(original_ids)) != len(original_ids):
        raise ValueError("Distinct canonical catalog identities")
    if candidate_ids is None:
        ids = original_ids
    else:
        if (isinstance(candidate_ids, (str, bytes)) or not isinstance(candidate_ids, Sequence)
                or len(candidate_ids) != len(original_ids)
                or any(type(cid) is not str for cid in candidate_ids)
                or set(candidate_ids) != set(original_ids)):
            raise ValueError("Candidate IDs must permute the complete catalog")
        ids = list(candidate_ids)
        by_id = dict(zip(original_ids, values, strict=True))
        values = [by_id[cid] for cid in ids]
    return lexical_stream(systems, users, ids, values)
