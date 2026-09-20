"""Public single-dialogue layouts for a synthetic trainable-encoder cost pilot.

No file, model, tokenizer, or feature-cache loading. The caller supplies a local
content-token tokenizer and authenticates the original metadata. The only field
read from evaluator query records is their declared integer query inventory.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from openjev.research.dialogue_copy_features import DONTCARE, NONE, lexical_stream


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value):
    return type(value) is int and value >= 0


def _sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _public_pairs(public):
    _require(isinstance(public, Mapping) and type(public.get('dialogue_id')) is str
             and public['dialogue_id'], 'Public dialogue identity')
    turns = public['turns']
    _require(_sequence(turns) and turns, 'Nonempty public turns')
    expected, systems, users = [], [], []
    for i, turn in enumerate(turns):
        _require(isinstance(turn, Mapping) and turn.get('speaker') in ('USER', 'SYSTEM')
                 and type(turn.get('utterance')) is str, 'Public role/text fields')
        if turn['speaker'] == 'USER':
            si = i - 1 if i and turns[i - 1]['speaker'] == 'SYSTEM' else None
            expected.append({'turn_index': i, 'previous_system_turn_index': si})
            systems.append('' if si is None else turns[si]['utterance'])
            users.append(turn['utterance'])
    _require(expected and _sequence(public['user_turns']), 'Public USER chronology')
    for item in public['user_turns']:
        _require(isinstance(item, Mapping) and _integer(item.get('turn_index'))
                 and (item.get('previous_system_turn_index') is None
                      or _integer(item['previous_system_turn_index'])), 'Strict public turn indices')
    _require(list(public['user_turns']) == expected, 'Complete public USER chronology')
    return systems, users, [item['turn_index'] for item in expected]


def _catalog(catalog_queries):
    _require(_sequence(catalog_queries) and catalog_queries, 'Supplied TRAIN catalog')
    result = {}
    for q in catalog_queries:
        _require(isinstance(q, Mapping) and type(q.get('query_id')) is str
                 and q['query_id'] not in result, 'Distinct catalog query IDs')
        _require(json.loads(q['query_id']) == [q['service'], q['slot']], 'Catalog identity')
        _require(all(type(q[k]) is str for k in ('service_description', 'slot_description', 'query_text')),
                 'Catalog public descriptions')
        text = 'Service: ' + q['service_description'] + '\nSlot: ' + q['slot_description']
        _require(q['query_text'] == text, 'Original query text convention')
        candidates = q['candidates']
        _require(_sequence(candidates) and candidates, 'Catalog candidates')
        ids = []
        for c in candidates:
            cid, value = c['id'], c['value']
            _require(type(cid) is str and cid not in ids, 'Distinct canonical candidates')
            ids.append(cid)
            if cid in (NONE, DONTCARE):
                _require(value is None, 'Reserved candidate value')
                label = 'NOT_MENTIONED (no constraint stated)' if cid == NONE else 'DONTCARE (no preference)'
            else:
                _require(type(value) is str and value.strip() and cid == 'value:' + value,
                         'Exact ontology candidate identity')
                label = value
            _require(c['text'] == text + '\nValue: ' + label, 'Original candidate text convention')
        _require(ids.count(NONE) == ids.count(DONTCARE) == 1, 'Both reserved candidate IDs')
        result[q['query_id']] = q
    return result


def build_dialogue_inputs(packet_dialogue, public_dialogue, catalog_queries, packet_queries,
                          lexical_layout, tokenize: Callable[[str], list[int]]):
    """Build one complete public USER stream, with separate supplied query states.

    Tokenization receives exact original strings and must return untruncated
    content IDs without CLS/SEP/padding. It is called once per distinct string
    within this dialogue. IDs retain original feature indices for later parity.
    No target, bin, unseen flag, annotated operation, or gold previous is read.
    """
    _require(callable(tokenize), 'Content tokenizer callable')
    systems, users, user_indices = _public_pairs(public_dialogue)
    did = public_dialogue['dialogue_id']
    _require(packet_dialogue['id'] == did == lexical_layout['id'], 'Dialogue identity join')
    turn_features = packet_dialogue['turns']
    _require(_sequence(turn_features) and len(turn_features) == len(users)
             and all(_integer(x) for x in turn_features), 'Complete original turn-feature map')
    _require(_sequence(packet_queries) and _sequence(packet_dialogue['queries']), 'Packet query inventory')
    query_ids = []
    for row in packet_dialogue['queries']:
        qi = row['query']
        _require(_integer(qi) and qi < len(packet_queries), 'Supplied query index')
        query_ids.append(qi)
    query_ids = sorted(set(query_ids))
    _require(query_ids, 'At least one supplied query')
    layout_ids = lexical_layout['query_ids']
    _require(_sequence(layout_ids) and all(_integer(x) for x in layout_ids)
             and list(layout_ids) == query_ids, 'Sorted independent supplied query layout')
    catalog = _catalog(catalog_queries)
    selected = []
    for qi in query_ids:
        entry = packet_queries[qi]
        _require(entry['split'] == 'train' and entry['id'] in catalog, 'TRAIN supplied schema only')
        q = catalog[entry['id']]
        _require((entry['service'], entry['slot']) == (q['service'], q['slot']), 'Schema service/slot join')
        _require(entry['candidate_ids'] == [c['id'] for c in q['candidates']]
                 and entry['candidate_values'] == [c['value'] for c in q['candidates']], 'Exact candidate identity/order')
        _require(_integer(entry['text']) and _sequence(entry['candidates'])
                 and len(entry['candidates']) == len(q['candidates'])
                 and all(_integer(x) for x in entry['candidates']), 'Schema feature-index map')
        selected.append((entry, q))
    cmax = max(len(q['candidates']) for _, q in selected)
    shape = lexical_layout['shape']
    _require(_sequence(shape) and all(_integer(x) for x in shape)
             and list(shape) == [len(users), len(selected), cmax, 10], 'Full public lexical shape')
    _require(_integer(lexical_layout['offset']), 'Original lexical offset')

    # All public/schema joins above precede tokenization; target fields are never inspected.
    tokens, original_features, text_ids, feature_text = [], [], {}, {}

    def add(text, original_feature):
        _require(original_feature not in feature_text or feature_text[original_feature] == text,
                 'Original feature ID maps to inconsistent public text')
        feature_text[original_feature] = text
        if text in text_ids:
            index = text_ids[text]
            _require(original_features[index] == original_feature, 'Duplicate text has inconsistent original feature ID')
            return index
        ids = tokenize(text)
        _require(type(ids) is list and ids and all(_integer(x) for x in ids), 'Nonempty content token IDs')
        index = len(tokens)
        text_ids[text] = index
        tokens.append(list(ids))
        original_features.append(original_feature)
        return index

    turn_ids = [add('System: ' + system + '\nUser: ' + user, feature)
                for system, user, feature in zip(systems, users, turn_features, strict=True)]
    query_text_ids, candidate_text_ids, candidate_ids = [], [], []
    lexical = np.zeros((len(users), len(selected), cmax, 10), dtype=np.float32)
    for position, (entry, q) in enumerate(selected):
        query_text_ids.append(add(q['query_text'], entry['text']))
        candidate_text_ids.append([add(c['text'], feature)
                                   for c, feature in zip(q['candidates'], entry['candidates'], strict=True)])
        ids = list(entry['candidate_ids'])
        candidate_ids.append(ids)
        values, _ = lexical_stream(systems, users, ids, entry['candidate_values'])
        lexical[:, position, :len(ids)] = values
    return {'dialogue_id': did, 'query_ids': query_ids, 'user_turn_indices': user_indices,
            'tokens': tokens, 'original_feature_ids': original_features, 'turn_text_ids': turn_ids,
            'query_text_ids': query_text_ids, 'candidate_text_ids': candidate_text_ids,
            'candidate_ids': candidate_ids, 'lexical': lexical}


def workload_profile(payload, *, chunk_tokens=254, chunk_batch_size=32):
    """Deterministic work geometry, not measured time, memory or model quality.

    Flatten unique texts in payload order, split content without truncation, add
    CLS/SEP per chunk, then pad each consecutive chunk batch to its maximum.
    """
    _require(type(chunk_tokens) is int and chunk_tokens > 0
             and type(chunk_batch_size) is int and chunk_batch_size > 0, 'Positive chunk dimensions')
    tokens = payload['tokens']
    _require(type(tokens) is list and tokens and all(type(x) is list and x and all(_integer(t) for t in x)
                                                   for x in tokens), 'Nonempty token lists')
    original = payload['original_feature_ids']
    _require(type(original) is list and len(original) == len(tokens) and all(_integer(x) for x in original)
             and len(set(original)) == len(original), 'Unique original feature map')
    turns, queries, candidates = payload['turn_text_ids'], payload['query_text_ids'], payload['candidate_text_ids']
    _require(type(turns) is list and turns and type(queries) is list and queries
             and type(candidates) is list and len(candidates) == len(queries), 'Public actor maps')
    _require(all(type(cs) is list and cs for cs in candidates), 'Candidate maps')
    mapped = turns + queries + [i for cs in candidates for i in cs]
    _require(all(_integer(i) and i < len(tokens) for i in mapped)
             and set(mapped) == set(range(len(tokens))), 'Complete text-map coverage')
    ids = payload['candidate_ids']
    _require(type(ids) is list and len(ids) == len(candidates), 'Canonical candidate maps')
    for cs, ci in zip(candidates, ids, strict=True):
        _require(type(ci) is list and len(ci) == len(cs) and all(type(x) is str for x in ci)
                 and len(set(ci)) == len(ci) and ci.count(NONE) == ci.count(DONTCARE) == 1,
                 'Candidate identity support')
    t, q, c = len(turns), len(queries), max(map(len, candidates))
    lexical = payload['lexical']
    _require(isinstance(lexical, np.ndarray) and lexical.dtype == np.float32
             and lexical.shape == (t, q, c, 10) and np.isfinite(lexical).all()
             and np.isin(lexical, (0, 1)).all(), 'Finite binary lexical actor')
    for j, cs in enumerate(candidates):
        _require(not lexical[:, j, len(cs):].any(), 'Padded candidate lexical slots must be zero')
    lengths = [min(chunk_tokens, len(ids) - start) + 2
               for ids in tokens for start in range(0, len(ids), chunk_tokens)]
    groups = [lengths[i:i + chunk_batch_size] for i in range(0, len(lengths), chunk_batch_size)]
    padded = sum(len(group) * max(group) for group in groups)
    return {'unique_texts': len(tokens), 'input_texts': len(tokens), 'content_tokens': sum(map(len, tokens)),
            'chunks': len(lengths), 'encoder_sequences': len(lengths), 'special_token_positions': 2 * len(lengths),
            'valid_token_positions': sum(lengths), 'padded_token_positions': padded,
            'padding_token_positions': padded - sum(lengths),
            'padded_attention_positions': sum(len(group) * max(group)**2 for group in groups),
            'encoder_calls': len(groups), 'chunk_tokens': chunk_tokens, 'chunk_batch_size': chunk_batch_size,
            'overlength_texts_chunked': sum(len(ids) > chunk_tokens for ids in tokens), 'truncated_tokens': 0,
            'max_chunk_tokens_with_special': max(lengths), 'public_user_turns': t, 'queries': q,
            'schema_text_occurrences': q, 'candidate_text_occurrences': sum(map(len, candidates)),
            'max_candidates': c, 'real_question_updates': t * q,
            'real_candidate_updates': t * sum(map(len, candidates)),
            'padded_candidate_positions': t * q * c, 'lexical_scalars': int(lexical.size),
            'lexical_bytes': int(lexical.nbytes)}


def select_representatives(profiles):
    """Fixed 10/50/90 percentile ranks floor(p*(N-1)), ties by dialogue ID.

    Input entries contain only dialogue_id and work metadata. This selector does
    not inspect labels, text, predictions, timing, or token content.
    """
    _require(_sequence(profiles) and len(profiles) >= 4, 'At least four workload profiles')
    seen = set()
    for entry in profiles:
        did = entry['dialogue_id']
        _require(type(did) is str and did and did not in seen, 'Distinct dialogue profiles')
        seen.add(did)
        _require(isinstance(entry['work'], Mapping) and
                 _integer(entry['work'].get('padded_attention_positions')), 'Attention-position work proxy')
    ranked = sorted(profiles, key=lambda x: (x['work']['padded_attention_positions'], x['dialogue_id']))
    result = []
    for percentile in (10, 50, 90):
        rank = percentile * (len(ranked) - 1) // 100
        entry = ranked[rank]
        result.append({'percentile': percentile, 'rank': rank, 'dialogue_id': entry['dialogue_id'],
                       'work': dict(entry['work'])})
    return result


def synthetic_targets(payload, *, offset=0):
    """Public schedule (USER step + query position + offset) modulo support.

    Returned int64 targets are separate from actor inputs and unrelated to SGD
    labels or scored-endpoint availability. Every real public query step is used.
    """
    _require(_integer(offset), 'Nonnegative synthetic schedule offset')
    work = workload_profile(payload)
    return np.asarray([[(t + q + offset) % len(ids) for q, ids in enumerate(payload['candidate_ids'])]
                       for t in range(work['public_user_turns'])], dtype=np.int64)
