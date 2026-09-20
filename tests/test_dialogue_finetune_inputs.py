"""Artificial public strings and fake tokenization only; no corpus or models."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from openjev.research.dialogue_finetune_inputs import (
    build_dialogue_inputs,
    select_representatives,
    synthetic_targets,
    workload_profile,
)
from openjev.research.dialogue_state_data import compile_schema, public_dialogue


def fixture():
    catalog = compile_schema([{'service_name': 'Paint', 'description': 'paint service', 'slots': [
        {'name': 'color', 'description': 'desired color', 'is_categorical': True,
         'possible_values': ['red', 'blue']},
        {'name': 'enabled', 'description': 'enable paint', 'is_categorical': True,
         'possible_values': ['True', 'False', 'None']},
    ]}]).catalog()
    entries = []
    for i, q in enumerate(catalog):
        entries.append({'id': q['query_id'], 'split': 'train', 'service': q['service'], 'slot': q['slot'],
                        'text': 10 * (i + 1), 'candidates': list(range(10 * (i + 1) + 1,
                                                                    10 * (i + 1) + 1 + len(q['candidates']))),
                        'candidate_ids': [c['id'] for c in q['candidates']],
                        'candidate_values': [c['value'] for c in q['candidates']]})
    raw = {'dialogue_id': 'synthetic-dialogue', 'services': ['UNUSED'], 'turns': [
        {'speaker': 'USER', 'utterance': 'hello'},
        {'speaker': 'SYSTEM', 'utterance': 'choose red'},
        {'speaker': 'USER', 'utterance': 'red'},
        {'speaker': 'SYSTEM', 'utterance': ''},
        {'speaker': 'USER', 'utterance': 'hello'},
        {'speaker': 'SYSTEM', 'utterance': 'blue?'},
        {'speaker': 'USER', 'utterance': 'not blue'},
    ]}
    public = public_dialogue(raw)
    packet = {'id': 'synthetic-dialogue', 'turns': [0, 1, 0, 2], 'user_text': ['ignored'],
              'queries': [{'query': 1, 'time': 3, 'label': 0, 'bin': 'clear'},
                          {'query': 0, 'time': 0, 'label': 2, 'bin': 'first_assignment'}]}
    layout = {'id': 'synthetic-dialogue', 'query_ids': [0, 1], 'offset': 100, 'shape': [4, 2, 5, 10]}
    return packet, public, catalog, entries, layout


def tokenize(text):
    return [ord(c) + 3 for c in text]


def build(items=None, tokenizer=tokenize):
    return build_dialogue_inputs(*(fixture() if items is None else items), tokenizer)


def resolved(payload, refs):
    return [payload['tokens'][i] for i in refs]


def test_exact_strings_feature_identity_and_complete_unscored_steps():
    calls = []
    items = fixture()
    payload = build(items, lambda text: calls.append(text) or tokenize(text))
    assert payload['query_ids'] == [0, 1]
    assert payload['user_turn_indices'] == [0, 2, 4, 6]
    assert payload['turn_text_ids'] == [0, 1, 0, 2]
    assert calls[:3] == ['System: \nUser: hello', 'System: choose red\nUser: red', 'System: blue?\nUser: not blue']
    assert len(calls) == len(set(calls)) == len(payload['tokens'])
    assert resolved(payload, payload['query_text_ids']) == [tokenize(q['query_text']) for q in items[2]]
    for q, refs in zip(items[2], payload['candidate_text_ids'], strict=True):
        assert resolved(payload, refs) == [tokenize(c['text']) for c in q['candidates']]
    assert payload['original_feature_ids'] == [0, 1, 2, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25]
    assert payload['lexical'].shape == (4, 2, 5, 10)
    assert not payload['lexical'][:, 0, 4].any()
    # The unscored red USER writes the public literal register, then it carries.
    assert payload['lexical'][1, 0, 2, 4] == payload['lexical'][2, 0, 2, 4] == 1
    assert payload['lexical'][3, 0, 3, 4] == 1  # No annotation-informed negation repair.


def test_actor_never_reads_annotation_values_or_annotation_availability():
    class QueryOnly(dict):
        def __getitem__(self, key):
            assert key == 'query', 'Annotation field was read'
            return super().__getitem__(key)

    original = build()
    items = fixture()
    items[0]['queries'] = [QueryOnly(query=1, label=object(), time=-999, bin=object()),
                           QueryOnly(query=0, label=object(), time=999), QueryOnly(query=1)]
    items[0]['user_text'] = object()
    for turn in items[1]['turns']:
        turn['frames'] = {'target': object()}
    changed = build(items)
    for key in original:
        if key == 'lexical':
            np.testing.assert_array_equal(original[key], changed[key])
        else:
            assert original[key] == changed[key]
    assert not ({'labels', 'bins', 'previous', 'targets', 'eligible'} & changed.keys())


def test_future_public_text_cannot_change_prefix_or_schema_inputs():
    first = build()
    items = fixture()
    items[1]['turns'][5]['utterance'] = 'completely different future system'
    items[1]['turns'][6]['utterance'] = 'red red future '
    second = build(items)
    assert resolved(first, first['turn_text_ids'][:3]) == resolved(second, second['turn_text_ids'][:3])
    assert resolved(first, first['query_text_ids']) == resolved(second, second['query_text_ids'])
    np.testing.assert_array_equal(first['lexical'][:3], second['lexical'][:3])


def test_candidate_permutation_preserves_canonical_features_and_reserved_none():
    first = build()
    items = fixture()
    order = [2, 3, 0, 1]
    q, entry = items[2][0], items[3][0]
    q['candidates'] = [q['candidates'][i] for i in order]
    for key in ('candidate_ids', 'candidate_values', 'candidates'):
        entry[key] = [entry[key][i] for i in order]
    second = build(items)
    assert second['candidate_ids'][0].index('reserved:NOT_MENTIONED') == 2
    np.testing.assert_array_equal(second['lexical'][:, 0, :4], first['lexical'][:, 0, order])
    assert resolved(second, second['candidate_text_ids'][0]) == [first['tokens'][first['candidate_text_ids'][0][i]] for i in order]


def test_query_permutation_and_added_unrelated_query_leave_resolved_stream_unchanged():
    first = build()
    items = fixture()
    items[3].reverse()
    for row in items[0]['queries']:
        row['query'] = 1 - row['query']
    second = build(items)
    assert resolved(first, first['turn_text_ids']) == resolved(second, second['turn_text_ids'])
    np.testing.assert_array_equal(first['lexical'][:, 0, :4], second['lexical'][:, 1, :4])
    np.testing.assert_array_equal(first['lexical'][:, 1], second['lexical'][:, 0])
    items = fixture()
    items[0]['queries'] = [items[0]['queries'][1]]
    items[4]['query_ids'], items[4]['shape'] = [0], [4, 1, 4, 10]
    single = build(items)
    assert resolved(first, first['turn_text_ids']) == resolved(single, single['turn_text_ids'])
    np.testing.assert_array_equal(first['lexical'][:, 0, :4], single['lexical'][:, 0])


@pytest.mark.parametrize('mutation', ['layout', 'candidate_order', 'query_text', 'candidate_text', 'dev', 'bool_query'])
def test_structural_mismatches_fail_before_tokenizer(mutation):
    items = fixture()
    if mutation == 'layout':
        items[4]['shape'][0] = 3
    elif mutation == 'candidate_order':
        items[3][0]['candidate_ids'].reverse()
    elif mutation == 'query_text':
        items[2][0]['query_text'] += ' gold label'
    elif mutation == 'candidate_text':
        items[2][0]['candidates'][0]['text'] += ' gold label'
    elif mutation == 'dev':
        items[3][0]['split'] = 'dev'
    else:
        items[0]['queries'][0]['query'] = True
    calls = []
    with pytest.raises(ValueError):
        build(items, lambda text: calls.append(text) or [1])
    assert calls == []


@pytest.mark.parametrize('mutation', ['missing_user', 'bool_time', 'feature_conflict', 'duplicate_text_feature_conflict'])
def test_public_and_feature_identity_failures(mutation):
    items = fixture()
    if mutation == 'missing_user':
        items[1]['user_turns'].pop(1)
    elif mutation == 'bool_time':
        items[1]['user_turns'][0]['turn_index'] = False
    elif mutation == 'feature_conflict':
        items[0]['turns'][1] = 0
    else:
        items[0]['turns'][2] = 3
    with pytest.raises(ValueError):
        build(items)


@pytest.mark.parametrize('ids', [[], [True], [-1], [1.5], (1, 2)])
def test_invalid_tokenizer_contract_rejected(ids):
    with pytest.raises(ValueError, match='content token IDs'):
        build(tokenizer=lambda _: ids)


def test_work_geometry_accounts_for_chunking_padding_and_all_public_heads():
    payload = build(tokenizer=lambda _: [11])
    # Fourteen unique strings: first has seven tokens, second five, remaining one each.
    payload['tokens'][0] = [11] * 7
    payload['tokens'][1] = [12] * 5
    work = workload_profile(payload, chunk_tokens=4, chunk_batch_size=3)
    # Chunk lengths incl specials: 6,5,6 | 3,3,3 | 3,3,3 | 3,3,3 | 3,3,3 | 3.
    assert work['content_tokens'] == 24
    assert work['chunks'] == work['encoder_sequences'] == 16
    assert work['valid_token_positions'] == 56
    assert work['special_token_positions'] == 32
    assert work['padded_token_positions'] == 57
    assert work['padding_token_positions'] == 1
    assert work['padded_attention_positions'] == 225
    assert work['encoder_calls'] == 6
    assert work['overlength_texts_chunked'] == 2 and work['truncated_tokens'] == 0
    assert work['real_question_updates'] == 8
    assert work['real_candidate_updates'] == 36
    assert work['padded_candidate_positions'] == 40
    assert work['lexical_scalars'] == 400 and work['lexical_bytes'] == 1600


def test_work_profile_rejects_orphan_text_and_nonzero_padding():
    payload = build()
    payload['tokens'].append([1])
    payload['original_feature_ids'].append(200)
    with pytest.raises(ValueError, match='coverage'):
        workload_profile(payload)
    payload = build()
    payload['lexical'][0, 0, 4, 0] = 1
    with pytest.raises(ValueError, match='Padded'):
        workload_profile(payload)


def test_representatives_use_fixed_attention_proxy_ranks_and_id_ties():
    profiles = [{'dialogue_id': f'd{i:02}', 'work': {'padded_attention_positions': i // 2,
                  'padded_token_positions': 1000 - i}} for i in range(11)]
    expected = select_representatives(profiles)
    assert [(x['percentile'], x['rank'], x['dialogue_id']) for x in expected] == [
        (10, 1, 'd01'), (50, 5, 'd05'), (90, 9, 'd09')]
    assert select_representatives(profiles[::-1]) == expected
    with pytest.raises(ValueError, match='Distinct dialogue'):
        select_representatives([{'dialogue_id': 'd', 'work': {'padded_attention_positions': 1}}] * 4)


def test_synthetic_labels_cover_all_public_query_steps_without_gold():
    payload = build()
    targets = synthetic_targets(payload)
    np.testing.assert_array_equal(targets, [[0, 1], [1, 2], [2, 3], [3, 4]])
    assert targets.dtype == np.int64
    np.testing.assert_array_equal(synthetic_targets(payload, offset=2), [[2, 3], [3, 4], [0, 0], [1, 1]])
    assert 'targets' not in payload


def test_builder_does_not_mutate_metadata():
    items = fixture()
    before = copy.deepcopy(items)
    build(items)
    assert items == before
