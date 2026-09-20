"""Pure artificial text/schema checks; no corpus, tokenizer, or model loads."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from openjev.research.dialogue_copy_features import DONTCARE, FEATURES, NONE
from openjev.research.dialogue_copy_features import lexical_stream as original_stream
from openjev.research.dialogue_finetune_inputs import build_dialogue_inputs
from openjev.research.dialogue_number_lexical import (
    lexical_observations,
    lexical_stream,
    number_matching_view,
)
from openjev.research.dialogue_state_data import compile_schema, public_dialogue


def run(users, values=('1', '2'), systems=None):
    ids = [NONE, DONTCARE] + ['value:' + v for v in values]
    return lexical_stream([''] * len(users) if systems is None else systems,
                          users, ids, [None, None, *values])


def fixture():
    q = compile_schema([{'service_name': 'Synthetic', 'description': 'room service', 'slots': [
        {'name': 'rooms', 'description': 'room count', 'is_categorical': True,
         'possible_values': ['1', '2']},
    ]}]).catalog()[0]
    public = public_dialogue({'dialogue_id': 'artificial', 'turns': [
        {'speaker': 'USER', 'utterance': 'hello'},
        {'speaker': 'SYSTEM', 'utterance': 'two rooms?'},
        {'speaker': 'USER', 'utterance': 'one please'},
        {'speaker': 'USER', 'utterance': 'thanks'},
        {'speaker': 'SYSTEM', 'utterance': 'one?'},
        {'speaker': 'USER', 'utterance': 'two'},
    ]})
    return public, q


def test_every_cardinal_and_canonical_decimal_zero_through_ninety_nine():
    small = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
             'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
             'seventeen', 'eighteen', 'nineteen']
    tens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    for value in range(100):
        if value < 20:
            word = small[value]
        else:
            word = tens[value // 10 - 2] + (' ' + small[value % 10] if value % 10 else '')
        assert number_matching_view('(' + word.upper() + ')') == '(' + str(value) + ')'
        assert number_matching_view(word.replace(' ', '-')) == str(value)
        assert number_matching_view(str(value)) == str(value)
    assert number_matching_view('ninety  nine, twenty - one') == '99, 21'


@pytest.mark.parametrize('text', [
    'one hundred', 'one thousand', 'one hundred and one', 'one million twenty one',
    'twenty first', 'twenty-first', 'ninety-ninth', 'first', '21st', '105', '005',
    'someone', 'threesome', 'twentyish', 'one2', '2one', 'x1', '1x',
    'to too for ate oh', 'one and two', 'twenty zero', 'twenty 1',
])
def test_unsupported_phrases_and_partial_tokens_are_not_rewritten(text):
    assert number_matching_view(text) == text


def test_no_spurious_small_candidate_from_unsupported_number():
    x, y = run(['one hundred', 'one thousand', 'twenty first', '105', '005'], ('1', '20', '5'))
    assert not x[:, 2:, :4].any()
    assert y.tolist() == [0] * 5
    x, y = run(['105', '005'], ('105', '005'))
    assert y.tolist() == [2, 3]  # Original exact literal support is retained.


def test_all_ten_features_recomputed_for_system_write_and_carry():
    x, y = run(['hello', 'one please', 'thanks', 'two'], systems=['two?', 'two?', 'one?', 'one?'])
    assert x.shape == (4, 4, 10) and x.dtype == np.float32 and y.dtype == np.int64
    assert len(FEATURES) == 10
    assert y.tolist() == [0, 2, 2, 3]
    assert x[:, :, 4].argmax(-1).tolist() == y.tolist()
    assert x[:, :, 5].argmax(-1).tolist() == [0, 0, 2, 2]
    assert x[0, 3, [1, 3]].tolist() == [1, 1]
    assert x[1, 2, [0, 2]].tolist() == [1, 1]
    assert x[:, 0, 6].tolist() == x[:, 1, 7].tolist() == [1] * 4
    assert not x[:, :, 8:].any()


def test_candidate_words_match_digits_and_phrase_length_uses_matching_view():
    x, y = run(['21', 'one or two', 'one bedroom'], ('twenty one', 'one', 'two', '1 bedroom'))
    assert y.tolist() == [2, 2, 5]
    assert x[1, 3:5, 0].tolist() == [1, 1] and not x[1, :, 2].any()
    assert x[2, [3, 5], 0].tolist() == [1, 1]
    assert x[2, 5, 2] == 1


def test_alias_collisions_are_not_merged_or_selected_by_original_length_or_order():
    x, y = run(['two', 'one', '1', 'twenty-one'], ('1', 'one', '2', '21', 'twenty one'))
    assert y.tolist() == [4, 4, 4, 4]
    assert x[1, [2, 3], 0].tolist() == [1, 1]
    assert x[3, [5, 6], 0].tolist() == [1, 1]
    assert not x[1:, :, 2].any()


def test_nonnumber_exact_parity_reserved_literal_none_and_original_boolean_cues():
    ids = [NONE, DONTCARE, 'value:True', 'value:False', 'value:None', 'value:north east']
    values = [None, None, 'True', 'False', 'None', 'north east']
    systems = ['True?', 'None?', 'north east?', 'False?']
    users = ['yes', 'no problem', 'None', 'not north east']
    expected = original_stream(systems, users, ids, values)
    actual = lexical_stream(systems, users, ids, values)
    for first, second in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(first, second)
    assert actual[1].tolist() == [0, 0, 4, 5]


def test_prefix_causality_candidate_permutation_and_input_immutability():
    ids = [NONE, DONTCARE, 'value:1', 'value:one', 'value:2']
    values = [None, None, '1', 'one', '2']
    users, systems = ['two', 'one', 'thanks'], ['one?', '', 'two?']
    before = copy.deepcopy((ids, values, users, systems))
    x, y = lexical_stream(systems, users, ids, values)
    longer, ly = lexical_stream(systems + ['one?'], users + ['one'], ids, values)
    np.testing.assert_array_equal(x, longer[:3])
    np.testing.assert_array_equal(y, ly[:3])
    perm = [4, 2, 1, 0, 3]
    permuted, py = lexical_stream(systems, users, [ids[i] for i in perm], [values[i] for i in perm])
    np.testing.assert_array_equal(x[:, perm], permuted)
    assert [ids[i] for i in y] == [[ids[i] for i in perm][i] for i in py]
    assert (ids, values, users, systems) == before


def test_public_wrapper_consumes_unscored_user_turns_and_never_gold_or_encoder_strings():
    public, catalog = fixture()

    class PublicOnly(dict):
        def __getitem__(self, key):
            assert key == 'candidates', 'Read query annotation or encoder text'
            return super().__getitem__(key)

    q = PublicOnly(candidates=[{'id': c['id'], 'value': c['value'], 'text': object()}
                               for c in catalog['candidates']], target=object(), query_text=object())
    for turn in public['turns']:
        turn['frames'] = object()
    x, y = lexical_observations(public, q)
    assert x.shape == (4, 4, 10) and y.tolist() == [0, 2, 2, 3]
    # The consecutive USER step has no preceding SYSTEM, rather than replaying an old one.
    assert not x[2, :, 1].any()
    ids = [c['id'] for c in catalog['candidates']]
    perm = [3, 0, 2, 1]
    px, py = lexical_observations(public, q, [ids[i] for i in perm])
    np.testing.assert_array_equal(x[:, perm], px)
    assert [ids[i] for i in y] == [[ids[i] for i in perm][i] for i in py]


def test_encoder_payload_and_original_feature_ids_are_unchanged_by_control():
    public, q = fixture()
    ids, values = [c['id'] for c in q['candidates']], [c['value'] for c in q['candidates']]
    packet = {'id': public['dialogue_id'], 'turns': [0, 1, 2, 3], 'queries': [{'query': 0}]}
    queries = [{'id': q['query_id'], 'split': 'train', 'service': q['service'], 'slot': q['slot'],
                'text': 4, 'candidates': [5, 6, 7, 8], 'candidate_ids': ids, 'candidate_values': values}]
    layout = {'id': public['dialogue_id'], 'query_ids': [0], 'shape': [4, 1, 4, 10], 'offset': 0}
    seen = []
    payload = build_dialogue_inputs(packet, public, [q], queries, layout,
                                   lambda text: seen.append(text) or [ord(c) for c in text])
    saved = copy.deepcopy(payload)
    x, _ = lexical_observations(public, q, payload['candidate_ids'][0])
    assert payload['lexical'][1, 0, 2, 0] == 0 and x[1, 2, 0] == 1
    assert seen[1] == 'System: two rooms?\nUser: one please'
    for key in payload:
        if key == 'lexical':
            np.testing.assert_array_equal(payload[key], saved[key])
        else:
            assert payload[key] == saved[key]


def test_empty_stream_and_validation():
    x, y = run([])
    assert x.shape == (0, 4, 10) and y.shape == (0,)
    with pytest.raises(TypeError):
        number_matching_view(1)
    with pytest.raises(ValueError):
        lexical_stream([], ['one'], [NONE, DONTCARE], [None, None])
    with pytest.raises(ValueError):
        lexical_stream([''], ['one'], [NONE, DONTCARE, 'value:1'], [None, None, 'one'])
    public, q = fixture()
    with pytest.raises(ValueError):
        lexical_observations(public, q, [NONE, DONTCARE, 'value:1'])
    with pytest.raises(ValueError):
        lexical_observations(public, q, [NONE, DONTCARE, 'value:1', 'value:1'])
    public['user_turns'].pop(1)
    with pytest.raises(ValueError):
        lexical_observations(public, q)
