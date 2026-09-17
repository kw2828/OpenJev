import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1]/'scripts/prepare_ruletaker.py'))


def test_model_input_allowlist_excludes_labels_and_proof_annotations():
    world = {'context': 'A is red. Red things are warm.', 'id': 'secret', 'proof': 'answer'}
    q = {'text': 'A is warm.', 'gold': 'true', 'depth': 2, 'representation': 'hidden', 'proof': 'hidden'}
    actual = MODULE['decision_input'](world, q)
    assert actual == MODULE['decision_input'](world | {'proof': 'changed'}, q | {'gold': 'false', 'depth': 5})
    assert set(actual) == {'context', 'question', 'candidates'}
    assert 'hidden' not in str(actual)


def test_world_fingerprint_is_order_insensitive():
    assert MODULE['world_key']('A is red. B is blue.') == MODULE['world_key'](' B IS BLUE. A is red. ')


def test_verification_intersection_and_label_disagreement():
    world = {'id': 'D2-1', 'questions': [{'id': 'x', 'text': 'A is red.', 'label': True,
                                        'meta': {'Qid': 'Q1', 'QDep': 0, 'strategy': 'proof'}}]}
    qs, excluded = MODULE['valid_questions'](world, {})
    assert not qs and excluded['not_in_problog_release'] == 1
    key = MODULE['verification_key']('D2-1', 'A IS RED')
    qs, excluded = MODULE['valid_questions'](world, {key: True})
    assert qs[0]['gold'] == 'true' and not excluded
    with pytest.raises(ValueError, match='disagree'):
        MODULE['valid_questions'](world, {key: False})


def test_problog_suffix_renumbering_does_not_change_alignment():
    world = {'id': 'D2-1', 'questions': [{'id': 'D2-1-11', 'text': 'A is red.', 'label': True,
                                        'meta': {'Qid': 'Q11', 'QDep': 2, 'strategy': 'proof'}}]}
    key = MODULE['verification_key']('D2-1', 'A is red.')
    qs, _ = MODULE['valid_questions'](world, {key: True, 'D2-1_11': False})
    assert qs[0]['id'] == 'D2-1-11' and qs[0]['gold'] == 'true'


def test_depth_shift_is_explicit_and_disjoint_from_training():
    assert MODULE['DEPTHS']['train'] == {0, 1, 2}
    assert MODULE['DEPTHS']['dev_in'] == {0, 1, 2}
    assert MODULE['DEPTHS']['dev_shift'] == {3, 4, 5}
    assert MODULE['DEPTHS']['train'].isdisjoint(MODULE['DEPTHS']['dev_shift'])
