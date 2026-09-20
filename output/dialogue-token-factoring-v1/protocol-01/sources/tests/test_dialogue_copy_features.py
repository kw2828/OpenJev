import numpy as np
import pytest

from openjev.research.dialogue_copy_features import DONTCARE, FEATURES, NONE, lexical_stream


def run(users, systems=None, values=None):
    values = [None, None, "Italian", "French", "None"] if values is None else values
    ids = [NONE, DONTCARE] + ["value:" + v for v in values[2:]]
    return lexical_stream([""] * len(users) if systems is None else systems, users, ids, values)


def test_literal_persistence_and_reserved_identity():
    x, y = run(["Italian", "nothing new", "None", "French"])
    assert len(FEATURES) == 10
    assert y.tolist() == [2, 2, 4, 3]
    assert x[:, :, 4].argmax(-1).tolist() == y.tolist()
    assert x[:, :, 5].argmax(-1).tolist() == [0, 2, 2, 4]
    assert x.dtype == np.float32
    assert x[:, 0, 6].tolist() == [1] * 4
    assert not x[:, 4, 6].any()


def test_negation_and_relevance_are_not_fabricated():
    x, y = run(["not Italian", "My friend's choice is French"])
    assert y.tolist() == [2, 3]  # A noisy observation, never a semantic label.
    assert x[0, 2, 0] == 1


def test_system_does_not_write_and_ties_carry():
    x, y = run(["thanks", "Italian", "thanks"], ["French?", "French?", "Italian?"])
    assert y.tolist() == [0, 2, 2]
    assert x[0, 3, 1] == 1
    x, y = run(["red", "red and tan"], values=[None, None, "red", "tan"])
    assert y.tolist() == [2, 2]
    assert not x[1, :, 2].any()


def test_longest_unique_and_word_boundaries():
    x, y = run(["northern", "north east"], values=[None, None, "north", "north east"])
    assert y.tolist() == [0, 3]
    assert x[1, 2:4, 0].tolist() == [1, 1]
    assert x[1, 2:4, 2].tolist() == [0, 1]


def test_causal_prefix_invariance():
    a, ay = run(["Italian", "thanks"])
    b, by = run(["Italian", "thanks", "French"])
    np.testing.assert_array_equal(a, b[:2])
    np.testing.assert_array_equal(ay, by[:2])


def test_permutation_moves_reserved_state_and_values():
    ids = [NONE, DONTCARE, "value:Italian", "value:French"]
    values = [None, None, "Italian", "French"]
    a, ay = lexical_stream(["", "", ""], ["thanks", "Italian", "French"], ids, values)
    perm = [3, 1, 0, 2]
    b, by = lexical_stream(["", "", ""], ["thanks", "Italian", "French"],
                           [ids[i] for i in perm], [values[i] for i in perm])
    np.testing.assert_array_equal(a[:, perm], b)
    assert [ids[i] for i in ay] == [[ids[i] for i in perm][j] for j in by]


def test_boolean_words_are_noisy_observations_and_do_not_write():
    x, y = run(["yes", "No thanks", "no problem, yes", "yesterday"],
               values=[None, None, "True", "False"])
    assert y.tolist() == [0, 0, 0, 0]
    assert x[:, 2, 8].tolist() == [1, 0, 1, 0]
    assert x[:, 3, 9].tolist() == [0, 1, 1, 0]
    assert not x[:, 0:2, 8:10].any()


def test_immutability_and_empty_stream():
    users, systems = ["Italian"], ["French?"]
    run(users, systems)
    assert users == ["Italian"] and systems == ["French?"]
    x, y = run([])
    assert x.shape == (0, 5, 10) and y.shape == (0,)


@pytest.mark.parametrize("values", [[None, None, ""], [None, None, "  "], [None, None, "Italian", "Italian"]])
def test_invalid_schema(values):
    with pytest.raises(ValueError):
        run(["Italian"], values=values)


def test_no_gold_argument_and_reject_misaligned_inputs():
    with pytest.raises(TypeError):
        lexical_stream([], [], [NONE, DONTCARE], [None, None], gold=[0])
    with pytest.raises(ValueError):
        lexical_stream([], ["yes"], [NONE, DONTCARE], [None, None])
    with pytest.raises(ValueError):
        lexical_stream([""], ["yes"], [NONE, DONTCARE, "value:French"], [None, None, "Italian"])
