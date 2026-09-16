import numpy as np
import pytest

from openjev.research.memory import DecisionHistory, expanded_current


def test_history_is_causal_and_resets_without_hidden_cooldown():
    first = np.array([1.,1.,.2,.1,.5,.8,.6])
    second = np.array([1.,0.,0.,0.,0.,.7,.5])
    history = DecisionHistory()
    before = history.features(first)
    assert len(before) == 17
    np.testing.assert_array_equal(before[7:13], np.zeros(6))
    np.testing.assert_array_equal(before[13:], [0,1,0,0])
    # The current outcome cannot change the saved pre-action features.
    history.update(first, True, 2)
    after = history.features(second)
    np.testing.assert_array_equal(after[7:13], first[1:])
    np.testing.assert_array_equal(after[13:], [1,.1,1,1])
    np.testing.assert_array_equal(before[13:], [0,1,0,0])
    for _ in range(30):
        history.update(second, False, 0)
    assert history.features(first)[14] == 1
    np.testing.assert_array_equal(DecisionHistory().features(first), before)


def test_capacity_control_uses_no_history_and_keeps_current_features():
    x = np.arange(7)/7
    result = expanded_current(x)
    assert len(result) == 17
    np.testing.assert_array_equal(result[:7], x)
    with pytest.raises(ValueError):
        expanded_current([float('nan')]*7)
