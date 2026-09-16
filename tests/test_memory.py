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


def test_saved_training_uses_identical_windows_and_current_features():
    import importlib.util
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('memory_doom',root/'research/memory_doom.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    protocol=json.loads((root/'research/protocols/memory-doom-v1.json').read_text())
    xs,ys,units=runner.training_data(protocol)
    assert [len(y) for y in ys]==[565,609,631,628,522]
    for rep in range(5):
        assert len(set(units[rep]))==16
        np.testing.assert_array_equal(np.array(xs['history'][rep])[:,:7],xs['current'][rep])
        np.testing.assert_array_equal(np.array(xs['expanded'][rep])[:,:7],xs['current'][rep])
