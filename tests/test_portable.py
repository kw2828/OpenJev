import numpy as np
import pytest

from openjev.research.memory import DecisionHistory
from openjev.research.portable import portable_features


def test_portable_head_is_invariant_to_removed_state_levels():
    x = np.array([1.,1.,.1,.1,.5,1.,.7])
    history = DecisionHistory().features(x)
    expected = portable_features(x,history)
    shifted = x.copy()
    shifted[4:] = [2.,.1,2.]
    shifted_history = history.copy()
    shifted_history[[4,5,6,10,11,12]] = 9.
    np.testing.assert_array_equal(portable_features(shifted,shifted_history),expected)
    assert expected.shape == (8,)
    assert portable_features(x).shape == (4,)
    history[15] = 1.
    assert portable_features(x,history)[5] == 1.
    with pytest.raises(ValueError):
        portable_features([float('nan')]*7)
