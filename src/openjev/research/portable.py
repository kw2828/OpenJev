"""A compact targeting representation without ammo, health or distance levels."""
import numpy as np


def portable_features(current, history=None):
    x = np.asarray(current, dtype=float)
    if x.shape != (7,) or not np.isfinite(x).all():
        raise ValueError('Seven finite current features required')
    # Scale alignment by the same target-width tolerance used by the rule.
    ratio = min(x[2]/max(.04, .65*x[3]), 3.)
    result = np.array([1.,x[1],ratio,x[3]])
    if history is None:
        return result
    h = np.asarray(history, dtype=float)
    if h.shape != (17,) or not np.isfinite(h).all():
        raise ValueError('Seventeen finite causal-history features required')
    return np.r_[result,h[13],h[15],h[14],h[16]]
