"""Causal, resettable history features for the frozen memory ablation."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DecisionHistory:
    previous: np.ndarray = field(default_factory=lambda: np.zeros(6))
    previous_fire: float = 0.
    previous_hit: float = 0.
    age: int = 10
    has_previous: float = 0.

    def features(self, current):
        current = np.asarray(current, dtype=float)
        if current.shape != (7,) or not np.isfinite(current).all():
            raise ValueError('Expected seven finite current-state features')
        return np.r_[current, self.previous, self.previous_fire,
                     min(self.age, 10)/10, self.previous_hit, self.has_previous]

    def update(self, current, issued_fire, hit):
        # Call only after the action outcome has arrived. No current outcome in features.
        self.previous = np.asarray(current, dtype=float)[1:].copy()
        self.previous_fire = float(issued_fire)
        self.previous_hit = float(hit > 0)
        self.age = 1 if issued_fire else min(self.age+1, 10)
        self.has_previous = 1.


def expanded_current(current):
    """17 dimensions, like history, but only deterministic current-observation terms."""
    x = np.asarray(current, dtype=float)
    if x.shape != (7,) or not np.isfinite(x).all():
        raise ValueError('Expected seven finite current-state features')
    return np.r_[x, x[1:]**2, x[1]*x[2], x[1]*x[3], x[2]*x[3], x[5]*x[6]]


def select_features(kind, current, history):
    if kind == 'current':
        return np.asarray(current, dtype=float)
    if kind == 'expanded':
        return expanded_current(current)
    if kind == 'history':
        return history.features(current)
    raise ValueError('Unknown feature kind')
