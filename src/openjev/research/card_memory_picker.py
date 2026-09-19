"""Three prospective public-only pickers over unchanged card-memory inputs.

A calls the frozen original chooser. B normalizes raw rows, then uses an
inclusive absolute 1e-12 score tie band. C uses B's scores/ties and prefers the
smallest unseen endpoint of a tied first-card pair. These are policy changes,
not probability calibration or extra memory updates. No RNG is constructed.
"""

import math

import numpy as np

from openjev.research.card_memory_task import PublicCardTracker

POLICIES = ('A', 'B', 'C')
TIE_ATOL = 1e-12


def _raw_rows(raw):
    if (not isinstance(raw, np.ndarray) or raw.shape != (52, 13)
            or raw.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(raw).all() or np.any(raw < 0)):
        raise ValueError('Expected finite nonnegative float32/64 [52,13] raw probabilities')
    copied = raw.astype(np.float64, copy=True)
    with np.errstate(over='ignore'):
        mass = copied.sum(axis=1, dtype=np.float64)
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError('Raw rows require finite strictly positive float64 mass')
    return copied, mass


def _tied_indices(scores, tolerance):
    """Freeze the direct float64 max-minus-score comparison, without rounding."""
    return np.flatnonzero(np.max(scores) - scores <= tolerance)


class CardPolicyTracker(PublicCardTracker):
    """Inherited actual-action metadata; every decision is a read-only query.

    Diagnostic tie_size counts unordered pairs in phase one and legal positions
    in phase two. unseen_tied_endpoints is the sorted unique first-phase list,
    empty in phase two. A diagnoses exact computed-score ties. Baction_ifC is
    B's action at this same public prefix for C, otherwise None. rowmass_max_abs
    is raw float64 row-sum deviation from one, before normalization/overrides.
    Returned arrays own their storage and are marked read-only; no deep tensor
    immutability or calibrated probabilities are implied.

    Instrumentation performs zero model calls and no state writes. A computes
    scores once for diagnostics and again inside the unchanged original chooser;
    B/C compute scores once. Whole-controller timing must charge this additional
    A choice pass. These instrumented timings do not establish a picker speedup
    under matched arithmetic, even when model forward/write counts are equal.
    """

    __slots__ = ('_policy',)

    def __init__(self, policy):
        if policy not in POLICIES or not isinstance(policy, str):
            raise ValueError('policy must be A, B or C')
        super().__init__()
        self._policy = policy

    @property
    def policy(self):
        return self._policy

    def probabilities(self, raw):
        if self._policy == 'A':
            return super().probabilities(raw)
        self._ready()
        values, mass = _raw_rows(raw)
        values /= mass[:, None]
        # Keep the original visibility/unseen rules and their ordering exactly.
        return super().probabilities(values)

    def decision(self, raw):
        self._ready()
        if self.terminal:
            raise ValueError('no decision is legal after native termination/budget')
        probabilities = self.probabilities(raw)
        _, mass = _raw_rows(raw)
        legal = np.flatnonzero(~self._matched)
        first = self._pending is None
        tolerance = 0.0 if self._policy == 'A' else TIE_ATOL
        if first:
            left, right = np.triu_indices(len(legal), k=1)
            scores = np.sum(probabilities[legal[left]] * probabilities[legal[right]], axis=1)
            tied = _tied_indices(scores, tolerance)
            baseline = int(legal[left[tied[0]]])
            endpoints = np.unique(np.concatenate((legal[left[tied]], legal[right[tied]])))
            unseen = endpoints[~self._seen[endpoints]].tolist()
        else:
            legal = legal[legal != self._pending]
            scores = probabilities[legal, self._obs[self._pending]]
            tied = _tied_indices(scores, tolerance)
            baseline = int(legal[tied[0]])
            unseen = []
        if self._policy == 'A':
            # Do not reimplement A's decision, even though diagnostics recompute scores.
            action = super().choose(raw)
        elif self._policy == 'C' and first and unseen:
            action = int(unseen[0])
        else:
            action = baseline
        diagnostics = {'tie_size': len(tied), 'firstphase': first,
                       'unseen_tied_endpoints': unseen, 'selected_unseen': bool(not self._seen[action]),
                       'Baction_ifC': baseline if self._policy == 'C' else None,
                       'rowmass_max_abs': float(np.max(np.abs(mass - 1)))}
        return action, probabilities, diagnostics

    def choose(self, raw, rng=None, random_chance=0):
        if (rng is not None or isinstance(random_chance, (bool, np.bool_))
                or not isinstance(random_chance, (int, float, np.integer, np.floating))
                or not math.isfinite(float(random_chance)) or random_chance != 0):
            raise ValueError('CardPolicyTracker decisions are deterministic; RNG/mixing is forbidden')
        return self.decision(raw)[0]
