"""Public-only bookkeeping for pinned POPGym ConcentrationHard.

Native source: proroklab/popgym@410d5aa626dae8024f498354d8781a0d1870c399.
There is no environment, deck, model or generator construction here. The
tracker stores no historical ranks; its current observation is public, including
the native mismatch frame captured before the pair is cleared. PublicTeacher is
separate training/evaluation label storage, never a learned-model input.
"""

import math
from dataclasses import dataclass

import numpy as np

POSITIONS = 52
RANKS = 13
HIDDEN = 13
MAX_ACTIONS = 104
REWARD_ATOL = 1e-8  # Also accepts an otherwise exact native reward saved as f32.


def _observation(value):
    if not isinstance(value, np.ndarray) or value.shape != (POSITIONS,):
        raise ValueError('observation must be a NumPy array with shape (52,)')
    if value.dtype != np.int64 or np.any((value < 0) | (value > HIDDEN)):
        raise ValueError('observation must contain int64 public ranks 0..13')
    return value.copy()


def _copy(value):
    result = value.copy()
    result.flags.writeable = False
    return result


def _initial(value):
    result = _observation(value)
    if np.any(result != HIDDEN):
        raise ValueError('native reset must hide every position')
    return result


def _index(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError('action must be an integer position')
    if not 0 <= int(value) < POSITIONS:
        raise ValueError('action is outside 0..51')
    return int(value)


@dataclass(frozen=True)
class Reveal:
    """Exactly one actual-action model-write event; no historical rank cache."""

    pos: int
    rank: int


class PublicCardTracker:
    """Shared picker metadata, including external seen/matched memory aids.

    Public arrays are detached read-only copies, not deeply immutable storage.
    ``observe`` admits all native actions, including wasteful matched/repeated
    selections. ``choose`` restricts all models to unmatched, nonpending cards.
    No model query or choice changes this object's actual-action state.
    """

    __slots__ = ('_matched', '_obs', '_pending', '_seen', '_step')

    def __init__(self):
        self._obs = None

    def _ready(self):
        if self._obs is None:
            raise ValueError('reset is required')

    def reset(self, obs):
        initial = _initial(obs)
        self._obs = initial
        self._seen = np.zeros(POSITIONS, dtype=bool)
        self._matched = np.zeros(POSITIONS, dtype=bool)
        self._pending = None
        self._step = 0

    @property
    def observation(self):
        self._ready()
        return _copy(self._obs)

    @property
    def seen(self):
        self._ready()
        return _copy(self._seen)

    @property
    def matched(self):
        self._ready()
        return _copy(self._matched)

    @property
    def pending(self):
        self._ready()
        return self._pending

    @property
    def step(self):
        self._ready()
        return self._step

    @property
    def terminal(self):
        self._ready()
        return self._step == MAX_ACTIONS or bool(np.all(self._matched))

    def probabilities(self, model_probs):
        """Override public ranks and unseen positions identically for all arms."""
        self._ready()
        if (not isinstance(model_probs, np.ndarray)
                or model_probs.shape != (POSITIONS, RANKS)
                or model_probs.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or not np.all(np.isfinite(model_probs))
                or np.any((model_probs < 0) | (model_probs > 1))
                or not np.allclose(model_probs.sum(axis=1), 1, rtol=0, atol=1e-5)):
            raise ValueError('model probabilities must be finite normalized float32/64 [52,13]')
        result = model_probs.astype(np.float64, copy=True)
        result[~self._seen] = 1 / RANKS
        visible = self._obs != HIDDEN
        result[visible] = 0
        result[np.flatnonzero(visible), self._obs[visible]] = 1
        return _copy(result)

    def choose(self, probs, rng=None, random_chance=0):
        """Lowest-index ties; caller RNG only for a declared random mixture.

        With probability ``random_chance`` choose uniformly among legal cards.
        A positive mixture consumes one random() draw then, if selected, one
        integers(legal_count) draw. A zero mixture consumes no random draws.
        """
        self._ready()
        if self.terminal:
            raise ValueError('no decision is legal after native termination/budget')
        if (isinstance(random_chance, (bool, np.bool_))
                or not isinstance(random_chance, (int, float, np.integer, np.floating))
                or not math.isfinite(float(random_chance)) or not 0 <= random_chance <= 1):
            raise ValueError('random_chance must be finite in [0,1]')
        distribution = self.probabilities(probs)
        legal = np.flatnonzero(~self._matched)
        if self._pending is not None:
            legal = legal[legal != self._pending]
        if random_chance > 0:
            if rng is None or not callable(getattr(rng, 'random', None)) or not callable(getattr(rng, 'integers', None)):
                raise ValueError('positive mixture requires an explicit compatible generator')
            if rng.random() < random_chance:
                return int(legal[int(rng.integers(len(legal)))])
        if self._pending is not None:
            return int(legal[np.argmax(distribution[legal, self._obs[self._pending]])])
        # Float64 computed-score ties only, with no epsilon or rank rounding.
        # Upper-triangle enumeration is lexicographic, so argmax keeps first ties.
        left, right = np.triu_indices(len(legal), k=1)
        scores = np.sum(distribution[legal[left]] * distribution[legal[right]], axis=1)
        return int(legal[left[int(np.argmax(scores))]])

    def observe(self, action, reward, after):
        """Accept one actual native transition atomically and return one reveal."""
        self._ready()
        if self.terminal:
            raise ValueError('transition after native termination/budget')
        action = _index(action)
        after = _observation(after)
        visible = self._matched.copy()
        visible[action] = True
        if self._pending is not None:
            visible[self._pending] = True
        if not np.array_equal(after != HIDDEN, visible):
            raise ValueError('returned frame must show matched cards and actual in-play positions')
        preserved = self._matched.copy()
        if self._pending is not None:
            preserved[self._pending] = True
        if not np.array_equal(after[preserved], self._obs[preserved]):
            raise ValueError('a still-visible public card changed rank')
        matched = self._matched.copy()
        pending = None
        if self._matched[action]:
            expected = -(1 if self._pending is None else 2) / MAX_ACTIONS
        elif self._pending is None:
            expected = 0.0
            pending = action
        elif action != self._pending and after[action] == after[self._pending]:
            expected = 2 / POSITIONS
            matched[[self._pending, action]] = True
        else:
            expected = -2 / MAX_ACTIONS
        if (isinstance(reward, (bool, np.bool_))
                or not isinstance(reward, (int, float, np.integer, np.floating))
                or not math.isfinite(float(reward))
                or not math.isclose(float(reward), expected, rel_tol=0, abs_tol=REWARD_ATOL)):
            raise ValueError('reward disagrees with the public native pair/phase')
        self._obs = after
        self._seen |= visible
        self._matched, self._pending = matched, pending
        self._step += 1
        return Reveal(action, int(after[action]))


class PublicTeacher:
    """Public-history labels and exact-map reference, isolated from the tracker."""

    def __init__(self):
        self._obs = None

    def _ready(self):
        if self._obs is None:
            raise ValueError('reset is required')

    def reset(self, obs):
        self._obs = _initial(obs)
        self._values = np.full(POSITIONS, -1, dtype=np.int64)
        self._last_visible = np.full(POSITIONS, -1, dtype=np.int32)
        self._step = 0

    def observe(self, after):
        self._ready()
        if self._step >= MAX_ACTIONS:
            raise ValueError('teacher exceeded native action budget')
        after = _observation(after)
        visible = after != HIDDEN
        known = visible & (self._values != -1)
        if np.any(self._values[known] != after[known]):
            raise ValueError('previously public card changed rank')
        self._step += 1
        self._values[visible] = after[visible]
        self._last_visible[visible] = self._step
        self._obs = after

    def targets(self, current_obs):
        """Before-action labels; ages since last public visibility, not write age."""
        self._ready()
        current = _observation(current_obs)
        if not np.array_equal(current, self._obs):
            raise ValueError('targets require the latest actually observed public frame')
        mask = (self._values != -1) & (current == HIDDEN)
        values = np.where(mask, self._values, -1).astype(np.int64)
        ages = np.where(mask, self._step - self._last_visible, -1).astype(np.int32)
        return _copy(values), _copy(mask), _copy(ages)

    def probabilities(self):
        self._ready()
        result = np.full((POSITIONS, RANKS), 1 / RANKS, dtype=np.float64)
        known = self._values != -1
        result[known] = 0
        result[np.flatnonzero(known), self._values[known]] = 1
        return _copy(result)
