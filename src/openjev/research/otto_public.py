"""Deterministic sampled-source adapter and immutable raw public OTTO views.

SourceTracking is supplied by the caller after authentication and normal import.
This module does not import OTTO, Gym or TensorFlow. The three sampling sites
adapt the MIT-licensed auroreloisy/otto-benchmark SourceTracking implementation,
commit a6aaef6507cffd2aff79291c1019f506f616bbef: _initial_hit (419),
_draw_a_source (413), and sampled-source _execute_action (201). Likelihood,
movement, found/nonfound and posterior-update formulas remain upstream-owned.

The integration replaces unseeded generators with three local PCG64 streams.
It preserves categorical distributions, not bitwise trajectories of the original
unseeded implementation. Initial/source draws are separate from subsequent hits;
identical episode seeds pair initialization across policies, not observations at
different positions. New instances replay a seed; restart advances its streams.
The simulator and draw log are evaluator-owned, never raw-actor inputs.
"""
from __future__ import annotations

import copy
from numbers import Integral, Real
from typing import NamedTuple

import numpy as np

CHANNELS = {"initial": 0, "source": 1, "hit": 2}
DEFAULT_CONFIG = {"Ndim": 2, "lambda_over_dx": 1.0, "R_dt": 1.0,
                  "Ngrid": None, "Nhits": None, "draw_source": True,
                  "norm_Poisson": "Euclidean"}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _config(config):
    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dictionary or None")
    values = {**DEFAULT_CONFIG, **(config or {})}
    if set(values) != set(DEFAULT_CONFIG) or values["draw_source"] is not True:
        raise ValueError("only sampled-source public configuration keys are supported")
    values["Ndim"] = _integer(values["Ndim"], "Ndim", 1)
    for name, minimum in (("Ngrid", 3), ("Nhits", 2)):
        if values[name] is not None:
            values[name] = _integer(values[name], name, minimum)
    for name, minimum in (("lambda_over_dx", 1), ("R_dt", 0)):
        value = values[name]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < minimum or name == "R_dt" and value == 0):
            raise ValueError(f"invalid public parameter: {name}")
    if values["norm_Poisson"] not in ("Euclidean", "Manhattan", "Chebyshev"):
        raise ValueError("unsupported public detection norm")
    return values


def seeded_environment(SourceTracking, seed, config=None, *, initial_hit=None):
    """Build an evaluator-owned subclass without changing upstream or global RNGs.

    Normal import and source authentication are the caller's responsibility.
    Forced-hit calls retain the upstream API and consume no hit RNG draw. The
    ordinary sampled route always delegates an explicit hit, including zero on
    the found branch (which upstream replaces with its terminal sentinel -2).
    An explicit positive initial_hit conditions initialization without an initial
    RNG draw. Its range is checked against the resolved upstream Nhits before
    restart builds the conditional prior or draws a source.
    Like SourceTracking itself, this object is not a post-terminal step guard;
    the episode controller must stop on done and implement its declared timeout.
    """
    seed = _integer(seed, "seed")
    values = _config(config)
    if initial_hit is not None:
        initial_hit = _integer(initial_hit, "initial_hit", 1)

    class SeededSourceTracking(SourceTracking):
        def __init__(self):
            self._public_rngs = {name: np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([seed, channel]))) for name, channel in CHANNELS.items()}
            self._public_draw_counts = dict.fromkeys(CHANNELS, 0)
            self._public_draw_log = []
            super().__init__(**values, initial_hit=initial_hit)

        def restart(self, initial_hit=None):
            if initial_hit is not None:
                initial_hit = _integer(initial_hit, "initial_hit", 1)
                if initial_hit >= self.Nhits:
                    raise ValueError("initial_hit must be less than the resolved Nhits")
            return super().restart(initial_hit=initial_hit)

        @property
        def draw_log(self):
            """Copied evaluator evidence. Probability vectors reveal hidden state."""
            return tuple(copy.deepcopy(self._public_draw_log))

        def _draw(self, channel, probabilities):
            probability = np.asarray(probabilities, dtype=np.float64)
            if (probability.ndim != 1 or not len(probability) or not np.isfinite(probability).all()
                    or np.any(probability < 0) or abs(float(probability.sum()) - 1.0) > 1e-10):
                raise ValueError("categorical probabilities must be finite, nonnegative and normalized")
            cumulative = np.cumsum(probability)
            mass = float(cumulative[-1])
            if mass <= 0:
                raise ValueError("categorical mass must be positive")
            # Normalizing the cumulative sum matches categorical choice's handling
            # of summation roundoff. No probability floor or support repair.
            cumulative /= mass
            uniform = float(self._public_rngs[channel].random())
            index = int(np.searchsorted(cumulative, uniform, side="right"))
            record = {"channel": channel, "index": self._public_draw_counts[channel],
                      "probabilities": probability.tolist(), "cdf_mass": mass,
                      "uniform": uniform, "selected_index": index}
            self._public_draw_log.append(record)
            self._public_draw_counts[channel] += 1
            return index

        def _initial_hit(self, hit=None):
            if hit is not None:
                return hit
            probability = np.zeros(self.Nhits)
            radius = np.arange(1, int(1000 * self.lambda_over_dx))
            shell_volume = self._volume_ball(radius + 0.5) - self._volume_ball(radius - 0.5)
            for h in range(1, self.Nhits):
                p = self._Poisson(self._mean_number_of_hits(radius), h)
                probability[h] = max(0, np.sum(p * shell_volume))
            probability /= np.sum(probability)
            return self._draw("initial", probability)

        def _draw_a_source(self):
            index = self._draw("source", self.p_source.flatten())
            self.source = np.array(np.unravel_index(index, shape=([self.N] * self.Ndim)), dtype=int)

        def _execute_action(self, action, hit=None, quiet=False):
            action = _integer(action, "action")
            if action >= self.Nactions or self.draw_source is not True:
                raise ValueError("invalid action or changed sampled-source mode")
            if hit is None:
                moved, _ = self._move(action, self.agent)
                norm = {"Manhattan": 1, "Euclidean": 2, "Chebyshev": np.inf}[self.norm_Poisson]
                distance = np.linalg.norm(np.asarray(moved) - np.asarray(self.source), ord=norm)
                if distance > 1e-10:  # exact pinned upstream EPSILON
                    mu = self._mean_number_of_hits(distance)
                    probability = np.zeros(self.Nhits)
                    total = 0
                    for h in range(self.Nhits - 1):
                        probability[h] = self._Poisson(mu, h)
                        total += self._Poisson(mu, h)
                    probability[self.Nhits - 1] = np.maximum(0, 1.0 - total)
                    hit = self._draw("hit", probability)
                else:
                    hit = 0  # no odor draw when found; upstream returns hit=-2
            return super()._execute_action(action, hit=hit, quiet=quiet)

    return SeededSourceTracking()


class PublicObservation(NamedTuple):
    position: tuple[int, ...]
    hit: int
    done: bool
    step: int
    valid_actions: tuple[int, ...]


def observation(env, step) -> PublicObservation:
    """Copy only current public coordinates/hit/boundary data; never expose env."""
    step = _integer(step, "step")
    position = tuple(_integer(value, "coordinate") for value in env.agent)
    if len(position) != env.Ndim or any(value >= env.N for value in position):
        raise ValueError("invalid public position")
    done = env.obs["done"]
    if not isinstance(done, (bool, np.bool_)):
        raise TypeError("public done flag must be boolean")
    raw_hit = env.obs["hit"]
    if isinstance(raw_hit, (bool, np.bool_)) or not isinstance(raw_hit, Integral):
        raise TypeError("public hit must be integer")
    hit = int(raw_hit)
    if (done and hit != -2) or (not done and not 0 <= hit < env.Nhits):
        raise ValueError("public hit disagrees with sampled-source termination")
    valid = () if done else tuple(action for action in range(env.Nactions) if env._move(action, list(position))[1])
    return PublicObservation(position, hit, bool(done), step, valid)
