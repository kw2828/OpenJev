"""Untrained DCT compression of additive evidence from public OTTO packets.

The supplied probability kernel is preserved, including numerical tail zeros.
One exact exclusion mask combines initial-prior zeros, visited non-source cells
and all observed zero likelihoods. Before projection, zero entries receive one
of two declared finite extensions. Those entries are always excluded on decode;
their extension can nevertheless affect retained cells at truncated DCT ranks.

No simulator, policy, source coordinates, seed, history or persistent decoded
belief is stored. DCT calls use type II, orthonormal normalization and one worker.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.fft import dctn, idctn

FIELDS = {"position", "hit", "done", "step", "valid_actions"}
EXTENSIONS = {"neutral", "nearest"}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _immutable(array, dtype):
    """An immutable bytes backing prevents callers from re-enabling writes."""
    array = np.asarray(array, dtype=dtype, order="C")
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def _valid_actions(position, size):
    return tuple(action for action in range(4)
                 if 0 <= position[action // 2] + (-1 if action % 2 == 0 else 1) < size)


def _packet(packet, model):
    if isinstance(packet, Mapping):
        values = dict(packet)
    elif isinstance(packet, tuple) and set(getattr(packet, "_fields", ())) == FIELDS:
        values = dict(zip(packet._fields, packet, strict=True))
    else:
        raise TypeError("packet must be a public mapping or named tuple")
    if set(values) != FIELDS:
        raise ValueError("packet must contain exactly the public observation fields")
    raw_position = values["position"]
    if not isinstance(raw_position, (tuple, list)) or len(raw_position) != 2:
        raise ValueError("position must contain two public coordinates")
    position = tuple(_integer(v, "coordinate") for v in raw_position)
    if any(v >= model.N for v in position):
        raise ValueError("position lies outside the grid")
    step = _integer(values["step"], "step")
    if not isinstance(values["done"], (bool, np.bool_)):
        raise TypeError("done must be Boolean")
    done = bool(values["done"])
    hit = values["hit"]
    if isinstance(hit, (bool, np.bool_)) or not isinstance(hit, Integral):
        raise TypeError("hit must be an integer category")
    hit = int(hit)
    if (done and hit != -2) or (not done and not 0 <= hit < model.NHITS):
        raise ValueError("terminal sentinel and hit category disagree")
    actions = values["valid_actions"]
    if not isinstance(actions, (tuple, list)):
        raise TypeError("valid_actions must be a sequence")
    actions = tuple(_integer(a, "action") for a in actions)
    if actions != (() if done else _valid_actions(position, model.N)):
        raise ValueError("valid_actions do not match the public position/boundary")
    return position, hit, done, step


@dataclass(frozen=True, slots=True, init=False)
class SpectralModel:
    """Shared immutable probability kernel and three conditioned log priors."""

    _kernel: np.ndarray
    _log_priors: np.ndarray
    N: int
    NHITS: int
    center: tuple[int, int]

    def __init__(self, kernel):
        kernel = np.asarray(kernel, dtype=np.float64)
        if (kernel.ndim != 3 or kernel.shape[0] != 4 or kernel.shape[1] != kernel.shape[2]
                or kernel.shape[1] % 2 != 1):
            raise ValueError("kernel must have shape (4, 2*N+1, 2*N+1)")
        size = (kernel.shape[1] - 1) // 2
        if size < 3 or size % 2 != 1:
            raise ValueError("N must be odd and at least three")
        if not np.isfinite(kernel).all() or np.any(kernel < 0) or np.any(kernel > 1):
            raise ValueError("kernel probabilities must be finite and within [0,1]")
        totals = kernel.sum(axis=0)
        expected = np.ones_like(totals)
        expected[size, size] = 0
        if np.max(np.abs(totals - expected)) > 1e-10 or np.any(kernel[:, size, size] != 0):
            raise ValueError("kernel must normalize off the excluded origin and be zero at the origin")
        if np.any(kernel[:, size + 1, size] <= 0):
            raise ValueError("nearest extension needs a positive distance-one likelihood in every category")
        object.__setattr__(self, "N", size)
        object.__setattr__(self, "NHITS", 4)
        object.__setattr__(self, "center", (size // 2, size // 2))
        object.__setattr__(self, "_kernel", _immutable(kernel, np.float64))
        priors = []
        for hit in (1, 2, 3):
            likelihood = self.likelihood(self.center, hit)
            support = likelihood > 0
            if not np.any(support):
                raise ValueError("initial hit has no possible source cell")
            log_prior = np.full((size, size), -np.inf, dtype=np.float64)
            log_prior[support] = np.log(likelihood[support])
            maximum = float(np.max(log_prior[support]))
            normalizer = maximum + np.log(np.exp(log_prior[support] - maximum).sum())
            log_prior[support] -= normalizer
            priors.append(log_prior)
        object.__setattr__(self, "_log_priors", _immutable(np.stack(priors), np.float64))

    @property
    def kernel(self):
        return self._kernel

    @property
    def initial_log_priors(self):
        return self._log_priors

    def likelihood(self, position, hit):
        x, y = position
        return self._kernel[hit, self.N - x:2 * self.N - x, self.N - y:2 * self.N - y]

    def start(self, initial_public, q, *, extension="neutral"):
        return SpectralMemory(self, initial_public, q, extension=extension)

    def start_exact(self, initial_public):
        return ExactLogMemory(self, initial_public)

    def storage_bytes(self):
        return {"kernel_array_bytes": self._kernel.nbytes, "initial_log_prior_array_bytes": self._log_priors.nbytes,
                "shared_array_bytes": self._kernel.nbytes + self._log_priors.nbytes,
                "scope": "Immutable array payload bytes; excludes Python object/allocator overhead."}


class _Memory:
    __slots__ = ("_excluded", "_values", "done", "extension", "initial_hit", "model", "position", "q", "step")

    def __init__(self, model, initial_public, q, extension):
        if not isinstance(model, SpectralModel):
            raise TypeError("model must be a shared SpectralModel")
        if extension not in EXTENSIONS:
            raise ValueError("extension must be neutral or nearest")
        q = _integer(q, "q", 1)
        if q > model.N:
            raise ValueError("q cannot exceed N")
        position, hit, done, step = _packet(initial_public, model)
        if position != model.center or hit not in (1, 2, 3) or done or step != 0:
            raise ValueError("initial packet must be a positive-hit nonterminal center observation at step zero")
        self.model, self.q, self.extension = model, q, extension
        self.initial_hit, self.position, self.step, self.done = hit, position, step, done
        self._values = _immutable(np.zeros((q, q)), np.float64)
        self._excluded = _immutable(~np.isfinite(model.initial_log_priors[hit - 1]), np.bool_)

    @property
    def excluded(self):
        return self._excluded

    def update(self, public_packet):
        """Atomically assimilate one completed public move, or mark termination."""
        if self.done:
            raise ValueError("cannot update after termination; start a fresh actor to reset")
        position, hit, done, step = _packet(public_packet, self.model)
        distance = sum(abs(a - b) for a, b in zip(position, self.position, strict=True))
        if step != self.step + 1 or distance > 1 or (distance == 0 and len(_valid_actions(self.position, self.model.N)) == 4):
            raise ValueError("nonconsecutive public step or impossible primitive movement")
        if done:
            if self._excluded[position]:
                raise ValueError("source cannot be found at a previously excluded cell")
            self.position, self.step, self.done = position, step, True
            return
        likelihood = self.model.likelihood(position, hit)
        excluded = self._excluded | (likelihood == 0)
        excluded[position] = True
        if np.all(excluded):
            raise ValueError("public evidence excludes every source cell")
        fill = 0.0 if self.extension == "neutral" else float(np.log(self.model.kernel[hit, self.model.N + 1, self.model.N]))
        increment = np.full((self.model.N, self.model.N), fill, dtype=np.float64)
        np.log(likelihood, out=increment, where=likelihood > 0)
        values = self._values + self._encode(increment)
        if not np.isfinite(values).all():
            raise ValueError("evidence state overflowed finite float64")
        new_values = _immutable(values, np.float64)
        new_excluded = _immutable(excluded, np.bool_)
        self._values, self._excluded = new_values, new_excluded
        self.position, self.step = position, step

    def decode(self):
        """Return temporary (normalized log probabilities, probabilities), no cache."""
        if self.done:
            raise ValueError("no decision belief after termination; start a fresh actor to reset")
        field = self._decode()
        logits = field + self.model.initial_log_priors[self.initial_hit - 1]
        logits[self._excluded] = -np.inf
        support = ~self._excluded
        if not np.any(support) or not np.isfinite(logits[support]).all():
            raise ValueError("invalid supported log evidence")
        maximum = float(np.max(logits[support]))
        shifted = logits[support] - maximum
        normalizer = float(np.log(np.exp(shifted).sum()))
        logs = np.full_like(logits, -np.inf)
        logs[support] = shifted - normalizer
        if not np.isfinite(logs[support]).all():
            raise ValueError("supported log probabilities overflowed")
        probabilities = np.exp(logs)
        if not np.isfinite(probabilities).all() or abs(float(probabilities.sum()) - 1) > 1e-10:
            raise ValueError("decoded probability mass is invalid")
        return logs, probabilities

    def belief(self):
        return self.decode()[1]

    def storage_bytes(self):
        return {"kind": "dct" if isinstance(self, SpectralMemory) else "exact_log", "q": self.q,
                "extension": self.extension, "evidence_array_bytes": self._values.nbytes,
                "support_mask_bytes": self._excluded.nbytes,
                "state_array_bytes": self._values.nbytes + self._excluded.nbytes,
                "shared_array_bytes": self.model.storage_bytes()["shared_array_bytes"],
                "shared_model": self.model.storage_bytes(), "persistent_full_belief": False,
                "persistent_history": False, "persistent_increment_cache": False,
                "workspace_scope": "Temporary N-by-N increment/transform/decoded/log/probability arrays and support masks; not an allocator peak measurement.",
                "float64_grid_bytes": 8 * self.model.N**2,
                "scalar_scope": "Public position, initial hit, step, done, q and extension; Python metadata overhead excluded."}


class SpectralMemory(_Memory):
    __slots__ = ()

    def __init__(self, model, initial_public, q, *, extension="neutral"):
        super().__init__(model, initial_public, q, extension)

    @property
    def coefficients(self):
        return self._values

    def _encode(self, increment):
        return dctn(increment, type=2, norm="ortho", workers=1)[:self.q, :self.q]

    def _decode(self):
        padded = np.zeros((self.model.N, self.model.N), dtype=np.float64)
        padded[:self.q, :self.q] = self._values
        return idctn(padded, type=2, norm="ortho", workers=1)


class ExactLogMemory(_Memory):
    __slots__ = ()

    def __init__(self, model, initial_public):
        super().__init__(model, initial_public, model.N, "neutral")

    @property
    def log_evidence(self):
        return self._values

    def _encode(self, increment):
        return increment

    def _decode(self):
        return self._values.copy()
