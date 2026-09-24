"""Compact raw observations and causal Gaussian linear-measurement memories.

Hybrid memories store sorted raw labels through event 118. At event 119 they
convert every currently held observation, including the new one, into fixed
linear sums; later writes add each new observation exactly once. Inference
conditions on those sums, never on an additional historical posterior.

The supplied grid, kernel and basis rules are fixed public structure. Bases,
factors and measurement matrices are reconstructed in temporary workspace and
never cached. Logical storage excludes Python object overhead and workspace.
SVD removes only directions below the declared machine-precision rank cutoff;
this numerical rank choice is reported, not an exact full-history guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import ndtr

VERSION = "measurement-memory-v1"
KINDS = ("spectral", "dct", "bins")
SIDE, POINTS, CAPACITY, RECENT_CAPACITY = 17, 289, 118, 98
LENGTH, AMPLITUDE, NUGGET, NOISE_VARIANCE = 1., 1., 1e-5, .09


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name, dtype=np.float64):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype)
             and value.shape == shape and np.isfinite(value).all(), "finite declared array: " + name)


def _owned(value):
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def _metadata(state):
    _require(type(state.step) is int and 0 <= state.step <= POINTS, "event count outside grid domain")
    values = (state.length, state.amplitude, state.nugget, state.noise_variance)
    _require(all(type(value) is float for value in values)
             and values == (LENGTH, AMPLITUDE, NUGGET, NOISE_VARIANCE), "fixed four kernel/noise scalars")


def _mask_bits(mask):
    return np.unpackbits(mask, axis=1, count=POINTS, bitorder="little")


def _mask_state(state, hybrid):
    _metadata(state)
    _require(isinstance(state.mask, np.ndarray) and state.mask.ndim == 2
             and state.mask.shape[0] > 0, "nonempty mask batch")
    batch = state.mask.shape[0]
    _array(state.mask, (batch, 37), "occupancy mask", np.uint8)
    _array(state.values, (batch, CAPACITY), "label/sum buffer")
    _require(np.all(state.mask[:, -1] <= 1), "unused mask bits must be zero")
    count = state.step if hybrid else min(state.step, CAPACITY)
    _require(np.all(_mask_bits(state.mask).sum(1) == count), "mask population matches state format")
    if state.step <= CAPACITY:
        _require(not np.any(state.values[:, count:]), "unused raw label slots must be zero")
    for name in ("mask", "values"):
        object.__setattr__(state, name, _owned(getattr(state, name)))


@dataclass(frozen=True, eq=False, slots=True)
class HybridMemory:
    mask: np.ndarray
    values: np.ndarray
    kind: str = "spectral"
    step: int = 0
    length: float = LENGTH
    amplitude: float = AMPLITUDE
    nugget: float = NUGGET
    noise_variance: float = NOISE_VARIANCE

    def __post_init__(self):
        _require(type(self.kind) is str and self.kind in KINDS, "unknown immutable basis kind")
        _mask_state(self, True)

    @property
    def array_bytes(self):
        return int(self.mask.nbytes + self.values.nbytes)

    @property
    def resident_bytes_per_context(self):
        # Four f64 hyperparameters + one i64 counter + one encoded kind byte.
        return self.array_bytes // self.mask.shape[0] + 41


@dataclass(frozen=True, eq=False, slots=True)
class PackedMemory:
    mask: np.ndarray
    values: np.ndarray
    step: int = 0
    length: float = LENGTH
    amplitude: float = AMPLITUDE
    nugget: float = NUGGET
    noise_variance: float = NOISE_VARIANCE

    def __post_init__(self):
        _mask_state(self, False)

    @property
    def array_bytes(self):
        return int(self.mask.nbytes + self.values.nbytes)

    @property
    def resident_bytes_per_context(self):
        return self.array_bytes // self.mask.shape[0] + 40


@dataclass(frozen=True, eq=False, slots=True)
class RecentMemory:
    ids: np.ndarray
    values: np.ndarray
    step: int = 0
    length: float = LENGTH
    amplitude: float = AMPLITUDE
    nugget: float = NUGGET
    noise_variance: float = NOISE_VARIANCE

    def __post_init__(self):
        _metadata(self)
        _require(isinstance(self.ids, np.ndarray) and self.ids.ndim == 2
                 and self.ids.shape[0] > 0, "nonempty recent batch")
        batch = self.ids.shape[0]
        _array(self.ids, (batch, RECENT_CAPACITY), "recent IDs", np.uint16)
        _array(self.values, (batch, RECENT_CAPACITY), "recent labels")
        used = min(self.step, RECENT_CAPACITY)
        _require((self.ids[:, :used] < POINTS).all() and not np.any(self.ids[:, used:])
                 and not np.any(self.values[:, used:]), "recent ID domain and zero padding")
        _require(all(len(np.unique(row[:used])) == used for row in self.ids), "unique retained recent IDs")
        for name in ("ids", "values"):
            object.__setattr__(self, name, _owned(getattr(self, name)))

    @property
    def array_bytes(self):
        return int(self.ids.nbytes + self.values.nbytes)

    @property
    def resident_bytes_per_context(self):
        return self.array_bytes // self.ids.shape[0] + 40


def _batch(batch):
    _require(type(batch) is int and batch > 0, "positive integer batch required")


def initial_hybrid(batch, kind="spectral"):
    _batch(batch)
    return HybridMemory(np.zeros((batch, 37), np.uint8), np.zeros((batch, CAPACITY)), kind)


def initial_packed(batch):
    _batch(batch)
    return PackedMemory(np.zeros((batch, 37), np.uint8), np.zeros((batch, CAPACITY)))


def initial_recent(batch):
    _batch(batch)
    return RecentMemory(np.zeros((batch, RECENT_CAPACITY), np.uint16), np.zeros((batch, RECENT_CAPACITY)))


def grid_points():
    ids = np.arange(POINTS)
    return np.stack((ids // SIDE, ids % SIDE), axis=-1).astype(np.float64) / 4 - 2


def basis(kind):
    """Regenerate an owned 118-by-289 basis without persistent numeric caches.

    Spectral modes are products of descending one-dimensional RBF eigensystems,
    sorted by descending eigenvalue product then mode indices. Each 1D vector
    has its first maximum-absolute entry positive. DCT-II modes are sorted by
    squared frequency then indices. Bins have unit weights and assign grid ID
    g to floor(118*g/289); these are consecutive-ID bins, not square cells.
    """
    _require(type(kind) is str and kind in KINDS, "unknown basis kind")
    if kind == "bins":
        result = np.zeros((CAPACITY, POINTS))
        result[np.arange(POINTS) * CAPACITY // POINTS, np.arange(POINTS)] = 1.
    else:
        if kind == "spectral":
            axis = np.arange(SIDE, dtype=np.float64) / 4 - 2
            eigenvalues, vectors = np.linalg.eigh(np.exp(-.5 * (axis[:, None] - axis[None]) ** 2))
            order = np.argsort(-eigenvalues, kind="stable")
            eigenvalues, vectors = eigenvalues[order], vectors[:, order]
            for column in range(SIDE):
                if vectors[np.argmax(np.abs(vectors[:, column])), column] < 0:
                    vectors[:, column] *= -1
            pairs = sorted(((i, j) for i in range(SIDE) for j in range(SIDE)),
                           key=lambda pair: (-eigenvalues[pair[0]] * eigenvalues[pair[1]], *pair))
        else:
            indices = np.arange(SIDE, dtype=np.float64)
            vectors = np.cos(np.pi * (indices[:, None] + .5) * indices[None] / SIDE) * np.sqrt(2 / SIDE)
            vectors[:, 0] /= np.sqrt(2)
            pairs = sorted(((i, j) for i in range(SIDE) for j in range(SIDE)),
                           key=lambda pair: (pair[0] ** 2 + pair[1] ** 2, *pair))
        result = np.stack([np.outer(vectors[:, i], vectors[:, j]).ravel() for i, j in pairs[:CAPACITY]])
    _require(np.isfinite(result).all(), "finite deterministic basis")
    return _owned(result)


def _ids(x):
    _require(np.isfinite(x).all() and np.all((x >= -2) & (x <= 2)), "coordinates inside fixed grid")
    coordinates = (x + 2) * 4
    _require(np.array_equal(coordinates, np.rint(coordinates)), "coordinates must be exactly on grid")
    integer = coordinates.astype(np.int64)
    return integer[..., 0] * SIDE + integer[..., 1]


def _event(state, x, y, batch):
    _require(state.step < POINTS, "finite unique stream exhausted")
    _array(x, (batch, 2), "public event coordinates")
    _array(y, (batch,), "public event labels")
    return _ids(x)


def _extend_raw(mask, values, ids, labels):
    old_bits = _mask_bits(mask)
    _require(not np.any(old_bits[np.arange(len(ids)), ids]), "repeated retained/observed coordinate")
    new_bits = old_bits.copy()
    new_bits[np.arange(len(ids)), ids] = 1
    coordinates, targets = [], []
    for b, index in enumerate(ids):
        old = np.flatnonzero(old_bits[b])
        position = np.searchsorted(old, index)
        coordinates.append(np.insert(old, position, index))
        targets.append(np.insert(values[b, :len(old)], position, labels[b]))
    return new_bits, coordinates, targets


def write_hybrid(state, x, y):
    """One causal update. Basis conversion is representation-only, not replay."""
    _require(type(state) is HybridMemory, "HybridMemory required")
    ids = _event(state, x, y, state.mask.shape[0])
    if state.step < CAPACITY:
        bits, _, targets = _extend_raw(state.mask, state.values, ids, y)
        values = np.zeros_like(state.values)
        values[:, :state.step + 1] = np.stack(targets)
    elif state.step == CAPACITY:
        bits, coordinates, targets = _extend_raw(state.mask, state.values, ids, y)
        phi = basis(state.kind)
        values = np.stack([phi[:, index] @ labels for index, labels in zip(coordinates, targets, strict=True)])
    else:
        bits = _mask_bits(state.mask)
        _require(not np.any(bits[np.arange(len(ids)), ids]), "repeated observed coordinate")
        bits[np.arange(len(ids)), ids] = 1
        values = state.values + basis(state.kind)[:, ids].T * y[:, None]
    _require(np.isfinite(values).all(), "finite one-event accumulation")
    return HybridMemory(np.packbits(bits, axis=1, bitorder="little"), values, state.kind, state.step + 1)


def write_packed(state, x, y):
    """Coordinate-only coverage retention; distance ties delete lowest grid ID.

    The mask represents retained locations. Detection of repeats after a
    location was discarded belongs to the upstream unique-stream contract.
    """
    _require(type(state) is PackedMemory, "PackedMemory required")
    ids = _event(state, x, y, state.mask.shape[0])
    bits, coordinates, targets = _extend_raw(state.mask, state.values, ids, y)
    values = np.zeros_like(state.values)
    grid = grid_points()
    for b, (indices, labels) in enumerate(zip(coordinates, targets, strict=True)):
        if len(indices) > CAPACITY:
            points = grid[indices]
            distances = np.sum((points[:, None] - points[None]) ** 2, axis=-1)
            np.fill_diagonal(distances, np.inf)
            drop = int(np.argmin(distances.min(axis=1)))
            bits[b, indices[drop]] = 0
            labels = np.delete(labels, drop)
        values[b, :len(labels)] = labels
    return PackedMemory(np.packbits(bits, axis=1, bitorder="little"), values, state.step + 1)


def _write_ordered(state, x, y, coverage):
    _require(type(state) is RecentMemory, "RecentMemory required")
    ids = _event(state, x, y, state.ids.shape[0])
    used = min(state.step, RECENT_CAPACITY)
    _require(not np.any(state.ids[:, :used] == ids[:, None]), "repeated retained recent coordinate")
    if used < RECENT_CAPACITY:
        order, values = state.ids.copy(), state.values.copy()
        order[:, used], values[:, used] = ids, y
    else:
        order = np.concatenate((state.ids, ids[:, None]), axis=1).astype(np.uint16)
        values = np.concatenate((state.values, y[:, None]), axis=1)
        if coverage:
            coordinates = grid_points()[order]
            distances = np.sum((coordinates[:, :, None] - coordinates[:, None]) ** 2, axis=-1)
            distances[:, np.arange(RECENT_CAPACITY + 1), np.arange(RECENT_CAPACITY + 1)] = np.inf
            drops = np.argmin(distances.min(axis=-1), axis=-1)
        else:
            drops = np.zeros(len(ids), dtype=np.int64)
        keep = np.arange(RECENT_CAPACITY + 1)[None] != drops[:, None]
        order, values = order[keep].reshape(-1, RECENT_CAPACITY), values[keep].reshape(-1, RECENT_CAPACITY)
    return RecentMemory(order, values, state.step + 1)


def write_recent(state, x, y):
    """Ordered packed FIFO control with 98 coordinate IDs and labels."""
    return _write_ordered(state, x, y, False)


def write_coverage98(state, x, y):
    """Ordered packed coverage control; nearest-distance ties delete oldest."""
    return _write_ordered(state, x, y, True)


def _kernel(x, z):
    delta = x[..., :, None, :] - z[..., None, :, :]
    return np.exp(-.5 * np.sum(delta * delta, axis=-1)) + NUGGET * np.all(delta == 0, axis=-1)


def _paths(state, paths, batch):
    _require(state.step > 0, "nonempty state required for prediction")
    _require(isinstance(paths, np.ndarray) and paths.ndim == 5 and paths.shape[1] > 0,
             "paths must be [B,Q,4,4,2]")
    _array(paths, (batch, paths.shape[1], 4, 4, 2), "public future paths")
    _ids(paths)


def _predict_one(ids, values, paths, phi=None):
    coordinates = grid_points()[ids]
    count = len(ids)
    if phi is None:
        transform, measurements, rank, cutoff = np.eye(count), values, count, 0.
    else:
        operator = phi[:, ids]
        left, singular, right = np.linalg.svd(operator, full_matrices=False)
        cutoff = float(np.finfo(np.float64).eps * max(operator.shape) * singular[0])
        selected = singular > cutoff
        rank = int(np.sum(selected))
        _require(rank > 0, "positive observed measurement rank")
        transform = right[selected]
        measurements = (left[:, selected].T @ values) / singular[selected]
    covariance = transform @ (_kernel(coordinates, coordinates) + NOISE_VARIANCE * np.eye(count)) @ transform.T
    factor = np.linalg.cholesky(covariance)
    # Average kernel rows before conditioning, preserving complete path covariance.
    flattened = paths.reshape(-1, 4, 2)
    cross = _kernel(flattened, coordinates).mean(axis=1) @ transform.T
    prior_variance = _kernel(flattened, flattened).sum(axis=(1, 2)) / 16
    mean = cross @ cho_solve((factor, True), measurements, check_finite=False)
    variance = prior_variance - np.sum(cross * cho_solve((factor, True), cross.T, check_finite=False).T, axis=1)
    _require(np.isfinite(mean).all() and np.isfinite(variance).all() and np.all(variance > 0),
             "positive finite latent path variance without clipping")
    risk = ndtr((mean - .5) / np.sqrt(variance))
    shape = paths.shape[:2]
    return mean.reshape(shape), variance.reshape(shape), risk.reshape(shape), rank, cutoff


def _result(records):
    return {"mean": np.stack([row[0] for row in records]),
            "variance": np.stack([row[1] for row in records]),
            "risk": np.stack([row[2] for row in records]),
            "rank": np.asarray([row[3] for row in records], dtype=np.int64),
            "rank_cutoff": np.asarray([row[4] for row in records], dtype=np.float64)}


def predict_hybrid(state, paths):
    _require(type(state) is HybridMemory, "HybridMemory required")
    _paths(state, paths, state.mask.shape[0])
    phi = basis(state.kind) if state.step > CAPACITY else None
    records = []
    for b, bits in enumerate(_mask_bits(state.mask)):
        ids = np.flatnonzero(bits)
        values = state.values[b] if phi is not None else state.values[b, :len(ids)]
        records.append(_predict_one(ids, values, paths[b], phi))
    return _result(records)


def predict_packed(state, paths):
    _require(type(state) is PackedMemory, "PackedMemory required")
    _paths(state, paths, state.mask.shape[0])
    return _result([_predict_one(np.flatnonzero(bits), state.values[b, :min(state.step, CAPACITY)], paths[b])
                    for b, bits in enumerate(_mask_bits(state.mask))])


def predict_recent(state, paths):
    _require(type(state) is RecentMemory, "RecentMemory required")
    _paths(state, paths, state.ids.shape[0])
    used = min(state.step, RECENT_CAPACITY)
    return _result([_predict_one(state.ids[b, :used], state.values[b, :used], paths[b])
                    for b in range(len(state.ids))])
