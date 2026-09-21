"""Independent NumPy replay of every row in a saved Bellman-target refresh.

File authentication, refresh/checkpoint provenance and study coverage belong to
the caller. This pure numerical API opens no data and runs no producer model,
optimizer or environment. It does perform one independent saved-checkpoint
readout of sixteen branches per selected TRAIN state, and reports that work.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/audit_otto_return_value.py"
HELPER_SHA256 = "ad3d8e92c0bae3b9dd573af40e1cbb49d1dc771ff23f30500667a46f94980de8"
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA256:
    raise ValueError("independent return-value auditor source pin")
_SPEC = importlib.util.spec_from_file_location("_bellman_independent_return_math", HELPER)
A = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = A
_SPEC.loader.exec_module(A)

VERSION = "otto-bellman-target-saved-audit-v1"
ATOL = RTOL = 1e-10
FLOAT_FIELDS = {
    "raw_masses": (4, 4), "weights": (4, 4), "branch_values": (16,),
    "branch_contributions": (4, 4), "costs": (4,), "minimum_costs": (),
    "targets_float64": (), "deployed_cost_gaps": (),
}
INTEGER_FIELDS = {"row_indices": (), "successors": (16, 2),
                  "minimum_actions": (), "deployed_actions": ()}
FIELDS = frozenset((*FLOAT_FIELDS, *INTEGER_FIELDS, "eligible_masks", "targets_float32"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _indices(value, upper):
    array = np.asarray(value)
    require(array.ndim == 1 and array.dtype.kind in "iu" and len(array) > 0
            and (array >= 0).all() and (array < upper).all(), "nonempty integer TRAIN row indices")
    require(len(np.unique(array)) == len(array), "no repeated TRAIN row indices")
    return array.astype(np.int64, copy=False)


def audit_refresh(payload, checkpoint, *, train, eligible_actions, kernels,
                  expected_row_indices, checkpoint_sha256, expected_checkpoint_sha256,
                  c0, check=None):
    """Replay an authenticated refresh, including every supplied selected row.

    payload is exactly the BellmanTargets arrays plus int64 row_indices.
    checkpoint is the original-format mlp8 NPZ mapping (metadata scalar arrays).
    train contains beliefs:f64[N,53,53], positions:i64[N,2], sensing_length:f64[N];
    other cached keys are ignored. eligible_actions covers the entire canonical
    TRAIN order. kernels maps numerical sensing lengths to f64[4,107,107].
    TRAIN sensing lengths are three/four only. expected_row_indices is the
    externally fixed full order or qualification subset; it is never selected
    here. Both checkpoint pins are caller-authenticated file identities.

    Float64 witnesses use atol=rtol=1e-10. Discrete decisions/masks/row joins
    are exact. The float32 target must be the exact cast of the *saved* float64
    target, whose independent reconstruction is separately tolerance-checked.
    This avoids claiming a second implementation rounds every borderline
    float32 value identically. Call counts describe this audit, not the study.
    """
    require(_pin(checkpoint_sha256) and _pin(expected_checkpoint_sha256)
            and checkpoint_sha256 == expected_checkpoint_sha256, "target checkpoint identity join")
    require(check is None or callable(check), "check must be callable or None")
    if check is not None:
        check()
    require(isinstance(train, Mapping) and {"beliefs", "positions", "sensing_length"} <= train.keys(),
            "canonical TRAIN array mapping")
    beliefs, positions, sensing = (train[k] for k in ("beliefs", "positions", "sensing_length"))
    require(isinstance(beliefs, np.ndarray) and beliefs.dtype == np.float64
            and beliefs.ndim == 3 and beliefs.shape[1:] == (53, 53) and len(beliefs) > 0,
            "float64 TRAIN beliefs[N,53,53]")
    total = len(beliefs)
    require(isinstance(positions, np.ndarray) and positions.dtype == np.int64
            and positions.shape == (total, 2) and (positions >= 0).all() and (positions < 53).all(),
            "int64 TRAIN positions[N,2]")
    require(isinstance(sensing, np.ndarray) and sensing.dtype == np.float64
            and sensing.shape == (total,) and np.isin(sensing, [3., 4.]).all(),
            "only training-supported sensing lengths three/four")
    require(isinstance(eligible_actions, (Sequence, np.ndarray)) and len(eligible_actions) == total,
            "eligibility for every TRAIN row")
    require(isinstance(kernels, Mapping), "public kernel mapping")
    indices = _indices(expected_row_indices, total)
    count = len(indices)
    require(isinstance(payload, Mapping) and set(payload) == FIELDS, "exact saved target array closure")
    layouts = {**{k: (shape, np.dtype("float64")) for k, shape in FLOAT_FIELDS.items()},
               **{k: (shape, np.dtype("int64")) for k, shape in INTEGER_FIELDS.items()},
               "eligible_masks": ((4,), np.dtype("bool")), "targets_float32": ((), np.dtype("float32"))}
    for name, (shape, dtype) in layouts.items():
        value = payload[name]
        require(isinstance(value, np.ndarray) and value.dtype == dtype
                and value.shape == (count, *shape) and np.isfinite(value).all(),
                f"saved {name} shape/dtype/finiteness")
    require((payload["raw_masses"] >= 0).all() and (payload["weights"] >= 1e-10).all()
            and (payload["deployed_cost_gaps"] >= 0).all(), "nonnegative masses/gaps and preserved positive floor")
    require(np.array_equal(payload["weights"], np.maximum(payload["raw_masses"], 1e-10)),
            "exact saved mass-to-floor identity")
    require(np.array_equal(payload["row_indices"], indices), "exact frozen TRAIN row order")
    require(type(c0) in (int, float) and math.isfinite(c0), "finite expected TRAIN baseline")
    weights = A.checkpoint(checkpoint, "mlp8", c0, np)
    verified_kernels = {}
    for lam in {float(sensing[i]) for i in indices}:
        require(lam in kernels, "selected state's public kernel")
        kernel = kernels[lam]
        require(isinstance(kernel, np.ndarray) and kernel.dtype == np.float64
                and kernel.shape == (4, 107, 107) and np.isfinite(kernel).all()
                and (kernel >= 0).all() and (kernel <= 1).all() and not kernel[:, 53, 53].any(),
                "finite exact-shape kernel with zero origin")
        verified_kernels[lam] = kernel

    compared = 0
    maximum = {name: 0. for name in FLOAT_FIELDS}
    for saved_index, train_index in enumerate(indices):
        if check is not None:
            check()
        belief, position, lam = beliefs[train_index], positions[train_index], float(sensing[train_index])
        require(np.isfinite(belief).all() and (belief >= 0).all()
                and float(belief.sum(dtype=np.float64)) <= 1 + 1e-6, "finite public TRAIN belief mass")
        allowed = eligible_actions[train_index]
        require(isinstance(allowed, (Sequence, np.ndarray)) and len(allowed) > 0
                and all(not isinstance(a, (bool, np.bool_)) and isinstance(a, (int, np.integer)) for a in allowed),
                "integer public action eligibility")
        allowed = [int(a) for a in allowed]
        pos = position.tolist()
        require(allowed == [a for a in range(4) if A.B.move(pos, a) != pos], "exact public inbounds mask")
        x, raw, floor = A.branches(belief, pos, verified_kernels[lam], lam, np)
        with np.errstate(over="raise", invalid="raise"):
            values = 64 * A.predict(x, weights, "mlp8", np)
            scores = A.costs(values, floor, np)
            contributions = floor * values.reshape(4, 4)
        require(np.isfinite(scores).all() and np.isfinite(contributions).all(), "finite reconstructed branches")
        minimum = scores[allowed].min()
        first_minimum = next(a for a in allowed if scores[a] == minimum)
        deployed = A.choice(scores.tolist(), allowed, True, np)
        successors = np.asarray([A.B.move(pos, a) for a in range(4) for _ in range(4)], np.int64)
        expected = {"row_indices": np.asarray(train_index, np.int64), "raw_masses": raw, "weights": floor,
                    "branch_values": values, "branch_contributions": contributions, "successors": successors,
                    "costs": scores, "eligible_masks": np.asarray([a in allowed for a in range(4)], bool),
                    "minimum_costs": minimum, "minimum_actions": np.asarray(first_minimum, np.int64),
                    "deployed_actions": np.asarray(deployed, np.int64), "targets_float64": minimum / 64,
                    "deployed_cost_gaps": scores[deployed] - minimum}
        with np.errstate(over="raise", invalid="raise"):
            expected["targets_float32"] = payload["targets_float64"][saved_index].astype(np.float32)
        for name, reference in expected.items():
            actual, reference = np.asarray(payload[name][saved_index]), np.asarray(reference)
            compared += actual.size
            if name in FLOAT_FIELDS:
                difference = np.abs(actual - reference)
                maximum[name] = max(maximum[name], float(difference.max()))
                require(bool((difference <= ATOL + RTOL * np.abs(reference)).all()), f"{name} row {int(train_index)}")
            elif name == "targets_float32":
                require(actual.tobytes() == reference.tobytes(), f"exact {name} row {int(train_index)}")
            else:
                require(np.array_equal(actual, reference), f"exact {name} row {int(train_index)}")
    return {"version": VERSION, "agreement": True, "rows": count,
            "checkpoint_sha256": checkpoint_sha256, "independent_helper_sha256": HELPER_SHA256,
            "numeric_items_compared": compared, "maximum_absolute_difference": max(maximum.values()),
            "maximum_absolute_difference_by_field": maximum, "atol": ATOL, "rtol": RTOL,
            "audit_checkpoint_readout_calls": count, "audit_branch_rows": 16 * count,
            "training_calls": 0, "simulator_calls": 0,
            "scope": "Every requested TRAIN row, independent saved-checkpoint NumPy readout; file provenance and full refresh coverage are caller checks. Float32 cast is exact against saved float64 target."}
