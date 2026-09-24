"""Float64 direct-solve controls for a fixed four-action residual readout.

The caller supplies already augmented frozen features, normalized teacher minus
parent-forecast errors, legal masks and scalar state weights. The weights must
already include episode, supported-row and action-count denominators. No labels,
episode weighting, cache extraction or model parameters are inferred here.

For each state this module stacks sqrt(weight) * P @ U @ B @ z against
sqrt(weight) * P @ target_error. U is the fixed orthonormal contrast basis below;
an all-true legal mask makes P the four-action centering matrix C. B has shape
(3, feature_dimension) and is flattened in row-major order. The returned head
increment U @ B is added to the parent's head, preserving its common-action
gauge. Production uses 29 augmented features and 87 increment coordinates.

OLS and positive ridge use NumPy's SVD-based lstsq with fixed rcond=1e-10. Ridge
stacks sqrt(1e-4) * I, penalizing every increment including the bias. A ridge
matrix whose numerical rank is still deficient at that cutoff is rejected, not
silently represented as a unique ridge solution. Rank and spectrum diagnostics
describe the matrix actually solved: data for OLS, augmented data for ridge.
Reuse the OLS spectrum when reporting the same data's rank for both controls.

All inputs are copied, no caller arrays or random state are mutated, and no
model/file/admission operations occur. The design has four rows per supplied
state, retaining zero-weight states as zero blocks. A 150,000-by-87 float64
design occupies about 104.4 MB; solver copies and LAPACK workspace cost extra.
Numerical failures raise ValueError. These are conventional least-squares
controls, not a calibrated-probability or autonomous-performance claim.
"""

from __future__ import annotations

import numpy as np

ACTION_DIM = 4
CONTRAST_DIM = 3
RCOND = 1e-10
RIDGE = 1e-4


def contrast_basis() -> np.ndarray:
    """Return an owned U[4,3]; dyadic entries make U'U=I and UU'=C exact."""
    return np.asarray(
        [[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]],
        dtype=np.float64,
    ) * .5


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _real_array(value, name: str, ndim: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
        _require(raw.ndim == ndim and raw.dtype.kind in "fiu", f"{name}: real numeric rank {ndim}")
        result = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, OverflowError) as error:
        raise ValueError(f"{name}: finite real numeric array") from error
    _require(bool(np.isfinite(result).all()), f"{name}: finite values")
    return result


def build_design(z, target_error, legal, weights) -> tuple[np.ndarray, np.ndarray]:
    """Build A[4N,3d], b[4N] for ||A vec(B)-b||², without an implicit divisor.

    z[N,d] includes the caller-owned bias feature; target_error[N,4] is already
    normalized. legal[N,4] must be boolean and weights[N] finite/nonnegative.
    A positive-weight state needs at least one legal action. A singleton legal
    set correctly supplies a zero contrast block. Zero-weight empty legal sets
    and N=0 are supported. All supplied numerical entries must still be finite.
    """
    features = _real_array(z, "z", 2)
    errors = _real_array(target_error, "target_error", 2)
    state_weights = _real_array(weights, "weights", 1)
    mask = np.asarray(legal)
    n, d = features.shape
    _require(d > 0, "z: positive feature dimension")
    _require(errors.shape == (n, ACTION_DIM), "target_error: shape [N,4]")
    _require(mask.shape == (n, ACTION_DIM) and mask.dtype.kind == "b", "legal: boolean shape [N,4]")
    _require(state_weights.shape == (n,) and bool((state_weights >= 0).all()), "weights: nonnegative shape [N]")
    mask = np.array(mask, dtype=np.float64, copy=True)
    count = mask.sum(axis=1)
    _require(bool(((count > 0) | (state_weights == 0)).all()), "positive-weight state needs a legal action")
    unsupported = state_weights == 0
    features[unsupported] = 0.
    errors[unsupported] = 0.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            projectors = -(mask[:, :, None] * mask[:, None, :]) / np.maximum(count, 1)[:, None, None]
            for action in range(ACTION_DIM):
                projectors[:, action, action] += mask[:, action]
            projected_basis = np.einsum("nij,jk->nik", projectors, contrast_basis())
            design = np.einsum("nik,nj->nikj", projected_basis, features).reshape(n * ACTION_DIM, CONTRAST_DIM * d)
            target = np.einsum("nij,nj->ni", projectors, errors)
            scale = np.sqrt(state_weights)
            design *= np.repeat(scale, ACTION_DIM)[:, None]
            target *= scale[:, None]
    except FloatingPointError as error:
        raise ValueError("nonfinite weighted design arithmetic") from error
    _require(bool(np.isfinite(design).all()) and bool(np.isfinite(target).all()), "finite weighted design")
    return design.copy(), target.reshape(n * ACTION_DIM).copy()


def _design(value) -> np.ndarray:
    result = _real_array(value, "design", 2)
    _require(result.shape[1] > 0 and result.shape[1] % CONTRAST_DIM == 0,
             "design: positive column count divisible by three")
    return result


def _spectrum_diagnostics(singular_values: np.ndarray, rows: int, columns: int, rank: int) -> dict:
    _require(singular_values.ndim == 1 and bool(np.isfinite(singular_values).all())
             and bool((singular_values >= 0).all()), "finite nonnegative singular spectrum")
    _require(0 <= rank <= min(rows, columns) and len(singular_values) == min(rows, columns), "valid numerical rank")
    largest = float(singular_values[0]) if singular_values.size else 0.
    cutoff = RCOND * largest
    retained = float(singular_values[rank - 1]) if rank else None
    condition = largest / retained if retained is not None else None
    _require(condition is None or np.isfinite(condition), "finite retained condition")
    return {"rows": rows, "coordinates": columns, "rank": rank,
            "full_column_rank": rank == columns, "singular_cutoff": cutoff,
            "largest_singular_value": largest, "smallest_retained_singular_value": retained,
            "retained_condition_number": condition,
            "condition_number": condition if rank == columns else None}


def design_diagnostics(design) -> dict:
    """Explicit extra SVD of data A; not called implicitly by solve_design."""
    matrix = _design(design)
    try:
        spectrum = np.linalg.svd(matrix, compute_uv=False)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise ValueError("design SVD failed") from error
    cutoff = RCOND * float(spectrum[0]) if spectrum.size else 0.
    rank = int(np.count_nonzero(spectrum > cutoff))
    return {"singular_values": spectrum.copy(), "rcond": RCOND,
            **_spectrum_diagnostics(spectrum, *matrix.shape, rank)}


def solve_design(design, target, *, mode: str) -> dict:
    """Solve the fixed OLS/ridge control; return owned coefficient arrays.

    Result keys: coefficients[3d], B[3,d], increment[4,d], singular_values and
    diagnostics. Diagnostics report b'b as objective_before, unpenalized
    data_loss, regularization, total_objective, and normal_residual, the L2 norm
    of A'(A coefficients-b)+lambda*coefficients (half the objective gradient).
    The all-zero increment is retained when the data supply no supported rows.
    """
    _require(type(mode) is str and mode in ("ols", "ridge"), "mode must be ols or ridge")
    matrix = _design(design)
    values = _real_array(target, "target", 1)
    rows, coordinates = matrix.shape
    _require(values.shape == (rows,), "target: one value per design row")
    penalty = RIDGE if mode == "ridge" else 0.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if penalty:
                solver_matrix = np.vstack((matrix, np.sqrt(penalty) * np.eye(coordinates)))
                solver_target = np.concatenate((values, np.zeros(coordinates)))
            else:
                solver_matrix, solver_target = matrix, values
            coefficients, _, rank, spectrum = np.linalg.lstsq(solver_matrix, solver_target, rcond=RCOND)
            _require(bool(np.isfinite(coefficients).all()), "finite solved coefficients")
            rank = int(rank)
            _require(not penalty or rank == coordinates, "ridge matrix numerically rank deficient at fixed cutoff")
            residual = matrix @ coefficients - values
            objective_before = float(values @ values)
            data_loss = float(residual @ residual)
            regularization = float(penalty * (coefficients @ coefficients))
            normal = matrix.T @ residual + penalty * coefficients
            normal_residual = float(np.sqrt(normal @ normal))
            objective = data_loss + regularization
            B = coefficients.reshape(CONTRAST_DIM, coordinates // CONTRAST_DIM)
            increment = contrast_basis() @ B
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise ValueError("least-squares solve or objective arithmetic failed") from error
    scalar_values = (objective_before, data_loss, regularization, objective, normal_residual)
    _require(all(np.isfinite(value) and value >= 0 for value in scalar_values)
             and bool(np.isfinite(increment).all()), "finite solution diagnostics")
    diagnostics = {"mode": mode, "rcond": RCOND, "ridge": penalty,
                   "spectrum_scope": "augmented" if penalty else "data",
                   "data_rows": rows, "feature_dimension": coordinates // CONTRAST_DIM,
                   "objective_before": objective_before, "data_loss": data_loss,
                   "regularization": regularization, "total_objective": objective,
                   "normal_residual": normal_residual,
                   **_spectrum_diagnostics(spectrum, *solver_matrix.shape, rank)}
    return {"coefficients": coefficients.copy(), "B": B.copy(), "increment": increment.copy(),
            "singular_values": spectrum.copy(), "diagnostics": diagnostics}
