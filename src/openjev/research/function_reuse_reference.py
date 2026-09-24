"""Public-only least-squares references for the disclosed function-reuse task.

Rows are demonstrations: Y = X @ W. Each public block is fit independently by
NumPy's minimum-norm least-squares solver, with rcond=None. At query time the
block with smallest mean squared few-shot residual is selected. Exact floating
ties select the first block; no tolerance or hidden basis identity is used.

The disclosed task uses dimension eight and sixteen demonstrations per block.
Positive smaller dimensions are supported for independent fabricated tests.
This is established linear algebra, not a trained neural architecture. Rank
deficiency is accepted and reported; it prevents a general recovery guarantee.
Duplicating noiseless rows preserves the mathematical solution, not necessarily
its last floating-point bits. No RNG, hidden matrices, query targets, file I/O,
or persistent global state is used. Reported storage covers retained numeric
bank arrays, not Python overhead, input buffers, or LAPACK's transient workspace.
"""
from __future__ import annotations

import numpy as np

VERSION = "function-reuse-reference-v1"
FIT_KEYS = {"version", "maps", "singular_values", "ranks", "full_rank", "metadata"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, ndim, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float64")
             and value.ndim == ndim and all(size > 0 for size in value.shape)
             and np.isfinite(value).all(), f"{name} must be a nonempty finite float64 array of rank {ndim}")


def _owned(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def storage_metadata(blocks, demonstrations, dimension):
    """Exact numeric payload formulas, with no claim about peak runtime memory."""
    for value in (blocks, demonstrations, dimension):
        _require(type(value) is int and value > 0, "positive integer storage dimensions")
    coefficients = blocks * dimension * dimension
    singular_values = blocks * min(demonstrations, dimension)
    return {
        "blocks": blocks,
        "demonstrations_per_block": demonstrations,
        "dimension": dimension,
        "learned_parameter_count": 0,
        "fitted_coefficient_count": coefficients,
        "fitted_coefficient_bytes": 8 * coefficients,
        "singular_value_bytes": 8 * singular_values,
        "rank_bytes": 8 * blocks,
        "full_rank_flag_bytes": blocks,
        "diagnostic_bytes": 8 * singular_values + 9 * blocks,
        "retained_numeric_state_bytes": 8 * (coefficients + singular_values + blocks) + blocks,
        "public_demonstration_bytes": 2 * blocks * demonstrations * dimension * 8,
        "raw_demonstrations_retained": False,
        "workspace_bytes_measured": False,
    }


def fit_blocks(block_x, block_y):
    """Fit public float64 [K,b,d] arrays and return an owned diagnostic bank.

    maps[k] has shape [d,d], with input coordinate first and output coordinate
    second. singular_values[k] has min(b,d) entries in NumPy's descending order.
    ranks is int64[K]; full_rank is bool[K], meaning rank equals d.
    """
    _array(block_x, 3, "block_x")
    _array(block_y, 3, "block_y")
    _require(block_x.shape == block_y.shape, "block_x and block_y must have identical [K,b,d] shape")
    blocks, demonstrations, dimension = block_x.shape
    maps = np.empty((blocks, dimension, dimension), np.float64)
    singular_values = np.empty((blocks, min(demonstrations, dimension)), np.float64)
    ranks = np.empty(blocks, np.int64)
    for block in range(blocks):
        try:
            matrix, _, rank, singular = np.linalg.lstsq(block_x[block], block_y[block], rcond=None)
        except np.linalg.LinAlgError as error:
            raise ValueError("least-squares solver did not converge") from error
        _require(np.isfinite(matrix).all() and np.isfinite(singular).all(), "nonfinite least-squares result")
        maps[block], singular_values[block], ranks[block] = matrix, singular, rank
    return {
        "version": VERSION,
        "maps": _owned(maps),
        "singular_values": _owned(singular_values),
        "ranks": _owned(ranks),
        "full_rank": _owned(ranks == dimension),
        "metadata": storage_metadata(blocks, demonstrations, dimension),
    }


def _validate_fit(fitted):
    _require(type(fitted) is dict and set(fitted) == FIT_KEYS and fitted["version"] == VERSION,
             "exact fitted-bank schema")
    maps = fitted["maps"]
    _array(maps, 3, "maps")
    blocks, dimension, outputs = maps.shape
    _require(dimension == outputs, "square fitted maps")
    metadata = fitted["metadata"]
    _require(type(metadata) is dict, "fitted metadata dictionary")
    demonstrations = metadata.get("demonstrations_per_block")
    expected = storage_metadata(blocks, demonstrations, dimension)
    _require(metadata == expected, "fitted storage metadata mismatch")
    singular = fitted["singular_values"]
    _array(singular, 2, "singular_values")
    _require(singular.shape == (blocks, min(demonstrations, dimension))
             and (singular >= 0).all() and (np.diff(singular, axis=1) <= 0).all(), "singular-value diagnostics")
    ranks, full_rank = fitted["ranks"], fitted["full_rank"]
    _require(isinstance(ranks, np.ndarray) and ranks.dtype == np.dtype("int64")
             and ranks.shape == (blocks,) and ((ranks >= 0) & (ranks <= min(demonstrations, dimension))).all(),
             "rank diagnostics")
    _require(isinstance(full_rank, np.ndarray) and full_rank.dtype == np.dtype("bool")
             and full_rank.shape == (blocks,) and np.array_equal(full_rank, ranks == dimension),
             "full-rank diagnostics")
    return maps


def _query_inputs(fewshot_x, fewshot_y, query_x, dimension):
    _array(fewshot_x, 2, "fewshot_x")
    _array(fewshot_y, 2, "fewshot_y")
    _array(query_x, 1, "query_x")
    _require(fewshot_x.shape == fewshot_y.shape and fewshot_x.shape[1] == dimension
             and query_x.shape == (dimension,), "few-shot and query dimensions must match fitted maps")


def predict(fitted, fewshot_x, fewshot_y, query_x):
    """Select using public few-shot MSE only, then return query_x @ selected W.

    Returns prediction[d], selected_block (int), block_residuals[K], and
    tie_indices[int64] for exact minima. No target for query_x is accepted.
    Selection does not use query_x or refit the stored maps on the few-shots.
    """
    maps = _validate_fit(fitted)
    _query_inputs(fewshot_x, fewshot_y, query_x, maps.shape[1])
    try:
        with np.errstate(over="raise", invalid="raise"):
            errors = np.einsum("fd,kdo->kfo", fewshot_x, maps) - fewshot_y[None]
            residuals = np.mean(np.square(errors), axis=(1, 2), dtype=np.float64)
            selected = int(np.argmin(residuals))
            prediction = query_x @ maps[selected]
    except FloatingPointError as error:
        raise ValueError("nonfinite query arithmetic") from error
    _require(np.isfinite(residuals).all() and np.isfinite(prediction).all(), "nonfinite query result")
    ties = np.flatnonzero(residuals == residuals[selected]).astype(np.int64)
    return {"prediction": _owned(prediction), "selected_block": selected,
            "block_residuals": _owned(residuals), "tie_indices": _owned(ties)}


def fewshot_only(fewshot_x, fewshot_y, query_x):
    """Minimum-norm control using only the public few-shots, without the bank.

    Returns prediction[d], rank (int), singular_values[min(F,d)], and fit, a
    one-block bank with the same diagnostics and storage accounting as
    fit_blocks. The top-level singular values are an owned convenience copy,
    not additional retained bank state. Rank deficiency is not rejected.
    """
    _array(fewshot_x, 2, "fewshot_x")
    _query_inputs(fewshot_x, fewshot_y, query_x, fewshot_x.shape[1])
    fitted = fit_blocks(fewshot_x[None], fewshot_y[None])
    try:
        with np.errstate(over="raise", invalid="raise"):
            prediction = query_x @ fitted["maps"][0]
    except FloatingPointError as error:
        raise ValueError("nonfinite few-shot prediction") from error
    _require(np.isfinite(prediction).all(), "nonfinite few-shot prediction")
    return {"prediction": _owned(prediction), "rank": int(fitted["ranks"][0]),
            "singular_values": _owned(fitted["singular_values"][0]), "fit": fitted}
