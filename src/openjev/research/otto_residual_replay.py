"""Pure cached residual estimators for an unadmitted prospective comparison.

This function consumes a validated in-memory cache, never files, checkpoints,
neural models, optimizers or a simulator. Source/cache provenance and phase admission
belong to its caller. Cached cues are already normalized; replay never projects
features or normalizes cues again. Every call starts with fresh episode state.

All residual arithmetic is float64. Actual query answers are copied bit for
bit. A residual method always adds its applied correction in float64 at a
nonquery and casts the result once to float32, even when correction is zero.
Consequently a nonquery signed zero need not retain its sign for these methods.
The two ordinary baselines instead copy their cached float32 scores directly.

``prewrite_correction`` is the applied multiplier times the prewrite mean at
every key step, including later queries. ``epistemic_variance`` is the RLS
posterior variance BEFORE the write and BEFORE read attenuation. It is NaN at
first queries and everywhere for non-RLS methods. It is not action confidence.

``work_counts`` records schedule geometry and selected executed operations, not
FLOPs or an exhaustive count of validation, copying and scalar arithmetic.
Feature-extraction work remains in the input cache and is not charged again
here. Every full RLS initialization and update includes the qualified reference's
Cholesky validation, and those calls are counted. ``state_bytes`` counts only
persistent estimator arrays for one active episode: it excludes input/output
arrays, cached traces, Python metadata and all temporary or validation buffers.
It is not peak RSS or a runtime/speed claim.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np

from openjev.research.bayesian_score_memory import BayesianScoreMemory
from openjev.research.otto_residual_contract import (
    KEY_DIM,
    MULTIPLIERS,
    RLS_METHODS,
    SCORE_DIM,
    method_config,
    require,
    validate_cache,
)

_WORK_FIELDS = (
    "cache_validation_calls", "cache_rows_validated", "episodes", "rows", "query_rows", "first_query_rows",
    "key_rows", "later_query_rows", "nonquery_rows", "neural_model_calls", "projection_calls", "optimizer_calls",
    "teacher_calls", "native_calls", "base_rows_copied", "query_rows_copied", "state_initializations",
    "residual_target_rows", "residual_target_coordinates", "applied_correction_rows", "applied_correction_coordinates",
    "action_score_rows", "action_score_coordinates", "float32_action_roundings", "last_error_decays",
    "last_error_decay_coordinates", "last_error_reads", "last_error_writes", "last_error_written_coordinates",
    "delta_matrix_decays", "delta_matrix_decay_coordinates", "delta_matrix_reads", "delta_matrix_read_terms",
    "delta_read_centerings", "delta_denominator_terms", "delta_matrix_writes", "delta_matrix_write_coordinates",
    "bayesian_initializations", "bayesian_predict_calls", "bayesian_observe_calls", "bayesian_direction_terms",
    "bayesian_variance_terms", "bayesian_mean_read_terms", "bayesian_mean_update_coordinates",
    "bayesian_covariance_update_entries", "bayesian_covariance_validations", "bayesian_cholesky_calls",
)


def _center(values):
    return values - np.sum(values * .25)


def _target(cache, row, counts):
    # This helper is called only at an eligible later query. Hidden teacher
    # scores and inactive shadow-prior rows are never consumed by replay.
    answer = cache["query_scores"][row].astype(np.float64, copy=True)
    shadow = cache["shadow_prior"][row].astype(np.float64, copy=True)
    result = _center((answer - shadow) / 64.)
    require(bool(np.isfinite(result).all()), "finite eligible centered residual target")
    counts["residual_target_rows"] += 1
    counts["residual_target_coordinates"] += SCORE_DIM
    return result


def _action(base, correction):
    result = (base.astype(np.float64, copy=True) + 64. * correction).astype(np.float32)
    require(bool(np.isfinite(result).all()), "finite corrected float32 action scores")
    return result


def _state_bytes(method):
    if method in ("pretrained", "joint_aux"):
        return 0
    if method == "last_error":
        return SCORE_DIM * np.dtype(np.float64).itemsize
    entries = SCORE_DIM * KEY_DIM
    if method in RLS_METHODS:
        entries += KEY_DIM if method == "rls_diagonal" else KEY_DIM**2
    return entries * np.dtype(np.float64).itemsize


def replay(cache, method, *, tau=None):
    """Replay one declared estimator over complete episodes with owned outputs.

    RLS requires a prospective-grid ``tau`` and uses prior covariance ``tau I``
    and observation variance one. Ordinary baselines reject a supplied ratio.
    Every episode excludes its first query from estimator reads and writes.
    RLS ``observe`` supplies the prewrite read on later queries, so no duplicate
    prediction call occurs. Shrink variants alter only application of the mean;
    the underlying posterior sees the same unattenuated residual regression.
    Any invalid input or arithmetic raises without changing caller-owned cache
    arrays. No output or state survives a failed call.
    """
    tau = method_config(method, tau)
    validate_cache(cache)
    rows = len(cache["base_action"])
    episodes = len(cache["episode_offsets"]) - 1
    counts = dict.fromkeys(_WORK_FIELDS, 0)
    query_rows = int(cache["query_mask"].sum())
    counts.update(cache_validation_calls=1, cache_rows_validated=rows, episodes=episodes, rows=rows,
                  query_rows=query_rows, first_query_rows=episodes, key_rows=rows - episodes,
                  later_query_rows=query_rows - episodes, nonquery_rows=rows - query_rows,
                  base_rows_copied=rows, query_rows_copied=query_rows)
    source = cache["joint_action"] if method == "joint_aux" else cache["base_action"]
    actions = source.copy()
    actions[cache["query_mask"]] = cache["query_scores"][cache["query_mask"]]
    corrections = np.zeros((rows, SCORE_DIM), dtype=np.float64)
    epistemic = np.full(rows, np.nan, dtype=np.float64)
    if method in ("pretrained", "joint_aux"):
        return {"method": method, "tau": tau, "action_scores": actions, "prewrite_correction": corrections,
                "epistemic_variance": epistemic, "work_counts": counts, "state_bytes": 0}

    rls = method in RLS_METHODS
    diagonal = method == "rls_diagonal"
    multiplier = MULTIPLIERS[method] if rls else 1.
    covariance_entries = KEY_DIM if diagonal else KEY_DIM**2
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for low, high in pairwise(cache["episode_offsets"]):
                counts["state_initializations"] += 1
                if rls:
                    posterior = BayesianScoreMemory(KEY_DIM, prior_variance=tau, noise_variance=1.,
                                                    mode="diagonal" if diagonal else "full")
                    counts["bayesian_initializations"] += 1
                    counts["bayesian_covariance_validations"] += 1
                    counts["bayesian_cholesky_calls"] += int(not diagonal)
                elif method == "last_error":
                    last = np.zeros(SCORE_DIM, dtype=np.float64)
                else:
                    matrix = np.zeros((SCORE_DIM, KEY_DIM), dtype=np.float64)

                # Complete P4 geometry was checked by validate_cache. Skipping
                # the first row also makes poisoned first cues/shadows inert.
                for row in range(int(low) + 1, int(high)):
                    query = bool(cache["query_mask"][row])
                    if method == "last_error":
                        last = .75 * last
                        mean = last.copy()
                        counts["last_error_decays"] += 1
                        counts["last_error_decay_coordinates"] += SCORE_DIM
                        counts["last_error_reads"] += 1
                        if query:
                            last = _target(cache, row, counts)
                            counts["last_error_writes"] += 1
                            counts["last_error_written_coordinates"] += SCORE_DIM
                    elif method == "trace_delta":
                        cue = cache["cues"][row].astype(np.float64, copy=True)
                        matrix = matrix * 1.
                        mean = _center(matrix @ cue)
                        counts["delta_matrix_decays"] += 1
                        counts["delta_matrix_decay_coordinates"] += SCORE_DIM * KEY_DIM
                        counts["delta_matrix_reads"] += 1
                        counts["delta_matrix_read_terms"] += SCORE_DIM * KEY_DIM
                        counts["delta_read_centerings"] += 1
                        if query:
                            target = _target(cache, row, counts)
                            denominator = 1e-6 + float(cue @ cue)
                            require(np.isfinite(denominator) and denominator > 0, "finite positive delta denominator")
                            matrix = matrix + .25 * np.outer(target - mean, cue) / denominator
                            require(bool(np.isfinite(matrix).all()), "finite updated delta matrix")
                            counts["delta_denominator_terms"] += KEY_DIM
                            counts["delta_matrix_writes"] += 1
                            counts["delta_matrix_write_coordinates"] += SCORE_DIM * KEY_DIM
                    else:
                        cue = cache["cues"][row].astype(np.float64, copy=True)
                        if query:
                            prediction = posterior.observe(cue, _target(cache, row, counts))
                            counts["bayesian_observe_calls"] += 1
                            counts["bayesian_mean_update_coordinates"] += SCORE_DIM * KEY_DIM
                            counts["bayesian_covariance_update_entries"] += covariance_entries
                            counts["bayesian_covariance_validations"] += 1
                            counts["bayesian_cholesky_calls"] += int(not diagonal)
                        else:
                            prediction = posterior.predict(cue)
                            counts["bayesian_predict_calls"] += 1
                        counts["bayesian_direction_terms"] += covariance_entries
                        counts["bayesian_variance_terms"] += KEY_DIM
                        counts["bayesian_mean_read_terms"] += SCORE_DIM * KEY_DIM
                        mean = prediction.mean
                        epistemic[row] = prediction.epistemic_variance

                    applied = multiplier * mean
                    require(bool(np.isfinite(applied).all()), "finite prewrite applied correction")
                    corrections[row] = applied
                    counts["applied_correction_rows"] += 1
                    counts["applied_correction_coordinates"] += SCORE_DIM
                    if not query:
                        actions[row] = _action(cache["base_action"][row], applied)
                        counts["action_score_rows"] += 1
                        counts["action_score_coordinates"] += SCORE_DIM
                        counts["float32_action_roundings"] += SCORE_DIM
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("nonfinite residual replay arithmetic") from error

    require(actions[cache["query_mask"]].tobytes() == cache["query_scores"][cache["query_mask"]].tobytes(),
            "actual query answers retain exact float32 bytes")
    require(bool(np.isfinite(actions).all()) and bool(np.isfinite(corrections).all()), "finite complete residual replay")
    require(bool(np.isfinite(epistemic[cache["key_mask"]]).all()) if rls else bool(np.isnan(epistemic).all()),
            "declared epistemic variance support")
    require(bool(np.isnan(epistemic[~cache["key_mask"]]).all()), "first query has no posterior read")
    return {"method": method, "tau": tau, "action_scores": actions, "prewrite_correction": corrections,
            "epistemic_variance": epistemic, "work_counts": counts, "state_bytes": _state_bytes(method)}
