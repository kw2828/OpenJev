"""Floor-free Bayesian reference for a static source and a supplied sensor kernel.

These functions have no environment, model, label-file or random-stream access.
Columns 0..3 of an OTTO likelihood mean hits 0, 1, 2, and >=3 respectively;
output column 4 is found. The source at the visited position is found before
sampling an odor. A kernel's origin row may therefore be exactly zero.

This is a separate probabilistic filter. For the public sampler, initialize it
from ``normalize_prior(cdf_law(initial_prior)["probabilities"])`` and thereafter
assimilate each public observation with ``observe``. Do not replace its state
with the legacy filter, whose 1e-10
evidence floor can leave subnormalized mass. Only the explicit initial helper
permits a recorded, bounded roundoff normalization. There is no probability
clipping, likelihood smoothing, evidence floor or silent support repair.

The formulas are exact for the supplied categorical kernel in real arithmetic,
subject to float64 roundoff and the stated normalization tolerance. CDF laws
assume an equally likely discrete uniform grid with the declared bit width;
matching that assumption to a runtime is a separate source-authentication step.
They do not claim a bitwise trajectory match or a continuous physical sensor.
Full source belief and kernel are privileged information, not learner inputs.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np

VERSION = "otto-predictive-belief-v1"
INITIAL_MASS_TOLERANCE = 1e-10
NORMALIZATION_ATOL = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, dimensions, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.ndim == dimensions, name + " must be a float64 ndarray")
    _require(np.isfinite(value).all() and (value >= 0).all(), name + " must be finite and nonnegative")
    return value


def _belief(belief):
    _array(belief, 1, "belief")
    _require(len(belief) > 0 and (belief <= 1).all(), "nonempty belief in [0,1]")
    mass = float(belief.sum(dtype=np.float64))
    _require(abs(mass - 1.) <= NORMALIZATION_ATOL, "belief must already be normalized")
    return belief


def _position(position, size):
    _require(isinstance(position, Integral) and not isinstance(position, (bool, np.bool_))
             and 0 <= position < size, "position index must be an in-bounds integer")
    return int(position)


def _likelihood(likelihood, size, position):
    _array(likelihood, 2, "likelihood")
    _require(likelihood.shape == (size, 4) and (likelihood <= 1).all(), "likelihood shape [N,4] and values in [0,1]")
    totals = likelihood.sum(axis=1, dtype=np.float64)
    valid = np.abs(totals - 1.) <= NORMALIZATION_ATOL
    # Native kernel has an all-zero origin. It is never an odor branch.
    valid[position] = valid[position] or bool((likelihood[position] == 0).all())
    _require(valid.all(), "every non-origin likelihood row must normalize")
    return likelihood


def _output(probabilities):
    _require(np.isfinite(probabilities).all() and (probabilities >= 0).all()
             and (probabilities <= 1. + NORMALIZATION_ATOL).all()
             and abs(float(probabilities.sum(dtype=np.float64)) - 1.) <= 3 * NORMALIZATION_ATOL,
             "predictive mass must be conserved")
    return probabilities


def cdf_law(probabilities, *, uniform_bits=53):
    """Categorical law for normalized CDF search on a finite uniform grid.

    The explicit assumption is U=j/2**uniform_bits, with equally likely integers
    j in [0,2**uniform_bits). For searchsorted(side="right"), category i contains
    ceil(2**bits * CDF[i]) - ceil(2**bits * CDF[i-1]) grid points. Equality at an
    internal CDF boundary therefore belongs to the next positive interval.
    A default 53-bit grid describes the intended PCG64 float64 conversion only
    after the caller authenticates the runtime source; no draws are inspected.

    First mirror cumsum(p)/cumsum(p)[-1]. Both cumulative-sum support loss and
    finite-grid quantization are reported, with no floor or support repair.
    All represented boundaries times 2**bits and integer counts through 2**53
    fit exactly in float64; integer differencing occurs in int64. Exact-zero
    origin kernel rows have no odor law and must not enter this API.
    """
    _array(probabilities, 1, "categorical probabilities")
    _require(len(probabilities) > 0, "nonempty categorical probabilities")
    _require(type(uniform_bits) is int and 1 <= uniform_bits <= 53, "uniform_bits must be an integer in [1,53]")
    raw_mass = float(probabilities.sum(dtype=np.float64))
    _require(np.isfinite(raw_mass) and abs(raw_mass - 1.) <= INITIAL_MASS_TOLERANCE,
             "categorical raw mass must be within 1e-10 of one")
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cdf_mass = float(cumulative[-1])
    _require(np.isfinite(cdf_mass) and cdf_mass > 0., "positive finite CDF mass")
    cumulative /= cdf_mass
    continuous = np.diff(np.concatenate((np.zeros(1, np.float64), cumulative)))
    cumulative_bins = np.ceil(np.ldexp(cumulative, uniform_bits)).astype(np.int64)
    bin_counts = np.diff(np.concatenate((np.zeros(1, np.int64), cumulative_bins)))
    effective = bin_counts.astype(np.float64) / (2**uniform_bits)
    _belief(effective)
    return {"probabilities": effective, "raw_mass": raw_mass, "cdf_mass": cdf_mass,
            "uniform_bits": uniform_bits,
            "max_abs_adjustment": float(np.abs(effective - probabilities).max()),
            "max_cdf_quantization_adjustment": float(np.abs(effective - continuous).max()),
            "cdf_positive_entries_rounded_to_zero": int(np.count_nonzero((probabilities > 0) & (continuous == 0))),
            "positive_entries_rounded_to_zero": int(np.count_nonzero((probabilities > 0) & (effective == 0)))}


def normalize_prior(prior, *, mass_tolerance=INITIAL_MASS_TOLERANCE):
    """Return an owned normalized initial prior plus its exact correction record.

    The tolerance can be tightened but cannot exceed 1e-10. A subfloor, empty or
    materially subnormalized legacy posterior is not an admissible initial prior.
    No provenance can be inferred from an array; initial/public-only provenance
    and subsequent observation order remain the caller's responsibility.
    """
    _array(prior, 1, "initial prior")
    _require(type(mass_tolerance) in (float, int) and 0 <= mass_tolerance <= INITIAL_MASS_TOLERANCE,
             "initial mass tolerance must be finite and in [0,1e-10]")
    _require(len(prior) > 0, "nonempty initial prior")
    mass = float(prior.sum(dtype=np.float64))
    _require(np.isfinite(mass) and mass > 0 and abs(mass - 1.) <= mass_tolerance,
             "initial prior mass is outside the recorded correction tolerance")
    normalized = prior / mass
    _belief(normalized)
    return {"belief": normalized, "input_mass": mass, "normalization_factor": 1. / mass,
            "correction_l1": float(np.abs(normalized - prior).sum(dtype=np.float64)),
            "mass_tolerance": float(mass_tolerance)}


def one_step(belief, likelihood, position_index, *, known_terminal=False):
    """Five-class next-outcome distribution before the next observation.

    ``known_terminal`` is only a previously observed terminal flag. It must never
    be set from a future realized flag or inferred by looking at the true source.
    On that normal-observation suffix the returned vector is exactly [0,0,0,0,1].
    The blind API deliberately accepts no such flag.
    """
    _belief(belief)
    position = _position(position_index, len(belief))
    _likelihood(likelihood, len(belief), position)
    _require(type(known_terminal) is bool, "known_terminal must be an explicit boolean")
    probabilities = np.zeros(5, dtype=np.float64)
    if known_terminal:
        probabilities[4] = 1.
    else:
        surviving = belief.copy()
        surviving[position] = 0.
        probabilities[:4] = np.sum(surviving[:, None] * likelihood, axis=0, dtype=np.float64)
        probabilities[4] = belief[position]
    return _output(probabilities)


def observe(belief, likelihood, position_index, odor=None, *, known_found=False):
    """Condition on one actual public observation, after forecasting that step.

    A nonterminal odor must be 0..3. For an observed found, pass ``odor=None``
    and ``known_found=True``; evidence is the prior source mass at the position.
    Zero-evidence observations fail, including impossible found. Positive
    evidence below the legacy floor is divided normally. If a positive product
    underflows float64 to zero, fail explicitly instead of losing its support.
    Absorption after a known found is handled by one_step's explicit flag, not
    repeated observation updates at new positions.
    """
    _belief(belief)
    position = _position(position_index, len(belief))
    _likelihood(likelihood, len(belief), position)
    _require(type(known_found) is bool, "known_found must be an explicit boolean")
    if known_found:
        _require(odor is None, "found has no odor observation")
        evidence = float(belief[position])
        _require(evidence > 0., "zero-evidence found observation")
        posterior = np.zeros_like(belief)
        posterior[position] = 1.
    else:
        _require(isinstance(odor, Integral) and not isinstance(odor, (bool, np.bool_))
                 and 0 <= odor < 4, "nonterminal odor must be an integer in [0,3]")
        surviving = belief.copy()
        surviving[position] = 0.
        joint = surviving * likelihood[:, int(odor)]
        if np.any((surviving > 0) & (likelihood[:, int(odor)] > 0) & (joint == 0)):
            raise FloatingPointError("positive observation mass underflowed float64")
        evidence = float(joint.sum(dtype=np.float64))
        _require(evidence > 0., "zero-evidence odor observation")
        posterior = joint / evidence
    _belief(posterior)
    return {"belief": posterior, "evidence": evidence, "known_terminal": known_found}


def blind_marginals(belief, likelihoods, positions):
    """Unconditional five-class marginals for a precommitted position sequence.

    At h, found mass is the sum of initial belief over UNIQUE positions through
    h. Odor masses sum over all remaining source cells. Earlier odors are
    integrated out, not replaced with an imagined realization. Repeated visits
    never add found mass twice. No actual future observation/found flag is an
    argument. Valid motion/bounds in the physical grid are caller obligations;
    this module validates flattened source-cell indices only.
    """
    _belief(belief)
    _array(likelihoods, 3, "likelihood sequence")
    _require(likelihoods.shape[1:] == (len(belief), 4) and len(likelihoods) > 0,
             "nonempty likelihood sequence [H,N,4]")
    _require(isinstance(positions, np.ndarray) and positions.dtype.kind in "iu"
             and positions.shape == (len(likelihoods),), "positions must be integer[H]")
    # Validate all steps before beginning the forecast. This has no stateful
    # partial result, and never returns an admitted prefix of a malformed path.
    for likelihood, position in zip(likelihoods, positions, strict=True):
        index = _position(position, len(belief))
        _likelihood(likelihood, len(belief), index)
    visited = np.zeros(len(belief), dtype=np.bool_)
    result = np.empty((len(likelihoods), 5), dtype=np.float64)
    for horizon, (likelihood, position) in enumerate(zip(likelihoods, positions, strict=True)):
        visited[int(position)] = True
        result[horizon, :4] = np.sum(belief[~visited, None] * likelihood[~visited], axis=0, dtype=np.float64)
        result[horizon, 4] = np.sum(belief[visited], dtype=np.float64)
        _output(result[horizon])
    return result
