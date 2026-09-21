"""Bounded explicit observation-backup targets from an externally frozen value.

This is numerical target construction, not a learning loop or convergence
claim. Every state uses one unchanged float64 sixteen-row FrozenValue call.
The target uses the true eligible minimum, independently of near-tie policy
selection. No checkpoint, corpus, environment or optimizer is opened here.
"""
from dataclasses import dataclass

import numpy as np

from openjev.research.otto_return_value import SCALE, FrozenValue, value_features
from openjev.research.otto_value_branches import EPSILON, explicit_scores, rl_branches, select_action

MAX_STATES = 32


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _owned(array):
    """Independent immutable C-order storage, including boolean/integer fields."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class BellmanTargets:
    """All branch witnesses, without retaining dense branch features or history.

    For B states, masses/weights/contributions are Bx4x4; values are Bx16 in
    action-major, hit-minor order; costs/masks Bx4; successors Bx16x2. Other
    arrays are length B. Floating witnesses are float64 except the explicit
    final float32 training cast. Values, contributions and costs are physical
    search costs; normalized training targets divide the minimum cost by64.
    """

    raw_masses: np.ndarray
    weights: np.ndarray
    branch_values: np.ndarray
    branch_contributions: np.ndarray
    successors: np.ndarray
    costs: np.ndarray
    eligible_masks: np.ndarray
    minimum_costs: np.ndarray
    targets_float64: np.ndarray
    targets_float32: np.ndarray
    minimum_actions: np.ndarray
    deployed_actions: np.ndarray
    deployed_cost_gaps: np.ndarray

    def counts(self):
        states = len(self.targets_float64)
        return {"states": states, "branch_constructions": states, "checkpoint_call_invocations": states,
                "evaluated_branch_rows": 16 * states, "action_costs": 4 * states,
                "training_target_casts": states, "optimizer_steps": 0}

    def diagnostics(self):
        """Describe all rows, including blocked actions and zero-mass branches."""
        cast_error = np.abs(self.targets_float32.astype(np.float64) - self.targets_float64)
        return {"scope": "All supplied states and all16 branches/four costs per state; no outcome selection.",
                "zero_raw_mass_branches": int(np.count_nonzero(self.raw_masses == 0)),
                "positive_subfloor_branches": int(np.count_nonzero((self.raw_masses > 0) & (self.raw_masses < EPSILON))),
                "negative_branch_values": int(np.count_nonzero(self.branch_values < 0)),
                "negative_action_costs": int(np.count_nonzero(self.costs < 0)),
                "negative_targets": int(np.count_nonzero(self.targets_float64 < 0)),
                "minimum_branch_value": float(self.branch_values.min()),
                "maximum_branch_value": float(self.branch_values.max()),
                "minimum_target": float(self.targets_float64.min()),
                "maximum_target": float(self.targets_float64.max()),
                "maximum_float32_cast_absolute_error": float(cast_error.max()),
                "deployed_action_differs_from_first_minimizer": int(np.count_nonzero(self.deployed_actions != self.minimum_actions)),
                "maximum_deployed_cost_gap": float(self.deployed_cost_gaps.max()),
                "retained_array_bytes": sum(getattr(self, name).nbytes for name in self.__slots__)}


def build_targets(frozen_value, beliefs, positions, kernels, sensing_lengths, eligible_actions, *, observe=None):
    """Build B<=32 targets; every state is evaluated with the same16-row shape.

    beliefs: float64[B,53,53]; positions: integer[B,2]. kernels is a sequence
    of B float64[4,107,107] public kernels (references may be shared). Sensing
    lengths and eligible-action sequences each have B entries. The head is an
    immutable FrozenValue created by the caller; its optional bound sensing
    scalar is unused because each state's supplied scalar enters its features.

    The inherited branch builder checks public mass, kernel origin and bounds;
    eligible actions affect selection only. Zero/subfloor branches, signed
    predictions and biased zero-input values remain present. No normalization,
    output floor, discount, terminal shortcut or fused homogeneous route.

    Optional observe(event, state_index, rows) receives only immutable scalar
    metadata: ("attempt", index,16) immediately before normalized(), and
    ("return", index,16) immediately after it returns. It receives no arrays;
    its return value is ignored. A caller can persist partial progress if a
    later state fails. Observation failures propagate without retry. Returned
    counts describe completed successful batches, not a failed partial call.
    """
    if not isinstance(frozen_value, FrozenValue):
        raise TypeError("target network must be an externally constructed FrozenValue")
    if observe is not None and not callable(observe):
        raise TypeError("observe must be callable or None")
    _require(isinstance(beliefs, np.ndarray) and beliefs.dtype == np.float64 and beliefs.ndim == 3
             and beliefs.shape[1:] == (53, 53) and 0 < len(beliefs) <= MAX_STATES,
             "bounded float64[B,53,53] beliefs required")
    _require(np.isfinite(beliefs).all() and (beliefs >= 0).all(), "invalid public beliefs")
    count = len(beliefs)
    positions = np.asarray(positions)
    _require(positions.shape == (count, 2) and positions.dtype.kind in "iu"
             and (positions >= 0).all() and (positions < 53).all(), "integer positions[B,2] in0..52 required")
    for name, sequence in (("kernels", kernels), ("sensing_lengths", sensing_lengths), ("eligible_actions", eligible_actions)):
        _require(isinstance(sequence, (list, tuple, np.ndarray)) and len(sequence) == count,
                 f"{name} must have exactly B entries")

    arrays = {"raw_masses": np.empty((count, 4, 4), dtype=np.float64),
              "weights": np.empty((count, 4, 4), dtype=np.float64),
              "branch_values": np.empty((count, 16), dtype=np.float64),
              "branch_contributions": np.empty((count, 4, 4), dtype=np.float64),
              "successors": np.empty((count, 16, 2), dtype=np.int64),
              "costs": np.empty((count, 4), dtype=np.float64),
              "eligible_masks": np.zeros((count, 4), dtype=bool),
              "minimum_costs": np.empty(count, dtype=np.float64),
              "minimum_actions": np.empty(count, dtype=np.int64),
              "deployed_actions": np.empty(count, dtype=np.int64),
              "deployed_cost_gaps": np.empty(count, dtype=np.float64)}
    for index in range(count):
        branches = rl_branches(beliefs[index], positions[index], kernels[index], eligible_actions[index])

        def physical_values(centered_z, successors, _kernel, index=index):
            features = value_features(centered_z, successors, sensing_lengths[index], dtype="float64")
            if observe is not None:
                observe("attempt", index, 16)
            normalized = frozen_value.normalized(features)
            if observe is not None:
                observe("return", index, 16)
            with np.errstate(over="raise", invalid="raise"):
                physical = SCALE * normalized
            _require(physical.shape == (16,) and physical.dtype == np.float64 and np.isfinite(physical).all(),
                     "finite float64[16] physical branch values required")
            arrays["branch_values"][index] = physical
            return physical

        costs = explicit_scores(branches, physical_values, arithmetic="float64")
        eligible = list(branches.eligible_actions)
        minimum = np.min(costs[eligible])
        first_minimizer = next(a for a in eligible if costs[a] == minimum)
        deployed = select_action(costs, branches.eligible_actions)
        arrays["raw_masses"][index], arrays["weights"][index] = branches.raw_masses, branches.weights
        arrays["successors"][index], arrays["costs"][index] = branches.successors, costs
        arrays["eligible_masks"][index, eligible] = True
        arrays["minimum_costs"][index], arrays["minimum_actions"][index] = minimum, first_minimizer
        arrays["deployed_actions"][index], arrays["deployed_cost_gaps"][index] = deployed, costs[deployed] - minimum
        with np.errstate(over="raise", invalid="raise"):
            arrays["branch_contributions"][index] = branches.weights * arrays["branch_values"][index].reshape(4, 4)

    with np.errstate(over="raise", invalid="raise"):
        arrays["targets_float64"] = arrays["minimum_costs"] / SCALE
        arrays["targets_float32"] = arrays["targets_float64"].astype(np.float32)
    _require(all(np.isfinite(array).all() for array in arrays.values()), "nonfinite target or witness")
    result = object.__new__(BellmanTargets)
    for name, array in arrays.items():
        object.__setattr__(result, name, _owned(array))
    return result
