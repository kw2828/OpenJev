"""Bounded public-belief planning for raw, undiscounted RockSample reward.

This is a classical finite candidate planner, not an optimal POMDP solver.
Four fixed serpentine tours supply exploitation continuations. Sensing evaluates
five fixed bundles at each of at most five public positions, with exact binary
outcome enumeration under the supplied belief. No simulator state is accepted.
A tour forecast includes its feasible exit; execution commits only the first
sample or the complete sensing bundle, so later replanning may revise the tour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

CHECK_MARGIN = 0.05
MAX_CHECKS = 64
EXIT_REWARD = 10.0


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _position(value, size):
    array = np.asarray(value)
    if (array.shape != (2,) or array.dtype.kind not in "iu"
            or np.any(array < 0) or array[0] >= size or array[1] >= size - 1):
        raise ValueError("position must be two integer coordinates outside the terminal column")
    return tuple(int(v) for v in array)


def _moves(start, end):
    """Row first, then column; both internal endpoints exclude the exit column."""
    dy, dx = end[0] - start[0], end[1] - start[1]
    return (2 if dy > 0 else 0,) * abs(dy) + (1 if dx > 0 else 3,) * abs(dx)


def _entropy(probability):
    if probability == 0 or probability == 1:
        return 0.0
    return -probability * math.log(probability) - (1 - probability) * math.log1p(-probability)


def _probability(belief, rock, position, work):
    work["check_probability_calls"] += 1
    probability = belief.check_probability(rock, position)
    if (isinstance(probability, (bool, np.bool_)) or not np.isscalar(probability)
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("check probability must be finite and in [0,1]")
    return float(probability)


def _routes(size):
    base = [(row, col) for row in range(size)
            for col in (range(size - 1) if row % 2 == 0 else range(size - 2, -1, -1))]
    return tuple(np.asarray([(size - 1 - row if flip_row else row,
                              size - 2 - col if flip_col else col) for row, col in base], dtype=np.int64)
                 for flip_row, flip_col in ((False, False), (False, True), (True, False), (True, True)))


@dataclass(frozen=True)
class _Option:
    kind: str
    value: float
    actions: tuple[int, ...]
    target: tuple[int, int] | None = None
    vantage: tuple[int, int] | None = None
    rocks: tuple[int, ...] = ()
    tour: int | None = None
    planned_steps: int | None = None

    def tie_key(self):
        return len(self.actions), self.actions, self.tour if self.tour is not None else -1


def _better(candidate, incumbent):
    return candidate.value > incumbent.value or (
        candidate.value == incumbent.value and candidate.tie_key() < incumbent.tie_key())


def _non_sensing(belief, position, sampled, remaining, routes, work):
    size = sampled.shape[0]
    distance_to_exit = size - 1 - position[1]
    best = _Option("exit", EXIT_REWARD if remaining >= distance_to_exit else 0.,
                   (1,) * min(distance_to_exit, remaining), planned_steps=min(distance_to_exit, remaining))
    work["non_sensing_calls"] += 1
    work["expected_reward_grid_calls"] += 1
    grid = np.asarray(belief.expected_rewards(), dtype=np.float64)
    if grid.shape != sampled.shape or not np.isfinite(grid).all():
        raise ValueError("expected rewards must be a finite size x (size-1) grid")
    # Negative, zero and previously sampled cells are never sample targets.
    eligible = (grid > 0) & ~sampled
    for index, route in enumerate(routes):
        work["tour_evaluations"] += 1
        cells = route[eligible[route[:, 0], route[:, 1]]]
        work["positive_target_visits_considered"] += len(cells)
        if not len(cells):
            continue
        previous = np.vstack((np.asarray(position), cells[:-1]))
        distances = np.abs(cells - previous).sum(axis=1)
        steps = np.cumsum(distances + 1) + size - 1 - cells[:, 1]
        values = np.cumsum(grid[cells[:, 0], cells[:, 1]]) + EXIT_REWARD
        feasible = np.flatnonzero(steps <= remaining)
        work["feasible_tour_prefixes"] += len(feasible)
        if not len(feasible):
            continue
        # Positive rewards make the last feasible prefix the highest-valued
        # prefix. Extra sample/travel costs cannot decrease total path+exit cost.
        last = int(feasible[-1])
        target = tuple(int(v) for v in cells[0])
        option = _Option("exploit", float(values[last]), _moves(position, target) + (4,),
                         target=target, tour=index, planned_steps=int(steps[last]))
        if _better(option, best):
            best = option
    return best


def _bundles(top):
    """Frozen family: top1, mixed top2/top4, repeated top1 twice/four times."""
    candidates = [(top[0],)]
    if len(top) >= 2:
        candidates.append(tuple(top[:2]))
    if len(top) >= 4:
        candidates.append(tuple(top[:4]))
    candidates.extend(((top[0],) * 2, (top[0],) * 4))
    return tuple(dict.fromkeys(candidates))


def _sensing_value(belief, vantage, rocks, sampled, remaining, routes, work):
    leaves = []

    def branch(current, depth, weight):
        if depth == len(rocks):
            work["outcome_leaves"] += 1
            continuation = _non_sensing(current, vantage, sampled, remaining, routes, work)
            leaves.append((weight, continuation.value))
            return
        probability = _probability(current, rocks[depth], vantage, work)
        for positive, likelihood in ((False, 1 - probability), (True, probability)):
            work["outcome_branches_considered"] += 1
            if likelihood == 0:
                work["zero_probability_branches_skipped"] += 1
                continue
            child_weight = weight * likelihood
            if child_weight == 0:
                # Do not silently discard positive-mass branches.
                raise FloatingPointError("positive sensing branch weight underflowed")
            work["condition_check_calls"] += 1
            child = current.condition_check(rocks[depth], vantage, positive)
            if child is current:
                raise ValueError("condition_check must return an independent belief copy")
            branch(child, depth + 1, child_weight)

    branch(belief, 0, 1.)
    total = math.fsum(weight for weight, _ in leaves)
    if not math.isclose(total, 1., rel_tol=0., abs_tol=1e-12):
        raise ArithmeticError("sensing outcome weights do not sum to one")
    work["maximum_branch_mass_error"] = max(work["maximum_branch_mass_error"], abs(total - 1.))
    work["maximum_leaves_per_bundle"] = max(work["maximum_leaves_per_bundle"], len(leaves))
    # No normalization or conditioning on favorable outcomes is applied here.
    return math.fsum(weight * value for weight, value in leaves)


def plan_action(belief, position, sampled, remaining_steps, checks_remaining, *, check_margin=CHECK_MARGIN):
    """Return a JSON-compatible primitive macro and bounded work diagnostics.

    ``sampled`` is a persistent boolean [size,size-1] ledger shared by every
    controller. ``checks_remaining`` is the unused portion of a global budget of
    64. Belief must expose size/rocks, expected_rewards(), check_probability(rock,
    position), and condition_check(rock,position,positive)->independent copy.
    Binary signs are enumerated sequentially; positive is a bool, never +/-1.

    A sensing macro commits travel plus the entire predetermined check bundle;
    its forecasts use the post-bundle exploitation/exit value. An exploitation
    macro commits only travel to its FIRST sample target and the sample action.
    Native reward is undiscounted: no gamma or implicit per-step reward penalty.
    If the exit is unreachable in the remaining horizon, use the available east
    moves for zero forecast reward. Tours always reserve enough time to exit.
    """
    size = _integer(belief.size, "size", 2)
    rocks = _integer(belief.rocks, "rocks", 1)
    position = _position(position, size)
    remaining = _integer(remaining_steps, "remaining_steps")
    check_budget = _integer(checks_remaining, "checks_remaining")
    if check_budget > MAX_CHECKS:
        raise ValueError("checks_remaining exceeds the fixed global cap of 64")
    if type(check_margin) not in (int, float) or not math.isfinite(check_margin) or check_margin < 0:
        raise ValueError("check_margin must be finite and nonnegative")
    sampled = np.asarray(sampled)
    if sampled.dtype != np.bool_ or sampled.shape != (size, size - 1):
        raise ValueError("sampled must be a boolean size x (size-1) public ledger")
    work = dict.fromkeys(("non_sensing_calls", "expected_reward_grid_calls", "tour_evaluations",
                         "positive_target_visits_considered", "feasible_tour_prefixes",
                         "vantages_considered", "vantages_feasible", "entropy_probability_calls",
                         "bundle_candidates", "bundles_evaluated", "bundles_infeasible",
                         "check_probability_calls", "condition_check_calls", "outcome_branches_considered",
                         "zero_probability_branches_skipped", "outcome_leaves", "maximum_leaves_per_bundle"), 0)
    work["maximum_branch_mass_error"] = 0.
    routes = _routes(size)
    baseline = _non_sensing(belief, position, sampled, remaining, routes, work)
    sensing = None
    positions = tuple(dict.fromkeys((position, (0, 0), (0, size - 2), (size - 1, 0), (size - 1, size - 2))))
    for vantage in positions:
        work["vantages_considered"] += 1
        travel = _moves(position, vantage)
        if check_budget == 0 or len(travel) + 1 + size - 1 - vantage[1] > remaining:
            continue
        work["vantages_feasible"] += 1
        entropies = []
        for rock in range(rocks):
            probability = _probability(belief, rock, vantage, work)
            work["entropy_probability_calls"] += 1
            entropies.append(_entropy(probability))
        top = sorted(range(rocks), key=lambda rock: (-entropies[rock], rock))[:4]
        for bundle in _bundles(top):
            work["bundle_candidates"] += 1
            duration = len(travel) + len(bundle)
            if len(bundle) > check_budget or duration + size - 1 - vantage[1] > remaining:
                work["bundles_infeasible"] += 1
                continue
            work["bundles_evaluated"] += 1
            value = _sensing_value(belief, vantage, bundle, sampled, remaining - duration, routes, work)
            candidate = _Option("sense", value, travel + tuple(5 + rock for rock in bundle),
                                vantage=vantage, rocks=bundle)
            if sensing is None or _better(candidate, sensing):
                sensing = candidate
    chosen = baseline
    if sensing is not None and sensing.value > baseline.value + check_margin:
        chosen = sensing
    if work["bundles_evaluated"] > 25 or work["outcome_leaves"] > 210 or work["maximum_leaves_per_bundle"] > 16:
        raise AssertionError("fixed candidate family exceeded its expansion bound")
    if work["tour_evaluations"] != 4 * (1 + work["outcome_leaves"]):
        raise AssertionError("each continuation must evaluate the same four tours")
    return {"kind": chosen.kind, "actions": list(chosen.actions), "committed_steps": len(chosen.actions),
            "expected_return": chosen.value, "non_sensing_return": baseline.value,
            "sensing_gain": chosen.value - baseline.value if chosen.kind == "sense" else 0.,
            "best_sensing_return": sensing.value if sensing is not None else None,
            "best_sensing_gain": sensing.value - baseline.value if sensing is not None else None,
            "first_sample_target": list(chosen.target) if chosen.target is not None else None,
            "sensing_position": list(chosen.vantage) if chosen.vantage is not None else None,
            "check_rocks": list(chosen.rocks), "checks_committed": len(chosen.rocks),
            "selected_tour": chosen.tour, "forecast_tour_steps": chosen.planned_steps,
            "check_margin": float(check_margin), "work": work,
            "scope": "Raw undiscounted finite candidate forecast; only the returned primitive macro is committed."}
