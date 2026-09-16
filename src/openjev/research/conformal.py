"""Split conformal sets for a frozen candidate scorer and labeled task.

The exchangeability assumption is external. Adjacent Doom frames do not meet it
merely because this code computes a quantile. Never interpret sets as safety.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass


def validate_probabilities(probabilities: Mapping[str, float]):
    if not probabilities or any(not isinstance(k, str) or not k for k in probabilities):
        raise ValueError("Nonempty candidate IDs and probabilities are required")
    values = list(probabilities.values())
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
        raise ValueError("Probabilities must be finite and between zero and one")
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Candidate probabilities must sum to one")


@dataclass(frozen=True)
class LabeledDecision:
    unit_id: str
    probabilities: Mapping[str, float]
    correct_id: str


@dataclass(frozen=True)
class SplitConformal:
    alpha: float
    threshold: float
    calibration_units: frozenset[str]
    scorer_signature: str
    task_signature: str

    @classmethod
    def fit(cls, examples, *, alpha, scorer_signature, task_signature):
        """One labeled decision per independently sampled calibration unit.

        Freeze model, prompt, candidate construction, and task before fitting.
        If units are episodes, preselect one timestep by a frozen sampling rule.
        """
        if not 0 < alpha < 1 or not scorer_signature or not task_signature:
            raise ValueError("Require alpha in (0, 1) and nonempty frozen signatures")
        scores, units = [], set()
        for row in examples:
            if not row.unit_id or row.unit_id in units:
                raise ValueError("One calibration example per unique, nonempty unit ID is required")
            validate_probabilities(row.probabilities)
            if row.correct_id not in row.probabilities:
                raise ValueError("Correct candidate must be present in every calibration example")
            units.add(row.unit_id)
            scores.append(1 - row.probabilities[row.correct_id])
        if not scores:
            raise ValueError("Calibration set must be nonempty")
        rank = math.ceil((len(scores) + 1) * (1 - alpha))
        # The +infinity order statistic becomes 1 since scores are bounded by 1.
        threshold = 1.0 if rank > len(scores) else sorted(scores)[rank - 1]
        return cls(alpha, threshold, frozenset(units), scorer_signature, task_signature)

    def predict(self, probabilities, *, unit_id, scorer_signature, task_signature):
        if scorer_signature != self.scorer_signature or task_signature != self.task_signature:
            raise ValueError("Frozen scorer or task changed; refit calibration on a separate split")
        if not unit_id or unit_id in self.calibration_units:
            raise ValueError("Prediction unit must be nonempty and held out from calibration")
        validate_probabilities(probabilities)
        # Ties are included; an empty set is valid and is never forced to argmax.
        return tuple(k for k, p in probabilities.items() if 1 - p <= self.threshold)


def semantic_group_entropy(probabilities, groups):
    """Entropy after summing declared equivalent candidates, not sampled-text SE.

    Group labels must come from an external, frozen equivalence definition.
    A grouping mistake is not statistical evidence of low uncertainty.
    """
    validate_probabilities(probabilities)
    if set(groups) != set(probabilities) or any(not g for g in groups.values()):
        raise ValueError("Every candidate needs exactly one nonempty semantic group")
    mass = {}
    for key, p in probabilities.items():
        mass[groups[key]] = mass.get(groups[key], 0.0) + p
    return -sum(p * math.log(p) for p in mass.values() if p > 0)
