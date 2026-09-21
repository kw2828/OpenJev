"""Selection-only restriction of the injected original OTTO value policy.

The model still scores all four actions, including blocked stays, exactly once.
Only final selection is restricted to the qualified public packet's in-bounds
directions. This is a changed policy, not a correction to the original action
contract. Public belief updates, reset and input ownership remain inherited.
No model runtime, simulator, native posterior, source or random seed is imported.
"""
from __future__ import annotations

import numpy as np

from openjev.research.otto_released_policy import EPSILON, ReleasedPolicyActor, _integer

VERSION = "otto-restricted-public-policy-v1"


def select_inbounds_action(scores, valid_actions):
    """Return the first near-minimum permitted action without changing costs.

    The subtraction and absolute-value test remain float32, matching original
    RLPolicy. This pure helper also accepts authenticated saved costs, without
    replaying a policy or inferring any counterfactual future observations.
    """
    if (not isinstance(scores, np.ndarray) or scores.shape != (4,) or scores.dtype != np.float32
            or not np.isfinite(scores).all()):
        raise ValueError("official policy must return four finite float32 costs")
    if not isinstance(valid_actions, (tuple, list)):
        raise TypeError("valid_actions must be an ordered public sequence")
    allowed = tuple(_integer(a, "valid action", 0, 3) for a in valid_actions)
    if not allowed or allowed != tuple(sorted(set(allowed))):
        raise ValueError("valid_actions must be nonempty, unique and in original action order")
    permitted = scores[list(allowed)]
    first = int(np.flatnonzero(np.abs(permitted - permitted.min()) < EPSILON)[0])
    return allowed[first]


class RestrictedPolicyActor(ReleasedPolicyActor):
    """One official score call followed by explicitly restricted selection.

    ``choose()`` returns ``(action, raw_float32_costs)`` with all four original
    costs unchanged. ``allowed_actions`` and ``selection_mask`` report selection
    eligibility separately. Only the restricted action becomes pending; calling
    choose twice or observing a different action is rejected by the inherited
    lifecycle. Prescribed public updates without a pending choice remain allowed
    for qualification, including a recorded blocked action from another policy.
    """

    __slots__ = ()

    @property
    def allowed_actions(self):
        return self._public["valid_actions"]

    @property
    def selection_mask(self):
        allowed = self.allowed_actions
        return tuple(a in allowed for a in range(4))

    def choose(self):
        if self._public["done"]:
            raise RuntimeError("cannot choose after source found")
        if self._pending_action is not None:
            raise RuntimeError("a chosen action is already awaiting its public outcome")
        # Do not call super().choose(): that would commit the unrestricted
        # action before this policy's single final selection.
        original_action, scores = self._policy._value_policy()
        original_action = _integer(original_action, "policy action", 0, 3)
        first = select_inbounds_action(scores, (0, 1, 2, 3))
        if original_action != first:
            raise ValueError("policy action disagrees with the upstream first near-tie rule")
        chosen = select_inbounds_action(scores, self.allowed_actions)
        raw_costs = scores.copy()
        self._pending_action = chosen
        return chosen, raw_costs
