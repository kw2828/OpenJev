"""Odor-history controls on a separate public-only OTTO model.

The live sampled-source environment is never passed to this class. The supplied
model must be constructed with draw_source=False and an explicit initial hit.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def normalized(probability):
    p = np.asarray(probability, dtype=np.float64)
    mass = float(p.sum())
    if not np.isfinite(p).all() or not np.isfinite(mass) or np.any(p < 0) or not mass > 0:
        raise ValueError("invalid public posterior")
    return p / mass


class OdorMemory:
    def __init__(self, model, policy_class, initial, window, policy_index):
        if model.draw_source or hasattr(model, "source"):
            raise ValueError("actor must use a separate public-only model")
        if window is not None and (type(window) is not int or window < 0):
            raise ValueError("window must be None or a nonnegative integer")
        if (initial.step != 0 or initial.done or initial.hit != model.initial_hit
                or tuple(model.agent) != initial.position):
            raise ValueError("initial public packet disagrees with actor initialization")
        if policy_index not in (0, 1):
            raise ValueError("only qualified one-step infotaxis policies are supported")
        self.model = model
        self.policy = policy_class(model, policy=policy_index, steps_ahead=1)
        self.policy_index = policy_index
        self.window = window
        self.prior = model.p_source.copy()
        self.visited = {initial.position}
        self.recent = deque(maxlen=window or 0)
        self.step = 0
        self.done = False
        self.validate()

    def observe(self, packet):
        if self.done or packet.step != self.step + 1:
            raise ValueError("public observations must be consecutive and stop at termination")
        if (len(packet.position) != self.model.Ndim
                or any(type(x) is not int or not 0 <= x < self.model.N for x in packet.position)
                or (packet.done and packet.hit != -2)
                or (not packet.done and not 0 <= packet.hit < self.model.Nhits)):
            raise ValueError("invalid public observation")
        self.model.agent = list(packet.position)
        self.step = packet.step
        self.done = packet.done
        if packet.done:
            self.model._update_after_hit(hit=-2, done=True)
        else:
            self.visited.add(packet.position)
            if self.window is None:
                self.model._update_after_hit(hit=packet.hit, done=False)
            else:
                self.recent.append((packet.position, packet.hit))
                p = self.prior.copy()
                for position in self.visited:
                    p[position] = 0
                p = normalized(p)
                for position, hit in self.recent:
                    likelihood = self.model._extract_N_from_2N(self.model.p_Poisson, list(position))[hit]
                    p = normalized(p * likelihood)
                self.model.p_source = p
                self.model.entropy = self.model._entropy(p)
                self.model.obs = {"hit": packet.hit, "done": False}
        self.validate()

    def validate(self):
        p = self.model.p_source
        if (not np.isfinite(p).all() or np.any(p < 0)
                or abs(float(p.sum()) - 1) > 1e-10):
            raise ValueError("public posterior lost finite normalized mass")
        if not self.done and any(p[position] != 0 for position in self.visited):
            raise ValueError("non-found visited source cell has nonzero probability")

    def choose(self):
        if self.done:
            raise ValueError("no action is allowed after finding the source")
        if self.policy_index == 1:
            action, scores = self.policy._space_aware_infotaxis()
        else:
            action, scores = self.policy._infotaxis()
        return int(action), np.asarray(scores).copy()
