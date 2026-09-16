"""Doom is a consumer of the general decision contract, not its schema."""

import json
from collections import deque

from .decisions import Candidate, DecisionRequest, Question
from .domain import STEERING, Decision

DEFAULT_INSTRUCTION = "Aim at visible enemies and fire when aligned. Keep scanning when no enemy is visible."
ACTIONS = {f"{s}_{f}": (s, f == "fire") for s in STEERING for f in ("wait", "fire")}


def doom_request(obs, instruction, history):
    motion = "strafe" if obs.scenario == "basic" else "turn"
    candidates = [
        Candidate(
            id=key,
            description=(f"{motion.capitalize()} {steer}" if steer != "hold" else "Keep the current aim")
            + (" and fire the weapon" if fire else " without firing"),
        )
        for key, (steer, fire) in ACTIONS.items()
    ]
    return DecisionRequest(
        context=json.dumps(
            {
                "observation": obs.to_dict(),
                "recent_steps": list(history),
                "controls": f"Only {motion} left/right, hold, and fire are executable. No forward movement or cover action.",
                "coordinates": "aim_error is negative when the enemy is left, positive when right, zero at crosshair.",
            }
        ),
        questions=[
            Question(
                id="doom_action",
                question=f"Which next action best follows this objective? {instruction}",
                candidates=candidates,
            )
        ],
    )


class LanguageDoomPolicy:
    name = "language"

    def __init__(self, service, instruction=DEFAULT_INSTRUCTION, owns_service=False):
        self.service = service
        self.instruction = instruction
        self.history = deque(maxlen=4)
        self.last_response = None
        self.owns_service = owns_service

    def reset(self):
        self.history.clear()
        self.last_response = None

    def decide(self, obs):
        response = self.service.decide(doom_request(obs, self.instruction, self.history))
        self.last_response = response.model_dump()
        answer = response.answers[0]
        steer, fire = ACTIONS[answer.choice]
        p = {s: sum(v for k, v in answer.probabilities.items() if ACTIONS[k][0] == s) for s in STEERING}
        firing = sum(v for k, v in answer.probabilities.items() if ACTIONS[k][1])
        self.history.append({"observation": obs.to_dict(), "selected_action": answer.choice})
        return Decision(steer, fire, p, firing, self.name, response.latency_ms)

    def close(self):
        if self.owns_service:
            self.service.close()
