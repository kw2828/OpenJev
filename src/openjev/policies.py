import os
import time
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np

from .domain import STEERING, Decision, Observation, hard_decision, teacher_action

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "doom.npz"


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class LocalPolicy:
    """One NumPy forward pass; no text decoder, no teacher fallback."""

    name = "local"

    def __init__(self, weights: Path = DEFAULT_WEIGHTS):
        with np.load(weights, allow_pickle=False) as data:
            self.weights = {k: data[k] for k in data.files}

    def logits(self, x):
        w = self.weights
        h = np.maximum(0, x @ w["w1"] + w["b1"])
        h = np.maximum(0, h @ w["w2"] + w["b2"])
        return h @ w["w3"] + w["b3"]

    def decide(self, obs: Observation) -> Decision:
        start = time.perf_counter()
        logits = self.logits(obs.features())
        p = softmax(logits[:3])
        fire = float(softmax(logits[3:])[1])
        return Decision(
            STEERING[int(p.argmax())],
            fire >= 0.5,
            dict(zip(STEERING, map(float, p))),
            fire,
            self.name,
            (time.perf_counter() - start) * 1000,
        )

    def close(self):
        pass


class RulePolicy:
    name = "rules"

    def decide(self, obs):
        start = time.perf_counter()
        decision = hard_decision(*teacher_action(obs), self.name)
        return replace(decision, latency_ms=(time.perf_counter() - start) * 1000)

    def close(self):
        pass


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def decide(self, obs):
        start = time.perf_counter()
        decision = Decision(
            str(self.rng.choice(STEERING)),
            bool(self.rng.integers(2)),
            {s: 1 / 3 for s in STEERING},
            0.5,
            self.name,
        )
        return replace(decision, latency_ms=(time.perf_counter() - start) * 1000)

    def close(self):
        pass


class JevPolicy:
    """Documented TypeSafe HTTP API. Explicit opt-in, bounded calls, no retries."""

    name = "jev"

    def __init__(self, *, max_calls=300, client=None):
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY on the server to enable Jev.")
        self.client = client or httpx.Client(timeout=httpx.Timeout(5, connect=2), follow_redirects=False)
        self.key = key
        self.calls = 0
        self.max_calls = max_calls

    def decide(self, obs):
        if self.calls >= self.max_calls:
            raise RuntimeError("Jev call budget reached. Restart the server to explicitly renew the budget.")
        self.calls += 1  # Reserve even when the outcome of a request is unknown.
        start = time.perf_counter()
        response = self.client.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": "jev-latest",
                "state": obs.to_dict(),
                "questions": {
                    "steer": {
                        "type": "choice",
                        "instructions": "Which direction should the Doom player steer to center the visible enemy? "
                        "aim_error is negative for left, positive for right; hold when abs(error)<0.07. "
                        "Scan right if no enemy is visible. In basic, steer means strafe.",
                        "criteria": {s: None for s in STEERING},
                    },
                    "fire": {
                        "type": "noul",
                        "instructions": "Should the player fire now? Only fire at a visible enemy near the crosshair "
                        "with ammo. Obey directive: pacifist never fires, conserve waits for a precise "
                        "shot, hunt fires whenever the enemy is approximately aligned.",
                    },
                },
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"TypeSafe returned HTTP {response.status_code}; run paused, no retry.")
        answers = response.json()["answers"]
        turn, fire = answers["steer"], answers["fire"]
        if turn["type"] != "choice" or fire["type"] != "noul":
            raise ValueError("Unexpected TypeSafe answer type")
        p = {k: float(v) for k, v in turn["probabilities"].items()}
        return Decision(
            turn["choice"],
            float(fire["noul"]) >= 0.5,
            p,
            float(fire["noul"]),
            self.name,
            (time.perf_counter() - start) * 1000,
            float(turn["confidence"]),
        )

    def close(self):
        self.client.close()


def make_policy(name, seed=0, instruction=None):
    if name == "language":
        from .decisions import DecisionService
        from .doom_adapter import DEFAULT_INSTRUCTION, LanguageDoomPolicy

        return LanguageDoomPolicy(DecisionService(), instruction or DEFAULT_INSTRUCTION, owns_service=True)
    if name == "local":
        return LocalPolicy()
    if name == "rules":
        return RulePolicy()
    if name == "random":
        return RandomPolicy(seed)
    if name == "jev":
        return JevPolicy()
    raise ValueError(f"Unknown policy: {name}")
