import json

import httpx
import pytest

from openjev.domain import Observation
from openjev.policies import JevPolicy


def response():
    return {
        "model": "jev-latest",
        "answers": {
            "steer": {
                "type": "choice",
                "choice": "left",
                "confidence": 0.8,
                "probabilities": {"left": 0.9, "hold": 0.05, "right": 0.05},
            },
            "fire": {"type": "noul", "noul": 0.7},
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


def test_documented_api_contract_and_budget(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-fixture-not-a-secret")
    calls = []

    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert body["questions"]["steer"]["type"] == "choice"
        assert body["questions"]["fire"]["type"] == "noul"
        assert body["state"]["directive"] == "hunt"
        return httpx.Response(200, json=response())

    policy = JevPolicy(max_calls=1, client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        decision = policy.decide(Observation())
        assert decision.steer == "left" and decision.fire
        assert decision.provider_confidence == 0.8
        with pytest.raises(RuntimeError, match="budget"):
            policy.decide(Observation())
        assert len(calls) == 1
    finally:
        policy.close()


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevPolicy()


@pytest.mark.parametrize("status", [401, 429, 500, 529])
def test_http_errors_are_not_retried_or_echoed(monkeypatch, status):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-fixture-not-a-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, text="private body"))
    )
    policy = JevPolicy(client=client)
    try:
        with pytest.raises(RuntimeError) as error:
            policy.decide(Observation())
        assert "private body" not in str(error.value)
        assert policy.calls == 1
    finally:
        policy.close()


def test_invalid_remote_enum_is_rejected(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-fixture-not-a-secret")
    data = response()
    data["answers"]["steer"]["choice"] = "shell_command"
    policy = JevPolicy(
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=data)))
    )
    try:
        with pytest.raises(ValueError, match="typed action"):
            policy.decide(Observation())
    finally:
        policy.close()
