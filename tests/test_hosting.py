"""Hosting boundaries and backend identity, without downloading weights."""

import pytest
from fastapi.testclient import TestClient

from openjev import server
from openjev.decisions import CPU_MODEL_ID, CPU_MODEL_REVISION, CPUScorer, DecisionRequest, DecisionService

PAYLOAD = {
    "context": "The appointment is Monday.",
    "questions": [{"id": "day", "question": "Which day?", "candidates": [
        {"id": "m", "description": "Monday"}, {"id": "t", "description": "Tuesday"},
    ]}],
}


def test_cpu_backend_reports_its_own_identity(monkeypatch):
    monkeypatch.setenv("OPENJEV_BACKEND", "cpu")

    class Scorer:
        def score(self, request):
            return []

    service = DecisionService(Scorer)
    try:
        result = service.decide(DecisionRequest.model_validate(PAYLOAD))
        assert result.model == CPU_MODEL_ID
        assert result.revision == CPU_MODEL_REVISION
        assert result.backend == "transformers-cpu-float32"
        assert service.status()["model"] == CPU_MODEL_ID
    finally:
        service.close()
    monkeypatch.setenv("OPENJEV_BACKEND", "typo")
    with pytest.raises(ValueError, match="mlx or cpu"):
        DecisionService()


def test_cpu_rejects_overlong_input_before_inference():
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["return_dict"] is False
            return [0] * 4097

    scorer = CPUScorer.__new__(CPUScorer)
    scorer.tokenizer = Tokenizer()
    with pytest.raises(ValueError, match="nothing was truncated"):
        scorer.score(DecisionRequest.model_validate(PAYLOAD))


def test_proxy_origin_and_paid_api_protection(monkeypatch):
    monkeypatch.setattr(server, "PUBLIC_ORIGIN", "https://demo.example.org")
    monkeypatch.setattr(server, "PUBLIC_DEMO", True)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-not-a-real-key")
    with TestClient(server.app) as client:
        headers = {"X-OpenJev": "1", "Origin": "https://demo.example.org"}
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/api/model").json()["public_demo"] is True
        assert client.post("/api/control", json={"command": "pause"}, headers=headers).status_code == 200
        assert client.post("/api/control", json={"command": "start", "policy": "jev"}, headers=headers).status_code == 403
        headers["Origin"] = "https://attacker.example"
        headers["X-Forwarded-Host"] = "attacker.example"
        assert client.post("/api/control", json={"command": "pause"}, headers=headers).status_code == 403
        assert client.get("/healthz", headers={"Host": "attacker.example"}).status_code == 400
