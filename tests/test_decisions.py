import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from openjev.decisions import (
    Candidate,
    DecisionRequest,
    DecisionService,
    MLXScorer,
    ModelBusy,
    ModelUnavailable,
    Question,
    messages_for,
    summarize_logits,
)
from openjev.domain import Observation
from openjev.doom_adapter import ACTIONS, LanguageDoomPolicy, doom_request
from openjev.server import Control, Session, app


def request():
    return DecisionRequest(
        context="The appointment is Monday.",
        questions=[
            Question(
                id="day",
                question="Which day?",
                candidates=[
                    Candidate(id="m", description="Monday"),
                    Candidate(id="t", description="Tuesday"),
                ],
            )
        ],
    )


def test_request_rejects_ambiguous_and_unbounded_inputs():
    for change in [
        lambda d: d.update(context=" "),
        lambda d: d.update(context="x" * 12001),
        lambda d: d.update(questions=d["questions"] * 2),
        lambda d: d["questions"][0]["candidates"][1].update(id="m"),
        lambda d: d["questions"][0]["candidates"][1].update(description="  MONDAY "),
        lambda d: d["questions"][0].update(candidates=[]),
        lambda d: d["questions"][0].update(id="<script>"),
        lambda d: d.update(model="https://attacker/model"),
    ]:
        data = request().model_dump()
        change(data)
        with pytest.raises(ValidationError):
            DecisionRequest.model_validate(data)


def test_candidate_order_and_caller_ids_do_not_change_model_prompt():
    r = request()
    q = r.questions[0]
    messages, ordered = messages_for(r.context, q)
    q.candidates.reverse()
    q.candidates[0].id = "other_id"
    assert messages_for(r.context, q)[0] == messages
    assert [c.description for c in ordered] == ["Monday", "Tuesday"]


def test_logits_return_only_candidate_distribution_and_report_excluded_mass():
    q = request().questions[0]
    a = summarize_logits([1, 2, 10], [0, 1], q.candidates, q.id, 3, 1)
    assert a.choice == "t"
    assert set(a.probabilities) == {"m", "t"}
    assert sum(a.probabilities.values()) == pytest.approx(1)
    assert a.candidate_token_mass < 0.001
    assert a.entropy_nats > 0
    with pytest.raises(ValueError):
        summarize_logits([1, np.nan], [0, 1], q.candidates, q.id, 3, 1)


class TestScorer:
    __test__ = False

    def score(self, r):
        return [
            summarize_logits(
                [2, 1],
                [0, 1],
                messages_for(r.context, q)[1],
                q.id,
                20,
                1,
            )
            for q in r.questions
        ]


def test_api_contract_and_protection_without_model_download():
    with TestClient(app) as client:
        old = app.state.decisions
        service = DecisionService(TestScorer)
        app.state.decisions = service
        try:
            assert client.post("/api/decide", json=request().model_dump()).status_code == 403
            assert (
                client.post(
                    "/api/decide",
                    headers={"x-openjev": "1", "origin": "https://other.test"},
                    json=request().model_dump(),
                ).status_code
                == 403
            )
            response = client.post("/api/decide", headers={"x-openjev": "1"}, json=request().model_dump())
            assert response.status_code == 200
            assert response.json()["answers"][0]["choice"] == "m"
            assert response.json()["generated_tokens"] == 0
            assert client.get("/api/model").json()["status"] == "ready"
        finally:
            app.state.decisions = old
            service.close()


def test_missing_model_is_explicit_not_a_rule_fallback():
    def missing():
        raise ModelUnavailable("not installed")

    service = DecisionService(missing)
    try:
        with pytest.raises(ModelUnavailable):
            service.decide(request())
        assert service.status()["status"] == "unavailable"
    finally:
        service.close()


def test_token_limit_rejects_instead_of_truncating_or_scoring():
    class TooLongTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "rendered"

        def encode(self, *args, **kwargs):
            return [0] * 4097

    scorer = MLXScorer.__new__(MLXScorer)
    scorer.tokenizer = TooLongTokenizer()
    with pytest.raises(ValueError, match="nothing was truncated"):
        scorer.score(request())


def test_model_has_bounded_queue_and_single_owner():
    entered, release = threading.Event(), threading.Event()
    owners = []

    class Blocking(TestScorer):
        def score(self, r):
            owners.append(threading.get_ident())
            entered.set()
            assert release.wait(5)
            return super().score(r)

    service = DecisionService(Blocking)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            running = caller.submit(service.decide, request())
            assert entered.wait(5)
            # Reserve the sole waiting slot, modeling one queued HTTP request.
            assert service.slots.acquire(blocking=False)
            try:
                with pytest.raises(ModelBusy):
                    service.decide(request())
            finally:
                service.slots.release()
                release.set()
            running.result()
        service.decide(request())
        assert len(set(owners)) == 1
    finally:
        release.set()
        service.close()


def test_doom_adapter_uses_general_contract_and_joint_action_mapping():
    class DoomScorer:
        def score(self, r):
            assert "Never fire" in r.questions[0].question
            q = r.questions[0]
            ordered = sorted(q.candidates, key=lambda c: c.description)
            logits = [10 if c.id == "left_wait" else 0 for c in ordered]
            return [summarize_logits(logits, list(range(6)), ordered, q.id, 20, 1)]

    service = DecisionService(DoomScorer)
    try:
        policy = LanguageDoomPolicy(service, "Never fire")
        obs = Observation(visible=True, ammo=10)
        r = doom_request(obs, "Never fire", [])
        assert {c.id for c in r.questions[0].candidates} == set(ACTIONS)
        decision = policy.decide(obs)
        assert decision.steer == "left" and not decision.fire
        assert decision.probabilities["left"] > 0.99
        assert len(policy.history) == 1
        policy.reset()
        assert len(policy.history) == 0 and policy.last_response is None
    finally:
        service.close()


def test_pause_cancels_inflight_language_action():
    entered, release = threading.Event(), threading.Event()

    class SlowScorer:
        def score(self, r):
            entered.set()
            assert release.wait(5)
            q = r.questions[0]
            ordered = sorted(q.candidates, key=lambda c: c.description)
            logits = [10 if c.id == "hold_fire" else 0 for c in ordered]
            return [summarize_logits(logits, list(range(6)), ordered, q.id, 20, 1)]

    service = DecisionService(SlowScorer)
    session = Session(service)
    session.control(Control(command="start", policy="language"))
    session.thread.start()
    try:
        assert entered.wait(5)
        session.control(Control(command="pause"))
        release.set()
        until = time.monotonic() + 5
        while time.monotonic() < until:
            with session.lock:
                snapshot = session.snapshot.copy()
            if snapshot.get("status") == "paused":
                break
            time.sleep(0.02)
        assert snapshot["status"] == "paused"
        assert snapshot["decisions"] == 0
        assert snapshot["decision"] is None
        assert snapshot["stats"]["ammo"] == 26
        assert snapshot["language_response"] is None  # Discard the canceled history/response.
    finally:
        release.set()
        session.stop.set()
        session.thread.join(5)
        service.close()
