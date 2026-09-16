import time

from fastapi.testclient import TestClient

from openjev.server import app


def wait_for(client, predicate, timeout=6):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        state = client.get("/api/state").json()
        if predicate(state):
            return state
        time.sleep(0.03)
    raise AssertionError(f"State did not arrive: {state.get('status')}")


def test_browser_controls_and_origin_protection():
    headers = {"x-openjev": "1"}
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        initial = wait_for(client, lambda s: s["status"] == "paused")
        assert initial["frame"]
        assert client.post("/api/control", json={"command": "start"}).status_code == 403
        assert (
            client.post(
                "/api/control",
                headers={**headers, "origin": "https://example.org"},
                json={"command": "start"},
            ).status_code
            == 403
        )
        assert client.post("/api/control", headers=headers, json={"command": "start"}).status_code == 200
        active = wait_for(client, lambda s: s.get("decisions", 0) >= 3)
        assert active["decision"]["source"] == "local"
        client.post("/api/control", headers=headers, json={"command": "pause"})
        paused = wait_for(client, lambda s: s["status"] == "paused")
        time.sleep(0.15)
        assert client.get("/api/state").json()["decisions"] == paused["decisions"]
        client.post("/api/control", headers=headers, json={"command": "configure", "directive": "pacifist"})
        wait_for(client, lambda s: s["config"]["directive"] == "pacifist")
        client.post("/api/control", headers=headers, json={"command": "restart", "seed": 43})
        reset = wait_for(client, lambda s: s["config"]["seed"] == 43 and s["status"] == "running")
        assert reset["stats"]["kills"] == 0


def test_unknown_control_and_bad_host():
    with TestClient(app) as client:
        assert (
            client.post("/api/control", headers={"x-openjev": "1"}, json={"command": "shell"}).status_code
            == 422
        )
        assert client.get("/api/state", headers={"host": "attacker.example"}).status_code == 400
