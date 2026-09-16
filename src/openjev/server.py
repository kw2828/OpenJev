import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .decisions import DecisionRequest, DecisionResponse, DecisionService, ModelBusy, ModelUnavailable, Text
from .domain import hard_decision
from .doom_adapter import DEFAULT_INSTRUCTION, LanguageDoomPolicy
from .game import Doom
from .policies import make_policy

STATIC = Path(__file__).parent / "static"
PUBLIC_DEMO = os.environ.get("OPENJEV_PUBLIC_DEMO") == "1"
# Explicit deployment configuration, never request-supplied forwarded headers.
PUBLIC_ORIGIN = os.environ.get("OPENJEV_PUBLIC_ORIGIN", "").rstrip("/")
if not PUBLIC_ORIGIN and os.environ.get("SPACE_HOST"):
    PUBLIC_ORIGIN = "https://" + os.environ["SPACE_HOST"]
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]
if PUBLIC_ORIGIN:
    parsed_origin = urlsplit(PUBLIC_ORIGIN)
    if (parsed_origin.scheme not in ("http", "https") or not parsed_origin.hostname
            or parsed_origin.path or parsed_origin.query or parsed_origin.fragment
            or parsed_origin.username or parsed_origin.password):
        raise ValueError("OPENJEV_PUBLIC_ORIGIN must be an http(s) origin without a path or credentials")
    ALLOWED_HOSTS.append(parsed_origin.hostname)



class Control(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Literal["start", "pause", "restart", "configure", "manual"]
    policy: Literal["local", "rules", "random", "jev", "manual", "language"] | None = None
    instruction: Text | None = Field(default=None, max_length=1500)
    scenario: Literal["defend_the_center", "defend_the_line", "basic"] | None = None
    directive: Literal["hunt", "conserve", "pacifist"] | None = None
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    steer: Literal["left", "hold", "right"] = "hold"
    fire: bool = False


class Session:
    """One engine owned by one worker; HTTP readers never touch ViZDoom."""

    def __init__(self, decision_service=None):
        self.decision_service = decision_service
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.config = {"policy": "local", "scenario": "defend_the_center", "directive": "hunt", "seed": 42}
        self.config["instruction"] = DEFAULT_INSTRUCTION
        self.running = False
        self.generation = 0
        self.manual = ("hold", False, 0.0)
        self.snapshot = {"status": "loading", "config": self.config.copy()}

    def control(self, c: Control):
        with self.lock:
            old = self.config.copy()
            for name in self.config:
                value = getattr(c, name, None)
                if value is not None:
                    self.config[name] = value
            if old != self.config or c.command == "restart":
                self.snapshot["decision"] = None
                self.snapshot["language_response"] = None
            if c.command == "manual":
                self.manual = (c.steer, c.fire, time.monotonic())
            if c.command == "pause":
                self.running = False
            if c.command in ("start", "restart"):
                self.running = True
            if c.command == "restart" or any(old[k] != self.config[k] for k in ("scenario", "seed")):
                self.generation += 1
            if self.snapshot.get("status") == "finished" and c.command == "start":
                self.generation += 1

    def loop(self):
        doom, policy, policy_name = None, None, None
        generation, decisions = -1, 0
        error = None
        # Preserve remote call reservations across resets and policy switches.
        remote_policy = None
        try:
            while not self.stop.is_set():
                start = time.perf_counter()
                with self.lock:
                    config, running, requested = self.config.copy(), self.running, self.generation
                    manual = self.manual
                try:
                    if requested != generation:
                        if doom:
                            doom.close()
                        doom = Doom(config["scenario"], config["seed"])
                        generation, decisions, error = requested, 0, None
                        with self.lock:
                            self.snapshot["decision"] = None
                        if isinstance(policy, LanguageDoomPolicy):
                            policy.reset()
                    if config["policy"] != policy_name:
                        if policy and policy is not remote_policy:
                            policy.close()
                        if config["policy"] == "language":
                            policy = LanguageDoomPolicy(self.decision_service, config["instruction"])
                        elif config["policy"] == "jev":
                            if remote_policy is None:
                                remote_policy = make_policy("jev")
                            policy = remote_policy
                        else:
                            policy = (
                                None
                                if config["policy"] == "manual"
                                else make_policy(config["policy"], config["seed"])
                            )
                        policy_name, error = config["policy"], None
                        with self.lock:
                            self.snapshot["decision"] = None
                    obs = doom.observe(config["directive"])
                    decision = None
                    if obs and running:
                        if (
                            isinstance(policy, LanguageDoomPolicy)
                            and policy.instruction != config["instruction"]
                        ):
                            policy.reset()
                            policy.instruction = config["instruction"]
                        if policy_name == "manual":
                            steer, fire, stamp = manual
                            if time.monotonic() - stamp > 0.4:
                                steer, fire = "hold", False
                            decision = hard_decision(steer, fire, "manual")
                        else:
                            decision = policy.decide(obs)
                        # A pause/restart received during a remote request cancels the action.
                        with self.lock:
                            valid = self.running and self.generation == generation and self.config == config
                        if valid:
                            doom.step(decision, obs)
                            decisions += 1
                        else:
                            decision = None
                            if isinstance(policy, LanguageDoomPolicy):
                                policy.reset()  # A canceled choice is not an executed history event.
                        error = None
                    frame = doom.frame()
                    encoded = None
                    if frame is not None:
                        output = io.BytesIO()
                        Image.fromarray(frame).save(output, format="JPEG", quality=85)
                        encoded = base64.b64encode(output.getvalue()).decode()
                    stats = doom.stats()
                    status = (
                        "error"
                        if error
                        else "finished"
                        if stats["finished"]
                        else "running"
                        if running
                        else "paused"
                    )
                    with self.lock:
                        prior = self.snapshot
                        self.snapshot = {
                            "status": status,
                            "config": config,
                            "stats": stats,
                            "decisions": decisions,
                            "observation": obs.to_dict() if obs else prior.get("observation"),
                            "decision": decision.to_dict() if decision else prior.get("decision"),
                            "frame": encoded or prior.get("frame"),
                            "error": error,
                            "loop_ms": (time.perf_counter() - start) * 1000,
                            "jev_calls": remote_policy.calls if remote_policy else 0,
                            "jev_available": not PUBLIC_DEMO and bool(os.environ.get("TYPESAFE_API_KEY")),
                            "language_response": policy.last_response
                            if isinstance(policy, LanguageDoomPolicy)
                            else None,
                        }
                        if stats["finished"]:
                            self.running = False
                except Exception as exc:  # noqa: BLE001 - worker boundary must pause on all engine/network failures
                    # Never put raw HTTP bodies, headers or API keys into the UI.
                    error = (
                        str(exc)
                        if isinstance(exc, (ValueError, RuntimeError))
                        else f"{type(exc).__name__}: run paused. Check server logs."
                    )
                    with self.lock:
                        self.running = False
                        self.snapshot.update(status="error", error=error)
                    self.stop.wait(0.2)
                self.stop.wait(max(0, 2 / 35 - (time.perf_counter() - start)))
        finally:
            if doom:
                doom.close()
            if policy and policy is not remote_policy:
                policy.close()
            if remote_policy:
                remote_policy.close()


@asynccontextmanager
async def lifespan(app):
    app.state.decisions = DecisionService()
    app.state.session = Session(app.state.decisions)
    app.state.session.thread.start()
    yield
    app.state.session.stop.set()
    app.state.session.thread.join(timeout=7)
    app.state.decisions.close()
    app.state.session.thread.join(timeout=7)


app = FastAPI(title="DecisionTics", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/app.js")
def script():
    return FileResponse(STATIC / "app.js", media_type="text/javascript")


@app.get("/style.css")
def style():
    return FileResponse(STATIC / "style.css", media_type="text/css")


@app.get("/decisions.js")
def decisions_script():
    return FileResponse(STATIC / "decisions.js", media_type="text/javascript")


@app.get("/api/state")
def state(request: Request):
    session = request.app.state.session
    with session.lock:
        return session.snapshot.copy()


@app.get("/api/model")
def model_status(request: Request):
    return {**request.app.state.decisions.status(), "public_demo": PUBLIC_DEMO}


def require_same_origin_request(request):
    if request.headers.get("x-openjev") != "1":
        raise HTTPException(403, "Same-origin request header required")
    origin = request.headers.get("origin")
    if origin and origin not in {str(request.base_url).rstrip("/"), PUBLIC_ORIGIN}:
        raise HTTPException(403, "Cross-origin requests are disabled")


@app.post("/api/decide", response_model=DecisionResponse)
def decide(payload: DecisionRequest, request: Request):
    require_same_origin_request(request)
    try:
        return request.app.state.decisions.decide(payload)
    except ModelBusy as exc:
        raise HTTPException(429, str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, "Model inference failed; no fallback scores were returned") from exc


@app.post("/api/control")
def control(c: Control, request: Request):
    # Custom header + JSON body + no CORS prevent other sites driving a localhost game/API bill.
    require_same_origin_request(request)
    if c.policy == "jev" and PUBLIC_DEMO:
        raise HTTPException(403, "Paid Jev API is disabled in public demo mode")
    if c.policy == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
        raise HTTPException(400, "TYPESAFE_API_KEY is not configured on the server")
    request.app.state.session.control(c)
    return {"ok": True}
