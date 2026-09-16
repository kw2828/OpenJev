import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .domain import hard_decision
from .game import Doom
from .policies import make_policy

STATIC = Path(__file__).parent / "static"


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Literal["start", "pause", "restart", "configure", "manual"]
    policy: Literal["local", "rules", "random", "jev", "manual"] | None = None
    scenario: Literal["defend_the_center", "defend_the_line", "basic"] | None = None
    directive: Literal["hunt", "conserve", "pacifist"] | None = None
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    steer: Literal["left", "hold", "right"] = "hold"
    fire: bool = False


class Session:
    """One engine owned by one worker; HTTP readers never touch ViZDoom."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.config = {"policy": "local", "scenario": "defend_the_center", "directive": "hunt", "seed": 42}
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
                    if config["policy"] != policy_name:
                        if policy and policy is not remote_policy:
                            policy.close()
                        if config["policy"] == "jev":
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
                    obs = doom.observe(config["directive"])
                    decision = None
                    if obs and running:
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
                            "jev_available": bool(os.environ.get("TYPESAFE_API_KEY")),
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
    app.state.session = Session()
    app.state.session.thread.start()
    yield
    app.state.session.stop.set()
    app.state.session.thread.join(timeout=7)


app = FastAPI(title="OpenJev", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/app.js")
def script():
    return FileResponse(STATIC / "app.js", media_type="text/javascript")


@app.get("/style.css")
def style():
    return FileResponse(STATIC / "style.css", media_type="text/css")


@app.get("/api/state")
def state(request: Request):
    session = request.app.state.session
    with session.lock:
        return session.snapshot.copy()


@app.post("/api/control")
def control(c: Control, request: Request):
    # Custom header + JSON body + no CORS prevent other sites driving a localhost game/API bill.
    if request.headers.get("x-openjev") != "1":
        raise HTTPException(403, "Same-origin control header required")
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "Cross-origin controls are disabled")
    if c.policy == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
        raise HTTPException(400, "TYPESAFE_API_KEY is not configured on the server")
    request.app.state.session.control(c)
    return {"ok": True}
