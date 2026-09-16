import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import vizdoom

from .game import Doom
from .policies import DEFAULT_WEIGHTS, make_policy


def run_episode(policy_name, scenario, seed, directive="hunt", record=None, instruction=None):
    policy = make_policy(policy_name, seed, instruction=instruction)
    rows, latencies, frames = [], [], []
    started = time.perf_counter()
    try:
        with Doom(scenario, seed) as doom:
            steps = 0
            last_obs = None
            while not doom.game.is_episode_finished():
                obs = doom.observe(directive)
                last_obs = obs
                decision = policy.decide(obs)
                latencies.append(decision.latency_ms)
                if record:
                    rows.append({"step": steps, "observation": obs.to_dict(), "decision": decision.to_dict()})
                    if policy_name == "language":
                        rows[-1]["language_response"] = policy.last_response
                    if steps % 2 == 0:
                        from PIL import Image

                        frames.append(Image.fromarray(doom.frame()).resize((320, 240)).quantize(colors=64))
                doom.step(decision, obs)
                steps += 1
            result = {
                "policy": policy_name,
                "scenario": scenario,
                "seed": seed,
                "directive": directive,
                "instruction": getattr(policy, "instruction", None),
                **doom.stats(),
                "decisions": steps,
                "remaining_ammo_before_last_action": last_obs.ammo if last_obs else None,
                "policy_latency_median_ms": float(np.median(latencies)),
                "policy_latency_p95_ms": float(np.percentile(latencies, 95)),
                "wall_seconds": time.perf_counter() - started,
            }
    finally:
        policy.close()
    if record:
        record = Path(record)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        record.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        if frames:
            # Each frame spans four 35 Hz game tics, so this is close to real-time.
            frames[0].save(
                record.with_suffix(".gif"),
                save_all=True,
                append_images=frames[1:],
                duration=114,
                loop=0,
                optimize=True,
            )
    return result


def evaluate(
    output, *, episodes=20, seed=1000, scenario="defend_the_center", policies=("local", "rules", "random")
):
    report = {
        "scope": "ViZDoom scenario with privileged visible-actor labels; no image recognition",
        "scenario": scenario,
        "frame_skip": 2,
        "seed_start": seed,
        "episodes_per_policy": episodes,
        "vizdoom": vizdoom.__version__,
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.machine(),
        "checkpoint_sha256": hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest(),
        "episodes": [],
        "summary": {},
    }
    for policy in policies:
        records = [run_episode(policy, scenario, s) for s in range(seed, seed + episodes)]
        report["episodes"].extend(records)
        report["summary"][policy] = {
            "mean_kills": float(np.mean([r["kills"] for r in records])),
            "mean_reward": float(np.mean([r["reward"] for r in records])),
            "mean_survival_game_seconds": float(np.mean([r["game_seconds"] for r in records])),
            "median_episode_policy_latency_ms": float(
                np.median([r["policy_latency_median_ms"] for r in records])
            ),
        }
        print(policy, json.dumps(report["summary"][policy]), flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
