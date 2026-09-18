"""Bounded learned-world-model pilot with separately saved execution evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research.reacher_world_models import make_world_model, repeat_index, sequence_loss
from openjev.research.robotics_reacher import ReacherEpisode, collect_episode, make_env

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("history", "gru", "rssm")
SOURCES = (
    "scripts/reacher_world_model_study.py",
    "scripts/audit_reacher_world_model_study.py",
    "src/openjev/research/robotics_reacher.py",
    "src/openjev/research/reacher_world_models.py",
    "src/openjev/research/reacher_physics_control.py",
    "tests/test_robotics_reacher.py",
    "tests/test_reacher_world_models.py",
    "tests/test_reacher_physics_control.py",
    "tests/test_reacher_world_model_study.py",
    "tests/test_audit_reacher_world_model_study.py",
    "research/robotics-requirements.txt",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runtime():
    import gymnasium.envs.mujoco.reacher_v5 as native
    import mujoco

    folder = Path(native.__file__).parent
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "gymnasium", "mujoco")},
        "native_source_sha256": sha(native.__file__),
        "native_xml_sha256": sha(folder / "assets/reacher.xml"),
        "mujoco_init_sha256": sha(mujoco.__file__),
    }


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    plan = {
        "version": 1, "sources": {p: sha(ROOT / p) for p in SOURCES},
        "runtime": runtime(), "kinds": list(KINDS), "fit_seeds": [211, 223, 239],
        "train_episodes": 768, "prediction_episodes": 96, "control_episodes": 32,
        "steps": 50, "epochs": 12, "batch_size": 32, "learning_rate": .001,
        "hidden_size": 64, "stochastic_size": 16, "window": 12, "width": 128,
        "rollout_horizon": 5, "rollout_weight": .5, "reward_scale": 4.,
        "kl_weight": .01, "kl_balance": .8, "free_nats": 1., "gradient_clip": 10.,
        "noise_std": .05, "ordinary_gap": 6, "shift_gap": 10,
        "train_seed": 64100001, "prediction_seed": 64200001,
        "control_seed": 64300001, "schedule_seed": 64400001,
        "noise_seed": 64500001, "exploration_seed": 64600001,
        "candidate_seed": 64700001, "filter_seed": 64800001,
        "bootstrap_seed": 64900001, "bootstrap_samples": 4096,
        "candidates": 64, "planning_horizon": 12, "action_block": 3,
        "particles": 32, "filter_bandwidth": .02,
        "threads": 2, "cap_seconds": 1800,
        "task": "Installed Gymnasium Reacher-v5, native 50 steps, explicit distance/control reward weights1/1. Preserve physical model, reset distribution and target. Add hidden Gaussian command noise std.05 before clipping; score native reward on applied action. Policy packet is angle cos/sin, static target, validity and age only. No velocity, fingertip displacement, reward input, applied action, noise realization or simulator state enters a learned policy.",
        "data": "One frozen common 768-episode mixed exploration corpus, half supplied-physics IK/PD plus held noise.12, quarter held Gaussian.3, quarter held Gaussian.8. Random commands/noise hold4 steps. This collector uses privileged state for exploration but its hidden state and expert labels are NOT learning targets. Train only next observed angle and executed reward, mask absent angle targets. Two6-step gaps at8+phase and28+phase; phase uniform0..3. Episode splits and independent streams are fixed before fitting.",
        "models": "Conventional autoregressive history MLP, GRU world model and diagonal Gaussian RSSM. No biological topology or Dreamer reproduction. Each receives same episodes, updates and optimizer; parameter/operation counts differ and are reported. Three paired fit seeds, final checkpoint only. RSSM samples at training; evaluation uses prior/posterior means and does not test uncertainty-aware planning.",
        "control": "Same32 untouched initial states and hidden noise per controller, ordinary6-step gaps and shifted10-step gaps; also full angle sensing as a competence panel. Same candidate bank per timestep/case, nominal action-conditioned imagined reward clipped to physical range[-2.5,0], no terminal value. Reset-memory variants of GRU/RSSM clear all learned state at blackout onset. Physics references use same candidate bank with native model and either true current state or a heuristic32-particle filter of the public history; both have supplied model knowledge. Zero and uniform policies are floors.",
        "criteria": "All fits and panels must complete. Physics known-state must reduce mean ordinary positive episode cost by>=10% versus zero action. A learned recurrent kind advances only if all3 fit seeds beat zero by>=10%, beat history MLP mean cost by>=5% in ordinary AND shifted panels, and mean blackout-reset cost is>=5% higher in both panels. Paired episode bootstrap intervals are descriptive, not architecture uncertainty; retain per-fit means and all cases. Prediction must also beat last-observed-angle persistence by>=10% one-step masked MSE in each fit. No topology sweep if these requirements fail. These are pilot continuation rules, not an ICLR acceptance or novel architecture claim.",
        "stop": "1800second cooperative whole-run cap, checked before optimizer batches, prediction roots, control steps and native-MPC cases, and before successful completion. An individual native operation is not preempted; an over-cap execution cannot be successful. No retries, substitute seeds, extensions, checkpoint selection or test-based tuning. All9 fits complete before control evaluation. On failure retain artifacts and stage counters; diagnose separately. Audit saved data only, no new fitting, policy decisions or MPC evaluations.",
    }
    write(out / "plan.json", plan)
    return {"path": str(out / "plan.json"), "sha256": sha(out / "plan.json")}


def validate(path, digest):
    if sha(path) != digest:
        raise ValueError("External frozen plan identity mismatch")
    plan = json.loads(path.read_text())
    if set(plan["sources"]) != set(SOURCES):
        raise ValueError("Frozen source membership mismatch")
    for name, value in plan["sources"].items():
        if sha(ROOT / name) != value:
            raise ValueError(f"Frozen source changed: {name}")
    if plan["runtime"] != runtime():
        raise ValueError("Frozen runtime changed")
    return plan


def check_cap(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Frozen whole-run wall-clock cap")


def schedule(plan, index, panel="ordinary"):
    valid = np.ones(plan["steps"] + 1, dtype=bool)
    if panel == "full":
        return valid
    phase = int(np.random.default_rng(plan["schedule_seed"] + index).integers(0, 4))
    gap = plan["shift_gap"] if panel == "shift" else plan["ordinary_gap"]
    for start in (8 + phase, 28 + phase):
        valid[start:start + gap] = False
    return valid


def save_records(out, records):
    arrays = {}
    for category in ("policy", "audit"):
        for key in records[0][category]:
            arrays[category + "__" + key] = np.stack([r[category][key] for r in records])
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    write(out.with_suffix(".json"), [r["metadata"] for r in records])


def load_records(path):
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    result = []
    for i, item in enumerate(meta):
        record = {"policy": {}, "audit": {}, "metadata": item}
        for key, value in arrays.items():
            category, name = key.split("__", 1)
            record[category][name] = value[i]
        result.append(record)
    return result


def learning_tensors(records):
    """Explicit allowlist: the training interface cannot receive privileged states."""
    return (
        torch.tensor(np.stack([r["policy"]["packets"] for r in records]), dtype=torch.float32),
        torch.tensor(np.stack([r["policy"]["commands"] for r in records]), dtype=torch.float32),
        torch.tensor(np.stack([r["audit"]["rewards"] for r in records]), dtype=torch.float32),
    )


def model_for(plan, kind, seed):
    torch.manual_seed(seed)
    return make_world_model(kind, **{k: plan[k] for k in (
        "hidden_size", "stochastic_size", "window", "width")})


def fit(plan, kind, seed, data, out, deadline, progress=None):
    begin = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    model = model_for(plan, kind, seed).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
    order = torch.Generator().manual_seed(seed + 4100000)
    updates, logs = 0, []
    for epoch in range(plan["epochs"]):
        permutation = torch.randperm(len(data[0]), generator=order)
        metrics = []
        for indices in permutation.split(plan["batch_size"]):
            if progress is not None:
                progress.update(epoch=epoch + 1, completed_updates=updates)
            check_cap(deadline)
            optimizer.zero_grad(set_to_none=True)
            loss, values = sequence_loss(model, *(x[indices] for x in data), **{
                k: plan[k] for k in ("rollout_horizon", "rollout_weight", "reward_scale", "kl_weight", "kl_balance", "free_nats")})
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            metrics.append({**values, "gradient_norm": float(norm)})
            updates += 1
        logs.append({"epoch": epoch + 1, **{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}})
        write(out / "training.json", logs)
    torch.save(model.state_dict(), out / "weights.pt")
    write(out / "training.json", logs)
    write(out / "completed.json", {
        "kind": kind, "seed": seed, "updates": updates,
        "parameters": sum(p.numel() for p in model.parameters()),
        "wall_seconds": time.monotonic() - begin,
        "files": {p.name: sha(p) for p in out.iterdir()},
    })
    return model.eval()


def candidate_bank(plan, step, count):
    horizon = min(plan["planning_horizon"], plan["steps"] - step)
    chunks = (horizon + plan["action_block"] - 1) // plan["action_block"]
    rng = np.random.default_rng(plan["candidate_seed"] + step)
    bank = rng.normal(0., .25, (count, plan["candidates"], chunks, 2))
    bank[:, plan["candidates"] // 2:] *= 3
    bank = np.repeat(bank, plan["action_block"], axis=2)[:, :, :horizon]
    bank = np.clip(bank, -1., 1.).astype(np.float32)
    bank[:, 0] = 0.
    for k, action in enumerate(((.1, 0), (-.1, 0), (0, .1), (0, -.1), (.2, .2), (-.2, -.2)), 1):
        if k < plan["candidates"]:
            bank[:, k] = action
    return bank


@torch.no_grad()
def learned_mpc(model, state, bank):
    n, k, horizon, _ = bank.shape
    indices = torch.arange(n).repeat_interleave(k)
    imagined = repeat_index(state, indices)
    score = torch.zeros(n * k)
    for t in range(horizon):
        imagined, _, reward = model.advance(imagined, torch.from_numpy(bank[:, :, t].reshape(n * k, 2)))
        if not torch.isfinite(reward).all():
            raise ValueError("Nonfinite imagined reward")
        score += reward.clamp(-2.5, 0.)
    selected = score.reshape(n, k).argmax(1).numpy()
    return bank[np.arange(n), selected, 0], score.reshape(n, k).numpy()


@torch.no_grad()
def prediction_record(plan, model, records, deadline=float("inf")):
    packets, commands, _ = learning_tensors(records)
    state = model.initial(len(records))
    one, rewards, multi, roots = [], [], [], []
    for t in range(plan["steps"]):
        check_cap(deadline)
        state = model.assimilate(state, packets[:, t])
        roots.append(state)
        state, angles, reward = model.advance(state, commands[:, t])
        one.append(angles.numpy())
        rewards.append(reward.numpy())
    h = plan["rollout_horizon"]
    for start in range(plan["steps"] - h + 1):
        check_cap(deadline)
        future = roots[start]
        for offset in range(h):
            future, angles, _ = model.advance(future, commands[:, start + offset])
        multi.append(angles.numpy())
    return {"one": np.stack(one, 1), "reward": np.stack(rewards, 1), "multi": np.stack(multi, 1)}


@torch.no_grad()
def learned_control(plan, model, panel, reset, deadline, progress=None):
    setup_start = time.perf_counter()
    count = plan["control_episodes"]
    envs = [ReacherEpisode(noise_std=plan["noise_std"]) for _ in range(count)]
    packets = np.stack([e.reset(plan["control_seed"] + i, schedule(plan, i, panel),
                              noise_seed=plan["noise_seed"] + 200000 + i)
                        for i, e in enumerate(envs)])
    state, previous_valid = model.initial(count), np.ones(count, bool)
    scores, times, predicted_angles, predicted_rewards = [], [], [], []
    setup_seconds = time.perf_counter() - setup_start
    try:
        for t in range(plan["steps"]):
            if progress is not None:
                progress["step"] = t
            check_cap(deadline)
            start = time.perf_counter()
            if reset:
                mask = torch.from_numpy(previous_valid & (packets[:, 6] == 0))
                empty = model.initial(count)
                state = {k: torch.where(mask.reshape((-1,) + (1,) * (v.ndim - 1)), empty[k], v) for k, v in state.items()}
            state = model.assimilate(state, torch.from_numpy(packets))
            actions, score = learned_mpc(model, state, candidate_bank(plan, t, count))
            state, angles, reward = model.advance(state, torch.from_numpy(actions))
            predicted_angles.append(angles.numpy())
            predicted_rewards.append(reward.numpy())
            times.append(time.perf_counter() - start)
            previous_valid = packets[:, 6] > .5
            packets = np.stack([e.step(a) for e, a in zip(envs, actions, strict=True)])
            scores.append(score)
        return [e.episode_record() for e in envs], {
            "candidate_scores": np.stack(scores, 1), "batch_decision_seconds": np.array(times),
            "planner_used": np.array(True),
            "selected_predicted_angles": np.stack(predicted_angles, 1),
            "selected_predicted_reward": np.stack(predicted_rewards, 1),
            "setup_seconds": np.array(setup_seconds),
        }
    finally:
        for env in envs:
            env.close()


def run(plan_path, expected, out):
    plan = validate(plan_path, expected)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + plan["cap_seconds"]
    torch.set_num_threads(plan["threads"])
    torch.use_deterministic_algorithms(True)
    write(out / "started.json", {"plan_sha256": expected, "unix_time": time.time()})
    progress = {"phase": "starting"}
    try:
        datasets = {}
        for split, count, base, stream in (
            ("train", plan["train_episodes"], plan["train_seed"], 0),
            ("prediction", plan["prediction_episodes"], plan["prediction_seed"], 100000),
        ):
            records = []
            for i in range(count):
                progress.update(phase="data", split=split, episode=i)
                check_cap(deadline)
                records.append(collect_episode(base + i, schedule(plan, stream + i),
                    noise_std=plan["noise_std"], noise_seed=plan["noise_seed"] + stream + i,
                    action_seed=plan["exploration_seed"] + stream + i, policy="mixed"))
            save_records(out / split, records)
            datasets[split] = records
            print(json.dumps({"completed_dataset": split, "episodes": count}), flush=True)
        tensors = learning_tensors(datasets["train"])
        models = {}
        for kind in plan["kinds"]:
            for seed in plan["fit_seeds"]:
                name = f"{kind}-{seed}"
                progress = {"phase": "fit", "model": name}
                models[name] = fit(plan, kind, seed, tensors, out / "fits" / name, deadline, progress)
                print(json.dumps({"completed_fit": name}), flush=True)
        for name, model in models.items():
            progress = {"phase": "prediction", "model": name}
            check_cap(deadline)
            folder = out / "predictions"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f"{name}.npz", **prediction_record(plan, model, datasets["prediction"], deadline))
        for panel in ("full", "ordinary", "shift"):
            folder = out / "control" / panel
            folder.mkdir(parents=True, exist_ok=False)
            for name, model in models.items():
                for reset in ((False,) if name.startswith("history") or panel == "full" else (False, True)):
                    label = name + ("-reset" if reset else "")
                    progress = {"phase": "control", "panel": panel, "model": label}
                    records, extra = learned_control(plan, model, panel, reset, deadline, progress)
                    save_records(folder / label, records)
                    np.savez_compressed(folder / f"{label}-planning.npz", **extra)
                    print(json.dumps({"completed_control": f"{panel}/{label}"}), flush=True)
            # Supplied-physics references and floors are appended by the shared control helper.
            for arm in ("known_state", "particle", "zero", "uniform"):
                progress = {"phase": "control", "panel": panel, "model": arm}
                records, extra = reference_control(plan, panel, arm, deadline, progress)
                save_records(folder / arm, records)
                np.savez_compressed(folder / f"{arm}-planning.npz", **extra)
                print(json.dumps({"completed_control": f"{panel}/{arm}"}), flush=True)
        validate(plan_path, expected)
        check_cap(deadline)
        write(out / "completed.json", {
            "status": "completed", "plan_sha256": expected,
            "wall_seconds": time.monotonic() - start, "fits": len(models),
            "astra_calls": 0,
            "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        })
    except BaseException as error:
        write(out / "failed.json", {"error": repr(error), "wall_seconds": time.monotonic() - start, "progress": progress})
        raise


def reference_control(plan, panel, arm, deadline, progress=None):
    from openjev.research.reacher_physics_control import ParticleFilter, PhysicsMPC

    setup_start = time.perf_counter()
    count = plan["control_episodes"]
    envs = [ReacherEpisode(noise_std=plan["noise_std"]) for _ in range(count)]
    packets = np.stack([e.reset(plan["control_seed"] + i, schedule(plan, i, panel),
                              noise_seed=plan["noise_seed"] + 200000 + i)
                        for i, e in enumerate(envs)])
    template = make_env()
    model = template.unwrapped.model
    planner = PhysicsMPC(model, frame_skip=2)
    filters = [ParticleFilter(model, p, seed=plan["filter_seed"] + i,
                 particles=plan["particles"], noise_std=plan["noise_std"],
                 measurement_std=plan["filter_bandwidth"], frame_skip=2)
               for i, p in enumerate(packets)] if arm == "particle" else []
    uniform = np.random.default_rng(plan["candidate_seed"] + 200000)
    previous = np.zeros((count, 2), dtype=np.float32)
    scores, times = [], []
    setup_seconds = time.perf_counter() - setup_start
    try:
        for t in range(plan["steps"]):
            if progress is not None:
                progress["step"] = t
            check_cap(deadline)
            begin = time.perf_counter()
            score = np.zeros((count, plan["candidates"]))
            if arm == "zero":
                actions = np.zeros((count, 2), dtype=np.float32)
            elif arm == "uniform":
                actions = uniform.uniform(-1., 1., (count, 2)).astype(np.float32)
            else:
                bank = candidate_bank(plan, t, count)
                actions = []
                for i in range(count):
                    check_cap(deadline)
                    if arm == "known_state":
                        record = envs[i].audit_record()
                        qpos, qvel = record["qpos"], record["qvel"]
                    else:
                        if t:
                            filters[i].update(previous[i], packets[i])
                        qpos, qvel = filters[i].estimate()
                    command, costs = planner.plan(qpos, qvel, bank[i])
                    actions.append(command)
                    score[i] = -costs
                actions = np.array(actions, dtype=np.float32)
            times.append(time.perf_counter() - begin)
            packets = np.stack([e.step(a) for e, a in zip(envs, actions, strict=True)])
            previous = actions
            scores.append(score)
        return [e.episode_record() for e in envs], {
            "candidate_scores": np.stack(scores, 1),
            "batch_decision_seconds": np.array(times),
            "planner_used": np.array(arm in ("known_state", "particle")),
            "setup_seconds": np.array(setup_seconds),
        }
    finally:
        for env in envs:
            env.close()
        template.close()


def audit(plan_path, expected, execution, out):
    from audit_reacher_world_model_study import audit_saved

    return audit_saved(validate(plan_path, expected), expected, execution, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.out)
    elif args.mode == "run":
        result = run(args.plan, args.expected_plan_sha256, args.out)
    else:
        result = audit(args.plan, args.expected_plan_sha256, args.execution, args.out)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
