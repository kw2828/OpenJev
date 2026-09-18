"""Bounded synthetic capacity measurement, not a scientific fit or score."""
import hashlib
import json
import time
from pathlib import Path

import torch

from openjev.research import reacher_memory_control as control
from openjev.research import reacher_memory_protocol as protocol
from openjev.research import reacher_memory_training as training

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "attempt"
OUT.mkdir(exist_ok=False)
begin = time.monotonic()
deadline = begin + 120


def check():
    if time.monotonic() >= deadline:
        raise TimeoutError("120-second synthetic capacity limit")


def write(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


try:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    plan = protocol.settings(engineering=True)
    namespace = "reacher-memory-engineering-capacity-v1"
    plan.update(rng_namespace=namespace, engineering=True,
                engineering_rng_namespaces=[n for n in protocol.ENGINEERING_NAMESPACES if n != namespace])
    plan["random_stream_contract"] = protocol.stream_contract(plan)
    protocol.validate_settings(plan, engineering=True)
    cfg = training.MemoryTrainingSettings()
    generator = torch.Generator().manual_seed(410)
    n, t = cfg.train_episodes, cfg.steps
    angles = torch.randn(n, t + 1, 2, generator=generator).cumsum(1) * .01
    packets = torch.zeros(n, t + 1, 8)
    packets[..., :2], packets[..., 2:4] = angles.cos(), angles.sin()
    packets[..., 4:6], packets[..., 6] = .1, 1.
    for start in (8, 30):
        packets[:, start:start + 6, :4] = 0.
        packets[:, start:start + 6, 6] = 0.
        packets[:, start:start + 6, 7] = torch.arange(1, 7) * cfg.dt
    commands = torch.randn(n, t, 2, generator=generator).clamp(-1, 1) * .1
    data = {"packets": packets, "commands": commands, "rewards": -commands.square().sum(-1) - .1}
    roles = {family: f"fit/{family}/pair0" for family in ("gru", "mlp")}
    initial = training.make_initialization(protocol.seed(plan, roles["gru"]), protocol.seed(plan, roles["mlp"]),
                                            cfg, role_names=roles)
    orders = training.make_orders(protocol.seed(plan, "fit/minibatch/pair0"), cfg, role_name="fit/minibatch/pair0")
    paths = ["src/openjev/research/reacher_memory_training.py", "src/openjev/research/reacher_memory_control.py",
             "src/openjev/research/reacher_memory_protocol.py", "src/openjev/research/reacher_bounded_history.py",
             "src/openjev/research/reacher_observation_baseline.py", "src/openjev/research/reacher_reward_residual.py",
             "src/openjev/research/reacher_world_models.py", "src/openjev/research/reacher_objective_training.py"]
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    write("started.json", {"scope": "synthetic capacity only; no native evaluation or scientific result",
        "source_sha256": sources, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan": plan, "data_seed": 410, "data_sha256": training.canonical_tensor_hash(data), "cap_seconds": 120})
    results = []
    for kind in protocol.ARMS:
        check()
        tick = time.perf_counter()
        trainer = training.make_trainer(kind, initial, orders, data, source_sha256=sources,
                                        runtime={"purpose": "synthetic-capacity", "threads": 2})
        logs = [trainer.train_next(deadline_check=check) for _ in range(4)]
        elapsed = time.perf_counter() - tick
        torch.save(trainer.export_checkpoint(), OUT / f"{kind}.pt")
        model = trainer.student.eval()
        root = model.assimilate(model.initial(64), packets[:64, 0])
        search_times = []
        for step in (0, 1, 2):
            inputs = protocol.draw_control_inputs(plan, step)
            started = time.perf_counter()
            result, _, _, _, work = control.score_search(plan, model, root, inputs, step, deadline)
            search_times.append(time.perf_counter() - started)
        row = {"kind": kind, "updates": 4, "whole_measurement_seconds": elapsed,
               "all_update_wall_seconds": [v["costs"]["batch_wall_seconds"] for v in logs],
               "whole_trainer_training_seconds": trainer.training_wall_seconds,
               "three_full_batch_cem_seconds": search_times,
               "parameters": sum(p.numel() for p in model.parameters()),
               "projected_3fit_training_seconds": trainer.training_wall_seconds / 4 * 1152 * 3,
               "projected_3fit_3panel_search_seconds": sum(search_times) / 3 * 50 * 3 * 3,
               "scope": "Synthetic full-size tensors and noninteractive search. No native utility measured."}
        results.append(row)
        write(f"{kind}.json", row)
        print(json.dumps(row), flush=True)
    for name, digest in sources.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    write("completed.json", {"status": "completed", "scope": "engineering_capacity_only", "rows": results,
          "wall_seconds": time.monotonic() - begin,
          "projected_training_seconds": sum(r["projected_3fit_training_seconds"] for r in results),
          "projected_search_seconds": sum(r["projected_3fit_3panel_search_seconds"] for r in results),
          "limitations": "Projection excludes variable horizon, real assimilation, serialization, reference physics, "
              "native environment steps, initialization, restoration, full audit and publication; not a promised run time."})
except BaseException as error:
    write("failed.json", {"error": repr(error), "wall_seconds": time.monotonic() - begin,
                           "scope": "engineering_capacity_only"})
    raise
