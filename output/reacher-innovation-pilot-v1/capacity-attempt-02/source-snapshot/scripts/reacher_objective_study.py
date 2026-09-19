"""Prospective paired objective study with fixed adaptive Reacher planning.

Preparation is separate from execution. No scored model is constructed by the
prepare path. All evaluation follows restoration of every final student.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import reacher_reward_residual_study as legacy
import reacher_search_study as search_study
import torch

from openjev.research import reacher_objective_protocol as protocol
from openjev.research import reacher_objective_training as training
from openjev.research import reacher_search_protocol as search_protocol
from openjev.research.reacher_random_streams import concrete_streams
from openjev.research.robotics_reacher import collect_episode

ROOT = Path(__file__).resolve().parents[1]
base = legacy.base
sha, read, write, require = search_protocol.sha, search_protocol.read, search_protocol.write, search_protocol.require
check_cap = base.check_cap
STUDY = "reacher-objective-ablation-v1"
SEARCH_PLAN = "evidence/reacher-search-v1/protocol/plan.json"
SEARCH_PLAN_SHA = "8d5da5ee146c95532721fd1b6582bc37a091d5417e5d8b68f73e241379f2428b"
SEARCH_AUDIT = "evidence/reacher-search-v1/audit/receipt.json"
SEARCH_AUDIT_SHA = "a510adcbc19d7dd6cb292d520a417f9e7d6146da8a6a8b16f605c54a6f4fa93f"
CAPACITY_PATH = "evidence/reacher-objective-ablation-v1/engineering-capacity-v1.json"
CAPACITY_SHA = "3d77541ef843e531a66e7de55c89401ef9083a6bcbcdabca442d59f3f5cadb63"
TRAINING_TEST_TORCH_SEEDS = {
    "engineering/training-tests/public_data": 410,
    "engineering/training-tests/student/initial": 1121,
    "engineering/training-tests/predictor/initial": 1123,
    "engineering/training-tests/predictor/alternate": 1151,
    "engineering/training-tests/minibatch": 1129,
    **{f"engineering/training-tests/student/calibration/{index}": 1201 + index for index in range(3)},
    **{f"engineering/training-tests/predictor/calibration/{index}": 1301 + index for index in range(3)},
}
AUDITOR_TEST_NUMPY_SEEDS = {"engineering/auditor-tests/public_data": 716}
AUDITOR_TEST_TORCH_SEEDS = {"engineering/auditor-tests/corrupted-minibatch-state": 72}
CRITERION = {"objective_improvement": .05, "every_pair_nonworse": True,
             "reset_increase": .05, "every_pair_reset_positive": True,
             "versus_zero_improvement": .10, "physics_versus_zero_improvement": .10,
             "panels": ["ordinary", "shift"], "expected_checks": 31}
SOURCES = search_study.SOURCES + (
    "src/openjev/research/reacher_latent_consistency.py",
    "tests/test_reacher_latent_consistency.py",
    "src/openjev/research/reacher_raw_endpoint.py",
    "tests/test_reacher_raw_endpoint.py",
    "src/openjev/research/reacher_objective_protocol.py",
    "tests/test_reacher_objective_protocol.py",
    "src/openjev/research/reacher_objective_training.py",
    "tests/test_reacher_objective_training.py",
    "scripts/reacher_objective_study.py",
    "tests/test_reacher_objective_study.py",
    "scripts/audit_reacher_objective_study.py",
    "tests/test_audit_reacher_objective_study.py",
)


def checked(path, digest):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == digest,
            f"Artifact identity mismatch: {path}")
    return path


def authenticated_sources():
    """Reuse only training data; authenticate the completed planning baseline."""
    source = legacy.default_training_source()
    parent = read(checked(ROOT / SEARCH_PLAN, SEARCH_PLAN_SHA))
    receipt = read(checked(ROOT / SEARCH_AUDIT, SEARCH_AUDIT_SHA))
    require(receipt["status"] == "completed" and receipt["saved_output_only"] is True
            and receipt["plan_sha256"] == SEARCH_PLAN_SHA
            and receipt["source_sha256"] == parent["sources"], "Completed search lineage")
    for name, digest in parent["sources"].items():
        checked(ROOT / name, digest)
    for name, digest in receipt["files"].items():
        checked((ROOT / SEARCH_AUDIT).parent / name, digest)
    return source, {
        "plan_path": SEARCH_PLAN, "plan_sha256": SEARCH_PLAN_SHA,
        "audit_path": SEARCH_AUDIT, "audit_receipt_sha256": SEARCH_AUDIT_SHA,
        "prior_costs": receipt["costs"],
        "training_weights_reused": False,
    }


def prior_streams():
    """Enumerate bound historical and explicit fixture generators without draws.

    These declarations cover known role-derived and literal fixture seeds, not
    every ambient RNG state that unrelated tests may have consumed.
    """
    parent = read(checked(ROOT / SEARCH_PLAN, SEARCH_PLAN_SHA))
    descriptors = list(parent["random_stream_contract"]["priors"])
    numpy_registries = [concrete_streams(read(checked(ROOT / row["plan_path"], row["plan_sha256"])),
                                        include_train=row["include_train"]) for row in descriptors]
    numpy_registries.append(search_protocol.registry(parent))
    descriptors.append({"plan_path": SEARCH_PLAN, "plan_sha256": SEARCH_PLAN_SHA,
                        "registry_kind": "named-search", "include_train": False})
    for excluded in parent["random_stream_contract"]["engineering_exclusions"]:
        numpy_registries.append(excluded["registry"])
        descriptors.append({"plan_path": SEARCH_PLAN, "plan_sha256": SEARCH_PLAN_SHA,
                            "registry_kind": "named-search-engineering",
                            "namespace": excluded["namespace"], "include_train": False})
    torch_seeds = {}
    for prefix, seeds in (("world", (211, 223, 239)), ("reward", (271, 283, 293))):
        for value in seeds:
            torch_seeds[f"{prefix}/student/{value}"] = value
            torch_seeds[f"{prefix}/minibatch/{value}"] = value + 4_100_000
    capacity = read(checked(ROOT / CAPACITY_PATH, CAPACITY_SHA))
    torch_seeds["engineering/capacity/synthetic/public_data"] = capacity["synthetic_data"]["seed"]
    torch_seeds["engineering/temporary-overwritten-construction"] = 0
    capacity_state = protocol.torch_generator_manifest({"capacity": capacity["synthetic_data"]["seed"]})
    require(capacity_state["capacity"]["initial_state_sha256"]
            == capacity["synthetic_data"]["initial_state_sha256"], "Capacity synthetic generator identity")
    descriptors.append({"artifact_path": CAPACITY_PATH, "artifact_sha256": CAPACITY_SHA,
                        "registry_kind": "additional-engineering-torch",
                        "roles": ["engineering/capacity/synthetic/public_data",
                                  "engineering/temporary-overwritten-construction"]})
    # These literal engineering seeds precede the role-derived study streams.
    # Binding the source and effective seeds lets the independent audit rebuild
    # their actual CPU generator states without executing the training tests.
    torch_seeds.update(TRAINING_TEST_TORCH_SEEDS)
    test_source = "tests/test_reacher_objective_training.py"
    descriptors.append({"artifact_path": test_source, "artifact_sha256": sha(ROOT / test_source),
                        "registry_kind": "additional-engineering-torch",
                        "roles": sorted(TRAINING_TEST_TORCH_SEEDS),
                        "seeds": dict(sorted(TRAINING_TEST_TORCH_SEEDS.items()))})
    numpy_registries.append(dict(AUDITOR_TEST_NUMPY_SEEDS))
    torch_seeds.update(AUDITOR_TEST_TORCH_SEEDS)
    audit_test_source = "tests/test_audit_reacher_objective_study.py"
    for kind, seeds in (("numpy", AUDITOR_TEST_NUMPY_SEEDS), ("torch", AUDITOR_TEST_TORCH_SEEDS)):
        descriptors.append({"artifact_path": audit_test_source, "artifact_sha256": sha(ROOT / audit_test_source),
                            "registry_kind": f"additional-engineering-{kind}",
                            "roles": sorted(seeds), "seeds": dict(sorted(seeds.items()))})
    return numpy_registries, [torch_seeds], descriptors


def save_torch(path, payload):
    with Path(path).open("xb") as handle:
        torch.save(payload, handle)


def save_npy(path, array):
    with Path(path).open("xb") as handle:
        np.save(handle, array, allow_pickle=False)


def file_members(folder, deadline=float("inf")):
    members = {}
    for path in sorted(folder.rglob("*")):
        check_cap(deadline)
        require(not path.is_symlink(), "Execution symlink is not an artifact")
        if path.is_file():
            members[str(path.relative_to(folder))] = sha(path)
    return members


@torch.no_grad()
def prediction_record(plan, model, records, deadline):
    """Use only native student decoders, equally for all training objectives."""
    begin = time.perf_counter()
    packets, commands, _ = base.learning_tensors(records)
    state = model.initial(len(records))
    roots, one, rewards = [], [], []
    for step in range(plan["steps"]):
        check_cap(deadline)
        state = model.assimilate(state, packets[:, step])
        roots.append(state)
        state, angles, reward = model.advance(state, commands[:, step])
        one.append(angles.numpy().copy())
        rewards.append(reward.numpy().copy())
    horizons = tuple(plan["auxiliary_horizons"])
    endpoints = {horizon: [] for horizon in horizons}
    imagined_advances = 0
    for root in range(plan["steps"]):
        future = roots[root]
        for offset in range(1, min(max(horizons), plan["steps"] - root) + 1):
            check_cap(deadline)
            future, angles, _ = model.advance(future, commands[:, root + offset - 1])
            imagined_advances += len(records)
            if offset in endpoints:
                endpoints[offset].append(angles.numpy().copy())
    values = {"one_step_angles": np.stack(one, 1), "one_step_rewards": np.stack(rewards, 1)}
    observed = packets[:, :, 6].numpy() > .5
    for horizon, predictions in endpoints.items():
        values[f"h{horizon}_angles"] = np.stack(predictions, 1)
        values[f"h{horizon}_valid"] = observed[:, :-horizon] & observed[:, horizon:]
    require(all(np.isfinite(value).all() for value in values.values()), "Nonfinite prediction output")
    return values, {
        "wall_seconds": time.perf_counter() - begin,
        "student_assimilations": len(records) * plan["steps"],
        "prefix_advances": len(records) * plan["steps"],
        "imagined_advances": imagined_advances,
        "teacher_or_auxiliary_calls": 0,
    }


@torch.no_grad()
def learned_control(plan, model, panel, reset, inputs_by_step, out, deadline, progress):
    begin = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    envs = []
    try:
        envs, packets = search_study.control_envs(plan, panel, deadline)
        schedules = np.stack([protocol.schedule(plan, index, panel)
                              for index in range(plan["control_episodes"])])
        state = model.initial(len(envs))
        masks = np.zeros((len(envs), plan["steps"]), dtype=bool)
        reset_packets = np.zeros((len(envs), plan["steps"], 8), dtype=np.float32)
        setup = time.perf_counter() - begin
        decisions, native_times, angles, rewards = [], [], [], []
        for step in range(plan["steps"]):
            progress["step"] = step
            check_cap(deadline)
            tick = time.perf_counter()
            inputs = search_protocol.load_inputs(inputs_by_step[step])
            state, mask = protocol.assimilate_control(model, state, torch.from_numpy(packets),
                                                      step, schedules, reset=reset)
            mask = np.asarray(mask, dtype=bool)
            masks[:, step] = mask
            reset_packets[mask, step] = packets[mask]
            result, raw, sizes, search_seconds = search_study.score_search(
                model, state, inputs, "cem256", step, plan, deadline)
            state, predicted_angles, predicted_reward = model.advance(
                state, torch.from_numpy(result.selected_actions.copy()))
            require(torch.isfinite(predicted_angles).all().item()
                    and torch.isfinite(predicted_reward).all().item(), "Nonfinite selected prediction")
            angles.append(predicted_angles.numpy().copy())
            rewards.append(predicted_reward.numpy().copy())
            search_protocol.save_trace(out / "decisions" / f"{step:03d}", result, raw, sizes, search_seconds)
            decisions.append(time.perf_counter() - tick)
            check_cap(deadline)
            tick = time.perf_counter()
            packets = np.stack([env.step(action) for env, action in zip(envs, result.selected_actions, strict=True)])
            native_times.append(time.perf_counter() - tick)
        search_study.save_episodes(out / "episodes", envs)
        search_protocol.save_npz(out / "executed_predictions.npz", angles=np.stack(angles, 1),
                                 rewards=np.stack(rewards, 1))
        search_protocol.save_npz(out / "reset-events.npz", mask=masks, packets=reset_packets)
        result = {"setup_seconds": setup, "decision_seconds": decisions, "native_step_seconds": native_times,
                  "row_wall_seconds": time.perf_counter() - begin,
                  "observation_assimilations": len(envs) * plan["steps"],
                  "executed_action_advances": len(envs) * plan["steps"]}
        write(out / "timings.json", result)
        check_cap(deadline)
        return result
    except BaseException:
        if envs and not (out / "episodes.npz").exists():
            search_study.save_partial_episodes(out / "partial-episodes", envs)
        raise
    finally:
        for env in envs:
            env.close()


def stream_contract(plan):
    numpy_prior, torch_prior, descriptions = prior_streams()
    return protocol.stream_contract(plan, numpy_prior, torch_prior, descriptions)


def prepare(out, *, cap_seconds, audit_cap_seconds):
    require(type(cap_seconds) is int and cap_seconds > 0 and type(audit_cap_seconds) is int
            and audit_cap_seconds > 0, "Explicit positive whole-run and audit caps required")
    source, previous = authenticated_sources()
    plan = protocol.settings()
    plan.update(training_source=source, search_source=previous, cap_seconds=cap_seconds,
                audit_cap_seconds=audit_cap_seconds, sources={name: sha(ROOT / name) for name in SOURCES},
                runtime=base.runtime(), engineering=False,
                criterion=dict(CRITERION),
                stop="One complete run and separate saved-output audit, each with its own frozen cap. "
                     "No retries, resumes, replacement seeds, cap extensions or checkpoint selection.")
    plan["random_stream_contract"] = stream_contract(plan)
    protocol.validate_settings(plan)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    return {"path": str(out / "plan.json"), "sha256": sha(out / "plan.json")}


def validate(path, expected):
    plan = read(checked(path, expected))
    require(plan["engineering"] is False, "Production execution requires a scored protocol")
    protocol.validate_settings(plan)
    require(plan["criterion"] == CRITERION, "Frozen decision rule changed")
    require(set(plan["sources"]) == set(SOURCES), "Frozen source membership")
    for name, digest in plan["sources"].items():
        checked(ROOT / name, digest)
    require(plan["runtime"] == base.runtime(), "Frozen runtime identity")
    source, previous = authenticated_sources()
    require(plan["training_source"] == source and plan["search_source"] == previous, "Frozen source lineage")
    require(plan["random_stream_contract"] == stream_contract(plan), "Prospective stream manifest changed")
    for key in ("cap_seconds", "audit_cap_seconds"):
        require(type(plan[key]) is int and plan[key] > 0, "Positive frozen cap")
    return plan


def prepare_training(plan, data, out, deadline):
    begin = time.perf_counter()
    settings = training.TrainingSettings.from_plan(plan)
    initializations, orders = {}, {}
    (out / "initializations").mkdir()
    (out / "orders").mkdir()
    for pair in protocol.PAIRS:
        check_cap(deadline)
        initial = training.make_initialization(protocol.seed(plan, f"fit/student/{pair}"),
                                               protocol.seed(plan, f"fit/predictor/{pair}"), settings)
        initializations[pair] = initial
        target = out / "initializations" / f"{pair}.pt"
        save_torch(target, initial)
        write(target.with_suffix(".json"), {
            "pair": pair, "file_sha256": sha(target), "settings": initial["settings"],
            "seeds": initial["seeds"], "hashes": initial["hashes"],
            "initialization_wall_seconds": initial["initialization_wall_seconds"],
        })
        order = protocol.minibatch_orders(plan, pair)
        orders[pair] = order
        target = out / "orders" / f"{pair}.npy"
        save_npy(target, order.numpy())
        write(target.with_suffix(".json"), {
            "pair": pair, "role": f"fit/minibatch/{pair}",
            "seed": protocol.seed(plan, f"fit/minibatch/{pair}"), "shape": list(order.shape),
            "file_sha256": sha(target), "tensor_sha256": training.canonical_tensor_hash({"orders": order}),
        })
    initialization_seconds = time.perf_counter() - begin
    calibration = training.calibrate([initializations[pair] for pair in protocol.PAIRS],
                                    dict(zip(("packets", "commands", "rewards"), data, strict=True)),
                                    settings, deadline_check=lambda: check_cap(deadline))
    save_torch(out / "calibration-gradients.pt", calibration["gradients"])
    write(out / "calibration.json", calibration["receipt"])
    write(out / "calibration-completed.json", {
        "files": {name: sha(out / name) for name in ("calibration.json", "calibration-gradients.pt")},
        "initialization_files": {f"initializations/{pair}.pt": sha(out / "initializations" / f"{pair}.pt")
                                 for pair in protocol.PAIRS},
        "optimizer_updates": 0, "unix_time": time.time(),
    })
    check_cap(deadline)
    return settings, initializations, orders, calibration["receipt"], {
        "initialization_and_order_seconds": initialization_seconds,
        "calibration_seconds": calibration["receipt"]["costs"]["wall_seconds"],
        "preparation_wall_seconds": time.perf_counter() - begin,
    }


def fit_one(plan, row, data, initialization, order, multiplier, out, deadline, progress):
    begin = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    settings = training.TrainingSettings.from_plan(plan)
    trainer = training.make_trainer(row["arm"], initialization,
                                   protocol.seed(plan, row["minibatch_role"]), multiplier, settings,
                                   source_sha256=plan["sources"], runtime=plan["runtime"])
    save_torch(out / "initial-weights.pt", trainer.student.state_dict())
    total_updates = plan["epochs"] * plan["train_episodes"] // plan["batch_size"]
    try:
        with (out / "training.jsonl").open("x") as log:
            for epoch in range(plan["epochs"]):
                check_cap(deadline)
                permutation = trainer.next_permutation()
                require(torch.equal(permutation, order[epoch]), "Paired minibatch ordering diverged")
                for batch, indices in enumerate(permutation.split(plan["batch_size"])):
                    progress.update(epoch=epoch, batch=batch, completed_updates=trainer.successful_updates)
                    witness = row["arm"] == "latent" and trainer.successful_updates in (0, total_updates - 1)
                    values = trainer.train_batch(*(value[indices] for value in data), epoch=epoch,
                                                 batch=batch, indices=indices, capture_ema_witness=witness,
                                                 deadline_check=lambda: check_cap(deadline))
                    log.write(json.dumps(values, allow_nan=False, separators=(",", ":")) + "\n")
                    log.flush()
                    if witness:
                        label = "first" if trainer.successful_updates == 1 else "last"
                        save_torch(out / f"ema-{label}.pt", trainer.ema_witness)
                        trainer.ema_witness = None
        require(trainer.successful_updates == total_updates, "Complete optimizer-update coverage")
        final_weights = trainer.save_weights(out / "weights.pt")
        checkpoint = trainer.export_checkpoint()
        save_torch(out / "checkpoint.pt", checkpoint)
        files = file_members(out, deadline)
        receipt = {
            "name": row["name"], "arm": row["arm"], "pair": row["pair"],
            "settings": settings.configuration(), "updates": trainer.successful_updates,
            "optimizer_steps": trainer.optimizer_steps, "ema_updates": trainer.ema_updates,
            "loss_multiplier": multiplier,
            "student_parameters": sum(parameter.numel() for parameter in trainer.student.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in trainer.named_trainable.values()),
            "initial_hashes": initialization["hashes"],
            "order_tensor_sha256": training.canonical_tensor_hash({"orders": order}),
            "final_weights": final_weights, "checkpoint_integrity_sha256": checkpoint["integrity_sha256"],
            "log_chain_sha256": trainer.log_chain_sha256,
            "wall_seconds": time.perf_counter() - begin, "files": files,
        }
        write(out / "completed.json", receipt)
        check_cap(deadline)
        return receipt
    except BaseException as error:
        try:
            save_torch(out / "partial-checkpoint.pt", trainer.export_checkpoint())
        except BaseException as checkpoint_error:  # noqa: BLE001 - retain both errors, then re-raise the primary failure.
            write(out / "partial-checkpoint-error.json", {"error": repr(checkpoint_error)})
        write(out / "failed.json", {"error": repr(error), "progress": dict(progress),
                                     "successful_updates": trainer.successful_updates,
                                     "optimizer_steps": trainer.optimizer_steps,
                                     "ema_updates": trainer.ema_updates,
                                     "wall_seconds": time.perf_counter() - begin})
        raise


def restore_students(plan, out, deadline, expected_plan_sha256):
    begin = time.perf_counter()
    settings = training.TrainingSettings.from_plan(plan)
    boundary = read(out / "all-fits-completed.json")
    require(boundary["plan_sha256"] == expected_plan_sha256 and boundary["fit_order"] == plan["fit_order"],
            "Restoration boundary differs from frozen plan")
    require(set(boundary["files"]) == {f"fits/{name}/completed.json" for name in plan["fit_order"]},
            "Restoration requires every scheduled fit")
    calibration_boundary = read(checked(out / "calibration-completed.json",
                                        boundary["calibration_completed_sha256"]))
    for name, digest in calibration_boundary["files"].items():
        checked(out / name, digest)
    calibration = read(out / "calibration.json")
    models, receipts = {}, {}
    for row in protocol.fit_manifest(plan):
        check_cap(deadline)
        folder = out / "fits" / row["name"]
        completed = read(checked(folder / "completed.json", boundary["files"][f"fits/{row['name']}/completed.json"]))
        require(all(completed[key] == row[key] for key in ("name", "arm", "pair")), "Restored fit identity")
        for name, digest in completed["files"].items():
            checked(folder / name, digest)
        initial_path = out / "initializations" / f"{row['pair']}.pt"
        checked(initial_path, calibration_boundary["initialization_files"][f"initializations/{row['pair']}.pt"])
        initial = torch.load(initial_path, weights_only=True, map_location="cpu")
        order_metadata = read(out / "orders" / f"{row['pair']}.json")
        order_path = checked(out / "orders" / f"{row['pair']}.npy", order_metadata["file_sha256"])
        order = torch.from_numpy(np.load(order_path, allow_pickle=False))
        require(completed["initial_hashes"] == initial["hashes"]
                and completed["order_tensor_sha256"] == training.canonical_tensor_hash({"orders": order})
                and torch.equal(order, protocol.minibatch_orders(plan, row["pair"])),
                "Restored initialization/minibatch lineage")
        multiplier = 0. if row["arm"] == "anchor" else calibration["multipliers"][row["arm"]]["value"]
        checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        updates = plan["epochs"] * plan["train_episodes"] // plan["batch_size"]
        require(checkpoint["arm"] == row["arm"]
                and checkpoint["minibatch_seed"] == protocol.seed(plan, row["minibatch_role"])
                and checkpoint["initialization"]["hashes"] == initial["hashes"]
                and checkpoint["initialization"]["seeds"] == initial["seeds"]
                and checkpoint["successful_updates"] == checkpoint["optimizer_steps"] == completed["updates"] == updates
                and checkpoint["ema_updates"] == (updates if row["arm"] == "latent" else 0)
                and checkpoint["loss_multiplier"] == completed["loss_multiplier"] == multiplier,
                "Restored arm, RNG, optimizer/EMA or calibrated objective differs from schedule")
        trainer = training.restore_checkpoint(checkpoint, expected_settings=settings,
                                              expected_source_sha256=plan["sources"],
                                              expected_runtime=plan["runtime"])
        deploy = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
        wanted = completed["final_weights"]["canonical_tensor_sha256"]
        require(training.canonical_tensor_hash(deploy) == wanted
                and training.canonical_tensor_hash(trainer.student.state_dict()) == wanted,
                "Restored student differs from saved deployment weights")
        require(trainer.successful_updates == completed["updates"], "Restored update boundary")
        models[row["name"]] = trainer.student.eval()
        receipts[row["name"]] = {
            "checkpoint_sha256": sha(folder / "checkpoint.pt"),
            "weights_sha256": sha(folder / "weights.pt"), "student_tensor_sha256": wanted,
            "successful_updates": trainer.successful_updates,
            "restored_auxiliary_for_validation_only": row["arm"] != "anchor",
        }
        del trainer
    write(out / "restored-models.json", {"models": receipts, "wall_seconds": time.perf_counter() - begin,
                                        "unix_time": time.time()})
    check_cap(deadline)
    return models


def collect_predictions(plan, out, deadline, progress):
    begin = time.perf_counter()
    records = []
    try:
        for index in range(plan["prediction_episodes"]):
            progress.update(episode=index, completed_episodes=len(records))
            check_cap(deadline)
            records.append(collect_episode(protocol.seed(plan, f"prediction/reset/{index}"),
                protocol.schedule(plan, index, "ordinary", "prediction"), noise_std=plan["noise_std"],
                noise_seed=protocol.seed(plan, f"prediction/actuator_noise/{index}"),
                action_seed=protocol.seed(plan, f"prediction/exploration/{index}"), policy="mixed"))
        base.save_records(out / "prediction", records)
        check_cap(deadline)
        return records, time.perf_counter() - begin
    except BaseException:
        if records and not (out / "prediction.npz").exists():
            base.save_records(out / "partial-prediction", records)
        write(out / "partial-prediction-state.json", {
            "completed_episodes": len(records), "progress": dict(progress),
            "wall_seconds": time.perf_counter() - begin,
            "unfinished_episode_record_available": False,
        })
        raise


def run(path, expected, out):
    begin = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan = validate(path, expected)
        check_cap(begin + plan["cap_seconds"])
        return run_validated(plan, expected, out, begin=begin, output_created=True,
                             final_validate=lambda: validate(path, expected))
    except BaseException as error:
        if not (out / "failed.json").exists():
            write(out / "failed.json", {"status": "failed", "plan_sha256": expected,
                                        "error": repr(error), "progress": {"phase": "validate-or-setup"},
                                        "wall_seconds": time.monotonic() - begin})
        raise


def run_validated(plan, expected, out, *, begin=None, final_validate=None, output_created=False):
    """Core also accepts explicitly synthetic engineering fixtures before freeze."""
    begin = time.monotonic() if begin is None else begin
    engineering = plan.get("engineering") is True
    protocol.validate_settings(plan, engineering=engineering)
    require(engineering or callable(final_validate), "Scored execution requires frozen validation")
    deadline = begin + plan["cap_seconds"]
    if output_created:
        require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()), "Exclusive empty attempt required")
    else:
        out.mkdir(parents=True, exist_ok=False)
    progress = {"phase": "copy-training"}
    try:
        check_cap(deadline)
        torch.set_num_threads(plan["threads"])
        torch.use_deterministic_algorithms(True)
        write(out / "started.json", {"plan_sha256": expected, "unix_time": time.time()})
        write(out / "random-streams.json", plan["random_stream_contract"])
        source = plan["training_source"]
        for name, digest in source["members"].items():
            search_study.copy_verified(ROOT / source["execution_path"] / name, out / name, digest, deadline)
        write(out / "train-provenance.json", source)
        train = base.load_records(out / "train")
        require(len(train) == plan["train_episodes"], "Complete authenticated training corpus required")
        data = base.learning_tensors(train)
        progress = {"phase": "calibration"}
        _settings, initializations, orders, calibration, prep_cost = prepare_training(plan, data, out, deadline)
        fits = []
        for row in protocol.fit_manifest(plan):
            progress = {"phase": "fit", "model": row["name"]}
            multiplier = 0. if row["arm"] == "anchor" else calibration["multipliers"][row["arm"]]["value"]
            fits.append(fit_one(plan, row, data, initializations[row["pair"]], orders[row["pair"]],
                                multiplier, out / "fits" / row["name"], deadline, progress))
            print(json.dumps({"completed_fit": row["name"]}), flush=True)
        write(out / "all-fits-completed.json", {
            "plan_sha256": expected, "fit_order": plan["fit_order"],
            "files": {f"fits/{name}/completed.json": sha(out / "fits" / name / "completed.json")
                      for name in plan["fit_order"]},
            "calibration_completed_sha256": sha(out / "calibration-completed.json"),
            "unix_time": time.time(), "elapsed_seconds": time.monotonic() - begin,
        })
        progress = {"phase": "restore-all-students"}
        models = restore_students(plan, out, deadline, expected)
        evaluation_started = time.monotonic() - begin
        write(out / "evaluation-started.json", {
            "plan_sha256": expected, "all_fits_completed_sha256": sha(out / "all-fits-completed.json"),
            "restored_models_sha256": sha(out / "restored-models.json"),
            "unix_time": time.time(), "elapsed_seconds": evaluation_started,
        })
        progress = {"phase": "prediction-collection"}
        prediction, prediction_collection_seconds = collect_predictions(plan, out, deadline, progress)
        (out / "predictions").mkdir()
        prediction_times = []
        for name, model in models.items():
            progress = {"phase": "prediction", "model": name}
            values, timing = prediction_record(plan, model, prediction, deadline)
            search_protocol.save_npz(out / "predictions" / f"{name}.npz", **values)
            write(out / "predictions" / f"{name}.json", timing)
            prediction_times.append(timing)
        tick = time.perf_counter()
        inputs = []
        for step in range(plan["steps"]):
            check_cap(deadline)
            stem = out / "innovations" / "control" / f"{step:03d}"
            search_protocol.save_inputs(stem, protocol.draw_control_inputs(plan, step), f"planner/control/{step}")
            inputs.append(stem)
        innovation_seconds = time.perf_counter() - tick
        control_times = []
        for row in protocol.execution_order(plan):
            progress = {"phase": "control", "panel": row["panel"], "model": row["label"]}
            folder = out / row["path"]
            if "fit" in row:
                timing = learned_control(plan, models[row["fit"]], row["panel"], row["reset"],
                                         inputs, folder, deadline, progress)
            else:
                timing = search_study.reference_control(plan, row["panel"], row["reference"],
                                                        inputs, folder, deadline, progress)
            control_times.append(timing)
            print(json.dumps({"completed_control": f"{row['panel']}/{row['label']}"}), flush=True)
        progress = {"phase": "final-validation"}
        if final_validate is not None:
            final_validate()
        costs = {
            **prep_cost, "fit_wall_seconds": sum(row["wall_seconds"] for row in fits),
            "restore_wall_seconds": read(out / "restored-models.json")["wall_seconds"],
            "prediction_collection_seconds": prediction_collection_seconds,
            "prediction_model_seconds": sum(row["wall_seconds"] for row in prediction_times),
            "innovation_generation_and_storage_seconds": innovation_seconds,
            "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in control_times),
            "control_setup_seconds": sum(row["setup_seconds"] for row in control_times),
            "control_decision_seconds": sum(sum(row["decision_seconds"]) for row in control_times),
            "control_native_step_seconds": sum(sum(row["native_step_seconds"]) for row in control_times),
            "new_fits": len(fits), "astra_calls": 0,
            "prior_costs": plan["search_source"]["prior_costs"],
            "accounting": "New execution includes source validation/copying, training-only calibration, all fitting, "
                          "restoration, fresh collection and scoring, native stepping, storage and final hashes. "
                          "Component timings are nested and are not all additive. Audit/publication are separate.",
        }
        write(out / "costs.json", costs)
        members = file_members(out, deadline)
        check_cap(deadline)
        total = time.monotonic() - begin
        write(out / "completed.json", {
            "status": "completed", "study": STUDY, "plan_sha256": expected, "fits": len(fits),
            "control_rows": len(control_times), "prediction_episodes": len(prediction), "astra_calls": 0,
            "wall_seconds": total, "evaluation_started_elapsed_seconds": evaluation_started,
            "cumulative_attempt_wall_seconds": plan["search_source"]["prior_costs"]["cumulative_attempt_wall_seconds"] + total,
            "files": members,
        })
        check_cap(deadline)
        return {"status": "completed", "fits": len(fits), "control_rows": len(control_times), "wall_seconds": total}
    except BaseException as error:
        if (out / "completed.json").exists():
            (out / "completed.json").rename(out / "over-cap-completion.json")
        write(out / "failed.json", {"status": "failed", "error": repr(error), "progress": progress,
                                    "wall_seconds": time.monotonic() - begin,
                                    "files": file_members(out)})
        raise


def audit(path, expected, execution, out):
    from audit_reacher_objective_study import audit_saved

    return audit_saved(validate(path, expected), expected, execution, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--cap-seconds", type=int)
    parser.add_argument("--audit-cap-seconds", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.out, cap_seconds=args.cap_seconds, audit_cap_seconds=args.audit_cap_seconds)
    else:
        require(args.plan is not None and args.expected_plan_sha256, "External frozen plan hash required")
        result = (run(args.plan, args.expected_plan_sha256, args.out) if args.mode == "run"
                  else audit(args.plan, args.expected_plan_sha256, args.execution, args.out))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
