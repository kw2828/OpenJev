"""Prospective explicit-cache comparison with actual model classes and fixed CEM.

This module does nothing on import. Preparation never draws scored examples or
constructs a scored model. Every final checkpoint is restored before any new
evaluation draw. Existing objective-study files remain frozen and unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import reacher_memory_study as previous
import torch

from openjev.research import reacher_cache_control as control
from openjev.research import reacher_cache_protocol as protocol
from openjev.research import reacher_cache_training as training
from openjev.research import reacher_search_protocol as serialization
from openjev.research.reacher_kinematic_control import KinematicMPC
from openjev.research.robotics_reacher import collect_episode, make_env

ROOT = Path(__file__).resolve().parents[1]
base, search_study = previous.base, previous.search_study
sha, read, write, require = previous.sha, previous.read, previous.write, previous.require
check_cap, checked, save_torch = previous.check_cap, previous.checked, previous.save_torch
file_members = previous.file_members
STUDY = "reacher-cache-ablation-v1"
MEMORY_PLAN = "evidence/reacher-memory-ablation-v1/protocol/plan.json"
MEMORY_PLAN_SHA = "05f09ee5425190f5d652aedb10af1641037035d85e906d04d86baf33552a5786"
MEMORY_AUDIT = "evidence/reacher-memory-ablation-v1/audit/receipt.json"
MEMORY_AUDIT_SHA = "2589383326a2312d7f738821da7fd11f82b9bee1b7168aba2e7f44aaa4946d5d"
NEW_SOURCES = (
    "src/openjev/research/reacher_cached_observation.py",
    "tests/test_reacher_cached_observation.py",
    "src/openjev/research/reacher_cache_training.py",
    "tests/test_reacher_cache_training.py",
    "src/openjev/research/reacher_cache_control.py",
    "tests/test_reacher_cache_control.py",
    "src/openjev/research/reacher_cache_protocol.py",
    "tests/test_reacher_cache_protocol.py",
    "scripts/reacher_cache_study.py",
    "tests/test_reacher_cache_study.py",
    "scripts/audit_reacher_cache_study.py",
    "tests/test_audit_reacher_cache_study.py",
)
SOURCES = previous.SOURCES + NEW_SOURCES


def settings_from_plan(plan):
    names = {field.name for field in fields(training.CacheTrainingSettings)}
    require(names <= set(plan), "Complete cache training settings")
    return training.CacheTrainingSettings(**{name: plan[name] for name in names})


def authenticated_sources():
    source, objective = previous.authenticated_sources()
    plan = read(checked(ROOT / MEMORY_PLAN, MEMORY_PLAN_SHA))
    receipt = read(checked(ROOT / MEMORY_AUDIT, MEMORY_AUDIT_SHA))
    require(receipt["status"] == "completed" and receipt["saved_output_only"] is True
            and receipt["plan_sha256"] == MEMORY_PLAN_SHA
            and receipt["source_sha256"] == plan["sources"], "Completed memory lineage")
    for name, digest in plan["sources"].items():
        checked(ROOT / name, digest)
    for name, digest in receipt["files"].items():
        checked((ROOT / MEMORY_AUDIT).parent / name, digest)
    return source, objective, {
        "plan_path": MEMORY_PLAN, "plan_sha256": MEMORY_PLAN_SHA,
        "audit_path": MEMORY_AUDIT, "audit_receipt_sha256": MEMORY_AUDIT_SHA,
        "prior_costs": receipt["costs"], "training_weights_reused": False,
        "previous_scientific_gate_passed": True,
    }


def prior_streams():
    numpy_prior, torch_prior, descriptors = previous.prior_streams()
    parent = read(checked(ROOT / MEMORY_PLAN, MEMORY_PLAN_SHA))
    contract = parent["random_stream_contract"]
    numpy_prior.append(contract["registry"])
    torch_prior.append(contract["torch_registry"])
    descriptors.append({"plan_path": MEMORY_PLAN, "plan_sha256": MEMORY_PLAN_SHA,
                        "registry_kind": "memory-scored"})
    for excluded in contract["engineering_exclusions"]:
        numpy_prior.append(excluded["registry"])
        torch_prior.append(excluded["torch_registry"])
        descriptors.append({"plan_path": MEMORY_PLAN, "plan_sha256": MEMORY_PLAN_SHA,
                            "namespace": excluded["namespace"], "registry_kind": "memory-engineering"})
    numpy_prior.append({"cache/engineering/fixture": 410})
    torch_prior.append({"cache/engineering/fixture": 410, "cache/engineering/restore": 0})
    descriptors.append({"registry_kind": "cache-engineering-literal", "numpy_seeds": [410],
                        "torch_seeds": [0, 410]})
    return numpy_prior, torch_prior, descriptors


def stream_contract(plan):
    return protocol.stream_contract(plan, *prior_streams())


def prepare(out, *, cap_seconds, audit_cap_seconds):
    for value in (cap_seconds, audit_cap_seconds):
        require(type(value) is int and value > 0, "Explicit positive execution and audit caps")
    source, objective, lineage = authenticated_sources()
    plan = protocol.settings()
    plan.update(training_source=source, objective_source=objective, memory_source=lineage, cap_seconds=cap_seconds,
                audit_cap_seconds=audit_cap_seconds, sources={name: sha(ROOT / name) for name in SOURCES},
                runtime=base.runtime(), engineering=False,
                stop="One whole execution and separate saved-output audit. No retries, resumes, replacement "
                     "seeds, cap extensions, selected checkpoints or post-outcome changes.")
    plan["random_stream_contract"] = stream_contract(plan)
    protocol.validate_settings(plan)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    return {"path": str(out / "plan.json"), "sha256": sha(out / "plan.json")}


def validate(path, expected):
    plan = read(checked(path, expected))
    require(plan["engineering"] is False, "Scored protocol required")
    protocol.validate_settings(plan)
    require(set(plan["sources"]) == set(SOURCES), "Exact cache source membership")
    for name, digest in plan["sources"].items():
        checked(ROOT / name, digest)
    require(plan["runtime"] == base.runtime(), "Frozen runtime changed")
    source, objective, lineage = authenticated_sources()
    require(plan["training_source"] == source and plan["objective_source"] == objective
            and plan["memory_source"] == lineage, "Frozen lineage")
    require(plan["random_stream_contract"] == stream_contract(plan), "Frozen random streams")
    for key in ("cap_seconds", "audit_cap_seconds"):
        require(type(plan[key]) is int and plan[key] > 0, "Positive frozen cap")
    return plan


def public_training_data(records):
    return dict(zip(("packets", "commands", "rewards"), base.learning_tensors(records), strict=True))


def prepare_training(plan, out, deadline):
    started = time.perf_counter()
    cfg = settings_from_plan(plan)
    initializations, orders = {}, {}
    (out / "initializations").mkdir()
    (out / "orders").mkdir()
    for pair in protocol.PAIRS:
        check_cap(deadline)
        roles = {"gru": f"fit/gru/{pair}", "mlp": f"fit/mlp/{pair}"}
        initial = training.make_initialization(protocol.seed(plan, roles["gru"]),
            protocol.seed(plan, roles["mlp"]), cfg, role_names=roles)
        order_role = f"fit/minibatch/{pair}"
        order = training.make_orders(protocol.seed(plan, order_role), cfg, role_name=order_role)
        for category, payload in (("initializations", initial), ("orders", order)):
            target = out / category / f"{pair}.pt"
            save_torch(target, payload)
            write(target.with_suffix(".json"), {"pair": pair, "file_sha256": sha(target),
                                               "integrity_sha256": payload["integrity_sha256"]})
        initializations[pair], orders[pair] = initial, order
    files = {f"{folder}/{pair}.{suffix}": sha(out / folder / f"{pair}.{suffix}")
             for folder in ("initializations", "orders") for pair in protocol.PAIRS for suffix in ("pt", "json")}
    write(out / "training-prepared.json", {"files": files, "optimizer_updates": 0,
                                           "wall_seconds": time.perf_counter() - started,
                                           "unix_time": time.time()})
    check_cap(deadline)
    return initializations, orders


def fit_one(plan, row, data, initial, order, out, deadline, progress):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    trainer = training.make_trainer(row["arm"], initial, order, data,
                                   source_sha256=plan["sources"], runtime=plan["runtime"])
    save_torch(out / "initial-weights.pt", trainer.student.state_dict())
    expected_updates = plan["epochs"] * trainer.settings.batches_per_epoch
    try:
        with (out / "training.jsonl").open("x") as log:
            for _ in range(expected_updates):
                progress.update(**trainer.cursor, completed_updates=trainer.successful_updates)
                values = trainer.train_next(deadline_check=lambda: check_cap(deadline))
                log.write(json.dumps(values, allow_nan=False, separators=(",", ":")) + "\n")
                log.flush()
        require(trainer.successful_updates == trainer.optimizer_steps == expected_updates, "Complete updates")
        save_torch(out / "weights.pt", trainer.student.state_dict())
        checkpoint = trainer.export_checkpoint()
        save_torch(out / "checkpoint.pt", checkpoint)
        receipt = {"name": row["name"], "arm": row["arm"], "pair": row["pair"],
                   "model_configuration": trainer.model_configuration, "settings": trainer.settings.configuration(),
                   "updates": trainer.successful_updates, "optimizer_steps": trainer.optimizer_steps,
                   "parameters": sum(p.numel() for p in trainer.student.parameters()),
                   "initialization_sha256": initial["integrity_sha256"], "orders_sha256": order["integrity_sha256"],
                   "data_sha256": trainer.data_sha256,
                   "student_tensor_sha256": training.canonical_tensor_hash(trainer.student.state_dict()),
                   "checkpoint_integrity_sha256": checkpoint["integrity_sha256"],
                   "log_chain_sha256": trainer.log_chain_sha256,
                   "trainer_setup_seconds": trainer.setup_wall_seconds,
                   "training_wall_seconds": trainer.training_wall_seconds,
                   "wall_seconds": time.perf_counter() - started, "files": file_members(out, deadline)}
        write(out / "completed.json", receipt)
        check_cap(deadline)
        return receipt
    except BaseException as error:
        try:
            save_torch(out / "partial-checkpoint.pt", trainer.export_checkpoint())
        except BaseException as checkpoint_error:  # noqa: BLE001 - preserve secondary failure, re-raise original.
            write(out / "partial-checkpoint-error.json", {"error": repr(checkpoint_error)})
        write(out / "failed.json", {"error": repr(error), "progress": dict(progress),
                                   "successful_updates": trainer.successful_updates,
                                   "optimizer_steps": trainer.optimizer_steps,
                                   "wall_seconds": time.perf_counter() - started})
        raise


def restore_students(plan, out, data, deadline, expected_plan_sha256):
    started = time.perf_counter()
    boundary = read(out / "all-fits-completed.json")
    require(boundary["plan_sha256"] == expected_plan_sha256 and boundary["fit_order"] == plan["fit_order"],
            "Restoration boundary identity")
    require(set(boundary["files"]) == {f"fits/{name}/completed.json" for name in plan["fit_order"]},
            "Every fit must finish before restoration")
    prepared = read(checked(out / "training-prepared.json", boundary["training_prepared_sha256"]))
    expected_prepared = {f"{folder}/{pair}.{suffix}" for folder in ("initializations", "orders")
                         for pair in protocol.PAIRS for suffix in ("pt", "json")}
    require(set(prepared["files"]) == expected_prepared, "Exact prepared initialization/order membership")
    for name, digest in prepared["files"].items():
        checked(out / name, digest)
    models, receipts = {}, {}
    cfg = settings_from_plan(plan)
    for row in protocol.fit_manifest(plan):
        check_cap(deadline)
        folder = out / "fits" / row["name"]
        done = read(checked(folder / "completed.json", boundary["files"][f"fits/{row['name']}/completed.json"]))
        require(all(done[k] == row[k] for k in ("name", "arm", "pair")), "Actual fit identity")
        require(set(done["files"]) == {"initial-weights.pt", "training.jsonl", "weights.pt", "checkpoint.pt"},
                "Exact completed fit membership")
        for name, digest in done["files"].items():
            checked(folder / name, digest)
        initial = torch.load(out / "initializations" / f"{row['pair']}.pt", weights_only=True, map_location="cpu")
        order = torch.load(out / "orders" / f"{row['pair']}.pt", weights_only=True, map_location="cpu")
        checkpoint = torch.load(folder / "checkpoint.pt", weights_only=True, map_location="cpu")
        roles = {family: f"fit/{family}/{row['pair']}" for family in ("gru", "mlp")}
        require(initial["roles"] == roles
                and initial["seeds"] == {family: protocol.seed(plan, role) for family, role in roles.items()},
                "Initialization roles/seeds must match planned streams")
        require(order["role"] == row["minibatch_role"]
                and order["seed"] == protocol.seed(plan, row["minibatch_role"]),
                "Order role/seed must match planned stream")
        require(done["initialization_sha256"] == initial["integrity_sha256"]
                and done["orders_sha256"] == order["integrity_sha256"]
                and done["data_sha256"] == training.canonical_tensor_hash(data)
                and done["checkpoint_integrity_sha256"] == checkpoint["integrity_sha256"]
                and done["log_chain_sha256"] == checkpoint["log_chain_sha256"]
                and done["model_configuration"] == checkpoint["model_configuration"]
                and done["settings"] == checkpoint["settings"] == cfg.configuration(),
                "Fit completion metadata must bind exact restored payload")
        require(done["updates"] == done["optimizer_steps"] == cfg.epochs * cfg.batches_per_epoch,
                "Complete fit schedule required")
        restored = training.restore_checkpoint(checkpoint, data, expected_kind=row["arm"],
            expected_settings=cfg, expected_initialization_sha256=initial["integrity_sha256"],
            expected_orders_sha256=order["integrity_sha256"],
            expected_data_sha256=training.canonical_tensor_hash(data),
            expected_source_sha256=plan["sources"], expected_runtime=plan["runtime"])
        weights = torch.load(folder / "weights.pt", weights_only=True, map_location="cpu")
        require(training.canonical_tensor_hash(weights) == done["student_tensor_sha256"]
                == training.canonical_tensor_hash(restored.student.state_dict())
                and restored.successful_updates == done["updates"], "Restored final student and update count")
        models[row["name"]] = restored.student.eval()
        receipts[row["name"]] = {"kind": row["arm"], "model_class": type(restored.student).__name__,
            "checkpoint_sha256": sha(folder / "checkpoint.pt"), "weights_sha256": sha(folder / "weights.pt"),
            "student_tensor_sha256": done["student_tensor_sha256"], "successful_updates": done["updates"],
            "configuration": restored.model_configuration}
    write(out / "restored-models.json", {"models": receipts, "wall_seconds": time.perf_counter() - started,
                                        "unix_time": time.time()})
    check_cap(deadline)
    return models


def collect_predictions(plan, out, deadline, progress):
    started, records = time.perf_counter(), []
    try:
        for index in range(plan["prediction_episodes"]):
            check_cap(deadline)
            progress.update(episode=index, completed_episodes=len(records))
            records.append(collect_episode(protocol.seed(plan, f"prediction/reset/{index}"),
                protocol.schedule(plan, index, "ordinary", "prediction"), noise_std=plan["noise_std"],
                noise_seed=protocol.seed(plan, f"prediction/actuator_noise/{index}"),
                action_seed=protocol.seed(plan, f"prediction/exploration/{index}"), policy="mixed"))
        base.save_records(out / "prediction", records)
        check_cap(deadline)
        return records, time.perf_counter() - started
    except BaseException:
        if records and not (out / "prediction.npz").exists():
            base.save_records(out / "partial-prediction", records)
        write(out / "partial-prediction-state.json", {"completed_episodes": len(records),
            "progress": dict(progress), "wall_seconds": time.perf_counter() - started,
            "unfinished_episode_record_available": False})
        raise


def control_cases(plan, panel):
    return [control.ControlCase(protocol.seed(plan, f"control/reset/{i}"),
                                protocol.seed(plan, f"control/actuator_noise/{i}"),
                                protocol.schedule(plan, i, panel)) for i in range(plan["control_episodes"])]


def json_arrays(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_arrays(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_arrays(item) for item in value]
    return value


def kinematic_control(plan, panel, inputs_by_step, out, deadline, progress):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    envs, template = [], None
    try:
        envs, packets = search_study.control_envs(plan, panel, deadline)
        template = make_env()
        observers = [KinematicMPC(template.unwrapped.model, p, frame_skip=2) for p in packets]
        previous_actions = np.zeros((len(envs), 2), dtype=np.float32)
        setup = time.perf_counter() - started
        scores, estimates, decisions, native_times = [], [], [], []
        for step in range(plan["steps"]):
            progress["step"] = step
            check_cap(deadline)
            tick = time.perf_counter()
            bank = protocol.common_bank(serialization.load_inputs(inputs_by_step[step]), step, plan)
            actions, batch_scores, batch_estimates = [], [], []
            for index, observer in enumerate(observers):
                check_cap(deadline)
                if step:
                    observer.update(previous_actions[index], packets[index])
                qpos, qvel = observer.estimate()
                action, costs = observer.plan(bank[index])
                actions.append(action)
                batch_scores.append(-costs)
                batch_estimates.append(np.concatenate((qpos, qvel)))
            actions = np.asarray(actions, dtype=np.float32)
            scores.append(np.asarray(batch_scores, dtype=np.float64))
            estimates.append(np.asarray(batch_estimates, dtype=np.float64))
            decisions.append(time.perf_counter() - tick)
            check_cap(deadline)
            tick = time.perf_counter()
            packets = np.stack([env.step(action) for env, action in zip(envs, actions, strict=True)])
            native_times.append(time.perf_counter() - tick)
            previous_actions = actions
        search_study.save_episodes(out / "episodes", envs)
        serialization.save_npz(out / "planning.npz", candidate_scores=np.stack(scores, 1),
                               planner_used=np.array(True), public_estimates=np.stack(estimates, 1))
        write(out / "observer-final.json", json_arrays([observer.snapshot() for observer in observers]))
        result = {"setup_seconds": setup, "decision_seconds": decisions, "native_step_seconds": native_times,
                  "row_wall_seconds": time.perf_counter() - started, "candidate_evaluations_per_decision": 64,
                  "information": "public packets, issued commands, supplied nominal model; no true state"}
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
        if template is not None:
            template.close()


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
    begin = time.monotonic() if begin is None else begin
    protocol.validate_settings(plan, engineering=plan.get("engineering") is True)
    require(plan.get("engineering") is True or callable(final_validate), "Scored execution requires validation")
    deadline = begin + plan["cap_seconds"]
    out = Path(out)
    if output_created:
        require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()), "Exclusive empty attempt")
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
        require(len(train) == plan["train_episodes"], "Complete training corpus")
        data = public_training_data(train)
        progress = {"phase": "prepare-training"}
        initializations, orders = prepare_training(plan, out, deadline)
        fits = []
        for row in protocol.fit_manifest(plan):
            progress = {"phase": "fit", "model": row["name"]}
            fits.append(fit_one(plan, row, data, initializations[row["pair"]], orders[row["pair"]],
                                out / "fits" / row["name"], deadline, progress))
            print(json.dumps({"completed_fit": row["name"]}), flush=True)
        write(out / "all-fits-completed.json", {"plan_sha256": expected, "fit_order": plan["fit_order"],
            "files": {f"fits/{name}/completed.json": sha(out / "fits" / name / "completed.json")
                      for name in plan["fit_order"]},
            "training_prepared_sha256": sha(out / "training-prepared.json"),
            "unix_time": time.time(), "elapsed_seconds": time.monotonic() - begin})
        progress = {"phase": "restore-all-models"}
        models = restore_students(plan, out, data, deadline, expected)
        evaluation_started = time.monotonic() - begin
        write(out / "evaluation-started.json", {"plan_sha256": expected,
            "all_fits_completed_sha256": sha(out / "all-fits-completed.json"),
            "restored_models_sha256": sha(out / "restored-models.json"),
            "unix_time": time.time(), "elapsed_seconds": evaluation_started})
        progress = {"phase": "prediction-collection"}
        prediction, collection_seconds = collect_predictions(plan, out, deadline, progress)
        (out / "predictions").mkdir()
        prediction_times = []
        for name, model in models.items():
            progress = {"phase": "prediction", "model": name}
            values, timing = control.prediction_record(plan, model, prediction, deadline)
            serialization.save_npz(out / "predictions" / f"{name}.npz", **values)
            write(out / "predictions" / f"{name}.json", timing)
            prediction_times.append(timing)
        tick, inputs = time.perf_counter(), []
        for step in range(plan["steps"]):
            check_cap(deadline)
            stem = out / "innovations" / "control" / f"{step:03d}"
            serialization.save_inputs(stem, protocol.draw_control_inputs(plan, step), f"planner/control/{step}")
            inputs.append(stem)
        innovation_seconds = time.perf_counter() - tick
        control_times = []
        for row in protocol.execution_order(plan):
            progress = {"phase": "control", "panel": row["panel"], "model": row["label"]}
            folder = out / row["path"]
            if "fit" in row:
                timing = control.learned_control(plan, models[row["fit"]], row["panel"], inputs,
                    folder, deadline, progress, cases=control_cases(plan, row["panel"]))
            elif row["reference"] == "public_kinematic":
                timing = kinematic_control(plan, row["panel"], inputs, folder, deadline, progress)
            else:
                timing = search_study.reference_control(plan, row["panel"], row["reference"],
                                                        inputs, folder, deadline, progress)
            control_times.append(timing)
            print(json.dumps({"completed_control": f"{row['panel']}/{row['label']}"}), flush=True)
        progress = {"phase": "final-validation"}
        if final_validate is not None:
            final_validate()
        costs = {"training_preparation_seconds": read(out / "training-prepared.json")["wall_seconds"],
            "fit_wall_seconds": sum(row["wall_seconds"] for row in fits),
            "restore_wall_seconds": read(out / "restored-models.json")["wall_seconds"],
            "prediction_collection_seconds": collection_seconds,
            "prediction_model_seconds": sum(row["wall_seconds"] for row in prediction_times),
            "innovation_generation_and_storage_seconds": innovation_seconds,
            "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in control_times),
            "control_setup_seconds": sum(row["setup_seconds"] for row in control_times),
            "control_decision_seconds": sum(sum(row["decision_seconds"]) for row in control_times),
            "control_native_step_seconds": sum(sum(row["native_step_seconds"]) for row in control_times),
            "new_fits": len(fits), "astra_calls": 0,
            "prior_costs": plan["memory_source"]["prior_costs"],
            "compute_matched": False,
            "accounting": "Whole execution includes validation, copying, all fitting/restoration, collection, "
                "planning, native stepping, traces and final hashes. Timings are nested and not all additive. "
                "Shared-host batch throughput, not isolated latency. Audits/publication are separate."}
        write(out / "costs.json", costs)
        members = file_members(out, deadline)
        check_cap(deadline)
        total = time.monotonic() - begin
        write(out / "completed.json", {"status": "completed", "study": STUDY, "plan_sha256": expected,
            "fits": len(fits), "control_rows": len(control_times), "prediction_episodes": len(prediction),
            "astra_calls": 0, "wall_seconds": total, "evaluation_started_elapsed_seconds": evaluation_started,
            "cumulative_attempt_wall_seconds": plan["memory_source"]["prior_costs"]["cumulative_attempt_wall_seconds"]
                + total, "files": members})
        check_cap(deadline)
        return {"status": "completed", "fits": len(fits), "control_rows": len(control_times), "wall_seconds": total}
    except BaseException as error:
        if (out / "completed.json").exists():
            (out / "completed.json").rename(out / "over-cap-completion.json")
        write(out / "failed.json", {"status": "failed", "plan_sha256": expected, "error": repr(error),
            "progress": progress, "wall_seconds": time.monotonic() - begin, "files": file_members(out)})
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--cap-seconds", type=int, required=True)
    prep.add_argument("--audit-cap-seconds", type=int, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--expected-plan-sha256", required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.out, cap_seconds=args.cap_seconds, audit_cap_seconds=args.audit_cap_seconds)
    else:
        result = run(args.plan, args.expected_plan_sha256, args.out)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
