"""Zero-fit reward-scoring intervention on six authenticated saved students.

The exposed-root diagnostic precedes fresh closed-loop evaluation. Neither
stage changes weights, planner budget, hyperparameters or the other stage.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import reacher_cache_study as previous
import torch

from openjev.research import reacher_geometry_control as control
from openjev.research import reacher_geometry_protocol as protocol
from openjev.research import reacher_search_protocol as artifacts
from openjev.research.robotics_reacher import make_env

ROOT = Path(__file__).resolve().parents[1]
base, search_study, training = previous.base, previous.search_study, previous.training
sha, read, write, require = previous.sha, previous.read, previous.write, previous.require
checked, check_cap, file_members = previous.checked, previous.check_cap, previous.file_members
STUDY = "reacher-geometry-score-v1"
CACHE_PLAN = "evidence/reacher-cache-ablation-v1/protocol/plan.json"
CACHE_PLAN_SHA = "7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa"
CACHE_AUDIT = "evidence/reacher-cache-ablation-v1/audit/receipt.json"
CACHE_AUDIT_SHA = "d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790"
CACHE_EXECUTION = "runs/reacher-cache-ablation-v1/attempt"
NEW_SOURCES = (
    "src/openjev/research/reacher_geometry_reward.py",
    "tests/test_reacher_geometry_reward.py",
    "src/openjev/research/reacher_geometry_control.py",
    "tests/test_reacher_geometry_control.py",
    "src/openjev/research/reacher_geometry_protocol.py",
    "tests/test_reacher_geometry_protocol.py",
    "scripts/reacher_geometry_study.py",
    "tests/test_reacher_geometry_study.py",
    "scripts/audit_reacher_geometry_study.py",
    "tests/test_audit_reacher_geometry_study.py",
)
SOURCES = previous.SOURCES + NEW_SOURCES


def inherited_members(parent):
    names = {"train.npz", "train.json"}
    for pair in protocol.PAIRS:
        names.update(f"{folder}/{pair}.{suffix}" for folder in ("initializations", "orders")
                     for suffix in ("pt", "json"))
        for arm in protocol.ARMS:
            names.update(f"fits/{arm}-{pair}/{member}" for member in (
                "initial-weights.pt", "training.jsonl", "weights.pt", "checkpoint.pt", "completed.json"))
    for panel in protocol.PANELS:
        names.update(f"control/{panel}/residual_gru-pair0/episodes.{suffix}" for suffix in ("npz", "json"))
    require(set(protocol.ARMS) <= set(parent["arms"]), "Inherited actual model families")
    return sorted(names)


def authenticate_parent(plan_path, plan_digest, audit_path, audit_digest, execution_path, *, engineering=False):
    """Bind completed prior evidence; never rerun its model or optimizer."""
    parent = read(checked(ROOT / plan_path, plan_digest))
    audit = read(checked(ROOT / audit_path, audit_digest))
    execution = ROOT / execution_path
    done = read(checked(execution / "completed.json", audit["execution_completed_sha256"]))
    require(audit["status"] == done["status"] == "completed" and audit["saved_output_only"] is True
            and audit["engineering"] is engineering and parent["engineering"] is engineering
            and audit["plan_sha256"] == done["plan_sha256"] == plan_digest
            and audit["source_sha256"] == parent["sources"]
            and audit["execution_members"] == done["files"], "Completed prior plan/audit/execution identity")
    require(set(parent["sources"]) == set(previous.SOURCES), "Exact inherited source membership")
    for name, digest in parent["sources"].items():
        checked(ROOT / name, digest)
    for name, digest in audit["files"].items():
        checked((ROOT / audit_path).parent / name, digest)
    names = inherited_members(parent)
    require(set(names) <= set(done["files"]), "Every inherited file was audited")
    members = {name: done["files"][name] for name in names}
    for name, digest in members.items():
        checked(execution / name, digest)
    source = {"plan_path": plan_path, "plan_sha256": plan_digest, "audit_path": audit_path,
        "audit_receipt_sha256": audit_digest, "execution_path": execution_path,
        "completed_sha256": audit["execution_completed_sha256"],
        "summary_sha256": audit["files"]["summary.json"], "members": members,
        "prior_costs": audit["costs"], "engineering": engineering,
        "training_weights_reused": True, "new_fits": 0,
        "previous_scientific_gate_passed": False if not engineering else None}
    return parent, source


def authenticated_sources():
    previous.authenticated_sources()
    return authenticate_parent(CACHE_PLAN, CACHE_PLAN_SHA, CACHE_AUDIT, CACHE_AUDIT_SHA, CACHE_EXECUTION)


def prior_streams():
    numpy_prior, torch_prior, descriptors = previous.prior_streams()
    parent = read(checked(ROOT / CACHE_PLAN, CACHE_PLAN_SHA))
    contract = parent["random_stream_contract"]
    for item, kind in [(contract, "cache-scored"),
                       *((item, "cache-engineering") for item in contract["engineering_exclusions"])]:
        numpy_prior.append(item["registry"])
        torch_prior.append(item["torch_registry"])
        descriptors.append({"plan_path": CACHE_PLAN, "plan_sha256": CACHE_PLAN_SHA,
                            "registry_kind": kind, "namespace": item.get("namespace")})
    return numpy_prior, torch_prior, descriptors


def stream_contract(plan):
    return protocol.stream_contract(plan, *prior_streams())


def prepare(out, *, cap_seconds, audit_cap_seconds):
    require(all(type(v) is int and v > 0 for v in (cap_seconds, audit_cap_seconds)), "Explicit positive caps")
    parent, source = authenticated_sources()
    plan = protocol.settings()
    plan.update(parent_source=source, cap_seconds=cap_seconds, audit_cap_seconds=audit_cap_seconds,
        sources={name: sha(ROOT / name) for name in SOURCES}, runtime=base.runtime(), engineering=False,
        stop="One whole execution and separate saved-output audit. No retries, resumes, replacement seeds, "
             "cap extensions, selected checkpoints or post-outcome changes.")
    require(all(plan[k] == parent[k] for k in ("hidden_size", "mlp_width", "dt", "noise_std")),
            "Unchanged inherited model configuration")
    plan["random_stream_contract"] = stream_contract(plan)
    protocol.validate_settings(plan)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    return {"path": str(out / "plan.json"), "sha256": sha(out / "plan.json")}


def validate(path, expected):
    plan = read(checked(path, expected))
    require(plan["engineering"] is False, "Scored protocol required")
    protocol.validate_settings(plan)
    require(set(plan["sources"]) == set(SOURCES), "Exact geometry source membership")
    for name, digest in plan["sources"].items():
        checked(ROOT / name, digest)
    require(plan["runtime"] == base.runtime(), "Frozen runtime changed")
    _, source = authenticated_sources()
    require(plan["parent_source"] == source and plan["random_stream_contract"] == stream_contract(plan),
            "Frozen lineage and random streams")
    require(all(type(plan[k]) is int and plan[k] > 0 for k in ("cap_seconds", "audit_cap_seconds")), "Frozen caps")
    return plan


def copy_inherited(plan, out, deadline):
    source = plan["parent_source"]
    for name, digest in source["members"].items():
        search_study.copy_verified(ROOT / source["execution_path"] / name, out / "inherited" / name,
                                   digest, deadline)
    for name, path, digest in (
        ("source-plan.json", ROOT / source["plan_path"], source["plan_sha256"]),
        ("source-audit.json", ROOT / source["audit_path"], source["audit_receipt_sha256"]),
        ("source-completed.json", ROOT / source["execution_path"] / "completed.json", source["completed_sha256"]),
        ("source-summary.json", (ROOT / source["audit_path"]).parent / "summary.json", source["summary_sha256"]),
    ):
        search_study.copy_verified(path, out / "inherited" / name, digest, deadline)
    write(out / "inheritance.json", source)


def restore_students(plan, out, deadline):
    """Construct only deployment students; no trainer or optimizer restoration."""
    started, models, receipts = time.perf_counter(), {}, {}
    parent = read(out / "inherited" / "source-plan.json")
    cfg = previous.settings_from_plan(parent)
    data = previous.public_training_data(base.load_records(out / "inherited" / "train"))
    data_hash = training.canonical_tensor_hash(data)
    for row in protocol.fit_manifest(plan):
        check_cap(deadline)
        folder = out / "inherited" / "fits" / row["name"]
        done = read(folder / "completed.json")
        for name, digest in done["files"].items():
            checked(folder / name, digest)
        checkpoint = torch.load(folder / "checkpoint.pt", weights_only=True, map_location="cpu")
        integrity = training.canonical_state_hash({k: v for k, v in checkpoint.items() if k != "integrity_sha256"})
        config = training.model_configuration(row["arm"], cfg)
        require(integrity == checkpoint["integrity_sha256"] == done["checkpoint_integrity_sha256"]
                and checkpoint["version"] == training.VERSION and checkpoint["failed"] is False
                and checkpoint["kind"] == row["arm"]
                and checkpoint["settings"] == done["settings"] == cfg.configuration()
                and checkpoint["model_configuration"] == done["model_configuration"] == config
                and checkpoint["source_sha256"] == parent["sources"]
                and checkpoint["runtime"] == parent["runtime"] == plan["runtime"]
                and checkpoint["data_sha256"] == done["data_sha256"] == data_hash,
                "Authenticated unchanged completed checkpoint configuration")
        updates = cfg.epochs * cfg.batches_per_epoch
        require(done["name"] == row["name"] and done["arm"] == row["arm"] and done["pair"] == row["pair"]
                and checkpoint["successful_updates"] == checkpoint["optimizer_steps"]
                == done["updates"] == done["optimizer_steps"] == updates
                and checkpoint["cursor"] == {"epoch": cfg.epochs, "batch": 0}, "Complete inherited fit schedule")
        for field, directory in (("initialization", "initializations"), ("orders", "orders")):
            original = torch.load(out / "inherited" / directory / f"{row['pair']}.pt",
                                  weights_only=True, map_location="cpu")
            require(training.canonical_state_hash(original) == training.canonical_state_hash(checkpoint[field])
                    and original["integrity_sha256"] == done[field + "_sha256"], "Inherited fit input identity")
        weights = torch.load(folder / "weights.pt", weights_only=True, map_location="cpu")
        digest = training.canonical_tensor_hash(weights)
        require(digest == done["student_tensor_sha256"] == training.canonical_tensor_hash(checkpoint["student_state"]),
                "Exact saved student tensors")
        # Isolated constructor seed0 is an excluded engineering role. Every
        # parameter is overwritten by the authenticated checkpoint before use.
        model = training._construct(row["arm"], cfg, 0)
        training._load_weights(model, checkpoint["student_state"])
        model.eval().requires_grad_(False)
        require(training.canonical_tensor_hash(model.state_dict()) == digest, "Deployment tensor restoration")
        control.model_identity(model, plan)
        snapshot = out / "model-states" / f"{row['name']}-before.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        previous.save_torch(snapshot, model.state_dict())
        models[row["name"]] = model
        receipts[row["name"]] = {"kind": row["arm"], "model_class": type(model).__name__,
            "checkpoint_sha256": sha(folder / "checkpoint.pt"), "weights_sha256": sha(folder / "weights.pt"),
            "student_tensor_sha256": digest, "successful_updates": updates, "configuration": config,
            "snapshot_path": snapshot.relative_to(out).as_posix(), "snapshot_sha256": sha(snapshot)}
    write(out / "all-models-restored.json", {"models": receipts, "optimizer_constructed": False,
        "new_updates": 0, "constructor_seed": 0, "constructor_rng_isolated_and_weights_overwritten": True,
        "wall_seconds": time.perf_counter() - started, "unix_time": time.time()})
    check_cap(deadline)
    return models


def deduplicate_sequences(slots):
    require(slots.dtype == np.float32 and slots.ndim == 3 and slots.shape[2] == 2,
            "Float32 identity slot sequences")
    indices, first, seen = [], [], {}
    for index, sequence in enumerate(slots):
        key = sequence.tobytes(order="C")
        if key not in seen:
            seen[key] = len(first)
            first.append(index)
        indices.append(seen[key])
    return slots[first].copy(), np.asarray(indices, dtype=np.int64), np.asarray(first, dtype=np.int64)


def save_scoring(stem, diagnostic):
    stem.parent.mkdir(parents=True, exist_ok=True)
    artifacts.save_npz(stem.with_suffix(".npz"), **diagnostic["arrays"])
    write(stem.with_suffix(".json"), diagnostic["metadata"])


def diagnostic_root(plan, models, episode, row, out, deadline, progress, template):
    begin = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    panel, index, step = row["panel"], row["case_index"], row["step"]
    default_source_path = f"inherited/control/{panel}/residual_gru-pair0/episodes"
    source_path = row.get("source_episode_path", default_source_path)
    require(source_path == default_source_path or plan.get("engineering") is True,
            "Only engineering capacity may supply its own development history")
    require(isinstance(source_path, str) and not Path(source_path).is_absolute()
            and ".." not in Path(source_path).parts, "Relative source episode attribution")
    inputs = protocol.diagnostic_inputs(plan, panel, index, step)
    artifacts.save_inputs(out / "innovations", inputs, f"planner/diagnostic/{panel}/{index}/{step}")
    bank = protocol.common_bank(inputs, step, plan)[0]
    slots, slot_ids = list(bank.copy()), [f"common/{i}" for i in range(64)]
    history = {"packets": episode["policy"]["packets"][:step + 1].copy(),
               "commands": episode["policy"]["commands"][:step].copy()}
    artifacts.save_npz(out / "history.npz", **history)
    (out / "states").mkdir()
    states, beliefs, search_order = {}, {}, []
    search_seconds, belief_seconds, union_score_seconds = 0., 0., 0.
    for fit, model in models.items():
        tick = time.perf_counter()
        state = search_study.belief_at(model, **history, deadline=deadline)
        elapsed = time.perf_counter() - tick
        belief_seconds += elapsed
        arrays = {key: value.detach().numpy().copy() for key, value in state.items()}
        artifacts.save_npz(out / "states" / f"{fit}.npz", **arrays)
        beliefs[fit] = {"history_sha256": sha(out / "history.npz"), "observation_assimilations": step + 1,
            "history_action_advances": step, "state_sha256": training.canonical_tensor_hash(state),
            "model_tensor_sha256": training.canonical_tensor_hash(model.state_dict()), "wall_seconds": elapsed}
        states[fit] = state
        for mode in protocol.SCORE_MODES:
            label = f"{fit}--{mode}"
            progress.update(fit=fit, score_mode=mode)
            result, raw, sizes, elapsed, _, diagnostic = control.score_search(
                plan, model, state, inputs, step, deadline, score_mode=mode,
                failure_out=out / "partial-search" / label)
            artifacts.save_trace(out / "search" / label, result, raw, sizes, elapsed)
            save_scoring(out / "search-scoring" / label, diagnostic)
            search_seconds += elapsed
            search_order.append({"fit": fit, "score_mode": mode, "label": label, "slot_index": len(slots)})
            slots.append(result.selected_sequences[0].copy())
            slot_ids.append(f"selected/{label}")
    slot_commands = np.stack(slots)
    commands, mapping, first = deduplicate_sequences(slot_commands)
    noise = protocol.diagnostic_noise(plan, panel, index, step)
    artifacts.save_npz(out / "union.npz", slot_commands=slot_commands, commands=commands,
                       slot_to_unique=mapping, unique_first_slots=first, noise=noise)
    root_record = {"step": step, "integration_state": episode["audit"]["integration_state"][step].copy()}
    artifacts.save_npz(out / "native-root.npz", integration_state=root_record["integration_state"])
    write(out / "root.json", {"panel": panel, "case_index": index, "step": step,
        "horizon": commands.shape[1], "source_controller": "residual_gru-pair0",
        "source_episode_path": source_path,
        "identity_slots": len(slots), "unique_sequence_count": len(commands),
        "slot_ids": slot_ids, "search_order": search_order, "beliefs": beliefs,
        "history_sha256": sha(out / "history.npz"), "deduplication": "first occurrence of exact float32 bytes",
        "scope": "exposed prior development histories; native hidden state used for offline labels only"})
    for fit, model in models.items():
        for mode in protocol.SCORE_MODES:
            tick = time.perf_counter()
            scores, diagnostic = control.score_bank(plan, model, states[fit], commands[None], mode,
                deadline, failure_out=out / "partial-union-scoring" / f"{fit}--{mode}")
            diagnostic["arrays"]["scores"] = scores
            save_scoring(out / "scores" / f"{fit}--{mode}", diagnostic)
            union_score_seconds += time.perf_counter() - tick
    completed, indices, tick = [], [], time.perf_counter()
    try:
        for candidate, sequence in enumerate(commands):
            for branch in range(plan["diagnostic_branches"]):
                progress.update(native_candidate=candidate, native_branch=branch)
                completed.append(search_study.native_branch(root_record, sequence, noise[branch], deadline, env=template))
                indices.append((candidate, branch))
        arrays = {key: np.stack([value[key] for value in completed]).reshape(
            len(commands), plan["diagnostic_branches"], *completed[0][key].shape) for key in completed[0]}
        artifacts.save_npz(out / "native.npz", **arrays)
    except BaseException:
        if completed:
            artifacts.save_npz(out / "native-partial.npz", indices=np.asarray(indices, dtype=np.int64),
                **{key: np.stack([value[key] for value in completed]) for key in completed[0]})
        raise
    timing = {"belief_seconds": belief_seconds, "search_seconds": search_seconds,
        "union_score_seconds": union_score_seconds, "native_and_storage_seconds": time.perf_counter() - tick,
        "row_wall_seconds": time.perf_counter() - begin, "identity_slots": len(slots),
        "unique_sequences": len(commands), "native_transitions": len(commands) * len(noise) * commands.shape[1],
        "observation_assimilations": len(models) * (step + 1), "history_action_advances": len(models) * step}
    write(out / "timings.json", timing)
    check_cap(deadline)
    return timing


def run(path, expected, out):
    begin, out = time.monotonic(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan = validate(path, expected)
        check_cap(begin + plan["cap_seconds"])
        return run_validated(plan, expected, out, begin=begin, output_created=True,
                             final_validate=lambda: validate(path, expected))
    except BaseException as error:
        if not (out / "failed.json").exists():
            write(out / "failed.json", {"status": "failed", "plan_sha256": expected, "error": repr(error),
                "progress": {"phase": "validate-or-setup"}, "wall_seconds": time.monotonic() - begin})
        raise


def run_validated(plan, expected, out, *, begin=None, final_validate=None, output_created=False):
    begin = time.monotonic() if begin is None else begin
    protocol.validate_settings(plan, engineering=plan.get("engineering") is True)
    require(callable(final_validate), "Every execution requires final source/lineage validation")
    deadline, out = begin + plan["cap_seconds"], Path(out)
    if output_created:
        require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()), "Exclusive empty attempt")
    else:
        out.mkdir(parents=True, exist_ok=False)
    progress = {"phase": "copy-inherited"}
    try:
        check_cap(deadline)
        torch.set_num_threads(plan["threads"])
        torch.use_deterministic_algorithms(True)
        write(out / "started.json", {"plan_sha256": expected, "unix_time": time.time()})
        write(out / "random-streams.json", plan["random_stream_contract"])
        copy_inherited(plan, out, deadline)
        progress = {"phase": "restore-all-models"}
        models = restore_students(plan, out, deadline)
        evaluation_started = time.monotonic() - begin
        write(out / "evaluation-started.json", {"plan_sha256": expected,
            "all_models_restored_sha256": sha(out / "all-models-restored.json"),
            "unix_time": time.time(), "elapsed_seconds": evaluation_started})
        episodes = {panel: base.load_records(out / "inherited" / "control" / panel / "residual_gru-pair0" / "episodes")
                    for panel in protocol.PANELS}
        diagnostic_times = []
        template = make_env()
        try:
            template.reset(seed=protocol.seed(plan, "diagnostic/native_template/0"))
            for row in protocol.diagnostic_manifest(plan):
                progress = {"phase": "diagnostic", **row}
                diagnostic_times.append(diagnostic_root(plan, models, episodes[row["panel"]][row["case_index"]],
                    row, out / row["path"], deadline, progress, template))
                print(json.dumps({"completed_diagnostic": row["path"]}), flush=True)
        finally:
            template.close()
        write(out / "diagnostic-completed.json", {"roots": len(diagnostic_times), "unix_time": time.time(),
            "native_transitions": sum(row["native_transitions"] for row in diagnostic_times),
            "timings": diagnostic_times, "plan_sha256": expected})
        inputs, tick = [], time.perf_counter()
        for step in range(plan["steps"]):
            check_cap(deadline)
            stem = out / "innovations" / "control" / f"{step:03d}"
            artifacts.save_inputs(stem, protocol.draw_control_inputs(plan, step), f"planner/control/{step}")
            inputs.append(stem)
        innovation_seconds, control_times = time.perf_counter() - tick, []
        for row in protocol.execution_order(plan):
            progress = {"phase": "control", "panel": row["panel"], "model": row["label"]}
            folder = out / row["path"]
            if "fit" in row:
                cases = [control.ControlCase(protocol.seed(plan, f"control/reset/{i}"),
                    protocol.seed(plan, f"control/actuator_noise/{i}"), protocol.schedule(plan, i, row["panel"]))
                    for i in range(plan["control_episodes"])]
                timing = control.learned_control(plan, models[row["fit"]], row["panel"], inputs, folder,
                    deadline, progress, cases=cases, score_mode=row["score_mode"])
            elif row["reference"] == "public_kinematic":
                timing = previous.kinematic_control(plan, row["panel"], inputs, folder, deadline, progress)
            else:
                timing = search_study.reference_control(plan, row["panel"], row["reference"], inputs,
                                                        folder, deadline, progress)
            control_times.append(timing)
            print(json.dumps({"completed_control": f"{row['panel']}/{row['label']}"}), flush=True)
        write(out / "control-completed.json", {"rows": len(control_times), "unix_time": time.time(),
            "plan_sha256": expected, "diagnostic_completed_sha256": sha(out / "diagnostic-completed.json")})
        progress = {"phase": "final-validation"}
        final_validate()
        restored = read(out / "all-models-restored.json")
        final_hashes = {name: training.canonical_tensor_hash(model.state_dict()) for name, model in models.items()}
        require(all(digest == restored["models"][name]["student_tensor_sha256"] for name, digest in final_hashes.items()),
                "Model weights unchanged across all diagnostic and control work")
        final_snapshots = {}
        for name, model in models.items():
            snapshot = out / "model-states" / f"{name}-after.pt"
            previous.save_torch(snapshot, model.state_dict())
            final_snapshots[snapshot.relative_to(out).as_posix()] = sha(snapshot)
        write(out / "final-models.json", {"student_tensor_sha256": final_hashes,
            "files": final_snapshots, "new_updates": 0})
        costs = {"restore_wall_seconds": restored["wall_seconds"], "new_fits": 0, "new_optimizer_steps": 0,
            "diagnostic_root_wall_seconds": sum(row["row_wall_seconds"] for row in diagnostic_times),
            "diagnostic_native_transitions": sum(row["native_transitions"] for row in diagnostic_times),
            "innovation_generation_and_storage_seconds": innovation_seconds,
            "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in control_times),
            "control_setup_seconds": sum(row["setup_seconds"] for row in control_times),
            "control_decision_seconds": sum(sum(row["decision_seconds"]) for row in control_times),
            "control_native_step_seconds": sum(sum(row["native_step_seconds"]) for row in control_times),
            "control_native_transitions": len(control_times) * plan["control_episodes"] * plan["steps"],
            "astra_calls": 0, "prior_costs": plan["parent_source"]["prior_costs"], "compute_matched": False,
            "accounting": "Whole execution includes validation, copying, six restorations, diagnostic scoring/native "
                "branches, control planning/native steps, complete scoring arrays and hashes. Nested timings are not "
                "additive. Shared-host batch throughput, not isolated latency. Audits/publication are separate."}
        write(out / "costs.json", costs)
        members = file_members(out, deadline)
        check_cap(deadline)
        total = time.monotonic() - begin
        write(out / "completed.json", {"status": "completed", "study": STUDY, "plan_sha256": expected,
            "new_fits": 0, "restored_models": len(models), "control_rows": len(control_times),
            "diagnostic_roots": len(diagnostic_times), "astra_calls": 0, "wall_seconds": total,
            "evaluation_started_elapsed_seconds": evaluation_started,
            "cumulative_attempt_wall_seconds": plan["parent_source"]["prior_costs"]["cumulative_attempt_wall_seconds"]
                + total, "files": members})
        check_cap(deadline)
        return {"status": "completed", "control_rows": len(control_times), "wall_seconds": total}
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
    result = prepare(args.out, cap_seconds=args.cap_seconds, audit_cap_seconds=args.audit_cap_seconds) \
        if args.command == "prepare" else run(args.plan, args.expected_plan_sha256, args.out)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
