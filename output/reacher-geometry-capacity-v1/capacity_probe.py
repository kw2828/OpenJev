"""Engineering capacity only; explicit authorization is required to execute.

run: authenticate all six saved students, probe all 12 model/score combinations,
then measure four whole learned rows, five reference rows and one native root.
audit: independently check saved score arithmetic and replay engineering states.
Both phases have separate fixed 1800-second cooperative caps and exclusive
outputs. Neither phase calculates utility, MSE, rank or winner summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
CAP_SECONDS = 1800
NAMESPACE = "reacher-geometry-engineering-capacity-v1"
PAIR = "pair0"
PANEL = "ordinary"
STEP = 12
SCOPE = "engineering_capacity_only_no_effectiveness_metrics"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Fixed 1800-second cooperative engineering-capacity cap")


def members(folder, deadline):
    result = {}
    for path in sorted(folder.rglob("*")):
        check(deadline)
        require(not path.is_symlink(), "No symlink capacity artifacts")
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def inventory_bytes(folder):
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def hardware_context():
    """Captured during this execution, never retroactively called a run record."""
    result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                            capture_output=True, text=True, check=False, timeout=10)
    return {"cpu_brand": result.stdout.strip() if result.returncode == 0 else None,
            "source": "sysctl during capacity execution", "shared_host": True,
            "recorded_unix_time": time.time()}


def projection(learned, references, diagnostic, probe_seconds, setup_seconds, hash_seconds,
               measured_bytes, projected_bytes):
    learned_seconds = sum(row["timing"]["row_wall_seconds"] for row in learned)
    reference_seconds = sum(row["timing"]["row_wall_seconds"] for row in references)
    hash_projection = hash_seconds * projected_bytes / max(measured_bytes, 1)
    components = {
        "36_learned_rows_seconds": 9 * learned_seconds,
        "15_reference_rows_seconds": 3 * reference_seconds,
        "48_diagnostic_roots_seconds": 48 * diagnostic["row_wall_seconds"],
        "authentication_copy_restoration_and_initial_setup_seconds": setup_seconds,
        "12_probe_conservative_extra_reserve_seconds": probe_seconds,
        "hashing_seconds_scaled_by_recorded_bytes": hash_projection,
    }
    total = sum(components.values())
    return {"components": components, "point_projection_seconds": total,
        "twice_point_projection_seconds": 2 * total,
        "projected_raw_file_bytes": projected_bytes,
        "formula": "9 * four learned rows + 3 * five references + 48 * one diagnostic root + recorded setup + 12 probe reserve + byte-scaled hashing",
        "assumptions": ["Each measured pair0 ordinary row represents its three fit pairs and three panels.",
            "Each reference ordinary row represents all three panels.",
            "The one engineering root represents all48 fixed roots, with actual unique union size recorded.",
            "The twelve probes expose fit variation but are not selected or additional scientific work; their cost is a conservative reserve.",
            "Compression, native stepping and scoring diagnostics storage are inside measured row totals."],
        "omissions": ["Full source/lineage audit and independent full-tree audit overhead beyond separately timed kernels.",
            "Cold filesystem behavior, nonlinear storage pressure, other shared-host load and publication packaging.",
            "Two times the point estimate is a descriptive margin, not a runtime guarantee or calibrated bound."],
        "not_effectiveness_evidence": True}


def run_probe(attempt):
    import numpy as np
    import reacher_geometry_study as study
    import torch

    from openjev.research import reacher_geometry_control as control
    from openjev.research import reacher_geometry_protocol as protocol
    from openjev.research import reacher_search_protocol as artifacts
    from openjev.research.robotics_reacher import ReacherEpisode, make_env

    begin = time.monotonic()
    deadline = begin + CAP_SECONDS
    attempt = attempt.resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    out = attempt / "execution"
    out.mkdir()
    progress = {"phase": "authenticate"}
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        _, parent_source = study.authenticated_sources()
        paths = [*study.SOURCES, Path(__file__).resolve().relative_to(ROOT).as_posix()]
        sources = {name: sha(ROOT / name) for name in paths}
        plan = protocol.settings(engineering=True)
        plan.update(rng_namespace=NAMESPACE, engineering=True,
            engineering_rng_namespaces=[name for name in protocol.ENGINEERING_NAMESPACES if name != NAMESPACE],
            parent_source=parent_source, runtime=study.base.runtime(), sources=sources,
            cap_seconds=CAP_SECONDS, audit_cap_seconds=CAP_SECONDS)
        plan["random_stream_contract"] = study.stream_contract(plan)
        protocol.validate_settings(plan, engineering=True)
        write(out / "capacity-plan.json", plan)
        write(out / "started.json", {"scope": SCOPE, "cap_seconds": CAP_SECONDS,
            "namespace": NAMESPACE, "plan_sha256": sha(out / "capacity-plan.json"),
            "source_sha256": sources, "runtime": plan["runtime"], "hardware": hardware_context(),
            "started_unix_time": time.time(),
            "coverage": {"restored_saved_students": 6, "full_batch_cem_probes": 12,
                "learned_rows": 4, "reference_rows": 5, "cases_per_row": 64, "steps": 50,
                "diagnostic_roots": 1, "root_step": STEP, "native_branches": 4,
                "diagnostic_horizon": 12},
            "outcome_policy": "Persist raw formula/replay evidence; do not calculate utility, error, ranking or selection summaries."})
        write(out / "random-streams.json", plan["random_stream_contract"])
        study.copy_inherited(plan, out, deadline)
        progress = {"phase": "restore_all_six"}
        models = study.restore_students(plan, out, deadline)
        cases = [control.ControlCase(protocol.seed(plan, f"control/reset/{i}"),
            protocol.seed(plan, f"control/actuator_noise/{i}"), protocol.schedule(plan, i, PANEL)) for i in range(64)]
        inputs = []
        for step in range(50):
            check(deadline)
            stem = out / "innovations" / "control" / f"{step:03d}"
            artifacts.save_inputs(stem, protocol.draw_control_inputs(plan, step), f"planner/control/{step}")
            inputs.append(stem)
        packets = []
        for case in cases:
            check(deadline)
            env = ReacherEpisode(noise_std=plan["noise_std"])
            try:
                packets.append(env.reset(case.reset_seed, case.sensor_schedule, noise_seed=case.noise_seed))
            finally:
                env.close()
        packets = np.stack(packets)
        artifacts.save_npz(out / "probe-public-packets.npz", packets=packets)
        setup_seconds = time.monotonic() - begin
        probes = []
        for fit, student in models.items():
            for mode in protocol.SCORE_MODES:
                progress = {"phase": "full_batch_probe", "fit": fit, "score_mode": mode}
                tick = time.perf_counter()
                with torch.no_grad():
                    root = student.assimilate(student.initial(64), torch.from_numpy(packets.copy()))
                label = protocol.policy_name(fit, mode)
                folder = out / "probes" / label
                folder.mkdir(parents=True)
                result, raw, sizes, seconds, work, diagnostic = control.score_search(
                    plan, student, root, artifacts.load_inputs(inputs[0]), 0, deadline,
                    score_mode=mode, failure_out=folder / "partial-scoring")
                artifacts.save_trace(folder / "trace", result, raw, sizes, seconds)
                study.save_scoring(folder / "scoring", diagnostic)
                row = {"fit": fit, "score_mode": mode, "wall_seconds": time.perf_counter() - tick,
                    "search_seconds": seconds, "geometry_seconds": work["geometry_seconds"],
                    "imagined_transitions": work["imagined_transitions"],
                    "diagnostic_array_bytes": work["diagnostic_array_bytes"],
                    "compressed_output_bytes": inventory_bytes(folder)}
                probes.append(row)
                write(folder / "timing.json", row)
                check(deadline)
        learned = []
        for arm in protocol.ARMS:
            fit = f"{arm}-{PAIR}"
            for mode in protocol.SCORE_MODES:
                label = protocol.policy_name(fit, mode)
                progress = {"phase": "whole_learned_row", "fit": fit, "score_mode": mode}
                folder = out / "control" / PANEL / label
                timing = control.learned_control(plan, models[fit], PANEL, inputs, folder, deadline,
                    progress, cases=cases, score_mode=mode)
                learned.append({"fit": fit, "score_mode": mode, "path": folder.relative_to(out).as_posix(),
                                "timing": timing, "compressed_output_bytes": inventory_bytes(folder)})
                print(json.dumps({"engineering_phase": "learned_row_complete", "row": label}), flush=True)
        references = []
        for name in protocol.REFERENCES:
            progress = {"phase": "whole_reference_row", "reference": name}
            folder = out / "control" / PANEL / name
            if name == "public_kinematic":
                timing = study.previous.kinematic_control(plan, PANEL, inputs, folder, deadline, progress)
            else:
                timing = study.search_study.reference_control(plan, PANEL, name, inputs, folder, deadline, progress)
            references.append({"name": name, "path": folder.relative_to(out).as_posix(), "timing": timing,
                               "compressed_output_bytes": inventory_bytes(folder)})
            print(json.dumps({"engineering_phase": "reference_row_complete", "row": name}), flush=True)
        source_path = f"control/{PANEL}/residual_gru-{PAIR}--learned/episodes"
        episode = study.base.load_records(out / source_path)[0]
        diagnostic_row = next(row for row in protocol.diagnostic_manifest(plan)
                              if row["panel"] == PANEL and row["case_index"] == 0 and row["step"] == STEP)
        diagnostic_row = {**diagnostic_row, "source_episode_path": source_path}
        progress = {"phase": "one_full_native_root", **diagnostic_row}
        template = make_env()
        try:
            template.reset(seed=protocol.seed(plan, "diagnostic/native_template/0"))
            diagnostic = study.diagnostic_root(plan, models, episode, diagnostic_row,
                out / diagnostic_row["path"], deadline, progress, template)
        finally:
            template.close()
        diagnostic_bytes = inventory_bytes(out / diagnostic_row["path"])
        write(out / "diagnostic-source.json", {"scope": SCOPE,
            "source_episode_path": source_path, "source_case_index": 0, "root_step": STEP,
            "source_episode_npz_sha256": sha((out / source_path).with_suffix(".npz")),
            "source_episode_json_sha256": sha((out / source_path).with_suffix(".json")),
            "prior_scored_history_used": False})
        restored = read(out / "all-models-restored.json")
        for name, student in models.items():
            check(deadline)
            digest = study.training.canonical_tensor_hash(student.state_dict())
            require(digest == restored["models"][name]["student_tensor_sha256"], "Capacity model tensors changed")
        for name, digest in sources.items():
            check(deadline)
            require(sha(ROOT / name) == digest, "Capacity source changed during execution")
        measured_bytes = inventory_bytes(out)
        projected_bytes = (9 * sum(row["compressed_output_bytes"] for row in learned)
            + 3 * sum(row["compressed_output_bytes"] for row in references) + 48 * diagnostic_bytes
            + inventory_bytes(out / "inherited"))
        tick = time.perf_counter()
        files = members(out, deadline)
        hashing_seconds = time.perf_counter() - tick
        estimate = projection(learned, references, diagnostic, sum(row["wall_seconds"] for row in probes),
                              setup_seconds, hashing_seconds, measured_bytes, projected_bytes)
        elapsed = time.monotonic() - begin
        check(deadline)
        receipt = {"status": "completed", "scope": SCOPE, "cap_seconds": CAP_SECONDS,
            "namespace": NAMESPACE, "source_sha256": sources, "new_fits": 0, "new_optimizer_updates": 0,
            "outcome_summaries_computed": False, "wall_seconds": elapsed, "files": files,
            "setup_seconds": setup_seconds, "probes": probes, "learned_rows": learned,
            "reference_rows": references, "diagnostic": diagnostic,
            "diagnostic_path": diagnostic_row["path"], "diagnostic_source_path": source_path,
            "hashing_seconds": hashing_seconds, "measured_file_bytes": measured_bytes,
            "projection": estimate, "script_sha256": sha(Path(__file__)),
            "audit_status": "not_started_separate_1800_second_phase"}
        write(out / "completed.json", receipt)
        check(deadline)
        return {"status": "completed", "scope": SCOPE, "receipt": str(out / "completed.json"),
                "receipt_sha256": sha(out / "completed.json"), "wall_seconds": elapsed,
                "point_projection_seconds": estimate["point_projection_seconds"]}
    except BaseException as error:
        if (out / "completed.json").exists():
            (out / "completed.json").rename(out / "incomplete-completion.json")
        write(out / "failed.json", {"status": "failed", "scope": SCOPE, "error": repr(error),
            "progress": progress, "wall_seconds": time.monotonic() - begin, "no_retry": True})
        raise


def audit_probe(attempt, expected_completed_sha256):
    import audit_reacher_geometry_study as audit
    import numpy as np

    from openjev.research.robotics_reacher import native_replay

    begin = time.monotonic()
    deadline = begin + CAP_SECONDS
    attempt = attempt.resolve()
    execution, out = attempt / "execution", attempt / "audit"
    out.mkdir(exist_ok=False)
    token = audit._DEADLINE.set(deadline)
    progress = {"phase": "authenticate_saved_capacity"}
    try:
        require(sha(execution / "completed.json") == expected_completed_sha256, "Capacity receipt SHA")
        completed = read(execution / "completed.json")
        require(completed["status"] == "completed" and completed["scope"] == SCOPE,
                "Completed engineering capacity only")
        require(completed["script_sha256"] == sha(Path(__file__)), "Capacity helper source changed")
        files = members(execution, deadline)
        files.pop("completed.json")
        require(files == completed["files"], "Exact completed capacity file membership/bytes")
        for name, digest in completed["source_sha256"].items():
            check(deadline)
            require(sha(ROOT / name) == digest, "Audit sources differ from measured sources")
        plan = read(execution / "capacity-plan.json")
        write(out / "started.json", {"scope": SCOPE, "cap_seconds": CAP_SECONDS,
            "completed_sha256": expected_completed_sha256, "script_sha256": sha(Path(__file__)),
            "new_model_calls": 0, "new_policy_calls": 0, "started_unix_time": time.time()})

        def check_score(stem, mode):
            check(deadline)
            tick = time.perf_counter()
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as data:
                keys = {"commands", "root_target", "predicted_angles", "learned_rewards", "selected_rewards"}
                if mode == "geometry":
                    keys |= {"geometry_" + key for key in audit.GEOMETRY_FIELDS}
                values = {key: data[key] for key in keys}
                counts = audit.audit_bank_arrays(plan, mode, values, values["commands"], values["root_target"])
                selected_samples = 0
                if "selected_angles" in data.files:
                    n = len(data["selected_angles"])
                    chosen = {"commands": data["selected_actions"][:, None, None],
                        "root_target": data["root_target"],
                        "predicted_angles": data["selected_angles"][:, None, None],
                        "learned_rewards": data["selected_learned_reward"][:, None, None],
                        "selected_rewards": data["selected_reward"][:, None, None]}
                    if mode == "geometry":
                        chosen.update({"geometry_" + key: data["selected_geometry_" + key][:, None, None]
                                       for key in audit.GEOMETRY_FIELDS})
                    audit.audit_bank_arrays(plan, mode, chosen, chosen["commands"], chosen["root_target"])
                    selected_samples = n
            return {"wall_seconds": time.perf_counter() - tick,
                "candidate_transition_samples": counts["imagined_transitions"],
                "selected_samples": selected_samples, "geometry_samples": counts["geometry_samples"]
                    + (selected_samples if mode == "geometry" else 0)}

        probe_checks = []
        for row in completed["probes"]:
            progress = {"phase": "probe_formula", "fit": row["fit"], "score_mode": row["score_mode"]}
            name = f"{row['fit']}--{row['score_mode']}"
            probe_checks.append(check_score(execution / "probes" / name / "scoring", row["score_mode"]))
        learned_checks, reference_checks = [], []
        for row in [*completed["learned_rows"], *completed["reference_rows"]]:
            progress = {"phase": "row_formula_and_replay", "path": row["path"]}
            folder = execution / row["path"]
            score_checks = [check_score(folder / "scoring" / f"{step:03d}", row["score_mode"])
                            for step in range(50)] if "fit" in row else []
            native_tick = time.perf_counter()
            records = audit.base.load_records(folder / "episodes", plan["control_episodes"])
            transitions, max_error = 0, 0.0
            for record in records:
                check(deadline)
                replay = native_replay(record)
                transitions += replay["transitions"]
                max_error = max(max_error, replay["max_abs_error"])
            result = {"path": row["path"], "formula_seconds": sum(item["wall_seconds"] for item in score_checks),
                "native_replay_seconds": time.perf_counter() - native_tick, "native_transitions": transitions,
                "native_max_abs_discrepancy": max_error, "score_checks": score_checks}
            (learned_checks if "fit" in row else reference_checks).append(result)
        progress = {"phase": "diagnostic_formula_and_native_replay"}
        diagnostic_folder = execution / completed["diagnostic_path"]
        diagnostic_tick = time.perf_counter()
        diagnostic_checks = []
        for fit in plan["fit_order"]:
            for mode in plan["score_modes"]:
                label = f"{fit}--{mode}"
                for category in ("search-scoring", "scores"):
                    diagnostic_checks.append(check_score(diagnostic_folder / category / label, mode))
        source = audit.base.load_records(
            execution / completed["diagnostic_source_path"], plan["control_episodes"])[0]
        with np.load(diagnostic_folder / "union.npz", allow_pickle=False) as data:
            commands, noise = data["commands"], data["noise"]
        with np.load(diagnostic_folder / "native.npz", allow_pickle=False) as data:
            saved = {key: data[key] for key in data.files}
        check(deadline)
        branches = audit.search_audit.replay_branches(source, STEP, commands, noise, saved)
        check(deadline)
        diagnostic_seconds = time.perf_counter() - diagnostic_tick
        learned_seconds = sum(row["formula_seconds"] + row["native_replay_seconds"] for row in learned_checks)
        reference_seconds = sum(row["native_replay_seconds"] for row in reference_checks)
        estimate = 9 * learned_seconds + 3 * reference_seconds + 48 * diagnostic_seconds
        elapsed = time.monotonic() - begin
        require(math.isfinite(estimate), "Finite descriptive audit timing projection")
        check(deadline)
        receipt = {"status": "completed", "scope": SCOPE, "completed_sha256": expected_completed_sha256,
            "wall_seconds": elapsed, "cap_seconds": CAP_SECONDS, "probe_checks": probe_checks,
            "learned_rows": learned_checks, "reference_rows": reference_checks,
            "diagnostic_formula_checks": diagnostic_checks, "diagnostic_wall_seconds": diagnostic_seconds,
            "diagnostic_native_replay": branches, "projected_kernel_audit_seconds": estimate,
            "twice_projected_kernel_audit_seconds": 2 * estimate,
            "projection_omits": "Full source/lineage validation, before/after weight checks, search-proposal reconstruction, state/counter checks, bootstrap and full artifact hashing/publication. Whole-tree rehearsal must measure remaining audit wiring.",
            "native_transitions_checked": sum(row["native_transitions"] for row in [*learned_checks, *reference_checks]) + branches["transitions"],
            "new_model_calls": 0, "new_policy_calls": 0, "new_fits": 0,
            "outcome_summaries_computed": False, "script_sha256": sha(Path(__file__)),
            "files": members(out, deadline)}
        write(out / "completed.json", receipt)
        check(deadline)
        return {"status": "completed", "scope": SCOPE, "receipt": str(out / "completed.json"),
                "receipt_sha256": sha(out / "completed.json"), "wall_seconds": elapsed,
                "projected_kernel_audit_seconds": estimate}
    except BaseException as error:
        if (out / "completed.json").exists():
            (out / "completed.json").rename(out / "incomplete-completion.json")
        write(out / "failed.json", {"status": "failed", "scope": SCOPE, "error": repr(error),
            "progress": progress, "wall_seconds": time.monotonic() - begin, "no_retry": True})
        raise
    finally:
        audit._DEADLINE.reset(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-engineering", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--attempt", type=Path, required=True)
    audit.add_argument("--expected-completed-sha256", required=True)
    args = parser.parse_args()
    require(args.execute_engineering, "Explicit root authorization required before capacity execution")
    result = run_probe(args.out) if args.command == "run" else audit_probe(args.attempt, args.expected_completed_sha256)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
