"""Explicit-input native orchestration for the two-observation controller.

This component allocates no seeds, generates no search innovations and launches
nothing on import. The enclosing runner must authenticate model, plan, cases and
all fifty saved SearchInputs stems. Native state is used only for trace storage;
the controller receives copies of current public packets and ACTUALLY issued
commands from the native record's policy allowlist. Terminal observation51 is
saved, with no fifty-first decision or assimilation.

Success retains legacy native/search/scoring formats plus per-decision public
boundary evidence. Failure is terminal: available model decisions may exceed
executed native steps, and both counts are recorded without pretending they are
aligned. No automatic retry, replacement case, fallback action or silent clip.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np

from openjev.research import reacher_search_protocol as artifacts
from openjev.research.reacher_cache_control import (
    PANELS,
    ControlCase,
    _plan,
    _public_arrays,
    _save_partial,
    _save_records,
    check_cap,
    require,
)
from openjev.research.reacher_two_observation_control import (
    TwoObservationController,
    model_identity,
    work_accounting,
)
from openjev.research.robotics_reacher import HORIZON, ReacherEpisode

VERSION = "reacher-two-observation-episode-v1"


def _states(rows):
    return {f"{label}__{key}": np.stack([row[key] for row in group], axis=1)
            for label, group in (("root", [row["root_state"] for row in rows]),
                                 ("carried", [row["carried_state"] for row in rows])) for key in group[0]}


def _public_record(record, *, step, returned_packet, expected_command=None):
    """Read only policy allowlist fields; never send the audit category upstream."""
    require(isinstance(record, dict) and isinstance(record.get("policy"), dict)
            and set(record["policy"]) == {"packets", "commands"}, "Exact native policy allowlist")
    packets, commands = record["policy"]["packets"], record["policy"]["commands"]
    _public_arrays(packets, commands)
    require(packets.shape == (step + 1, 8) and commands.shape == (step, 2), "Native public prefix lengths")
    require(np.array_equal(packets[-1], returned_packet), "Native returned packet differs from recorded packet")
    if expected_command is not None:
        require(step > 0 and np.array_equal(commands[-1], expected_command),
                "Native actually issued command differs from selected command")
    return packets[-1].copy(), None if step == 0 else commands[-1].copy()


def _save_decision(out, step, decision):
    artifacts.save_trace(out / "decisions" / f"{step:03d}", decision.search_result,
                         decision.raw_rewards, decision.callback_sizes, decision.metadata["search_seconds"])
    artifacts.save_npz(out / "scoring" / f"{step:03d}.npz", **decision.diagnostics["arrays"])
    artifacts.write(out / "scoring" / f"{step:03d}.json", decision.diagnostics["metadata"])
    artifacts.write(out / "controller-decisions" / f"{step:03d}.json", decision.metadata)


def learned_control(plan, model, panel, inputs_by_step, out, deadline=float("inf"), progress=None, *, cases):
    """Run exactly fifty decisions with caller-supplied cases and saved inputs.

    Input paths are SearchInputs stems, whose .npz/.json bytes are hashed before
    and after loading. inputs.json retains their exact identities; the enclosing
    publication must include those external files. The row has no RNG authority.

    Timings include validation, input loading, controller work, native stepping,
    native-record copying, trace writes, final aggregation, closure and hashing.
    The terminal receipt's own write is unavoidably outside its wall timestamp.
    All other I/O is charged. Native source/physics replay and learned numerical
    verification belong to the future independent whole-execution auditor.
    """
    begin = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    phase, current_step, controller = "validation", None, None
    envs, initialized, closed, decisions, input_rows = [], [], set(), [], []
    decision_times, native_times, input_times, storage_times, native_cases = [], [], [], [], []
    cleanup_errors = []
    active_decision_seconds = active_native_seconds = 0.0
    active_native_cases = 0
    setup = 0.0

    def set_phase(value):
        nonlocal phase
        phase = value
        if progress is not None:
            progress.update(phase=value, step=current_step, completed_model_decisions=len(decisions))

    def close_all():
        for index, env in enumerate(envs):
            if index in closed:
                continue
            closed.add(index)  # A failing close is attempted once, never retried.
            try:
                env.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(error)

    try:
        check_cap(deadline)
        _plan(plan)
        require(plan["dt"] == .02, "Two-observation component supports dt=.02")
        require(panel in PANELS, "Unknown sensing panel")
        require(type(plan["control_episodes"]) is int and plan["control_episodes"] > 0
                and len(cases) == plan["control_episodes"] and all(type(case) is ControlCase for case in cases),
                "Explicit complete control cases")
        require(isinstance(inputs_by_step, (list, tuple)) and len(inputs_by_step) == HORIZON
                and all(isinstance(value, (str, Path)) for value in inputs_by_step),
                "Exactly fifty saved SearchInputs stems required")
        stems = tuple(Path(value) for value in inputs_by_step)
        require(progress is None or isinstance(progress, dict), "Mutable progress dict or None")
        require(all(stem.with_suffix(suffix).is_file() for stem in stems for suffix in (".npz", ".json")),
                "All saved innovation files must exist before native setup")
        identity = model_identity(model, plan)
        for name in ("scoring", "controller-decisions"):
            (out / name).mkdir()
        artifacts.write(out / "started.json", {"version": VERSION, "status": "started", "panel": panel,
            "model": identity, "cases": len(cases), "steps": HORIZON,
            "input_stems": [str(stem) for stem in stems],
            "case_bindings": [{"reset_seed": case.reset_seed, "noise_seed": case.noise_seed,
                               "sensor_schedule": case.sensor_schedule.tolist()} for case in cases],
            "seed_allocation": "none; explicit caller-supplied cases"})
        set_phase("setup")
        packets = []
        for case in cases:
            check_cap(deadline)
            env = ReacherEpisode(noise_std=plan["noise_std"])
            envs.append(env)
            require(env.dt == plan["dt"], "Native/model timestep alignment")
            value = env.reset(case.reset_seed, case.sensor_schedule, noise_seed=case.noise_seed)
            initialized.append(env)
            packet, _ = _public_record(env.episode_record(), step=0, returned_packet=value)
            packets.append(packet)
        packets = np.stack(packets)
        controller = TwoObservationController(plan, model, len(envs))
        issued = None
        setup = time.perf_counter() - begin
        for step in range(HORIZON):
            current_step, active_native_cases = step, 0
            active_decision_seconds = active_native_seconds = 0.0
            set_phase("decision")
            check_cap(deadline)
            tick = time.perf_counter()
            try:
                _public_arrays(packets, issued)
                input_tick = time.perf_counter()
                stem = stems[step]
                hashes = {suffix: artifacts.sha(stem.with_suffix(suffix)) for suffix in (".npz", ".json")}
                inputs = artifacts.load_inputs(stem)
                require(hashes == {suffix: artifacts.sha(stem.with_suffix(suffix)) for suffix in hashes},
                        "Saved innovations changed while loading")
                require(inputs.initial.shape[0] == len(envs), "Innovation case count")
                input_times.append(time.perf_counter() - input_tick)
                input_rows.append({"step": step, "stem": str(stem), "files_sha256": hashes,
                                   "input_identities": dict(inputs.identities())})
                decision = controller.decide(packets.copy(), inputs,
                    issued_action=None if issued is None else issued.copy(), deadline=deadline,
                    failure_out=out / "partial-controller" / f"{step:03d}")
                require(decision.metadata["step"] == step and decision.metadata["model"] == identity,
                        "Controller decision identity/clock")
                _public_arrays(packets, decision.action)
                require(decision.action.shape == (len(envs), 2), "Batch-aligned selected commands")
                require(np.array_equal(decision.action, decision.search_result.selected_actions),
                        "Controller action differs from search selection")
                # Retain compact state/work evidence, not all fifty candidate banks in RAM.
                decisions.append({"root_state": decision.root_state, "carried_state": decision.carried_state,
                    "metadata": copy.deepcopy(decision.metadata),
                    "angles": decision.diagnostics["arrays"]["selected_angles"].copy(),
                    "reward": decision.diagnostics["arrays"]["selected_learned_reward"].copy(),
                    "imagined_transitions": decision.search_result.imagined_transitions})
                storage_tick = time.perf_counter()
                _save_decision(out, step, decision)
                storage_times.append(time.perf_counter() - storage_tick)
            finally:
                active_decision_seconds = time.perf_counter() - tick
            decision_times.append(active_decision_seconds)
            check_cap(deadline)
            set_phase("native_step")
            tick = time.perf_counter()
            try:
                next_packets, actual_commands = [], []
                for env, action in zip(envs, decision.action, strict=True):
                    check_cap(deadline)
                    value = env.step(action.copy())
                    active_native_cases += 1
                    require(env.step_index == step + 1 and env.finished == (step + 1 == HORIZON),
                            "Native episode terminated early or violated its fifty-action clock")
                    packet, actual = _public_record(env.episode_record(), step=step + 1,
                                                    returned_packet=value, expected_command=action)
                    next_packets.append(packet)
                    actual_commands.append(actual)
                packets, issued = np.stack(next_packets), np.stack(actual_commands)
            finally:
                active_native_seconds = time.perf_counter() - tick
            native_times.append(active_native_seconds)
            native_cases.append(active_native_cases)
        set_phase("finalize")
        check_cap(deadline)
        require(all(env.finished and env.step_index == HORIZON for env in envs), "Complete native50 termination")
        require(model_identity(model, plan) == identity, "Inference changed model identity")
        tick = time.perf_counter()
        _save_records(out / "episodes", [env.episode_record() for env in envs])
        states = _states(decisions)
        artifacts.save_npz(out / "states.npz", **states)
        artifacts.save_npz(out / "executed_predictions.npz",
            angles=np.stack([d["angles"] for d in decisions], 1),
            rewards=np.stack([d["reward"] for d in decisions], 1))
        artifacts.write(out / "state-work.json", {"version": VERSION, "model": identity,
            "steps": [d["metadata"] for d in decisions], "state_arrays_bytes": sum(v.nbytes for v in states.values()),
            "aggregate_model_work": work_accounting(model, assimilate_samples=len(envs) * HORIZON,
                advance_samples=len(envs) * HORIZON + sum(d["imagined_transitions"] for d in decisions)),
            "auxiliary_teacher_calls": 0, "reset_calls": 0,
            "neural_verification": "Recorded source-bound outputs, not independent neural replay"})
        artifacts.write(out / "inputs.json", {"version": VERSION, "steps": input_rows,
            "scope": "Exact external input files must accompany any independently auditable release."})
        finalize_seconds = time.perf_counter() - tick
        set_phase("cleanup")
        tick = time.perf_counter()
        close_all()
        cleanup_seconds = time.perf_counter() - tick
        if cleanup_errors:
            raise cleanup_errors[0]
        check_cap(deadline)
        timing = {"version": VERSION, "setup_seconds": setup, "decision_seconds": decision_times,
            "controller_seconds": [d["metadata"]["controller_seconds"] for d in decisions],
            "input_load_hash_seconds": input_times, "decision_trace_write_seconds": storage_times,
            "native_step_and_record_seconds": native_times, "native_step_seconds": native_times,
            "native_cases_per_step": native_cases, "finalize_seconds": finalize_seconds,
            "cleanup_seconds": cleanup_seconds, "row_payload_wall_seconds": time.perf_counter() - begin,
            "observation_assimilations": len(envs) * HORIZON, "executed_action_advances": len(envs) * HORIZON,
            "scope": "Decision includes input loading, controller and trace writes; native includes record copying/validation; sub-times overlap, do not add them again."}
        artifacts.write(out / "timings.json", timing)
        set_phase("terminal_manifest")
        files = {str(path.relative_to(out)): artifacts.sha(path) for path in sorted(out.rglob("*")) if path.is_file()}
        check_cap(deadline)
        completed = {"version": VERSION, "status": "completed", "panel": panel, "cases": len(envs),
            "model_decisions": HORIZON, "native_steps_per_case": [HORIZON] * len(envs),
            "public_packets_per_case": HORIZON + 1, "row_wall_seconds": time.perf_counter() - begin,
            "wall_excludes": "Only the final completed.json receipt write; all payload, cleanup and hashing included.",
            "files": files}
        artifacts.write(out / "completed.json", completed)
        if progress is not None:
            progress.update(phase="completed", step=HORIZON, completed_model_decisions=HORIZON)
        return {**timing, "row_wall_seconds": completed["row_wall_seconds"]}
    except BaseException as error:
        failure_phase = phase
        close_tick = time.perf_counter()
        close_all()
        failure_cleanup_seconds = time.perf_counter() - close_tick
        for close_error in cleanup_errors:
            if close_error is not error:
                error.add_note(f"Environment cleanup failed: {type(close_error).__name__}: {close_error}")
        # Independent best-effort saves preserve other evidence even if one write fails.
        saves = []
        if initialized:
            saves.append(lambda: _save_partial(out / "partial-episodes", initialized))
        if decisions:
            saves.append(lambda: artifacts.save_npz(out / "partial-states.npz", **_states(decisions)))
        if controller is not None:
            def save_controller():
                snapshot = controller.snapshot()
                arrays = snapshot.pop("state")
                selected = snapshot.pop("previous_selected_action")
                if selected is not None:
                    arrays["last_selected_action"] = selected
                artifacts.save_npz(out / "partial-controller-state.npz", **arrays)
                artifacts.write(out / "partial-controller-state.json", snapshot)
            saves.append(save_controller)
        saves.append(lambda: artifacts.write(out / "partial-inputs.json", {"steps": input_rows}))
        for save in saves:
            try:
                save()
            except BaseException as preservation_error:  # noqa: BLE001
                error.add_note(f"Partial artifact preservation failed: {type(preservation_error).__name__}: {preservation_error}")
        failure = {"version": VERSION, "status": "failed", "phase": failure_phase, "step": current_step,
            "exception_type": type(error).__name__, "message": str(error), "notes": list(getattr(error, "__notes__", [])),
            "initialized_cases": len(initialized), "completed_steps_by_case": [env.step_index for env in initialized],
            "completed_model_decisions": len(decisions), "completed_decision_seconds": decision_times,
            "completed_native_step_seconds": native_times, "active_decision_seconds": active_decision_seconds,
            "active_native_seconds": active_native_seconds, "active_native_cases": active_native_cases,
            "setup_seconds": setup, "failure_cleanup_seconds": failure_cleanup_seconds,
            "completed_state_work": [copy.deepcopy(d["metadata"]) for d in decisions],
            "controller_failure": None if controller is None else copy.deepcopy(getattr(controller, "_failed", None)),
            "row_wall_seconds": time.perf_counter() - begin,
            "warning": "Saved model decisions can include commands not executed by all cases. This attempt is not a complete row."}
        try:
            artifacts.write(out / "failed.json", failure)
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Failure receipt write failed: {type(preservation_error).__name__}: {preservation_error}")
        raise
