"""Retrospective paired decision decomposition, using only pinned saved outputs.

This script never imports a model, optimizer or simulator. It separates phase
contributions under the original full-episode denominator from the differently
weighted primary metric. No new scientific pass/fail gate is constructed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-prequery-decision-diagnosis-v1"
SEEDS = (275000001, 275000002, 275000003)
ARCHITECTURES = ("innovation", "innovation_gru")
METRICS = ("agreement", "gap")
PHASES = ("initial", "post")
CELLS = ("CC", "CW", "WC", "WW")


def require(value, message):
    if not value:
        raise ValueError(message)


def decisions(q, p, legal):
    """Independent implementation of the held strict float32 1e-10 tie rule."""
    require(isinstance(q, np.ndarray) and q.ndim == 2 and q.shape[1:] == (4,)
            and q.dtype == np.float32 and isinstance(p, np.ndarray)
            and p.dtype == np.float32 and p.shape == q.shape
            and isinstance(legal, np.ndarray) and legal.dtype == np.bool_
            and legal.shape == q.shape, "float32 score and Boolean legal geometry")
    require(np.isfinite(q).all() and np.isfinite(p).all() and legal.any(axis=1).all(),
            "finite scores and at least one legal action")
    qmin = np.where(legal, q, np.float32(np.inf)).min(axis=1)
    pmin = np.where(legal, p, np.float32(np.inf)).min(axis=1)
    with np.errstate(over="ignore"):
        pn = legal & ((p - pmin[:, None]) < np.float32(1e-10))
        qn = legal & ((q - qmin[:, None]) < np.float32(1e-10))
    require(pn.any(axis=1).all() and qn.any(axis=1).all(), "nonempty near-minimum sets")
    action = pn.argmax(axis=1).astype(np.int64)
    rows = np.arange(len(q))
    return {"action": action, "correct": qn[rows, action],
            "gap": q[rows, action].astype(np.float64) - qmin.astype(np.float64)}


def decompose(base, candidate, steps, episode_index, identities):
    """Per-episode additive contributions, without dropping empty episodes.

    Transition gaps and masses divide by full episode nonquery count. They are
    contributions, not conditional cell means. Group averages happen later.
    """
    require(isinstance(identities, list) and bool(identities), "declared episodes")
    ids = [x["episode_id"] for x in identities]
    require(len(ids) == len(set(ids)), "unique episode identities")
    require(isinstance(steps, np.ndarray) and steps.dtype == np.int64 and steps.ndim == 1
            and isinstance(episode_index, np.ndarray) and episode_index.dtype == np.int64
            and episode_index.shape == steps.shape, "integer row identity vectors")
    require(((steps > 0) & (steps % 4 != 0)).all()
            and ((episode_index >= 0) & (episode_index < len(ids))).all()
            and len(set(zip(episode_index.tolist(), steps.tolist(), strict=True))) == len(steps),
            "unique positive nonquery episode-step rows")
    for value in (base, candidate):
        require(set(value) == {"action", "correct", "gap"}, "complete decision fields")
        for name, dtype in (("action", np.int64), ("correct", np.bool_), ("gap", np.float64)):
            v = value[name]
            require(isinstance(v, np.ndarray) and v.shape == steps.shape and v.dtype == dtype,
                    "decision field geometry")
        require(((value["action"] >= 0) & (value["action"] < 4)).all()
                and np.isfinite(value["gap"]).all() and (value["gap"] >= 0).all(),
                "valid decisions and finite nonnegative gaps")
    episodes, transitions = [], []
    for i, identity in enumerate(identities):
        whole = episode_index == i
        n = int(whole.sum())
        initial, post = whole & (steps <= 3), whole & (steps >= 5)
        record = {**identity, "rows": n, "initial_rows": int(initial.sum()), "post_rows": int(post.sum())}
        for metric, field in (("agreement", "correct"), ("gap", "gap")):
            b, c = base[field].astype(np.float64), candidate[field].astype(np.float64)
            for phase, mask in (("full", whole), ("primary", post)):
                count = int(mask.sum())
                for name, values in (("baseline", b), ("candidate", c)):
                    record[f"{name}_{phase}_{metric}"] = math.fsum(values[mask]) / count if count else 0.
            for phase, mask in (("initial", initial), ("post", post)):
                record[f"{phase}_{metric}_contribution"] = math.fsum((c - b)[mask]) / n if n else 0.
            record[f"reweight_{metric}"] = record[f"post_{metric}_contribution"] - (
                record[f"candidate_primary_{metric}"] - record[f"baseline_primary_{metric}"])
        for phase, selected in (("initial", initial), ("post", post)):
            for cell in CELLS:
                mask = selected & (base["correct"] == (cell[0] == "C")) & (
                    candidate["correct"] == (cell[1] == "C"))
                bg = math.fsum(base["gap"][mask]) / n if n else 0.
                cg = math.fsum(candidate["gap"][mask]) / n if n else 0.
                transitions.append({"episode_id": identity["episode_id"], "phase": phase, "cell": cell,
                    "count": int(mask.sum()), "mass": int(mask.sum()) / n if n else 0.,
                    "baseline_gap": bg, "candidate_gap": cg, "gap_delta": cg - bg})
        episodes.append(record)
    return {"episodes": episodes, "transitions": transitions}


def pin(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def close_enough(left, right, label):
    residual = abs(left - right)
    require(residual <= 1e-12 * max(1., abs(left), abs(right)), label)
    return residual


def groups(identities):
    yield "overall", list(range(len(identities)))
    for regime in ("lambda3", "lambda4"):
        yield regime, [i for i, x in enumerate(identities) if x["regime"] == regime]
        for arm in ("analytic", "neural", "period4_hold"):
            yield f"{regime}/{arm}", [i for i, x in enumerate(identities)
                                      if x["regime"] == regime and x["arm"] == arm]
        for case in range(6):
            yield f"{regime}/case{case}", [i for i, x in enumerate(identities)
                                          if x["regime"] == regime and x["case"] == case]


def original_join(worker, terminal, *, script, output, plan, plan_pin):
    """Authenticate an original worker to its exact successful supervising parent."""
    require(terminal["status"] == "completed" and terminal["returncode"] == 0
            and terminal["group_absent"] is True and terminal["cleanup"]["errors"] == []
            and terminal["cleanup"]["reaped"] is True and terminal["timed_out"] is False
            and terminal["error"] is terminal["clock_error"] is None,
            "successful original parent")
    command = terminal["command"]
    prefix = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts" / script)]
    if script.startswith("train_"):
        prefix.append("run")
    require(command[:len(prefix)] == prefix, "original worker command")
    tail = command[len(prefix):]
    require(len(tail) % 2 == 0 and len(set(tail[::2])) == len(tail) // 2,
            "unique original command arguments")
    options = dict(zip(tail[::2], tail[1::2], strict=True))
    require(options["--plan"] == str(plan) and options["--plan-sha256"] == plan_pin
            == worker["plan_sha256"] and options["--output"] == str(output), "original plan/output binding")
    launch_path = Path(options["--supervision"])
    require(launch_path.is_absolute() and launch_path.is_relative_to(ROOT)
            and pin(launch_path)["sha256"] == worker["supervision_sha256"], "original launch hash")
    launch = json.loads(launch_path.read_text())
    require(all(terminal[k] == v for k, v in launch.items()), "original launch retained in terminal")
    require(terminal["started_ns"] <= worker["started_ns"] < worker["finished_ns"]
            <= terminal["finished_ns"] <= terminal["deadline_ns"], "original worker interval")
    return options


def execute(args):
    started = time.monotonic_ns()
    require(args.plan.is_absolute() and args.output.is_absolute() and args.supervision.is_absolute(),
            "absolute plan/output/supervision")
    require(pin(args.plan)["sha256"] == args.plan_sha256, "external plan pin")
    plan = json.loads(args.plan.read_text())
    require(plan["version"] == VERSION and plan["status"] == "frozen_before_saved_array_read"
            and plan["output"] == str(args.output), "exact diagnosis plan")
    require(args.output.is_relative_to(ROOT / "output" / VERSION), "separate diagnostic output")
    args.output.mkdir(exist_ok=False)
    launch = json.loads(args.supervision.read_text())
    receipt = {"version": VERSION, "status": "started", "plan_sha256": args.plan_sha256,
               "supervision_sha256": pin(args.supervision)["sha256"], "model_calls": 0,
               "optimizer_calls": 0, "native_calls": 0, "npz_decodes": 0, "started_ns": started,
               "clock": "time.monotonic_ns; original supervisor independently enforces suspend-aware deadline"}
    def interrupted(_signum, _frame):
        raise InterruptedError("diagnostic interrupted by original supervisor")

    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        require(launch["cap_seconds"] == 120 and launch["pid"] == os.getpid()
                and launch["pgid"] == os.getpgid(0) and launch["parent_pid"] == os.getppid()
                and launch["cwd"] == str(ROOT)
                and launch["deadline_ns"] - launch["started_ns"] == 120 * 10**9
                and launch["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]["sha256"]
                and launch["clock_source_sha256"] == plan["sources"]["src/openjev/research/suspend_clock.py"]["sha256"]
                and launch["command"] == [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
                                          *sys.argv[1:]], "exact original diagnostic supervisor")
        for name, expected in plan["sources"].items():
            require(pin(ROOT / name) == expected, "frozen diagnostic source " + name)
        inputs = {}
        for role, item in plan["inputs"].items():
            path = Path(item["path"])
            require(path.is_absolute() and path.is_relative_to(ROOT) and not path.is_symlink()
                    and pin(path) == {k: item[k] for k in ("sha256", "bytes")}, "pinned input " + role)
            inputs[role] = path
        training = json.loads(inputs["training_receipt"].read_text())
        audit = json.loads(inputs["audit_receipt"].read_text())
        require(training["status"] == audit["status"] == "completed"
                and training["fits_completed"] == 12 and audit["agreement"] is True
                and audit["counts"]["required_passed"] == 51, "closed original failed study")
        require(training["inputs"]["collection_plan"] == plan["inputs"]["collection_plan"],
                "training selects original collection plan")
        for audit_role, input_role in (("plan", "training_plan"), ("worker", "training_receipt"),
                                       ("terminal", "training_terminal")):
            require(audit["producer_inputs"][audit_role] == plan["inputs"][input_role], "audit producer join")
        for phase, worker in (("training", training), ("audit", audit)):
            script = "train_otto_prequery_calibration.py" if phase == "training" else "audit_otto_prequery_calibration.py"
            options = original_join(worker, json.loads(inputs[phase + "_terminal"].read_text()),
                script=script, output=inputs[phase + "_receipt"].parent, plan=inputs["training_plan"],
                plan_pin=plan["inputs"]["training_plan"]["sha256"])
            if phase == "audit":
                for name, role in (("worker", "training_receipt"), ("terminal", "training_terminal")):
                    require(options["--" + name] == str(inputs[role])
                            and options["--" + name + "-sha256"] == plan["inputs"][role]["sha256"],
                            "original audit selected training input")
        for role in ("windows", "window_metadata", "summary", *(f"{a}_{o}-{s}" for a in ARCHITECTURES
                                                              for o in ("mse", "aux") for s in SEEDS)):
            path = inputs[role]
            require(pin(path) == training["files"][path.name], "original training payload " + role)
        metadata = json.loads(inputs["window_metadata"].read_text())
        cohort = json.loads(inputs["collection_plan"].read_text())["cohort"]
        identities = [{k: x[k] for k in ("episode_id", "regime", "case", "arm")}
                      for x in cohort if x["stage"] == "valid"]
        require(len(identities) == 36 and [x["episode_id"] for x in identities] == metadata["episode_ids"],
                "exact ordered originating identities")
        with np.load(inputs["windows"], allow_pickle=False) as archive:
            windows = {name: archive[name] for name in archive.files}
        receipt["npz_decodes"] += 1
        counts = metadata["counts"]
        require(counts["windows"] == len(windows["lengths"]) == 4609, "complete census geometry")
        row, age = np.nonzero(windows["nonquery_mask"])
        require(len(row) == 13788, "all nonquery census rows")
        steps = windows["step_offsets"][row] + age.astype(np.int64)
        indices = windows["episode_index"][row]
        expected = [(i, t) for i, length in enumerate(windows["episode_lengths"])
                    for t in range(int(length)) if t % 4]
        require(list(zip(indices.tolist(), steps.tolist(), strict=True)) == expected,
                "complete disjoint nonquery rows including final tails")
        q, legal = windows["targets"][row, age], windows["legal"][row, age]
        original = json.loads(inputs["summary"].read_text())
        prior_models = {(m["family"], m["seed"]): m for m in original["models"]}
        paired, all_episodes, all_transitions, residuals, transition_groups = [], [], [], [], []
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                values = []
                for objective in ("mse", "aux"):
                    name = f"{architecture}_{objective}"
                    with np.load(inputs[f"{name}-{seed}"], allow_pickle=False) as archive:
                        require(set(archive.files) == {"predictions", "prior", "prior_mask"}, "prediction payload schema")
                        p = archive["predictions"]
                    receipt["npz_decodes"] += 1
                    require(p.shape == windows["targets"].shape and p.dtype == np.float32
                            and np.isfinite(p).all(), "complete finite window predictions")
                    values.append(decisions(q, p[row, age], legal))
                result = decompose(*values, steps, indices, identities)
                for item in result["episodes"]:
                    all_episodes.append({"architecture": architecture, "seed": seed, **item})
                    transitions = [t for t in result["transitions"] if t["episode_id"] == item["episode_id"]]
                    require(len(transitions) == 8 and sum(t["count"] for t in transitions) == item["rows"],
                            "exhaustive transition cells")
                    for metric in METRICS:
                        d = item[f"candidate_full_{metric}"] - item[f"baseline_full_{metric}"]
                        residuals.append(close_enough(d, item[f"initial_{metric}_contribution"] +
                                                     item[f"post_{metric}_contribution"], "episode additive identity"))
                    for phase in PHASES:
                        cells = {t["cell"]: t for t in transitions if t["phase"] == phase}
                        residuals.append(close_enough(cells["WC"]["mass"] - cells["CW"]["mass"],
                            item[f"{phase}_agreement_contribution"], "agreement transition identity"))
                        residuals.append(close_enough(math.fsum(t["gap_delta"] for t in cells.values()),
                            item[f"{phase}_gap_contribution"], "gap transition identity"))
                all_transitions.extend({"architecture": architecture, "seed": seed, **x}
                                       for x in result["transitions"])
                for group, selected in groups(identities):
                    records = [result["episodes"][i] for i in selected]
                    require(bool(records), "nonempty declared group")
                    fields = [k for k in records[0] if k not in identities[0]
                              and k not in ("rows", "initial_rows", "post_rows")]
                    aggregated = {k: math.fsum(x[k] for x in records) / len(records) for k in fields}
                    paired.append({"architecture": architecture, "seed": seed, "group": group,
                        "episodes": len(records), "supported_full": sum(x["rows"] > 0 for x in records),
                        "supported_initial": sum(x["initial_rows"] > 0 for x in records),
                        "supported_primary": sum(x["post_rows"] > 0 for x in records),
                        "rows": sum(x["rows"] for x in records),
                        "initial_rows": sum(x["initial_rows"] for x in records),
                        "post_rows": sum(x["post_rows"] for x in records),
                        "initial_contribution_mass": math.fsum(x["initial_rows"] / x["rows"]
                            if x["rows"] else 0. for x in records) / len(records),
                        "post_contribution_mass": math.fsum(x["post_rows"] / x["rows"]
                            if x["rows"] else 0. for x in records) / len(records), **aggregated})
                    selected_ids = {x["episode_id"] for x in records}
                    for phase in PHASES:
                        for cell in CELLS:
                            ts = [x for x in result["transitions"] if x["episode_id"] in selected_ids
                                  and x["phase"] == phase and x["cell"] == cell]
                            require(len(ts) == len(records), "every declared episode in grouped transition")
                            transition_groups.append({"architecture": architecture, "seed": seed, "group": group,
                                "phase": phase, "cell": cell, "episodes": len(records),
                                "count": sum(x["count"] for x in ts), **{k: math.fsum(x[k] for x in ts) / len(ts)
                                    for k in ("mass", "baseline_gap", "candidate_gap", "gap_delta")}})
                    if group not in ("overall", "lambda3", "lambda4"):
                        continue
                    for label, objective in (("baseline", "mse"), ("candidate", "aux")):
                        model = prior_models[(f"{architecture}_{objective}", seed)]["metrics"]
                        for phase, source in (("full", "full"), ("primary", "postcorrection")):
                            target = model[source]["overall"] if group == "overall" else model[source]["by_regime"][group]
                            for metric, name in (("agreement", "episode_weighted_agreement"),
                                                 ("gap", "episode_weighted_raw_gap")):
                                residuals.append(close_enough(aggregated[f"{label}_{phase}_{metric}"], target[name],
                                                             "published metric reconciliation"))
        means, transition_means = [], []
        for architecture in ARCHITECTURES:
            for group, _ in groups(identities):
                rows = [x for x in paired if x["architecture"] == architecture and x["group"] == group]
                require(len(rows) == 3, "all paired fit seeds")
                means.append({"architecture": architecture, "group": group, "seeds": list(SEEDS),
                    **{k: rows[0][k] for k in ("episodes", "supported_full", "supported_initial", "supported_primary",
                        "rows", "initial_rows", "post_rows", "initial_contribution_mass", "post_contribution_mass")},
                    **{k: math.fsum(x[k] for x in rows) / 3 for k in fields}})
                for phase in PHASES:
                    for cell in CELLS:
                        ts = [x for x in transition_groups if x["architecture"] == architecture
                              and x["group"] == group and x["phase"] == phase and x["cell"] == cell]
                        require(len(ts) == 3, "all seeds in mean transition")
                        transition_means.append({"architecture": architecture, "group": group,
                            "seeds": list(SEEDS), "phase": phase, "cell": cell,
                            "count_mean": math.fsum(x["count"] for x in ts) / 3,
                            **{k: math.fsum(x[k] for x in ts) / 3
                               for k in ("mass", "baseline_gap", "candidate_gap", "gap_delta")}})
        summary = {"version": VERSION, "scope": "retrospective saved-output diagnosis; original FAIL51/55 unchanged",
            "paired": paired, "means": means, "transition_groups": transition_groups,
            "transition_means": transition_means, "published_and_additive_checks": len(residuals),
            "max_absolute_reconciliation_residual": max(residuals),
            "row_exposures": len(row) * 6, "distinct_rows": len(row), "initial_rows": int((steps <= 3).sum()),
            "post_rows": int((steps >= 5).sum()), "episode_pairs": len(all_episodes),
            "transition_cells": len(all_transitions), "array_decodes": receipt["npz_decodes"]}
        write(args.output / "summary.json", summary)
        for name, records in (("episodes.csv", all_episodes), ("transitions.csv", all_transitions)):
            with (args.output / name).open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
        for role, path in inputs.items():
            require(pin(path) == {k: plan["inputs"][role][k] for k in ("bytes", "sha256")}, "unchanged input " + role)
        for name, expected in plan["sources"].items():
            require(pin(ROOT / name) == expected, "unchanged diagnostic source " + name)
        require(pin(args.plan)["sha256"] == args.plan_sha256
                and pin(args.supervision)["sha256"] == receipt["supervision_sha256"], "unchanged plan and launch")
        receipt.update(status="completed", files={p.name: pin(p) for p in args.output.iterdir()},
            checks=len(residuals), max_residual=max(residuals), finished_ns=time.monotonic_ns(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
        receipt["wall_seconds"] = (receipt["finished_ns"] - started) / 1e9
        require(receipt["wall_seconds"] < 120 and receipt["peak_rss_bytes"] < 1024**3
                and sum(v["bytes"] for v in receipt["files"].values()) < 16 * 1024**2, "bounded diagnostic resources")
        write(args.output / "receipt.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("status", "npz_decodes", "checks", "max_residual", "wall_seconds")}))
    except BaseException as error:
        signal.signal(signal.SIGTERM, old_handler)
        receipt.update(status="failed", error=repr(error), finished_ns=time.monotonic_ns())
        try:
            if (args.output / "receipt.json").exists():
                (args.output / "receipt.json").rename(args.output / "receipt.invalid.json")
            write(args.output / "failure.json", receipt)
        except BaseException as secondary:  # noqa: BLE001 - preserve original interruption or failure
            error.add_note("Failure record could not be published: " + repr(secondary))
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
