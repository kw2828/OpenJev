"""Prepare and package a closed separate-prior release using opaque saved bytes.

Neither mode imports a scientific module, opens an array, or runs a model.
Prepare requires externally pinned successful phase records and finished
publication assets. Archive reuses the unchanged qualified tar/gzip engine.
The inventory and packaging outputs live outside the scientific evidence tree.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/package_otto_separate_prior.py"
ENGINE = "scripts/package_otto_score_forecasts.py"
ENGINE_PIN = "c5a7099bf42c4ddb11c7d25b555a8c4ff237ff60721a028f7bc0758449916429"
PLOTTER = "scripts/plot_otto_separate_prior.py"
PLOT_HELPER = "scripts/plot_otto_prequery_calibration.py"
PLOT_HELPER_PIN = "69c3542f693e8bdceb474da0beb8880b81a8614360107119fcc97173f9653403"
STUDY = ROOT / "output/otto-separate-prior-v1"
PACKAGE = ROOT / "output/otto-separate-prior-package-v1"
VERSION = "otto-separate-prior-package-v1"
LIMITS = {"seconds": 300, "rss_bytes": 1024**3, "output_bytes": 6 * 1024**3}
NAME = "openjev-otto-separate-prior-v1.tar.gz"
ROLES = {"collection_plan", "collection_receipt", "collection_terminal", "training_plan",
         "training_receipt", "training_terminal", "audit_receipt", "audit_terminal", "figure_receipt", "precollection_freeze"}
REQUIRED_PUBLICATIONS = {"research/otto-separate-prior-results.md",
                         "research/otto-separate-prior-engineering.md",
                         "research/otto-separate-prior-protocol.md",
                         "docs/assets/otto-separate-prior-forecast.png",
                         "docs/assets/otto-separate-prior-forecast.svg",
                         "docs/assets/otto-separate-prior-costs.png",
                         "docs/assets/otto-separate-prior-costs.svg"}
LICENSES = {"LICENSE", "third_party/otto/LICENSE", "third_party/otto/LICENSE-zoo",
            "tmp/otto-source-review-01/LICENSE"}
CELLS = ("innovation_shared_mse", "innovation_shared_aux", "innovation_separate_mse", "innovation_separate_aux",
         "gru_shared_mse", "gru_shared_aux", "gru_separate_mse", "gru_separate_aux")
SEEDS = (285000001, 285000002, 285000003)
COLLECTION_PAYLOADS = {f"{name}.jsonl.gz" for name in
    ("work", "weights", "forwards", "transitions", "samples", "annotations")} | {
    "started.json", "runtime.json", "setup.json", "deployment.json", "cohort.json",
    "episode-boundaries.jsonl", "episodes.jsonl", "train.npz", "valid.npz", "costs.json",
    "summary.json", "train-selection.json"}
TRAINING_PAYLOADS = {"started.json", "runtime.json", "progress.jsonl", "work.jsonl", "fits.json", "summary.json",
    "training-history.npz", "training-history.json", "validation-history.npz", "validation-history.json",
    "validation-windows.npz", "validation-windows.json", "prediction-hold.npz"} | {
    f"{prefix}{cell}-{seed}.npz" for cell in CELLS for seed in SEEDS
    for prefix in ("", "prediction-", "training-prediction-")}
FIGURE_PAYLOADS = {f"otto-separate-prior-{kind}.{suffix}" for kind in ("forecast", "costs")
                  for suffix in ("png", "svg")} | {
    "individual-points.csv", "family-means.csv", "conditions.csv",
    "gates.csv", "costs.csv", "plotted-values.json"}
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
GATE_SIZES = {"innovation_mechanism": 19, "gru_mechanism": 19, "architecture": 29}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts and path.is_file()
        and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular evidence")
    return path


def relative(path):
    return str(regular(path).relative_to(ROOT))


def digest(path, check=lambda: None):
    path = regular(path)
    value, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            check(); value.update(block); size += len(block)
    return {"sha256": value.hexdigest(), "bytes": size}


def read(path):
    path = regular(path)
    require(path.stat().st_size <= 32 * 1024**2, "bounded JSON metadata")
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def output_directory(path):
    require(path.is_absolute() and path.is_relative_to(PACKAGE) and path != PACKAGE
        and ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "exclusive packaging output outside scientific evidence")
    PACKAGE.mkdir(exist_ok=True)
    path.mkdir(exist_ok=False)


def parent_join(worker, terminal, command_prefix, plan, plan_pin, output, add, cap):
    require(worker["status"] == "completed" and terminal["status"] == "completed"
        and terminal["returncode"] == 0 and terminal["timed_out"] is False
        and terminal["group_absent"] is True and terminal["cleanup"]["reaped"] is True
        and terminal["cleanup"]["errors"] == [] and terminal["error"] is terminal["clock_error"] is None
        and terminal["started_ns"] <= worker["started_ns"] < worker["finished_ns"]
        <= terminal["finished_ns"] <= terminal["deadline_ns"], "successful original phase and parent")
    require(terminal["cap_seconds"] == cap and terminal["cwd"] == str(ROOT)
        and terminal["deadline_ns"] == terminal["started_ns"] + cap * 10**9
        and terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
        and terminal["wall_seconds"] == terminal["elapsed_ns"] / 10**9
        and type(terminal["pid"]) is int and terminal["pid"] > 1
        and terminal["pid"] == terminal["pgid"] and terminal["pid"] != terminal["parent_pid"]
        and terminal["clock_source_sha256"] == CLOCK_PIN and terminal["watchdog_sha256"] == SUPERVISOR_PIN,
        "original exact process allocation, clock and cleanup timing")
    command = list(terminal["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:len(command_prefix)] == command_prefix, "original phase command prefix")
    tail = command[len(command_prefix):]
    require(len(tail) % 2 == 0 and len(set(tail[::2])) == len(tail) // 2, "unique original phase options")
    options = dict(zip(tail[::2], tail[1::2], strict=True))
    require(options["--plan"] == str(plan) and options["--plan-sha256"] == plan_pin
        == worker["plan_sha256"] and options["--output"] == str(output), "original plan and output joins")
    launch_path = regular(options["--supervision"])
    add(launch_path, {"sha256": worker["supervision_sha256"], "bytes": launch_path.stat().st_size})
    launch = read(launch_path)
    require(all(terminal[k] == v for k, v in launch.items()), "terminal retains original launch")
    require(read(output / "started.json")["launch"] == launch
        and worker["wall_seconds"] == (worker["finished_ns"] - worker["started_ns"]) / 10**9,
        "worker retained original launch and elapsed scope")
    return options


def prepare(args):
    start = time.monotonic_ns()
    output_directory(args.output)
    receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
               "array_decodes": 0, "model_calls": 0, "native_calls": 0, "optimizer_calls": 0}

    def check():
        require(time.monotonic_ns() - start < LIMITS["seconds"] * 10**9, "inventory deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "inventory RSS cap")
        require(sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
                < LIMITS["output_bytes"] - 1024**2, "inventory output cap with failure reserve")

    def interrupt(_signal, _frame):
        raise InterruptedError("inventory hard time cap")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["seconds"])
        require(digest(args.bindings, check)["sha256"] == args.bindings_sha256, "external completion bindings pin")
        binding = read(args.bindings)
        require(set(binding) == {"version", "roles", "publications", "inherited_snapshots"}
            and binding["version"] == VERSION and set(binding["roles"]) == ROLES,
            "exact completion bindings schema")
        members, completion_checks, dependencies, source_bindings, historical_sources = {}, [], {}, {}, {}
        opaque_json = []
        snapshots = binding["inherited_snapshots"]
        require(isinstance(snapshots, dict), "explicit historical publication snapshot mapping")

        def add(path, expected=None):
            name = relative(path); actual = digest(path, check)
            require(expected is None or actual == {k: expected[k] for k in ("sha256", "bytes")}, "immutable member " + name)
            require(name not in members or members[name] == actual, "one member identity")
            members[name] = actual
            return name

        def pinned_path(name, expected):
            """Never substitute current publication text for an old pinned copy."""
            original = Path(name)
            original = original if original.is_absolute() else ROOT / original
            require(original.is_relative_to(ROOT), "contained original dependency")
            key = str(original.relative_to(ROOT))
            path = snapshots.get(key + "@" + expected["sha256"], snapshots.get(key, key))
            actual = digest(path, check)
            require(actual == {k: expected[k] for k in ("sha256", "bytes")}, "original bytes or explicit exact snapshot " + key)
            if path != key:
                add(path, expected)
            return regular(path)

        for path in sorted(STUDY.rglob("*")):
            require(not path.is_symlink(), "no linked study artifact")
            if path.is_file():
                add(path)
        require(bool(members), "completed scientific evidence tree exists")
        initial_study = {name: d for name, d in members.items() if (ROOT / name).is_relative_to(STUDY)}
        add(SELF); add(ENGINE, {"sha256": ENGINE_PIN, "bytes": regular(ENGINE).stat().st_size})
        add(args.bindings)
        for name in LICENSES:
            add(name)
        roles, records = {}, {}
        for role, item in binding["roles"].items():
            path = regular(item["path"])
            require(path.is_relative_to(STUDY), "current phase evidence belongs to study tree")
            add(path, item); roles[role] = path; records[role] = read(path)

        freeze = records["precollection_freeze"]
        require(freeze["status"] == "frozen_before_empirical_collection" and freeze["capacity_admitted"] is True
            and freeze["no_retry_or_replacement_seeds"] is True and freeze["scientific_runs_started"] is False
            and freeze["planned_fit_count"] == 24 and freeze["planned_optimizer_updates"] == 17280
            and freeze["distinct_conditions"] == 39 and freeze["gate_counts"] == GATE_SIZES
            and len(freeze["sources"]) == 140, "complete prospective 140-source freeze")
        for role in ("collection_plan",):
            item = freeze["inputs"][role]
            require(regular(item["path"]) == roles[role]
                and {k: item[k] for k in ("sha256", "bytes")} == members[relative(roles[role])],
                "precollection manifest binds selected " + role)

        def closed_record(record, path, expected):
            require(set(record["files"]) == expected, "complete declared payload set")
            require({p.name for p in path.parent.iterdir()} == set(record["files"]) | {"receipt.json"}, "closed selected receipt directory")
            for name, item in record["files"].items():
                require(Path(name).name == name, "flat selected phase payload")
                add(path.parent / name, item)

        for phase in ("collection", "training"):
            plan, worker = records[phase + "_plan"], records[phase + "_receipt"]
            require(plan["version"] == worker["version"] == "otto-separate-prior-" + phase + "-v1"
                and worker["complete"] is True and worker["sources"] == plan["sources"]
                and worker["inputs"] == plan["inputs"] and not worker.get("pending")
                and worker.get("pending_emission") is None and worker.get("pending_episode") is None
                and worker.get("pending_action") is None and not worker.get("cleanup_errors"), "complete phase closure")
            closed_record(worker, roles[phase + "_receipt"],
                          COLLECTION_PAYLOADS if phase == "collection" else TRAINING_PAYLOADS)
            executable = ".venv-otto-released-native/bin/python" if phase == "collection" else ".venv/bin/python"
            script = "collect_otto_separate_prior.py" if phase == "collection" else "train_otto_separate_prior.py"
            options = parent_join(worker, records[phase + "_terminal"], [str(ROOT / executable), str(ROOT / "scripts" / script), "run"],
                roles[phase + "_plan"], binding["roles"][phase + "_plan"]["sha256"], roles[phase + "_receipt"].parent, add,
                7200 if phase == "collection" else 21600)
            require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"}, "exact phase arguments")
        require(records["collection_receipt"]["completed_episodes"] == 90
            and records["collection_receipt"]["train_episodes"] == 54
            and records["collection_receipt"]["valid_episodes"] == 36
            and records["collection_receipt"]["training_updates"] == 0
            and records["collection_receipt"]["native_inputs"] == records["collection_plan"]["native_inputs"]
            and records["training_receipt"]["fits_completed"] == 24
            and records["training_receipt"]["optimizer_steps"] == 17280, "complete declared workload")
        training_inputs = records["training_plan"]["inputs"]
        for role in ("collection_plan", "collection_receipt", "collection_terminal"):
            require(training_inputs[role] == {"path": str(roles[role]), **members[relative(roles[role])]},
                    "training binds selected collection " + role)
        capacity_paths = {}
        capacity_records = {}
        for role in ("plan", "receipt", "terminal"):
            item = training_inputs["capacity_" + role]
            path = regular(item["path"])
            require(path.is_relative_to(STUDY), "current capacity evidence in study tree")
            add(path, item); capacity_paths[role] = path; capacity_records[role] = read(path)
            frozen = freeze["inputs"]["capacity_" + role]
            require(regular(frozen["path"]) == path
                    and {k:frozen[k] for k in ("sha256","bytes")} == members[relative(path)],
                    "precollection capacity identity " + role)
        capplan, capworker = capacity_records["plan"], capacity_records["receipt"]
        require(capplan["version"] == capworker["version"] == "otto-separate-prior-capacity-v1"
            and capplan["status"] == "frozen_before_synthetic_work" and capworker["complete"] is True
            and capworker["admitted"] is True and capworker["completed_families"] == list(CELLS)
            and capworker["optimizer_updates"] == 8 and capworker["sources"] == capplan["sources"]
            and capworker["pending"] is capworker["pending_emission"] is None,
            "complete admitted synthetic capacity")
        closed_record(capworker, capacity_paths["receipt"],
                      {"started.json", "runtime.json", "synthetic.json", "work.jsonl", "summary.json"})
        options = parent_join(capworker, capacity_records["terminal"],
            [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/qualify_otto_separate_prior_capacity.py"), "run"],
            capacity_paths["plan"], training_inputs["capacity_plan"]["sha256"], capacity_paths["receipt"].parent, add, 180)
        require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"}, "exact capacity arguments")
        audit = records["audit_receipt"]
        require(audit["agreement"] is True and audit["version"] == "otto-separate-prior-saved-audit-v1", "independent audit agreement")
        closed_record(audit, roles["audit_receipt"], {"started.json", "audit.json"})
        for role, source in (("plan", "training_plan"), ("worker", "training_receipt"), ("terminal", "training_terminal")):
            require(audit["producer_inputs"][role] == {"path": str(roles[source]),
                    **members[relative(roles[source])]}, "audit authenticates selected original training")
        options = parent_join(audit, records["audit_terminal"], [str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/audit_otto_separate_prior.py")], roles["training_plan"],
            binding["roles"]["training_plan"]["sha256"], roles["audit_receipt"].parent, add, 240)
        require(set(options) == {"--plan", "--plan-sha256", "--worker", "--worker-sha256", "--terminal",
            "--terminal-sha256", "--supervision", "--output"}
            and options["--worker"] == str(roles["training_receipt"])
            and options["--worker-sha256"] == binding["roles"]["training_receipt"]["sha256"]
            and options["--terminal"] == str(roles["training_terminal"])
            and options["--terminal-sha256"] == binding["roles"]["training_terminal"]["sha256"], "exact original audit inputs")
        figure = records["figure_receipt"]
        require(figure["status"] == "completed" and figure["version"] == "otto-separate-prior-plot-v1"
            and figure["condition_rows"] == 39 and figure["individual_points"] == 198
            and figure["family_mean_points"] == 64 and figure["fit_cost_rows"] == 24,
            "finished complete publication figure")
        require(set(figure["gates"]) == set(GATE_SIZES), "all three separate gate outcomes")
        for name, size in GATE_SIZES.items():
            gate = figure["gates"][name]
            require(set(gate) == {"conditions", "passed", "total", "passes"}
                and gate["total"] == len(gate["conditions"]) == len(set(gate["conditions"])) == size
                and type(gate["passed"]) is int and 0 <= gate["passed"] <= size
                and type(gate["passes"]) is bool and gate["passes"] == (gate["passed"] == size),
                "complete named gate metadata; either outcome is publishable")
        closed_record(figure, roles["figure_receipt"], FIGURE_PAYLOADS)
        figure_inputs = figure["inputs"]
        for role in ("audit_receipt", "audit_terminal", "training_receipt"):
            path = roles[role]
            require(figure_inputs[str(path)] == {"path": str(path), **members[relative(path)]},
                    "figure belongs to selected closed phase " + role)
        summary_path = roles["training_receipt"].parent / "summary.json"
        require(figure_inputs[str(summary_path)] == {"path": str(summary_path),
            **records["training_receipt"]["files"]["summary.json"]}, "figure belongs to selected full summary")
        add(PLOTTER, figure_inputs[str(ROOT / PLOTTER)])
        helper_pin = figure_inputs[str(ROOT / PLOT_HELPER)]
        require(helper_pin["sha256"] == PLOT_HELPER_PIN, "qualified plot helper identity")
        add(PLOT_HELPER, helper_pin)

        # Current source closure is complete. Older qualification receipts are
        # retained verbatim; their earlier source hashes are not claims that a
        # mutable current pathname still contains the earlier bytes.
        for plan in (records["collection_plan"], records["training_plan"], capplan, freeze):
            for source, pin in plan["sources"].items():
                require(source in freeze["sources"] and freeze["sources"][source] == pin,
                        "scientific source agrees with prospective freeze " + source)
                candidate = snapshots.get(source + "@" + pin, snapshots.get(source, source))
                actual = digest(candidate, check)
                require(actual["sha256"] == pin, "exact current frozen source " + source)
                archived = add(candidate, actual)
                source_bindings.setdefault(source, {})[pin] = archived
        for name in sorted(initial_study):
            if not name.endswith(".json"):
                continue
            if initial_study[name]["bytes"] > 32 * 1024**2:
                opaque_json.append(name)
                continue
            record = read(ROOT / name)
            if not isinstance(record, dict):
                continue
            for field in ("sources", "sources_before", "sources_after"):
                mapping = record.get(field, {})
                if not isinstance(mapping, dict):
                    continue
                for source, pin in mapping.items():
                    if not isinstance(pin, str) or len(pin) != 64:
                        continue
                    candidate = snapshots.get(source + "@" + pin, snapshots.get(source, source))
                    actual = digest(candidate, check) if (ROOT / candidate).is_file() else None
                    included = actual is not None and actual["sha256"] == pin
                    archived = add(candidate, actual) if included else None
                    historical_sources[name + ":" + field + ":" + source] = {
                        "original_path": source, "sha256": pin, "archived_path": archived,
                        "scope": "matching source included" if included else "historical receipt witness; prior source bytes not supplied"}
            for field in ("inputs", "native_inputs"):
                values = record.get(field, {})
                require(isinstance(values, (dict, list)), "recorded input descriptors are mapping or ordered list")
                items = values.items() if isinstance(values, dict) else enumerate(values)
                for role, item in items:
                    if not isinstance(item, dict) or not {"path", "sha256", "bytes"} <= set(item):
                        continue
                    original = str(Path(item["path"]).relative_to(ROOT)) if Path(item["path"]).is_absolute() else item["path"]
                    resolved = pinned_path(item["path"], item)
                    entry = {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"],
                             "resolved_path": str(resolved), "included": relative(resolved) in members,
                             "container": "mapping" if isinstance(values, dict) else "ordered_list",
                             "record_role": item.get("role"), "key_or_index": role}
                    dependencies[name + ":" + field + ":" + str(role)] = entry
                    if original in snapshots or original + "@" + item["sha256"] in snapshots:
                        require(relative(resolved) in members, "explicit historical snapshot archived")
        publications = binding["publications"]
        require(isinstance(publications, list) and REQUIRED_PUBLICATIONS <= {relative(p) for p in publications},
                "final results and canonical figure required")
        for name in publications:
            add(name)
        for artifact in ("forecast", "costs"):
            for suffix in ("png", "svg"):
                canonical = f"docs/assets/otto-separate-prior-{artifact}.{suffix}"
                if canonical in {relative(p) for p in publications}:
                    item = figure["files"][f"otto-separate-prior-{artifact}.{suffix}"]
                    require(members[canonical] == {k: item[k] for k in ("sha256", "bytes")},
                            "canonical plot copy matches selected figure " + canonical)
        for role in ("collection_receipt", "training_receipt", "audit_receipt", "figure_receipt"):
            equals = {"status": "completed"}
            if role == "audit_receipt":
                equals["agreement"] = True
            completion_checks.append({"path": relative(roles[role]), "equals": equals})
        for role in ("collection_terminal", "training_terminal", "audit_terminal"):
            completion_checks.append({"path": relative(roles[role]),
                "equals": {"status": "completed", "returncode": 0, "timed_out": False, "group_absent": True}})
        completion_checks.extend([
            {"path": relative(capacity_paths["receipt"]),
             "equals": {"status": "completed", "complete": True, "admitted": True}},
            {"path": relative(capacity_paths["terminal"]),
             "equals": {"status": "completed", "returncode": 0, "timed_out": False, "group_absent": True}}])
        require(sum(d["bytes"] + 4096 for d in members.values()) + 16 * 1024**2 < LIMITS["output_bytes"],
                "bounded raw archive headroom")
        require({relative(p): digest(p, check) for p in STUDY.rglob("*") if p.is_file()} == initial_study,
                "scientific artifact membership and bytes unchanged during preparation")
        require(digest(args.bindings, check)["sha256"] == args.bindings_sha256, "unchanged completion bindings")
        for name in (SELF, ENGINE, PLOTTER, PLOT_HELPER):
            require(digest(name, check) == members[name], "unchanged packaging/presentation source")
        restore = ("# Restoring the separate-prior evidence\n\n"
            "Verify SHA256SUMS, then extract the tar without rewriting original evidence. The manifest records every member hash. "
            "All current scientific phase files, qualification attempts, final checkpoints, predictions, publication assets and licenses are preserved.\n\n"
            "This archive is not a standalone runtime. External native weights, kernels, interpreters and inherited evidence are listed in dependencies. "
            "Obtain those exact bytes from their original releases and restore the recorded absolute paths and pinned runtime before strict numerical replay. "
            "No relocation adapter is provided. Byte inspection does not require scientific execution.\n\n"
            "Explicit inherited_snapshots map historical publication bytes to archive members. Current README/index publication copies do not replace those old bytes. "
            "historical_source_witnesses distinguish included matching source bytes from earlier qualification versions recorded only by hash; those older versions require their original snapshots. "
            "The external inventory and packaging receipt are outside the tar to avoid self-reference. Technical closure is separate from the three scientific gates (19, 19 and 29 conditions); all 39 records are retained and no favorable outcome is required.\n")
        inventory = {"version": VERSION, "status": "prepared_not_archived", "limits": LIMITS,
            "members": dict(sorted(members.items())), "completion_checks": completion_checks,
            "completion_bindings": binding["roles"], "source_bindings": source_bindings,
            "historical_source_witnesses": historical_sources,
            "opaque_large_json_members": opaque_json,
            "opaque_json_scope": "Hash-bound and archived in full; JSON over 32 MiB is not decoded for historical dependency extraction.",
            "named_gates": figure["gates"], "scientific_condition_records": 39,
            "prospective_source_count": len(freeze["sources"]),
            "dependencies": dependencies, "inherited_snapshots": snapshots,
            "publications": publications, "restore_text": restore}
        write(args.output / "inventory.json", inventory)
        inventory_pin = digest(args.output / "inventory.json", check)
        check(); finish = time.monotonic_ns()
        receipt.update(status="completed", members=len(members), inventory=inventory_pin,
            started_ns=start, finished_ns=finish, wall_seconds=(finish-start)/1e9)
        write(args.output / "receipt.json", receipt)
        check()
        print(json.dumps({"status": "prepared_not_archived", "inventory": receipt["inventory"]}), flush=True)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(status="failed", error=repr(error))
        try:
            if (args.output / "receipt.json").exists():
                (args.output / "receipt.json").rename(args.output / "receipt.invalid.json")
            write(args.output / "receipt.json", receipt)
        except BaseException as secondary:  # noqa: BLE001 - preserve primary preparation failure
            error.add_note("Failure receipt publication: " + repr(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def archive(args):
    require(args.output.is_absolute() and args.output.is_relative_to(PACKAGE) and args.output != PACKAGE,
            "separate packaging output root")
    require(digest(ENGINE)["sha256"] == ENGINE_PIN and digest(args.manifest)["sha256"] == args.manifest_sha256,
            "qualified engine and external prepared inventory")
    preparation = read(args.manifest.parent / "receipt.json")
    require(args.manifest.name == "inventory.json" and preparation["version"] == VERSION
        and preparation["status"] == "completed" and preparation["inventory"] == digest(args.manifest)
        and {p.name for p in args.manifest.parent.iterdir()} == {"inventory.json", "receipt.json"},
        "successful closed inventory preparation")
    inventory = read(args.manifest)
    require(SELF in inventory["members"] and "LICENSE" in inventory["members"], "adapter and root license in tar")
    spec = importlib.util.spec_from_file_location("_separate_prior_opaque_packager", ROOT / ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION, module.NAME, module.LIMITS = VERSION, NAME, LIMITS
    module.execute(SimpleNamespace(manifest=args.manifest, manifest_sha256=args.manifest_sha256, output=args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("prepare")
    plan.add_argument("--bindings", type=Path, required=True)
    plan.add_argument("--bindings-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("archive")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        archive(args)


if __name__ == "__main__":
    main()
