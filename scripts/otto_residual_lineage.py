"""Authenticate the closed prior study as checkpoint provenance, using stdlib only.

This is not a new scientific audit or admission of any empirical phase. The
three immutable anchors bind the original plan and successful process closures,
including the independent saved-output audit. That audit checked numerical
checkpoint contents, optimizer journals and the chronological pre-DEV barrier.
Here every producer/auditor payload is hashed as opaque bytes; only selected
JSON metadata is parsed. No journal, metric, array or checkpoint is decoded,
and no old TEST file or TEST evaluator is opened or imported.

The historical scientific outcome remains FAIL 6/13. Technical provenance of
TRAIN-derived weights does not promote that method or reopen its TEST split.
The returned runtime is the prior recorded runtime, not a current-env check.
The caller must bind this entire result in a new plan, reauthenticate before
decoding, and verify each returned checkpoint descriptor immediately at load.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = "output/otto-query-memory-v1"
PINS = {
    "training_plan": (f"{STUDY}/training-plan-01.json", "9ea6b7422116ca3a708e93d2339f4209ddad8a695feac9d31c590f758aac140a"),
    "training_closure": (f"{STUDY}/training-closure-01.json", "0829963b311342638a7d18fd9f5681047da8825e6d147cbd4f2189811a322821"),
    "dev_audit_closure": (f"{STUDY}/dev-audit-closure-01.json", "7b9d4b591ee603d29dd7a369c402095cb377a0156257b314a436ce36a4acfa47"),
}
SEEDS = (309000001, 309000002, 309000003)
FITS = ("pretrained", "joint_aux", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled")
VIEWS = ("pretrained", "joint_aux", "last_error", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled", "trace_no_write")
SELECTED = ("pretrained", "joint_aux", "trace_delta")
PRODUCER = "scripts/train_otto_query_memory.py"
AUDITOR = "scripts/audit_otto_query_memory.py"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
TRAIN_VERSION = "otto-query-memory-training-v1"
AUDIT_VERSION = "otto-query-memory-dev-saved-audit-v1"
TRAIN_LIMITS = {"seconds": 21600, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
AUDIT_LIMITS = {"seconds": 600, "rss_bytes": 2 * 1024**3, "output_bytes": 256 * 1024**2}
PAYLOADS = {"started.json", "runtime.json", "progress.jsonl", "work.jsonl", "fits.json", "forks.json",
            "train-views.json", "dev-views.json", "summary.json", "dev-barrier.json",
            "train-history.npz", "train-history.json", "dev-history.npz", "dev-history.json"}
PAYLOADS |= {f"checkpoint-{family}-{seed}.npz" for seed in SEEDS for family in FITS}
PAYLOADS |= {f"{stage}-prediction-{view}-{seed}.npz" for stage in ("train", "dev") for seed in SEEDS for view in VIEWS}
INPUT_ROLES = {"collection_plan", "collection_receipt", "collection_terminal", "engineering",
               "capacity_plan", "capacity_receipt", "capacity_terminal"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts and path.is_file()
            and not any(part.is_symlink() for part in (path, *path.parents)), "regular contained lineage file")
    return path


def descriptor(value):
    path = regular(value)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
            size += len(block)
    require(size == path.stat().st_size, "stable lineage byte count")
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


def _pin(record):
    require(isinstance(record, dict) and set(record) == {"sha256", "bytes"}
            and type(record["sha256"]) is str and len(record["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in record["sha256"])
            and type(record["bytes"]) is int and record["bytes"] >= 0, "exact byte descriptor")
    return record


def _bytes(record):
    return {key: record[key] for key in ("sha256", "bytes")}


class Evidence:
    def __init__(self):
        self.records = {}

    def add(self, role, path, expected=None):
        actual = descriptor(path)
        if isinstance(expected, str):
            require(actual["sha256"] == expected, "immutable lineage anchor: " + role)
        elif expected is not None:
            require(_bytes(actual) == _pin(expected), "lineage byte identity: " + role)
        require(role not in self.records or self.records[role] == actual, "consistent lineage role")
        self.records[role] = actual
        return actual

    def external(self, role, record):
        require(isinstance(record, dict) and set(record) == {"path", "sha256", "bytes"}, "external lineage descriptor")
        return self.add(role, record["path"], _bytes(record))

    def read(self, role):
        record = self.records[role]
        raw = regular(record["path"]).read_bytes()
        require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
                "metadata unchanged before parsing")
        value = json.loads(raw)
        require(isinstance(value, dict), "lineage metadata object")
        return value

    def phase(self, label, receipt, expected):
        directory = regular(self.records[label + "_receipt"]["path"]).parent
        require(set(receipt["files"]) == expected
                and {p.name for p in directory.iterdir()} == expected | {"receipt.json"}, "exact closed " + label + " roster")
        for name, pin in receipt["files"].items():
            require(Path(name).name == name and name not in (".", ".."), "simple payload name")
            self.add(label + "_payload:" + name, directory / name, pin)
        return directory

    def process(self, label, receipt, sources, *, cap, script, audit=False):
        parent = self.read(label + "_terminal")
        prefix = "dev-audit" if audit else "training"
        self.add(label + "_launch", f"{STUDY}/{prefix}-native-01.launch.json", receipt["supervision_sha256"])
        launch = self.read(label + "_launch")
        require(parent["status"] == "completed" and type(parent["returncode"]) is int and parent["returncode"] == 0
                and parent["timed_out"] is False and parent["group_absent"] is True
                and parent["cleanup"]["reaped"] is True and parent["cleanup"]["group_absent"] is True
                and parent["cleanup"]["errors"] == [] and parent["timing_available"] is True
                and parent["error"] is parent["clock_error"] is None and parent["cap_seconds"] == cap
                and parent["cwd"] == str(ROOT) and parent["pid"] == parent["pgid"] != parent["parent_pid"]
                and parent["watchdog_sha256"] == sources[SUPERVISOR] and parent["clock_source_sha256"] == sources[CLOCK]
                and parent["deadline_ns"] == parent["started_ns"] + cap * 10**9
                and parent["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
                <= parent["finished_ns"] <= parent["deadline_ns"]
                and all(parent[key] == value for key, value in launch.items()), "successful original " + label + " supervisor")
        command = list(parent["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        base = [str(ROOT / ".venv/bin/python"), str(ROOT / script)] + ([] if audit else ["run"])
        require(command[:len(base)] == base and (len(command) - len(base)) % 2 == 0, "original lineage command")
        keys, values = command[len(base)::2], command[len(base) + 1::2]
        require(len(keys) == len(set(keys)), "unique lineage command options")
        options = dict(zip(keys, values, strict=True))
        expected = {"--plan": self.records["training_plan"]["path"], "--plan-sha256": self.records["training_plan"]["sha256"],
                    "--supervision": self.records[label + "_launch"]["path"],
                    "--output": str(Path(self.records[label + "_receipt"]["path"]).parent)}
        if audit:
            for option, role in (("worker", "training_receipt"), ("terminal", "training_terminal")):
                expected.update({"--" + option: self.records[role]["path"], "--" + option + "-sha256": self.records[role]["sha256"]})
        require(options == expected, "original process exact input/output joins")
        started = self.read(label + "_payload:started.json")
        require(started["launch"] == launch and started["started_ns"] == receipt["started_ns"], "original worker launch joins")
        if audit:
            require(started["version"] == AUDIT_VERSION and started["inputs"] == receipt["inputs"]
                    and started["authenticated_before_numerical_reads"] is True, "auditor pre-read authentication")
        else:
            require(started["request"] == {"mode": "run", **{key[2:].replace("-", "_"): value for key, value in options.items()}},
                    "training started request joins")


def _fit_lineage(evidence, training):
    fits = evidence.read("training_payload:fits.json")["fits"]
    forks = evidence.read("training_payload:forks.json")["forks"]
    barrier = evidence.read("training_payload:dev-barrier.json")
    require([(row["family"], row["seed"]) for row in fits] == [(family, seed) for seed in SEEDS for family in FITS]
            and [row["seed"] for row in forks] == list(SEEDS), "all18 completed same-seed fits and forks")
    by = {(row["family"], row["seed"]): row for row in fits}
    for seed in SEEDS:
        parent = by["pretrained", seed]
        projection = None
        for family in FITS:
            row = by[family, seed]
            name = f"checkpoint-{family}-{seed}.npz"
            matrix = family not in ("pretrained", "joint_aux")
            epochs = 80 if family == "pretrained" else 40
            require(row["checkpoint_path"] == name and row["checkpoint"] == training["files"][name]
                    and row["epochs"] == epochs and row["steps"] == epochs * 9
                    and row["episode_exposures"] == epochs * 54 and row["optimizer_initial_state_entries"] == 0
                    and row["stage"] == ("pretrain" if family == "pretrained" else "branch")
                    and row["mode"] == (family if matrix else "none")
                    and row["slow_mode"] == ("frozen" if matrix else "joint"), "final fixed-epoch fit identity")
            if family == "pretrained":
                require(row["pretrained_checkpoint"] is None and row["frozen_slow_unchanged"] is None, "fresh pretrained parent")
            else:
                require(row["pretrained_checkpoint"] == {"path": parent["checkpoint_path"], **parent["checkpoint"]}
                        and row["initial_slow"] == parent["final_slow"], "exact same-seed pretrained fork")
                for name, witness in parent["final"]["tensors"].items():
                    require(row["initial"]["tensors"][name] == witness, "initial branch uses parent tensors")
                if matrix:
                    require(row["frozen_slow_unchanged"] is True and row["final_slow"] == parent["final_slow"], "frozen slow witness")
                    current = row["initial"]["tensors"]["projection.weight"]
                    require(current["shape"] == [8, 28] and current["dtype"] == "torch.float32" and current["bytes"] == 896
                            and (projection is None or projection == current), "shared seeded projection witness")
                    projection = current
                else:
                    require(row["frozen_slow_unchanged"] is None, "joint branch makes no frozen claim")
        expected = {"seed": seed, "pretrained_checkpoint": parent["checkpoint"], "pretrained_checkpoint_path": parent["checkpoint_path"],
                    "memory_initial_projection": projection,
                    "branches": [{"family": family, **{key: by[family, seed][key] for key in
                        ("initial", "initial_slow", "checkpoint_path", "checkpoint")}} for family in FITS[1:]]}
        require(forks[list(SEEDS).index(seed)] == expected, "complete original same-seed fork record")
    expected = {f"checkpoint-{family}-{seed}.npz" for seed in SEEDS for family in FITS}
    expected |= {f"train-prediction-{view}-{seed}.npz" for seed in SEEDS for view in VIEWS}
    expected |= {"fits.json", "forks.json", "train-views.json", "train-history.json", "train-history.npz"}
    require(barrier["event"] == "all18_checkpoints_and24_TRAIN_views_closed_before_DEV"
            and barrier["fits"] == 18 and barrier["train_views"] == 24 and barrier["optimizer_steps"] == 7560
            and barrier["episode_exposures"] == 45360 and type(barrier["work_sequence"]) is int and barrier["work_sequence"] > 7560
            and set(barrier["files"]) == expected and len(expected) == 47
            and all(pin == training["files"][name] for name, pin in barrier["files"].items()), "durable all-fit pre-DEV barrier")
    return {family: {str(seed): evidence.records[f"training_payload:checkpoint-{family}-{seed}.npz"].copy()
                    for seed in SEEDS} for family in SELECTED}


def authenticate():
    """Return immutable prior evidence and exactly nine permitted checkpoint pins."""
    evidence = Evidence()
    for role, (path, pin) in PINS.items():
        evidence.add(role, path, pin)
    plan, closure, audit_closure = (evidence.read(role) for role in ("training_plan", "training_closure", "dev_audit_closure"))
    sources = plan["sources"]
    require(plan["version"] == TRAIN_VERSION and plan["status"] == "frozen_before_fitting"
            and plan["limits"] == TRAIN_LIMITS and set(plan["payloads"]) == PAYLOADS and len(PAYLOADS) == 80
            and len(sources) == 145 and {PRODUCER, AUDITOR, CLOCK, SUPERVISOR} <= set(sources), "frozen original recipe and145 sources")
    for name, pin in sources.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and descriptor(name)["sha256"] == pin,
                "unchanged frozen source: " + name)
    require(set(plan["inputs"]) == INPUT_ROLES, "complete inherited evidence inputs")
    for role, pin in plan["inputs"].items():
        evidence.external("prior_input:" + role, pin)
    require(closure["status"] == "original_training_closure_authenticated" and closure["frozen_source_count"] == 145
            and closure["payload_count"] == 80 and closure["technical_complete"] is False
            and closure["requires_independent_saved_audit"] is True
            and closure["numerical_array_decodes"] == closure["model_calls"] == 0
            and set(closure["inputs"]) == {"training_plan", "training_receipt", "training_terminal"}, "original training closure anchor")
    for role, pin in closure["inputs"].items():
        require(pin["path"] == str(ROOT / f"{STUDY}/" / {"training_plan": "training-plan-01.json", "training_receipt": "training-01/receipt.json",
                                                      "training_terminal": "training-native-01.terminal.json"}[role]), "canonical training evidence path")
        evidence.external(role, pin)
    training = evidence.read("training_receipt")
    require(training["version"] == TRAIN_VERSION and training["status"] == "completed" and training["complete"] is True
            and training["plan_sha256"] == evidence.records["training_plan"]["sha256"] and training["sources"] == sources
            and training["inputs"] == plan["inputs"] and training["limits"] == TRAIN_LIMITS
            and training["fits_completed"] == 18 and training["optimizer_steps"] == 7560 and training["episode_exposures"] == 45360
            and training["teacher_calls"] == training["native_calls"] == training["test_array_decodes"] == 0
            and training["technical_complete"] is False and training["pending"] is training["pending_emission"] is None
            and training["requires_successful_original_supervisor"] is training["requires_independent_saved_audit"] is True
            and training["peak_rss_bytes"] <= TRAIN_LIMITS["rss_bytes"], "complete original training producer")
    evidence.phase("training", training, PAYLOADS)
    require(sum(pin["bytes"] for pin in training["files"].values()) == closure["payload_bytes"] <= TRAIN_LIMITS["output_bytes"], "producer byte bound")
    evidence.process("training", training, sources, cap=21600, script=PRODUCER)
    checkpoints = _fit_lineage(evidence, training)
    runtime = evidence.read("training_payload:runtime.json")
    require(plan["runtime"]["executable"] == str(ROOT / ".venv/bin/python")
            and all(runtime[key] == value for key, value in plan["runtime"].items())
            and runtime["torch_threads"] == runtime["interop_threads"] == 1 and runtime["deterministic"] is True
            and runtime["cuda_used"] is runtime["mps_used"] is False, "recorded prior CPU runtime")
    require(audit_closure["status"] == "original_DEV_audit_closure_authenticated" and audit_closure["gate_passed"] is False
            and audit_closure["conditions_passed"] == 6 and audit_closure["conditions_total"] == 13
            and audit_closure["test_evaluation_admitted"] is False and audit_closure["numerical_array_decodes_by_handoff"] == 0,
            "closed historical scientific FAIL6/13 with TEST unused")
    for role, name in (("receipt", "dev-audit-01/receipt.json"), ("terminal", "dev-audit-native-01.terminal.json"), ("audit", "dev-audit-01/audit.json")):
        actual = evidence.external("dev_audit_" + role, audit_closure[role])
        require(actual["path"] == str(ROOT / STUDY / name), "canonical audit evidence path")
    audit = evidence.read("dev_audit_receipt")
    expected_inputs = {key: evidence.records[role] for key, role in
                       (("plan", "training_plan"), ("worker", "training_receipt"), ("terminal", "training_terminal"))}
    require(audit["version"] == AUDIT_VERSION and audit["status"] == "completed" and audit["complete"] is audit["agreement"] is True
            and audit["pending"] is None and audit["sources"] == sources and audit["inputs"] == expected_inputs
            and audit["source_plan_sha256"] == evidence.records["training_plan"]["sha256"]
            and audit["limits"] == AUDIT_LIMITS and audit["peak_rss_bytes"] <= AUDIT_LIMITS["rss_bytes"]
            and audit["requires_successful_original_supervisor"] is True and audit["array_decodes"] == 70
            and audit["training_updates_checked"] == 7560 and audit["episode_exposures_checked"] == 45360
            and audit["paired_work_calls_checked"] == 27432
            and all(audit[key] == 0 for key in ("model_calls", "optimizer_calls", "teacher_calls", "simulator_calls", "test_array_decodes")),
            "independent successful saved-output audit, not a scientific pass")
    evidence.phase("dev_audit", audit, {"started.json", "audit.json"})
    require(audit["files"]["audit.json"] == _bytes(evidence.records["dev_audit_audit"])
            and sum(pin["bytes"] for pin in audit["files"].values()) <= AUDIT_LIMITS["output_bytes"], "audit result identity and byte bound")
    evidence.process("dev_audit", audit, sources, cap=600, script=AUDITOR, audit=True)
    require(evidence.read("training_terminal")["finished_ns"] <= evidence.read("dev_audit_terminal")["started_ns"], "audit starts after original training closure")
    # Recheck all authenticated bytes and sources at the handoff boundary.
    for record in evidence.records.values():
        require(descriptor(record["path"]) == record, "lineage evidence unchanged at return")
    for name, pin in sources.items():
        require(descriptor(name)["sha256"] == pin, "frozen source unchanged at return")
    return {"sources": dict(sources), "checkpoints": checkpoints, "evidence": evidence.records, "runtime": plan["runtime"]}
