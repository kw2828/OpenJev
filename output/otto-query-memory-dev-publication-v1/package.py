"""Package closed TRAIN/DEV evidence as opaque bytes, preserving FAIL 6/13."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
STUDY = "output/otto-query-memory-v1"
COLLECTION_PUBLICATION = "output/otto-query-memory-collection-publication-v1"
RENDER_ENGINEERING = "output/otto-query-memory-render-engineering-v1"
PINS = {
    f"{STUDY}/training-plan-01.json": "9ea6b7422116ca3a708e93d2339f4209ddad8a695feac9d31c590f758aac140a",
    f"{STUDY}/training-01/receipt.json": "1e8303c5c632b1d4e1f781e5645aae36b380e06097da7a011bbdcdaff87b4794",
    f"{STUDY}/training-native-01.terminal.json": "650bad61972649a2b3adac08ad1a7b248d731b91b31f52039ee9e1e2a9511c7e",
    f"{STUDY}/dev-audit-01/receipt.json": "d666f72b2b7e25e132216f09164fb476e9399d94d718de528681c483dbfc280a",
    f"{STUDY}/dev-audit-native-01.terminal.json": "96307093c122b56e9d5aaebcb6eb89b58c777da2406769918deb2889e78b4936",
    f"{STUDY}/dev-audit-closure-01.json": "7b9d4b591ee603d29dd7a369c402095cb377a0156257b314a436ce36a4acfa47",
}
COLLECTION_DEPENDENCY = {
    "url": "https://github.com/kw2828/OpenJev/releases/download/otto-query-memory-collection-v1/otto-query-memory-collection-v1.tar.gz",
    "release": "https://github.com/kw2828/OpenJev/releases/tag/otto-query-memory-collection-v1",
    "sha256": "720f65933c36fb565e4a838b52acefe2d7829e8c56f67df71b70a4ecf0839ddd",
    "bytes": 37194452,
    "scope": "required companion collection evidence; includes all17 collection payloads and108 paths",
}
SCOPE = "opaque complete TRAIN/DEV evidence companion; scientific gate FAIL6/13; TEST remains unadmitted"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(value):
    path = Path(value)
    if path.is_absolute():
        require(path.is_relative_to(ROOT), "contained original absolute path")
        path = path.relative_to(ROOT)
    require(".." not in path.parts and path.parts, "safe repository-relative member")
    return str(path)


def payload(value):
    name = relative(value)
    path = ROOT / name
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "regular repository file")
    return path.read_bytes()


def pin_bytes(value):
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


class Package:
    def __init__(self):
        self.members = {}

    def add(self, value, pin=None):
        name, content = relative(value), payload(value)
        if pin is not None:
            actual = pin_bytes(content)
            require(actual["sha256"] == pin if isinstance(pin, str) else actual == pin, "pinned archive input: " + name)
        require(name not in self.members or self.members[name] == content, "consistent duplicate archive member")
        self.members[name] = content
        return name

    def metadata(self, value, pin=None):
        return json.loads(self.members[self.add(value, pin)])

    def phase(self, label, receipt, expected):
        directory = f"{STUDY}/{label}-01"
        require(set(receipt["files"]) == set(expected), "exact " + label + " payload roster")
        require({p.name for p in (ROOT / directory).iterdir()} == set(expected) | {"receipt.json"}, "closed " + label + " inventory")
        for name, pin in receipt["files"].items():
            require(Path(name).name == name, "simple phase payload filename")
            self.add(f"{directory}/{name}", pin)

    def process(self, label, receipt, *, script, cap, plan_record, worker_record=None, training_terminal=None):
        prefix = f"{STUDY}/{label}-native-01"
        parent = self.metadata(prefix + ".terminal.json")
        launch = self.metadata(prefix + ".launch.json", receipt["supervision_sha256"])
        require(parent["status"] == "completed" and type(parent["returncode"]) is int and parent["returncode"] == 0
                and parent["timed_out"] is False and parent["group_absent"] is True and parent["cleanup"]["reaped"] is True
                and parent["cleanup"]["group_absent"] is True and parent["cleanup"]["errors"] == []
                and parent["error"] is parent["clock_error"] is None and parent["timing_available"] is True
                and parent["cap_seconds"] == cap and parent["cwd"] == str(ROOT)
                and parent["pid"] == parent["pgid"] != parent["parent_pid"]
                and parent["deadline_ns"] == parent["started_ns"] + cap * 10**9
                and parent["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
                <= parent["finished_ns"] <= parent["deadline_ns"] and all(parent[k] == v for k, v in launch.items()),
                "original successful " + label + " parent and launch")
        require(parent["watchdog_sha256"] == pin_bytes(payload("scripts/supervise_dialogue_observation_v2.py"))["sha256"]
                and parent["clock_source_sha256"] == pin_bytes(payload("src/openjev/research/suspend_clock.py"))["sha256"],
                "unchanged qualified supervisor and clock")
        started = self.metadata(f"{STUDY}/{label}-01/started.json")
        require(started["launch"] == launch and started["started_ns"] == receipt["started_ns"], "original worker start joins")
        command = list(parent["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        base = [str(ROOT / ".venv/bin/python"), str(ROOT / script)]
        if label != "dev-audit":
            base.append("run")
        require(command[:len(base)] == base and (len(command) - len(base)) % 2 == 0, "original process command")
        keys, values = command[len(base)::2], command[len(base) + 1::2]
        require(len(keys) == len(set(keys)), "unique original process options")
        options = dict(zip(keys, values, strict=True))
        expected = {"--plan": plan_record["path"], "--plan-sha256": plan_record["sha256"],
                    "--supervision": str(ROOT / (prefix + ".launch.json")), "--output": str(ROOT / f"{STUDY}/{label}-01")}
        if label == "dev-audit":
            expected.update({"--worker": worker_record["path"], "--worker-sha256": worker_record["sha256"],
                             "--terminal": training_terminal["path"], "--terminal-sha256": training_terminal["sha256"]})
        require(options == expected, "original process input and output identities")
        for suffix in (".detached-intent.json", ".detached-launch.json"):
            self.add(prefix + suffix)
        self.add(f"{STUDY}/{label}-launch-command-01.json")


def main():
    output = BASE / "attempt-01"
    output.mkdir(exist_ok=False)
    package = Package()
    for name, digest in PINS.items():
        package.add(name, digest)
    plan = package.metadata(f"{STUDY}/training-plan-01.json")
    training = package.metadata(f"{STUDY}/training-01/receipt.json")
    audit = package.metadata(f"{STUDY}/dev-audit-01/receipt.json")
    closure = package.metadata(f"{STUDY}/dev-audit-closure-01.json")
    train_closure = package.metadata(f"{STUDY}/training-closure-01.json")
    training_plan = {"path": str(ROOT / f"{STUDY}/training-plan-01.json"), **pin_bytes(package.members[f"{STUDY}/training-plan-01.json"])}
    training_receipt = {"path": str(ROOT / f"{STUDY}/training-01/receipt.json"), **pin_bytes(package.members[f"{STUDY}/training-01/receipt.json"])}
    training_terminal = {"path": str(ROOT / f"{STUDY}/training-native-01.terminal.json"), **pin_bytes(package.members[f"{STUDY}/training-native-01.terminal.json"])}
    require(plan["status"] == "frozen_before_fitting" and len(plan["sources"]) == 145 and len(plan["payloads"]) == 80,
            "fixed145-source80-payload training plan")
    require(training["status"] == "completed" and training["complete"] is True and training["fits_completed"] == 18
            and training["optimizer_steps"] == 7560 and training["episode_exposures"] == 45360
            and training["plan_sha256"] == training_plan["sha256"] and training["sources"] == plan["sources"]
            and training["inputs"] == plan["inputs"] and training["pending"] is training["pending_emission"] is None
            and training["test_array_decodes"] == training["teacher_calls"] == training["native_calls"] == 0,
            "complete original TRAIN/DEV producer")
    require(audit["status"] == "completed" and audit["complete"] is audit["agreement"] is True
            and audit["pending"] is None and audit["sources"] == plan["sources"]
            and audit["source_plan_sha256"] == training_plan["sha256"]
            and audit["inputs"] == {"plan": training_plan, "worker": training_receipt, "terminal": training_terminal}
            and all(audit[name] == 0 for name in ("model_calls", "optimizer_calls", "teacher_calls", "simulator_calls", "test_array_decodes")),
            "complete independent DEV audit with TEST unused")
    require(closure["status"] == "original_DEV_audit_closure_authenticated" and closure["gate_passed"] is False
            and closure["conditions_passed"] == 6 and closure["conditions_total"] == 13
            and closure["test_evaluation_admitted"] is False, "immutable failed scientific continuation gate")
    for role in ("audit", "receipt", "terminal"):
        pin = closure[role]
        package.add(pin["path"], {key: pin[key] for key in ("bytes", "sha256")})
    require(closure["audit"]["sha256"] == audit["files"]["audit.json"]["sha256"], "failed gate closure joins saved audit")
    require(train_closure["status"] == "original_training_closure_authenticated"
            and train_closure["inputs"] == {"training_plan": training_plan, "training_receipt": training_receipt,
                                            "training_terminal": training_terminal}, "training closure joins original evidence")
    package.phase("training", training, plan["payloads"])
    package.phase("dev-audit", audit, {"started.json", "audit.json"})
    package.process("training", training, script="scripts/train_otto_query_memory.py", cap=21600, plan_record=training_plan)
    package.process("dev-audit", audit, script="scripts/audit_otto_query_memory.py", cap=600, plan_record=training_plan,
                    worker_record=training_receipt, training_terminal=training_terminal)
    for name, digest in plan["sources"].items():
        package.add(name, digest)
    for pin in plan["inputs"].values():
        package.add(pin["path"], {key: pin[key] for key in ("bytes", "sha256")})
    capacity_plan = package.metadata(plan["inputs"]["capacity_plan"]["path"])
    capacity = package.metadata(plan["inputs"]["capacity_receipt"]["path"])
    require(capacity["status"] == "completed" and capacity["complete"] is capacity["admitted"] is True
            and capacity["plan_sha256"] == plan["inputs"]["capacity_plan"]["sha256"]
            and capacity["sources"] == capacity_plan["sources"] and capacity["inputs"] == capacity_plan["inputs"]
            and capacity["completed_family_count"] == capacity["optimizer_updates"] == 5
            and capacity["pending"] is capacity["pending_emission"] is None, "complete admitted capacity check")
    package.phase("capacity", capacity, {"started.json", "runtime.json", "synthetic.json", "work.jsonl", "summary.json"})
    package.process("capacity", capacity, script="scripts/qualify_otto_query_memory_capacity.py", cap=240,
                    plan_record=plan["inputs"]["capacity_plan"])
    engineering_path = relative(plan["inputs"]["engineering"]["path"])
    engineering = package.metadata(engineering_path)
    require(engineering["status"] == "passed" and engineering["source_before"] == engineering["source_after"]
            and set(engineering["files"]) == {"started.json", "command-1.log", "command-2.log", "audit-capacity.json"},
            "complete successful training qualification")
    for name, pin in engineering["files"].items():
        package.add(str(Path(engineering_path).parent / name), pin)
    for name, pin in engineering["source_after"].items():
        package.add(name, pin)
    collection = package.metadata(plan["inputs"]["collection_plan"]["path"])
    require(len(collection["native_inputs"]) == 8, "all eight native input dependencies declared")
    for suffix in ("attempt-01/receipt.json", "attempt-01/MANIFEST.json", "attempt-01/DEPENDENCIES.json",
                   "attempt-01/SHA256SUMS", "release-verification-01.json", "COLLECTION-README.md", "package.py"):
        package.add(f"{COLLECTION_PUBLICATION}/{suffix}")
    collection_package = package.metadata(f"{COLLECTION_PUBLICATION}/attempt-01/receipt.json")
    require(collection_package["status"] == "passed" and collection_package["all_members_byte_identical"] is True
            and collection_package["archive"]["sha256"] == COLLECTION_DEPENDENCY["sha256"]
            and collection_package["archive"]["bytes"] == COLLECTION_DEPENDENCY["bytes"], "exact published collection companion")
    collection_release = package.metadata(f"{COLLECTION_PUBLICATION}/release-verification-01.json")
    require(collection_release["status"] == "verified" and any(row["browser_download_url"] == COLLECTION_DEPENDENCY["url"]
            and row["digest"] == "sha256:" + COLLECTION_DEPENDENCY["sha256"] and row["size"] == COLLECTION_DEPENDENCY["bytes"]
            for row in collection_release["assets"]), "saved remote collection digest verification")
    # Hash the local companion as opaque bytes, but do not duplicate it or any
    # collection arrays/journals in this archive. Do not extract the companion.
    dependency_path = f"{COLLECTION_PUBLICATION}/attempt-01/otto-query-memory-collection-v1.tar.gz"
    require(pin_bytes(payload(dependency_path)) == {key: COLLECTION_DEPENDENCY[key] for key in ("bytes", "sha256")},
            "local opaque companion matches published dependency")
    for name in ("LICENSE", "third_party/otto/README.md", "third_party/otto/LICENSE", "third_party/otto/LICENSE-zoo",
                 "research/otto-learned-reference-and-symmetry.md", "scripts/render_otto_query_memory.py",
                 "tests/test_render_otto_query_memory.py", f"{STUDY}/training-planning-command-01.json",
                 f"{STUDY}/training-planning-command-01.started.json", f"{STUDY}/capacity-planning-command-01.json",
                 f"{STUDY}/capacity-planning-command-01.started.json"):
        package.add(name)
    renderer = package.metadata(f"{RENDER_ENGINEERING}/renderer-qualified.json")
    require(renderer["status"] == "passed" and renderer["empirical_reads"] == renderer["scientific_calls"] == 0,
            "separate successful fabricated renderer qualification")
    for name, pin in renderer["sources"].items():
        package.add(name, pin)
    previous = renderer["original_attempt"]
    attempt = package.metadata(previous["path"], {key: previous[key] for key in ("bytes", "sha256")})
    attempt_dir = Path(relative(previous["path"])).parent
    package.add(str(attempt_dir / "started.json"))
    for command in attempt["commands"]:
        package.add(str(attempt_dir / command["log"]), {key: command[key] for key in ("bytes", "sha256")})
    lint = renderer["followup_lint"]
    package.add(f"{RENDER_ENGINEERING}/{lint['log']}", {key: lint[key] for key in ("bytes", "sha256")})
    rendered = package.metadata(f"{STUDY}/render-01.json")
    require(rendered["stage"] == "dev" and rendered["synthetic"] is False
            and rendered["original_audit_closure_authenticated"] is True and rendered["record_count"] == 24
            and rendered["source_audit"] == closure["audit"]
            and rendered["source_closure"]["sha256"] == PINS[f"{STUDY}/dev-audit-closure-01.json"]
            and rendered["renderer_sha256"] == pin_bytes(package.members["scripts/render_otto_query_memory.py"])["sha256"]
            and set(rendered["files"]) == {"query-memory-dev.md", "query-memory-dev.png"}, "rendered failed DEV report provenance")
    for name, pin in rendered["files"].items():
        package.add(f"research/otto-query-memory-dev-results/{name}", pin)
    for name in ("package.py", "EVIDENCE-README.md", "release-notes.md"):
        package.add(str(BASE.relative_to(ROOT) / name))
    require(not any(name.startswith(f"{STUDY}/collection-01/") and name != f"{STUDY}/collection-01/receipt.json"
                    for name in package.members), "collection payloads remain in explicit companion archive")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    original = dict(package.members)
    package.members["EVIDENCE-README.md"] = payload(BASE / "EVIDENCE-README.md")
    package.members["DEPENDENCIES.json"] = encoded({"repository": "https://github.com/kw2828/OpenJev", "repository_commit": commit,
        "scope": SCOPE, "collection_evidence_archive": COLLECTION_DEPENDENCY,
        "native_inputs_not_bundled": collection["native_inputs"], "training_runtime": plan["runtime"],
        "native_runtime": collection["runtime"], "self_contained_executable_reproduction": False,
        "audit_authentication": "Saved-audit authentication also hashes the eight native inputs despite making no native calls.",
        "absolute_paths": "Original receipts preserve their original absolute paths. Relocated reproduction needs a distinct plan; do not rewrite original receipts.",
        "test_evaluation_admitted": False})
    manifest = {"kind": "opaque_train_dev_evidence", "scope": SCOPE, "repository_commit": commit,
        "training_payloads": 80, "training_payload_bytes": sum(pin["bytes"] for pin in training["files"].values()),
        "trained_checkpoints": 18, "train_predictions": 24, "dev_predictions": 24, "audit_payloads": 2,
        "capacity_payloads": 5, "training_source_records": 145,
        "gate_passed": False, "conditions_passed": 6, "conditions_total": 13, "test_evaluation_admitted": False,
        "numerical_decodes": 0, "trajectory_journal_parses": 0, "scientific_calls": 0,
        "files": {name: pin_bytes(value) for name, value in sorted(package.members.items())}}
    package.members["MANIFEST.json"] = encoded(manifest)
    archive = output / "otto-query-memory-dev-v1.tar.gz"
    with (
        archive.open("xb") as stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for name, content in sorted(package.members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(content), 0o644, 0
            tar.addfile(info, io.BytesIO(content))
    with archive.open("rb") as stream:
        os.fsync(stream.fileno())
    verified = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            require(member.isfile() and member.name not in verified and member.name in package.members,
                    "exact regular unique archive member")
            require(tar.extractfile(member).read() == package.members[member.name], "byte-identical opaque archive roundtrip")
            verified.add(member.name)
    require(verified == set(package.members), "complete archive member roundtrip")
    for name, content in original.items():
        require(payload(name) == content, "input remained unchanged during packaging")
    require(pin_bytes(payload(dependency_path)) == {key: COLLECTION_DEPENDENCY[key] for key in ("bytes", "sha256")},
            "unchanged opaque collection dependency")
    archive_pin = pin_bytes(archive.read_bytes())
    (output / "MANIFEST.json").write_bytes(package.members["MANIFEST.json"])
    (output / "DEPENDENCIES.json").write_bytes(package.members["DEPENDENCIES.json"])
    (output / "SHA256SUMS").write_text(f"{archive_pin['sha256']}  {archive.name}\n")
    receipt = {"status": "passed", "scope": SCOPE, "archive": {"name": archive.name, **archive_pin},
        "member_count": len(package.members), "all_members_byte_identical": True,
        "training_payloads": 80, "trained_checkpoints": 18, "train_predictions": 24, "dev_predictions": 24,
        "audit_payloads": 2, "capacity_payloads": 5, "training_source_records": 145,
        "gate_passed": False, "conditions_passed": 6, "conditions_total": 13, "test_evaluation_admitted": False,
        "numerical_decodes": 0, "trajectory_journal_parses": 0, "scientific_calls": 0,
        "collection_dependency": COLLECTION_DEPENDENCY, "self_contained_executable_reproduction": False,
        "repository_commit": commit, "packager": pin_bytes(Path(__file__).read_bytes())}
    (output / "receipt.json").write_bytes(encoded(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
