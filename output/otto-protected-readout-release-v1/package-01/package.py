"""Create an exclusive, byte-verified release package without scientific decoding.

Only plans, receipts, launch/terminal records and provenance JSON are decoded.
Scientific outputs and historical model/kernel artifacts are opaque bytes.
No inference, array loader, simulator, optimizer, Git mutation or upload occurs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
STUDY = ROOT / "output/otto-protected-readout-v1"
REPORT = ROOT / "output/otto-protected-readout-report-v1"
REPORT_PIN = "93f64c02cb715a0bd38f8bca8089a55f8637e280f4347fd1797932be682131ee"
OFFICIAL_COMMIT = "1467029f399dc5eeac8652499a9c8326ecab4575"
OFFICIAL_BASE = f"https://raw.githubusercontent.com/C0PEP0D/otto/{OFFICIAL_COMMIT}"
WEIGHT_URL = OFFICIAL_BASE + "/zoo/models/zoo_model_2_3_2/zoo_model_2_3_2"
WEIGHT_PIN = "1efb73aa38e0fd8b08d6d03059c0db4664da3afb8eaff1d7ab9b363f8e7ad37d"
WEIGHT_PATH = ROOT / "output/otto-pretrained-reference-v2/qualification-01/original.weights-legacy.h5"
FAMILIES = ("pretrained", "frozen_aux", "frozen_spo", "joint_aux", "joint_spo")
SEEDS = (301000001, 301000002, 301000003)
START = time.monotonic()
BUNDLES = {"worker": {}, "audit": {}}
PINS = {}
DEPS = []


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check():
    require(time.monotonic() - START < 600, "bounded packaging deadline")


def canonical(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), "regular contained source: " + str(path))
    return path


def descriptor(path):
    path = canonical(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            check(); digest.update(chunk); size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def bind(path, expected=None):
    path = canonical(path)
    actual = descriptor(path)
    if isinstance(expected, str):
        require(actual["sha256"] == expected, "pinned source hash: " + str(path))
    elif expected is not None:
        require(all(actual[key] == expected[key] for key in ("sha256", "bytes") if key in expected),
                "exact saved descriptor: " + str(path))
    name = path.relative_to(ROOT).as_posix()
    require(name not in PINS or PINS[name] == actual, "unchanged bound bytes: " + name)
    PINS[name] = actual
    return path


def read(path, expected=None):
    path = bind(path, expected)
    require(path.suffix == ".json" and path.stat().st_size <= 2 * 1024**2, "bounded metadata JSON only")
    return json.loads(path.read_text())


def save(name, value):
    path = OUT / name
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return path


def add(bundle, path, expected=None):
    path = bind(path, expected)
    name = path.relative_to(ROOT).as_posix()
    BUNDLES[bundle][name] = PINS[name]
    return path


def add_tree(bundle, directory):
    require(directory.is_dir() and not directory.is_symlink(), "real evidence directory")
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "no evidence symlinks")
        if path.is_file():
            add(bundle, path)


def phase(bundle, directory, expected_count):
    receipt = read(directory / "receipt.json")
    require(receipt["status"] == "completed" and len(receipt["files"]) == expected_count,
            "completed exact phase inventory")
    require({p.name for p in directory.iterdir()} == set(receipt["files"]) | {"receipt.json"},
            "closed phase file membership")
    for name, pin in receipt["files"].items():
        require(Path(name).name == name, "flat phase payload")
        add(bundle, directory / name, pin)
    add(bundle, directory / "receipt.json")
    return receipt


def supervisor(bundle, prefix, receipt, script, cap, expected, mode=None, executable=".venv/bin/python"):
    launch_path, terminal_path = Path(str(prefix) + ".launch.json"), Path(str(prefix) + ".terminal.json")
    launch = read(launch_path, receipt["supervision_sha256"])
    terminal = read(terminal_path)
    require(terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timed_out"] is False
        and terminal["error"] is terminal["clock_error"] is None
        and terminal["group_absent"] is terminal["cleanup"]["reaped"] is True
        and terminal["cleanup"]["errors"] == [] and terminal["cap_seconds"] == cap
        and terminal["cwd"] == str(ROOT)
        and terminal["clock_source_sha256"] == "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
        and terminal["watchdog_sha256"] == "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
        and terminal["deadline_ns"] == terminal["started_ns"] + cap * 10**9
        and terminal["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
        <= terminal["finished_ns"] <= terminal["deadline_ns"], "genuine successful original supervisor")
    require(all(terminal[key] == value for key, value in launch.items()), "unchanged original launch identity")
    command = list(terminal["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    head = [str(ROOT / executable), str(ROOT / script)] + ([] if mode is None else [mode])
    options = {**expected, "--supervision": str(launch_path)}
    require(command[:len(head)] == head and len(command) == len(head) + 2 * len(options), "original command size")
    pairs = command[len(head):]
    require(len(set(pairs[::2])) == len(options) and dict(zip(pairs[::2], pairs[1::2], strict=True)) == options,
            "original input and output argument joins")
    for path in sorted(prefix.parent.glob(prefix.name + ".*")):
        add(bundle, path)
    return terminal


def dependency(owner, section, role, value):
    require(isinstance(value, dict) and set(("path", "sha256", "bytes")) <= set(value), "registered dependency descriptor")
    path = bind(value["path"], value)
    name = path.relative_to(ROOT).as_posix()
    current = path.is_relative_to(STUDY) or path.is_relative_to(REPORT)
    external = path == WEIGHT_PATH
    if external:
        require(value["sha256"] == WEIGHT_PIN and value["bytes"] == 53581916, "exact public official weight identity")
    elif not any(name in members for members in BUNDLES.values()):
        add("worker" if path.suffix in (".npz", ".h5") else "audit", path, value)
    DEPS.append({"owner": owner, "section": section, "role": role, "original_descriptor": value,
        "repository_path": name, "classification": "current_study" if current else "historical_registered_input",
        "delivery": "immutable_upstream_dependency" if external else "included",
        **({"url": WEIGHT_URL, "license": "MIT", "notice": "OTTO-UPSTREAM-LICENSE.txt"} if external else {})})


def archive(bundle, manifest_path):
    archive_path = OUT / f"otto-protected-readout-v1-{bundle}-evidence.tar.gz"
    members = {**BUNDLES[bundle], manifest_path.relative_to(ROOT).as_posix(): descriptor(manifest_path)}
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(members):
                    check()
                    path = ROOT / name
                    require(descriptor(path) == members[name], "source unchanged before archive")
                    info = tar.gettarinfo(str(path), arcname=name)
                    require(info.isfile() and not Path(info.name).is_absolute() and ".." not in Path(info.name).parts,
                            "safe regular relative archive member")
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
        raw.flush(); os.fsync(raw.fileno())
    seen, total = set(), 0
    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            check()
            require(member.isfile() and member.name in members and member.name not in seen, "exact unique roundtrip member")
            require(not Path(member.name).is_absolute() and ".." not in Path(member.name).parts, "safe roundtrip path")
            stream = tar.extractfile(member)
            digest, count = hashlib.sha256(), 0
            for chunk in iter(lambda: stream.read(1024**2), b""):
                check(); digest.update(chunk); count += len(chunk)
            require({"sha256": digest.hexdigest(), "bytes": count} == members[member.name]
                    and member.size == count, "byte-for-byte archive roundtrip")
            seen.add(member.name); total += count
    require(seen == set(members), "complete archive membership")
    return {"path": archive_path.relative_to(ROOT).as_posix(), **descriptor(archive_path),
            "members": len(seen), "payload_files": len(BUNDLES[bundle]), "uncompressed_member_bytes": total,
            "manifest": {"path": manifest_path.relative_to(ROOT).as_posix(), **descriptor(manifest_path)},
            "roundtrip": "passed"}


def main():
    require({p.name for p in OUT.iterdir()} == {"package.py"}, "new exclusive package directory")
    save("started.json", {"status": "started", "scope": __doc__, "seconds_cap": 600,
                           "script": descriptor(__file__), "report_receipt_sha256": REPORT_PIN})
    try:
        report_receipt = read(REPORT / "report-01/receipt.json", REPORT_PIN)
        require(report_receipt["status"] == "completed" and report_receipt["array_decodes"] == 0, "closed saved-only report")
        collection_plan = read(STUDY / "collection-plan-01.json")
        train_plan = read(STUDY / "training-plan-01.json")
        collection = phase("worker", STUDY / "collection-01", 18)
        training = phase("worker", STUDY / "training-01", 58)
        audit = phase("audit", STUDY / "audit-01", 2)
        phase("audit", REPORT / "report-01", 13)
        require(collection["complete"] is training["complete"] is True and training["fits_completed"] == 15
                and training["optimizer_steps"] == 6480 and audit["agreement"] is True and audit["failures"] == [],
                "complete current study and agreeing audit")
        require(collection["sources"] == collection_plan["sources"] and collection["inputs"] == collection_plan["inputs"]
            and training["sources"] == train_plan["sources"] and training["inputs"] == train_plan["inputs"], "registered input/source identity")
        require(collection["plan_sha256"] == descriptor(STUDY / "collection-plan-01.json")["sha256"]
                and training["plan_sha256"] == descriptor(STUDY / "training-plan-01.json")["sha256"], "original plan identity")
        checkpoints = {f"{family}-{seed}.npz" for family in FAMILIES for seed in SEEDS}
        require(checkpoints <= training["files"].keys()
                and all(f"{prefix}{name}" in training["files"] for prefix in ("prediction-", "training-prediction-") for name in checkpoints),
                "all fifteen checkpoints and all thirty model predictions")
        require(len(collection_plan["cohort"]) == 90, "all ninety declared paths")
        for phase_name, receipt, script, cap in (("collection", collection, "scripts/collect_otto_protected_readout.py", 7200),
                                                ("training", training, "scripts/train_otto_protected_readout.py", 14400)):
            plan_path = STUDY / f"{phase_name}-plan-01.json"
            terminal = supervisor("worker", STUDY / f"{phase_name}-supervisor-01", receipt, script, cap,
                {"--plan": str(plan_path), "--plan-sha256": descriptor(plan_path)["sha256"],
                 "--output": str(STUDY / f"{phase_name}-01")}, mode="run",
                executable=".venv-otto-released-native/bin/python" if phase_name == "collection" else ".venv/bin/python")
            started = read(STUDY / f"{phase_name}-01/started.json")
            require(started["launch"] == read(STUDY / f"{phase_name}-supervisor-01.launch.json")
                    and started["started_ns"] == receipt["started_ns"], "original worker start witness")
        expected = {"--output": str(STUDY / "audit-01")}
        for role, value in audit["producer_inputs"].items():
            bind(value["path"], value)
            expected["--" + role], expected["--" + role + "-sha256"] = value["path"], value["sha256"]
        supervisor("audit", STUDY / "audit-supervisor-01", audit,
                   "scripts/audit_otto_protected_readout.py", 300, expected)
        require(read(STUDY / "audit-01/started.json")["launch"] == read(STUDY / "audit-supervisor-01.launch.json"),
                "original audit start witness")
        # Preserve every current engineering attempt and all prepublication metadata.
        for path in sorted(STUDY.iterdir()):
            if path.name in ("collection-01", "training-01", "audit-01"):
                continue
            if path.is_dir():
                add_tree("audit" if "engineering" in path.name else "worker", path)
            elif path.is_file():
                add("audit" if path.name.startswith("audit-") else "worker", path)
        add_tree("audit", REPORT)
        visual = read(REPORT / "visual-review-01.json")
        require(visual["status"] == "passed" and visual["report_receipt_sha256"] == REPORT_PIN and len(visual["files"]) == 3,
                "root visual review of the closed report")
        for path, pin in visual["files"].items():
            bind(path, pin)
        for path, pin in report_receipt["inputs"].items():
            bind(path, pin)
        # Include actual prospectively pinned sources, including historical source witnesses.
        source_rows = {}
        for owner, mapping in (("collection-plan", collection_plan["sources"]), ("training-plan", train_plan["sources"]),
                               ("audit-receipt", audit["sources"])):
            for name, pin in mapping.items():
                path = add("audit", name, pin)
                record = source_rows.setdefault(path.relative_to(ROOT).as_posix(), {**descriptor(path), "owners": []})
                record["owners"].append(owner)
        for name in ("scripts/report_otto_protected_readout.py", "scripts/plot_otto_action_focused.py",
                     "scripts/plot_otto_prequery_calibration.py", "tests/test_report_otto_protected_readout.py"):
            path = add("audit", name)
            source_rows.setdefault(name, {**descriptor(path), "owners": ["qualified-presentation"]})
        for owner, plan in (("collection-plan", collection_plan), ("training-plan", train_plan)):
            for section in ("inputs", "native_inputs"):
                for role, value in plan.get(section, {}).items():
                    dependency(owner, section, role, value)
        # The saved audit authenticates the original historical capacity phase.
        capacity_path = canonical(train_plan["inputs"]["capacity_receipt"]["path"])
        phase("audit", capacity_path.parent, 5)
        terminal_path = canonical(train_plan["inputs"]["capacity_terminal"]["path"])
        prefix = terminal_path.name.removesuffix(".terminal.json")
        for path in sorted(terminal_path.parent.glob(prefix + ".*")):
            add("audit", path)
        # Preserve official retrieval/extraction provenance, without decoding model artifacts.
        for name in ("tmp/otto-pretrained-reference-01/receipt.json", "output/otto-pretrained-reference-v1/extraction-01/receipt.json",
                     "tmp/otto-pretrained-reference-01/zoo_model_2_3_2.config", "tmp/otto-source-review-01/LICENSE"):
            add("audit", name)
        official_license = OUT / "OTTO-UPSTREAM-LICENSE.txt"
        with urllib.request.urlopen(OFFICIAL_BASE + "/LICENSE", timeout=30) as response:
            content = response.read(65537)
        require(len(content) <= 65536 and content.startswith(b"MIT License")
                and b"Copyright (c) 2022 by Christophe Eloy and Aurore Loisy" in content, "exact upstream MIT notice")
        with official_license.open("xb") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        provenance = save("third-party-provenance.json", {
            "official_teacher": {"repository": "https://github.com/C0PEP0D/otto", "commit": OFFICIAL_COMMIT,
                "original_path": "zoo/models/zoo_model_2_3_2/zoo_model_2_3_2", "url": WEIGHT_URL,
                "sha256": WEIGHT_PIN, "bytes": 53581916, "license": "MIT",
                "license_url": OFFICIAL_BASE + "/LICENSE", "license_file": official_license.relative_to(ROOT).as_posix(),
                "delivery": "Immutable upstream dependency; original weight bytes are not duplicated in this package.",
                "local_weight_path": WEIGHT_PATH.relative_to(ROOT).as_posix(),
                "derived_tensor_path": "output/otto-pretrained-reference-v1/extraction-01/tensors.npz"},
            "native_source": {"repository": "https://github.com/auroreloisy/otto-benchmark",
                "commit": "a6aaef6507cffd2aff79291c1019f506f616bbef", "license": "MIT",
                "copyright": "2023 Aurore Loisy", "license_file": "tmp/otto-source-review-01/LICENSE"},
            "scope": "Original third-party notices retained; OpenJev does not claim authorship of the teacher or native environment."})
        source_manifest = save("source-manifest.json", {"files": dict(sorted(source_rows.items())),
            "repository_head_at_packaging": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scope": "Exact registered source bytes are bundled; commit identity does not replace file hashes."})
        dependency_manifest = save("historical-dependencies.json", {"dependencies": DEPS,
            "scope": "All directly registered inputs are bundled except the byte-identical official teacher weights, supplied by immutable upstream URL. "
                     "Historical plans and receipts retain their transitive descriptors unchanged. This package does not rebundle every historical experiment. "
                     "The complete historical capacity phase is included because the current saved audit reads it.",
            "original_record_paths_rewritten": False,
            "restore": "Archive member paths are relative to the original repository root. Original absolute paths inside records remain unchanged; "
                       "a different checkout location does not automatically satisfy the original command/path checks."})
        for path in (official_license, provenance, source_manifest, dependency_manifest, Path(__file__), OUT / "started.json"):
            add("audit", path)
        # A manifest in each archive supplies exact members of both disjoint payload sets.
        for name in list(BUNDLES["worker"]):
            BUNDLES["audit"].pop(name, None)
        manifests = {}
        for bundle in ("worker", "audit"):
            manifests[bundle] = save(f"{bundle}-evidence-manifest.json", {"version": "otto-protected-readout-release-v1",
                "bundle": bundle, "files": dict(sorted(BUNDLES[bundle].items())), "scientific_array_decodes": 0,
                "scope": "Original immutable evidence bytes. Technical closure is distinct from the scientific acceptance gate.",
                "companion_archives_required": True,
                "historical_dependencies": dependency_manifest.relative_to(ROOT).as_posix(),
                "source_manifest": source_manifest.relative_to(ROOT).as_posix()})
        results = {bundle: archive(bundle, manifests[bundle]) for bundle in ("worker", "audit")}
        for name, pin in PINS.items():
            require(descriptor(ROOT / name) == pin, "unchanged original bytes after roundtrip: " + name)
        sums = {Path(result["path"]).name: {k: result[k] for k in ("sha256", "bytes")} for result in results.values()}
        for path in (*manifests.values(), source_manifest, dependency_manifest, provenance, official_license):
            sums[path.name] = descriptor(path)
        with (OUT / "SHA256SUMS").open("x") as stream:
            for name, pin in sorted(sums.items()):
                stream.write(f"{pin['sha256']}  {name}\n")
            stream.flush(); os.fsync(stream.fileno())
        receipt = {"status": "passed", "version": "otto-protected-readout-package-v1", "archives": results,
            "report_receipt_sha256": REPORT_PIN, "source_files": len(source_rows), "registered_dependency_edges": len(DEPS),
            "collection_payloads": 18, "training_payloads": 58, "audit_payloads": 2, "report_payloads": 13,
            "collection_trajectories": 90, "trained_checkpoints": 15, "saved_prediction_files": 31,
            "array_decodes": 0, "model_calls": 0, "optimizer_calls": 0, "native_calls": 0,
            "original_records_rewritten": False, "git_mutations": 0, "uploads": 0,
            "checksums": descriptor(OUT / "SHA256SUMS"), "wall_seconds": time.monotonic() - START,
            "files": {p.name: descriptor(p) for p in sorted(OUT.iterdir()) if p.is_file()}}
        save("byte-roundtrip-verification.json", receipt)
        print(json.dumps({"status": "passed", "archives": results,
                          "verification_receipt": descriptor(OUT / "byte-roundtrip-verification.json")}, indent=2), flush=True)
    except BaseException as error:
        save("failed.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                            "wall_seconds": time.monotonic() - START, "arrays_decoded": 0})
        raise


if __name__ == "__main__":
    main()
