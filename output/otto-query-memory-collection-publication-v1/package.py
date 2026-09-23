"""Package authenticated collection bytes without decoding arrays or journals."""
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
STUDY = "output/otto-query-memory-v1"
PINS = {
    f"{STUDY}/collection-plan-01.json": "624b7fcebcdaa4df9d3b2079127ec40b9a0e3e043864034fd7869296cc1a2645",
    f"{STUDY}/collection-01/receipt.json": "8ae3f84ef8669e69a88edb5dfed875aeeb66a9fac6702e9e68ae1c8910da9fa4",
    f"{STUDY}/collection-native-01.launch.json": "bb8d3f24b7ee43b2496b724e455e145e9aec063e904c86130f030e4c2af8e7c4",
    f"{STUDY}/collection-native-01.terminal.json": "9852c7fa2f2b8645170f9794efd43766a7ad3078cf5979af016d088cb97bbbff",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def pin_bytes(payload):
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def payload(name):
    path = ROOT / name
    require(not Path(name).is_absolute() and ".." not in Path(name).parts
            and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "regular repository file required")
    return path.read_bytes()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main():
    output = BASE / "attempt-01"
    output.mkdir(exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    members = {name: payload(name) for name in PINS}
    for name, digest in PINS.items():
        require(pin_bytes(members[name])["sha256"] == digest, "original collection metadata pin")
    plan = json.loads(members[f"{STUDY}/collection-plan-01.json"])
    receipt = json.loads(members[f"{STUDY}/collection-01/receipt.json"])
    terminal = json.loads(members[f"{STUDY}/collection-native-01.terminal.json"])
    require(receipt["complete"] is True and receipt["completed_episodes"] == 108
            and len(receipt["files"]) == 17, "completed exact collection roster")
    require(terminal["status"] == "completed" and type(terminal["returncode"]) is int
            and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["group_absent"] is True and terminal["cleanup"]["reaped"] is True
            and terminal["cleanup"]["group_absent"] is True and terminal["cleanup"]["errors"] == []
            and terminal["error"] is terminal["clock_error"] is None, "original collection closure")
    for name, pin in receipt["files"].items():
        require(Path(name).name == name, "simple closed collection payload name")
        relative = f"{STUDY}/collection-01/{name}"
        members[relative] = payload(relative)
        require(pin_bytes(members[relative]) == pin, "unchanged opaque collection payload")
    for name, digest in plan["sources"].items():
        members[name] = payload(name)
        require(pin_bytes(members[name])["sha256"] == digest, "frozen collection source identity")
    for pin in plan["inputs"].values():
        members[pin["path"]] = payload(pin["path"])
        require(pin_bytes(members[pin["path"]]) == {k: pin[k] for k in ("bytes", "sha256")},
                "original direct collection admission input")
    additional = [
        f"{STUDY}/collection-native-01.detached-intent.json",
        f"{STUDY}/collection-native-01.detached-launch.json",
        f"{STUDY}/collection-launch-command-01.json",
        "LICENSE", "third_party/otto/README.md", "third_party/otto/LICENSE",
        "third_party/otto/LICENSE-zoo", "research/otto-learned-reference-and-symmetry.md",
        "output/otto-query-memory-study-engineering-v1/attempt-02/started.json",
        "output/otto-query-memory-study-engineering-v1/attempt-02/command-1.log",
        "output/otto-query-memory-study-engineering-v1/attempt-02/command-2.log",
    ]
    for name in additional:
        members[name] = payload(name)
    members["COLLECTION-README.md"] = (BASE / "COLLECTION-README.md").read_bytes()
    members["DEPENDENCIES.json"] = encoded({
        "repository": "https://github.com/kw2828/OpenJev", "repository_commit": commit,
        "included_source_records": plan["sources"], "included_direct_inputs": plan["inputs"],
        "external_native_inputs_not_bundled": plan["native_inputs"], "runtime": plan["runtime"],
        "scope": "collection evidence archive, not a self-contained native environment or rerun",
        "upstream_provenance": "research/otto-learned-reference-and-symmetry.md",
        "absolute_paths": "original receipts retain original paths; relocation requires a distinct reproduction plan",
    })
    manifest = {
        "kind": "opaque_collection_evidence", "collection_payloads": 17,
        "collection_payload_bytes": sum(p["bytes"] for p in receipt["files"].values()),
        "collection_source_records": len(plan["sources"]), "repository_commit": commit,
        "files": {name: pin_bytes(value) for name, value in sorted(members.items())},
        "numerical_decodes": 0, "trajectory_journal_parses": 0, "scientific_calls": 0,
    }
    members["MANIFEST.json"] = encoded(manifest)
    archive = output / "otto-query-memory-collection-v1.tar.gz"
    with archive.open("xb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
    verified = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            require(member.isfile() and member.name not in verified and member.name in members,
                    "exact regular unique archive member")
            data = tar.extractfile(member).read()
            require(data == members[member.name], "opaque byte-identical archive roundtrip")
            verified[member.name] = pin_bytes(data)
    require(set(verified) == set(members), "complete archive roundtrip")
    for name, pin in manifest["files"].items():
        if name not in ("DEPENDENCIES.json", "COLLECTION-README.md"):
            require(pin_bytes(payload(name)) == pin, "source remained unchanged during packaging")
    archive_pin = pin_bytes(archive.read_bytes())
    (output / "MANIFEST.json").write_bytes(members["MANIFEST.json"])
    (output / "DEPENDENCIES.json").write_bytes(members["DEPENDENCIES.json"])
    (output / "SHA256SUMS").write_text(f"{archive_pin['sha256']}  {archive.name}\n")
    result = {"status": "passed", "scope": manifest["kind"], "archive": {"name": archive.name, **archive_pin},
              "member_count": len(members), "all_members_byte_identical": True,
              "collection_payloads": 17, "collection_payload_bytes": manifest["collection_payload_bytes"],
              "source_records": len(plan["sources"]), "numerical_decodes": 0,
              "trajectory_journal_parses": 0, "scientific_calls": 0,
              "test_evaluation_admitted": False, "repository_commit": commit,
              "packager": pin_bytes(Path(__file__).read_bytes())}
    (output / "receipt.json").write_bytes(encoded(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
