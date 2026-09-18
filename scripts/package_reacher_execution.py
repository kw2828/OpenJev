"""Package an already audited execution, checking every archived byte."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(audit, expected, execution, archive, manifest, url):
    receipt_path = audit / "receipt.json"
    if sha(receipt_path) != expected:
        raise ValueError("External audit receipt mismatch")
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "completed" or not receipt["saved_output_only"]:
        raise ValueError("Completed saved-output audit required")
    for name, digest in receipt["files"].items():
        if sha(audit / name) != digest:
            raise ValueError("Audit member changed")
    members = {**receipt["execution_members"], "completed.json": receipt["execution_completed_sha256"]}
    actual = {str(path.relative_to(execution)) for path in execution.rglob("*") if path.is_file()}
    if actual != set(members) or any(path.is_symlink() for path in execution.rglob("*")):
        raise ValueError("Execution membership/symlink mismatch")
    for name, digest in members.items():
        if sha(execution / name) != digest:
            raise ValueError(f"Execution changed: {name}")
    if archive.exists() or manifest.exists():
        raise FileExistsError("Exclusive archive/manifest paths required")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x:gz") as stream:
        for name in sorted(members):
            stream.add(execution / name, arcname=f"execution/{name}", recursive=False)
    with tarfile.open(archive, "r:gz") as stream:
        archived = stream.getmembers()
        if {item.name for item in archived} != {f"execution/{name}" for name in members}:
            raise ValueError("Archive member mismatch")
        for item in archived:
            name = item.name.removeprefix("execution/")
            if not item.isfile() or hashlib.sha256(stream.extractfile(item).read()).hexdigest() != members[name]:
                raise ValueError(f"Archived bytes changed: {name}")
    result = {
        "audit_receipt_sha256": expected,
        "execution_receipt_sha256": receipt["execution_completed_sha256"],
        "packaging_source_sha256": sha(Path(__file__)),
        "archive_name": archive.name, "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size, "member_count": len(members),
        "contents": "Complete original execution, including every fit, prediction, control trajectory and planning record. No new fits or decisions.",
        "url": url,
        "reproduction_note": "Audit reproduction also requires repository source, pinned runtime and any upstream training-source artifacts bound by the protocol. These are not replaced by this archive.",
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.audit, args.expected_audit_receipt_sha256, args.execution,
                             args.archive, args.manifest, args.url), indent=2))
