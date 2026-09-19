"""Bundle all nine fitted models and results; explicitly excludes raw traces."""

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/reacher-two-observation-study-v1/model-release-v1"
EXECUTION = ROOT / "runs/reacher-two-observation-study-v1/attempt"
AUDIT = ROOT / "evidence/reacher-two-observation-study-v1/audit"
REVIEW = ROOT / "output/reacher-two-observation-study-v1/results-review.json"
EXPECTED_COMPLETED = "cf90efcf33d7f23f4f7777c534c3285ebbb4ef90f604019f96cd8e4c24b63bfc"
EXPECTED_AUDIT = "ce77c492ce9266b0c9cabb6f404ad10f55f8739806204b6bcaf49c21e2fbf24f"
EXPECTED_REVIEW = "a685eeead30dca4caa391b37bbd7760ad5b97a877e95106d51ea0384c3ec2be5"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    assert sha(EXECUTION / "completed.json") == EXPECTED_COMPLETED
    assert sha(AUDIT / "receipt.json") == EXPECTED_AUDIT
    assert sha(REVIEW) == EXPECTED_REVIEW
    done = json.loads((EXECUTION / "completed.json").read_text())
    protocol = ROOT / "evidence/reacher-two-observation-study-v1/protocol"
    plan = json.loads((protocol / "plan.json").read_text())
    audit = json.loads((AUDIT / "receipt.json").read_text())
    assert audit["status"] == "completed" and done["new_fits"] == 3 and done["inherited_fits"] == 6
    members = {}
    for folder in [EXECUTION / "fits", EXECUTION / "inherited/fits"]:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                assert not path.is_symlink()
                assert sha(path) == done["files"][path.relative_to(EXECUTION).as_posix()]
                members[path.relative_to(ROOT).as_posix()] = path
    weight_names = [name for name in members if name.endswith("/weights.pt")]
    assert len(weight_names) == 9
    for name, expected in plan["sources"].items():
        path = ROOT / name
        assert sha(path) == expected
        members[name] = path
    for folder in [protocol, AUDIT, ROOT / "evidence/reacher-two-observation-study-v1/report"]:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                assert not path.is_symlink()
                members[path.relative_to(ROOT).as_posix()] = path
    for path in [EXECUTION / "completed.json", EXECUTION / "results.json", REVIEW,
                 ROOT / "output/reacher-two-observation-study-v1/results-review.md", Path(__file__)]:
        members[path.relative_to(ROOT).as_posix()] = path
    OUT.mkdir(parents=True, exist_ok=False)
    rows = [{"path": name, "bytes": path.stat().st_size, "sha256": sha(path)}
            for name, path in sorted(members.items())]
    write(OUT / "members.json", {"scope": "All nine fitted models, fit logs, frozen source/protocol and audited results",
        "raw_execution_traces_included": False, "models": weight_names, "members": rows,
        "execution_completed_sha256": EXPECTED_COMPLETED, "audit_receipt_sha256": EXPECTED_AUDIT,
        "independent_review_sha256": EXPECTED_REVIEW})
    archive = OUT / "openjev-two-observation-models-and-results.tar.gz"
    with archive.open("xb") as destination, gzip.GzipFile(fileobj=destination, mode="wb", mtime=0,
                                                        filename="", compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as bundle:
            for row in rows:
                path = ROOT / row["path"]
                info = bundle.gettarinfo(str(path), arcname=row["path"])
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with path.open("rb") as stream:
                    bundle.addfile(info, stream)
    expected = {row["path"]: row for row in rows}
    seen = set()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            assert member.isfile() and member.name in expected and member.name not in seen
            row = expected[member.name]
            assert member.size == row["bytes"]
            assert hashlib.file_digest(bundle.extractfile(member), "sha256").hexdigest() == row["sha256"]
            seen.add(member.name)
    assert seen == set(expected)
    write(OUT / "verification.json", {"status": "verified", "all_members_reopened": len(seen),
        "models": 9, "raw_execution_traces_included": False, "new_model_or_native_calls": 0,
        "archive": archive.name, "archive_bytes": archive.stat().st_size, "archive_sha256": sha(archive),
        "members_sha256": sha(OUT / "members.json"), "source_sha256": sha(Path(__file__)),
        "scientific_continuation": "FAIL (24/25), unchanged"})
    print(json.dumps({"archive_bytes": archive.stat().st_size, "members": len(seen), "models": 9}))


if __name__ == "__main__":
    main()
