"""Package complete saved experiment evidence without running the environment."""

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/mystery-path-qualification-v1"
BASE = ROOT / "evidence/mystery-path-qualification-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    bindings = json.loads((BASE / "bindings.json").read_text())
    for name, expected in bindings["sources"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    assert (OUTPUT / "results-review.json").is_file()
    files = {ROOT / name for name in bindings["sources"]}
    files.update(p for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.update(p for p in OUTPUT.rglob("*") if p.is_file() and not any(
        part.startswith(("release-", "publication-", "__pycache__")) for part in p.relative_to(OUTPUT).parts))
    files.update([ROOT / "README.md", ROOT / "research/mystery-path-memory-qualification.md"])
    files = sorted(files)
    release = OUTPUT / "release-v1"
    release.mkdir(exist_ok=False)
    members = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())} for p in files}
    (release / "members.json").write_text(json.dumps(members, indent=2)+"\n")
    archive = release / "openjev-mystery-path-full-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for path in files:
            output.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    with tarfile.open(archive, "r:gz") as saved:
        assert len(saved.getmembers()) == len(members)
        for member in saved.getmembers():
            data = saved.extractfile(member).read()
            assert {"bytes": len(data), "sha256": sha(data)} == members[member.name]
    sums = "".join(f"{sha(p.read_bytes())}  {p.name}\n" for p in (archive, release / "members.json"))
    (release / "SHA256SUMS").write_text(sums)
    verified = {"status": "verified", "members": len(members),
                "raw_bytes": sum(item["bytes"] for item in members.values()),
                "archive_bytes": archive.stat().st_size, "archive_sha256": sha(archive.read_bytes()),
                "manifest_sha256": sha((release / "members.json").read_bytes())}
    (release / "verified.json").write_text(json.dumps(verified, indent=2)+"\n")
    print(json.dumps(verified))


if __name__ == "__main__":
    main()
