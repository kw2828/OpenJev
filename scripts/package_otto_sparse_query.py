"""Package an externally pinned, exact saved-evidence inventory; no science calls."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import signal
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-sparse-query-package-v1"
LIMITS = {"seconds": 300, "rss_bytes": 1024**3, "output_bytes": 1024**3}
NAME = "openjev-otto-sparse-query-v1.tar.gz"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contained(name):
    value = Path(name)
    require(not value.is_absolute() and ".." not in value.parts, "relative member path")
    value = ROOT / value
    require(value.is_file() and not any(p.is_symlink() for p in (value, *value.parents)), "regular nonlink member")
    return value


def digest(path, check):
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            check()
            size += len(block)
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": size}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute(args):
    start = time.monotonic_ns()
    out = args.output
    require(out.is_absolute() and out.is_relative_to(ROOT) and ".." not in out.parts
            and not any(p.is_symlink() for p in out.parents), "contained exclusive output")
    out.mkdir(exist_ok=False)
    receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
               "model_calls": 0, "native_calls": 0, "array_decodes": 0, "optimizer_calls": 0}

    def check():
        require(time.monotonic_ns() - start < LIMITS["seconds"] * 10**9, "package deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in out.iterdir())
                <= LIMITS["output_bytes"], "package RSS/output cap")

    def interrupt(_signum, _frame):
        raise InterruptedError("package hard time cap")

    try:
        signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["seconds"])
        require(digest(args.manifest, check)["sha256"] == args.manifest_sha256, "external inventory SHA")
        inventory = json.loads(args.manifest.read_text())
        require(inventory["version"] == VERSION and inventory["status"] == "prepared_not_archived"
                and inventory["limits"] == LIMITS, "exact packaging scope")
        members = inventory["members"]
        require(str(Path(__file__).resolve().relative_to(ROOT)) in members, "packager source included")
        require(sum(d["bytes"] + 4096 for d in members.values()) + 16 * 1024**2 < LIMITS["output_bytes"],
                "conservative raw archive/output headroom")
        for name, expected in members.items():
            require(digest(contained(name), check) == expected, "unchanged original member")
        for rule in inventory["completion_checks"]:
            require(rule["path"] in members, "completion prerequisite is hash-bound archive member")
            record = json.loads(contained(rule["path"]).read_text())
            require(all(record[k] == v and type(record[k]) is type(v) for k, v in rule["equals"].items()),
                    "successful original receipt/terminal prerequisites")
        with (out / NAME).open("xb") as raw:
            with (
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed,
                tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive,
            ):
                for name, expected in sorted(members.items()):
                    check()
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = expected["bytes"], 0o644, 0
                    with contained(name).open("rb") as source:
                        archive.addfile(info, source)
            raw.flush()
            os.fsync(raw.fileno())
        verified = set()
        with gzip.open(out / NAME, "rb") as decoded:
            with tarfile.open(fileobj=decoded, mode="r|") as archive:
                for entry in archive:
                    require(entry.isfile() and entry.name in members and entry.name not in verified,
                            "exact unique regular archive member")
                    h, size = hashlib.sha256(), 0
                    with archive.extractfile(entry) as stream:
                        for block in iter(lambda: stream.read(1024**2), b""):
                            check()
                            size += len(block)
                            h.update(block)
                    require({"sha256": h.hexdigest(), "bytes": size} == members[entry.name], "archive byte readback")
                    verified.add(entry.name)
            while decoded.read(1024**2):
                check()
        require(verified == set(members), "complete archive coverage and gzip trailer")
        for name, expected in members.items():
            require(digest(contained(name), check) == expected, "original bytes unchanged after packaging")
        require(digest(args.manifest, check)["sha256"] == args.manifest_sha256, "unchanged external inventory")
        packed = digest(out / NAME, check)
        write(out / "manifest.json", {**inventory, "status": "archived_verified", "archive": {"name": NAME, **packed},
            "inventory_sha256": args.manifest_sha256, "member_count": len(members)})
        with (out / "RESTORE.md").open("x") as stream:
            stream.write(inventory["restore_text"])
            stream.flush()
            os.fsync(stream.fileno())
        files = {p.name: digest(p, check) for p in out.iterdir()}
        with (out / "SHA256SUMS").open("x") as stream:
            for name, descriptor in sorted(files.items()):
                stream.write(descriptor["sha256"] + "  " + name + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        files["SHA256SUMS"] = digest(out / "SHA256SUMS", check)
        finish = time.monotonic_ns()
        receipt.update(status="completed", files=files, inventory_sha256=args.manifest_sha256,
            verified_members=len(members), original_bytes=sum(d["bytes"] for d in members.values()),
            started_ns=start, finished_ns=finish, wall_seconds=(finish-start)/1e9)
        write(out / "receipt.json", receipt)
        check()
        print(json.dumps({"status": "completed", "receipt": digest(out / "receipt.json", check)}), flush=True)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(status="failed", error=repr(error))
        try:
            if (out / "receipt.json").exists():
                (out / "receipt.json").rename(out / "receipt.invalid.json")
            write(out / "receipt.json", receipt)
        except BaseException as secondary:  # noqa: BLE001 - preserve packaging failure and original artifacts
            error.add_note(f"Failure publication: {secondary!r}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
