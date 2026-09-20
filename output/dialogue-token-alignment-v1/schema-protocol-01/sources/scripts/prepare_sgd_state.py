"""Fetch pinned SGD train/dev only, then prepare text-only categorical packets."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from openjev.research.dialogue_state_data import (
    BINS,
    DONTCARE,
    NOT_MENTIONED,
    compile_schema,
    parse_dialogue,
    require,
    text_identity,
)

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "google-research-datasets/dstc8-schema-guided-dialogue"
COMMIT = "e852981ae34990f4358979625854259302feaa78"
DEFAULT_SOURCE = ROOT / "runs/sgd-state-v1/source/all"
DEFAULT_DATA = ROOT / "runs/sgd-state-v1/data"
IMPLEMENTATION = ("scripts/prepare_sgd_state.py", "src/openjev/research/dialogue_state_data.py",
                  "tests/test_dialogue_state_data.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def download(url):
    return subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--max-time", "120", url])


def git_blob(payload):
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def fetch(source, *, reuse=ROOT / "runs/sgd-state-v1/source", workers=8):
    require(type(workers) is int and 1 <= workers <= 8, "Download concurrency must be 1..8")
    source.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    files, failures = {}, []
    try:
        url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
        tree_bytes = download(url)
        tree = json.loads(tree_bytes)
        require(not tree.get("truncated", True), "Incomplete repository tree")
        entries = {x["path"]: x for x in tree["tree"] if x["type"] == "blob"
                   and x["path"].split("/")[0] in ("train", "dev")
                   and (x["path"].endswith("/schema.json") or "/dialogues_" in x["path"])}
        require(set(entries) == {"train/schema.json", "dev/schema.json"}
                | {f"train/dialogues_{n:03d}.json" for n in range(1, 128)}
                | {f"dev/dialogues_{n:03d}.json" for n in range(1, 21)}, "Pinned train/dev membership")
        write(source / "fetch-started.json", {
            "repository": REPOSITORY, "commit": COMMIT, "utc": datetime.now(UTC).isoformat(),
            "metadata_url": url, "metadata_sha256": hashlib.sha256(tree_bytes).hexdigest(),
            "files": {name: {"git_blob_sha1": x["sha"], "bytes": x["size"]} for name, x in entries.items()},
            "maximum_concurrent_requests": workers, "automatic_retry": False, "test_contents": False})

        def one(name):
            entry = entries[name]
            old = reuse / name
            reused = old.is_file()
            raw_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{name}"
            payload = old.read_bytes() if reused else download(raw_url)
            require(len(payload) == entry["size"] and git_blob(payload) == entry["sha"], f"Git blob mismatch: {name}")
            json.loads(payload)
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            return name, {"sha256": hashlib.sha256(payload).hexdigest(), "git_blob_sha1": entry["sha"],
                          "bytes": len(payload), "url": raw_url, "reused_verified_copy": reused}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(one, name): name for name in sorted(entries)}
            for future in as_completed(pending):
                try:
                    name, record = future.result()
                    files[name] = record
                    if len(files) % 10 == 0 or len(files) == len(entries):
                        print(json.dumps({"fetched_files": len(files), "of": len(entries)}), flush=True)
                except Exception as exc:  # noqa: BLE001 - preserve all concurrent fetch failures
                    failures.append({"path": pending[future], "error_type": type(exc).__name__, "error": str(exc)})
        require(not failures, f"Fetch failures: {failures}")
        write(source / "fetch-receipt.json", {
            "status": "completed", "repository": REPOSITORY, "commit": COMMIT,
            "license": "CC-BY-SA-4.0", "license_url": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/LICENSE.txt",
            "started_sha256": sha(source / "fetch-started.json"), "files": files,
            "file_count": len(files), "bytes": sum(x["bytes"] for x in files.values()),
            "train_dialogue_shards": 127, "dev_dialogue_shards": 20,
            "test_dialogue_contents_accessed": False, "maximum_concurrent_requests": workers,
            "automatic_retry": False, "wall_seconds": time.perf_counter() - started})
    except BaseException as exc:
        try:
            write(source / "failed.json", {"status": "failed", "error_type": type(exc).__name__,
                                          "error": str(exc), "files_completed": files, "failures": failures})
        except BaseException as secondary:  # noqa: BLE001 - retain original exception
            exc.add_note(f"Failure receipt also failed: {secondary}")
        raise


def authenticate_source(source):
    receipt = json.loads((source / "fetch-receipt.json").read_text())
    require(receipt["status"] == "completed" and receipt["commit"] == COMMIT, "Source completion")
    require(sha(source / "fetch-started.json") == receipt["started_sha256"], "Source started digest")
    expected = set(receipt["files"]) | {"fetch-started.json", "fetch-receipt.json"}
    require({str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()} == expected, "Source exact membership")
    require(len(receipt["files"]) == 149 and not any(p.is_symlink() for p in source.rglob("*")), "Source files")
    for name, record in receipt["files"].items():
        path = source / name
        require(Path(name).parts[0] in ("train", "dev") and ".." not in Path(name).parts,
                "Source path")
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == record["sha256"]
                and len(payload) == record["bytes"] and git_blob(payload) == record["git_blob_sha1"],
                f"Source digest: {name}")
    return receipt


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    context = {}
    try:
        source_receipt = authenticate_source(source)
        source_pin = sha(source / "fetch-receipt.json")
        code = {name: sha(ROOT / name) for name in IMPLEMENTATION}
        write(out / "started.json", {"study": "sgd-state-v1", "commit": COMMIT,
                                     "source_receipt_sha256": source_pin, "implementation_sha256": code,
                                     "scope": "categorical conditional state tracking; no model calls or test dialogues"})
        snapshots = out / "implementation"
        for name in IMPLEMENTATION:
            destination = snapshots / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
            require(sha(destination) == code[name], "Implementation snapshot")
        schemas = {split: compile_schema(json.loads((source / split / "schema.json").read_text()))
                   for split in ("train", "dev")}
        write(out / "catalog.json", {split: schemas[split].catalog() for split in schemas})
        train_texts, dev_texts = defaultdict(list), defaultdict(list)
        exclusions, stats = [], {}
        for split in ("train", "dev"):
            schema = schemas[split]
            counts, bins, labels_by_service, labels_by_query = Counter(), Counter(), Counter(), Counter()
            dontcare_bins = Counter()
            seen_ids = set()
            with (out / f"{split}-dialogues.jsonl").open("x") as features, (out / f"{split}-labels.jsonl").open("x") as targets:
                for path in sorted((source / split).glob("dialogues_*.json")):
                    context = {"split": split, "file": str(path)}
                    raw_dialogues = json.loads(path.read_text())
                    require(isinstance(raw_dialogues, list), "Shard must contain list")
                    for raw in raw_dialogues:
                        context["dialogue_id"] = raw.get("dialogue_id")
                        public, labels = parse_dialogue(raw, schema, train_services=set(schemas["train"].services))
                        identifier = public["dialogue_id"]
                        require(identifier not in seen_ids, f"Duplicate dialogue id within {split}: {identifier}")
                        seen_ids.add(identifier)
                        counts["dialogues_read"] += 1
                        identity = text_identity(public)
                        if split == "dev" and identity in train_texts:
                            exclusions.append({"dialogue_id": identifier, "source_file": str(path.relative_to(source)),
                                               "normalized_text_sha256": identity,
                                               "matching_train_dialogue_ids": train_texts[identity]})
                            counts["dialogues_excluded_as_train_duplicates"] += 1
                            continue
                        (train_texts if split == "train" else dev_texts)[identity].append(identifier)
                        counts["dialogues_retained"] += 1
                        counts["turns"] += len(public["turns"])
                        counts["user_turns"] += len(public["user_turns"])
                        counts["query_sequences"] += len({r["query_id"] for r in labels})
                        features.write(json.dumps(public, ensure_ascii=False, separators=(",", ":")) + "\n")
                        for row in labels:
                            counts["labels"] += 1
                            counts["not_mentioned"] += row["label_id"] == NOT_MENTIONED
                            counts["dontcare"] += row["label_id"] == DONTCARE
                            counts["ontology_value"] += row["label_id"].startswith("value:")
                            counts["unseen_service_labels"] += row["unseen_service"]
                            bins[row["bin"]] += 1
                            if row["is_dontcare"]:
                                dontcare_bins[row["bin"]] += 1
                            labels_by_service[row["service"]] += 1
                            labels_by_query[row["query_id"]] += 1
                            targets.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    print(json.dumps({"parsed": str(path.relative_to(source)), "dialogues": counts["dialogues_read"],
                                      "labels": counts["labels"]}), flush=True)
            stats[split] = {"counts": dict(counts), "bins": {b: bins[b] for b in BINS},
                            "dontcare_bins": {b: dontcare_bins[b] for b in BINS},
                            "labels_by_service": dict(labels_by_service), "labels_by_query": dict(labels_by_query),
                            "categorical_slots": len(schema.queries), "services": len(schema.services)}
        duplicates = {"normalization": "Ordered speaker roles + NFKC/casefold/whitespace-normalized text, entire dialogue",
                      "exclusion_rule": "Exclude dev matches to any train text; retain all train; within-split duplicates retained and reported",
                      "excluded_dev": exclusions,
                      "train_duplicate_groups": [v for v in train_texts.values() if len(v) > 1],
                      "dev_duplicate_groups_retained": [v for v in dev_texts.values() if len(v) > 1]}
        write(out / "duplicates.json", duplicates)
        write(out / "stats.json", stats)
        authenticate_source(source)
        require(sha(source / "fetch-receipt.json") == source_pin, "Source receipt changed")
        require(all(sha(ROOT / name) == digest for name, digest in code.items()), "Implementation changed")
        files = {str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(out.rglob("*")) if p.is_file()}
        write(out / "completed.json", {"status": "completed", "study": "sgd-state-v1", "commit": COMMIT,
                                       "source": str(source.resolve()), "source_receipt_sha256": source_pin,
                                       "source_files": source_receipt["files"], "files": files,
                                       "implementation_sha256": code, "stats": stats,
                                       "test_contents_accessed": False, "new_model_calls": 0,
                                       "wall_seconds": time.perf_counter() - started,
                                       "scope": "Preparation only. All labels/bins/unseen flags evaluator-only. "
                                                "Use public_prefix for causal model inputs; full files also contain later utterances. "
                                                "Query service is supplied; this is not service routing or full DST."})
    except BaseException as exc:
        try:
            write(out / "failed.json", {"status": "failed", "context": context,
                                       "error_type": type(exc).__name__, "error": str(exc)})
        except BaseException as secondary:  # noqa: BLE001 - retain original error
            exc.add_note(f"Failure preservation also failed: {secondary}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fetch", "prepare", "all"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.mode in ("fetch", "all"):
        fetch(args.source, workers=args.workers)
    if args.mode in ("prepare", "all"):
        prepare(args.source, args.out)
