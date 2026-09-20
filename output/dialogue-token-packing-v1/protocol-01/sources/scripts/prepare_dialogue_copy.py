"""Prepare causal lexical observations from authenticated public dialogues.

No new neural/encoder calls and no test data. Labels never enter lexical_stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from openjev.research.dialogue_copy_features import FEATURES, lexical_stream

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/prepare_dialogue_copy.py", "src/openjev/research/dialogue_copy_features.py",
           "tests/test_dialogue_copy_features.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def authenticate(folder, external):
    if sha(folder / "completed.json") != external or (folder / "failed.json").exists():
        raise ValueError("Preparation input completion mismatch")
    receipt = json.loads((folder / "completed.json").read_text())
    if receipt["status"] != "completed":
        raise ValueError("Incomplete input")
    for name, entry in receipt["files"].items():
        member = folder / name
        digest = entry["sha256"] if isinstance(entry, dict) else entry
        if not member.resolve().is_relative_to(folder.resolve()) or sha(member) != digest:
            raise ValueError("Input payload mismatch: " + name)
    return receipt


def prepare(packet_path, data_path, out, packet_sha256, data_sha256):
    packet_path, data_path, out = map(Path, (packet_path, data_path, out))
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        authenticate(packet_path, packet_sha256)
        authenticate(data_path, data_sha256)
        sources = {name: sha(ROOT / name) for name in SOURCES}
        write(out / "started.json", {"packet_completed_sha256": packet_sha256,
              "data_completed_sha256": data_sha256, "source_sha256": sources})
        packet = json.loads((packet_path / "packet.json").read_text())
        metadata, pieces, offset, counts = {}, [], 0, {}
        for split in ("train", "dev"):
            cohort = packet["cohorts"][split]
            requested = {d["id"] for d in cohort}
            public = {}
            with (data_path / f"{split}-dialogues.jsonl").open() as f:
                for line in f:
                    d = json.loads(line)
                    if d["dialogue_id"] in requested:
                        public[d["dialogue_id"]] = d
            if set(public) != requested:
                raise ValueError("Missing public dialogue")
            metadata[split] = []
            real_steps, candidate_steps = 0, 0
            for d in cohort:
                raw = public[d["id"]]
                system, user = [], []
                for turn in raw["user_turns"]:
                    ti, si = turn["turn_index"], turn["previous_system_turn_index"]
                    if raw["turns"][ti]["speaker"] != "USER" or (si is not None and si >= ti):
                        raise ValueError("Public turn chronology")
                    user.append(raw["turns"][ti]["utterance"])
                    system.append("" if si is None else raw["turns"][si]["utterance"])
                if user != d["user_text"] or len(user) != len(d["turns"]):
                    raise ValueError("Original encoder/public turn alignment")
                qids = sorted({r["query"] for r in d["queries"]})
                c = max(len(packet["queries"][qi]["candidates"]) for qi in qids)
                values = np.zeros((len(user), len(qids), c, len(FEATURES)), np.float32)
                for q, qi in enumerate(qids):
                    entry = packet["queries"][qi]
                    x, _ = lexical_stream(system, user, entry["candidate_ids"], entry["candidate_values"])
                    values[:, q, :x.shape[1]] = x
                    candidate_steps += len(user) * x.shape[1]
                metadata[split].append({"id": d["id"], "query_ids": qids,
                                        "offset": offset, "shape": list(values.shape)})
                pieces.append(values.reshape(-1))
                offset += values.size
                real_steps += len(user) * len(qids)
            counts[split] = {"dialogues": len(cohort), "unique_questions": sum(len(d["query_ids"]) for d in metadata[split]),
                             "public_question_steps": real_steps, "real_candidate_steps": candidate_steps,
                             "scored_queries": sum(len(d["queries"]) for d in cohort)}
        array = np.concatenate(pieces)
        if array.size != offset or not np.isfinite(array).all() or not np.isin(array, [0., 1.]).all():
            raise ValueError("Invalid lexical payload")
        np.save(out / "lexical.npy", array, allow_pickle=False)
        write(out / "index.json", {"features": list(FEATURES), "cohorts": metadata})
        authenticate(packet_path, packet_sha256)
        authenticate(data_path, data_sha256)
        if any(sha(ROOT / name) != digest for name, digest in sources.items()):
            raise ValueError("Preparation source changed")
        write(out / "completed.json", {"status": "completed", "packet_completed_sha256": packet_sha256,
              "data_completed_sha256": data_sha256, "source_sha256": sources, "features": list(FEATURES),
              "files": {name: sha(out / name) for name in ("started.json", "index.json", "lexical.npy")},
              "counts": counts, "float32_scalars": int(array.size), "bytes": int(array.nbytes),
              "encoder_calls": 0, "neural_calls": 0, "test_contents_accessed": False,
              "wall_seconds": time.perf_counter() - started,
              "scope": "Public strings and supplied schema only. Each question independent; all public prefix turns retained."})
        return sha(out / "completed.json")
    except BaseException as error:
        try:
            write(out / "failed.json", {"status": "failed", "error": str(error), "error_type": type(error).__name__})
        except BaseException as secondary:  # noqa: BLE001 - preserve original exception
            error.add_note("Failure receipt: " + str(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "data", "out", "packet-sha256", "data-sha256"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    print(json.dumps({"completed_sha256": prepare(a.packet, a.data, a.out, a.packet_sha256, a.data_sha256)}))
