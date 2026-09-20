"""Synthetic saved artifacts only. No corpus, model or tokenizer execution.

The complete controller fixture substitutes the historical lineage authenticator
with an explicit synthetic parent. All current run manifests, twelve fit joins,
public token work, journals, supervisor, raw arrays and scoring remain real.
Separate tests cover the lineage pin-first boundary and unsafe manifests.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("observation_report", ROOT / "scripts/report_dialogue_observation.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def jsonl(path, values):
    path.write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in values))


def files(path):
    return {p.relative_to(path).as_posix(): r.descriptor(p) for p in path.rglob("*")
            if p.is_file() and p != path / "completed.json"}


def invariant(profiles):
    return {**r.state_counts(profiles), "mass_above_one_count": 0,
            **dict.fromkeys(r.MAX_KEYS, 0.), "mass_min": .1, "mass_max": .6, "tolerance": 2e-6}


def actor(did, split, qi):
    return {"split": split, "dialogue_id": did, "tokens": [[7] * 255] + [[8 + i] for i in range(7)],
            "original_feature_ids": list(range(8)), "turn_text_ids": [0, 1, 2], "query_text_ids": [3],
            "candidate_text_ids": [[4, 5, 6, 7]], "candidate_ids": [[r.metrics.NONE, r.metrics.DONTCARE, "value:True", "value:False"]],
            "query_ids": [qi], "user_turn_indices": [0, 2, 4], "lexical_shape": [3, 1, 4, 10], "lexical_offset": 0}


def endpoint_records(a, query):
    result = []
    for i, (bin_name, label) in enumerate((("unmentioned_retention", 0), ("first_assignment", 2), ("assigned_retention", 2))):
        stratum = bin_name if bin_name != "first_assignment" else "changed"
        result.append({"split": a["split"], "dialogue_id": a["dialogue_id"], "source_row_index": i,
                       "time": i, "turn_index": a["user_turn_indices"][i], "query_position": 0, "query_index": a["query_ids"][0],
                       "query_id": query["id"], "service": query["service"], "slot": "slot", "label_index": label,
                       "label_id": query["candidate_ids"][label], "bin": bin_name, "stratum": stratum,
                       "stratum_index": r.metrics.STRATA.index(stratum), "unseen": query["service"] == "unseen", "dontcare": False})
    return {"split": a["split"], "dialogue_id": a["dialogue_id"], "rows": result,
            "literal_registers": {"original": [[0, 2, 2]], "numbers": [[0, 2, 2]]}}


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(r, "ROOT", root)
    original_bind = r.bind
    monkeypatch.setattr(r, "bind", lambda mapping, root=root: original_bind(mapping, root))
    monkeypatch.setattr(r, "CONFIG", {**r.CONFIG, "epochs": 2, "effective_batch": 2})
    monkeypatch.setattr(r, "DEV_ENDPOINTS", 6)
    for name in ("scripts/report_dialogue_observation.py", "tests/test_report_dialogue_observation.py",
                 "src/openjev/research/dialogue_observation_metrics.py", "tests/test_dialogue_observation_metrics.py", r.WATCHDOG):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Artificial source identity fixture only\n")
    prepared = root / "prepared"
    prepared.mkdir()
    queries = [{"split": split, "id": "q" + str(i), "service": service, "slot": "slot",
                "candidate_ids": [r.metrics.NONE, r.metrics.DONTCARE, "value:True", "value:False"],
                "candidate_values": [None, None, "True", "False"]}
               for i, (split, service) in enumerate((("train", "seen"), ("dev", "seen"), ("dev", "unseen")))]
    dump(root / "runs/sgd-state-v1/features-02/packet.json", {"queries": queries})
    actors = [actor(did, "train", 0) for did in ("shared", "b", "c")]
    # Same bare ID across TRAIN/DEV is valid.
    actors += [actor("shared", "dev", 1), actor("e", "dev", 2)]
    profiles = { (a["split"], a["dialogue_id"]): r.actor_work(a) for a in actors }
    records = [endpoint_records(a, queries[a["query_ids"][0]]) for a in actors]
    for split in ("train", "dev"):
        jsonl(prepared / f"actors-{split}.jsonl", [a for a in actors if a["split"] == split])
        jsonl(prepared / f"targets-{split}.jsonl", [a for a in records if a["split"] == split])
    dump(prepared / "workloads.json", {"profiles": [{"split": s, "dialogue_id": d, "work": p} for (s, d), p in profiles.items()]})
    orders = {"seeds": list(r.metrics.SEEDS), "epochs": 2, "dialogue_ids": ["shared", "b", "c"],
              "orders": {str(seed): [[0, 1, 2], [2, 0, 1]] for seed in r.metrics.SEEDS}}
    dump(prepared / "orders.json", orders)
    parent = {"cohort_sizes": {"train": 3, "dev": 2}, "loss_counts": {"train": {"0": 3, "1": 3, "2": 3}, "dev": {"0": 2, "1": 2, "2": 2}},
              "loss_weights": [1., 1., 1.], "input_sha256": {"runs/sgd-state-v1/features-02/packet.json": r.sha(root / "runs/sgd-state-v1/features-02/packet.json")}}
    meta = r.prepared_metadata(prepared, parent, queries)
    expected = r.expected_work(meta)
    run = root / "run"
    run.mkdir()
    freeze = root / "freeze"
    freeze.mkdir()
    sources = {name: r.sha(root / name) for name in ("scripts/report_dialogue_observation.py", "tests/test_report_dialogue_observation.py",
                "src/openjev/research/dialogue_observation_metrics.py", "tests/test_dialogue_observation_metrics.py", r.WATCHDOG)}
    allocation = {"limits": {"wall_seconds": 100, "rss_bytes": 10**9, "output_bytes": 10**8, "mps_driver_bytes": 10**9}}
    plan = {"expected": expected, "prepared": str(prepared), "source_sha256": sources, "allocation": allocation}
    dump(freeze / "plan.json", plan)
    dump(run / "plan.json", plan)
    dump(run / "allocation.json", allocation)
    jsonl(run / "evaluation-rows.jsonl", meta["rows"])
    dump(run / "references.json", meta["references"])
    args = SimpleNamespace(run=run, completed_sha256="", plan=freeze / "plan.json", plan_sha256=r.sha(freeze / "plan.json"),
                           out=root / "report", launch=root / "launch.json", launch_sha256="",
                           terminal=root / "terminal.json", terminal_sha256="")
    dump(run / "started.json", {"request": {"out": str(run)}})
    fit_records = []
    for name in r.FIT_ORDER:
        arm, seed = name.rsplit("-", 1)
        directory = run / name
        directory.mkdir()
        events = []
        for epoch, order in enumerate(orders["orders"][seed]):
            for start in range(0, len(order), 2):
                ids = [orders["dialogue_ids"][i] for i in order[start:start + 2]]
                batch = [profiles[("train", did)] for did in ids]
                events.append({"epoch": epoch, "batch_start": start, "dialogue_ids": ids, "microbatches": len(ids), "endpoint_count": 3 * len(ids),
                               "weighted_training_loss": .3, "encoder_work": r.work_sum(batch), "invariants": invariant(batch),
                               "wall_seconds": .01, "memory_gradient_squared_norm": .1, "gradient_norm_before_clip": .8,
                               "encoder_gradient_squared_norms": dict.fromkeys(r.GRADIENT_KEYS, .2 if arm.startswith("trainable") else None)})
        jsonl(directory / "updates.jsonl", events)
        logs = np.full((6, 12), -np.inf, np.float32)
        for i, row in enumerate(meta["rows"]):
            probs = [.1] * 4
            probs[row["label_index"]] = .7
            logs[i, :4] = np.log(probs).astype(np.float32)
        np.savez(directory / "predictions.npz", row_indices=np.arange(6, dtype=np.int64), log_probs=logs)
        (directory / "weights.pt").write_bytes(b"opaque synthetic weights, never load")
        counts = {**expected["per_fit"], "encoding_attempts": expected["per_fit"]["encoding_passes"],
                  "memory_forward_attempts": expected["per_fit"]["memory_forwards"], "backward_attempts": expected["per_fit"]["backward_calls"],
                  "optimizer_attempts": expected["per_fit"]["optimizer_updates"], "encoder_forward_attempts": expected["per_fit"]["encoder_calls"],
                  "encoder_forward_returns": expected["per_fit"]["encoder_calls"]}
        dev = [p for (s, _), p in profiles.items() if s == "dev"]
        fit = {"status": "completed", "version": r.STUDY, "fit_id": name, "arm": arm, "seed": int(seed),
               "initial_sha256": {"encoder": "a" * 64, "memory": str(int(seed) - 6900) * 64},
               "final_encoder_sha256": "b" * 64 if arm.startswith("trainable") else "a" * 64,
               "counts": counts, "encoder_work": expected["encoder_work_per_fit"],
               "invariants": r.merge_invariants([*[e["invariants"] for e in events], invariant(dev)]),
               "training_seconds": .1, "evaluation": {"rows": 6, "wall_seconds": .1, "encoder_work": r.work_sum(dev), "invariants": invariant(dev)},
               "checkpoint": {**r.descriptor(directory / "weights.pt"), "wall_seconds": .1}, "wall_seconds": .4,
               "files": files(directory)}
        dump(directory / "completed.json", fit)
        fit_records.append({"fit_id": name, "completed_sha256": r.sha(directory / "completed.json")})
    done = {"version": r.STUDY, "status": "completed", "phase": "train", "fit_count": 12, "fits": fit_records,
            "plan_sha256": args.plan_sha256, "quality_scoring_in_runner": False, "official_test_opened": False,
            "external_model_api_calls": 0, "counts": expected["all_fits"], "wall_seconds": 6., "peak_rss_bytes": 100,
            "sampled_mps_driver_max_bytes": 100, "sampled_mps_current_max_bytes": 80, "files": files(run)}
    dump(run / "completed.json", done)
    args.completed_sha256 = r.sha(run / "completed.json")
    command = [sys.executable, "-u", str(root / "scripts/study_dialogue_observation.py"), "train", "--plan", str(args.plan),
               "--plan-sha256", args.plan_sha256, "--out", str(run)]
    dump(args.launch, {"command": command, "pid": 123, "pgid": 123, "started_unix": 100., "cap_seconds": 100,
                       "watchdog_sha256": sources[r.WATCHDOG]})
    dump(args.terminal, {"command": command, "pgid": 123, "returncode": 0, "timed_out": False, "error": None,
                         "group_absent": True, "wall_seconds": 7., "finished_unix": 106.5})
    args.launch_sha256, args.terminal_sha256 = r.sha(args.launch), r.sha(args.terminal)
    monkeypatch.setattr(r, "authenticate_lineage", lambda *_: (plan, parent, {"synthetic_parent": True}))
    return SimpleNamespace(args=args, meta=meta, plan=plan, parent=parent, expected=expected, done=done, root=root)


def reseal(study):
    run = study.args.run
    for item in study.done["fits"]:
        path = run / item["fit_id"] / "completed.json"
        fit = r.read(path)
        fit["files"] = files(path.parent)
        if "checkpoint" in fit:
            fit["checkpoint"].update(fit["files"]["weights.pt"])
        dump(path, fit)
        item["completed_sha256"] = r.sha(path)
    study.done["files"] = files(run)
    dump(run / "completed.json", study.done)
    study.args.completed_sha256 = r.sha(run / "completed.json")


def test_complete_synthetic_saved_controller_and_report(study):
    receipt = r.execute(study.args)
    assert receipt["technical_validity_passed"] is True and receipt["scientific_checks_total"] == 7
    assert receipt["scientific_checks_passed"] == 5 and receipt["continuation_passed"] is False
    summary = r.read(study.args.out / "summary.json")
    assert len(summary["fits"]) == 12 and len(summary["factorial"]["contrasts"]) == 4
    assert summary["fits"][r.FIT_ORDER[0]]["panels"]["unseen"]["macro_three"]["accuracy"] == 1
    assert summary["fits"][r.FIT_ORDER[0]]["panels"]["all"]["micro"]["nll"] == pytest.approx(-math.log(.7), abs=1e-7)
    assert summary["references"]["original"]["panels"]["all"]["micro"]["accuracy"] == 1
    assert summary["costs"]["training_seconds"] == pytest.approx(1.2)
    assert summary["costs"]["execution_wall_seconds"] == 6
    assert "FAIL (5/7)" in (study.args.out / "report.md").read_text()
    r.manifest(study.args.out, receipt["files"], terminal="receipt.json")
    with pytest.raises(FileExistsError):
        r.execute(study.args)


@pytest.mark.parametrize("mutation", ["missing_fit", "reordered_fit", "payload_corruption", "foreign_payload"])
def test_invalid_execution_fails_before_predictions_and_preserves_receipt(study, monkeypatch, mutation):
    if mutation == "missing_fit":
        study.done["fits"].pop()
        dump(study.args.run / "completed.json", study.done)
        study.args.completed_sha256 = r.sha(study.args.run / "completed.json")
    elif mutation == "reordered_fit":
        study.done["fits"][0], study.done["fits"][1] = study.done["fits"][1], study.done["fits"][0]
        dump(study.args.run / "completed.json", study.done)
        study.args.completed_sha256 = r.sha(study.args.run / "completed.json")
    elif mutation == "payload_corruption":
        (study.args.run / r.FIT_ORDER[0] / "weights.pt").write_bytes(b"changed")
    else:
        (study.args.run / "unexpected.txt").write_text("foreign")
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Prediction decode before authentication"))
    with pytest.raises(ValueError):
        r.execute(study.args)
    failure = r.read(study.args.out / "failed.json")
    assert failure["scientific_result_qualified"] is False
    assert not (study.args.out / "summary.json").exists()


@pytest.mark.parametrize("mutation", ["order", "denominator", "microbatch", "encoder_work", "unscored_update", "gradient", "initial_pair"])
def test_self_consistent_resigned_receipts_still_reject_missing_actual_coverage(study, mutation):
    name = r.FIT_ORDER[1]
    directory = study.args.run / name
    if mutation == "initial_pair":
        fit = r.read(directory / "completed.json")
        fit["initial_sha256"]["memory"] = "e" * 64
        dump(directory / "completed.json", fit)
    else:
        rows = list(r.lines(directory / "updates.jsonl"))
        if mutation == "order": rows[0]["dialogue_ids"].reverse()
        elif mutation == "denominator": rows[0]["endpoint_count"] -= 1
        elif mutation == "microbatch": rows[0]["microbatches"] -= 1
        elif mutation == "encoder_work": rows[0]["encoder_work"]["encoder_sequences"] -= 1
        elif mutation == "unscored_update": rows[0]["invariants"]["real_question_updates"] -= 1
        else: rows[0]["encoder_gradient_squared_norms"][next(iter(r.GRADIENT_KEYS))] = .1
        jsonl(directory / "updates.jsonl", rows)
    reseal(study)
    with pytest.raises(ValueError):
        r.execute(study.args)
    assert not (study.args.out / "summary.json").exists()


@pytest.mark.parametrize("mutation", ["row", "supported_infinity", "padding", "mass"])
def test_authenticated_prediction_arrays_are_still_validated(study, mutation):
    path = study.args.run / r.FIT_ORDER[-1] / "predictions.npz"
    with np.load(path, allow_pickle=False) as data:
        rows, logs = data["row_indices"], data["log_probs"]
    if mutation == "row": rows = rows[::-1]
    elif mutation == "supported_infinity": logs[0, 1] = -np.inf
    elif mutation == "padding": logs[0, 4] = -1000.
    else: logs[0, :4] -= .1
    np.savez(path, row_indices=rows, log_probs=logs)
    reseal(study)
    with pytest.raises(ValueError):
        r.execute(study.args)
    assert r.read(study.args.out / "failed.json")["model_calls"] == 0


def test_public_token_geometry_counts_long_chunks_and_unscored_turns():
    a = actor("x", "train", 0)
    work = r.actor_work(a)
    assert work["encoder_sequences"] == 9 and work["content_tokens"] == 262
    assert work["padded_token_positions"] == 9 * 256
    assert work["padded_attention_positions"] == 9 * 256**2
    assert work["real_question_updates"] == 3 and work["truncated_tokens"] == 0
    # Work does not accept a supervised endpoint mask as a substitute for public time.
    a["turn_text_ids"].append(1)
    a["lexical_shape"][0] = 4
    assert r.actor_work(a)["real_question_updates"] == 4


@pytest.mark.parametrize("mutation", ["timeout", "group", "command", "cap", "source", "elapsed"])
def test_supervisor_invalidity_precedes_array_loading(study, monkeypatch, mutation):
    launch, end = r.read(study.args.launch), r.read(study.args.terminal)
    if mutation == "timeout": end["timed_out"] = True
    elif mutation == "group": end["group_absent"] = False
    elif mutation == "command": end["command"][-1] = "elsewhere"
    elif mutation == "cap": launch["cap_seconds"] += 1
    elif mutation == "source": launch["watchdog_sha256"] = "e" * 64
    else: end["wall_seconds"] = 5
    dump(study.args.launch, launch)
    dump(study.args.terminal, end)
    study.args.launch_sha256, study.args.terminal_sha256 = r.sha(study.args.launch), r.sha(study.args.terminal)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("No prediction read on invalid supervision"))
    with pytest.raises(ValueError):
        r.execute(study.args)


def test_manifest_rejects_symlink(tmp_path):
    source = tmp_path / "opaque"
    source.write_text("opaque")
    directory = tmp_path / "artifacts"
    directory.mkdir()
    (directory / "alias").symlink_to(source)
    with pytest.raises(ValueError, match="symbolic"):
        r.manifest(directory, {"alias": r.descriptor(source)})


def test_untrusted_lineage_pin_is_checked_before_decode(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text("not JSON")
    monkeypatch.setattr(r, "read", lambda *_: pytest.fail("Untrusted plan decoded"))
    with pytest.raises(ValueError, match="External frozen"):
        r.authenticate_lineage(tmp_path, {"plan_sha256": "0" * 64}, plan, "0" * 64)


def test_original_exception_survives_secondary_failure_write(study, monkeypatch):
    original = RuntimeError("first failure")
    monkeypatch.setattr(r, "authenticate_run", lambda *_: (_ for _ in ()).throw(original))
    write = r.write
    def fail_receipt(path, value):
        if path.name == "failed.json":
            raise OSError("disk error")
        return write(path, value)
    monkeypatch.setattr(r, "write", fail_receipt)
    with pytest.raises(RuntimeError, match="first failure") as exc:
        r.execute(study.args)
    assert exc.value is original and "disk error" in " ".join(original.__notes__)
