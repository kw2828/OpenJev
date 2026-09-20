"""Synthetic saved-byte/metric checks only; no models, encoder or corpus calls."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_joint_test", ROOT / "scripts/report_dialogue_joint.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
b = r.base


def metric(hits, n=100):
    return {"count": n, "correct": hits, "accuracy": hits/n if n else None, "nll": .5 if n else None,
            "brier": .25 if n else None, "zero_target_probabilities": 0}


def gate_rows():
    rows = {}
    for arm in r.ARMS:
        joint = arm.startswith("joint")
        for seed in r.SEEDS:
            panels = {}
            for panel in r.PANELS:
                hits = 52 if joint else 50
                if panel == "seen":
                    hits = 49 if joint else 50
                panels[panel] = {"strata": {s: metric(hits) for s in r.v2.old.STRATA},
                    "macro_three": {"accuracy": hits/100, "nll": .5, "brier": .25},
                    "micro": metric(hits), "revision": metric(49 if joint else 50),
                    "subgroups": {"assigned_true": metric(60 if joint else 50),
                                  "assigned_false": metric(0, 0), "dontcare": metric(60 if joint else 50)},
                    "bins": {key: metric(hits) for key in b.BINS}}
            rows[f"{arm}-{seed}"] = {"metrics": panels}
    return rows


def test_exact_all_fifteen_inclusive_boundaries_and_false_is_not_a_pass_requirement():
    result = r.criteria(gate_rows())
    assert result["passed"] and result["checks_passed"] == result["checks_total"] == 15
    assert len({x["name"] for x in result["checks"]}) == 15


@pytest.mark.parametrize("kind", ["true", "dc", "macro", "seen", "revision", "nll", "missing", "pair"])
def test_each_margin_support_and_pair_condition_can_fail_independently(kind):
    rows = gate_rows()
    m = rows["joint_scalar-4101"]["metrics"]
    if kind in ("true", "dc"):
        m["unseen"]["subgroups"]["assigned_true" if kind == "true" else "dontcare"]["correct"] -= 1
    elif kind in ("macro", "seen"):
        m["seen" if kind == "seen" else "unseen"]["strata"]["changed"]["correct"] -= 1
    elif kind == "revision":
        m["unseen"]["revision"]["correct"] -= 1
    elif kind == "nll":
        for seed in r.SEEDS:
            rows[f"joint_scalar-{seed}"]["metrics"]["unseen"]["micro"]["nll"] = math.nextafter(.5, math.inf)
    elif kind == "missing":
        m["unseen"]["subgroups"]["dontcare"] = metric(0, 0)
    else:
        for seed, hits in zip(r.SEEDS, (56, 50, 50), strict=True):
            for item in rows[f"joint_scalar-{seed}"]["metrics"]["unseen"]["strata"].values():
                item["correct"] = hits
    result = r.criteria(rows)
    assert not result["passed"]
    assert all(x["passed"] for x in result["checks"] if x["name"].startswith("readout/"))


def test_infinite_nll_is_not_floored_or_favorably_compared():
    rows = gate_rows()
    rows["independent_scalar-4101"]["metrics"]["unseen"]["micro"]["nll"] = None
    assert not r.criteria(rows)["passed"]
    p = np.array([[0., 1., 0.]], np.float32)
    item = b.score(p, np.array([0]), np.array([True]))
    assert item["nll"] is None and item["zero_target_probabilities"] == 1


def test_missing_fit_fails_without_crashing_and_family_is_equal_seed_mean():
    rows = gate_rows()
    del rows["joint_scalar-4102"]
    assert not r.criteria(rows)["passed"]
    rows = gate_rows()
    for seed, v in zip(r.SEEDS, (.2, .5, .8), strict=True):
        rows[f"joint_scalar-{seed}"]["metrics"]["unseen"]["micro"]["nll"] = v
    family = r.family_metrics(rows)["joint_scalar"]["unseen"]
    assert family["micro"]["nll"] == .5
    assert family["subgroups"]["assigned_false"]["accuracy"] is None


def packet():
    qs, ds = [], []
    for i, unseen in enumerate((False, True)):
        qs.append({"split": "dev", "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:True", "value:FALSE"],
                   "candidate_values": [None, None, "True", "FALSE"], "candidates": [0, 1, 2, 3]})
        ds.append({"id": "synthetic-"+str(i), "turns": [0]*5, "queries": [
            {"query": i, "time": t, "label": label, "bin": kind, "unseen": unseen, "dontcare": label == 1}
            for t, (label, kind) in enumerate(zip((0, 2, 2, 1, 2),
                ("unmentioned_retention", "first_assignment", "assigned_retention", "revision", "revision"), strict=True))]})
    return {"queries": qs, "cohorts": {"train": [copy.deepcopy(ds[0])], "dev": ds}}


def test_boolean_catalog_not_word_substring_and_reserved_labels_are_separate():
    data = packet()
    expected, _ = b.expected_records(data)
    masks = r.subgroup_masks(data, expected)
    assert masks["assigned_true"].sum() == 6 and masks["dontcare"].sum() == 2
    assert masks["assigned_false"].sum() == 0
    data["queries"][1]["candidate_values"][3] = "false friend"
    assert r.subgroup_masks(data, expected)["assigned_true"].sum() == 3


def test_public_work_includes_unscored_and_padded_queries_and_candidates():
    ds = [{"turns": [0]*3, "queries": [{"query": 0}, {"query": 1}]}, {"turns": [0], "queries": [{"query": 0}]}]
    qs = [{"candidate_ids": [0, 1, 2]}, {"candidate_ids": [0, 1, 2, 3]}]
    independent, joint = [r.observation_work(ds, qs, mode) for mode in ("independent", "joint")]
    assert independent["turn_projection_positions"] == 6
    assert joint["turn_projection_positions"] == 48
    assert joint["real_question_updates"] == 7 and joint["real_candidate_updates"] == 24
    assert joint["emitted_bytes"] == 48*384*4
    assert joint["schema_query_projection_positions"] == 4


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def seal(folder, fields):
    done = {**fields, "files": {p.relative_to(folder).as_posix(): {"sha256": b.sha(p), "bytes": p.stat().st_size}
            for p in folder.rglob("*") if p.is_file() and p.name != "completed.json"}}
    # Nested fit completed receipts are payloads too.
    done["files"].update({p.relative_to(folder).as_posix(): {"sha256": b.sha(p), "bytes": p.stat().st_size}
                          for p in folder.rglob("completed.json") if p.parent != folder})
    put(folder / "completed.json", done)
    return done


def witness(head, dialogs):
    inv = r.v2.empty_invariants()
    c = r.v2.actor_counts(dialogs)
    n = c["executed_question_steps"]
    inv.update(forward_calls=1, forward_returned=1, advance_calls=c["advance_calls"], advance_returned=c["advance_calls"],
        valid_turns=sum(len(d["turns"]) for d in dialogs), executed_valid_question_slots=n,
        real_question_updates=c["real_question_steps"], incoming_checks=n, feature_checks=n, result_checks=n,
        incoming_max_sum_error=1e-7, feature_max_sum_error=1e-7, result_max_sum_error=1e-7)
    if head == "scalar":
        inv.update(mass_checks=n, mass_min=.2, mass_max=.4)
    return inv


def build_joint(root, data, packet_dir, lexical):
    folder, cap, raw = [root / n for n in ("joint", "capacity", "raw")]
    for p in (folder, cap, raw):
        p.mkdir()
    (raw / "raw.json").write_text("synthetic only")
    raw_done = seal(raw, {"status": "completed"})
    del raw_done
    offset, cohorts, counts = 0, {}, {}
    indices = []
    for split, dialogs in data["cohorts"].items():
        entries, t, tq, tc = [], 0, 0, 0
        for d in dialogs:
            ids = sorted({q["query"] for q in d["queries"]})
            sizes = [len(data["queries"][q]["candidate_ids"]) for q in ids]
            shape = [len(d["turns"]), len(ids), max(sizes)]
            entries.append({"id": d["id"], "query_ids": ids, "offset": offset, "shape": shape})
            mask = np.broadcast_to(np.arange(max(sizes))[None, :] < np.array(sizes)[:, None], shape)
            indices += np.where(mask, 0, -1).ravel().tolist()
            offset += math.prod(shape)
            t += shape[0]
            tq += shape[0]*shape[1]
            tc += shape[0]*sum(sizes)
        cohorts[split] = entries
        counts[split] = {"dialogues": len(dialogs), "public_turns": t, "public_question_steps": tq, "real_candidate_steps": tc}
    index = {"version": r.CACHE_VERSION, "template": r.TEMPLATE, "embedding_width": 384, "padding_index": -1,
             "unique_texts": 1, "index_count": offset, "cohorts": cohorts, "counts": counts}
    put(folder / "index.json", index)
    np.save(folder / "indices.npy", np.asarray(indices, np.int64))
    emb = np.zeros((1, 384), np.float32)
    emb[0, 0] = 1
    np.save(folder / "embeddings.npy", emb)
    source_map = {n: b.sha(root/n) for n in ("scripts/prepare_dialogue_joint.py", "tests/test_prepare_dialogue_joint.py", "research/dialogue-joint-protocol.md")}
    for n in source_map:
        p = folder / "sources" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((root/n).read_bytes())
    encoder = b.read(packet_dir / "encoder-plan.json")
    (folder / "original-encoder-source.py").write_bytes((packet_dir / "encoder-source.py").read_bytes())
    prep = {"version": r.CACHE_VERSION, "encoder": r.ENCODER, "revision": r.REVISION, "model_files_sha256": encoder["model_files_sha256"],
        "template": r.TEMPLATE, "device": "mps", "dtype": "float32", "chunk_tokens": 254, "batch_size": 128,
        "capacity_unique_texts": 512, "capacity_seconds": 120, "encoding_seconds": 900, "disk_cap_bytes": r.DISK_CAP,
        "source_sha256": source_map, "runtime": {"synthetic": True}, "original_encoder_source_sha256": encoder["source_sha256"],
        "inputs": {name: {"path": p.name, "completed_sha256": b.sha(p / "completed.json")}
                   for name, p in (("packet", packet_dir), ("lexical", lexical), ("data", raw))},
        "protocol_path": "research/dialogue-joint-protocol.md", "protocol_sha256": source_map["research/dialogue-joint-protocol.md"],
        "unique_texts": 1, "index_count": offset, "counts": counts,
        "layout_files": {n: b.sha(folder/n) for n in ("index.json", "indices.npy")}}
    put(folder / "plan.json", prep)
    put(folder / "started.json", {"status": "started"})
    profile = {"input_tokens": 3, "encoder_tokens_with_special": 5, "padded_token_slots": 5,
               "overlength_texts_chunked": 0, "truncated_tokens": 0, "encoder_calls": 1, "encoder_sequences": 1}
    put(cap / "token-profile.json", profile)
    common = {"status": "completed", "version": r.CACHE_VERSION, "plan_sha256": b.sha(folder / "plan.json"),
        "source_sha256": source_map, "runtime": prep["runtime"], "model_files_sha256": prep["model_files_sha256"],
        "encoder": r.ENCODER, "revision": r.REVISION, "device": "mps", "unique_texts_encoded": 1, "full_unique_texts": 1,
        "no_retry": True, "work": {**profile, "encoder_seconds": .1}, "counts": counts,
        "progress": {"encoder_calls_attempted": 1, "encoder_calls_returned": 1, "encoded_sequences": 1, "encoded_texts": 1}}
    projection = {"projected_encoder_seconds": .1, "observed_nonencoding_seconds": .1, "projected_total_seconds": .2,
                  "maximum_projected_seconds": 720., "projected_cache_bytes": 10000, "maximum_cache_bytes": r.DISK_CAP, "encoding_permitted": True}
    seal(cap, {**common, "phase": "capacity", "wall_seconds": .2, "projection": projection})
    receipt = seal(folder, {**common, "phase": "encode", "test_contents_accessed": False, "parameter_training_steps": 0,
        "wall_seconds": .3, "capacity_path": "capacity", "capacity_completed_sha256": b.sha(cap / "completed.json"),
        **{k+"_completed_sha256": v["completed_sha256"] for k, v in prep["inputs"].items()}})
    receipt["cache_file_bytes"] = sum(v["bytes"] for v in receipt["files"].values())
    put(folder / "completed.json", receipt)
    return folder, raw


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    for name in r.SOURCES:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT/name).read_bytes())
    run, packet_dir, lexical, prior = [root/n for n in ("run", "packet", "lexical", "prior")]
    for p in (run, packet_dir, lexical, prior):
        p.mkdir()
    data = packet()
    put(packet_dir / "packet.json", data)
    (packet_dir / "encoder-source.py").write_text("# synthetic encoder provenance only")
    put(packet_dir / "encoder-plan.json", {"source_sha256": b.sha(packet_dir / "encoder-source.py"),
        "model_files_sha256": {n: "a"*64 for n in r.MODEL_FILES}})
    (packet_dir / "features.npy").write_bytes(b"opaque, unused saved array")
    seal(packet_dir, {"status": "completed", "cohorts": {"dev": {"queries": 10, "dialogues": 2}}, "wall_seconds": 1.})
    for name in ("started.json", "index.json", "lexical.npy"):
        (lexical/name).write_bytes(b"opaque saved lexical bytes")
    seal(lexical, {"status": "completed", "packet_completed_sha256": b.sha(packet_dir / "completed.json"),
        "source_sha256": {r.V2_PATH: r.V2_SHA}, "wall_seconds": .1})
    joint, raw = build_joint(root, data, packet_dir, lexical)
    monkeypatch.setattr(r, "DATA_COMPLETED", b.sha(raw / "completed.json"))
    prior_plan = {"study": "dialogue-copy-v2", "config": r.v2.CONFIG, "practical_checks": r.v2.PRACTICAL_CHECKS,
        "runtime": {k: "synthetic" for k in ("python", "torch", "numpy", "platform")},
        "packet_completed_sha256": b.sha(packet_dir / "completed.json"), "lexical_completed_sha256": b.sha(lexical / "completed.json"),
        "source_sha256": {n: b.sha(root/n) for n in r.v2.SOURCES}, "loss_counts": {"0": 1, "1": 1, "2": 3},
        "loss_weights": [5/3, 5/3, 5/9], "optimizer_updates_per_fit": 20}
    put(prior / "plan.json", prior_plan)
    originals = {}
    for seed in r.SEEDS:
        for method in r.v2.METHODS:
            row = {"method": method, "seed": seed, "status": "completed", "parameters": 99458,
                "initial_tensors_sha256": "a"*64, "configuration": {"class": "MonitoredCopyMemoryV2", "method": method,
                "implementation_version": "dialogue-copy-memory-v2-normalized"}}
            originals[f"{method}-{seed}"] = row
            put(prior / "fits" / f"{method}-{seed}" / "completed.json", row)
    put(prior / "completed.json", {"status": "completed", "fit_count": 15, "plan_sha256": b.sha(prior / "plan.json"), "fits": list(originals.values())})
    monkeypatch.setattr(r, "OLD_PLAN", b.sha(prior / "plan.json"))
    monkeypatch.setattr(r, "OLD_COMPLETED", b.sha(prior / "completed.json"))
    configs = {arm: r.configuration_from_parent(originals[arm.rsplit('_', 1)[1]+"-4101"]["configuration"]) for arm in r.ARMS}
    plan = {**prior_plan, "study": r.STUDY, "config": r.CONFIG, "practical_checks": r.CHECKS, "implementation_version": r.VERSION,
        "wall_cap_seconds": 3600., "normalization_tolerance": 2e-6, "execution_files": 52, "expected_fits": 12,
        "source_sha256": {n: b.sha(root/n) for n in r.SOURCES}, "joint_completed_sha256": b.sha(joint / "completed.json"),
        "paths": {"packet": "packet", "lexical": "lexical", "joint": "joint", "old_study": "prior"},
        "old_plan_sha256": r.OLD_PLAN, "old_completed_sha256": r.OLD_COMPLETED, "expected_configurations": configs}
    put(run / "plan.json", plan)
    put(run / "started.json", {"status": "started", "plan_sha256": b.sha(run / "plan.json"), "runtime": plan["runtime"], "wall_cap_seconds": 3600.})
    expected, _ = b.expected_records(data)
    p = np.zeros((10, 12), np.float32)
    p[:, :4] = .1
    p[np.arange(10), expected["labels"]] = .7
    fits = []
    for seed in r.SEEDS:
        for arm in r.ARMS:
            mode, head = arm.rsplit("_", 1)
            folder = run / "fits" / f"{arm}-{seed}"
            folder.mkdir(parents=True)
            (folder / "weights.pt").write_bytes(b"opaque, no deserialization")
            np.savez_compressed(folder / "dev-predictions.npz", **expected, probabilities=p, choice=p.argmax(-1))
            tr, dev = data["cohorts"]["train"], data["cohorts"]["dev"]
            ledger = [{"indices": [0], "epoch": e, "update": e+1, "loss": .5, "supervised_queries": 5,
                "actor_shapes": r.v2.work_counts(tr, data["queries"]), "observation_work": r.observation_work(tr, data["queries"], mode),
                "invariants": witness(head, tr)} for e in range(20)]
            (folder / "batches.jsonl").write_text("".join(json.dumps(x)+"\n" for x in ledger))
            inv, shapes, _, losses = r.v2.audit_batches(ledger, tr, data["queries"], head, training=True)
            eb = {"indices": [0, 1], "actor_shapes": r.v2.work_counts(dev, data["queries"]),
                  "observation_work": r.observation_work(dev, data["queries"], mode), "invariants": witness(head, dev)}
            evaluation = {"queries": 10, "wall_seconds": .2, "batches": [eb], **{k: eb[k] for k in ("actor_shapes", "observation_work", "invariants")}}
            record = {"status": "completed", "method": arm, "head": head, "observation": mode, "seed": seed,
                "epochs": 20, "updates": 20, "training_queries": 100, "training_losses": losses,
                "training_actor_shapes": shapes, "training_observation_work": r.audit_observations(ledger, tr, data["queries"], mode),
                "training_invariants": inv, "initial_tensors_sha256": "a"*64, "original_initial_tensors_sha256": "a"*64,
                "common_initial_tensors_sha256": "a"*64, "configuration": configs[arm], "parameters": 99458,
                "parameters_with_final_gradient": 99000, "train_wall_seconds": 1., "fit_wall_seconds_before_receipt": 1.3,
                "evaluation": evaluation, "weights_sha256": b.sha(folder / "weights.pt"),
                "predictions_sha256": b.sha(folder / "dev-predictions.npz"), "batches_sha256": b.sha(folder / "batches.jsonl")}
            put(folder / "completed.json", record)
            fits.append(record)
    np.savez_compressed(run / "references.npz", **expected, pred_none=np.zeros(10, np.int64), pred_literal=np.full(10, 2, np.int64))
    seal(run, {"status": "completed", "study": r.STUDY, "plan_sha256": b.sha(run / "plan.json"), "fit_count": 12, "fits": fits,
        "external_model_api_calls": 0, "encoder_calls": 0, "normalization_tolerance": 2e-6, "resume_authorized": False,
        "joint_completed_sha256": plan["joint_completed_sha256"], "wall_seconds": 17., "references": {"file": "references.npz",
        "queries": 10, "wall_seconds": .1, "sha256": b.sha(run / "references.npz")}})
    return root, run, packet_dir, lexical, joint


def invoke(tree, out):
    root, run, packet_dir, lexical, joint = tree
    return r.report(run, packet_dir, lexical, joint, b.sha(run / "plan.json"), b.sha(run / "completed.json"), out, root=root)


def test_full_synthetic_52_file_report_checks_work_support_gate_and_png(tree, tmp_path):
    out = tmp_path / "report"
    receipt = invoke(tree, out)
    assert receipt["technical_validity_passed"] and not receipt["continuation_passed"]
    assert receipt["checks_total"] == 15 and receipt["fit_count"] == 12
    assert receipt["neural_calls"] == receipt["encoder_calls"] == receipt["checkpoint_deserializations"] == 0
    summary = b.read(out / "summary.json")
    assert summary["technical_validity"]["exact_execution_files"] == 52
    assert summary["optimizer_updates"] == 240 and summary["supervised_presentations"] == 1200
    assert summary["families"]["joint_scalar"]["unseen"]["subgroups"]["assigned_true"]["count_per_fit"] == 3
    assert summary["families"]["joint_scalar"]["unseen"]["subgroups"]["assigned_false"]["accuracy"] is None
    assert summary["rows"]["joint_scalar-4101"]["training_observation_work"]["turn_projection_positions"] == 400
    assert (out / "comparison.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        invoke(tree, out)


@pytest.mark.parametrize("kind", ["weights", "extra", "source", "initial", "config", "observation", "invariants", "label", "order", "cost", "arm"])
def test_resealed_wrong_inputs_or_arithmetic_are_rejected(tree, tmp_path, kind):
    root, run, _, _, _ = tree
    done = b.read(run / "completed.json")
    record = done["fits"][3]
    folder = run / "fits/joint_scalar-4101"
    if kind == "weights":
        (folder / "weights.pt").write_bytes(b"different bytes")
    elif kind == "extra":
        (run / "unpermitted").write_text("extra")
    elif kind == "source":
        (root / "src/openjev/research/dialogue_joint_memory.py").write_text("changed")
    elif kind == "initial":
        record["initial_tensors_sha256"] = "b"*64
    elif kind == "config":
        record["configuration"]["representation_control"] = "wrong"
    elif kind == "observation":
        record["training_observation_work"]["turn_projection_positions"] -= 1
    elif kind == "invariants":
        record["training_invariants"]["feature_checks"] -= 1
    elif kind == "label":
        with np.load(folder / "dev-predictions.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        a["labels"][0] = 2
        np.savez_compressed(folder / "dev-predictions.npz", **a)
        record["predictions_sha256"] = b.sha(folder / "dev-predictions.npz")
    elif kind == "order":
        record["evaluation"]["batches"][0]["indices"] = [1, 0]
    elif kind == "cost":
        done["wall_seconds"] = 1.
    else:
        record["observation"] = "independent"
    put(folder / "completed.json", record)
    seal(run, done)
    out = tmp_path / "bad-report"
    with pytest.raises(ValueError):
        invoke(tree, out)
    assert (out / "failed.json").exists() and not (out / "receipt.json").exists()


@pytest.mark.parametrize("kind", ["embedding", "layout", "index", "model", "capacity"])
def test_joint_cache_resealed_geometry_model_and_capacity_reject(tree, tmp_path, kind):
    root, run, packet_dir, lexical, joint = tree
    plan = b.read(run / "plan.json")
    if kind == "embedding":
        x = np.zeros((1, 384), np.float32)
        np.save(joint / "embeddings.npy", x)
    elif kind == "layout":
        x = b.read(joint / "index.json")
        x["cohorts"]["dev"][0]["query_ids"] = [1]
        put(joint / "index.json", x)
    elif kind == "index":
        x = np.load(joint / "indices.npy", allow_pickle=False)
        x[0] = -1
        np.save(joint / "indices.npy", x)
    elif kind == "model":
        x = b.read(joint / "completed.json")
        x["model_files_sha256"]["model.safetensors"] = "b"*64
        put(joint / "completed.json", x)
    else:
        cap = b.read(root / "capacity/completed.json")
        cap["projection"]["projected_total_seconds"] = 721.
        put(root / "capacity/completed.json", cap)
        x = b.read(joint / "completed.json")
        x["capacity_completed_sha256"] = b.sha(root / "capacity/completed.json")
        put(joint / "completed.json", x)
    saved = seal(joint, b.read(joint / "completed.json"))
    saved["cache_file_bytes"] = sum(v["bytes"] for v in saved["files"].values())
    put(joint / "completed.json", saved)
    plan["joint_completed_sha256"] = b.sha(joint / "completed.json")
    with pytest.raises(ValueError):
        r.authenticate_joint(joint, packet_dir, lexical, packet(), plan, root, {})


def test_failure_receipt_failure_keeps_original_cause(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("original audit failure")
    monkeypatch.setattr(r, "audit_saved", boom)
    original_write = b.write
    def write(path, value):
        if Path(path).name == "failed.json":
            raise OSError("disk full")
        original_write(path, value)
    monkeypatch.setattr(b, "write", write)
    with pytest.raises(ValueError, match="original audit failure") as error:
        r.report(tmp_path, tmp_path, tmp_path, tmp_path, "a"*64, "b"*64, tmp_path / "out")
    assert "disk full" in str(error.value.__notes__)
