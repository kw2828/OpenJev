"""Tiny artificial checks only; never invokes the frozen sixteen-cell cost screen."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import qualify_dialogue_token_projection as q


@pytest.fixture(autouse=True)
def isolate():
    before = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(before)


def geometry():
    # B2/T3/Q2/C4/L5, including one padded dummy NONE query on dialogue 1.
    return {"version": "dialogue-token-packing-geometry-v1", "maximum_shape": [2, 3, 2, 4, 5],
        "dialogs": [{"turn_token_lengths": [5, 2, 3], "candidate_counts": [4, 3]},
                    {"turn_token_lengths": [2, 1], "candidate_counts": [3]}],
        "derived_counts": {"actor_shapes": {"padded_candidate_positions": 48, "padded_query_positions": 12,
            "padded_turn_positions": 6, "real_question_steps": 8, "real_turns": 5},
            "observation_work": {"emitted_bytes": 46080, "emitted_float32_scalars": 11520,
                "emitted_token_mask_bytes": 30, "emitted_token_prior_bytes": 120,
                "evidence_query_projection_positions": 16, "padded_token_positions": 30,
                "pooling_evidence_positions": 48, "pooling_score_positions": 240,
                "raw_cache_prior_bytes_read": 52, "raw_cache_token_bytes_read": 19968,
                "real_candidate_updates": 27, "real_public_turns": 5, "real_question_updates": 8,
                "schema_candidate_projection_positions": 16, "schema_query_projection_positions": 4,
                "token_key_projection_positions": 30, "turn_projection_positions": 48, "valid_token_positions": 13},
            "active_candidate_rows_including_dummy_none": 29, "dummy_none_question_updates": 2,
            "valid_turn_question_slots_including_dummy_none": 10,
            "active_candidate_token_interactions_including_dummy_none": 82,
            "real_candidate_token_interactions": 79, "masked_candidate_rows_in_full_padded_head": 19,
            "masked_candidate_rows_on_valid_turns": 11, "padded_turn_candidate_rows": 8}}


def tiny(mode="slot", head="scalar", index=12):
    return {"index": index, "name": "synthetic", "shape": [2, 3, 2, 4, 5], "mode": mode, "head": head, "seed": 410}


def progress():
    return {k+s: 0 for k in ("parity_forward", "parity_backward", "update_forward", "update_backward", "optimizer_steps")
            for s in ("_attempted", "_returned")}


def fake_results():
    rows, events = [], []
    for case in q.CASES:
        ratio = .90 if case["index"] < 4 else 1.10
        paths = {"original": {"seconds": ratio}, "projected": {"seconds": 1.}}
        row = {"case": copy.deepcopy(case), "parity": {"passed": True}, "warm": copy.deepcopy(paths), "measured": []}
        events.extend({"kind": kind, "case": case["name"], "path": path} for kind in ("parity", "warm") for path in q.PATHS)
        for pair, order in enumerate(q.RECIPE["pair_orders"]):
            row["measured"].append({"pair": pair, "order": list(order), "paths": copy.deepcopy(paths), "speed_ratio": ratio})
            events.extend({"kind": "measured", "case": case["name"], "path": path, "pair": pair} for path in order)
        rows.append(row)
    return rows, events


def test_dense_cases_and_recipe_remain_unchanged():
    assert q.CASES == q.packing.CASES
    assert q.packing.RECIPE["pair_orders"][0] == ["original", "packed"]
    assert q.CASES[:12] == q.old.CASES
    assert q.old.RECIPE["optimizer_updates"] == 120 and q.old.RECIPE["pair_orders"][0][1] == "batched"
    assert [x["seed"] for x in q.CASES[12:]] == [91301, 91302, 91303, 91304]
    for k in ("output_atol", "output_rtol", "gradient_atol", "gradient_rtol", "loss_atol", "loss_rtol"):
        assert q.RECIPE[k] == q.old.RECIPE[k]


def test_exact_192_events_160_updates_and_inclusive_boundaries():
    rows, events = fake_results()
    result = q.aggregate(rows, events, 6*1024**3)
    assert result["engineering_admission"] and not result["full_training_authorized"]
    assert (result["operation_records"], result["optimizer_updates"], result["parity_forward_backward_passes"]) == (192, 160, 32)


@pytest.mark.parametrize("index", [0, 4, 11, 12, 15])
def test_every_dense_and_geometry_threshold_required(index):
    rows, events = fake_results()
    for pair in rows[index]["measured"]:
        pair["paths"]["original"]["seconds"] -= .0001
        pair["speed_ratio"] -= .0001
    assert not q.aggregate(rows, events, 1)["engineering_admission"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "seed", "order", "nan", "ratio"])
def test_corrupt_membership_and_time_rejected(kind):
    rows, events = fake_results()
    if kind == "missing":
        events.pop()
    elif kind == "duplicate":
        events.append(events[-1])
    elif kind == "seed":
        rows[15]["case"]["seed"] += 1
    elif kind == "order":
        rows[15]["measured"][0]["order"].reverse()
    elif kind == "nan":
        rows[15]["measured"][0]["paths"]["projected"]["seconds"] = float("nan")
    else:
        rows[15]["measured"][0]["speed_ratio"] = 9.
    with pytest.raises(ValueError):
        q.aggregate(rows, events, 1)


def test_parity_and_rss_are_separate_failures():
    rows, events = fake_results()
    rows[15]["parity"]["passed"] = False
    result = q.aggregate(rows, events, 6*1024**3+1)
    assert not result["all_parity_passed"] and not result["rss_passed"] and not result["engineering_admission"]


@pytest.mark.parametrize("kind", ["bool", "max", "count", "foreign_field"])
def test_geometry_rejects_forged_counts_and_scope(kind):
    g = geometry()
    if kind == "bool":
        g["dialogs"][0]["candidate_counts"][0] = True
    elif kind == "max":
        g["maximum_shape"][-1] += 1
    elif kind == "count":
        g["derived_counts"]["dummy_none_question_updates"] -= 1
    else:
        g["dialogs"][0]["label"] = 0
    with pytest.raises(ValueError):
        q.geometry_counts(g)


def test_actual_assembler_geometry_dummy_none_and_label_independence():
    study, _, _ = q.backend()
    case, g = tiny(), geometry()
    data = q.geometry_sample(case, g)
    actor, labels, _ = study.make_batch(*data, case["mode"])
    assert q.old.digest(data) == q.old.digest(q.geometry_sample(case, g))
    assert study.observation_work(data[0], data[1], actor) == q.geometry_counts(g)["observation_work"]
    counts = q.mask_work(*(actor[i].numpy() for i in (1, 4, 6)))
    assert counts["packed_token_key_positions"] == 13 and counts["packed_evidence_positions"] == 29
    assert counts["packed_score_positions"] == 145 and counts["pooling_groups"] == 2
    assert actor[4][1, 1, 0] and not actor[4][1, 1, 1:].any()
    changed = copy.deepcopy(data)
    for d in changed[0]:
        for r in d["queries"]:
            r["label"] = (r["label"]+1) % len(changed[1][r["query"]]["candidates"])
    other, new_labels, _ = study.make_batch(*changed, case["mode"])
    assert q.old.digest(other) == q.old.digest(actor) and not torch.equal(labels, new_labels)


@pytest.mark.parametrize("mode,head", q.old.ARMS)
def test_tiny_geometry_actual_output_all_gradient_and_monitor_parity(mode, head):
    study, _, classes = q.backend()
    case = tiny(mode, head)
    data = q.geometry_sample(case, geometry())
    actor, _, _ = study.make_batch(*data, mode)
    expected = q.mask_work(*(actor[i].numpy() for i in (1, 4, 6)))
    models = {}
    for name, cls in classes.items():
        torch.manual_seed(410)
        models[name] = cls(head, attention_mode=mode)
    events, work = [], progress()
    result = q.parity(models, data, case, torch.tensor([.5, 1., 2.]), study, torch, work, events.append, lambda: None, expected)
    assert result["passed"] and set(result["inputs"]) == set(map(str, range(8)))
    assert [e["path"] for e in events] == ["original", "projected"]
    assert events[1]["packing"]["positions_and_scalars"] == expected
    assert work["parity_backward_returned"] == 2 and work["active"]["path"] == "projected"
    assert result["outputs"]["negative_infinity_elements"] > 0
    models["projected"].last_packing_work["packed_evidence_positions"] -= 1
    with pytest.raises(ValueError, match="packing work"):
        q.packing_record(models["projected"], expected)


def test_tiny_case_canonical_postwarm_adam_state_resets_and_complete_work(monkeypatch):
    study, _, classes = q.backend()
    observed = []
    original = q.timed_update
    def capture(model, optimizer, *args):
        observed.append(q.old.digest((model.state_dict(), optimizer.state_dict())))
        return original(model, optimizer, *args)
    # Test-only interception of the called update; no production helper mutation.
    monkeypatch.setattr(q, "timed_update", capture)
    events, work = [], progress()
    result = q.run_case(tiny(), geometry(), torch.tensor([.5, 1., 2.]), study, torch, classes, work, events.append, lambda: None)
    assert len(observed) == 10 and len(set(observed[2:])) == 1
    assert observed[2] == result["measured_initial_sha256"] and observed[0] != observed[2]
    assert work["optimizer_steps_returned"] == 10 and work["parity_backward_returned"] == 2 and len(events) == 12
    assert all("packing" in e for e in events if e["path"] == "projected")
    assert all("packing" not in e for e in events if e["path"] == "original")
    assert all(e["seconds"] >= sum(e["phase_seconds"].values())-.000001 for e in events if e["kind"] != "parity")


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    names = {q.PRIOR_SOURCE, q.GEOMETRY} | {f"prior/{i}.py" for i in range(44)}
    for name in names | set(q.ADD_SOURCES):
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("artificial source "+name)
    gp = tmp_path/q.GEOMETRY
    gp.unlink()
    q.write(gp, geometry())
    monkeypatch.setattr(q, "GEOMETRY_SHA256", q.sha(gp))
    monkeypatch.setattr(q, "RECIPE", {**q.RECIPE, "geometry_sha256": q.sha(gp)})
    parent = {"source_sha256": {n: q.sha(tmp_path/n) for n in names}, "loss_counts": {"0": 1, "1": 2, "2": 3}, "loss_weights": [2., 1., 2/3]}
    parent_dir = tmp_path/"parent"
    parent_dir.mkdir()
    q.write(parent_dir/"plan.json", parent)
    for name in ("training-plan.json", "batching-plan.json", "batching-completed.json"):
        q.write(parent_dir/name, {"synthetic": True})
    monkeypatch.setattr(q, "PARENT_PLAN_SHA256", q.sha(parent_dir/"plan.json"))
    q.write(parent_dir/"completed.json", {"status": "completed", "phase": "freeze", "plan_sha256": q.sha(parent_dir/"plan.json"),
        "source_sha256": parent["source_sha256"], "model_calls": 0, "no_retry": True})
    def fake_parent(path, digest, *, complete=True):
        q.bind(path, digest)
        result = q.read(path)
        assert len(result["source_sha256"]) == 46
        return result
    monkeypatch.setattr(q, "parent_plan", fake_parent)
    args = SimpleNamespace(packing_plan=parent_dir/"plan.json", packing_plan_sha256=q.sha(parent_dir/"plan.json"),
        protocol=tmp_path/q.ADD_SOURCES[-1], protocol_sha256=q.sha(tmp_path/q.ADD_SOURCES[-1]), out=tmp_path/"freeze")
    digest = q.freeze(args)
    return args, digest


def test_synthetic_freeze_full51_closure_no_calls_and_exclusive(frozen):
    args, digest = frozen
    p = q.validate_plan(args.out/"plan.json", digest)
    assert len(p["source_sha256"]) == 51 and p["loss_weights"] == [2., 1., 2/3]
    assert q.read(args.out/"completed.json")["model_calls"] == 0
    with pytest.raises(FileExistsError):
        q.freeze(args)


@pytest.mark.parametrize("kind", ["source", "snapshot", "runtime", "recipe", "loss", "parent_done", "parent_identity", "geometry", "failed"])
def test_corrupt_envelope_before_backend(frozen, kind):
    args, expected = frozen
    planfile = args.out/"plan.json"
    p = q.read(planfile)
    if kind == "source":
        args.protocol.write_text("changed")
    elif kind == "snapshot":
        (args.out/"sources"/q.ADD_SOURCES[0]).write_text("changed")
    elif kind == "failed":
        q.write(args.out/"failed.json", {"status": "failed"})
    elif kind == "parent_done":
        (args.out/"packing-completed.json").write_text("{}")
    else:
        if kind == "runtime":
            p["runtime"]["python"] = "different"
        elif kind == "recipe":
            p["recipe"]["whole_cap_seconds"] = 301
        elif kind == "loss":
            p["loss_weights"][0] = 99.
        elif kind == "parent_identity":
            p["packing_plan_sha256"] = "0"*64
        else:
            p["geometry_sha256"] = "0"*64
        planfile.unlink()
        q.write(planfile, p)
        expected = q.sha(planfile)
    with pytest.raises(ValueError):
        q.validate_plan(planfile, expected)


def test_run_validation_failure_preserves_no_work(frozen, monkeypatch):
    args, digest = frozen
    args.protocol.write_text("changed")
    monkeypatch.setattr(q, "backend", lambda: pytest.fail("No backend allowed"))
    out = args.out.parent/"attempt"
    with pytest.raises(ValueError):
        q.run(SimpleNamespace(plan=args.out/"plan.json", plan_sha256=digest, out=out))
    failed = q.read(out/"failed.json")
    assert failed["progress"]["optimizer_steps_attempted"] == 0 and not failed["resume_authorized"]
    assert not (out/"completed.json").exists()


def test_pinned_prior_helper_source_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    p = tmp_path/q.PRIOR_SOURCE
    p.parent.mkdir()
    p.write_text("not the reviewed source")
    with pytest.raises(ValueError, match="hash mismatch"):
        q.parent_plan(tmp_path/"unused.json", "0"*64)


def test_projection_counts_include_bias_fill_and_reference_only_widths():
    valid = np.array([[True, False]], dtype=bool)
    candidates = np.array([[[True, True, False], [True, False, False]]], dtype=bool)
    token = np.array([[[True, False, True, False, False], [False]*5]], dtype=bool)
    work = q.mask_work(valid, candidates, token)
    assert work["packed_turn_projection_positions"] == 3 and work["dense_turn_projection_positions"] == 12
    assert work["turn_projection_calls"] == 1 and work["empty_turn_projection_calls"] == 0
    assert work["projected_bias_fill_positions"] == 12 and work["projected_skipped_positions"] == 9
    assert work["projected_scatter_scalars"] == 3*64 and work["dense_projected_scalars"] == 12*64
    record = q.packing_record(SimpleNamespace(last_packing_work=work), work)
    assert set(record["reference_only_fields"]) == {"scatter_evidence_scalars", "dense_evidence_scalars", "dense_turn_projection_positions"}
    assert not set(record["reference_only_fields"]) & set(record["float32_payload_bytes"])
    assert record["float32_payload_bytes"]["projected_scatter_scalars"] == 3*64*4
    valid[:] = False
    work = q.mask_work(valid, candidates, token)
    assert work["packed_turn_projection_positions"] == 0 and work["projected_skipped_positions"] == 12
    assert work["turn_projection_calls"] == work["empty_turn_projection_calls"] == 1
    assert work["projected_scatter_bytes_float32"] == 0 and work["dense_projected_bytes_float32"] == 12*64*4


def test_changed_counter_width_is_rejected():
    valid = np.array([[True]], dtype=bool)
    mask = np.ones((1, 1, 2), dtype=bool)
    tokens = np.ones((1, 1, 3), dtype=bool)
    expected = q.mask_work(valid, mask, tokens)
    wrong = dict(expected, projected_scatter_scalars=2*32)
    with pytest.raises(ValueError, match="packing work"):
        q.packing_record(SimpleNamespace(last_packing_work=wrong), expected)
