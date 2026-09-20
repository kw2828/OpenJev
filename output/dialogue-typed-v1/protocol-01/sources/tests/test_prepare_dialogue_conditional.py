"""Synthetic metadata only. No corpus, encoders, models or RNG calls."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_dialogue_conditional.py"
SPEC = importlib.util.spec_from_file_location("conditional_prep", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture():
    packet = {"queries": [], "cohorts": {}}
    catalog, lexical, token = {}, {"features": list(m.FIELDS), "cohorts": {}}, {
        "version": "dialogue-token-minilm-v1", "width": 384, "cohorts": {},
        "original_feature_indices": list(range(12)), "unique_contexts": 12, "context_occurrences": 12}
    ids, values = [m.NONE, m.DONTCARE, "value:true", "value:false"], [None, None, "true", "false"]
    for qi, split in enumerate(("train", "dev")):
        service = "service_" + split
        qid = json.dumps([service, "slot"], separators=(",", ":"))
        q = {"id": qid, "split": split, "service": service, "slot": "slot", "text": 12,
             "candidates": [13, 14, 15, 16], "candidate_ids": ids.copy(), "candidate_values": values.copy()}
        packet["queries"].append(q)
        catalog[split] = [{"query_id": qid, "service": service, "slot": "slot", "query_text": "NEVER_USE",
                           "candidates": [{"id": cid, "value": v, "text": "NEVER_USE"}
                                          for cid, v in zip(ids, values, strict=True)]}]
        rows = []
        for t, label, bin_name in [(0, 0, "unmentioned_retention"), (1, 2, "first_assignment"),
                                   (3, 3, "revision"), (4, 3, "assigned_retention"), (5, 1, "revision")]:
            rows.append({"time": t, "query": qi, "label": label, "bin": bin_name,
                         "unseen": split == "dev", "dontcare": label == 1})
        packet["cohorts"][split] = [{"id": "same-id-across-splits", "turns": list(range(qi*6, qi*6+6)),
                                     "user_text": ["NEVER_USE"]*6, "queries": rows[::-1]}]
        lexical["cohorts"][split] = [{"id": "same-id-across-splits", "query_ids": [qi],
                                      "offset": qi*240, "shape": [6, 1, 4, 10]}]
        token["cohorts"][split] = [{"id": "same-id-across-splits", "query_ids": [qi], "offset": qi*6, "shape": [6]}]
    arrays = {"text-token-counts.npy": np.full(12, 3, np.int64), "offsets.npy": np.arange(13, dtype=np.int64)*5,
              "chunk-offsets.npy": np.arange(13, dtype=np.int64), "chunk-lengths.npy": np.full(12, 3, np.int64),
              "context-indices.npy": np.arange(12, dtype=np.int64)}
    shapes = {"features.npy": (17, 384), "lexical.npy": (480,), "tokens.npy": (60, 384), "priors.npy": (60,)}
    return packet, catalog, lexical, token, arrays, shapes


def prepared_fixture():
    p, c, lex, tok, a, shapes = fixture()
    q = m.catalog_metadata(p, c, shapes["features.npy"][0])
    layouts = m.validate_metadata(p, lex, tok, a, q, shapes)
    return p, q, layouts, a


def test_admission_alignment_whitelist_and_source_order():
    p, q, layouts, a = prepared_fixture()
    rows = list(m.row_ledger(p, q, layouts, a))
    assert len(rows) == 10
    assert [r["source_row_index"] for r in rows[:5]] == [4, 3, 2, 1, 0]
    assert [r["admission"] for r in rows[:5]] == ["first_public_turn", "admitted", "missing_adjacent_scored_predecessor", "admitted", "admitted"]
    assert rows[1]["previous_current_index"] == 0
    assert rows[1]["cache"] == {"pooled_index": 1, "query_feature_index": 12,
        "candidate_feature_indices": [13, 14, 15, 16], "token_context_index": 1, "token_start": 5, "token_stop": 10,
        "lexical_start": 40, "lexical_candidates": 4, "lexical_stride": 10}
    assert "NEVER_USE" not in json.dumps(rows) + json.dumps(q)
    groups = m.aggregate(rows)
    dev = groups["dev/unseen"]
    assert dev["adjacent_eligible_rows"] == 3 and dev["changed"] == 2 and dev["retained"] == 1
    assert dev["values"] == {"none": 0, "true": 1, "false": 1, "dontcare": 1, "other": 0}
    assert dev["sum_score_positions"] == 60 and dev["max_score_positions"] == 20
    assert dev["sum_float_input_bytes"]["tokens"] == 3*4*(5*384+5+384+4*384+40+4)
    assert groups["dev/seen"]["scored_rows"] == 0


def test_exact_candidate_id_remapping_and_absent_value():
    p, q, layouts, a = prepared_fixture()
    other = copy.deepcopy(q[0])
    other["candidate_ids"] = [m.NONE, m.DONTCARE, "value:false", "value:true"]
    other["candidate_values"] = [None, None, "false", "true"]
    other["query_index"] = 2
    q.append(other)
    d = p["cohorts"]["train"][0]
    d["queries"] = [{"time": 0, "query": 0, "label": 2, "bin": "first_assignment", "unseen": False, "dontcare": False},
                    {"time": 1, "query": 2, "label": 2, "bin": "revision", "unseen": False, "dontcare": False}]
    layouts[("train", d["id"])]["lexical"]["query_ids"] = [0, 2]
    layouts[("train", d["id"])]["lexical"]["shape"][1] = 2
    rows = list(m.row_ledger(p, q, layouts, a))
    assert rows[1]["previous_current_index"] == 3 and rows[1]["current_label_index"] == 2
    other["candidate_ids"][3] = "value:else"
    rows = list(m.row_ledger(p, q, layouts, a))
    assert rows[1]["admission"] == "previous_candidate_absent"
    assert rows[1]["previous_current_index"] is None
    assert m.aggregate(rows)["train/all"]["adjacent_eligible_rows"] == 1


@pytest.mark.parametrize("field,value", [("time", True), ("time", -1), ("query", True), ("label", True),
                                         ("label", 12), ("unseen", 1), ("dontcare", 1), ("bin", "foreign")])
def test_invalid_scored_metadata_rejected_even_excluded(field, value):
    p, q, layouts, a = prepared_fixture()
    first = next(x for x in p["cohorts"]["train"][0]["queries"] if x["time"] == 0)
    first[field] = value
    with pytest.raises((ValueError, IndexError)):
        list(m.row_ledger(p, q, layouts, a))


def test_duplicate_and_bad_adjacent_bin_rejected():
    p, q, layouts, a = prepared_fixture()
    d = p["cohorts"]["train"][0]
    d["queries"].append(copy.deepcopy(d["queries"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        list(m.row_ledger(p, q, layouts, a))
    d["queries"].pop()
    next(r for r in d["queries"] if r["time"] == 1)["bin"] = "assigned_retention"
    with pytest.raises(ValueError, match="transition bin"):
        list(m.row_ledger(p, q, layouts, a))


def test_future_text_and_labels_do_not_change_past_rows():
    p, q, layouts, a = prepared_fixture()
    before = list(m.row_ledger(p, q, layouts, a))
    d = p["cohorts"]["train"][0]
    d["user_text"] = [object()]*6
    last = next(r for r in d["queries"] if r["time"] == 5)
    last.update(label=0, dontcare=False, bin="clear")
    after = list(m.row_ledger(p, q, layouts, a))
    assert before[:4] == after[:4]


@pytest.mark.parametrize("kind", ["context", "offset", "chunks", "lexical", "token", "bool_index", "duplicate_dialogue"])
def test_full_layout_corruption_rejected(kind):
    p, c, lex, tok, a, shapes = fixture()
    if kind == "context":
        a["context-indices.npy"][0] = 1
    elif kind == "offset":
        a["offsets.npy"][1] += 1
    elif kind == "chunks":
        a["chunk-lengths.npy"][0] += 1
    elif kind == "lexical":
        lex["cohorts"]["dev"][0]["offset"] += 10
    elif kind == "token":
        tok["cohorts"]["dev"][0]["query_ids"] = [0]
    elif kind == "bool_index":
        p["cohorts"]["train"][0]["turns"][0] = True
    else:
        p["cohorts"]["train"].append(copy.deepcopy(p["cohorts"]["train"][0]))
        lex["cohorts"]["train"].append(copy.deepcopy(lex["cohorts"]["train"][0]))
        tok["cohorts"]["train"].append(copy.deepcopy(tok["cohorts"]["train"][0]))
    q = m.catalog_metadata(p, c, 17)
    with pytest.raises(ValueError):
        m.validate_metadata(p, lex, tok, a, q, shapes)


@pytest.mark.parametrize("kind", ["duplicate", "reserved", "embedding", "catalog"])
def test_catalog_corruption(kind):
    p, c, *_ = fixture()
    if kind == "duplicate":
        p["queries"].append(copy.deepcopy(p["queries"][0]))
    elif kind == "reserved":
        p["queries"][0]["candidate_ids"][0] = "value:None"
    elif kind == "embedding":
        p["queries"][0]["text"] = True
    else:
        c["train"][0]["candidates"][2]["value"] = "TRUE"
    with pytest.raises(ValueError):
        m.catalog_metadata(p, c, 17)


def test_header_only_float_and_whitelisted_integer_decode(tmp_path, monkeypatch):
    # NaN values are deliberately untouched: this stage inherits numeric validation.
    floats = tmp_path / "features.npy"
    np.save(floats, np.full((2, 384), np.nan, np.float32), allow_pickle=False)
    ints = tmp_path / "offsets.npy"
    np.save(ints, np.array([0, 3], np.int64), allow_pickle=False)
    original = np.load
    def guarded(path, **kwargs):
        assert Path(path).name in m.INTEGER_FILES
        return original(path, **kwargs)
    monkeypatch.setattr(m.np, "load", guarded)
    assert m.float_header(floats) == (2, 384)
    assert m.int_array(ints).tolist() == [0, 3]
    with pytest.raises(ValueError, match="Only integer"):
        m.int_array(floats)
    with floats.open("ab") as f:
        f.write(b"extra")
    with pytest.raises(ValueError, match="payload size"):
        m.float_header(floats)


def test_strict_json_and_unsafe_path(tmp_path):
    p = tmp_path / "metadata.json"
    for content in ('{"a":1,"a":2}', '{"a":NaN}'):
        p.write_text(content)
        with pytest.raises(ValueError):
            m.read(p)
    with pytest.raises(ValueError):
        m.safe(tmp_path, "../outside")


def test_exclusive_attempt_failure_and_late_demotion(tmp_path, monkeypatch):
    def success(out, _budget, _args):
        m.write(out / "small.json", {"value": 1})
        return {"source_sha256": {}, "plan_sha256": "a"*64}
    out = tmp_path / "success"
    digest = m.attempt("freeze", out, success, None)
    assert digest == m.sha(out / "completed.json")
    with pytest.raises(FileExistsError):
        m.attempt("freeze", out, success, None)
    def fails(out, budget, _args):
        (out / "rows.jsonl").write_text('{"partial":true}\n')
        budget.progress["rows_written"] = 1
        raise ValueError("original failure")
    out = tmp_path / "failure"
    with pytest.raises(ValueError, match="original failure"):
        m.attempt("prepare", out, fails, SimpleNamespace(plan="frozen/plan.json", plan_sha256="f"*64))
    assert m.read(out / "failed.json")["progress"]["rows_written"] == 1
    assert m.read(out / "failed.json")["request"]["plan_sha256"] == "f"*64
    original = m.Budget.storage
    def late(self, extra=0):
        if (self.out / "completed.json").exists():
            raise TimeoutError("late cap")
        return original(self, extra)
    monkeypatch.setattr(m.Budget, "storage", late)
    out = tmp_path / "late"
    with pytest.raises(TimeoutError):
        m.attempt("freeze", out, success, None)
    assert not (out / "completed.json").exists()
    assert (out / "late-completion.json").exists() and (out / "failed.json").exists()


def test_storage_and_wall_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUTPUT_CAP", 1)
    with pytest.raises(ValueError, match="storage cap"):
        m.attempt("freeze", tmp_path / "small", lambda *_: {}, None)
    b = m.Budget(tmp_path, 0.)
    monkeypatch.setattr(m.time, "monotonic", lambda: m.WALL_CAP+1)
    with pytest.raises(TimeoutError):
        b.check()


@pytest.mark.parametrize("corruption", ["runtime", "source", "pin", "input", "snapshot"])
def test_plan_authentication_before_prepare(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(m, "source_map", lambda *_: {"source.py": "b"*64})
    plan = {"version": m.VERSION, "runtime": m.runtime(), "source_sha256": {"source.py": "b"*64},
            "inputs": {k: {"path": v[0], "sha256": v[1]} for k, v in m.INPUTS.items()},
            "limits": {"wall_seconds": m.WALL_CAP, "output_bytes": m.OUTPUT_CAP}}
    if corruption == "runtime":
        plan["runtime"]["python"] = "foreign"
    elif corruption == "source":
        plan["source_sha256"]["source.py"] = "c"*64
    elif corruption == "input":
        plan["inputs"]["data"]["sha256"] = "d"*64
    m.write(tmp_path / "plan.json", plan)
    digest = m.sha(tmp_path / "plan.json")
    m.write(tmp_path / "completed.json", {"status": "completed", "phase": "freeze", "plan_sha256": digest,
            "source_sha256": plan["source_sha256"], "no_retry": True, "files": m.manifest(tmp_path)})
    if corruption == "snapshot":
        (tmp_path / "extra").write_text("unlisted")
    with pytest.raises(ValueError):
        m.validate_plan(tmp_path / "plan.json", "a"*64 if corruption == "pin" else digest,
                        m.Budget(tmp_path, m.time.monotonic()))


def test_binding_hash_and_byte_record_validation(tmp_path):
    p = tmp_path / "payload"
    p.write_bytes(b"test")
    budget = m.Budget(tmp_path, m.time.monotonic())
    verified = {}
    m.binding(p, m.sha(p), budget, verified)
    assert budget.progress["hashed_files"] == 1
    with pytest.raises(ValueError, match="Conflicting"):
        m.binding(p, "0"*64, budget, verified)
    with pytest.raises(ValueError, match="File record"):
        m.file_entry({"bytes": True, "sha256": "0"*64})


def test_empty_support_not_invented():
    result = m.aggregate([])
    assert len(result) == 6
    assert all(r["reasons"]["admitted"] == r["changed"] == r["retained"] == 0 for r in result.values())


@pytest.mark.parametrize("kind", ["train_unseen", "known_dev", "inconsistent", "bool_layout"])
def test_panel_and_layout_type_checks(kind):
    p, c, lex, tok, a, shapes = fixture()
    q = m.catalog_metadata(p, c, 17)
    if kind == "bool_layout":
        lex["cohorts"]["train"][0]["offset"] = False
        with pytest.raises(ValueError, match="integer types"):
            m.validate_metadata(p, lex, tok, a, q, shapes)
        return
    layouts = m.validate_metadata(p, lex, tok, a, q, shapes)
    if kind == "train_unseen":
        for r in p["cohorts"]["train"][0]["queries"]:
            r["unseen"] = True
    elif kind == "known_dev":
        q[1]["service"] = q[0]["service"]
    else:
        p["cohorts"]["dev"][0]["queries"][0]["unseen"] = False
    with pytest.raises(ValueError, match="panel flag|marked unseen"):
        list(m.row_ledger(p, q, layouts, a))


def test_freeze_positive_and_resealed_snapshot_rejection(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    protocol = root / m.PROTOCOL
    protocol.parent.mkdir()
    protocol.write_text("synthetic protocol")
    inp = root / "input.json"
    m.write(inp, {"status": "completed"})
    inputs = {"fake": ("input.json", m.sha(inp))}
    sources = {m.PROTOCOL: m.sha(protocol)}
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(m, "INPUTS", inputs)
    monkeypatch.setattr(m, "source_map", lambda *_: sources.copy())
    monkeypatch.setattr(m, "receipts", lambda *_args, **_kwargs: (
        {"token_plan": {"model_files_sha256": {}}},
        {str(inp): {"sha256": m.sha(inp), "bytes": inp.stat().st_size}}))
    out = root / "freeze"
    m.attempt("freeze", out, m.freeze_body, None)
    pin = m.sha(out / "plan.json")
    assert m.validate_plan(out / "plan.json", pin, m.Budget(out, m.time.monotonic()))["version"] == m.VERSION
    snap = out / "sources" / m.PROTOCOL
    snap.write_text("changed")
    completed = m.read(out / "completed.json")
    completed["files"]["sources/"+m.PROTOCOL] = {"sha256": m.sha(snap), "bytes": snap.stat().st_size}
    (out / "completed.json").write_bytes(m.encoded(completed))
    with pytest.raises(ValueError, match="source snapshot identity"):
        m.validate_plan(out / "plan.json", pin, m.Budget(out, m.time.monotonic()))


def test_hash_loop_obeys_timeout(tmp_path):
    p = tmp_path / "opaque.bin"
    p.write_bytes(b"no numerical interpretation")
    def stop():
        raise TimeoutError("hash timeout")
    with pytest.raises(TimeoutError, match="hash timeout"):
        m.sha(p, stop)


def test_no_model_or_experiment_imports():
    import ast
    tree = ast.parse(PATH.read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(x and x.split(".")[0] in {"torch", "transformers", "openjev"} for x in imports)
