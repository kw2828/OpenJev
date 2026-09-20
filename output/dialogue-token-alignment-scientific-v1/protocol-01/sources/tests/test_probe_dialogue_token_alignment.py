"""Small artificial unit/integration checks, not the capacity measurement."""
from __future__ import annotations

import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import probe_dialogue_token_alignment as p
import pytest


def tiny_metadata():
    rows = []
    for i, length in enumerate((2, 4, 3, 5, 2, 3)):
        rows.append({"row_index": i, "query_index": 0, "candidate_count": 4,
                     "current_label_index": 2, "previous_current_index": 1, "derived_bin": "revision",
                     "cache": {"pooled_index": 0, "query_feature_index": 1,
                               "candidate_feature_indices": [2, 3, 4, 5],
                               "token_start": i*10, "token_stop": i*10+length,
                               "lexical_start": i*40, "lexical_candidates": 4, "lexical_stride": 10}})
    index = {"unique_texts": 4, "queries": [{"query_index": 0,
              "candidate_token_ids": [0, 1, 2, 3], "candidate_feature_indices": [2, 3, 4, 5]}]}
    schema = p.common.schema_metadata(index, np.array([0, 2, 5, 9, 11], np.int64))
    headers = {"features": {"shape": [8, 384]}, "tokens": {"shape": [60, 384]},
               "priors": {"shape": [60]}, "lexical": {"shape": [240]}}
    return rows, schema, headers


@pytest.fixture
def small_recipe(monkeypatch):
    recipe = {**p.RECIPE, "batch_size": 2, "microbatch_size": 1, "epochs": 1}
    monkeypatch.setattr(p, "RECIPE", recipe)
    monkeypatch.setattr(p.common, "CONFIG", {**p.common.CONFIG, "batch_size": 2, "microbatch_size": 1, "epochs": 1})
    return recipe


def test_public_geometry_never_copies_labels_or_previous_target():
    rows, _, _ = tiny_metadata()
    a = p.public_row(rows[0])
    changed = {**rows[0], "current_label_index": 999, "previous_current_index": 888, "derived_bin": "poison"}
    assert a == p.public_row(changed)
    assert set(a) == {"row_index", "query_index", "candidate_count", "cache", "previous_current_index"}
    assert a["previous_current_index"] == 0


def test_geometry_uses_actual_microbatch_padding(small_recipe):
    rows, schema, _ = tiny_metadata()
    g = p.effective_geometry(rows[:2], schema)
    assert g["microbatches"] == 2
    assert g["padded_context_token_positions"] == 6
    assert g["padded_schema_token_positions"] == 32
    assert g["padded_pairwise_positions"] == 96
    assert g["padded_comparison_positions"] == 56
    assert p.workload(g) == 384*64*38+4*64*64*56+3*64*96
    assert g["padded_pairwise_positions"] < p.common.geometry(rows[:2], schema)["padded_pairwise_positions"]


def test_schedule_covers_every_seed_epoch_and_tail(small_recipe):
    rows, schema, _ = tiny_metadata()
    orders = {seed: np.arange(5, dtype=np.int64)[None] for seed in p.RECIPE["seeds"]}
    values = p.schedule(rows[:5], rows, orders, schema)
    assert len(values["train"]) == len(values["evaluation"]) == 9
    assert sum(v["geometry"]["rows"] for v in values["train"]) == 15
    assert sum(v["geometry"]["rows"] for v in values["evaluation"]) == 18
    for phase in values:
        strata = p.stratify(values[phase])
        assert sum(s["batches_per_arm"] for s in strata) == len(values[phase])
        for s in strata:
            assert s["representative"]["workload"] == s["maximum_workload"]
            assert s["component_maxima"]["rows"] >= s["representative"]["geometry"]["rows"]


def test_strata_ties_choose_first_member_without_label_selection():
    records = [{"workload": 1, "geometry": {"rows": 2}, "index": i} for i in range(6)]
    assert [s["representative_index"] for s in p.stratify(records)] == [0, 2, 4]
    with pytest.raises(ValueError, match="Too few"):
        p.stratify(records[:2])


def timing_fixture(seconds=1.):
    plan = {"geometry_strata": {phase: [{"stratum": i, "batches_per_arm": i+1} for i in range(3)]
                               for phase in ("train", "evaluation")}}
    events = [{"phase": phase, "stratum": i, "method": method, "repeat": r,
               "warmup": r == 0, "seconds": seconds}
              for phase in plan["geometry_strata"] for i in range(3) for method in p.METHODS for r in range(4)]
    return plan, events


def test_projection_exact_boundary_all_arms_and_evaluation():
    plan, events = timing_fixture(80.)
    result = p.project(plan, events, 0., p.RECIPE["rss_bytes"])
    assert len(events) == 72 and len(result["cells"]) == 18
    assert result["projected_seconds"] == 2880 and result["admitted"]
    assert not p.project(plan, events, .0001, 0)["admitted"]
    assert not p.project(plan, events, 0, p.RECIPE["rss_bytes"]+1)["admitted"]
    events[1]["seconds"] = 81.
    assert not p.project(plan, events, 0, 0)["admitted"]


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nan", "warm_inf", "bad_warm", "foreign", "bool_repeat"])
def test_projection_rejects_incomplete_or_invalid_timings(corruption):
    plan, events = timing_fixture()
    if corruption == "missing": events.pop()
    elif corruption == "duplicate": events[-1] = events[0]
    elif corruption == "nan": events[1]["seconds"] = float("nan")
    elif corruption == "warm_inf": events[0]["seconds"] = float("inf")
    elif corruption == "bad_warm": events[1]["warmup"] = True
    elif corruption == "foreign": events[1]["method"] = "other"
    else: events[1]["repeat"] = True
    with pytest.raises(ValueError): p.project(plan, events, 0, 0)


def test_synthetic_arrays_are_small_repeatable_and_priors_are_valid():
    _, schema, headers = tiny_metadata()
    arrays, syn, first = p.synthetic_inputs({"feature_headers": headers}, schema, 410)
    second = p.synthetic_inputs({"feature_headers": headers}, schema, 410)[2]
    assert first == second and arrays["tokens"].shape == (60, 384)
    assert arrays["tokens"].bank.shape == (1024, 384)
    np.testing.assert_array_equal(arrays["tokens"][4:7], arrays["tokens"][[4, 5, 6]])
    np.testing.assert_allclose(arrays["priors"][4:7].sum(), 1.)
    np.testing.assert_allclose(syn["priors"][2:5].sum(), 1.)
    with pytest.raises(ValueError, match="address"):
        arrays["tokens"][[60]]
    with pytest.raises(ValueError, match="nonempty"):
        arrays["priors"][4:4]


@pytest.mark.parametrize("method", p.METHODS)
def test_tiny_real_model_event_uses_shared_actor_and_one_optimizer_step(method, small_recipe):
    import torch
    torch.set_num_threads(1)
    rows, schema, headers = tiny_metadata()
    rows = [p.public_row(r) for r in rows[:2]]
    arrays, syn, _ = p.synthetic_inputs({"feature_headers": headers}, schema, 410)
    models, _ = p.common.init_models(torch, 410)
    model = models[method]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    progress = dict.fromkeys(("events_started", "events_completed", "optimizer_attempted", "optimizer_returned",
                             "forward_attempted", "forward_returned", "backward_attempted", "backward_returned"), 0)
    identity = {"phase": "train", "stratum": 0, "method": method, "repeat": 0, "warmup": True}
    log = io.StringIO()
    record = p.event(torch, model, optimizer, method, "train", rows, arrays, {0: [0, 1, 2, 3]}, syn,
                     np.array([2, 1], np.int64), np.array([.5, 2.], np.float32), log, identity, progress, lambda: None)
    assert progress["optimizer_returned"] == 1
    assert progress["forward_returned"] == progress["backward_returned"] == 2
    assert record["rows"] == 2 and record["seconds"] > 0
    assert json.loads(log.getvalue())["synthetic_weighted_loss"] == record["synthetic_weighted_loss"]
    assert not any("max_" in key for key in record["work"])
    before = {n: v.detach().clone() for n, v in model.named_parameters()}
    identity = {**identity, "phase": "evaluation"}
    record = p.event(torch, model, optimizer, method, "evaluation", rows, arrays, {0: [0, 1, 2, 3]}, syn,
                     np.array([2, 1], np.int64), np.ones(2, np.float32), io.StringIO(), identity, progress, lambda: None)
    assert record["output_materialized_bytes"] == 2*12*4
    assert len(record["synthetic_output_sha256"]) == 64
    assert progress["optimizer_returned"] == 1
    assert all(torch.equal(before[n], v) for n, v in model.named_parameters())


def test_hash_timeout_is_checked_inside_loop(tmp_path):
    path = tmp_path/"opaque-float"; path.write_bytes(b"a"*(2*1024**2))
    def fail(): raise TimeoutError("stop hashing")
    with pytest.raises(TimeoutError, match="hashing"): p.sha(path, fail)


@pytest.mark.parametrize("kind", ["hash", "runtime", "recipe"])
def test_run_rejects_before_backend_and_preserves_request(tmp_path, monkeypatch, kind):
    plan = {"version": p.VERSION, "recipe": copy.deepcopy(p.RECIPE), "runtime": p.common.base.runtime(), "no_retry": True}
    if kind == "runtime": plan["runtime"] = {"forged": True}
    if kind == "recipe": plan["recipe"]["batch_size"] = 99
    path = tmp_path/"plan.json"; p.write(path, plan)
    pin = "0"*64 if kind == "hash" else p.sha(path)
    monkeypatch.setattr(p, "backend", lambda: pytest.fail("Backend ran before technical admission"))
    out = tmp_path/"run"
    with pytest.raises(ValueError):
        p.run(SimpleNamespace(phase="run", plan=str(path), plan_sha256=pin, out=str(out)))
    failure = p.read(out/"failed.json")
    assert failure["request"]["plan_sha256"] == pin
    assert failure["progress"]["events_started"] == 0
    assert not (out/"completed.json").exists()
    with pytest.raises(FileExistsError):
        p.run(SimpleNamespace(phase="run", plan=str(path), plan_sha256=pin, out=str(out)))


def test_late_terminal_failure_is_demoted(tmp_path):
    args = SimpleNamespace(phase="run", out=str(tmp_path/"run"), plan_sha256="a"*64)
    with pytest.raises(RuntimeError, match="late"), p.attempt(args) as (out, _start, _progress, _check):
        p.write(out/"completed.json", {"status": "completed"})
        raise RuntimeError("late")
    assert (out/"late-completion.json").exists()
    assert p.read(out/"failed.json")["error"] == "RuntimeError('late')"


def test_byte_cap_rejects_terminal_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "RECIPE", {**p.RECIPE, "output_bytes": 1})
    with pytest.raises(ValueError, match="byte cap"):
        p.finish(tmp_path, p.time.perf_counter(), {}, lambda: None)
    assert not (tmp_path/"completed.json").exists()


def test_partial_microbatch_matches_one_full_weighted_update(monkeypatch):
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.linspace(-.1, .1, 384))

        def forward(self, candidates, candidate_mask, **unused):
            return (candidates @ self.weight).masked_fill(~candidate_mask, -torch.inf).log_softmax(-1)

    rows, schema, headers = tiny_metadata()
    rows = [p.public_row(r) for r in rows[:3]]
    arrays, syn, _ = p.synthetic_inputs({"feature_headers": headers}, schema, 410)
    models, records = [], []
    for micro in (2, 3):
        monkeypatch.setattr(p, "RECIPE", {**p.RECIPE, "microbatch_size": micro})
        model = Tiny()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        progress = dict.fromkeys(("events_started", "events_completed", "optimizer_attempted", "optimizer_returned",
                                 "forward_attempted", "forward_returned", "backward_attempted", "backward_returned"), 0)
        record = p.event(torch, model, optimizer, "flat_stratum", "train", rows, arrays,
                         {0: [0, 1, 2, 3]}, syn, np.array([2, 0, 1], np.int64),
                         np.array([.5, 2., 3.], np.float32), io.StringIO(), {}, progress, lambda: None)
        assert progress["optimizer_returned"] == 1
        models.append(model)
        records.append(record)
    assert records[0]["synthetic_weighted_loss"] == pytest.approx(records[1]["synthetic_weighted_loss"], abs=1e-6)
    torch.testing.assert_close(models[0].weight.grad, models[1].weight.grad, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(models[0].weight, models[1].weight, atol=1e-7, rtol=1e-5)
