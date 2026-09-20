"""Synthetic rows, cache arrays and tiny fake-scientific models only."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/study_dialogue_conditional.py"
SPEC = importlib.util.spec_from_file_location("conditional_study", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture():
    arrays = {"features": np.arange(20*384, dtype=np.float32).reshape(20, 384)/10000,
              "lexical": np.zeros(100, np.float32),
              "tokens": np.arange(9*384, dtype=np.float32).reshape(9, 384)/10000,
              "priors": np.concatenate([np.full(n, 1/n, np.float32) for n in (2, 3, 4)])}
    rows = []
    for i, (c, start, stop, lex, prior, target, bin_name, literal) in enumerate([
            (3, 0, 2, 0, 0, 0, "unmentioned_retention", 2),
            (4, 2, 5, 30, 2, 2, "assigned_retention", 3),
            (3, 5, 9, 70, 0, 2, "first_assignment", 0)]):
        arrays["lexical"][lex+10*literal+4] = 1.
        rows.append({"row_index": 10+i, "admission": "admitted", "split": "train",
                     "candidate_count": c, "previous_current_index": prior,
                     "current_label_index": target, "derived_bin": bin_name, "unseen": False,
                     "current_value_group": "other", "cache": {
                         "pooled_index": i, "query_feature_index": 6,
                         "candidate_feature_indices": list(range(7, 7+c)),
                         "token_start": start, "token_stop": stop,
                         "lexical_start": lex, "lexical_candidates": c, "lexical_stride": 10}})
    for array in arrays.values():
        array.flags.writeable = False
    return rows, arrays


@pytest.mark.parametrize("mode", m.MODES)
def test_actor_excludes_current_target_flags_and_uses_previous_value(mode):
    rows, arrays = fixture()
    actor, work = m.make_actor(rows, arrays, mode)
    changed = copy.deepcopy(rows)
    for row in changed:
        row.update(current_label_index=-1000, derived_bin="not-a-class", unseen="not-a-panel",
                   current_value_group="not-a-category", current_candidate_id="forbidden", future_label="forbidden")
    other, other_work = m.make_actor(changed, arrays, mode)
    assert work == other_work and actor.keys() == other.keys()
    for name in actor:
        assert np.array_equal(actor[name], other[name]), name
    changed[0]["previous_current_index"] = 2
    altered, _ = m.make_actor(changed, arrays, mode)
    assert altered["previous_onehot"][0].tolist() == [0., 0., 1., 0.]
    assert not np.array_equal(altered["previous_onehot"], actor["previous_onehot"])
    with pytest.raises(ValueError, match="Current target"):
        m.supervision(changed)


@pytest.mark.parametrize("mode", m.MODES)
def test_actor_padding_order_ownership_and_work_are_exact(mode):
    rows, arrays = fixture()
    chosen = [rows[2], rows[0]]
    actor, work = m.make_actor(chosen, arrays, mode)
    assert actor["candidate_mask"].shape == (2, 3) and actor["candidate_mask"].all()
    assert np.array_equal(actor["query"], arrays["features"][[6, 6]])
    assert np.array_equal(actor["candidates"][0], arrays["features"][[7, 8, 9]])
    assert work["rows"] == 2 and work["supported_candidate_positions"] == work["scorer_positions"] == 6
    assert work["float_input_scalars"] == sum(v.size for v in actor.values() if v.dtype == np.float32)
    assert work["boolean_input_bytes"] == sum(v.nbytes for v in actor.values() if v.dtype == bool)
    if mode == "mean":
        assert work["attention_score_positions"] == 0
        assert np.array_equal(actor["observation"], arrays["features"][[2, 0]])
        assert "token_prior" not in actor
    else:
        assert work["supported_token_positions"] == 6 and work["padded_token_positions"] == 8
        assert work["attention_score_positions"] == 24
        assert actor["token_mask"].tolist() == [[True]*4, [True, True, False, False]]
        assert np.all(actor["observation"][1, 2:] == 0) and np.all(actor["token_prior"][1, 2:] == 0)
    old = arrays["features"].copy()
    actor["candidates"][:] = -1
    assert np.array_equal(arrays["features"], old)


@pytest.mark.parametrize("field,bad", [("pooled_index", -1), ("query_feature_index", True),
                                       ("token_start", -1), ("token_stop", 10),
                                       ("lexical_start", -1), ("lexical_stride", 11)])
def test_negative_and_invalid_cache_addresses_fail_before_numpy_wraparound(field, bad):
    rows, arrays = fixture()
    rows[0]["cache"][field] = bad
    with pytest.raises(ValueError):
        m.make_actor(rows, arrays, "candidate")


@pytest.mark.parametrize("bad", [-1, True, 3, .5])
def test_previous_index_strict_range(bad):
    rows, arrays = fixture()
    rows[0]["previous_current_index"] = bad
    with pytest.raises(ValueError, match="Previous index"):
        m.make_actor(rows, arrays, "mean")


def test_objective_is_recomputed_from_admitted_training_rows():
    rows, _ = fixture()
    data = rows+[rows[0], rows[0]]
    value = m.objective(data)
    assert value["counts"] == [3, 1, 1]
    assert value["weights"] == [5/9, 5/3, 5/3]
    for n, weight in zip(value["counts"], value["weights"], strict=True):
        assert n*weight == pytest.approx(5/3)
    with pytest.raises(ValueError, match="Missing admitted"):
        m.objective(rows[:2])


def test_references_require_valid_literal_onehot_and_keep_global_ids():
    rows, arrays = fixture()
    refs = m.reference_arrays(rows, arrays["lexical"])
    assert refs["row_indices"].tolist() == [10, 11, 12]
    assert refs["previous_indices"].tolist() == [0, 2, 0]
    assert refs["literal_indices"].tolist() == [2, 3, 0]
    for bad in (0., .5, np.nan):
        lexical = arrays["lexical"].copy()
        lexical[24] = bad
        with pytest.raises(ValueError, match="Literal current"):
            m.reference_arrays(rows, lexical)


def test_headers_and_training_arrays_are_float32_readonly_memmaps(tmp_path, monkeypatch):
    _, arrays = fixture()
    names = {"features": "features.npy", "lexical": "lexical.npy", "tokens": "tokens.npy", "priors": "priors.npy"}
    paths = {}
    for key, value in arrays.items():
        paths[key] = tmp_path/names[key]
        np.save(paths[key], value)
    monkeypatch.setattr(m, "float_paths", lambda: paths)
    headers = m.headers()
    loaded = m.load_arrays({"feature_headers": headers})
    assert all(isinstance(v, np.memmap) and not v.flags.writeable for v in loaded.values())
    np.save(paths["priors"], np.ones(9, np.float64))
    with pytest.raises(ValueError, match="header shape/dtype"):
        m.load_arrays({"feature_headers": headers})


def test_work_schedules_count_partial_batches_and_preserve_epoch_order(monkeypatch):
    rows, _ = fixture()
    monkeypatch.setitem(m.CONFIG, "batch_size", 2)
    orders = np.array([[2, 0, 1], [1, 2, 0]], np.int64)
    for mode in m.MODES:
        result = m.work_schedule(rows, orders, mode)
        assert result["batches"] == 4 and result["totals"]["rows"] == 6
        assert result["totals"]["supported_candidate_positions"] == 20
    assert m.work_schedule(rows, orders, "slot") == m.work_schedule(rows, orders, "candidate")


def test_freeze_body_does_not_load_float_arrays_or_construct_models(tmp_path, monkeypatch):
    rows, arrays = fixture()
    out, prepared = tmp_path/"out", tmp_path/"prepared"
    out.mkdir(); prepared.mkdir()
    m.write(prepared/"completed.json", {"fixture": True})
    monkeypatch.setattr(m, "authenticate_prepared", lambda *_a, **_k: ({"plan_sha256": "parent", "files": {}}, {}))
    monkeypatch.setattr(m, "metadata", lambda *_a: ([], {"train": rows, "dev": rows}, {}))
    monkeypatch.setattr(m, "headers", lambda: {k: {"shape": list(v.shape)} for k, v in arrays.items()})
    monkeypatch.setattr(m, "source_map", lambda *_a: {})
    monkeypatch.setitem(m.CONFIG, "epochs", 2)
    monkeypatch.setitem(m.CONFIG, "batch_size", 2)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Numeric feature/model path reached freeze")
    monkeypatch.setattr(m, "backend", forbidden)
    monkeypatch.setattr(m, "load_arrays", forbidden)
    budget = m.Budget(out, m.time.monotonic())
    args = SimpleNamespace(prepared=prepared, prepared_sha256=m.sha(prepared/"completed.json"))
    done = m.freeze_body(out, budget, args)
    plan = m.read(out/"plan.json")
    assert done["model_calls"] == done["float_feature_arrays_decoded"] == 0
    assert plan["objective"]["counts"] == [1, 1, 1] and plan["updates_per_fit"] == 4
    for seed in m.SEEDS:
        saved = np.load(out/plan["orders"][str(seed)]["file"], allow_pickle=False)
        rng = np.random.default_rng(seed)
        assert np.array_equal(saved, np.stack([rng.permutation(3) for _ in range(2)]))
        assert m.sha(out/plan["orders"][str(seed)]["file"]) == plan["orders"][str(seed)]["sha256"]


def test_output_invariants_check_raw_support_and_mass():
    import torch
    mask = torch.tensor([[True, True, False]])
    scores = torch.tensor([[.25, .75, 0.]]).log()
    stats = m.invariants(scores, mask)
    assert stats["rows"] == 1 and stats["supported_candidates"] == 2
    assert stats["max_abs_mass_error"] < 2e-6
    for bad in (scores+1, torch.tensor([[-torch.inf, 0., -torch.inf]]), torch.zeros(1, 3)):
        with pytest.raises(ValueError):
            m.invariants(bad, mask)


@pytest.mark.parametrize("mutation", ["external_pin", "recipe", "extra_member", "payload"])
def test_frozen_plan_refuses_pin_recipe_closure_and_payload_corruption(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(m, "source_map", lambda *_a: {})
    m.write(tmp_path/"started.json", {"fixture": True})
    m.write(tmp_path/"prepared-completed.json", {"fixture": True})
    for seed in m.SEEDS:
        np.save(tmp_path/f"orders-{seed}.npy", np.array([[0]], np.int64))
    plan = {"version": m.VERSION, "config": copy.deepcopy(m.CONFIG), "limits": m.LIMITS,
            "runtime": m.runtime(), "expected_fits": m.EXPECTED_FITS, "source_sha256": {},
            "prepared_completed_sha256": m.sha(tmp_path/"prepared-completed.json")}
    if mutation == "recipe":
        plan["config"]["epochs"] += 1
    m.write(tmp_path/"plan.json", plan)
    pin = m.sha(tmp_path/"plan.json")
    m.write(tmp_path/"completed.json", {"status": "completed", "phase": "freeze", "plan_sha256": pin,
                                       "files": m.prep.manifest(tmp_path)})
    budget = m.Budget(tmp_path, m.time.monotonic())
    if mutation != "recipe":
        assert m.validate_plan(tmp_path/"plan.json", pin, budget) == plan
    if mutation == "external_pin":
        pin = "0"*64
    elif mutation == "extra_member":
        (tmp_path/"failed.json").write_text("{}")
    elif mutation == "payload":
        (tmp_path/"started.json").write_text("changed")
    with pytest.raises(ValueError):
        m.validate_plan(tmp_path/"plan.json", pin, budget)


def fake_training(tmp_path, monkeypatch):
    """Full runner control flow with tiny synthetic linear fake models, not the scientific scorer."""
    import torch
    rows, arrays = fixture()
    dev = copy.deepcopy(rows)
    for i, row in enumerate(dev):
        row.update(row_index=20+i, split="dev")
    seeds = (410, 411, 412)
    monkeypatch.setattr(m, "SEEDS", seeds)
    expected = [f"{mode}-{seed}" for seed, modes in zip(seeds, m.ARM_ORDERS, strict=True) for mode in modes]
    monkeypatch.setattr(m, "EXPECTED_FITS", expected)
    for key, value in {"seeds": list(seeds), "epochs": 1, "batch_size": 2, "threads": 1}.items():
        monkeypatch.setitem(m.CONFIG, key, value)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    parent, cache = tmp_path/"protocol", tmp_path/"cache"
    parent.mkdir(); cache.mkdir()
    paths = {}
    for key, value in arrays.items():
        filename = {"features": "features.npy", "lexical": "lexical.npy", "tokens": "tokens.npy", "priors": "priors.npy"}[key]
        paths[key] = cache/filename
        np.save(paths[key], value)
    monkeypatch.setattr(m, "float_paths", lambda: paths)
    orders = {}
    schedules = {}
    for seed in seeds:
        order = np.array([[2, 0, 1]], np.int64)
        name = f"orders-{seed}.npy"
        np.save(parent/name, order)
        orders[str(seed)] = {"file": name, "sha256": m.sha(parent/name), "shape": [1, 3]}
        for mode in m.MODES:
            schedules[f"{mode}-{seed}"] = {"training": m.work_schedule(rows, order, mode), "evaluation": m.work_schedule(dev, [range(3)], mode)}
    plan = {"source_sha256": {}, "runtime": m.runtime(), "prepared_path": str(tmp_path/"not-read"),
            "prepared_completed_sha256": "synthetic", "objective": m.objective(rows),
            "admitted_train_rows": 3, "admitted_dev_rows": 3, "updates_per_fit": 2,
            "evaluation_batches_per_fit": 2, "feature_headers": m.headers(), "orders": orders, "work_schedules": schedules}
    m.write(parent/"plan.json", plan)
    monkeypatch.setattr(m, "validate_plan", lambda *_a: plan)
    monkeypatch.setattr(m, "authenticate_prepared", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(m, "metadata", lambda *_a: ([], {"train": rows, "dev": dev}, {}))
    monkeypatch.setattr(m, "source_map", lambda *_a: {})
    class Tiny(torch.nn.Module):
        def __init__(self, mode, **_kwargs):
            super().__init__()
            self.mode = mode
            self.weight = torch.nn.Parameter(torch.tensor([.02, .03]))
            if mode != "mean":
                self.attention = torch.nn.Parameter(torch.tensor(.04))

        def forward(self, candidates, lexical, previous_onehot, candidate_mask, **_kwargs):
            logits = candidates[..., 0]*self.weight[0]+lexical[..., 4]*self.weight[1]+.1*previous_onehot
            if self.mode != "mean":
                logits = logits+candidates[..., 1]*self.attention
            return logits.masked_fill(~candidate_mask, -torch.inf).log_softmax(-1)

        def configuration(self):
            return {"synthetic_fake_model": True, "mode": self.mode}

    def copier(source, target, *, include_attention=False):
        with torch.no_grad():
            target.weight.copy_(source.weight)
            if include_attention:
                target.attention.copy_(source.attention)
    monkeypatch.setattr(m, "backend", lambda: (torch, Tiny, copier, ("weight",), ("attention",)))
    args = SimpleNamespace(phase="train", plan=parent/"plan.json", plan_sha256=m.sha(parent/"plan.json"), out=tmp_path/"run")
    return args, plan, expected


def test_nine_fit_fake_pipeline_pairs_initializers_orders_and_complete_prediction_membership(tmp_path, monkeypatch):
    args, plan, expected = fake_training(tmp_path, monkeypatch)
    result = m.execute(args)
    done = m.read(args.out/"completed.json")
    assert result == m.sha(args.out/"completed.json")
    assert done["completed_fits"] == expected and done["quality_metrics_computed"] is False
    assert done["progress"]["totals"]["optimizer_attempted"] == done["progress"]["totals"]["optimizer_returned"] == 18
    assert done["progress"]["totals"]["forward_returned"] == 36
    assert len(done["files"]) == 42
    for seed in m.SEEDS:
        records = [m.read(args.out/"fits"/f"{mode}-{seed}"/"completed.json") for mode in m.MODES]
        assert len({r["initial_common_sha256"] for r in records}) == 1
        assert records[1]["initial_attention_sha256"] == records[2]["initial_attention_sha256"]
        assert records[0]["initial_attention_sha256"] is None
        for mode, record in zip(m.MODES, records, strict=True):
            directory = args.out/"fits"/f"{mode}-{seed}"
            with np.load(directory/"dev-predictions.npz", allow_pickle=False) as saved:
                assert set(saved.files) == {"row_indices", "log_probs"}
                assert saved["row_indices"].tolist() == [20, 21, 22]
                assert saved["log_probs"].shape == (3, 12) and saved["log_probs"].dtype == np.float32
                assert np.isneginf(saved["log_probs"][:, 4:]).all()
            events = [json.loads(line) for line in (directory/"updates.jsonl").read_text().splitlines()]
            assert [e["row_indices"] for e in events] == [[12, 10], [11]]
            assert record["orders_sha256"] == plan["orders"][str(seed)]["sha256"]
            assert record["training_normalization"]["rows"] == 3 and record["evaluation"]["rows"] == 3
    with pytest.raises(FileExistsError):
        m.execute(args)


def test_partial_evaluation_and_original_error_are_preserved(tmp_path, monkeypatch):
    args, _, _ = fake_training(tmp_path, monkeypatch)
    original = m.make_actor
    def broken(rows, arrays, mode):
        if rows[0]["row_index"] == 22:
            raise RuntimeError("synthetic evaluation failure")
        return original(rows, arrays, mode)
    monkeypatch.setattr(m, "make_actor", broken)
    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        m.execute(args)
    assert not (args.out/"completed.json").exists()
    failed = m.read(args.out/"failed.json")
    assert failed["progress"]["completed_fits"] == []
    active = args.out/"fits"/"mean-410"
    assert (active/"partial-weights.pt").exists()
    with np.load(active/"partial-dev-predictions.npz", allow_pickle=False) as saved:
        assert saved["row_indices"].tolist() == [20, 21]
    assert failed["progress"]["totals"]["evaluation_rows"] == 2


def test_failed_receipt_error_does_not_replace_original_exception(tmp_path, monkeypatch):
    args = SimpleNamespace(phase="freeze", out=tmp_path/"run")
    error = RuntimeError("original")
    def body(*_args):
        raise error
    original = m.write
    def write(path, value):
        if Path(path).name == "failed.json":
            raise OSError("receipt unavailable")
        original(path, value)
    monkeypatch.setattr(m, "freeze_body", body)
    monkeypatch.setattr(m, "write", write)
    with pytest.raises(RuntimeError, match="original") as caught:
        m.execute(args)
    assert caught.value is error and "receipt unavailable" in " ".join(error.__notes__)


def test_completion_is_demoted_when_final_hash_cap_check_fails(tmp_path, monkeypatch):
    args = SimpleNamespace(phase="freeze", out=tmp_path/"run")
    monkeypatch.setattr(m, "freeze_body", lambda *_args: {"model_calls": 0})
    original = m.sha
    def sha(path, check=lambda: None):
        if Path(path) == args.out/"completed.json":
            raise TimeoutError("late cap")
        return original(path, check)
    monkeypatch.setattr(m, "sha", sha)
    with pytest.raises(TimeoutError, match="late cap"):
        m.execute(args)
    assert not (args.out/"completed.json").exists()
    assert (args.out/"late-completion.json").exists() and (args.out/"failed.json").exists()
