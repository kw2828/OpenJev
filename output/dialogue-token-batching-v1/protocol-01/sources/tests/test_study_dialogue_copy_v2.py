"""Tiny synthetic monitor parity and fake orchestration, no corpus or fit."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location("study_dialogue_copy_v2_tested", ROOT / "scripts/study_dialogue_copy_v2.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
finally:
    sys.path.pop(0)


@pytest.fixture(autouse=True)
def synthetic_scope():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(threads)


def fixture():
    features = np.arange(12*384, dtype=np.float32).reshape(12, 384)/10000
    queries = [{"text": 4, "candidates": [6, 7, 8]}, {"text": 5, "candidates": [6, 7, 8, 9]}]
    ds = [{"id": "a", "turns": [0, 1, 2], "queries": [
        {"query": 1, "time": 0, "label": 2, "bin": "first_assignment", "unseen": False},
        {"query": 0, "time": 1, "label": 1, "bin": "first_assignment", "unseen": True},
        {"query": 1, "time": 2, "label": 2, "bin": "assigned_retention", "unseen": False}],
        "layout": {"shape": [3, 2, 4, 10], "query_ids": [0, 1], "offset": 0}},
        {"id": "b", "turns": [3], "queries": [
        {"query": 0, "time": 0, "label": 0, "bin": "unmentioned_retention", "unseen": True}],
        "layout": {"shape": [1, 1, 3, 10], "query_ids": [0], "offset": 240}}]
    return ds, queries, features, np.zeros(270, np.float32)


def assert_coverage(audit, *, forwards=1, mass=True):
    assert audit["forward_calls"] == audit["forward_returned"] == forwards
    assert audit["advance_calls"] == audit["advance_returned"] == 3*forwards
    assert audit["valid_turns"] == 4*forwards
    assert audit["executed_valid_question_slots"] == 8*forwards
    assert audit["real_question_updates"] == 7*forwards
    for key in ("incoming_checks", "feature_checks", "result_checks"):
        assert audit[key] == 8*forwards
    assert audit["mass_checks"] == (8*forwards if mass else 0)
    assert all(audit[k] <= 2e-6 for k in study.MAX_KEYS)


@pytest.mark.parametrize("method", study.old.METHODS)
def test_monitor_leaves_actual_v2_outputs_and_parameter_gradients_unchanged(method):
    kwargs = {"projection_dim": 4, "hidden_dim": 7, "gru_width": 3}
    plain = study.DialogueCopyMemoryV2(method, **kwargs)
    watched = study.MonitoredCopyMemoryV2(method, **kwargs)
    watched.load_state_dict(plain.state_dict())
    ds, *rest = fixture()
    actor, labels, _ = study.old.make_batch(ds, *rest)
    before = [x.clone() for x in actor]
    watched.begin_batch(actor, ds)
    expected, actual = plain(*actor), watched(*actor)
    assert torch.equal(actual, expected)
    eligible = labels != -100
    expected_loss = torch.nn.functional.cross_entropy(expected[eligible], labels[eligible])
    actual_loss = torch.nn.functional.cross_entropy(actual[eligible], labels[eligible])
    assert torch.isfinite(expected_loss) and torch.isfinite(actual_loss)
    expected_loss.backward()
    actual_loss.backward()
    for (name, a), (_, b) in zip(plain.named_parameters(), watched.named_parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None, name
        else:
            assert torch.equal(a.grad, b.grad), name
    assert all(torch.equal(a, b) for a, b in zip(actor, before, strict=True))
    assert_coverage(watched.audit, mass=method in ("scalar", "selective", "selective_no_lexical"))
    assert watched.configuration()["implementation_version"] == study.VERSION
    assert watched.configuration()["class"] == "MonitoredCopyMemoryV2"


def test_audit_coverage_uses_public_layout_not_gold_eligibility():
    ds, queries, features, lexical = fixture()
    net = study.MonitoredCopyMemoryV2("scalar", projection_dim=4, hidden_dim=7, gru_width=3)
    actor, _, _ = study.old.make_batch(ds, queries, features, lexical)
    net.begin_batch(actor, ds)
    result = net(*actor)
    first = dict(net.audit)
    mutated = copy.deepcopy(ds)
    mutated[0]["queries"].pop(1)
    for d in mutated:
        for row in d["queries"]:
            row["label"] = 1
            row["bin"] = "revision"
    actor2, _, _ = study.old.make_batch(mutated, queries, features, lexical)
    net.begin_batch(actor2, mutated)
    assert torch.equal(result, net(*actor2))
    assert first == net.audit
    assert first["real_question_updates"] == 7  # Only4 original labels, then3.


def test_raw_prior_failure_is_not_hidden_by_v2_or_final_softmax():
    ds, *rest = fixture()
    actor, _, _ = study.old.make_batch(ds, *rest)
    net = study.MonitoredCopyMemoryV2("scalar", projection_dim=4, hidden_dim=7, gru_width=3)
    net.begin_batch(actor, ds)
    original = net.initial
    def corrupt(mask, none_index=None):
        state = original(mask, none_index)
        state["log_b"][..., 0] = .01
        return state
    net.initial = corrupt
    with pytest.raises(ValueError, match="incoming normalization"):
        net(*actor)
    assert net.audit["forward_calls"] == 1 and net.audit["forward_returned"] == 0
    assert net.audit["feature_checks"] == 0


def test_actual_feature_hook_detects_bad_probability_feature_without_mutating_input():
    net = study.MonitoredCopyMemoryV2("scalar", projection_dim=4, hidden_dim=7, gru_width=3)
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    net.active_check = mask, torch.ones(1, 1, dtype=torch.bool)
    features = torch.zeros(1, 1, 2, 36)
    features[..., -2] = 1.
    old = features.clone()
    with pytest.raises(ValueError, match="feature normalization"):
        net._feature_check(None, (features,))
    assert torch.equal(features, old)


def test_evaluation_retains_membership_and_exposes_all_batch_coverage(tmp_path):
    net = study.MonitoredCopyMemoryV2("selective", projection_dim=4, hidden_dim=7, gru_width=3)
    progress = {}
    receipt = study.evaluate(net, *fixture(), tmp_path / "p.npz", lambda: None, progress)
    assert receipt["queries"] == 4 and len(receipt["batches"]) == 1
    assert receipt["batches"][0]["indices"] == [0, 1]
    assert_coverage(receipt["invariants"])
    assert receipt["actor_shapes"]["real_question_steps"] == 7
    with np.load(tmp_path / "p.npz", allow_pickle=False) as arrays:
        assert arrays["dialogue"].tolist() == ["a", "a", "a", "b"]
        assert arrays["query"].tolist() == [1, 0, 1, 0]
    assert "_evaluation_arrays" not in progress


def fake_lifecycle(monkeypatch, tmp_path, *, fail_method=None, bad_initial=None):
    ds, queries, features, lexical = fixture()
    root = tmp_path / "run"
    root.mkdir()
    study.write(root / "plan.json", {"synthetic": True})
    args = SimpleNamespace(out=root, packet=tmp_path, lexical=tmp_path, old_study=tmp_path,
                           plan_sha256=study.sha(root / "plan.json"))
    config = {**study.CONFIG, "epochs": 1}
    monkeypatch.setattr(study, "CONFIG", config)
    originals = {f"{m}-{s}": {"initial_tensors_sha256": "bad" if m == bad_initial else "paired"}
                 for s in study.old.SEEDS for m in study.old.METHODS}
    monkeypatch.setattr(study, "validate", lambda *a: ({"loss_weights": [1., 1., 1.], "optimizer_updates_per_fit": 1}, originals))
    monkeypatch.setattr(study.old, "load_inputs", lambda *a: ({"train": ds, "dev": ds}, queries, features, lexical))
    monkeypatch.setattr(study.old, "tensor_digest", lambda *a: "paired")
    def references(ds, queries, path):
        study.save_npz(path, {"synthetic": np.zeros(1)})
        return {"sha256": study.sha(path)}
    monkeypatch.setattr(study.old, "reference_predictions", references)
    events = []
    monkeypatch.setattr(study.torch, "manual_seed", lambda seed: events.append(("init_seed", seed)))
    monkeypatch.setattr(study.np.random, "default_rng", lambda seed: SimpleNamespace(permutation=lambda n: np.arange(n)[::-1]))
    monkeypatch.setattr(study.torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(study.torch, "set_num_interop_threads", lambda *a: None)
    monkeypatch.setattr(study.torch, "use_deterministic_algorithms", lambda *a: None)

    class FakeModel(torch.nn.Module):
        def __init__(self, method, **kwargs):
            super().__init__()
            events.append(("construct", method))
            if method == fail_method:
                raise ValueError("synthetic constructor failure")
            self.method = method
            self.weight = torch.nn.Parameter(torch.tensor(.1))
            self.audit = study.empty_invariants()

        def begin_batch(self, actor, dialogues):
            self.audit = study.empty_invariants()
            _, t = actor[1].shape
            q = actor[4].shape[1]
            n = int(actor[1].sum())*q
            self.audit.update(forward_calls=1, forward_returned=1, advance_calls=t, advance_returned=t,
                              valid_turns=int(actor[1].sum()), executed_valid_question_slots=n,
                              real_question_updates=sum(d["layout"]["shape"][0]*d["layout"]["shape"][1] for d in dialogues),
                              incoming_checks=n, feature_checks=n, result_checks=n)

        def forward(self, turns, valid, query, candidates, mask, lexical):
            events.append(("forward", self.method, self.training))
            values = self.weight*torch.arange(mask.shape[-1])
            output = values.expand(turns.shape[0], turns.shape[1], mask.shape[1], -1)
            return output.masked_fill(~mask[:, None], -torch.inf).log_softmax(-1)

        def configuration(self):
            return {"method": self.method, "implementation_version": study.VERSION, "synthetic": True}

    class FakeOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameters = list(parameters)
            events.append(("optimizer", kwargs))
        def zero_grad(self, **kwargs):
            for p in self.parameters:
                p.grad = None
        def step(self):
            events.append(("step",))  # Deliberately no parameter update.

    monkeypatch.setattr(study, "MonitoredCopyMemoryV2", FakeModel)
    monkeypatch.setattr(study.torch.optim, "AdamW", FakeOptimizer)
    return args, events


def test_fake_all15_fits_full_closure_ledger_and_common_pairing(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path)
    result = study.train(args)
    receipt = study.read(args.out / "completed.json")
    assert result["completed_sha256"] == study.sha(args.out / "completed.json")
    assert len(receipt["fits"]) == 15 and len(receipt["files"]) == 63
    assert len(list(args.out.rglob("*.*"))) == 64
    assert len([e for e in events if e[0] == "step"]) == 15
    assert [r["method"] for r in receipt["fits"]] == study.old.METHODS*3
    for r in receipt["fits"]:
        path = args.out / "fits" / f"{r['method']}-{r['seed']}" / "batches.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["indices"] == [1, 0]
        assert rows[0]["supervised_queries"] == 4 and rows[0]["update"] == 1
        assert r["initial_tensors_sha256"] == r["original_initial_tensors_sha256"] == r["common_initial_tensors_sha256"]
        assert r["training_queries"] == 4 and r["updates"] == 1
    for name, metadata in receipt["files"].items():
        assert study.sha(args.out / name) == metadata["sha256"]
    with pytest.raises(ValueError, match="already attempted"):
        study.train(args)


def test_constructor_failure_does_not_reuse_previous_fits_work(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path, fail_method="scalar")
    with pytest.raises(ValueError, match="constructor failure"):
        study.train(args)
    failed = study.read(args.out / "failed.json")
    assert len(failed["completed_fits"]) == 1
    assert failed["active_progress"]["fit"] == "scalar-4101"
    assert failed["active_progress"]["completed_optimizer_steps"] == 0
    assert len([e for e in events if e[0] == "step"]) == 1


def test_wrong_original_initialization_fails_before_optimizer(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path, bad_initial="readout")
    with pytest.raises(ValueError, match="Original initialization"):
        study.train(args)
    assert not any(e[0] in ("optimizer", "step", "forward") for e in events)
    assert (args.out / "fits/readout-4101/partial-weights.pt").exists()


def test_failure_after_returned_optimizer_step_preserves_count_and_flushed_prefix(tmp_path, monkeypatch):
    args, _ = fake_lifecycle(monkeypatch, tmp_path)
    def failed_evaluation(*args):
        raise RuntimeError("synthetic evaluation failure")
    monkeypatch.setattr(study, "evaluate", failed_evaluation)
    with pytest.raises(RuntimeError, match="evaluation failure"):
        study.train(args)
    failed = study.read(args.out / "failed.json")
    assert failed["active_progress"]["completed_optimizer_steps"] == 1
    assert failed["active_progress"]["completed_supervised_queries"] == 4
    assert len((args.out / "fits/readout-4101/batches.jsonl").read_text().splitlines()) == 1
    assert (args.out / "fits/readout-4101/partial-weights.pt").exists()


def test_started_and_failure_receipt_errors_preserve_original(tmp_path, monkeypatch):
    args, _ = fake_lifecycle(monkeypatch, tmp_path)
    original = OSError("started write failed")
    def broken(path, data):
        if Path(path).name == "started.json":
            raise original
        raise OSError("secondary failure receipt")
    monkeypatch.setattr(study, "write", broken)
    monkeypatch.setattr(study.old, "write", broken)
    with pytest.raises(OSError) as caught:
        study.train(args)
    assert caught.value is original
    assert any("secondary failure receipt" in n for n in original.__notes__)


def test_late_completion_cap_demotes_and_completed_work_is_not_active_twice(tmp_path, monkeypatch):
    args, _ = fake_lifecycle(monkeypatch, tmp_path)
    clock = [0.]
    monkeypatch.setattr(study.time, "perf_counter", lambda: clock[0])
    write = study.write
    def delayed(path, data):
        write(path, data)
        if Path(path) == args.out / "completed.json":
            clock[0] = study.CAP+1
    monkeypatch.setattr(study, "write", delayed)
    with pytest.raises(TimeoutError):
        study.train(args)
    assert not (args.out / "completed.json").exists()
    assert (args.out / "invalid-completion.json").exists()
    failed = study.read(args.out / "failed.json")
    assert len(failed["completed_fits"]) == 15 and failed["active_progress"] == {}


def test_preflight_cap_failure_happens_before_model_construction(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path)
    clock = [0.]
    monkeypatch.setattr(study.time, "perf_counter", lambda: clock[0])
    loader = study.old.load_inputs
    def delayed(*args):
        result = loader(*args)
        clock[0] = study.CAP+1
        return result
    monkeypatch.setattr(study.old, "load_inputs", delayed)
    with pytest.raises(TimeoutError):
        study.train(args)
    assert events == [] and (args.out / "failed.json").exists()


def test_partial_prediction_write_failure_keeps_original_error(tmp_path, monkeypatch):
    error = ValueError("primary invariant failure")
    progress = {"_evaluation_arrays": {"labels": [1], "probabilities": [np.array([.3, .7])]}, "evaluation_rows": 1}
    def bad(*args):
        raise OSError("partial I/O")
    monkeypatch.setattr(study, "save_npz", bad)
    study.preserve_failure(tmp_path, error, [], progress, None, tmp_path, study.time.perf_counter())
    assert any("partial I/O" in n for n in error.__notes__)
    receipt = study.read(tmp_path / "failed.json")
    assert receipt["active_progress"] == {"evaluation_rows": 1}


@pytest.mark.parametrize("key,bad", [("loss_weights", [2., 1., 1.]), ("loss_counts", {"0": 99}),
                                      ("optimizer_updates_per_fit", 1279)])
def test_rehashed_plan_cannot_change_original_objective_or_update_budget(tmp_path, monkeypatch, key, bad):
    prior = {"loss_weights": [1., 1., 1.], "loss_counts": {"0": 1, "1": 1, "2": 1},
             "optimizer_updates_per_fit": 1280, "packet_completed_sha256": "p", "lexical_completed_sha256": "l"}
    plan = {**prior, "study": study.STUDY, "implementation_version": study.VERSION,
            "config": study.CONFIG, "practical_checks": study.CHECKS, "runtime": study.runtime(),
            "wall_cap_seconds": study.CAP, "normalization_tolerance": study.TOLERANCE,
            "source_sha256": dict.fromkeys(study.SOURCES, "synthetic"), "paths": {},
            "old_plan_sha256": study.OLD_PLAN, "old_completed_sha256": study.OLD_COMPLETED}
    path = tmp_path / "plan.json"
    study.write(path, plan)
    args = SimpleNamespace(out=tmp_path, plan_sha256=study.sha(path), old_study=tmp_path, packet=tmp_path, lexical=tmp_path)
    # Only immutable-source/receipt byte validators are stubbed; objective guard
    # below is the actual runner's validation after the external plan is read.
    monkeypatch.setattr(study, "file_check", lambda *a: None)
    monkeypatch.setattr(study, "path_map", lambda *a: {})
    monkeypatch.setattr(study, "old_identity", lambda *a: (prior, {}))
    monkeypatch.setattr(study.old, "authenticate", lambda *a: None)
    study.validate(args)
    plan[key] = bad
    path.write_text(json.dumps(plan))
    args.plan_sha256 = study.sha(path)
    with pytest.raises(ValueError, match="objective/update budget"):
        study.validate(args)
