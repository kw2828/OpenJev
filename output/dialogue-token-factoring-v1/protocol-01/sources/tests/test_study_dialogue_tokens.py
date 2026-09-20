"""Tiny public-input and lifecycle tests; no encoder, corpus or real fit."""
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
    spec = importlib.util.spec_from_file_location("study_token_test", ROOT / "scripts/study_dialogue_tokens.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
finally:
    sys.path.pop(0)


@pytest.fixture(autouse=True)
def engineering_only():
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
    offsets = np.array([0, 2, 5, 6, 10], np.int64)
    tokens = np.arange(10*384, dtype=np.float32).reshape(10, 384)/10000
    priors = np.concatenate([np.full(n, 1/n, np.float32) for n in np.diff(offsets)])
    for d in ds:
        d["token_contexts"] = np.asarray(d["turns"], np.int64)
    return ds, queries, features, np.zeros(270, np.float32), (tokens, offsets, priors)


@pytest.mark.parametrize("head", ["readout", "scalar"])
@pytest.mark.parametrize("mode", ["slot", "candidate"])
def test_existing_monitor_preserves_tokens_adapter_outputs_and_parameter_gradients(head, mode):
    ds, queries, features, lexical, tokens = fixture()
    actor, labels, _ = study.make_batch(ds, queries, features, lexical, tokens, mode)
    plain = study.DialogueTokenMemory(head, attention_mode=mode, projection_dim=4, hidden_dim=7, gru_width=3)
    watched = study.MonitoredTokenMemory(head, attention_mode=mode, projection_dim=4, hidden_dim=7, gru_width=3)
    watched.load_state_dict(plain.state_dict())
    watched.begin_batch(actor, ds)
    expected, actual = plain(*actor), watched(*actor)
    assert torch.equal(expected, actual)
    eligible = labels != -100
    for output in (expected, actual):
        torch.nn.functional.cross_entropy(output[eligible], labels[eligible]).backward()
    for (name, a), (_, b) in zip(plain.named_parameters(), watched.named_parameters(), strict=True):
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad), name
    assert watched.audit["real_question_updates"] == 7
    assert watched.audit["executed_valid_question_slots"] == watched.audit["feature_checks"] == 8
    assert watched.audit["mass_checks"] == (8 if head == "scalar" else 0)
    assert watched.configuration()["class"] == "MonitoredTokenMemory"
    assert watched.configuration()["normalized_parent_version"] == study.v2.VERSION


def test_token_assembly_shares_context_without_candidate_repeats_and_counts_work():
    ds, queries, features, lexical, cache = fixture()
    actor, labels, bins = study.make_batch(ds, queries, features, lexical, cache, "slot")
    other, labels2, bins2 = study.make_batch(ds, queries, features, lexical, cache, "candidate")
    assert torch.equal(labels, labels2) and torch.equal(bins, bins2)
    assert all(torch.equal(a, b) for a, b in zip(actor, other, strict=True))
    assert actor[0].shape == (2, 3, 4, 384)
    assert torch.equal(actor[0][0, 1, :3], torch.from_numpy(cache[0][2:5]))
    assert (actor[0][1, 1:] == 0).all() and not actor[6][1, 1:].any()
    assert (actor[7][~actor[6]] == 0).all()
    work = study.observation_work(ds, queries, actor)
    assert work["turn_projection_positions"] == 48
    assert work["emitted_bytes"] == 2*3*4*384*4
    assert work["raw_cache_token_bytes_read"] == 10*384*4
    assert work["valid_token_positions"] == 10 and work["padded_token_positions"] == 24
    assert work["pooling_score_positions"] == 192 and work["pooling_evidence_positions"] == 48
    assert work["evidence_query_projection_positions"] == 16
    assert work["real_candidate_updates"] == 24


def test_labels_and_scoring_eligibility_do_not_select_tokens_updates():
    ds, queries, features, lexical, tokens = fixture()
    original, _, _ = study.make_batch(ds, queries, features, lexical, tokens, "candidate")
    changed = copy.deepcopy(ds)
    changed[0]["queries"].pop(1)  # Still keep this public supplied question's full stream.
    for d in changed:
        for row in d["queries"]:
            row["label"] = 1
            row["bin"] = "revision"
            row["unseen"] = not row["unseen"]
    other, _, _ = study.make_batch(changed, queries, features, lexical, tokens, "candidate")
    assert all(torch.equal(a, b) for a, b in zip(original, other, strict=True))
    assert study.observation_work(ds, queries, original) == study.observation_work(changed, queries, other)


@pytest.mark.parametrize("mutation", ["missing_turn", "out_of_range", "foreign_feature", "foreign_query", "nonfinite", "extra_index", "prior"])
def test_cache_alignment_rejects_public_context_or_payload_corruption(mutation):
    ds, queries, _, _, (tokens, offsets, priors) = fixture()
    entries = [{"id": d["id"], "offset": sum(len(x["turns"]) for x in ds[:i]),
                "shape": [len(d["turns"])], "query_ids": list(d["layout"]["query_ids"])} for i, d in enumerate(ds)]
    index = {"cohorts": {"train": entries, "dev": []}, "original_feature_indices": list(range(4))}
    ix = np.arange(4, dtype=np.int64)
    cohorts = {"train": ds, "dev": []}
    study.align_tokens(cohorts, queries, index, tokens, offsets, priors, ix)
    if mutation == "missing_turn":
        index["cohorts"]["train"][0]["shape"][0] -= 1
    elif mutation == "out_of_range":
        ix[0] = 4
    elif mutation == "foreign_feature":
        index["original_feature_indices"][0] = 99
    elif mutation == "foreign_query":
        index["cohorts"]["train"][0]["query_ids"].reverse()
    elif mutation == "nonfinite":
        tokens[0, 0] = np.nan
    elif mutation == "prior":
        priors[0] = .2
    else:
        ix = np.append(ix, 0)
    with pytest.raises(ValueError):
        study.align_tokens(cohorts, queries, index, tokens, offsets, priors, ix)


def test_configuration_preserves_rng_parent_initializer_and_whole_four_arm_pairing():
    rng = torch.random.get_rng_state().clone()
    configs = study.expected_configurations()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(configs) == set(study.ARMS)
    assert {x["parameters"] for x in configs.values()} == {173186}
    assert {x["parent_parameters"] for x in configs.values()} == {99458}
    whole = []
    for arm in study.ARMS:
        mode, head = arm.rsplit("_", 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(410)
            parent = study.v2.MonitoredCopyMemoryV2(head)
            torch.manual_seed(410)
            child = study.MonitoredTokenMemory(head, attention_mode=mode)
        assert study.old.tensor_digest(parent) == study.parent_initial_digest(child)
        whole.append(study.old.tensor_digest(child))
        assert configs[arm]["attention_mode"] == mode
    assert len(set(whole)) == 1


def fake_lifecycle(monkeypatch, tmp_path, *, bad_initial=False):
    ds, queries, features, lexical, tokens = fixture()
    run = tmp_path / "run"
    run.mkdir()
    study.write(run / "plan.json", {"synthetic": True})
    args = SimpleNamespace(out=run, packet=tmp_path, lexical=tmp_path, tokens=tmp_path, old_study=tmp_path,
        plan_sha256=study.sha(run / "plan.json"), tokens_completed_sha256="fake")
    monkeypatch.setattr(study, "CONFIG", {**study.CONFIG, "epochs": 1})
    originals = {f"{m}-{s}": {"initial_tensors_sha256": "wrong" if bad_initial else "parent"}
                 for m in ("readout", "scalar") for s in study.SEEDS}
    configuration = lambda method, mode: {"method": method, "attention_mode": mode, "implementation_version": study.VERSION, "synthetic": True}
    plan = {"loss_weights": [1., 1., 1.], "optimizer_updates_per_fit": 1, "tokens_completed_sha256": "fake",
            "expected_configurations": {a: configuration(a.rsplit("_", 1)[1], a.rsplit("_", 1)[0]) for a in study.ARMS}}
    monkeypatch.setattr(study, "validate", lambda *a: (plan, originals))
    monkeypatch.setattr(study, "load_inputs", lambda *a: ({"train": ds, "dev": ds}, queries, features, lexical, tokens))
    monkeypatch.setattr(study.old, "tensor_digest", lambda *a: "paired")
    monkeypatch.setattr(study, "parent_initial_digest", lambda *a: "parent")
    events = []
    monkeypatch.setattr(study.torch, "manual_seed", lambda s: events.append(("init", s)))
    monkeypatch.setattr(study.np.random, "default_rng", lambda s: SimpleNamespace(permutation=lambda n: np.arange(n)[::-1]))
    for name in ("set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms"):
        monkeypatch.setattr(study.torch, name, lambda *a: None)
    def references(ds, queries, path):
        study.save_npz(path, {"synthetic": np.zeros(1)})
        return {"sha256": study.sha(path)}
    monkeypatch.setattr(study.old, "reference_predictions", references)
    class FakeModel(torch.nn.Module):
        def __init__(self, method, **kwargs):
            super().__init__()
            events.append(("construct", method))
            self.method = method
            self.mode = kwargs["attention_mode"]
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
        def forward(self, turns, valid, query, candidates, mask, lexical, token_mask, token_prior):
            events.append(("forward", self.method, self.training, turns.ndim))
            output = (self.weight*torch.arange(mask.shape[-1])).expand(turns.shape[0], turns.shape[1], mask.shape[1], -1)
            return output.masked_fill(~mask[:, None], -torch.inf).log_softmax(-1)
        def configuration(self):
            return configuration(self.method, self.mode)
    class FakeOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameters = list(parameters)
            events.append(("optimizer", kwargs))
        def zero_grad(self, **kwargs):
            for p in self.parameters:
                p.grad = None
        def step(self):
            events.append(("step",))  # No parameter update or actual optimizer.
    monkeypatch.setattr(study, "MonitoredTokenMemory", FakeModel)
    monkeypatch.setattr(study.torch.optim, "AdamW", FakeOptimizer)
    return args, events


def test_all12_fake_fits_strict_membership_pairing_orders_and_observation_cost(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path)
    result = study.train(args)
    done = study.read(args.out / "completed.json")
    assert result["completed_sha256"] == study.sha(args.out / "completed.json")
    assert len(done["fits"]) == 12 and len(done["files"]) == 51
    assert type(done["process_lifetime_peak_rss_bytes"]) is int and done["process_lifetime_peak_rss_bytes"] > 0
    assert done["process_memory_scope"] == study.MEMORY_SCOPE
    assert set(done["files"])|{"completed.json"} == study.expected_members()
    assert [r["method"] for r in done["fits"]] == list(study.ARMS)*3
    assert sum(e[0] == "step" for e in events) == 12
    for r in done["fits"]:
        assert 0 < r["process_lifetime_peak_rss_bytes"] <= done["process_lifetime_peak_rss_bytes"]
        assert r["observation_work_scope"] == study.OBSERVATION_WORK_SCOPE
        assert r["initial_tensors_sha256"] == r["common_initial_tensors_sha256"] == "paired"
        assert r["parent_initial_tensors_sha256"] == r["original_initial_tensors_sha256"] == "parent"
        folder = args.out / "fits" / f"{r['method']}-{r['seed']}"
        ledger = [json.loads(x) for x in (folder / "batches.jsonl").read_text().splitlines()]
        assert len(ledger) == 1 and ledger[0]["indices"] == [1, 0]
        assert ledger[0]["observation_work"] == r["training_observation_work"]
        assert ledger[0]["supervised_queries"] == r["training_queries"] == 4
        assert r["evaluation"]["observation_work"]["turn_projection_positions"] == 48
    for name, entry in done["files"].items():
        assert study.sha(args.out / name) == entry["sha256"]
    with pytest.raises(ValueError, match="already attempted"):
        study.train(args)


def test_wrong_pin_fails_before_auth_or_construction(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path)
    args.plan_sha256 = "bad"
    with pytest.raises(ValueError, match="Changed file"):
        study.train(args)
    assert events == [] and {p.name for p in args.out.iterdir()} == {"plan.json"}


def test_wrong_initializer_fails_before_optimizer_or_any_forward(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path, bad_initial=True)
    with pytest.raises(ValueError, match="V2 initialization"):
        study.train(args)
    assert not any(e[0] in ("optimizer", "step", "forward") for e in events)
    assert (args.out / "fits/slot_readout-4101/partial-weights.pt").exists()


def test_later_evaluation_failure_preserves_paid_optimizer_prefix(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path)
    def fail(*args):
        raise RuntimeError("evaluation injected")
    monkeypatch.setattr(study, "evaluate", fail)
    with pytest.raises(RuntimeError, match="evaluation injected"):
        study.train(args)
    fail = study.read(args.out / "failed.json")
    assert fail["active_progress"]["completed_optimizer_steps"] == 1
    assert fail["active_progress"]["completed_supervised_queries"] == 4
    assert len(fail["completed_fits"]) == 0 and sum(e[0] == "step" for e in events) == 1
    assert (args.out / "fits/slot_readout-4101/batches.jsonl").stat().st_size > 0


def test_authentication_time_is_inside_cap_and_no_model_is_constructed(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path)
    clock = [0.]
    monkeypatch.setattr(study.time, "perf_counter", lambda: clock[0])
    load = study.load_inputs
    def delayed(*a):
        result = load(*a)
        clock[0] = study.CAP+1
        return result
    monkeypatch.setattr(study, "load_inputs", delayed)
    with pytest.raises(TimeoutError):
        study.train(args)
    assert events == [] and (args.out / "failed.json").exists()


def test_late_cap_demotes_completion_and_does_not_duplicate_last_fit(tmp_path, monkeypatch):
    args, _ = fake_lifecycle(monkeypatch, tmp_path)
    clock = [0.]
    monkeypatch.setattr(study.time, "perf_counter", lambda: clock[0])
    write = study.write
    def delayed(path, payload):
        write(path, payload)
        if Path(path) == args.out / "completed.json":
            clock[0] = study.CAP+1
    monkeypatch.setattr(study, "write", delayed)
    with pytest.raises(TimeoutError):
        study.train(args)
    assert not (args.out / "completed.json").exists() and (args.out / "invalid-completion.json").exists()
    failed = study.read(args.out / "failed.json")
    assert len(failed["completed_fits"]) == 12 and failed["active_progress"] == {}


def test_started_failure_and_secondary_receipt_error_preserve_original(tmp_path, monkeypatch):
    args, _ = fake_lifecycle(monkeypatch, tmp_path)
    original = OSError("primary started error")
    def broken(path, data):
        if Path(path).name == "started.json":
            raise original
        raise OSError("secondary disk error")
    monkeypatch.setattr(study, "write", broken)
    monkeypatch.setattr(study.old, "write", broken)
    with pytest.raises(OSError) as error:
        study.train(args)
    assert error.value is original and any("secondary disk" in n for n in original.__notes__)


@pytest.mark.parametrize("mutation", ["external_pin", "packet_lineage", "test_scope"])
def test_cache_external_identity_and_parent_scope_before_loading_payload(tmp_path, monkeypatch, mutation):
    parent = {"packet_completed_sha256": "packet", "lexical_completed_sha256": "lexical"}
    receipt = {**parent, "status": "completed", "test_contents_accessed": False}
    if mutation == "packet_lineage":
        receipt["packet_completed_sha256"] = "other"
    elif mutation == "test_scope":
        receipt["test_contents_accessed"] = True
    study.write(tmp_path / "completed.json", receipt)
    args = SimpleNamespace(tokens=tmp_path, tokens_completed_sha256=study.sha(tmp_path / "completed.json"))
    if mutation == "external_pin":
        args.tokens_completed_sha256 = "wrong"
    called = []
    monkeypatch.setitem(sys.modules, "prepare_dialogue_tokens", SimpleNamespace(
        authenticate_cache=lambda *a: called.append(True)))
    with pytest.raises(ValueError):
        study.cache_identity(args, parent)
    assert called == []


@pytest.mark.parametrize("key,bad", [("loss_weights", [2., 1., 1.]), ("loss_counts", {"0": 99}),
                                    ("optimizer_updates_per_fit", 1279), ("wall_cap_seconds", 7200.)])
def test_resealed_plan_cannot_change_recipe_or_cap(tmp_path, monkeypatch, key, bad):
    prior = {"loss_weights": [1., 1., 1.], "loss_counts": {"0": 1, "1": 1, "2": 1},
             "optimizer_updates_per_fit": 1280, "packet_completed_sha256": "p", "lexical_completed_sha256": "l"}
    plan = {**prior, "study": study.STUDY, "implementation_version": study.VERSION, "config": study.CONFIG,
        "practical_checks": study.CHECKS, "runtime": study.runtime(), "wall_cap_seconds": study.CAP,
        "normalization_tolerance": study.TOLERANCE, "source_sha256": dict.fromkeys(study.SOURCES, "synthetic"),
        "paths": {}, "old_plan_sha256": study.OLD_PLAN, "old_completed_sha256": study.OLD_COMPLETED,
        "tokens_completed_sha256": "j", "execution_files": 52, "expected_fits": 12,
        "expected_configurations": {arm: {} for arm in study.ARMS}}
    study.write(tmp_path / "plan.json", plan)
    args = SimpleNamespace(out=tmp_path, plan_sha256=study.sha(tmp_path / "plan.json"),
        old_study=tmp_path, packet=tmp_path, lexical=tmp_path, tokens=tmp_path, tokens_completed_sha256="j")
    monkeypatch.setattr(study, "file_check", lambda *a: None)
    monkeypatch.setattr(study, "path_map", lambda *a: {})
    monkeypatch.setattr(study, "old_identity", lambda *a: (prior, {}))
    monkeypatch.setattr(study, "cache_identity", lambda *a: None)
    monkeypatch.setattr(study.old, "authenticate", lambda *a: None)
    study.validate(args)
    plan[key] = bad
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    args.plan_sha256 = study.sha(tmp_path / "plan.json")
    with pytest.raises(ValueError, match="objective/update budget|configuration/runtime"):
        study.validate(args)
