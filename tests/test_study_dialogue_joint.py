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
    spec = importlib.util.spec_from_file_location("study_joint_test", ROOT / "scripts/study_dialogue_joint.py")
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
    offset, cursor, chunks = 0, 0, []
    for d in ds:
        shape = d["layout"]["shape"][:3]
        d["joint_layout"] = {"id": d["id"], "query_ids": d["layout"]["query_ids"], "offset": offset, "shape": shape}
        chunk = np.full(shape, -1, np.int64)
        for t in range(shape[0]):
            for q, qi in enumerate(d["layout"]["query_ids"]):
                for c in range(len(queries[qi]["candidates"])):
                    chunk[t, q, c] = cursor
                    cursor += 1
        chunks.append(chunk.ravel())
        offset += chunk.size
    embeddings = np.arange(cursor*384, dtype=np.float32).reshape(cursor, 384)/10000
    return ds, queries, features, np.zeros(270, np.float32), (embeddings, np.concatenate(chunks))


@pytest.mark.parametrize("head", ["readout", "scalar"])
@pytest.mark.parametrize("mode", ["independent", "joint"])
def test_existing_monitor_preserves_joint_adapter_outputs_and_parameter_gradients(head, mode):
    ds, queries, features, lexical, joint = fixture()
    actor, labels, _ = study.make_batch(ds, queries, features, lexical, joint, mode)
    plain = study.DialogueJointMemory(head, projection_dim=4, hidden_dim=7, gru_width=3)
    watched = study.MonitoredJointMemory(head, projection_dim=4, hidden_dim=7, gru_width=3)
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
    assert watched.configuration()["class"] == "MonitoredJointMemory"
    assert watched.configuration()["normalized_parent_version"] == study.v2.VERSION


def test_joint_assembly_changes_only_public_observation_and_counts_paid_dense_work():
    ds, queries, features, lexical, joint = fixture()
    independent, labels, bins = study.make_batch(ds, queries, features, lexical, joint, "independent")
    actor, labels2, bins2 = study.make_batch(ds, queries, features, lexical, joint, "joint")
    assert torch.equal(labels, labels2) and torch.equal(bins, bins2)
    assert all(torch.equal(a, b) for a, b in zip(independent[1:], actor[1:], strict=True))
    assert torch.equal(actor[0][0, 0, 0, 0], torch.from_numpy(joint[0][0]))
    assert torch.equal(actor[0][0, 0, 1, 0], torch.from_numpy(joint[0][3]))
    assert (actor[0][0, :, 0, 3] == 0).all()
    assert (actor[0][1, 1:] == 0).all() and (actor[0][1, :, 1] == 0).all()
    a, b = (study.observation_work(ds, queries, v) for v in (independent, actor))
    assert a["turn_projection_positions"] == 6 and b["turn_projection_positions"] == 48
    assert a["emitted_bytes"] == 6*384*4 and b["emitted_bytes"] == 48*384*4
    assert a["real_candidate_updates"] == b["real_candidate_updates"] == 24
    assert study.old.work_counts(ds, independent) == study.old.work_counts(ds, actor)


def test_labels_and_scoring_eligibility_do_not_select_joint_updates():
    ds, queries, features, lexical, joint = fixture()
    original, _, _ = study.make_batch(ds, queries, features, lexical, joint, "joint")
    changed = copy.deepcopy(ds)
    changed[0]["queries"].pop(1)  # Still keep this public supplied question's full stream.
    for d in changed:
        for row in d["queries"]:
            row["label"] = 1
            row["bin"] = "revision"
            row["unseen"] = not row["unseen"]
    other, _, _ = study.make_batch(changed, queries, features, lexical, joint, "joint")
    assert all(torch.equal(a, b) for a, b in zip(original, other, strict=True))
    assert study.observation_work(ds, queries, original) == study.observation_work(changed, queries, other)


@pytest.mark.parametrize("mutation", ["missing_turn", "out_of_range", "candidate_padding", "foreign_query", "nonfinite", "extra_index"])
def test_cache_alignment_rejects_layout_or_payload_corruption(mutation):
    ds, queries, _, _, (embeddings, indices) = fixture()
    index = {"cohorts": {"train": [copy.deepcopy(d["joint_layout"]) for d in ds], "dev": []}}
    cohorts = {"train": ds, "dev": []}
    study.align_joint(cohorts, queries, index, embeddings, indices)
    if mutation == "missing_turn":
        index["cohorts"]["train"][0]["shape"][0] -= 1
    elif mutation == "out_of_range":
        indices[0] = len(embeddings)
    elif mutation == "candidate_padding":
        indices[3] = 0
    elif mutation == "foreign_query":
        index["cohorts"]["train"][0]["query_ids"].reverse()
    elif mutation == "nonfinite":
        embeddings[0, 0] = np.nan
    else:
        indices = np.append(indices, -1)
    with pytest.raises(ValueError):
        study.align_joint(cohorts, queries, index, embeddings, indices)


def test_configuration_capture_preserves_rng_and_exact_v2_initial_schema():
    rng = torch.random.get_rng_state().clone()
    configs = study.expected_configurations()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(configs) == set(study.ARMS)
    assert {x["parameters"] for x in configs.values()} == {99458}
    for head in ("readout", "scalar"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(410)
            parent = study.v2.MonitoredCopyMemoryV2(head)
            torch.manual_seed(410)
            child = study.MonitoredJointMemory(head)
        assert study.old.tensor_digest(parent) == study.old.tensor_digest(child)
        assert configs["independent_"+head] == configs["joint_"+head]


def fake_lifecycle(monkeypatch, tmp_path, *, bad_initial=False):
    ds, queries, features, lexical, joint = fixture()
    run = tmp_path / "run"
    run.mkdir()
    study.write(run / "plan.json", {"synthetic": True})
    args = SimpleNamespace(out=run, packet=tmp_path, lexical=tmp_path, joint=tmp_path, old_study=tmp_path,
        plan_sha256=study.sha(run / "plan.json"), joint_completed_sha256="fake")
    monkeypatch.setattr(study, "CONFIG", {**study.CONFIG, "epochs": 1})
    originals = {f"{m}-{s}": {"initial_tensors_sha256": "wrong" if bad_initial else "paired"}
                 for m in ("readout", "scalar") for s in study.SEEDS}
    configuration = lambda method: {"method": method, "implementation_version": study.VERSION, "synthetic": True}
    plan = {"loss_weights": [1., 1., 1.], "optimizer_updates_per_fit": 1, "joint_completed_sha256": "fake",
            "expected_configurations": {a: configuration(a.rsplit("_", 1)[1]) for a in study.ARMS}}
    monkeypatch.setattr(study, "validate", lambda *a: (plan, originals))
    monkeypatch.setattr(study, "load_inputs", lambda *a: ({"train": ds, "dev": ds}, queries, features, lexical, joint))
    monkeypatch.setattr(study.old, "tensor_digest", lambda *a: "paired")
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
            events.append(("forward", self.method, self.training, turns.ndim))
            output = (self.weight*torch.arange(mask.shape[-1])).expand(turns.shape[0], turns.shape[1], mask.shape[1], -1)
            return output.masked_fill(~mask[:, None], -torch.inf).log_softmax(-1)
        def configuration(self):
            return configuration(self.method)
    class FakeOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameters = list(parameters)
            events.append(("optimizer", kwargs))
        def zero_grad(self, **kwargs):
            for p in self.parameters:
                p.grad = None
        def step(self):
            events.append(("step",))  # No parameter update or actual optimizer.
    monkeypatch.setattr(study, "MonitoredJointMemory", FakeModel)
    monkeypatch.setattr(study.torch.optim, "AdamW", FakeOptimizer)
    return args, events


def test_all12_fake_fits_strict_membership_pairing_orders_and_observation_cost(tmp_path, monkeypatch):
    args, events = fake_lifecycle(monkeypatch, tmp_path)
    result = study.train(args)
    done = study.read(args.out / "completed.json")
    assert result["completed_sha256"] == study.sha(args.out / "completed.json")
    assert len(done["fits"]) == 12 and len(done["files"]) == 51
    assert set(done["files"])|{"completed.json"} == study.expected_members()
    assert [r["method"] for r in done["fits"]] == list(study.ARMS)*3
    assert sum(e[0] == "step" for e in events) == 12
    for r in done["fits"]:
        assert r["initial_tensors_sha256"] == r["original_initial_tensors_sha256"] == r["common_initial_tensors_sha256"]
        folder = args.out / "fits" / f"{r['method']}-{r['seed']}"
        ledger = [json.loads(x) for x in (folder / "batches.jsonl").read_text().splitlines()]
        assert len(ledger) == 1 and ledger[0]["indices"] == [1, 0]
        assert ledger[0]["observation_work"] == r["training_observation_work"]
        assert ledger[0]["supervised_queries"] == r["training_queries"] == 4
        assert r["evaluation"]["observation_work"]["turn_projection_positions"] == (6 if r["observation"] == "independent" else 48)
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
    assert (args.out / "fits/independent_readout-4101/partial-weights.pt").exists()


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
    assert (args.out / "fits/independent_readout-4101/batches.jsonl").stat().st_size > 0


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
    args = SimpleNamespace(joint=tmp_path, joint_completed_sha256=study.sha(tmp_path / "completed.json"))
    if mutation == "external_pin":
        args.joint_completed_sha256 = "wrong"
    called = []
    monkeypatch.setitem(sys.modules, "prepare_dialogue_joint", SimpleNamespace(
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
        "joint_completed_sha256": "j", "execution_files": 52, "expected_fits": 12,
        "expected_configurations": {arm: {} for arm in study.ARMS}}
    study.write(tmp_path / "plan.json", plan)
    args = SimpleNamespace(out=tmp_path, plan_sha256=study.sha(tmp_path / "plan.json"),
        old_study=tmp_path, packet=tmp_path, lexical=tmp_path, joint=tmp_path, joint_completed_sha256="j")
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
