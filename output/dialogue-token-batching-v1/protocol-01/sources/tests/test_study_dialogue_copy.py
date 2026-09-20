"""Synthetic packing and fake lifecycle checks; no corpus or scientific fits."""

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
    spec = importlib.util.spec_from_file_location("study_dialogue_copy_under_test", ROOT / "scripts/study_dialogue_copy.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
finally:
    sys.path.pop(0)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    setter = torch.set_num_threads
    setter(1)
    try:
        yield
    finally:
        setter(previous)


def fixture():
    features = np.arange(12 * 384, dtype=np.float32).reshape(12, 384) / 10000
    queries = [{"text": 4, "candidates": [6, 7, 8]}, {"text": 5, "candidates": [6, 7, 8, 9]}]
    ds = [
        {"id": "a", "turns": [0, 1, 2], "queries": [
            {"query": 1, "time": 0, "label": 2, "bin": "first_assignment", "unseen": False},
            {"query": 0, "time": 1, "label": 1, "bin": "first_assignment", "unseen": True},
            {"query": 1, "time": 2, "label": 2, "bin": "assigned_retention", "unseen": False}],
         "layout": {"id": "a", "query_ids": [0, 1], "shape": [3, 2, 4, 10], "offset": 0}},
        {"id": "b", "turns": [3], "queries": [
            {"query": 0, "time": 0, "label": 0, "bin": "unmentioned_retention", "unseen": True}],
         "layout": {"id": "b", "query_ids": [0], "shape": [1, 1, 3, 10], "offset": 240}},
    ]
    lexical = np.arange(270, dtype=np.float32) / 1000
    return ds, queries, features, lexical


def test_exact_batch_layout_and_padding():
    ds, queries, features, lexical = fixture()
    actor, labels, bins = study.make_batch(ds, queries, features, lexical)
    turns, valid, query, candidate, mask, lex = actor
    assert turns.shape == (2, 3, 384) and query.shape == (2, 2, 384)
    assert candidate.shape == (2, 2, 4, 384) and lex.shape == (2, 3, 2, 4, 10)
    assert valid.tolist() == [[True, True, True], [True, False, False]]
    assert labels.tolist() == [[[-100, 2], [1, -100], [-100, 2]], [[0, -100], [-100, -100], [-100, -100]]]
    assert bins[0].tolist() == [[0, 2], [2, 0], [0, 1]]
    assert mask[1, 1].tolist() == [True, False, False, False]
    np.testing.assert_array_equal(lex[0], lexical[:240].reshape(3, 2, 4, 10))
    np.testing.assert_array_equal(lex[1, 0, 0, :3], lexical[240:].reshape(3, 10))
    assert torch.count_nonzero(lex[1, 1:]) == 0
    assert torch.count_nonzero(turns[1, 1:]) == 0


def test_label_bin_unseen_changes_and_removed_eligibility_do_not_enter_actor():
    ds, queries, features, lexical = fixture()
    before, _, _ = study.make_batch(ds, queries, features, lexical)
    altered = copy.deepcopy(ds)
    for d in altered:
        for r in d["queries"]:
            r["label"], r["bin"], r["unseen"] = 1, "revision", not r["unseen"]
    # Keep the supplied independent schema streams, remove a loss record only.
    altered[0]["queries"].pop(1)
    after, labels, _ = study.make_batch(altered, queries, features, lexical)
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert labels[0, 1, 0] == -100


def test_earlier_unscored_turns_remain_in_every_independent_question_prefix():
    ds, queries, features, lexical = fixture()
    actor, labels, _ = study.make_batch(ds, queries, features, lexical)
    # q0 is first requested at time1, but its time0 public input and lexical
    # features are present. Gold eligibility never becomes a recurrent mask.
    assert labels[0, 0, 0] == -100 and actor[1][0, 0]
    np.testing.assert_array_equal(actor[0][0, 0], features[0])
    np.testing.assert_array_equal(actor[5][0, :, 0], lexical[:240].reshape(3, 2, 4, 10)[:, 0])
    solo, _, _ = study.make_batch(ds[:1], queries, features, lexical)
    for full, alone in zip(actor, solo, strict=True):
        torch.testing.assert_close(full[:1], alone, rtol=0, atol=0)


def test_work_counts_charge_all_padded_streams_and_separate_real_question_steps():
    ds, queries, features, lexical = fixture()
    actor, _, _ = study.make_batch(ds, queries, features, lexical)
    assert study.work_counts(ds, actor) == {
        "real_turns": 4, "padded_turn_positions": 6, "padded_query_positions": 12,
        "padded_candidate_positions": 48, "real_question_steps": 7}


@pytest.mark.parametrize("time", [-1, 3, True])
def test_invalid_target_time_is_rejected_before_indexing(time):
    ds, queries, features, lexical = fixture()
    ds[0]["queries"][0]["time"] = time
    with pytest.raises(ValueError, match="target|Target"):
        study.make_batch(ds, queries, features, lexical)


def test_duplicate_target_rejected():
    ds, queries, features, lexical = fixture()
    ds[0]["queries"].append(copy.deepcopy(ds[0]["queries"][0]))
    with pytest.raises(ValueError, match="Duplicated"):
        study.make_batch(ds, queries, features, lexical)


@pytest.mark.parametrize("label", [-1, 4, True, 1.5])
def test_invalid_target_label_rejected(label):
    ds, queries, features, lexical = fixture()
    ds[0]["queries"][0]["label"] = label
    with pytest.raises(ValueError, match="target label"):
        study.make_batch(ds, queries, features, lexical)


class IndexedScores:
    """Hand-computable scorer spy; no learned modules or RNG."""
    def eval(self):
        return self

    def __call__(self, turns, valid, query, candidates, mask, lexical):
        b, t, q, c = turns.shape[0], turns.shape[1], mask.shape[1], mask.shape[2]
        out = torch.full((b, t, q, c), -torch.inf)
        for i in range(b):
            for time in range(t):
                for j in range(q):
                    choice = (time + j) % int(mask[i, j].sum())
                    out[i, time, j, choice] = 0.
        return out


def test_evaluation_preserves_original_order_and_uses_correct_packed_question(tmp_path):
    ds, queries, features, lexical = fixture()
    path = tmp_path / "predictions.npz"
    receipt = study.evaluate(IndexedScores(), ds, queries, features, lexical, path)
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {"probabilities", "labels", "choice", "bin", "unseen", "dialogue", "time", "query"}
        assert saved["dialogue"].tolist() == ["a", "a", "a", "b"]
        assert saved["time"].tolist() == [0, 1, 2, 0]
        assert saved["query"].tolist() == [1, 0, 1, 0]
        assert saved["choice"].tolist() == [1, 1, 3, 0]
        assert saved["labels"].tolist() == [2, 1, 2, 0]
        assert saved["probabilities"].shape == (4, 12)
        assert np.count_nonzero(saved["probabilities"][:, 4:]) == 0
        np.testing.assert_array_equal(saved["probabilities"].sum(-1), np.ones(4))
    assert receipt["queries"] == 4 and receipt["actor_shapes"]["real_question_steps"] == 7


def test_evaluation_rejects_bad_scores(tmp_path):
    class BadScores(IndexedScores):
        def __call__(self, *args):
            return super().__call__(*args) * torch.nan
    with pytest.raises(ValueError, match="probabilities"):
        study.evaluate(BadScores(), *fixture(), tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


def fake_lifecycle(monkeypatch, tmp_path, *, fail_method=None):
    ds, queries, features, lexical = fixture()
    events = []
    config = {**study.CONFIG, "epochs": 1, "batch_size": 32}
    monkeypatch.setattr(study, "CONFIG", config)
    monkeypatch.setattr(study, "validate", lambda args: {"loss_weights": [1., 1., 1.]})
    monkeypatch.setattr(study, "load_inputs", lambda *args: ({"train": ds, "dev": ds}, queries, features, lexical))
    monkeypatch.setattr(study, "reference_predictions", lambda *args: {"synthetic": True})
    monkeypatch.setattr(study.torch, "manual_seed", lambda seed: events.append(("seed", seed)))
    monkeypatch.setattr(study.torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(study.torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(study.torch, "use_deterministic_algorithms", lambda x: None)
    monkeypatch.setattr(study.np.random, "default_rng", lambda seed: SimpleNamespace(permutation=lambda n: np.arange(n)))

    class FakeModel(torch.nn.Module):
        def __init__(self, method, **kwargs):
            super().__init__()
            events.append(("construct", method))
            if method == fail_method:
                raise RuntimeError("synthetic constructor failure")
            self.method = method
            self.weight = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, turns, valid, query, candidates, mask, lexical):
            events.append(("forward", self.method))
            c = candidates.shape[2]
            logits = (self.weight * torch.arange(c)).expand(turns.shape[0], turns.shape[1], mask.shape[1], c)
            return logits.masked_fill(~mask[:, None], -torch.inf).log_softmax(-1)

        def configuration(self):
            return {"synthetic": True, "method": self.method}

    class FakeOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameters = list(parameters)

        def zero_grad(self, **kwargs):
            for p in self.parameters:
                p.grad = None

        def step(self):
            events.append(("fake_optimizer_step",))
            # Intentionally no optimizer update or scientific fitting.

    monkeypatch.setattr(study, "DialogueCopyMemory", FakeModel)
    monkeypatch.setattr(study.torch.optim, "AdamW", FakeOptimizer)
    args = SimpleNamespace(packet="unused", lexical="unused", out=str(tmp_path), plan_sha256="0" * 64)
    return args, events


def test_fake_whole_lifecycle_has_all15_final_members_and_exact_completed_work(monkeypatch, tmp_path):
    args, events = fake_lifecycle(monkeypatch, tmp_path)
    study.train(args)
    done = json.loads((tmp_path / "completed.json").read_text())
    assert done["fit_count"] == 15
    assert [(r["method"], r["seed"]) for r in done["fits"]] == [
        (method, seed) for seed in study.SEEDS for method in study.METHODS]
    assert len([e for e in events if e[0] == "fake_optimizer_step"]) == 15
    for row in done["fits"]:
        assert row["updates"] == 1 and row["training_queries"] == 4
        assert row["training_actor_shapes"]["real_question_steps"] == 7
        folder = tmp_path / "fits" / f"{row['method']}-{row['seed']}"
        assert {p.name for p in folder.iterdir()} == {"completed.json", "weights.pt", "dev-predictions.npz"}
        assert study.sha(folder / "weights.pt") == row["weights_sha256"]
        assert study.sha(folder / "dev-predictions.npz") == row["predictions_sha256"]


def test_constructor_failure_records_new_identity_zero_progress_and_retains_previous_fit(monkeypatch, tmp_path):
    args, _ = fake_lifecycle(monkeypatch, tmp_path, fail_method="scalar")
    with pytest.raises(RuntimeError, match="constructor failure"):
        study.train(args)
    failure = json.loads((tmp_path / "failed.json").read_text())
    assert len(failure["completed_fits"]) == 1
    assert failure["active_progress"]["method"] == "scalar"
    assert failure["active_progress"]["completed_optimizer_steps"] == 0
    assert not (tmp_path / "completed.json").exists()
    assert (tmp_path / "fits" / "readout-4101" / "completed.json").is_file()


def test_evaluation_failure_retains_completed_fake_step_counts(monkeypatch, tmp_path):
    args, _ = fake_lifecycle(monkeypatch, tmp_path)
    original = RuntimeError("synthetic evaluation failure")
    def fail(*args):
        raise original
    monkeypatch.setattr(study, "evaluate", fail)
    with pytest.raises(RuntimeError) as observed:
        study.train(args)
    assert observed.value is original
    saved = json.loads((tmp_path / "failed.json").read_text())
    assert saved["completed_fits"] == []
    assert saved["active_progress"]["completed_optimizer_steps"] == 1
    assert saved["active_progress"]["completed_supervised_queries"] == 4
    assert len(saved["active_progress"]["completed_epoch_losses"]) == 1
    assert (tmp_path / "fits" / "readout-4101" / "weights.pt").is_file()
    assert not (tmp_path / "completed.json").exists()


def test_failure_receipt_error_never_replaces_original_exception(monkeypatch, tmp_path):
    original = RuntimeError("original failure")
    def bad_write(*args):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(study, "write", bad_write)
    study.failure(tmp_path, original, {"active_progress": {"completed_optimizer_steps": 3}})
    assert "synthetic disk failure" in original.__notes__[0]


def test_wrong_external_plan_hash_refuses_before_inputs_or_fit_directory(monkeypatch, tmp_path):
    (tmp_path / "plan.json").write_text("{}")
    args = SimpleNamespace(out=str(tmp_path), packet="unused", lexical="unused", plan_sha256="0" * 64)
    def forbidden(*args):
        raise AssertionError("inputs must not be read after bad plan hash")
    monkeypatch.setattr(study, "load_inputs", forbidden)
    with pytest.raises(ValueError, match="Plan digest"):
        study.train(args)
    assert not (tmp_path / "fits").exists()
