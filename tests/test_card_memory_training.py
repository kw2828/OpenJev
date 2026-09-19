"""Tiny public-event fixtures only. Any random model construction uses seed410."""

import copy
import json
import math

import pytest
import torch
from torch import nn

from openjev.research import card_memory_training as training
from openjev.research.card_associative_memory import MODES, CardAssociativeMemory


def data(n=2, steps=5):
    positions = torch.arange(steps).remainder(52).repeat(n, 1)
    ranks = positions.remainder(13)
    valid = torch.ones((n, steps), dtype=torch.bool)
    targets = torch.full((n, steps, 52), -1, dtype=torch.int64)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    ages = torch.full_like(targets, -1, dtype=torch.int32)
    last_clock = [-1] * 52
    for step in range(steps):
        for position, clock in enumerate(last_clock):
            if clock >= 0 and step - clock >= 1:
                mask[:, step, position] = True
                targets[:, step, position] = position % 13
                ages[:, step, position] = step - clock
        last_clock[step % 52] = step + 1
    return {"positions": positions, "ranks": ranks, "valid": valid,
            "targets": targets, "target_mask": mask, "ages": ages}


class ToyMemory(nn.Module):
    def __init__(self, *, constant=False):
        super().__init__()
        self.bias = nn.Parameter(torch.arange(13, dtype=torch.float32) / 10)
        self.scale = nn.Parameter(torch.tensor(0.3))
        self.constant = constant
        self.mode = "toy"
        self.predictions = []
        self.events = []

    def init_state(self, batch, device):
        return {"S": torch.zeros(batch, 13, device=device), "P": None}

    def predict(self, state, queries):
        logits = self.bias[None] + (0 if self.constant else state["S"] * self.scale)
        logits = logits.expand(len(queries), -1)[:, None].expand(-1, 52, -1)
        self.predictions.append(logits.detach().clone())
        return logits

    def write(self, state, pos, rank, valid):
        self.events.append((pos.clone(), rank.clone(), valid.clone()))
        new = torch.where(valid[:, None], state["S"] + torch.nn.functional.one_hot(rank, 13), state["S"])
        return {"S": new, "P": None}, {"writes": int(valid.sum())}


def model(mode):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        return CardAssociativeMemory(mode)


def test_equal_episode_and_boundary_weighting_not_pooled_queries():
    values = data()
    # Episode0 keeps one easy target in each boundary; episode1 has more queries.
    values["target_mask"][0, :, 1:] = False
    values["targets"][0, :, 1:] = -1
    values["ages"][0, :, 1:] = -1
    student = ToyMemory(constant=True)
    loss, counts = training.batch_loss(student, values, [0, 1])
    log_prob = student.bias.log_softmax(0)
    expected_ep0 = -log_prob[0]
    expected_ep1 = torch.stack([-log_prob[:k].mean() for k in (1, 2, 3)]).mean()
    assert loss.item() == pytest.approx(((expected_ep0 + expected_ep1) / 2).item())
    pooled = -torch.cat([log_prob[:1]] * 3 + [log_prob[:k] for k in (1, 2, 3)]).mean()
    assert abs(loss.item() - pooled.item()) > 0.01
    assert counts["eligible_episodes"] == 2
    assert counts["query_boundaries"] == 6
    assert counts["query_cards"] == 9


def test_full_shape_synthetic_fixture_has_causal_refresh_ages():
    validated = training.validate_data(data(16, 104))
    assert validated["ages"][0, 54, 0] == 1
    assert validated["targets"][0, 54, 0] == 0


def test_future_reveal_changes_only_subsequent_predictions():
    original = data(1, 5)
    changed = copy.deepcopy(original)
    changed["ranks"][0, 3] = 12
    # Position3 is not queried before its first reveal or while still visible.
    a, b = ToyMemory(), ToyMemory()
    training.batch_loss(a, original, [0])
    training.batch_loss(b, changed, [0])
    assert all(torch.equal(a.predictions[t], b.predictions[t]) for t in range(4))
    assert not torch.equal(a.predictions[4], b.predictions[4])


def test_labels_do_not_enter_model_inputs():
    original = data(1, 5)
    changed = copy.deepcopy(original)
    changed["target_mask"][:, 3, 0] = False
    changed["targets"][:, 3, 0] = -1
    changed["ages"][:, 3, 0] = -1
    a, b = ToyMemory(), ToyMemory()
    training.batch_loss(a, original, [0])
    training.batch_loss(b, changed, [0])
    assert all(torch.equal(x, y) for x, y in zip(a.predictions, b.predictions, strict=True))


@pytest.mark.parametrize("mode", MODES)
def test_actual_six_modes_finite_gradients_and_no_data_mutation(mode):
    student, values = model(mode), data()
    saved = copy.deepcopy(values)
    loss, counts = training.batch_loss(student, values, [1, 0])
    loss.backward()
    assert torch.isfinite(loss)
    assert student.key_embedding.weight.grad is not None
    assert student.value_embedding.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
    assert counts["valid_reveals"] == 10
    assert counts["predict_calls_completed"] == counts["write_calls_completed"] == 5
    assert all(torch.equal(values[k], saved[k]) for k in values)


def test_empty_targets_differentiable_zero():
    values = data(1, 1)
    student = ToyMemory()
    loss, counts = training.batch_loss(student, values, [0])
    loss.backward()
    assert loss.item() == 0
    assert counts["eligible_episodes"] == 0
    assert all(torch.equal(p.grad, torch.zeros_like(p)) for p in student.parameters())


def test_padding_sanitized_and_all_padding_tail_skipped():
    values = data(2, 5)
    values["valid"][0, 3:] = False
    values["valid"][:, 4] = False
    for key in ("targets", "ages"):
        values[key][~values["valid"]] = -1
    values["target_mask"][~values["valid"]] = False
    values["positions"][~values["valid"]] = -999
    values["ranks"][~values["valid"]] = 999
    student = ToyMemory()
    _, counts = training.batch_loss(student, values, [0, 1])
    assert counts["write_calls_completed"] == 4
    assert counts["valid_reveals"] == 7
    assert counts["padded_write_examples"] == 1
    assert student.events[-1][0].tolist() == [0, 3]
    assert student.events[-1][1].tolist() == [0, 3]


def test_violating_padding_write_rejected():
    values = data(2, 5)
    values["valid"][0, 4] = False
    values["target_mask"][0, 4] = False
    values["targets"][0, 4] = values["ages"][0, 4] = -1
    student = ToyMemory()
    original = student.write

    def bad(*args):
        state, diag = original(*args)
        state["S"] = state["S"] + 1
        return state, diag

    student.write = bad
    with pytest.raises(ValueError, match="Padding write"):
        training.batch_loss(student, values, [0, 1])


@pytest.mark.parametrize("field,value", [("positions", torch.float32), ("ranks", torch.int32),
                                         ("valid", torch.int64), ("ages", torch.int64),
                                         ("targets", torch.float32), ("target_mask", torch.uint8)])
def test_strict_data_dtypes(field, value):
    values = data()
    values[field] = values[field].to(value)
    with pytest.raises(ValueError, match="CPU"):
        training.validate_data(values)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["valid"].__setitem__((0, 1), False), "prefix"),
    (lambda d: d["positions"].__setitem__((0, 1), 52), "range"),
    (lambda d: d["ranks"].__setitem__((0, 1), 13), "range"),
    (lambda d: d["targets"].__setitem__((0, 2, 0), 12), "previously"),
    (lambda d: d["targets"].__setitem__((0, 0, 0), 0), "Unqueried"),
    (lambda d: d["ages"].__setitem__((0, 2, 0), 2), "age"),
    (lambda d: d["ages"].__setitem__((0, 2, 0), 0), "age"),
])
def test_bad_public_data_rejected(mutation, match):
    values = data()
    mutation(values)
    with pytest.raises(ValueError, match=match):
        training.validate_data(values)


def test_cannot_supervise_current_first_reveal():
    values = data()
    values["target_mask"][0, 2, 2] = True
    values["targets"][0, 2, 2] = 2
    values["ages"][0, 2, 2] = 1
    with pytest.raises(ValueError, match="previously"):
        training.validate_data(values)


def test_static_rank_consistency():
    values = data()
    values["positions"][0, -1] = 0
    with pytest.raises(ValueError, match="cannot change"):
        training.validate_data(values)


@pytest.mark.parametrize("indices", [[0, 0], [], [-1], [2], [True], [0.0]])
def test_indices_rejected(indices):
    with pytest.raises(ValueError, match="indices"):
        training.batch_loss(ToyMemory(), data(), indices)


def test_evaluation_balancing_counts_and_mode_preserved():
    student, values = ToyMemory(constant=True), data(3, 5)
    values["target_mask"][0, 3:] = False
    values["targets"][0, 3:] = values["ages"][0, 3:] = -1
    expected, _ = training.batch_loss(student, values, [0, 1, 2])
    before = {k: v.clone() for k, v in student.state_dict().items()}
    a = training.evaluate(student, values, batch_size=2)
    b = training.evaluate(student, values, batch_size=1)
    assert a["all"]["ce"] == pytest.approx(float(expected.detach()))
    assert a["all"] == b["all"]
    assert a["all"]["eligible_episodes"] == 3
    assert a["all"]["query_boundaries"] == 7
    assert a["all"]["query_cards"] == 13
    assert a["age_gt32"]["ce"] is None
    assert student.training
    assert all(torch.equal(before[k], v) for k, v in student.state_dict().items())
    assert all(p.grad is None for p in student.parameters())
    assert len(a["per_episode"]) == 3


def test_age_gt32_is_strict_and_separately_episode_balanced():
    values = data(2, 36)
    values["target_mask"][:] = False
    values["targets"][:] = values["ages"][:] = -1
    for row, step, age in ((0, 33, 32), (0, 34, 33), (0, 35, 34), (1, 35, 34)):
        values["target_mask"][row, step, 0] = True
        values["targets"][row, step, 0] = 0
        values["ages"][row, step, 0] = age
    result = training.evaluate(ToyMemory(constant=True), values)
    assert result["all"]["query_cards"] == 4
    assert result["age_gt32"]["query_cards"] == 3
    assert result["age_gt32"]["query_boundaries"] == 3
    assert result["age_gt32"]["eligible_episodes"] == 2
    assert result["all"]["ce"] == pytest.approx(result["age_gt32"]["ce"], abs=1e-6)


def test_brier_and_accuracy_analytic_uniform():
    student = ToyMemory(constant=True)
    with torch.no_grad():
        student.bias.zero_()
    result = training.evaluate(student, data(1, 3))
    assert result["all"]["ce"] == pytest.approx(math.log(13))
    assert result["all"]["brier"] == pytest.approx(12 / 13)
    assert result["all"]["accuracy"] == 1


def test_finite_prediction_and_evaluation_failure_restore_mode():
    student = ToyMemory()
    student.predict = lambda state, queries: torch.full((len(queries), 52, 13), float("nan"))
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        training.evaluate(student, data())
    assert student.training


def test_tiny_training_final_only_exact_schedule_and_reproducibility(tmp_path):
    orders = torch.tensor([[2, 0, 1], [1, 2, 0]])
    student, other = model("delta"), model("delta")
    values = data(3, 4)
    first = training.train_fit(student, values, orders, tmp_path / "one", batch_size=2)
    second = training.train_fit(other, values, orders, tmp_path / "two", batch_size=2)
    assert first["counts"] == {"backward_attempted": 4, "backward_completed": 4,
                                "optimizer_attempted": 4, "optimizer_returned": 4,
                                "successful_updates": 4, "completed_epochs": 2}
    assert first["final_weights_sha256"] == second["final_weights_sha256"]
    checkpoint = torch.load(tmp_path / "one/final-checkpoint.pt", weights_only=True)
    assert checkpoint["configuration"]["mode"] == "delta"
    assert torch.equal(checkpoint["orders"], orders)
    assert checkpoint["counts"]["successful_updates"] == 4
    assert first["evaluation_calls"] == 0
    assert first["work"]["valid_reveals"] == 24
    assert set(first["files"]) == {"started.json", "fit-settings.json", "batches.jsonl", "epoch-000.json",
                                    "epoch-001.json", "final-checkpoint.pt"}
    assert not list((tmp_path / "one").glob("*partial*"))
    rows = [json.loads(row) for row in (tmp_path / "one/batches.jsonl").read_text().splitlines()]
    assert [row["indices"] for row in rows] == [[2, 0], [1], [1, 2], [0]]
    for meta in first["files"].values():
        assert len(meta["sha256"]) == 64 and meta["bytes"] > 0


def test_failed_fit_preserves_prefix_no_checkpoint_or_retry(tmp_path):
    student = ToyMemory()
    original = student.write

    def fail(*args):
        if len(student.events) == 4:
            raise RuntimeError("injected forward failure")
        return original(*args)

    student.write = fail
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        training.train_fit(student, data(2, 4), torch.tensor([[0, 1]]), out, batch_size=1)
    failed = json.loads((out / "failed.json").read_text())
    assert failed["counts"]["successful_updates"] == 1
    assert failed["work"]["write_calls_attempted"] == 5
    assert failed["work"]["write_calls_completed"] == 4
    assert not (out / "completed.json").exists()
    assert not list(out.glob("*.pt"))
    with pytest.raises(FileExistsError):
        training.train_fit(ToyMemory(), data(), torch.tensor([[0, 1]]), out)


def test_bad_orders_and_expired_deadline_preserved_without_updates(tmp_path):
    for label, orders, deadline, error in (
        ("bad", [[0, 0]], None, ValueError),
        ("late", [[0, 1]], 0.0, TimeoutError),
    ):
        out = tmp_path / label
        with pytest.raises(error):
            training.train_fit(ToyMemory(), data(), torch.tensor(orders), out, wall_deadline=deadline)
        failed = json.loads((out / "failed.json").read_text())
        assert failed["counts"]["optimizer_attempted"] == 0
        assert not list(out.glob("*.pt"))


def test_late_completion_demoted_and_original_error_preserved(tmp_path, monkeypatch):
    original = training._json

    def writer(path, value):
        original(path, value)
        if path.name == "completed.json":
            raise TimeoutError("late output boundary")

    monkeypatch.setattr(training, "_json", writer)
    out = tmp_path / "late"
    with pytest.raises(TimeoutError, match="late output"):
        training.train_fit(ToyMemory(), data(1, 3), torch.tensor([[0]]), out)
    assert not (out / "completed.json").exists()
    assert not (out / "final-checkpoint.pt").exists()
    assert (out / "invalid-completion.json").exists()
    assert (out / "invalid-final-checkpoint.pt").exists()
    assert (out / "failed.json").exists()


def test_failure_writer_cannot_mask_original(tmp_path, monkeypatch):
    original = training._json

    def writer(path, value):
        if path.name == "failed.json":
            raise OSError("disk failure")
        return original(path, value)

    monkeypatch.setattr(training, "_json", writer)
    with pytest.raises(TimeoutError, match="deadline"):
        training.train_fit(ToyMemory(), data(), torch.tensor([[0, 1]]), tmp_path / "x", wall_deadline=0)
