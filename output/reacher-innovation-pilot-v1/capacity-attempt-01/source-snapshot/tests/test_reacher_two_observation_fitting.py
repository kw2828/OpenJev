"""Serialization tests with a fake trainer, no model forward or optimizer calls."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from openjev.research import reacher_two_observation_fitting as fitting


def read(path):
    return json.loads(path.read_text())


def sealed(body):
    return {**copy.deepcopy(body), "integrity_sha256": fitting.canonical_state_hash(body)}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = SimpleNamespace(fail_constructor=False, fail_before=None, fail_after=None, corrupt_row=None,
                        corrupt_checkpoint=None, export_failure=False, trainers=[])
    h.settings = fitting.training.TrainingSettings(hidden_size=3, train_episodes=2, steps=2,
                                                   epochs=2, batch_size=1, rollout_horizon=1)
    h.initial = {"synthetic_weight": torch.tensor([.1, .2], dtype=torch.float32)}
    h.orders = sealed({"orders": torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
                       "role": "synthetic_explicit_order", "seed": 410,
                       "initial_rng": torch.tensor([1], dtype=torch.uint8),
                       "final_rng": torch.tensor([2], dtype=torch.uint8)})
    h.data = {"packets": torch.zeros(2, 3, 8), "commands": torch.zeros(2, 2, 2), "rewards": torch.zeros(2, 2)}
    h.kwargs = {"settings": h.settings, "expected_initial_sha256": fitting.canonical_tensor_hash(h.initial),
        "expected_orders_sha256": fitting.canonical_state_hash(h.orders),
        "expected_data_sha256": fitting.canonical_tensor_hash(h.data),
        "provenance": {"scope": "synthetic_only", "pair": "fixture"},
        "source_sha256": {"fake_trainer": "b" * 64}, "runtime": {"scope": "fake_no_optimizer"},
        "pair": "fixture", "name": "two_observation_gru-fixture", "deadline": float("inf"), "progress": {}}
    h.out = tmp_path / "fit"

    class FakeTrainer:
        def __init__(self, initial, orders, public_data, **kwargs):
            if h.fail_constructor:
                raise KeyboardInterrupt("synthetic constructor failure")
            assert initial is h.initial and orders is h.orders and public_data is h.data
            assert kwargs["expected_orders_sha256"] == fitting.canonical_state_hash(orders)
            self.settings = kwargs["settings"]
            self.weights = copy.deepcopy(initial)
            self.student = SimpleNamespace(state_dict=lambda: self.weights, parameters=lambda: list(self.weights.values()))
            self.model_configuration = {"model_class": "TwoObservationHistoryGRUWorldModel", "synthetic_only": True}
            self.successful_updates = self.optimizer_steps = 0
            self.log_chain_sha256 = fitting.canonical_state_hash([])
            self.failed, self.failure, self.last_attempt = False, None, None
            self.setup_wall_seconds = self.training_wall_seconds = 0.
            self.rows = []
            h.trainers.append(self)

        @property
        def cursor(self):
            epoch, batch = divmod(self.successful_updates, self.settings.batches_per_epoch)
            return {"epoch": epoch, "batch": batch}

        def train_next(self, *, deadline_check):
            # Verify each PREVIOUS row is already visible on disk before the next call.
            with (h.out / "training.jsonl").open() as log:
                assert len(log.readlines()) == self.successful_updates
            deadline_check()
            update = self.successful_updates + 1
            if update == h.fail_before:
                self.failed, self.failure = True, {"synthetic": "before_update"}
                self.last_attempt = {"status": "failed", "completed_neural_sample_calls": {"synthetic": 0}}
                raise KeyboardInterrupt("synthetic before update")
            epoch, batch = self.cursor.values()
            self.optimizer_steps += 1  # Counter simulation only; no optimizer exists.
            self.successful_updates += 1
            self.weights["synthetic_weight"] = self.weights["synthetic_weight"] + .01
            if update == h.fail_after:
                self.failed, self.failure = True, {"synthetic": "after_update"}
                self.last_attempt = {"status": "failed", "completed_neural_sample_calls": {"synthetic": 3}}
                raise TimeoutError("synthetic after completed optimizer step")
            indices = h.orders["orders"][epoch, batch:batch + 1]
            row = {"kind": fitting.training.KIND, "update": update, "optimizer_steps": update,
                "epoch": epoch, "batch": batch, "indices": indices.tolist(),
                "indices_sha256": fitting.canonical_tensor_hash({"indices": indices}),
                "previous_log_sha256": self.log_chain_sha256, "loss": .123,
                "anchor": {"synthetic_only": True}, "gradient_norm": .4, "gradient_clipped": False,
                "work": {"synthetic_only": True}, "costs": {"batch_wall_seconds": 0.},
                "observed_neural_sample_calls": {"gru_cell_sample_calls": 0, "linear_layer_sample_calls": 0},
                "rng_identity_sha256": "c" * 64, "future_extra_field": {"must_survive": [1, 2, 3]}}
            if h.corrupt_row == "indices":
                row["indices"] = [5]
            if h.corrupt_row == "cursor":
                row["epoch"] += 1
            if h.corrupt_row == "previous":
                row["previous_log_sha256"] = "f" * 64
            row["log_sha256"] = fitting.canonical_state_hash(row)
            if h.corrupt_row == "digest":
                row["log_sha256"] = "e" * 64
            self.log_chain_sha256 = row["log_sha256"]
            self.last_attempt = {"status": "completed", "completed_neural_sample_calls": {"synthetic": 0}}
            self.rows.append(copy.deepcopy(row))
            return row

        def export_checkpoint(self):
            if h.export_failure:
                raise RuntimeError("synthetic export failure")
            body = {"version": fitting.training.VERSION, "kind": fitting.training.KIND,
                "settings": self.settings.configuration(), "model_configuration": self.model_configuration,
                "initialization": {"weights": h.initial, "tensor_sha256": h.kwargs["expected_initial_sha256"]},
                "orders": h.orders, "data_sha256": h.kwargs["expected_data_sha256"],
                "provenance": h.kwargs["provenance"], "source_sha256": h.kwargs["source_sha256"], "runtime": h.kwargs["runtime"],
                "student_state": self.weights, "optimizer_state": {"synthetic_only": True, "steps": self.optimizer_steps},
                "optimizer_group_names": [["synthetic_weight"]], "successful_updates": self.successful_updates,
                "optimizer_steps": self.optimizer_steps, "cursor": self.cursor, "failed": self.failed,
                "failure": self.failure, "last_attempt": self.last_attempt, "log_chain_sha256": self.log_chain_sha256,
                "rng": {"synthetic_only": True}, "setup_wall_seconds": 0., "training_wall_seconds": 0.,
                "restoration_wall_seconds": 0., "future_checkpoint_field": {"preserved": True}}
            body = copy.deepcopy(body)
            if h.corrupt_checkpoint == "class":
                body["model_configuration"]["model_class"] = "GRUResidualRewardWorldModel"
            if h.corrupt_checkpoint == "settings":
                body["settings"]["epochs"] += 1
            if h.corrupt_checkpoint == "order":
                body["orders"]["orders"][0, 0] = 0
            if h.corrupt_checkpoint == "initial":
                body["initialization"]["weights"]["synthetic_weight"][0] += 1
            if h.corrupt_checkpoint == "data":
                body["data_sha256"] = "f" * 64
            if h.corrupt_checkpoint == "provenance":
                body["provenance"]["scope"] = "changed"
            if h.corrupt_checkpoint == "runtime":
                body["runtime"]["scope"] = "changed"
            if h.corrupt_checkpoint == "source":
                body["source_sha256"]["fake_trainer"] = "d" * 64
            if h.corrupt_checkpoint == "weights":
                body["student_state"]["synthetic_weight"][0] += 1
            if h.corrupt_checkpoint == "cursor":
                body["cursor"]["epoch"] = 0
            if h.corrupt_checkpoint == "chain":
                body["log_chain_sha256"] = "f" * 64
            return sealed(body)

    monkeypatch.setattr(fitting.training, "TwoObservationTrainer", FakeTrainer)
    h.run = lambda: fitting.fit_one(h.initial, h.orders, h.data, h.out, **h.kwargs)
    return h


def test_exact_five_success_files_full_logs_and_checkpoint_fields(harness):
    h = harness
    before_rng = torch.get_rng_state().clone()
    receipt = h.run()
    assert torch.equal(before_rng, torch.get_rng_state())
    assert {p.name for p in h.out.iterdir()} == fitting.PAYLOAD_FILES | {"completed.json"}
    assert read(h.out / "completed.json") == receipt
    assert receipt["updates"] == receipt["optimizer_steps"] == receipt["flushed_updates"] == 4
    assert receipt["cursor"] == {"epoch": 2, "batch": 0}
    assert receipt["orders_sha256"] == fitting.canonical_state_hash(h.orders) != h.orders["integrity_sha256"]
    assert h.kwargs["progress"]["phase"] == "completed"
    rows = [json.loads(line) for line in (h.out / "training.jsonl").read_text().splitlines()]
    assert rows == h.trainers[0].rows
    assert all(row["future_extra_field"] == {"must_survive": [1, 2, 3]} for row in rows)
    checkpoint = torch.load(h.out / "checkpoint.pt", weights_only=True)
    assert checkpoint["future_checkpoint_field"] == {"preserved": True}
    assert checkpoint["rng"] == {"synthetic_only": True}
    assert checkpoint["optimizer_state"]["steps"] == 4
    assert checkpoint["log_chain_sha256"] == rows[-1]["log_sha256"] == receipt["log_chain_sha256"]
    weights = torch.load(h.out / "weights.pt", weights_only=True)
    assert fitting.canonical_tensor_hash(weights) == receipt["student_tensor_sha256"]
    assert fitting.canonical_tensor_hash(torch.load(h.out / "initial-weights.pt", weights_only=True)) == h.kwargs["expected_initial_sha256"]
    assert all(fitting._sha(h.out / name, float("inf")) == sha for name, sha in receipt["files"].items())
    assert receipt["wall_seconds"] >= receipt["constructor_seconds"] + receipt["update_call_seconds"]
    assert receipt["log_validation_flush_seconds"] > 0 and receipt["payload_export_write_seconds"] > 0


def test_existing_output_is_never_replaced(harness):
    h = harness
    h.out.mkdir()
    (h.out / "sentinel").write_text("original")
    with pytest.raises(FileExistsError):
        h.run()
    assert h.trainers == [] and (h.out / "sentinel").read_text() == "original"


@pytest.mark.parametrize("kind", ["initial", "orders", "orders_integrity_instead_of_full", "data", "pair", "name", "deadline"])
def test_external_binding_and_name_rejections_before_trainer(harness, kind):
    h = harness
    if kind == "initial":
        h.kwargs["expected_initial_sha256"] = "f" * 64
    elif kind == "orders":
        h.kwargs["expected_orders_sha256"] = "f" * 64
    elif kind == "orders_integrity_instead_of_full":
        h.kwargs["expected_orders_sha256"] = h.orders["integrity_sha256"]
    elif kind == "data":
        h.kwargs["expected_data_sha256"] = "f" * 64
    elif kind in {"pair", "name"}:
        h.kwargs[kind] = "../escape"
    elif kind == "deadline":
        h.kwargs["deadline"] = -1
    with pytest.raises((ValueError, TimeoutError)):
        h.run()
    assert h.trainers == [] and (h.out / "failed.json").exists()
    assert not (h.out / "completed.json").exists()


def test_constructor_failure_receipt_without_fabricated_checkpoint(harness):
    h = harness
    h.fail_constructor = True
    with pytest.raises(KeyboardInterrupt, match="constructor failure"):
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["phase"] == "constructor" and failure["optimizer_steps"] == 0
    assert not (h.out / "partial-checkpoint.pt").exists()


@pytest.mark.parametrize(("mode", "expected_steps", "expected_logs"), [("before", 1, 1), ("after", 2, 1)])
def test_partial_checkpoint_keeps_completed_step_even_without_returned_row(harness, mode, expected_steps, expected_logs):
    h = harness
    setattr(h, f"fail_{mode}", 2)
    with pytest.raises((KeyboardInterrupt, TimeoutError)):
        h.run()
    failure = read(h.out / "failed.json")
    checkpoint = torch.load(h.out / "partial-checkpoint.pt", weights_only=True)
    assert failure["optimizer_steps"] == checkpoint["optimizer_steps"] == expected_steps
    assert failure["successful_updates"] == expected_steps and failure["flushed_updates"] == expected_logs
    assert len((h.out / "training.jsonl").read_text().splitlines()) == expected_logs
    assert checkpoint["failed"] is True and checkpoint["last_attempt"]["status"] == "failed"
    assert not (h.out / "completed.json").exists() and not (h.out / "unflushed-update.json").exists()


@pytest.mark.parametrize("kind", ["indices", "cursor", "previous", "digest"])
def test_log_mismatch_retains_unflushed_row_and_actual_partial_counters(harness, kind):
    h = harness
    h.corrupt_row = kind
    with pytest.raises(ValueError):
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["optimizer_steps"] == 1 and failure["flushed_updates"] == 0
    assert (h.out / "training.jsonl").read_text() == ""
    assert read(h.out / "unflushed-update.json")["update"] == 1
    assert (h.out / "partial-checkpoint.pt").exists()


@pytest.mark.parametrize("kind", ["class", "settings", "order", "initial", "data", "provenance", "runtime", "source", "weights", "cursor", "chain"])
def test_final_checkpoint_mismatch_cannot_complete(harness, kind):
    h = harness
    h.corrupt_checkpoint = kind
    with pytest.raises(ValueError):
        h.run()
    assert read(h.out / "failed.json")["optimizer_steps"] == 4
    assert not (h.out / "completed.json").exists()


def test_final_write_crossing_cap_invalidates_completion_marker(harness, monkeypatch):
    h = harness
    original_cap = fitting._cap
    def cap(deadline):
        if (h.out / "completed.json").exists():
            raise TimeoutError("synthetic terminal write crossed cap")
        return original_cap(deadline)
    monkeypatch.setattr(fitting, "_cap", cap)
    with pytest.raises(TimeoutError, match="terminal write"):
        h.run()
    assert not (h.out / "completed.json").exists()
    assert (h.out / "invalid-completion.json").exists()
    assert read(h.out / "failed.json")["phase"] == "completion_write"
    assert torch.load(h.out / "partial-checkpoint.pt", weights_only=True)["optimizer_steps"] == 4


def test_log_fsync_failure_preserves_returned_but_unflushed_row(harness, monkeypatch):
    h = harness
    original_fsync = fitting.os.fsync
    calls = 0
    def fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:  # first is initial weights; second is the first update log.
            raise OSError("synthetic fsync failure")
        return original_fsync(fd)
    monkeypatch.setattr(fitting.os, "fsync", fsync)
    with pytest.raises(OSError, match="fsync failure"):
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["flushed_updates"] == 0 and failure["optimizer_steps"] == 1
    assert read(h.out / "unflushed-update.json")["update"] == 1
    assert failure["trainer_failed"] is False  # Outer attempt still terminal.


def test_secondary_checkpoint_export_error_does_not_mask_original(harness):
    h = harness
    h.fail_after = 1
    h.export_failure = True
    with pytest.raises(TimeoutError, match="completed optimizer step") as caught:
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["partial_checkpoint_error"]["message"] == "synthetic export failure"
    assert any("Partial checkpoint failed" in note for note in caught.value.__notes__)


def test_corrupt_or_partial_completion_write_preserved_but_never_accepted(harness, monkeypatch):
    h = harness
    original = fitting._json
    def writing(path, value):
        if path.name == "completed.json":
            path.write_text('{"status":')
            raise OSError("synthetic completion write failure")
        return original(path, value)
    monkeypatch.setattr(fitting, "_json", writing)
    with pytest.raises(OSError, match="completion write failure"):
        h.run()
    assert (h.out / "invalid-completion.json").read_text() == '{"status":'
    assert not (h.out / "completed.json").exists()
    assert read(h.out / "failed.json")["optimizer_steps"] == 4


def test_final_payload_io_failure_keeps_timing_and_full_partial_state(harness, monkeypatch):
    h = harness
    original = fitting._save_torch
    def save(path, payload):
        if path.name == "weights.pt":
            raise OSError("synthetic final weight write failure")
        return original(path, payload)
    monkeypatch.setattr(fitting, "_save_torch", save)
    with pytest.raises(OSError, match="final weight write"):
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["phase"] == "final_payload" and failure["payload_export_write_seconds"] > 0
    assert failure["flushed_updates"] == 4
    assert torch.load(h.out / "partial-checkpoint.pt", weights_only=True)["optimizer_steps"] == 4


def test_hash_failure_is_terminal_with_measured_hash_prefix(harness, monkeypatch):
    h = harness
    monkeypatch.setattr(fitting, "_sha", lambda *args: (_ for _ in ()).throw(TimeoutError("synthetic hash cap")))
    with pytest.raises(TimeoutError, match="hash cap"):
        h.run()
    failure = read(h.out / "failed.json")
    assert failure["phase"] == "manifest" and failure["file_hash_seconds"] > 0
    assert failure["optimizer_steps"] == 4 and not (h.out / "completed.json").exists()
