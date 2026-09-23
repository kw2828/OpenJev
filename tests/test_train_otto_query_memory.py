"""Fabricated producer admission, owned forks, canonical outputs and DEV barrier.

These fixtures do not open empirical arrays/checkpoints or launch any worker.
Synthetic model checks are intentionally small; the one fit uses query-only
single-step episodes and one locally patched epoch, not the scientific schedule.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import otto_query_memory_data as data
from openjev.research import otto_query_memory_gate as gate
from openjev.research import otto_query_memory_metrics as metrics
from openjev.research import otto_query_memory_model as models
from openjev.research import otto_query_memory_training as core

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_query_memory_producer_fixture", ROOT / "scripts/train_otto_query_memory.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fabricated_history(lengths=(65, 9, 1, 33, 5, 1), *, stage="train"):
    offsets = np.array([0, *np.cumsum(lengths)], np.int64)
    total = int(offsets[-1])
    features = np.linspace(-.2, .6, total * 31, dtype=np.float32).reshape(total, 31)
    targets = np.zeros((total, 4), np.float32)
    correction = np.zeros(total, np.bool_)
    identities = []
    for index, (low, high) in enumerate(pairwise(offsets)):
        steps = np.arange(high - low)
        features[low:high, 15] = steps / 2188
        features[low:high, 16] = (steps % 4) / 2188
        features[low:high, 17] = 1
        targets[low:high] = np.array([31, 34, 33, 32], np.float32) + index / 8 + steps[:, None] * np.array(
            [.25, -.125, .5, -.25], np.float32)
        correction[low:high] = steps % 4 == 0
        regime = "lambda3" if index % 6 < 3 else "lambda4"
        case = index // 6
        identities.append({"stage": stage, "episode_id": f"fabricated:{stage}:{index}", "episode_index": index,
            "seed": (303000001 if regime == "lambda3" else 304000001) + case,
            "case": case, "regime": regime, "arm": ("analytic", "neural", "period4_hold")[index % 3]})
    legal = np.ones((total, 4), np.bool_)
    legal[1::3, 3] = False
    flat = {"features": features, "raw_q": targets, "legal": legal,
            "actions": np.zeros(total, np.int64), "correction": correction, "episode_offsets": offsets}
    return data.project_census(flat, identities, query_period=4, expected_stage=stage), identities


def new_parent():
    parent = runner.construct_fit(models, "pretrained", runner.SEEDS[0])
    with torch.no_grad():
        parent.slow.output.weight.copy_(torch.arange(112).sin().reshape(4, 28) * .01)
        parent.slow.action_residual.weight.copy_(torch.arange(112).cos().reshape(4, 28) * .002)
        parent.slow.action_residual.bias.copy_(torch.tensor([.1, -.2, .3, -.1]))
    return parent


def numerical_run(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.np, run.torch, run.models = np, torch, models
    run.data, run.metrics, run.core = data, metrics, core
    clock = iter(range(1000, 1000000, 1000))
    run.clock = SimpleNamespace(now_ns=lambda: next(clock))
    run.check = lambda: None
    run.events = []
    run.event = lambda value, filename="progress.jsonl": run.events.append((filename, value))
    return run


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), **runner.descriptor(path)}


def test_static_schedule_and_payloads_match_gate_and_capacity_contract():
    assert runner.VIEWS == gate.FAMILIES
    assert runner.SEEDS == gate.FIT_SEEDS
    assert len(runner.PAYLOADS) == 80 and len(runner.COLLECTION_PAYLOADS) == 17
    assert sum(runner.EPOCHS.values()) * len(runner.SEEDS) * 9 == 7560
    assert sum(runner.EPOCHS.values()) * len(runner.SEEDS) * 54 == 45360
    assert sum(name.startswith("checkpoint-") for name in runner.PAYLOADS) == 18
    assert sum(name.startswith("train-prediction-") for name in runner.PAYLOADS) == 24
    assert sum(name.startswith("dev-prediction-") for name in runner.PAYLOADS) == 24
    assert all("test-prediction" not in name for name in runner.PAYLOADS)
    assert runner.CONFIG["test_decode"] is False
    assert {runner.CORE, runner.CORE_TEST, runner.SELF, runner.TEST} <= runner.NEW_COMPONENTS
    assert {runner.GATE, "tests/test_otto_query_memory_gate.py", runner.CAPACITY_SCRIPT} <= runner.NEW_SOURCES


@pytest.mark.parametrize("missing", sorted(runner.AUDIT_COMPONENTS))
def test_training_cannot_admit_source_pinned_but_unqualified_auditor(missing, monkeypatch):
    qualified = {name: {"sha256": "0" * 64, "bytes": 1}
                 for name in runner.NEW_COMPONENTS if name != missing}
    receipt = {"status": "passed", "source_before": qualified, "source_after": qualified,
               "commands": [{"returncode": 0, "timed_out": False, "reaped": True}]}
    monkeypatch.setattr(runner, "read", lambda path: receipt)
    monkeypatch.setattr(runner, "descriptor", lambda path: pytest.fail("qualification must precede source reads"))
    with pytest.raises(ValueError, match="all current fabricated component qualifications"):
        runner.authenticate_engineering("fabricated-receipt.json", {})


def test_owned_five_forks_share_slow_bytes_and_four_seeded_nonzero_projections():
    parent = new_parent()
    before = runner.state_witness(parent)
    rng = torch.random.get_rng_state().clone()
    branches = [runner.construct_fit(models, family, runner.SEEDS[0], parent.slow.state_dict())
                for family in runner.BRANCHES]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert runner.state_witness(parent) == before
    slow = runner.state_witness(parent, slow_only=True)
    assert all(runner.state_witness(item, slow_only=True) == slow for item in branches)
    for name in parent.slow.state_dict():
        assert len({item.slow.state_dict()[name].data_ptr() for item in [parent, *branches]}) == 6
    projections = [item.projection.weight for item in branches[1:]]
    assert all(torch.equal(weight, projections[0]) and bool((weight != 0).any()) for weight in projections)
    assert len({weight.data_ptr() for weight in projections}) == 4
    assert branches[0].parameter_metadata()["effective_count"] == 6112
    assert all(item.parameter_metadata()["count"] == 6336 and item.parameter_metadata()["effective_count"] == 224
               for item in branches[1:])
    with torch.no_grad():
        projections[0].add_(1)
    assert all(torch.equal(weight, projections[1]) for weight in projections[1:])
    assert runner.state_witness(parent) == before


@pytest.mark.parametrize("view", runner.VIEWS)
def test_canonical_clone_preserves_weights_flags_and_rng_without_parent_alias(view):
    parent = new_parent()
    family = runner.view_parent(view)
    trained = parent if family == "pretrained" else runner.construct_fit(models, family, runner.SEEDS[0], parent.slow.state_dict())
    before = runner.state_witness(trained)
    flags = [p.requires_grad for p in trained.parameters()]
    rng = torch.random.get_rng_state().clone()
    clone = runner.canonical_clone(models, trained, view, runner.SEEDS[0])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert runner.state_witness(trained) == before and [p.requires_grad for p in trained.parameters()] == flags
    assert clone.slow.mode == "frozen" and not clone.training
    assert all(p.requires_grad == name.startswith("action_residual.") for name, p in clone.slow.named_parameters())
    assert runner.state_witness(clone, slow_only=True) == runner.state_witness(trained, slow_only=True)
    for name, value in clone.slow.state_dict().items():
        assert value.data_ptr() != trained.slow.state_dict()[name].data_ptr()
    if clone.projection is not None:
        assert clone.projection.weight.requires_grad
        assert torch.equal(clone.projection.weight, trained.projection.weight)
        assert clone.projection.weight.data_ptr() != trained.projection.weight.data_ptr()


def test_missing_or_extra_slow_state_fails_before_any_branch_forward():
    source = new_parent().slow.state_dict()
    for state in ({name: value for name, value in source.items() if name != "output.bias"},
                  {**source, "extra": torch.zeros(1)}):
        with pytest.raises(ValueError, match="state keys"):
            runner.construct_fit(models, "trace_delta", runner.SEEDS[0], state)


def test_canonical_flat_outputs_no_write_parity_padding_carry_and_loss(tmp_path):
    history, identities = fabricated_history()
    source = new_parent()
    trained = runner.construct_fit(models, "trace_delta", runner.SEEDS[0], source.slow.state_dict())
    with torch.no_grad():
        trained.projection.weight.mul_(.7)
    run = numerical_run(tmp_path)
    observed = []
    reference = runner.canonical_clone(models, source, "pretrained", runner.SEEDS[0])

    def inspect(_module, _args, kwargs):
        observed.append((torch.is_grad_enabled(), kwargs["lengths"].tolist(), kwargs["carry"].fast.absolute_step.tolist()))
        assert kwargs["carry"].fast.matrix.grad_fn is None
        assert torch.isnan(kwargs["query_scores"][~kwargs["query_mask"]]).all()

    hook = reference.register_forward_pre_hook(inspect, with_kwargs=True)
    expected, record = run.predict(reference, "pretrained", runner.SEEDS[0], history, identities, "train")
    hook.remove()
    assert observed == [(False, [32, 9, 1, 32, 5, 1], [0] * 6),
                        (False, [32, 0, 0, 1, 0, 0], [32, 9, 1, 32, 5, 1]),
                        (False, [1, 0, 0, 0, 0, 0], [64, 9, 1, 33, 5, 1])]
    assert set(expected) == set(runner.PREDICTION_FIELDS)
    assert record["rows"] == 114 and record["chunks"] == 3
    assert len(run.events) == 6 and [event[1]["event"] for event in run.events] == ["attempt", "return"] * 3
    for name in ("action_prediction", "base_prediction", "slow_action_prediction"):
        assert expected[name][history["query_mask"]].tobytes() == history["query_scores"][history["query_mask"]].tobytes()
    no_write = runner.canonical_clone(models, trained, "trace_no_write", runner.SEEDS[0])
    saved, no_write_record = run.predict(no_write, "trace_no_write", runner.SEEDS[0], history, identities, "train")
    assert runner.same_frozen_predictions(saved, expected, no_write=True)
    assert no_write_record["work_counts"]["memory_matrix_writes"] == 0
    active = runner.canonical_clone(models, trained, "trace_delta", runner.SEEDS[0])
    active_saved, active_record = run.predict(active, "trace_delta", runner.SEEDS[0], history, identities, "train")
    assert runner.same_frozen_predictions(active_saved, expected)
    assert active_record["work_counts"]["memory_matrix_writes"] > 0
    assert active_saved["action_prediction"].tobytes() != expected["action_prediction"].tobytes()

    # Separate scalar reconstruction over saved rows, not the producer loss helper.
    nq, prior = [], []
    for low, high in zip(history["episode_offsets"][:-1], history["episode_offsets"][1:], strict=True):
        local_nq, local_prior = [], []
        for row in range(int(low), int(high)):
            if not history["query_mask"][row]:
                allowed = history["legal"][row]
                p = expected["action_prediction"][row, allowed].astype(np.float64) / 64
                q = history["targets"][row, allowed].astype(np.float64) / 64
                local_nq.append(float(np.mean(((p - p.mean()) - (q - q.mean())) ** 2)))
            elif history["prior_mask"][row]:
                p = expected["corrected_shadow_prior"][row].astype(np.float64) / 64
                q = history["targets"][row].astype(np.float64) / 64
                local_prior.append(float(np.mean(((p - p.mean()) - (q - q.mean())) ** 2)))
        nq.append(sum(local_nq) / len(local_nq) if local_nq else 0.)
        prior.append(sum(local_prior) / len(local_prior) if local_prior else 0.)
    assert record["loss"]["nonquery"] == pytest.approx(sum(nq) / 6, abs=1e-7)
    assert record["loss"]["prior"] == pytest.approx(sum(prior) / 6, abs=1e-7)


def test_off_schedule_targets_cannot_enter_canonical_predictions(tmp_path):
    history, identities = fabricated_history((9, 5, 1, 9, 5, 1))
    alternate = dict(history)
    alternate["targets"] = history["targets"].copy()
    alternate["targets"][~history["query_mask"]] += np.array([8, -4, 16, -8], np.float32)
    run = numerical_run(tmp_path)
    parent = new_parent()
    a, first = run.predict(runner.canonical_clone(models, parent, "pretrained", runner.SEEDS[0]),
        "pretrained", runner.SEEDS[0], history, identities, "train")
    b, second = run.predict(runner.canonical_clone(models, parent, "pretrained", runner.SEEDS[0]),
        "pretrained", runner.SEEDS[0], alternate, identities, "train")
    assert all(a[name].tobytes() == b[name].tobytes() for name in a)
    assert first["loss"]["nonquery"] != second["loss"]["nonquery"]


@pytest.mark.parametrize("stage", ("test", "TEST", "dev"))
def test_sealed_splits_reject_before_path_or_numpy_access(stage, tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.check = lambda: pytest.fail("must reject before even phase checks")
    with pytest.raises(ValueError, match="never decodes TEST|barrier before DEV"):
        run.decode_split(stage)
    assert run.receipt["test_array_decodes"] == 0 and not run.dev_allowed


def barrier_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.check = lambda: None
    run.receipt.update(fits_completed=18, optimizer_steps=7560, episode_exposures=45360)
    fits, views = [], []
    for seed in runner.SEEDS:
        for family in runner.FITS:
            name = f"checkpoint-{family}-{seed}.npz"
            (tmp_path / name).write_bytes(b"fabricated checkpoint metadata only\n")
            fits.append({"family": family, "seed": seed, "checkpoint_path": name, "checkpoint": runner.descriptor(tmp_path / name)})
        for view in runner.VIEWS:
            name = f"train-prediction-{view}-{seed}.npz"
            (tmp_path / name).write_bytes(b"fabricated prediction metadata only\n")
            views.append({"view": view, "seed": seed, "prediction_path": name, "prediction": runner.descriptor(tmp_path / name)})
    for name in ("fits.json", "forks.json", "train-views.json", "train-history.json", "train-history.npz"):
        (tmp_path / name).write_bytes(b"fabricated metadata only\n")
    return run, fits, views


def test_dev_barrier_opens_only_after_all_bytes_and_durable_journal(tmp_path, monkeypatch):
    run, fits, views = barrier_fixture(tmp_path, monkeypatch)
    events = []

    def event(value):
        assert not run.dev_allowed
        assert (tmp_path / "dev-barrier.json").is_file()
        events.append(value)

    run.event = event
    run.close_training(fits, views)
    assert run.dev_allowed and len(events) == 1
    barrier = runner.read(tmp_path / "dev-barrier.json")
    assert len(barrier["files"]) == 47
    assert all(runner.descriptor(tmp_path / name) == value for name, value in barrier["files"].items())


def test_dev_decode_has_durable_cross_journal_witness_before_numpy_access(tmp_path, monkeypatch):
    run, fits, views = barrier_fixture(tmp_path, monkeypatch)
    run.sequence = 123
    events = []
    run.event = lambda value: events.append(value)
    run.close_training(fits, views)
    assert runner.read(tmp_path / "dev-barrier.json")["work_sequence"] == 123
    (tmp_path / "dev.npz").write_bytes(b"fabricated opaque split; never decoded")
    run.collection_dir = tmp_path
    run.collection_receipt = {"files": {"dev.npz": runner.descriptor(tmp_path / "dev.npz")}}

    def attempted_load(*_args, **_kwargs):
        assert events[-1] == {"event": "dev_decode_after_barrier", "work_sequence": 123,
                             "barrier": runner.descriptor(tmp_path / "dev-barrier.json")}
        raise RuntimeError("fabricated stop before decoding")

    run.np = SimpleNamespace(load=attempted_load)
    with pytest.raises(RuntimeError, match="stop before decoding"):
        run.decode_split("dev")


@pytest.mark.parametrize("failure", ("missing_fit", "bad_bytes", "duplicate_path", "journal_failure"))
def test_failed_dev_barrier_never_grants_access(tmp_path, monkeypatch, failure):
    run, fits, views = barrier_fixture(tmp_path, monkeypatch)
    run.event = lambda *_: None
    if failure == "missing_fit":
        fits.pop()
    elif failure == "bad_bytes":
        (tmp_path / fits[-1]["checkpoint_path"]).write_bytes(b"tampered fixture")
    elif failure == "duplicate_path":
        fits[-1]["checkpoint_path"] = fits[0]["checkpoint_path"]
    else:
        def failed_event(*_):
            raise OSError("fabricated fsync failure")
        run.event = failed_event
    with pytest.raises((ValueError, OSError)):
        run.close_training(fits, views)
    assert not run.dev_allowed


def test_one_fabricated_query_only_epoch_records_all_nine_updates_and_durable_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setitem(runner.EPOCHS, "pretrained", 1)
    history, _ = fabricated_history((1,) * 54)
    run = numerical_run(tmp_path)
    model, fit = run.fit("pretrained", runner.SEEDS[0], history)
    assert fit["steps"] == 9 and fit["episode_exposures"] == 54
    assert fit["initial"] == fit["final"] == runner.state_witness(model)
    assert set(fit["optimizer_final_steps"].values()) == {9}
    assert fit["totals"]["backward_chunks"] == 0
    assert fit["totals"]["forward_rows"] == 54
    assert run.receipt["fits_completed"] == 1 and run.receipt["optimizer_steps"] == 9
    assert sorted(fit["episode_orders"][0]) == list(range(54))
    assert runner.descriptor(tmp_path / fit["checkpoint_path"]) == fit["checkpoint"]
    attempts = [row for _, row in run.events if row["event"] == "attempt"]
    returns = [row for _, row in run.events if row["event"] == "return"]
    assert len(attempts) == len(returns) == 9
    assert [row["episode_indices"] for row in attempts] == [row["result"]["episode_indices"] for row in returns]
    assert run.receipt["pending"] is None


def process_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    phase = tmp_path / "fabricated-phase"
    phase.mkdir()
    inputs = {"collection_plan": put(tmp_path / "plan.json", {"fixture": True})}
    inputs["collection_receipt"] = put(phase / "receipt.json", {"fixture": True})
    launch_path = tmp_path / "original.launch.json"
    command = [str(tmp_path / ".venv-otto-released-native/bin/python"), str(tmp_path / runner.COLLECTOR), "run",
        "--plan", inputs["collection_plan"]["path"], "--plan-sha256", inputs["collection_plan"]["sha256"],
        "--supervision", str(launch_path), "--output", str(phase)]
    launch = {"command": command, "started_ns": 100, "deadline_ns": 7200 * 10**9 + 100,
              "cap_seconds": 7200, "clock_source_sha256": runner.CLOCK_PIN,
              "watchdog_sha256": runner.SUPERVISOR_PIN, "pid": 101, "pgid": 101,
              "parent_pid": 100, "cwd": str(tmp_path)}
    pin = put(launch_path, launch)
    parent = {**launch, "status": "completed", "returncode": 0, "timed_out": False, "group_absent": True,
              "cleanup": {"reaped": True, "group_absent": True, "errors": []}, "error": None,
              "clock_error": None, "finished_ns": 1100, "timing_available": True}
    put(phase / "started.json", {"started_ns": 200, "launch": launch, "request": {"mode": "run", **{
        key[2:].replace("-", "_"): value for key, value in zip(command[3::2], command[4::2], strict=True)}}})
    inputs["collection_terminal"] = put(tmp_path / "original.terminal.json", parent)
    receipt = {"started_ns": 200, "finished_ns": 1000, "supervision_sha256": pin["sha256"]}
    return inputs, receipt, parent, launch


def test_original_successful_parent_requires_exact_launch_identity_and_argument_join(tmp_path, monkeypatch):
    inputs, receipt, parent, _ = process_fixture(tmp_path, monkeypatch)
    assert runner.successful_process(inputs, "collection", receipt, runner.COLLECTOR,
        ".venv-otto-released-native/bin/python", 7200) == parent
    parent["parent_pid"] += 10
    put(Path(inputs["collection_terminal"]["path"]), parent)
    with pytest.raises(ValueError, match="launch identity"):
        runner.successful_process(inputs, "collection", receipt, runner.COLLECTOR,
            ".venv-otto-released-native/bin/python", 7200)


@pytest.mark.parametrize("change", ({"returncode": 1}, {"returncode": False}, {"timed_out": True},
                                  {"timed_out": 0}, {"group_absent": False}, {"group_absent": 1},
                                  {"timing_available": False}, {"pgid": 555},
                                  {"cleanup": {"reaped": True, "group_absent": False, "errors": []}},
                                  {"finished_ns": 7200 * 10**9 + 101}, {"status": "launched"}))
def test_spawn_or_incomplete_terminal_is_not_success(tmp_path, monkeypatch, change):
    inputs, receipt, parent, _ = process_fixture(tmp_path, monkeypatch)
    parent.update(change)
    put(Path(inputs["collection_terminal"]["path"]), parent)
    with pytest.raises(ValueError, match="original successful"):
        runner.successful_process(inputs, "collection", receipt, runner.COLLECTOR,
            ".venv-otto-released-native/bin/python", 7200)


@pytest.mark.parametrize("change", ("started_clock", "started_launch", "request"))
def test_worker_started_record_must_join_original_launch_and_request(tmp_path, monkeypatch, change):
    inputs, receipt, _, _ = process_fixture(tmp_path, monkeypatch)
    path = Path(inputs["collection_receipt"]["path"]).parent / "started.json"
    started = runner.read(path)
    if change == "started_clock":
        started["started_ns"] += 1
    elif change == "started_launch":
        started["launch"]["parent_pid"] += 10
    else:
        started["request"]["output"] = str(tmp_path / "different-worker")
    put(path, started)
    with pytest.raises(ValueError, match="worker started/launch|worker request"):
        runner.successful_process(inputs, "collection", receipt, runner.COLLECTOR,
            ".venv-otto-released-native/bin/python", 7200)


def test_closed_payload_inventory_rejects_extra_or_missing_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "one.bin").write_bytes(b"fabricated")
    receipt = {"files": {"one.bin": runner.descriptor(tmp_path / "one.bin")}}
    put(tmp_path / "receipt.json", receipt)
    assert runner.closed_files(tmp_path / "receipt.json", receipt, {"one.bin"}) == tmp_path
    (tmp_path / "extra.bin").write_bytes(b"undeclared")
    with pytest.raises(ValueError, match="inventory"):
        runner.closed_files(tmp_path / "receipt.json", receipt, {"one.bin"})
    (tmp_path / "extra.bin").unlink()
    (tmp_path / "one.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload hash"):
        runner.closed_files(tmp_path / "receipt.json", receipt, {"one.bin"})


def test_missing_capacity_roles_fail_before_any_input_read(monkeypatch):
    monkeypatch.setattr(runner, "descriptor", lambda *_: pytest.fail("role admission must run first"))
    with pytest.raises(ValueError, match="exact required"):
        runner.authenticate_inputs({name: {} for name in runner.ROLES if not name.startswith("capacity_")})


def test_capacity_unadmitted_receipt_fails_before_source_import(monkeypatch):
    plan = {"version": runner.CAPACITY_VERSION, "status": "frozen_before_synthetic_work"}
    receipt = {"version": runner.CAPACITY_VERSION, "status": "completed", "complete": True, "admitted": False}
    monkeypatch.setattr(runner, "read", lambda value: plan if value == "plan" else receipt)
    monkeypatch.setattr(runner, "load", lambda *_: pytest.fail("unadmitted phase must not import"))
    with pytest.raises(ValueError, match="admitted new memory capacity"):
        runner.authenticate_capacity({"capacity_plan": {"path": "plan"}, "capacity_receipt": {"path": "receipt"}}, {})


def test_journal_failure_retains_pending_emission_without_terminal_success(tmp_path, monkeypatch):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.check = lambda: None

    def reject(_fd):
        raise OSError("fabricated journal fsync failure")

    monkeypatch.setattr(runner.os, "fsync", reject)
    with pytest.raises(OSError, match="fsync failure"):
        run.event({"event": "attempt", "call_id": 7}, "work.jsonl")
    assert run.receipt["pending_emission"] == {"file": "work.jsonl", "record": {"event": "attempt", "call_id": 7}}
    assert run.receipt["status"] == "started" and run.receipt["technical_complete"] is False


def test_failed_admission_preserves_original_failure_receipt_and_never_calls_body(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.signal, "signal", lambda *_: None)
    out = tmp_path / "failed-producer"
    run = runner.Run(SimpleNamespace(output=out))

    def reject():
        raise ValueError("fabricated admission rejection")

    run.admit = reject
    run.body = lambda: pytest.fail("body must remain unreachable")
    with pytest.raises(ValueError, match="admission rejection"):
        run.execute()
    result = runner.read(out / "receipt.json")
    assert result["status"] == "failed" and result["complete"] is False
    assert result["technical_complete"] is False and result["test_array_decodes"] == 0
    assert result["fits_completed"] == result["optimizer_steps"] == 0
    assert "fabricated admission rejection" in result["traceback"]


def test_bitwise_frozen_comparison_rejects_signed_zero_and_different_shape():
    values = {name: np.zeros((2, 4), np.float32) for name in runner.PREDICTION_FIELDS}
    changed = copy.deepcopy(values)
    changed["base_prediction"][0, 0] = -0.
    with pytest.raises(ValueError, match="base_prediction"):
        runner.same_frozen_predictions(changed, values)
    changed = copy.deepcopy(values)
    changed["shadow_prior"] = changed["shadow_prior"].reshape(4, 2)
    with pytest.raises(ValueError, match="shadow_prior"):
        runner.same_frozen_predictions(changed, values)
