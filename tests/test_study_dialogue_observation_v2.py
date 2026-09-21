"""Artificial metadata and tiny CPU routes; no pretrained assets or real data."""
from __future__ import annotations

import ast
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import study_dialogue_observation_v2 as s

from openjev.research.suspend_clock import ClockError, SuspendClock


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    state = SimpleNamespace(ns=10_000_000_000, broken=False, clocks=[])

    def reader():
        if state.broken:
            raise OSError("Artificial native clock failure")
        return state.ns

    def factory():
        clock = SuspendClock(reader)
        clock.backend = "mach_continuous_time"  # Pretend native identity; all readings are injected.
        state.clocks.append(clock)
        return clock

    monkeypatch.setattr(s, "SuspendClock", factory)
    return state


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def allocation_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "ROOT", tmp_path)
    protocol = tmp_path / s.PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_text("Artificial prospective protocol")
    failed = tmp_path / s.FAILED_MANIFEST
    save(failed, {"status": "failed_technical_timing", "resume_permitted": False,
                  "partial_scoring_permitted": False, "quality_metrics_opened": False,
                  "individual_predictions_decoded": False, "weights_loaded": False})
    monkeypatch.setattr(s, "FAILED_MANIFEST_PIN", s.sha(failed))
    value = {"version": s.ALLOCATION_VERSION, "prepared_completed_sha256": s.cost.PREPARED_PIN,
             "prepared_plan_sha256": s.cost.PREPARED_PLAN_PIN,
             "cost_completed": {"path": str(tmp_path / "cost.json"), "sha256": "a" * 64},
             "cost_audit": {"path": str(tmp_path / "audit.json"), "sha256": "b" * 64},
             "failed_attempt": {"path": str(failed), "sha256": s.sha(failed)},
             "protocol": {"path": str(protocol), "sha256": s.sha(protocol)},
             "limits": {"wall_seconds": 60, "rss_bytes": 8 * 1024**3,
                        "mps_driver_bytes": 8 * 1024**3, "output_bytes": 10 * 1024**2},
             "freeze_limits": {"wall_seconds": 30, "rss_bytes": 8 * 1024**3, "output_bytes": 10 * 1024**2}}
    path = tmp_path / "allocation.json"
    save(path, value)
    return path, value


def test_external_allocation_has_no_defaults_and_uses_exact_pin(tmp_path, monkeypatch):
    path, value = allocation_fixture(tmp_path, monkeypatch)
    assert s.allocation(path, s.sha(path)) == value
    with pytest.raises(ValueError, match="identity"):
        s.allocation(path, "0" * 64)


@pytest.mark.parametrize("mutation", ["missing", "extra", "zero_wall", "bool_bytes", "nan_wall", "wrong_parent"])
def test_invalid_allocation_rejected(tmp_path, monkeypatch, mutation):
    path, value = allocation_fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        del value["limits"]
    elif mutation == "extra":
        value["automatic_admission"] = True
    elif mutation == "zero_wall":
        value["limits"]["wall_seconds"] = 0
    elif mutation == "bool_bytes":
        value["freeze_limits"]["rss_bytes"] = True
    elif mutation == "nan_wall":
        value["limits"]["wall_seconds"] = float("nan")
    else:
        value["prepared_completed_sha256"] = "0" * 64
    save(path, value)
    with pytest.raises(ValueError):
        s.allocation(path, s.sha(path))


def test_recursive_membership_and_tamper(tmp_path):
    save(tmp_path / "fit" / "completed.json", {"status": "completed"})
    (tmp_path / "fit" / "opaque-weights.pt").write_bytes(b"never deserialized")
    expected = s.members(tmp_path)
    assert set(expected) == {"fit/completed.json", "fit/opaque-weights.pt"}
    s.check_manifest(tmp_path, expected)
    (tmp_path / "extra").touch()
    with pytest.raises(ValueError, match="closure"):
        s.check_manifest(tmp_path, expected)
    (tmp_path / "extra").unlink()
    (tmp_path / "fit" / "opaque-weights.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity"):
        s.check_manifest(tmp_path, expected)


def canonical_fixture(tmp_path):
    query = {"id": '["Fake","flag"]', "split": "dev", "service": "Fake", "slot": "flag",
             "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:True", "value:False"],
             "candidate_values": [None, None, "True", "False"]}
    rows = [{"split": "dev", "dialogue_id": "same", "source_row_index": i, "time": t,
             "query_position": 0, "query_index": 0, "query_id": query["id"], "service": "Fake", "slot": "flag",
             "label_index": c, "label_id": query["candidate_ids"][c], "bin": b, "unseen": True}
            for i, (t, c, b) in enumerate(((1, 2, "first_assignment"), (0, 0, "unmentioned_retention")))]
    record = {"split": "dev", "dialogue_id": "same", "rows": rows,
              "literal_registers": {"original": [[0, 0]], "numbers": [[0, 2]]}}
    (tmp_path / "targets-dev.jsonl").write_text(json.dumps(record) + "\n")
    return query, record


def test_canonical_rows_preserve_source_order_and_exact_public_values(tmp_path):
    query, record = canonical_fixture(tmp_path)
    rows, refs = s.canonical_evaluation(tmp_path, [query])
    assert [r["time"] for r in rows] == [1, 0]
    assert [r["row_index"] for r in rows] == [0, 1]
    assert rows[0]["candidate_values"] == [None, None, "True", "False"]
    assert "boolean" not in rows[0]
    assert refs["original"] == [0, 0] and refs["numbers"] == [2, 0]
    record["rows"][0]["label_id"] = "value:False"
    (tmp_path / "targets-dev.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="support"):
        s.canonical_evaluation(tmp_path, [query])


def test_public_actor_assembly_has_no_target_argument_or_input():
    class PublicOnly(dict):
        def __getitem__(self, key):
            assert key not in ("label", "bin", "previous", "targets")
            return super().__getitem__(key)

    record = PublicOnly(split="train", dialogue_id="d", lexical_shape=[2, 1, 3, 10], lexical_offset=0,
                        tokens=[[1]], turn_text_ids=[0, 0], query_text_ids=[0], candidate_text_ids=[[0, 0, 0]])
    arrays = {"original": np.zeros(60, np.float32), "numbers": np.ones(60, np.float32)}
    a = s.payload_actor(record, arrays, "frozen_original")
    b = s.payload_actor(record, arrays, "trainable_original")
    np.testing.assert_array_equal(a["lexical"], b["lexical"])
    assert not {"labels", "targets", "previous"} & a.keys()
    assert s.payload_actor(record, arrays, "frozen_numbers")["lexical"].sum() == 60


def toy_workload():
    one = dict.fromkeys(s.cost.ENCODER_KEYS, 0)
    one["encoder_calls"] = 1
    one.update(real_question_updates=2, public_user_turns=2)
    return {"splits": {"train": {"totals": {k: v * 2 for k, v in one.items()}}, "dev": {"totals": one}}}


def test_expected_full_counts_include_every_public_step_and_encoder_component():
    parent = {"cohort_sizes": {"train": 2017, "dev": 2363}, "config": {"epochs": 20, "effective_batch": 32},
              "loss_counts": {"train": {"0": 32718, "1": 14331, "2": 4692}, "dev": {"0": 38662, "1": 17959, "2": 5708}}}
    work = toy_workload()
    work["splits"]["train"]["totals"]["encoder_calls"] = 3444
    work["splits"]["dev"]["totals"]["encoder_calls"] = 4121
    expected = s.expected_counts(parent, work)
    assert expected["per_fit"]["optimizer_updates"] == 1280
    assert expected["all_fits"]["training_dialogue_visits"] == 484080
    assert expected["all_fits"]["training_endpoints"] == 12417840
    assert expected["all_fits"]["evaluation_endpoints"] == 747948
    assert expected["all_fits"]["encoder_calls"] == 876012
    assert set(expected["encoder_work_per_fit"]) == set(s.cost.ENCODER_KEYS)


def test_failure_before_allocation_and_exclusive_output_preserve_identity(tmp_path):
    args = SimpleNamespace(command="freeze", allocation=tmp_path / "missing", allocation_sha256="0" * 64,
                           prepared=tmp_path / "prepared", out=tmp_path / "attempt")
    with pytest.raises(FileNotFoundError):
        s.execute(args)
    failed = s.read(args.out / "failed.json")
    assert failed["request"]["allocation_sha256"] == "0" * 64
    before = (args.out / "failed.json").read_bytes()
    with pytest.raises(FileExistsError):
        s.execute(args)
    assert (args.out / "failed.json").read_bytes() == before


def test_late_completion_output_failure_demotes_success(tmp_path, monkeypatch):
    path, value = allocation_fixture(tmp_path, monkeypatch)
    value["freeze_limits"]["output_bytes"] = 4096
    save(path, value)
    def fake_freeze(args, *_):
        (args.out / "oversize").write_bytes(b"x" * 4097)
        return {"plan_sha256": "1" * 64}
    monkeypatch.setattr(s, "freeze", fake_freeze)
    args = SimpleNamespace(command="freeze", allocation=path, allocation_sha256=s.sha(path),
                           prepared=tmp_path / "prepared", out=tmp_path / "attempt")
    with pytest.raises(ValueError, match="Output cap"):
        s.execute(args)
    assert (args.out / "failed.json").exists() and (args.out / "invalid-completion.json").exists()
    assert not (args.out / "completed.json").exists()


def test_forward_failure_keeps_paid_encoder_work_partial_audit_and_returned_bad_logs(tmp_path):
    import torch

    class FakeMemory:
        def begin_batch(self, *_):
            self.audit = {"checks": 0}

        def __call__(self, *_args, **_kwargs):
            self.audit["checks"] = 2
            return torch.tensor([[[[float("nan"), 0.]]]])

    route = s.TorchRoute.__new__(s.TorchRoute)
    route.torch, route.parent, route.trainable = torch, {"tokenizer_ids": {}}, False
    route.encoder, route.memory = object(), FakeMemory()
    route.counts, route.work, route.audit = Counter(), Counter(), Counter()
    route.artifact_dir, route.progress = tmp_path, {"fit_id": "fake-fit", "operation": "training"}
    route.encode_tokens = lambda *_args, **_kwargs: (torch.ones(1, 2), {"encoder_calls": 3})
    actor = [None, None, None, None, torch.ones(1, 1, 2, dtype=torch.bool)]
    route.build_actor = lambda *_args, **_kwargs: (actor, None)
    route.merge = lambda target, source: target.update(source)
    payload = {"tokens": [[1]], "turn_text_ids": [0], "query_text_ids": [0], "candidate_text_ids": [[0, 0]],
               "candidate_ids": [["reserved:NOT_MENTIONED", "reserved:DONTCARE"]], "lexical": [],
               "lexical_shape": [1, 1, 2, 10], "split": "train", "dialogue_id": "fake"}
    with pytest.raises(ValueError, match="Finite supported"):
        route.forward(payload, True)
    assert route.work["encoder_calls"] == 3 and route.audit["checks"] == 2
    assert route.counts["encoding_passes"] == route.counts["memory_forwards"] == 1
    assert np.isnan(np.load(tmp_path / "offending-log-probs.npy", allow_pickle=False)).any()
    assert s.read(tmp_path / "offending-context.json")["context"]["fit_id"] == "fake-fit"


def test_effective_update_uses_unequal_endpoint_denominator_and_immediate_backward():
    import torch

    from openjev.research.dialogue_finetune_training import supervised_loss

    torch.set_num_threads(1)
    parameter = torch.nn.Parameter(torch.tensor([.2, -.1, .4], dtype=torch.float64))
    counts, work, seen = Counter(), Counter(), []
    budget = SimpleNamespace(synchronize=lambda _torch: None)
    route = SimpleNamespace(torch=torch, counts=counts, work=work, budget=budget,
                            empty=lambda: Counter(), merge=lambda a, b: a.update(b),
                            optimizer=torch.optim.SGD([parameter], lr=.1))
    def forward(payload, _training):
        if seen:
            assert parameter.grad is not None  # Prior dialogue backward occurred already.
        return parameter.log_softmax(-1)[None, None, None, :].expand(1, 2, 1, 3), {"calls": 1}, {"steps": 2}
    def loss(logp, rows, weights, denominator):
        seen.append((len(rows), denominator))
        return supervised_loss(logp, rows, weights, denominator)
    def step():
        route.optimizer.step()
        counts["optimizer_updates"] += 1
        return {}
    route.forward, route.loss, route.step = forward, loss, step
    base = {"lexical_shape": [2, 1, 3, 10], "lexical_offset": 0}
    actors = {("train", name): base for name in ("a", "b")}
    targets = {("train", "a"): [{"time": 0, "query_position": 0, "label_index": 0, "stratum_index": 0},
                                 {"time": 1, "query_position": 0, "label_index": 1, "stratum_index": 1}],
               ("train", "b"): [{"time": 1, "query_position": 0, "label_index": 2, "stratum_index": 2}]}
    lexical = {"original": np.zeros(60, np.float32)}
    initial = parameter.detach().clone().requires_grad_()
    expected = -(initial.log_softmax(-1) * torch.tensor([.5, 1.2, 3.7], dtype=torch.float64)).sum() / 3
    expected.backward()
    event = s.update(route, ["a", "b"], actors, targets, lexical, "frozen_original", [.5, 1.2, 3.7], {})
    assert seen == [(2, 3), (1, 3)] and counts["backward_calls"] == 2 and counts["optimizer_updates"] == 1
    assert event["endpoint_count"] == 3 and event["microbatches"] == 2
    torch.testing.assert_close(parameter.detach(), initial.detach() - .1 * initial.grad, atol=1e-14, rtol=1e-14)


def test_evaluation_preserves_raw_logs_order_and_partial_prefix_on_failure(tmp_path):
    import torch

    torch.set_num_threads(1)
    logs = torch.tensor([[[[-.3132616875, -1.3132616875]], [[-1.3132616875, -.3132616875]]]], dtype=torch.float32)
    route = SimpleNamespace(torch=torch, counts=Counter(), empty=lambda: Counter(), merge=lambda a, b: a.update(b),
                            budget=SimpleNamespace(synchronize=lambda _torch: None, storage=lambda: None),
                            forward=lambda *_: (logs, {"calls": 1}, {"steps": 2}))
    record = {"split": "dev", "dialogue_id": "d", "candidate_ids": [["n", "d"]],
              "lexical_shape": [2, 1, 2, 10], "lexical_offset": 0}
    targets = [{"source_row_index": i, "time": t, "query_position": 0, "query_index": 9,
                "label_index": 0, "bin": "unmentioned_retention"} for i, t in enumerate((1, 0))]
    rows = [{**r, "row_index": i, "dialogue_id": "d", "candidate_ids": ["n", "d"]} for i, r in enumerate(targets)]
    arrays = {"original": np.zeros(40, np.float32)}
    good = tmp_path / "good"
    good.mkdir()
    s.evaluate(route, {("dev", "d"): record}, {("dev", "d"): targets}, arrays, "frozen_original", rows, good, {})
    with np.load(good / "predictions.npz", allow_pickle=False) as result:
        np.testing.assert_array_equal(result["log_probs"][:, :2], logs.numpy()[0, [1, 0], 0])
        assert np.isneginf(result["log_probs"][:, 2:]).all()
        assert result["log_probs"].dtype == np.float32 and result["row_indices"].dtype == np.int64
    assert not (good / "partial-log-probs.npy").exists()
    bad = tmp_path / "bad"
    bad.mkdir()
    changed = copy.deepcopy(rows)
    changed[1]["time"] = 1
    with pytest.raises(ValueError, match="membership/order"):
        s.evaluate(route, {("dev", "d"): record}, {("dev", "d"): targets}, arrays, "frozen_original", changed, bad, {})
    assert np.load(bad / "partial-row-indices.npy", allow_pickle=False).tolist() == [0, -1]
    assert not (bad / "predictions.npz").exists()


def fake_campaign(tmp_path, monkeypatch, fail_load=None):
    """Exercise the actual twelve-fit controller without any encoder or neural calls."""
    import torch

    prepared, frozen, out = (tmp_path / name for name in ("prepared", "frozen", "run"))
    for path in (prepared, frozen, out):
        path.mkdir()
    ids = [f"d{i}" for i in range(33)]
    orders = {str(seed): [list(range(33)) if epoch % 2 == 0 else list(reversed(range(33)))
                          for epoch in range(20)] for seed in s.SEEDS}
    save(prepared / "orders.json", {"seeds": s.SEEDS, "epochs": 20, "dialogue_ids": ids, "orders": orders})
    for name in ("allocation.json", "references.json", "plan.json"):
        save(frozen / name, {})
    (frozen / "evaluation-rows.jsonl").write_text('{}\n')
    parent = {"cohort_sizes": {"train": 33, "dev": 1}, "config": {"epochs": 20, "effective_batch": 32},
              "loss_counts": {"train": {"0": 33}, "dev": {"0": 1}}, "loss_weights": [1., 1., 1.]}
    per_dialogue = dict.fromkeys(s.cost.ENCODER_KEYS, 1)
    totals = {**per_dialogue, "real_question_updates": 1, "public_user_turns": 1}
    workload = {"splits": {"train": {"totals": {k: 33 * v for k, v in totals.items()}},
                           "dev": {"totals": totals}}}
    expected = s.expected_counts(parent, workload)
    plan = {"prepared": str(prepared), "expected": expected, "allocation_sha256": "a" * 64, "source_sha256": {}}
    actors = {("train", did): {} for did in ids} | {("dev", "same-id-is-allowed"): {}}
    monkeypatch.setattr(s, "validate_plan", lambda *_: (plan, parent))
    monkeypatch.setattr(s, "load_training_inputs", lambda *_: (actors, {}, {}))
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *_: None)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(s.cost.qualified, "tensor_digest", lambda obj: obj)
    constructed, calls = [], []

    class Route:
        def __init__(self, _parent, seed, arm, _budget):
            if len(constructed) == fail_load:
                raise RuntimeError("Artificial next model load failure")
            constructed.append((seed, arm))
            self.seed, self.arm, self.trainable = seed, arm, arm.startswith("trainable_")
            self.initial = {"encoder": "e" * 64, "memory": str(seed).zfill(64)}
            self.encoder = "f" * 64 if self.trainable else self.initial["encoder"]
            self.counts, self.work, self.audit = Counter(), Counter(), Counter()

        def checkpoint(self, path):
            path.write_bytes(b"artificial opaque weight artifact")
            return {"sha256": s.sha(path), "bytes": path.stat().st_size, "wall_seconds": .01}

        def observe(self, visits, training):
            self.work.update({k: v * visits for k, v in per_dialogue.items()})
            self.audit.update(dict.fromkeys(expected["state_counts_per_fit"], visits))
            self.counts.update(dict.fromkeys(("encoding_attempts", "encoding_passes", "memory_forward_attempts",
                                              "memory_forwards", "encoder_forward_attempts", "encoder_forward_returns"), visits))
            if training:
                self.counts.update(dict.fromkeys(("training_dialogue_visits", "training_endpoints",
                                                  "backward_attempts", "backward_calls"), visits))
                self.counts.update(optimizer_attempts=1, optimizer_updates=1)
            else:
                self.counts.update(evaluation_forwards=visits, evaluation_endpoints=visits)

    def fake_update(route, batch_ids, *_args):
        calls.append((route.seed, route.arm, list(batch_ids)))
        route.observe(len(batch_ids), True)
        return {"dialogue_ids": batch_ids, "microbatches": len(batch_ids), "endpoint_count": len(batch_ids)}

    def fake_evaluate(route, _actors, _targets, _lexical, _arm, _rows, directory, _progress):
        before = route.counts["optimizer_updates"], route.counts["backward_calls"]
        route.observe(1, False)
        assert before == (route.counts["optimizer_updates"], route.counts["backward_calls"])
        (directory / "predictions.npz").write_bytes(b"artificial opaque prediction artifact")
        return {"rows": 1}

    monkeypatch.setattr(s, "TorchRoute", Route)
    monkeypatch.setattr(s, "update", fake_update)
    monkeypatch.setattr(s, "evaluate", fake_evaluate)
    args = SimpleNamespace(plan=frozen / "plan.json", plan_sha256="b" * 64, out=out)
    budget = SimpleNamespace(storage=lambda: None, synchronize=lambda *_: None, max_driver=0, max_current=0)
    return args, budget, expected, constructed, calls, ids


def test_all_twelve_fits_match_initializers_orders_and_tail_without_dev_updates(tmp_path, monkeypatch):
    args, budget, expected, constructed, calls, ids = fake_campaign(tmp_path, monkeypatch)
    progress = {}
    result = s.train(args, {}, budget, progress)
    assert constructed == [(seed, arm) for seed in s.SEEDS for arm in s.ARMS]
    assert result["counts"] == expected["all_fits"] and result["fit_count"] == 12
    assert len(calls) == 12 * 20 * 2
    for seed in s.SEEDS:
        order_lists = [[batch for ss, aa, batch in calls if ss == seed and aa == arm] for arm in s.ARMS]
        assert all(order == order_lists[0] for order in order_lists)
        assert order_lists[0][:4] == [ids[:32], ids[32:], list(reversed(ids))[0:32], [ids[0]]]
        records = [s.read(args.out / f"{arm}-{seed}" / "completed.json") for arm in s.ARMS]
        assert all(record["initial_sha256"] == records[0]["initial_sha256"] for record in records)
    assert result["quality_scoring_in_runner"] is False


def test_next_model_load_failure_does_not_mislabel_previous_fit_witnesses(tmp_path, monkeypatch):
    args, budget, _expected, _constructed, _calls, _ids = fake_campaign(tmp_path, monkeypatch, fail_load=1)
    progress = {}
    with pytest.raises(RuntimeError, match="next model load"):
        s.train(args, {}, budget, progress)
    assert progress["fit_id"] == s.FIT_ORDER[1] and progress["completed_fits"] == 1
    assert progress["operation"] == "fresh-model-load"
    assert not any(key.startswith("active_") for key in progress)
    assert "epoch" not in progress and "microbatch" not in progress
    assert (args.out / s.FIT_ORDER[0] / "completed.json").exists()


def supervision_fixture(tmp_path, monkeypatch, fake_clock):
    path, spec = allocation_fixture(tmp_path, monkeypatch)
    plan = tmp_path / "plan.json"
    save(plan, {"allocation_sha256": s.sha(path)})
    for name in (s.CLOCK_SOURCE, s.SUPERVISOR_SOURCE, "scripts/study_dialogue_observation_v2.py"):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("Artificial metadata-only source fixture")
    args = SimpleNamespace(command="train", plan=plan, plan_sha256=s.sha(plan),
                           out=tmp_path / "run", supervision=tmp_path / "parent.launch.json")
    argv = [str(tmp_path / "scripts/study_dialogue_observation_v2.py"), "train", "--plan", str(plan),
            "--plan-sha256", args.plan_sha256, "--out", str(args.out), "--supervision", str(args.supervision)]
    monkeypatch.setattr(s.sys, "argv", argv)
    monkeypatch.setattr(s.os, "getpid", lambda: 444)
    monkeypatch.setattr(s.os, "getpgrp", lambda: 444)
    monkeypatch.setattr(s.os, "getppid", lambda: 333)
    launch = {"version": "dialogue-observation-supervision-v2", "command": [s.sys.executable, "-u", *argv],
              "cwd": str(Path.cwd()), "pid": 444, "pgid": 444, "parent_pid": 333,
              "clock_backend": "mach_continuous_time", "started_ns": fake_clock.ns - 1_000_000_000,
              "deadline_ns": fake_clock.ns + 59_000_000_000, "cap_seconds": 60,
              "watchdog_sha256": s.sha(tmp_path / s.SUPERVISOR_SOURCE),
              "clock_source_sha256": s.sha(tmp_path / s.CLOCK_SOURCE)}
    save(args.supervision, launch)
    return args, spec, launch


def test_shared_parent_deadline_is_not_restarted_at_worker_start(tmp_path, monkeypatch, fake_clock):
    args, spec, launch = supervision_fixture(tmp_path, monkeypatch, fake_clock)
    clock = s.SuspendClock()
    started = fake_clock.ns
    deadline = s.validate_supervision(args, launch, clock, started, spec["limits"])
    assert deadline.started_ns < started and deadline.remaining_ns() == 59_000_000_000
    budget = s.Budget(tmp_path, deadline, spec["limits"])
    monkeypatch.setattr(s.time, "time", lambda: -(10**30))
    monkeypatch.setattr(s.time, "perf_counter", lambda: 0.)
    fake_clock.ns = deadline.expires_ns
    with pytest.raises(TimeoutError):
        budget.check()


@pytest.mark.parametrize("field,value", [("pid", 555), ("pgid", 555), ("parent_pid", 555),
    ("clock_backend", "CLOCK_MONOTONIC"), ("cap_seconds", 61), ("started_ns", 11_000_000_000),
    ("deadline_ns", 70_000_000_000), ("clock_source_sha256", "0" * 64), ("watchdog_sha256", "0" * 64),
    ("command", ["different-command"])])
def test_supervision_identity_rejects_mismatches(tmp_path, monkeypatch, fake_clock, field, value):
    args, spec, launch = supervision_fixture(tmp_path, monkeypatch, fake_clock)
    launch[field] = value
    with pytest.raises(ValueError):
        s.validate_supervision(args, launch, s.SuspendClock(), fake_clock.ns, spec["limits"])


def test_missing_launch_times_out_before_training_or_model_import(tmp_path, monkeypatch, fake_clock):
    args, _spec, _launch = supervision_fixture(tmp_path, monkeypatch, fake_clock)
    args.supervision.unlink()
    calls = []
    monkeypatch.setattr(s, "train", lambda *_: calls.append("must not enter"))
    monkeypatch.setattr(s.time, "sleep", lambda _: setattr(fake_clock, "ns", fake_clock.ns + 5_000_000_000))
    with pytest.raises(TimeoutError):
        s.execute(args)
    failed = s.read(args.out / "failed.json")
    assert not calls and failed["error_type"] == "TimeoutError" and failed["no_retry"] is True
    assert failed["elapsed_ns"] == 5_000_000_000


def test_root_clock_records_parent_enclosure_and_original_phase_scope(tmp_path, monkeypatch, fake_clock):
    args, _spec, launch = supervision_fixture(tmp_path, monkeypatch, fake_clock)

    def fake_train(*_):
        fake_clock.ns += 2_000_000_000
        return {"fits": [], "quality_scoring_in_runner": False}

    monkeypatch.setattr(s, "train", fake_train)
    s.execute(args)
    start, done = s.read(args.out / "started.json"), s.read(args.out / "completed.json")
    assert start["started_ns"] == 10_000_000_000 and start["finished_ns"] is None
    assert done["parent_started_ns"] == launch["started_ns"] and done["deadline_ns"] == launch["deadline_ns"]
    assert done["elapsed_ns"] == 2_000_000_000 and done["wall_seconds"] == 2.
    assert done["supervision_sha256"] == s.sha(args.supervision)
    assert "perf_counter" in done["phase_timing_scope"] and done["no_retry"] is True


def test_suspend_during_completion_write_invalidates_late_success(tmp_path, monkeypatch, fake_clock):
    args, _spec, launch = supervision_fixture(tmp_path, monkeypatch, fake_clock)
    monkeypatch.setattr(s, "train", lambda *_: {"quality_scoring_in_runner": False})
    original_write = s.write

    def delayed_write(path, value):
        original_write(path, value)
        if path.name == "completed.json":
            fake_clock.ns = launch["deadline_ns"]

    monkeypatch.setattr(s, "write", delayed_write)
    with pytest.raises(TimeoutError):
        s.execute(args)
    assert not (args.out / "completed.json").exists() and (args.out / "invalid-completion.json").exists()
    failed = s.read(args.out / "failed.json")
    assert failed["finished_ns"] == launch["deadline_ns"] and failed["error_type"] == "TimeoutError"


def test_clock_failure_retains_original_exception_with_null_elapsed(tmp_path, monkeypatch, fake_clock):
    args, _spec, _launch = supervision_fixture(tmp_path, monkeypatch, fake_clock)

    def fail(*_):
        fake_clock.broken = True
        raise RuntimeError("Original artificial training failure")

    monkeypatch.setattr(s, "train", fail)
    with pytest.raises(RuntimeError, match="Original artificial"):
        s.execute(args)
    failed = s.read(args.out / "failed.json")
    assert failed["error_type"] == "RuntimeError" and failed["timing_available"] is False
    assert all(failed[key] is None for key in ("finished_ns", "elapsed_ns", "wall_seconds"))
    assert failed["started_ns"] == 10_000_000_000 and failed["no_retry"] is True


def test_native_clock_initialization_failure_is_receipted(tmp_path, monkeypatch):
    args = SimpleNamespace(command="freeze", out=tmp_path / "run", allocation=tmp_path / "none", allocation_sha256="x")

    def unavailable():
        raise ClockError("Artificial unavailable native backend")

    monkeypatch.setattr(s, "SuspendClock", unavailable)
    with pytest.raises(ClockError):
        s.execute(args)
    failed = s.read(args.out / "failed.json")
    assert failed["timing_available"] is False and failed["clock_backend"] is None
    assert failed["wall_seconds"] is None and (args.out / "started.json").exists()


def test_clock_break_on_success_path_cannot_complete(tmp_path, monkeypatch, fake_clock):
    args, _spec, _launch = supervision_fixture(tmp_path, monkeypatch, fake_clock)

    def nominal_return(*_):
        fake_clock.broken = True
        return {}

    monkeypatch.setattr(s, "train", nominal_return)
    with pytest.raises(ClockError):
        s.execute(args)
    assert not (args.out / "completed.json").exists()
    assert s.read(args.out / "failed.json")["timing_available"] is False


@pytest.mark.parametrize("key", ["resume_permitted", "partial_scoring_permitted", "quality_metrics_opened",
                                "individual_predictions_decoded", "weights_loaded"])
def test_failed_prior_attempt_cannot_be_relabelled(tmp_path, monkeypatch, key):
    path, value = allocation_fixture(tmp_path, monkeypatch)
    manifest = Path(value["failed_attempt"]["path"])
    record = s.read(manifest)
    record[key] = True
    save(manifest, record)
    monkeypatch.setattr(s, "FAILED_MANIFEST_PIN", s.sha(manifest))
    value["failed_attempt"]["sha256"] = s.sha(manifest)
    save(path, value)
    with pytest.raises(ValueError, match="failed, unscored and unresumed"):
        s.allocation(path, s.sha(path))


def test_scientific_functions_and_model_route_are_ast_identical_to_v1():
    base = Path(__file__).resolve().parents[1] / "scripts"
    def nodes(name):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse((base / name).read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old, new = nodes("study_dialogue_observation.py"), nodes("study_dialogue_observation_v2.py")
    unchanged = ["expected_counts", "canonical_evaluation", "load_training_inputs", "payload_actor",
                 "TorchRoute", "update", "evaluate", "train"]
    assert all(old[name] == new[name] for name in unchanged)


def test_exact_64_source_closure_preserves_all_old_53(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "ROOT", tmp_path)
    old_names = [f"old/cost-{i}.py" for i in range(44)]
    for name in [*old_names, *s.OLD_NEW_SOURCES, *s.NEW_SOURCES]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Artificial source " + name)
    historical = {name: s.sha(tmp_path / name) for name in [*old_names, *s.OLD_NEW_SOURCES]}
    save(tmp_path / s.V1_PLAN, {"source_sha256": historical})
    monkeypatch.setattr(s, "V1_PLAN_PIN", s.sha(tmp_path / s.V1_PLAN))
    cost_plan = {"source_sha256": {name: historical[name] for name in old_names}}
    result = s.source_map(cost_plan)
    assert len(result) == 64 and {k: result[k] for k in historical} == historical
    (tmp_path / s.OLD_NEW_SOURCES[0]).write_text("Unauthorized frozen-source change")
    with pytest.raises(ValueError, match="source changed"):
        s.source_map(cost_plan)
