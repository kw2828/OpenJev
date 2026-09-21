"""Artificial saved packets and fake inference only; no pretrained assets."""
from __future__ import annotations

import copy
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("calibration_replay_tested", ROOT / "scripts/replay_dialogue_calibration.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class Tensor:
    """Only the already-CPU detach/numpy boundary needed by this controller."""
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Budget:
    def __init__(self):
        self.ticks = 0
        self.progress = {}
        self.syncs = 0

    def check(self):
        pass

    def storage(self):
        pass

    def elapsed(self):
        self.ticks += 1
        return self.ticks / 100

    def sync(self, _torch):
        self.syncs += 1


def work():
    return {"input_texts": 2, "content_tokens": 3, "encoder_sequences": 2, "encoder_calls": 1,
            "special_token_positions": 4, "valid_token_positions": 7, "padded_token_positions": 8,
            "padded_attention_positions": 32, "padding_token_positions": 1, "overlength_texts_chunked": 0,
            "truncated_tokens": 0, "public_user_turns": 2, "real_question_updates": 2}


def invariants():
    return {"forward_calls": 1, "forward_returned": 1, "advance_calls": 2, "advance_returned": 2,
            "valid_turns": 2, "executed_valid_question_slots": 2, "real_question_updates": 2,
            "incoming_checks": 2, "feature_checks": 2, "result_checks": 2, "mass_checks": 2,
            "mass_above_one_count": 0, "tolerance": 2e-6, "mass_min": .3, "mass_max": .5,
            "incoming_max_sum_error": 0., "feature_max_sum_error": 0., "result_max_sum_error": 0.,
            "mass_max_overshoot": 0.}


def actor(did, split="dev", offset=0):
    return {"split": split, "dialogue_id": did, "query_ids": [7], "user_turn_indices": [1, 3],
            "tokens": [[4], [5, 6]], "turn_text_ids": [0, 1], "query_text_ids": [0],
            "candidate_text_ids": [[0, 1, 0]], "candidate_ids": [["NONE", "DC", "value"]],
            "lexical_offset": offset, "lexical_shape": [2, 1, 3, 10]}


def endpoints(start=0):
    return [{"row_index": start + t, "source_row_index": t, "time": t, "turn_index": 2*t + 1,
             "query_position": 0, "query_index": 7, "query_id": "service.slot", "service": "service",
             "slot": "slot", "candidate_ids": ["NONE", "DC", "value"], "candidate_values": ["", "", "value"]}
            for t in range(2)]


def raw():
    block = np.full((2, 12), -np.inf, dtype=np.float32)
    block[:, :3] = np.log(np.array([.6, .3, .1], dtype=np.float64)).astype(np.float32)
    return block


def result(did="d0"):
    return SimpleNamespace(log_probs=Tensor(raw()[:, :3].reshape(1, 2, 1, 3)), split="dev",
                           analysis_role="qualification", dialogue_id=did, query_ids=(7,),
                           user_turn_indices=(1, 3), candidate_ids=(("NONE", "DC", "value"),),
                           encoder_work={k: work()[k] for k in replay.ENCODER_KEYS}, invariants=invariants())


def test_projection_charges_all_twelve_and_three_work_metrics():
    cases = [{"case_seconds": 2., "work": {"encoder_calls": 2, "padded_attention_positions": 8,
                                           "real_question_updates": 4}} for _ in range(24)]
    totals = {"encoder_calls": 10, "padded_attention_positions": 100, "real_question_updates": 20}
    answer = replay.projection(cases, [3.] * 12, totals, 7.)
    assert answer["measures"]["encoder_calls"]["projected_seconds"] == 120
    assert answer["measures"]["padded_attention_positions"]["projected_seconds"] == 300
    assert answer["measures"]["real_question_updates"]["projected_seconds"] == 120
    assert answer["total_seconds"] == 343 and answer["admitted"]


def test_projection_inclusive_fixed_boundary_and_no_case_selection():
    cases = [{"case_seconds": 1., "work": dict.fromkeys(replay.WORK_MEASURES, 1)} for _ in range(24)]
    totals = dict.fromkeys(replay.WORK_MEASURES, 1)
    assert replay.projection(cases, [1.] * 12, totals, 1776.)["total_seconds"] == 1800
    assert replay.projection(cases, [1.] * 12, totals, 1776.)["admitted"]
    assert not replay.projection(cases, [1.] * 12, totals, 1776.00001)["admitted"]
    cases[-1]["case_seconds"] = 200.
    assert not replay.projection(cases, [1.] * 12, totals, 1.)["admitted"]
    with pytest.raises(ValueError, match="All 24"):
        replay.projection(cases[:-1], [1.] * 12, totals, 1.)


def test_public_gather_exact_order_and_no_target_requirement():
    rows = endpoints(4)
    logs, ids = replay.gather_public(result(), rows[::-1])
    assert ids.tolist() == [5, 4]
    assert np.array_equal(logs, raw()[::-1])
    rows[0]["label_index"] = 1
    with pytest.raises(ValueError, match="public addresses only"):
        replay.gather_public(result(), rows)


@pytest.mark.parametrize("field,value", [("query_index", 8), ("turn_index", 2), ("time", True),
                                         ("candidate_ids", ["DC", "NONE", "value"])])
def test_public_gather_rejects_changed_identities(field, value):
    rows = endpoints()
    rows[0][field] = value
    with pytest.raises(ValueError):
        replay.gather_public(result(), rows)


def test_parity_raw_identity_underflow_and_first_tie():
    x = raw()
    x[:, :3] = np.array([-np.log(2), -np.log(2), -1000], dtype=np.float32)
    answer = replay.parity(x, x.copy(), endpoints())
    assert answer["maximum_supported_log_error"] == 0 and answer["canonical_first_argmax_equal"]
    y = x.copy()
    y[:, :2] += np.array([-1e-7, 1e-7], dtype=np.float32)
    with pytest.raises(ValueError, match="parity failed"):
        replay.parity(y, x, endpoints())


@pytest.mark.parametrize("defect", ["nonfinite", "padding", "mass", "drift"])
def test_parity_rejects_raw_defects(defect):
    x = raw()
    if defect == "nonfinite":
        x[0, 1] = -np.inf
    elif defect == "padding":
        x[0, 8] = -100
    elif defect == "mass":
        x[0, 0] += .1
    else:
        x[:, :3] = np.log([.600002, .299998, .1]).astype(np.float32)
    with pytest.raises(ValueError):
        replay.parity(x, raw(), endpoints())


def test_payload_separates_variants_and_keeps_complete_turns():
    a = actor("d0")
    arrays = {"original": np.zeros(60, np.float32), "numbers": np.ones(60, np.float32)}
    p = replay.payload(a, arrays, "trainable_numbers", "qualify")
    assert p["source_split"] == "dev" and p["analysis_role"] == "qualification"
    assert p["turn_text_ids"] == a["turn_text_ids"] and np.all(p["lexical"] == 1)
    assert "lexical" not in a and "analysis_role" not in a


def test_missing_public_update_or_encoder_work_is_rejected():
    replay.validate_work(result(), work())
    wrong = result()
    wrong.invariants["advance_returned"] = 1
    with pytest.raises(ValueError, match="Every full public"):
        replay.validate_work(wrong, work())
    wrong = result()
    wrong.encoder_work["padded_attention_positions"] -= 1
    with pytest.raises(ValueError, match="Every encoder"):
        replay.validate_work(wrong, work())


class Route:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.work = Counter()
        self.payloads = []

    def forward_public(self, payload):
        assert not ({"label_index", "targets", "temperature", "stratum"} & set(payload))
        self.payloads.append(payload)
        self.n += 1
        value = result(payload["dialogue_id"])
        value.split, value.analysis_role = payload["split"], payload["analysis_role"]
        self.work.update(value.encoder_work)
        return value

    def verify_parameters_unchanged(self):
        return {"encoder": "e" * 64, "memory": "f" * 64}

    def snapshot(self):
        counts = {p + "_" + suffix: (1 if p in ("checkpoint_load", "encoder_load") else self.n)
                  for p in ("checkpoint_load", "encoder_load", "public_forward", "encoding",
                            "memory_forward", "encoder_forward") for suffix in ("attempts", "returns")}
        return {"synthetic_injection": False, "encoder_device": "mps:0", "memory_device": "cpu",
                "dtype": "float32", "optimizer_created": False, "temperature_applied": False,
                "counts": counts, "encoder_work": dict(self.work), "restored_sha256": self.verify_parameters_unchanged()}


def collect(value, rows):
    assert value.split == "train" and all(r["label_index"] == 2 for r in rows)
    return {"log_probs": Tensor(raw()), "row_indices": Tensor(np.array([r["row_index"] for r in rows], np.int64))}


@pytest.fixture
def saved_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "SAMPLE_SIZE", 2)
    monkeypatch.setattr(replay, "DEV_ENDPOINTS", 4)
    prepared, original, oldrun = (tmp_path / name for name in ("prepared", "original", "oldrun"))
    for directory in (prepared, original, oldrun):
        directory.mkdir()
    actors = [actor("d0", offset=0), actor("d1", offset=60)]
    replay.common.write_lines(original / "actors-dev.jsonl", actors)
    train = [{**a, "split": "train", "source_split": "train", "analysis_role": "calibration"} for a in actors]
    replay.common.write_lines(prepared / "actors.jsonl", train)
    rows = [{**r, "dialogue_id": did, "split": "train", "source_split": "train", "analysis_role": "calibration",
             "label_index": 2, "label_id": "value"} for i, did in enumerate(("d0", "d1")) for r in endpoints(2*i)]
    replay.common.write_lines(prepared / "evaluation-rows.jsonl", rows)
    replay.common.write(prepared / "selection.json", {"selected_ids": ["d0", "d1"]})
    replay.common.write(prepared / "workloads.json", {
        "profiles": [{"split": "train", "dialogue_id": did, "work": work()} for did in ("d0", "d1")],
        "all": {"totals": {k: 2*v for k, v in work().items()}}})
    cases = [{"dialogue_id": did, "work": work(), "row_indices": [2*i, 2*i + 1], "endpoints": endpoints(2*i)}
             for i, did in enumerate(("d0", "d1"))]
    replay.common.write(prepared / "replay-cases.json", {"cases": cases})
    for directory in (prepared, original):
        for variant in ("original", "numbers"):
            np.save(directory / f"lexical-{variant}.npy", np.zeros(120, dtype=np.float32))
    for fit_id in replay.FIT_ORDER:
        directory = oldrun / fit_id
        directory.mkdir()
        arm, seed = fit_id.rsplit("-", 1)
        replay.common.write(directory / "completed.json", {
            "status": "completed", "fit_id": fit_id, "arm": arm, "seed": int(seed),
            "checkpoint": {"sha256": "a"*64}, "files": {"weights.pt": {"sha256": "a"*64}},
            "final_encoder_sha256": "e"*64})
        replay.save_packet(directory / "predictions.npz", np.concatenate([raw(), raw()]), np.arange(4, dtype=np.int64))
    prep_receipt = {"status": "completed", "counts": {"scored_endpoints": 4}, "wall_seconds": 5.}
    qualifications, events, routes = {}, [], []

    def auth_output(path, pin, ctx, phase):
        events.append("authenticate-" + phase)
        assert pin and ctx["plan_sha256"] == "p"*64
        return prep_receipt if phase == "prepare" else qualifications["receipt"]

    def auth_terminal(path, pin, receipt):
        events.append("terminal")
        assert pin and receipt["status"] == "completed"
        return {"wall_seconds": 7.}

    def loader(checkpoint, **kwargs):
        assert "encoder_factory" not in kwargs and "memory_factory" not in kwargs and "encoder_device" not in kwargs
        assert checkpoint.name == "weights.pt" and kwargs["expected_encoder_sha256"] == "e"*64
        route = Route(**kwargs)
        routes.append(route)
        return route

    api = SimpleNamespace(torch=SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda: None)), load=loader, collect=collect)

    def fake_backend():
        events.append("backend")
        assert events.index("terminal") < events.index("backend")
        return api

    monkeypatch.setattr(replay.common, "authenticate_output", auth_output)
    monkeypatch.setattr(replay.common, "authenticate_terminal", auth_terminal)
    monkeypatch.setattr(replay, "backend", fake_backend)
    budget = Budget()
    ctx = {"science": {"fit_order": replay.FIT_ORDER}, "original_prepared": original,
           "prior": SimpleNamespace(run=oldrun), "parent": {"snapshot": "fake", "model_files": {}, "tokenizer_ids": {}},
           "progress": budget.progress, "plan_sha256": "p"*64}

    def args(command):
        out = tmp_path / command
        out.mkdir()
        replay.common.write(out / "started.json", {"synthetic": True})
        return SimpleNamespace(command=command, prepared=prepared, prepared_sha256="b"*64,
                               prepared_terminal=tmp_path / "prep-terminal.json", prepared_terminal_sha256="c"*64,
                               qualification=tmp_path / "qualify", qualification_sha256="d"*64,
                               qualification_terminal=tmp_path / "qual-terminal.json", qualification_terminal_sha256="e"*64,
                               out=out)

    return SimpleNamespace(args=args, ctx=ctx, budget=budget, qualification=qualifications,
                           routes=routes, events=events, api=api, train=train, rows=rows, profiles={d: work() for d in ("d0", "d1")})


def test_complete_fake_twelve_fit_qualification_then_inference(saved_fixture):
    f = saved_fixture
    qualification_args = f.args("qualify")
    qualified = replay.body(qualification_args, f.ctx, f.budget)
    assert qualified["fit_order"] == replay.FIT_ORDER and qualified["completed_forwards"] == 24
    assert qualified["parity_passed"] and qualified["projection"]["admitted"]
    assert qualified["projection"]["preparation_seconds"] == 7.  # actual parent terminal, not receipt's 5
    assert all(item["witness"]["counts"]["public_forward_returns"] == 2 for item in qualified["fits"])
    assert len(replay.common.members(qualification_args.out)) == 50
    f.qualification["receipt"] = {**qualified, "status": "completed"}
    f.events.clear()
    infer_args = f.args("infer")
    answer = replay.body(infer_args, f.ctx, f.budget)
    assert answer["completed_forwards"] == 24 and answer["total_endpoint_rows"] == 48
    assert len(f.routes) == 24 and answer["task_metrics_computed"] is False
    assert len(replay.common.members(infer_args.out)) == 37
    for fit_id in replay.FIT_ORDER:
        with np.load(infer_args.out / fit_id / "predictions.npz", allow_pickle=False) as packet:
            assert set(packet.files) == {"row_indices", "log_probs"}
            assert np.array_equal(packet["row_indices"], np.arange(4))
            assert np.array_equal(packet["log_probs"], np.concatenate([raw(), raw()]))
        assert not (infer_args.out / fit_id / "partial-log-probs.npy").exists()


def test_terminal_failure_precedes_backend_or_payload_decode(saved_fixture, monkeypatch):
    f = saved_fixture

    def rejected(*_):
        raise ValueError("Missing successful parent exit")

    monkeypatch.setattr(replay.common, "authenticate_terminal", rejected)
    monkeypatch.setattr(replay.common, "read", lambda *_: pytest.fail("Task metadata read before terminal"))
    with pytest.raises(ValueError, match="parent exit"):
        replay.body(f.args("qualify"), f.ctx, f.budget)
    assert "backend" not in f.events


def test_denied_qualification_does_not_load_models(saved_fixture):
    f = saved_fixture
    f.qualification["receipt"] = {"status": "completed", "prepared_sha256": "b"*64,
        "prepared_terminal_sha256": "c"*64, "fit_order": replay.FIT_ORDER, "parity_passed": True,
        "projection": {"admitted": False}}
    with pytest.raises(ValueError, match="Complete admitted"):
        replay.body(f.args("infer"), f.ctx, f.budget)
    assert "backend" not in f.events and not f.routes


def test_next_load_failure_clears_previous_fit_witness(saved_fixture):
    f = saved_fixture
    f.ctx["progress"].update(route={"previous_fit": True}, case_index=1, endpoint_rows=99, dialogue_id="old")

    def fail(*_args, **_kwargs):
        raise RuntimeError("Synthetic load failure")

    f.api.load = fail
    with pytest.raises(RuntimeError, match="load failure"):
        replay.load_route(f.api, f.ctx, replay.FIT_ORDER[1], f.budget)
    assert f.ctx["progress"]["fit_id"] == replay.FIT_ORDER[1]
    assert f.ctx["progress"]["route"] is None and f.ctx["progress"]["case_index"] is None
    assert f.ctx["progress"]["endpoint_rows"] == 0 and f.ctx["progress"]["dialogue_id"] is None


def test_inference_partial_prefix_survives_failed_forward(saved_fixture, tmp_path):
    f = saved_fixture
    directory = tmp_path / "failed-fit"
    directory.mkdir()
    route = Route()
    original_forward = route.forward_public

    def fail_second(actor_payload):
        if route.n == 1:
            raise RuntimeError("Synthetic second dialogue failure")
        return original_forward(actor_payload)

    route.forward_public = fail_second
    grouped = {did: [r for r in f.rows if r["dialogue_id"] == did] for did in ("d0", "d1")}
    progress = {"completed_forwards": 0}
    with pytest.raises(RuntimeError, match="second dialogue"):
        replay.infer_fit(f.api, route, directory, f.train, grouped,
                         {"original": np.zeros(120, np.float32)}, "frozen_original", f.profiles, 4, f.budget, progress)
    assert progress["completed_forwards"] == 1 and progress["dialogue_id"] == "d1"
    assert np.load(directory / "partial-row-indices.npy").tolist() == [0, 1, -1, -1]
    assert np.array_equal(np.load(directory / "partial-log-probs.npy")[:2], raw())
    assert not (directory / "predictions.npz").exists() and not (directory / "completed.json").exists()


def test_cleanup_flush_error_preserves_original_forward_failure(saved_fixture, tmp_path, monkeypatch):
    f = saved_fixture
    directory = tmp_path / "flush-failure"
    directory.mkdir()
    original_open = np.lib.format.open_memmap

    class FlushFailure:
        def __init__(self, array):
            self.array = array
            self.calls = 0

        def __setitem__(self, key, value):
            self.array[key] = value

        def flush(self):
            self.calls += 1
            if self.calls == 2:
                raise OSError("Synthetic cleanup flush failure")
            self.array.flush()

    def opened(path, **kwargs):
        array = original_open(path, **kwargs)
        return FlushFailure(array) if path.name == "partial-log-probs.npy" else array

    monkeypatch.setattr(np.lib.format, "open_memmap", opened)
    route = Route()
    forward = route.forward_public

    def fail_second(actor_payload):
        if route.n == 1:
            raise RuntimeError("Primary forward failure")
        return forward(actor_payload)

    route.forward_public = fail_second
    grouped = {did: [r for r in f.rows if r["dialogue_id"] == did] for did in ("d0", "d1")}
    with pytest.raises(RuntimeError, match="Primary forward failure") as captured:
        replay.infer_fit(f.api, route, directory, f.train, grouped,
                         {"original": np.zeros(120, np.float32)}, "frozen_original", f.profiles, 4,
                         f.budget, {"completed_forwards": 0})
    assert any("Synthetic cleanup flush failure" in note for note in captured.value.__notes__)


def test_canonical_join_rejects_reordered_or_missing_rows(saved_fixture):
    f = saved_fixture
    rows = copy.deepcopy(f.rows)
    rows[1], rows[2] = rows[2], rows[1]
    with pytest.raises(ValueError, match="Canonical TRAIN"):
        replay.canonical_calibration(f.train, rows, ["d0", "d1"])
    with pytest.raises(ValueError, match="Complete nonempty"):
        replay.canonical_calibration(f.train, f.rows[:2], ["d0", "d1"])


def test_cli_requires_prior_terminals_and_does_not_add_infer_fields_to_qualify():
    base = ["--plan", "plan", "--plan-sha256", "p", "--supervision", "launch", "--out", "out",
            "--prepared", "prepared", "--prepared-sha256", "s"]
    with pytest.raises(SystemExit):
        replay.parse_args(["qualify", *base])
    prior = ["--prepared-terminal", "terminal", "--prepared-terminal-sha256", "t"]
    args = replay.parse_args(["qualify", *base, *prior])
    assert not hasattr(args, "qualification")
    with pytest.raises(SystemExit):
        replay.parse_args(["infer", *base, *prior])
