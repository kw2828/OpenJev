"""Artificial public streams and a fake inference route, never neural assets."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("runtime_pilot_tested", ROOT / "scripts/pilot_dialogue_runtime.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Budget:
    def __init__(self):
        self.progress = {}
        self.now = 1.
        self.slow_fit = None
        self.extra_nonblock_seconds = 0.

    def check(self):
        pass

    def storage(self):
        pass

    def sync(self, _torch):
        self.now += .001

    def elapsed(self):
        slow = self.progress.get("fit_id") == self.slow_fit and self.progress.get("block") == "verification"
        self.now += .1 if slow else .001
        value = self.now + self.extra_nonblock_seconds
        return value


def profile_work():
    return {"unique_texts": 1, "input_texts": 1, "content_tokens": 1, "chunks": 1,
            "encoder_sequences": 1, "special_token_positions": 2, "valid_token_positions": 3,
            "padded_token_positions": 3, "padding_token_positions": 0, "padded_attention_positions": 9,
            "encoder_calls": 1, "overlength_texts_chunked": 0, "truncated_tokens": 0,
            "max_chunk_tokens_with_special": 3, "public_user_turns": 1, "queries": 1,
            "schema_text_occurrences": 1, "candidate_text_occurrences": 3, "max_candidates": 3,
            "real_question_updates": 1, "real_candidate_updates": 3, "padded_candidate_positions": 3,
            "lexical_scalars": 30, "lexical_bytes": 120, "max_content_tokens_per_text": 1,
            "chunk_tokens": 254, "chunk_batch_size": 32}


def actor(did):
    return {"split": "dev", "dialogue_id": did, "query_ids": [7], "user_turn_indices": [1],
            "tokens": [[8]], "turn_text_ids": [0], "query_text_ids": [0], "candidate_text_ids": [[0, 0, 0]],
            "candidate_ids": [["NONE", "DC", "red"]], "lexical_offset": 0, "lexical_shape": [1, 1, 3, 10]}


def endpoint(i):
    return {"row_index": i, "source_row_index": 0, "time": 0, "turn_index": 1, "query_position": 0,
            "query_index": 7, "query_id": "service/slot", "service": "service", "slot": "slot",
            "candidate_ids": ["NONE", "DC", "red"], "candidate_values": [None, None, "red"]}


def invariant():
    return {**dict.fromkeys(("forward_calls", "forward_returned", "advance_calls", "advance_returned",
                            "valid_turns", "executed_valid_question_slots", "real_question_updates",
                            "incoming_checks", "feature_checks", "result_checks", "mass_checks"), 1),
            "mass_above_one_count": 0, "tolerance": 2e-6, "mass_min": .4, "mass_max": .4,
            "incoming_max_sum_error": 0., "feature_max_sum_error": 0., "result_max_sum_error": 0., "mass_max_overshoot": 0.}


def raw(n=1):
    logs = np.full((n, 12), -np.inf, np.float32)
    logs[:, :3] = np.log([.6, .3, .1]).astype(np.float32)
    return logs


class Route:
    def __init__(self, state, out, **kwargs):
        self.progress, self.out, self.kwargs = state, out, kwargs
        self.count = 0
        self.work = Counter()
        self.seen = []
        self.forecast_bytes = None

    def forward_public(self, payload):
        assert payload["split"] == payload["source_split"] == "dev"
        assert payload["analysis_role"] == "qualification"
        assert not ({"label_id", "label_index", "target", "stratum", "temperature"} & set(payload))
        path = self.out / self.progress["fit_id"] / "forecast.json"
        if self.progress["block"] == "verification":
            assert path.is_file(), "Forecast must be on disk before first verification model call"
            contents = path.read_bytes()
            self.forecast_bytes = contents if self.forecast_bytes is None else self.forecast_bytes
            assert contents == self.forecast_bytes
            assert json.loads(contents)["verification_forwards_started"] == 0
        else:
            assert not path.exists(), "No forecast exists before the complete estimator block"
        self.count += 1
        self.seen.append((self.progress["block"], payload["dialogue_id"]))
        work = {k: profile_work()[k] for k in pilot.replay.ENCODER_KEYS}
        self.work.update(work)
        return SimpleNamespace(log_probs=Tensor(raw()[:, :3].reshape(1, 1, 1, 3)), split="dev",
                               analysis_role="qualification", dialogue_id=payload["dialogue_id"], query_ids=(7,),
                               candidate_ids=(("NONE", "DC", "red"),), user_turn_indices=(1,),
                               encoder_work=work, invariants=invariant())

    def verify_parameters_unchanged(self):
        return {"memory": "m"*64, "encoder": "e"*64}

    def snapshot(self):
        counts = {prefix + "_" + suffix: (1 if prefix in ("encoder_load", "checkpoint_load") else self.count)
                  for prefix in ("encoder_load", "checkpoint_load", "public_forward", "encoding", "memory_forward", "encoder_forward")
                  for suffix in ("attempts", "returns")}
        return {"synthetic_injection": False, "encoder_device": "mps", "memory_device": "cpu", "dtype": "float32",
                "optimizer_created": False, "temperature_applied": False, "counts": counts,
                "encoder_work": dict(self.work), "restored_sha256": self.verify_parameters_unchanged()}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "DEV_ENDPOINTS", 2363)
    original, prepared, oldrun, out = (tmp_path / name for name in ("original", "prepared", "oldrun", "out"))
    for path in (original, prepared, oldrun, out):
        path.mkdir()
    profiles = [{"split": "dev", "dialogue_id": f"dev-{i:04}", "work": profile_work()} for i in range(2363)]
    actors = [actor(p["dialogue_id"]) for p in profiles]
    rows = [{**endpoint(i), "split": "dev", "dialogue_id": p["dialogue_id"],
             "label_id": "forbidden evaluator marker", "stratum": "do not use"} for i, p in enumerate(profiles)]
    legacy = [{"dialogue_id": profiles[i]["dialogue_id"], "work": profile_work(),
               "row_indices": [i], "endpoints": [endpoint(i)]} for i in range(2)]
    calibration = [{"split": "train", "dialogue_id": f"cal-{i}", "work": profile_work()} for i in range(512)]
    pilot.common.write(original / "workloads.json", {"profiles": profiles})
    pilot.common.write_lines(original / "actors-dev.jsonl", actors)
    pilot.common.write_lines(oldrun / "evaluation-rows.jsonl", rows)
    pilot.common.write(prepared / "replay-cases.json", {"cases": legacy})
    pilot.common.write(prepared / "workloads.json", {"profiles": calibration,
                        "all": {"totals": {k: v * 512 for k, v in profile_work().items()}}})
    for variant in ("original", "numbers"):
        np.save(original / f"lexical-{variant}.npy", np.zeros(30, np.float32))
    for fit_id in pilot.FIT_ORDER:
        target = oldrun / fit_id
        target.mkdir()
        arm, seed = fit_id.rsplit("-", 1)
        pilot.common.write(target / "completed.json", {"status": "completed", "fit_id": fit_id,
                           "arm": arm, "seed": int(seed), "checkpoint": {"sha256": "a"*64},
                           "files": {"weights.pt": {"sha256": "a"*64}}, "final_encoder_sha256": "e"*64})
        pilot.replay.save_packet(target / "predictions.npz", raw(2363), np.arange(2363, dtype=np.int64))
    budget = Budget()
    ctx = {"original_prepared": original, "prepared": prepared, "prior": SimpleNamespace(run=oldrun),
           "science": {"fit_order": pilot.FIT_ORDER}, "parent": {"snapshot": "fake", "model_files": {}, "tokenizer_ids": {}},
           "progress": budget.progress, "preparation_receipt": {"counts": {"dialogues": 512}},
           "preparation_terminal": {"wall_seconds": 7.},
           "old_qualification": {"projection": {"admitted": False}, "parity_passed": True}}
    routes = []

    def load(_path, **kwargs):
        assert "encoder_factory" not in kwargs and "memory_factory" not in kwargs and "encoder_device" not in kwargs
        value = Route(budget.progress, out, **kwargs)
        routes.append(value)
        return value

    api = SimpleNamespace(torch=SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda: None)), load=load)

    def backend():
        assert (out / "sample.json").exists()
        return api

    monkeypatch.setattr(pilot.replay, "backend", backend)
    pilot.common.write(out / "started.json", {"synthetic": True})
    return SimpleNamespace(args=SimpleNamespace(out=out), ctx=ctx, budget=budget, api=api,
                           routes=routes, profiles=profiles, calibration=calibration, legacy=legacy)


def test_public_selection_is_complete_disjoint_and_label_free(fixture):
    f = fixture
    sample, actors, _profiles, endpoints, calibration = pilot.public_inputs(f.ctx, f.budget)
    assert sample["counts"]["selected_dialogues"] == 128
    assert len(actors) == 130 and sample["warmup_ids"] == ["dev-0000", "dev-0001"]
    assert not set(sample["estimator"]["dialogue_ids"]) & set(sample["verification"]["dialogue_ids"])
    assert not set(sample["warmup_ids"]) & set(sample["estimator"]["dialogue_ids"] + sample["verification"]["dialogue_ids"])
    assert all(set(r) == set(pilot.PUBLIC_ENDPOINT_KEYS) for rows in endpoints.values() for r in rows)
    assert calibration == {"dialogue_count": 512, "encoder_calls": 512,
                           "padded_attention_positions": 4608, "real_question_updates": 512}
    assert sample["geometry"]["used_for_admission"] is False
    assert sample["geometry"]["joint_batch_coverage"]["status"] == "unknown"


def test_complete_twelve_fit_fake_pilot_and_separately_finalized_projection(fixture):
    f = fixture
    metadata = pilot.body(f.args, f.ctx, f.budget)
    assert metadata["completed_forwards"] == 1560 and metadata["timed_forwards"] == 1536
    assert metadata["warmup_forwards"] == 24 and metadata["calibration_forward_calls"] == 0
    assert metadata["optimizer_created"] is metadata["temperature_applied"] is False
    assert len(f.routes) == 12 and all(r.count == 130 for r in f.routes)
    assert not (f.args.out / "projection.json").exists()
    # Common lifecycle reauthentication can take nontrivial time after body.
    f.budget.extra_nonblock_seconds = 11.
    metadata = pilot.body.finalize(f.args, f.ctx, f.budget, metadata)
    assert metadata["pilot_nonblock_overhead_seconds"] >= 11
    assert metadata["projection"]["all_verifications_passed"]
    assert metadata["projection"]["components"]["preparation_parent_wall_seconds"] == 7
    assert len(pilot.common.members(f.args.out)) == 51
    sample = pilot.common.read(f.args.out / "sample.json")
    order = sample["warmup_ids"] + sample["estimator"]["dialogue_ids"] + sample["verification"]["dialogue_ids"]
    for route, fit in zip(f.routes, metadata["fits"], strict=True):
        assert [did for _block, did in route.seen] == order
        assert [name for name, _did in route.seen] == ["warmup"] * 2 + ["estimator"] * 64 + ["verification"] * 64
        directory = f.args.out / fit["fit_id"]
        with np.load(directory / "predictions.npz", allow_pickle=False) as packet:
            assert set(packet.files) == {"row_indices", "log_probs"}
            assert packet["row_indices"].tolist() == [int(did.split("-")[1]) for did in order]
            assert np.array_equal(packet["log_probs"], raw(130))
        forecast = pilot.common.read(directory / "forecast.json")
        assert fit["estimator"]["finished_elapsed_seconds"] <= forecast["created_elapsed_seconds"]
        assert forecast["created_elapsed_seconds"] <= fit["forecast_published_elapsed_seconds"] <= fit["verification"]["started_elapsed_seconds"]
        assert forecast["prediction"] == fit["verification_result"]["prediction"]
        assert not (directory / "partial-log-probs.npy").exists()


def test_slow_verification_is_reported_without_refitting_or_dropping_any_fit(fixture):
    f = fixture
    f.budget.slow_fit = pilot.FIT_ORDER[0]
    metadata = pilot.body(f.args, f.ctx, f.budget)
    result = pilot.finalize(f.args, f.ctx, f.budget, metadata)["projection"]
    assert metadata["completed_forwards"] == 1560 and len(metadata["fits"]) == 12
    assert result["verification"][pilot.FIT_ORDER[0]]["passed"] is False
    assert result["all_verifications_passed"] is False and result["admitted"] is False
    assert result["verifications_total"] == 12 and result["verifications_passed"] == 11
    for fit in metadata["fits"]:
        forecast = pilot.common.read(f.args.out / fit["fit_id"] / "forecast.json")
        assert forecast["prediction"] == fit["verification_result"]["prediction"]
        assert fit["verification_result"]["prediction"]["estimator_seconds"] == fit["estimator"]["seconds"]


def test_parity_failure_preserves_raw_prefix_and_stops_no_retry(fixture, monkeypatch):
    f = fixture
    forward = Route.forward_public

    def changed(self, payload):
        result = forward(self, payload)
        if self.count == 3:
            result.log_probs = Tensor(np.log(np.array([.1, .3, .6])).astype(np.float32).reshape(1, 1, 1, 3))
        return result

    monkeypatch.setattr(Route, "forward_public", changed)
    with pytest.raises(ValueError, match="parity failed"):
        pilot.body(f.args, f.ctx, f.budget)
    assert len(f.routes) == 1 and f.routes[0].count == 3
    directory = f.args.out / pilot.FIT_ORDER[0]
    saved = np.load(directory / "partial-log-probs.npy", allow_pickle=False)
    assert np.all(np.isfinite(saved[:3, :3])) and np.all(np.isneginf(saved[3:]))
    assert not (directory / "completed.json").exists() and not (directory / "forecast.json").exists()
    assert f.budget.progress["route"]["counts"]["public_forward_returns"] == 3


def test_forecast_is_immutable_after_verification_begins(fixture, monkeypatch):
    f = fixture
    forward = Route.forward_public

    def mutate(self, payload):
        result = forward(self, payload)
        if self.progress["block"] == "verification" and self.progress["block_position"] == 63:
            path = self.out / self.progress["fit_id"] / "forecast.json"
            value = json.loads(path.read_text())
            value["prediction"]["predicted_seconds"] *= 10
            path.write_text(json.dumps(value))
        return result

    monkeypatch.setattr(Route, "forward_public", mutate)
    with pytest.raises(ValueError, match="Unchanged forecast"):
        pilot.body(f.args, f.ctx, f.budget)
    assert len(f.routes) == 1 and f.routes[0].count == 130
    assert not (f.args.out / pilot.FIT_ORDER[0] / "completed.json").exists()


def test_failure_witness_error_never_masks_original_forward_failure(fixture, monkeypatch):
    f = fixture
    snapshot = Route.snapshot

    def failing_forward(self, _payload):
        self.count = 1
        raise RuntimeError("Primary synthetic forward failure")

    def failing_snapshot(self):
        if self.count:
            raise OSError("Secondary synthetic snapshot failure")
        return snapshot(self)

    monkeypatch.setattr(Route, "forward_public", failing_forward)
    monkeypatch.setattr(Route, "snapshot", failing_snapshot)
    with pytest.raises(RuntimeError, match="Primary synthetic forward failure") as captured:
        pilot.body(f.args, f.ctx, f.budget)
    assert any("Secondary synthetic snapshot failure" in note for note in captured.value.__notes__)
    assert len(f.routes) == 1 and not (f.args.out / pilot.FIT_ORDER[0] / "completed.json").exists()


def test_diagnostic_geometry_never_adds_an_admission_gate(fixture):
    f = fixture
    changed = copy.deepcopy(f.calibration)
    changed[0]["work"]["max_content_tokens_per_text"] = 99999
    path = f.ctx["prepared"] / "workloads.json"
    table = pilot.common.read(path)
    table["profiles"] = changed
    path.write_text(json.dumps(table))
    sample, *_ = pilot.public_inputs(f.ctx, f.budget)
    assert sample["geometry"]["uncovered_calibration"]["timed_union"]["outside_any_field_count"] == 1
    assert sample["geometry"]["used_for_admission"] is False


def test_failed_v1_admission_must_remain_preserved(fixture, monkeypatch):
    f = fixture
    f.ctx["old_qualification"]["projection"]["admitted"] = True
    monkeypatch.setattr(pilot, "public_inputs", lambda *_: pytest.fail("Wrong lineage decoded public inputs"))
    with pytest.raises(ValueError, match="failed V1 admission"):
        pilot.body(f.args, f.ctx, f.budget)


def test_cli_has_no_model_selection_or_new_threshold_flags():
    args = pilot.parse_args(["--plan", "plan", "--plan-sha256", "a"*64, "--supervision", "launch", "--out", "out"])
    assert args.command == "pilot" and set(vars(args)) == {"command", "plan", "plan_sha256", "supervision", "out"}
