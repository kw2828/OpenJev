"""Outer-driver wiring with fake fitting/evaluation and tiny constant tensors.

No model, optimizer, real corpus/split, scientific RNG, native environment,
subprocess or network operation executes. Actual filesystem hashes and JSON /
safe tensor serialization exercise the receipt boundaries.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

DRIVER = Path(__file__).with_name("run_pilot.py")


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location("innovation_pilot_driver_fake_tests", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_runtime_metadata_is_safe_for_checkpoint_hashing(driver):
    from openjev.research.reacher_objective_training import canonical_state_hash

    actual = driver.runtime()
    assert type(actual["torch"]) is str and actual["torch"] == str(torch.__version__)
    assert type(actual["numpy"]) is str and actual["numpy"] == str(np.__version__)
    digest = canonical_state_hash({"binding": {"runtime": actual}})
    assert digest == canonical_state_hash({"binding": {"runtime": json.loads(json.dumps(actual))}})


@pytest.fixture
def harness(driver, tmp_path, monkeypatch):
    module = driver
    protocol, attempt = tmp_path / "protocol", tmp_path / "attempt"
    protocol.mkdir()
    monkeypatch.setattr(module, "PROTOCOL", protocol)
    monkeypatch.setattr(module, "ATTEMPT", attempt)
    events, fit_receipts = [], {}
    options = {"published_wrong": False, "remote_wrong": False, "fail_fit": None,
               "fit_error": None, "receipt_defect": None, "fit_file_tamper": False,
               "evaluation_tamper": None, "evaluation_error": None}
    runtime = {"fixture": "no numerical backend", "torch_threads": 1}
    sources = {"synthetic/source.py": "a" * 64}
    configs = {variant: dataclasses.asdict(module.PilotConfig(variant=variant, hidden_size=4,
        context_size=4, dt=0.02, noise_std=0.05, eta=0.1, variance_min=1e-4,
        variance_max=4.0, epochs=2, batch_size=1, variance_score_weight=0.1)) for variant in module.VARIANTS}
    packets = torch.zeros(1, 6, 8, dtype=torch.float32)
    packets[..., 6] = 1
    public = {"packets": packets, "commands": torch.zeros(1, 5, 2), "rewards": torch.zeros(1, 5)}
    prepared = {name + ".pt": {key: value.clone() for key, value in public.items()}
                for name in ("train", "dev6", "dev10")}
    pairs = []
    for pair in range(3):
        weights = {"fixture_weight": torch.tensor([float(pair + 1)])}
        orders = torch.tensor([[0], [0]], dtype=torch.int64)
        prepared[f"initial-pair{pair}.pt"], prepared[f"orders-pair{pair}.pt"] = weights, orders
        pairs.append({"pair": pair, "initial_tensor_sha256": module.canonical_tensor_hash(weights),
                      "orders_tensor_sha256": module.canonical_tensor_hash({"orders": orders})})
    for name, payload in prepared.items():
        module.save(protocol / name, payload)
    plan = {"runtime": runtime, "source_sha256": sources,
            "prepared_members": {name: {"sha256": module.sha(protocol / name), "bytes": (protocol / name).stat().st_size}
                                 for name in prepared},
            "cap_seconds": 100, "pairs": pairs, "variants": list(module.VARIANTS),
            "configurations": configs, "fit_count": 12, "updates_per_fit": 2,
            "evaluations": ["initial", "final"], "panels": ["dev6", "dev10"],
            "data_tensor_sha256": {name: module.canonical_tensor_hash(public) for name in ("train", "dev6", "dev10")}}
    module.write_json(protocol / "plan.json", plan)
    digest, published = module.sha(protocol / "plan.json"), "1" * 40
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.value, perf_counter=lambda: clock.value))
    monkeypatch.setattr(module, "runtime", lambda: runtime)

    def forbidden(*args, **kwargs):
        raise AssertionError("No model, RNG, real preparation or external call allowed")
    monkeypatch.setattr(module, "InnovationContextWorldModel", forbidden)
    monkeypatch.setattr(module, "prepare_data", forbidden)
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch, "randperm", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)

    def subprocess_output(command, **kwargs):
        events.append(("subprocess", tuple(command)))
        if command == ["git", "show", f"{published}:{protocol / 'plan.json'}"]:
            return b"wrong published bytes" if options["published_wrong"] else (protocol / "plan.json").read_bytes()
        assert command == ["gh", "api", f"repos/kw2828/OpenJev/commits/{published}"]
        return json.dumps({"sha": "2" * 40 if options["remote_wrong"] else published}).encode()
    monkeypatch.setattr(module.subprocess, "check_output", subprocess_output)

    def sources_check(actual):
        assert actual == sources
        events.append(("source_check",))
    monkeypatch.setattr(module, "check_sources", sources_check)

    def load(name, actual_plan):
        assert actual_plan == plan
        assert module.sha(protocol / name) == plan["prepared_members"][name]["sha256"]
        events.append(("load", name))
        payload = prepared[name]
        return {key: value.clone() for key, value in payload.items()} if isinstance(payload, dict) else payload.clone()
    monkeypatch.setattr(module, "load", load)

    def fit(weights, orders, data, out, **kwargs):
        name, variant = out.name, kwargs["config"].variant
        events.append(("fit_started", name))
        assert kwargs["source_sha256"] == sources and kwargs["runtime"] == runtime
        assert kwargs["expected_initial_sha256"] == module.canonical_tensor_hash(weights)
        assert kwargs["expected_orders_sha256"] == module.canonical_tensor_hash({"orders": orders})
        assert kwargs["expected_data_sha256"] == module.canonical_tensor_hash(data)
        assert kwargs["deadline"] == 100
        out.mkdir(parents=True, exist_ok=False)
        if name == options["fail_fit"]:
            module.write_json(out / "partial.json", {"synthetic": True, "completed_optimizer_steps": 1})
            raise options["fit_error"] or RuntimeError("deliberate fake fit failure")
        final = {key: value + (1 + list(module.VARIANTS).index(variant)) for key, value in weights.items()}
        module.save(out / "weights.pt", final)
        receipt = {"status": "completed", "config": dataclasses.asdict(kwargs["config"]),
            "counts": {"attempted_updates": 2, "optimizer_steps": 2, "flushed_updates": 2},
            "adam": {"betas": (0.9, 0.999), "eps": 1e-8},
            "files": {"weights.pt": {"sha256": module.sha(out / "weights.pt"), "bytes": (out / "weights.pt").stat().st_size}},
            "final_weights_sha256": module.canonical_tensor_hash(final)}
        if options["receipt_defect"] == "status":
            receipt["status"] = "failed"
        elif options["receipt_defect"] == "counts":
            receipt["counts"]["flushed_updates"] = 1
        elif options["receipt_defect"] == "config":
            receipt["config"]["eta"] = 0.2
        elif options["receipt_defect"] == "semantic_weights":
            receipt["final_weights_sha256"] = "f" * 64
        module.write_json(out / "completed.json", receipt)
        if options["receipt_defect"] == "stored_receipt":
            stored = json.loads((out / "completed.json").read_text())
            stored["unbound_extra"] = True
            (out / "completed.json").write_text(json.dumps(stored))
        if options["fit_file_tamper"]:
            (out / "weights.pt").write_bytes(b"modified after helper receipt")
        fit_receipts[name] = receipt
        events.append(("fit_returned", name))
        return receipt
    monkeypatch.setattr(module, "fit_one", fit)

    def evaluate(weights, data, out, **kwargs):
        # These assertions apply to EVERY evaluation, not only the first.
        assert len([event for event in events if event[0] == "fit_returned"]) == 12
        boundary = json.loads((attempt / "all-fits-completed.json").read_text())
        assert len(boundary["fits"]) == 12
        assert kwargs["expected_weights_sha256"] == module.canonical_tensor_hash(weights), "Semantic weight binding"
        assert kwargs["expected_data_sha256"] == module.canonical_tensor_hash(data)
        assert kwargs["source_sha256"] == sources and kwargs["runtime"] == runtime
        assert kwargs["deadline"] == 100
        relative = out.relative_to(attempt / "evaluation").parts
        events.append(("evaluate", *relative, kwargs["expected_weights_sha256"]))
        if options["evaluation_error"] is not None:
            raise options["evaluation_error"]
        out.mkdir(parents=True, exist_ok=False)
        summary = {"status": "completed", "synthetic_fake_evaluation": True,
                   "weight_sha256": kwargs["expected_weights_sha256"], "new_optimizer_steps": 0}
        module.write_json(out / "completed.json", summary)
        if options["evaluation_tamper"] and len([event for event in events if event[0] == "evaluate"]) == 1:
            original = attempt / "fits" / "pair0-constant"
            name = "completed.json" if options["evaluation_tamper"] == "receipt" else "weights.pt"
            (original / name).write_bytes(b"changed after all-fits boundary")
        return summary
    monkeypatch.setattr(module, "evaluate_development", evaluate)
    return SimpleNamespace(module=module, plan=plan, digest=digest, published=published,
        protocol=protocol, attempt=attempt, events=events, options=options, clock=clock,
        fit_receipts=fit_receipts, run=lambda: module.run(digest, published))


def test_twelve_complete_fits_precede_all_forty_eight_evaluations_and_tuple_receipts_roundtrip(harness):
    h = harness
    h.run()
    fits = [event[1] for event in h.events if event[0] == "fit_returned"]
    expected_fits = [f"pair{pair}-{variant}" for pair in range(3) for variant in h.module.VARIANTS]
    assert fits == expected_fits
    evaluation = [event for event in h.events if event[0] == "evaluate"]
    assert len(evaluation) == 48
    expected_rows = [(name, stage, panel) for name in expected_fits for stage in ("initial", "final") for panel in ("dev6", "dev10")]
    assert [event[1:4] for event in evaluation] == expected_rows
    assert max(i for i, event in enumerate(h.events) if event[0] == "fit_returned") < min(
        i for i, event in enumerate(h.events) if event[0] == "evaluate")
    for name, receipt in h.fit_receipts.items():
        assert receipt["adam"]["betas"] == (0.9, 0.999)
        stored = json.loads((h.attempt / "fits" / name / "completed.json").read_text())
        assert stored["adam"]["betas"] == [0.9, 0.999]
    completed = json.loads((h.attempt / "completed.json").read_text())
    assert completed["fits"] == 12 and completed["evaluations"] == 48
    assert completed["development_only"] and not completed["native_control_measured"]
    assert completed["gate_status"] == "requires_saved_output_review"
    assert len(json.loads((h.attempt / "evaluation-index.json").read_text())) == 48
    assert not (h.attempt / "failed.json").exists()


@pytest.mark.parametrize("error", [RuntimeError("fake fit failure"), KeyboardInterrupt("fake interruption")])
def test_failed_fit_stops_every_evaluation_and_preserves_partial_prefix(harness, error):
    h = harness
    h.options.update(fail_fit="pair1-age", fit_error=error)
    with pytest.raises(type(error)) as caught:
        h.run()
    assert caught.value is error
    assert len([event for event in h.events if event[0] == "fit_returned"]) == 5
    assert not any(event[0] == "evaluate" for event in h.events)
    assert (h.attempt / "fits/pair1-age/partial.json").exists()
    assert not (h.attempt / "all-fits-completed.json").exists()
    assert not (h.attempt / "completed.json").exists()
    failed = json.loads((h.attempt / "failed.json").read_text())
    assert failed["phase"] == "fit" and failed["current"] == "pair1-age" and failed["no_retry"]


@pytest.mark.parametrize("defect", ["status", "counts", "config", "stored_receipt"])
def test_invalid_or_differently_serialized_fit_receipt_stops_before_evaluation(harness, defect):
    h = harness
    h.options["receipt_defect"] = defect
    with pytest.raises(ValueError, match="completed fit|required|receipt differ"):
        h.run()
    assert not any(event[0] == "evaluate" for event in h.events)
    assert (h.attempt / "failed.json").exists()


def test_final_fit_file_hash_is_checked_immediately_on_return(harness):
    h = harness
    h.options["fit_file_tamper"] = True
    with pytest.raises(ValueError, match="Fit artifact changed"):
        h.run()
    assert not any(event[0] == "evaluate" for event in h.events)


@pytest.mark.parametrize("tamper,reason", [("receipt", "Completed fit receipt changed"), ("weights", "Final weight file changed")])
def test_completed_receipt_and_weight_file_rebound_before_final_evaluation(harness, tamper, reason):
    h = harness
    h.options["evaluation_tamper"] = tamper
    with pytest.raises(ValueError, match=reason):
        h.run()
    assert len([event for event in h.events if event[0] == "fit_returned"]) == 12
    assert len([event for event in h.events if event[0] == "evaluate"]) == 2
    assert not (h.attempt / "completed.json").exists()


def test_semantic_final_weight_digest_is_forwarded_to_evaluation_guard(harness):
    h = harness
    h.options["receipt_defect"] = "semantic_weights"
    with pytest.raises(AssertionError, match="Semantic weight binding"):
        h.run()
    assert len([event for event in h.events if event[0] == "evaluate"]) == 2
    assert not (h.attempt / "completed.json").exists()


@pytest.mark.parametrize("defect", ["published_wrong", "remote_wrong", "external_sha", "prepared_member"])
def test_admission_refusal_precedes_attempt_creation_and_all_fits(harness, defect):
    h = harness
    if defect in h.options:
        h.options[defect] = True
    elif defect == "prepared_member":
        (h.protocol / "train.pt").write_bytes(b"tampered input")
    with pytest.raises(ValueError):
        h.module.run("0" * 64 if defect == "external_sha" else h.digest, h.published)
    assert not h.attempt.exists()
    assert not any(event[0] in {"load", "fit_started", "evaluate"} for event in h.events)
    if defect == "external_sha":
        assert not h.events


@pytest.mark.parametrize("failure_write_fails", [False, True])
def test_late_completion_serialization_expires_cap_and_preserves_primary_exception(harness, monkeypatch, failure_write_fails):
    h = harness
    original_write = h.module.write_json
    def write(path, value):
        if path == h.attempt / "failed.json" and failure_write_fails:
            raise OSError("deliberate failure-receipt write error")
        original_write(path, value)
        if path == h.attempt / "completed.json":
            h.clock.value = 100.0
    monkeypatch.setattr(h.module, "write_json", write)
    with pytest.raises(ValueError, match="cap exceeded during terminal serialization") as caught:
        h.run()
    assert len([event for event in h.events if event[0] == "evaluate"]) == 48
    assert not (h.attempt / "completed.json").exists()
    invalid = json.loads((h.attempt / "invalid-completion.json").read_text())
    assert invalid["fits"] == 12 and invalid["evaluations"] == 48
    if failure_write_fails:
        assert not (h.attempt / "failed.json").exists()
        assert any("Failure preservation failed" in note and "write error" in note for note in caught.value.__notes__)
    else:
        failed = json.loads((h.attempt / "failed.json").read_text())
        assert "terminal serialization" in failed["error"] and failed["no_retry"]


def test_original_fit_error_survives_failure_receipt_write_error(harness, monkeypatch):
    h = harness
    primary = RuntimeError("primary fit exception")
    h.options.update(fail_fit="pair0-constant", fit_error=primary)
    original_write = h.module.write_json
    def write(path, value):
        if path == h.attempt / "failed.json":
            raise OSError("secondary serialization exception")
        original_write(path, value)
    monkeypatch.setattr(h.module, "write_json", write)
    with pytest.raises(RuntimeError) as caught:
        h.run()
    assert caught.value is primary
    assert any("secondary serialization exception" in note for note in primary.__notes__)
    assert not any(event[0] == "evaluate" for event in h.events)
    assert (h.attempt / "fits/pair0-constant/partial.json").exists()


def test_existing_attempt_is_never_reused_or_overwritten(harness):
    h = harness
    h.attempt.mkdir()
    marker = h.attempt / "retained-failure.bin"
    marker.write_bytes(b"original failure")
    with pytest.raises(FileExistsError):
        h.run()
    assert marker.read_bytes() == b"original failure"
    assert not any(event[0] in {"load", "fit_started", "evaluate"} for event in h.events)
