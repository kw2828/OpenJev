"""Synthetic protocol and optimizer checks, with no real data or engine calls."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import chess
import pytest
import torch
from torch.nn import functional as F

from openjev.research.chess_candidate import ARMS, CandidateChess
from openjev.research.chess_candidate_eval import CachedPositions

ROOT = Path(__file__).resolve().parents[1]
PLAN_HASH = "a" * 64
DATA_HASH = "b" * 64


def load_study():
    spec = importlib.util.spec_from_file_location(
        "candidate_study_test", ROOT / "scripts/chess_candidate_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study():
    return load_study()


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
    torch.use_deterministic_algorithms(deterministic)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def synthetic_rows():
    fens = [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 12 20",
        "7k/8/8/8/3Pp3/8/8/7K b - d3 0 12",
        "7k/P7/8/8/8/8/8/7K w - - 7 19",
        "7k/6R1/8/5K2/8/8/8/8 b - - 0 1",
    ]
    return [
        {
            "id": f"synthetic-{i}",
            "game_id": i,
            "fen": fen,
            "target_uci": max(m.uci() for m in chess.Board(fen).legal_moves),
            "target_value": (-1) ** i * (0.1 + i * 0.1),
        }
        for i, fen in enumerate(fens)
    ]


def tiny_protocol(study, monkeypatch, *, count=5, batch=3, microbatch=2, epochs=1):
    protocol = copy.deepcopy(study.PROTOCOL)
    protocol.update(
        seeds=[97],
        train_examples=count,
        epochs=epochs,
        batch_size=batch,
        microbatch_size=microbatch,
        updates_per_fit=epochs * ((count + batch - 1) // batch),
        training_device="cpu",
        candidate_chunk_size=8,
    )
    monkeypatch.setattr(study, "PROTOCOL", protocol)
    return protocol


def test_full_schedule_visits_every_position_and_preserves_shared_initialization(study):
    rng_before = torch.random.get_rng_state().clone()
    starts = []
    for seed in (97, 109, 127):
        schedule = study.schedule(seed)
        assert len(schedule) == 1536
        assert [step["step"] for step in schedule] == list(range(1, 1537))
        assert sum(len(step["indices"]) for step in schedule) == 32768 * 6
        for epoch in range(1, 7):
            subset = [step for step in schedule if step["epoch"] == epoch]
            assert len(subset) == 256 and all(len(step["indices"]) == 128 for step in subset)
            assert sorted(i for step in subset for i in step["indices"]) == list(range(32768))
        starts.append(schedule[0]["indices"])
        reference = CandidateChess("direct", seed=seed).state_dict()
        for arm in ARMS:
            model = CandidateChess(arm, seed=seed)
            assert set(reference) == set(model.state_dict())
            assert all(torch.equal(value, reference[key]) for key, value in model.state_dict().items())
            assert study.schedule(seed) == schedule
    assert starts[0] != starts[1] != starts[2]
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    configs = study.configs()
    assert configs == study.configs() and len(configs) == 12
    assert {(c["arm"], c["seed"]) for c in configs} == {(a, s) for a in ARMS for s in (97, 109, 127)}
    assert all(c["name"] == f"{c['arm']}-{c['seed']}" for c in configs)


def test_active_parameter_and_compute_matching_limits_are_explicit(study):
    counts = {arm: CandidateChess(arm, seed=97).parameter_counts() for arm in ARMS}
    assert len({v["stored"] for v in counts.values()}) == 1
    assert counts["direct"]["active"] == 33185
    assert len({counts[a]["active"] for a in ARMS if a != "direct"}) == 1
    assert counts["action_only"]["active"] > counts["direct"]["active"]
    assert "NOT matched" in study.PROTOCOL["matching"]


def test_schedule_rejects_an_inconsistent_declared_update_budget(study, monkeypatch):
    monkeypatch.setitem(study.PROTOCOL, "updates_per_fit", 1535)
    with pytest.raises(ValueError, match="budget"):
        study.schedule(97)


@pytest.fixture
def signature_root(study, tmp_path, monkeypatch):
    root = tmp_path / "source"
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "SOURCES", ["sentinel.py"])
    for name in (
        "sentinel.py",
        study.ENGINE,
        "uv.lock",
        "pyproject.toml",
        "old-input.json",
        study.data.TRAIN,
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic source marker, never used as model data\n")
    for name in study.PREFLIGHTS:
        write(
            root / name,
            {
                "status": "completed",
                "arms": {arm: {"weights_unchanged": True} for arm in ARMS},
                "sources": {"sentinel.py": study.sha(root / "sentinel.py")},
            },
        )

    def exclusions(_):
        return {
            "states": ["synthetic-excluded-state"],
            "files": {"old-input.json": study.sha(root / "old-input.json")},
            "counts": {"synthetic": 1},
        }

    monkeypatch.setattr(study.data, "collect_exclusions", exclusions)

    def forbidden(*_, **__):
        raise AssertionError("Prepare must not generate data, evaluate or train")

    monkeypatch.setattr(study.data, "generate", forbidden)
    monkeypatch.setattr(study, "fit", forbidden)
    monkeypatch.setattr(study, "evaluate", forbidden)
    return root


def test_prepare_binds_complete_schedule_sources_parameters_and_preflights(study, signature_root, tmp_path):
    plan_path = study.prepare(tmp_path / "plan")
    plan = study.verify_plan(plan_path)
    assert len(plan["configurations"]) == 12 and len(plan["arena_schedule"]) == 288
    assert len(plan["openings"]) == 16
    assert set(plan["parameters"]) == set(ARMS)
    assert plan["parameters"]["direct"]["active"] == 33185
    assert plan["parameters"]["delta"] == plan["parameters"]["full_afterstate"]
    assert set(plan["preflights"]) == set(study.PREFLIGHTS)
    assert plan["training_source_sha256"] == study.sha(signature_root / study.data.TRAIN)
    prepared = study.read(plan_path.parent / "prepared.json")
    assert prepared["training_or_scoring_started"] is False
    assert prepared["plan_sha256"] == study.sha(plan_path)
    with pytest.raises(FileExistsError):
        study.prepare(plan_path.parent)


@pytest.mark.parametrize("change", ["source", "engine", "exclusion", "lock", "plan", "training", "preflight"])
def test_frozen_sources_inputs_and_budgets_cannot_change(study, signature_root, tmp_path, change):
    plan_path = study.prepare(tmp_path / "plan")
    if change == "plan":
        plan = study.read(plan_path)
        plan["protocol"]["microbatch_size"] = 1
        write(plan_path, plan)
    elif change == "preflight":
        path = signature_root / study.PREFLIGHTS[0]
        preflight = study.read(path)
        preflight["arms"]["delta"]["weights_unchanged"] = False
        write(path, preflight)
    else:
        name = {
            "source": "sentinel.py",
            "engine": study.ENGINE,
            "exclusion": "old-input.json",
            "lock": "uv.lock",
            "training": study.data.TRAIN,
        }[change]
        with (signature_root / name).open("a") as stream:
            stream.write("changed\n")
    with pytest.raises(ValueError):
        study.verify_plan(plan_path)


def test_verified_exclusion_snapshot_can_be_reused_without_collecting_again(
    study, signature_root, tmp_path, monkeypatch
):
    plan_path = study.prepare(tmp_path / "plan")
    exclusions = study.data.collect_exclusions(signature_root)

    def forbidden(*_):
        raise AssertionError("Unexpected second exclusion collection")

    monkeypatch.setattr(study.data, "collect_exclusions", forbidden)
    assert study.verify_plan(plan_path, exclusions=exclusions) == study.read(plan_path)
    exclusions["states"].append("new-state")
    with pytest.raises(ValueError):
        study.verify_plan(plan_path, exclusions=exclusions)


@pytest.fixture
def gate_inputs(study, monkeypatch):
    monkeypatch.setitem(study.PROTOCOL, "bootstrap_replicates", 20)
    metrics, predictions, regret = {}, {}, {}
    eval_rows = {
        split: [{"id": f"{split}-{i}", "game_id": f"game-{i // 10}"} for i in range(100)]
        for split in ("dev", "shift")
    }
    for config in study.configs():
        name = config["name"]
        correct = 40 if config["arm"] == "delta" else 30
        metrics[name] = {
            split: {
                "agreement": correct / 100,
                "target_nll": 2.0,
                "value_mae": 0.4,
                "mean_confidence": 0.3,
            }
            for split in eval_rows
        }
        predictions[name] = {split: [{"correct": i < correct} for i in range(100)] for split in eval_rows}
        regret[name] = {split: 0.3 if config["arm"] == "delta" else 0.5 for split in eval_rows}
    return metrics, predictions, regret, eval_rows, {"gate": {"passed": True}, "gate_passed": True}


def test_gate_requires_all_four_control_panel_comparisons_and_arena(study, gate_inputs):
    result = study.comparisons(*gate_inputs)
    assert result["continuation_passed"] is True
    assert len(result["checks"]) == 5
    assert sum(check["passed"] for check in result["checks"]) == 5
    assert len(result["comparisons"]) == 6
    assert {(row["comparator"], row["split"]) for row in result["comparisons"]} == {
        (a, s) for a in ("direct", "action_only", "full_afterstate") for s in ("dev", "shift")
    }
    for row in result["comparisons"]:
        assert row["relative_reduction"] == pytest.approx(0.4)
        assert row["agreement_interval"]["mean"] == pytest.approx(0.1)
        assert row["agreement_interval"]["games"] == 10
    gate_inputs[-1]["gate_passed"] = False
    gate_inputs[-1]["gate"]["passed"] = False
    assert not study.comparisons(*gate_inputs)["continuation_passed"]


@pytest.mark.parametrize(
    ("reference", "delta", "passed"),
    [
        (0.5, 0.4, True),
        (0.5, 0.45, False),
        (0.5, -0.1, True),
        (0.5, 0.40000005, False),
        (0.0, -0.1, False),
        (-0.1, -0.2, False),
    ],
)
def test_signed_regret_gate_requires_positive_reference_and_handles_boundary(
    study, gate_inputs, reference, delta, passed
):
    for seed in study.PROTOCOL["seeds"]:
        for split in ("dev", "shift"):
            for arm in ("direct", "action_only"):
                gate_inputs[2][f"{arm}-{seed}"][split] = reference
            gate_inputs[2][f"delta-{seed}"][split] = delta
    result = study.comparisons(*gate_inputs)
    assert result["continuation_passed"] is passed
    for row in result["comparisons"]:
        if row["primary"]:
            assert row["reference_bounded_regret"] == pytest.approx(reference)
            assert row["delta_bounded_regret"] == pytest.approx(delta)
            expected = (reference - delta) / reference if reference > 0 else None
            assert row["relative_reduction"] == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize("reference", ["direct", "action_only"])
@pytest.mark.parametrize("split", ["dev", "shift"])
def test_one_primary_control_panel_failure_blocks_gate(study, gate_inputs, reference, split):
    for seed in study.PROTOCOL["seeds"]:
        gate_inputs[2][f"{reference}-{seed}"][split] = 0.31
    result = study.comparisons(*gate_inputs)
    assert not result["continuation_passed"]
    assert sum(not check["passed"] for check in result["checks"]) == 1


def test_full_afterstate_quality_is_descriptive_and_cannot_replace_a_primary_control(study, gate_inputs):
    for seed in study.PROTOCOL["seeds"]:
        for split in ("dev", "shift"):
            gate_inputs[2][f"full_afterstate-{seed}"][split] = -1.0
    assert study.comparisons(*gate_inputs)["continuation_passed"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_regret_cannot_enter_a_gate(study, gate_inputs, value):
    gate_inputs[2]["delta-97"]["dev"] = value
    with pytest.raises(ValueError, match="Nonfinite regret"):
        study.comparisons(*gate_inputs)


def test_all_prediction_vectors_shortened_together_still_fail_panel_coverage(study, gate_inputs):
    for prediction in gate_inputs[1].values():
        prediction["dev"].pop()
    with pytest.raises(ValueError):
        study.comparisons(*gate_inputs)


@pytest.mark.parametrize("change", ["fit", "split", "prediction", "grade", "extra", "short_rows"])
def test_gate_rejects_incomplete_or_extra_fit_membership(study, gate_inputs, change):
    metrics, predictions, regret, _, _ = gate_inputs
    if change == "fit":
        del metrics["delta-127"]
    elif change == "split":
        del metrics["delta-127"]["shift"]
    elif change == "prediction":
        del predictions["delta-127"]
    elif change == "grade":
        del regret["delta-127"]
    elif change == "extra":
        metrics["unplanned"] = copy.deepcopy(metrics["delta-127"])
    else:
        predictions["delta-127"]["dev"].pop()
    with pytest.raises(ValueError):
        study.comparisons(*gate_inputs)


@pytest.mark.parametrize("arm", ARMS)
def test_microbatch_gradients_match_one_full_position_weighted_loss(study, monkeypatch, tmp_path, arm):
    tiny_protocol(study, monkeypatch, count=5, batch=5, microbatch=2)
    cache = CachedPositions(synthetic_rows())
    assert len(set(cache.counts.tolist())) > 1
    config = next(c for c in study.configs() if c["arm"] == arm)
    indices = study.schedule(97)[0]["indices"]
    inputs, targets, values = cache.batch(indices, arm, device="cpu")
    reference = CandidateChess(arm, seed=97)
    logits, prediction, _ = reference(**inputs, depth=4, candidate_chunk_size=8)
    ce = F.cross_entropy(logits, targets)
    mse = F.mse_loss(prediction, values)
    expected_loss = ce + 0.5 * mse
    expected_loss.backward()
    gradients = {name: None if p.grad is None else p.grad.clone() for name, p in reference.named_parameters()}
    captures = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        captures.append([None if p.grad is None else p.grad.clone() for p in parameters])
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    out = tmp_path / arm
    study.fit(config, cache, out, PLAN_HASH, DATA_HASH)
    assert len(captures) == 1
    assert len(captures[0]) == len(gradients)
    for (name, expected), actual in zip(gradients.items(), captures[0], strict=True):
        if expected is None:
            assert actual is None, name
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-6, msg=name)
    journal = study.rows(out / "learning.jsonl")
    assert len(journal) == 1
    assert journal[0]["policy_ce"] == pytest.approx(float(ce.detach()), abs=2e-6)
    assert journal[0]["value_mse"] == pytest.approx(float(mse.detach()), abs=2e-6)
    assert journal[0]["loss"] == pytest.approx(float(expected_loss.detach()), abs=2e-6)


@pytest.mark.parametrize("arm", ARMS)
def test_tiny_cpu_fit_preserves_journals_unused_heads_and_exact_candidate_budgets(
    study, monkeypatch, tmp_path, arm
):
    protocol = tiny_protocol(study, monkeypatch)
    cache = CachedPositions(synthetic_rows())
    config = next(c for c in study.configs() if c["arm"] == arm)
    initial = study.make_model(config)
    out = tmp_path / arm
    receipt = study.fit(config, cache, out, PLAN_HASH, DATA_HASH)
    trained = study.load_model(out / "weights.pt", config, PLAN_HASH)
    assert receipt == study.read(out / "training.json")
    assert receipt["status"] == "completed" and receipt["updates"] == 2
    assert receipt["examples_seen"] == 5 and receipt["training_seconds"] >= 0
    assert receipt["plan_sha256"] == PLAN_HASH and receipt["data_receipt_sha256"] == DATA_HASH
    assert receipt["cache_sha256"] == cache.metadata["cache_sha256"]
    assert receipt["initial_state_sha256"] == study.digest_state(initial)
    assert receipt["weights_sha256"] == study.sha(out / "weights.pt")
    assert receipt["learning_sha256"] == study.sha(out / "learning.jsonl")
    assert any(
        not torch.equal(value, initial.state_dict()[key])
        for key, value in trained.state_dict().items()
        if key.startswith("encoder.")
    )
    assert all(
        torch.equal(value, initial.state_dict()[key])
        for key, value in trained.state_dict().items()
        if key.startswith("aux_head.")
    )
    if arm == "direct":
        assert all(
            torch.equal(value, initial.state_dict()[key])
            for key, value in trained.state_dict().items()
            if key.startswith("action_projection.")
        )
    candidates = sum(len(menu) for menu in cache.menus)
    refinement = 0 if arm == "direct" else candidates * 2
    afterstate = candidates * 4 if arm == "full_afterstate" else 0
    expected = {
        "candidate_evaluations": candidates,
        "root_core_iterations": 5 * 4,
        "candidate_refinement_iterations": refinement,
        "successor_root_iterations": afterstate,
        "total_core_iterations": 20 + refinement + afterstate,
        "delta_convolutions": candidates if arm == "delta" else 0,
        "successor_encoders": candidates if arm == "full_afterstate" else 0,
    }
    assert receipt["computation"] == expected
    journal = study.rows(out / "learning.jsonl")
    assert [r["microbatches"] for r in journal] == [2, 1]
    assert [r["examples"] for r in journal] == [3, 2]
    for key, total in expected.items():
        assert sum(row[key] for row in journal) == total
    for row, step in zip(journal, study.schedule(97), strict=True):
        assert row["step"] == step["step"] and row["epoch"] == step["epoch"]
        assert row["root_depth"] == 4 and row["branch_depth"] == 2
        assert row["loss"] == pytest.approx(
            row["policy_ce"] + protocol["value_loss_weight"] * row["value_mse"], abs=2e-6
        )
        assert row["sampled_mps_allocated_bytes_after_backward"] == 0
    with pytest.raises(FileExistsError):
        study.fit(config, cache, out, PLAN_HASH, DATA_HASH)


@pytest.fixture
def saved_tiny_fit(study, monkeypatch, tmp_path):
    tiny_protocol(study, monkeypatch, batch=5)
    cache = CachedPositions(synthetic_rows())
    config = next(c for c in study.configs() if c["arm"] == "delta")
    out = tmp_path / "fit"
    study.fit(config, cache, out, PLAN_HASH, DATA_HASH)
    audited = study.audit_training(out, config, PLAN_HASH, DATA_HASH, cache.metadata, cache.rows)
    assert audited["updates"] == 1 and audited["examples_seen"] == 5
    return out, config, cache


@pytest.mark.parametrize(
    "fault",
    [
        "epoch",
        "indices_sha256",
        "microbatches",
        "candidate_evaluations",
        "root_core_iterations",
        "total_core_iterations",
        "delta_convolutions",
        "loss",
        "gradient_norm",
        "allocation",
        "allocation_type",
        "allocation_bool",
        "initial_state",
        "cache",
        "total",
        "negative_time",
    ],
)
def test_training_audit_rejects_rehashed_semantic_corruption(study, saved_tiny_fit, fault):
    out, config, cache = saved_tiny_fit
    receipt = study.read(out / "training.json")
    journal = study.rows(out / "learning.jsonl")
    if fault == "initial_state":
        receipt["initial_state_sha256"] = "c" * 64
    elif fault == "cache":
        receipt["cache_sha256"] = "c" * 64
    elif fault == "total":
        receipt["computation"]["candidate_evaluations"] += 1
    elif fault == "negative_time":
        receipt["training_seconds"] = -1
    elif fault == "indices_sha256":
        journal[0]["indices_sha256"] = "c" * 64
    elif fault == "gradient_norm":
        journal[0]["gradient_norm"] = -1
    elif fault == "allocation":
        journal[0]["sampled_mps_allocated_bytes_after_backward"] = -1
    elif fault == "allocation_type":
        journal[0]["sampled_mps_allocated_bytes_after_backward"] = 1.0
    elif fault == "allocation_bool":
        journal[0]["sampled_mps_allocated_bytes_after_backward"] = True
    else:
        journal[0][fault] += 1
    (out / "learning.jsonl").write_text("".join(json.dumps(row) + "\n" for row in journal))
    receipt["learning_sha256"] = study.sha(out / "learning.jsonl")
    write(out / "training.json", receipt)
    with pytest.raises(ValueError):
        study.audit_training(out, config, PLAN_HASH, DATA_HASH, cache.metadata, cache.rows)
