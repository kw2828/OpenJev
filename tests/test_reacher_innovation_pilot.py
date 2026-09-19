"""Tiny synthetic410 pilot tests; no native environment or real corpus."""

import json
import time
from dataclasses import replace

import pytest
import torch

from openjev.research import reacher_innovation_pilot as pilot
from openjev.research.reacher_innovation_context import VARIANTS, InnovationContextWorldModel
from openjev.research.reacher_objective_training import canonical_tensor_hash


@pytest.fixture(autouse=True)
def engineering():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(410)
            yield
    finally:
        torch.set_num_threads(threads)


def config(**changes):
    return replace(pilot.PilotConfig("normalized", 4, 4, .02, .05, .1, 1e-4, 4., 1, 2, .1), **changes)


def data(episodes=3, steps=6, missing=(2, 3)):
    angles = torch.linspace(-.4, .8, episodes * (steps + 1) * 2).reshape(episodes, steps + 1, 2)
    packets = torch.zeros(episodes, steps + 1, 8)
    packets[..., :2], packets[..., 2:4] = angles.cos(), angles.sin()
    packets[..., 4:6] = torch.tensor([.1, -.09])
    last = 0
    for step in range(steps + 1):
        if step in missing:
            packets[:, step, :4] = 0
            packets[:, step, 7] = (step - last) * .02
        else:
            packets[:, step, 6] = 1
            last = step
    commands = torch.linspace(-.7, .8, episodes * steps * 2).reshape(episodes, steps, 2)
    return {"packets": packets, "commands": commands, "rewards": -.2 - commands.square().sum(-1)}


def weights(cfg=None):
    cfg = config() if cfg is None else cfg
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        return {name: value.clone() for name, value in InnovationContextWorldModel(**cfg.model_kwargs()).state_dict().items()}


def args(values, initial, cfg=None):
    return {"config": config() if cfg is None else cfg, "source_sha256": {"synthetic": "0" * 64},
            "runtime": {"scope": "synthetic engineering410", "threads": 1},
            "expected_data_sha256": canonical_tensor_hash(values),
            "expected_initial_sha256": canonical_tensor_hash(initial), "deadline": time.monotonic() + 60}


def fit(out, values=None, cfg=None):
    values, cfg = data() if values is None else values, config() if cfg is None else cfg
    initial = weights(cfg)
    orders = torch.arange(len(values["packets"]) - 1, -1, -1).expand(cfg.epochs, -1).clone()
    kwargs = args(values, initial, cfg)
    receipt = pilot.fit_one(initial, orders, values, out,
        expected_orders_sha256=canonical_tensor_hash({"orders": orders}), **kwargs)
    return receipt, initial, orders


def evaluate(out, values=None, cfg=None, initial=None):
    values = data(episodes=3, steps=18, missing=(2, 3, 9, 10)) if values is None else values
    cfg, initial = config() if cfg is None else cfg, weights() if initial is None else initial
    kwargs = args(values, initial, cfg)
    kwargs["expected_weights_sha256"] = kwargs.pop("expected_initial_sha256")
    return pilot.evaluate_development(initial, values, out, **kwargs)


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("variant", VARIANTS)
def test_fit_all_variants_exact_orders_tail_batch_and_checkpoint(tmp_path, variant):
    cfg = config(variant=variant, epochs=2)
    values, before = data(), torch.get_rng_state().clone()
    receipt, initial, orders = fit(tmp_path / "fit", values, cfg)
    assert torch.equal(before, torch.get_rng_state())
    assert receipt["counts"] == {"attempted_updates": 4, "optimizer_steps": 4, "flushed_updates": 4}
    folder = tmp_path / "fit"
    assert set(receipt["files"]) == {"started.json", "initial-weights.pt", "epoch-orders.pt", "training.jsonl", "weights.pt", "checkpoint.pt"}
    assert {p.name for p in folder.iterdir()} == set(receipt["files"]) | {"completed.json"}
    for name, row in receipt["files"].items():
        assert pilot._file_sha(folder / name) == row["sha256"]
        assert (folder / name).stat().st_size == row["bytes"]
    logs = [json.loads(line) for line in (folder / "training.jsonl").read_text().splitlines()]
    assert [row["indices"] for row in logs] == [[2, 1], [0], [2, 1], [0]]
    assert [row["update"] for row in logs] == [1, 2, 3, 4]
    assert all(set(row["gradient_norm_before_clip"]) == {"backbone", "variance_head"} for row in logs)
    assert all(row["fit_elapsed_seconds_before_log"] >= row["update_seconds_before_log"] > 0 for row in logs)
    assert logs[0]["work"]["expected_action_cost_samples"] == 2 * (6 + 2 * 5)
    saved = torch.load(folder / "checkpoint.pt", weights_only=True)
    assert saved["counts"] == receipt["counts"] and saved["resume_authorized"] is False
    assert saved["binding"]["config"]["variant"] == variant
    assert canonical_tensor_hash(saved["weights"]) == receipt["final_weights_sha256"]
    assert all(state["step"] == 4 for state in saved["optimizer"]["state"].values())
    assert canonical_tensor_hash(initial) == receipt["initial_sha256"]
    assert torch.equal(orders, torch.load(folder / "epoch-orders.pt", weights_only=True))


def test_auxiliary_does_not_rescale_backbone_when_both_clip_groups_active(tmp_path):
    values = data(episodes=1)
    values["rewards"].fill_(-1e4)  # Forces the main gradient clip to be active.
    results, logs = [], []
    for label, weight in (("off", 0.), ("on", 1e5)):
        fit(tmp_path / label, values, config(variance_score_weight=weight))
        results.append(torch.load(tmp_path / label / "weights.pt", weights_only=True))
        logs.append(json.loads((tmp_path / label / "training.jsonl").read_text()))
    assert all(row["gradient_clipped"]["backbone"] for row in logs)
    assert logs[1]["gradient_clipped"]["variance_head"]
    initial = weights()
    for name in results[0]:
        if name.startswith("variance_head."):
            assert torch.equal(results[0][name], initial[name])
            assert not torch.equal(results[1][name], initial[name])
        else:
            torch.testing.assert_close(results[0][name], results[1][name], rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["initial_hash", "orders_hash", "data_hash", "duplicate_order", "wrong_shape", "runtime"])
def test_admission_rejection_preserves_failure_before_model_construction(tmp_path, monkeypatch, bad):
    initial, values = weights(), data()
    orders = torch.tensor([[2, 1, 0]])
    kwargs = args(values, initial)
    if bad == "initial_hash": kwargs["expected_initial_sha256"] = "f" * 64
    elif bad == "data_hash": kwargs["expected_data_sha256"] = "f" * 64
    elif bad == "duplicate_order": orders[0, 0] = 1
    elif bad == "wrong_shape": orders = orders[:, :2]
    elif bad == "runtime": kwargs["runtime"] = {}
    order_hash = "f" * 64 if bad == "orders_hash" else canonical_tensor_hash({"orders": orders})
    # Weight hash is checked inside construction wrapper but before constructing a model.
    if bad != "initial_hash":
        monkeypatch.setattr(pilot, "_model", lambda *a: pytest.fail("No model construction after invalid input"))
    with pytest.raises(ValueError):
        pilot.fit_one(initial, orders, values, tmp_path / "fit", expected_orders_sha256=order_hash, **kwargs)
    failed = read(tmp_path / "fit/failed.json")
    assert failed["counts"]["optimizer_steps"] == 0 and failed["automatic_retry"] is False
    assert not (tmp_path / "fit/completed.json").exists()


def test_failed_second_update_preserves_first_step_and_optimizer_without_resume(tmp_path, monkeypatch):
    original = pilot.innovation_sequence_loss
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate update failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(pilot, "innovation_sequence_loss", fail_second)
    with pytest.raises(RuntimeError, match="deliberate"):
        fit(tmp_path / "fit")
    failed = read(tmp_path / "fit/failed.json")
    assert failed["counts"] == {"attempted_updates": 2, "optimizer_steps": 1, "flushed_updates": 1}
    checkpoint = torch.load(tmp_path / "fit/partial-checkpoint.pt", weights_only=True)
    assert checkpoint["counts"] == failed["counts"]
    assert all(value["step"] == 1 for value in checkpoint["optimizer"]["state"].values())
    with pytest.raises(FileExistsError):
        fit(tmp_path / "fit")


def test_cap_after_completion_demotes_only_new_receipt(tmp_path, monkeypatch):
    original = pilot._check
    def check(deadline):
        original(deadline)
        if (tmp_path / "fit/completed.json").exists():
            raise TimeoutError("after completion")
    monkeypatch.setattr(pilot, "_check", check)
    with pytest.raises(TimeoutError, match="after completion"):
        fit(tmp_path / "fit", data(episodes=1))
    assert (tmp_path / "fit/invalid-completion.json").exists()
    assert (tmp_path / "fit/failed.json").exists()
    assert not (tmp_path / "fit/completed.json").exists()


def test_development_keeps_every_episode_and_post_reacquisition_h1_h3(tmp_path):
    values = data(episodes=3, steps=18, missing=(2, 3, 9, 10))
    summary = evaluate(tmp_path / "dev", values)
    predicted = torch.load(tmp_path / "dev/predictions.pt", weights_only=True)
    assert summary["split"] == "development" and len(summary["per_episode"]) == 3
    assert summary["counts"] == {"optimizer_steps": 0, "completed_batches": 2}
    assert predicted["one_step_mean"].shape == (3, 18, 4)
    assert predicted["open_loop_mean"].shape == (3, 14, 5, 4)
    expected_roots = torch.zeros(3, 14, dtype=torch.bool)
    expected_roots[:, [4, 11]] = True
    assert torch.equal(predicted["post_reacquisition_root_mask"], expected_roots)
    for case, row in enumerate(summary["per_episode"]):
        assert row["episode"] == case and row["post_reacquisition_complete_two_roots"]
        direct = []
        for root in (4, 11):
            direct.append([(predicted["open_loop_mean"][case, root, h - 1] - values["packets"][case, root + h, :4]).square().mean()
                           for h in (1, 3)])
        assert row["post_reacquisition_mean_mse"] == pytest.approx(float(torch.tensor(direct).mean()))
        assert row["post_reacquisition_h1_mse"] == pytest.approx(float(torch.tensor(direct)[:, 0].mean()))
        assert row["post_reacquisition_h3_mse"] == pytest.approx(float(torch.tensor(direct)[:, 1].mean()))
        assert row["counts"]["post_reacquisition_complete_roots"] == 2
    receipt = read(tmp_path / "dev/completed.json")
    assert receipt["new_optimizer_steps"] == 0
    assert set(receipt["files"]) == {"started.json", "predictions.pt", "summary.json"}


def test_development_missing_recovery_coverage_is_null_not_partial_winner(tmp_path):
    summary = evaluate(tmp_path / "dev", data(episodes=1, steps=10, missing=(2, 3, 8, 9)))
    row = summary["per_episode"][0]
    assert row["counts"]["reacquisition_targets"] == 2
    assert row["counts"]["post_reacquisition_complete_roots"] == 1
    assert row["post_reacquisition_mean_mse"] is None and not row["post_reacquisition_complete_two_roots"]


def test_development_private_forecasts_do_not_use_future_observations(tmp_path):
    left = data(episodes=1, steps=10, missing=(2, 3))
    right = {name: value.clone() for name, value in left.items()}
    right["packets"][:, 5:, :4] *= -1
    evaluate(tmp_path / "left", left)
    evaluate(tmp_path / "right", right)
    a = torch.load(tmp_path / "left/predictions.pt", weights_only=True)
    b = torch.load(tmp_path / "right/predictions.pt", weights_only=True)
    torch.testing.assert_close(a["open_loop_mean"][:, :5], b["open_loop_mean"][:, :5], rtol=0, atol=0)
    torch.testing.assert_close(a["one_step_mean"][:, :5], b["one_step_mean"][:, :5], rtol=0, atol=0)
    assert not torch.equal(a["one_step_mean"][:, 5:], b["one_step_mean"][:, 5:])


def test_development_missing_targets_no_loss_or_bound_fraction_claim(tmp_path):
    summary = evaluate(tmp_path / "dev", data(episodes=1, steps=6, missing=tuple(range(1, 7))))
    row = summary["per_episode"][0]
    assert row["one_step_angle_mse"] is row["residual_moment_score"] is row["variance_lower_bound_fraction"] is None
    assert row["counts"]["visible_targets"] == 0 and row["one_step_reward_mse"] >= 0


def test_evaluation_never_constructs_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("Evaluation must not construct Adam"))
    evaluate(tmp_path / "dev", data(episodes=1))
