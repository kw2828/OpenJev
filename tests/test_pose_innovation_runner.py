"""Synthetic runner boundaries only; no corpus, empirical fits, or native calls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import run_pose_innovation as runner
import torch
from train_pose_adaptation import forecast as frozen_cv1_forecast

from openjev.research.pose_transport import PoseTransport
from openjev.research.rigid_motion import so3_exp


@pytest.fixture(autouse=True)
def isolated_engineering_runtime():
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield
    torch.use_deterministic_algorithms(old_determinism)
    torch.set_num_threads(old_threads)


def original_ids():
    return np.column_stack((np.repeat(np.arange(30), 24),
                            np.tile(np.arange(24) * 50, 30))).astype(np.int64)


def test_exact_parent_split_all_folds_without_cross_window_leakage():
    ids = original_ids()
    all_test, all_train = set(), set()
    for fold in range(3):
        indices, train, test = runner.split_indices(ids, fold)
        assert indices.shape == (240,) and train.dtype == test.dtype == np.bool_
        assert train.sum() == 144 and test.sum() == 96
        np.testing.assert_array_equal(train, ~test)
        np.testing.assert_array_equal(indices, np.flatnonzero(ids[:, 0] % 3 == fold))
        train_parents, test_parents = set(ids[indices[train], 0]), set(ids[indices[test], 0])
        assert test_parents == {fold + 3 * rank for rank in (1, 4, 7, 9)}
        assert train_parents == {fold + 3 * rank for rank in (0, 2, 3, 5, 6, 8)}
        backbone_parents = set(range(30)) - train_parents - test_parents
        assert len(backbone_parents) == 20
        assert train_parents.isdisjoint(test_parents | backbone_parents)
        assert test_parents.isdisjoint(backbone_parents)
        for parent in train_parents | test_parents:
            which = ids[indices, 0] == parent
            assert which.sum() == 24 and len(set(train[which].tolist())) == 1
        all_train.update(train_parents)
        all_test.update(test_parents)
    assert len(all_train) == 18 and len(all_test) == 12
    assert all_train.isdisjoint(all_test) and all_train | all_test == set(range(30))


@pytest.mark.parametrize("kind", ["dtype", "start", "parent", "order", "short", "fold"])
def test_split_rejects_unbound_identity(kind):
    ids, fold = original_ids(), 0
    if kind == "dtype":
        ids = ids.astype(np.float32)
    elif kind == "start":
        ids[12, 1] += 1
    elif kind == "parent":
        ids[12, 0] = 1
    elif kind == "order":
        ids[[0, 1]] = ids[[1, 0]]
    elif kind == "short":
        ids = ids[:-1]
    else:
        fold = 3
    with pytest.raises(ValueError):
        runner.split_indices(ids, fold)


def ridge_data():
    generator = torch.Generator().manual_seed(410)
    return (torch.randn(12, 9, generator=generator),
            torch.randn(12, 25, 6, generator=generator),
            np.array([True] * 8 + [False] * 4))


@pytest.mark.parametrize("poison", [float("nan"), 1e20])
def test_held_out_labels_cannot_change_ridge_fit_or_any_prediction(poison):
    features, targets, train = ridge_data()
    first, first_fit = runner.ridge_predict(features, targets, train)
    changed = targets.clone()
    changed[~train] = poison
    second, second_fit = runner.ridge_predict(features, changed, train)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for key in first_fit:
        np.testing.assert_array_equal(first_fit[key], second_fit[key])
    changed[train] += 2
    trained_change, _ = runner.ridge_predict(features, changed, train)
    assert not torch.equal(first, trained_change)


def test_held_out_features_cannot_change_train_centering_or_weights():
    features, targets, train = ridge_data()
    first, first_fit = runner.ridge_predict(features, targets, train)
    changed = features.clone()
    changed[~train] = torch.arange(9).float() + 100
    second, second_fit = runner.ridge_predict(changed, targets, train)
    for key in first_fit:
        np.testing.assert_array_equal(first_fit[key], second_fit[key])
    torch.testing.assert_close(first[train], second[train], rtol=0, atol=0)
    assert not torch.equal(first[~train], second[~train])


def test_ridge_has_unpenalized_signed_intercept():
    features, targets, train = ridge_data()
    targets[train] = -2.5
    predicted, fit = runner.ridge_predict(features, targets, train)
    torch.testing.assert_close(predicted, torch.full_like(predicted, -2.5), rtol=0, atol=0)
    np.testing.assert_array_equal(fit["weights"], np.zeros_like(fit["weights"]))


def test_shuffle_retains_complete_tuples_endpoints_and_summary_information():
    tokens = torch.linspace(-1, 1, 7 * 31 * 55).reshape(7, 31, 55)
    permutations = runner.interior_permutations(7, 410)
    np.testing.assert_array_equal(permutations, runner.interior_permutations(7, 410))
    assert len({tuple(row) for row in permutations}) == 7
    for row in permutations:
        np.testing.assert_array_equal(np.sort(row), np.arange(31))
        assert row[0] == 0 and row[-1] == 30
    before = tokens.clone()
    shuffled = runner.variant_tokens(tokens, "shuffled", permutations)
    for i, row in enumerate(permutations):
        torch.testing.assert_close(shuffled[i], tokens[i, row], rtol=0, atol=0)
    torch.testing.assert_close(shuffled[:, [0, -1]], tokens[:, [0, -1]], rtol=0, atol=0)
    torch.testing.assert_close(runner.summary_features(shuffled), runner.summary_features(tokens),
                               rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(tokens, before, rtol=0, atol=0)


def test_noerror_changes_only_error_channels_without_mutating_cache():
    tokens = torch.linspace(-1, 1, 2 * 31 * 55).reshape(2, 31, 55)
    before = tokens.clone()
    result = runner.variant_tokens(tokens, "noerror", runner.interior_permutations(2, 410))
    torch.testing.assert_close(result[..., :49], tokens[..., :49], rtol=0, atol=0)
    assert torch.count_nonzero(result[..., 49:]) == 0
    assert result.data_ptr() != tokens.data_ptr()
    torch.testing.assert_close(tokens, before, rtol=0, atol=0)


def test_error_only_shuffle_preserves_public_chronology_and_residual_endpoints():
    tokens = torch.linspace(-1, 1, 3 * 31 * 55).reshape(3, 31, 55)
    before = tokens.clone()
    permutations = runner.interior_permutations(3, 410)
    result = runner.variant_tokens(tokens, "error_shuffled", permutations)
    torch.testing.assert_close(result[..., :49], tokens[..., :49], rtol=0, atol=0)
    for i, row in enumerate(permutations):
        torch.testing.assert_close(result[i, :, 49:], tokens[i, row, 49:], rtol=0, atol=0)
    torch.testing.assert_close(result[:, [0, -1]], tokens[:, [0, -1]], rtol=0, atol=0)
    torch.testing.assert_close(runner.summary_features(result), runner.summary_features(tokens),
                               rtol=2e-6, atol=2e-7)
    assert not torch.equal(result[:, 1:-1, 49:], tokens[:, 1:-1, 49:])
    torch.testing.assert_close(tokens, before, rtol=0, atol=0)


def synthetic_backbone_inputs():
    scales = torch.tensor([.05, .03, .01, .01])
    model = PoseTransport("body", scales, hidden=6).eval().requires_grad_(False)
    with torch.no_grad():
        model.readout.weight.copy_(torch.linspace(-.12, .15, 36).reshape(6, 6))
        model.readout.bias.copy_(torch.linspace(-.02, .02, 6))
    t = torch.arange(32).float()[None, :, None]
    p = t.square() * torch.tensor([.0002, -.0001, .00005])
    angles = t * torch.tensor([.004, -.002, .003])
    r = so3_exp(angles)
    actions = torch.linspace(-.1, .2, 56 * 40).reshape(1, 56, 40)
    return model, p, r, actions[:, :31], actions[:, 31:]


def test_cache_future_forecast_is_exact_frozen_cv1_loop_on_real_synthetic_gru():
    model, p, r, past, future = synthetic_backbone_inputs()
    one_p, one_r, got_p, got_r = runner.causal_cache(model, p, r, past, future)
    with torch.no_grad():
        expected_p, expected_r = frozen_cv1_forecast(model, p, r, past, future, "gru")
        generic_p, _ = model.forecast(p, r, torch.cat((past, future), 1))
    assert one_p.shape == (1, 31, 3) and one_r.shape == (1, 31, 3, 3)
    torch.testing.assert_close(got_p, expected_p, rtol=0, atol=0)
    torch.testing.assert_close(got_r, expected_r, rtol=0, atol=0)
    assert not torch.equal(got_p, generic_p), "fixture must distinguish CV1 from generic CV16"
    assert all(not value.requires_grad for value in (one_p, one_r, got_p, got_r))


def test_successor_observation_cannot_change_its_own_prior_prediction():
    model, p, r, past, future = synthetic_backbone_inputs()
    first = runner.causal_cache(model, p, r, past, future)
    changed_p, changed_r = p.clone(), r.clone()
    changed_p[:, 13] += torch.tensor([.4, -.2, .1])
    changed_r[:, 13] = so3_exp(torch.tensor([[.2, .1, -.1]])) @ changed_r[:, 13]
    second = runner.causal_cache(model, changed_p, changed_r, past, future)
    for a, b in zip(first[:2], second[:2], strict=True):
        torch.testing.assert_close(a[:, :13], b[:, :13], rtol=0, atol=0)
    assert not torch.equal(first[0][:, 13], second[0][:, 13])
    assert not torch.equal(first[1][:, 13], second[1][:, 13])


def test_later_future_actions_cannot_change_context_errors_or_earlier_predictions():
    model, p, r, past, future = synthetic_backbone_inputs()
    first = runner.causal_cache(model, p, r, past, future)
    changed = future.clone()
    changed[:, 8:] = 4
    second = runner.causal_cache(model, p, r, past, changed)
    for a, b in zip(first[:2], second[:2], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(first[2:], second[2:], strict=True):
        torch.testing.assert_close(a[:, :8], b[:, :8], rtol=0, atol=0)
    assert not torch.equal(first[2][:, 8:], second[2][:, 8:])


@pytest.mark.parametrize("field", ["p", "r", "past", "future"])
def test_cache_rejects_unexpected_sequence_boundary(field):
    model, p, r, past, future = synthetic_backbone_inputs()
    values = {"p": p, "r": r, "past": past, "future": future}
    values[field] = values[field][:, :-1]
    with pytest.raises(ValueError):
        runner.causal_cache(model, **values)


def synthetic_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("source.py")
    source.write_text("value = 1\n")
    bound = Path("input.bin")
    bound.write_bytes(b"synthetic-bound-input")
    out = Path("experiment")
    (out / "source-snapshot").mkdir(parents=True)
    (out / "source-snapshot" / source).write_bytes(source.read_bytes())
    protocol = {"bindings": {str(bound): runner.sha(bound)}, "sources": {str(source): runner.sha(source)},
                "runtime": {"python": runner.platform.python_version(), "torch": str(torch.__version__),
                            "numpy": str(np.__version__), "platform": runner.platform.platform(), "threads": 1}}
    runner.write_json(out / "protocol.json", protocol)
    return out, runner.sha(out / "protocol.json")


def forbidden_call(*args, **kwargs):
    raise AssertionError("data/model work must not occur")


def test_wrong_external_digest_rejected_before_output_or_data(tmp_path, monkeypatch):
    out, _ = synthetic_protocol(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "load_data", forbidden_call)
    monkeypatch.setattr(runner, "make_model", forbidden_call)
    with pytest.raises(ValueError, match="protocol pin"):
        runner.run(out, "0" * 64)
    assert not (out / "run-01").exists()


@pytest.mark.parametrize("changed", ["source.py", "input.bin", "experiment/source-snapshot/source.py"])
def test_changed_source_binding_or_snapshot_rejected_before_work(tmp_path, monkeypatch, changed):
    out, digest = synthetic_protocol(tmp_path, monkeypatch)
    Path(changed).write_bytes(b"changed")
    monkeypatch.setattr(runner, "load_data", forbidden_call)
    with pytest.raises(ValueError):
        runner.run(out, digest)
    assert not (out / "run-01").exists()


def test_recorded_runtime_mismatch_rejected(tmp_path, monkeypatch):
    out, _ = synthetic_protocol(tmp_path, monkeypatch)
    path = out / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["runtime"]["numpy"] = "different-runtime"
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "load_data", forbidden_call)
    with pytest.raises(ValueError, match="runtime"):
        runner.run(out, runner.sha(path))
    assert not (out / "run-01").exists()


def test_existing_attempt_is_exclusive_and_preserved(tmp_path, monkeypatch):
    out, digest = synthetic_protocol(tmp_path, monkeypatch)
    (out / "run-01").mkdir()
    sentinel = out / "run-01" / "sentinel"
    sentinel.write_bytes(b"preserved")
    monkeypatch.setattr(runner, "load_data", forbidden_call)
    with pytest.raises(FileExistsError):
        runner.run(out, digest)
    assert sentinel.read_bytes() == b"preserved"
    assert list((out / "run-01").iterdir()) == [sentinel]


def test_data_failure_preserves_original_and_failed_counts(tmp_path, monkeypatch):
    out, digest = synthetic_protocol(tmp_path, monkeypatch)
    original = RuntimeError("synthetic load failure")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(runner, "load_data", fail)
    with pytest.raises(RuntimeError) as caught:
        runner.run(out, digest)
    assert caught.value is original
    failed = json.loads((out / "run-01/failed.json").read_text())
    assert failed["status"] == "failed"
    assert failed["fits_completed"] == failed["rows_completed"] == failed["optimizer_updates"] == 0
    assert (out / "run-01/started.json").exists()
    assert not (out / "run-01/completed.json").exists()


def test_started_write_failure_is_preserved_in_failure_receipt(tmp_path, monkeypatch):
    out, digest = synthetic_protocol(tmp_path, monkeypatch)
    original, writer = OSError("synthetic started write failure"), runner.write_json

    def write(path, payload):
        if Path(path).name == "started.json":
            raise original
        return writer(path, payload)

    monkeypatch.setattr(runner, "write_json", write)
    monkeypatch.setattr(runner, "load_data", forbidden_call)
    with pytest.raises(OSError) as caught:
        runner.run(out, digest)
    assert caught.value is original
    assert json.loads((out / "run-01/failed.json").read_text())["status"] == "failed"


def test_failure_receipt_error_does_not_replace_original(tmp_path, monkeypatch):
    out, digest = synthetic_protocol(tmp_path, monkeypatch)
    original, writer = RuntimeError("original failure"), runner.write_json

    def fail(*args, **kwargs):
        raise original

    def write(path, payload):
        if Path(path).name == "failed.json":
            raise OSError("secondary receipt failure")
        return writer(path, payload)

    monkeypatch.setattr(runner, "load_data", fail)
    monkeypatch.setattr(runner, "write_json", write)
    with pytest.raises(RuntimeError) as caught:
        runner.run(out, digest)
    assert caught.value is original
    assert any("secondary receipt failure" in note for note in original.__notes__)


def test_failure_demotes_any_premature_completion(tmp_path, monkeypatch):
    out, digest = synthetic_protocol(tmp_path, monkeypatch)
    original = RuntimeError("injected late failure")

    def fail(*args, **kwargs):
        runner.write_json(out / "run-01/completed.json", {"status": "completed", "synthetic": True})
        raise original

    monkeypatch.setattr(runner, "load_data", fail)
    with pytest.raises(RuntimeError) as caught:
        runner.run(out, digest)
    assert caught.value is original
    assert not (out / "run-01/completed.json").exists()
    retained = [json.loads(path.read_text()) for path in (out / "run-01").glob("*.json")]
    assert {"status": "completed", "synthetic": True} in retained
    assert json.loads((out / "run-01/failed.json").read_text())["status"] == "failed"
