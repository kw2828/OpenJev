"""Synthetic NumPy/tensor evidence only; no real data, model, or random calls."""
from __future__ import annotations

import copy
import json

import audit_pose_innovation as audit
import numpy as np
import pytest
import torch


def ids():
    return np.column_stack((np.repeat(np.arange(30), 24), np.tile(np.arange(24)*50, 30)))


def metric(value, parents):
    values = np.broadcast_to(np.asarray(value), (len(parents),)).copy()
    mean = float(values.mean())
    return {"mse": mean, "rmse": mean**.5, "parent_ids": list(parents), "parent_mse": values.tolist(),
            "horizon_mse": [mean]*25}


def aggregate_fixture(candidate=.8):
    result = {}
    for seed in audit.SEEDS:
        for fold in audit.FOLDS:
            index, train, test = audit.split(ids(), fold)
            result[f"fold-{fold}-{seed}"] = {v: {part: {e: metric(candidate if v == "recurrent" else 1.,
                np.unique(ids()[index[mask], 0]).tolist()) for e in audit.ENDPOINTS}
                for part, mask in (("train", train), ("test", test))} for v in audit.VARIANTS}
    return result


def test_coverage_and_exact_method_order():
    import run_pose_innovation as producer
    assert audit.NEURAL == producer.NEURAL
    assert audit.VARIANTS == producer.VARIANTS
    assert len(audit.expected_members()) == 380
    families, _, screen = audit.aggregate(aggregate_fixture())
    assert screen["passed"] and screen["checks_passed"] == screen["total_checks"] == 368
    assert len(families["test"]["base"]["position"]["parent_ids"]) == 12
    assert len(families["train"]["base"]["position"]["parent_ids"]) == 18


@pytest.mark.parametrize("fold", range(3))
def test_exact_split_and_disjoint_backbone(fold):
    index, train, test = audit.split(ids(), fold)
    assert index.shape == (240,) and train.sum() == 144 and test.sum() == 96
    assert set(ids()[index[test], 0]) == {fold+3*r for r in (1, 4, 7, 9)}
    assert (ids()[index, 0] % 3 == fold).all()
    bad = ids(); bad[0] = bad[1]
    with pytest.raises(ValueError):
        audit.split(bad, fold)


@pytest.mark.parametrize("candidate,expected", [(.81, True), (np.nextafter(.81, 1), False)])
def test_inclusive_mean_margin(candidate, expected):
    _, _, screen = audit.aggregate(aggregate_fixture(candidate))
    mean = [x for x in screen["checks"] if x["name"].endswith("/mean")]
    assert all(x["passed"] is expected for x in mean)


def test_one_bad_pair_cannot_hide_behind_good_family():
    rows = aggregate_fixture(.4)
    entry = rows["fold-0-1101"]["recurrent"]["test"]
    for endpoint in audit.ENDPOINTS:
        entry[endpoint] = metric(1.1, entry[endpoint]["parent_ids"])
    _, _, screen = audit.aggregate(rows)
    assert not screen["passed"]
    assert all(x["passed"] for x in screen["checks"] if x["name"].endswith("/mean"))
    assert all(not x["passed"] for x in screen["checks"] if x["name"].endswith("/fold-0-1101"))


def test_three_bad_parents_fail_even_with_good_pairs():
    rows = aggregate_fixture(.1)
    for seed in audit.SEEDS:
        for fold in audit.FOLDS:
            for endpoint in audit.ENDPOINTS:
                entry = rows[f"fold-{fold}-{seed}"]["recurrent"]["test"][endpoint]
                rows[f"fold-{fold}-{seed}"]["recurrent"]["test"][endpoint] = metric([1.1, .1, .1, .1], entry["parent_ids"])
    _, _, screen = audit.aggregate(rows)
    assert all(not x["passed"] and x["parents_nonworse"] == 9 for x in screen["checks"] if x["name"].endswith("/parents"))
    assert all(x["passed"] for x in screen["checks"] if "/fold-" in x["name"])


def test_train_errors_do_not_enter_test_gate_and_zero_reference_fails():
    rows = aggregate_fixture(.8)
    for group in rows.values():
        for endpoint in audit.ENDPOINTS:
            e = group["recurrent"]["train"][endpoint]
            group["recurrent"]["train"][endpoint] = metric(100., e["parent_ids"])
    families, _, screen = audit.aggregate(rows)
    assert screen["passed"] and families["train"]["recurrent"]["position"]["mse"] == 100
    for group in rows.values():
        for endpoint in audit.ENDPOINTS:
            e = group["base"]["test"][endpoint]
            group["base"]["test"][endpoint] = metric(0., e["parent_ids"])
    assert not audit.aggregate(rows)[2]["passed"]


def test_left_noncommuting_root_frame_correction_and_residual_roundtrip():
    root = audit.rotation_exp(np.array([[.4, -.2, .1]]))
    base_r = audit.rotation_exp(np.array([[[.1, .3, -.4], [.2, -.1, .5]]]))
    base_p = np.array([[[1., 2., 3.], [2., 3., 4.]]])
    correction = np.array([[[1., -.5, .2, .5, .3, -.1], [-.3, .1, .7, -.2, .4, .1]]])
    p, r = audit.corrected(base_p, base_r, root, correction)
    recovered = audit.residual(base_p, base_r, p, r, root, [.1, .1])
    np.testing.assert_allclose(recovered, correction, atol=1e-13)
    assert not np.allclose(r, base_r @ audit.rotation_exp(.1*correction[..., 3:]))
    assert max(x.max() for x in audit.errors(p, r, p, r).values()) < 1e-28


def test_float32_torch_geometry_and_ridge_match_independent_arithmetic():
    from run_pose_innovation import ridge_predict, summary_features

    from openjev.research.pose_innovation import apply_correction, residual_targets
    from openjev.research.rigid_motion import so3_exp
    # Deterministic arithmetic fixture, no generator or neural construction.
    tokens = torch.sin(torch.arange(6*31*55, dtype=torch.float32).reshape(6, 31, 55)*.037)
    target = torch.cos(torch.arange(6*25*6, dtype=torch.float32).reshape(6, 25, 6)*.021)*.04
    train = np.array([True, True, False, True, False, True])
    for kind, feature in (("ridge_summary", summary_features(tokens)), ("ridge_ordered", tokens.flatten(1))):
        produced, saved = ridge_predict(feature, target, train)
        expected, witness = audit.ridge_check(saved, tokens.numpy(), target.numpy(), train, kind)
        np.testing.assert_allclose(expected, produced.numpy(), rtol=2e-5, atol=2e-6)
        assert witness < 1e-7
        poisoned = target.numpy().copy(); poisoned[~train] = np.nan
        np.testing.assert_array_equal(audit.ridge_check(saved, tokens.numpy(), poisoned, train, kind)[0], expected)
        bad = copy.deepcopy(saved); bad["weights"][0, 0] += .2
        with pytest.raises(ValueError, match="normal-equation"):
            audit.ridge_check(bad, tokens.numpy(), target.numpy(), train, kind)
    base_p = target[..., :3]
    base_r = so3_exp(target[..., 3:])
    root = so3_exp(torch.tensor([[.5, -.4, .2]]).expand(6, -1))
    p, r = apply_correction(base_p, base_r, root, target, torch.tensor([.1, .1]))
    xp, xr = audit.corrected(base_p.numpy(), base_r.numpy(), root.numpy(), target.numpy())
    np.testing.assert_allclose(p.numpy(), xp, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(r.numpy(), xr, atol=2e-6, rtol=2e-6)
    y = residual_targets(base_p, base_r, p, r, root, torch.tensor([.1, .1]))
    expected = audit.residual(base_p.numpy(), base_r.numpy(), p.numpy(), r.numpy(), root.numpy(), [.1, .1])
    np.testing.assert_allclose(y.numpy(), expected, atol=5e-6, rtol=2e-5)


def weights(variant):
    shapes = ({"network.0.weight": (9, 165), "network.0.bias": (9,), "network.2.weight": (150, 9), "network.2.bias": (150,)}
              if variant == "summary" else {"gru.weight_ih_l0": (24, 55), "gru.weight_hh_l0": (24, 8),
                  "gru.bias_ih_l0": (24,), "gru.bias_hh_l0": (24,), "head.weight": (150, 8), "head.bias": (150,)})
    return {k: torch.zeros(s, dtype=torch.float32) for k, s in shapes.items()}


@pytest.mark.parametrize("variant", audit.NEURAL)
def test_checkpoint_schema_and_zero_head(tmp_path, variant):
    path = tmp_path/"state.pt"
    value = weights(variant)
    torch.save(value, path)
    audit.checkpoint(path, variant, True)
    head = "network.2.bias" if variant == "summary" else "head.bias"
    value[head][0] = 1
    torch.save(value, path)
    with pytest.raises(ValueError, match="zero initial"):
        audit.checkpoint(path, variant, True)
    value[head][0] = float("nan")
    torch.save(value, path)
    with pytest.raises(ValueError, match="checkpoint tensor"):
        audit.checkpoint(path, variant)


def fake_complete(tmp_path):
    experiment = tmp_path/"experiment"
    run = experiment/"run-01"
    run.mkdir(parents=True)
    identity = ids()
    original_p = np.zeros((720, 57, 3), np.float32)
    original_r = np.broadcast_to(np.eye(3, dtype=np.float32), (720, 57, 3, 3)).copy()
    actions = np.zeros((720, 56, 40), np.float32)
    parent = tmp_path/audit.PARENT/"run-01"; parent.mkdir(parents=True)
    protocol = {"sources": {}}
    audit.write_json(experiment/"protocol.json", protocol)
    digest = audit.sha(experiment/"protocol.json")
    audit.write_json(run/"started.json", {"protocol_sha256": digest})
    fits, rows, caches = [], [], []
    for seed in audit.SEEDS:
        for fold in audit.FOLDS:
            stem = f"fold-{fold}-{seed}"
            index, train, test = audit.split(identity, fold)
            p, r = original_p[index], original_r[index]
            np.savez_compressed(parent/(stem+"-normalization.npz"), motion_scales=np.ones(4, np.float32),
                action_mean=np.zeros(40, np.float32), action_scale=np.ones(40, np.float32),
                train_indices=np.flatnonzero(identity[:, 0] % 3 != fold))
            np.savez_compressed(parent/f"fold-{fold}-slow-{seed}-cache.npz", p=original_p[:, 32:], R=original_r[:, 32:])
            tokens = np.zeros((240, 31, 55), np.float32); tokens[..., 2] = np.tanh(np.float32(1))
            c = {"ids": identity[index], "indices": index, "train_mask": train, "test_mask": test,
                "tokens": tokens, "target": np.zeros((240, 25, 6), np.float32), "base_p": p[:, 32:], "base_R": r[:, 32:],
                "target_p": p[:, 32:], "target_R": r[:, 32:], "root_R": r[:, 31], "one_errors": np.zeros((240, 31, 6), np.float32),
                "permutations": np.broadcast_to(np.arange(31), (240, 31)).copy(), "context_p": p[:, :32], "context_R": r[:, :32],
                "past_actions": actions[index, :31], "one_p": p[:, 1:32], "one_R": r[:, 1:32], "motion_scales": np.ones(4, np.float32)}
            np.savez_compressed(run/(stem+"-cache.npz"), **c)
            caches.append({"seed": seed, "fold": fold, "seconds": .1, "replay_max": {"p": 0., "R": 0.},
                           "cache_sha256": audit.sha(run/(stem+"-cache.npz"))})
            np.save(run/(stem+"-orders.npy"), np.broadcast_to(np.flatnonzero(train), (40, 144)))
            for kind, features in (("ridge_summary", np.concatenate((tokens[:, -1], tokens.mean(1), tokens[:, -1]-tokens[:, 0]), -1)),
                                   ("ridge_ordered", tokens.reshape(240, -1))):
                x = features.astype(float); x /= np.linalg.norm(x, axis=1, keepdims=True)
                np.savez_compressed(run/(stem+"-"+kind+"-weights.npz"), x_mean=x[train].mean(0),
                    y_mean=np.zeros(150), weights=np.zeros((x.shape[1], 150)))
            for variant in audit.NEURAL:
                fs = stem+"-"+variant
                torch.save(weights(variant), run/(fs+"-initial.pt")); torch.save(weights(variant), run/(fs+".pt"))
                np.save(run/(fs+"-losses.npy"), np.zeros(200))
                fit = {"seed": seed, "fold": fold, "variant": variant, "updates": 200,
                    "parameters": 2994 if variant == "summary" else 2910, "seconds": .2,
                    "initial_sha256": audit.sha(run/(fs+"-initial.pt")), "final_sha256": audit.sha(run/(fs+".pt")),
                    "head_latency_ms": [1.]*12, "warmup_ms": [1.]*3, "first_loss": 0., "last_loss": 0.}
                audit.write_json(run/(fs+"-fit.json"), fit); fits.append(fit)
            for variant in audit.VARIANTS:
                prefix = stem+"-"+variant
                np.savez_compressed(run/(prefix+"-predictions.npz"), p=p[:, 32:], R=r[:, 32:], correction=c["target"])
                physical = {"position_rmse_m": 0., "rotation_rmse_rad": 0., "composite": 0.,
                            "rotation_orthogonality_max": 0., "rotation_determinant_max_error": 0.}
                row = {"seed": seed, "fold": fold, "variant": variant, "train": physical, "test": physical}
                audit.write_json(run/(prefix+"-evaluation.json"), row); rows.append(row)
    done = {"status": "completed", "protocol_sha256": digest, "fits": fits, "rows": rows, "caches": caches,
            "optimizer_updates": 9000, "external_panel_calls": 0, "frozen_backbone_batch_calls": 9,
            "wall_seconds": 100., "files": audit.members(run)}
    audit.write_json(run/"completed.json", done)
    return experiment, digest, audit.sha(run/"completed.json"), protocol, (original_p, original_r, actions, identity)


def test_full_saved_tree_preserves_failed_screen_and_exclusive_output(tmp_path, monkeypatch):
    experiment, protocol_sha, completion_sha, protocol, original = fake_complete(tmp_path)
    monkeypatch.setattr(audit, "validate_inputs", lambda *_: protocol)
    monkeypatch.setattr(audit, "original_training", lambda *_: original)
    out = tmp_path/"audit"
    receipt = audit.audit(experiment, out, protocol_sha256=protocol_sha, completed_sha256=completion_sha, root=tmp_path)
    assert receipt["status"] == "completed" and receipt["qualification_passed"] is False
    assert len(receipt["execution_members"]) == 380 and receipt["total_checks"] == 368
    assert set(receipt["files"]) == {"summary.json", "window-errors.npz", "README.md"}
    assert audit.read_json(out/"summary.json")["counts"]["optimizer_updates"] == 9000
    with pytest.raises(FileExistsError):
        audit.audit(experiment, out, protocol_sha256=protocol_sha, completed_sha256=completion_sha, root=tmp_path)
    # Corruption remains rejected even though the fixture replaces lineage auth.
    (experiment/"run-01/extra.json").write_text("{}")
    failed = tmp_path/"failed-audit"
    with pytest.raises(ValueError, match="380"):
        audit.audit(experiment, failed, protocol_sha256=protocol_sha, completed_sha256=completion_sha, root=tmp_path)
    assert audit.read_json(failed/"failed.json")["status"] == "failed"


@pytest.mark.parametrize("mutation", ["mask", "root", "tokens", "errors", "permutation", "futuretarget"])
def test_cache_corruption(tmp_path, mutation):
    experiment, _, _, _, original = fake_complete(tmp_path)
    c = audit.arrays(experiment/"run-01/fold-0-1101-cache.npz")
    if mutation == "mask": c["test_mask"][0] = True
    elif mutation == "root": c["root_R"][0] = audit.rotation_exp(np.array([.1, 0., 0.])).astype(np.float32)
    elif mutation == "tokens": c["tokens"][0, 0, 49] = .5
    elif mutation == "errors": c["one_errors"][0, 0, 0] = .5
    elif mutation == "permutation": c["permutations"][0, 0] = 1
    else: c["target_p"][0, 0, 0] = .2
    with pytest.raises(ValueError):
        audit.validate_cache(c, 0, 1101, original, tmp_path)


def test_external_pin_failure_is_retained(tmp_path):
    experiment = tmp_path/"experiment"; experiment.mkdir()
    (experiment/"protocol.json").write_text(json.dumps({"study": "fake"}))
    out = tmp_path/"audit"
    with pytest.raises(ValueError, match="external protocol"):
        audit.audit(experiment, out, protocol_sha256="0"*64, completed_sha256="1"*64, root=tmp_path)
    assert (out/"failed.json").is_file()
