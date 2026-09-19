"""Post hoc saved-output diagnostic. Never imports a model or native engine."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix("")
AUDIT_SHA = "2589383326a2312d7f738821da7fd11f82b9bee1b7168aba2e7f44aaa4946d5d"
PLAN_SHA = "05f09ee5425190f5d652aedb10af1641037035d85e906d04d86baf33552a5786"
ARMS = ("residual_gru", "current_gru", "bounded_gru", "packet_mlp")
PANELS = ("full", "ordinary", "shift")
READS = {}


def read(path, expected):
    path = ROOT / path
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    assert actual == expected, (str(path), actual, expected)
    READS[str(path.relative_to(ROOT))] = actual
    return content


def read_json(path, expected):
    return json.loads(read(path, expected))


def native_source(name, expected):
    paths = list((ROOT / ".venv-robotics/lib").glob("python*/site-packages/gymnasium/envs/mujoco/" + name))
    assert len(paths) == 1
    read(paths[0], expected)


def clipped_cost_moments(commands, sigma):
    """Independent float64 truncated-normal first two moments of sum X_i^2.

    Integration recurrence I_n = a^(n-1)phi(a)-b^(n-1)phi(b)+(n-1)I_(n-2).
    This fixed study uses sigma=.05; no general large-sigma promise is made.
    """
    assert sigma == 0.05
    u = np.clip(np.asarray(commands, dtype=np.float64), -1, 1)
    a, b = (-1 - u) / sigma, (1 - u) / sigma
    erf = np.vectorize(math.erf, otypes=[float])
    phi_a = np.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    phi_b = np.exp(-b * b / 2) / math.sqrt(2 * math.pi)
    inside = 0.5 * (erf(b / math.sqrt(2)) - erf(a / math.sqrt(2)))
    integrals = [inside, phi_a - phi_b]
    for n in range(2, 5):
        integrals.append(a ** (n - 1) * phi_a - b ** (n - 1) * phi_b + (n - 1) * integrals[n - 2])
    moments = {}
    for power in (2, 4):
        moments[power] = 1 - inside + sum(
            math.comb(power, k) * u ** (power - k) * sigma**k * integrals[k]
            for k in range(power + 1)
        )
    variance = moments[4] - moments[2] ** 2
    assert np.min(variance) >= -1e-14
    return moments[2].sum(-1), np.maximum(variance, 0).sum(-1)


def metric(error, mask):
    values = error[mask]
    if not values.size:
        return None
    return {"transitions": int(mask.sum()), "mse": float(np.mean(values**2)),
            "rmse": float(np.sqrt(np.mean(values**2))), "bias": float(values.mean()),
            "mae": float(np.mean(np.abs(values)))}


def diagnose(ep, pred, expected_cost):
    packets = ep["policy__packets"]
    commands = ep["policy__commands"]
    n, steps = commands.shape[:2]
    assert packets.shape == (n, steps + 1, 8)
    assert pred["angles"].shape == (n, steps, 4) and pred["rewards"].shape == (n, steps)
    assert all(np.isfinite(value).all() for value in (*ep.values(), *pred.values()))
    reward = ep["audit__rewards"]
    distance = -ep["audit__reward_dist"]
    cost = -ep["audit__reward_ctrl"]
    raw = ep["audit__raw_obs"]
    assert np.allclose(reward, -distance - cost, rtol=0, atol=1e-14)
    assert np.allclose(cost, (ep["audit__applied_actions"] ** 2).sum(-1), rtol=0, atol=1e-14)
    assert np.allclose(distance, np.linalg.norm(raw[:, 1:, -2:], axis=-1), rtol=0, atol=1e-14)
    mean_cost, var_cost = clipped_cost_moments(commands, 0.05)
    # rhat = residual_head - E[actuator_cost]. Dhat below is the negated
    # learned residual, not a supervised pure-distance head.
    predicted_distance = -pred["rewards"].astype(float) - mean_cost
    reward_error = pred["rewards"].astype(float) - reward
    distance_error = predicted_distance - distance
    cost_noise = cost - mean_cost
    assert np.allclose(reward_error, -distance_error + cost_noise, rtol=0, atol=2e-15)
    angle_error = pred["angles"].astype(float) - raw[:, 1:, :4]
    root_seen, end_seen = packets[:, :-1, 6] > 0.5, packets[:, 1:, 6] > 0.5
    masks = {"all": np.ones((n, steps), bool), "root_visible": root_seen,
             "root_missing": ~root_seen, "endpoint_visible": end_seen,
             "endpoint_missing": ~end_seen, "both_visible": root_seen & end_seen}
    result = {"groups": {key: {"reward": metric(reward_error, mask),
                "learned_residual_distance": metric(distance_error, mask),
                "angle_features": metric(angle_error, mask)} for key, mask in masks.items()}}
    result.update({"mean_episode_cost": float(-reward.sum(1).mean()),
                   "mean_episode_distance_cost": float(distance.sum(1).mean()),
                   "mean_episode_actuator_cost": float(cost.sum(1).mean()),
                   "actuator_cost_innovation_mse": float(np.mean(cost_noise**2)),
                   "mean_analytic_actuator_cost_variance": float(var_cost.mean()),
                   "distance_actuator_cross_term": float(-2 * np.mean(distance_error * cost_noise)),
                   "distance_population_sd": float(distance.std()),
                   "negative_learned_distance_fraction": float(np.mean(predicted_distance < 0)),
                   "selected_reward_positive_fraction": float(np.mean(pred["rewards"] > 0)),
                   "selected_reward_below_search_clip_fraction": float(np.mean(pred["rewards"] < -2.5)),
                   "native_distance_identity_max_abs_error": float(np.max(np.abs(distance - np.linalg.norm(raw[:, 1:, -2:], axis=-1))))})
    if expected_cost is not None:
        assert np.allclose(-reward.sum(1), expected_cost, rtol=0, atol=1e-12)
    # Timing canary only: compare TRUE final qpos FK with cached native reward.
    # Do not evaluate decoded predicted angles as a purported exact reward.
    q = ep["audit__qpos"][:, 1:]
    finger = np.stack((0.1 * np.cos(q[..., 0]) + 0.11 * np.cos(q[..., 0] + q[..., 1]),
                       0.1 * np.sin(q[..., 0]) + 0.11 * np.sin(q[..., 0] + q[..., 1])), axis=-1)
    vec_error = finger - q[..., 2:4] - raw[:, 1:, -2:]
    dist_error = np.linalg.norm(finger - q[..., 2:4], axis=-1) - distance
    result["qpos_fk_timing_canary"] = {"native_vector_error_max_m": float(np.linalg.norm(vec_error, axis=-1).max()),
        "native_vector_error_rms_m": float(np.sqrt(np.mean(np.sum(vec_error**2, axis=-1)))),
        "native_distance_error_max_m": float(np.abs(dist_error).max()),
        "native_distance_error_rms_m": float(np.sqrt(np.mean(dist_error**2))),
        "exact_native_reward_replacement_valid": bool(np.abs(dist_error).max() < 1e-10)}
    per_case = {"reward_mse": np.mean(reward_error**2, axis=1),
                "distance_mse": np.mean(distance_error**2, axis=1),
                "angles_mse": np.mean(angle_error**2, axis=(1, 2)),
                "cost": -reward.sum(1)}
    return result, per_case


def paired_interval(values):
    """Normal approximation over case clusters, conditional on all fixed fits."""
    values = np.asarray(values)
    estimate = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {"mean_difference": estimate, "conditional_case_cluster_se": se,
            "approximate_normal_95_interval": [estimate - 1.96 * se, estimate + 1.96 * se],
            "case_clusters": len(values)}


def main():
    receipt_path = Path("evidence/reacher-memory-ablation-v1/audit/receipt.json")
    receipt = read_json(receipt_path, AUDIT_SHA)
    assert receipt["status"] == "completed" and receipt["engineering"] is False
    plan = read_json("evidence/reacher-memory-ablation-v1/protocol/plan.json", PLAN_SHA)
    assert receipt["plan_sha256"] == PLAN_SHA
    summary = read_json(receipt_path.with_name("summary.json"), receipt["files"]["summary.json"])
    execution = Path("runs/reacher-memory-ablation-v1/attempt")
    completion = read_json(execution / "completed.json", receipt["execution_completed_sha256"])
    assert completion["status"] == "completed"
    for path in ("src/openjev/research/robotics_reacher.py", "src/openjev/research/reacher_reward_residual.py",
                 "src/openjev/research/reacher_memory_control.py", "src/openjev/research/reacher_world_models.py",
                 "scripts/reacher_world_model_study.py", "scripts/audit_reacher_memory_study.py",
                 "scripts/audit_reacher_objective_study.py"):
        read(path, receipt["source_sha256"][path])
    native_source("reacher_v5.py", receipt["runtime"]["native_source_sha256"])
    native_source("assets/reacher.xml", receipt["runtime"]["native_xml_sha256"])
    assert set(plan["fit_order"]) == {f"{arm}-pair{pair}" for arm in ARMS for pair in range(3)}
    assert plan["control_episodes"] == 64 and plan["steps"] == 50

    def arrays(rel):
        import io
        content = read(execution / rel, receipt["execution_members"][rel])
        with np.load(io.BytesIO(content), allow_pickle=False) as z:
            return {key: z[key] for key in z.files}

    checks = clipped_cost_moments(np.array([[0., 0.]]), 0.05)
    assert np.allclose(checks[0], 0.005, atol=1e-14)
    assert np.allclose(checks[1], 0.000025, atol=1e-14)
    rows, cases = {}, {}
    for panel in PANELS:
        rows[panel], cases[panel] = {}, {}
        for name in plan["fit_order"]:
            rel = f"control/{panel}/{name}"
            rows[panel][name], cases[panel][name] = diagnose(
                arrays(rel + "/episodes.npz"), arrays(rel + "/executed_predictions.npz"),
                summary["control"][panel][name]["episode_costs"])
    common_rows, common_cases = {}, {}
    ep = arrays("prediction.npz")
    for name in plan["fit_order"]:
        pred = arrays(f"predictions/{name}.npz")
        common_rows[name], common_cases[name] = diagnose(ep, {"angles": pred["one_step_angles"],
                                                           "rewards": pred["one_step_rewards"]}, None)
        assert math.isclose(common_rows[name]["groups"]["all"]["reward"]["mse"],
                            summary["prediction"][name]["one_step_reward_mse"], rel_tol=1e-12)
    rows["common_prediction"], cases["common_prediction"] = common_rows, common_cases
    families = {}
    for panel in rows:
        families[panel] = {}
        for arm in ARMS:
            fits = [rows[panel][f"{arm}-pair{p}"] for p in range(3)]
            record = {key: float(np.mean([fit[key] for fit in fits])) for key in (
                "mean_episode_cost", "mean_episode_distance_cost", "mean_episode_actuator_cost",
                "actuator_cost_innovation_mse", "mean_analytic_actuator_cost_variance", "distance_actuator_cross_term")}
            for group in fits[0]["groups"]:
                record[group] = {key: None if fits[0]["groups"][group][key] is None else {
                    m: float(np.mean([fit["groups"][group][key][m] for fit in fits]))
                    for m in ("mse", "bias", "mae")}
                    for key in ("reward", "learned_residual_distance", "angle_features")}
            families[panel][arm] = record
    contrasts = {}
    for panel in cases:
        contrasts[panel] = {}
        for other in ARMS[1:]:
            entry = {}
            for key in ("reward_mse", "distance_mse", "angles_mse", "cost"):
                diff = np.stack([cases[panel][f"residual_gru-pair{p}"][key]
                                 - cases[panel][f"{other}-pair{p}"][key] for p in range(3)])
                entry[key] = {**paired_interval(diff.mean(0)), "three_paired_fit_differences": diff.mean(1).tolist()}
            contrasts[panel][other] = entry
    result = {"status": "completed_post_hoc_descriptive_diagnostic", "study": "reacher-memory-ablation-v1",
              "plan_sha256": PLAN_SHA, "audit_receipt_sha256": AUDIT_SHA,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "new_model_calls": 0, "new_native_calls": 0, "new_rollouts": 0, "rng_draws": 0,
              "coverage": {"fits": 12, "control_panels": 3, "control_rows": 36,
                           "cases_per_control_row": 64, "steps": 50, "common_prediction_episodes": 96},
              "authenticated_reads": READS, "family_means": families, "all_fit_rows": rows,
              "persistent_minus_comparator": contrasts,
              "angle_derived_reward_status": "Not computed. True final-qpos FK does not exactly match native cached distance at the same step; a decoder-to-FK replacement is not an exact native reward diagnostic.",
              "limits": ["Post hoc diagnostic, not a new prospective gate or counterfactual policy test.",
                         "Control trajectories differ across models; common prediction inputs are identical.",
                         "Intervals cluster cases after averaging three fixed fits and do not measure training-seed uncertainty; normal approximation, no multiplicity correction.",
                         "Learned residual is trained on total reward minus expected action cost; its discrepancy against distance includes correlated disturbance effects, not a pure supervised distance error.",
                         "Action cost variance alone is not the irreducible variance of total reward because distance also depends on realized action noise.",
                         "Only selected-action one-step predictions are evaluated; no unchosen native branches or multistep reward truth are available, so planner ranking quality and reward bottleneck causality remain untested.",
                         "No current cache-study artifacts were read."]}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUT.with_suffix('.json')), "authenticated_files": len(READS),
                      "control_rows": 36, "common_prediction_fits": 12,
                      "new_model_calls": 0, "new_native_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
