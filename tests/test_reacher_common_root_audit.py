"""Synthetic engineering410 common-root evidence; no scientific histories."""
from __future__ import annotations

import ast
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium.envs.mujoco.reacher_v5")

from openjev.research import reacher_common_root_audit as audit
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_common_root_branches import build_union, run_branches
from openjev.research.reacher_geometry_physics import PhysicsGeometryCEM
from openjev.research.reacher_tracking_control import GainPlanningBank
from openjev.research.reacher_tracking_dynamics import TrackingDynamicsEpisode, nominal_model
from openjev.research.reacher_tracking_rollout import jsonable, save_decision


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), allow_nan=False))


def save_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def dict_inputs(value):
    return dict(zip(("initial", "random_extra", "cem/1", "cem/2", "cem/3"),
                    (value.initial, value.random_extra, *value.cem), strict=True))


def object_inputs(values, chunks):
    return SearchInputs(values["initial"][:, :, :chunks], values["random_extra"][:, :, :chunks],
                        tuple(values[f"cem/{j}"][:, :, :chunks] for j in range(1, 4)))


def seal(folder, work, wall):
    write(folder/"completed.json", {"status": "completed", "files": {
        p.relative_to(folder).as_posix(): audit.file_hash(p)
        for p in folder.rglob("*") if p.is_file() and p != folder/"completed.json"},
        "wall_seconds": wall, "work": work})


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    started = time.perf_counter(); folder = tmp_path_factory.mktemp("common-root-audit-engineering410")
    rng = np.random.default_rng(410); banks = []
    for _ in range(2):
        banks.append(dict_inputs(SearchInputs(rng.normal(size=(1, 64, 8, 2)), rng.normal(size=(1, 192, 8, 2)),
                      tuple(rng.normal(size=(1, k, 8, 2)) for k in (64, 64, 63)))))
    noise = rng.normal(0, .05, size=(4, 24, 2))
    env = TrackingDynamicsEpisode(horizon=200, target_path=np.tile([.12, -.04], (201, 1)),
        gear_multiplier=np.full(200, .7), sensor_schedule=np.ones(201, bool), noise=np.zeros((200, 2)))
    try:
        packet = env.reset(410)
        root = {"source_role": "nominal", "step": 0, "root": jsonable(env.episode_record()["audit"]["decision_states"][0]),
                "true_gain": .7, "public_packet": packet.tolist(), "source_episode_sha256": "1"*64,
                "source_completed_sha256": "2"*64}
    finally:
        env.close()
    write(folder/"root.json", root)
    nominal = nominal_model(); q = np.array(root["root"]["qpos"])[None]; v = np.array(root["root"]["qvel"])[None]
    searches = {}; results = {}
    for assumption, gain in (("nominal", 1.), ("actual", .7)):
        for name in audit.SEARCHES:
            h = 24 if name == "long_a" else 12; label = assumption+"--"+name
            planner = GainPlanningBank(nominal, allowed_gains=np.array([gain]), steps=200, noise_std=.05,
                                       planning_horizon=h, action_block=3)
            inputs = object_inputs(banks[int(name == "short_b")], h//3)
            result, journal, snapshot = planner.plan(q.copy(), v.copy(), packet[None, 4:6].copy(), gain,
                                                     inputs, step=0, deadline=time.monotonic()+120)
            trace = {"step": 0, "arm": assumption, "root_qpos": q[0].copy(), "root_qvel": v[0].copy(),
                "public_packet": packet.copy(), "public_qpos": q[0].copy(), "public_qvel": v[0].copy(),
                "planning_gain": gain, "action": result.selected_actions[0].copy(), "search": result,
                "journal": journal, "bank_snapshot": snapshot}
            sf = folder/"searches"/label
            (sf/"decisions").mkdir(parents=True)
            save_decision(sf, trace, inputs)
            results[label] = result
            searches[label] = audit.read_npz(sf/"decisions/000.npz", audit.COMMON | audit.PLANNED)
    selected = []
    for assumption in audit.ASSUMPTIONS:
        a, b, long = (results[assumption+"--"+name] for name in audit.SEARCHES)
        ab = b if b.scores[0, b.selected_ids[0]] > a.scores[0, a.selected_ids[0]] else a
        selected.extend(row.selected_sequences[0] for row in (a, ab, long))
    union = build_union(results["nominal--long_a"].sequences[0, :64], selected)
    save_npz(folder/"union.npz", union)
    for assumption, gain in (("nominal", 1.), ("actual", .7)):
        model = audit.checked_model(nominal, gain)
        scorer = PhysicsGeometryCEM(model, steps=200, planning_horizon=24, action_block=3, noise_std=.05)
        scores, journal = scorer.score_bank(q.copy(), v.copy(), packet[None, 4:6].copy(), union["unique"][None].copy(), step=0)
        values = journal["banks"][0]["arrays"]
        score12 = np.zeros(len(union["unique"]), np.float32)
        for offset in range(12):score12 += np.clip(values["geometry_reward"][0, :, offset], -2.5, 0.)
        save_npz(folder/f"cross/{assumption}.npz", {"sequences": union["unique"],
            "predicted_angles": values["predicted_angles"][0], "raw_rewards": values["geometry_reward"][0],
            "scores_12": score12, "scores_24": scores[0], "qpos": values["qpos"][0], "qvel": values["qvel"][0]})
        write(folder/f"cross/{assumption}.json", {"configuration": journal["configuration"],
                                                 "journalmetadata": journal["banks"][0]["metadata"]})
    run_branches(nominal, root=root["root"], true_gain=.7, sequences=union["unique"], noise=noise,
                 out=folder/"branches", deadline=time.monotonic()+120)
    k = len(union["unique"]); work = {"search_candidate": 24576, "selected_advance": 6,
                                     "cross_score": 48*k, "native_branch": 96*k}
    seal(folder, work, time.perf_counter()-started)
    return {"folder": folder, "nominal": nominal, "root": root, "noise": noise,
            "a": banks[0], "b": banks[1], "searches": searches, "union": union, "work": work}


def invoke(e, folder=None, **changes):
    args = {"nominal_model": e["nominal"], "inputs_a": e["a"], "inputs_b": e["b"],
            "expected_root": e["root"], "expected_noise": e["noise"], **changes}
    return audit.audit_slot(folder or e["folder"], **args)


def test_complete_actual_saved_slot_native_and_score_audit(evidence):
    result = invoke(evidence)
    assert result["status"] == "completed" and result["work"] == evidence["work"]
    assert result["identity_slots"] == 70 and result["unique_sequences"] == len(evidence["union"]["unique"])
    assert result["max_native_abs_error"] == 0
    assert result["max_geometry_abs_error"] <= 5e-7
    assert result["native_transitions_checked"] == sum(evidence["work"].values())
    assert result["new_model_calls"] == result["new_rng_draws"] == result["new_planner_calls"] == 0


def test_union_exact_slots_and_first_occurrence(evidence):
    derived = audit.verify_union(evidence["union"], evidence["searches"])
    for name, value in derived.items():np.testing.assert_array_equal(value, evidence["union"][name])
    assert derived["slot_to_unique"].shape == (70,)
    assert 1 <= len(derived["unique"]) <= 70
    assert not np.signbit(derived["slots"][derived["slots"] == 0]).any()


@pytest.mark.parametrize("key", ["slots", "unique", "slot_to_unique", "first_slot"])
def test_union_identity_corruption(evidence, key):
    value = {name: v.copy() for name, v in evidence["union"].items()}
    value[key].flat[-1] += 1 if value[key].dtype == np.int64 else .01
    with pytest.raises(ValueError, match="union"):
        audit.verify_union(value, evidence["searches"])


def test_restart_tie_uses_A_and_nonbest_winner_rejected(evidence):
    searches = copy.deepcopy(evidence["searches"])
    for assumption in audit.ASSUMPTIONS:
        for name in audit.SEARCHES:
            row = searches[assumption+"--"+name]; row["scores"][:] = 0.; row["selected_id"][...] = 0
        searches[assumption+"--short_a"]["sequences"][0] = .1
        searches[assumption+"--short_b"]["sequences"][0] = -.1
    result = audit.rebuild_union(searches)
    np.testing.assert_array_equal(result["slots"][64], result["slots"][65])
    np.testing.assert_array_equal(result["slots"][67], result["slots"][68])
    assert np.all(result["slots"][65] == np.float32(.1))
    searches["nominal--short_a"]["selected_id"][...] = 1
    with pytest.raises(ValueError, match="global best"):
        audit.rebuild_union(searches)


@pytest.mark.parametrize("kind", ["gain", "root", "step", "source_hash", "source_role", "public_packet"])
def test_expected_root_cannot_self_certify(evidence, kind):
    saved = copy.deepcopy(evidence["root"])
    if kind == "gain":saved["true_gain"] = 1.
    elif kind == "root":saved["root"]["qvel"][0] += .01
    elif kind == "step":saved["step"] = 50
    elif kind == "source_hash":saved["source_episode_sha256"] = "3"*64
    elif kind == "source_role":saved["source_role"] = "true_state"
    else:saved["public_packet"][4] += .01
    with pytest.raises(ValueError, match="authenticated"):
        audit.validate_root(saved, evidence["root"], evidence["nominal"])


@pytest.mark.parametrize("kind", ["prefix_score", "full_score", "geometry", "qvel", "angles", "gain_config", "count", "time"])
def test_cross_arithmetic_state_and_paid_cost_corruption(evidence, kind):
    folder=evidence["folder"]
    values=audit.read_npz(folder/"cross/actual.npz", {"sequences", "predicted_angles", "raw_rewards", "scores_12", "scores_24", "qpos", "qvel"})
    meta=json.loads((folder/"cross/actual.json").read_text())
    if kind == "prefix_score":values["scores_12"][0] += .01
    elif kind == "full_score":values["scores_24"][0] += .01
    elif kind == "geometry":values["raw_rewards"][0, 0] += .01
    elif kind == "qvel":values["qvel"][0, 1, 0] += .01
    elif kind == "angles":values["predicted_angles"][0, 0, 0] += .01
    elif kind == "gain_config":meta["configuration"]["model_binary_sha256"] = "0"*64
    elif kind == "count":meta["journalmetadata"]["native_substeps_completed"] -= 1
    else:meta["journalmetadata"]["native_seconds"] = meta["journalmetadata"]["wall_seconds"]+1.
    with pytest.raises(ValueError):
        audit.audit_cross(values,meta,nominal=evidence["nominal"],root=evidence["root"],gain=.7,
                          sequences=evidence["union"]["unique"])


@pytest.mark.parametrize("kind", ["noise", "state", "reward", "effort", "mask", "counter", "root", "time", "gain"])
def test_branch_resealed_corruption(evidence, tmp_path, kind):
    folder=tmp_path/"branches";shutil.copytree(evidence["folder"]/"branches",folder)
    receipt=json.loads((folder/"completed.json").read_text()); started=json.loads((folder/"started.json").read_text())
    with np.load(folder/"data.npz") as f:values={key:f[key] for key in f.files}
    if kind == "noise":values["noise"][0,0,0] += .01
    elif kind == "state":values["integration_states"][0,0,1,-1] += .01
    elif kind == "reward":values["rewards"][0,0,0] += .01
    elif kind == "effort":values["effort"][0,0,0] += .01
    elif kind == "mask":values["recorded"][0,0,0] = False
    elif kind == "counter":receipt["counters"]["restore_calls_completed"] -= 1
    elif kind == "root":started["root"]["qpos"][0] += .01
    elif kind == "time":values["time"][0,0,1] += .01
    elif kind == "gain":started["configuration"]["true_gain"] = receipt["configuration"]["true_gain"] = 1.
    save_npz(folder/"data.npz",values);write(folder/"started.json",started)
    receipt["files"]={name:audit.file_hash(folder/name) for name in receipt["files"]};write(folder/"completed.json",receipt)
    with pytest.raises(ValueError):
        audit.audit_branches(folder,nominal=evidence["nominal"],root=evidence["root"],
                             sequences=evidence["union"]["unique"],noise=evidence["noise"])


@pytest.mark.parametrize("kind", ["union", "search_gain", "search_winner", "failed", "unpaid", "work", "extra"])
def test_whole_slot_resealed_or_incomplete_rejected(evidence,tmp_path,kind):
    folder=tmp_path/"slot";shutil.copytree(evidence["folder"],folder)
    receipt=json.loads((folder/"completed.json").read_text())
    if kind == "union":
        path=folder/"union.npz"
        with np.load(path) as f:values={k:f[k] for k in f.files}
        values["slot_to_unique"][-1]=0;save_npz(path,values)
    elif kind in ("search_gain","search_winner"):
        path=folder/"searches/actual--short_a/decisions/000.npz"
        with np.load(path) as f:values={k:f[k] for k in f.files}
        if kind == "search_gain":values["planning_gain"][...]=1.
        else:values["selected_id"][...]=(int(values["selected_id"])+1)%256
        save_npz(path,values)
    elif kind == "failed":write(folder/"failed.json",{})
    elif kind == "extra":write(folder/"extra.json",{})
    elif kind == "unpaid":receipt["wall_seconds"]=0.
    elif kind == "work":receipt["work"]["selected_advance"]=0
    receipt["files"]={name:audit.file_hash(folder/name) for name in receipt["files"]}
    write(folder/"completed.json",receipt)
    with pytest.raises(ValueError):invoke(evidence,folder)


def test_external_noise_and_expired_deadline(evidence):
    noise=evidence["noise"].copy();noise[0,0,0]+=.01
    with pytest.raises(ValueError,match="noise"):invoke(evidence,expected_noise=noise)
    with pytest.raises(TimeoutError):invoke(evidence,deadline=time.perf_counter()-1.)


def test_no_search_scorer_branch_runner_or_rng_import():
    tree=ast.parse(Path(audit.__file__).read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert imports==["__future__","pathlib","openjev.research.reacher_tracking_audit"]
    assert not any(isinstance(n,ast.Global) for n in ast.walk(tree))
