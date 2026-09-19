"""Prospective search checks on synthetic artifacts and tiny engineering fixtures.

No inherited learned checkpoint or real training cohort is loaded by this file.
"""

import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openjev.research import reacher_search_protocol as protocol
from openjev.research.reacher_adaptive_search import SearchInputs, search
from openjev.research.reacher_world_models import GRUWorldModel
from openjev.research.robotics_reacher import ReacherEpisode, make_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reacher_search_study as study


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(834)
    yield
    torch.set_num_threads(threads)


def synthetic_plan():
    # Protocol metadata only; no execution artifact or performance is read.
    return json.loads((ROOT / "evidence/reacher-reward-residual-control-v2/protocol/plan.json").read_text())


def innovations(cases=2):
    rng = np.random.default_rng(834)
    return SearchInputs(rng.normal(size=(cases, 64, 4, 2)),
                        rng.normal(size=(cases, 192, 4, 2)),
                        tuple(rng.normal(size=(cases, count, 4, 2)) for count in (64, 64, 63)))


def small_protocol():
    plan = synthetic_plan()
    plan.update(study="reacher-search-engineering-unit-v1",
                rng_namespace="reacher-search-engineering-unit-v1", engineering_rng_namespaces=[],
                control_episodes=3, diagnostic_episodes=2,
                diagnostic_branches=4, planners=list(protocol.PLANNERS))
    plan["random_stream_contract"] = protocol.stream_contract(plan)
    return plan


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return protocol.sha(path)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """A complete synthetic parent lineage; no real learned tensor is loaded."""
    parent = synthetic_plan()
    previous_plans = [(row, (ROOT / row["plan_path"]).read_bytes())
                      for row in parent["random_stream_contract"]["priors"]]
    root = tmp_path / "synthetic-repo"
    root.mkdir()
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study.base, "runtime", lambda: {"synthetic_fixture": True})
    for name in study.SOURCES:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("Synthetic source identity: " + name)
    for row, contents in previous_plans:
        file = root / row["plan_path"]
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(contents)
    parent["sources"] = {name: protocol.sha(root / name) for name in study.recovery.SOURCES}
    parent["runtime"] = study.base.runtime()
    parent_sha = put(root / "parent-plan.json", parent)
    execution = root / "completed-parent"
    execution.mkdir()
    for name in study.INHERITED_FILES:
        (execution / name).write_bytes(b"Synthetic inherited source bytes: " + name.encode())
    for name in parent["fit_order"]:
        folder = execution / "fits" / name
        folder.mkdir(parents=True)
        kind, seed = name.rsplit("-", 1)
        (folder / "initial-weights.pt").write_bytes(b"Paired synthetic initial: " + seed.encode())
        (folder / "weights.pt").write_bytes(b"Synthetic final: " + name.encode())
        put(folder / "training.json", [{"synthetic_fixture": True}])
        put(folder / "completed.json", {"kind": kind, "seed": int(seed), "files": {
            file: protocol.sha(folder / file) for file in ("initial-weights.pt", "weights.pt", "training.json")
        }})
    members = {name: protocol.sha(execution / name) for name in study.inherited_members(parent)}
    completed_sha = put(execution / "completed.json", {"status": "completed", "files": members,
                        "plan_sha256": parent_sha, "wall_seconds": 10.0})
    summary_sha = put(root / "audit" / "summary.json", {"synthetic_fixture": True})
    costs = {"inherited_fit_wall_seconds": 6.0, "cumulative_attempt_wall_seconds": 100.0,
             "new_evaluation_wall_seconds": 10.0}
    audit_sha = put(root / "audit" / "receipt.json", {
        "status": "completed", "version": parent["study"], "plan_sha256": parent_sha,
        "saved_output_only": True, "source_sha256": parent["sources"], "runtime": parent["runtime"],
        "execution_members": members, "execution_completed_sha256": completed_sha,
        "files": {"summary.json": summary_sha}, "costs": costs,
    })
    source = {"plan_path": "parent-plan.json", "plan_sha256": parent_sha,
              "audit_path": "audit/receipt.json", "audit_receipt_sha256": audit_sha,
              "execution_path": "completed-parent", "completed_sha256": completed_sha,
              "summary_sha256": summary_sha, "members": members, "prior_costs": costs}
    result = study.prepare(tmp_path / "protocol", parent_source=source)
    path = Path(result["path"])
    return SimpleNamespace(root=root, source=source, parent=parent, path=path,
                           digest=result["sha256"], plan=protocol.read(path), execution=execution)


def build_complete_synthetic_execution(tmp_path, monkeypatch):
    """Build an engineering-only complete tree for independent auditor tests.

    Only lineage and model construction are injected. Every fresh decision,
    proposal, native transition, diagnostic root, timing and receipt is real.
    The two arms deliberately use random GRUs: this cannot establish efficacy.
    """
    fixture = prepared.__wrapped__(tmp_path, monkeypatch)
    plan, parent = copy.deepcopy(fixture.plan), copy.deepcopy(fixture.parent)
    for settings in (plan, parent):
        settings.update(fit_order=["free-271", "residual-271"], fit_seeds=[271],
                        hidden_size=4, train_episodes=1, control_episodes=1)
    plan.update(study="reacher-search-engineering-whole-tree-v1",
                rng_namespace="reacher-search-engineering-whole-tree-v1",
                diagnostic_episodes=1, diagnostic_branches=1)
    plan["engineering_rng_namespaces"] = [name for name in plan.get("engineering_rng_namespaces", [])
                                         if name != plan["rng_namespace"]]
    source = copy.deepcopy(fixture.source)
    source["plan_sha256"] = put(fixture.root / source["plan_path"], parent)
    generator = torch.Generator().manual_seed(834)
    initial = GRUWorldModel(hidden_size=4).state_dict()
    for name in plan["fit_order"]:
        folder = fixture.execution / "fits" / name
        final = {key: torch.randn(value.shape, generator=generator) * .05
                 for key, value in initial.items()}
        torch.save(initial, folder / "initial-weights.pt")
        torch.save(final, folder / "weights.pt")
        kind, seed = name.rsplit("-", 1)
        put(folder / "completed.json", {"kind": kind, "seed": int(seed), "files": {
            file: protocol.sha(folder / file)
            for file in ("initial-weights.pt", "weights.pt", "training.json")
        }})
    record = study.collect_episode(834, np.ones(51, dtype=bool), noise_std=plan["noise_std"],
                                   noise_seed=835, action_seed=836, policy="mixed")
    study.base.save_records(fixture.execution / "train", [record])
    source["members"] = {name: protocol.sha(fixture.execution / name)
                         for name in study.inherited_members(parent)}
    source["completed_sha256"] = put(fixture.execution / "completed.json", {
        "status": "completed", "files": source["members"],
        "plan_sha256": source["plan_sha256"], "wall_seconds": 10.0,
    })
    receipt = protocol.read(fixture.root / source["audit_path"])
    receipt.update(plan_sha256=source["plan_sha256"], execution_members=source["members"],
                   execution_completed_sha256=source["completed_sha256"])
    source["audit_receipt_sha256"] = put(fixture.root / source["audit_path"], receipt)
    plan["parent_source"] = source
    plan["execution_order"] = study.execution_order(plan)
    plan["random_stream_contract"] = study.stream_contract(plan)
    digest = put(fixture.path, plan)
    monkeypatch.setattr(study, "validate", lambda path, expected: copy.deepcopy(plan))
    monkeypatch.setattr(study.legacy, "model_for", lambda plan, kind, seed: GRUWorldModel(hidden_size=4))
    execution = tmp_path / "complete-engineering-execution"
    study.run(fixture.path, digest, execution)
    put(tmp_path / "synthetic-fixture.json", {
        "scope": "engineering_only_random_weights_no_efficacy", "root": str(fixture.root),
        "execution": str(execution), "plan": str(fixture.path), "plan_sha256": digest,
        "parent": parent,
    })
    return SimpleNamespace(root=fixture.root, execution=execution, parent=parent, plan=plan,
                           path=fixture.path, digest=digest, initial=initial)


def test_prepare_freezes_full_scope_all_six_fits_and_three_prior_registries(prepared):
    plan = study.validate(prepared.path, prepared.digest)
    assert plan["fit_order"] == prepared.parent["fit_order"]
    assert len(plan["parent_source"]["members"]) == 33
    assert len(plan["execution_order"]) == 66
    assert sum("fit" in row for row in plan["execution_order"]) == 54
    assert sum("reference" in row for row in plan["execution_order"]) == 12
    assert plan["diagnostic_episodes"] == 16 and plan["diagnostic_branches"] == 4
    assert plan["control_episodes"] == 64 and plan["steps"] == 50
    assert len(plan["random_stream_contract"]["priors"]) == 3
    assert plan["random_stream_contract"]["priors"][0]["include_train"] is True
    assert plan["competence_criteria"]["memory_qualification"] is False
    assert plan["rng_namespace"] == protocol.SCORED_NAMESPACE
    assert plan["engineering_rng_namespaces"] == list(protocol.ENGINEERING_NAMESPACES)
    contract = plan["random_stream_contract"]
    assert [row["namespace"] for row in contract["engineering_exclusions"]] == plan["engineering_rng_namespaces"]
    current_seeds = set(contract["registry"].values())
    current_states = {value["initial_state_sha256"] for value in contract["generators"].values()}
    for previous in contract["engineering_exclusions"]:
        assert previous["coverage"] == {"control_episodes": 64, "diagnostic_episodes": 16,
                                        "steps": 50, "diagnostic_branches": 4}
        assert len(previous["registry"]) == 1149 and len(previous["generators"]) == 1277
        assert not current_seeds.intersection(previous["registry"].values())
        assert not current_states.intersection(value["initial_state_sha256"]
                                                for value in previous["generators"].values())
    for key, value in prepared.parent.items():
        if key not in study.CHANGED_FIELDS:
            assert plan[key] == value


@pytest.mark.parametrize("key", ["fit_order", "fit_seeds", "kinds", "steps", "control_episodes",
                                   "planning_horizon", "action_block", "noise_std", "ordinary_gap",
                                   "shift_gap", "hidden_size", "cap_seconds", "diagnostic_episodes",
                                   "diagnostic_branches", "planners", "panels", "references",
                                   "execution_order", "search_parameters", "search_criteria",
                                   "competence_criteria", "legacy_seed_fields_unused", "rng_namespace",
                                   "engineering_rng_namespaces"])
def test_rehashing_does_not_allow_scope_or_mechanism_drift(prepared, key):
    plan = copy.deepcopy(prepared.plan)
    old = plan[key]
    if isinstance(old, bool):
        plan[key] = not old
    elif isinstance(old, dict):
        plan[key]["unauthorized"] = True
    elif isinstance(old, list):
        plan[key] = old[:-1]
    elif isinstance(old, str):
        plan[key] = old + "-unauthorized"
    else:
        plan[key] = old + 1
    digest = put(prepared.path, plan)
    with pytest.raises(ValueError):
        study.validate(prepared.path, digest)


def test_external_plan_hash_source_runtime_and_manifest_are_checked(prepared, monkeypatch):
    with pytest.raises(ValueError, match="Identity mismatch"):
        study.validate(prepared.path, "0" * 64)
    manifest = copy.deepcopy(prepared.plan)
    first = next(iter(manifest["random_stream_contract"]["registry"]))
    manifest["random_stream_contract"]["registry"][first] += 1
    digest = put(prepared.path, manifest)
    with pytest.raises(ValueError, match="manifest"):
        study.validate(prepared.path, digest)
    put(prepared.path, prepared.plan)
    monkeypatch.setattr(study.base, "runtime", lambda: {"drifted": True})
    with pytest.raises(ValueError, match="runtime"):
        study.validate(prepared.path, prepared.digest)


@pytest.mark.parametrize("file", ["initial-weights.pt", "weights.pt", "training.json", "completed.json"])
def test_any_changed_inherited_fit_file_rejects_authentication(prepared, file):
    changed = prepared.execution / "fits" / "residual-293" / file
    changed.write_bytes(b"Unexpected changed checkpoint")
    with pytest.raises(ValueError, match="Identity mismatch"):
        study.authenticate_parent(prepared.source)


def test_subset_or_extra_checkpoint_cannot_be_inherited(prepared):
    source = copy.deepcopy(prepared.source)
    source["members"].pop("fits/residual-293/weights.pt")
    with pytest.raises(ValueError, match="All six"):
        study.authenticate_parent(source)
    source = copy.deepcopy(prepared.source)
    source["members"]["fits/residual-293/alternate.pt"] = "0" * 64
    with pytest.raises(ValueError, match="All six"):
        study.authenticate_parent(source)


def test_failed_parent_cannot_supply_completed_lineage(prepared):
    put(prepared.execution / "failed.json", {"status": "failed"})
    with pytest.raises(ValueError, match="must be completed"):
        study.authenticate_parent(prepared.source)


def test_frozen_implementation_bytes_are_checked(prepared):
    source = prepared.root / "src/openjev/research/reacher_search_protocol.py"
    source.write_text("Changed after preparation")
    with pytest.raises(ValueError, match="Identity mismatch"):
        study.validate(prepared.path, prepared.digest)


def test_all_six_load_before_first_fresh_draw_and_failure_is_preserved(prepared, tmp_path, monkeypatch):
    loaded = []

    class SyntheticModel:
        def __init__(self, label):
            self.label = label

        def load_state_dict(self, weights, strict):
            assert weights == {"synthetic": True} and strict
            loaded.append(self.label)

        def requires_grad_(self, flag):
            assert flag is False
            return self

        def eval(self):
            return self

    monkeypatch.setattr(study.legacy, "model_for", lambda plan, kind, seed: SyntheticModel(f"{kind}-{seed}"))
    monkeypatch.setattr(study.torch, "load", lambda path, **kwargs: {"synthetic": True})
    out = tmp_path / "execution"

    def stop_at_first_draw(plan, prefix, count):
        assert loaded == plan["fit_order"]
        ready = protocol.read(out / "inherited-fits-ready.json")
        boundary = protocol.read(out / "evaluation-started.json")
        assert ready["fit_order"] == loaded and ready["new_fits"] == 0
        assert boundary["inherited_fits_ready_sha256"] == protocol.sha(out / "inherited-fits-ready.json")
        assert boundary["random_streams_sha256"] == protocol.sha(out / "random-streams.json")
        assert prefix == "planner/control/0" and count == 64
        raise RuntimeError("Synthetic boundary stop, no draws or learned calls")

    monkeypatch.setattr(protocol, "draw_inputs", stop_at_first_draw)
    with pytest.raises(RuntimeError, match="Synthetic boundary stop"):
        study.run(prepared.path, prepared.digest, out)
    failure = protocol.read(out / "failed.json")
    assert failure["status"] == "failed" and failure["progress"]["phase"] == "fresh-innovations"
    assert not (out / "completed.json").exists()
    assert len(list((out / "fits").glob("*/weights.pt"))) == 6
    with pytest.raises(FileExistsError):
        study.run(prepared.path, prepared.digest, out)


def test_whole_run_cap_includes_initial_validation(prepared, tmp_path, monkeypatch):
    clock = [0.0]
    original = study.validate

    def expensive_validation(path, digest):
        plan = original(path, digest)
        clock[0] = 3600.0
        return plan

    monkeypatch.setattr(study, "validate", expensive_validation)
    monkeypatch.setattr(study.time, "monotonic", lambda: clock[0])
    out = tmp_path / "over-cap"
    with pytest.raises(TimeoutError):
        study.run(prepared.path, prepared.digest, out)
    failure = protocol.read(out / "failed.json")
    assert failure["progress"]["phase"] == "validate" and failure["wall_seconds"] == 3600
    assert not (out / "inherited-fits-ready.json").exists()
    assert not (out / "completed.json").exists()


@pytest.fixture
def engineering_inputs(tmp_path):
    plan = small_protocol()
    plan.update(control_episodes=1, diagnostic_episodes=1, diagnostic_branches=1)
    plan["random_stream_contract"] = protocol.stream_contract(plan)
    stems = []
    for step in range(50):
        prefix = f"planner/control/{step}"
        stem = tmp_path / "engineering-innovations" / f"{step:03d}"
        protocol.save_inputs(stem, protocol.draw_inputs(plan, prefix, 1), prefix)
        stems.append(stem)
    return plan, stems


def test_complete_random_model_and_physics_rows_share_draws_and_pass_saved_native_replay(
        engineering_inputs, tmp_path):
    import audit_reacher_search_study as auditor

    plan, stems = engineering_inputs
    model = TrackingGRU().eval().requires_grad_(False)
    learned = tmp_path / "learned-cem-row"
    reference = tmp_path / "physics-row"
    deadline = time.monotonic() + 120
    learned_timing = study.learned_control(plan, model, "ordinary", "cem256", stems, learned,
                                          deadline, {})
    reference_timing = study.reference_control(plan, "ordinary", "known_state", stems, reference,
                                              deadline, {})
    seen_before_audit = len(model.seen_roots)
    learned_records = auditor.base.load_records(learned / "episodes", 1)
    reference_records = auditor.base.load_records(reference / "episodes", 1)
    for records in (learned_records, reference_records):
        replay = auditor.base.native_replay(records[0])
        assert replay["transitions"] == 50 and replay["max_abs_error"] == 0
        assert replay["new_policy_calls"] == 0 and replay["saved_output_only"] is True
    np.testing.assert_array_equal(learned_records[0]["audit"]["qpos"][0],
                                  reference_records[0]["audit"]["qpos"][0])
    np.testing.assert_array_equal(learned_records[0]["audit"]["actuator_noise"],
                                  reference_records[0]["audit"]["actuator_noise"])
    with np.load(reference / "planning.npz", allow_pickle=False) as saved:
        physics_scores = saved["candidate_scores"].copy()
    total_transitions = 0
    for step in range(50):
        inputs = protocol.load_inputs(stems[step])
        trace = auditor.audit_trace(plan, inputs, learned / "decisions" / f"{step:03d}", step=step)
        result = trace["result"]
        np.testing.assert_array_equal(result.selected_actions[0], learned_records[0]["policy"]["commands"][step])
        bank = study.common_bank(inputs, step, plan)
        np.testing.assert_array_equal(result.sequences[:, :64], bank)
        selected = int(physics_scores[0, step].argmax())
        np.testing.assert_array_equal(bank[0, selected, 0], reference_records[0]["policy"]["commands"][step])
        total_transitions += result.imagined_transitions_per_case
    assert total_transitions == 136704
    assert len(model.seen_roots) == seen_before_audit  # The auditor never reruns the learned model.
    assert learned_timing["observation_assimilations"] == learned_timing["executed_action_advances"] == 50
    for timing in (learned_timing, reference_timing):
        assert len(timing["decision_seconds"]) == len(timing["native_step_seconds"]) == 50
        assert timing["row_wall_seconds"] >= timing["setup_seconds"] + sum(timing["decision_seconds"])


def test_control_failure_preserves_partial_native_records_and_does_not_claim_completion(
        engineering_inputs, tmp_path, monkeypatch):
    plan, stems = engineering_inputs
    original = study.score_search

    def capped(model, state, inputs, method, step, plan, deadline):
        if step == 1:
            raise TimeoutError("Synthetic cooperative cap")
        return original(model, state, inputs, method, step, plan, deadline)

    monkeypatch.setattr(study, "score_search", capped)
    out = tmp_path / "partial-row"
    with pytest.raises(TimeoutError, match="cooperative cap"):
        study.learned_control(plan, GRUWorldModel(hidden_size=4).eval(), "ordinary", "rs64", stems,
                              out, float("inf"), {})
    assert (out / "partial-episodes.npz").exists() and (out / "decisions/000.npz").exists()
    assert not (out / "episodes.npz").exists() and not (out / "timings.json").exists()
    with np.load(out / "partial-episodes.npz", allow_pickle=False) as saved:
        assert saved["policy__commands"].shape == (1, 1, 2)
        assert saved["audit__integration_state"].shape[1] == 2


def test_mid_batch_native_failure_preserves_unequal_histories_and_closes_every_case(
        tmp_path, monkeypatch):
    plan = small_protocol()
    plan["control_episodes"] = 2
    plan["random_stream_contract"] = protocol.stream_contract(plan)
    stem = tmp_path / "engineering-innovations" / "000"
    prefix = "planner/control/0"
    protocol.save_inputs(stem, protocol.draw_inputs(plan, prefix, 2), prefix)
    original = study.control_envs
    closed = []

    def environments(plan, panel, deadline):
        envs, packets = original(plan, panel, deadline)
        for index, env in enumerate(envs):
            close = env.close

            def tracked_close(index=index, close=close):
                closed.append(index)
                close()

            monkeypatch.setattr(env, "close", tracked_close)

        def fail_step(action):
            raise RuntimeError("Synthetic second-case native failure")

        monkeypatch.setattr(envs[1], "step", fail_step)
        return envs, packets

    monkeypatch.setattr(study, "control_envs", environments)
    out = tmp_path / "ragged-partial-row"
    with pytest.raises(RuntimeError, match="second-case native failure"):
        study.learned_control(plan, GRUWorldModel(hidden_size=4).eval(), "ordinary", "rs64",
                              [stem], out, float("inf"), {})
    partial = out / "partial-episodes"
    assert protocol.read(partial / "manifest.json") == {"completed_steps_by_case": [1, 0]}
    for index, length in enumerate((1, 0)):
        with np.load(partial / f"{index:03d}.npz", allow_pickle=False) as saved:
            assert saved["policy__commands"].shape[1] == length
            assert saved["audit__integration_state"].shape[1] == length + 1
    assert (out / "decisions/000.npz").exists()
    assert not (out / "episodes.npz").exists() and not (out / "timings.json").exists()
    assert closed == [0, 1]


@pytest.fixture
def terminal_root():
    episode = ReacherEpisode(noise_std=0.0)
    try:
        episode.reset(834, np.ones(51, dtype=bool), noise_seed=835)
        for _ in range(47):
            episode.step(np.zeros(2, dtype=np.float32))
        yield episode.audit_record()
    finally:
        episode.close()


def test_native_branch_restores_complete_state_time_limit_and_explicit_noise(terminal_root):
    commands = np.array([[0.95, -0.95], [0.15, -0.2], [0.3, 0.1]], dtype=np.float32)
    noise = np.array([[0.2, -0.3], [-0.04, 0.08], [0.0, 0.02]], dtype=np.float64)
    template = make_env()
    try:
        template.reset(seed=837)
        first = study.native_branch(terminal_root, commands, noise, float("inf"), env=template)
        second = study.native_branch(terminal_root, commands, noise, float("inf"), env=template)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first["integration_state"][0], terminal_root["integration_state"])
        np.testing.assert_array_equal(first["applied_actions"], np.clip(commands.astype(float) + noise, -1, 1))
        assert first["truncated"].tolist() == [False, False, True]
        assert not first["terminated"].any() and template._elapsed_steps == 50
        assert len(first["integration_state"]) == 4 and len(first["rewards"]) == 3
        np.testing.assert_allclose(first["rewards"], first["reward_dist"] + first["reward_ctrl"], atol=0, rtol=0)
        with pytest.raises(ValueError, match="time boundary"):
            study.native_branch(terminal_root, np.zeros((4, 2), dtype=np.float32),
                                np.zeros((4, 2), dtype=np.float64), float("inf"), env=template)
        with pytest.raises(TimeoutError):
            study.native_branch(terminal_root, commands, noise, -float("inf"), env=template)
    finally:
        template.close()


def test_role_manifest_covers_every_generator_and_draws_nothing():
    before = np.random.get_state()
    plan = small_protocol()
    contract = plan["random_stream_contract"]
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert contract["draws_for_manifest"] == 0 and contract["full_horizon_innovations"] is True
    assert len(contract["registry"]) == 345
    assert len(contract["generators"]) == 351
    assert len(set(contract["registry"].values())) == 345
    assert len({v["initial_state_sha256"] for v in contract["generators"].values()}) == 351
    for branch in range(4):
        assert f"diagnostic/branch_noise/1/3/{branch}" in contract["registry"]
    for child in ("initial", "process_noise", "resample"):
        assert f"planner/particle_filter/2/{child}" in contract["generators"]


def test_role_contract_rejects_aliases_inside_run_and_with_any_prior(monkeypatch):
    plan = small_protocol()
    old_seed = next(iter(plan["random_stream_contract"]["registry"].values()))
    with pytest.raises(ValueError, match="Prior root seed collision"):
        protocol.stream_contract(plan, prior_registries=({"old/actual-offset-role": old_seed},))
    monkeypatch.setattr(protocol, "named_seed", lambda namespace, role: 42)
    with pytest.raises(ValueError, match="Prospective root seed collision"):
        protocol.stream_contract(plan)


def test_engineering_namespace_cannot_be_reused_as_a_scored_namespace():
    plan = small_protocol()
    plan["engineering_rng_namespaces"] = [plan["rng_namespace"]]
    with pytest.raises(ValueError, match="Engineering root seed collision"):
        protocol.stream_contract(plan)
    plan["rng_namespace"] = " "
    with pytest.raises(ValueError, match="explicit RNG namespace"):
        protocol.registry(plan)


def test_manifest_rejects_equal_generator_states_even_when_integer_seeds_differ(monkeypatch):
    plan = small_protocol()
    monkeypatch.setattr(protocol, "generator_manifest", lambda registry: {
        name: {"initial_state_sha256": "0" * 64} for name in registry
    })
    with pytest.raises(ValueError, match="generator-state collision"):
        protocol.stream_contract(plan)


def test_search_draws_roundtrip_with_full_horizon_even_near_termination(tmp_path):
    plan = small_protocol()
    inputs = protocol.draw_inputs(plan, "planner/control/47", count=3)
    assert inputs.initial.shape == (3, 64, 4, 2)
    assert inputs.random_extra.shape == (3, 192, 4, 2)
    assert [v.shape for v in inputs.cem] == [(3, 64, 4, 2), (3, 64, 4, 2), (3, 63, 4, 2)]
    stem = tmp_path / "draws"
    protocol.save_inputs(stem, inputs, "planner/control/47")
    restored = protocol.load_inputs(stem)
    assert restored.identities() == inputs.identities()
    assert protocol.read(stem.with_suffix(".json"))["unused_anchor_draws"] == 7
    changed = protocol.draw_inputs(plan, "planner/control/46", count=3)
    assert changed.identities() != inputs.identities()
    with pytest.raises(FileExistsError):
        protocol.save_inputs(stem, inputs, "planner/control/47")


def test_diagnostic_roots_and_sensor_panels_share_phase_but_preserve_shift():
    plan = small_protocol()
    ordinary = protocol.schedule(plan, 0, "ordinary", split="diagnostic")
    shifted = protocol.schedule(plan, 0, "shift", split="diagnostic")
    full = protocol.schedule(plan, 0, "full", split="diagnostic")
    phase = protocol.phase(plan, 0, split="diagnostic")
    assert full.all() and (~ordinary).sum() == 12 and (~shifted).sum() == 20
    assert np.flatnonzero(~ordinary)[0] == np.flatnonzero(~shifted)[0] == 8 + phase
    roots = protocol.root_steps(plan, 0)
    assert roots == (6, 10 + phase, 14 + phase, 47)
    assert not ordinary[roots[1]] and ordinary[roots[2]]
    assert not shifted[roots[2]]  # Ordinary recovery remains inside the longer shifted blackout.


def test_public_history_has_no_future_or_unobserved_angle_or_velocity_dependency():
    plan = small_protocol()
    valid = protocol.schedule(plan, 0, "ordinary", split="diagnostic")
    root = protocol.root_steps(plan, 0)[2]
    qpos = np.arange(51 * 4, dtype=float).reshape(51, 4) / 500
    qpos[:, 2:4] = [0.1, -0.1]
    record = {"audit": {"qpos": qpos, "qvel": np.zeros((51, 4))},
              "policy": {"commands": np.zeros((50, 2), dtype=np.float32)}, "metadata": {"dt": 0.02}}
    expected = protocol.public_history(record, valid, root)
    changed = copy.deepcopy(record)
    changed["audit"]["qpos"][~valid, :2] += 100
    changed["audit"]["qpos"][root + 1:] += 1000
    changed["audit"]["qvel"][:] = 999
    changed["policy"]["commands"][root:] = 1
    actual = protocol.public_history(changed, valid, root)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["packets"].shape == (root + 1, 8)
    assert actual["commands"].shape == (root, 2)
    assert not actual["packets"][~valid[:root + 1], :4].any()


def test_saved_full_cem_trace_reconstructs_all_paid_proposals(tmp_path):
    import audit_reacher_search_study as auditor

    raw_rewards = []
    sizes = []

    def score(bank):
        raw = -np.square(bank).sum(-1)
        raw_rewards.append(raw)
        sizes.append(bank.shape[1])
        total = np.zeros(bank.shape[:2], dtype=np.float32)
        for step in range(bank.shape[2]):
            total += np.clip(raw[:, :, step], -2.5, 0)
        return total

    plan = small_protocol()
    inputs = innovations()
    result = search("cem256", inputs, score, step=47)
    stem = tmp_path / "trace"
    protocol.save_trace(stem, result, np.concatenate(raw_rewards, 1), sizes, 0.001)
    checked = auditor.audit_trace(plan, inputs, stem, step=47)
    np.testing.assert_array_equal(checked["result"].selected_sequences, result.selected_sequences)
    assert checked["result"].candidate_ids[-1] == "cem/3/mean"
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as saved:
        assert saved["chunks"].shape == (2, 256, 1, 2)
        assert saved["raw_rewards"].shape == (2, 256, 3)


class TrackingGRU(GRUWorldModel):
    def __init__(self):
        super().__init__(hidden_size=4)
        self.seen_roots = []

    def advance(self, state, action):
        self.seen_roots.append({key: value.detach().clone() for key, value in state.items()})
        return super().advance(state, action)


def root_state(model, cases=2):
    packet = torch.zeros(cases, 8)
    packet[:, :2] = 1
    packet[:, 4:6] = torch.tensor([0.12, -0.05])
    packet[:, 6] = 1
    return model.assimilate(model.initial(cases), packet)


@pytest.mark.parametrize("method,sizes", [("rs64", [64]), ("rs256", [256]),
                                         ("cem256", [64, 64, 64, 64])])
@pytest.mark.parametrize("step,horizon", [(0, 12), (47, 3), (49, 1)])
def test_every_score_callback_starts_from_same_unmodified_belief(method, sizes, step, horizon):
    model = TrackingGRU().eval()
    state = root_state(model)
    before = {key: value.clone() for key, value in state.items()}
    result, _, callback_sizes, _ = study.score_search(model, state, innovations(), method, step,
                                                      synthetic_plan(), float("inf"))
    assert result.horizon == horizon
    assert len(model.seen_roots) == len(sizes) * horizon
    for callback, size in enumerate(sizes):
        first = model.seen_roots[callback * horizon]
        for key, value in before.items():
            torch.testing.assert_close(first[key], value.repeat_interleave(size, 0), rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(state[key], before[key], rtol=0, atol=0)
    assert result.candidate_evaluations_per_case == sum(sizes)
    assert result.imagined_transitions_per_case == sum(sizes) * horizon
    assert callback_sizes == sizes


def test_score_search_cap_and_terminal_boundary_precede_model_calls():
    model = TrackingGRU().eval()
    state = root_state(model)
    with pytest.raises(TimeoutError):
        study.score_search(model, state, innovations(), "cem256", 0, synthetic_plan(), -float("inf"))
    assert not model.seen_roots
    with pytest.raises(ValueError):
        study.score_search(model, state, innovations(), "cem256", 50, synthetic_plan(), float("inf"))
    assert not model.seen_roots


def test_all_methods_preserve_the_common_first_bank_and_scores():
    model = GRUWorldModel(hidden_size=4).eval()
    state = root_state(model)
    draws = innovations()
    results = [study.score_search(model, state, draws, method, 47, synthetic_plan(), float("inf"))[0]
               for method in ("rs64", "rs256", "cem256")]
    for result in results[1:]:
        np.testing.assert_array_equal(result.sequences[:, :64], results[0].sequences)
        np.testing.assert_allclose(result.scores[:, :64], results[0].scores, rtol=1e-6, atol=1e-6)
    assert results[-1].candidate_ids[-1] == "cem/3/mean"
    assert results[-1].stages[-1].mean_candidate_id == 255


def test_scorer_applies_physical_reward_clipping_before_summing():
    class CommandRewardGRU(GRUWorldModel):
        def advance(self, state, action):
            next_state, angles, _ = super().advance(state, action)
            return next_state, angles, 12 * action[:, 0] - 0.7

    model = CommandRewardGRU(hidden_size=4).eval()
    result, _, _, _ = study.score_search(model, root_state(model), innovations(), "cem256", 47,
                                        synthetic_plan(), float("inf"))
    raw = 12 * result.sequences[..., 0] - 0.7
    expected = np.clip(raw, -2.5, 0).sum(-1)
    np.testing.assert_allclose(result.scores, expected, rtol=1e-6, atol=1e-6)
    assert np.any(raw > 0) and np.any(raw < -2.5)


def test_model_outputs_cannot_be_nonfinite_even_if_clipping_would_hide_infinity():
    class NonfiniteRewardGRU(GRUWorldModel):
        def advance(self, state, action):
            next_state, angles, reward = super().advance(state, action)
            return next_state, angles, reward + float("inf")

    model = NonfiniteRewardGRU(hidden_size=4).eval()
    with pytest.raises(ValueError, match="finite"):
        study.score_search(model, root_state(model), innovations(), "rs64", 47,
                           synthetic_plan(), float("inf"))
