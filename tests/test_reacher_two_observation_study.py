"""Fake orchestration only: no model, optimizer, environment or seed draws."""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest
import reacher_two_observation_study as study
import torch

ROOT_FILES = {"started.json", "random-streams.json", "inheritance.json", "training-inputs.json",
    "inherited-models-restored.json", "training-started.json", "all-fits-completed.json",
    "all-models-restored.json", "evaluation-started.json", "control-completed.json", "final-models.json",
    "costs.json", "results.json"}


def expected_members(settings):
    names = set(ROOT_FILES)
    names.update("inherited/" + name for name in study.protocol.inherited_members(settings))
    for label, suffixes in (("cache", ("plan", "audit", "completed", "summary")),
                           ("prerequisite", ("plan", "audit", "completed", "summary", "terminal"))):
        names.update(f"inherited/lineage/{label}-{suffix}.json" for suffix in suffixes)
    for row in study.protocol.fit_manifest(settings):
        if row["new_fit"]:
            names.update(f"fits/{row['name']}/{member}" for member in study.protocol.FIT_MEMBERS)
        else:
            names.add(f"model-states/{row['name']}-prefit.pt")
        names.update(f"model-states/{row['name']}-{phase}.pt" for phase in ("before", "after"))
    names.update(f"innovations/control/{t:03d}.{suffix}" for t in range(50) for suffix in ("json", "npz"))
    for row in study.protocol.execution_order(settings):
        path = row["path"]
        names.update(f"{path}/{member}" for member in ("episodes.npz", "episodes.json", "timings.json"))
        if "fit" in row:
            names.update(f"{path}/{member}" for member in ("states.npz", "state-work.json", "executed_predictions.npz"))
            names.update(f"{path}/{kind}/{t:03d}.{suffix}" for kind in ("decisions", "scoring")
                for t in range(50) for suffix in ("json", "npz"))
            if row["arm"] == "two_observation_gru":
                names.update(f"{path}/{member}" for member in ("started.json", "inputs.json", "completed.json"))
                names.update(f"{path}/controller-decisions/{t:03d}.json" for t in range(50))
        else:
            names.add(f"{path}/planning.npz")
            if row["reference"] in study.protocol.PHYSICS_REFERENCES:
                names.add(f"{path}/physics-final.json")
                names.update(f"{path}/{kind}/{t:03d}.{suffix}" for kind in ("decisions", "physics")
                    for t in range(50) for suffix in ("json", "npz"))
                if row["reference"] != "known_state":
                    names.add(f"{path}/observer-final.json")
    return names


def touch(path, value=b"synthetic artifact, not a checkpoint or native trace\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def fake_model(value=1.):
    state = {"fake": torch.tensor([value], dtype=torch.float32)}
    return SimpleNamespace(state_dict=lambda: state)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    old_threads, old_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    h = SimpleNamespace(events=[], fail_fit=None, fail_row=None, fail_stage=None, validation_calls=0)
    settings = study.protocol.settings(engineering=True)
    settings.update(hidden_size=4, train_episodes=3, epochs=2, batch_size=2, control_episodes=1, threads=1)
    h.settings = settings
    h.context = SimpleNamespace(settings=settings, streams=object(), cache_plan={},
        cache_source={"prior_costs": {"cumulative_attempt_wall_seconds": 10.}},
        prerequisite_source={"prior_costs": {"cumulative_attempt_wall_seconds": 20.}})
    h.plan = {"settings": settings, "engineering": True, "cap_seconds": 1000,
        "audit_cap_seconds": 1000, "random_stream_contract": {"explicit_fake": True},
        "sources": {"fake": "a" * 64}, "runtime": {"scope": "fake_no_model_or_native"},
        "lineage": {"cache": {"plan_sha256": "b" * 64, "audit_receipt_sha256": "c" * 64,
            "members": {name: "d" * 64 for name in study.protocol.inherited_members(settings)}}}}
    h.out, h.expected = tmp_path / "attempt", "e" * 64
    h.names = expected_members(settings)
    h.models = {row["name"]: fake_model(i + 1.) for i, row in enumerate(study.protocol.fit_manifest(settings))}
    h.data = {"fake_public": torch.tensor([1.], dtype=torch.float32)}
    h.initials = {pair: {"states": {"gru": {"fake": torch.tensor([.1])}}} for pair in study.protocol.PAIRS}
    h.orders = {pair: {"whole_sealed_payload": pair} for pair in study.protocol.PAIRS}

    def validate():
        h.validation_calls += 1
        h.events.append("validate")
        if h.fail_stage == "validation" or h.fail_stage == "final" and h.validation_calls == 2:
            raise KeyboardInterrupt("synthetic validation failure")
        return h.plan, h.context

    def copied(plan, context, out, deadline):
        h.events.append("copy")
        for name in h.names:
            if name.startswith("inherited/"):
                touch(out / name)
        study.write(out / "inheritance.json", {"synthetic": True})

    def load_inputs(plan, context, out, deadline):
        h.events.append("inputs")
        receipt = {"data_sha256": study.training.canonical_tensor_hash(h.data), "wall_seconds": 0.,
            "pairs": {pair: {"initial_tensor_sha256": study.training.canonical_tensor_hash(h.initials[pair]["states"]["gru"]),
                "orders_sha256": study.training.canonical_state_hash(h.orders[pair])} for pair in study.protocol.PAIRS}}
        study.write(out / "training-inputs.json", receipt)
        return h.data, h.initials, h.orders, receipt

    def restore(plan, expected, context, out, deadline, inputs, *, stage):
        h.events.append("restore:" + stage)
        if h.fail_stage == "restore:" + stage:
            raise KeyboardInterrupt("synthetic restore failure")
        rows = [row for row in study.protocol.fit_manifest(settings) if stage == "before" or not row["new_fit"]]
        if stage == "before":
            assert [event for event in h.events if event.startswith("fit:")] == ["fit:pair0", "fit:pair1", "fit:pair2"]
            assert (out / "all-fits-completed.json").exists()
        bindings = {}
        for row in rows:
            model = h.models[row["name"]]
            prefix = "fits" if row["new_fit"] else "inherited/fits"
            checkpoint = out / prefix / row["name"] / "checkpoint.pt"
            weights = checkpoint.with_name("weights.pt")
            snapshot = out / "model-states" / f"{row['name']}-{stage}.pt"
            touch(snapshot)
            bindings[row["name"]] = {"student_tensor_sha256": study.training.canonical_tensor_hash(model.state_dict()),
                "checkpoint_path": str(checkpoint.relative_to(out)), "checkpoint_sha256": study.sha(checkpoint),
                "weights_sha256": study.sha(weights)}
        receipt = {"models": bindings, "wall_seconds": 0.}
        study.write(out / ("inherited-models-restored.json" if stage == "prefit" else "all-models-restored.json"), receipt)
        return {row["name"]: h.models[row["name"]] for row in rows}, receipt

    def fit(initial, orders, data, out, **kwargs):
        pair = kwargs["pair"]
        h.events.append("fit:" + pair)
        assert "restore:prefit" in h.events and "draw" not in h.events
        assert initial is h.initials[pair]["states"]["gru"] and orders is h.orders[pair] and data is h.data
        assert kwargs["expected_orders_sha256"] == study.training.canonical_state_hash(orders)
        assert kwargs["source_sha256"] == h.plan["sources"] and kwargs["runtime"] == h.plan["runtime"]
        assert kwargs["provenance"]["pair"] == pair
        out.mkdir(parents=True)
        if h.fail_fit == pair:
            study.write(out / "failed.json", {"optimizer_steps": 1, "status": "failed"})
            raise KeyboardInterrupt("synthetic partial fit")
        for name in study.fitting.PAYLOAD_FILES:
            touch(out / name)
        receipt = {"name": kwargs["name"], "optimizer_steps": 4, "wall_seconds": 0.}
        study.write(out / "completed.json", receipt)
        return receipt

    def draw(settings, step, *, contract):
        assert contract is h.context.streams
        assert (h.out / "all-fits-completed.json").exists() and (h.out / "all-models-restored.json").exists()
        assert (h.out / "evaluation-started.json").exists()
        h.events.append("draw")
        return step

    def save_input(stem, value, role):
        touch(stem.with_suffix(".npz"))
        touch(stem.with_suffix(".json"))

    def row(settings, model_or_ref, panel, inputs, out, deadline, progress, *, cases, bound=None):
        relative = str(out.relative_to(h.out))
        h.events.append("row:" + relative)
        assert h.events.count("draw") == 50 and "restore:before" in h.events and len(inputs) == 50
        assert cases == ["explicit_public_case"]
        if h.fail_row == relative:
            touch(out / "partial-episodes.json")
            raise KeyboardInterrupt("synthetic native/controller failure")
        for name in h.names:
            if name.startswith(relative + "/"):
                touch(h.out / name)
        return {"row_wall_seconds": 0., "setup_seconds": 0., "decision_seconds": [0.] * 50,
            "native_step_seconds": [0.] * 50}

    def load_records(path):
        return [{"audit": {"rewards": -np.ones(50, dtype=np.float64)}}]

    h.validate = validate
    monkeypatch.setattr(study, "copy_inherited", copied)
    monkeypatch.setattr(study, "load_training_inputs", load_inputs)
    monkeypatch.setattr(study, "restore_students", restore)
    monkeypatch.setattr(study.fitting, "fit_one", fit)
    monkeypatch.setattr(study.streams, "draw_control_inputs", draw)
    monkeypatch.setattr(study.artifacts, "save_inputs", save_input)
    monkeypatch.setattr(study, "control_cases", lambda *args: ["explicit_public_case"])
    monkeypatch.setattr(study.control, "learned_control", row)
    monkeypatch.setattr(study.episode, "learned_control", row)
    monkeypatch.setattr(study, "reference_control", row)
    monkeypatch.setattr(study.base, "load_records", load_records)
    monkeypatch.setattr(study, "_experiment", lambda: SimpleNamespace(expected_members=expected_members))
    yield h
    torch.set_num_threads(old_threads)
    torch.use_deterministic_algorithms(old_deterministic)


def run_harness(h):
    return study.run_validated(h.plan, h.expected, h.out, h.context, final_validate=h.validate)


def test_full_fake_lifecycle_all_fits_all_rows_boundaries_and_members(harness):
    h = harness
    done = run_harness(h)
    assert done["new_fits"] == 3 and done["restored_models"] == 9 and done["control_rows"] == 42
    assert done["new_optimizer_steps"] == 12 and h.validation_calls == 2
    assert h.events.index("restore:prefit") < h.events.index("fit:pair0") < h.events.index("restore:before") < h.events.index("draw")
    assert h.events.count("draw") == 50
    assert len([event for event in h.events if event.startswith("row:")]) == 42
    assert set(done["files"]) == h.names and len(done["files"]) == 8119
    assert set(study.file_members(h.out)) == h.names | {"completed.json"}
    assert all(study.sha(h.out / name) == digest for name, digest in done["files"].items())
    costs = study.read(h.out / "costs.json")
    assert costs["new_optimizer_steps"] == 12 and costs["coverage"]["control_rows"] == 42
    assert done["cumulative_attempt_wall_seconds"] == 20. + done["wall_seconds"]
    assert not study.read(h.out / "results.json")["continuation_gate"]["passed"]


@pytest.mark.parametrize("failure", ["validation", "restore:prefit", "restore:before", "final"])
def test_failure_preserves_baseexception_no_completion_or_retry(harness, failure):
    h = harness
    h.fail_stage = failure
    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        run_harness(h)
    failed = study.read(h.out / "failed.json")
    assert failed["exception_type"] == "KeyboardInterrupt" and not (h.out / "completed.json").exists()
    if failure != "final":
        assert "draw" not in h.events
    with pytest.raises(FileExistsError):
        run_harness(h)


def test_fit_failure_prevents_later_fit_and_all_fresh_draws(harness):
    h = harness
    h.fail_fit = "pair1"
    with pytest.raises(KeyboardInterrupt, match="partial fit"):
        run_harness(h)
    assert "fit:pair2" not in h.events and "draw" not in h.events
    assert not (h.out / "all-fits-completed.json").exists()
    assert study.read(h.out / "fits/two_observation_gru-pair1/failed.json")["optimizer_steps"] == 1


def test_row_failure_keeps_partial_and_stops_next_row(harness):
    h = harness
    path = study.protocol.execution_order(h.settings)[1]["path"]
    h.fail_row = path
    with pytest.raises(KeyboardInterrupt, match="controller failure"):
        run_harness(h)
    assert (h.out / path / "partial-episodes.json").exists()
    assert len([event for event in h.events if event.startswith("row:")]) == 2
    assert not (h.out / "control-completed.json").exists()


def test_missing_manifest_member_prevents_success(harness, monkeypatch):
    h = harness
    expected = h.names | {"required-but-missing.json"}
    monkeypatch.setattr(study, "_experiment", lambda: SimpleNamespace(expected_members=lambda _: expected))
    with pytest.raises(ValueError, match="artifact membership"):
        run_harness(h)
    assert not (h.out / "completed.json").exists()


def test_weight_mutation_during_controls_prevents_success(harness, monkeypatch):
    h = harness
    original = study.control.learned_control
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        args[1].state_dict()["fake"].add_(1.)
        return result
    monkeypatch.setattr(study.control, "learned_control", mutate)
    with pytest.raises(ValueError, match="Evaluation changed"):
        run_harness(h)


def test_cap_after_completion_demotes_success_marker(harness, monkeypatch):
    h = harness
    original = study.check_cap
    def cap(deadline):
        if (h.out / "completed.json").exists():
            raise TimeoutError("synthetic post-write cap")
        original(deadline)
    monkeypatch.setattr(study, "check_cap", cap)
    with pytest.raises(TimeoutError, match="post-write"):
        run_harness(h)
    assert (h.out / "invalid-completion.json").is_file() and (h.out / "failed.json").is_file()
    assert not (h.out / "completed.json").exists()


def test_cap_before_any_fitting_or_draw(harness):
    h = harness
    with pytest.raises(TimeoutError):
        study.run_validated(h.plan, h.expected, h.out, h.context,
            begin=time.monotonic() - 2000, final_validate=h.validate)
    assert h.events == ["validate"] and (h.out / "failed.json").exists()


def test_cli_preflight_failure_is_exclusive_and_preserved(tmp_path, monkeypatch):
    out = tmp_path / "attempt"
    def bad(*args, **kwargs):
        raise KeyboardInterrupt("explicit authentication failure")
    monkeypatch.setattr(study, "validate", bad)
    with pytest.raises(KeyboardInterrupt):
        study.run(tmp_path / "plan.json", "a" * 64, out, engineering=True)
    assert study.read(out / "failed.json")["progress"]["phase"] == "validate-or-setup"
    with pytest.raises(FileExistsError):
        study.run(tmp_path / "plan.json", "a" * 64, out)


def test_failure_preservation_error_does_not_replace_original(harness, monkeypatch):
    h = harness
    h.fail_stage = "validation"
    original = study.write
    def bad(path, value):
        if path.name == "failed.json":
            raise OSError("synthetic full disk")
        original(path, value)
    monkeypatch.setattr(study, "write", bad)
    with pytest.raises(KeyboardInterrupt, match="validation failure") as caught:
        run_harness(h)
    assert any("full disk" in note for note in caught.value.__notes__)


def test_plan_drift_before_fitting_rejected(harness):
    h = harness
    changed = copy.deepcopy(h.plan)
    changed["runtime"]["scope"] = "changed"
    with pytest.raises(ValueError, match="Plan identity"):
        study.run_validated(h.plan, h.expected, h.out, h.context,
            final_validate=lambda: (changed, h.context))
    assert h.events == []


def test_external_order_binding_uses_whole_payload(harness):
    h = harness
    row = study.protocol.training_manifest(h.settings)[0]
    binding = study.fit_provenance(h.plan, h.expected, row)
    assert binding["pair"] == "pair0" and binding["plan_sha256"] == h.expected
    assert set(binding["original_members"]) == {"initializations/pair0.pt", "orders/pair0.pt", "train.npz", "train.json"}


@pytest.fixture
def reference_fake(tmp_path, monkeypatch):
    h = SimpleNamespace(envs=[], observers=[], roots=[], audit_reads=0, seed_roles=[], close_calls=0,
                        noise=None, corrupt_command=False, fail_step=None, allow_audit=False, bound=object())
    h.settings = study.protocol.settings(engineering=True)
    h.settings.update(control_episodes=1)
    h.out = tmp_path / "reference"
    packet = np.array([1, 1, 0, 0, .1, .1, 1, 0], dtype=np.float32)

    class Env:
        dt = .02
        def __init__(self, *, noise_std):
            assert noise_std == .05
            self.step_index, self.finished, self.commands = 0, False, []
            h.envs.append(self)
        def reset(self, seed, schedule, *, noise_seed):
            assert seed == 410 and noise_seed == 410 and len(schedule) == 51
            return packet.copy()
        def step(self, action):
            if self.step_index == h.fail_step:
                raise KeyboardInterrupt("fake native failure")
            self.commands.append(action.copy() + (.1 if h.corrupt_command else 0.))
            self.step_index += 1
            self.finished = self.step_index == 50
            return packet.copy()
        def episode_record(self):
            return {"policy": {"commands": np.array(self.commands, dtype=np.float32)}}
        def audit_record(self):
            assert h.allow_audit, "Unprivileged reference accessed actual state"
            h.audit_reads += 1
            return {"qpos": np.array([0., 0., .1, .1]), "qvel": np.zeros(4)}
        def close(self):
            h.close_calls += 1

    class Observer:
        def __init__(self, model, observed, **kwargs):
            assert np.array_equal(observed, packet)
            self.updates = []
            h.observers.append(self)
        def update(self, issued, observed):
            assert np.array_equal(observed, packet) and np.array_equal(issued, np.zeros(2, dtype=np.float32))
            self.updates.append(issued.copy())
        def estimate(self):
            return np.array([0., 0., .1, .1]), np.zeros(4)
        def snapshot(self):
            return {"updates": len(self.updates), "public_inputs_only": True}

    class Physics:
        def __init__(self, model, **kwargs):
            assert kwargs["planning_horizon"] == 12 and kwargs["action_block"] == 3
            self.last = None
        def plan(self, qpos, qvel, target, inputs, *, step, deadline):
            assert inputs == "explicit_inputs" and np.array_equal(target, packet[None, 4:6])
            h.roots.append((qpos.copy(), qvel.copy(), target.copy()))
            result = SimpleNamespace(selected_actions=np.zeros((1, 2), np.float32), scores=np.zeros((1, 256)))
            self.last = {"roots": {"qpos": qpos, "qvel": qvel, "public_target": target},
                "banks": [{"arrays": {"geometry_reward": np.zeros((1, 64, min(12, 50-step)), np.float32)}} for _ in range(4)],
                "selected": None, "wall_seconds": 0.}
            return result, self.last
        def snapshot(self):
            return {"last_operation": self.last, "synthetic_only": True}

    def seed(settings, role, *, contract):
        assert contract is h.bound
        h.seed_roles.append(role)
        return 410
    monkeypatch.setattr(study, "ReacherEpisode", Env)
    monkeypatch.setattr(study, "make_env", lambda: SimpleNamespace(unwrapped=SimpleNamespace(model=object()),
        close=lambda: setattr(h, "close_calls", h.close_calls + 1)))
    monkeypatch.setattr(study, "ParticleFilter", Observer)
    monkeypatch.setattr(study, "KinematicObserver", Observer)
    monkeypatch.setattr(study, "PhysicsGeometryCEM", Physics)
    monkeypatch.setattr(study.streams, "seed", seed)
    monkeypatch.setattr(study.artifacts, "load_inputs", lambda stem: "explicit_inputs")
    monkeypatch.setattr(study.artifacts, "save_trace", lambda stem, *args: touch(stem.with_suffix(".json")))
    monkeypatch.setattr(study.control, "_save_records", lambda stem, rows: touch(stem.with_suffix(".json")))
    monkeypatch.setattr(study.control, "_save_partial", lambda stem, envs: touch(stem.with_suffix(".json")))
    # No generator is allocated: this stand-in proves explicit role routing only.
    monkeypatch.setattr(study.np.random, "default_rng", lambda seed: SimpleNamespace(
        uniform=lambda lo, hi, shape: np.zeros(shape, dtype=np.float64)))
    h.cases = [study.control.ControlCase(410, 410, np.ones(51, dtype=np.bool_))]
    return h


@pytest.mark.parametrize("reference", study.protocol.REFERENCES)
def test_reference_wrapper_public_privileged_boundaries_and_actual_issued_commands(reference_fake, reference):
    h = reference_fake
    h.allow_audit = reference == "known_state"
    timing = study.reference_control(h.settings, "ordinary", reference, ["input"] * 50,
        h.out, cases=h.cases, bound=h.bound)
    assert h.envs[0].step_index == 50 and len(h.envs[0].commands) == 50
    assert h.audit_reads == (50 if reference == "known_state" else 0)
    assert timing["privileged_state_reads"] == h.audit_reads
    assert h.close_calls == (2 if reference in study.protocol.PHYSICS_REFERENCES else 1)
    assert len(h.roots) == (50 if reference in study.protocol.PHYSICS_REFERENCES else 0)
    if reference in ("particle", "public_kinematic"):
        assert len(h.observers[0].updates) == 49
    expected = ["planner/particle_filter/0"] if reference == "particle" else ["floor/uniform/0"] if reference == "uniform" else []
    assert h.seed_roles == expected


def test_reference_rejects_native_command_disagreement(reference_fake):
    h = reference_fake
    h.corrupt_command = True
    with pytest.raises(ValueError, match="issued commands"):
        study.reference_control(h.settings, "ordinary", "zero", ["input"] * 50,
            h.out, cases=h.cases, bound=h.bound)
    assert h.envs[0].step_index == 1 and h.close_calls == 1
    assert study.read(h.out / "failed.json")["completed_steps_by_case"] == [1]


def test_reference_failure_preserves_paid_physics_prefix_and_closes(reference_fake):
    h = reference_fake
    h.fail_step = 1
    with pytest.raises(KeyboardInterrupt, match="fake native"):
        study.reference_control(h.settings, "ordinary", "particle", ["input"] * 50,
            h.out, cases=h.cases, bound=h.bound)
    assert h.envs[0].step_index == 1 and len(h.roots) == 2 and h.close_calls == 2
    assert (h.out / "partial-episodes.json").exists() and (h.out / "partial-physics.npz").exists()


def test_production_modules_never_use_old_protocol_seed_or_optimizer_restore():
    import inspect
    source = inspect.getsource(study)
    assert "protocol.seed(" not in source and "protocol.schedule(" not in source
    assert "restore_checkpoint(" not in source and "torch.optim." not in source
    assert "training.make_initialization(" not in source and "training.make_orders(" not in source


def test_exact_member_contract_matches_independent_preparation_layer():
    from openjev.research import reacher_two_observation_experiment as experiment
    settings = study.protocol.settings(engineering=True)
    assert set(experiment.expected_members(settings)) == expected_members(settings)
    assert len(experiment.expected_members(settings)) == 8119


@pytest.fixture
def inherited_files(tmp_path, monkeypatch):
    """Handwritten checkpoint metadata and tensors; no seeded construction."""
    settings = study.protocol.settings(engineering=True)
    settings.update(train_episodes=2, epochs=1, batch_size=1, hidden_size=3)
    cfg = study.cache.settings_from_plan(settings)
    parent = {**settings, "sources": {"synthetic": "a" * 64}, "runtime": {"synthetic": True}}
    row = study.protocol.restore_manifest(settings)[0]
    folder = tmp_path / "inherited" / "fits" / row["name"]
    folder.mkdir(parents=True)
    weights = {"fake": torch.tensor([1.], dtype=torch.float32)}
    initial = {"states": {"gru": {"fake": torch.tensor([.2])}},
               "tensor_hashes": {"gru": study.training.canonical_tensor_hash({"fake": torch.tensor([.2])})}}
    initial["integrity_sha256"] = study.training.canonical_state_hash(initial)
    orders = {"orders": torch.tensor([[1, 0]], dtype=torch.int64)}
    orders["integrity_sha256"] = study.training.canonical_state_hash(orders)
    config = study.training.model_configuration(row["arm"], cfg)
    checkpoint = {"version": study.training.VERSION, "failed": False, "kind": row["arm"],
        "settings": cfg.configuration(), "model_configuration": config,
        "source_sha256": parent["sources"], "runtime": parent["runtime"], "data_sha256": "b" * 64,
        "successful_updates": 2, "optimizer_steps": 2, "cursor": {"epoch": 1, "batch": 0},
        "initialization": initial, "orders": orders, "student_state": weights}
    done = {"name": row["name"], "arm": row["arm"], "pair": row["pair"], "settings": cfg.configuration(),
        "model_configuration": config, "data_sha256": "b" * 64, "updates": 2, "optimizer_steps": 2,
        "initialization_sha256": initial["integrity_sha256"], "orders_sha256": orders["integrity_sha256"],
        "student_tensor_sha256": study.training.canonical_tensor_hash(weights)}
    h = SimpleNamespace(settings=settings, parent=parent, row=row, folder=folder, out=tmp_path,
        checkpoint=checkpoint, done=done, initials={row["pair"]: initial}, orders={row["pair"]: orders}, constructed=[])

    def save():
        # Explicit tampering of this handbuilt temporary fixture only.
        for name in study.protocol.FIT_MEMBERS:
            (folder / name).unlink(missing_ok=True)
        h.checkpoint["integrity_sha256"] = study.training.canonical_state_hash(
            {k: v for k, v in h.checkpoint.items() if k != "integrity_sha256"})
        study.cache.save_torch(folder / "checkpoint.pt", h.checkpoint)
        study.cache.save_torch(folder / "weights.pt", weights)
        study.cache.save_torch(folder / "initial-weights.pt", initial["states"]["gru"])
        touch(folder / "training.jsonl")
        h.done["checkpoint_integrity_sha256"] = h.checkpoint["integrity_sha256"]
        h.done["files"] = {name: study.sha(folder / name) for name in study.protocol.FIT_MEMBERS if name != "completed.json"}
        study.write(folder / "completed.json", h.done)
    h.save = save
    save()

    class Fake:
        def __init__(self):
            self.state = {}
        def eval(self):
            return self
        def requires_grad_(self, flag):
            assert flag is False
            return self
        def state_dict(self):
            return self.state
    Fake.__name__ = row["model_class"]
    def construct(kind, actual_cfg, seed):
        assert kind == row["arm"] and actual_cfg == cfg and seed == 0
        h.constructed.append(kind)
        return Fake()
    monkeypatch.setattr(study.training, "_construct", construct)
    monkeypatch.setattr(study.training, "_load_weights", lambda model, values: setattr(model, "state", copy.deepcopy(values)))
    monkeypatch.setattr(study.control, "model_identity", lambda model, p: {"kind": row["arm"]})
    return h


def call_inherited(h):
    return study._inherited_student(h.settings, h.parent, h.out, h.row, "b" * 64, h.initials, h.orders)


def test_inherited_deployment_uses_actual_class_tensors_without_optimizer(inherited_files, monkeypatch):
    h = inherited_files
    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer construction forbidden")
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    model, _, config, digest, updates = call_inherited(h)
    assert type(model).__name__ == h.row["model_class"] and config == h.checkpoint["model_configuration"]
    assert updates == 2 and digest == h.done["student_tensor_sha256"] and h.constructed == [h.row["arm"]]


@pytest.mark.parametrize("mutation", ["class", "settings", "source", "runtime", "data", "updates", "pair", "initial", "orders", "weights"])
def test_resealed_inherited_corruption_rejected_before_construction(inherited_files, mutation):
    h = inherited_files
    if mutation == "class":
        h.checkpoint["kind"] = "encoded_current_gru"
    elif mutation == "settings":
        h.checkpoint["settings"]["dt"] = .03
    elif mutation == "source":
        h.checkpoint["source_sha256"] = {"different": "c" * 64}
    elif mutation == "runtime":
        h.checkpoint["runtime"] = {"different": True}
    elif mutation == "data":
        h.checkpoint["data_sha256"] = "c" * 64
    elif mutation == "updates":
        h.checkpoint["optimizer_steps"] = 1
    elif mutation == "pair":
        h.done["pair"] = "pair1"
    elif mutation == "initial":
        h.checkpoint["initialization"] = copy.deepcopy(h.checkpoint["initialization"])
        h.checkpoint["initialization"]["states"]["gru"]["fake"].add_(1.)
    elif mutation == "orders":
        h.checkpoint["orders"] = copy.deepcopy(h.checkpoint["orders"])
        h.checkpoint["orders"]["orders"] = torch.tensor([[0, 1]])
    elif mutation == "weights":
        h.checkpoint["student_state"] = {"fake": torch.tensor([9.])}
    h.save()
    with pytest.raises(ValueError):
        call_inherited(h)
    assert h.constructed == []


@pytest.fixture
def history_files(inherited_files, monkeypatch):
    from openjev.research import reacher_two_observation_training_audit as saved_audit
    h = inherited_files
    h.row = study.protocol.training_manifest(h.settings)[0]
    h.folder = h.out / "fits" / h.row["name"]
    h.folder.mkdir(parents=True)
    h.plan = {"settings": h.settings, "sources": h.parent["sources"], "runtime": h.parent["runtime"],
        "lineage": {"cache": {"plan_sha256": "c" * 64, "audit_receipt_sha256": "d" * 64,
            "members": {name: "e" * 64 for name in study.protocol.inherited_members(h.settings)}}}}
    h.expected = "f" * 64
    cfg = study.training_settings(h.settings)
    config = saved_audit.model_configuration(cfg.configuration())
    initial = h.initials["pair0"]["states"]["gru"]
    initial_hash = study.training.canonical_tensor_hash(initial)
    order_hash = study.training.canonical_state_hash(h.orders["pair0"])
    provenance = study.fit_provenance(h.plan, h.expected, h.row)
    weights = {"fake": torch.tensor([2.])}
    h.checkpoint = {"version": study.history_training.VERSION, "kind": "two_observation_gru",
        "settings": cfg.configuration(), "model_configuration": config, "failed": False, "failure": None,
        "data_sha256": "b" * 64, "source_sha256": h.plan["sources"], "runtime": h.plan["runtime"],
        "provenance": provenance, "successful_updates": 2, "optimizer_steps": 2,
        "cursor": {"epoch": 1, "batch": 0}, "log_chain_sha256": "1" * 64,
        "initialization": {"weights": initial, "tensor_sha256": initial_hash},
        "orders": h.orders["pair0"], "student_state": weights}
    h.done = {"status": "completed", "name": h.row["name"], "pair": "pair0", "arm": "two_observation_gru",
        "settings": cfg.configuration(), "model_configuration": config, "data_sha256": "b" * 64,
        "source_sha256": h.plan["sources"], "runtime": h.plan["runtime"], "provenance": provenance,
        "updates": 2, "optimizer_steps": 2, "flushed_updates": 2, "cursor": {"epoch": 1, "batch": 0},
        "log_chain_sha256": "1" * 64, "initialization_sha256": initial_hash, "orders_sha256": order_hash,
        "student_tensor_sha256": study.training.canonical_tensor_hash(weights)}
    def save():
        for name in study.protocol.FIT_MEMBERS:
            (h.folder / name).unlink(missing_ok=True)
        h.checkpoint["integrity_sha256"] = study.training.canonical_state_hash(
            {k: v for k, v in h.checkpoint.items() if k != "integrity_sha256"})
        study.cache.save_torch(h.folder / "checkpoint.pt", h.checkpoint)
        study.cache.save_torch(h.folder / "initial-weights.pt", initial)
        study.cache.save_torch(h.folder / "weights.pt", weights)
        touch(h.folder / "training.jsonl")
        h.done["checkpoint_integrity_sha256"] = h.checkpoint["integrity_sha256"]
        h.done["files"] = {name: study.sha(h.folder / name) for name in study.fitting.PAYLOAD_FILES}
        study.write(h.folder / "completed.json", h.done)
    h.save = save
    save()
    class FakeHistory:
        def __init__(self):
            self.state = {}
        def configuration(self):
            return config
        def eval(self):
            return self
        def requires_grad_(self, flag):
            assert not flag
            return self
        def state_dict(self):
            return self.state
    FakeHistory.__name__ = "TwoObservationHistoryGRUWorldModel"
    def construct(actual_cfg):
        assert actual_cfg == cfg
        h.constructed.append("history")
        return FakeHistory()
    monkeypatch.setattr(study.history_training, "_construct", construct)
    monkeypatch.setattr(study.history_training, "_load_weights", lambda model, values: setattr(model, "state", copy.deepcopy(values)))
    monkeypatch.setattr(study.history_control, "model_identity", lambda model, settings: {"kind": "two_observation_gru"})
    return h


def call_history(h):
    return study._history_student(h.plan, h.expected, h.out, h.row, "b" * 64, h.initials, h.orders)


def test_history_deployment_never_constructs_or_restores_adam(history_files, monkeypatch):
    h = history_files
    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer or trainer restore forbidden")
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(study.history_training, "restore_checkpoint", forbidden)
    model, _, config, digest, updates = call_history(h)
    assert type(model).__name__ == "TwoObservationHistoryGRUWorldModel"
    assert config == h.checkpoint["model_configuration"] and digest == h.done["student_tensor_sha256"] and updates == 2
    assert h.constructed == ["history"]


@pytest.mark.parametrize("mutation", ["class", "failed", "source", "runtime", "provenance", "order_scope", "initial", "weights", "flushed"])
def test_history_resealed_corruption_rejected_before_class_construction(history_files, mutation):
    h = history_files
    if mutation == "class":
        h.checkpoint["kind"] = "residual_gru"
    elif mutation == "failed":
        h.checkpoint["failed"] = True
    elif mutation in ("source", "runtime", "provenance"):
        field = "source_sha256" if mutation == "source" else mutation
        h.checkpoint[field] = {"changed": "a" * 64}
    elif mutation == "order_scope":
        h.done["orders_sha256"] = h.orders["pair0"]["integrity_sha256"]
    elif mutation == "initial":
        h.checkpoint["initialization"] = {"weights": {"fake": torch.tensor([99.])}, "tensor_sha256": "a" * 64}
    elif mutation == "weights":
        h.checkpoint["student_state"] = {"fake": torch.tensor([99.])}
    elif mutation == "flushed":
        h.done["flushed_updates"] = 1
    h.save()
    with pytest.raises(ValueError):
        call_history(h)
    assert h.constructed == []
