"""Synthetic byte trees only; no scientific namespace/model/native execution."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openjev.research import reacher_two_observation_experiment as experiment
from openjev.research import reacher_two_observation_protocol as protocol
from openjev.research import reacher_two_observation_streams as streams


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def fake_runtime():
    return {"python": "synthetic", "platform": "synthetic", "packages": dict.fromkeys(("numpy", "torch", "gymnasium", "mujoco"), "synthetic"),
            "native_source_sha256": "a" * 64, "native_xml_sha256": "b" * 64, "mujoco_init_sha256": "c" * 64}


def make_parent(root, name, plan, members, *, gate=False):
    plan_path = f"{name}/plan.json"
    plan_sha = write(root, plan_path, plan)
    execution = f"{name}/execution"
    for path, payload in members.items():
        write(root, f"{execution}/{path}", payload)
    file_hashes = {path: experiment.sha(root / execution / path) for path in members}
    done = {"status": "completed", "plan_sha256": plan_sha, "files": file_hashes, "wall_seconds": .5}
    done_sha = write(root, f"{execution}/completed.json", done)
    summary_sha = write(root, f"{name}/audit/summary.json", {"continuation_gate": {"passed": gate}})
    readme_sha = write(root, f"{name}/audit/README.md", b"Synthetic parent audit metadata only.\n")
    audit = {"status": "completed", "engineering": True, "saved_output_only": True,
             "plan_sha256": plan_sha, "source_sha256": plan["sources"], "runtime": plan["runtime"],
             "execution_completed_sha256": done_sha, "execution_members": file_hashes,
             "files": {"summary.json": summary_sha, "README.md": readme_sha},
             "costs": {"synthetic_prior_wall_seconds": 0.}}
    audit_path = f"{name}/audit/receipt.json"
    audit_sha = write(root, audit_path, audit)
    refs = {"plan_path": plan_path, "plan_sha256": plan_sha, "audit_path": audit_path,
            "audit_receipt_sha256": audit_sha, "execution_path": execution}
    bound = {**refs, "completed_sha256": done_sha, "summary_sha256": summary_sha, "members": file_hashes}
    return refs, bound


@pytest.fixture
def tree(tmp_path):
    settings = protocol.settings(engineering=True)
    settings.update(control_episodes=2, train_episodes=4, epochs=1, batch_size=2, hidden_size=3, mlp_width=7)
    runtime = fake_runtime()
    inherited = {f"synthetic/frozen-{i:03d}.py": write(tmp_path, f"synthetic/frozen-{i:03d}.py", f"# synthetic old source {i}\n".encode()) for i in range(90)}
    for name in experiment.SOURCE_PATHS_NEW:
        write(tmp_path, name, f"# synthetic additive source {name}\n".encode())
    cache_plan = {**copy.deepcopy(settings), "study": "reacher-cache-ablation-v1", "runtime": runtime,
                  "sources": dict(list(inherited.items())[:70]), "cap_seconds": 10}
    members = {name: ("synthetic audited member " + name).encode() for name in protocol.inherited_members(settings)}
    cache_ref, cache_source = make_parent(tmp_path, "cache", cache_plan, members)
    roles = protocol.role_manifest(protocol.settings(engineering=True))["root_roles"]
    numpy = [{"old/reset": 2001, "planner/particle_filter/0": 2002}]
    torch = [{"old/order": 2003}]
    descriptors = [{"scope": "synthetic prior maps only"}]
    own = []
    for i, namespace in enumerate(streams.GEOMETRY_MEMORY_NAMESPACES):
        values = {role: 10000 + i * 1000 + j for j, role in enumerate(roles)}
        own.append({"namespace": namespace, "registry": values, "generators": streams.generator_manifest(values)})
    current = own[1]
    prior = {**current, "namespace_exclusions": [row for row in own if row is not current],
        "prior_numpy_registry_sha256": [experiment.identity(value) for value in numpy],
        "prior_torch_registry_sha256": [experiment.identity(value) for value in torch], "priors": descriptors,
        "engineering_literal_exclusions": [{"numpy_registry": {"older/literal": 410},
            "generators": streams.generator_manifest({"older/literal": 410}),
            "torch_registry": {"older/construction": 0}, "torch_generators": streams.torch_generator_manifest({"older/construction": 0})}]}
    for i, field in enumerate(("geometry_study_exclusions", "cache_study_exclusions", "memory_study_exclusions")):
        values = {f"{field}/old/reset": 3000 + i}
        prior[field] = [{"namespace": f"synthetic-{field}", "registry": values, "generators": streams.generator_manifest(values)}]
    prior["prior_bindings"] = {"numpy_count": len(numpy), "torch_count": len(torch),
        "numpy_hash_list_sha256": experiment.identity(prior["prior_numpy_registry_sha256"]),
        "torch_hash_list_sha256": experiment.identity(prior["prior_torch_registry_sha256"]),
        "descriptors_sha256": experiment.identity(descriptors)}
    parent = {"study": "reacher-geometry-memory-v1", "engineering": True, "runtime": runtime,
              "sources": inherited, "cap_seconds": 10, "cache_source": cache_source, "random_stream_contract": prior}
    prerequisite_ref, _ = make_parent(tmp_path, "prerequisite", parent, {"synthetic.json": b"{}"}, gate=False)
    prerequisite_ref.update(terminal_path=prerequisite_ref["audit_path"], terminal_sha256=prerequisite_ref["audit_receipt_sha256"],
                           terminal_kind="engineering_completed_audit")
    known = {**copy.deepcopy(streams.REQUIRED_LITERAL_CALLS), **copy.deepcopy(experiment.ADDITIONAL_LITERAL_CALLS)}
    literals, extra = [], {}
    for name, kinds in known.items():
        if name not in experiment.SOURCE_PATHS_NEW:
            extra[name] = write(tmp_path, name, b"# synthetic separately bound engineering source\n")
        literals.append({"source_path": name, "source_sha256": experiment.sha(tmp_path / name),
            "numpy_registry": kinds["numpy"], "numpy_generators": streams.generator_manifest(kinds["numpy"]),
            "torch_registry": kinds["torch"], "torch_generators": streams.torch_generator_manifest(kinds["torch"])})
    historical = {"numpy": numpy, "torch": torch, "descriptors": descriptors,
                  "literal_calls": literals, "engineering_sources": extra}
    return SimpleNamespace(root=tmp_path, settings=settings, runtime=runtime,
        refs={"cache": cache_ref, "prerequisite": prerequisite_ref}, historical=historical, parent=parent)


@pytest.fixture(autouse=True)
def forbid_new_scored_allocation(monkeypatch):
    original = streams.named_seed

    def only_engineering(namespace, role):
        assert namespace in protocol.ENGINEERING_NAMESPACES, "No scored allocation in tests"
        return original(namespace, role)

    monkeypatch.setattr(streams, "named_seed", only_engineering)


def prepare(tree, **kwargs):
    arguments = {"lineage_refs": tree.refs, "historical_registries": tree.historical,
                 "runtime": tree.runtime, "cap_seconds": 10, "audit_cap_seconds": 20,
                 "engineering_evidence": {}, "engineering": True}
    arguments.update(kwargs)
    return experiment.prepare(tree.root, tree.settings, **arguments)


def test_complete_engineering_binding_is_unfrozen_and_revalidates_original_bytes(tree):
    plan = prepare(tree)
    assert plan["status"] == "prepared_unfrozen"
    assert len(plan["sources"]) == 116 and len(plan["lineage"]["cache"]["members"]) == 44
    assert plan["lineage"]["prerequisite"]["recorded_gate_passed"] is False
    assert plan["lineage"]["prerequisite"]["previous_scientific_gate_passed"] is None
    expected = write(tree.root, "prepared/plan.json", plan)
    result = experiment.validate_plan(plan, expected, root=tree.root, runtime=tree.runtime,
                                     engineering=True, plan_path=tree.root / "prepared/plan.json")
    assert result.execution_authorized and result.engineering
    assert result.cache_source == plan["lineage"]["cache"]
    assert result.prerequisite_plan == tree.parent
    assert result.settings == tree.settings
    assert result.streams.registry == plan["random_stream_contract"]["registry"]
    assert len(result.history["geometry_memory"]) == 5
    assert all(len(row["registry"]) == 508 for row in result.history["geometry_memory"])
    assert len(result.history["numpy_priors"]) == 5  # original + three namespace sets + old literal
    assert result.history["torch_priors"][-1]["registry"] == {"older/construction": 0}


def test_prior_partial_engineering_namespace_expands_full64_without_new_scored_derivation(tree):
    parent = copy.deepcopy(tree.parent)
    old = parent["random_stream_contract"]
    namespace = old["namespace"]
    assert namespace == "reacher-geometry-memory-engineering-unit-v1"
    roles = protocol.role_manifest(tree.settings)["root_roles"]
    old["registry"] = {role: int.from_bytes(hashlib.sha256(f"OpenJev/{namespace}/{role}".encode()).digest()[:8], "big") for role in roles}
    old["generators"] = streams.generator_manifest(old["registry"])
    reference, _ = make_parent(tree.root, "partial-parent", parent, {"synthetic.json": b"{}"})
    reference.update(terminal_path=reference["audit_path"], terminal_sha256=reference["audit_receipt_sha256"],
                     terminal_kind="engineering_completed_audit")
    tree.refs["prerequisite"] = reference
    plan = prepare(tree)
    expanded = next(row for row in plan["random_stream_contract"]["history"]["geometry_memory"] if row["name"] == namespace)
    assert len(expanded["registry"]) == 508 and len(expanded["generators"]) == 636
    assert all(expanded["registry"][role] == value for role, value in old["registry"].items())
    assert expanded["generators"]["planner/particle_filter/63/resample"]["spawn_key"] == [2]


def test_authenticated_prerequisite_cannot_switch_original_cache_parent(tree):
    parent = copy.deepcopy(tree.parent)
    parent["cache_source"]["plan_sha256"] = "d" * 64
    reference, _ = make_parent(tree.root, "switched-parent", parent, {"synthetic.json": b"{}"})
    reference.update(terminal_path=reference["audit_path"], terminal_sha256=reference["audit_receipt_sha256"],
                     terminal_kind="engineering_completed_audit")
    tree.refs["prerequisite"] = reference
    with pytest.raises(ValueError, match="exact same completed cache"):
        prepare(tree)


def test_exact_execution_file_contract_preserves_all_models_rows_and_lineage(tree):
    names = experiment.expected_members(tree.settings)
    assert len(names) == len(set(names)) == 8119
    assert len(experiment.expected_members(tree.settings, include_completed=True)) == 8120
    assert "completed.json" not in names
    assert len([name for name in names if name.startswith("inherited/lineage/")]) == 9
    assert len([name for name in names if name.startswith("model-states/")]) == 24
    assert len([name for name in names if name.startswith("fits/")]) == 15
    assert len([name for name in names if name.startswith("innovations/")]) == 100
    for panel in protocol.PANELS:
        assert f"control/{panel}/two_observation_gru-pair2/controller-decisions/049.json" in names
        assert f"control/{panel}/cached_gru-pair2/scoring/049.npz" in names
        assert f"control/{panel}/particle/observer-final.json" in names
        assert f"control/{panel}/public_kinematic/observer-final.json" in names
        assert f"control/{panel}/known_state/physics/049.npz" in names
        assert f"control/{panel}/zero/planning.npz" in names


@pytest.mark.parametrize("target", ["train.npz", "initializations/pair2.pt", "orders/pair2.json", "fits/cached_gru-pair2/checkpoint.pt"])
def test_any_inherited44_corruption_blocks_before_seed_derivation(tree, target):
    (tree.root / "cache/execution" / target).write_bytes(b"corrupted")
    with patch.object(streams, "stream_contract", side_effect=AssertionError("Must not allocate")), pytest.raises(ValueError, match="digest mismatch"):
        prepare(tree)


@pytest.mark.parametrize("target", ["synthetic/frozen-089.py", "src/openjev/research/reacher_two_observation_control.py"])
def test_changed_source_blocks_validation(tree, target):
    plan = prepare(tree)
    payload = json.dumps(plan).encode()
    (tree.root / target).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="digest mismatch|changed"):
        experiment.validate_plan(plan, hashlib.sha256(payload).hexdigest(), root=tree.root,
            runtime=tree.runtime, engineering=True, plan_bytes=payload)


def test_missing_auditor_source_blocks_preparation(tree):
    (tree.root / "scripts/audit_reacher_two_observation_study.py").unlink()
    with patch.object(streams, "stream_contract", side_effect=AssertionError("Must not allocate")), pytest.raises(ValueError, match="Required regular file missing"):
        prepare(tree)


@pytest.mark.parametrize("mutation", ["missing_prior", "changed_prior", "descriptors", "literal", "extra_source", "source_alias"])
def test_historical_closure_and_literal_bytes_are_not_caller_assertions(tree, mutation):
    history = tree.historical
    if mutation == "missing_prior":
        history["numpy"] = []
    elif mutation == "changed_prior":
        history["torch"][0]["old/order"] += 1
    elif mutation == "descriptors":
        history["descriptors"][0]["scope"] = "omitted prior"
    elif mutation == "literal":
        history["literal_calls"] = history["literal_calls"][:-1]
    elif mutation == "extra_source":
        (tree.root / next(iter(history["engineering_sources"]))).write_bytes(b"changed driver")
    else:
        name = experiment.SOURCE_PATHS_NEW[0]
        history["engineering_sources"][name] = experiment.sha(tree.root / name)
    with pytest.raises(ValueError):
        prepare(tree)


@pytest.mark.parametrize("cap", [0, -1, True, 1.0, float("nan"), float("inf")])
def test_explicit_finite_builtin_integer_caps_before_parent_reads(tree, cap):
    with patch.object(experiment, "_lineage", side_effect=AssertionError("No parent reads")), pytest.raises(ValueError, match="positive integer"):
        prepare(tree, cap_seconds=cap)


def test_recipe_change_and_runtime_drift_rejected(tree):
    tree.settings["epochs"] = 2
    with pytest.raises(ValueError, match="Original paired training recipe"):
        prepare(tree)
    tree.settings["epochs"] = 1
    changed = copy.deepcopy(tree.runtime)
    changed["packages"]["torch"] = "different"
    with pytest.raises(ValueError, match="runtime differs"):
        prepare(tree, runtime=changed)


def test_mode_promotion_cannot_reuse_engineering_parents(tree):
    settings = protocol.settings()
    with patch.object(streams, "stream_contract", side_effect=AssertionError("No scored allocation")), pytest.raises(ValueError, match="Completed parent"):
        experiment.prepare(tree.root, settings, lineage_refs=tree.refs, historical_registries=tree.historical,
            runtime=tree.runtime, cap_seconds=10, audit_cap_seconds=20, engineering_evidence={}, engineering=False)


@pytest.mark.parametrize("mutation", ["bytes", "object", "extra_key", "both_inputs", "no_inputs", "engineering_freeze"])
def test_external_plan_bytes_and_exact_envelope_boundary(tree, mutation):
    plan = prepare(tree)
    payload = json.dumps(plan).encode()
    expected = hashlib.sha256(payload).hexdigest()
    kwargs = {"plan_bytes": payload}
    if mutation == "bytes":
        kwargs["plan_bytes"] = payload + b"\n"
    elif mutation == "object":
        plan["cap_seconds"] += 1
    elif mutation == "extra_key":
        plan["unauthenticated_shortcut"] = True
        plan["content_sha256"] = experiment.identity({k: v for k, v in plan.items() if k != "content_sha256"})
        kwargs["plan_bytes"] = json.dumps(plan).encode()
        expected = hashlib.sha256(kwargs["plan_bytes"]).hexdigest()
    elif mutation == "both_inputs":
        kwargs["plan_path"] = "unused.json"
    elif mutation == "no_inputs":
        kwargs = {}
    else:
        kwargs["freeze_ref"] = {"path": "unused.json", "sha256": "a" * 64}
    with pytest.raises(ValueError):
        experiment.validate_plan(plan, expected, root=tree.root, runtime=tree.runtime, engineering=True, **kwargs)


def test_paths_duplicate_keys_symlinks_and_external_plan_cannot_escape(tree, tmp_path):
    digest = write(tree.root, "safe.json", b'{"same": 1, "same": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        experiment.read_bound(tree.root, "safe.json", digest)
    for name in ("../safe.json", "/safe.json", "a//b", "a/./b", "a\\b"):
        with pytest.raises(ValueError):
            experiment.checked(tree.root, name, digest)
    (tree.root / "linked.json").symlink_to(tree.root / "safe.json")
    with pytest.raises(ValueError, match="Symlink"):
        experiment.read_bound(tree.root, "linked.json", digest)
    with pytest.raises(ValueError, match="under the explicit"):
        experiment.validate_plan({}, "a" * 64, root=tree.root, runtime=tree.runtime,
                                 engineering=True, plan_path=tmp_path.parent / "outside.json")


def qualification(root, kind, settings, sources, observed_runtime):
    raw_sha = write(root, f"qualification/{kind}/raw.json", {"scope": "synthetic measured evidence only"})
    review_sha = write(root, f"qualification/{kind}/review.md", b"Synthetic review, never production readiness.\n")
    receipt = {"schema": f"reacher-two-observation-{kind}-qualification-v1", "status": "completed",
        "engineering": True, "study": protocol.STUDY, "source_sha256": sources, "runtime": observed_runtime,
        "files": {"raw.json": raw_sha, "review.md": review_sha}, "wall_seconds": 1.}
    if kind == "rehearsal":
        receipt.update(new_fits=3, inherited_models=6, restored_final_models=9, control_rows=42,
                       saved_output_audit_completed=True, native_replay_max_abs_error=0., scientific_draws=0)
    else:
        receipt.update(full_shape={key: settings[key] for key in ("hidden_size", "train_episodes", "batch_size", "steps", "control_episodes", "planning_horizon", "action_block")},
            coverage=protocol.coverage(settings), review_sha256=review_sha, projected_seconds={"execution": 12., "audit": 8.},
            measured_components={name: {"wall_seconds": .1, "units": 1, "artifact": "raw.json"}
                for name in ("training", "learned_control", "physics_references", "audit", "storage_and_hashing")})
    path = f"qualification/{kind}/receipt.json"
    return {"path": path, "sha256": write(root, path, receipt)}, receipt


def test_capacity_and_rehearsal_gate_without_any_scored_allocation(tree):
    # Exercise admission arithmetic only, never prepare(settings(scored)).
    settings = protocol.settings()
    sources = experiment._sources(tree.root, tree.parent["sources"])
    refs = {kind: qualification(tree.root, kind, settings, sources, tree.runtime)[0] for kind in ("rehearsal", "capacity")}
    result = experiment._engineering_evidence(tree.root, refs, settings, sources, tree.runtime, 20, 10, False)
    assert set(result) == {"rehearsal", "capacity"}
    for evidence in ({}, {"rehearsal": refs["rehearsal"]}):
        with pytest.raises(ValueError, match="rehearsal and measured capacity"):
            experiment._engineering_evidence(tree.root, evidence, settings, sources, tree.runtime, 20, 10, False)
    with pytest.raises(ValueError, match="Caps must exceed"):
        experiment._engineering_evidence(tree.root, refs, settings, sources, tree.runtime, 12, 10, False)


@pytest.mark.parametrize("mutation", ["fit_only", "audit_incomplete", "wrong_sources", "missing_measurement", "unmeasured", "wrong_shape", "missing_review", "corrupt_raw"])
def test_readiness_receipts_require_retained_full_measured_work(tree, mutation):
    settings = protocol.settings()
    sources = experiment._sources(tree.root, tree.parent["sources"])
    refs, receipts = {}, {}
    for kind in ("rehearsal", "capacity"):
        refs[kind], receipts[kind] = qualification(tree.root, kind, settings, sources, tree.runtime)
    kind = "rehearsal" if mutation in ("fit_only", "audit_incomplete") else "capacity"
    value = receipts[kind]
    if mutation == "fit_only":
        value["control_rows"] = 0
    elif mutation == "audit_incomplete":
        value["saved_output_audit_completed"] = False
    elif mutation == "wrong_sources":
        value["source_sha256"].pop(next(iter(sources)))
    elif mutation == "missing_measurement":
        del value["measured_components"]["storage_and_hashing"]
    elif mutation == "unmeasured":
        value["measured_components"]["training"]["wall_seconds"] = 0
    elif mutation == "wrong_shape":
        value["full_shape"]["hidden_size"] = 3
    elif mutation == "missing_review":
        value["review_sha256"] = "f" * 64
    else:
        write(tree.root, "qualification/capacity/raw.json", b"corrupt")
    refs[kind]["sha256"] = write(tree.root, refs[kind]["path"], value)
    with pytest.raises(ValueError):
        experiment._engineering_evidence(tree.root, refs, settings, sources, tree.runtime, 20, 10, False)


def test_scientific_freeze_is_separate_exact_external_binding_without_seed_calls(tree):
    plan = {"content_sha256": "d" * 64, "sources": {"synthetic.py": "e" * 64}, "engineering_evidence": {"synthetic": "only"}}
    expected = "f" * 64
    body = {"schema": "reacher-two-observation-freeze-v1", "status": "frozen", "study": protocol.STUDY,
            "engineering": False, "plan_sha256": expected, "content_sha256": plan["content_sha256"],
            "source_sha256": plan["sources"], "runtime": tree.runtime,
            "engineering_evidence_sha256": experiment.identity(plan["engineering_evidence"]), "source_commit": "1" * 40, "no_retry": True}
    ref = {"path": "freeze.json", "sha256": write(tree.root, "freeze.json", body)}
    experiment._freeze(tree.root, ref, plan, expected, tree.runtime)
    with pytest.raises(ValueError, match="authenticated scientific freeze"):
        experiment._freeze(tree.root, None, plan, expected, tree.runtime)
    body["no_retry"] = False
    ref["sha256"] = write(tree.root, "freeze.json", body)
    with pytest.raises(ValueError, match="Exact scientific freeze"):
        experiment._freeze(tree.root, ref, plan, expected, tree.runtime)


def test_no_model_or_native_imports_and_no_model_file_deserialization():
    source = Path(experiment.__file__).read_text()
    tree = ast.parse(source)
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert not any(any(forbidden in name for forbidden in ("torch", "mujoco", "gymnasium", "training", "world_model", "study")) for name in imports)
    assert "torch.load" not in source and "pickle" not in source and "np.load" not in source
