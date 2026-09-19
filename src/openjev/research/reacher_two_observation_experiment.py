"""Byte authentication and preparation for the two-observation experiment.

This layer never loads Torch payloads, constructs models, fits, steps a simulator
or samples evaluation data. Preparation does derive the requested namespace's
seed/state manifest, but only AFTER all admission checks. Scientific preparation
requires retained whole-tree/capacity evidence; execution additionally requires
an externally authenticated freeze. Engineering uses explicit engineering
parents/settings and cannot be promoted to scientific execution.

The enclosing runner owns exclusive directories, failure receipts, deadlines,
tensor semantics/restoration and actual execution. A successful validation here
does not certify fitted weights, optimizer arithmetic or empirical effectiveness.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from openjev.research import reacher_two_observation_protocol as protocol
from openjev.research import reacher_two_observation_streams as streams

SCHEMA = "reacher-two-observation-experiment-v1"
ARTIFACT_VERSION = "reacher-two-observation-members-v1"
_HELPERS = ("history", "training", "control", "protocol", "audit", "results", "episode",
            "streams", "fitting", "training_audit", "experiment")
SOURCE_PATHS_NEW = tuple(path for name in _HELPERS for path in (
    f"src/openjev/research/reacher_two_observation_{name}.py",
    f"tests/test_reacher_two_observation_{name}.py")) + (
    "scripts/reacher_two_observation_study.py", "tests/test_reacher_two_observation_study.py",
    "scripts/audit_reacher_two_observation_study.py", "tests/test_audit_reacher_two_observation_study.py")
ROOT_PAYLOADS = ("started.json", "random-streams.json", "inheritance.json", "training-inputs.json",
    "inherited-models-restored.json", "training-started.json", "all-fits-completed.json",
    "all-models-restored.json", "evaluation-started.json", "control-completed.json",
    "final-models.json", "costs.json", "results.json")
LINEAGE_SUFFIXES = {"cache": ("plan", "audit", "completed", "summary"),
                    "prerequisite": ("plan", "audit", "completed", "summary", "terminal")}
TRAINING_FIELDS = ("train_episodes", "epochs", "batch_size", "hidden_size", "mlp_width", "steps", "dt",
    "noise_std", "learning_rate", "gradient_clip", "rollout_horizon", "rollout_weight", "reward_scale",
    "kl_weight", "kl_balance", "free_nats", "optimizer", "residual_reward")
ADDITIONAL_LITERAL_CALLS = {
    "tests/test_reacher_two_observation_training_audit.py": {"numpy": {}, "torch": {"synthetic_historical_orders": 410}},
    "tests/test_audit_reacher_two_observation_study.py": {"numpy": {}, "torch": {"delegated_synthetic_historical_orders": 410}},
    "output/reacher-two-observation-control-v1/integrate_training.py": {"numpy": {}, "torch": {"delegated_seed410_fixture": 410}},
    "scripts/reacher_two_observation_study.py": {"numpy": {}, "torch": {"inherited_overwritten_constructor": 0, "history_overwritten_constructor": 410}},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("Finite canonical JSON required") from error


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _relative(value):
    require(isinstance(value, str) and value and "\\" not in value
            and not PurePosixPath(value).is_absolute()
            and all(part not in ("", ".", "..") for part in value.split("/")), "Safe relative path required")
    return value


def _path(root, relative):
    root = Path(root).resolve(strict=True)
    current = root
    for part in _relative(relative).split("/"):
        current = current / part
        require(not current.is_symlink(), "Symlink artifact/source paths are not permitted")
    require(current.is_file(), f"Required regular file missing: {relative}")
    return current


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked(root, relative, expected):
    require(_sha(expected), "Explicit lowercase SHA256 required")
    path = _path(root, relative)
    require(sha(path) == expected, f"Artifact/source digest mismatch: {relative}")
    return path


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(payload):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON token: {value}")
    value = json.loads(payload, object_pairs_hook=_object, parse_constant=invalid)
    require(type(value) is dict, "JSON object required")
    canonical(value)
    return value


def read_bound(root, relative, expected):
    require(_sha(expected), "Explicit SHA256 required")
    payload = _path(root, relative).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected, f"JSON digest mismatch: {relative}")
    return _decode(payload)


def _map(value, label, *, allow_empty=False):
    require(type(value) is dict and (allow_empty or value)
            and all(_sha(digest) for digest in value.values()), f"{label}: file hash map required")
    for name in value:
        _relative(name)
    return value


def _finite(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"{label}: finite {'positive' if positive else 'nonnegative'} value required")
    return float(value)


def runtime():
    """Old runtime fingerprint without importing model, native or gym modules."""
    packages = {name: importlib.metadata.distribution(name) for name in ("numpy", "torch", "gymnasium", "mujoco")}
    gym = packages["gymnasium"]
    return {"python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: item.version for name, item in packages.items()},
        "native_source_sha256": sha(gym.locate_file("gymnasium/envs/mujoco/reacher_v5.py")),
        "native_xml_sha256": sha(gym.locate_file("gymnasium/envs/mujoco/assets/reacher.xml")),
        "mujoco_init_sha256": sha(packages["mujoco"].locate_file("mujoco/__init__.py"))}


def _runtime(value):
    require(type(value) is dict and set(value) == {"python", "platform", "packages", "native_source_sha256", "native_xml_sha256", "mujoco_init_sha256"},
            "Complete runtime fingerprint required")
    require(all(isinstance(value[key], str) and value[key] for key in ("python", "platform"))
            and type(value["packages"]) is dict and set(value["packages"]) == {"numpy", "torch", "gymnasium", "mujoco"}
            and all(isinstance(item, str) and item for item in value["packages"].values())
            and all(_sha(value[key]) for key in ("native_source_sha256", "native_xml_sha256", "mujoco_init_sha256")), "Runtime versions/native source hashes required")


def expected_members(settings, *, include_completed=False):
    """Exact successful execution membership, independently rederived by audit."""
    protocol.validate_settings(settings, engineering=settings.get("engineering") is True)
    require(type(include_completed) is bool, "Explicit completion membership flag")
    names = set(ROOT_PAYLOADS)
    names.update(f"inherited/{name}" for name in protocol.inherited_members(settings))
    names.update(f"inherited/lineage/{kind}-{suffix}.json" for kind, suffixes in LINEAGE_SUFFIXES.items() for suffix in suffixes)
    for row in protocol.fit_manifest(settings):
        name = row["name"]
        if row["new_fit"]:
            names.update(f"fits/{name}/{member}" for member in protocol.FIT_MEMBERS)
        else:
            names.add(f"model-states/{name}-prefit.pt")
        names.update(f"model-states/{name}-{suffix}.pt" for suffix in ("before", "after"))
    names.update(f"innovations/control/{step:03d}.{suffix}" for step in range(50) for suffix in ("npz", "json"))
    for row in protocol.execution_order(settings):
        prefix = row["path"]
        local = {"episodes.npz", "episodes.json", "timings.json"}
        if "fit" in row:
            local.update(("executed_predictions.npz", "states.npz", "state-work.json"))
            local.update(f"{folder}/{step:03d}.{suffix}" for folder in ("decisions", "scoring") for step in range(50) for suffix in ("npz", "json"))
            if row["fit"].startswith("two_observation_gru-"):
                local.update(("started.json", "inputs.json", "completed.json"))
                local.update(f"controller-decisions/{step:03d}.json" for step in range(50))
        else:
            local.add("planning.npz")
            if row["reference"] in protocol.PHYSICS_REFERENCES:
                local.add("physics-final.json")
                local.update(f"{folder}/{step:03d}.{suffix}" for folder in ("decisions", "physics") for step in range(50) for suffix in ("npz", "json"))
                if row["reference"] != "known_state":
                    local.add("observer-final.json")
        names.update(f"{prefix}/{name}" for name in local)
    if include_completed:
        names.add("completed.json")
    return sorted(names)


def _parent(root, reference, *, kind, engineering):
    keys = {"plan_path", "plan_sha256", "audit_path", "audit_receipt_sha256", "execution_path"}
    if kind == "prerequisite":
        keys |= {"terminal_path", "terminal_sha256", "terminal_kind"}
    require(type(reference) is dict and set(reference) == keys, "Exact explicit parent descriptor required")
    plan = read_bound(root, reference["plan_path"], reference["plan_sha256"])
    audit = read_bound(root, reference["audit_path"], reference["audit_receipt_sha256"])
    _relative(reference["execution_path"])
    done_path = f"{reference['execution_path']}/completed.json"
    done = read_bound(root, done_path, audit["execution_completed_sha256"])
    expected_study = "reacher-cache-ablation-v1" if kind == "cache" else "reacher-geometry-memory-v1"
    require(plan.get("study") == expected_study and plan.get("engineering") is engineering
            and audit.get("engineering") is engineering and audit.get("saved_output_only") is True
            and audit.get("status") == done.get("status") == "completed"
            and audit.get("plan_sha256") == done.get("plan_sha256") == reference["plan_sha256"]
            and audit.get("source_sha256") == plan.get("sources")
            and audit.get("runtime") == plan.get("runtime")
            and audit.get("execution_members") == done.get("files"), "Completed parent plan/audit/execution binding")
    _map(plan["sources"], "Parent sources")
    require(len(plan["sources"]) == (70 if kind == "cache" else 90), "Exact inherited source count")
    _map(done["files"], "Parent execution members")
    _map(audit["files"], "Parent audit members")
    require(set(audit["files"]) == {"summary.json", "README.md"}, "Exact parent audit payload membership")
    require(_finite(done["wall_seconds"], "Parent wall") <= _finite(plan["cap_seconds"], "Parent cap", positive=True), "Parent execution exceeded cap")
    audit_dir = str(PurePosixPath(reference["audit_path"]).parent)
    for name, digest in audit["files"].items():
        checked(root, f"{audit_dir}/{name}", digest)
    summary = read_bound(root, f"{audit_dir}/summary.json", audit["files"]["summary.json"])
    gate = summary["continuation_gate"]
    require(type(gate) is dict and type(gate.get("passed")) is bool, "Actual parent gate state required")
    if not engineering:
        require(reference["plan_sha256"] == (protocol.CACHE_PLAN_SHA256 if kind == "cache" else protocol.PREREQUISITE_PLAN_SHA256), "Pinned scientific parent plan")
        require(gate["passed"] is (kind == "prerequisite"), "Preserve failed cache gate and positive prerequisite")
    names = protocol.inherited_members(protocol.settings(engineering=engineering)) if kind == "cache" else []
    require(set(names) <= set(done["files"]), "All44 original training/pairing/checkpoint files must have been audited")
    members = {name: done["files"][name] for name in names}
    for name, digest in members.items():
        checked(root, f"{reference['execution_path']}/{name}", digest)
    source = {**reference, "completed_path": done_path, "completed_sha256": audit["execution_completed_sha256"],
        "summary_path": f"{audit_dir}/summary.json", "summary_sha256": audit["files"]["summary.json"],
        "members": members, "prior_costs": copy.deepcopy(audit["costs"]), "engineering": engineering,
        "previous_scientific_gate_passed": gate["passed"] if not engineering else None,
        "recorded_gate_passed": gate["passed"]}
    if kind == "prerequisite":
        terminal = read_bound(root, reference["terminal_path"], reference["terminal_sha256"])
        terminal_kind = reference["terminal_kind"]
        if engineering and terminal_kind == "engineering_completed_audit":
            require(reference["terminal_path"] == reference["audit_path"]
                    and reference["terminal_sha256"] == reference["audit_receipt_sha256"]
                    and terminal == audit, "Explicit engineering audit terminal binding")
        else:
            require(terminal_kind == "independent_terminal_verification" and terminal.get("status") == "completed"
                    and terminal.get("engineering") is engineering
                    and terminal.get("plan_sha256") == reference["plan_sha256"]
                    and terminal.get("execution_completed_sha256") == source["completed_sha256"]
                    and terminal.get("audit_receipt_sha256") == reference["audit_receipt_sha256"]
                    and terminal.get("audit_summary_sha256") == source["summary_sha256"]
                    and terminal.get("source_sha256") == plan["sources"]
                    and type(terminal.get("execution_exit_code")) is int and terminal["execution_exit_code"] == 0
                    and type(terminal.get("audit_exit_code")) is int and terminal["audit_exit_code"] == 0,
                    "Independent completed parent terminal binding")
            if not engineering:
                require(reference["terminal_sha256"] == protocol.PREREQUISITE_TERMINAL_SHA256
                        and terminal.get("qualification_passed") is True
                        and terminal.get("checks_passed") == terminal.get("total_checks") == 25
                        and terminal.get("bound_sources_verified") == 90
                        and terminal.get("restored_checkpoints") == 12 and terminal.get("control_rows") == 51
                        and terminal.get("native_control_transitions_checked") == 163200
                        and terminal.get("native_nominal_candidate_transitions_checked") == 78741504
                        and terminal.get("native_nominal_selected_transitions_checked") == 28800
                        and terminal.get("native_max_abs_error") == 0
                        and terminal.get("new_fits") == terminal.get("new_optimizer_steps") == terminal.get("audit_model_calls") == 0,
                        "Pinned successful scientific prerequisite and complete replay coverage")
    return plan, source


def _lineage(root, refs, settings, engineering):
    require(type(refs) is dict and set(refs) == {"cache", "prerequisite"}, "Both explicit parent roles required")
    cache_plan, cache_source = _parent(root, refs["cache"], kind="cache", engineering=engineering)
    parent, parent_source = _parent(root, refs["prerequisite"], kind="prerequisite", engineering=engineering)
    inherited = parent["cache_source"]
    require(all(inherited[key] == cache_source[key] for key in ("plan_sha256", "audit_receipt_sha256", "completed_sha256", "summary_sha256"))
            and all(inherited["members"].get(name) == digest for name, digest in cache_source["members"].items()), "Prerequisite used the exact same completed cache inputs")
    require(all(parent["sources"].get(name) == digest for name, digest in cache_plan["sources"].items()), "Unchanged cache sources within frozen90 closure")
    require(all(canonical(settings[key]) == canonical(cache_plan[key]) for key in TRAINING_FIELDS), "Original paired training recipe and dimensions must remain unchanged")
    return cache_plan, cache_source, parent, parent_source


def _sources(root, inherited):
    require(len(inherited) == 90 and not set(inherited) & set(SOURCE_PATHS_NEW), "Unchanged inherited90 plus additive sources only")
    result = {}
    for name, digest in inherited.items():
        checked(root, name, digest)
        result[name] = digest
    for name in SOURCE_PATHS_NEW:
        result[name] = sha(_path(root, name))
    require(len(result) == 116, "Complete116 scientific source closure")
    return result


def _history(root, parent, source, supplied, sources):
    fields = {"numpy", "torch", "descriptors", "literal_calls", "engineering_sources"}
    require(type(supplied) is dict and set(supplied) == fields, "Explicit exhaustive historical maps and source-bound literal calls required")
    prior = parent["random_stream_contract"]
    for kind in ("numpy", "torch"):
        require(type(supplied[kind]) is list and [identity(row) for row in supplied[kind]] == prior[f"prior_{kind}_registry_sha256"], "Exact ordered inherited registry inventory required")
    require(canonical(supplied["descriptors"]) == canonical(prior["priors"]), "Exact inherited descriptor inventory")
    bindings = prior["prior_bindings"]
    require(len(supplied["numpy"]) == bindings["numpy_count"] and len(supplied["torch"]) == bindings["torch_count"]
            and identity(prior["prior_numpy_registry_sha256"]) == bindings["numpy_hash_list_sha256"]
            and identity(prior["prior_torch_registry_sha256"]) == bindings["torch_hash_list_sha256"]
            and identity(supplied["descriptors"]) == bindings["descriptors_sha256"], "Original complete historical bindings")
    result = {"lineage_sha256": source["plan_sha256"], "numpy_priors": [], "torch_priors": [],
              "geometry_memory": [], "literal_calls": copy.deepcopy(supplied["literal_calls"])}

    def add(kind, name, values, recorded=None):
        if not values:  # Empty historical Torch registries have no consumed state.
            require(kind == "torch" and values == {}, "Only empty historical Torch maps may be omitted from state identities")
            return
        function = streams.generator_manifest if kind == "numpy" else streams.torch_generator_manifest
        actual = function(values)
        if recorded is not None:
            require(canonical(recorded) == canonical(actual), "Historical inline generator identity")
        result[f"{kind}_priors"].append({"name": name, "artifact_sha256": source["plan_sha256"], "registry": copy.deepcopy(values), "generators": actual})

    for kind in ("numpy", "torch"):
        for index, values in enumerate(supplied[kind]):
            add(kind, f"inherited/{kind}/{index}", values)
    for field in ("geometry_study_exclusions", "cache_study_exclusions", "memory_study_exclusions"):
        require(type(prior[field]) is list and prior[field], "Every inherited namespace exclusion must remain present")
        for index, row in enumerate(prior[field]):
            add("numpy", f"{field}/{index}/{row['namespace']}", row["registry"], row["generators"])
            if row.get("torch_registry"):
                add("torch", f"{field}/{index}/{row['namespace']}", row["torch_registry"], row["torch_generators"])
    for index, row in enumerate(prior["engineering_literal_exclusions"]):
        add("numpy", f"older-literal/{index}", row["numpy_registry"], row["generators"])
        if row.get("torch_registry"):
            add("torch", f"older-literal/{index}", row["torch_registry"], row["torch_generators"])
    own = {row["namespace"]: row for row in prior["namespace_exclusions"]}
    require(len(own) == 4 and prior["namespace"] not in own, "All other prior geometry-memory namespaces required")
    own[prior["namespace"]] = {"namespace": prior["namespace"], "registry": prior["registry"], "generators": prior["generators"]}
    require(set(own) == set(streams.GEOMETRY_MEMORY_NAMESPACES), "All five completed geometry-memory namespaces required")
    roles = protocol.role_manifest(protocol.settings(engineering=True))["root_roles"]
    for namespace in streams.GEOMETRY_MEMORY_NAMESPACES:
        row = own[namespace]
        values = dict(row["registry"])
        require(set(values) <= set(roles), "Known prior geometry-memory roles only")
        require(canonical(streams.generator_manifest(values)) == canonical(row["generators"]), "Prior geometry-memory root/child identities")
        # Only a completed engineering parent's own current registry may be
        # smaller. Expand its OLD namespace, never this study's scored namespace.
        if set(values) != set(roles):
            require(parent["engineering"] is True and namespace == prior["namespace"], "Historical full64 exclusions cannot be truncated")
            full = {role: int.from_bytes(hashlib.sha256(f"OpenJev/{namespace}/{role}".encode()).digest()[:8], "big") for role in roles}
            require(all(full[role] == value for role, value in values.items()), "Historical engineering role derivation")
            values = full
        result["geometry_memory"].append({"name": namespace, "artifact_sha256": source["plan_sha256"],
                                          "registry": values, "generators": streams.generator_manifest(values)})
    _map(supplied["engineering_sources"], "Separate engineering source map", allow_empty=True)
    require(not set(supplied["engineering_sources"]) & set(sources), "Separate engineering sources must not shadow scientific sources")
    for name, digest in supplied["engineering_sources"].items():
        checked(root, name, digest)
    indexed = {}
    for row in supplied["literal_calls"]:
        name = row["source_path"]
        require(name not in indexed, "Unique literal source attribution")
        indexed[name] = row
        require((sources | supplied["engineering_sources"]).get(name) == row["source_sha256"], "Every literal source must belong to authenticated scientific/engineering source closure")
        checked(root, name, row["source_sha256"])
    for name, required in ADDITIONAL_LITERAL_CALLS.items():
        require(name in indexed and all(indexed[name][f"{kind}_registry"].get(role) == value
                    for kind in ("numpy", "torch") for role, value in required[kind].items()), "New known literal engineering/constructor call omitted")
    return result


def _engineering_evidence(root, evidence, settings, sources, observed_runtime, cap_seconds, audit_cap_seconds, engineering):
    require(type(evidence) is dict, "Explicit engineering evidence dictionary required")
    if engineering:
        require(evidence == {}, "An engineering attempt is not its own scientific readiness evidence")
        return {}
    require(set(evidence) == {"rehearsal", "capacity"}, "Completed whole-tree rehearsal and measured capacity are mandatory before scored allocation")
    authenticated = {}
    for kind, reference in evidence.items():
        require(type(reference) is dict and set(reference) == {"path", "sha256"}, "Explicit engineering receipt reference")
        receipt = read_bound(root, reference["path"], reference["sha256"])
        require(receipt.get("schema") == f"reacher-two-observation-{kind}-qualification-v1"
                and receipt.get("status") == "completed" and receipt.get("engineering") is True
                and receipt.get("study") == protocol.STUDY
                and receipt.get("source_sha256") == sources and receipt.get("runtime") == observed_runtime,
                "Engineering qualification must bind complete current source/runtime and completed status")
        _map(receipt["files"], "Retained engineering evidence files")
        base = str(PurePosixPath(reference["path"]).parent)
        for name, digest in receipt["files"].items():
            checked(root, f"{base}/{name}", digest)
        _finite(receipt["wall_seconds"], "Measured engineering whole wall", positive=True)
        if kind == "rehearsal":
            require(receipt.get("new_fits") == 3 and receipt.get("inherited_models") == 6
                    and receipt.get("restored_final_models") == 9 and receipt.get("control_rows") == 42
                    and receipt.get("saved_output_audit_completed") is True
                    and receipt.get("native_replay_max_abs_error") == 0
                    and receipt.get("scientific_draws") == 0, "Complete retained whole-tree rehearsal required; a tiny fit-only test is insufficient")
        else:
            expected_shape = {key: settings[key] for key in ("hidden_size", "train_episodes", "batch_size", "steps", "control_episodes", "planning_horizon", "action_block")}
            require(receipt.get("full_shape") == expected_shape and receipt.get("coverage") == protocol.coverage(settings), "Capacity must cover the full target study shape/work")
            measurements = receipt["measured_components"]
            require(type(measurements) is dict and set(measurements) == {"training", "learned_control", "physics_references", "audit", "storage_and_hashing"}, "Actual measurements of all critical components required")
            for row in measurements.values():
                require(type(row) is dict and set(row) == {"wall_seconds", "units", "artifact"}
                        and type(row["units"]) is int and row["units"] > 0
                        and row["artifact"] in receipt["files"], "Measured component must reference retained raw evidence")
                _finite(row["wall_seconds"], "Measured component wall", positive=True)
            projection = receipt["projected_seconds"]
            require(type(projection) is dict and set(projection) == {"execution", "audit"}, "Explicit reviewed capacity projections")
            require(_finite(projection["execution"], "Projected execution", positive=True) < cap_seconds
                    and _finite(projection["audit"], "Projected audit", positive=True) < audit_cap_seconds, "Caps must exceed measured-study projections")
            require(_sha(receipt.get("review_sha256")) and receipt["review_sha256"] in receipt["files"].values(), "Independent capacity interpretation must be retained and hash-bound")
        authenticated[kind] = {**reference, "receipt": receipt}
    return authenticated


def _caps(cap_seconds, audit_cap_seconds):
    require(all(type(value) is int and value > 0 for value in (cap_seconds, audit_cap_seconds)), "Explicit positive integer execution/audit caps; no defaults")


def prepare(root, settings, *, lineage_refs, historical_registries, runtime, cap_seconds,
            audit_cap_seconds, engineering_evidence, engineering=False):
    """Authenticate and assemble an UNFROZEN envelope, without writing files.

    Engineering evidence is {} for engineering attempts. Scientific mode requires
    separately completed, externally hash-bound qualification receipts. Their
    schemas are checked here; this function never invents a timing cap or receipt.
    The caller must retain its exclusive started/failed preparation receipts.
    """
    require(type(engineering) is bool, "Explicit engineering mode")
    protocol.validate_settings(settings, engineering=engineering)
    _caps(cap_seconds, audit_cap_seconds)
    _runtime(runtime)
    cache_plan, cache_source, parent, parent_source = _lineage(root, lineage_refs, settings, engineering)
    sources = _sources(root, parent["sources"])
    require(runtime == cache_plan["runtime"] == parent["runtime"], "Current runtime differs from authenticated inheritance")
    evidence = _engineering_evidence(root, engineering_evidence, settings, sources, runtime, cap_seconds, audit_cap_seconds, engineering)
    history = _history(root, parent, parent_source, historical_registries, sources)
    contract = streams.stream_contract(settings, history=history)  # LAST: no scored derivation before readiness.
    plan = {"schema": SCHEMA, "study": protocol.STUDY, "engineering": engineering,
        "status": "prepared_unfrozen", "settings": copy.deepcopy(settings),
        "lineage": {"cache": cache_source, "prerequisite": parent_source}, "sources": sources,
        "runtime": copy.deepcopy(runtime), "historical_registries": copy.deepcopy(historical_registries),
        "random_stream_contract": contract, "cap_seconds": cap_seconds, "audit_cap_seconds": audit_cap_seconds,
        "engineering_evidence": evidence,
        "artifact_contract": {"version": ARTIFACT_VERSION, "payload_member_count": len(expected_members(settings)),
            "payload_members_sha256": identity(expected_members(settings)), "complete_member_count": len(expected_members(settings, include_completed=True)),
            "new_fits": 3, "inherited_models": 6, "restored_models": 9, "control_rows": 42,
            "lineage_sidecars": {key: list(value) for key, value in LINEAGE_SUFFIXES.items()}}}
    plan["content_sha256"] = identity(plan)
    return plan


@dataclass(frozen=True)
class ValidatedExperiment:
    settings: dict
    streams: streams.BoundStreams
    cache_plan: dict
    cache_source: dict
    prerequisite_plan: dict
    prerequisite_source: dict
    history: dict
    plan_sha256: str
    engineering: bool
    execution_authorized: bool


def _freeze(root, reference, plan, expected_plan_sha256, observed_runtime):
    require(type(reference) is dict and set(reference) == {"path", "sha256"}, "Externally authenticated scientific freeze required")
    freeze = read_bound(root, reference["path"], reference["sha256"])
    require(set(freeze) == {"schema", "status", "study", "engineering", "plan_sha256", "content_sha256", "source_sha256", "runtime", "engineering_evidence_sha256", "source_commit", "no_retry"}
            and freeze["schema"] == "reacher-two-observation-freeze-v1" and freeze["status"] == "frozen"
            and freeze["study"] == protocol.STUDY and freeze["engineering"] is False
            and freeze["plan_sha256"] == expected_plan_sha256 and freeze["content_sha256"] == plan["content_sha256"]
            and freeze["source_sha256"] == plan["sources"] and freeze["runtime"] == observed_runtime
            and freeze["engineering_evidence_sha256"] == identity(plan["engineering_evidence"])
            and isinstance(freeze["source_commit"], str) and len(freeze["source_commit"]) == 40
            and set(freeze["source_commit"]) <= set("0123456789abcdef") and freeze["no_retry"] is True,
            "Exact scientific freeze/source/readiness/no-retry binding")


def validate_plan(plan, expected_plan_sha256, *, root, runtime, engineering=False,
                  plan_path=None, plan_bytes=None, freeze_ref=None):
    """Reauthenticate bytes, complete inheritance, sources, readiness and RNGs.

    Supply exactly one original plan_path (relative to root) or plan_bytes, so the
    externally expected file SHA is actually checked. Production additionally
    needs a separately authenticated freeze receipt; engineering must not use it.
    Use this at study entry and exit, within the enclosing whole-run deadline.
    """
    require(type(engineering) is bool and _sha(expected_plan_sha256), "Explicit mode and external plan SHA")
    require((plan_path is None) != (plan_bytes is None), "Exactly one original plan path or bytes required")
    if plan_path is not None:
        incoming = Path(plan_path)
        if incoming.is_absolute():
            try:
                incoming = incoming.relative_to(Path(root).resolve(strict=True))
            except ValueError as error:
                raise ValueError("Plan must reside under the explicit repository root") from error
        payload = _path(root, incoming.as_posix()).read_bytes()
    else:
        require(type(plan_bytes) is bytes, "Original serialized plan bytes required")
        payload = plan_bytes
    require(hashlib.sha256(payload).hexdigest() == expected_plan_sha256
            and canonical(_decode(payload)) == canonical(plan), "External plan bytes and supplied object differ")
    require(type(plan) is dict and plan.get("schema") == SCHEMA and plan.get("study") == protocol.STUDY
            and plan.get("engineering") is engineering and plan.get("status") == "prepared_unfrozen"
            and plan.get("content_sha256") == identity({key: value for key, value in plan.items() if key != "content_sha256"}), "Sealed preparation envelope")
    ref_keys = {"plan_path", "plan_sha256", "audit_path", "audit_receipt_sha256", "execution_path"}
    refs = {key: {field: source[field] for field in ref_keys | ({"terminal_path", "terminal_sha256", "terminal_kind"} if key == "prerequisite" else set())}
            for key, source in plan["lineage"].items()}
    evidence = {key: {field: row[field] for field in ("path", "sha256")} for key, row in plan["engineering_evidence"].items()}
    expected = prepare(root, plan["settings"], lineage_refs=refs, historical_registries=plan["historical_registries"],
        runtime=runtime, cap_seconds=plan["cap_seconds"], audit_cap_seconds=plan["audit_cap_seconds"],
        engineering_evidence=evidence, engineering=engineering)
    require(canonical(expected) == canonical(plan), "Prepared lineage/sources/runtime/settings/streams changed")
    if engineering:
        require(freeze_ref is None, "Engineering cannot consume a scientific freeze")
    else:
        _freeze(root, freeze_ref, plan, expected_plan_sha256, runtime)
    cache_plan = read_bound(root, refs["cache"]["plan_path"], refs["cache"]["plan_sha256"])
    parent = read_bound(root, refs["prerequisite"]["plan_path"], refs["prerequisite"]["plan_sha256"])
    history = plan["random_stream_contract"]["history"]
    bound = streams.validate_stream_contract(plan["settings"], plan["random_stream_contract"], history=history)
    return ValidatedExperiment(copy.deepcopy(plan["settings"]), bound, cache_plan, copy.deepcopy(plan["lineage"]["cache"]),
        parent, copy.deepcopy(plan["lineage"]["prerequisite"]), copy.deepcopy(history), expected_plan_sha256, engineering, True)
