"""Prepare complete geometry-memory publication assets after final authorization.

Source-only preparation until explicitly invoked with completed audit/report and
previous verified release identities. No research imports, inference, native
replay, RNG, network or import-time file reads. No automatic retry or clobber.
The streaming primitives are copied from the reviewed geometry-score packager.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
STUDY = "reacher-geometry-memory-v1"
EXPECTED_PLAN = "23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee"
EXPECTED_READINESS = "4fb83c5eafc84e3522f5b0188277f06ac1cc6ef33f0f2bba56993e305a25cb95"
PREVIOUS_STUDY = "reacher-geometry-score-v1"
PREVIOUS_PLAN = "94c90f7585303f4edd88e1b0cb0dc8a488a2e2f07237aae4cfb489ade2068a2b"
PREVIOUS_AUDIT = "9cd5fdde29601ae5227afb9b3725be1fc855d4c7ba72aa7ca4c83248ae8de72e"
MAX_ASSET_BYTES = PART_BYTES = 1_400_000_000
CHUNK_BYTES = 8 * 1024 * 1024
ARMS = ("residual_gru", "encoded_current_gru", "cached_gru", "cached_mlp")
PAIRS = ("pair0", "pair1", "pair2")
PANELS = ("full", "ordinary", "shift")
PHYSICS = ("known_state", "particle", "public_kinematic")
REFERENCES = (*PHYSICS, "zero", "uniform")
FIT_FILES = ("initial-weights.pt", "training.jsonl", "weights.pt", "checkpoint.pt", "completed.json")
PREPARATION_TREES = ("output/reacher-geometry-memory-rehearsal-v1", "output/reacher-geometry-memory-capacity-v1")
SIDECARS = ("packaging-started.json", "manifest.json", "receipt.json", "README.md", "SHA256SUMS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'Nonfinite JSON number')
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def digest(value):
    require(isinstance(value, str) and len(value) == 64
            and set(value) <= set('0123456789abcdef'), 'External SHA-256 required')
    return value


def child(folder, name):
    require(isinstance(name, str) and name and '\\' not in name, 'Safe relative path')
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and all(part not in ('', '.', '..') for part in rel.parts)
            and rel.as_posix() == name, 'Canonical relative member path')
    result = Path(folder).joinpath(*rel.parts)
    require(not Path(folder).is_symlink() and not any(parent.is_symlink()
            for parent in [result, *result.parents] if parent != parent.parent), 'No symlink paths')
    return result


def checked(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing regular member: ' + str(path))
    require(sha(path) == digest(expected), 'Member hash changed: ' + str(path))
    return path


def files(folder):
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink(), 'Required real directory: ' + str(folder))
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'No symlinks in artifact tree')
        require(path.is_file() or path.is_dir(), 'No special artifact files')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = path
    return result


def bind_members(folder, members, extras=()):
    require(isinstance(members, dict), 'Member hash mapping required')
    actual = files(folder)
    require(set(actual) == set(members) | set(extras), 'Exact artifact membership: ' + str(folder))
    for name, expected in members.items():
        checked(child(folder, name), expected)
    return set(actual.values())


def positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, label)
    return value


def verify_archive(archive, manifest):
    expected = {row['path']: row for row in manifest}
    require(len(expected) == len(manifest), 'Unique archived member paths')
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            require(member.isfile() and member.name in expected and member.name not in seen,
                    'Exact regular archive member')
            require(member.name == PurePosixPath(member.name).as_posix()
                    and not PurePosixPath(member.name).is_absolute()
                    and '..' not in PurePosixPath(member.name).parts, 'Safe archive path')
            row = expected[member.name]
            require(member.size == row['bytes'] and member.mode == 0o644 and member.mtime == 0
                    and member.uid == member.gid == 0 and member.uname == member.gname == '',
                    'Archive member length and deterministic header')
            with tar.extractfile(member) as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256'],
                        'Archive member bytes')
            seen.add(member.name)
    require(seen == set(expected), 'Complete archive membership')


def bound_argument(value):
    require(isinstance(value, str) and '=' in value, 'Expected repository-relative path=SHA256')
    name, expected = value.rsplit('=', 1)
    return name, digest(expected)


def write_archive(root, paths, archive):
    rows = []
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        require(child(root, relative) == path and path.is_file(), 'Repository-contained regular input')
        rows.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    with (
        archive.open('xb') as raw,
        gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=1) as gz,
        tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar,
    ):
        for row in rows:
            path = root / row['path']
            require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                    'Input changed before archiving')
            info = tarfile.TarInfo(row['path'])
            info.size, info.mode, info.mtime = row['bytes'], 0o644, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with path.open('rb') as handle:
                tar.addfile(info, handle)
    verify_archive(archive, rows)
    return rows


def split_assets(archive, out, *, max_asset_bytes=MAX_ASSET_BYTES, part_bytes=PART_BYTES):
    require(0 < part_bytes <= max_asset_bytes < 2_000_000_000, 'Every asset below 2 GB')
    archive_bytes, archive_sha = archive.stat().st_size, sha(archive)
    if archive_bytes <= max_asset_bytes:
        return [{'name': archive.name, 'bytes': archive_bytes, 'sha256': archive_sha}], False
    parts = []
    with archive.open('rb') as source:
        while True:
            first = source.read(min(CHUNK_BYTES, part_bytes))
            if not first:
                break
            destination = out / f'{archive.name}.part{len(parts):03d}'
            count, remaining = 0, part_bytes
            with destination.open('xb') as handle:
                data = first
                while data:
                    handle.write(data)
                    count += len(data)
                    remaining -= len(data)
                    if remaining == 0:
                        break
                    data = source.read(min(CHUNK_BYTES, remaining))
            parts.append({'name': destination.name, 'bytes': count, 'sha256': sha(destination)})
    joined = hashlib.sha256()
    total = 0
    for part in parts:
        path = out / part['name']
        require(0 < path.stat().st_size == part['bytes'] <= part_bytes and sha(path) == part['sha256'],
                'Part identity and size')
        with path.open('rb') as handle:
            while data := handle.read(CHUNK_BYTES):
                joined.update(data)
                total += len(data)
    require(total == archive_bytes and joined.hexdigest() == archive_sha, 'Concatenated parts exactly recover archive')
    return parts, True


def fit_names():
    return [f"{arm}-{pair}" for pair in PAIRS for arm in ARMS]


def inherited_members():
    names = {"train.npz", "train.json"}
    names |= {f"{folder}/{pair}.{ext}" for folder in ("initializations", "orders")
              for pair in PAIRS for ext in ("pt", "json")}
    names |= {f"fits/{fit}/{name}" for fit in fit_names() for name in FIT_FILES}
    require(len(names) == 74, "Independent inheritance coverage")
    return names


def expected_members():
    """Independent filename-only coverage; no research module is imported."""
    names = {"started.json", "random-streams.json", "inheritance.json", "all-models-restored.json",
             "evaluation-started.json", "control-completed.json", "final-models.json", "costs.json"}
    names |= {"inherited/" + name for name in inherited_members()}
    names |= {f"inherited/{parent}-{part}.json" for parent in ("cache", "geometry")
              for part in ("plan", "audit", "completed", "summary")}
    names |= {f"model-states/{fit}-{phase}.pt" for fit in fit_names() for phase in ("before", "after")}
    names |= {f"innovations/control/{step:03d}.{ext}" for step in range(50) for ext in ("npz", "json")}
    for panel in PANELS:
        for label in (*fit_names(), *REFERENCES):
            prefix = f"control/{panel}/{label}"
            names |= {f"{prefix}/{name}" for name in ("episodes.npz", "episodes.json", "timings.json")}
            if label in fit_names():
                names |= {f"{prefix}/{name}" for name in ("executed_predictions.npz", "states.npz", "state-work.json")}
                folders = ("decisions", "scoring")
            else:
                names.add(f"{prefix}/planning.npz")
                folders = ()
                if label in PHYSICS:
                    names.add(f"{prefix}/physics-final.json")
                    folders = ("decisions", "physics")
                if label in ("particle", "public_kinematic"):
                    names.add(f"{prefix}/observer-final.json")
            names |= {f"{prefix}/{folder}/{step:03d}.{ext}" for folder in folders
                      for step in range(50) for ext in ("npz", "json")}
    require(len(names) == 9505, "Independent full scientific file coverage")
    return names


def validate_gate(gate):
    """Verify all retained identities and inequalities, without selecting methods."""
    names = set()
    for comparator in ("encoded_current_gru", "cached_gru"):
        for panel in ("ordinary", "shift"):
            names.add(f"gap_mean/{panel}/{comparator}")
            names |= {f"gap_pair/{panel}/{comparator}/{pair}" for pair in PAIRS}
        names.add(f"full_mean/{comparator}")
    names |= {f"competence/{panel}/residual_gru-{pair}" for panel in ("ordinary", "shift") for pair in PAIRS}
    names.add("competence/ordinary/known_state")
    checks = gate["checks"]
    require(len(checks) == len(names) == 25 and {row["name"] for row in checks} == names,
            "All original twenty-five check identities")
    for row in checks:
        require(row["comparison"] == "le" and type(row["passed"]) is bool
                and type(row["left"]) in (int, float) and type(row["right"]) in (int, float)
                and math.isfinite(row["left"]) and math.isfinite(row["right"])
                and row["passed"] is (row["left"] <= row["right"]), "Recorded gate arithmetic")
    require(type(gate["passed"]) is bool and gate["passed"] is all(row["passed"] for row in checks),
            "Completed audit may pass or fail the unchanged scientific gate")


def validate_weight_boundaries(execution, plan, summary):
    """Bind every weight/checkpoint boundary using the completed byte audit.

    Tensor/Adam semantics were independently audited already. Packaging does
    not deserialize checkpoints, instantiate students or claim a new tensor audit.
    """
    source = plan["cache_source"]
    require(set(source["members"]) == inherited_members(), "Exact twelve-fit inheritance")
    parent_done = read(execution / "inherited/cache-completed.json")
    parent_audit = read(execution / "inherited/cache-audit.json")
    restored = read(execution / "all-models-restored.json")
    final = read(execution / "final-models.json")
    require(set(restored["models"]) == set(final["student_tensor_sha256"]) == set(fit_names())
            and restored["optimizer_constructed"] is False and restored["new_updates"] == final["new_updates"] == 0,
            "All twelve unchanged deployment boundaries")
    for name, value in source["members"].items():
        require(parent_done["files"].get(name) == parent_audit["execution_members"].get(name) == value,
                "Each inherited byte was covered by the parent audit")
        checked(execution / "inherited" / name, value)
    for pair in PAIRS:
        for arm in ARMS:
            name = f"{arm}-{pair}"
            folder = execution / "inherited/fits" / name
            done, restored_fit, audited = read(folder / "completed.json"), restored["models"][name], summary["inherited_fits"][name]
            require(set(done["files"]) == set(FIT_FILES) - {"completed.json"}
                    and (done["name"], done["arm"], done["pair"]) == (name, arm, pair), "Actual original fit identity")
            for member, value in done["files"].items():
                checked(folder / member, value)
            tensor = digest(done["student_tensor_sha256"])
            require(restored_fit["kind"] == audited["arm"] == arm and audited["pair"] == pair
                    and restored_fit["model_class"] == plan["model_classes"][arm]
                    and restored_fit["student_tensor_sha256"] == audited["student_tensor_sha256"]
                    == final["student_tensor_sha256"][name] == tensor
                    and audited["observed_before_after_equal"] is True
                    and restored_fit["configuration"] == done["model_configuration"] == audited["model_configuration"],
                    "All saved actual-class and unchanged tensor identities")
            require(restored_fit["weights_sha256"] == sha(folder / "weights.pt")
                    and restored_fit["checkpoint_sha256"] == sha(folder / "checkpoint.pt")
                    and restored_fit["snapshot_path"] == f"model-states/{name}-before.pt", "Deployment source files")
            checked(execution / restored_fit["snapshot_path"], restored_fit["snapshot_sha256"])
            checked(execution / f"model-states/{name}-after.pt", final["files"][f"model-states/{name}-after.pt"])


def validate_scored(root, audit_sha, terminal_sha):
    # Authenticate the external completed audit before reading any outcome.
    folder = root / f"evidence/{STUDY}"
    audit_dir, execution = folder / "audit", root / f"runs/{STUDY}/attempt"
    audit = read(checked(audit_dir / "receipt.json", audit_sha))
    require(audit["status"] == "completed" and audit["version"] == STUDY
            and audit["engineering"] is False and audit["saved_output_only"] is True,
            "Completed independent scored audit required first")
    plan = read(checked(folder / "protocol/plan.json", EXPECTED_PLAN))
    readiness = read(checked(folder / "protocol/readiness.json", EXPECTED_READINESS))
    require(plan["study"] == STUDY and plan["engineering"] is False and len(plan["sources"]) == 90
            and plan["arms"] == list(ARMS) and plan["pairs"] == list(PAIRS)
            and plan["panels"] == list(PANELS) and plan["references"] == list(REFERENCES)
            and plan["score_modes"] == ["geometry"] and plan["fit_order"] == fit_names()
            and len(plan["execution_order"]) == 51 and plan["control_episodes"] == 64
            and plan["steps"] == 50 and plan["planning_horizon"] == 12 and plan["action_block"] == 3
            and plan["planner"] == plan["physics_planner"] == "cem256", "Exact frozen scientific coverage")
    require(readiness["status"] == "prepared_before_scored_execution" and readiness["plan_sha256"] == EXPECTED_PLAN
            and readiness["scored_random_samples_drawn"] is False
            and readiness["models_selected_by_capacity_performance"] is False
            and readiness["frozen_execution_cap_seconds"] == plan["cap_seconds"] == 5700
            and readiness["frozen_audit_cap_seconds"] == plan["audit_cap_seconds"] == 3300, "Original readiness and caps")
    paths = {checked(child(root, name), value) for name, value in plan["sources"].items()}
    require(set(files(folder / "protocol")) == {"plan.json", "readiness.json"}, "Exact protocol directory")
    paths |= set(files(folder / "protocol").values())
    done = read(checked(execution / "completed.json", audit["execution_completed_sha256"]))
    require(done["status"] == "completed" and done["study"] == STUDY
            and done["plan_sha256"] == audit["plan_sha256"] == EXPECTED_PLAN
            and audit["source_sha256"] == plan["sources"] and audit["runtime"] == plan["runtime"]
            and done["new_fits"] == done["diagnostic_roots"] == done["astra_calls"] == 0
            and done["restored_models"] == 12 and done["control_rows"] == 51, "Completed full execution identity")
    require(set(done["files"]) == expected_members() and audit["execution_members"] == done["files"],
            "Exact9505 scientific members and full checkpoint coverage")
    paths |= bind_members(execution, done["files"], ("completed.json",))
    require(set(audit["files"]) == {"summary.json", "README.md"}, "Exact independent audit outputs")
    paths |= bind_members(audit_dir, audit["files"], ("receipt.json",))
    summary = read(audit_dir / "summary.json")
    require(summary["status"] == "completed" and summary["version"] == STUDY and summary["engineering"] is False
            and summary["saved_output_only"] is True and summary["plan_sha256"] == EXPECTED_PLAN
            and summary["execution_completed_sha256"] == audit["execution_completed_sha256"]
            and summary["costs"] == audit["costs"] and summary["new_model_calls"]
            == summary["new_policy_calls"] == summary["new_fits"] == 0, "Authenticated saved-output summary")
    require(set(summary["inherited_fits"]) == set(fit_names()) and set(summary["control"]) == set(PANELS), "All fits/panels")
    for panel in PANELS:
        require(set(summary["control"][panel]) == set(fit_names()) | set(REFERENCES)
                and all(len(row["episode_costs"]) == 64 for row in summary["control"][panel].values()), "Every row and native case")
    counts = {"native_control_transitions_checked": 163200,
              "native_nominal_candidate_transitions_checked": 78741504,
              "native_nominal_selected_transitions_checked": 28800,
              "public_observer_transitions_checked": 3 * 64 * 49 * 33}
    require(all(summary[key] == count for key, count in counts.items()), "All separate native replay scopes")
    require(summary["phase_boundary"]["all_twelve_restored_before_fresh_controls"] is True,
            "Complete model restoration precedes fresh data")
    validate_gate(summary["continuation_gate"])
    validate_weight_boundaries(execution, plan, summary)
    require(positive(done["wall_seconds"], "Execution time") <= 5700
            and positive(summary["costs"]["audit_validation_wall_seconds"], "Audit time") <= 3300, "Both frozen caps respected")
    for kind in ("cache", "geometry"):
        source = plan[kind + "_source"]
        for suffix, value in (("plan", source["plan_sha256"]), ("audit", source["audit_receipt_sha256"]),
                              ("completed", source["completed_sha256"]), ("summary", source["summary_sha256"])):
            checked(execution / f"inherited/{kind}-{suffix}.json", value)
    started = read(execution / "started.json")
    launch, confirmation = read(folder / "launch.json"), read(folder / "launch-confirmation.json")
    require(readiness["prepared_unix_time"] <= launch["launch_unix_time"] <= started["unix_time"]
            and launch["status"] == "launch_requested_before_child_execution"
            and launch["plan_sha256"] == confirmation["plan_sha256"] == EXPECTED_PLAN
            and launch["readiness_sha256"] == EXPECTED_READINESS and launch["source_snapshot"] == plan["sources"]
            and launch["automatic_retry"] is False and launch["execution_cap_seconds"] == 5700
            and launch["audit_cap_seconds"] == 3300, "Original prelaunch history retained")
    require(confirmation["status"] == "observed_running" and confirmation["restored_checkpoints"] == 12
            and confirmation["all_models_restored_sha256"] == sha(execution / "all-models-restored.json")
            and confirmation["evaluation_started_sha256"] == sha(execution / "evaluation-started.json"), "Original confirmation bound")
    terminal_path = checked(folder / "terminal-verification.json", terminal_sha)
    terminal = read(terminal_path)
    require(terminal["status"] == "completed" and terminal["plan_sha256"] == EXPECTED_PLAN
            and terminal["execution_completed_sha256"] == audit["execution_completed_sha256"]
            and terminal["audit_receipt_sha256"] == audit_sha and terminal["audit_summary_sha256"] == audit["files"]["summary.json"]
            and terminal["execution_exit_code"] == terminal["audit_exit_code"] == 0
            and terminal["bound_sources_verified"] == 90 and terminal["restored_checkpoints"] == 12
            and terminal["control_rows"] == 51 and terminal["new_fits"] == terminal["new_optimizer_steps"]
            == terminal["audit_model_calls"] == 0 and terminal["total_checks"] == 25
            and terminal["qualification_passed"] is summary["continuation_gate"]["passed"]
            and terminal["checks_passed"] == sum(row["passed"] for row in summary["continuation_gate"]["checks"]), "Externally verified terminal")
    for key in counts.keys() - {"public_observer_transitions_checked"}:
        require(terminal[key] == summary[key], "Terminal coverage " + key)
    require(terminal["native_max_abs_error"] == summary["native_max_abs_error"]
            and terminal["execution_wall_seconds"] == done["wall_seconds"]
            and terminal["audit_wall_seconds"] == summary["costs"]["audit_validation_wall_seconds"], "Terminal numerical identity")
    paths |= {path for path in folder.iterdir() if path.is_file()}
    require(all(not path.is_symlink() for path in paths), "No symlink metadata")
    return plan, audit, summary, readiness, paths


def validate_reporting(root, directory, expected, plan, audit, summary, audit_sha):
    receipt = read(checked(directory / "receipt.json", expected))
    inputs = {"plan_sha256": EXPECTED_PLAN, "audit_receipt_sha256": audit_sha,
              "audit_summary_sha256": audit["files"]["summary.json"],
              "execution_completed_sha256": audit["execution_completed_sha256"],
              "execution_member_count": 9505, "frozen_source_count": 90}
    require(receipt["status"] == "completed" and receipt["study"] == STUDY
            and receipt["engineering"] is False and receipt["inputs"] == inputs
            and receipt["scope"] == "saved-artifact reporting and fixed-case schematic replay only"
            and receipt["source_sha256"] == plan["sources"]
            and receipt["new_model_calls"] == receipt["new_policy_calls"] == receipt["new_native_calls"] == 0
            and receipt["control_rows"] == 51 and receipt["learned_rows"] == 36
            and receipt["reference_rows"] == 15 and receipt["inherited_models"] == 12,
            "Completed report covers all authenticated scientific inputs")
    # The external receipt fixes exact file membership; future output hashes are
    # never guessed here. Minimum named outputs prevent an empty report receipt.
    required = {"report.json", "tables.md", "fixed-case-replay.gif", "replay-preview.png"}
    require(required <= set(receipt["files"]), "Numeric report, tables and fixed replay are mandatory")
    require(all(PurePosixPath(name).parent == PurePosixPath(".") for name in receipt["files"]),
            "Reporting files are flat named artifacts")
    paths = bind_members(directory, receipt["files"], ("receipt.json",))
    paths.add(checked(root / f"output/{STUDY}/render.py", receipt["renderer_source_sha256"]))
    report, gate = read(directory / "report.json"), summary["continuation_gate"]
    require(report["status"] == "completed" and report["study"] == STUDY and report["engineering"] is False
            and report["inputs"] == inputs and report["continuation_gate"] == gate
            and report["costs"] == summary["costs"] and report["replay"] == receipt["replay"]
            and report["paired_descriptive_comparisons"] == summary["paired_descriptive_comparisons"]
            and report["new_model_calls"] == report["new_native_calls"] == 0
            and receipt["gate_passed"] is gate["passed"]
            and receipt["checks_passed"] == sum(row["passed"] for row in gate["checks"]),
            "Report retains the unchanged scientific gate, costs and limitations")
    rows = report["control_rows"]
    require(len(rows) == 51 and {(row["panel"], row["policy"]) for row in rows}
            == {(panel, label) for panel in PANELS for label in (*fit_names(), *REFERENCES)},
            "Report has all twelve fits and five references on three panels")
    for row in rows:
        source = summary["control"][row["panel"]][row["policy"]]
        require(row["mean_native_cost"] == source["mean_cost"], "Report means match independent audit")
    require(report["work_counts"] == receipt["work_counts"], "Report work-count identity")
    replay = receipt["replay"]
    require(set(replay["models"]) == {f"{arm}-pair0" for arm in ARMS}
            and replay["frames"] == 50 and replay["frame_duration_ms"] == 100
            and replay["playback_seconds"] == 5 and replay["native_episode_seconds"] == 1
            and replay["native_pixels"] is False, "All four preselected schematic policies")
    for label, binding in replay["models"].items():
        require(binding["case_index"] == 0 and binding["pair"] == "pair0"
                and binding["panel"] == "ordinary", "Replay selection was fixed before results")
        for key, extension in (("episodes_npz_sha256", "npz"), ("episodes_json_sha256", "json")):
            require(binding[key] == audit["execution_members"][f"control/ordinary/{label}/episodes.{extension}"],
                    "Replay draws only from audited saved native records")
        require(math.isclose(binding["native_cost"], summary["control"]["ordinary"][label]["episode_costs"][0],
                             rel_tol=1e-12, abs_tol=1e-12), "Replay caption reports the fixed case")
    return paths


def previous_release_dependency(root, verification_path, expected, package_directory, plan):
    """Require the existing verified release; never silently use a local fallback.

    This authenticates a saved publication attestation and every small sidecar.
    It does not make network requests or claim fresh remote reachability.
    The earlier large archives remain explicit external lineage dependencies.
    """
    verification = read(checked(verification_path, expected))
    require(verification["status"] == "published_and_verified"
            and verification["repository"] == "kw2828/OpenJev"
            and verification["release_tag"] == "research-reacher-geometry-score-v1"
            and verification["target_commit"] == verification["resolved_tag_commit"]
            and verification["release_url"] == "https://github.com/kw2828/OpenJev/releases/tag/research-reacher-geometry-score-v1"
            and verification["audit_receipt_sha256"] == PREVIOUS_AUDIT
            and verification["new_model_calls"] == verification["new_native_calls"] == 0,
            "Mandatory previously published and verified geometry release")
    receipt_path = checked(package_directory / "receipt.json", verification["package_receipt_sha256"])
    receipt = read(receipt_path)
    manifest_path = checked(package_directory / "manifest.json", receipt["manifest_sha256"])
    manifest = read(manifest_path)
    require(receipt["status"] == "verified" and receipt["study"] == manifest["study"] == PREVIOUS_STUDY
            and receipt["plan_sha256"] == plan["geometry_source"]["plan_sha256"] == PREVIOUS_PLAN
            and receipt["audit_receipt_sha256"] == plan["geometry_source"]["audit_receipt_sha256"] == PREVIOUS_AUDIT
            and receipt["execution_completed_sha256"] == plan["geometry_source"]["completed_sha256"]
            and len(receipt["source_sha256"]) == 80
            and all(plan["sources"].get(name) == value for name, value in receipt["source_sha256"].items())
            and verification["scientific_status"] == receipt["scientific_status"],
            "Previous verified release is the exact completed geometry ancestor")
    bundles = manifest["archives"]
    require(len(bundles) == 2 and receipt["archives"] == [
        {key: value for key, value in bundle.items() if key != "members"} for bundle in bundles],
        "Previous manifest and receipt archive identity")
    assets = receipt["release_assets"]
    require(len({row["name"] for row in assets}) == len(assets), "Previous release asset names are unique")
    by_name = {row["name"]: row for row in assets}
    prep_members = {}
    for bundle in bundles:
        rows = bundle["members"]
        require(len(rows) == bundle["member_count"] == len({row["path"] for row in rows})
                and bundle["all_members_reopened_and_verified"] is True,
                "Every original archive member was reopened and checked")
        digest(bundle["sha256"])
        for row in rows:
            child(root, row["path"])
            digest(row["sha256"])
            require(type(row["bytes"]) is int and row["bytes"] >= 0, "Original member byte count")
        ordered = bundle["ordered_release_assets"]
        require(ordered and len(set(ordered)) == len(ordered)
                and all(name in by_name for name in ordered)
                and sum(by_name[name]["bytes"] for name in ordered) == bundle["bytes"]
                and all(by_name[name]["bundle"] == bundle["name"] for name in ordered),
                "Previous split assets recover the original archive length")
        if bundle["split_for_release"]:
            require(bundle["concatenation_verified"] is True, "Previous archive reassembly was verified")
        else:
            require(len(ordered) == 1 and by_name[ordered[0]]["sha256"] == bundle["sha256"],
                    "Previous unsplit archive identity")
        if bundle["scope"] == "engineering_only_no_effectiveness_claim":
            prep_members = {row["path"]: row for row in rows}
    require(receipt["member_count"] == sum(bundle["member_count"] for bundle in bundles)
            and set(by_name) == {name for bundle in bundles for name in bundle["ordered_release_assets"]},
            "Previous archive coverage is complete")
    ancestor_failures = (
        "output/reacher-geometry-capacity-v1/attempt/execution/failed.json",
        "output/reacher-geometry-rehearsal-v1/attempt-01/audit/failed.json",
    )
    require(all(name in prep_members for name in ancestor_failures), "Previous release retains both original failed preparations")
    paths = {verification_path, *(package_directory / name for name in SIDECARS)}
    expected_remote = {}
    for row in assets:
        child(package_directory, row["name"])
        require(type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_ASSET_BYTES,
                "Previously published assets satisfy release size bound")
        expected_remote[row["name"]] = {"bytes": row["bytes"], "sha256": digest(row["sha256"])}
    for name in SIDECARS:
        path = child(package_directory, name)
        require(path.is_file(), "Every original publication sidecar required")
        expected_remote[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    remote = verification["assets"]
    require(len(remote) == len(expected_remote) == len({row["name"] for row in remote})
            and {row["name"] for row in remote} == set(expected_remote), "Exact previously verified remote asset set")
    for row in remote:
        require(all(row[key] == expected_remote[row["name"]][key] for key in ("bytes", "sha256"))
                and row["browser_download_url"].startswith(
                    "https://github.com/kw2828/OpenJev/releases/download/research-reacher-geometry-score-v1/"),
                "Every prior remote asset matches its original digest and size")
    downloads = verification["public_sidecar_downloads"]
    require(len(downloads) == 3 and {row["name"] for row in downloads} == {"receipt.json", "manifest.json", "SHA256SUMS"}
            and all(all(row[key] == expected_remote[row["name"]][key] for key in ("bytes", "sha256")) for row in downloads),
            "Previous public sidecars were downloaded and authenticated")
    sums = {}
    for line in (package_directory / "SHA256SUMS").read_text().splitlines():
        value, name = line.split("  ", 1)
        require(name not in sums, "Unique previous checksum entry")
        sums[name] = digest(value)
    require(set(sums) == set(expected_remote) - {"SHA256SUMS"}
            and all(value == expected_remote[name]["sha256"] for name, value in sums.items()), "Previous exact checksum sidecar")
    dependency = {"mode": "mandatory_previously_verified_release", "verification_sha256": expected,
        "verification_path": verification_path.relative_to(root).as_posix(),
        "repository": verification["repository"], "release_id": verification["release_id"],
        "release_url": verification["release_url"], "release_tag": verification["release_tag"],
        "target_commit": verification["target_commit"], "resolved_tag_commit": verification["resolved_tag_commit"],
        "published_at": verification["published_at"], "publisher_sha256": verification["publisher_sha256"],
        "verified_utc": verification["verified_utc"], "package_receipt_sha256": sha(receipt_path),
        "manifest_sha256": sha(manifest_path), "archives": receipt["archives"],
        "release_assets": remote, "retained_failed_ancestors": {name: prep_members[name] for name in ancestor_failures},
        "limit": "Previous complete preparation archives remain separately published dependencies. This package carries their authenticated manifests and publication receipt, not duplicate large archive bytes. No fresh network check or offline complete historical closure is claimed."}
    return paths, dependency


def preparation_files(root, readiness, sources):
    """Authenticate and retain the entire two new preparation trees.

    Any additional attempt must first be explicitly bound into a revised,
    externally authenticated publication contract. It cannot be silently omitted
    or treated as an authenticated frozen preparation.
    """
    paths = set()
    for tree in PREPARATION_TREES:
        folder = child(root, tree)
        require({path.name for path in folder.iterdir() if path.is_dir()} == {"attempt-01"},
                "Unexpected preparation attempt requires an explicit publication-contract update")
        paths |= set(files(folder).values())
    paths |= {checked(child(root, name), expected) for name, expected in readiness["inputs"].items()}
    rehearsal = root / PREPARATION_TREES[0] / "attempt-01"
    capacity = root / PREPARATION_TREES[1] / "attempt-01"
    terminals = (rehearsal / "execution/completed.json", rehearsal / "audit/receipt.json", capacity / "execution/completed.json")
    for path in terminals:
        relative = path.relative_to(root).as_posix()
        terminal = read(checked(path, readiness["inputs"][relative]))
        require(terminal["status"] == "completed", "Original preparation completed state")
        mapping = terminal["files"]
        hashes = {name: value["sha256"] if isinstance(value, dict) else value for name, value in mapping.items()}
        require(bind_members(path.parent, hashes, (path.name,)) <= paths, "All completed engineering members retained")
        for name, value in mapping.items():
            if isinstance(value, dict):
                require(child(path.parent, name).stat().st_size == value["bytes"], "Engineering member byte count")
    rehearsal_plan = read(rehearsal / "plan.json")
    require(rehearsal_plan["engineering"] is True and rehearsal_plan["sources"] == sources
            and rehearsal_plan["control_episodes"] == 1
            and rehearsal_plan["rng_namespace"] == "reacher-geometry-memory-engineering-whole-tree-v1", "Exact engineering rehearsal identity")
    fixture_name = "tests/reacher_geometry_memory_fixture.py"
    snapshots = {**sources, fixture_name: rehearsal_plan["fixture_source_sha256"]}
    require(bind_members(rehearsal / "source-snapshot", snapshots) <= paths, "Exact ninety sources plus fixture snapshots")
    paths.add(checked(child(root, fixture_name), rehearsal_plan["fixture_source_sha256"]))
    rehearsal_audit = read(rehearsal / "audit/receipt.json")
    require(rehearsal_audit["engineering"] is True and rehearsal_audit["source_sha256"] == sources
            and rehearsal_audit["plan_sha256"] == sha(rehearsal / "plan.json")
            and rehearsal_audit["execution_completed_sha256"] == sha(rehearsal / "execution/completed.json"),
            "Rehearsal audit belongs to its completed engineering execution")
    capacity_done = read(capacity / "execution/completed.json")
    capacity_plan = read(capacity / "execution/capacity-plan.json")
    binding = read(capacity / "execution/source-binding.json")
    expected_sources = {**sources, "output/reacher-geometry-memory-v1/capacity_probe.py":
                        readiness["inputs"]["output/reacher-geometry-memory-v1/capacity_probe.py"]}
    require(capacity_plan["engineering"] is True and capacity_plan["sources"] == sources
            and capacity_plan["rng_namespace"] == "reacher-geometry-memory-engineering-capacity-v1"
            and capacity_done["source_sha256"] == binding["sources"] == expected_sources
            and capacity_done["plan_sha256"] == sha(capacity / "execution/capacity-plan.json")
            and capacity_done["new_fits"] == capacity_done["optimizer_updates"] == 0
            and capacity_done["outcome_summaries_computed"] is False, "Unchanged synthetic capacity scope")
    require(bind_members(capacity / "source-snapshot", expected_sources) <= paths, "Exact capacity source snapshots")
    capacity_audit_path = capacity / "audit/completed.json"
    capacity_audit = read(checked(capacity_audit_path, readiness["inputs"][capacity_audit_path.relative_to(root).as_posix()]))
    require(capacity_audit["status"] == "completed" and capacity_audit["scope"] == capacity_done["scope"]
            and capacity_audit["execution_completed_sha256"] == sha(capacity / "execution/completed.json")
            and capacity_audit["new_model_calls"] == capacity_audit["new_fits"] == 0
            and capacity_audit["outcome_summaries_computed"] is False
            and set(files(capacity / "audit")) == {"started.json", "completed.json"}, "Complete capacity audit scope")
    paths |= {checked(child(root, name), value) for name, value in sources.items()}
    paths.add(root / f"evidence/{STUDY}/protocol/readiness.json")
    return paths


def archive_bundle(root, paths, out, name, scope):
    archive = out / name
    rows = write_archive(root, paths, archive)
    assets, split = split_assets(archive, out)
    for asset in assets:
        asset["bundle"], asset["scope"] = name, scope
    return {"name": name, "scope": scope, "bytes": archive.stat().st_size, "sha256": sha(archive),
        "member_count": len(rows), "members": rows, "all_members_reopened_and_verified": True,
        "split_for_release": split, "concatenation_verified": split,
        "ordered_release_assets": [row["name"] for row in assets]}, assets


def storage_preflight(out, collections, readiness):
    # Conservative gzip expansion plus tar padding, complete local archives,
    # duplicate release segments and an additional 8 GiB reserve. Input raw files
    # already exist and are never removed to free space.
    uncompressed = sum(path.stat().st_size + 1024 for paths in collections for path in paths)
    archive_bound = math.ceil(1.01 * (uncompressed + 10240 * len(collections))) + 2**20
    required = max(readiness["required_free_disk_bytes"], 2 * archive_bound + 8 * 2**30)
    available = shutil.disk_usage(out).free
    require(available >= required, "Insufficient space for complete verified archives and split copies")
    return {"input_bytes_with_header_allowance": uncompressed, "conservative_archive_bytes": archive_bound,
        "required_free_bytes": required, "observed_free_bytes": available,
        "scope": "Preflight bound, not measured final compression or a disk reservation"}


def package(args):
    require(args.completed_authorized is True, "Explicit completed-study packaging authorization required before any reads")
    out = child(ROOT, args.out)
    protected = [ROOT / f"runs/{STUDY}/attempt", ROOT / f"evidence/{STUDY}",
                 child(ROOT, args.reporting_directory), child(ROOT, args.previous_package_directory),
                 *(child(ROOT, tree) for tree in PREPARATION_TREES)]
    require(not out.exists() and not any(out.is_relative_to(path) or path.is_relative_to(out) for path in protected),
            "Exclusive output must be separate from every input tree")
    out.mkdir(parents=True, exist_ok=False)
    stage = "authorized-preflight"
    try:
        package_source_sha = sha(Path(__file__))
        write(out / "packaging-started.json", {"scope": "Authorized packaging only; scientific inputs not yet authenticated",
            "requested_plan_sha256": args.expected_plan_sha256,
            "requested_audit_receipt_sha256": args.expected_audit_receipt_sha256,
            "requested_reporting_receipt_sha256": args.expected_reporting_receipt_sha256,
            "requested_terminal_verification_sha256": args.expected_terminal_verification_sha256,
            "requested_previous_release_verification_sha256": args.expected_previous_release_verification_sha256,
            "package_source_sha256": package_source_sha, "automatic_retry": False,
            "started_utc": datetime.datetime.now(datetime.UTC).isoformat()})
        require(args.expected_plan_sha256 == EXPECTED_PLAN, "Exact frozen plan required")
        audit_sha = digest(args.expected_audit_receipt_sha256)
        report_sha = digest(args.expected_reporting_receipt_sha256)
        terminal_sha = digest(args.expected_terminal_verification_sha256)
        previous_sha = digest(args.expected_previous_release_verification_sha256)
        stage = "authenticate-completed-scientific-evidence"
        plan, audit, summary, readiness, scientific = validate_scored(ROOT, audit_sha, terminal_sha)
        stage = "authenticate-completed-report"
        scientific |= validate_reporting(ROOT, child(ROOT, args.reporting_directory), report_sha, plan, audit, summary, audit_sha)
        stage = "authenticate-mandatory-previous-release"
        previous_files, previous_dependency = previous_release_dependency(ROOT,
            child(ROOT, args.previous_release_verification), previous_sha, child(ROOT, args.previous_package_directory), plan)
        scientific |= previous_files
        stage = "authenticate-all-new-preparation"
        preparation = preparation_files(ROOT, readiness, plan["sources"])
        scientific |= {child(ROOT, name) for name in ("LICENSE", "pyproject.toml", "uv.lock")}
        scientific.add(Path(__file__).resolve())
        for declaration in args.reporting_file:
            name, expected = bound_argument(declaration)
            scientific.add(checked(child(ROOT, name), expected))
        stage = "storage-preflight"
        storage = storage_preflight(out, (scientific, preparation), readiness)
        stage = "archive-scored-evidence"
        scored_bundle, scored_assets = archive_bundle(ROOT, scientific, out,
            "reacher-geometry-memory-scored-execution-and-audit.tar.gz", "completed_scored_study_all_results")
        stage = "archive-engineering-preparation"
        prep_bundle, prep_assets = archive_bundle(ROOT, preparation, out,
            "reacher-geometry-memory-engineering-preparation-all-attempts.tar.gz", "engineering_only_no_effectiveness_claim")
        stage = "verify-manifest-and-sidecars"
        bundles, assets = [scored_bundle, prep_bundle], [*scored_assets, *prep_assets]
        require(len({row["name"] for row in assets}) == len(assets), "Unique release asset names")
        write(out / "manifest.json", {"study": STUDY, "archives": bundles,
            "previous_release_dependency": previous_dependency,
            "historical_dependency_limit": "Complete current execution and all new preparation are local. Earlier complete preparation, including failed ancestors, remains bound to the mandatory verified previous release. This is a verification bundle, not an offline complete runnable historical closure.",
            "supervision_limit": "Original prelaunch, confirmation and externally verified terminal records are retained; no retrospective supervision records are invented."})
        for bundle in bundles:
            for row in bundle["members"]:
                path = checked(child(ROOT, row["path"]), row["sha256"])
                require(path.stat().st_size == row["bytes"], "Input size unchanged after streamed verification")
        # Recheck the external trust roots after serialization, closing the
        # validation-to-archive mutation gap rather than only comparing the
        # originals with whichever bytes happened to be archived.
        stage = "reauthenticate-all-input-bindings"
        again = validate_scored(ROOT, audit_sha, terminal_sha)
        require(again[:4] == (plan, audit, summary, readiness), "Scientific trust roots stayed unchanged")
        validate_reporting(ROOT, child(ROOT, args.reporting_directory), report_sha, plan, audit, summary, audit_sha)
        repeated_previous = previous_release_dependency(ROOT, child(ROOT, args.previous_release_verification),
            previous_sha, child(ROOT, args.previous_package_directory), plan)
        require(repeated_previous == (previous_files, previous_dependency), "Previous release bindings stayed unchanged")
        require(preparation_files(ROOT, readiness, plan["sources"]) == preparation, "All preparation inputs stayed unchanged")
        for declaration in args.reporting_file:
            name, expected = bound_argument(declaration)
            checked(child(ROOT, name), expected)
        require(sha(Path(__file__)) == package_source_sha, "Package source unchanged throughout this attempt")
        stage = "write-final-sidecars"
        gate = summary["continuation_gate"]
        receipt = {"status": "verified", "study": STUDY, "plan_sha256": EXPECTED_PLAN,
            "readiness_sha256": EXPECTED_READINESS, "audit_receipt_sha256": audit_sha,
            "reporting_receipt_sha256": report_sha, "terminal_verification_sha256": terminal_sha,
            "previous_release_verification_sha256": previous_sha,
            "execution_completed_sha256": audit["execution_completed_sha256"], "source_sha256": plan["sources"],
            "package_source_sha256": package_source_sha, "manifest_sha256": sha(out / "manifest.json"),
            "archives": [{key: value for key, value in bundle.items() if key != "members"} for bundle in bundles],
            "member_count": sum(bundle["member_count"] for bundle in bundles), "release_assets": assets,
            "storage_preflight": storage, "previous_release_dependency": previous_dependency,
            "raw_coverage": {"execution_manifest_members": 9505, "inherited_fits": 12, "new_fits": 0,
                "control_rows": 51, "control_cases_per_row": 64, "diagnostic_roots": 0, "frozen_source_count": 90,
                "native_control_transitions_checked": 163200, "native_nominal_candidate_transitions_checked": 78741504,
                "native_nominal_selected_transitions_checked": 28800, "public_observer_transitions_checked": 310464},
            "scientific_status": {"continuation_passed": gate["passed"],
                "checks_passed": sum(row["passed"] for row in gate["checks"]), "checks_total": 25, "packaging_changes_gate": False},
            "new_model_calls": 0, "new_native_calls": 0, "new_optimizer_steps": 0,
            "uploaded": False, "release_created": False, "verified_utc": datetime.datetime.now(datetime.UTC).isoformat()}
        write(out / "receipt.json", receipt)
        with (out / "README.md").open("x") as handle:
            handle.write("# Verified geometry-memory study assets\n\n"
                "Every result is retained whether the scientific continuation gate passes or fails. The full 9,505-member execution, twelve inherited models, 51 control rows, 90 frozen sources, independent audit and completed report/replay are included. No new fit, policy call or native step occurs during packaging. No upload has occurred.\n\n"
                "The separate engineering archive contains both complete new preparation trees and their exact source snapshots. Earlier failed preparation ancestors remain in the mandatory, previously verified geometry release identified by manifest.json. Those large historical archives are not duplicated here. This is a verification bundle with explicit external dependencies, not an offline complete runnable history.\n\n"
                "For a split archive, concatenate its ordered_release_assets from receipt.json in the given order. These are consecutive byte segments, not separate tar archives. Verify the resulting compressed archive size and SHA-256, then inspect its gzip tar contents against manifest.json. Every uncompressed member was reopened, streamed and hashed; all split bytes were checked to recover the complete archive. Every release asset is at most 1,400,000,000 bytes.\n\n"
                "The prior publication receipt records remote validation at its stated time. This helper performs no network request and does not claim current remote reachability. Original failures and partial packaging bytes are retained; there is no automatic retry or overwrite.\n")
        sum_paths = [out / name for name in SIDECARS if name != "SHA256SUMS"] + [out / row["name"] for row in assets]
        with (out / "SHA256SUMS").open("x") as handle:
            handle.write("".join(f"{sha(path)}  {path.name}\n" for path in sum_paths))
        require(all((out / row["name"]).stat().st_size == row["bytes"] and sha(out / row["name"]) == row["sha256"]
                    for row in assets), "Final release-asset byte verification")
        return receipt
    except BaseException as error:
        if (out / "receipt.json").exists():
            (out / "receipt.json").rename(out / "incomplete-receipt.json")
        write(out / "packaging-failed.json", {"status": "failed", "stage": stage, "error": repr(error),
            "scope": "Packaging only; all partial archives and original attempts preserved",
            "plan_sha256": args.expected_plan_sha256, "new_model_calls": 0, "new_native_calls": 0,
            "uploaded": False, "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-authorized", action="store_true")
    parser.add_argument("--expected-plan-sha256", default=EXPECTED_PLAN)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--expected-reporting-receipt-sha256", required=True)
    parser.add_argument("--expected-terminal-verification-sha256", required=True)
    parser.add_argument("--expected-previous-release-verification-sha256", required=True)
    parser.add_argument("--reporting-directory", default=f"evidence/{STUDY}/report")
    parser.add_argument("--previous-release-verification", default=f"evidence/{PREVIOUS_STUDY}/scored-publication/release-verification.json")
    parser.add_argument("--previous-package-directory", default=f"output/{PREVIOUS_STUDY}/publication-v1")
    parser.add_argument("--reporting-file", action="append", default=[], help="Additional final prose: repository-relative-path=SHA256")
    parser.add_argument("--out", default=f"output/{STUDY}/publication-v1")
    args = parser.parse_args()
    receipt = package(args)
    print(json.dumps({"status": receipt["status"], "release_assets": receipt["release_assets"],
        "receipt_sha256": sha(child(ROOT, args.out) / "receipt.json")}))


if __name__ == "__main__":
    main()
