"""Tiny synthetic packaging contracts. Never opens any study run or receipt."""

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("package_publication.py")
SPEC = importlib.util.spec_from_file_location("cache_publication_synthetic", MODULE_PATH)
pkg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pkg)


class Fixture:
    def __init__(self, root):
        self.root = root
        study = pkg.STUDY
        self.protocol = root / f"evidence/{study}/protocol"
        self.audit_folder = root / f"evidence/{study}/audit"
        self.execution = root / f"runs/{study}/attempt"
        self.figures = root / f"evidence/{study}/figures"
        self.replay_folder = root / f"evidence/{study}/replay"
        output = root / f"output/{study}"
        self.launcher = output / "launch.py"
        self.put(self.launcher, b"synthetic launcher only")
        for name in ("LICENSE", "pyproject.toml", "uv.lock"):
            self.put(root / name, b"synthetic dependency only")
        for name in ("report_results.py", "render_replay.py"):
            self.put(output / name, b"synthetic renderer only")
        self.packager = output / "package_publication.py"
        self.put(self.packager, MODULE_PATH.read_bytes())
        sources = {}
        for index in range(70):
            path = root / f"synthetic-source/{index:02d}.py"
            self.put(path, f"# synthetic source {index}\n".encode())
            sources[path.relative_to(root).as_posix()] = pkg.sha(path)
        self.plan = {"study": study, "engineering": False, "synthetic_test_only": True,
            "sources": sources, "runtime": {"native_xml_sha256": "a" * 64},
            "prediction_episodes": 96, "control_episodes": 64, "train_episodes": 768,
            "epochs": 48, "batch_size": 32, "steps": 50, "cap_seconds": 3600,
            "audit_cap_seconds": 300, "arms": list(pkg.ARMS), "pairs": list(pkg.PAIRS),
            "panels": list(pkg.PANELS), "references": list(pkg.REFERENCES)}
        self.save(self.protocol / "plan.json", self.plan)
        self.plan_sha = pkg.sha(self.protocol / "plan.json")
        validation = self.put(root / "synthetic-validation/check.json", b"synthetic prelaunch check")
        self.freeze = {"status": "frozen_before_scored_execution", "plan_sha256": self.plan_sha,
            "source_count": 70, "inherited_sources_unchanged": 58, "runtime": self.plan["runtime"],
            "no_evaluation_outcomes_used_for_freeze": True, "scored_execution_started": False,
            "execution_cap_seconds": 3600, "audit_cap_seconds": 300,
            "launcher_sha256": pkg.sha(self.launcher),
            "validation_artifacts": {validation.relative_to(root).as_posix(): pkg.sha(validation)}}
        self.save(self.protocol / "freeze.json", self.freeze)
        self.freeze_sha = pkg.sha(self.protocol / "freeze.json")
        names = {f"{arm}-{pair}" for arm in pkg.ARMS for pair in pkg.PAIRS}
        for name in names:
            for filename in ("checkpoint.pt", "weights.pt", "completed.json"):
                self.put(self.execution / f"fits/{name}/{filename}", f"synthetic {name} {filename}".encode())
            for panel in pkg.PANELS:
                for filename in ("episodes.npz", "episodes.json", "decisions/000.npz"):
                    self.put(self.execution / f"control/{panel}/{name}/{filename}", b"synthetic native trace")
        self.completed = {"status": "completed", "study": study, "plan_sha256": self.plan_sha,
            "files": {name: pkg.sha(path) for name, path in pkg.files(self.execution).items()},
            "fits": 15, "control_rows": 60, "prediction_episodes": 96,
            "astra_calls": 0, "wall_seconds": 10.}
        self.save(self.execution / "completed.json", self.completed)
        self.summary = {"status": "completed", "engineering": False, "plan_sha256": self.plan_sha,
            "execution_completed_sha256": pkg.sha(self.execution / "completed.json"),
            "saved_output_only": True, "costs": {"audit_validation_wall_seconds": 1.},
            "new_model_calls": 0, "new_policy_calls": 0, "new_fits": 0,
            "fits": {name: {} for name in names}, "prediction": {name: {} for name in names},
            "control": {panel: {name: {"episode_costs": [10.] * 64} for name in names | set(pkg.REFERENCES)}
                        for panel in pkg.PANELS},
            "coverage": {"fits": 15, "control_rows": 60, "learned_control_rows": 45,
                "reference_rows": 15, "control_cases_per_row": 64, "optimizer_updates_per_fit": 1152,
                "total_optimizer_updates": 17280}, "native_transitions_checked": 235200,
            "phase_boundary": {"all_fifteen_fits_restored_before_evaluation": True},
            "continuation_gate": {"passed": False, "checks": [{"passed": i != 0} for i in range(28)]}}
        self.save(self.audit_folder / "summary.json", self.summary)
        self.put(self.audit_folder / "README.md", b"synthetic audit only")
        self.audit = {"status": "completed", "version": study, "engineering": False,
            "saved_output_only": True, "plan_sha256": self.plan_sha, "source_sha256": sources,
            "runtime": self.plan["runtime"], "execution_completed_sha256": self.summary["execution_completed_sha256"],
            "execution_members": self.completed["files"], "costs": self.summary["costs"],
            "files": {name: pkg.sha(self.audit_folder / name) for name in ("summary.json", "README.md")}}
        self.save(self.audit_folder / "receipt.json", self.audit)
        self.audit_sha = pkg.sha(self.audit_folder / "receipt.json")
        supervision = output / "supervision"
        self.save(supervision / "started.json", {"plan_sha256": self.plan_sha, "retries": 0,
            "execution": self.execution.relative_to(root).as_posix(),
            "audit": self.audit_folder.relative_to(root).as_posix(), "frozen_execution_cap_seconds": 3600,
            "frozen_audit_cap_seconds": 300, "launcher_sha256": pkg.sha(self.launcher)})
        self.save(supervision / "completed.json", {"plan_sha256": self.plan_sha, "status": "completed",
            "audit_receipt_sha256": self.audit_sha, "continuation_passed": False,
            "passed_checks": 27, "native_transitions_checked": 235200, "whole_supervision_seconds": 12.})
        for name in ("execution.log", "audit.log"):
            self.put(supervision / name, b"synthetic closed log")
        report_names = ("report.json", "tables.md", "native-costs.png", "utility-vs-cost.png", "paired-comparisons.png")
        for name in report_names:
            self.put(self.figures / name, b"synthetic final report")
        self.report = {"status": "completed", "engineering": False,
            "scope": "completed_scored_development_study", "inputs": {
                "plan_sha256": self.plan_sha, "audit_receipt_sha256": self.audit_sha,
                "audit_summary_sha256": self.audit["files"]["summary.json"],
                "execution_completed_sha256": self.audit["execution_completed_sha256"],
                "execution_member_count": len(self.audit["execution_members"]), "frozen_source_count": 70},
            "new_model_calls": 0, "new_simulator_calls": 0, "wall_seconds": 1.,
            "source_sha256": pkg.sha(output / "report_results.py"),
            "files": {name: pkg.sha(self.figures / name) for name in report_names}}
        self.save(self.figures / "receipt.json", self.report)
        self.report_sha = pkg.sha(self.figures / "receipt.json")
        replay_names = ("first-ordinary-case-pair0.gif", "layout-preview.png")
        for name in replay_names:
            self.put(self.replay_folder / name, b"synthetic final replay")
        self.replay = {"status": "completed", "engineering": False,
            "scope": "fixed_first_case_schematic_scored_replay", "plan_sha256": self.plan_sha,
            "audit_receipt_sha256": self.audit_sha, "audit_summary_sha256": self.audit["files"]["summary.json"],
            "execution_completed_sha256": self.audit["execution_completed_sha256"], "source_sha256": sources,
            "case_index": 0, "pair": "pair0", "panel": "ordinary", "frames": 50,
            "executed_steps": list(range(1, 51)), "frame_duration_ms": 100, "playback_duration_seconds": 5.,
            "native_episode_duration_seconds": 1., "slowdown_factor": 5,
            "layout": {"columns": 3, "rows": 2, "model_panels": 5, "legend_panels": 1, "width": 1500, "height": 930},
            "models": {}, "new_model_calls": 0, "new_policy_calls": 0, "new_simulator_calls": 0,
            "geometry": {"xml_sha256": "a" * 64}, "files": {name: pkg.sha(self.replay_folder / name) for name in replay_names},
            "renderer_source_sha256": pkg.sha(output / "render_replay.py")}
        for arm in pkg.ARMS:
            name = f"{arm}-pair0"
            self.replay["models"][name] = {"case_index": 0, "pair": "pair0", "panel": "ordinary",
                "episodes_npz_sha256": self.completed["files"][f"control/ordinary/{name}/episodes.npz"],
                "episodes_json_sha256": self.completed["files"][f"control/ordinary/{name}/episodes.json"],
                "fit_completed_sha256": self.completed["files"][f"fits/{name}/completed.json"],
                "weights_sha256": self.completed["files"][f"fits/{name}/weights.pt"],
                "final_cumulative_native_cost": 10.}
        self.save(self.replay_folder / "receipt.json", self.replay)
        self.replay_sha = pkg.sha(self.replay_folder / "receipt.json")

    @staticmethod
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def save(self, path, data):
        return self.put(path, (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode())

    def authenticate(self):
        return pkg.validate_scored(self.root, self.plan_sha, self.freeze_sha, self.audit_sha)

    def args(self):
        return SimpleNamespace(completed_authorized=True, expected_plan_sha256=self.plan_sha,
            expected_freeze_sha256=self.freeze_sha, expected_audit_receipt_sha256=self.audit_sha,
            expected_reporting_receipt_sha256=self.report_sha, expected_replay_receipt_sha256=self.replay_sha,
            reporting_directory=f"evidence/{pkg.STUDY}/figures", replay_directory=f"evidence/{pkg.STUDY}/replay",
            reporting_file=[], engineering_receipt=[], out=f"output/{pkg.STUDY}/synthetic-publication")


class SyntheticChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_authorization_precedes_all_file_reads(self):
        with (patch.object(pkg, "read", side_effect=AssertionError("opened a study artifact")),
              self.assertRaisesRegex(ValueError, "authorization")):
            pkg.package(SimpleNamespace(completed_authorized=False))

    def test_incomplete_audit_precedes_plan_and_execution_reads(self):
        folder = self.root / f"evidence/{pkg.STUDY}/audit"
        Fixture.put(folder / "receipt.json", b'{"status":"running","engineering":false,"saved_output_only":true}')
        with self.assertRaisesRegex(ValueError, "audit required first"):
            pkg.validate_scored(self.root, "a" * 64, "b" * 64, pkg.sha(folder / "receipt.json"))

    def test_complete_synthetic_bundle_retains_failed_gate_and_rejects_overwrite(self):
        case = Fixture(self.root)
        plan, audit, summary, members = case.authenticate()
        self.assertFalse(summary["continuation_gate"]["passed"])
        self.assertTrue(all(self.root / name in members for name in plan["sources"]))
        with patch.object(pkg, "ROOT", self.root), patch.object(pkg, "EXPECTED_PLAN", case.plan_sha), \
             patch.object(pkg, "EXPECTED_FREEZE", case.freeze_sha), patch.object(pkg, "__file__", str(case.packager)):
            result = pkg.package(case.args())
            self.assertEqual(result["scientific_status"], {"continuation_passed": False, "checks_passed": 27,
                                                         "checks_total": 28, "packaging_changes_gate": False})
            self.assertEqual(result["raw_coverage"]["fits"], 15)
            self.assertTrue(result["archive"]["all_members_reopened_and_verified"])
            with self.assertRaisesRegex(ValueError, "Exclusive"):
                pkg.package(case.args())
        self.assertEqual(audit["execution_members"], case.completed["files"])

    def test_packaging_failure_retains_partial_archive_and_forbids_retry(self):
        case = Fixture(self.root)

        def fail(_root, _paths, archive):
            archive.write_bytes(b"retained partial archive")
            raise RuntimeError("synthetic archive write failure")

        with (patch.object(pkg, "ROOT", self.root), patch.object(pkg, "EXPECTED_PLAN", case.plan_sha),
              patch.object(pkg, "EXPECTED_FREEZE", case.freeze_sha), patch.object(pkg, "__file__", str(case.packager)),
              patch.object(pkg, "write_archive", side_effect=fail)):
            with self.assertRaisesRegex(RuntimeError, "synthetic archive"):
                pkg.package(case.args())
            folder = self.root / case.args().out
            self.assertEqual((folder / "reacher-cache-scored-execution-and-audit.tar.gz").read_bytes(),
                             b"retained partial archive")
            self.assertEqual(pkg.read(folder / "packaging-failed.json")["status"], "failed")
            self.assertFalse((folder / "receipt.json").exists())
            with self.assertRaisesRegex(ValueError, "Exclusive"):
                pkg.package(case.args())

    def test_missing_and_corrupt_bound_primary_artifacts_reject(self):
        case = Fixture(self.root)
        path = self.root / next(iter(case.plan["sources"]))
        original = path.read_bytes()
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            case.authenticate()
        path.write_bytes(original)
        next(iter(pkg.files(case.execution).values())).unlink()
        with self.assertRaises(ValueError):
            case.authenticate()

    def test_extra_primary_member_and_changed_freeze_reject(self):
        case = Fixture(self.root)
        extra = case.execution / "unbound-selection.json"
        extra.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "Exact artifact membership"):
            case.authenticate()
        extra.unlink()
        (case.protocol / "freeze.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            case.authenticate()

    def test_reporting_and_replay_boundaries(self):
        case = Fixture(self.root)
        args = (self.root, case.replay_folder, case.replay_sha, case.plan, case.plan_sha,
                case.audit_sha, case.audit, case.summary)
        pkg.validate_replay(*args)
        pkg.validate_reporting(self.root, case.figures, case.report_sha, case.plan_sha, case.audit_sha, case.audit)
        case.replay["models"]["cached_mlp-pair0"]["case_index"] = 1
        case.save(case.replay_folder / "receipt.json", case.replay)
        bad = list(args); bad[2] = pkg.sha(case.replay_folder / "receipt.json")
        with self.assertRaisesRegex(ValueError, "Fixed replay row"):
            pkg.validate_replay(*bad)
        (case.figures / "paired-comparisons.png").unlink()
        with self.assertRaises(ValueError):
            pkg.validate_reporting(self.root, case.figures, case.report_sha, case.plan_sha, case.audit_sha, case.audit)

    def test_deterministic_archive_reopen_and_corruption(self):
        source = self.root / "input.bin"
        source.write_bytes(bytes(range(256)) * 3)
        one, two = self.root / "one.tar.gz", self.root / "two.tar.gz"
        rows = pkg.write_archive(self.root, {source}, one)
        pkg.write_archive(self.root, {source}, two)
        self.assertEqual(pkg.sha(one), pkg.sha(two))
        rows[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "member bytes"):
            pkg.verify_archive(one, rows)

    def test_split_parts_reconstruct_exact_archive(self):
        archive = self.root / "archive.tar.gz"
        archive.write_bytes(bytes(range(256)) * 3 + b"tail")
        parts, split = pkg.split_assets(archive, self.root, max_asset_bytes=200, part_bytes=127)
        self.assertTrue(split)
        joined = b"".join((self.root / row["name"]).read_bytes() for row in parts)
        self.assertEqual(hashlib.sha256(joined).hexdigest(), pkg.sha(archive))
        self.assertEqual(sum(row["bytes"] for row in parts), archive.stat().st_size)
        self.assertTrue(all(row["bytes"] <= 127 for row in parts))
        with self.assertRaises(FileExistsError):
            pkg.split_assets(archive, self.root, max_asset_bytes=200, part_bytes=127)

    def engineering_bundle(self, index=0):
        scope = pkg.ENGINEERING_SCOPE_BINDINGS[index]
        folder = self.root / f"evidence/synthetic-engineering-{index}"
        raw = self.root / f"raw-engineering-{index}.bin"
        raw.write_bytes(b"synthetic engineering archive input")
        archive = self.root / f"synthetic-engineering-{index}.tar.gz"
        rows = pkg.write_archive(self.root, {raw}, archive)
        folder.mkdir(parents=True)
        Fixture.save(self, folder / "summary.json", {"scope": scope[1]})
        Fixture.save(self, folder / "bundle-manifest.json", {"scope": scope[2],
            "members": {row["path"]: {key: value for key, value in row.items() if key != "path"} for row in rows}})
        receipt = {"status": "completed", "scope": scope[0], "files": {
            name: pkg.sha(folder / name) for name in ("summary.json", "bundle-manifest.json")},
            "archive": {"path": archive.relative_to(self.root).as_posix(), "sha256": pkg.sha(archive),
                        "bytes": archive.stat().st_size}}
        Fixture.save(self, folder / "receipt.json", receipt)
        (folder / "SHA256SUMS").write_text("synthetic checksum sidecar")
        return folder, receipt

    put = staticmethod(Fixture.put)

    def test_optional_capacity_and_rehearsal_scope_schemas_are_accepted(self):
        declarations = []
        for index in range(2):
            folder, _ = self.engineering_bundle(index)
            declarations.append(f"{(folder / 'receipt.json').relative_to(self.root).as_posix()}={pkg.sha(folder / 'receipt.json')}")
        assets, members = pkg.engineering_assets(self.root, declarations)
        self.assertEqual(len(assets), 2)
        self.assertTrue(all(row["scope"] == "engineering_only_not_scientific_result" for row in assets))
        self.assertEqual(len(members), 8)

    def test_optional_engineering_scope_documents_must_remain_bound(self):
        folder, receipt = self.engineering_bundle()
        declaration = f"{(folder / 'receipt.json').relative_to(self.root).as_posix()}={pkg.sha(folder / 'receipt.json')}"
        (folder / "summary.json").write_text('{"scope":"scientific effectiveness"}')
        with self.assertRaisesRegex(ValueError, "hash changed"):
            pkg.engineering_assets(self.root, [declaration])
        (folder / "summary.json").unlink()
        receipt["files"].pop("summary.json")
        Fixture.save(self, folder / "receipt.json", receipt)
        declaration = f"{(folder / 'receipt.json').relative_to(self.root).as_posix()}={pkg.sha(folder / 'receipt.json')}"
        with self.assertRaisesRegex(ValueError, "scope documents"):
            pkg.engineering_assets(self.root, [declaration])

    def test_optional_engineering_receipt_text_cannot_override_scientific_summary(self):
        folder, receipt = self.engineering_bundle()
        Fixture.save(self, folder / "summary.json", {"scope": "scientific effectiveness"})
        receipt["files"]["summary.json"] = pkg.sha(folder / "summary.json")
        Fixture.save(self, folder / "receipt.json", receipt)
        declaration = f"{(folder / 'receipt.json').relative_to(self.root).as_posix()}={pkg.sha(folder / 'receipt.json')}"
        with self.assertRaisesRegex(ValueError, "engineering-only scope"):
            pkg.engineering_assets(self.root, [declaration])

    def test_safe_paths_and_digest_arguments(self):
        for name in ("../outside", "/absolute", "nested/../escape", "nested//alias", "nested\\escape"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pkg.child(self.root, name)
        with self.assertRaisesRegex(ValueError, "path=SHA"):
            pkg.bound_argument("unbound.md")
        with self.assertRaisesRegex(ValueError, "SHA"):
            pkg.bound_argument("bound.md=short")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticChecks))
    if args.receipt:
        pkg.write(args.receipt, {"scope": "synthetic_packaging_tests_only_no_study_artifacts_read",
            "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "passed": result.wasSuccessful(), "packager_sha256": pkg.sha(MODULE_PATH),
            "test_sha256": pkg.sha(Path(__file__)), "new_model_calls": 0, "new_native_calls": 0,
            "synthetic_global_patches": "Only isolated temp-root expected plan/freeze and helper path for full package test."})
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
