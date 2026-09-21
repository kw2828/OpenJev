"""One artificial render plus authentication rejection checks. No real inputs."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("observation_figure", ROOT / "plot.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def fixture():
    fits = {}
    for i, arm in enumerate(p.ARMS):
        for j, seed in enumerate(p.SEEDS):
            panels = {}
            for panel, count in (("seen", 2000), ("unseen", 1000)):
                base = (.80 if panel == "seen" else .61) + .005 * i + .002 * j
                if arm == "trainable_numbers" and panel == "unseen":
                    base = .619 + .006 * j
                strata = {name: {"count": count, "correct": round(count * (base + offset)),
                                  "accuracy": round(count * (base + offset)) / count,
                                  "error": 1 - round(count * (base + offset)) / count}
                          for name, offset in (("unmentioned_retention", .15), ("assigned_retention", .03), ("changed", -.18))}
                if arm == "trainable_numbers" and panel == "seen":
                    correct = round(count * (.827 + .002 * j))
                    strata["assigned_retention"].update(correct=correct, accuracy=correct / count, error=1 - correct / count)
                panels[panel] = {"strata": strata, "macro_three": {"accuracy": math.fsum(v["accuracy"] for v in strata.values()) / 3},
                                 "micro": {"count": 3 * count, "nll": 1.01 - i * .018 + j * .004,
                                           "brier": .51 - i * .011 + j * .003}}
            fits[f"{arm}-{seed}"] = {"panels": panels}
    def macro(panel):
        return sum((Fraction(v["correct"], v["count"]) for v in panel["strata"].values()), Fraction()) / 3

    def errors(panel):
        value = panel["strata"]["assigned_retention"]
        return Fraction(value["count"] - value["correct"], value["count"])

    checks, unseen_deltas = [], []
    for name, _label, _rule, threshold, comparison, _scale, _unit in p.CONDITIONS:
        if threshold is None:
            wins = sum(v > 0 for v in unseen_deltas)
            checks.append({"name": name, "wins": wins, "required": 2, "passed": wins >= 2, "arithmetic": "exact"})
        else:
            panel = "unseen" if name.startswith("unseen") else "seen"
            getter = macro if "macro" in name else errors if "assigned" in name else lambda v, n=name: v["micro"]["nll" if "nll" in n else "brier"]
            primary = [getter(fits[f"trainable_numbers-{seed}"]["panels"][panel]) for seed in p.SEEDS]
            control = [getter(fits[f"frozen_numbers-{seed}"]["panels"][panel]) for seed in p.SEEDS]
            deltas = [a - b for a, b in zip(primary, control, strict=True)]
            exact = all(isinstance(v, Fraction) for v in deltas)
            value = sum(deltas, Fraction()) / 3 if exact else math.fsum(primary) / 3 - math.fsum(control) / 3
            boundary = Fraction(str(threshold)) if exact else threshold
            if name == "unseen_macro_gain_1pp":
                unseen_deltas = deltas
            checks.append({"name": name, "paired_seed_values": list(map(float, deltas)), "mean": float(value),
                           "threshold": threshold, "comparison": comparison,
                           "passed": value >= boundary if comparison == "ge" else value <= boundary})
    return {"version": p.REPORT_VERSION, "status": "completed", "synthetic": True,
            "technical_validity_passed": True, "technical_complete_fits": 12,
            "plan_sha256": "a" * 64, "execution_completed_sha256": "b" * 64, "fits": fits,
            "continuation": {"passed": all(c["passed"] for c in checks), "checks_passed": sum(c["passed"] for c in checks),
                             "checks_total": 7, "complete_fit_membership": True, "checks": checks}}


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    report = out / "synthetic-report"
    report.mkdir()
    summary = fixture()
    p.write(report / "started.json", {"synthetic": True})
    p.write(report / "summary.json", summary)
    (report / "report.md").write_text("SYNTHETIC FIXTURE - NOT RESULTS\n")
    receipt = {"version": p.REPORT_VERSION, "status": "completed", "synthetic": True,
               "technical_validity_passed": True, "model_calls": 0, "plan_sha256": "a" * 64,
               "execution_completed_sha256": "b" * 64, "continuation_passed": summary["continuation"]["passed"],
               "launch_sha256": "c" * 64, "terminal_sha256": "d" * 64,
               "scientific_checks_total": 7, "scientific_checks_passed": summary["continuation"]["checks_passed"],
               "files": {name: p.descriptor(report / name) for name in ("started.json", "summary.json", "report.md")}}
    p.write(report / "receipt.json", receipt)
    audit = out / "synthetic-audit"
    audit.mkdir()
    audited = {"version": p.AUDIT_VERSION, "status": "completed", "agreement": True, "synthetic": True,
               "fits": copy.deepcopy(summary["fits"]), "continuation": copy.deepcopy(summary["continuation"]),
               "plan_sha256": "a" * 64, "execution_completed_sha256": "b" * 64,
               "fit_cells": 144, "literal_cells": 24, "scalar_checks": 1000}
    p.write(audit / "started.json", {"synthetic": True})
    p.write(audit / "summary.json", audited)
    audit_receipt = {"version": p.AUDIT_VERSION, "status": "completed", "agreement": True, "synthetic": True,
                     "model_calls": 0, "plan_sha256": "a" * 64, "execution_completed_sha256": "b" * 64,
                     "launch_sha256": "c" * 64, "terminal_sha256": "d" * 64,
                     "producer_receipt_sha256": p.sha(report / "receipt.json"),
                     "producer_summary_sha256": p.sha(report / "summary.json"),
                     "continuation_passed": summary["continuation"]["passed"],
                     "scientific_checks_passed": summary["continuation"]["checks_passed"],
                     "fit_cells": 144, "literal_cells": 24, "scalar_checks": 1000,
                     "files": {name: p.descriptor(audit / name) for name in ("started.json", "summary.json")}}
    p.write(audit / "receipt.json", audit_receipt)
    args = SimpleNamespace(report=report, summary_sha256=p.sha(report / "summary.json"),
                           receipt_sha256=p.sha(report / "receipt.json"), audit=audit,
                           audit_receipt_sha256=p.sha(audit / "receipt.json"),
                           audit_summary_sha256=p.sha(audit / "summary.json"),
                           failed_manifest=p.FAILED_MANIFEST, out=out / "render", synthetic=True)
    original_read = p.read
    decoded_quality = []

    def observed_read(path):
        if Path(path).name == "summary.json":
            decoded_quality.append(str(path))
        return original_read(path)

    p.read = observed_read
    failures = []
    bad = copy.copy(args)
    bad.summary_sha256 = "0" * 64
    bad.out = out / "wrong-pin"
    try:
        p.execute(bad)
        raise AssertionError("Wrong external pin accepted")
    except ValueError as error:
        assert "pins" in str(error) and not decoded_quality
        assert (bad.out / "failed.json").exists()
        failures.append("External pin rejection before quality decode; failure retained")
    (report / "extra.txt").write_text("Artificial unexpected member")
    bad = copy.copy(args)
    bad.out = out / "extra-member"
    try:
        p.execute(bad)
        raise AssertionError("Unexpected report member accepted")
    except ValueError as error:
        assert "membership" in str(error) and not decoded_quality
        failures.append("Extra report payload rejected before quality decode; failure retained")
    (report / "extra.txt").unlink()
    for name, change, expected in (
        ("failed-audit", {"agreement": False}, "independent audit"),
        ("wrong-audit-join", {"producer_summary_sha256": "0" * 64}, "exact completed production report"),
    ):
        destination = out / (name + "-input")
        shutil.copytree(audit, destination)
        altered = dict(audit_receipt, **change)
        (destination / "receipt.json").unlink()
        p.write(destination / "receipt.json", altered)
        bad = copy.copy(args)
        bad.audit, bad.audit_receipt_sha256, bad.out = destination, p.sha(destination / "receipt.json"), out / name
        try:
            p.execute(bad)
            raise AssertionError("Invalid audit accepted")
        except ValueError as error:
            assert expected in str(error) and not decoded_quality
            assert (bad.out / "failed.json").exists()
            failures.append(name + " rejected before either quality summary decode; failure retained")
    bad = copy.copy(args)
    bad.audit, bad.out = out / "absent-audit", out / "missing-audit"
    try:
        p.execute(bad)
        raise AssertionError("Missing audit accepted")
    except FileNotFoundError:
        assert not decoded_quality and (bad.out / "failed.json").exists()
        failures.append("Missing audit rejected before either quality summary decode; failure retained")
    p.read = original_read
    native_loader, actual_render = p.load_clock, p.render

    def broken_loader():
        raise RuntimeError("Injected clock initialization failure")

    bad = copy.copy(args)
    bad.out = out / "clock-initialization-failure"
    p.load_clock = broken_loader
    try:
        p.execute(bad)
        raise AssertionError("Broken clock accepted")
    except RuntimeError:
        failure = original_read(bad.out / "failed.json")
        assert failure["timing_available"] is False and failure["wall_seconds"] is None
        failures.append("Clock initialization failure preserved with unavailable timing")
    finally:
        p.load_clock = native_loader

    native_loader()
    deadline_class = sys.modules["observation_figure_suspend_clock"].Deadline
    bad = copy.copy(args)
    bad.out = out / "late-publication"

    class LateClock:
        backend = "injected-lifecycle-only"

        def now_ns(self):
            return 60_000_000_000 if (bad.out / "receipt.json").exists() else 0

        def deadline_after(self, seconds):
            return deadline_class(self, 0, int(seconds * 1e9))

    def placeholder_render(_summary, _values, destination, _synthetic):
        for name in ("observation.png", "observation.svg", "retention-and-conditions.png", "retention-and-conditions.svg"):
            (destination / name).write_text("SYNTHETIC CLOCK TEST PLACEHOLDER, NOT AN IMAGE\n")

    p.load_clock, p.render = lambda: LateClock, placeholder_render
    try:
        p.execute(bad)
        raise AssertionError("Deadline equality after publication accepted")
    except TimeoutError:
        assert (bad.out / "invalid-receipt.json").exists() and (bad.out / "failed.json").exists()
        assert not (bad.out / "receipt.json").exists()
        failures.append("Deadline equality after publication invalidates success; synthetic placeholders only")
    finally:
        p.load_clock, p.render = native_loader, actual_render
    result = p.execute(args)
    p.write(out / "smoke-receipt.json", {"status": "completed", "synthetic": True, "model_calls": 0,
            "figure_source": p.descriptor(ROOT / "plot.py"), "smoke_source": p.descriptor(Path(__file__)),
            "negative_checks": failures, "render_receipt": p.descriptor(args.out / "receipt.json"),
            "render_wall_seconds": result["wall_seconds"],
            "scope": "One artificial aggregate render only; no actual study metrics read. Fixed prior-failure metadata is bound separately and its referenced files are never opened."})
    print(json.dumps({"status": "completed", "synthetic": True, "out": str(out), "checks": failures}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
