"""Conspicuous artificial summaries. Never reads any real study result."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("synthetic_lexical_plot", ROOT/"scripts/plot_dialogue_qwen_lexical_ablation.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
report, audit = HERE/"synthetic-report", HERE/"synthetic-audit"
report.mkdir(); audit.mkdir()
fits = {v: {} for v in ("original", "no_flags")}
checks = {}
for a, arm in enumerate(p.ARMS):
    for variant in fits:
        fits[variant][arm] = {"cells": {s: {"rows": p.SUPPORT[s], "metrics": {}} for s in ("all", "changed", "retained")}}
    for s, metric, *_ in p.PANELS:
        for w, weighting in enumerate(p.WEIGHTINGS):
            original = {"accuracy": .75, "error": .14, "nll": 1.2, "brier": .3}[metric]+a*.01+w*.003
            delta = {"accuracy": .005 if a == 0 else -.015, "error": -.023 if a == 0 else -.004,
                     "nll": -.13 if a == 0 else .00023, "brier": -.008 if a == 0 else .00009}[metric]
            new = original+delta
            for variant, value in (("original", original), ("no_flags", new)):
                fits[variant][arm]["cells"][s]["metrics"].setdefault(metric, {})[weighting] = value
            threshold = {"accuracy": -.01, "error": -.02, "nll": 0., "brier": 0.}[metric]
            passed = delta >= threshold if metric == "accuracy" else delta <= threshold
            checks[f"{arm}/{s}/{metric}/{weighting}"] = {"difference": new-original, "threshold": threshold,
                "relation": ">=" if metric == "accuracy" else "<=", "passed": passed}
arms = {a: all(v["passed"] for k, v in checks.items() if k.startswith(a+"/")) for a in p.ARMS}
decision = {"passed": all(arms.values()), "arms": arms, "checks": checks, "total_checks": 16,
            "checks_passed": sum(c["passed"] for c in checks.values())}
summary = {"status": "completed", "version": p.REPORT, "experiment_id": p.EXPERIMENT,
    "technical_validity_passed": True, "synthetic": True, "plan_sha256": "1"*64, "execution_completed_sha256": "2"*64,
    "support": p.SUPPORT, "services": [f"synthetic-service-{i}" for i in range(6)], "continuation": decision, **fits}
p.write(report/"started.json", {"synthetic": True})
p.write(report/"summary.json", summary)
(report/"report.md").write_text("SYNTHETIC FIXTURE ONLY. Not real outcomes.\n")
receipt = {"status": "completed", "version": p.REPORT, "experiment_id": p.EXPERIMENT, "technical_validity_passed": True,
    "synthetic": True, "plan_sha256": "1"*64, "execution_completed_sha256": "2"*64,
    "continuation_passed": decision["passed"], "model_calls": 0, "tokenizer_calls": 0, "checkpoint_deserializations": 0,
    "files": {n: p.item(report/n) for n in ("started.json", "summary.json", "report.md")}}
p.write(report/"receipt.json", receipt)
checked = {"status": "completed", "version": p.AUDIT, "synthetic": True, "agreement": True,
    "rows_per_arm": 7819, "metric_cells": 84, "decision_checks": 16,
    "plan_sha256": "1"*64, "execution_completed_sha256": "2"*64,
    "producer_summary_sha256": p.digest(report/"summary.json"), "continuation": decision, **copy.deepcopy(fits)}
p.write(audit/"started.json", {"synthetic": True})
p.write(audit/"summary.json", checked)
audit_receipt = {"status": "completed", "version": p.AUDIT, "synthetic": True, "agreement": True,
    "producer_summary_sha256": p.digest(report/"summary.json"), "producer_receipt_sha256": p.digest(report/"receipt.json"),
    "plan_sha256": "1"*64, "execution_completed_sha256": "2"*64, "metric_cells": 84, "decision_checks": 16,
    "continuation_passed": decision["passed"], "model_calls": 0, "tokenizer_calls": 0, "checkpoint_deserializations": 0,
    "files": {n: p.item(audit/n) for n in ("started.json", "summary.json")}}
p.write(audit/"receipt.json", audit_receipt)
args = SimpleNamespace(summary=report/"summary.json", summary_sha256=p.digest(report/"summary.json"),
    receipt=report/"receipt.json", receipt_sha256=p.digest(report/"receipt.json"), audit=audit/"receipt.json",
    audit_sha256=p.digest(audit/"receipt.json"), synthetic=True, out=HERE/"figure-01")
real_read = p.read
decoded = []
def guarded_read(path):
    decoded.append(str(path))
    if Path(path).name == "summary.json":
        raise AssertionError("Quality decoded before authentication rejected corrupt envelope")
    return real_read(path)
p.read = guarded_read
bad = copy.copy(args); bad.out = HERE/"rejected-pin"; bad.summary_sha256 = "0"*64
try:
    p.execute(bad)
    raise AssertionError("Wrong pin accepted")
except ValueError as error:
    assert "pin" in str(error) and decoded == []
bad = copy.copy(args); bad.out = HERE/"rejected-membership"
extra = report/"unmanifested.txt"; extra.write_text("artificial membership damage")
try:
    p.execute(bad)
    raise AssertionError("Extra membership accepted")
except ValueError as error:
    assert "membership" in str(error) and not any(Path(x).name == "summary.json" for x in decoded)
finally:
    extra.rename(HERE/"removed-artificial-unmanifested.txt")
p.read = real_read
p.execute(args)
values = p.read(args.out/"plotted-values.json")
assert len(values["panels"]) == 8
assert sum(len(v["arms"]) for v in values["panels"].values()) == 16
assert values["continuation"]["checks_passed"] == 8
result = {"synthetic": True, "source_sha256": p.digest(ROOT/"scripts/plot_dialogue_qwen_lexical_ablation.py"),
    "checks": {"bad_pin_before_any_input_decode": True, "extra_member_before_quality_decode": True,
               "both_failures_receipted": all((HERE/n/"failed.json").is_file() for n in ("rejected-pin", "rejected-membership")),
               "panels": 8, "plotted_values": 32, "decision_markers": 16, "synthetic_checks_passed": 8},
    "figure_receipt_sha256": p.digest(args.out/"receipt.json")}
p.write(HERE/"smoke-result.json", result)
print(json.dumps(result, indent=2))
