"""Render only a completed, independently checked lexical-ablation report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path

VERSION = "dialogue-qwen-lexical-ablation-figure-v1"
REPORT = "dialogue-qwen-lexical-ablation-report-v1"
AUDIT = "dialogue-qwen-lexical-result-audit-v1"
EXPERIMENT = "dialogue-qwen-lexical-ablation-v1"
LIMITS = {"wall_seconds": 60, "rss_bytes": 2*1024**3, "output_bytes": 64*1024**2}
ARMS = ("current", "history4")
WEIGHTINGS = ("row", "equal_service")
SUPPORT = {"all": 7819, "changed": 578, "retained": 7241, "unmentioned_retention": 4032, "assigned_retention": 3209}
PANELS = (("changed", "accuracy", "Changed-state accuracy", 100., "%", "Drop at most 1 percentage point"),
          ("retained", "error", "Retained-state error", 100., "%", "Drop at least 2 percentage points"),
          ("all", "nll", "Overall log loss", 1., "nats", "No increase"),
          ("all", "brier", "Overall Brier score", 1., "squared probability error", "No increase"))
SCOPE = ("Saved-summary presentation with external report and independent-audit pins. The audit checks primary "
         "all/changed/retained accuracy/error/NLL/Brier and all 16 decisions; the plotter only checks their "
         "authenticated agreement and mapping. No prediction, model or cache reads. Exposed official TRAIN, "
         "correct previous value supplied, same Qwen model. No architecture, autonomous-memory, calibration "
         "or causal speed claim. This removes both noisy lexical matches and genuine full-prefix register information.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def item(path):
    return {"sha256": digest(path), "bytes": Path(path).stat().st_size}


def pin(value):
    return type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")


def read(path):
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON: "+value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def closure(path, receipt, members, bindings):
    root = path.parent
    require(set(receipt["files"]) == members and
            {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == members | {"receipt.json"},
            "Exact artifact membership")
    for name, descriptor in receipt["files"].items():
        target = root/name
        require(not target.is_symlink() and target.resolve().is_relative_to(root)
                and set(descriptor) == {"sha256", "bytes"} and type(descriptor["bytes"]) is int
                and item(target) == descriptor, "Authenticated payload: "+name)
        bindings[target] = descriptor


def authenticate(args):
    summary_path, receipt_path, audit_path = (Path(getattr(args, n)).resolve() for n in ("summary", "receipt", "audit"))
    require(summary_path.name == "summary.json" and receipt_path.name == audit_path.name == "receipt.json"
            and summary_path.parent == receipt_path.parent and audit_path.parent != receipt_path.parent,
            "Separate report/audit directories")
    bindings = {}
    for path, expected in ((summary_path, args.summary_sha256), (receipt_path, args.receipt_sha256), (audit_path, args.audit_sha256)):
        require(pin(expected) and digest(path) == expected, "External report/audit pin")
        bindings[path] = item(path)
    # Receipt decoding is control metadata. Neither quality summary is decoded until both full closures bind.
    receipt, audit = read(receipt_path), read(audit_path)
    require(receipt["status"] == audit["status"] == "completed" and receipt["version"] == REPORT
            and audit["version"] == AUDIT and receipt["experiment_id"] == EXPERIMENT
            and receipt["technical_validity_passed"] is audit["agreement"] is True,
            "Completed technically admitted report and agreeing independent audit")
    require(audit["producer_receipt_sha256"] == args.receipt_sha256
            and audit["producer_summary_sha256"] == args.summary_sha256, "Audit binds plotted report")
    require(all(receipt[k] == audit[k] == 0 for k in ("model_calls", "tokenizer_calls", "checkpoint_deserializations")),
            "Saved-only input scope")
    closure(receipt_path, receipt, {"started.json", "summary.json", "report.md"}, bindings)
    closure(audit_path, audit, {"started.json", "summary.json"}, bindings)
    summary, checked = read(summary_path), read(audit_path.parent/"summary.json")
    require(summary["status"] == checked["status"] == "completed" and summary["version"] == REPORT
            and checked["version"] == AUDIT and summary["experiment_id"] == EXPERIMENT
            and summary["technical_validity_passed"] is checked["agreement"] is True, "Summary status/identity")
    for name in ("plan_sha256", "execution_completed_sha256"):
        require(pin(summary[name]) and summary[name] == receipt[name] == audit[name] == checked[name], "Same plan/run lineage")
    require(checked["producer_summary_sha256"] == args.summary_sha256 and checked["rows_per_arm"] == 7819
            and checked["metric_cells"] == audit["metric_cells"] == 84
            and checked["decision_checks"] == audit["decision_checks"] == 16, "Audited primary scope")
    require(all(v.get("synthetic", False) is bool(args.synthetic) for v in (summary, receipt, audit, checked)),
            "Explicit matching synthetic scope")
    require(summary["support"] == SUPPORT and set(summary["original"]) == set(summary["no_flags"]) == set(ARMS)
            and len(summary["services"]) == len(set(summary["services"])) == 6, "Complete fixed report cohort")
    decision = summary["continuation"]
    require(decision == checked["continuation"] and decision["passed"] is receipt["continuation_passed"]
            is audit["continuation_passed"], "Same audited continuation")
    expected = {f"{arm}/{s}/{m}/{w}" for arm in ARMS for s, m, *_ in PANELS for w in WEIGHTINGS}
    require(set(decision["checks"]) == expected and decision["total_checks"] == 16
            and all(type(v["passed"]) is bool for v in decision["checks"].values())
            and decision["checks_passed"] == sum(v["passed"] for v in decision["checks"].values()), "All 16 decision components")
    require(decision["arms"] == {a: all(v["passed"] for k, v in decision["checks"].items() if k.startswith(a+"/")) for a in ARMS}
            and decision["passed"] is all(decision["arms"].values()), "Per-arm and joint conjunctions")
    return summary, checked, bindings


def plotted_values(summary, audited):
    data = {}
    for s, metric, title, scale, unit, requirement in PANELS:
        for weighting in WEIGHTINGS:
            entries = {}
            for arm in ARMS:
                values = {}
                for variant in ("original", "no_flags"):
                    cell = summary[variant][arm]["cells"][s]
                    value = cell["metrics"][metric][weighting]
                    confirmed = audited[variant][arm]["cells"][s]
                    require(cell["rows"] == confirmed["rows"] == SUPPORT[s] and type(value) in (float, int)
                            and math.isfinite(value) and value >= 0
                            and (metric == "nll" or value <= (2 if metric == "brier" else 1))
                            and math.isclose(value, confirmed["metrics"][metric][weighting], abs_tol=2e-12, rel_tol=2e-12),
                            "Finite metric agrees with independent audit")
                    values[variant] = value
                check = summary["continuation"]["checks"][f"{arm}/{s}/{metric}/{weighting}"]
                require(math.isclose(values["no_flags"]-values["original"], check["difference"], abs_tol=2e-12, rel_tol=2e-12),
                        "Displayed decision delta agrees")
                entries[arm] = {**values, "decision": check}
            data[f"{s}/{metric}/{weighting}"] = {"title": title, "stratum": s, "metric": metric, "weighting": weighting,
                  "scale": scale, "unit": unit, "requirement": requirement, "rows": SUPPORT[s], "arms": entries}
    return data


def render(summary, data, out, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    colors = {"original": "#346384", "no_flags": "#ba5c27"}
    fig, axes = plt.subplots(4, 2, figsize=(12, 11.8), facecolor="white")
    for row, (s, metric, *_rest) in enumerate(PANELS):
        for col, weighting in enumerate(WEIGHTINGS):
            panel, ax = data[f"{s}/{metric}/{weighting}"], axes[row, col]
            scale, all_values = panel["scale"], []
            for y, arm in enumerate(ARMS):
                entry = panel["arms"][arm]
                values = [entry[v]*scale for v in ("original", "no_flags")]
                all_values.extend(values)
                ax.plot(values, [y, y], color="#bdc5cb", lw=1.5, zorder=1)
                for value, variant, marker in zip(values, colors, ("o", "D"), strict=True):
                    ax.scatter(value, y, s=45, color=colors[variant], marker=marker, zorder=3)
                passed = entry["decision"]["passed"]
                decimals = 4 if scale == 100 else 6
                ax.text(.01, y+.31, f"{values[0]:.{decimals}f} → {values[1]:.{decimals}f}",
                        transform=ax.get_yaxis_transform(), fontsize=9, color="#33424e", va="center")
                ax.text(.99, y+.31, "PASS" if passed else "FAIL", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=9, fontweight="bold", color="#246443" if passed else "#a33030")
            low, high = min(all_values), max(all_values)
            margin = max(.015 if scale == 100 else .0002, (high-low)*.12)
            ax.set_xlim(max(0., low-margin), high+margin)
            ax.set_ylim(1.7, -.35)
            ax.set_yticks([0, 1], ["Current", "History4"] if col == 0 else ["", ""])
            ax.tick_params(axis="y", length=0)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(axis="x", color="#e0e5e9", lw=.7)
            ax.set_axisbelow(True)
            ax.spines[["left", "top", "right"]].set_visible(False)
            ax.set_title(panel["title"]+" | "+("Row" if col == 0 else "Equal service"), loc="left", fontsize=11, fontweight="bold")
            ax.set_xlabel(panel["unit"]+"; "+panel["requirement"].lower(), fontsize=9)
            if synthetic:
                ax.text(.5, .48, "SYNTHETIC", transform=ax.transAxes, ha="center", va="center", fontsize=23,
                        color="#a33030", rotation=12, alpha=.18, fontweight="bold")
    rule = summary["continuation"]
    title = "SYNTHETIC FIXTURE ONLY: NOT REAL RESULTS" if synthetic else "Qwen lexical-input ablation"
    fig.suptitle(title, x=.06, y=.987, ha="left", fontsize=17, fontweight="bold")
    fig.text(.06, .954, "Exposed official TRAIN | Correct previous value supplied | Same Qwen model", fontsize=10)
    fig.text(.06, .931, f"Continuation {'PASS' if rule['passed'] else 'FAIL'}: {rule['checks_passed']}/16 components. "
             +"  ".join(a+": "+("PASS" if rule["arms"][a] else "FAIL") for a in ARMS), fontsize=10)
    handles = [Line2D([], [], linestyle="none", marker=m, color=colors[v], label=label) for v, m, label in
               (("original", "o", "Original lexical inputs"), ("no_flags", "D", "No lexical flags"))]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.965, .915), ncol=2, frameon=False, fontsize=10)
    fig.text(.06, .891, "578 changed; 7,241 retained (4,032 unmentioned + 3,209 assigned) per arm. Dot axes are zoomed.", fontsize=9)
    fig.text(.06, .04, "Labels: original → no flags. Full precision is embedded in the assets and plotted-values.json. "
             "PASS/FAIL uses the frozen boundaries.", fontsize=9)
    fig.text(.06, .023, "Removing flags also removes public prefix-register information. No architecture, causal-speed or autonomous-memory claim.", fontsize=9)
    fig.subplots_adjust(left=.12, right=.97, top=.852, bottom=.10, wspace=.18, hspace=.65)
    metadata = {"Description": json.dumps({"synthetic": synthetic, "scope": SCOPE, "panels": data}, sort_keys=True)}
    fig.savefig(out/"comparison.png", dpi=180, facecolor="white", metadata=metadata)
    fig.savefig(out/"comparison.svg", facecolor="white", metadata={**metadata, "Date": None})
    plt.close(fig)


def peak_rss():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform == "darwin" else n*1024)


def execute(args):
    out = Path(args.out).resolve()
    parents = {Path(getattr(args, n)).resolve().parent for n in ("summary", "receipt", "audit")}
    require(all(not out.is_relative_to(p) and not p.is_relative_to(out) for p in parents), "Separate output tree")
    out.mkdir(parents=True, exist_ok=False)
    start, previous, source = time.monotonic(), None, {}
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    def check():
        require(time.monotonic()-start <= LIMITS["wall_seconds"], "Figure wall cap")
        require(peak_rss() <= LIMITS["rss_bytes"], "Figure RSS cap")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Figure output cap")
    def timeout(*_):
        raise TimeoutError("Figure wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        source = {Path(__file__).name: digest(__file__)}
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": source, "limits": LIMITS, "scope": SCOPE})
        summary, audited, bindings = authenticate(args)
        data = plotted_values(summary, audited)
        check()
        write(out/"plotted-values.json", {"version": VERSION, "synthetic": bool(args.synthetic), "scope": SCOPE,
              "panels": data, "continuation": summary["continuation"]})
        render(summary, data, out, bool(args.synthetic))
        for path, descriptor in bindings.items():
            require(item(path) == descriptor, "End input identity")
        require(digest(__file__) == source[Path(__file__).name], "End source identity")
        check()
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "synthetic": bool(args.synthetic),
              "scope": SCOPE, "request": request, "source_sha256": source, "summary_sha256": args.summary_sha256,
              "report_receipt_sha256": args.receipt_sha256, "audit_receipt_sha256": args.audit_sha256,
              "execution_completed_sha256": summary["execution_completed_sha256"], "plan_sha256": summary["plan_sha256"],
              "continuation_passed": summary["continuation"]["passed"],
              "files": {n: item(out/n) for n in ("started.json", "plotted-values.json", "comparison.png", "comparison.svg")},
              "input_members": {str(p): d for p, d in bindings.items()}, "limits": LIMITS, "cpu_threads": 1,
              "wall_seconds": time.monotonic()-start, "process_lifetime_peak_rss_bytes": peak_rss(),
              "model_calls": 0, "prediction_reads": 0, "no_retry": True})
        check()
    except BaseException as error:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"late-receipt.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "source_sha256": source,
                  "request": request, "error": repr(error), "wall_seconds": time.monotonic()-start,
                  "model_calls": 0, "prediction_reads": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - retain original failure
            error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "summary-sha256", "receipt", "receipt-sha256", "audit", "audit-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--synthetic", action="store_true", help="Require every input summary/receipt to be marked synthetic")
    execute(parser.parse_args())
