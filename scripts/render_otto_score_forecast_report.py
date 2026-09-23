"""Render complete audited score forecasts from two saved JSON files only.

The caller supplies the closed producer summary and independent audit.json.
This presentation tool requires their full summaries to agree exactly; it does
not reopen checkpoints, arrays, collection journals or scientific runtimes.
Original process admission remains the caller's responsibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

VERSION = "otto-score-forecast-render-v1"
KINDS = ("residual_gru", "direct_gru", "history_mlp", "current_mlp")
SEEDS = (225001, 225002, 225003)
REGIMES = ("lambda3", "lambda4")
COLLECTORS = ("analytic", "neural", "period4_hold")
LABELS = {"hold": "Hold query Q", "residual_gru": "Residual GRU", "direct_gru": "Direct GRU",
          "history_mlp": "History MLP", "current_mlp": "Current MLP"}
COLORS = {"hold": "#333333", "residual_gru": "#1765A5", "direct_gru": "#D17C17",
          "history_mlp": "#7552A1", "current_mlp": "#258474"}
METRICS = (("episode_weighted_raw_gap", "Raw teacher-score gap (lower is better)", 1.),
           ("episode_weighted_agreement", "Teacher near-minimum agreement (%)", 100.))
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "regular nonsymlink input")
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
            size += len(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": size}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path, rows):
    require(bool(rows), "nonempty complete export")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
        stream.flush()
        os.fsync(stream.fileno())


def authenticate(summary, audit):
    require(summary["version"] == "otto-score-forecast-training-v1"
            and audit["version"] == "otto-score-forecast-saved-audit-v1"
            and audit["agreement"] is True, "completed agreeing score-forecast evidence")
    require(canonical(summary) == canonical(audit["summary"]), "exact full independently audited summary")
    expected = [(family, seed) for seed in SEEDS for family in KINDS]
    require([(r["family"], r["seed"]) for r in summary["models"]] == expected,
            "all twelve final fits in declared order")
    require([(r["family"], r["seed"]) for r in audit["fits"]] == expected,
            "all twelve audited fit records")
    checks = summary["required"]
    require(len(checks) == len({c["name"] for c in checks}) == summary["required_conditions"] == 45
            and all(type(c["passes"]) is bool for c in checks), "all forty-five fixed conditions")
    count = sum(c["passes"] for c in checks)
    require(count == summary["required_passed"] == audit["counts"]["required_passed"]
            and audit["counts"]["required_conditions"] == 45
            and summary["forecast_continuation"] is (count == 45), "unchanged complete continuation rule")
    require(summary["train_counts"]["episodes"] == 54 and summary["validation_counts"]["episodes"] == 36
            and audit["counts"]["collection_episodes"] == 90
            and audit["counts"]["fits"] == 12 and audit["counts"]["prediction_files"] == 13,
            "complete training and forecast coverage")
    return [{"family": "hold", "seed": None, "metrics": summary["hold"]}, *summary["models"]]


def tables(records):
    rows, points, means = [], [], []
    for record in records:
        family, seed, report = record["family"], record["seed"], record["metrics"]
        name = family if seed is None else f"{family}@{seed}"
        require(set(report["by_regime"]) == set(REGIMES)
                and set(report["by_collector"]) == set(COLLECTORS), "every setting and collector retained")
        scopes = [("all", report), *[(arm, report["by_collector"][arm]) for arm in COLLECTORS]]
        for collector, scoped in scopes:
            require(set(scoped["by_regime"]) == set(REGIMES), "both settings within every collector")
            groups = [("all", scoped["overall"]), *[(r, scoped["by_regime"][r]) for r in REGIMES]]
            for regime, group in groups:
                require(set(group["by_age"]) == {"1", "2", "3"}, "all three nonquery ages")
                ages = [("all", group), *[(str(age), group["by_age"][str(age)]) for age in (1, 2, 3)]]
                for age, values in ages:
                    rows.append({"model": name, "family": family, "seed": seed, "collector": collector,
                                 "regime": regime, "age": age,
                                 **{k: v for k, v in values.items() if k != "by_age"}})
        for regime in REGIMES:
            values = report["by_regime"][regime]
            for metric, _, multiplier in METRICS:
                value = values[metric]
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                        "finite nonnegative plotted metric")
                if metric.endswith("agreement"):
                    require(value <= 1, "agreement fraction bound")
                points.append({"model": name, "family": family, "seed": seed, "regime": regime,
                               "metric": metric, "raw_value": value, "display_value": value * multiplier,
                               "episode_weight_mass": values["weight_mass"]})
    for family in KINDS:
        for regime in REGIMES:
            for metric, _, multiplier in METRICS:
                selected = [p for p in points if (p["family"], p["regime"], p["metric"])
                            == (family, regime, metric)]
                require({p["seed"] for p in selected} == set(SEEDS) and len(selected) == 3,
                        "equal mean over all three fixed fits")
                value = math.fsum(p["raw_value"] for p in selected) / 3
                means.append({"family": family, "regime": regime, "metric": metric,
                              "raw_value": value, "display_value": value * multiplier, "seeds": list(SEEDS)})
    require(len(rows) == 624 and len(points) == 52 and len(means) == 16, "complete export and figure geometry")
    return rows, points, means


def figure(output, summary, points, means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey="row")
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.23, wspace=.15, hspace=.36)
    families = ("hold", *KINDS)
    for row, (metric, label, _) in enumerate(METRICS):
        ceiling = max(p["display_value"] for p in points if p["metric"] == metric)
        for column, regime in enumerate(REGIMES):
            ax = axes[row, column]
            selected = [p for p in points if p["regime"] == regime and p["metric"] == metric]
            held = next(p for p in selected if p["family"] == "hold")
            ax.axhline(held["display_value"], color=COLORS["hold"], linestyle="--", linewidth=1., alpha=.7)
            ax.plot(0, held["display_value"], marker="x", color=COLORS["hold"], markersize=8, linestyle="none")
            for x, family in enumerate(KINDS, start=1):
                for seed, shift, marker in zip(SEEDS, (-.14, 0., .14), ("o", "s", "^"), strict=True):
                    value = next(p["display_value"] for p in selected if p["family"] == family and p["seed"] == seed)
                    ax.plot(x + shift, value, marker=marker, markersize=6, markerfacecolor="white",
                            markeredgecolor=COLORS[family], linestyle="none", zorder=3)
                mean = next(m["display_value"] for m in means
                            if (m["family"], m["regime"], m["metric"]) == (family, regime, metric))
                ax.plot((x - .23, x + .23), (mean, mean), color=COLORS[family], linewidth=2.5, zorder=4)
            ax.set_xticks(range(5), [LABELS[f].replace(" ", "\n", 1) for f in families])
            ax.set_xlim(-.45, 4.45)
            ax.set_ylim(0, 105 if metric.endswith("agreement") else (ceiling * 1.12 if ceiling else 1))
            ax.grid(axis="y", alpha=.18)
            if column == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"Sensing length {regime[-1]} | 6 paired VALID cases", fontsize=12, pad=12)
    count = summary["required_passed"]
    outcome = "PASS" if summary["forecast_continuation"] else "FAIL"
    fig.suptitle("Can remembered planner scores predict the next three decisions?", fontsize=17, y=.975)
    fig.text(.5, .915, f"Forced-path VALID forecast | 12 fits + hold-Q | Fixed screen: {outcome} ({count}/45)",
             ha="center", fontsize=12)
    handles = [Line2D([], [], color="#444444", marker=m, markerfacecolor="white", linestyle="none", label=f"Seed {s}")
               for s, m in zip(SEEDS, ("o", "s", "^"), strict=True)]
    handles += [Line2D([], [], color="#444444", linewidth=2.5, label="Mean of all 3 seeds"),
                Line2D([], [], color=COLORS["hold"], linestyle="--", marker="x", label="Hold-Q baseline")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .14), ncol=5, frameon=False)
    fig.text(.5, .105, "Every fit is shown. Equal episode weights; query rows excluded. "
             "Zero-support episodes retain zero mass contribution.", ha="center", fontsize=9)
    fig.text(.5, .067, "Gap uses raw teacher-score units; agreement permits the teacher's strict float32 near-minimum set. "
             "Shared vertical scales within each metric.", ha="center", fontsize=9)
    fig.text(.5, .029, "These are planner-imitation forecasts on collected paths, not autonomous search results, "
             "true action regret or evidence of novelty.", ha="center", fontsize=9)
    for suffix in ("png", "svg"):
        with (output / f"score-forecast.{suffix}").open("xb") as stream:
            fig.savefig(stream, format=suffix, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.2)
            stream.flush()
            os.fsync(stream.fileno())
    plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(all(path.is_absolute() for path in (args.summary, args.audit, args.output)), "absolute input/output paths")
    require(not any(p.is_symlink() for p in args.output.parents), "nonsymlink output parents")
    args.output.mkdir(parents=False, exist_ok=False)
    start = time.monotonic()
    receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
               "scope": "Saved JSON presentation only; original process admission is supplied by the caller."}

    def check():
        require(time.monotonic() - start < LIMITS["seconds"], "render time bound")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
                <= LIMITS["output_bytes"] - 1024**2, "render RSS/output bounds with failure reserve")

    try:
        inputs = {name: descriptor(getattr(args, name)) for name in ("summary", "audit")}
        require(all(d["bytes"] <= 16 * 1024**2 for d in inputs.values()), "bounded saved JSON inputs")
        receipt.update(inputs=inputs, source=descriptor(Path(__file__).absolute()))
        summary, audit = (json.loads(p.read_text()) for p in (args.summary, args.audit))
        records = authenticate(summary, audit)
        rows, points, means = tables(records)
        check()
        write_csv(args.output / "forecast-metrics.csv", rows)
        write_csv(args.output / "family-means.csv", means)
        write_csv(args.output / "conditions.csv", summary["required"])
        write_csv(args.output / "fits.csv", audit["fits"])
        write_json(args.output / "plotted-values.json", {"version": VERSION, "individual_points": points,
            "family_means": means, "required": summary["required"], "scope": summary["scope"],
            "train_counts": summary["train_counts"], "validation_counts": summary["validation_counts"],
            "support": summary["support"], "audit_limitations": audit["limitations"],
            "stage_seconds": {key: summary[key] for key in ("setup_seconds", "fitting_seconds", "validation_seconds")}})
        receipt["matplotlib_version"] = figure(args.output, summary, points, means)
        check()
        require(all(descriptor(getattr(args, name)) == value for name, value in inputs.items()), "unchanged saved inputs")
        require(descriptor(Path(__file__).absolute()) == receipt["source"], "unchanged renderer source")
        receipt.update(status="completed", required_passed=summary["required_passed"], required_conditions=45,
            forecast_continuation=summary["forecast_continuation"], metric_rows=len(rows), individual_points=len(points),
            family_mean_points=len(means), files={p.name: descriptor(p) for p in sorted(args.output.iterdir()) if p.is_file()},
            wall_seconds=time.monotonic() - start)
        write_json(args.output / "receipt.json", receipt)
        check()
        print(json.dumps({"status": "completed", "receipt": descriptor(args.output / "receipt.json")}), flush=True)
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc(), wall_seconds=time.monotonic() - start)
        try:
            if (args.output / "receipt.json").exists():
                (args.output / "receipt.json").rename(args.output / "receipt.invalid.json")
            receipt["files"] = {p.name: descriptor(p) for p in args.output.iterdir() if p.is_file()}
            write_json(args.output / "receipt.json", receipt)
        except BaseException as publication:  # noqa: BLE001 - preserve original presentation failure
            error.add_note(f"Failure receipt publication: {publication!r}")
        raise


if __name__ == "__main__":
    main()
