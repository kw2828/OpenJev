"""Present closed ranking-diagnostic JSON only; no arrays or model calls.

Every fitted seed and every fixed contrast is retained in the CSV exports.
The hash-bound producer supplies byte/process guards only, not numeric reducers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/plot_otto_ranking_diagnosis.py"
HELPER = "scripts/diagnose_otto_ranking.py"
VERSION = "otto-ranking-diagnosis-plot-v1"
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
REGIMES = ("lambda3", "lambda4")
FAMILIES = tuple(f"{a}_{r}_{o}" for a in ("innovation", "gru")
                 for r in ("shared", "separate") for o in ("mse", "aux"))
SEEDS = (285000001, 285000002, 285000003)
LABELS = tuple(("Explicit" if f.startswith("innovation") else "GRU") + "\n"
               + f.split("_")[1] + "/" + f.split("_")[2].upper() for f in FAMILIES)
COLORS = ("#819abc", "#355f98", "#58a2d3", "#006b9b", "#a0bcb0", "#557e6b", "#54ab87", "#00734c")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            h.update(part)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def load_helper(args):
    require(args.plan.is_absolute() and args.plan.is_relative_to(ROOT)
            and not any(p.is_symlink() for p in (args.plan, *args.plan.parents)), "regular external plan")
    require(digest(args.plan)["sha256"] == args.plan_sha256 and args.plan.stat().st_size <= 16 * 1024**2,
            "external bounded plan hash")
    plan = json.loads(args.plan.read_text())
    for name in (HELPER, "src/openjev/research/suspend_clock.py"):
        require(digest(ROOT / name) == plan["sources"][name], "frozen byte helper before import")
    spec = importlib.util.spec_from_file_location("_ranking_plot_byte_helpers", ROOT / HELPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, plan


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = ";".join(str(v) for v in item) if isinstance(item, list) else item
    return result


def export_rows(records, paired=False):
    for record in records:
        identity = {k: v for k, v in record.items() if k != "stats"}
        if paired:
            stats = {k: v for k, v in record["stats"].items() if k != "cells"}
            for cell in ("CC", "CW", "WC", "WW"):
                yield flatten({**identity, **stats, "cell": cell, "cell_statistics": record["stats"]["cells"][cell]})
        else:
            yield flatten({**identity, **record["stats"]})


class Plot:
    def __init__(self, args, helper, plan):
        self.args, self.helper, self.plan, self.out = args, helper, plan, args.output
        self.clock = helper.SuspendClock()
        self.start = self.clock.now_ns()
        self.created = False
        self.bound = {}
        self.receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
                        "started_ns": self.start, "model_calls": 0, "array_decodes": 0,
                        "new_scientific_gate": False, "scope": "presentation of audited saved JSON"}

    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS["seconds"] * 10**9, "presentation time bound")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "presentation memory bound")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) < LIMITS["output_bytes"] - 1024**2,
                "presentation output bound")

    def bind(self, path, expected):
        self.check()
        actual = self.helper.pin(path)
        require(actual == expected if isinstance(expected, dict) else actual["sha256"] == expected,
                "unchanged input: " + str(path))
        self.bound[str(path)] = actual

    def authenticate(self):
        self.bind(self.args.plan, self.args.plan_sha256)
        self.bind(self.args.closure, self.args.closure_sha256)
        closure = self.helper.read(self.args.closure)
        require(closure["status"] == "completed" and set(closure["phases"]) == {"diagnosis", "audit"}
                and closure["plan"] == {"path": str(self.args.plan), **self.bound[str(self.args.plan)]},
                "closed exact diagnosis and audit")
        self.helper.authenticate(self.plan, self.check)
        for name, pin in self.plan["sources"].items():
            self.bind(ROOT / name, pin)
        for item in self.plan["inputs"].values():
            self.bound[item["path"]] = {k: item[k] for k in ("sha256", "bytes")}
        workers, results, options = {}, {}, {}
        for phase in ("diagnosis", "audit"):
            record = closure["phases"][phase]
            for role in ("worker", "terminal"):
                item = record[role]
                self.bind(Path(item["path"]), {k: item[k] for k in ("sha256", "bytes")})
            worker_path, terminal_path = (Path(record[r]["path"]) for r in ("worker", "terminal"))
            worker, terminal = self.helper.read(worker_path), self.helper.read(terminal_path)
            require(worker_path == Path(self.plan["outputs"][phase]) / "receipt.json"
                    and worker["version"] == self.helper.VERSION and worker["phase"] == phase
                    and worker["sources"] == self.plan["sources"] and worker["inputs"] == self.plan["inputs"]
                    and worker["limits"] == self.helper.LIMITS and worker["npz_decodes"] == 26
                    and worker["requires_successful_original_supervisor"] is True
                    and worker["new_scientific_gate"] is False
                    and all(worker[k] == 0 for k in ("model_calls", "native_calls", "optimizer_calls")), "closed saved-only phase")
            filename = "diagnose_otto_ranking.py" if phase == "diagnosis" else "audit_otto_ranking_diagnosis.py"
            opts = self.helper.original_join(worker, terminal, script=filename, output=worker_path.parent,
                plan=self.args.plan, plan_pin=self.args.plan_sha256, cap=240)
            expected_options = {"--plan", "--plan-sha256", "--supervision", "--output"}
            if phase == "audit":
                expected_options |= {"--worker", "--worker-sha256", "--terminal", "--terminal-sha256"}
                for role in ("worker", "terminal"):
                    require(opts["--" + role] == closure["phases"]["diagnosis"][role]["path"]
                            and opts["--" + role + "-sha256"] == closure["phases"]["diagnosis"][role]["sha256"],
                            "audit selected this original diagnostic")
            require(set(opts) == expected_options, "exact original argument set")
            result_name = phase + ".json"
            require(set(worker["files"]) == {"started.json", result_name}
                    and {p.name for p in worker_path.parent.iterdir()} == {"started.json", result_name, "receipt.json"},
                    "closed phase payload set")
            for name, pin in worker["files"].items():
                self.bind(worker_path.parent / name, pin)
            launch = Path(opts["--supervision"])
            self.bind(launch, worker["supervision_sha256"])
            require(self.helper.read(worker_path.parent / "started.json") == {
                "phase": phase, "launch": self.helper.read(launch), "plan_sha256": self.args.plan_sha256}, "startup joins")
            require(record["wall_seconds"] == terminal["wall_seconds"]
                    and record["peak_rss_bytes"] == worker["peak_rss_bytes"], "closure accounting matches worker")
            workers[phase], options[phase] = worker, opts
            results[phase] = self.helper.read(worker_path.parent / result_name, limit_bytes=self.helper.LIMITS["output_bytes"])
        audit = results["audit"]
        require(workers["audit"]["agreement"] is audit["agreement"] is True
                and audit["version"] == "otto-ranking-diagnosis-saved-audit-v1"
                and workers["diagnosis"]["counts"] == workers["audit"]["counts"] == audit["counts"] == results["diagnosis"]["counts"],
                "complete independent arithmetic agreement")
        expected_producer = {"worker": closure["phases"]["diagnosis"]["worker"],
            "terminal": closure["phases"]["diagnosis"]["terminal"],
            "launch": {"path": options["diagnosis"]["--supervision"], **self.bound[options["diagnosis"]["--supervision"]]},
            "result": {"path": str(Path(self.plan["outputs"]["diagnosis"]) / "diagnosis.json"),
                       **workers["diagnosis"]["files"]["diagnosis.json"]}}
        require(workers["audit"]["producer_inputs"] == audit["producer_inputs"] == expected_producer,
                "independent audit binds every displayed diagnostic")
        self.receipt["original_audit_checks"] = audit["checks"]
        self.receipt["counts"] = audit["counts"]
        return results["diagnosis"]

    def json(self, name, value):
        self.helper.write(self.out / name, value)

    def csv(self, name, rows):
        rows = iter(rows)
        first = next(rows)
        count = 1
        with (self.out / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(first), lineterminator="\n")
            writer.writeheader(); writer.writerow(first)
            for record in rows:
                writer.writerow(record); count += 1
                if count % 512 == 0:
                    self.check()
            stream.flush(); os.fsync(stream.fileno())
        return count

    def save(self, fig, name):
        for suffix in ("png", "svg"):
            self.check()
            with (self.out / f"{name}.{suffix}").open("xb") as stream:
                fig.savefig(stream, format=suffix, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.18)
                stream.flush(); os.fsync(stream.fileno())

    def figures(self, result):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none",
                             "axes.spines.top": False, "axes.spines.right": False})
        fields = (("legal_centered_mse", "Centered MSE (raw score squared) | lower is better", 1.),
                  ("raw_gap", "Raw teacher-cost gap | lower is better", 1.),
                  ("agreement", "Teacher near-best agreement (%) | higher is better", 100.))
        points, means = [], []
        selected = [r for r in result["individual_groups"] if r["scope"] == "full" and r["group"] in REGIMES]
        selected_means = [r for r in result["individual_seed_means"] if r["scope"] == "full" and r["group"] in REGIMES]
        require(len(selected) == 50 and len(selected_means) == 16, "all individual figure seeds and means")
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey="col")
        for ri, regime in enumerate(REGIMES):
            for ci, (field, title, scale) in enumerate(fields):
                ax = axes[ri, ci]
                for fi, family in enumerate(FAMILIES):
                    records = sorted((r for r in selected if r["family"] == family and r["group"] == regime), key=lambda r: r["seed"])
                    require([r["seed"] for r in records] == list(SEEDS), "all three seeds")
                    for offset, record in zip((-.17, 0, .17), records, strict=True):
                        value = record["stats"]["metrics"][field] * scale
                        require(math.isfinite(value), "finite plotted value")
                        ax.scatter(fi + offset, value, s=25, color=COLORS[fi], alpha=.65, zorder=3)
                        points.append({"family": family, "seed": record["seed"], "regime": regime, "scope": "full",
                                       "metric": field, "raw_value": record["stats"]["metrics"][field], "display_value": value,
                                       "episodes": record["stats"]["episodes"], "weight_mass": record["stats"]["weight_mass"]})
                    record = next(r for r in selected_means if r["family"] == family and r["group"] == regime)
                    value = record["stats"]["metrics"][field] * scale
                    ax.scatter(fi, value, s=65, marker="D", color=COLORS[fi], edgecolor="white", linewidth=.7, zorder=4)
                    means.append({"family": family, "regime": regime, "scope": "full", "metric": field,
                                  "raw_value": record["stats"]["metrics"][field], "display_value": value})
                hold = next(r for r in selected if r["family"] == "hold" and r["group"] == regime)
                value = hold["stats"]["metrics"][field] * scale
                ax.axhline(value, color="#484848", ls="--", lw=1.3)
                points.append({"family": "hold", "seed": None, "regime": regime, "scope": "full", "metric": field,
                               "raw_value": hold["stats"]["metrics"][field], "display_value": value,
                               "episodes": hold["stats"]["episodes"], "weight_mass": hold["stats"]["weight_mass"]})
                ax.set_title(f"Sensing length {regime[-1]} | {title}", fontsize=10)
                ax.set_xticks(range(8), LABELS, rotation=35, ha="right", fontsize=8)
                ax.grid(axis="y", alpha=.2)
                if field == "agreement":
                    ax.set_ylim(0, 100)
                else:
                    ax.set_ylim(bottom=0)
        fig.suptitle("Retrospective teacher imitation: all fitted seeds and the held-score control", fontsize=16, y=.98)
        fig.text(.5, .075, "Small dots: three fixed seeds | diamonds: arithmetic seed means | dashed line: hold\n"
                 "Full nonquery scope. Equal episode weights; all 18 episodes per setting remain in the denominator, including zero support.",
                 ha="center", fontsize=10)
        fig.text(.5, .025, "Fixed-path forecasts, not autonomous control. Descriptive diagnosis only; no new efficacy decision.", ha="center", fontsize=10)
        fig.subplots_adjust(top=.9, bottom=.24, hspace=.8, wspace=.25)
        self.save(fig, "otto-ranking-levels"); plt.close(fig)

        names = list(dict.fromkeys(r["contrast"] for r in result["paired_seed_means"]))
        require(len(names) == 12, "all twelve directed contrasts")
        labels = [n.replace("innovation_to_gru.", "Explicit to GRU: ").replace("innovation.", "Explicit: ")
                  .replace("gru.", "GRU: ").replace("_to_", " to ").replace("_", " ") for n in names]
        contrasts = []
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True, sharex="col")
        for ri, regime in enumerate(REGIMES):
            for ci, (field, title, scale) in enumerate((fields[1], fields[2])):
                ax = axes[ri, ci]
                for scope, offset, color, marker in (("full", -.12, "#286f96", "o"),
                                                      ("postcorrection", .12, "#be613e", "s")):
                    for ni, name in enumerate(names):
                        record = next(r for r in result["paired_seed_means"] if r["contrast"] == name
                                      and r["scope"] == scope and r["group"] == regime)
                        value = math.fsum(cell["delta"][field] for cell in record["stats"]["cells"].values())
                        ax.scatter(value * scale, ni + offset, color=color, marker=marker, s=36,
                                   label=scope if ni == 0 else None, zorder=3)
                        contrasts.append({"contrast": name, "regime": regime, "scope": scope, "metric": field,
                                          "raw_delta": value, "display_delta": value * scale, "seed_count": 3})
                ax.axvline(0, color="#444444", lw=1)
                ax.set_yticks(range(12), labels, fontsize=8)
                ax.set_title(f"Sensing length {regime[-1]} | candidate minus baseline", fontsize=11)
                ax.set_xlabel("Raw gap change (negative favors candidate)" if field == "raw_gap"
                              else "Agreement change in percentage points (positive favors candidate)")
                ax.grid(axis="x", alpha=.2)
                ax.legend(loc="best", fontsize=8)
        axes[0, 0].invert_yaxis()
        fig.suptitle("Every fixed contrast: full versus postcorrection teacher imitation", fontsize=16, y=.975)
        fig.text(.5, .065, "All three seed-paired differences averaged; no contrast or winning seed selected. Postcorrection: nonquery steps >= 5.\n"
                 "Each scope uses its own within-episode row denominator; zero-support episodes remain in the fixed episode denominator.",
                 ha="center", fontsize=10)
        fig.text(.5, .025, "Fixed-path forecasts, not autonomous control. Scope changes also change weights; phase CSVs preserve the full-scope denominator.",
                 ha="center", fontsize=10)
        fig.subplots_adjust(left=.25, right=.98, top=.9, bottom=.16, hspace=.35, wspace=.15)
        self.save(fig, "otto-ranking-contrasts"); plt.close(fig)
        require(len(points) == 150 and len(means) == 48 and len(contrasts) == 96, "full plotted coverage")
        return {"individual_points": points, "individual_means": means, "contrast_means": contrasts}

    def run(self):
        try:
            require(self.out.is_absolute() and self.out.parent == ROOT / "output/otto-ranking-diagnosis-v1"
                    and not any(p.is_symlink() for p in (self.out, *self.out.parents)), "new presentation output only")
            self.out.mkdir(exist_ok=False)
            self.created = True
            self.bind(ROOT / SELF, digest(ROOT / SELF))
            result = self.authenticate()
            exports = {}
            for table, filename in (("individual_groups", "individual-seeds.csv"),
                    ("individual_seed_means", "individual-means.csv"), ("paired_groups", "paired-seeds.csv"),
                    ("paired_seed_means", "paired-means.csv"), ("phase_groups", "phase-seeds.csv"),
                    ("phase_seed_means", "phase-means.csv")):
                exports[filename] = self.csv(filename, export_rows(result[table], paired=not table.startswith("individual")))
                require(exports[filename] == result["counts"][table] * (1 if table.startswith("individual") else 4),
                        "all exported groups and transition cells")
            plotted = self.figures(result)
            plotted.update(version=VERSION, definitions=result["definitions"], export_rows=exports,
                           scope="audited retrospective teacher imitation; no new efficacy decision")
            self.json("plotted-values.json", plotted)
            for path, expected in self.bound.items():
                self.check(); require(self.helper.pin(Path(path)) == expected, "input unchanged at closure")
            self.receipt.update(status="completed", inputs=self.bound, csv_rows=exports,
                individual_points=150, individual_means=48, contrast_means=96,
                files={p.name: self.helper.pin(p) for p in sorted(self.out.iterdir()) if p.is_file()})
            self.check()
            finish = self.clock.now_ns()
            self.receipt.update(finished_ns=finish, wall_seconds=(finish - self.start) / 1e9)
            self.json("receipt.json", self.receipt); self.check()
            print(json.dumps({"status": "completed", "receipt": self.helper.pin(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            if self.created:
                try:
                    if (self.out / "receipt.json").exists():
                        (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                    self.receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc(), inputs=self.bound,
                        files={p.name: self.helper.pin(p) for p in self.out.iterdir() if p.is_file()})
                    self.json("receipt.json", self.receipt)
                except BaseException as secondary:  # noqa: BLE001 - retain original presentation failure
                    error.add_note("Failure publication: " + repr(secondary))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("plan", "closure"):
        parser.add_argument("--" + role, type=Path, required=True)
        parser.add_argument("--" + role + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    helper, plan = load_helper(args)
    Plot(args, helper, plan).run()


if __name__ == "__main__":
    main()
