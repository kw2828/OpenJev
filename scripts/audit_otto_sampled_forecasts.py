"""Independent saved-only sampled forecast audit with unchanged census metrics.

The private, hash-pinned original auditor supplies process/file authentication,
scalar forecast metrics, saved fit checks and all 45 rules. This adapter changes
only experiment identities, complete collection admission, and reconstruction of
the uniformly sampled TRAIN windows/IPW. It imports no training, sampling,
model, environment or filter implementation. NumPy is used only for saved data
and deterministic replay of recorded sampling/permutation indices.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = "scripts/audit_otto_score_forecasts.py"
BASE_PIN = "c944b2edf7b053c65cb43b3e24ee81a549a1584808cb3fbf0553c28593e9d2ce"
SELF = "scripts/audit_otto_sampled_forecasts.py"
TEST = "tests/test_audit_otto_sampled_forecasts.py"
VERSION = "otto-sampled-forecast-saved-audit-v1"
PRODUCER = "scripts/train_otto_sampled_forecasts.py"
PRODUCER_VERSION = "otto-sampled-forecast-training-v1"
COLLECTOR = "scripts/collect_otto_sampled_forecasts.py"
COLLECTOR_VERSION = "otto-sampled-forecast-collection-v1"
SEEDS = (235001, 235002, 235003)
SELECTION_START = 23600001
COLLECTION_SECONDS = 7200


def _base():
    path = ROOT / BASE_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_PIN:
        raise ValueError("unchanged qualified forecast audit base")
    spec = importlib.util.spec_from_file_location("_sampled_forecast_audit_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _base()
FAMILIES, REGIMES, ARMS = base.FAMILIES, base.REGIMES, base.ARMS
CONFIG = {**base.CONFIG, "fit_seeds": list(SEEDS),
          "training_labels": "uniform_min8_disjoint_windows_per_episode",
          "selection_seed_start": SELECTION_START,
          "training_weight": "(W/k)/(54*full_episode_nonquery_rows)"}
LIMITATIONS = [*base.LIMITATIONS,
    ("TRAIN sampling indices, selected bytes and non-renormalized inverse-inclusion weights are reconstructed; "
     "original teacher-score and public replay truth remain authenticated producer evidence.")]
CONFIGURATION = {"SELF": SELF, "TEST": TEST, "VERSION": VERSION, "PRODUCER": PRODUCER,
    "PRODUCER_VERSION": PRODUCER_VERSION, "COLLECTOR": COLLECTOR, "COLLECTOR_VERSION": COLLECTOR_VERSION,
    "SEEDS": SEEDS, "CONFIG": CONFIG, "LIMITATIONS": LIMITATIONS}
for _name, _value in CONFIGURATION.items():
    setattr(base, _name, _value)


def cohort():
    result, global_case = [], 0
    for stage, size, starts in (("train", 9, (23100001, 23200001)), ("valid", 6, (23300001, 23400001))):
        for regime, start in zip(REGIMES, starts, strict=True):
            for case in range(size):
                shift = global_case % 3
                for arm in ARMS[shift:] + ARMS[:shift]:
                    result.append({"stage": stage, "episode_index": len(result), "regime": regime,
                        "seed": start + case, "case": case, "initial_hit": 1 + case % 3,
                        "arm": arm, "episode_id": f"{stage}:{regime}:{start + case}:{arm}"})
                global_case += 1
    return result


base.cohort = cohort
continuation_rules = base.continuation_rules
scalar_metrics = base.scalar_metrics


class Audit(base.Audit):
    def authenticate(self):
        _, self.worker, self.parent, self.run = self.phase(
            self.args.plan, self.args.worker, self.args.terminal, PRODUCER, ".venv/bin/python", 600)
        self.require(self.worker["version"] == PRODUCER_VERSION and self.worker["fits_completed"] == 12
                     and self.worker["teacher_calls"] == self.worker["native_calls"] == 0,
                     "complete sampled forecast fitting without scientific collection")
        roles = self.plan["inputs"]
        self.require(set(roles) == {"collection_plan", "collection_receipt", "collection_terminal", "engineering"},
                     "exact training source roles")
        self.collection_plan, self.collection_worker, self.collection_parent, self.collection = self.phase(
            self.path(roles["collection_plan"]["path"]), self.path(roles["collection_receipt"]["path"]),
            self.path(roles["collection_terminal"]["path"]), COLLECTOR,
            ".venv-otto-released-native/bin/python", COLLECTION_SECONDS)
        c = self.collection_worker
        payloads = {name + ".jsonl.gz" for name in ("work", "weights", "forwards", "transitions", "samples", "annotations")}
        payloads.update({"started.json", "runtime.json", "setup.json", "deployment.json", "cohort.json",
            "episode-boundaries.jsonl", "episodes.jsonl", "train.npz", "valid.npz", "train-selection.json", "costs.json", "summary.json"})
        self.require(self.collection_plan["version"] == COLLECTOR_VERSION
                     and self.collection_plan["status"] == "frozen_before_collection"
                     and self.collection_plan["limits"] == {"native_seconds": COLLECTION_SECONDS,
                         "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
                     and set(c["files"]) == set(self.collection_plan["payloads"]) == payloads,
                     "new complete collection allocation and eighteen payloads")
        self.require(c["version"] == COLLECTOR_VERSION and c["completed_episodes"] == 90
                     and c["train_episodes"] == 54 and c["valid_episodes"] == 36
                     and c["training_updates"] == 0, "complete new ninety-path collection")
        self.equal(self.collection_plan["cohort"], cohort(), "exact prospective collection cohort")
        self.episodes = list(self.rows(self.collection / "episodes.jsonl"))
        self.require(len(self.episodes) == 90, "all complete episode records")
        for row, identity in zip(self.episodes, cohort(), strict=True):
            self.equal({key: row[key] for key in identity}, identity, "ordered episode identity")
            steps = row["steps"]
            self.require(type(steps) is int and 1 <= steps <= 2188 and row["rows"] == row["updates"] == steps
                         and row["final_update_assimilated"] is True and type(row["found"]) is bool
                         and row["censored"] is (not row["found"]) and (row["found"] or steps == 2188)
                         and row["corrections"] == (steps + 3) // 4, "all complete path tails and final updates")
            self.require(all(type(row[key]) is int and row[key] >= 0 for key in
                             ("teacher_calls", "deployed_queries", "annotation_only", "deferred_annotations", "unlabeled_rows"))
                         and row["teacher_calls"] == row["deployed_queries"] + row["annotation_only"]
                         and row["teacher_calls"] + row["unlabeled_rows"] == steps
                         and row["deferred_annotations"] <= row["annotation_only"], "one returned score per labeled state")
            if row["stage"] == "valid":
                self.require(row["teacher_calls"] == steps and row["unlabeled_rows"] == row["deferred_annotations"] == 0,
                             "full VALID teacher census unchanged")
        engineering = self.read(roles["engineering"]["path"])
        self.require(engineering["status"] == "passed"
                     and engineering["sources_before"] == engineering["sources_after"]
                     and all(row["exit_code"] == 0 for row in engineering["results"]), "qualified engineering scope")

    def reconstruct(self, stage):
        if stage != "train":
            return super().reconstruct(stage)
        np = self.np
        flat = self.arrays(self.collection / "train.npz")
        selections = self.read(self.collection / "train-selection.json")
        identities = [row for row in self.episodes if row["stage"] == "train"]
        self.require(len(identities) == len(selections) == 54, "all TRAIN episodes and selections")
        self.require(set(flat) == {"features", "raw_q", "legal", "actions", "correction", "episode_offsets", "label_mask"},
                     "exact sampled flat array keys")
        offsets = flat["episode_offsets"]
        self.require(offsets.dtype == np.int64 and offsets.shape == (55,) and offsets[0] == 0
                     and bool((np.diff(offsets) >= 1).all()) and bool((np.diff(offsets) <= 2188).all()), "full TRAIN offsets")
        total = int(offsets[-1])
        for name, dtype, shape in (("features", np.float32, (total, 31)), ("raw_q", np.float32, (total, 4)),
                                   ("legal", np.bool_, (total, 4)), ("actions", np.int64, (total,)),
                                   ("correction", np.bool_, (total,)), ("label_mask", np.bool_, (total,))):
            self.require(flat[name].dtype == dtype and flat[name].shape == shape, "exact TRAIN " + name)
        self.require(bool(np.isfinite(flat["features"]).all()) and bool(flat["legal"].any(axis=1).all()), "finite public features and legal support")
        missing = flat["raw_q"][~flat["label_mask"]]
        self.require(missing.tobytes() == np.zeros(missing.shape, np.float32).tobytes(), "unscored positive-zero placeholders")
        lengths = [int(hi - lo) for lo, hi in pairwise(offsets)]
        records = []
        for index, (identity, length) in enumerate(zip(identities, lengths, strict=True)):
            population = (length + 3) // 4
            count = min(8, population)
            seed = SELECTION_START + index
            rng = np.random.Generator(np.random.PCG64(seed))
            starts = sorted(int(value) * 4 for value in rng.choice(population, size=count, replace=False))
            expected = {"episode_id": identity["episode_id"], "version": "otto-sampled-forecast-data-v1",
                "length": length, "seed": seed, "period": 4, "population_windows": population,
                "selected_windows": count, "start_offsets": starts, "inclusion_numerator": count,
                "inclusion_denominator": population, "inclusion_probability": count / population,
                "algorithm": "numpy.Generator(PCG64(seed)).choice(W,size=k,replace=False); sorted"}
            self.equal(selections[index], expected, "independent fixed uniform TRAIN selection")
            self.require(all(type(selections[index][key]) is type(value) for key, value in expected.items())
                         and all(type(value) is int for value in selections[index]["start_offsets"]), "exact selection scalar types")
            records.append(expected)
        n = sum(record["selected_windows"] for record in records)
        full_counts = [length - (length + 3) // 4 for length in lengths]
        output = {"features": np.zeros((n, 4, 31), np.float32), "query_scores": np.zeros((n, 4), np.float32),
            "targets": np.zeros((n, 4, 4), np.float32), "legal": np.zeros((n, 4, 4), np.bool_),
            "valid_mask": np.zeros((n, 4), np.bool_), "nonquery_mask": np.zeros((n, 4), np.bool_),
            "lengths": np.zeros(n, np.int64), "episode_index": np.zeros(n, np.int64),
            "step_offsets": np.zeros(n, np.int64), "nonquery_weights": np.zeros((n, 4), np.float64),
            "episode_lengths": np.asarray(lengths, np.int64), "episode_nonquery_counts": np.asarray(full_counts, np.int64)}
        window = 0
        for index, (identity, record, length) in enumerate(zip(identities, records, lengths, strict=True)):
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            self.require(identity["start_row"] == lo and identity["end_row"] == hi and identity["steps"] == length,
                         "complete full-history row ranges")
            x, actions = flat["features"][lo:hi], flat["actions"][lo:hi]
            steps = np.arange(length)
            self.require(np.array_equal(x[:, 15], (steps / 2188).astype(np.float32))
                         and np.array_equal(x[:, 16], ((steps % 4) / 2188).astype(np.float32))
                         and bool((x[:, 17] == 1).all())
                         and np.array_equal(flat["correction"][lo:hi], steps % 4 == 0)
                         and bool(((actions >= 0) & (actions < 4)).all())
                         and bool(flat["legal"][lo:hi][steps, actions].all()), "full public chronology and legal actions")
            self.require(int(flat["label_mask"][lo:hi].sum()) == identity["teacher_calls"], "all physically labeled rows accounted")
            row_weight = (record["population_windows"] / record["selected_windows"]) / (54 * full_counts[index]) if full_counts[index] else 0.
            for start in record["start_offsets"]:
                size = min(4, length - start)
                sl = slice(lo + start, lo + start + size)
                self.require(bool(flat["label_mask"][sl].all()) and bool(np.isfinite(flat["raw_q"][sl]).all()), "selected labels complete and finite")
                for key, source in (("features", "features"), ("targets", "raw_q"), ("legal", "legal")):
                    output[key][window, :size] = flat[source][sl]
                output["query_scores"][window] = flat["raw_q"][lo + start]
                output["valid_mask"][window, :size] = True
                output["nonquery_mask"][window, 1:size] = True
                output["lengths"][window] = size
                output["episode_index"][window] = index
                output["step_offsets"][window] = start
                output["nonquery_weights"][window, 1:size] = row_weight
                window += 1
            self.check()
        selected_rows = int(output["valid_mask"].sum())
        counts = {"episodes": 54, "windows": n, "rows": selected_rows, "query_rows": n,
            "nonquery_rows": selected_rows - n, "zero_support_episodes": sum(value == 0 for value in full_counts),
            "query_only_windows": int((output["lengths"] == 1).sum()), "full_rows": total,
            "population_windows": sum(record["population_windows"] for record in records), "full_nonquery_rows": sum(full_counts)}
        meta = {"version": "otto-score-forecast-data-v1", "episode_ids": [row["episode_id"] for row in identities],
            "episode_regimes": [row["regime"] for row in identities], "episode_splits": ["train"] * 54,
            "counts": counts, "sampling": {"version": "otto-sampled-forecast-data-v1",
                "unit": "disjoint period-four TRAIN windows", "episode_denominator": 54, "window_cap": 8,
                "selections": records, "weight_formula": "(population_windows/selected_windows)/(54*full_episode_nonquery_rows)",
                "realized_weight_mass": float(output["nonquery_weights"].sum()), "renormalized": False,
                "shared_across_fits": True, "validation": "unchanged full census"}}
        self.counts.update(train_rows=total, train_windows=n, selected_train_rows=selected_rows,
                           selected_train_nonquery_rows=selected_rows - n)
        return output, meta, identities

    def saved_windows(self, stage, windows, meta):
        saved = self.arrays(self.run / f"{stage}-windows.npz")
        self.require(set(saved) == set(windows), "exact persisted window keys")
        for key, expected in windows.items():
            value = saved[key]
            self.require(value.dtype == expected.dtype and value.shape == expected.shape
                         and value.tobytes() == expected.tobytes(), "independent persisted window bytes " + key)
        self.equal(self.read(self.run / f"{stage}-windows.json"), meta, "persisted identities, selection and weights")

    def body(self):
        self.authenticate()
        import numpy as np
        self.np = np
        runtime = self.read(self.run / "runtime.json")
        self.require(runtime["threads"] == runtime["interop_threads"] == 1
                     and runtime["deterministic"] is True and runtime["cuda_used"] is runtime["mps_used"] is False,
                     "inherited deterministic CPU training runtime")
        train, train_meta, _ = self.reconstruct("train")
        self.saved_windows("training", train, train_meta)
        del train
        fits = self.training(train_meta)
        windows, meta, identities = self.reconstruct("valid")
        saved_windows = self.arrays(self.run / "validation-windows.npz")
        self.require(set(saved_windows) == set(windows), "exact saved validation window schema")
        for key, value in windows.items():
            actual = saved_windows[key]
            self.require(actual.dtype == value.dtype and actual.shape == value.shape
                         and actual.tobytes() == value.tobytes(), "reconstructed validation bytes " + key)
        del saved_windows
        self.equal(self.read(self.run / "validation-windows.json"), meta, "validation identities and coverage")
        models, hold = self.predictions(windows, meta, identities, fits)
        support = {regime: {str(age): len({row["case"] for row, length in zip(identities, windows["episode_lengths"], strict=True)
                                          if row["regime"] == regime and int(length) > age})
                           for age in (1, 2, 3)} for regime in REGIMES}
        rules = continuation_rules(models, hold, support)
        summary = self.read(self.run / "summary.json")
        expected = {"version": PRODUCER_VERSION, "models": models, "hold": hold, "support": support,
                    "required": rules, "required_passed": sum(row["passes"] for row in rules),
                    "required_conditions": 45, "forecast_continuation": all(row["passes"] for row in rules),
                    "requires_successful_original_supervisor_and_saved_audit": True,
                    "train_counts": train_meta["counts"], "validation_counts": meta["counts"],
                    "scope": "Forced-path forecasts on fresh VALID, not autonomous performance or true action regret."}
        for field in ("setup_seconds", "fitting_seconds", "validation_seconds"):
            number = summary[field]
            self.require(type(number) in (int, float) and math.isfinite(number) and number >= 0,
                         "finite physical forecast stage " + field)
            expected[field] = number
        self.require(math.fsum(fit["wall_seconds"] for fit in fits) <= summary["fitting_seconds"] + 1e-9,
                     "fit intervals contained in fitting phase")
        self.require(math.fsum(summary[key] for key in ("setup_seconds", "fitting_seconds", "validation_seconds"))
                     <= self.worker["wall_seconds"] + 1e-9, "physical training stages bounded by original worker")
        self.equal(summary, expected, "all saved forecast reductions and criteria")
        payloads = {"started.json", "runtime.json", "progress.jsonl", "fits.json", "validation-windows.npz",
                    "validation-windows.json", "prediction-hold.npz", "summary.json",
                    "training-windows.npz", "training-windows.json"}
        for fit in fits:
            payloads.update((fit["checkpoint_path"], "prediction-" + fit["checkpoint_path"]))
        self.require(set(self.worker["files"]) == payloads, "exact complete forecast payload membership")
        self.counts.update(required_conditions=45, required_passed=expected["required_passed"],
                           collection_episodes=90, prediction_files=13, source_files=len(self.plan["sources"]),
                           training_payloads=len(payloads), collection_payloads=len(self.collection_worker["files"]))
        self.result = {"version": VERSION, "agreement": True, "summary": expected,
                       "fits": fits, "counts": dict(self.counts), "limitations": LIMITATIONS}


# Use the untouched original supervised audit lifecycle and command parser.
base.Audit = Audit

if __name__ == "__main__":
    base.main()
