"""Sampled TRAIN adapter; unchanged forecast optimizer, models and census VALID.

The private base module is loaded only after its exact source hash is checked.
The explicit configuration table below changes experiment identities and seeds,
not the inherited optimizer/loss/prediction/45-rule implementations. Disk sources
remain unchanged. Only selected, positively witnessed teacher rows enter TRAIN.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = "scripts/train_otto_score_forecasts.py"
BASE_PIN = "ddb61875b041c2f8887199c17e1140be9d7abe4de4634fb68706fc0805e6a601"
SELF = "scripts/train_otto_sampled_forecasts.py"
TEST = "tests/test_train_otto_sampled_forecasts.py"
AUDIT = "scripts/audit_otto_sampled_forecasts.py"
AUDIT_TEST = "tests/test_audit_otto_sampled_forecasts.py"
SAMPLED = "src/openjev/research/otto_sampled_forecast_data.py"
PROTOCOL = "research/otto-sampled-forecast-protocol.md"
VERSION = "otto-sampled-forecast-training-v1"
COLLECTOR = "scripts/collect_otto_sampled_forecasts.py"
COLLECTOR_VERSION = "otto-sampled-forecast-collection-v1"
SEEDS = (235001, 235002, 235003)
SELECTION_START = 23600001
COLLECTION_SECONDS = 7200


def _base():
    path = ROOT / BASE_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_PIN:
        raise ValueError("unchanged qualified forecast training base")
    spec = importlib.util.spec_from_file_location("_sampled_forecast_training_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _base()
require = base.require
KINDS = base.KINDS
CONFIG = {**base.CONFIG, "fit_seeds": list(SEEDS),
          "training_labels": "uniform_min8_disjoint_windows_per_episode",
          "selection_seed_start": SELECTION_START,
          "training_weight": "(W/k)/(54*full_episode_nonquery_rows)"}
NEW = base.NEW | {BASE_PATH, SELF, TEST, AUDIT, AUDIT_TEST, SAMPLED, PROTOCOL,
                  "tests/test_otto_sampled_forecast_data.py"}
CONFIGURATION = {"SELF": SELF, "TEST": TEST, "AUDIT": AUDIT, "AUDIT_TEST": AUDIT_TEST,
                 "VERSION": VERSION, "PROTOCOL": PROTOCOL, "SEEDS": SEEDS,
                 "CONFIG": CONFIG, "NEW": NEW}
for _name, _value in CONFIGURATION.items():
    setattr(base, _name, _value)


def collection_inputs(inputs):
    """Admit only the new fully closed 90-path sampled collection."""
    require(set(inputs) == set(base.ROLES), "exact sampled training inputs")
    for d in inputs.values():
        require(base.descriptor(d["path"]) == {k: d[k] for k in ("sha256", "bytes")}, "input hash")
    plan = base.read(inputs["collection_plan"]["path"])
    receipt = base.read(inputs["collection_receipt"]["path"])
    parent = base.read(inputs["collection_terminal"]["path"])
    require(plan["version"] == receipt["version"] == COLLECTOR_VERSION
            and plan["status"] == "frozen_before_collection"
            and plan["limits"]["native_seconds"] == COLLECTION_SECONDS
            and receipt["status"] == "completed" and receipt["complete"] is True
            and receipt["plan_sha256"] == inputs["collection_plan"]["sha256"]
            and receipt["sources"] == plan["sources"] and receipt["inputs"] == plan["inputs"]
            and receipt["native_inputs"] == plan["native_inputs"], "complete new sampled collection")
    require(receipt["completed_episodes"] == 90 and receipt["train_episodes"] == 54
            and receipt["valid_episodes"] == 36 and receipt["training_updates"] == 0
            and not receipt.get("pending") and receipt.get("pending_episode") is None
            and receipt.get("pending_action") is None and receipt.get("pending_emission") is None
            and not receipt.get("cleanup_errors"), "all paths closed before fitting")
    require(parent["status"] == "completed" and parent["returncode"] == 0 and not parent["timed_out"]
            and parent["group_absent"] and parent["cleanup"]["reaped"]
            and parent["cleanup"]["errors"] == [] and parent["error"] is parent["clock_error"] is None
            and parent["cap_seconds"] == COLLECTION_SECONDS and parent["clock_source_sha256"] == base.CLOCK_PIN
            and parent["watchdog_sha256"] == base.SUPERVISOR_PIN
            and parent["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
            <= parent["finished_ns"] <= parent["deadline_ns"], "original successful sampled collection supervisor")
    command = list(parent["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:3] == [str(ROOT / ".venv-otto-released-native/bin/python"),
            str(ROOT / COLLECTOR), "run"] and len(command[3:]) == 8
            and len(set(command[3::2])) == 4, "original sampled collection command")
    opts = dict(zip(command[3::2], command[4::2], strict=True))
    directory = base.regular(inputs["collection_receipt"]["path"]).parent
    require(set(opts) == {"--plan", "--plan-sha256", "--supervision", "--output"}
            and opts["--plan"] == str(base.regular(inputs["collection_plan"]["path"]))
            and opts["--plan-sha256"] == inputs["collection_plan"]["sha256"]
            and opts["--output"] == str(directory), "original collection input/output join")
    launch = base.read(opts["--supervision"])
    require(base.descriptor(opts["--supervision"])["sha256"] == receipt["supervision_sha256"]
            and all(parent[k] == v for k, v in launch.items()), "original collection launch")
    require({p.name for p in directory.iterdir()} == set(receipt["files"]) | {"receipt.json"}, "closed collection inventory")
    for name, d in receipt["files"].items():
        require(Path(name).name == name and base.descriptor(directory / name) == d, "saved collection payload")
    for name, pin in plan["sources"].items():
        require(base.descriptor(name)["sha256"] == pin, "unchanged collection source")
    for section in ("inputs", "native_inputs"):
        for d in plan[section].values():
            require(base.descriptor(d["path"]) == {k: d[k] for k in ("sha256", "bytes")}, "unchanged collection input")
    engineering = base.read(inputs["engineering"]["path"])
    require(engineering["status"] == "passed" and engineering["sources_before"] == engineering["sources_after"]
            and all(x["exit_code"] == 0 for x in engineering["results"]), "qualified sampled components")
    for name, pin in engineering["sources_after"].items():
        require(base.descriptor(name)["sha256"] == pin, "qualified component unchanged")
    return plan, receipt, directory


base.collection_inputs = collection_inputs


def selected_training(np, flat, identities, selections, sampled):
    """Project selected score slices before passing anything to the pure builder."""
    require(set(flat) == base.ARRAYS | {"label_mask"}, "exact sampled TRAIN arrays")
    require(len(identities) == len(selections) == 54, "all complete TRAIN identities and selections")
    offsets = flat["episode_offsets"]
    require(offsets.dtype == np.int64 and offsets.shape == (55,) and offsets[0] == 0
            and bool((np.diff(offsets) >= 1).all()) and bool((np.diff(offsets) <= 2188).all()), "full TRAIN offsets")
    total = int(offsets[-1])
    for name, dtype, shape in (("features", np.float32, (total, 31)), ("raw_q", np.float32, (total, 4)),
                               ("legal", np.bool_, (total, 4)), ("actions", np.int64, (total,)),
                               ("correction", np.bool_, (total,)), ("label_mask", np.bool_, (total,))):
        require(flat[name].dtype == dtype and flat[name].shape == shape, "exact TRAIN array " + name)
    missing = flat["raw_q"][~flat["label_mask"]]
    require(missing.tobytes() == np.zeros(missing.shape, np.float32).tobytes(), "unscored rows are exact positive-zero placeholders")
    episodes, scores = [], []
    for index, (identity, selection) in enumerate(zip(identities, selections, strict=True)):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        require(identity["stage"] == "train" and identity["episode_index"] == index
                and selection == {"episode_id": identity["episode_id"],
                                  **sampled.select_windows(hi - lo, SELECTION_START + index)},
                "exact predeclared selection and identity")
        require(np.array_equal(flat["correction"][lo:hi], np.arange(hi - lo) % 4 == 0), "virtual correction schedule")
        episode = {"id": identity["episode_id"], "regime": identity["regime"], "split": "train",
                   "features": flat["features"][lo:hi], "legal": flat["legal"][lo:hi], "actions": flat["actions"][lo:hi]}
        selected = {}
        for start in selection["start_offsets"]:
            stop = min(start + 4, hi - lo)
            require(bool(flat["label_mask"][lo + start:lo + stop].all()), "every selected label was returned")
            selected[start] = flat["raw_q"][lo + start:lo + stop].copy()
        episodes.append(episode); scores.append(selected)
    return episodes, sampled.build_sampled_windows(episodes, selections, scores)


class Run(base.Run):
    def episodes(self, stage):
        if stage != "train":
            return super().episodes(stage)
        np = self.np
        with np.load(self.collection_dir / "train.npz", allow_pickle=False) as z:
            flat = {key: z[key] for key in z.files}
            require(len(z.files) == len(flat), "unique TRAIN array members")
        identities = [r for r in self.collection_plan["cohort"] if r["stage"] == "train"]
        sampled = base.load(ROOT / SAMPLED, "_sampled_forecast_data")
        episodes, windows = selected_training(np, flat, identities,
                                              base.read(self.collection_dir / "train-selection.json"), sampled)
        self.npz("training-windows.npz", {k: v for k, v in windows.items() if isinstance(v, np.ndarray)})
        base.write(self.out / "training-windows.json", {k: v for k, v in windows.items() if not isinstance(v, np.ndarray)})
        original = self.data

        def build_windows(rows):
            if rows is episodes:
                return windows
            require(all(row.get("split") == "valid" for row in rows), "only census VALID delegates to original helper")
            return original.build_windows(rows)

        self.data = SimpleNamespace(build_windows=build_windows, forecast_metrics=original.forecast_metrics)
        return episodes, identities


# Only this private loaded module is configured. Its inherited body resolves the
# new Run class and configuration through its own globals, never another import.
base.Run = Run
criteria = base.criteria
centered_loss = base.centered_loss
batch_loss = base.batch_loss


if __name__ == "__main__":
    base.main()
