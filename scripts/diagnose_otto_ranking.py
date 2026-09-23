"""Authenticated saved-output ranking diagnosis. No fitting or model imports.

The independent checker may reuse byte and process authentication from this
module. Numerical reducers are imported only by the diagnostic run itself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.suspend_clock import SuspendClock

VERSION = "otto-ranking-diagnosis-v1"
LIMITS = {"seconds": 240, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
STUDY = ROOT / "output/otto-separate-prior-v1"
OUTPUT = ROOT / "output" / VERSION
SEEDS = (285000001, 285000002, 285000003)
FAMILIES = ("innovation_shared_mse", "innovation_shared_aux", "innovation_separate_mse",
            "innovation_separate_aux", "gru_shared_mse", "gru_shared_aux",
            "gru_separate_mse", "gru_separate_aux")
SOURCES = ("scripts/diagnose_otto_ranking.py", "tests/test_diagnose_otto_ranking.py",
           "src/openjev/research/otto_ranking_diagnosis.py", "tests/test_otto_ranking_diagnosis.py",
           "scripts/audit_otto_ranking_diagnosis.py", "tests/test_audit_otto_ranking_diagnosis.py",
           "scripts/supervise_dialogue_observation_v2.py", "src/openjev/research/suspend_clock.py")
PROTOCOL = "research/otto-ranking-diagnosis-protocol.md"
FIXED_INPUTS = {
    "training_plan": "training-plan-01.json", "training_receipt": "training-01/receipt.json",
    "training_terminal": "training-supervision-01.terminal.json",
    "audit_receipt": "audit-01/receipt.json", "audit_terminal": "audit-supervision-01.terminal.json",
    "audit_result": "audit-01/audit.json", "collection_plan": "collection-plan-01.json",
    "windows": "training-01/validation-windows.npz",
    "window_metadata": "training-01/validation-windows.json", "hold": "training-01/prediction-hold.npz",
    **{f"prediction-{family}-{seed}": f"training-01/prediction-{family}-{seed}.npz"
       for seed in SEEDS for family in FAMILIES},
}


def require(value, message):
    if not value:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts
            and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "contained regular file: " + str(path))
    return path


def pin(path):
    path = regular(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return {"sha256": value.hexdigest(), "bytes": path.stat().st_size}


def descriptor(path):
    return {"path": str(regular(path)), **pin(path)}


def read(path, limit_bytes=80 * 1024**2):
    path = regular(path)
    require(type(limit_bytes) is int and 0 < limit_bytes <= LIMITS["output_bytes"]
            and path.stat().st_size <= limit_bytes, "bounded saved JSON")
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def runtime():
    return {"python": sys.version, "executable": sys.executable,
            "numpy": importlib.metadata.version("numpy"), "platform": sys.platform}


def original_join(worker, terminal, *, script, output, plan, plan_pin, cap):
    """Require the original successful worker/parent, not a replacement observer."""
    require(worker["status"] == terminal["status"] == "completed"
            and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["group_absent"] is terminal["cleanup"]["reaped"] is True
            and terminal["cleanup"]["errors"] == []
            and terminal["error"] is terminal["clock_error"] is None, "original phase completed")
    require(terminal["cap_seconds"] == cap and terminal["cwd"] == str(ROOT)
            and terminal["deadline_ns"] == terminal["started_ns"] + cap * 10**9
            and terminal["started_ns"] <= worker["started_ns"] < worker["finished_ns"]
            <= terminal["finished_ns"] <= terminal["deadline_ns"], "original native-clock interval")
    prefix = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts" / script)]
    if script.startswith("train_") or script == "diagnose_otto_ranking.py":
        prefix.append("run")
    command = terminal["command"]
    require(command[:len(prefix)] == prefix, "original command prefix")
    tail = command[len(prefix):]
    require(len(tail) % 2 == 0 and len(set(tail[::2])) == len(tail) // 2,
            "unique original arguments")
    options = dict(zip(tail[::2], tail[1::2], strict=True))
    require(options["--plan"] == str(plan) and options["--plan-sha256"] == plan_pin
            == worker["plan_sha256"] and options["--output"] == str(output), "original plan/output")
    launch_path = regular(options["--supervision"])
    require(pin(launch_path)["sha256"] == worker["supervision_sha256"], "original launch pin")
    launch = read(launch_path)
    require(all(terminal[key] == value for key, value in launch.items()), "original launch retained")
    require(terminal["pid"] == terminal["pgid"] and terminal["pid"] != terminal["parent_pid"]
            and terminal["watchdog_sha256"] == pin(ROOT / SOURCES[-2])["sha256"]
            and terminal["clock_source_sha256"] == pin(ROOT / SOURCES[-1])["sha256"],
            "original process allocation and immutable supervisor")
    return options


def authenticate(plan, check=lambda: None):
    """Authenticate descriptors and closed provenance; never decode an array."""
    require(plan["version"] == VERSION and plan["status"] == "frozen_before_saved_array_read"
            and plan["limits"] == LIMITS and plan["runtime"] == runtime(), "frozen diagnostic contract")
    require(set(plan["sources"]) == set(SOURCES), "complete diagnostic/checker sources")
    for name, expected in plan["sources"].items():
        check(); require(pin(ROOT / name) == expected, "unchanged diagnostic source: " + name)
    require(set(plan["inputs"]) == set(FIXED_INPUTS) | {"qualification", "protocol"},
            "complete fixed saved inputs")
    paths = {}
    for role, item in plan["inputs"].items():
        check(); path = regular(item["path"])
        require(pin(path) == {k: item[k] for k in ("sha256", "bytes")}, "input pin: " + role)
        if role in FIXED_INPUTS:
            require(path == STUDY / FIXED_INPUTS[role], "fixed original input identity")
        paths[role] = path
    require(paths["protocol"] == ROOT / PROTOCOL, "specified retrospective protocol")
    qualification = read(paths["qualification"])
    require(qualification["status"] == "completed" and qualification["all_passed"] is True
            and qualification["sources"] == plan["sources"]
            and qualification["empirical_array_reads"] == 0, "fabricated qualification binds held sources")
    training, audit = read(paths["training_receipt"]), read(paths["audit_receipt"])
    training_plan = read(paths["training_plan"])
    require(training["complete"] is True and training["fits_completed"] == 24
            and training["optimizer_steps"] == 17280 and audit["agreement"] is True
            and audit["failures"] == [], "complete independently audited original study")
    require(training["inputs"]["collection_plan"] == plan["inputs"]["collection_plan"]
            and training["sources"] == training_plan["sources"]
            and len(training_plan["sources"]) == 140, "original 140-source study")
    for name, expected in training_plan["sources"].items():
        check(); actual = pin(ROOT / name)
        require(actual == expected if isinstance(expected, dict) else actual["sha256"] == expected,
                "unchanged original scientific source: " + name)
    for role, original in (("plan", "training_plan"), ("worker", "training_receipt"),
                           ("terminal", "training_terminal")):
        require(audit["producer_inputs"][role] == plan["inputs"][original], "original audit joins")
    for role, worker, script, cap in (("training", training, "train_otto_separate_prior.py", 21600),
                                     ("audit", audit, "audit_otto_separate_prior.py", 240)):
        options = original_join(worker, read(paths[role + "_terminal"]), script=script,
            output=paths[role + "_receipt"].parent, plan=paths["training_plan"],
            plan_pin=plan["inputs"]["training_plan"]["sha256"], cap=cap)
        if role == "audit":
            for option, key in (("worker", "training_receipt"), ("terminal", "training_terminal")):
                require(options["--" + option] == str(paths[key])
                        and options["--" + option + "-sha256"] == plan["inputs"][key]["sha256"],
                        "auditor selected original producer")
    for role in ("windows", "window_metadata", "hold", *(key for key in paths if key.startswith("prediction-"))):
        require(pin(paths[role]) == training["files"][paths[role].name], "producer-bound payload: " + role)
    require(pin(paths["audit_result"]) == audit["files"]["audit.json"], "closed audit result")
    reference = read(paths["audit_result"])["summary"]
    require({key: (g["passed"], g["total"], g["passes"]) for key, g in reference["gates"].items()}
            == {"innovation_mechanism": (11, 19, False), "gru_mechanism": (10, 19, False),
                "architecture": (14, 29, False)}, "original failed decisions stay closed")
    return paths, reference


class Run:
    """Byte-only lifecycle shared with the independently implemented checker."""

    def __init__(self, args, kind="diagnosis"):
        self.args, self.kind, self.out = args, kind, args.output
        self.clock = SuspendClock(); self.start = self.clock.now_ns(); self.created = False
        self.receipt = {"version": VERSION, "phase": kind, "status": "started", "limits": LIMITS,
            "started_ns": self.start, "model_calls": 0, "native_calls": 0, "optimizer_calls": 0,
            "npz_decodes": 0, "requires_successful_original_supervisor": True}
        try:
            require(kind in ("diagnosis", "audit") and self.out.is_absolute()
                    and self.out.is_relative_to(OUTPUT) and self.out != OUTPUT
                    and ".." not in self.out.parts
                    and not any(p.is_symlink() for p in (self.out, *self.out.parents)), "exclusive output root")
            self.out.mkdir(exist_ok=False); self.created = True
            require(pin(args.plan)["sha256"] == args.plan_sha256, "external frozen plan pin")
            self.plan = read(args.plan)
            require(self.plan["outputs"][kind] == str(self.out), "predeclared phase output")
            self.receipt.update(plan_sha256=args.plan_sha256, supervision_sha256=pin(args.supervision)["sha256"],
                                sources=self.plan["sources"], inputs=self.plan["inputs"])
            self.launch = read(args.supervision)
            require(self.launch["cap_seconds"] == LIMITS["seconds"] and self.launch["pid"] == os.getpid()
                    and self.launch["pgid"] == os.getpgid(0) and self.launch["parent_pid"] == os.getppid()
                    and self.launch["cwd"] == str(ROOT)
                    and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["seconds"] * 10**9
                    and self.launch["clock_backend"] == self.clock.backend, "current original supervisor")
            script = "diagnose_otto_ranking.py" if kind == "diagnosis" else "audit_otto_ranking_diagnosis.py"
            require(Path(sys.argv[0]).resolve() == ROOT / "scripts" / script
                    and self.launch["command"] == [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts" / script), *sys.argv[1:]]
                    and self.launch["watchdog_sha256"] == self.plan["sources"][SOURCES[-2]]["sha256"]
                    and self.launch["clock_source_sha256"] == self.plan["sources"][SOURCES[-1]]["sha256"],
                    "exact command and current supervisor sources")
            require(all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")), "one numerical thread")
            self.old_handler = signal.signal(signal.SIGTERM, self.interrupt)
            self.publish("started.json", {"phase": kind, "launch": self.launch, "plan_sha256": args.plan_sha256})
            self.paths, self.reference = authenticate(self.plan, self.check)
        except BaseException as error:
            self.fail(error)
            raise

    @staticmethod
    def interrupt(_signum, _frame):
        raise InterruptedError("original diagnostic supervisor interrupted worker")

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original phase deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "phase RSS bound")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) < LIMITS["output_bytes"] - 1024**2,
                "phase output bound with failure reserve")

    def publish(self, name, value):
        require(Path(name).name == name and name.endswith(".json"), "flat JSON result")
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.check()
        used = sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        require(used + len(encoded) < LIMITS["output_bytes"] - 1024**2, "encoded output admitted before write")
        with (self.out / name).open("xb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())

    def finish(self, payloads, extra=None):
        for name, value in payloads.items():
            self.publish(name, value)
        for item in self.plan["inputs"].values():
            self.check(); require(pin(Path(item["path"])) == {k: item[k] for k in ("sha256", "bytes")}, "inputs unchanged")
        for name, expected in self.plan["sources"].items():
            require(pin(ROOT / name) == expected, "diagnostic/checker sources unchanged")
        require(pin(self.args.plan)["sha256"] == self.args.plan_sha256
                and pin(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"],
                "external plan and original launch unchanged")
        files = {p.name: pin(p) for p in sorted(self.out.iterdir()) if p.is_file()}
        self.check(); finish = self.clock.now_ns()
        self.receipt.update(extra or {})
        self.receipt.update(status="completed", finished_ns=finish, wall_seconds=(finish-self.start)/1e9,
                            files=files)
        self.publish("receipt.json", self.receipt); self.check()
        print(json.dumps({"status": "completed", "phase": self.kind, "receipt": pin(self.out / "receipt.json")}), flush=True)
        signal.signal(signal.SIGTERM, self.old_handler)

    def fail(self, error):
        if not self.created:
            return
        self.receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        try:
            try:
                finish = self.clock.now_ns()
                self.receipt.update(finished_ns=finish, wall_seconds=(finish-self.start)/1e9)
            except BaseException as clock_error:  # noqa: BLE001 - failure publication must survive clock failure
                self.receipt.update(finished_ns=None, wall_seconds=None, clock_error=repr(clock_error))
            if (self.out / "receipt.json").exists():
                (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
            self.receipt["files"] = {p.name: pin(p) for p in self.out.iterdir() if p.is_file()}
            write(self.out / "receipt.json", self.receipt)
        except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
            error.add_note("Failure receipt publication: " + repr(secondary))


def plan(args):
    require(args.output.is_absolute() and args.output.parent == OUTPUT, "plan in separate diagnostic tree")
    OUTPUT.mkdir(exist_ok=True)
    sources = {name: pin(ROOT / name) for name in SOURCES}
    inputs = {role: descriptor(STUDY / name) for role, name in FIXED_INPUTS.items()}
    inputs.update(protocol=descriptor(ROOT / PROTOCOL), qualification=descriptor(args.qualification))
    value = {"version": VERSION, "status": "frozen_before_saved_array_read", "sources": sources,
        "inputs": inputs, "limits": LIMITS, "runtime": runtime(),
        "outputs": {kind: str(OUTPUT / (kind + "-01")) for kind in ("diagnosis", "audit")},
        "scope": "Retrospective diagnostic of an exposed failed study; no new predictions, promotion or efficacy gate."}
    authenticate(value)
    write(args.output, value)
    print(json.dumps({"status": "frozen_before_saved_array_read", "plan": descriptor(args.output)}), flush=True)


def hydrate_windows(arrays, metadata):
    """Restore the exact metadata types expected by the held array contract."""
    return {**arrays, "version": metadata["version"],
            **{key: tuple(metadata[key]) for key in ("episode_ids", "episode_regimes")}}


def run(args):
    job = Run(args)
    try:
        import numpy as np

        from openjev.research import otto_ranking_diagnosis as metrics
        paths = job.paths
        meta = read(paths["window_metadata"])
        with np.load(paths["windows"], allow_pickle=False) as archive:
            windows = {key: archive[key] for key in archive.files}
        job.receipt["npz_decodes"] += 1
        windows = hydrate_windows(windows, meta)
        identities = [{key: row[key] for key in ("episode_id", "regime", "case", "arm")}
                      for row in read(paths["collection_plan"])["cohort"] if row["stage"] == "valid"]
        require(len(identities) == 36 and [row["episode_id"] for row in identities] == meta["episode_ids"],
                "all original VALID identities in order")
        predictions = {}
        for family, seed in [("hold", None), *((family, seed) for seed in SEEDS for family in FAMILIES)]:
            job.check(); role = "hold" if family == "hold" else f"prediction-{family}-{seed}"
            with np.load(paths[role], allow_pickle=False) as archive:
                require(set(archive.files) == ({"predictions"} if family == "hold" else {"predictions", "prior", "prior_mask"}),
                        "original prediction container")
                predictions[family, seed] = archive["predictions"]
            job.receipt["npz_decodes"] += 1
        result = metrics.analyze(windows, predictions, identities)
        job.check()
        job.finish({"diagnosis.json": result}, {"counts": result["counts"], "new_scientific_gate": False})
    except BaseException as error:
        job.fail(error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    freeze = subs.add_parser("plan")
    freeze.add_argument("--qualification", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    execute = subs.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--supervision", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan(args) if args.mode == "plan" else run(args)


if __name__ == "__main__":
    main()
