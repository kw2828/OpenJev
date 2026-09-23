"""Saved arithmetic diagnosis of an interrupted study; never qualifies its training.

This reads authenticated saved arrays without importing a model or optimizer.
The original training parent is unavailable, so technical_complete remains false
and none of the original continuation gates can become eligible.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/diagnose_otto_action_focused_interruption.py"
TEST = "tests/test_diagnose_otto_action_focused_interruption.py"
PROTOCOL = "research/otto-action-focused-interruption-protocol.md"
ORIGINAL = "scripts/audit_otto_action_focused.py"
ORIGINAL_PIN = "4ee295fc8d77b81b6df869811b471770f47af2d6f5a763e040a21ad7d0578729"
VERSION = "otto-action-focused-interruption-diagnostic-v1"
COMPONENTS = {SELF, TEST}
LAUNCHER_COMPONENTS = {"scripts/launch_otto_detached_phase.py", "tests/test_launch_otto_detached_phase.py"}
ROLES = ("training_plan", "worker", "launch", "interruption", "engineering", "launcher_engineering")
LIMITS = {"seconds": 240, "rss_bytes": 2 * 1024**3, "output_bytes": 256 * 1024**2}
LIMITATIONS = [
    "Original training supervisor terminal is missing; exit, reaping and paid parent bounds are unverified.",
    "Arithmetic agreement is descriptive and cannot repair the original failed technical gate.",
    "Training timings and resource usage are worker-reported only, not closed-parent measurements.",
    "No model, optimizer, teacher or native simulator is executed by this diagnostic.",
    "Saved forecasts and gradient claims are authenticated, not independently rerun.",
    "Collection and historical capacity require their actual closed original supervisors.",
    "Forced-path imitation does not establish autonomous utility or architectural novelty.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular evidence")
    return path


def descriptor(value):
    path = regular(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(value):
    return json.loads(regular(value).read_text())


def load_original():
    require(descriptor(ORIGINAL)["sha256"] == ORIGINAL_PIN, "immutable original saved auditor")
    spec = importlib.util.spec_from_file_location("_interrupted_action_focused_original", ROOT / ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


original = load_original()
write = original.write
CLOCK, CLOCK_PIN = original.CLOCK, original.CLOCK_PIN
SUPERVISOR, SUPERVISOR_PIN = original.SUPERVISOR, original.SUPERVISOR_PIN
PRODUCER_VERSION, CAPACITY_SCOPE = original.PRODUCER_VERSION, original.CAPACITY_SCOPE


def runtime_record():
    return {"python": sys.version, "executable": sys.executable,
            "distributions": dict(sorted((x.metadata["Name"], x.version)
                                          for x in importlib.metadata.distributions()))}


def original_terminal_path(launch_path):
    path = Path(launch_path)
    require(path.name.endswith(".launch.json"), "original launch suffix")
    return path.with_name(path.name[:-len(".launch.json")] + ".terminal.json")


def check_observation(observation, inputs, worker):
    require(observation["status"] == "original_parent_closure_unavailable"
            and observation["original_terminal_exists"] is False
            and observation["process_group_present"] is False
            and observation["processes_present"] == {"supervisor": False, "worker": False}
            and observation["scientific_retry_launched"] is False,
            "saved interruption observation cannot establish process closure")
    require(observation["worker_receipt_sha256"] == inputs["worker"]["sha256"]
            and observation["worker_complete"] is worker["complete"] is True
            and observation["worker_status"] == worker["status"] == "completed"
            and observation["fits_completed"] == worker["fits_completed"] == 12
            and observation["optimizer_steps"] == worker["optimizer_steps"] == 8640
            and observation["worker_reported_seconds"] == worker["wall_seconds"],
            "interruption observation joins exact worker")
    launch_name = str(Path(inputs["launch"]["path"]).relative_to(ROOT))
    require(observation["files"].get(launch_name) ==
            {k: inputs["launch"][k] for k in ("sha256", "bytes")}, "observed original launch pin")
    for name, pin in observation["files"].items():
        require({k: descriptor(name)[k] for k in ("sha256", "bytes")} == pin,
                "unchanged observation evidence")


def engineering(inputs, role="engineering", components=None):
    components = COMPONENTS if components is None else components
    pin = inputs[role]
    record = read(pin["path"])
    require(record["status"] == "passed" and record["sources_before"] == record["sources_after"]
            and set(record["sources_after"]) == components and record["results"]
            and all(type(row["exit_code"]) is int and row["exit_code"] == 0 for row in record["results"]),
            "qualified fabricated diagnostic engineering")
    for name, digest in record["sources_after"].items():
        require(descriptor(name)["sha256"] == digest, "unchanged qualified diagnostic source")
    directory = Path(pin["path"]).parent
    require({p.name for p in directory.iterdir()} == set(record["files"]) | {"receipt.json"},
            "exact engineering inventory")
    for name, item in record["files"].items():
        require(Path(name).name == name and
                all(descriptor(directory / name)[key] == item[key] for key in ("sha256", "bytes")),
                "engineering payload pin")
    return record


def freeze(args):
    inputs = {}
    for role in ROLES:
        item = descriptor(getattr(args, role))
        require(item["sha256"] == getattr(args, role + "_sha256"), "external input " + role)
        inputs[role] = item
    plan, worker = read(inputs["training_plan"]["path"]), read(inputs["worker"]["path"])
    require(plan["version"] == PRODUCER_VERSION and plan["status"] == "frozen_before_fitting"
            and plan["config"] == original.CONFIG, "original fixed training plan")
    require(plan["runtime"] == runtime_record(), "unchanged original training runtime")
    require(worker["sources"] == plan["sources"] and worker["inputs"] == plan["inputs"],
            "worker retains original source and input maps")
    require(worker["plan_sha256"] == inputs["training_plan"]["sha256"]
            and worker["supervision_sha256"] == inputs["launch"]["sha256"], "exact worker plan and launch joins")
    terminal = original_terminal_path(inputs["launch"]["path"])
    require(not terminal.exists(), "original parent terminal must remain unavailable")
    check_observation(read(inputs["interruption"]["path"]), inputs, worker)
    engineering(inputs)
    engineering(inputs, "launcher_engineering", LAUNCHER_COMPONENTS)
    sources = dict(plan["sources"])
    require(sources.get(ORIGINAL) == ORIGINAL_PIN, "original independent arithmetic frozen before fitting")
    for name, pin in sources.items():
        require(descriptor(name)["sha256"] == pin, "unchanged original source " + name)
    for name in (*sorted(COMPONENTS | LAUNCHER_COMPONENTS), PROTOCOL):
        require(name not in sources, "new diagnostic cannot overwrite original source")
        sources[name] = descriptor(name)["sha256"]
    write(args.output, {"version": VERSION, "status": "frozen_before_saved_array_decode",
        "inputs": inputs, "sources": sources, "runtime": runtime_record(), "limits": LIMITS,
        "original_terminal_path": str(terminal), "original_process_closure": False,
        "technical_complete": False, "original_audit": "unavailable_not_replaced",
        "scientific_calls": {"model": 0, "optimizer": 0, "teacher": 0, "native": 0},
        "limits_scope": "new diagnostic only; no claim about original training parent bounds",
        "limitations": LIMITATIONS})


def ineligible_gates(rows):
    require(rows[0]["name"] == "common.technical_complete" and rows[0]["value"] is False
            and rows[0]["passes"] is False, "technical condition always false")
    values = original.gate_decisions(rows)
    require(all(gate["passes"] is False for gate in values.values()), "no original gate passes")
    return {name: {**gate, "eligible": False,
            "reason": "Original training parent closure unavailable; original audit not completed"}
            for name, gate in values.items()}


class Diagnostic(original.Audit):
    def __init__(self, args):
        super().__init__(args)
        self.diagnostic_args = args
        self.receipt.update(version=VERSION, arithmetic_agreement=False, original_process_closure=False,
            technical_complete=False, original_audit="unavailable_not_replaced", limits=LIMITS,
            native_calls=0, model_calls=0, optimizer_calls=0, teacher_calls=0, limitations=LIMITATIONS)
        self.receipt.pop("agreement", None)

    def admit(self):
        self.require(self.sha(self.path(CLOCK)) == CLOCK_PIN, "qualified clock before import")
        spec = importlib.util.spec_from_file_location("_interruption_diagnostic_clock", ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        args = self.diagnostic_args
        while not args.supervision.exists():
            self.require(self.clock.now_ns() - self.start < 5 * 10**9, "new diagnostic launch available")
            time.sleep(.01)
        self.launch = self.read(args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        self.require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
            and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
            and self.launch["cwd"] == str(ROOT) == str(Path.cwd())
            and self.launch["cap_seconds"] == LIMITS["seconds"]
            and self.launch["clock_backend"] == self.clock.backend
            and self.launch["clock_source_sha256"] == CLOCK_PIN
            and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
            and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
            and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["seconds"] * 10**9,
            "actual bounded new diagnostic process")
        self.require(self.sha(self.path(args.plan)) == args.plan_sha256, "external frozen diagnostic plan")
        self.diagnostic_plan = self.read(args.plan)
        dp = self.diagnostic_plan
        self.require(dp["version"] == VERSION and dp["status"] == "frozen_before_saved_array_decode"
            and dp["limits"] == LIMITS and dp["original_process_closure"] is False
            and dp["technical_complete"] is False and dp["original_audit"] == "unavailable_not_replaced"
            and dp["runtime"] == runtime_record(), "frozen diagnostic scope and runtime")
        self.inputs = dp["inputs"]
        self.require(set(self.inputs) == set(ROLES), "exact diagnostic inputs")
        for pin in self.inputs.values():
            self.pinned(self.path(pin["path"]), pin)
        self.plan = self.read(self.inputs["training_plan"]["path"])
        self.require(self.plan["version"] == PRODUCER_VERSION and self.plan["status"] == "frozen_before_fitting"
            and self.plan["config"] == original.CONFIG and self.plan["runtime"] == runtime_record()
            and self.plan["limits"] == {"seconds": 14400, "rss_bytes": 4 * 1024**3,
                                       "output_bytes": 2 * 1024**3}, "fixed original training allocation")
        self.require(set(dp["sources"]) == set(self.plan["sources"]) | COMPONENTS | LAUNCHER_COMPONENTS | {PROTOCOL}
            and {k: dp["sources"][k] for k in self.plan["sources"]} == self.plan["sources"]
            and dp["sources"][ORIGINAL] == ORIGINAL_PIN, "additive original and diagnostic source closure")
        for name, pin in dp["sources"].items():
            self.require(self.sha(self.path(name)) == pin, "frozen source before array decode " + name)
        engineering(self.inputs)
        engineering(self.inputs, "launcher_engineering", LAUNCHER_COMPONENTS)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.require(os.environ.get(name) == "1", "single numerical thread")
        self.args = SimpleNamespace(plan=Path(self.inputs["training_plan"]["path"]),
            worker=Path(self.inputs["worker"]["path"]), terminal=None)
        self.receipt.update(plan_sha256=args.plan_sha256, inputs=self.inputs, sources=dp["sources"],
                            supervision_sha256=self.sha(args.supervision))
        write(self.out / "started.json", {"started_ns": self.start, "launch": self.launch,
            "inputs": self.inputs, "original_process_closure": False, "technical_complete": False})

    def phase(self, plan_path, worker_path, terminal_path, script, interpreter, cap):
        if script != original.PRODUCER:
            return super().phase(plan_path, worker_path, terminal_path, script, interpreter, cap)
        self.require(terminal_path is None and plan_path == self.args.plan and worker_path == self.args.worker
                     and interpreter == ".venv/bin/python" and cap == 14400,
                     "only exact interrupted training branch lacks parent closure")
        plan, worker = self.read(plan_path), self.read(worker_path)
        launch_path = self.path(self.inputs["launch"]["path"])
        launch = self.read(launch_path)
        terminal = original_terminal_path(launch_path)
        self.require(str(terminal) == self.diagnostic_plan["original_terminal_path"] and not terminal.exists(),
                     "original training terminal remains unavailable")
        check_observation(self.read(self.inputs["interruption"]["path"]), self.inputs, worker)
        command = list(launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        self.require(command[:3] == [str(ROOT / interpreter), str(ROOT / script), "run"]
            and len(command) == 11 and len(set(command[3::2])) == 4, "original worker command identity")
        options = dict(zip(command[3::2], command[4::2], strict=True))
        self.require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"}
            and options["--plan"] == str(plan_path)
            and options["--plan-sha256"] == self.sha(plan_path) == worker["plan_sha256"]
            and options["--supervision"] == str(launch_path)
            and options["--output"] == str(worker_path.parent)
            and self.sha(launch_path) == worker["supervision_sha256"], "original launch and worker joins")
        self.require(launch["cwd"] == str(ROOT) and launch["cap_seconds"] == cap
            and launch["clock_source_sha256"] == CLOCK_PIN and launch["watchdog_sha256"] == SUPERVISOR_PIN
            and launch["deadline_ns"] == launch["started_ns"] + cap * 10**9,
            "recorded original allocation intent, not verified parent bound")
        self.require(worker["version"] == PRODUCER_VERSION and worker["status"] == "completed"
            and worker["complete"] is True and worker["requires_successful_original_supervisor"] is True
            and worker["limits"] == plan["limits"] and worker["fits_completed"] == 12
            and worker["optimizer_steps"] == 8640 and worker["native_calls"] == worker["teacher_calls"] == 0,
            "complete worker-reported fitting scope")
        self.require(type(worker["started_ns"]) is int and type(worker["finished_ns"]) is int
            and launch["started_ns"] <= worker["started_ns"] < worker["finished_ns"]
            and math.isfinite(worker["wall_seconds"])
            and worker["wall_seconds"] == (worker["finished_ns"] - worker["started_ns"]) / 1e9,
            "worker-reported timestamp arithmetic only")
        self.equal(self.read(worker_path.parent / "started.json")["launch"], launch, "saved original worker launch")
        self.equal(worker["sources"], plan["sources"], "original training source map")
        self.equal(worker["inputs"], plan["inputs"], "original training input map")
        self.require(not worker.get("cleanup_errors") and not worker.get("pending")
            and worker.get("pending_emission") is None and worker.get("pending_episode") is None
            and worker.get("pending_action") is None, "no worker-reported unresolved operation")
        directory = worker_path.parent
        self.require({p.name for p in directory.iterdir()} == set(worker["files"]) | {"receipt.json"}
            and len(worker["files"]) == 49, "exact original worker payload inventory")
        for name, pin in worker["files"].items():
            self.require(Path(name).name == name, "flat original worker payload name")
            self.pinned(self.path(directory / name), pin)
        self.require(type(worker["peak_rss_bytes"]) is int and worker["peak_rss_bytes"] > 0,
                     "worker-reported resource observation")
        for name, pin in plan["sources"].items():
            self.require(self.sha(self.path(name)) == pin, "unchanged original training source")
        for pin in plan["inputs"].values():
            self.pinned(self.path(pin["path"]), pin)
        runtime = self.read(directory / "runtime.json")
        self.equal({k: runtime[k] for k in plan["runtime"]}, plan["runtime"], "saved original runtime matches frozen runtime")
        self.require(runtime["numpy"] == plan["runtime"]["distributions"]["numpy"]
                     and runtime["torch"] == plan["runtime"]["distributions"]["torch"], "saved framework versions")
        self.counts["unverified_training_phases"] += 1
        return plan, worker, None, directory

    def body(self):
        self.authenticate()
        import numpy as np
        self.np = np
        runtime = self.read(self.run / 'runtime.json')
        self.require(runtime['threads'] == runtime['interop_threads'] == 1 and runtime['deterministic'] is True
                     and runtime['cuda_used'] is runtime['mps_used'] is False, 'qualified deterministic CPU runtime')
        train, train_meta, _ = self.reconstruct('train')
        self.saved_windows('training-history',train,train_meta)
        self.train_reports = []
        fits = self.training(train,train_meta)
        del train
        windows, meta, identities = self.reconstruct('valid')
        self.saved_windows('validation-windows',windows,meta)
        history, history_meta = self.validation_history(windows,meta)
        history.update({key: meta[key] for key in ('episode_ids','episode_regimes')})
        models, hold = self.predictions(windows,meta,identities,fits,history)
        lengths = windows['episode_lengths']
        chunks = sum((max(int(v) for v in lengths[first:first+6])+31)//32 for first in range(0,36,6))
        prediction_work = {'forward_chunks':chunks,'forward_rows':meta['counts']['rows'],
                           'prior_rows':history_meta['counts']['prior_rows']}
        for row in models:
            row['prediction_work'] = dict(prediction_work)
        self.require(self.worker['validation_forward_chunks'] == 12*chunks
                     and self.worker['validation_forward_rows'] == 12*meta['counts']['rows'], 'complete census inference accounting')
        rules = original.continuation_rules(models,hold,technical_complete=False)
        expected = {'version':PRODUCER_VERSION,'models':models,'hold':hold,'required':rules,
            'required_passed':sum(r['passes'] for r in rules),'required_conditions':41,
            'gates':original.gate_decisions(rules),
            'technical_complete_pending_saved_audit':True,'scientific_conditions':40,
            'scientific_passed':sum(r['passes'] for r in rules[1:]),
            'requires_successful_original_supervisor_and_saved_audit':True,
            'train_counts':train_meta['counts'],'validation_counts':meta['counts'],
            'validation_history_counts':history_meta['counts'],
            'capacity_evidence_scope':CAPACITY_SCOPE,
            'scope':'Forced-path forecasts only, with separate objective and architecture gates. No omnibus all-41 gate, autonomous performance, new-shift evidence or true action regret.'}
        summary = self.read(self.run / 'summary.json')
        for key in ('setup_seconds','fitting_seconds','validation_seconds'):
            self.require(type(summary[key]) in (int,float) and math.isfinite(summary[key]) and summary[key] >= 0,
                         'finite original physical stage')
            expected[key] = summary[key]
        self.require(math.fsum(f['wall_seconds'] for f in fits) <= summary['fitting_seconds']+1e-9
                     and math.fsum(summary[k] for k in ('setup_seconds','fitting_seconds','validation_seconds'))
                     <= self.worker['wall_seconds']+1e-9, 'worker-reported stage cost containment only')
        self.equal(summary,expected,'independent provisional producer summary and all 40 scientific records')
        payloads = {'started.json','runtime.json','progress.jsonl','work.jsonl','fits.json','summary.json',
            'training-history.npz','training-history.json','validation-history.npz','validation-history.json',
            'validation-windows.npz','validation-windows.json','prediction-hold.npz'}
        for fit in fits:
            payloads.update((fit['checkpoint_path'],'prediction-'+fit['checkpoint_path'],
                             'training-prediction-'+fit['checkpoint_path']))
        self.require(len(payloads) == 49 and set(self.worker['files']) == payloads, 'exact final 49-payload closure')
        for directory, worker in ((self.run,self.worker),(self.collection,self.collection_worker)):
            for name,pin in worker['files'].items():
                self.pinned(directory/name,pin)
        for descriptor in self.plan['inputs'].values():
            self.pinned(self.path(descriptor['path']),descriptor)
        self.counts.update(required_conditions=41, required_passed=expected['required_passed'],
            collection_episodes=90, prediction_files=25, source_files=len(self.plan['sources']),
            training_payloads=49, collection_payloads=18)
        self.result = {'version': VERSION, 'arithmetic_agreement': True,
            'original_process_closure': False, 'original_training_process_closure': False,
            'technical_complete': False, 'eligible_for_continuation': False, 'original_audit': 'unavailable_not_replaced', 'original_continuation_eligible': False,
            'producer_summary': expected,
            'summary': {**expected, 'gates': ineligible_gates(rules), 'technical_complete': False,
                        'eligible_for_continuation': False, 'original_training_process_closure': False},
            'gates': ineligible_gates(rules),
            'fits': fits, 'training_forecast_checks': self.train_reports, 'counts': dict(self.counts),
            'timing_scope': 'worker-reported only; original parent exit, reaping and bounds unverified',
            'worker_reported_wall_seconds': self.worker['wall_seconds'],
            'worker_reported_peak_rss_bytes': self.worker['peak_rss_bytes'],
            'scientific_calls': {'model': 0, 'optimizer': 0, 'teacher': 0, 'native': 0},
            'limitations': LIMITATIONS}

    def execute(self):
        self.require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and '..' not in self.out.parts
            and not any(p.is_symlink() for p in self.out.parents), 'exclusive contained diagnostic output')
        self.out.mkdir(parents=False, exist_ok=False)
        old_handler = signal.getsignal(signal.SIGTERM)

        def interrupted(_signum, _frame):
            raise InterruptedError('new saved-diagnostic supervisor stopped worker')

        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.admit()
            self.body()
            for name, pin in self.diagnostic_plan['sources'].items():
                self.require(self.sha(self.path(name)) == pin, 'sources unchanged after diagnostic')
            for pin in self.inputs.values():
                self.pinned(self.path(pin['path']), pin)
            args = self.diagnostic_args
            self.require(self.sha(args.plan) == args.plan_sha256
                and self.sha(args.supervision) == self.receipt['supervision_sha256'], 'unchanged diagnostic plan and launch')
            self.require(not Path(self.diagnostic_plan['original_terminal_path']).exists(),
                         'original parent terminal still unavailable')
            self.require(self.result['technical_complete'] is False and self.result['original_process_closure'] is False
                and all(g['eligible'] is False and g['passes'] is False for g in self.result['gates'].values()),
                'diagnostic cannot promote original study')
            write(self.out / 'diagnostic.json', self.result)
            self.check()
            files = {name: {k: v for k, v in self.descriptor(self.out / name).items() if k != 'path'}
                     for name in ('started.json', 'diagnostic.json')}
            self.require({p.name for p in self.out.iterdir()} == set(files), 'exact diagnostic output inventory')
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', arithmetic_agreement=True, files=files, counts=dict(self.counts),
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished-self.start)/1e9,
                requires_successful_original_diagnostic_supervisor=True)
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            print(json.dumps({'status': 'completed', 'arithmetic_agreement': True,
                'original_process_closure': False, 'technical_complete': False,
                'receipt': self.descriptor(self.out / 'receipt.json')}), flush=True)
        except BaseException as error:
            self.failure = True
            self.receipt.update(status='failed', arithmetic_agreement=False, error=repr(error),
                traceback=traceback.format_exc(), counts=dict(self.counts))
            try:
                if (self.out / 'receipt.json').exists():
                    (self.out / 'receipt.json').rename(self.out / 'receipt.invalid.json')
                self.receipt['files'] = {p.name: {k: v for k, v in self.descriptor(p).items() if k != 'path'}
                                         for p in self.out.iterdir() if p.is_file()}
                write(self.out / 'receipt.json', self.receipt)
            except BaseException as publication:  # noqa: BLE001 - preserve original failure
                error.add_note(f'Failure publication: {publication!r}')
            raise
        finally:
            signal.signal(signal.SIGTERM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    plan = sub.add_parser('plan')
    for role in ROLES:
        flag = role.replace('_', '-')
        plan.add_argument('--' + flag, type=Path, required=True)
        plan.add_argument('--' + flag + '-sha256', required=True)
    plan.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run')
    for role in ('plan', 'supervision', 'output'):
        run.add_argument('--' + role, type=Path, required=True)
    run.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    require(Path(sys.executable).absolute() == ROOT / '.venv/bin/python' and Path.cwd() == ROOT,
            'original general CPU interpreter and checkout')
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and '..' not in args.output.parts
            and not any(p.is_symlink() for p in args.output.parents), 'contained exclusive absolute output')
    if args.mode == 'plan':
        freeze(args)
    else:
        require(args.plan.is_absolute() and args.supervision.is_absolute(), 'absolute execution inputs')
        Diagnostic(args).execute()


if __name__ == '__main__':
    main()
