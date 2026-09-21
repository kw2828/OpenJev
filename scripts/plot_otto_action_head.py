"""Authenticate complete saved action-head results and render every frozen arm.

This is visualization, not an independent statistical or execution audit. No
simulator, fitted model, training array or checkpoint is imported/deserialized.
Only the two prospectively chosen case-zero trajectories are animated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-action-head-figure-v1"
FAMILIES = ("dct16_neutral", "dct16_nearest", "recent32_hard", "full_bayes")
SEEDS = (7901, 7902, 7903)
PLANNERS = ("full_bayes", "dct16_neutral", "dct16_nearest", "recent32_hard")
FITS = tuple(f"{f}@{seed}" for seed in SEEDS for f in FAMILIES)
ARMS = FITS + tuple(f"{f}@planner" for f in PLANNERS)
DISPLAY = tuple(f"{f}@{mode}" for f in PLANNERS for mode in ("planner", *SEEDS))
NAMES = {"full_bayes": "Full belief", "dct16_neutral": "DCT16 neutral",
         "dct16_nearest": "DCT16 nearest", "recent32_hard": "Recent32 + hard"}
COLORS = {"full_bayes": "#477765", "dct16_neutral": "#367eb2",
          "dct16_nearest": "#cf8b30", "recent32_hard": "#877098"}
COHORTS = {"base": 690001, "shift": 700001}
CAPS = {"native_seconds": 300, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_SHA = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SCOPE = ("Visualization of complete authenticated saved summaries and selected public paths. "
         "No independent posterior, score, training or timing verification; no new simulation or model calls.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite numerical value required")
    return float(value)


def payload_names():
    names = {"started.json", "imports.json", "qualification.json", "qualification.jsonl",
             "public-kernel-base.npz", "public-kernel-shift.npz", "fits.json", "summary.json",
             "evaluation-episodes.jsonl", "evaluation-transitions.jsonl"}
    for split in ("train", "validation"):
        names.update({f"{split}-episodes.json", f"{split}-transitions.jsonl", f"{split}-targets.npz"})
        names.update(f"{split}-{f}.npz" for f in FAMILIES)
    for fit in FITS:
        family, seed = fit.split("@")
        names.update({f"head-{family}-{seed}.npz", f"fit-{family}-{seed}.json"})
    return names


def checked_manifest(run, receipt, check):
    expected = payload_names()
    require(len(expected) == 48 and set(receipt["files"]) == expected, "complete fixed 48 payload manifest")
    require({str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()} == expected | {"receipt.json"},
            "exact closed run membership; partial/late-failure output rejected")
    for name, item in receipt["files"].items():
        check()
        path = run / name
        require(not path.is_symlink() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                f"saved payload mismatch: {name}")


def authenticate(args, check):
    run = args.run.resolve()
    require(sha(run / "receipt.json") == args.receipt_sha256
            and sha(args.terminal) == args.terminal_sha256, "external receipt/terminal pins")
    receipt, terminal = read(run / "receipt.json"), read(args.terminal)
    require(receipt["status"] == terminal["status"] == "completed" and receipt["completed_fits"] == 12
            and receipt["completed_episodes"] == 1536 and terminal["returncode"] == 0
            and terminal["timed_out"] is False and terminal["group_absent"] is True
            and terminal["error"] is None and terminal["clock_error"] is None
            and terminal["timing_available"] is True and terminal["cleanup"]["errors"] == []
            and terminal["cleanup"]["reaped"] is True, "successful complete run and parent required")
    checked_manifest(run, receipt, check)
    started = read(run / "started.json")
    request, launch = started["request"], started["launch"]
    plan_path = Path(request["plan"])
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    require(sha(plan_path) == receipt["plan_sha256"] == request["plan_sha256"], "run plan binding")
    plan = read(plan_path)
    require(plan["version"] == "otto-action-head-v1" and plan["status"] == "frozen_before_run"
            and plan["configuration"]["arms"] == list(ARMS)
            and plan["configuration"]["evaluation_first_seeds"] == COHORTS
            and plan["configuration"]["cases_per_regime"] == 48
            and plan["configuration"]["evaluation_horizon"] == 2188
            and receipt["sources"] == plan["sources"], "fixed study identity")
    command = terminal["command"]
    require(command.count("--output") == command.count("--plan") == command.count("--plan-sha256") == 1,
            "unambiguous actual parent command")
    def command_path(flag):
        p = Path(command[command.index(flag) + 1])
        return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    require(command == launch["command"] and command_path("--output") == run
            and command_path("--plan") == plan_path.resolve()
            and command[command.index("--plan-sha256") + 1] == receipt["plan_sha256"]
            and Path(terminal["cwd"]).resolve() == ROOT, "actual command/output/plan join")
    supervision = Path(request["supervision"])
    if not supervision.is_absolute():
        supervision = ROOT / supervision
    require(sha(supervision) == receipt["supervision_sha256"] and read(supervision) == launch,
            "worker's launch receipt")
    for key in ("pid", "pgid", "parent_pid", "started_ns", "deadline_ns", "clock_backend", "cap_seconds",
                "watchdog_sha256", "clock_source_sha256"):
        require(launch[key] == terminal[key], f"supervisor identity {key}")
    require(terminal["clock_backend"] == receipt["clock_backend"] in ("mach_continuous_time", "CLOCK_BOOTTIME")
            and terminal["cap_seconds"] == plan["limits"]["native_seconds"]
            and terminal["started_ns"] <= receipt["started_ns"] <= receipt["finished_ns"]
            <= terminal["finished_ns"] < terminal["deadline_ns"]
            and terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
            and terminal["deadline_ns"] == terminal["started_ns"] + terminal["cap_seconds"] * 10**9,
            "strict successful native deadline enclosure")
    for name, pin in plan["sources"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and not (ROOT / path).is_symlink()
                and sha(ROOT / path) == pin, f"local frozen source {name}")
    for name, pin in plan["upstream_sources"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and sha(ROOT / "tmp/otto-source-review-01" / path) == pin, f"upstream source {name}")
    require(launch["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]
            and launch["clock_source_sha256"] == plan["sources"][CLOCK], "supervisor source pins")
    return run, receipt, terminal, plan_path


def load_values(run, receipt):
    summary = read(run / "summary.json")
    fits = read(run / "fits.json")
    require(summary["episodes"] == 1536 and set(summary["regimes"]) == set(COHORTS)
            and [f["arm"] for f in fits] == list(FITS), "all 12 fits and both regimes")
    require(summary["inherited_gate_revised"] is False
            and summary["learned_architecture_advantage_established"] is False, "claim scope remains fixed")
    for fit in fits:
        family, seed = fit["arm"].split("@")
        require(read(run / f"fit-{family}-{seed}.json") == fit
                and receipt["files"][fit["checkpoint"]]["sha256"] == fit["checkpoint_sha256"], "complete fit record join")
    rows = [json.loads(line) for line in (run / "evaluation-episodes.jsonl").read_text().splitlines()]
    expected = []
    for ri, (regime, first) in enumerate(COHORTS.items()):
        for case in range(48):
            rotation = (ri * 48 + case) % 16
            expected.extend((regime, first + case, case // 6, 1 + case % 6 // 2, arm)
                            for arm in ARMS[rotation:] + ARMS[:rotation])
    require(len(rows) == 1536 and [tuple(r[k] for k in ("regime", "seed", "block", "initial_hit", "arm"))
                                  for r in rows] == expected, "complete ordered 1,536 episode membership")
    for r in rows:
        require(type(r["found"]) is bool and type(r["steps"]) is int and 1 <= r["steps"] <= 2188
                and (r["found"] or r["steps"] == 2188), "capped failures retained")
    plotted = {"scope": SCOPE, "display_order": list(DISPLAY), "regimes": {},
               "fits": [{k: f[k] for k in ("arm", "selected_epoch", "parameters", "input_dim", "storage")}
                        for f in fits]}
    all_conditions = []
    for regime, first in COHORTS.items():
        record = summary["regimes"][regime]
        require(set(record["means"]) == set(ARMS) and set(record["criteria"]) == set(FAMILIES[:2])
                and type(record["full_head_competent"]) is bool, "complete arm and decision cells")
        conditions = [value for c in record["criteria"].values() for value in c["checks"].values()]
        require(len(conditions) == 20 and all(type(v) is bool for v in conditions), "twenty fixed candidate conditions per regime")
        all_conditions.extend(conditions)
        for criterion in record["criteria"].values():
            require(criterion["passes"] == all(criterion["checks"].values()), "criterion status consistency")
        selected = [r for r in rows if r["regime"] == regime]
        values = []
        for arm in DISPLAY:
            metrics = record["means"][arm]
            require(1 <= finite(metrics["steps"]) <= 2188 and 0 <= finite(metrics["found"]) <= 1
                    and finite(metrics["controller_seconds"]) >= 0, "finite metric ranges")
            values.append({"arm": arm, "mean_capped_moves": metrics["steps"],
                           "mean_controller_ms": 1000 * metrics["controller_seconds"],
                           "weighted_success": metrics["found"], "evolving_state_bytes": metrics["state_bytes"],
                           "found": sum(r["found"] for r in selected if r["arm"] == arm), "episodes": 48})
        plotted["regimes"][regime] = {"weights": record["weights"], "values": values,
            "full_head_competent": record["full_head_competent"], "candidate_checks_passed": sum(conditions),
            "candidate_checks_total": 20, "case_zero_seed": first}
    competent = all(s["full_head_competent"] for s in plotted["regimes"].values())
    require(summary["readout_pilot_passes"] is (competent and all(all_conditions))
            and receipt["readout_pilot_passes"] == summary["readout_pilot_passes"], "overall decision consistency")
    plotted.update(full_head_competent=competent, readout_pilot_passes=summary["readout_pilot_passes"],
                   candidate_checks_passed=sum(all_conditions), candidate_checks_total=40)
    return plotted, rows


def figures(output, plotted, check):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.subplots_adjust(left=.13, right=.98, bottom=.14, top=.82, wspace=.44, hspace=.35)
    status = lambda value: "PASS" if value else "FAIL"
    fig.text(.035, .965, "Can direct action heads preserve search utility and reduce complete computation?", fontsize=23, weight="bold")
    fig.text(.035, .933, "1,536 autonomous episodes | All 12 fitted heads and 4 unchanged planners | 48 paired cases per regime", fontsize=14)
    fig.text(.035, .9, f"Full-belief head competence: {status(plotted['full_head_competent'])}    "
             f"Pilot: {status(plotted['readout_pilot_passes'])}    Candidate checks: "
             f"{plotted['candidate_checks_passed']}/40 passed    No architecture novelty established", fontsize=14, weight="bold")
    fig.text(.035, .867, "Solid rows: fitted seeds 7901/7902/7903. Hatched rows: original analytic planner. Every fitted seed remains visible.", fontsize=11)
    for col, regime in enumerate(COHORTS):
        record = plotted["regimes"][regime]
        title = ("Baseline: known lambda 3" if regime == "base" else "Shift: known lambda 4, TRAIN lambda 3 only")
        title += f" | Full-head competence {status(record['full_head_competent'])}"
        for row, key in enumerate(("mean_capped_moves", "mean_controller_ms")):
            ax = axes[row, col]
            values = [v[key] for v in record["values"]]
            maximum = max(values)
            require(maximum > 0, "positive plotted axis extent")
            labels = []
            for i, (item, value) in enumerate(zip(record["values"], values, strict=True)):
                family, mode = item["arm"].split("@")
                labels.append(f"{NAMES[family]} | {'planner' if mode == 'planner' else 'seed ' + mode}")
                ax.barh(i, value, height=.68, color=COLORS[family], alpha=.50 if mode == "planner" else .9,
                        hatch="///" if mode == "planner" else None, edgecolor="white", linewidth=.7)
                suffix = f"  ({item['found']}/48 found)" if row == 0 else ""
                ax.text(value + maximum * .012, i, f"{value:.2f}{suffix}", va="center", fontsize=9)
            ax.set_yticks(range(16), labels, fontsize=9)
            ax.set_ylim(15.7, -.7)
            ax.set_xlim(0, maximum * (1.34 if row == 0 else 1.2))
            for boundary in (3.5, 7.5, 11.5):
                ax.axhline(boundary, color="#64748b", linewidth=.65, alpha=.35)
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
            ax.set_title(title if row == 0 else "Complete controller cost per episode", loc="left", fontsize=12, weight="bold", pad=12)
            ax.set_xlabel("Mixture-weighted capped moves, lower is better" if row == 0 else "Mixture-weighted milliseconds, lower is better")
    notes = ["Both regimes use their own initial-hit mixture. Found counts are unweighted. All failed searches contribute the full 2,188-move horizon.",
             "Controller time includes feature construction, standardization, inference/planning, legal selection, updates and allocated setup; one rotated CPU pass.",
             "Training and collection cost are separate. Evolving actor state excludes immutable learned weights/standardizers, shared tables and temporary workspaces.",
             "Full belief has 362,788 head parameters versus 199,396 for compact/recent heads. Matching hidden widths does not equalize model capacity.",
             "Known lambda 4 and longer horizons are transfer tests. No seed/fill selection, prior-gate reversal or learned-architecture claim."]
    for i, note in enumerate(notes):
        fig.text(.035, .105 - i * .018, note, fontsize=10, color="#475569")
    check()
    fig.savefig(output / "action-head.png", dpi=160)
    fig.savefig(output / "action-head.pdf")
    plt.close(fig)


def replay_paths(run, rows, check):
    paths = {}
    with (run / "evaluation-transitions.jsonl").open() as stream:
        for index, line in enumerate(stream):
            if index % 1000 == 0:
                check()
            record = json.loads(line)
            regime = record["regime"]
            if record["seed"] != COHORTS[regime]:
                continue
            key = (regime, record["arm"])
            require(record["arm"] in ARMS, "replay arm membership")
            position = record["public"]["position"]
            require(len(position) == 2 and all(type(v) is int and 0 <= v < 53 for v in position), "public replay coordinates")
            if record["kind"] == "reset":
                require(key not in paths and record["public"]["step"] == 0, "unique replay reset")
                paths[key] = {"positions": [position], "source": record["source_evaluation_only"], "seed": record["seed"]}
            else:
                require(record["kind"] == "step" and key in paths
                        and record["step"] == record["public"]["step"] == len(paths[key]["positions"]), "consecutive replay steps")
                paths[key]["positions"].append(position)
    require(len(paths) == 32, "all 16 preselected arms in both regimes")
    for row in rows:
        if row["seed"] != COHORTS[row["regime"]]:
            continue
        path = paths[row["regime"], row["arm"]]
        require(len(path["positions"]) == row["steps"] + 1 and path["source"] == row["source_evaluation_only"], "replay episode join")
        path.update(steps=row["steps"], found=row["found"])
    for regime in COHORTS:
        require(len({tuple(paths[regime, arm]["source"]) for arm in ARMS}) == 1, "paired replay source")
    return paths


def replays(output, paths, check):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    records = {}
    for regime, seed in COHORTS.items():
        maximum = max(paths[regime, arm]["steps"] for arm in ARMS)
        stride = max(1, math.ceil(maximum / 159))
        frames = list(range(0, maximum + 1, stride))
        if frames[-1] != maximum:
            frames.append(maximum)
        require(len(frames) <= 160 and frames[0] == 0 and frames[-1] == maximum, "bounded fixed replay sampling")
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=85)
        fig.subplots_adjust(left=.035, right=.985, bottom=.06, top=.88, hspace=.42, wspace=.18)
        fig.suptitle(f"{regime.title()} case zero, seed {seed}: all 16 arms", fontsize=17, weight="bold", y=.975)
        subtitle = fig.text(.5, .937, "", ha="center", fontsize=10)
        fig.text(.5, .018, "Saved paths only; source stars are viewer-only. Playback timing is illustrative, not measured latency.", ha="center", fontsize=9)
        artists = {}
        for ax, arm in zip(axes.flat, DISPLAY, strict=True):
            family, mode = arm.split("@")
            path = paths[regime, arm]
            ax.set(xlim=(-1, 53), ylim=(53, -1), aspect="equal", xticks=[], yticks=[])
            ax.set_facecolor("#f1f5f9")
            ax.scatter(path["source"][1], path["source"][0], marker="*", s=95, color="#a43c31", zorder=4)
            trail, = ax.plot([], [], color=COLORS[family], linewidth=1.15, alpha=.75)
            dot, = ax.plot([], [], "o", color=COLORS[family], markersize=5)
            artists[arm] = trail, dot, ax.set_title("", fontsize=8)
        images = []
        for move in frames:
            check()
            subtitle.set_text(f"Common move index {move} | Fixed stride {stride}, final included | Columns: planner, seeds 7901/7902/7903")
            for arm, (trail, dot, title) in artists.items():
                path = paths[regime, arm]
                upto = min(move, path["steps"])
                positions = np.asarray(path["positions"][:upto + 1])
                trail.set_data(positions[:, 1], positions[:, 0])
                dot.set_data([positions[-1, 1]], [positions[-1, 0]])
                family, mode = arm.split("@")
                status = "found" if move >= path["steps"] and path["found"] else "capped" if move >= path["steps"] else "searching"
                title.set_text(f"{NAMES[family]} | {mode}\nMove {upto}: {status}")
            fig.canvas.draw()
            frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            images.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        plt.close(fig)
        images[0].save(output / f"first-case-{regime}.gif", save_all=True, append_images=images[1:],
                       duration=[170] * (len(images) - 1) + [1500], loop=0, optimize=False)
        images[-1].convert("RGB").save(output / f"first-case-{regime}-final.png")
        records[regime] = {"seed": seed, "case_index": 0, "arms": list(DISPLAY), "stride": stride,
                           "frame_move_indices": frames, "maximum_moves": maximum, "source_markers": "viewer-only",
                           "paths": {a: paths[regime, a] for a in DISPLAY}}
        images.clear()
    return records


def execute(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_pin = sha(Path(__file__))
    clock = started = None
    previous_handler = signal.getsignal(signal.SIGALRM)
    receipt = {"version": VERSION, "status": "started", "source_sha256": source_pin, "limits": CAPS,
               "model_calls": 0, "environment_calls": 0, "scope": SCOPE}
    def check():
        require(clock.now_ns() - started < CAPS["native_seconds"] * 10**9, "native plotting cap")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        receipt["peak_rss_bytes"] = rss
        require(rss <= CAPS["rss_bytes"] and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= CAPS["output_bytes"], "plot storage/RSS cap")
    def alarm(_signum, _frame):
        raise TimeoutError("plotting emergency wall cap")
    try:
        require(sha(ROOT / CLOCK) == CLOCK_SHA, "qualified clock source")
        spec = importlib.util.spec_from_file_location("otto_action_plot_clock", ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        clock = module.SuspendClock()
        started = clock.now_ns()
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, CAPS["native_seconds"])
        write(out / "started.json", {"request": {k: str(v) for k, v in vars(args).items()}, **receipt})
        run, original, _terminal, plan_path = authenticate(args, check)
        values, rows = load_values(run, original)
        receipt.update(producer_receipt_sha256=args.receipt_sha256, terminal_sha256=args.terminal_sha256,
                       producer_summary_sha256=original["files"]["summary.json"]["sha256"],
                       plan_sha256=original["plan_sha256"], run=str(run),
                       full_head_competent=values["full_head_competent"], readout_pilot_passes=values["readout_pilot_passes"])
        write(out / "plotted-values.json", values)
        figures(out, values, check)
        paths = replay_paths(run, rows, check)
        replay = replays(out, paths, check)
        write(out / "replay-receipt.json", {"scope": "Protocol-preselected case zero only; no simulation",
              "producer_receipt_sha256": args.receipt_sha256, "terminal_sha256": args.terminal_sha256,
              "transitions_sha256": original["files"]["evaluation-transitions.jsonl"]["sha256"],
              "source_sha256": source_pin, "cases": replay})
        require(sha(run / "receipt.json") == args.receipt_sha256 and sha(args.terminal) == args.terminal_sha256
                and sha(plan_path) == original["plan_sha256"] and sha(Path(__file__)) == source_pin, "end input/source pins")
        checked_manifest(run, original, check)
        check()
        finished = clock.now_ns()
        receipt.update(status="completed", started_ns=started, finished_ns=finished,
                       clock_backend=clock.backend, elapsed_ns=finished - started,
                       wall_seconds=(finished - started) / 1e9, timing_available=True,
                       files={p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()})
        write(out / "receipt.json", receipt)
        check()
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(status="failed", error=repr(error), finished_ns=None, elapsed_ns=None,
                       wall_seconds=None, timing_available=False)
        try:
            if (out / "receipt.json").exists():
                (out / "receipt.json").rename(out / "receipt-before-failure.json")
            write(out / "failed.json", receipt)
        except OSError as secondary:
            error.add_note(f"Preserving plotting failure: {secondary!r}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "terminal", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--terminal-sha256", required=True)
    execute(parser.parse_args())
