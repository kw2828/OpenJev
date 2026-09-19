"""Plot all paired gap comparisons from the completed, externally bound audit.

Reads saved JSON only. No model, simulator, training or evaluation code is used.
Existing report artifacts remain untouched; output is exclusive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = "reacher-memory-ablation-v1"
REPORT_SHA = "61596ed1562c73da64ac8e6c4da744e40e305531e0e0e6c539e1f05808cb95c7"
AUDIT_SHA = "2589383326a2312d7f738821da7fd11f82b9bee1b7168aba2e7f44aaa4946d5d"
PANELS = ("ordinary", "shift")
COMPARATORS = ("current_gru", "bounded_gru", "packet_mlp")
PAIRS = ("pair0", "pair1", "pair2")
LABELS = {"current_gru":"Current-packet GRU", "bounded_gru":"3-packet GRU", "packet_mlp":"Packet MLP"}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def checked(path, digest):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == digest, f"Bound artifact mismatch: {path}")
    return path


def files(folder, mapping, expected):
    require(set(mapping) == set(expected), "Exact bound artifact membership")
    for name, digest in mapping.items():
        checked(Path(folder)/name, digest)


def authenticate(args):
    require(args.expected_report_sha256 == REPORT_SHA and args.expected_audit_sha256 == AUDIT_SHA,
            "Explicit final report and audit hashes required")
    report_receipt = read(checked(args.report_receipt, args.expected_report_sha256))
    audit_receipt = read(checked(args.audit_receipt, args.expected_audit_sha256))
    require(report_receipt["status"] == audit_receipt["status"] == "completed"
            and report_receipt["engineering"] is audit_receipt["engineering"] is False
            and audit_receipt["saved_output_only"] is True
            and report_receipt["scope"] == "completed_scored_development_study", "Completed scored scope only")
    require(report_receipt["inputs"]["audit_receipt_sha256"] == args.expected_audit_sha256
            and report_receipt["inputs"]["plan_sha256"] == audit_receipt["plan_sha256"]
            and report_receipt["inputs"]["audit_summary_sha256"] == audit_receipt["files"]["summary.json"]
            and report_receipt["inputs"]["execution_completed_sha256"] == audit_receipt["execution_completed_sha256"],
            "Final report and audit identity")
    files(args.report_receipt.parent, report_receipt["files"], ("report.json", "tables.md", "native-costs.png", "utility-vs-cost.png"))
    files(args.audit_receipt.parent, audit_receipt["files"], ("summary.json", "README.md"))
    for folder in (args.report_receipt.parent, args.audit_receipt.parent):
        require(not (folder/"failed.json").exists() and not (folder/"over-cap-receipt.json").exists(), "No failed or over-cap output")
    plan = read(checked(ROOT/f"evidence/{STUDY}/protocol/plan.json", audit_receipt["plan_sha256"]))
    require(plan["study"] == STUDY and plan["engineering"] is False
            and plan["sources"] == audit_receipt["source_sha256"], "Frozen scored plan identity")
    checked(ROOT/f"output/{STUDY}/report_results.py", report_receipt["source_sha256"])
    report = read(args.report_receipt.parent/"report.json")
    summary = read(args.audit_receipt.parent/"summary.json")
    require(report["bindings"] == report_receipt["inputs"] and report["fits"] == summary["fits"]
            and report["continuation_gate"] == summary["continuation_gate"]
            and summary["status"] == "completed" and summary["engineering"] is False
            and summary["plan_sha256"] == audit_receipt["plan_sha256"]
            and summary["saved_output_only"] is True and summary["new_model_calls"] == 0, "Final saved summary binding")
    names = {f"{arm}-{pair}" for arm in ("residual_gru",*COMPARATORS) for pair in PAIRS}
    references = {"known_state", "particle", "public_kinematic", "zero", "uniform"}
    require(set(summary["fits"]) == names and len(summary["continuation_gate"]["checks"]) == 25, "All12 fits and25 checks retained")
    for panel in ("full",*PANELS):
        require(set(summary["control"][panel]) == names|references, "Every fit and reference retained")
    return plan, summary, report_receipt


def calculate(plan, summary):
    rows = []
    for panel in PANELS:
        for comparator in COMPARATORS:
            persistent, baseline = [], []
            for pair in PAIRS:
                for arm, values in (("residual_gru",persistent),(comparator,baseline)):
                    row = summary["control"][panel][f"{arm}-{pair}"]
                    cases = np.asarray(row["episode_costs"], dtype=float)
                    require(cases.shape == (plan["control_episodes"],) and np.isfinite(cases).all()
                            and math.isclose(float(cases.mean()),row["mean_cost"],rel_tol=1e-12,abs_tol=1e-12), "Complete paired native case means")
                    require(row["mean_cost"] > 0, "Strictly positive native mean cost")
                    values.append(row["mean_cost"])
            family_persistent, family_baseline = float(np.mean(persistent)), float(np.mean(baseline))
            rows.append({"panel":panel, "comparator":comparator, "pair_ids":list(PAIRS),
                "persistent_pair_costs":persistent, "comparator_pair_costs":baseline,
                "paired_improvement_percent":[100*(1-left/right) for left,right in zip(persistent,baseline,strict=True)],
                "persistent_family_mean_cost":family_persistent, "comparator_family_mean_cost":family_baseline,
                "family_improvement_percent":100*(1-family_persistent/family_baseline)})
    return rows


def plot(rows, path, passed, checks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    require(len(rows) == 6, "All six gap comparisons required")
    fig, axes = plt.subplots(1,2,figsize=(13.5,6.4),sharex=True,sharey=True)
    colors = ("#007C91", "#D27828", "#775CA8")
    shapes = ("o", "^", "s")
    offsets = (-.19,0,.19)
    coordinates = [value for row in rows for value in (*row["paired_improvement_percent"],row["family_improvement_percent"])]
    left, right = min(0,min(coordinates)), max(0,max(coordinates))
    for ax,panel in zip(axes,PANELS,strict=True):
        ax.axvspan(left-4,0,color="#FFF2EE",zorder=0)
        ax.axvline(0,color="#777777",linewidth=1.2,zorder=1)
        for position,comparator in enumerate(COMPARATORS):
            row = next(value for value in rows if value["panel"] == panel and value["comparator"] == comparator)
            for index,value in enumerate(row["paired_improvement_percent"]):
                y = position+offsets[index]
                ax.scatter(value,y,s=67,color=colors[index],marker=shapes[index],edgecolor="white",linewidth=.7,zorder=3)
                ax.annotate(f"{value:+.2f}%",(value,y),xytext=(6,0),textcoords="offset points",ha="left",va="center",fontsize=8.5,color=colors[index])
            mean = row["family_improvement_percent"]
            ax.scatter(mean,position+.35,marker="D",s=62,color="#222222",zorder=4)
            ax.annotate(f"{mean:+.2f}% family",(mean,position+.35),xytext=(6,0),textcoords="offset points",ha="left",va="center",fontsize=8.5,fontweight="bold")
        ax.set_title("6-step observation gaps" if panel == "ordinary" else "10-step gap shift",loc="left",fontsize=13,pad=13)
        ax.set_yticks(range(3),[f"vs {LABELS[name]}" for name in COMPARATORS],fontsize=10)
        ax.set_xlabel("Native-cost improvement of persistent GRU (%)",fontsize=10)
        ax.set_xlim(left-3,right+7)
        ax.set_ylim(2.6,-.5)
        ax.grid(axis="x",color="#E2E2E2",linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y",length=0)
        ax.spines[["top","right","left"]].set_visible(False)
    fig.suptitle(f"Persistent public memory: every paired comparison | {'PASS' if passed else 'FAIL'} ({checks}/25)",
                 x=.045,y=.98,ha="left",fontsize=17,fontweight="bold")
    fig.text(.045,.908,"Positive: persistent GRU has lower native cost. Negative: comparator has lower cost. All three paired fits retained.",fontsize=10.5,color="#444444")
    handles = [Line2D([],[],marker=shapes[i],linestyle="",color=colors[i],label=f"Pair {i}") for i in range(3)]
    handles.append(Line2D([],[],marker="D",linestyle="",color="#222222",label="Change in family mean cost"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,.185),ncol=4,frameon=False,fontsize=10)
    fig.text(.045,.035,"Dots: 100 x (1 - persistent pair cost / comparator pair cost). Diamond: 100 x (1 - mean persistent cost / mean comparator cost).\n"
             "The family percentage is not the mean of pair percentages. These are descriptive results from three paired fits in one environment, not confidence intervals.\n"
             "Uneven fit gains and losing MLP comparisons remain visible. This does not establish biological novelty or identify which retained information explains the gain.",
             fontsize=9,color="#444444",va="bottom")
    fig.subplots_adjust(left=.17,right=.98,top=.80,bottom=.27,wspace=.15)
    fig.savefig(path,dpi=190,facecolor="white",metadata={"Description":"All paired comparisons from completed scored Reacher memory study"})
    plt.close(fig)


def render(args):
    out=Path(args.out)
    require(not out.exists(), "Exclusive paired-figure directory required")
    started=time.monotonic()
    try:
        plan,summary,report_receipt=authenticate(args)
        rows=calculate(plan,summary)
        checks=sum(row["passed"] for row in summary["continuation_gate"]["checks"])
        out.mkdir(parents=True,exist_ok=False)
        plot(rows,out/"paired-improvements.png",summary["continuation_gate"]["passed"],checks)
        receipt={"status":"completed","study":STUDY,"engineering":False,
            "plan_sha256":report_receipt["inputs"]["plan_sha256"],
            "report_receipt_sha256":args.expected_report_sha256,"audit_receipt_sha256":args.expected_audit_sha256,
            "audit_summary_sha256":report_receipt["inputs"]["audit_summary_sha256"],
            "source_sha256":sha(__file__),"new_model_calls":0,"new_simulator_calls":0,
            "wall_seconds":time.monotonic()-started,"comparisons":rows,
            "pair_formula":"100 * (1 - persistent_pair_cost / comparator_pair_cost)",
            "family_formula":"100 * (1 - mean(persistent_pair_costs) / mean(comparator_pair_costs))",
            "family_is_not_mean_of_pair_percentages":True,"all_fit_pairs_retained":True,
            "files":{"paired-improvements.png":sha(out/"paired-improvements.png")}}
        write(out/"receipt.json",receipt)
        return {"status":"completed","receipt_sha256":sha(out/"receipt.json"),"out":str(out)}
    except BaseException as error:
        out.mkdir(parents=True,exist_ok=True)
        write(out/"failed.json",{"status":"failed","error":repr(error),"wall_seconds":time.monotonic()-started})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-receipt",type=Path,required=True)
    parser.add_argument("--expected-report-sha256",required=True)
    parser.add_argument("--audit-receipt",type=Path,required=True)
    parser.add_argument("--expected-audit-sha256",required=True)
    parser.add_argument("--out",type=Path,required=True)
    print(json.dumps(render(parser.parse_args())))


if __name__ == "__main__":
    main()
