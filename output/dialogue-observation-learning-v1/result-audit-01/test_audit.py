"""Small artificial saved artifacts only; no real evaluator or model access."""
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("independent_observation_audit", HERE / "audit.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
ROOT = HERE.parents[2]
report_spec = importlib.util.spec_from_file_location("synthetic_observation_report", ROOT / "scripts/report_dialogue_observation.py")
producer = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(producer)  # Integration fixture only; never imported by auditor.


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))


def manifest(path):
    return {p.relative_to(path).as_posix():a.describe(p) for p in path.rglob("*") if p.is_file() and p.name != "completed.json"}


def sample():
    rows=[]
    for unseen in (False,True):
        for t,(bin_name,target) in enumerate((("unmentioned_retention",0),("assigned_retention",2),("first_assignment",3))):
            ids=[a.NONE,a.DC,"value:True","value:False"]
            rows.append({"row_index":len(rows),"split":"dev","dialogue_id":str(unseen),"source_row_index":t,
                         "query_index":int(unseen),"time":t,"service":str(unseen),"slot":"flag","candidate_ids":ids,
                         "candidate_values":[None,None,"True","False"],"label_index":target,"label_id":ids[target],"bin":bin_name,"unseen":unseen})
    logs=np.full((6,12),-np.inf,np.float32)
    logs[:,:4]=np.log([[.5,.1,.3,.1],[.1,.1,.7,.1],[.1,.1,.2,.6]]*2).astype(np.float32)
    refs={"row_indices":list(range(6)),"original":[0,2,2]*2,"numbers":[0,2,3]*2}
    return rows,logs,refs


@pytest.fixture
def complete(tmp_path,monkeypatch):
    root=tmp_path/"repo";root.mkdir()
    monkeypatch.setattr(a,"ROOT",root)
    sources={}
    names=[a.REPORTER,a.METRICS,a.WATCHDOG,"tests/test_report_dialogue_observation.py","tests/test_dialogue_observation_metrics.py"]
    names += [f"synthetic/source-{i}.py" for i in range(48)]
    for name in names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# synthetic identity "+name)
        sources[name]=a.sha(path)
    monkeypatch.setattr(a,"REPORTER_PIN",sources[a.REPORTER]);monkeypatch.setattr(a,"METRICS_PIN",sources[a.METRICS])
    prior=root/"parents";prior.mkdir()
    dump(prior/"completed.json",{"status":"completed"})
    dump(prior/"audit.json",{"status":"completed","agreement":True,"execution_completed_sha256":a.sha(prior/"completed.json")})
    (prior/"protocol.md").write_text("Synthetic prospective allocation")
    allocation={"limits":{"wall_seconds":100,"rss_bytes":10**9,"output_bytes":10**8,"mps_driver_bytes":10**9},
                "cost_completed":{"path":str(prior/"completed.json"),"sha256":a.sha(prior/"completed.json")},
                "cost_audit":{"path":str(prior/"audit.json"),"sha256":a.sha(prior/"audit.json")},
                "protocol":{"path":str(prior/"protocol.md"),"sha256":a.sha(prior/"protocol.md")}}
    run=root/"run";run.mkdir();freeze=root/"freeze";freeze.mkdir();report=root/"report";report.mkdir()
    rows,logs,refs=sample()
    text="".join(json.dumps(r)+"\n" for r in rows)
    for folder in (run,freeze):
        (folder/"evaluation-rows.jsonl").write_text(text);dump(folder/"references.json",refs);dump(folder/"allocation.json",allocation)
    expected={"fits":12,"per_fit":{"evaluation_endpoints":6,"optimizer_updates":1},
              "all_fits":{"evaluation_endpoints":72,"optimizer_updates":12},"encoder_work_per_fit":{"input_texts":1},"state_counts_per_fit":{"forward_calls":1}}
    plan={"version":a.STUDY,"fit_order":a.FIT_ORDER,"source_sha256":sources,"allocation":allocation,
          "allocation_sha256":a.sha(run/"allocation.json"),"expected":expected,
          "evaluation_rows_sha256":a.sha(run/"evaluation-rows.jsonl"),"references_sha256":a.sha(run/"references.json")}
    dump(freeze/"plan.json",plan);dump(run/"plan.json",plan)
    pin=a.sha(freeze/"plan.json")
    dump(run/"started.json",{"request":{"command":"train","plan_sha256":pin,"out":str(run)}})
    dump(freeze/"started.json",{"synthetic":True})
    for name in names:
        path=freeze/"sources"/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((root/name).read_bytes())
    dump(freeze/"completed.json",{"status":"completed","phase":"freeze","version":a.STUDY,"plan_sha256":pin,"source_sha256":sources,"files":manifest(freeze)})
    packets={};fits=[]
    for name in a.FIT_ORDER:
        arm,seed=name.rsplit("-",1);folder=run/name;folder.mkdir()
        packets[name]={"row_indices":np.arange(6,dtype=np.int64),"log_probs":logs.copy()}
        np.savez(folder/"predictions.npz",**packets[name])
        (folder/"weights.pt").write_bytes(b"opaque never-deserialized checkpoint")
        (folder/"updates.jsonl").write_text('{"synthetic":true}\n')
        done={"version":a.STUDY,"status":"completed","fit_id":name,"arm":arm,"seed":int(seed),"evaluation":{"rows":6},
              "initial_sha256":{"encoder":"a"*64,"memory":"b"*64},"final_encoder_sha256":("c" if arm.startswith("trainable") else "a")*64,
              "counts":expected["per_fit"],"encoder_work":expected["encoder_work_per_fit"],"invariants":expected["state_counts_per_fit"],"files":manifest(folder)}
        dump(folder/"completed.json",done);fits.append({"fit_id":name,"completed_sha256":a.sha(folder/"completed.json")})
    done={"status":"completed","phase":"train","version":a.STUDY,"fit_count":12,"fits":fits,"plan_sha256":pin,
          "allocation_sha256":plan["allocation_sha256"],"source_sha256":sources,"wall_seconds":5.,"peak_rss_bytes":100,
          "sampled_mps_driver_max_bytes":100,"sampled_mps_current_max_bytes":50,"counts":expected["all_fits"],
          "quality_scoring_in_runner":False,"official_test_opened":False,"external_model_api_calls":0}
    done["files"]={p.relative_to(run).as_posix():a.describe(p) for p in run.rglob("*") if p.is_file()}
    dump(run/"completed.json",done)
    command=[sys.executable,str(root/"scripts/study_dialogue_observation.py"),"train","--plan",str(freeze/"plan.json"),"--plan-sha256",pin,"--out",str(run)]
    launch=root/"launch.json";terminal=root/"terminal.json"
    dump(launch,{"command":command,"pid":123,"pgid":123,"cap_seconds":100,"watchdog_sha256":sources[a.WATCHDOG],"started_unix":100.})
    dump(terminal,{"command":command,"pgid":123,"returncode":0,"error":None,"timed_out":False,"group_absent":True,"finished_unix":105.5,"wall_seconds":6.})
    summary=producer.aggregate(packets,rows,refs)
    summary.update(version=a.REPORT,status="completed",technical_validity_passed=True,technical_complete_fits=12,
                   execution_completed_sha256=a.sha(run/"completed.json"),plan_sha256=pin)
    dump(report/"summary.json",summary);dump(report/"started.json",{"synthetic":True});(report/"report.md").write_text("Synthetic only")
    receipt={"status":"completed","version":a.REPORT,"technical_validity_passed":True,"execution_completed_sha256":a.sha(run/"completed.json"),
             "plan_sha256":pin,"launch_sha256":a.sha(launch),"terminal_sha256":a.sha(terminal),"execution_source_sha256":sources,
             "source_sha256":{k:sources[k] for k in names[:2]+names[3:5]},"files":manifest(report),
             "continuation_passed":summary["continuation"]["passed"],"scientific_checks_passed":summary["continuation"]["checks_passed"],"scientific_checks_total":7}
    dump(report/"receipt.json",receipt)
    args=SimpleNamespace(run=run,completed_sha256=a.sha(run/"completed.json"),plan=freeze/"plan.json",plan_sha256=pin,
                         launch=launch,launch_sha256=a.sha(launch),terminal=terminal,terminal_sha256=a.sha(terminal),
                         report=report,report_receipt_sha256=a.sha(report/"receipt.json"),report_summary_sha256=a.sha(report/"summary.json"),out=root/"result")
    return args


def test_full_synthetic_artifact_audit_matches_production_arithmetic(complete):
    receipt=a.execute(complete)
    assert receipt["agreement"] is True and receipt["fit_cells"]==144 and receipt["literal_cells"]==24
    assert receipt["scientific_checks_passed"]==5 and receipt["continuation_passed"] is False
    assert receipt["scalar_checks"] > 1000
    result=a.read(complete.out/"summary.json")
    assert result["fits"][a.FIT_ORDER[0]]["panels"]["all"]["micro"]["accuracy"]==1
    assert result["references"]["original"]["panels"]["unseen"]["micro"]["correct"]==2
    with pytest.raises(FileExistsError): a.execute(complete)


def test_closed_form_tie_underflow_and_candidate_brier():
    rows,logs,_=sample();info=a.layout(rows)
    logs[0,:4]=[0.,-1000.,-1000.,-1000.]
    rows[0]["label_index"]=2;rows[0]["label_id"]="value:True"
    logs[1,:4]=np.log([.4,.1,.4,.1]).astype(np.float32)
    result=a.score(logs,np.arange(6,dtype=np.int64),a.layout(rows))
    expected_nll=math.fsum([-float(logs[i,r["label_index"]]) for i,r in enumerate(rows)])/6
    assert result["panels"]["all"]["micro"]["nll"]==expected_nll
    expected_brier=[]
    for i,row in enumerate(rows):
        p=[math.exp(float(x)) for x in logs[i,:4]]
        expected_brier.append(math.fsum((v-(j==row["label_index"]))**2 for j,v in enumerate(p)))
    assert result["panels"]["all"]["micro"]["brier"]==pytest.approx(math.fsum(expected_brier)/6,abs=1e-15)
    assert result["validation"]["exact_top1_tie_rows"]==1 and result["validation"]["float64_exponent_underflow_positions"]==3
    assert info["rows"]==6


@pytest.mark.parametrize("mutation",["metric","gate","terminal","checkpoint","membership"])
def test_corruption_is_rejected_and_failure_preserved(complete,monkeypatch,mutation):
    if mutation in ("metric","gate"):
        summary=a.read(complete.report/"summary.json")
        if mutation=="metric": summary["fits"][a.FIT_ORDER[0]]["panels"]["unseen"]["micro"]["nll"] += .01
        else: summary["continuation"]["checks"][0]["passed"]=True
        dump(complete.report/"summary.json",summary)
        receipt=a.read(complete.report/"receipt.json");receipt["files"]["summary.json"]=a.describe(complete.report/"summary.json")
        dump(complete.report/"receipt.json",receipt)
        complete.report_summary_sha256=a.sha(complete.report/"summary.json");complete.report_receipt_sha256=a.sha(complete.report/"receipt.json")
    elif mutation=="terminal":
        value=a.read(complete.terminal);value["timed_out"]=True;dump(complete.terminal,value);complete.terminal_sha256=a.sha(complete.terminal)
    elif mutation=="checkpoint": (complete.run/a.FIT_ORDER[0]/"weights.pt").write_bytes(b"changed")
    else:
        value=a.read(complete.run/"completed.json");value["fits"].pop();dump(complete.run/"completed.json",value);complete.completed_sha256=a.sha(complete.run/"completed.json")
    if mutation not in ("metric","gate"):
        monkeypatch.setattr(np,"load",lambda *_args,**_kwargs: pytest.fail("Arrays read before authentication"))
    with pytest.raises(ValueError): a.execute(complete)
    assert a.read(complete.out/"failed.json")["agreement"] is False
    assert not (complete.out/"receipt.json").exists()


def test_exact_inclusive_margins_strict_wins_and_missing_support():
    rows,logs,_=sample();base=a.score(logs,np.arange(6,dtype=np.int64),a.layout(rows))
    fits={name:copy.deepcopy(base) for name in a.FIT_ORDER}
    for seed in a.SEEDS:
        for arm in a.ARMS:
            fit=fits[f"{arm}-{seed}"]
            for panel in a.PANELS:
                for s in a.STRATA:
                    fit["panels"][panel]["strata"][s].update(count=200,correct=100,incorrect=100)
        for s in a.STRATA:
            fits[f"trainable_numbers-{seed}"]["panels"]["unseen"]["strata"][s].update(correct=102,incorrect=98)
        # Exactly +.005 seen assigned error and -.01 seen macro.
        fit=fits[f"trainable_numbers-{seed}"]["panels"]["seen"]["strata"]
        for s,hits in zip(a.STRATA,(98,99,97),strict=True): fit[s].update(correct=hits,incorrect=200-hits)
    gate=a.decision(fits)
    assert gate["passed"] is True and gate["checks_passed"]==7
    fits["trainable_numbers-6903"]["panels"]["seen"]["strata"]["assigned_retention"].update(correct=98,incorrect=102)
    assert not a.decision(fits)["checks"][-2]["passed"]
    fits["frozen_numbers-6901"]["panels"]["unseen"]["strata"]["changed"]["count"]=0
    assert a.decision(fits)["checks"][0]["mean"] is None and not a.decision(fits)["passed"]
    del fits[a.FIT_ORDER[-1]]
    with pytest.raises(ValueError): a.decision(fits)


def test_literal_reserved_identity_and_bad_raw_support():
    rows,logs,refs=sample();info=a.layout(rows)
    refs["original"][0]=1
    with pytest.raises(ValueError,match="DONTCARE"): a.literal(refs["original"],refs["row_indices"],rows,info)
    rows[0]["candidate_values"][0]="None"
    with pytest.raises(ValueError,match="reserved"): a.layout(rows)
    for error in ("infinity","padding","mass"):
        corrupt=logs.copy()
        if error=="infinity": corrupt[0,2]=-np.inf
        elif error=="padding": corrupt[0,4]=-1000.
        else: corrupt[0,:4]-=.01
        with pytest.raises(ValueError): a.score(corrupt,np.arange(6,dtype=np.int64),info)
