"""Pure envelope/control tests and tiny artificial parity, no real screen/corpus."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
SPEC = importlib.util.spec_from_file_location("qualify_token_test", ROOT/"scripts/qualify_dialogue_token_batching.py")
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


@pytest.fixture(autouse=True)
def isolated_tiny():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def tiny_case(mode="slot", head="scalar"):
    return {"index": 0, "name": "synthetic", "shape": [2, 3, 2, 4, 5], "mode": mode, "head": head, "seed": 410}


def empty_progress():
    return {k+suffix: 0 for k in ("parity_forward", "parity_backward", "update_forward", "update_backward", "optimizer_steps")
            for suffix in ("_attempted", "_returned")}


def fake_rows():
    rows, events = [], []
    for case in q.CASES:
        a = .90 if case["index"] < 4 else 1.10
        paths = {"original": {"seconds": a}, "batched": {"seconds": 1.}}
        row = {"case": copy.deepcopy(case), "parity": {"passed": True}, "warm": copy.deepcopy(paths), "measured": []}
        for kind in ("parity", "warm"):
            for path in ("original", "batched"):
                events.append({"kind": kind, "case": case["name"], "path": path})
        for i, order in enumerate(q.RECIPE["pair_orders"]):
            row["measured"].append({"pair": i, "order": list(order), "paths": copy.deepcopy(paths), "speed_ratio": a})
            events.extend({"kind": "measured", "case": case["name"], "path": p, "pair": i} for p in order)
        rows.append(row)
    return rows, events


def test_all12_aggregate_inclusive_thresholds_counts_no_scientific_admission():
    rows, events = fake_rows()
    result = q.aggregate(rows, events, 6*1024**3)
    assert result["engineering_admission"] and not result["full_training_authorized"]
    assert result["optimizer_updates"] == 120 and result["parity_forward_backward_passes"] == 24
    assert len(result["cells"]) == 12


@pytest.mark.parametrize("which", [0, 4, 11])
def test_each_cell_speed_threshold_is_required(which):
    rows, events = fake_rows()
    for pair in rows[which]["measured"]:
        pair["paths"]["original"]["seconds"] -= .00001
        pair["speed_ratio"] -= .00001
    assert not q.aggregate(rows, events, 100)["engineering_admission"]


def test_paired_median_is_not_ratio_of_pooled_medians():
    rows, events = fake_rows()
    for pair, a, b in zip(rows[4]["measured"], [2., 3., 40., 50.], [1., 1., 100., 100.], strict=True):
        pair["paths"] = {"original": {"seconds": a}, "batched": {"seconds": b}}
        pair["speed_ratio"] = a/b
    assert q.aggregate(rows, events, 100)["cells"][4]["median_speed_ratio"] == 1.25


@pytest.mark.parametrize("kind", ["missing", "duplicate", "foreign", "order", "nan", "ratio"])
def test_bad_coverage_or_timing_rejected(kind):
    rows, events = fake_rows()
    if kind == "missing":
        events.pop()
    elif kind == "duplicate":
        events.append(events[-1])
    elif kind == "foreign":
        rows[0]["case"]["seed"] += 1
    elif kind == "order":
        rows[0]["measured"][0]["order"].reverse()
    elif kind == "nan":
        rows[0]["measured"][0]["paths"]["original"]["seconds"] = float("nan")
    else:
        rows[0]["measured"][0]["speed_ratio"] = 4
    with pytest.raises(ValueError):
        q.aggregate(rows, events, 100)


def test_parity_and_process_rss_failure_are_retained_not_exception():
    rows, events = fake_rows()
    rows[1]["parity"]["passed"] = False
    result = q.aggregate(rows, events, 6*1024**3+1)
    assert not result["engineering_admission"] and not result["all_parity_passed"] and not result["rss_passed"]


def test_sample_ragged_public_layout_matches_original_assembler_and_is_seeded():
    study, _, _ = q.backend()
    case = tiny_case()
    data = q.sample(case)
    actor, labels, _ = study.make_batch(*data, case["mode"])
    assert list(actor[0].shape) == [2, 3, 5, 384] and actor[4].shape == (2, 2, 4)
    assert not actor[1][1, 2] and actor[4][1, 1, 0] and not actor[4][1, 1, 1:].any()
    assert not actor[6][1, 2].any() and (actor[7][~actor[6]] == 0).all()
    assert q.digest(data) == q.digest(q.sample(case))
    old_digest = q.digest(actor)
    altered = copy.deepcopy(data)
    for d in altered[0]:
        for r in d["queries"]:
            r["label"] = (r["label"]+1) % len(altered[1][r["query"]]["candidates"])
    changed, other_labels, _ = study.make_batch(*altered, case["mode"])
    assert q.digest(changed) == old_digest and not torch.equal(labels, other_labels)


@pytest.mark.parametrize("mode,head", q.ARMS)
def test_tiny_actual_monitored_output_all_input_and_parameter_gradient_parity(mode, head):
    study, _, classes = q.backend()
    case = tiny_case(mode, head)
    data = q.sample(case)
    models = {}
    for name, cls in classes.items():
        torch.manual_seed(410)
        models[name] = cls(head, attention_mode=mode)
    assert q.digest(models["original"].state_dict()) == q.digest(models["batched"].state_dict())
    progress, events = empty_progress(), []
    result = q.parity(models, data, case, torch.tensor([.5, 1., 2.]), study, torch, progress, events.append, lambda: None)
    assert result["passed"] and len(events) == 2 and progress["parity_backward_returned"] == 2
    assert set(result["inputs"]) == set(map(str, range(8))) and result["inputs"]["0"] is not None
    assert all(e["monitor"]["advance_returned"] == 3 for e in events)
    assert result["outputs"]["negative_infinity_elements"] > 0


@pytest.mark.parametrize("kind", ["nan", "inf_gradient", "support", "difference"])
def test_numerical_or_support_failure_rejected(kind):
    a = torch.tensor([1., 2.])
    b = a.clone()
    label = "Gradient"
    if kind == "nan":
        b[0] = float("nan")
    elif kind == "inf_gradient":
        a[0] = b[0] = float("-inf")
    elif kind == "support":
        b[0] = float("-inf")
        label = "Outputs"
    else:
        b[0] += .01
    with pytest.raises(ValueError):
        q.compare(a, b, 1e-5, 1e-4, label)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    prior_names = {"scripts/study_dialogue_tokens.py", "src/openjev/research/dialogue_token_memory.py"}
    prior_names |= {f"synthetic/prior-{i}.py" for i in range(33)}
    for name in prior_names | set(q.ADD_SOURCES):
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic pinned source "+name)
    parent = {"study": "dialogue-token-v1", "source_sha256": {n: q.sha(tmp_path/n) for n in prior_names},
        "runtime": q.runtime(), "config": {k: q.RECIPE[k] for k in ("learning_rate", "weight_decay", "gradient_clip", "threads", "projection_dim", "hidden_dim", "gru_width")},
        "loss_counts": {"0": 1, "1": 2, "2": 3}, "loss_weights": [2., 1., 2/3]}
    parent_path = tmp_path/"old-plan.json"
    q.write(parent_path, parent)
    protocol = tmp_path/q.ADD_SOURCES[-1]
    args = SimpleNamespace(out=tmp_path/"freeze", training_plan=parent_path, training_plan_sha256=q.sha(parent_path),
                           protocol=protocol, protocol_sha256=q.sha(protocol))
    pin = q.freeze(args)
    return args, pin


def test_freeze_is_metadata_only_exact40sources_and_no_retry(frozen):
    args, pin = frozen
    p = q.validate_plan(args.out/"plan.json", pin)
    assert len(p["source_sha256"]) == 40 and p["recipe"]["optimizer_updates"] == 120
    assert q.read(args.out/"completed.json")["model_calls"] == 0
    with pytest.raises(FileExistsError):
        q.freeze(args)


@pytest.mark.parametrize("kind", ["source", "runtime", "recipe", "weights", "snapshot", "completion"])
def test_plan_binding_corruption_rejected(frozen, kind):
    args, pin = frozen
    path = args.out/"plan.json"
    plan = q.read(path)
    if kind == "source":
        (q.ROOT/q.ADD_SOURCES[0]).write_text("changed")
    elif kind == "snapshot":
        (args.out/"sources"/q.ADD_SOURCES[0]).write_text("changed")
    elif kind == "completion":
        (args.out/"completed.json").unlink()
        q.write(args.out/"completed.json", {"status": "failed"})
    else:
        if kind == "runtime":
            plan["runtime"]["torch"] = "different"
        elif kind == "recipe":
            plan["recipe"]["whole_cap_seconds"] = 301
        else:
            plan["loss_weights"][0] = 100
        path.unlink()
        q.write(path, plan)
        pin = q.sha(path)
    with pytest.raises((ValueError, KeyError)):
        q.validate_plan(path, pin)


def test_exclusive_timeout_preserves_ledger_paid_prefix_and_demotes_completion(tmp_path):
    out = tmp_path/"attempt"
    with pytest.raises(TimeoutError, match="synthetic"), q.attempt(out, "run", {}, capped=True) as (folder, _, progress, _):
        (folder/"operations.jsonl").write_text('{"completed":1}\n')
        progress["optimizer_steps_attempted"] = 2
        progress["optimizer_steps_returned"] = 1
        q.write(folder/"completed.json", {"status": "completed"})
        raise TimeoutError("synthetic timeout")
    done = q.read(out/"failed.json")
    assert done["progress"]["optimizer_steps_returned"] == 1 and done["no_retry"]
    assert (out/"late-completion.json").exists() and not (out/"completed.json").exists()
    with pytest.raises(FileExistsError), q.attempt(out, "run", {}):
        pass


def test_original_postwarm_canonical_reset_and_alternating_pair_order(monkeypatch):
    # Tiny real modules, but the update is fake; verifies state reset without a cost screen.
    class Model(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(173186))
    study, _, _ = q.backend()
    monkeypatch.setattr(q, "parity", lambda *a: {"passed": True})
    before, events = [], []
    def step(model, optimizer, data, case, weight, study, torch, progress, check):
        before.append(float(model.w[0].detach()))
        with torch.no_grad():
            model.w.add_(1 if len(before) == 1 else 2)
        return {"seconds": 1.}
    monkeypatch.setattr(q, "update", step)
    result = q.run_case(tiny_case(), torch.ones(3), study, torch, {"original": Model, "batched": Model}, {}, events.append, lambda: None)
    assert before == [0., 0.]+[1.]*8
    assert [e["path"] for e in events[2:]] == [p for pair in q.RECIPE["pair_orders"] for p in pair]
    assert len(result["measured"]) == 4


def test_complete_fake_outer_run_retains_all144_operations_and_false_admission(frozen, tmp_path, monkeypatch):
    args, pin = frozen
    fake_torch = SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None,
        use_deterministic_algorithms=lambda b: None, tensor=lambda v, **kw: v, float32="float32")
    monkeypatch.setattr(q, "backend", lambda: (None, fake_torch, {}))
    rows, events = fake_rows()
    rows[0]["parity"]["passed"] = False
    def case_call(case, weight, study, torch, classes, progress, emit, check):
        for row in events:
            if row["case"] == case["name"]:
                emit(row)
        for prefix, n in (("parity_forward", 2), ("parity_backward", 2), ("update_forward", 10),
                          ("update_backward", 10), ("optimizer_steps", 10)):
            progress[prefix+"_attempted"] += n
            progress[prefix+"_returned"] += n
        return rows[case["index"]]
    monkeypatch.setattr(q, "run_case", case_call)
    call = SimpleNamespace(plan=args.out/"plan.json", plan_sha256=pin, out=tmp_path/"run")
    completed = q.run(call)
    done = q.read(call.out/"completed.json")
    assert completed == q.sha(call.out/"completed.json") and done["status"] == "completed"
    assert not done["engineering_admission"] and not done["full_training_authorized"]
    assert done["progress"]["optimizer_steps_returned"] == 120
    assert len((call.out/"operations.jsonl").read_text().splitlines()) == 144
    assert len(list((call.out/"cases").glob("*.json"))) == 12
    with pytest.raises(FileExistsError):
        q.run(call)
