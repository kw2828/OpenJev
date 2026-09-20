"""Pure count boundaries and synthetic saved-file authentication, without models."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dialogue_copy_report_test", ROOT / "scripts/report_dialogue_copy.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)
base = reporter.base


def test_source_closure_matches_runner_literal_without_importing_runner():
    tree = ast.parse((ROOT / "scripts/study_dialogue_copy.py").read_text())
    declared = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SOURCES" for target in node.targets))
    assert len(declared) == 13 and set(declared) == reporter.REQUIRED_SOURCES


def gate_fixture():
    counts = {"readout": 122, "scalar": 125, "selective": 126, "selective_no_lexical": 110, "candidate_gru": 126}
    rows = {}
    for method in reporter.METHODS:
        for seed in reporter.SEEDS:
            metric = {"strata": {s: {"count": 200, "correct": counts[method]} for s in reporter.STRATA},
                      "micro": {"nll": .4}, "revision": {"count": 100, "correct": 60}}
            rows[f"{method}-{seed}"] = {"metrics": {p: copy.deepcopy(metric) for p in reporter.PANELS}}
    literal = {"strata": {s: {"count": 200, "correct": 120} for s in reporter.STRATA},
               "revision": {"count": 100, "correct": 61}}
    references = {"literal": {p: copy.deepcopy(literal) for p in reporter.PANELS}}
    return rows, references


def checks(rows, references):
    return {c["name"]: c for c in reporter.criteria(rows, references)["checks"]}


def test_all_thirteen_exact_inclusive_boundaries():
    rows, refs = gate_fixture()
    result = reporter.criteria(rows, refs)
    assert result["passed"] and result["checks_total"] == result["checks_passed"] == 13
    assert checks(rows, refs)["seen/macro_gain_over_literal_3pp"]["left"] == .63
    assert checks(rows, refs)["seen/macro_gain_over_scalar_half_pp"]["right"] == .63
    assert checks(rows, refs)["seen/revision_within_1pp_literal"]["right"] == .60


@pytest.mark.parametrize("name", ["macro_gain_over_literal_3pp", "macro_gain_over_scalar_half_pp",
                                  "revision_within_1pp_literal"])
def test_one_count_below_accuracy_boundary_fails(name):
    rows, refs = gate_fixture()
    target = rows["selective-4101"]["metrics"]["seen"]
    if name.startswith("revision"):
        target["revision"]["correct"] -= 1
    else:
        target["strata"]["changed"]["correct"] -= 1
    assert not checks(rows, refs)["seen/" + name]["passed"]


def test_family_mean_does_not_replace_two_strict_pairs():
    rows, refs = gate_fixture()
    for seed, correct in zip(reporter.SEEDS, (125, 125, 128), strict=True):
        for stratum in reporter.STRATA:
            rows[f"selective-{seed}"]["metrics"]["seen"]["strata"][stratum]["correct"] = correct
    result = checks(rows, refs)
    assert result["seen/macro_gain_over_literal_3pp"]["passed"]
    assert not result["seen/strict_macro_win_at_least_two_pairs"]["passed"]


def test_best_conventional_is_family_mean_and_ablation_never_selected():
    rows, refs = gate_fixture()
    for seed in reporter.SEEDS:
        for stratum in reporter.STRATA:
            rows[f"selective_no_lexical-{seed}"]["metrics"]["seen"]["strata"][stratum]["correct"] = 200
    assert reporter.criteria(rows, refs)["passed"]
    for stratum in reporter.STRATA:
        rows["readout-4101"]["metrics"]["seen"]["strata"][stratum]["correct"] = 200
    result = checks(rows, refs)["seen/macro_nonworse_best_conventional"]
    assert not result["passed"] and result["strongest"] == "readout"


@pytest.mark.parametrize("kind", ["zero_bin", "zero_revision", "infinite_nll", "nan_nll", "missing_fit", "missing_literal"])
def test_missing_required_evidence_cannot_pass(kind):
    rows, refs = gate_fixture()
    if kind == "zero_bin":
        rows["scalar-4101"]["metrics"]["unseen"]["strata"]["changed"]["count"] = 0
    elif kind == "zero_revision":
        refs["literal"]["unseen"]["revision"]["count"] = 0
    elif kind in ("infinite_nll", "nan_nll"):
        rows["selective-4101"]["metrics"]["seen"]["micro"]["nll"] = None if kind == "infinite_nll" else float("nan")
    elif kind == "missing_fit":
        del rows["selective-4101"]
    else:
        refs.clear()
    result = reporter.criteria(rows, refs)
    assert not result["passed"] and result["checks_total"] == 13


def public_packet():
    queries, dialogs = [], []
    for i, unseen in enumerate((False, True)):
        queries.append({"split": "dev", "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:None", "value:other"],
                        "candidates": [0, 1, 2, 3]})
        dialogs.append({"id": f"dialog-{i}", "turns": [0] * 5, "queries": [
            {"query": i, "time": t, "label": label, "bin": name, "unseen": unseen, "dontcare": label == 1}
            for t, (label, name) in enumerate(zip((0, 2, 2, 3, 1),
                ("unmentioned_retention", "first_assignment", "assigned_retention", "revision", "revision"), strict=True))]})
    return {"queries": queries, "cohorts": {"dev": dialogs, "train": [copy.deepcopy(dialogs[0])]}}


@pytest.fixture
def tree(tmp_path):
    root, run, packet, lexical = (tmp_path / name for name in ("root", "run", "packet", "lexical"))
    for path in (root, run, packet, lexical):
        path.mkdir()
    for name in reporter.REQUIRED_SOURCES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        actual = ROOT / name
        target.write_bytes(actual.read_bytes() if actual.exists() else b"synthetic source fixture")
    data = public_packet()
    base.write(packet / "packet.json", data)
    for name in ("encoder-plan.json", "encoder-source.py", "features.npy"):
        (packet / name).write_bytes(b"synthetic opaque payload, no encoder")
    base.write(packet / "completed.json", {"status": "completed", "cohorts": {"dev": {"queries": 10, "dialogues": 2}},
        "wall_seconds": 1., "unique_texts": 4, "encoder_sequences": 4, "encoder_tokens_with_special": 12,
        "files": {p.name: base.sha(p) for p in packet.iterdir()}})
    for name in ("started.json", "index.json", "lexical.npy"):
        (lexical / name).write_bytes(b"synthetic lexical payload")
    base.write(lexical / "completed.json", {"status": "completed", "packet_completed_sha256": base.sha(packet / "completed.json"),
        "source_sha256": {reporter.BASE_PATH: reporter.BASE_SHA256},
        "files": {p.name: base.sha(p) for p in lexical.iterdir()}})
    plan = {"study": "dialogue-copy-v1", "runtime": {"python": "synthetic"},
            "config": {"methods": list(reporter.METHODS), "seeds": list(reporter.SEEDS), "epochs": 2, "batch_size": 32,
                       "projection_dim": 64, "hidden_dim": 64, "gru_width": 16},
            "practical_checks": reporter.PRACTICAL_CHECKS, "packet_completed_sha256": base.sha(packet / "completed.json"),
            "lexical_completed_sha256": base.sha(lexical / "completed.json"),
            "source_sha256": {name: base.sha(root / name) for name in reporter.REQUIRED_SOURCES}}
    base.write(run / "plan.json", plan)
    expected, _ = base.expected_records(data)
    p = np.zeros((10, 12), np.float32)
    p[:, :4] = .1
    p[np.arange(10), expected["labels"]] = .7
    arrays = {**expected, "probabilities": p, "choice": p.argmax(-1)}
    records = []
    for seed in reporter.SEEDS:
        for method in reporter.METHODS:
            folder = run / "fits" / f"{method}-{seed}"
            folder.mkdir(parents=True)
            (folder / "weights.pt").write_bytes(b"opaque weight bytes, no Torch deserialization")
            np.savez_compressed(folder / "dev-predictions.npz", **arrays)
            record = {"method": method, "seed": seed, "status": "completed", "epochs": 2,
                      "updates": 2, "training_queries": 10, "training_losses": [.5, .4],
                      "training_actor_shapes": {"real_turns": 10, "padded_turn_positions": 10,
                          "padded_query_positions": 10, "padded_candidate_positions": 40, "scored_queries": 10,
                          "real_question_steps": 10, "real_candidate_steps": 40},
                      "parameters": 100, "parameters_with_final_gradient": 90,
                      "configuration": {"class": "DialogueCopyMemory", "method": method, "parameters": 100,
                                        "projection_dim": 64, "hidden_dim": 64, "gru_width": 16},
                      "initial_tensors_sha256": ("b" if method == "candidate_gru" else "a") * 64,
                      "train_wall_seconds": 1., "evaluation": {"queries": 10, "wall_seconds": .2, "diagnostic": 2},
                      "weights_sha256": base.sha(folder / "weights.pt"),
                      "predictions_sha256": base.sha(folder / "dev-predictions.npz")}
            base.write(folder / "completed.json", record)
            records.append(record)
    np.savez_compressed(run / "references.npz", **expected, pred_none=np.zeros(10, dtype=np.int64),
                        pred_literal=np.full(10, 2, dtype=np.int64))
    completed = {"status": "completed", "fit_count": 15, "fits": records, "external_model_api_calls": 0,
                 "wall_seconds": 20., "plan_sha256": base.sha(run / "plan.json"),
                 "references": {"file": "references.npz", "queries": 10, "wall_seconds": .01,
                                "sha256": base.sha(run / "references.npz")}}
    base.write(run / "completed.json", completed)
    return root, run, packet, lexical


def invoke(tree, out):
    root, run, packet, lexical = tree
    return reporter.report(run, packet, lexical, base.sha(run / "plan.json"), base.sha(run / "completed.json"), out, root=root)


def test_full_synthetic_report_preserves_failed_gate_references_and_hashes(tree, tmp_path):
    out = tmp_path / "report"
    receipt = invoke(tree, out)
    assert receipt["status"] == "completed" and not receipt["continuation_passed"]
    assert receipt["fit_count"] == 15 and len(receipt["execution_members"]) == 48
    assert receipt["neural_calls"] == receipt["encoder_calls"] == 0
    assert receipt["reused_metric_source_sha256"] == reporter.BASE_SHA256
    summary = base.read(out / "summary.json")
    assert len(summary["rows"]) == 15
    assert summary["references"]["literal"]["seen"]["revision"]["count"] == 2
    assert summary["families"]["selective"]["seen"]["micro"]["brier"] == pytest.approx(.12)
    assert summary["families"]["selective"]["seen"]["micro"]["nll"] == pytest.approx(-math_log_point7(), abs=1e-7)
    assert (out / "comparison.png").read_bytes().startswith(b"\x89PNG")
    for name, item in receipt["files"].items():
        assert base.sha(out / name) == item["sha256"]
    with pytest.raises(FileExistsError):
        invoke(tree, out)


def math_log_point7():
    return float(np.log(np.float32(.7)))


@pytest.mark.parametrize("kind", ["prediction", "weights", "lexical", "source", "missing_source", "extra_source", "criteria",
                                  "initial_pair", "configuration", "missing_configuration", "fit_order", "missing_fit",
                                  "extra_root_file", "extra_nested_file", "reference", "labels", "failure"])
def test_corruption_rejected_with_preserved_failure(tree, tmp_path, kind):
    root, run, _, lexical = tree
    folder = run / "fits/selective-4101"
    completion = base.read(run / "completed.json")
    plan = base.read(run / "plan.json")
    if kind in ("prediction", "weights"):
        (folder / ("dev-predictions.npz" if kind == "prediction" else "weights.pt")).write_bytes(b"tampered")
    elif kind == "lexical":
        (lexical / "lexical.npy").write_bytes(b"tampered")
    elif kind == "source":
        (root / "scripts/report_dialogue_copy.py").write_text("tampered")
    elif kind in ("missing_source", "extra_source", "criteria"):
        if kind == "missing_source":
            del plan["source_sha256"][reporter.BASE_PATH]
        elif kind == "extra_source":
            (root / "foreign.py").write_bytes(b"new undeclared source")
            plan["source_sha256"]["foreign.py"] = base.sha(root / "foreign.py")
        else:
            plan["practical_checks"]["macro_gain_over_literal"] = 0
        (run / "plan.json").write_text(json.dumps(plan))
        completion["plan_sha256"] = base.sha(run / "plan.json")
    elif kind in ("initial_pair", "configuration", "missing_configuration"):
        own = base.read(folder / "completed.json")
        if kind == "initial_pair":
            own["initial_tensors_sha256"] = "c" * 64
        elif kind == "missing_configuration":
            del own["configuration"]
        else:
            own["configuration"]["method"] = "scalar"
        (folder / "completed.json").write_text(json.dumps(own))
        completion["fits"][2] = own
    elif kind == "fit_order":
        completion["fits"][0], completion["fits"][1] = completion["fits"][1], completion["fits"][0]
    elif kind == "missing_fit":
        (run / "fits/candidate_gru-4103").rename(run / "fits/foreign")
    elif kind in ("extra_root_file", "extra_nested_file"):
        target = run / ("partial.npz" if kind == "extra_root_file" else "unexpected/partial.npz")
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"retained foreign partial output")
    elif kind == "reference":
        (run / "references.npz").write_bytes(b"tampered")
    elif kind == "labels":
        with np.load(folder / "dev-predictions.npz", allow_pickle=False) as saved:
            a = {key: saved[key] for key in saved.files}
        a["labels"][0] = 2
        np.savez_compressed(folder / "dev-predictions.npz", **a)
        own = base.read(folder / "completed.json")
        own["predictions_sha256"] = base.sha(folder / "dev-predictions.npz")
        (folder / "completed.json").write_text(json.dumps(own))
        completion["fits"][2] = own
    else:
        (run / "failed.json").write_text("{}")
    (run / "completed.json").write_text(json.dumps(completion))
    out = tmp_path / "failed-report"
    with pytest.raises(ValueError):
        invoke(tree, out)
    assert (out / "failed.json").exists() and not (out / "receipt.json").exists()
