"""Handbuilt public layouts and saved-choice arithmetic; no corpus or models."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("copy_v2_error_diagnostic", Path(__file__).with_name("analyze.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    old = audit.historical()
    values = [[None, None, " True ", "false"], [None, None, "red", "blue"],
              [None, None, "FALSE", "true"]]
    queries = [{"split": "dev", "candidate_values": v,
                "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE"] + ["value:" + s for s in v[2:]]}
               for v in values]
    dialogs = []
    entries, blocks, offset = [], [], 0
    specifications = [("a", [0, 1], [[2, 2], [3, 2], [1, 3], [1, 3]], False,
                       [[2, 2], [3, 2], [3, 3], [3, 3]]),
                      ("b", [2], [[0], [3], [2], [1]], True, [[0], [3], [2], [2]])]
    for name, qids, labels, unseen, literals in specifications:
        rows, previous = [], {}
        block = np.zeros((4, len(qids), 4, 10), np.float32)
        block[:, :, 0, 6] = 1
        block[:, :, 1, 7] = 1
        for t, golds in enumerate(labels):
            for column, (q, y) in enumerate(zip(qids, golds, strict=True)):
                prior = previous.get(q, 0)
                kind = ("unmentioned_retention" if prior == y == 0 else "assigned_retention" if prior == y
                        else "first_assignment" if prior == 0 else "clear" if y == 0 else "revision")
                rows.append({"query": q, "time": t, "label": y, "bin": kind, "unseen": unseen})
                previous[q] = y
                block[t, column, literals[t][column], 4] = 1
                block[t, column, 0 if t == 0 else literals[t-1][column], 5] = 1
                if y >= 2:
                    block[t, column, y, 0] = block[t, column, y, 2] = 1
                    if q in (0, 2):
                        cue = 8 if values[q][y].strip().casefold() == "true" else 9
                        block[t, column, y, cue] = 1
        dialogs.append({"id": name, "turns": [0, 1, 2, 3], "queries": rows})
        entries.append({"id": name, "query_ids": qids, "offset": offset, "shape": list(block.shape)})
        blocks.append(block.ravel())
        offset += block.size
    packet = {"queries": queries, "cohorts": {"train": [], "dev": dialogs}}
    index = {"features": list(old.LEXICAL_FEATURES), "cohorts": {"train": [], "dev": entries}}
    lexical = np.concatenate(blocks)
    truth, evidence = old.extract(packet, index, lexical)
    return old, packet, index, lexical, truth, evidence


def test_boolean_ontology_not_candidate_index_and_dontcare_is_separate():
    old, packet, _, _, truth, evidence = fixture()
    masks = audit.diagnostic_groups(packet, truth, evidence, old)
    for panel in audit.PANELS:
        assert masks[panel + "/assigned_boolean"].sum() == 2
        assert masks[panel + "/assigned_boolean/true"].sum() == 1
        assert masks[panel + "/assigned_boolean/false"].sum() == 1
        assert not np.any(masks[panel + "/assigned_boolean"] & masks[panel + "/label/dontcare"])
    assert truth["labels"][masks["seen/assigned_boolean/true"]].tolist() == [2]
    assert truth["labels"][masks["unseen/assigned_boolean/true"]].tolist() == [3]
    assert masks["seen/label/dontcare"].sum() == 2
    assert masks["unseen/label/dontcare"].sum() == 1


def test_wrong_revision_partition_prior_prediction_and_literal_difference():
    old, packet, _, _, truth, evidence = fixture()
    masks = audit.diagnostic_groups(packet, truth, evidence, old)
    choice = truth["labels"].copy()
    # Seen bool revision true->false is stale after a correct prior.
    choice[2] = 2
    # Next bool revision false->DONTCARE is stale after the already-wrong prior.
    choice[4] = 3
    # Color red->blue is an unrelated wrong value, not stale.
    choice[5] = 0
    row = audit.measure(choice, masks["seen/bin/revision"], truth, evidence, old)
    assert row["count"] == row["wrong"] == 3
    assert row["revision_stale_prior_gold"] == 2
    assert row["revision_other_wrong"] == 1
    assert row["revision_stale_after_prior_correct"] == 1
    assert row["revision_stale_after_prior_wrong"] == 1
    assert row["literal_correct"] == row["literal_only"] == 2
    assert row["both_wrong"] == 1 and row["neural_only"] == 0
    assert row["accuracy_difference_from_literal_pp"] == pytest.approx(-200 / 3)


def test_first_assigned_prediction_is_not_a_revision_or_stale_carry():
    old, packet, _, _, truth, evidence = fixture()
    choice = truth["labels"].copy()
    choice[0] = 0
    row = audit.measure(choice, audit.diagnostic_groups(packet, truth, evidence, old)["seen/all"], truth, evidence, old)
    assert row["wrong"] == 1 and row["revision_stale_prior_gold"] == 0
    assert evidence["previous_record"][0] == -1
    assert evidence["previous_record"][8] == -1  # New dialogue resets the scored-prefix mapping.


def test_nonadjacent_scored_frames_are_not_counted_as_adjacent():
    old, packet, _, _, truth, evidence = fixture()
    evidence = {k: v.copy() for k, v in evidence.items()}
    evidence["gap"][2] = 2
    choice = truth["labels"].copy()
    choice[2] = 2
    row = audit.measure(choice, audit.diagnostic_groups(packet, truth, evidence, old)["seen/bin/revision"], truth, evidence, old)
    assert row["revision_stale_nonadjacent_public_step"] == 1
    assert row["revision_stale_adjacent_public_step"] == 0


def test_zero_support_is_null_and_three_fit_pooling_keeps_exact_rows():
    old, packet, _, _, truth, evidence = fixture()
    masks = audit.diagnostic_groups(packet, truth, evidence, old)
    masks["empty"] = np.zeros(len(truth["labels"]), bool)
    rows = {}
    for method in audit.METHODS:
        for seed in audit.SEEDS:
            choice = truth["labels"].copy()
            if seed == audit.SEEDS[0]:
                choice[0] = 0
            rows[f"{method}-{seed}"] = {k: audit.measure(choice, m, truth, evidence, old) for k, m in masks.items()}
    family = audit.pool(rows, masks)["scalar"]
    assert family["seen/assigned_boolean"]["count_per_fit"] == 2
    assert family["seen/assigned_boolean"]["correct"] == pytest.approx(5 / 3)
    assert family["seen/assigned_boolean"]["accuracy"] == pytest.approx(5 / 6)
    assert rows["scalar-4101"]["seen/assigned_boolean"]["correct"] == 1
    assert family["empty"]["accuracy"] is family["empty"]["accuracy_difference_from_literal_pp"] is None


@pytest.mark.parametrize("mutation", ["offset", "bin", "order", "literal"])
def test_historical_public_alignment_checks_are_retained(mutation):
    old, packet, index, lexical, _, _ = fixture()
    if mutation == "offset":
        index["cohorts"]["dev"][0]["offset"] = 1
    elif mutation == "bin":
        packet["cohorts"]["dev"][0]["queries"][2]["bin"] = "first_assignment"
    elif mutation == "order":
        index["features"][0], index["features"][1] = index["features"][1], index["features"][0]
    else:
        lexical[4] = 1  # Add NONE alongside the true literal.
    with pytest.raises(ValueError):
        old.extract(packet, index, lexical)


def saved_arrays(truth):
    arrays = {k: v.copy() for k, v in truth.items()}
    p = np.eye(12, dtype=np.float32)[truth["labels"]]
    arrays.update(probabilities=p, choice=p.argmax(1))
    return arrays


@pytest.mark.parametrize("mutation", [None, "cohort", "choice", "mass", "padding", "nan"])
def test_saved_choices_and_probability_integrity(tmp_path, mutation):
    _, _, _, _, truth, evidence = fixture()
    arrays = saved_arrays(truth)
    if mutation == "cohort":
        arrays["labels"][0] = 3
    elif mutation == "choice":
        arrays["choice"][0] = 3
    elif mutation == "mass":
        arrays["probabilities"][0, 2] = .9
    elif mutation == "padding":
        arrays["probabilities"][0, 2] = 0
        arrays["probabilities"][0, 11] = 1
        arrays["choice"][0] = 11
    elif mutation == "nan":
        arrays["probabilities"][0, 2] = np.nan
    path = tmp_path / "predictions.npz"
    np.savez(path, **arrays)
    if mutation is None:
        np.testing.assert_array_equal(audit.validate_saved(path, truth, evidence), truth["labels"])
    else:
        with pytest.raises(ValueError):
            audit.validate_saved(path, truth, evidence)


def test_literal_reference_reconstructs_from_saved_public_feature(tmp_path):
    _, _, _, _, truth, evidence = fixture()
    path = tmp_path / "references.npz"
    np.savez(path, **truth, pred_none=np.zeros(len(truth["labels"]), int), pred_literal=evidence["literal"])
    np.testing.assert_array_equal(audit.validate_saved(path, truth, evidence, reference=True), evidence["literal"])
    bad = evidence["literal"].copy()
    bad[0] = 0
    np.savez(path, **truth, pred_none=np.zeros(len(truth["labels"]), int), pred_literal=bad)
    with pytest.raises(ValueError, match="Literal/reference"):
        audit.validate_saved(path, truth, evidence, reference=True)


def test_exact_execution_membership_rejects_failed_or_extra_payload(tmp_path):
    for method in audit.METHODS:
        for seed in audit.SEEDS:
            folder = tmp_path / "fits" / f"{method}-{seed}"
            folder.mkdir(parents=True)
            for name in ("completed.json", "weights.pt", "dev-predictions.npz", "batches.jsonl"):
                (folder / name).write_text("")
    for name in ("plan.json", "started.json", "completed.json", "references.npz"):
        (tmp_path / name).write_text("")
    assert len(audit.execution_members(tmp_path)) == 64
    (tmp_path / "failed.json").write_text("{}")
    with pytest.raises(ValueError, match="64-file"):
        audit.execution_members(tmp_path)


@pytest.mark.parametrize("fail_receipt", [False, True])
def test_failure_preserves_original_and_no_retry(tmp_path, monkeypatch, fail_receipt):
    args = SimpleNamespace(out=str(tmp_path), **{key: "0" * 64 for key in
        ("source_sha256", "tests_sha256", "plan_sha256", "completed_sha256", "report_receipt_sha256",
         "packet_completed_sha256", "lexical_completed_sha256")})
    original_write = audit.write

    def write(path, value):
        if fail_receipt and Path(path).name == "failed.json":
            raise OSError("secondary-write")
        return original_write(path, value)

    monkeypatch.setattr(audit, "write", write)
    with pytest.raises(ValueError, match="Input hash mismatch") as caught:
        audit.run(args)
    assert not (tmp_path / "receipt.json").exists()
    if fail_receipt:
        assert "secondary-write" in " ".join(caught.value.__notes__)
    else:
        failed = json.loads((tmp_path / "failed.json").read_text())
        assert failed["progress"]["fits_checked"] == []
    with pytest.raises(ValueError, match="no retry"):
        audit.run(args)


def test_source_pin_checked_without_executing_untrusted_script(tmp_path):
    path = tmp_path / audit.OLD_PATH
    path.parent.mkdir(parents=True)
    path.write_text("raise AssertionError('must-not-execute')")
    with pytest.raises(ValueError, match="Historical diagnostic source changed"):
        audit.historical(tmp_path)


def test_render_keeps_descriptive_scope_and_null_denominators():
    old, packet, _, _, truth, evidence = fixture()
    masks = audit.diagnostic_groups(packet, truth, evidence, old)
    rows = {f"{method}-{seed}": {k: audit.measure(truth["labels"], v, truth, evidence, old) for k, v in masks.items()}
            for method in audit.METHODS for seed in audit.SEEDS}
    result = {"families": audit.pool(rows, masks), "literal": {
        k: audit.measure(evidence["literal"], v, truth, evidence, old) for k, v in masks.items()},
        "limits": ["No causal claim."]}
    result = copy.deepcopy(result)
    result["families"]["scalar"]["seen/assigned_boolean/true"]["accuracy"] = None
    text = audit.render(result)
    assert "selects no new winner" in text and "No causal claim." in text
    assert "| seen | scalar |" in text and "| - |" in text
