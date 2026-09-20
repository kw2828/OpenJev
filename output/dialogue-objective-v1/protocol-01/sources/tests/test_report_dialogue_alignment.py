"""Small saved-array and behavioral-rule fixtures; no corpus or model calls."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import report_dialogue_alignment as report


def rows_fixture():
    ids = ("reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:true", "value:false")
    rows = []
    for i, (old, new, service, dialogue, held) in enumerate([
            (0, 2, "s1", "d1", True), (0, 1, "s1", "d1", True),
            (0, 0, "s1", "d2", True), (2, 2, "s1", "d2", True),
            (0, 2, "s2", "d3", True), (0, 0, "s2", "d3", True),
            (2, 0, "s3", "d4", False)]):
        transition = (report.base.BINS[0] if old == new == 0 else report.base.BINS[1] if old == new else
                      report.base.BINS[2] if old == 0 else report.base.BINS[4] if new == 0 else report.base.BINS[3])
        rows.append({"row_index": i, "split": "train", "admission": "admitted", "candidate_count": 4,
                     "candidate_types": [0, 1, 2, 3], "current_label_index": new, "previous_current_index": old,
                     "current_candidate_id": ids[new], "previous_candidate_id": ids[old],
                     "current_value_group": report.base.VALUES[new], "derived_bin": transition,
                     "heldout_service": held, "service": service, "dialogue_id": dialogue, "query_id": service+"/q",
                     "query_index": 0})
    return rows


def packet(rows):
    values = np.full((len(rows), 4), .1)
    values[np.arange(len(rows)), [r["current_label_index"] for r in rows]] = .7
    logs = np.full((len(rows), 12), -np.inf, np.float32)
    logs[:, :4] = np.log(values)
    return {"row_indices": np.asarray([r["row_index"] for r in rows], np.int64), "log_probs": logs}


def references(rows):
    return {"row_indices": np.asarray([r["row_index"] for r in rows], np.int64),
            "previous_indices": np.asarray([r["previous_current_index"] for r in rows], np.int64),
            "literal_indices": np.asarray([r["current_label_index"] for r in rows], np.int64)}


def passing_fits():
    fits = {}
    for name in report.FITS:
        aligned = name.startswith("token_aligned-")
        fits[name] = {"decisions": {
            "accuracy": {"numerator": 52 if aligned else 50, "denominator": 100},
            "wrong_selected_branch": {"numerator": 18 if aligned else 20, "denominator": 100},
            **{key: {"numerator": 21 if aligned else 20, "denominator": 200} for key in
               ("retained_error", "true_false_positive_rate", "dontcare_false_positive_rate")}}}
    return fits


def test_exact_inclusive_22_checks_and_both_controls():
    got = report.continuation(passing_fits())
    assert got["passed"] and got["total_checks"] == got["checks_passed"] == 22
    fits = passing_fits()
    for seed in report.SEEDS: fits[f"token_mean-{seed}"]["decisions"]["accuracy"]["numerator"] = 51
    assert not report.continuation(fits)["passed"]


def test_paired_harm_cannot_hide_in_good_mean():
    fits = passing_fits()
    for seed, correct in zip(report.SEEDS, (49, 65, 65), strict=True):
        fits[f"token_aligned-{seed}"]["decisions"]["accuracy"]["numerator"] = correct
    got = report.continuation(fits)
    assert all(c["passed"] for c in got["checks"] if c["name"] == "mean_accuracy")
    assert not got["passed"]


@pytest.mark.parametrize("metric", ["accuracy", "wrong_selected_branch", "retained_error", "true_false_positive_rate", "dontcare_false_positive_rate"])
def test_empty_denominators_fail_closed(metric):
    fits = passing_fits()
    for fit in fits.values(): fit["decisions"][metric] = {"numerator": 0, "denominator": 0}
    assert not report.continuation(fits)["passed"]


def test_rare_false_positives_fail_even_with_behavioral_gains():
    fits = passing_fits()
    for seed in report.SEEDS: fits[f"token_aligned-{seed}"]["decisions"]["dontcare_false_positive_rate"]["numerator"] = 22
    got = report.continuation(fits)
    assert not got["passed"]
    assert all(c["passed"] for c in got["checks"] if c["name"] in ("mean_accuracy", "mean_wrong_selected_branch"))


def test_selected_branch_is_not_maximum_branch_mass():
    rows = rows_fixture()[:1]
    p = packet(rows)
    # Concrete has mass .60, but NONE is the largest individual candidate.
    p["log_probs"][0, :4] = np.log([.35, .05, .31, .29])
    _, choice, _ = report.base.predictions(rows, p)
    got = report.decision_vectors(rows, choice)
    assert got["wrong_selected_branch"].tolist() == [1]
    assert got["wrong_value"].tolist() == [0]
    got = report.decision_vectors(rows, np.asarray([3]))
    assert got["wrong_selected_branch"].tolist() == [0]
    assert got["wrong_value"].tolist() == [1]


def test_loss_decomposition_uses_branch_sum_and_keeps_extreme_logs():
    rows = rows_fixture()[:1]; p = packet(rows)
    p["log_probs"][0, :4] = np.log([.35, .05, .31, .29])
    got = report.branch_losses(rows, p)
    assert got["branch_nll"][0] == pytest.approx(-np.log(.6), abs=1e-7)
    assert got["within_branch_nll"][0] == pytest.approx(-np.log(.31/.6), abs=1e-7)
    p["log_probs"][0, :4] = [0., -1000., -2000., -1000.]
    got = report.branch_losses(rows, p)
    assert got["branch_nll"][0] == 1000. and got["within_branch_nll"][0] == 1000.


def test_microbatch_padding_and_schema_bytes_are_paid_locally():
    rows = rows_fixture()[:2]
    for row, length in zip(rows, (2, 5), strict=True): row["cache"] = {"token_start": 0, "token_stop": length}
    plain, n, padding = report.public_work(rows, "flat_stratum", {0: [2, 3, 4, 5]})
    token, _, _ = report.public_work(rows, "token_mean", {0: [2, 3, 4, 5]})
    assert n == 1 and padding == 0 and plain["padded_token_positions"] == 10
    assert token["schema_supported_pairwise_positions"] == 7*14
    assert token["schema_padded_pairwise_positions"] == 2*4*5*5
    assert token["schema_schema_float_input_bytes"] == 4*2*4*5*385
    assert "schema_padded_pairwise_positions" not in plain


def test_all_nine_full_groups_and_opposite_repair_harm_counts():
    rows = rows_fixture()
    packets = {name: packet(rows) for name in report.FITS}
    for seed in report.SEEDS:
        # Baseline misses row zero; aligned repairs it but breaks rows one and four.
        packets[f"flat_stratum-{seed}"]["log_probs"][0, :4] = np.log([.7, .1, .1, .1])
        packets[f"token_aligned-{seed}"]["log_probs"][[1, 4], :4] = np.log([.7, .1, .1, .1])
    result = report.aggregate(rows, packets, references(rows))
    assert len(result["fits"]) == 9 and result["groups_per_fit"] == 144
    pair = result["pairs"]["flat_stratum"]["6201"][report.PRIMARY]
    assert pair["wrong_to_correct"] == 1 and pair["correct_to_wrong"] == 2
    assert pair["aligned_minus_control"]["accuracy"]["row"] == pytest.approx(-1/3)
    assert set(result["references"]["literal"]["all/all"]) == {"rows", "services", "dialogues", "schema_queries", "accuracy"}


def test_equal_service_metric_differs_from_pooled_row():
    rows = rows_fixture()
    mask = report.base.group_masks(rows)[report.PRIMARY]
    group = report.base.layout(rows, mask)
    mean = report.base.means(np.asarray([1., 1., 0., 0., 10., 0., 0.]), group)
    assert mean["row"] == 4 and mean["equal_service"] == 5.5


def test_raw_log_nll_does_not_floor_underflow():
    rows = rows_fixture()[:1]; p = packet(rows)
    p["log_probs"][0, :4] = [0., -1000., -1000., -1000.]
    values, _, _ = report.base.predictions(rows, p)
    assert values["nll"].tolist() == [1000.] and values["brier"].tolist() == [2.]


def test_partial_fit_rejected_before_prediction_decode():
    rows = rows_fixture()
    with pytest.raises(ValueError, match="nine"):
        report.aggregate(rows, {next(iter(report.FITS)): object()}, {})
    with pytest.raises(ValueError, match="nine"):
        report.continuation({})


def test_row_join_and_mass_corruption_rejected():
    rows = rows_fixture(); p = packet(rows)
    p["row_indices"] = p["row_indices"][::-1]
    with pytest.raises(ValueError, match="identity"): report.base.predictions(rows, p)
    p = packet(rows); p["log_probs"][0, :4] = -.1
    with pytest.raises(ValueError, match="mass"): report.base.predictions(rows, p)
    rows = copy.deepcopy(rows); rows[0]["previous_candidate_id"] = "value:true"
    with pytest.raises(ValueError, match="Atomic"): report.base.validate_rows(rows)


def test_opaque_manifest_mutation_and_foreign_members_rejected(tmp_path):
    (tmp_path/"weights.pt").write_bytes(b"opaque weights, never deserialized")
    (tmp_path/"completed.json").write_text("{}")
    entry = {"sha256": report.digest(tmp_path/"weights.pt"), "bytes": (tmp_path/"weights.pt").stat().st_size}
    report.check_manifest(tmp_path, {"weights.pt": entry}, {"weights.pt"})
    (tmp_path/"weights.pt").write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash/size"): report.check_manifest(tmp_path, {"weights.pt": entry}, {"weights.pt"})
    (tmp_path/"foreign").write_text("unexpected")
    with pytest.raises(ValueError, match="membership"): report.check_manifest(tmp_path, {"weights.pt": entry}, {"weights.pt"})


def test_partial_authentication_fails_without_decoding_arrays(tmp_path, monkeypatch):
    report.write(tmp_path/"completed.json", {"status": "failed"})
    report.write(tmp_path/"plan.json", {})
    report.write(tmp_path/"started.json", {})
    def forbidden(*_args, **_kwargs): raise AssertionError("Prediction decoding before complete membership")
    monkeypatch.setattr(report.np, "load", forbidden)
    with pytest.raises(ValueError, match="completed campaign"):
        report.authenticate_run(tmp_path, report.digest(tmp_path/"completed.json"))


def test_complete_saved_only_render_and_exclusive_output(tmp_path, monkeypatch):
    """Exercise saved scoring/output, with authentication isolated from fixtures."""
    run, out = tmp_path/"run", tmp_path/"out"; run.mkdir()
    rows = rows_fixture()
    for name in report.FIT_ORDER:
        path = run/"fits"/name; path.mkdir(parents=True)
        np.savez(path/"predictions.npz", **packet(rows))
    np.savez(run/"references.npz", **references(rows))
    files = {p.relative_to(run).as_posix(): {"sha256": report.digest(p), "bytes": p.stat().st_size}
             for p in run.rglob("*") if p.is_file()}
    done = {"files": files, "source_sha256": {}, "plan_sha256": "a"*64, "wall_seconds": 2.,
            "process_lifetime_peak_rss_bytes": 1024, "process_load_start": {}, "process_load_end": {}}
    report.write(run/"completed.json", done)
    records = {name: {"wall_seconds": .1, "training_wall_seconds": .07, "evaluation": {"wall_seconds": .02, "work": {}},
                     "counts": {}, "training_work": {}, "configuration": {"parameters": 1}} for name in report.FIT_ORDER}
    cached = {"wall_seconds": .1, "payload_bytes": 1, "work": {}}
    monkeypatch.setattr(report, "authenticate_run", lambda *_: (rows, done, {"split": {}, "objective": {}}, records, 1, cached))
    args = SimpleNamespace(run=run, run_sha256=report.digest(run/"completed.json"), out=out)
    assert report.execute(args) is False
    receipt = report.read(out/"receipt.json")
    assert receipt["status"] == "completed" and receipt["model_calls"] == 0 and receipt["technical_validity_passed"]
    assert set(receipt["files"]) == {"summary.json", "report.md"}
    assert report.read(out/"summary.json")["continuation"]["total_checks"] == 22
    assert "All nine" in (out/"report.md").read_text()
    with pytest.raises(FileExistsError): report.execute(args)


def test_failure_preserves_original_and_zero_quality_reads(tmp_path, monkeypatch):
    error = RuntimeError("authentication stop")
    def failed(*_args): raise error
    monkeypatch.setattr(report, "authenticate_run", failed)
    args = SimpleNamespace(run=tmp_path, run_sha256="b"*64, out=tmp_path/"out")
    with pytest.raises(RuntimeError, match="authentication stop"): report.execute(args)
    result = json.loads((args.out/"failed.json").read_text())
    assert result["model_calls"] == result["encoder_calls"] == 0 and result["no_retry"]
    assert not (args.out/"summary.json").exists()


def test_capacity_order_binding_uses_real_payload_schema_not_orders(tmp_path):
    paths = [tmp_path/name for name in ("capacity", "freeze", "run")]
    filename = "orders-6201.npy"
    expected = np.tile(np.asarray([2, 0, 1], np.int64), (20, 1))
    for path in paths:
        path.mkdir(); np.save(path/filename, expected, allow_pickle=False)
    pin = report.digest(paths[0]/filename)
    # This is the actual capacity shape. There intentionally is no orders key.
    capacity = {"payloads": {filename: {"sha256": pin, "bytes": (paths[0]/filename).stat().st_size}}}
    item = {"file": filename, "sha256": pin, "shape": [20, 3]}
    np.testing.assert_array_equal(report.copied_order(item, capacity, 6201, paths, 3), expected)
    mutated = expected.copy(); mutated[0] = [0, 0, 2]
    np.save(paths[2]/filename, mutated, allow_pickle=False)
    with pytest.raises(ValueError, match="Copied order"): report.copied_order(item, capacity, 6201, paths, 3)
