"""Small synthetic saved-reader checks; no empirical files or model forwards."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spatial_study_audit_test", ROOT/"scripts/audit_otto_spatial_study.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
H = A.load(ROOT/A.HELPER, "spatial_study_audit_independent_helper_test")


def exported(kind):
    tensors = {name: np.ones(shape, np.float32) for name, shape in A.shapes_for(kind).items()}
    for name, value in tensors.items():
        if "bias" in name or name == "readout_weight_1":
            value.fill(0)
    return {**tensors, "version": np.array("otto-spatial-value-v1"), "kind": np.array(kind),
            "input_dim": np.array(11028, np.int64), "c0": np.array(.4, np.float32)}


@pytest.mark.parametrize(("kind", "count"), [("spatial", 737), ("neighbor_free", 737), ("cnn", 801),
                                               ("dense128", 1411841), ("statistics", 225)])
def test_initial_checkpoint_exact_shapes_and_common_constant_function(kind, count):
    saved = exported(kind)
    assert A.checkpoint(saved, kind, float(np.float32(.4)), np, initial=True) == count
    saved["readout_weight_1"][0, 0] = 1
    with pytest.raises(ValueError, match="zero final"):
        A.checkpoint(saved, kind, float(np.float32(.4)), np, initial=True)
    assert A.checkpoint(saved, kind, float(np.float32(.4)), np) == count


def test_centering_preserves_asymmetric_subfloor_mass_at_every_corner():
    beliefs = np.zeros((5, 53, 53), np.float64)
    beliefs[:, 1, 52] = 1e-20
    beliefs[4] = 0
    positions = np.array([[0, 0], [0, 52], [52, 0], [52, 52], [26, 26]], np.int64)
    before = beliefs.copy()
    centered = A.center_batch(beliefs, positions, np)
    for i, (x, y) in enumerate(positions):
        assert np.array_equal(centered[i, 52-x:105-x, 52-y:105-y], beliefs[i])
        assert np.count_nonzero(centered[i]) == (i < 4)
    assert np.array_equal(beliefs, before)
    with pytest.raises(ValueError, match="float64"):
        A.center_batch(beliefs.astype(np.float32), positions, np)


def test_saved_parity_uses_reference_relative_term_and_preserves_signed_values():
    reference = np.array([0., -1e-300, -3.], np.float64)
    assert A.compare_values(reference.copy(), reference, np) == 0
    actual = reference.copy()
    actual[0] = 1e-8
    assert A.compare_values(actual, reference, np) == 1e-8
    actual[0] = np.nextafter(1e-8, np.inf)
    with pytest.raises(ValueError, match="parity"):
        A.compare_values(actual, reference, np)


def test_epoch_tail_is_row_weighted_and_all_three_training_operations_are_required():
    order = np.arange(129, dtype=np.int64)[::-1].copy()
    digest = hashlib.sha256(order.tobytes()).hexdigest()
    order_row = {"fit_id": "spatial@10101", "epoch": 1, "rows": 129, "order": order.tolist(), "sha256": digest}
    rows = []
    for batch, (size, loss) in enumerate(((128, 1.), (1, 130.))):
        indices = order[batch*128:batch*128+size]
        rows.append({"fit_id": "spatial@10101", "epoch": 1, "batch": batch, "rows": size,
                     "batch_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                     "loss": loss, "gradient_norm": 0., "update_index": batch+1})
    curve = {"fit_id": "spatial@10101", "epoch": 1, "rows": 129, "updates": 2,
             "training_mse_normalized": 2., "order_sha256": digest,
             "scope": "row-weighted mean of pre-update minibatch losses; not final-checkpoint MSE"}

    class Work:
        def __init__(self):
            self.taken = []

        def take(self, channel, context):
            self.taken.append((channel, context.copy()))
            return .125

    work = Work()
    assert A.epoch_records("spatial@10101", 1, order, order_row, iter(rows), curve, work, np, H) == .75
    assert [channel for channel, _ in work.taken] == ["training_forward", "backward", "optimizer_update"]*2
    rows[-1]["batch_indices_sha256"] = "wrong"
    with pytest.raises(ValueError, match="batch identities"):
        A.epoch_records("spatial@10101", 1, order, order_row, iter(rows), curve, Work(), np, H)


def test_exact_full_workload_and_payload_closure():
    assert len(A.payload_names()) == 56
    assert sum(A.COUNTS.values()) == 171090
    assert A.COUNTS["training_forward"] == A.COUNTS["backward"] == A.COUNTS["optimizer_update"] == 15*80*44
    assert A.COUNTS["numpy_prediction"] == A.COUNTS["torch_prediction"] == 15*(350+70)
    assert all(f"predictions-{kind}-{seed}.npz" in A.payload_names() for kind in A.KINDS for seed in A.SEEDS)


def test_audit_rejects_unbounded_threads_before_numpy_model_import(monkeypatch):
    for name, value in A.THREADS.items():
        monkeypatch.setenv(name, value)
    A.require_threads()
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    with pytest.raises(ValueError, match="before NumPy import"):
        A.require_threads()


def test_family_means_retain_every_seed_and_no_scalar_gate():
    fits = []
    for i, seed in enumerate(A.SEEDS):
        for kind in A.KINDS:
            panel = {"rows": 1, "mse_normalized": float(i), "mae_physical": float(2*i),
                     "negative_predictions": i, "minimum_normalized": -float(i), "maximum_normalized": float(i+1)}
            fits.append({"fit_id": f"{kind}@{seed}", "kind": kind, "metrics": {"train": panel, "valid": panel},
                         "fit_seconds": 1., "diagnostic_seconds": .25})
    summary = A.aggregate(fits)
    assert all(group["valid"]["mse_normalized"] == 1. for group in summary["family_means"].values())
    assert len(summary["metrics"]) == 15 and summary["scalar_admission_gate"] is None
    assert summary["alias_floor"] is None and summary["learned_architecture_advantage_established"] is False
    with pytest.raises(ValueError, match="fifteen"):
        A.aggregate(fits[:-1])


def test_interrupted_saved_replay_keeps_pending_attempt_without_retry(tmp_path):
    audit = A.Audit(SimpleNamespace(output=tmp_path))
    audit.check = lambda: None
    audit.clock = SimpleNamespace(now_ns=lambda: 100)

    class FailingReadout:
        @staticmethod
        def normalized(*_):
            raise RuntimeError("synthetic interruption")

    context = {"fit_id": "spatial@10101", "split": "train", "offset": 0}
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        audit.predict(FailingReadout(), np.zeros((1, 105, 105)), np.zeros((1, 2), np.int64), np.ones(1), context)
    assert audit.readout_attempts == 1 and audit.readout_returns == audit.readout_rows == 0
    assert audit.pending == {"id": 1, **context, "rows": 1}
    records = [json.loads(line) for line in (tmp_path/"readouts.jsonl").read_text().splitlines()]
    assert records == [{"id": 1, **context, "rows": 1, "event": "attempt"}]
    with pytest.raises(ValueError, match="allocation"):
        audit.predict(FailingReadout(), np.zeros((1, 105, 105)), np.zeros((1, 2), np.int64), np.ones(1), context)
