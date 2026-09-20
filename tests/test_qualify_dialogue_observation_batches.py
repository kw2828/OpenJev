"""Artificial metadata, fake CPU encoder and synthetic targets only."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import qualify_dialogue_observation_batches as q


def metadata():
    ids = [f"train-{i:04}" for i in range(2017)]
    profiles = []
    metrics = set(q.SINGLE_METRICS) | set(q.BATCH_METRICS)
    for split, names in (("train", ids), ("dev", ["dev-0", "dev-1", "dev-2", "dev-3"])):
        for i, did in enumerate(names):
            work = dict.fromkeys(metrics, 1)
            for j, metric in enumerate(q.SINGLE_METRICS):
                if i == j % 3:
                    work[metric] = 10
            profiles.append({"split": split, "dialogue_id": did, "work": work})
    orders = {"dialogue_ids": ids, "orders": {str(seed): [list(range(shift, 2017)) + list(range(shift))]
              for seed, shift in ((6901, 0), (6902, 32), (6903, 64))}}
    by_id = {p["dialogue_id"]: p["work"] for p in profiles if p["split"] == "train"}
    maxima = {}
    for j, metric in enumerate(q.BATCH_METRICS):
        seed = 6901 + j % 3
        indices = orders["orders"][str(seed)][0][:32]
        dids = [ids[i] for i in indices]
        maxima[metric] = {"seed": seed, "epoch": 0, "batch_start": 0,
                          "dialogue_indices": indices, "dialogue_ids": dids,
                          "value": sum(by_id[d][metric] for d in dids)}
    return {"profiles": profiles}, {"maxima": maxima}, orders


def test_fixed_case_selection_and_complete_paid_counts():
    cases = q.select_cases(*metadata())
    assert len(cases) == 10 and q.expected_counts(cases) == q.EXPECTED
    assert [c["kind"] for c in cases] == ["batch"] * 3 + ["single"] * 3 + ["tail"] + ["eval"] * 3
    assert [c["seed"] for c in cases[:3]] == [6901, 6902, 6903]
    assert cases[6]["dialogue_ids"] == ["train-2016"]
    assert [c["dialogue_ids"] for c in cases[3:6]] == [["train-0000"], ["train-0001"], ["train-0002"]]


def test_case_selection_is_stable_to_metadata_order_and_ties():
    workloads, batches, orders = metadata()
    expected = q.select_cases(workloads, batches, orders)
    workloads["profiles"].reverse()
    batches["maxima"] = dict(reversed(list(batches["maxima"].items())))
    actual = q.select_cases(workloads, batches, orders)
    # Reasons retain declared metric order, not input mapping insertion order.
    for left, right in zip(expected, actual, strict=True):
        assert {k: v for k, v in left.items() if k != "maximizes"} == {k: v for k, v in right.items() if k != "maximizes"}
        if "maximizes" in left:
            assert sorted(left["maximizes"]) == sorted(right["maximizes"])


@pytest.mark.parametrize("mutation", ["missing_metric", "wrong_sum", "wrong_members", "duplicate_profile"])
def test_corrupt_metadata_rejected(mutation):
    workloads, batches, orders = metadata()
    record = batches["maxima"][q.BATCH_METRICS[0]]
    if mutation == "missing_metric":
        batches["maxima"].pop(q.BATCH_METRICS[0])
    elif mutation == "wrong_sum":
        record["value"] += 1
    elif mutation == "wrong_members":
        record["dialogue_ids"].reverse()
    else:
        workloads["profiles"].append(workloads["profiles"][0])
    with pytest.raises(ValueError):
        q.select_cases(workloads, batches, orders)


def test_synthetic_targets_use_only_positions_and_public_support():
    class PositionOnly(dict):
        def __getitem__(self, key):
            assert key in ("time", "query_position")
            return super().__getitem__(key)

    payload = {"turn_text_ids": [0, 1, 2], "candidate_ids": [["n", "d", "a"], ["n", "d", "b", "c"]]}
    positions = [PositionOnly(time=2, query_position=1, label_index=object(), bin=object()),
                 PositionOnly(time=0, query_position=0)]
    assert q.synthetic_rows(payload, positions, 7) == [
        {"time": 2, "query_position": 1, "label_index": 2, "stratum_index": 1},
        {"time": 0, "query_position": 0, "label_index": 1, "stratum_index": 1}]
    with pytest.raises(ValueError, match="Duplicate"):
        q.synthetic_rows(payload, positions * 2, 0)
    with pytest.raises(ValueError, match="coordinates"):
        q.synthetic_rows(payload, [{"time": True, "query_position": 0}], 0)


def save(path, value):
    path.write_text(json.dumps(value))


def prepared_fixture(tmp_path, monkeypatch):
    root, prepared = tmp_path / "root", tmp_path / "prepared"
    root.mkdir()
    prepared.mkdir()
    monkeypatch.setattr(q, "ROOT", root)
    sources = {}
    for i in range(39):
        name = f"source-{i}.py"
        (root / name).write_text(f"# artificial {i}\n")
        sources[name] = q.sha(root / name)
    asset = root / "asset.bin"
    asset.write_bytes(b"opaque fake model bytes, never loaded")
    for name in q.PREPARED_MEMBERS - {"plan.json"}:
        save(prepared / name, {})
    save(prepared / "source-pins.json", sources)
    plan = {"preparation_version": 2, "study": "dialogue-observation-learning-v1", "runtime": q.qualified.runtime(),
            "config": {"arms": q.ARMS}, "data_files": q.qualified.manifest(prepared),
            "source_sha256": sources, "input_sha256": {asset.name: q.sha(asset)},
            "model_files": {"fake": {"path": str(asset), "sha256": q.sha(asset)}}}
    save(prepared / "plan.json", plan)
    done = {"status": "completed", "phase": "prepare", "encoder_calls": 0,
            "model_weights_loaded": False, "official_test_opened": False,
            "plan_sha256": q.sha(prepared / "plan.json"), "files": q.qualified.manifest(prepared)}
    save(prepared / "completed.json", done)
    monkeypatch.setattr(q, "PREPARED_PLAN_PIN", q.sha(prepared / "plan.json"))
    monkeypatch.setattr(q, "PREPARED_PIN", q.sha(prepared / "completed.json"))
    return prepared, plan


def test_exact_prepared_auth_and_tamper_before_actor_decode(tmp_path, monkeypatch):
    prepared, plan = prepared_fixture(tmp_path, monkeypatch)
    assert q.authenticate_prepared(prepared, q.PREPARED_PIN) == plan
    (prepared / "actors-train.jsonl").write_text("not JSON or valid actor data")
    with pytest.raises(ValueError, match="payload identity"):
        q.authenticate_prepared(prepared, q.PREPARED_PIN)


@pytest.mark.parametrize("mutation", ["source", "extra", "pin", "runtime"])
def test_prepared_identity_runtime_and_closure_fail_closed(tmp_path, monkeypatch, mutation):
    prepared, _plan = prepared_fixture(tmp_path, monkeypatch)
    pin = q.PREPARED_PIN
    if mutation == "source":
        (q.ROOT / "source-0.py").write_text("changed")
    elif mutation == "extra":
        (prepared / "extra.txt").touch()
    elif mutation == "pin":
        pin = "0" * 64
    else:
        monkeypatch.setattr(q.qualified, "runtime", lambda: {"different": True})
    with pytest.raises(ValueError):
        q.authenticate_prepared(prepared, pin)


def test_attempt_failure_preserves_request_progress_and_exclusive_output(tmp_path):
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="deliberate"), q.attempt(out, "fake", {"pin": "external"}) as (_, progress, _):
        progress["completed_updates"] = 2
        raise ValueError("deliberate")
    failed = q.read(out / "failed.json")
    assert failed["request"] == {"pin": "external"}
    assert failed["progress"]["completed_updates"] == 2
    before = (out / "failed.json").read_bytes()
    with pytest.raises(FileExistsError), q.attempt(out, "fake", {}):
        pass
    assert (out / "failed.json").read_bytes() == before


def test_late_output_cap_demotes_completion(tmp_path, monkeypatch):
    monkeypatch.setitem(q.CAPS, "output_bytes", 4096)
    out = tmp_path / "capped"
    with pytest.raises(ValueError, match="Output cap"), q.attempt(out, "fake", {}) as (_, _, _check):
        q.write(out / "completed.json", {"status": "completed"})
        (out / "oversized").write_bytes(b"x" * 4097)
    assert (out / "invalid-completion.json").exists() and (out / "failed.json").exists()
    assert not (out / "completed.json").exists()


def test_run_auth_failure_precedes_neural_dispatch_and_preserves_attempt(tmp_path, monkeypatch):
    def fail(*_):
        raise ValueError("bad authenticated plan")
    monkeypatch.setattr(q, "authenticate_plan", fail)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="bad authenticated plan"):
        q.run(tmp_path / "missing", "external", out)
    assert q.read(out / "failed.json")["progress"]["operation"] == "authenticate"
    assert not (out / "events.jsonl").exists()


def test_tiny_fake_cpu_full_path_counts_parity_gradients_checkpoints_and_offsets(tmp_path, monkeypatch):
    import study_dialogue_copy_v2 as monitored
    import torch
    from torch import nn

    import openjev.research.dialogue_finetune_training as loss_module
    from openjev.research.dialogue_finetune_inputs import workload_profile
    from openjev.research.dialogue_trainable_encoder import encode_token_lists

    torch.set_num_threads(1)

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Module()
            self.embeddings.word_embeddings = nn.Embedding(30, 8)
            self.encoder = nn.Module()
            layer = nn.Module()
            layer.attention = nn.Module()
            layer.attention.self = nn.Module()
            layer.attention.self.query = nn.Linear(8, 8)
            self.encoder.layer = nn.ModuleList([layer])

        def to(self, *_args, **_kwargs):
            return self  # Explicit test-only fake MPS placement; every tensor is CPU.

        def forward(self, input_ids, attention_mask):
            x = self.embeddings.word_embeddings(input_ids)
            return SimpleNamespace(last_hidden_state=x + self.encoder.layer[0].attention.self.query(x).tanh())

    class Factory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeEncoder()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=Factory))
    original_memory = monitored.MonitoredCopyMemoryV2
    monkeypatch.setattr(monitored, "MonitoredCopyMemoryV2",
                        lambda method: original_memory(method, input_dim=8, projection_dim=4, hidden_dim=6))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _n: None)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    for name in ("synchronize", "empty_cache"):
        monkeypatch.setattr(torch.mps, name, lambda: None)
    for name in ("driver_allocated_memory", "current_allocated_memory"):
        monkeypatch.setattr(torch.mps, name, lambda: 0)

    root, prepared, cost = tmp_path / "root", tmp_path / "prepared", tmp_path / "cost"
    for directory in (root, prepared, cost):
        directory.mkdir()
    monkeypatch.setattr(q, "ROOT", root)
    (cost / "plan.json").write_text("artificial cost plan")
    base = {"tokens": [[1], [2], [3], [4], [5], [6]], "turn_text_ids": [0, 1],
            "query_text_ids": [2], "candidate_text_ids": [[3, 4, 5]], "query_ids": [0],
            "candidate_ids": [["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:blue"]],
            "original_feature_ids": list(range(6)), "lexical_shape": [2, 1, 3, 10]}
    actors, positions, profiles, index = [], [], [], []
    for i, (split, did) in enumerate((("train", "same"), ("train", "other"), ("dev", "same"))):
        actor = {**base, "split": split, "dialogue_id": did, "lexical_offset": i * 60}
        actors.append(actor)
        positions.append({"split": split, "dialogue_id": did, "rows": [
            {"time": 0, "query_position": 0, "label_index": "must not be used", "bin": "invalid"},
            {"time": 1, "query_position": 0, "label_index": "must not be used", "bin": "invalid"}]})
        if did == "other":
            positions[-1]["rows"].pop(0)  # Unequal eligible endpoint counts within the effective batch.
        profiles.append({"split": split, "dialogue_id": did,
                         "work": workload_profile({**actor, "lexical": np.zeros((2, 1, 3, 10), np.float32)})})
        index.append({"split": split, "dialogue_id": did, "offset": i * 60, "shape": [2, 1, 3, 10],
                      "scored_rows": len(positions[-1]["rows"])})
    for split in ("train", "dev"):
        for name, values in (("actors", actors), ("targets", positions)):
            (prepared / f"{name}-{split}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values if v["split"] == split))
    save(prepared / "index.json", index)
    save(prepared / "workloads.json", {"profiles": profiles})
    for kind in ("original", "numbers"):
        np.save(prepared / f"lexical-{kind}.npy", np.zeros(180, np.float32))
    special = {"cls_id": 28, "sep_id": 29, "pad_id": 0}
    torch.manual_seed(q.CONFIG["seed"])
    encoder = FakeEncoder()
    vectors, _ = encode_token_lists(encoder, base["tokens"], **special, trainable=False)
    reference = root / "runs/sgd-state-v1/features-02/features.npy"
    reference.parent.mkdir(parents=True)
    np.save(reference, vectors.numpy())
    cases = [{"case_id": "batch-0", "kind": "batch", "split": "train", "dialogue_ids": ["same", "other"], "iterations": 2},
             {"case_id": "tail", "kind": "tail", "split": "train", "dialogue_ids": ["same"], "iterations": 1},
             {"case_id": "dev-single", "kind": "eval", "split": "dev", "dialogue_ids": ["same"], "iterations": 2}]
    plan = {"prepared": str(prepared), "cases": cases}
    parent = {"snapshot": "fake-only", "tokenizer_ids": special, "loss_weights": [.5, 1.2, 3.7]}
    monkeypatch.setattr(q, "authenticate_plan", lambda *_: (plan, parent))
    monkeypatch.setattr(q, "EXPECTED", q.expected_counts(cases))
    observed_offsets = []
    original_synthetic = q.synthetic_rows

    def capture(payload, coords, offset):
        observed_offsets.append(offset)
        return original_synthetic(payload, coords, offset)
    monkeypatch.setattr(q, "synthetic_rows", capture)
    original_loss, denominators = loss_module.supervised_loss, []

    def capture_loss(logp, rows, weights, denominator):
        denominators.append((denominator, len(rows)))
        return original_loss(logp, rows, weights, denominator)
    monkeypatch.setattr(loss_module, "supervised_loss", capture_loss)
    out = tmp_path / "result"
    pin = q.run(cost, "synthetic", out)
    summary, done = q.read(out / "summary.json"), q.read(out / "completed.json")
    assert pin == q.sha(out / "completed.json")
    assert {k: summary["counts"][k] for k in q.EXPECTED} == q.EXPECTED
    assert len(summary["checkpoints"]) == 4
    assert summary["official_targets_used"] is False
    assert all(p["max_abs"] <= q.CONFIG["parity_tolerance"] for c in summary["cells"] for p in c["parity"])
    assert observed_offsets[:3] == [0, 0, 0]  # Selected coordinate validation.
    assert observed_offsets[3:19] == [0, 1, 2, 3] * 4
    assert denominators == [(3, 2), (3, 1)] * 8 + [(2, 2)] * 4
    assert {p.name for p in out.iterdir()} == set(done["files"]) | {"completed.json"}
    for cell in summary["cells"]:
        unchanged = cell["initial_sha256"]["encoder"] == cell["final_encoder_sha256"]
        assert unchanged == (cell["arm"].startswith("frozen") or cell["kind"] == "eval")
        if cell["kind"] == "eval":
            assert all(e["phase"] == "eval" and "encoder_gradient_squared_norms" not in e for e in cell["events"])
