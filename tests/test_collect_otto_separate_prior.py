"""Tiny fabricated collector routing/closure fixtures; no native or teacher calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research.otto_released_policy import PublicBeliefView, _packet
from openjev.research.otto_sampled_forecast_data import select_windows

SPEC = importlib.util.spec_from_file_location(
    "separate_prior_collector_tests", Path(__file__).resolve().parents[1] / "scripts/collect_otto_separate_prior.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def select(scores, allowed):
    subset = scores[list(allowed)]
    return allowed[int(np.flatnonzero(np.abs(subset - subset.min()) < 1e-10)[0])]


def test_complete_fixed_cohort_rotation_and_initial_hits():
    assert MODULE.VERSION == "otto-separate-prior-collection-v1"
    assert MODULE.CONFIGURATION["selection_seed_first"] == 286000001
    assert MODULE.FIRST == {"train": {"lambda3": 281000001, "lambda4": 282000001},
                            "valid": {"lambda3": 283000001, "lambda4": 284000001}}
    rows = MODULE.cohort()
    assert len(rows) == 90
    assert len(MODULE.cohort("train")) == 54 and len(MODULE.cohort("valid")) == 36
    assert [r["episode_index"] for r in rows] == list(range(90))
    assert len({r["episode_id"] for r in rows}) == 90
    for case_index in range(30):
        group = rows[case_index * 3:(case_index + 1) * 3]
        offset = case_index % 3
        assert [r["arm"] for r in group] == list(MODULE.ARMS[offset:] + MODULE.ARMS[:offset])
        assert len({(r["stage"], r["regime"], r["seed"], r["case"]) for r in group}) == 1
        assert all(r["initial_hit"] == 1 + r["case"] % 3 for r in group)
    for stage, counts in (("train", 9), ("valid", 6)):
        for regime, first in MODULE.FIRST[stage].items():
            assert {r["seed"] for r in rows if r["stage"] == stage and r["regime"] == regime} == set(range(first, first + counts))


def test_virtual_correction_including_initial_and_final_steps():
    assert [MODULE.virtual_query(n) for n in range(9)] == [0, 0, 0, 0, 4, 4, 4, 4, 8]
    assert MODULE.virtual_query(2187) == 2184
    for invalid in (True, -1, 2188, 1.0):
        with pytest.raises(ValueError, match="preaction step"):
            MODULE.virtual_query(invalid)


@pytest.mark.parametrize("arm,expected_queries", [("analytic", 0), ("neural", 9), ("period4_hold", 3)])
def test_exact_one_teacher_per_row_and_distinct_virtual_corrections(arm, expected_queries):
    cache, queries, annotations, corrections, calls = None, 0, 0, 0, []
    for step in range(9):
        def teacher(deployed, step=step):
            calls.append((step, deployed))
            return np.asarray([4, 3, 2, 1], dtype=np.float32)
        action, raw, cache, deployed, correction = MODULE.route_scores(
            arm, step, (0, 1, 2, 3), 1, teacher, select, cache)
        assert action == (1 if arm == "analytic" else 3)
        assert raw.dtype == np.float32
        queries += deployed; annotations += not deployed; corrections += correction
    assert len(calls) == 9 and queries == expected_queries and annotations == 9 - expected_queries
    assert corrections == 3


def test_skipped_held_action_selected_before_annotation_without_refresh():
    original = np.asarray([0, 4, 2, 3], dtype=np.float32)
    action, _, cache, queried, _ = MODULE.route_scores(
        "period4_hold", 0, (0, 1, 2, 3), 1, lambda _q: original, select, None)
    assert action == 0 and queried and not cache.flags.writeable
    original[:] = 99  # The held state owns its bytes.
    before, events = cache.tobytes(), []
    def held_select(scores, allowed):
        events.append("selected")
        return select(scores, allowed)
    def annotation(deployed):
        assert not deployed and events == ["selected"]
        events.append("annotation")
        return np.asarray([8, 0, 7, 6], dtype=np.float32)
    action, labels, after, queried, correction = MODULE.route_scores(
        "period4_hold", 1, (1, 2, 3), 1, annotation, held_select, cache)
    assert action == 2 and labels[1] == 0 and not queried and not correction
    assert after is cache and after.tobytes() == before
    # A later legal-mask change is applied to the complete retained four-score vector.
    assert MODULE.route_scores("period4_hold", 2, (0, 1), 1, annotation_without_order, select, cache)[0] == 0


def annotation_without_order(_deployed):
    return np.asarray([8, 0, 7, 6], dtype=np.float32)


def test_analytic_action_ignores_annotation_and_teacher_failures_propagate():
    action, _, cache, deployed, correction = MODULE.route_scores(
        "analytic", 0, (0, 1, 2, 3), 2, annotation_without_order, select, None)
    assert (action, cache, deployed, correction) == (2, None, False, True)
    error = OSError("annotation failed")
    def fail(_query):
        raise error
    with pytest.raises(OSError) as captured:
        MODULE.route_scores("analytic", 1, (0, 1, 2, 3), 2, fail, select, None)
    assert captured.value is error


def test_illegal_memory_mutation_and_missing_first_query_rejected():
    cache = np.arange(4, dtype=np.float32)
    def corrupt(_query):
        cache[0] += 1
        return np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="modify held memory"):
        MODULE.route_scores("period4_hold", 1, (0, 1, 2, 3), 1, corrupt, select, cache)
    with pytest.raises(ValueError, match="earlier correction"):
        MODULE.route_scores("period4_hold", 1, (0, 1, 2, 3), 1, annotation_without_order, select, None)


def test_offsets_keep_singletons_and_full_horizon_tail():
    assert MODULE.episode_offsets([1, 3, 4, 5, 2188]) == [0, 1, 4, 8, 13, 2201]
    for invalid in ([], [0], [True], [2189]):
        with pytest.raises(ValueError, match="episode lengths"):
            MODULE.episode_offsets(invalid)


def test_stage_export_retains_exact_float_bytes_and_all_episode_offsets(tmp_path, monkeypatch):
    run = MODULE.Run(SimpleNamespace(output=tmp_path))
    run.runtime = SimpleNamespace(np=np)
    run.check = lambda: None
    # Only the output descriptor containment boundary is adapted for pytest's temp root.
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    for identity in MODULE.cohort("train"):
        i = len(run.rows)
        run.rows.append({**identity, "steps": 1, "start_row": i, "end_row": i + 1})
        run.buffer["features"].append(np.full(31, -0., dtype=np.float32))
        run.buffer["raw_q"].append(np.asarray([4., 3., 2., -0.], dtype=np.float32))
        run.buffer["legal"].append(np.asarray([True, True, True, False]))
        run.buffer["actions"].append(2)
        run.buffer["correction"].append(True)
        run.label_mask.append(True)
        run.selections.append({"episode_id": identity["episode_id"], **select_windows(1, 286000001+i)})
    run.save_stage("train")
    with np.load(tmp_path / "train.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(MODULE.ARRAY_KEYS) | {"label_mask"}
        assert saved["features"].shape == (54, 31) and np.signbit(saved["features"]).all()
        assert np.signbit(saved["raw_q"][:, 3]).all()
        assert saved["episode_offsets"].tolist() == list(range(55))
        assert saved["actions"].dtype == np.int64 and saved["correction"].dtype == np.bool_
    assert run.datasets["train"]["rows"] == 54 and all(not v for v in run.buffer.values())


def test_episode_completion_waits_for_durable_return(tmp_path):
    run = MODULE.Run(SimpleNamespace(output=tmp_path))
    identity = MODULE.cohort()[0]
    events, error = [], OSError("completion fsync failed")
    def episode(_identity):
        run.pending_episode = dict(identity)
        return {**identity, "steps": 1, "start_row": 0, "end_row": 1,
                "source_evaluation_only": [1, 2], "draws_evaluation_only": []}
    def publish(name, record):
        events.append(name)
        if record.get("event") == "return":
            raise error
    run.episode, run.append_durable = episode, publish
    run.ledger.flush = lambda: events.append("flushed")
    run.check = lambda: None
    with pytest.raises(OSError) as caught:
        run.complete_episode(identity, {})
    assert caught.value is error and run.rows == [] and run.pending_episode == identity
    assert events == ["flushed", "episodes.jsonl", "episode-boundaries.jsonl"]


@pytest.mark.parametrize("arm,expected", [("analytic", 0), ("neural", 9), ("period4_hold", 3)])
def test_train_live_routing_scores_only_deployed_queries(arm, expected):
    cache, calls = None, []
    for step in range(9):
        def teacher(deployed, step=step):
            assert deployed
            calls.append(step)
            return np.asarray([4, 3, 2, 1], dtype=np.float32)
        action, raw, cache, queried, correction = MODULE.route_scores(
            arm, step, (0, 1, 2, 3), 1, teacher, select, cache, annotate=False)
        assert queried is (raw is not None)
        assert action == (1 if arm == "analytic" else 3)
        assert correction == (step % 4 == 0)
    assert len(calls) == expected


def deferred_fixture(tmp_path, failure=None, length=3):
    kernel = np.full((4, 107, 107), .25, dtype=np.float64)
    kernel[:, 53, 53] = 0
    view = PublicBeliefView(kernel)
    view._reset(1)
    history, events = [], []

    def packet(step, done=False):
        pos = [26-step, 26] if step <= 20 else [6, 26-(step-20)]
        return {"position": pos, "step": step, "hit": -2 if done else (1 if step == 0 else 0),
                "done": done, "valid_actions": [] if done else [0, 1, 2, 3]}

    for step in range(length):
        before = MODULE.posterior_witness(view)
        after = packet(step+1, step == length-1)
        view._observe(_packet(after, step+1))
        history.append({"public": packet(step), "posterior": before, "action": 0 if step < 20 else 2,
                        "after": after, "posterior_after": MODULE.posterior_witness(view)})
    run = MODULE.Run(SimpleNamespace(output=tmp_path))
    run.check = lambda: None
    run.public_view, run.packet, run.sample_windows = PublicBeliefView, _packet, select_windows
    run.kernels = {"lambda3": kernel}
    run.readonly = lambda v: v
    run.active_actor = SimpleNamespace(public=history[-1]["after"], belief=view.p_source)
    run.label_mask = [False, True, False] + [False] * (length-3)
    run.buffer["raw_q"] = [np.zeros(4, dtype=np.float32), np.full(4, 99, dtype=np.float32),
                            np.zeros(4, dtype=np.float32)] + [np.zeros(4, dtype=np.float32) for _ in range(length-3)]

    class Policy:
        def __init__(self, env, model, sym_avg):
            assert sym_avg
            self.env = env

        def _value_policy(self):
            if failure is not None:
                raise failure
            return 0, np.full(4, self.env.agent[0], dtype=np.float32)

    class Ledger:
        context = None

        def call(self, channel, callback):
            events.append(("call", channel))
            return callback()

        def emit(self, name, value):
            events.append(("emit", name, value))

        def flush(self):
            events.append(("flush",))

    run.ledger = Ledger()
    run.runtime = SimpleNamespace(np=np, policy=Policy, model=None)
    run.reference = SimpleNamespace(ForwardRecorder=lambda *_: None)
    return run, MODULE.cohort("train")[0], history, events


def test_deferred_scores_reuse_live_labels_and_replay_every_final_update(tmp_path):
    run, identity, history, events = deferred_fixture(tmp_path)
    held = np.asarray([4, 3, 2, 1], dtype=np.float32)
    before = held.tobytes(), run.active_actor.belief.tobytes(), dict(run.active_actor.public)
    assert run.annotate_train(identity, history, 0, held) == (2, 1)
    assert run.label_mask == [True, True, True]
    assert [float(v[0]) for v in run.buffer["raw_q"]] == [26., 99., 24.]
    assert events[0][0] == "emit" and events[0][2]["kind"] == "selection" and events[1] == ("flush",)
    assert events.count(("call", "teacher_score")) == 2
    assert events.count(("call", "annotation_public_update")) == 3
    assert len(run.selections) == 1 and run.selections[0]["start_offsets"] == [0]
    assert before == (held.tobytes(), run.active_actor.belief.tobytes(), run.active_actor.public)


def test_deferred_teacher_failure_retains_selection_without_fabricated_label(tmp_path):
    error = OSError("fabricated deferred teacher failure")
    run, identity, history, events = deferred_fixture(tmp_path, error)
    with pytest.raises(OSError) as caught:
        run.annotate_train(identity, history, 0, None)
    assert caught.value is error and len(run.selections) == 1
    assert run.label_mask == [False, True, False]
    assert not any(e[0] == "emit" and e[2]["kind"] == "score" for e in events)


def test_deferred_replay_rejects_corrupt_public_witness_before_teacher(tmp_path):
    run, identity, history, events = deferred_fixture(tmp_path)
    history[0]["posterior"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="replay reset"):
        run.annotate_train(identity, history, 0, None)
    assert ("call", "teacher_score") not in events


def test_declared_teacher_cap_preserves_complete_validation_and_all_paths():
    train = 18*2188 + 36*(547 + 8*3)
    valid = 36*2188
    assert train == 59940 and valid == 78768
    assert MODULE.CALL_CAPS["teacher_score"] == MODULE.CALL_CAPS["tensorflow_value"] == train + valid
    assert MODULE.CALL_CAPS["native_step"] == 90*2188
    assert MODULE.CALL_CAPS["annotation_public_update"] == 54*2188


def test_query_union_keeps_unselected_boundaries_and_query_only_tail():
    # A fabricated projection makes the distinction explicit without changing
    # the production selector: only 4..7 are loss-window rows; 0,8,12 are inputs.
    selection = {"length": 13, "start_offsets": [4]}
    assert MODULE.selected_steps(selection) == (4, 5, 6, 7)
    assert MODULE.train_annotation_steps(selection) == (0, 4, 5, 6, 7, 8, 12)
    assert selection == {"length": 13, "start_offsets": [4]}
    assert MODULE.train_annotation_steps({"length": 1, "start_offsets": [0]}) == (0,)


def test_deferred_union_scores_unselected_queries_without_enlarging_loss_sample(tmp_path):
    run, identity, history, events = deferred_fixture(tmp_path, length=37)
    selection = select_windows(37, 286000001)
    loss_rows = {step for start in selection["start_offsets"] for step in range(start, min(start+4, 37))}
    query_rows = set(range(0, 37, 4))
    extra_queries = query_rows - loss_rows
    assert len(selection["start_offsets"]) == 8 and len(extra_queries) == 2
    required = loss_rows | query_rows
    held = np.asarray([4, 3, 2, 1], np.float32)
    before = held.tobytes(), run.active_actor.belief.tobytes(), dict(run.active_actor.public)
    count, bindings = run.annotate_train(identity, history, 0, held)
    assert count == len(required - {1}) and bindings == 1
    assert {step for step, labeled in enumerate(run.label_mask) if labeled} == required | {1}
    assert run.buffer["raw_q"][1].tobytes() == np.full(4, 99, np.float32).tobytes()
    assert run.selections == [{"episode_id": identity["episode_id"], **selection}]
    scores = [e[2] for e in events if e[0] == "emit" and e[2]["kind"] == "score"]
    assert {r["step"] for r in scores} == required - {1}
    assert extra_queries <= {r["step"] for r in scores}
    assert events.count(("call", "annotation_public_update")) == 37
    assert before == (held.tobytes(), run.active_actor.belief.tobytes(), run.active_actor.public)


def test_missing_unselected_query_failure_cannot_publish_a_label(tmp_path):
    error = OSError("extra query annotation failed")
    run, identity, history, events = deferred_fixture(tmp_path, error, length=37)
    selection = select_windows(37, 286000001)
    selected = {step for start in selection["start_offsets"] for step in range(start, min(start+4, 37))}
    extra = min(set(range(0, 37, 4)) - selected)
    run.label_mask = [True] * 37
    run.label_mask[extra] = False
    before = run.buffer["raw_q"][extra].tobytes()
    with pytest.raises(OSError) as caught:
        run.annotate_train(identity, history, 0, None)
    assert caught.value is error and run.label_mask[extra] is False
    assert run.buffer["raw_q"][extra].tobytes() == before
    assert len(run.selections) == 1 and events[1] == ("flush",)
    assert events.count(("call", "annotation_public_update")) == extra
    assert not any(e[0] == "emit" and e[2]["kind"] == "score" for e in events)


def test_export_rejects_complete_loss_windows_with_missing_query_anchors(tmp_path):
    run = MODULE.Run(SimpleNamespace(output=tmp_path))
    run.runtime, run.check = SimpleNamespace(np=np), lambda: None
    for index, identity in enumerate(MODULE.cohort("train")):
        length = 37 if index == 0 else 1
        start = len(run.label_mask)
        selection = select_windows(length, 286000001+index)
        selected = {step for offset in selection["start_offsets"] for step in range(offset, min(offset+4, length))}
        run.rows.append({**identity, "steps": length, "start_row": start, "end_row": start+length})
        run.selections.append({"episode_id": identity["episode_id"], **selection})
        for step in range(length):
            run.buffer["features"].append(np.zeros(31, np.float32))
            run.buffer["raw_q"].append(np.zeros(4, np.float32))
            run.buffer["legal"].append(np.ones(4, np.bool_))
            run.buffer["actions"].append(0)
            run.buffer["correction"].append(step % 4 == 0)
            run.label_mask.append(step in selected)
    with pytest.raises(ValueError, match="query anchors"):
        run.save_stage("train")
    assert not list(tmp_path.iterdir()) and run.datasets == {}
