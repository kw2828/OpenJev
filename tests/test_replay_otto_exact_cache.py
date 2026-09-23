"""Tiny fabricated public histories and journal bytes, never saved science."""
from __future__ import annotations

import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import otto_exact_value_cache as cache
from openjev.research import otto_released_policy as public

SOURCE = Path(__file__).resolve().parents[1] / "scripts/replay_otto_exact_cache.py"
SPEC = importlib.util.spec_from_file_location("_test_exact_replay", SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def packet(position, hit, done, step):
    return {"position": list(position), "hit": hit, "done": done, "step": step,
            "valid_actions": [] if done else [a for a in range(4) if public._move(a, position)[1]]}


def fixture():
    cells = [(stage, regime, arm) for stage in ("train", "valid") for regime in ("lambda3", "lambda4")
             for arm in ("analytic", "neural", "period4_hold")]
    first = [("train", "lambda3", "analytic"), ("train", "lambda4", "neural")]
    cells = first + [c for c in cells if c not in first]
    cohort = [{"stage": stage, "regime": regime, "arm": arm, "episode_id": f"{stage}:{regime}:{arm}",
               "episode_index": i, "seed": 100+i, "case": 0, "initial_hit": 1}
              for i, (stage, regime, arm) in enumerate(cells)]
    kernels = {r: np.zeros((4, 107, 107), dtype=np.float64) for r in ("lambda3", "lambda4")}
    samples, forwards, transitions, completed = [], [], [], []
    for index, identity in enumerate(cohort[:2]):
        view = public.PublicBeliefView(kernels[identity["regime"]])
        view._reset(1)
        current = packet((26, 26), 1, False, 0)
        state = m.witness(view)
        transitions.append({"kind": "reset", **identity, "public": current, "posterior_after": state,
                            "source_evaluation_only": [2, 7]})
        for step in range(2):
            deployed = identity["arm"] == "neural"
            row_index = 2*index + step
            inputs, masses = cache.build_inputs(view)
            assert not inputs.any()  # Every nonterminal fabricated posterior remains zero.
            context = {"phase": "teacher_score", **identity, "step": step,
                       "deployed_query": deployed, "annotation_only": not deployed}
            samples.append({**identity, "step": step, "row_index": row_index, "public": current,
                "posterior": state, "deployed_query": deployed, "annotation_only": not deployed,
                "action": 0, "legal": [a in current["valid_actions"] for a in range(4)],
                "raw_q": "intentionally unused", "features": "intentionally unused"})
            forwards.append({"context": context, "ordinal": row_index + 1, "seconds": .25,
                "symmetry_average": True, "input_shape": [16, 105, 105],
                "branch_masses": masses.tolist(), "values": [4.] * 16})
            successor, _ = public._move(0, current["position"])
            done = index == 0 and step == 1
            following = packet(successor, -2 if done else 0, done, step+1)
            view._observe(public._packet(following, step+1))
            following_witness = m.witness(view)
            transitions.append({"kind": "step", **identity, "step": step+1, "row_index": row_index,
                "action": 0, "deployed_query": deployed, "public": following,
                "posterior_before": state, "posterior_after": following_witness, "native_p_end": float(done)})
            current, state = following, following_witness
        if index == 0:
            completed.append({**identity, "steps": 2, "found": True, "censored": False,
                "start_row": 0, "end_row": 2, "final_public": current, "updates": 2,
                "final_update_assimilated": True})
    return {"samples": samples, "forwards": forwards, "transitions": transitions,
            "cohort": cohort, "completed": completed, "pending": cohort[1], "kernels": kernels,
            "expected_rows": 4, "expected_resets": 2}


def replay(data, emit=lambda _: None):
    engine = m.Replay(np, public, cache, emit=emit)
    args = {k: iter(v) if k in ("samples", "forwards", "transitions") else v for k, v in data.items()}
    return engine, engine.run(**args)


def test_full_partial_replay_zero_mass_final_update_all_cells_and_no_cross_episode_hits():
    data, emitted = fixture(), []
    engine, result = replay(data, emitted.append)
    assert result["agreement"] is result["continuation"] is True
    assert result["totals"] == {"requests": 4, "hits": 2, "misses": 2, "evictions": 0,
                                "original_forward_seconds": 1., "hit_forward_seconds": .5}
    assert [r["hit"] for r in emitted] == [False, True, False, True]
    assert [r["partial_episode"] for r in emitted] == [False, False, True, True]
    assert len(result["by_stage_regime_arm"]) == 12
    assert all(r["requests"] == 0 and r["unstarted_episodes"] == 1 for r in result["by_stage_regime_arm"][2:])
    assert result["maximum_cache_key_bytes"] == 16*105*105*4
    assert engine.counts == {"public_resets": 2, "public_updates": 4, "branch_builds": 4,
                             "historical_value_callbacks": 2, "requests": 4}
    assert emitted[1]["posterior_after"]["mass"] == 1. and emitted[-1]["posterior_after"]["mass"] == 0.
    assert all(row["passed"] for row in result["conditions"]) and engine.pending is None


@pytest.mark.parametrize("corruption,message", [("mass", "branch masses"), ("value", "cache value"),
                                                ("posterior", "reset witness"), ("order", "preaction"),
                                                ("extra", "all original returned")])
def test_replay_rejects_value_mass_public_order_and_extra_record_corruption(corruption, message):
    data = fixture()
    if corruption == "mass":
        data["forwards"][0]["branch_masses"][0][0] = float(np.nextafter(np.float32(1e-10), np.float32(1)))
    elif corruption == "value":
        data["forwards"][1]["values"][0] = float(np.nextafter(np.float32(4), np.float32(5)))
    elif corruption == "posterior":
        data["transitions"][0]["posterior_after"] = {"mass": 0., "sha256": "0"*64, "exact": True}
    elif corruption == "order":
        data["samples"][1]["step"] = 0
    else:
        data["forwards"].append(copy.deepcopy(data["forwards"][-1]))
    with pytest.raises(ValueError, match=message):
        replay(data)


def test_publication_failure_retains_unacknowledged_request_and_progress():
    data, failure = fixture(), OSError("fabricated output write failure")

    def emit(_):
        raise failure

    engine = m.Replay(np, public, cache, emit=emit)
    with pytest.raises(OSError) as caught:
        engine.run(**{k: iter(v) if k in ("samples", "forwards", "transitions") else v for k, v in data.items()})
    assert caught.value is failure and engine.counts["requests"] == 0
    assert engine.counts["branch_builds"] == engine.counts["public_updates"] == 1
    assert engine.pending["step"] == 0 and engine.pending["episode_id"] == data["cohort"][0]["episode_id"]


def test_original_work_stack_keeps_unreturned_tail_and_joins_returned_forward_time():
    identity = {"episode_id": "synthetic", "episode_index": 0}
    context = {"phase": "teacher_score", **identity, "step": 0}
    outer = {"call_id": 1, "channel": "teacher_score", "parent_call_id": None, "context": context}
    inner = {"call_id": 2, "channel": "tensorflow_value", "parent_call_id": 1, "context": context}
    tail_context = {**context, "step": 1}
    tail_outer = {**outer, "call_id": 3, "context": tail_context}
    tail_inner = {**inner, "call_id": 4, "parent_call_id": 3, "context": tail_context}
    records = [{"event": "attempt", **outer}, {"event": "attempt", **inner},
               {"event": "return", **inner, "seconds": .25, "instrumented_seconds": .5, "excluded_io_seconds": .25},
               {"event": "return", **outer, "seconds": .5, "instrumented_seconds": .75, "excluded_io_seconds": .25},
               {"event": "attempt", **tail_outer}, {"event": "attempt", **tail_inner}]
    expected = {"teacher_score": {"attempted": 2, "returned": 1, "seconds": .5},
                "tensorflow_value": {"attempted": 2, "returned": 1, "seconds": .25}}
    work = m.Work(iter(records), expected, [tail_outer, tail_inner], [identity])
    work.forward(context, 1, .25)
    work.finish()
    assert work.counts == expected and work.stack == [tail_outer, tail_inner]
    bad = m.Work(iter(records), expected, [tail_outer, tail_inner], [identity])
    with pytest.raises(ValueError, match="recorded time"):
        bad.forward(context, 1, .5)


def test_complete_gzip_eof_requires_trailer_and_whole_record():
    encoded = b'{"x":1}\n{"x":2}\n'
    valid = gzip.compress(encoded, mtime=0)
    with gzip.GzipFile(fileobj=io.BytesIO(valid)) as stream:
        rows = m.Rows(stream)
        assert list(rows) == [{"x": 1}, {"x": 2}] and rows.eof
    with gzip.GzipFile(fileobj=io.BytesIO(valid[:-4])) as stream, pytest.raises(EOFError):
        list(m.Rows(stream))
    with pytest.raises(ValueError, match="whole bounded"):
        list(m.Rows(io.BytesIO(b'{"x":1}')))


def test_late_failure_demotes_completed_receipt_without_losing_error(tmp_path):
    output = tmp_path / "attempt"
    run = m.Run(SimpleNamespace(output=output, plan=tmp_path / "plan", plan_sha256="p", supervision=tmp_path / "s"))
    error = RuntimeError("fabricated late failure")

    def bind():
        (output / "receipt.json").write_text('{"status":"completed"}')
        raise error

    run.bind = bind
    with pytest.raises(RuntimeError) as caught:
        run.execute()
    assert caught.value is error
    assert json.loads((output / "receipt.invalid.json").read_text())["status"] == "completed"
    failure = json.loads((output / "receipt.json").read_text())
    assert failure["status"] == "failed" and failure["agreement"] is False and "late failure" in failure["error"]
