import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_shared_prefix.py"
SPEC = importlib.util.spec_from_file_location("benchmark_shared_prefix", PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_workload_has_all_prespecified_cells_and_unique_ids():
    rows = benchmark.requests()
    assert len(rows) == len({r["id"] for r in rows}) == 54
    for questions in (1, 2, 4):
        for size in ("short", "long"):
            for candidates in (2, 4, 12):
                group = [r for r in rows if (r["question_count"], r["context_size"], r["candidate_count"])
                         == (questions, size, candidates)]
                assert len(group) == 3
                assert all(len(r["request"]["questions"]) == questions for r in group)
                assert all(len(q["candidates"]) == candidates for r in group for q in r["request"]["questions"])


def result(probability=.7, mass=.5, choice="a"):
    return {"answers": [{"id": "field", "choice": choice,
                         "probabilities": {"a": probability, "b": 1 - probability},
                         "candidate_token_mass": mass}], "candidate_logits": [[2., 1.]]}


def test_parity_rejects_choice_probability_mass_and_identity_changes():
    reference = result()
    assert benchmark.compare(reference, result())["pass"]
    assert benchmark.compare(reference, result(.704))["pass"]
    assert not benchmark.compare(reference, result(.706))["pass"]
    assert not benchmark.compare(reference, result(mass=.506))["pass"]
    assert not benchmark.compare(reference, result(choice="b"))["pass"]
    altered = result()
    altered["answers"][0]["id"] = "other"
    with pytest.raises(ValueError):
        benchmark.compare(reference, altered)


def test_artifact_writes_cannot_overwrite_or_publish_nan(tmp_path):
    target = tmp_path / "record.json"
    benchmark.write(target, {"value": 1})
    with pytest.raises(FileExistsError):
        benchmark.write(target, {"value": 2})
    with pytest.raises(ValueError):
        benchmark.write(tmp_path / "invalid.json", {"value": float("nan")})
