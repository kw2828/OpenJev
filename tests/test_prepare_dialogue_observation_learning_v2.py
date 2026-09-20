"""Lineage and namespace correction; no corpus or neural execution."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("observation_preparation_v2", SCRIPTS / "prepare_dialogue_observation_learning_v2.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_recipe_and_caps_unchanged_from_preserved_failed_preparation():
    assert prep.CONFIG == prep.first.CONFIG
    assert prep.CAPS == prep.first.CAPS
    assert set(prep.first.INPUTS.items()) <= set(prep.INPUTS.items())
    assert set(prep.first.SOURCES) < set(prep.SOURCES)


def test_qualified_and_failed_source_lineage_are_both_bound(monkeypatch):
    monkeypatch.setattr(prep.qualified, "sources", lambda: {"qualified.py": "unchanged"})
    failed = {**dict.fromkeys(prep.first.SOURCES, "old"), "qualified.py": "unchanged"}
    monkeypatch.setattr(prep, "read", lambda _: failed)
    checked = []
    monkeypatch.setattr(prep, "bind", lambda files: checked.append(dict(files)))
    assert prep.inherited_sources({"source_sha256": {"qualified.py": "unchanged"}}) == failed
    assert checked == [failed]
    failed.pop(prep.first.SOURCES[0])
    with pytest.raises(ValueError, match="Complete failed-version"):
        prep.inherited_sources({"source_sha256": {"qualified.py": "unchanged"}})


def test_changed_qualification_lineage_rejected_before_binding(monkeypatch):
    monkeypatch.setattr(prep.qualified, "sources", lambda: {"qualified.py": "new"})
    monkeypatch.setattr(prep, "read", lambda _: pytest.fail("Read after changed qualification"))
    with pytest.raises(ValueError, match="All inherited"):
        prep.inherited_sources({"source_sha256": {"qualified.py": "old"}})
