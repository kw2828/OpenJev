"""Focused terminal-contract repair fixtures; no saved scientific records."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = load('return_consistency_v2_test_subject', 'scripts/diagnose_otto_return_consistency_v2.py')
fixtures = load('return_consistency_v1_fixture_provider', 'tests/test_otto_return_consistency.py')


def corrected_events():
    for row in fixtures.events():
        if row['public']['done']:
            row['public']['valid_actions'] = []
        yield row


def test_terminal_empty_stream_reaches_same_sixteen_prefixes():
    consumed = []
    def stream():
        for row in corrected_events():
            consumed.append(row)
            yield row
            if row['episode_id'] == v2.BASE.SELECTED['valid'][0] and row.get('step') == 8:
                raise AssertionError('read past last required VALID prefix')
    metadata = fixtures.metadata()
    got = v2.BASE.teacher_join(stream(), metadata)
    old_synthetic = fixtures.d.teacher_join(fixtures.events(), metadata)
    assert got == old_synthetic and len(got) == 16
    assert len(consumed) == 1 + 79 + 1 + 8
    terminal = consumed[79]
    assert terminal['public']['done'] and terminal['public']['hit'] == -2
    assert terminal['public']['valid_actions'] == []
    assert terminal['allowed_actions'] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match='eligible actions'):
        fixtures.d.teacher_join(corrected_events(), metadata)


def test_all144_saved_arithmetic_and_eighteen_panels_unchanged():
    metadata, parity = fixtures.metadata(), fixtures.parity()
    teachers = v2.BASE.teacher_join(corrected_events(), metadata)
    actual = v2.BASE.diagnose(parity, metadata, teachers)
    expected = fixtures.d.diagnose(parity, metadata, fixtures.d.teacher_join(fixtures.events(), metadata))
    assert len(actual) == 144 and actual == expected
    assert sum(len(row['branches']) for row in actual) == 2304
    summary = v2.BASE.summarize(actual)
    previous = fixtures.d.summarize(expected)
    assert summary['version'] == v2.VERSION
    previous['version'] = v2.VERSION
    assert summary == previous
    assert sum(len(panels) for panels in summary['fit_split'].values()) == 18


@pytest.mark.parametrize('actions', [[0, 1, 2, 3], [0], (), None, False])
def test_terminal_requires_exact_empty_list(actions):
    value = fixtures.public(79, done=True)
    value['valid_actions'] = actions
    with pytest.raises(ValueError, match='terminal packet'):
        v2.packet(value)


@pytest.mark.parametrize('actions', [[], [0], [3, 2, 1, 0], [0, 1, 2, 2], [True, 1, 2, 3]])
def test_nonterminal_grid_eligibility_remains_strict(actions):
    value = fixtures.public(2)
    value['valid_actions'] = actions
    with pytest.raises(ValueError):
        v2.packet(value)


@pytest.mark.parametrize('done,hit', [(True, 0), (True, 3), (False, -2), (1, -2), (True, True)])
def test_hit_and_done_contract_stays_exact(done, hit):
    value = fixtures.public(2)
    value.update(done=done, hit=hit, valid_actions=[] if done else [0, 1, 2, 3])
    with pytest.raises(ValueError):
        v2.packet(value)


def test_terminal_identity_and_input_ownership():
    value = fixtures.public(79, done=True)
    value['valid_actions'] = []
    original = copy.deepcopy(value)
    assert v2.packet(value) is value and value == original
    corner = fixtures.public(2)
    corner.update(position=[0, 52], valid_actions=[1, 2])
    assert v2.packet(corner) == fixtures.d.packet(corner)


def test_eight_source_closure_and_inherited_functions_remain_bound():
    assert len(v2.SOURCES) == 8
    assert v2.INHERITED_SOURCES == frozenset(fixtures.d.SOURCES)
    assert v2.SOURCES == set(fixtures.d.SOURCES) | v2.ADDITIONS
    assert v2.BASE.VERSION == 'otto-return-consistency-v2'
    assert v2.BASE.SOURCES == v2.SOURCES
    assert v2.BASE.packet is v2.packet
    assert v2.BASE.CONFIGURATION == fixtures.d.CONFIGURATION
    assert v2.BASE.LIMITS == fixtures.d.LIMITS
    assert hashlib.sha256((ROOT / v2.ORIGINAL).read_bytes()).hexdigest() == v2.ORIGINAL_PIN
    for name in ('analyze_row', 'select', 'diagnose', 'summarize', 'teacher_join'):
        actual, old = getattr(v2.BASE, name), getattr(fixtures.d, name)
        assert actual.__code__.co_code == old.__code__.co_code
        assert actual.__code__.co_filename == old.__code__.co_filename
    with pytest.raises(ValueError):
        v2.BASE.allowed([])  # Selection/step masks were not relaxed globally.


def test_original_source_pin_rejects_modified_dependency(tmp_path, monkeypatch):
    path = tmp_path / v2.ORIGINAL
    path.parent.mkdir(parents=True)
    path.write_text('raise AssertionError("must not execute changed source")\n')
    monkeypatch.setattr(v2, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='source pin changed'):
        v2.load_original()
