import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('pin_screen', Path(__file__).parents[1]/'scripts/chess_pin_factor_screen.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_frozen_membership_and_promotion_mirror():
    assert m.indices() == list(range(0, 32768, 32))
    assert len(m.indices()) == 1024
    assert m.mirror_uci('a7a8n') == 'a2a1n'
    assert m.mirror_uci('e1g1') == 'e8g8'


def test_coverage_distinguishes_counts_and_identity():
    a, b = [0, 4, 12, 60], [0, 4, 20, 60]
    records = [{'before': [a], 'candidates': [
        {'after': [a], 'factors': [[*a, 0]]},
        {'after': [b], 'factors': [[*a, 2], [*b, 1]]},
        {'after': [], 'factors': [[*a, 2]]}]}]
    summary = m.summarize(records)
    assert summary == {'roots': 1, 'roots_with_pins': 1, 'roots_with_changed_candidates': 1,
        'roots_with_candidate_variation': 1, 'candidates': 3, 'candidates_with_pins': 2,
        'changed_candidates': 2, 'counts_changed_candidates': 1, 'identity_only_changed_candidates': 1,
        'witness_roles_by_owner_retained_added_removed': [[1, 1, 2], [0, 0, 0]]}
