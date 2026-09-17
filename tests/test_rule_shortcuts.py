import runpy
from pathlib import Path

MODULE = runpy.run_path(str(Path(__file__).parents[1]/'scripts/audit_rule_shortcuts.py'))


def test_negation_match_is_word_based_and_case_insensitive():
    assert MODULE['has_not']('A is NOT red.')
    assert not MODULE['has_not']('A notices B.')


def test_four_group_metric_exposes_label_negation_imbalance():
    # A surface-only rule can have 90% raw accuracy and 50% macro accuracy.
    rows = [(0, 1, 3)]*9+[(1, 0, 3)]*9+[(1, 1, 3)]+[(0, 0, 3)]
    predictions = [1-neg for _, neg, _ in rows]
    result = MODULE['evaluate'](predictions, rows)
    assert result['accuracy'] == .9
    assert result['four_group_macro_accuracy'] == .5
