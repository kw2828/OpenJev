"""Metadata parsing checks; no actual ledgers or scientific arrays are read."""
import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1]/'scripts/review_otto_spatial_seeds.py'
    spec = importlib.util.spec_from_file_location('_test_spatial_seed_review', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_named_nested_seed_maps_exclude_unrelated_counts_and_descriptors():
    m = module()
    value = {'seed': 16100001, 'rows': 16100002, 'seeds': {'lambda3': [1, 2]},
             'enabled_seed': True, 'not_a_seed_number': 1.25,
             'inputs': {'seed_review': {'path': 'example', 'bytes': 16100003, 'sha256': 'abc'}}}
    assert list(m.seed_fields(value)) == [('$.seed', 16100001), ('$.seeds.lambda3[]', 1), ('$.seeds.lambda3[]', 2)]


def test_ranges_do_not_invent_values_between_disjoint_seeds():
    assert module().ranges({9, 1, 2, 4, 5}) == [[1, 2], [4, 5], [9, 9]]


def test_source_token_scope_does_not_match_probability_decimal():
    m = module()
    assert [int(x.group()) for x in m.TOKEN.finditer('FIRST = 16100001; uniform = 0.16100002')] == [16100001]


def test_current_reservations_handle_both_filename_styles():
    m = module()
    assert m.current('scripts/audit_otto_spatial_control.py')
    assert m.current('research/otto-spatial-control-protocol.md')
    assert not m.current('scripts/study_otto_conditioning_control.py')


def test_ledger_inventory_includes_stopped_collection_and_allocated_rows():
    m = module()
    assert m.LEDGER.fullmatch('output/otto-coverage-v1/collection-01/collection-episodes.jsonl')
    assert m.LEDGER.fullmatch('output/otto-conditioning-control-v1/run-01/evaluation.jsonl')
    assert not m.LEDGER.fullmatch('output/otto-spatial-study-v1/publication-verify-01/repo/output/run/episodes.jsonl')


def test_exact_proposed_allocation():
    m = module()
    assert len(m.PROPOSED) == 75
    assert 16100024 in m.PROPOSED and 16100025 not in m.PROPOSED
    assert 16500003 in m.PROPOSED and 16500004 not in m.PROPOSED
