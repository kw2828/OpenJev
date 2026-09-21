"""Artificial metadata and streaming keys only; no scientific arrays or actors."""
import hashlib
import random

import pytest

from openjev.research import otto_coverage_selection as m


def counts(values):
    return dict(zip(m.STRATA, values, strict=True))


def metadata(values=(1000, 900, 800, 700, 1200, 989)):
    rows = []
    for (regime, hit), count in zip(m.STRATA, values, strict=True):
        for _ in range(count):
            rows.append({'row_index': len(rows), 'stage': 'train', 'regime': regime, 'initial_hit': hit})
    return rows


def oracle(identities, quota):
    return sorted((hashlib.sha256(('otto-coverage-v1|'+identity).encode('utf-8')).hexdigest(), identity)
                  for identity in identities)[:quota]


def test_integer_largest_remainders_has_explicit_unequal_cell_oracle():
    assert m.apportion(counts((4, 3, 2, 1, 0, 0)), 7) == counts((3, 2, 1, 1, 0, 0))


def test_equal_remainders_tie_by_lexicographic_cell_not_mapping_insertion_order():
    values = dict(reversed(list(counts((1, 1, 1, 1, 1, 1)).items())))
    assert m.apportion(values, 3) == counts((1, 1, 1, 0, 0, 0))
    assert list(m.apportion(values, 3)) == list(m.STRATA)


@pytest.mark.parametrize('total', [0, 1, 9, 10])
def test_allocation_conserves_total_and_never_exceeds_cell_population(total):
    source = counts((4, 3, 2, 1, 0, 0))
    result = m.apportion(source, total)
    assert sum(result.values()) == total
    assert all(0 <= result[key] <= source[key] for key in m.STRATA)


@pytest.mark.parametrize(('quota', 'expected'), [(0, (0, 0, 0)), (1, (1, 0, 0)), (2, (1, 1, 0)),
                                               (5, (2, 2, 1)), (93, (31, 31, 31))])
def test_collectors_use_all_fixed_seeds_and_ascending_remainders(quota, expected):
    assert m.split_collectors(quota) == dict(zip((10101, 10102, 10103), expected, strict=True))


def test_full_synthetic_cohort_preserves_all_six_strata_and_total_mix():
    result = m.coverage_quotas(iter(metadata()))
    assert [(r['regime'], r['initial_hit']) for r in result] == list(m.STRATA)
    assert [r['student_rows'] for r in result] == [499, 449, 399, 350, 599, 494]
    assert [r['teacher_rows'] for r in result] == [501, 451, 401, 350, 601, 495]
    assert sum(r['student_rows'] for r in result) == 2790
    assert sum(r['teacher_rows'] for r in result) == 2799
    for row in result:
        assert sum(row['collector_rows'].values()) == row['student_rows']
        assert row['teacher_rows']+row['student_rows'] == row['original_rows']
        assert max(row['collector_rows'].values())-min(row['collector_rows'].values()) <= 1


def test_outcomes_targets_and_numerical_beliefs_cannot_affect_quotas():
    class Forbidden:
        def __bool__(self):
            raise AssertionError('outcome used')

        def __float__(self):
            raise AssertionError('target used')

    baseline = metadata()
    decorated = [{**row, 'found': Forbidden(), 'target': Forbidden(), 'belief': Forbidden(),
                  'total_steps': Forbidden(), 'score': Forbidden()} for row in baseline]
    assert m.coverage_quotas(decorated) == m.coverage_quotas(baseline)


@pytest.mark.parametrize('defect', ['short', 'long', 'order', 'duplicate', 'valid_split', 'unknown_cell', 'bool_hit', 'bool_index'])
def test_canonical_original_metadata_required(defect):
    rows = metadata()
    if defect == 'short':
        rows.pop()
    elif defect == 'long':
        rows.append({**rows[-1], 'row_index': 5589})
    elif defect == 'order':
        rows[0], rows[1] = rows[1], rows[0]
    elif defect == 'duplicate':
        rows[1] = rows[0]
    elif defect == 'valid_split':
        rows[0]['stage'] = 'valid'
    elif defect == 'unknown_cell':
        rows[0]['regime'] = 'lambda5'
    elif defect == 'bool_hit':
        rows[0]['initial_hit'] = True
    else:
        rows[0]['row_index'] = False
    with pytest.raises(ValueError):
        m.coverage_quotas(rows)


@pytest.mark.parametrize('defect', ['negative', 'bool_count', 'float_count', 'missing', 'extra', 'empty_counts', 'bool_hit'])
def test_bad_apportionment_counts_fail_closed(defect):
    source = counts((1, 2, 3, 4, 5, 6))
    if defect in ('negative', 'bool_count', 'float_count'):
        source[m.STRATA[0]] = {'negative': -1, 'bool_count': True, 'float_count': 1.5}[defect]
    elif defect == 'missing':
        source.pop(m.STRATA[0])
    elif defect == 'extra':
        source['lambda5', 1] = 1
    elif defect == 'empty_counts':
        source = counts((0, 0, 0, 0, 0, 0))
    else:
        value = source.pop(('lambda3', 1))
        source['lambda3', True] = value
    with pytest.raises(ValueError):
        m.apportion(source, 3)


@pytest.mark.parametrize('total', [-1, True, 1.5, 22])
def test_bad_or_infeasible_total_rejected(total):
    with pytest.raises(ValueError):
        m.apportion(counts((1, 2, 3, 4, 5, 6)), total)


@pytest.mark.parametrize('order', ['forward', 'reverse', 'shuffled'])
def test_streaming_selection_equals_complete_sort_and_is_order_invariant(order):
    identities = [m.student_identity(f'train:lambda3:{130000+i}:mlp8@10101', i % 13) for i in range(90)]
    stream = list(identities)
    if order == 'reverse':
        stream.reverse()
    elif order == 'shuffled':
        random.Random(37).shuffle(stream)
    selector = m.SaltedSelector(11)
    for identity in stream:
        selector.offer(identity, lambda identity=identity: {'identity_copy': identity, 'numeric_belief': [0.5, 0.5]})
        assert len(selector) <= 11
    selected = selector.finish()
    assert [(row['key'], row['identity']) for row in selected] == oracle(identities, 11)
    assert len({row['identity'] for row in selected}) == 11
    assert all(row['payload']['identity_copy'] == row['identity'] for row in selected)
    assert all(row['payload']['numeric_belief'] == [0.5, 0.5] for row in selected)  # No belief deduplication.


def test_payload_is_lazy_and_does_not_snapshot_rejected_candidates():
    identities = [m.teacher_identity(i) for i in range(30)]
    ordered = [identity for _, identity in oracle(identities, 30)]
    selector, called = m.SaltedSelector(3), []
    for identity in ordered:
        def snapshot(identity=identity):
            called.append(identity)
            return {'captured': identity}
        assert selector.offer(identity, snapshot) == (identity in ordered[:3])
    assert called == ordered[:3]
    assert len(selector.finish()) == 3


def test_payload_factory_failure_preserves_previous_selection():
    ordered = [identity for _, identity in oracle(['first', 'second'], 2)]
    selector = m.SaltedSelector(1)
    selector.offer(ordered[1], lambda: 'previous snapshot')
    before = selector.finish()

    def fail():
        raise RuntimeError('snapshot failed')

    with pytest.raises(RuntimeError, match='snapshot failed'):
        selector.offer(ordered[0], fail)
    assert selector.finish() == before


def test_digest_collision_uses_identity_lexicographic_tie(monkeypatch):
    monkeypatch.setattr(m, 'selection_key', lambda identity: ('0'*64, identity))
    selector = m.SaltedSelector(2)
    for identity in ('z', 'b', 'a', 'c'):
        selector.offer(identity)
    assert [row['identity'] for row in selector.finish()] == ['a', 'b']


def test_retained_duplicate_rejected_without_second_snapshot():
    selector = m.SaltedSelector(2)
    selector.offer('same', lambda: 'first')
    with pytest.raises(ValueError, match='duplicate'):
        selector.offer('same', lambda: pytest.fail('duplicate snapshot'))
    assert len(selector) == 1


def test_unretained_duplicate_detection_is_explicitly_the_callers_responsibility():
    ordered = [identity for _, identity in oracle(['first', 'second'], 2)]
    selector = m.SaltedSelector(1)
    selector.offer(ordered[1])
    selector.offer(ordered[0])
    assert selector.offer(ordered[1]) is False  # Already evicted; no full-stream identity set retained.
    assert [row['identity'] for row in selector.finish()] == [ordered[0]]


def test_underfilled_cell_fails_instead_of_cross_cell_replacement():
    selector = m.SaltedSelector(2)
    selector.offer('only-one')
    with pytest.raises(ValueError, match='insufficient'):
        selector.finish()


def test_zero_quota_stores_nothing_and_never_snapshots():
    selector = m.SaltedSelector(0)
    assert selector.offer('candidate', lambda: pytest.fail('zero quota snapshot')) is False
    assert selector.finish() == []


def test_identity_serialization_and_canonical_utf8_key():
    assert m.teacher_identity(27) == 'teacher:27'
    identity = m.student_identity('train:lambda4:130001:mlp8@10102', 0)
    assert identity == 'train:lambda4:130001:mlp8@10102:0'
    assert m.selection_key(identity) == oracle([identity], 1)[0]
    assert m.selection_key('episode:é:1') == oracle(['episode:é:1'], 1)[0]


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_invalid_quota_rejected(value):
    with pytest.raises(ValueError):
        m.SaltedSelector(value)
    with pytest.raises(ValueError):
        m.split_collectors(value)


@pytest.mark.parametrize('identity', ['', None, 1])
def test_invalid_identity_rejected(identity):
    with pytest.raises(ValueError):
        m.SaltedSelector(1).offer(identity)


@pytest.mark.parametrize('index', [-1, True, 5589])
def test_invalid_teacher_identity(index):
    with pytest.raises(ValueError):
        m.teacher_identity(index)


def test_noncallable_payload_factory_rejected():
    with pytest.raises(ValueError, match='callable'):
        m.SaltedSelector(1).offer('valid', object())
