"""Synthetic algebra and decision invariants for saved typed-output diagnosis."""
import copy

import diagnose_dialogue_typed as d
import numpy as np
import pytest


def rows_for(targets=(2, 3, 1), services=('a', 'a', 'b')):
    ids = ['reserved:NOT_MENTIONED', 'reserved:DONTCARE', 'value:A', 'value:B']
    result = []
    for i, (target, service) in enumerate(zip(targets, services, strict=True)):
        result.append({'row_index': i, 'split': 'train', 'admission': 'admitted',
                       'dialogue_id': 'd'+str(i//2), 'query_id': 'q'+service, 'service': service,
                       'heldout_service': True, 'candidate_count': 4, 'candidate_types': [0, 1, 4, 4],
                       'current_label_index': target, 'previous_current_index': 0,
                       'current_candidate_id': ids[target], 'previous_candidate_id': ids[0],
                       'derived_bin': 'first_assignment' if target != 0 else 'unmentioned_retention',
                       'current_value_group': d.report.VALUES[[0, 1, 4, 4][target]]})
    return result


def packet(rows, probabilities=None):
    p = np.asarray(probabilities if probabilities is not None else [[.4, .1, .26, .24]]*len(rows))
    logs = np.full((len(rows), 12), -np.inf, dtype=np.float32)
    logs[:, :4] = np.log(p)
    return {'row_indices': np.asarray([r['row_index'] for r in rows], np.int64), 'log_probs': logs}


def test_mass_branch_is_not_actual_decision_and_partition_is_exhaustive():
    rows = rows_for()
    p = packet(rows, [[.4, .1, .26, .24], [.1, .1, .45, .35], [.1, .7, .1, .1]])
    q = d.row_quantities(rows, p)
    assert q['selected_branch'].tolist() == [0, 2, 1]
    assert q['target_branch'].tolist() == [2, 2, 1]
    assert q['error_code'].tolist() == [1, 2, 0]
    assert q['decisions']['mass_branch_disagrees'].tolist() == [True, False, False]
    assert q['decisions']['mass_branch_correct'].all()
    assert q['decisions']['conditional_value_correct'].tolist() == [True, False, True]
    assert q['loss']['value'][2] == 0
    np.testing.assert_allclose(q['loss']['total'], q['loss']['branch']+q['loss']['value'], atol=1e-12)
    cell = d.describe(rows, q, np.ones(3, bool))
    assert [cell['decisions'][k] for k in d.DECISIONS[:3]] == [1, 1, 1]
    assert cell['selected_branch_confusion'] == [[0, 0, 0], [0, 1, 0], [1, 0, 1]]
    assert cell['loss']['total']['row'] != cell['loss']['total']['equal_service']
    for weighting in ('row', 'equal_service', 'equal_dialogue'):
        assert cell['loss']['total'][weighting] == pytest.approx(cell['loss']['branch'][weighting]+cell['loss']['value'][weighting])


def test_stable_underflow_and_unrepaired_small_negative_branch_nll():
    rows = rows_for((2,), ('a',)); p = packet(rows)
    p['log_probs'][0, :4] = [0, -1200, -1100, -1101]
    q = d.row_quantities(rows, p)
    assert q['loss']['total'][0] == 1100
    assert q['loss']['branch'][0] == pytest.approx(1100-np.log1p(np.exp(-1)))
    assert q['loss']['value'][0] == pytest.approx(np.log1p(np.exp(-1)))
    rows = rows_for((0,), ('a',)); p = packet(rows)
    p['log_probs'][0, :4] = [-1e-8, -30, -np.log(2)+5e-7, -np.log(2)+5e-7]
    # Above has total mass near two, so reject it rather than repair it.
    with pytest.raises(ValueError, match='Unnormalized'):
        d.row_quantities(rows, p)
    rows = rows_for((2,), ('a',)); p = packet(rows)
    p['log_probs'][0, :4] = [-30, -30, -np.log(2)+5e-7, -np.log(2)+5e-7]
    q = d.row_quantities(rows, p)
    assert -2e-6 < q['loss']['branch'][0] < 0
    assert q['maximum_probability_mass_error'] < 2e-6
    assert q['loss']['total'][0] == pytest.approx(q['loss']['branch'][0]+q['loss']['value'][0])


def test_padding_ties_and_candidate_permutation_without_ties():
    rows = rows_for((3,), ('a',)); p = packet(rows, [[.1, .1, .4, .4]])
    q = d.row_quantities(rows, p)
    assert q['decisions']['top1_tie'].tolist() == [True]
    assert q['decisions']['wrong_value'].tolist() == [True]  # first supported argmax wins
    p['log_probs'][0, 11] = -1
    with pytest.raises(ValueError, match='Supported/masked'):
        d.row_quantities(rows, p)
    p = packet(rows, [[.1, .1, .5, .3]]); original = d.row_quantities(rows, p)
    shuffled = copy.deepcopy(rows); permuted = copy.deepcopy(p); order = [2, 1, 3, 0]
    shuffled[0]['candidate_types'] = [rows[0]['candidate_types'][i] for i in order]
    shuffled[0]['current_label_index'] = order.index(3)
    shuffled[0]['previous_current_index'] = order.index(0)
    permuted['log_probs'][0, :4] = p['log_probs'][0, order]
    result = d.row_quantities(shuffled, permuted)
    for c in d.COMPONENTS: np.testing.assert_array_equal(original['loss'][c], result['loss'][c])
    assert result['error_code'].tolist() == original['error_code'].tolist()


def test_empty_means_and_paired_flips_with_distinct_rate_denominators():
    rows = rows_for((2, 2, 2), ('a', 'a', 'b'))
    base = d.row_quantities(rows, packet(rows, [[.1, .1, .7, .1], [.1, .1, .7, .1], [.7, .1, .1, .1]]))
    candidate = d.row_quantities(rows, packet(rows, [[.7, .1, .1, .1], [.1, .1, .7, .1], [.1, .1, .7, .1]]))
    selected = np.ones(3, bool)
    result = d.paired_group(rows, base, candidate, selected)
    assert result['correctness_counts'] == {'CC': 1, 'CW': 1, 'WC': 1, 'WW': 0}
    assert result['accuracy_difference'] == 0
    assert result['repair_rate'] == 1 and result['harm_rate'] == .5
    empty = d.paired_group(rows, base, candidate, ~selected)
    assert empty['accuracy_difference'] is empty['repair_rate'] is empty['harm_rate'] is None
    assert all(v is None for v in empty['loss_difference']['total'].values())
    for side in (base, candidate):
        cell = d.describe(rows, side, ~selected)
        assert cell['rows'] == sum(cell['decisions'].values()) == 0
        assert all(v is None for v in cell['loss']['branch'].values())
    contributions = d.contributions(rows, base, candidate, selected)
    for partition in contributions.values():
        assert sum(cell['rows'] for cell in partition.values()) == 3
        for c in d.COMPONENTS:
            for weight in ('row', 'equal_service'):
                assert sum(cell['loss_difference'][c][weight] for cell in partition.values()) == pytest.approx(result['loss_difference'][c][weight])
    # Same errors cancel in row weighting but not in equal-service weighting.
    assert result['loss_difference']['total']['row'] == pytest.approx(0)
    assert result['loss_difference']['total']['equal_service'] != pytest.approx(0)


def synthetic_published(rows, packets):
    output = {'fits': {}, 'continuation_allowed': False, 'continuation': {'checks_passed': 5, 'total_checks': 9}}
    masks = d.groups(rows)
    for name, p in packets.items():
        values, _, _ = d.report.predictions(rows, p)
        output['fits'][name] = {'cells': {key: d.report.describe(values, d.report.layout(rows, mask))
                                        for key, mask in masks.items() if key.split('/')[0] in ('all', 'heldout_service', 'seen_service')}}
    return output


def test_all_twelve_and_all_four_contrasts_preserve_published_metrics():
    rows = rows_for((2, 0, 1), ('a', 'a', 'b')); rows[-1]['heldout_service'] = False
    packets = {name: packet(rows) for name in d.report.FITS}
    published = synthetic_published(rows, packets)
    result = d.aggregate(rows, packets, published)
    assert set(result['fits']) == d.report.FITS
    assert set(result['pairs']) == set(d.PAIRS)
    for pair in result['pairs'].values():
        assert set(pair) == {'6101', '6102', '6103'}
        assert all(p['groups']['heldout_service/changed']['accuracy_difference'] == 0 for p in pair.values())
    bad = copy.deepcopy(published)
    bad['fits']['flat_balanced-6101']['cells']['heldout_service/changed']['nll']['row'] += 1
    with pytest.raises(ValueError, match='Published raw NLL'): d.aggregate(rows, packets, bad)
    packets.pop('typed_balanced-6101')
    with pytest.raises(ValueError, match='Every final fit'): d.aggregate(rows, packets, published)


def test_initialization_failure_preserves_original_even_if_receipt_write_fails(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    def fail_digest(_): raise ValueError('original initialization failure')
    monkeypatch.setattr(d.report, 'digest', fail_digest)
    destination = tmp_path/'recorded'
    with pytest.raises(ValueError, match='original initialization failure'):
        d.execute(Namespace(out=str(destination)))
    assert 'original initialization failure' in json.loads((destination/'failed.json').read_text())['error']
    def fail_write(*_): raise OSError('receipt write failure')
    monkeypatch.setattr(d.report, 'write', fail_write)
    with pytest.raises(ValueError, match='original initialization failure') as exc:
        d.execute(Namespace(out=str(tmp_path/'unwritable')))
    assert any('receipt write failure' in note for note in exc.value.__notes__)
