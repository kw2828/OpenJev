# SPDX-License-Identifier: GPL-3.0-only
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import chess_pin_quality_analysis_v2 as analysis


def metrics(arms):
    return {f'{a}-{seed}': {p: {'agreement': .40 if a == 'joint' else .37,
            'target_nll': 1., 'examples': 2048} for p in analysis.SPLITS}
            for a in ('base', *arms) for seed in analysis.SEEDS}


def test_all_comparators_panels_and_seed_floor_are_required():
    arms = (*analysis.CORE_ARMS, 'union:child', 'union:rotated')
    values = metrics(arms)
    checks = analysis.gate(values, arms)
    assert len(checks) == 18 and all(r['passed'] for r in checks)
    values['union:rotated-97']['shift']['agreement'] = .406
    checks = analysis.gate(values, arms)
    failed = [r for r in checks if not r['passed']]
    assert len(failed) == 1 and failed[0]['comparator'] == 'union:rotated'
    assert failed[0]['mean_gain'] > .01 and failed[0]['minimum_paired_seed_gain'] < -.005
    values = metrics(arms)
    for seed in analysis.SEEDS: values[f'pairwise-{seed}']['dev']['agreement'] = .391
    assert not next(r for r in analysis.gate(values, arms) if r['comparator'] == 'pairwise' and r['split'] == 'dev')['passed']
    for seed in analysis.SEEDS: values[f'base-{seed}']['dev']['agreement'] = .401
    assert not next(r for r in analysis.gate(values, arms) if r['comparator'] == 'base' and r['split'] == 'dev')['passed']


def test_contract_rejects_missing_controls_and_incomplete_metrics():
    arms = analysis.CORE_ARMS
    assert len(analysis.gate(metrics(arms), arms)) == 14
    for bad in (arms[1:], (*arms, 'union:rotated', 'union:child'), (*arms, 'union:child', 'union:child'),
                (*arms, 'union:child', 'union:union', 'union:edits'), (*arms, 'unknown')):
        with pytest.raises(ValueError): analysis.validate_arms(bad)
    for bad in ('missing', 'extra', 'nan', 'short', 'negative'):
        values = metrics(arms)
        if bad == 'missing': del values['wldn-97']
        if bad == 'extra': values['another-97'] = values['wldn-97']
        if bad == 'nan': values['joint-97']['dev']['agreement'] = float('nan')
        if bad == 'short': values['joint-97']['dev']['examples'] = 2047
        if bad == 'negative': values['joint-97']['dev']['target_nll'] = -1.
        with pytest.raises(ValueError): analysis.gate(values, arms)


def test_game_bootstrap_keeps_seed_pairing_and_position_weights():
    # Two unequal games: the first contributes 1/3 on one root, the second -1 on three.
    ids = ['a', 'b', 'b', 'b']
    delta = [[1, -1, -1, -1], [0, -1, -1, -1], [0, -1, -1, -1]]
    result = analysis.game_interval(delta, ids, 'dev')
    assert result['point_gain'] == pytest.approx(-2/3)
    assert result['percentile95'] == pytest.approx([-1., 1/3])
    assert result['games'] == 2 and result['roots'] == 4 and result['draws'] == 2000
    # Independent direct game concatenation reproduces all quantiles.
    draws = np.random.default_rng(996101).integers(0, 2, (2000, 2))
    source_games = [[1/3], [-1., -1., -1.]]
    brute = [np.mean(source_games[a]+source_games[b]) for a,b in draws]
    assert result['percentile95'] == pytest.approx(np.quantile(brute, [.025, .975]))
    assert analysis.game_interval(delta, ids, 'shift')['seed'] == 996102
    with pytest.raises(ValueError): analysis.game_interval(delta[:2], ids, 'dev')
    with pytest.raises(ValueError): analysis.game_interval([[.5]*4]*3, ids, 'dev')


def test_full_analysis_counts_records_and_rejects_identity_corruption():
    arms = analysis.CORE_ARMS
    rows = {p: [{'id': f'{p}:{i}', 'game_id': i//128, 'target_uci': 'a2a3'} for i in range(2048)]
            for p in analysis.SPLITS}
    records = {}
    for arm in ('base', *arms):
        for seed in analysis.SEEDS:
            for split in analysis.SPLITS:
                cutoff = 900 if arm == 'joint' else 800
                records[arm, seed, split] = [{'index': i, 'id': r['id'], 'game_id': r['game_id'],
                    'choice': 'a2a3' if i < cutoff else 'a2a4', 'correct': i < cutoff, 'target_nll': 1.}
                    for i,r in enumerate(rows[split])]
    result = analysis.analyze(records, rows, arms)
    assert result['prediction_records'] == 98304 and result['quality_gate_passed']
    assert len(result['gate_checks']) == len(result['conditional_game_bootstrap']) == 14
    assert all(r['mean_gain'] == 100/2048 for r in result['gate_checks'])
    records['joint',97,'dev'][0]['game_id'] = 'wrong'
    with pytest.raises(ValueError): analysis.analyze(records, rows, arms)


def test_integer_zero_ids_preserve_groups_without_coercion_or_bool_aliasing():
    delta = [[1,-1,-1,-1],[0,-1,-1,-1],[0,-1,-1,-1]]
    integer_ids = [0,600,600,600]
    integer = analysis.game_interval(delta,integer_ids,'dev')
    named = analysis.game_interval(delta,['a','b','b','b'],'dev')
    assert integer == named and integer_ids == [0,600,600,600]
    assert integer['point_gain'] == pytest.approx(-2/3)
    for bad in ([],[0,'0'],[False,0],[False,True],[0.,1.],['','a'],[None,None]):
        with pytest.raises(ValueError): analysis.validate_game_ids(bad)
