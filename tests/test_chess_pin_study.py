# SPDX-License-Identifier: GPL-3.0-only
import copy

import numpy as np
import pytest
import torch

from openjev.research import chess_pin_study as study
from openjev.research.chess_child_graph_cache import ChildGraphCache, write_cache
from openjev.research.chess_training_inputs import TrainingInputs
from test_chess_training_inputs import fixture


def native_inputs(tmp_path):
    boards, _, features, pins = fixture()
    write_cache([{'fen': b.fen(en_passant='fen')} for b in boards], tmp_path/'graphs', provenance={'test': True})
    data = TrainingInputs(features, pins); graph = ChildGraphCache(tmp_path/'graphs').batch([2, 1, 0])
    return data.batch(torch.tensor([2, 1, 0]), graph)


@pytest.mark.parametrize('arm', list(study.PARAMETERS))
def test_native_initialization_and_three_finite_artificial_updates(tmp_path, arm):
    args, factors = native_inputs(tmp_path); rng = torch.get_rng_state().clone()
    model = study.make_head(arm, 97)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(study.logits(model, arm, args, factors), args[4])
    opt = study.optimizer(model); initial = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(3):
        result = study.update(model, arm, opt, args, factors, torch.zeros(3, dtype=torch.long))
        assert result['loss'] >= 0 and result['gradient_norm'] >= 0 and result['update_seconds'] > 0
    assert not torch.equal(initial['output.weight'], model.output.weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with pytest.raises(ValueError): study.logits(model, 'unknown', args, factors)


def test_illegal_target_rejected_before_weights_change_and_arm_mismatch_rejected(tmp_path):
    args, factors = native_inputs(tmp_path); model = study.make_head('joint', 97); opt = study.optimizer(model)
    before = copy.deepcopy(model.state_dict()); targets = torch.zeros(3, dtype=torch.long); targets[1] = args[3].shape[1]-1
    assert not args[3][1, targets[1]]
    with pytest.raises(ValueError): study.update(model, 'joint', opt, args, factors, targets)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    for arm in ('separable', 'wldn', 'counts', 'pairwise', 'union:child'):
        with pytest.raises(ValueError): study.logits(model, arm, args, factors)


def test_schedule_matches_frozen_historical_rule_without_changing_rng():
    rng = torch.get_rng_state().clone(); batches = list(study.schedule(32768, 6, 128, 109))
    assert len(batches) == 1536 and torch.equal(rng, torch.get_rng_state())
    for epoch in range(6):
        actual = torch.cat([index for e, index in batches if e == epoch]).numpy()
        expected = np.random.default_rng(700000+100*109+epoch).permutation(32768)
        assert np.array_equal(actual, expected)
    assert study.index_digest(batches[0][1]) != study.index_digest(batches[1][1])
    with pytest.raises(ValueError): list(study.schedule(3, 1, 2, 97))


def metrics():
    return {f'{arm}-{seed}': {s: {'agreement': .3, 'examples': 2048} for s in ('dev', 'shift')}
            for arm in ('wldn', *study.UNION_ARMS) for seed in study.SEEDS}


def test_comparator_selection_keeps_all_core_controls_and_panel_winners_with_deterministic_ties():
    m = metrics(); assert study.select_comparators(m)['fit_arms'] == list(study.CORE_ARMS)
    for seed in study.SEEDS:
        m[f'child-{seed}']['dev']['agreement'] = .31
        m[f'edits-{seed}']['shift']['agreement'] = .32
    result = study.select_comparators(m)
    assert result['additional_union_arms'] == ['union:child', 'union:edits']
    assert result['fit_arms'][:7] == list(study.CORE_ARMS)
    # The helper uses all completed arm metrics; a failed study gate cannot erase a winner.
    for seed in study.SEEDS: m[f'rotated-{seed}']['dev']['agreement'] = .4
    assert study.select_comparators(m)['additional_union_arms'] == ['union:edits', 'union:rotated']
    m['wldn-97']['dev']['agreement'] = float('nan')
    with pytest.raises(ValueError): study.select_comparators(m)
