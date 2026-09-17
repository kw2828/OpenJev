import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('minigrid')
pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cue_study_test', ROOT / 'scripts/cue_memory_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_protocol_substitutes_only_explicit_training_settings():
    original = importlib.util.spec_from_file_location('original_memory_test', ROOT / 'scripts/recurrent_world_study.py')
    module = importlib.util.module_from_spec(original)
    original.loader.exec_module(module)
    changed = {key for key in module.PROTOCOL if study.PROTOCOL[key] != module.PROTOCOL[key]}
    assert changed == {'version', 'seeds', 'training_seed_base', 'evaluation_seed_start',
                       'probe_seed_start', 'fit_order_seed', 'claim'}
    assert module.MemoryBatch is study.MemoryBatch
    assert study.base.MemoryBatch is study.CueMemoryBatch
    assert study.base.fit.__code__.co_code == module.fit.__code__.co_code


@pytest.mark.parametrize('size', [11, 17, 23])
def test_swapped_cue_only_changes_cue_and_keeps_reward_target(size):
    with study.CueMemoryBatch(4, size, 67000) as native, study.SwappedCueBatch(4, size, 67000) as swapped:
        first, second = native.reset(), swapped.reset()
        assert np.all(np.any(first != second, axis=-1))
        for a, b in zip(native.envs, swapped.envs, strict=True):
            assert a.success_pos == b.success_pos
            assert a.failure_pos == b.failure_pos
            assert a.step_count == b.step_count == 0
            np.testing.assert_array_equal(a.agent_pos, b.agent_pos)
            assert a.agent_dir == b.agent_dir
            diffs = np.any(a.grid.encode() != b.grid.encode(), axis=-1)
            assert diffs.sum() == 1
            assert diffs[1, size // 2 - 1]
        # A fresh reset must regenerate the map before swapping, never accumulate swaps.
        third, fourth = native.reset(), swapped.reset()
        assert np.all(np.any(third != fourth, axis=-1))


def test_branch_reversal_counts_timeouts_in_denominator():
    first = {'episodes': [{'seed': 1, 'success': True, 'timeout': False},
                          {'seed': 2, 'success': True, 'timeout': False},
                          {'seed': 3, 'success': False, 'timeout': True}]}
    second = {'episodes': [{'seed': 1, 'success': False, 'timeout': False},
                           {'seed': 2, 'success': False, 'timeout': True},
                           {'seed': 3, 'success': True, 'timeout': False}]}
    result = study.branch_reversal(first, second)
    assert result['assigned_pairs'] == 3
    assert result['both_reach_branch'] == result['reversed_branch'] == 1
    assert result['branch_reversal_fraction'] == pytest.approx(1 / 3)


def test_memory_gate_requires_consistent_fits_and_reset_effect():
    summary = {'averages': {arm: {'11': {'success': value}} for arm, value in
                           [('recurrent_ppo', .9), ('current_ppo', .5), ('recurrent_ppo_reset', .5)]},
               'per_fit': {'recurrent_ppo': {'11': [{'success': .9}] * 3}}}
    assert all(study.memory_gate(summary).values())
    summary['per_fit']['recurrent_ppo']['11'][0] = {'success': .69}
    assert not study.memory_gate(summary)['each_fit_same_size_success']
    summary['averages']['recurrent_ppo_reset']['11']['success'] = .9
    assert not study.memory_gate(summary)['drop_on_reset']
