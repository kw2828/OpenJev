"""Fabricated training integration, with no generator or timing assertions."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_joint_reuse_training as current
import run_finite_update_learning as frozen

SEED = 943101
CONFIG = {'batch_size': 2, 'learning_rate': .003, 'gradient_clip': 5.,
          'prefix_updates': 2, 'joint_updates': 4, 'fit_cap_seconds': 30.}
ATOL, RTOL = 1e-9, 1e-9


def fabricated():
    endings = (None, 1, 3, 8, None)
    prefix = torch.zeros((5, 9, 31), dtype=torch.float32)
    lengths = torch.tensor([9, 2, 4, 9, 9], dtype=torch.int64)
    rows = torch.tensor([0, -1, -1, -1, 1], dtype=torch.int64)
    for row, end in enumerate(endings):
        prefix[row, 0, 4 + row % 4] = prefix[row, 0, 9] = 1
        for step in range(1, int(lengths[row])):
            prefix[row, step, (row + step) % 4] = 1
            prefix[row, step, 8 if step == end else 4 + (row + 2 * step) % 4] = 1
    active = torch.arange(9)[None] < lengths[:, None]
    indices = torch.tensor([0, 4], dtype=torch.int64)
    data = {'prefix': prefix[indices].clone(), 'lengths': lengths[indices].clone(),
            'actions': torch.tensor([[0, 2], [1, 3]], dtype=torch.int64),
            'observations': torch.tensor([[1, 4], [2, 0]], dtype=torch.int64),
            'blind_costs': torch.tensor([.21, -.11, .03, -.13], dtype=torch.float64).expand(2, 2, 4).clone(),
            'observed_costs': torch.tensor([-.17, .19, -.03, .01], dtype=torch.float64).expand(2, 2, 4).clone(),
            'blind_survival': torch.tensor([[.91, .82], [.93, .86]], dtype=torch.float64),
            'observed_survival': torch.tensor([[.89, .87], [.92, .9]], dtype=torch.float64),
            'observed_probabilities': torch.tensor([.17, .22, .25, .29, .07], dtype=torch.float64).expand(2, 2, 5).clone()}
    prefixes = {'prefix': prefix, 'lengths': lengths, 'event_mask': active, 'endpoint_rows': rows}
    return data, prefixes


def execute(folder, arm, implementation):
    folder.mkdir()
    data, prefixes = fabricated()
    counts, work = current.new_counts(), current.new_structural_work()
    if implementation == 'frozen':
        model, row = frozen.train(arm, SEED, data, prefixes, CONFIG, folder, counts, work, lambda: None)
    else:
        model, row = current.train(arm, SEED, data, prefixes, CONFIG, folder, counts, work,
                                   lambda: None, implementation=implementation)
    allocation = json.loads((folder / row['allocation']['path']).read_text())
    return {'model': model, 'row': row, 'allocation': allocation, 'counts': counts, 'work': work,
            'folder': folder, 'prefixes': prefixes}


def compare_adam(first, second, *, exact):
    assert first['param_groups'] == second['param_groups']
    assert first['state'].keys() == second['state'].keys()
    for key, state in first['state'].items():
        assert state.keys() == second['state'][key].keys()
        for name, value in state.items():
            other = second['state'][key][name]
            assert value['kind'] == other['kind'] == 'tensor'
            assert value['shape'] == other['shape'] and value['dtype'] == other['dtype']
            np.testing.assert_allclose(value['values'], other['values'], atol=0 if exact else ATOL,
                                       rtol=0 if exact else RTOL)


def compare_runs(first, second, *, exact):
    assert first['counts'] == second['counts']
    a, b = first['allocation'], second['allocation']
    for key in ('status', 'termination', 'arm', 'prefix_updates', 'joint_updates', 'max_seconds',
                'accepted_updates', 'attempted_updates', 'accepted_prefix_updates', 'accepted_joint_updates',
                'joint_cursor', 'metadata', 'work', 'update_work', 'final_summary_work'):
        assert a[key] == b[key]
    for left, right in zip(a['trace'], b['trace'], strict=True):
        for key in ('kind', 'stage', 'accepted', 'rolled_back', 'cursor_before', 'cursor_attempted', 'cursor_retained',
                    'work_delta'):
            assert left[key] == right[key]
        assert left['result']['diagnostics'] == right['result']['diagnostics']
        assert left['result']['work'] == right['result']['work']
        assert left['result']['loss'] == pytest.approx(right['result']['loss'], abs=0 if exact else 1e-10,
                                                     rel=0 if exact else 1e-10)
    assert a['boundary']['optimizer_reset'] is b['boundary']['optimizer_reset'] is True
    assert a['stages'][1]['joint_cursor_start'] == b['stages'][1]['joint_cursor_start'] == 0
    for label in ('initial', 'boundary', 'final'):
        stem = f'{label}-{first["row"]["arm"]}-{SEED}'
        with np.load(first['folder'] / (stem + '.npz'), allow_pickle=False) as left, \
                np.load(second['folder'] / (stem + '.npz'), allow_pickle=False) as right:
            assert left.files == right.files
            for name in left.files:
                np.testing.assert_allclose(left[name], right[name], atol=0 if exact or label != 'final' else ATOL,
                                           rtol=0 if exact or label != 'final' else RTOL)
        name = f'{label}-optimizer-{first["row"]["arm"]}-{SEED}.json'
        compare_adam(json.loads((first['folder'] / name).read_text()),
                     json.loads((second['folder'] / name).read_text()), exact=exact or label != 'final')
    assert (first['folder'] / 'training-orders.jsonl').read_bytes() == (second['folder'] / 'training-orders.jsonl').read_bytes()


@pytest.mark.parametrize('arm', current.ARMS)
def test_integrated_separate_and_reuse_match_frozen_training_with_empty_and_partial_batches(tmp_path, arm):
    original = execute(tmp_path / 'original', arm, 'frozen')
    separate = execute(tmp_path / 'separate', arm, 'separate')
    reuse = execute(tmp_path / 'reuse', arm, 'reuse')
    compare_runs(original, separate, exact=True)
    compare_runs(original, reuse, exact=False)
    assert original['work'] == separate['work']
    assert separate['row']['implementation'] == 'separate' and reuse['row']['implementation'] == 'reuse'
    assert reuse['row']['integration_version'] == current.VERSION
    assert reuse['counts']['accepted_prefix_steps'] == 2 and reuse['counts']['accepted_joint_steps'] == 4
    assert reuse['counts']['optimizer_steps'] == 6 and reuse['counts']['fit_count'] == 1
    joint = [row for row in reuse['allocation']['trace'] if row['kind'] == 'joint']
    assert len(joint) == 4
    assert any(row['result']['diagnostics']['eligible'] == 0 for row in joint)
    assert any(row['result']['diagnostics']['batch_size'] == 1 for row in joint)
    for cursor, row in enumerate(joint):
        epoch, offset = divmod(cursor, 3)
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, epoch, 818]))).permutation(5)
        assert row['result']['diagnostics']['indices'] == order[offset * 2:(offset + 1) * 2].tolist()
    initial = np.load(reuse['folder'] / f'initial-{arm}-{SEED}.npz', allow_pickle=False)
    boundary = np.load(reuse['folder'] / f'boundary-{arm}-{SEED}.npz', allow_pickle=False)
    final = np.load(reuse['folder'] / f'final-{arm}-{SEED}.npz', allow_pickle=False)
    try:
        np.testing.assert_array_equal(initial['cost_logits'], boundary['cost_logits'])
        assert not np.array_equal(boundary['cost_logits'], final['cost_logits'])
        assert any(not np.array_equal(initial[name], boundary[name]) for name in current.DYNAMICS)
    finally:
        initial.close()
        boundary.close()
        final.close()
    optimizer = json.loads((reuse['folder'] / f'final-optimizer-{arm}-{SEED}.json').read_text())
    assert len(optimizer['state']) == 4
    assert all(value['step']['values'] == 4 for value in optimizer['state'].values())
    assert all(np.count_nonzero(value['exp_avg']['values']) > 0 for value in optimizer['state'].values())
    actual = reuse['work'][arm]
    assert actual['training_blind'] == actual['training_observed'] == {}
    assert actual['training_prefix']['probability_field_calls'] == 2  # Prefix stage only.
    assert actual['joint_reuse_shared']['probability_field_calls'] == 4
    assert actual['joint_reuse_shared']['factorization_calls'] == 4
    assert actual['joint_reuse_shared']['rounded_transition_calls'] == (4 if arm == 'rounded' else 0)
    nonempty = sum(row['result']['diagnostics']['eligible'] > 0 for row in joint)
    endpoints = sum(row['result']['diagnostics']['eligible'] for row in joint)
    assert actual['joint_reuse_shared']['operator_marginal_sum_calls'] == nonempty
    assert actual['joint_reuse_endpoint_prefix']['reset_emission_rows'] == endpoints
    assert actual['joint_reuse_endpoint_prefix']['prefix_filter_rows'] == 8 * endpoints
    assert actual['joint_reuse_prefix_nll']['prefix_nll_rows'] == sum(
        row['result']['diagnostics']['valid_events'] for row in joint)
    assert actual['joint_reuse_blind']['cost_readout_rows'] == actual['joint_reuse_observed']['cost_readout_rows'] == 2 * endpoints
    assert all(not block for route, block in separate['work'][arm].items() if route in current.REUSE_ROUTES)


@pytest.mark.parametrize('mode', ['', 'cached', None, 0, True])
def test_mode_rejection_precedes_controller_model_or_output(tmp_path, mode):
    counts, work = current.new_counts(), current.new_structural_work()
    with pytest.raises(ValueError, match='explicit separate or reuse'):
        current.train('rounded', SEED, {}, {}, {}, tmp_path, counts, work, lambda: None, implementation=mode)
    assert not any(counts.values()) and list(tmp_path.iterdir()) == []


def test_mode_is_explicit_not_a_silent_default(tmp_path):
    with pytest.raises(TypeError, match='implementation'):
        current.train('rounded', SEED, {}, {}, {}, tmp_path, {}, {}, lambda: None)


def test_frozen_prefix_optimizer_snapshot_checkpoint_functions_are_unchanged_source():
    def methods(path):
        tree = ast.parse(Path(path).read_text())
        train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
        return {node.name: node for node in train.body if isinstance(node, ast.FunctionDef)}
    old, new = methods(frozen.__file__), methods(current.__file__)
    for name in ('validate', 'optimizer', 'snapshot_model', 'restore_model', 'snapshot_optimizer', 'checkpoint'):
        assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False)
    old_update = old['update']
    old_joint = next(node for node in old_update.body if isinstance(node, ast.If)).orelse
    first = next(i for i, node in enumerate(old_joint) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'loss')
    last = next(i for i, node in enumerate(old_joint) if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'batch_events')
    new_joint = next(node for node in new['update'].body if isinstance(node, ast.If)).orelse
    branch = next(node for node in new_joint if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                  and node.test.left.id == 'implementation')
    assert [ast.dump(node, include_attributes=False) for node in old_joint[first:last]] == [
        ast.dump(node, include_attributes=False) for node in branch.body]


def test_original_failure_preserves_exact_allocation_and_partial_reuse_work(tmp_path, monkeypatch):
    marker = TimeoutError('fabricated original failure')
    marker.allocation_result = {'status': 'FAILED_TIMEOUT', 'trace': [{'accepted': False}]}
    marker.joint_reuse_work = {'shared': {'probability_field_calls': 1}}
    marker.rounded_model_work = {'probability_field_calls': 1}

    def fail(*args, **kwargs):
        raise marker

    monkeypatch.setattr(current, 'run_update_allocation', fail)
    with pytest.raises(TimeoutError) as caught:
        current.train('rounded', SEED, {}, {}, CONFIG, tmp_path, current.new_counts(),
                      current.new_structural_work(), lambda: None, implementation='reuse')
    assert caught.value is marker
    assert json.loads((tmp_path / f'allocation-rounded-{SEED}.json').read_text()) == marker.allocation_result
    assert json.loads((tmp_path / f'failed-joint-reuse-rounded-{SEED}.json').read_text()) == marker.joint_reuse_work
    assert json.loads((tmp_path / f'failed-normalization-rounded-{SEED}.json').read_text()) == marker.rounded_model_work
    assert not (tmp_path / 'fits.jsonl').exists()
