"""Fabricated head intervention through the unchanged qualified fit arithmetic."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_head_training as current
import run_finite_joint_reuse_training as frozen

from openjev.research.finite_head_initialization import expected_head_work

SEED = 946101
CONFIG = {'batch_size': 2, 'learning_rate': .003, 'gradient_clip': 5.,
          'prefix_updates': 2, 'joint_updates': 4, 'fit_cap_seconds': 30.}


def fabricated():
    endings = (None, 1, 3, 8, None)
    prefix = torch.zeros((5, 9, 31), dtype=torch.float32)
    lengths = torch.tensor([9, 2, 4, 9, 9], dtype=torch.int64)
    endpoint_rows = torch.tensor([0, -1, -1, -1, 1], dtype=torch.int64)
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
    return data, {'prefix': prefix, 'lengths': lengths, 'event_mask': active, 'endpoint_rows': endpoint_rows}


def state_digest(arrays):
    result = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        header = json.dumps([name, value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
        result.update(len(header).to_bytes(8, 'little'))
        result.update(header)
        result.update(value.tobytes(order='C'))
    return result.hexdigest()


def execute(folder, arm, module):
    folder.mkdir()
    data, prefixes = fabricated()
    counts, work = module.new_counts(), module.new_structural_work()
    model, row = module.train(arm, SEED, data, prefixes, CONFIG, folder, counts, work,
                              lambda: None, implementation='reuse')
    allocation = json.loads((folder / row['allocation']['path']).read_text())
    states, optimizers = {}, {}
    for label in ('initial', 'boundary', 'final'):
        with np.load(folder / f'{label}-{arm}-{SEED}.npz', allow_pickle=False) as archive:
            states[label] = {name: archive[name].copy() for name in archive.files}
        optimizers[label] = json.loads((folder / f'{label}-optimizer-{arm}-{SEED}.json').read_text())
    return {'model': model, 'row': row, 'allocation': allocation, 'counts': counts, 'work': work,
            'states': states, 'optimizers': optimizers}


def test_three_arm_fit_preserves_parent_and_prefix_integrity_with_real_joint_updates(tmp_path):
    runs = {arm: execute(tmp_path / arm, arm, current) for arm in current.ARMS}
    parent = execute(tmp_path / 'parent', 'rounded', frozen)
    anchor, random = runs['rounded_anchor'], runs['rounded_random']
    assert anchor['counts'] == parent['counts']
    assert anchor['work']['rounded_anchor'] == parent['work']['rounded']
    for label in ('initial', 'boundary', 'final'):
        for name in anchor['states'][label]:
            np.testing.assert_array_equal(anchor['states'][label][name], parent['states'][label][name])
        assert anchor['optimizers'][label] == parent['optimizers'][label]
    for a, b in zip(anchor['allocation']['trace'], parent['allocation']['trace'], strict=True):
        assert a['result'] == b['result']
        assert a['model_before_sha256'] == b['model_before_sha256'] and a['model_retained_sha256'] == b['model_retained_sha256']
        assert a['optimizer_before_sha256'] == b['optimizer_before_sha256'] and a['optimizer_retained_sha256'] == b['optimizer_retained_sha256']
    for label in ('initial', 'boundary'):
        for name in current.DYNAMICS:
            np.testing.assert_array_equal(anchor['states'][label][name], random['states'][label][name])
        assert anchor['optimizers'][label] == random['optimizers'][label]
    assert [r['result']['loss'] for r in anchor['allocation']['trace'][:2]] == [
        r['result']['loss'] for r in random['allocation']['trace'][:2]]
    np.testing.assert_array_equal(random['states']['initial']['cost_logits'],
                                  runs['matched_free_random']['states']['initial']['cost_logits'])
    joint_orders = []
    for arm, run in runs.items():
        row, allocation = run['row'], run['allocation']
        assert row['integration_version'] == current.VERSION and row['implementation'] == 'reuse'
        assert row['updates'] == 4 and row['accepted_prefix_updates'] == 2 and row['attempted_updates'] == 6
        assert row['parameter_metadata']['parameter_count'] == 352
        assert row['head_initialization_work'] == expected_head_work(arm)
        assert row['dynamics_boundary_hash_evaluations'] == row['head_boundary_hash_evaluations'] == 3
        for exported in (allocation['metadata'], allocation['final_summary']):
            assert exported['model_metadata'] == row['model_metadata']
            assert exported['parameter_metadata'] == row['parameter_metadata']
            assert exported['head_initialization_work'] == expected_head_work(arm)
        assert row['model_metadata']['privileged_readout_initialization'] is (arm == 'rounded_anchor')
        for checkpoint in allocation['checkpoints']:
            label, meta = checkpoint['label'], checkpoint['metadata']
            dynamics = state_digest({name: run['states'][label][name] for name in current.DYNAMICS})
            head = state_digest({'cost_logits': run['states'][label]['cost_logits']})
            assert meta['dynamics_state_sha256'] == row['dynamics_boundary_sha256'][label] == dynamics
            assert meta['head_state_sha256'] == row['head_boundary_sha256'][label] == head
        assert row['head_boundary_sha256']['initial'] == row['head_boundary_sha256']['boundary']
        assert row['head_boundary_sha256']['boundary'] != row['head_boundary_sha256']['final']
        np.testing.assert_array_equal(run['states']['initial']['cost_logits'], run['states']['boundary']['cost_logits'])
        assert any(not np.array_equal(run['states']['initial'][name], run['states']['boundary'][name])
                   for name in current.DYNAMICS)
        assert all(not np.array_equal(run['states']['boundary'][name], run['states']['final'][name])
                   for name in run['states']['final'])
        assert len(run['optimizers']['boundary']['state']) == 3
        assert len(run['optimizers']['final']['state']) == 4
        assert all(item['step']['values'] == 4 for item in run['optimizers']['final']['state'].values())
        joint = [r for r in allocation['trace'] if r['kind'] == 'joint']
        joint_orders.append([r['result']['diagnostics']['indices'] for r in joint])
        for cursor, step in enumerate(joint):
            epoch, offset = divmod(cursor, 3)
            order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, epoch, 818]))).permutation(5)
            assert step['result']['diagnostics']['indices'] == order[offset * 2:(offset + 1) * 2].tolist()
        assert run['counts']['optimizer_steps'] == run['counts']['accepted_optimizer_steps'] == 6
        assert run['counts']['fit_count'] == 1
    assert joint_orders[0] == joint_orders[1] == joint_orders[2]


def test_numerical_update_and_optimizer_snapshot_source_are_identical_to_parent():
    def methods(path):
        tree = ast.parse(Path(path).read_text())
        train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
        return {node.name: node for node in train.body if isinstance(node, ast.FunctionDef)}
    old, new = methods(frozen.__file__), methods(current.__file__)
    for name in ('update', 'optimizer', 'snapshot_model', 'restore_model', 'snapshot_optimizer'):
        assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False)


@pytest.mark.parametrize('mode', ['separate', '', 'cached', None, 0, True])
def test_only_explicit_reuse_is_admitted_before_model_or_controller(tmp_path, mode):
    counts, work = current.new_counts(), current.new_structural_work()
    with pytest.raises(ValueError, match='explicit reuse'):
        current.train('rounded_anchor', SEED, {}, {}, {}, tmp_path, counts, work,
                      lambda: None, implementation=mode)
    assert not any(counts.values()) and list(tmp_path.iterdir()) == []


def test_reuse_has_no_silent_default(tmp_path):
    with pytest.raises(TypeError, match='implementation'):
        current.train('rounded_anchor', SEED, {}, {}, {}, tmp_path, {}, {}, lambda: None)


def test_failure_preserves_head_and_original_partial_work_without_success_row(tmp_path, monkeypatch):
    marker = TimeoutError('fabricated original failure')
    marker.allocation_result = {'status': 'FAILED_TIMEOUT', 'trace': [{'accepted': False}]}
    marker.joint_reuse_work = {'shared': {'probability_field_calls': 1}}
    marker.rounded_model_work = {'probability_field_calls': 1}
    marker.head_initialization_work = expected_head_work('rounded_random')
    def fail(*args, **kwargs):
        raise marker
    monkeypatch.setattr(current, 'run_update_allocation', fail)
    with pytest.raises(TimeoutError) as caught:
        current.train('rounded_random', SEED, {}, {}, CONFIG, tmp_path, current.new_counts(),
                      current.new_structural_work(), lambda: None, implementation='reuse')
    assert caught.value is marker
    for name, expected in (('allocation', marker.allocation_result),
                           ('failed-joint-reuse', marker.joint_reuse_work),
                           ('failed-normalization', marker.rounded_model_work),
                           ('failed-head-initialization', marker.head_initialization_work)):
        assert json.loads((tmp_path / f'{name}-rounded_random-{SEED}.json').read_text()) == expected
    assert not (tmp_path / 'fits.jsonl').exists()
