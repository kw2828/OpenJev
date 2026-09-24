"""Fabricated six-arm trainer qualification; no generated or empirical inputs."""
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
import run_finite_action_range_training as current
import run_finite_head_training as frozen

from openjev.research.finite_action_range_loss import HEAD_ARMS, LOSS_KINDS, LOSS_WORK_KEYS, loss_work_counts
from openjev.research.finite_head_initialization import expected_head_work

SEED = 948101
CONFIG = {'batch_size': 2, 'learning_rate': .003, 'gradient_clip': 5.,
          'prefix_updates': 2, 'joint_updates': 3, 'fit_cap_seconds': 30.}


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


def test_six_loss_fits_preserve_prefix_pairing_mse_trajectory_and_charged_work(tmp_path):
    runs = {arm: execute(tmp_path / arm, arm, current) for arm in current.ARMS}
    references = {transport: execute(tmp_path / f'parent-{transport}', HEAD_ARMS[f'{transport}_mse'], frozen)
                  for transport in ('rounded', 'free')}
    for transport in ('rounded', 'free'):
        baseline = runs[f'{transport}_mse']
        parent = references[transport]
        assert baseline['counts'] == parent['counts']
        assert baseline['work'][f'{transport}_mse'] == parent['work'][HEAD_ARMS[f'{transport}_mse']]
        for label in ('initial', 'boundary', 'final'):
            for name in baseline['states'][label]:
                np.testing.assert_array_equal(baseline['states'][label][name], parent['states'][label][name])
            assert baseline['optimizers'][label] == parent['optimizers'][label]
        for left, right in zip(baseline['allocation']['trace'], parent['allocation']['trace'], strict=True):
            assert left['result']['loss'] == right['result']['loss']
            assert left['result']['work'] == right['result']['work']
            assert left['model_retained_sha256'] == right['model_retained_sha256']
            assert left['optimizer_retained_sha256'] == right['optimizer_retained_sha256']
            diagnostics = dict(left['result']['diagnostics'])
            if left['kind'] == 'joint':
                assert diagnostics.pop('loss_kind') == 'mse'
                assert diagnostics.pop('loss_work')['wrapper_calls'] == 1
            assert diagnostics == right['result']['diagnostics']
        for kind in ('double', 'range'):
            variant = runs[f'{transport}_{kind}']
            for label in ('initial', 'boundary'):
                for name in baseline['states'][label]:
                    np.testing.assert_array_equal(baseline['states'][label][name], variant['states'][label][name])
                assert baseline['optimizers'][label] == variant['optimizers'][label]
            assert [row['result']['loss'] for row in baseline['allocation']['trace'][:2]] == [
                row['result']['loss'] for row in variant['allocation']['trace'][:2]]
    all_orders = []
    for arm, run in runs.items():
        row, allocation = run['row'], run['allocation']
        assert row['integration_version'] == current.VERSION and row['implementation'] == 'reuse'
        assert row['loss_kind'] == LOSS_KINDS[arm] and row['updates'] == 3 and row['accepted_prefix_updates'] == 2
        assert row['attempted_updates'] == 5 and row['parameter_metadata']['parameter_count'] == 352
        assert row['model_metadata']['privileged_readout_initialization'] is False
        assert row['model_metadata']['arm'] == arm and row['model_metadata']['head_arm'] == HEAD_ARMS[arm]
        assert row['head_initialization_work'] == expected_head_work(HEAD_ARMS[arm])
        assert allocation['metadata']['loss_work'] == loss_work_counts()
        assert allocation['metadata']['loss_kind'] == allocation['final_summary']['loss_kind'] == LOSS_KINDS[arm]
        for exported in (allocation['metadata'], allocation['final_summary']):
            assert exported['model_metadata'] == row['model_metadata']
            assert exported['parameter_metadata'] == row['parameter_metadata']
        joint = [item for item in allocation['trace'] if item['kind'] == 'joint']
        all_orders.append([item['result']['diagnostics']['indices'] for item in joint])
        totals = {key: sum(item['result']['diagnostics']['loss_work'][key] for item in joint) for key in LOSS_WORK_KEYS}
        assert totals == row['loss_work'] == allocation['final_summary']['loss_work']
        assert totals['wrapper_calls'] == 3
        assert totals['zero_endpoint_batches'] == sum(item['result']['diagnostics']['eligible'] == 0 for item in joint)
        eligible_horizons = sum(item['result']['diagnostics']['eligible'] * 2 for item in joint)
        assert totals['blind_error_entries'] == (0 if LOSS_KINDS[arm] == 'mse' else 4 * eligible_horizons)
        assert totals['range_max_rows'] == (eligible_horizons if LOSS_KINDS[arm] == 'range' else 0)
        assert row['dynamics_boundary_hash_evaluations'] == row['head_boundary_hash_evaluations'] == 3
        for checkpoint in allocation['checkpoints']:
            label, meta = checkpoint['label'], checkpoint['metadata']
            dynamics = state_digest({name: run['states'][label][name] for name in current.DYNAMICS})
            head = state_digest({'cost_logits': run['states'][label]['cost_logits']})
            assert meta['dynamics_state_sha256'] == row['dynamics_boundary_sha256'][label] == dynamics
            assert meta['head_state_sha256'] == row['head_boundary_sha256'][label] == head
        assert row['head_boundary_sha256']['initial'] == row['head_boundary_sha256']['boundary']
        assert row['head_boundary_sha256']['boundary'] != row['head_boundary_sha256']['final']
        assert len(run['optimizers']['boundary']['state']) == 3 and len(run['optimizers']['final']['state']) == 4
        assert all(state['step']['values'] == 3 for state in run['optimizers']['final']['state'].values())
        assert run['counts']['optimizer_steps'] == run['counts']['accepted_optimizer_steps'] == 5
    assert all(order == all_orders[0] for order in all_orders)
    for arm, run in runs.items():
        np.testing.assert_array_equal(run['states']['initial']['cost_logits'], runs['rounded_mse']['states']['initial']['cost_logits'])
    # Matching transport is functional, not equality of the raw transition logits.
    rounded = current.learned_model('rounded_mse', SEED).probability_fields()['transition']
    free = current.learned_model('free_mse', SEED).probability_fields()['transition']
    torch.testing.assert_close(rounded, free, atol=1e-12, rtol=0)


def test_optimizer_and_state_snapshot_primitives_are_unchanged_source():
    def methods(path):
        tree = ast.parse(Path(path).read_text())
        train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
        return {node.name: node for node in train.body if isinstance(node, ast.FunctionDef)}
    old, new = methods(frozen.__file__), methods(current.__file__)
    for name in ('optimizer', 'snapshot_model', 'restore_model', 'snapshot_optimizer', 'checkpoint'):
        assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False)


@pytest.mark.parametrize('mode', ['separate', '', None, 0, True])
def test_only_explicit_reuse_is_admitted_before_construction(tmp_path, mode):
    counts, work = current.new_counts(), current.new_structural_work()
    with pytest.raises(ValueError, match='explicit reuse'):
        current.train('rounded_mse', SEED, {}, {}, {}, tmp_path, counts, work, lambda: None, implementation=mode)
    assert not any(counts.values()) and not list(tmp_path.iterdir())


def test_failure_preserves_loss_and_original_partial_work(tmp_path, monkeypatch):
    marker = TimeoutError('fabricated original failure')
    marker.allocation_result = {'status': 'FAILED_TIMEOUT', 'trace': [{'accepted': False}]}
    marker.joint_reuse_work = {'shared': {'probability_field_calls': 1}}
    marker.rounded_model_work = {'probability_field_calls': 1}
    marker.action_range_loss_work = loss_work_counts()
    marker.action_range_loss_work['wrapper_calls'] = 1
    def fail(*args, **kwargs):
        raise marker
    monkeypatch.setattr(current, 'run_update_allocation', fail)
    with pytest.raises(TimeoutError) as caught:
        current.train('rounded_range', SEED, {}, {}, CONFIG, tmp_path, current.new_counts(),
                      current.new_structural_work(), lambda: None, implementation='reuse')
    assert caught.value is marker
    for name, expected in (('allocation', marker.allocation_result),
                           ('failed-joint-reuse', marker.joint_reuse_work),
                           ('failed-normalization', marker.rounded_model_work),
                           ('failed-action-range-loss', marker.action_range_loss_work)):
        assert json.loads((tmp_path / f'{name}-rounded_range-{SEED}.json').read_text()) == expected
    assert not (tmp_path / 'fits.jsonl').exists()

