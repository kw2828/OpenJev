import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('connectome_parameter_audit', Path(__file__).resolve().parents[1]/'scripts/chess_connectome_parameter_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def state(seed=17):
    generator = torch.Generator().manual_seed(seed)
    weight, bias = torch.empty(2, 3, 1, 1), torch.empty(2)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
    nn.init.uniform_(bias, -1/math.sqrt(3), 1/math.sqrt(3), generator=generator)
    return {'input_projection.weight': weight, 'input_projection.bias': bias,
            'output_projection.weight': torch.zeros(3, 2, 1, 1), 'output_projection.bias': torch.zeros(3),
            'edge_log_gain': torch.full((5,), math.log(math.expm1(1.))), 'node_bias': torch.zeros(128),
            'node_slots': torch.arange(128), 'slot_occupancy': torch.ones(128)}


def test_exact_initialization_and_global_rng_preserved():
    inputs = state()
    rng = torch.get_rng_state().clone()
    report = audit.parameter_summary(inputs, 17)
    assert report['input_weight_change']['max'] == report['input_bias_change']['max'] == 0
    assert report['output_projection_is_zero']
    assert report['uncovered_slots'] == 0
    assert torch.equal(rng, torch.get_rng_state())


def test_known_parameter_movements_are_descriptive_not_forward_calls():
    inputs = state()
    inputs['output_projection.weight'][0, 0, 0, 0] = 2
    inputs['input_projection.bias'] += .25
    inputs['edge_log_gain'][:] = math.log(math.expm1(2.))
    report = audit.parameter_summary(inputs, 17)
    assert not report['output_projection_is_zero']
    assert report['output_weight']['max'] == 2
    assert report['input_bias_change']['mean'] == pytest.approx(.25)
    assert report['edge_magnitude']['mean'] == pytest.approx(2.)
    assert report['fraction_edge_magnitudes_changed_over_1e_6'] == 1


@pytest.mark.parametrize('values', [[], [float('nan')], [float('inf')]])
def test_invalid_statistics_fail(values):
    with pytest.raises(ValueError):
        audit.stats(torch.tensor(values))


def test_incorrect_occupancy_rejected():
    inputs = state()
    inputs['slot_occupancy'][0] = 0
    with pytest.raises(ValueError, match='occupancy'):
        audit.parameter_summary(inputs, 17)


def test_summary_matches_known_population_moments():
    report = audit.stats(torch.tensor([-2., 0., 2.]))
    assert report['count'] == 3
    assert report['mean'] == 0 and report['median'] == 0
    assert report['rms'] == pytest.approx(math.sqrt(8/3))


def test_invalid_source_preserves_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError):
        audit.run(tmp_path/'report')
    assert audit.read(tmp_path/'report/failed.json')['status'] == 'failed'
    assert not (tmp_path/'report/summary.json').exists()
