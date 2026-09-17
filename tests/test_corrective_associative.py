import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from openjev.research.corrective_associative import VARIANTS, CorrectiveAssociativeHead
from openjev.research.learned_associative import LearnedAssociativeHead


def head(mode):
    return CorrectiveAssociativeHead(mode, input_dim=7, latent_dim=4, classes=3, top_k=2)


def test_attractive_update_is_unchanged_and_corrective_updates_are_distinct():
    x = torch.randn(6, 7)
    old = LearnedAssociativeHead('recurrent', input_dim=7, latent_dim=4, classes=3, top_k=2)
    attractive = head('attractive_2')
    attractive.load_state_dict(old.state_dict())
    assert torch.equal(old(x), attractive(x))
    outputs = []
    for mode in ('inhibitory_2', 'residual_2', 'residual_3'):
        candidate = head(mode)
        candidate.load_state_dict(old.state_dict())
        outputs.append(candidate(x))
        assert not torch.allclose(outputs[-1], old(x))
        candidate.correction_rate = 0.
        control = head('feedforward')
        control.load_state_dict(candidate.state_dict())
        assert torch.allclose(candidate(x), control(x), atol=1e-5)
    assert not torch.allclose(outputs[0], outputs[1])


def test_all_parameters_train_and_counts_stay_matched():
    x, y = torch.randn(12, 7), torch.arange(12) % 3
    counts = []
    for mode in VARIANTS:
        model = head(mode)
        model.initialize_prototypes(x, y)
        torch.nn.functional.cross_entropy(model(x), (y+1) % 3).backward()
        counts.append(model.cost()['trainable_parameters'])
        for p in model.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert len(set(counts)) == 1
    assert head('residual_3').cost()['matrix_multiply_accumulates_per_query'] > head('residual_2').cost()['matrix_multiply_accumulates_per_query']


def test_configured_completion_counts_all_fits_and_rejects_missing(tmp_path):
    import json
    module = runpy.run_path(str(Path(__file__).parents[1]/'scripts/corrective_associative_study.py'))
    with pytest.raises(ValueError, match='missing'):
        module['write_execution_record'](tmp_path/'completed.json', {'status': 'completed', 'fits': 12})
    for mode in module['PROTOCOL']['modes']:
        for seed in module['PROTOCOL']['seeds']:
            (tmp_path/f'{mode}-{seed}.json').write_text('{}')
    module['write_execution_record'](tmp_path/'completed.json', {'status': 'completed', 'fits': 12})
    assert json.loads((tmp_path/'completed.json').read_text())['fits'] == 18
