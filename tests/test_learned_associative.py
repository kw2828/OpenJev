import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from openjev.research.learned_associative import MODES, LearnedAssociativeHead


def model(mode, **kwargs):
    return LearnedAssociativeHead(mode, input_dim=7, latent_dim=4, classes=3, top_k=2, **kwargs)


def test_parameters_match_and_every_head_backpropagates_to_all_parameters():
    counts = []
    x = torch.randn(12, 7)
    y = torch.arange(12) % 3
    for mode in MODES:
        head = model(mode)
        head.initialize_prototypes(x, y)
        counts.append(head.cost()['trainable_parameters'])
        loss = torch.nn.functional.cross_entropy(head(x), (y+1) % 3)
        loss.backward()
        for parameter in head.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    assert len(set(counts)) == 1


def test_zero_recurrence_and_fully_anchored_recurrence_match_feedforward():
    x = torch.randn(5, 7)
    base = model('feedforward')
    for head in (model('recurrent', steps=0), model('recurrent', anchor=1.)):
        head.load_state_dict(base.state_dict())
        assert torch.allclose(head(x), base(x), atol=1e-5)


def test_class_permutation_equivariance_and_checkpoint_roundtrip(tmp_path):
    x = torch.randn(5, 7)
    order = torch.tensor([2, 0, 1])
    for mode in MODES:
        head = model(mode).eval()
        before = head(x).detach()
        path = tmp_path/f'{mode}.pt'
        torch.save(head.state_dict(), path)
        restored = model(mode).eval()
        restored.load_state_dict(torch.load(path, weights_only=True))
        assert torch.equal(before, restored(x).detach())
        with torch.no_grad():
            head.prototypes.copy_(head.prototypes[order])
        assert torch.allclose(head(x), before[:, order], atol=1e-5)


def test_development_loader_never_opens_reserved_parts(tmp_path, monkeypatch):
    import json

    import numpy as np

    from openjev.research.text_distillation import file_hash
    module = runpy.run_path(str(Path(__file__).parents[1]/'scripts/learned_associative_study.py'))
    manifest = {'protocol': {'version': 'associative-text-v1'}, 'parts': {}}
    for name in ('train', 'tune'):
        (tmp_path/f'{name}.json').write_text(json.dumps([{'label': 0, 'text_sha256': name}]))
        np.save(tmp_path/f'{name}.npy', np.ones((1, 7), dtype=np.float32), allow_pickle=False)
        manifest['parts'][name] = {'rows_sha256': file_hash(tmp_path/f'{name}.json'),
                                   'vectors_sha256': file_hash(tmp_path/f'{name}.npy')}
    manifest['parts']['test'] = {'rows_sha256': 'must not read'}
    manifest['parts']['calibration'] = {'rows_sha256': 'must not read'}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.stem not in ('test', 'calibration')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    _, parts = module['load_development'](tmp_path)
    assert set(parts) == {'train', 'tune'}
