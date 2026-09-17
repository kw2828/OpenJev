import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DEMO = runpy.run_path(str(ROOT/'scripts/bound_rule_demo.py'))


def test_published_model_uses_relation_direction_and_preserves_question_ids():
    model, package = DEMO['load_model'](ROOT/'evidence/bound-rules-v1/weights/learned_16-17.json')
    context = ('Mira is red. Mira visits Nemi. If someone is red and they visit Nemi then they are warm. '
               'Warm people are kind.')
    questions = [{'id': 'caller-id', 'question': 'Mira is kind.'}]
    a = DEMO['decide'](model, package, context, questions)['answers'][0]
    b = DEMO['decide'](model, package, context.replace('Mira visits Nemi', 'Nemi visits Mira'), questions)['answers'][0]
    assert a['id'] == b['id'] == 'caller-id'
    assert a['choice'] == 'true' and b['choice'] == 'false'
    assert sum(a['probabilities'].values()) == sum(b['probabilities'].values()) == 1.


def test_public_weight_files_cover_every_fit_with_no_best_seed_filter():
    files = list((ROOT/'evidence/bound-rules-v1/weights').glob('*.json'))
    assert len(files) == 12
    for path in files:
        model, package = DEMO['load_model'](path)
        assert sum(p.numel() for p in model.parameters()) == 225
        assert package['seed'] in (17, 29, 43)


def test_duplicate_question_ids_are_rejected():
    model, package = DEMO['load_model'](ROOT/'evidence/bound-rules-v1/weights/learned_16-17.json')
    with pytest.raises(ValueError, match='Distinct'):
        DEMO['decide'](model, package, 'Mira is red.', [{'id': 'x', 'question': 'Mira is red.'}]*2)
