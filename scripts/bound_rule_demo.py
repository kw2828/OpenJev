"""Run a published compact operator on controlled-English facts and rules."""
import argparse
import json
from pathlib import Path

import torch

from openjev.research.bound_rules import BoundRuleNet, collate, ground


def load_model(path):
    package = json.loads(Path(path).read_text())
    model = BoundRuleNet('learned', package['steps'])
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in package['weights'].items()}, strict=True)
    return model.eval(), package


@torch.no_grad()
def decide(model, package, context, questions):
    # IDs are caller-owned and are never parsed as English.
    if not questions or len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Distinct question IDs required')
    graph = ground(context, [q['question'] for q in questions], package['collapse_entities'])
    p, change = model(collate([graph]))
    return {'model': 'OpenJev bound-rule operator', 'seed': package['seed'], 'steps': package['steps'],
        'scope': 'Controlled RuleTaker-style English, handwritten parser plus learned conjunction',
        'probability_semantics': 'Uncalibrated fuzzy logical scores, not guaranteed confidence',
        'answers': [{'id': q['id'], 'choice': 'true' if value >= .5 else 'false',
                     'probabilities': {'true': float(value), 'false': 1.-float(value)},
                     'last_step_change': float(delta)} for q, value, delta in zip(questions, p, change, strict=True)]}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('request', type=Path)
    p.add_argument('--weights', type=Path, default=Path('evidence/bound-rules-v1/weights/learned_16-17.json'))
    args = p.parse_args()
    model, package = load_model(args.weights)
    request = json.loads(args.request.read_text())
    print(json.dumps(decide(model, package, request['context'], request['questions']), indent=2, allow_nan=False))
