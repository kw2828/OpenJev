"""Synthetic IID coverage sanity check. This is not an OpenJev quality result."""
import json

import numpy as np

from openjev.research.conformal import LabeledDecision, SplitConformal

rng = np.random.default_rng(20260916)
rows = []
for i in range(2500):
    p = rng.dirichlet(np.ones(6))
    rows.append(LabeledDecision(f'iid-{i}', {str(j): float(x) for j, x in enumerate(p)},
                                str(rng.choice(6, p=p))))
model = SplitConformal.fit(rows[:500], alpha=.1, scorer_signature='synthetic-oracle-v1',
                           task_signature='iid-six-class-v1')
sets = [model.predict(row.probabilities, unit_id=row.unit_id,
                      scorer_signature='synthetic-oracle-v1', task_signature='iid-six-class-v1')
        for row in rows[500:]]
print(json.dumps({
    'kind': 'synthetic IID software sanity check, not model accuracy or Doom safety',
    'seed': 20260916, 'calibration_n': 500, 'test_n': 2000, 'alpha': .1,
    'threshold': model.threshold,
    'empirical_coverage': sum(row.correct_id in s for row, s in zip(rows[500:], sets))/len(sets),
    'mean_set_size': sum(map(len, sets))/len(sets),
}, indent=2))
