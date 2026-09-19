"""Descriptive training-expert comparison from sealed caches; no inference."""
from __future__ import annotations

import json
from pathlib import Path

from audit_pose_coordination import error_arrays, load_prediction, load_public, members, pooled, reduce_errors, sha, write_json

EXP = Path('output/pose-coordination-v1/experiment-01')
REPORT = Path('output/pose-coordination-v1/report-01')
OUT = Path('output/pose-coordination-v1/diagnosis-01')
EXPECTED = '711251ab051e0415b0b61ee234e5e59a5420b60246bc9e4565d2ec7e48461ac1'


def run():
    completed = EXP / 'run-01/completed.json'
    if sha(completed) != EXPECTED:
        raise ValueError('completion binding')
    done = json.loads(completed.read_text())
    source = EXP / 'run-01'
    before = members(source)
    if done['files'] != {k: v for k, v in before.items() if k != 'completed.json'}:
        raise ValueError('cache seal')
    protocol = json.loads((EXP / 'protocol.json').read_text())
    data = Path(protocol['data'])
    if protocol['data_hashes'] != {p.name: sha(p) for p in data.iterdir() if p.is_file()}:
        raise ValueError('data changed')
    p, r, _a, ids = load_public(data, 'train')
    rows = {}
    for variant in ('fast', 'slow'):
        for seed in (1101, 1202, 1303):
            pp, rr = load_prediction(source / f'train-{variant}-{seed}.npz')
            rows[f'{variant}-{seed}'] = reduce_errors(error_arrays(pp, rr, p[:, 32:], r[:, 32:]), ids)
    families = {v: pooled([rows[f'{v}-{s}'] for s in (1101, 1202, 1303)]) for v in ('fast', 'slow')}
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / 'training-experts.json', {
        'status': 'completed', 'scope': 'post-outcome descriptive analysis of sealed training caches',
        'execution_completed_sha256': EXPECTED, 'protocol_sha256': sha(EXP / 'protocol.json'),
        'audited_summary_sha256': sha(REPORT / 'summary.json'),
        'rows': rows, 'families': families, 'windows': 720, 'parents': 30, 'horizon': 25,
        'target_conversion': 'Independent NumPy arithmetic from bound training data; no training or forecast call.',
        'limits': ['Training windows were seen by the frozen experts. This is not held-out evidence.',
                   'The result diagnoses an association between training errors and gate choices, not a causal proof.',
                   'No gate refit, model call, optimizer call, sweep or outcome selection.']})
    if members(source) != before:
        raise ValueError('evidence changed')
    write_json(OUT / 'receipt.json', {'status': 'completed', 'source_sha256': sha(__file__),
        'inputs': {'completion': EXPECTED, 'protocol': sha(EXP / 'protocol.json'),
                   'data': protocol['data_hashes'], 'execution_files': before},
        'files': members(OUT), 'new_model_calls': 0, 'new_optimizer_calls': 0, 'new_random_draws': 0})
    print(json.dumps({v: {k: families[v][k]['rmse'] for k in ('position', 'rotation')} for v in families}))


if __name__ == '__main__':
    run()
