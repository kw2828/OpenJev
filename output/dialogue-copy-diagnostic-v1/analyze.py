"""Posthoc saved-choice complementarity; no model, encoder, RNG or test access."""
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METHODS = ('current', 'gru', 'attention', 'gated_delta', 'kalman', 'innovation_kalman', 'carry')
SEEDS = (1729, 2718, 3141)
PINS = {
    'runs/sgd-state-v1/study-01/plan.json': '8c62bd5853299f29fc232b044b69f24d1d0bf4fe9e1a860954a1515d1a617a38',
    'runs/sgd-state-v1/study-01/completed.json': 'd3b59edc9000237fe5e9cc34cf91d9d416bd2a3561b60044aeda2a02610cbc44',
    'runs/sgd-state-v1/features-02/completed.json': 'e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308',
    'runs/sgd-state-v1/data/completed.json': '677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12',
    'output/dialogue-memory-v1/independent-audit-01.json': '55d1b8a39b2694cda7df60d4721aeca4f65732f290ac5c99035ff58f97545090',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    with (OUT / name).open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def run():
    bound = dict(PINS)
    for name, digest in bound.items():
        assert sha(ROOT / name) == digest, name
    study = ROOT / 'runs/sgd-state-v1/study-01'
    features = ROOT / 'runs/sgd-state-v1/features-02'
    data = ROOT / 'runs/sgd-state-v1/data'
    done, plan = read(study / 'completed.json'), read(study / 'plan.json')
    assert done['status'] == 'completed' and done['fit_count'] == 21
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == digest
        bound[name] = digest
    feature_receipt = read(features / 'completed.json')
    for name, digest in feature_receipt['files'].items():
        assert sha(features / name) == digest
        bound[str((features / name).relative_to(ROOT))] = digest
    prepared = read(data / 'completed.json')
    text_path = data / 'dev-dialogues.jsonl'
    assert sha(text_path) == prepared['files']['dev-dialogues.jsonl']['sha256']
    bound[str(text_path.relative_to(ROOT))] = sha(text_path)
    public = {d['dialogue_id']: d for d in map(json.loads, text_path.read_text().splitlines())}
    packet = read(features / 'packet.json')
    schema = packet['queries']
    patterns = {}
    booleans = set()
    for qi, q in enumerate(schema):
        values = q['candidate_values']
        if q['split'] != 'dev':
            continue
        assert values[:2] == [None, None]
        words = [v.strip().casefold() for v in values[2:]]
        patterns[qi] = [(i + 2, word, re.compile(r'(?<!\w)' + re.escape(word) + r'(?!\w)'))
                        for i, word in enumerate(words)]
        if set(words) == {'true', 'false'}:
            booleans.add(qi)
    expected = {k: [] for k in ('labels', 'bin', 'unseen', 'dialogue', 'time', 'query')}
    reconstructed, current_mentions, current_unique, target_mentions, boolean = [], [], [], [], []
    for d in packet['cohorts']['dev']:
        original = public[d['id']]
        turns = [original['turns'][t['turn_index']]['utterance'] for t in original['user_turns']]
        assert turns == d['user_text'] and len(turns) == len(d['turns'])
        tracks = {}
        for qi in sorted({r['query'] for r in d['queries']}):
            state, history = 0, []
            for utterance in turns:
                hits = [(i, len(word)) for i, word, pattern in patterns[qi]
                        if pattern.search(utterance.casefold())]
                longest = max((length for _, length in hits), default=-1)
                winners = [i for i, length in hits if length == longest]
                if len(winners) == 1:
                    state = winners[0]
                history.append((state, {i for i, _ in hits}, len(winners) == 1))
            tracks[qi] = history
        for r in d['queries']:
            qi, ti, label = r['query'], r['time'], r['label']
            value, hits, unique = tracks[qi][ti]
            reconstructed.append(value)
            current_mentions.append(bool(hits))
            current_unique.append(unique)
            target_mentions.append(label in hits)
            boolean.append(qi in booleans)
            for name, value in (('labels', label), ('bin', r['bin']), ('unseen', r['unseen']),
                                ('dialogue', d['id']), ('time', ti), ('query', qi)):
                expected[name].append(value)
    expected = {k: np.asarray(v) for k, v in expected.items()}
    literal = np.asarray(reconstructed)
    mention, unique, target_mention, boolean = map(np.asarray, (current_mentions, current_unique,
                                                              target_mentions, boolean))
    ref_path = study / 'references.npz'
    assert sha(ref_path) == done['references']['sha256']
    bound[str(ref_path.relative_to(ROOT))] = sha(ref_path)
    with np.load(ref_path, allow_pickle=False) as z:
        assert np.array_equal(literal, z['pred_literal'])
        for k, value in expected.items():
            assert np.array_equal(value, z[k]), k
    labels = expected['labels']
    n = len(labels)
    assert n == 62329
    panels = {'all': np.ones(n, bool), 'seen': ~expected['unseen'], 'unseen': expected['unseen']}
    label_masks = {'not_mentioned': labels == 0, 'dontcare': labels == 1, 'ontology': labels >= 2}
    slot_masks = {'boolean_true_false': boolean, 'nonboolean': ~boolean}
    mention_masks = {'current_mention': mention, 'no_current_mention': ~mention}
    changed = np.isin(expected['bin'], ['first_assignment', 'revision', 'clear'])
    groups = {}
    for panel, pm in panels.items():
        groups[panel] = pm
        for category, masks in [('label', label_masks), ('slot', slot_masks), ('mention', mention_masks),
                                ('transition', {b: expected['bin'] == b for b in
                                                ['first_assignment', 'revision', 'clear',
                                                 'assigned_retention', 'unmentioned_retention']})]:
            for key, mask in masks.items():
                groups[f'{panel}/{category}/{key}'] = pm & mask
        for slot, sm in slot_masks.items():
            for label, lm in label_masks.items():
                for name, mm in mention_masks.items():
                    groups[f'{panel}/joint/{slot}/{label}/{name}'] = pm & sm & lm & mm
    lc = literal == labels

    def outcomes(nc, mask):
        count = int(mask.sum())
        quadrants = {'both_correct': int((mask & lc & nc).sum()),
                     'literal_only': int((mask & lc & ~nc).sum()),
                     'neural_only': int((mask & ~lc & nc).sum()),
                     'both_wrong': int((mask & ~lc & ~nc).sum())}
        assert sum(quadrants.values()) == count
        return {'count': count, **quadrants,
                'literal_accuracy': float(lc[mask].mean()) if count else None,
                'neural_accuracy': float(nc[mask].mean()) if count else None,
                'oracle_union_accuracy': float((lc | nc)[mask].mean()) if count else None}

    def macro(correct, pm):
        strata = [pm & (expected['bin'] == 'unmentioned_retention'),
                  pm & (expected['bin'] == 'assigned_retention'), pm & changed]
        return float(sum((Fraction(int((correct & m).sum()), int(m.sum())) for m in strata),
                         Fraction()) / 3)

    rows, primary = {}, {}
    assert [f"{r['method']}-{r['seed']}" for r in done['fits']] == [f'{m}-{s}' for s in SEEDS for m in METHODS]
    for record in done['fits']:
        name = f"{record['method']}-{record['seed']}"
        folder = study / 'fits' / name
        assert read(folder / 'completed.json') == record
        for filename, digest in [('completed.json', sha(folder / 'completed.json')),
                                 ('weights.pt', record['weights_sha256']),
                                 ('dev-predictions.npz', record['predictions_sha256'])]:
            assert sha(folder / filename) == digest
            bound[str((folder / filename).relative_to(ROOT))] = digest
        with np.load(folder / 'dev-predictions.npz', allow_pickle=False) as z:
            for key, value in expected.items():
                assert np.array_equal(z[key], value)
            assert np.array_equal(z['choice'], z['probabilities'].argmax(-1))
            nc = z['choice'] == labels
        rows[name] = {panel: {**outcomes(nc, pm), 'literal_macro': macro(lc, pm),
                              'neural_macro': macro(nc, pm), 'oracle_union_macro': macro(lc | nc, pm)}
                      for panel, pm in panels.items()}
        if record['method'] == 'innovation_kalman':
            primary[str(record['seed'])] = {name: outcomes(nc, mask) for name, mask in groups.items()}
    families = {method: {panel: {key: float(np.mean([rows[f'{method}-{seed}'][panel][key]
                                                     for seed in SEEDS]))
                                 for key in ('literal_accuracy', 'neural_accuracy', 'oracle_union_accuracy',
                                             'literal_macro', 'neural_macro', 'oracle_union_macro')}
                        for panel in panels} for method in METHODS}
    aggregate = {}
    for name in groups:
        sample = primary[str(SEEDS[0])][name]
        aggregate[name] = {'count_per_fit': sample['count'],
                           **{key: None if sample[key] is None else
                              float(np.mean([primary[str(seed)][name][key] for seed in SEEDS]))
                              for key in sample if key != 'count'}}
    result = {
        'status': 'completed', 'study': 'dialogue-copy-diagnostic-v1', 'posthoc': True,
        'source_sha256': sha(Path(__file__)), 'input_sha256': bound,
        'scope': 'Exposed official development data only. No newly trained or evaluated model. '
                 'Union correctness is a target-aware per-query oracle between one fixed neural seed '
                 'and the literal rule, then averaged across seeds; never an achieved hybrid score. '
                 'No cross-seed oracle, policy selection, threshold tuning or changed study gate.',
        'definitions': {'current_mention': 'Any bounded casefolded ontology value in current USER text; public-only',
                        'boolean': 'Exact casefolded ontology set {true,false}; numeric 0/1 slots are nonboolean',
                        'target_mention': 'Gold current label is one of current matched candidates; evaluator-only',
                        'macro': 'Equal unmentioned-retention, assigned-retention and changed stratum accuracies',
                        'group_accuracy': 'Micro accuracy within each named subgroup; counts repeat across seeds',
                        'prefix': 'Every USER utterance from dialogue start, including unscored service gaps',
                        'literal_ties': 'Unique longest Unicode-bounded casefolded value; ties retain prior state'},
        'counts': {'queries_per_fit': n, 'fits': 21, 'primary_fits': 3,
                   'literal_reconstruction_mismatches': 0, 'boolean_slot_definitions': len(booleans),
                   'current_mention_queries': int(mention.sum()), 'unique_longest_current_queries': int(unique.sum()),
                   'gold_target_current_mention_queries': int(target_mention.sum()),
                   'new_model_calls': 0, 'new_encoder_calls': 0, 'test_dialogue_contents_accessed': False},
        'families': families, 'all_fit_panels': rows, 'primary_groups': aggregate,
        'primary_groups_per_seed': primary,
    }
    for name, digest in bound.items():
        assert sha(ROOT / name) == digest, name
    write('diagnostic.json', result)
    print(json.dumps({'families_primary': families['innovation_kalman'], 'counts': result['counts'],
                      'diagnostic_sha256': sha(OUT / 'diagnostic.json')}, indent=2))


if __name__ == '__main__':
    try:
        run()
    except BaseException as exc:
        write('failed.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
