"""Prepare grouped RuleTaker development data from the pinned authors' release.

Only train/dev members are opened. No formal representation or proof is exposed
by decision_input; those annotations cannot become accidental model inputs.
"""
import argparse
import hashlib
import json
import random
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from openjev.research.text_distillation import digest, file_hash, write_new

URL = 'https://aristo-data-public.s3-us-west-2.amazonaws.com/ruletaker/rule-reasoning-dataset-V2020.2.5.zip'
SHA256 = '080c8bca836603d9fea040e5a242bbd281f631255ca83428b7141ebf7f3deb0c'
ROOT = 'rule-reasoning-dataset-V2020.2.5.0/'
SEED = 59271
PARTS = [('train', 'depth-2', 'train', 2000), ('dev_in', 'depth-2', 'dev', 300),
         ('dev_shift', 'depth-5', 'dev', 300)]
DEPTHS = {'train': {0, 1, 2}, 'dev_in': {0, 1, 2}, 'dev_shift': {3, 4, 5}}


def decision_input(world, question):
    return {'context': world['context'],
            'question': 'Under the rulebase\'s closed-world semantics, is this statement true? '+question['text'],
            'candidates': [{'id': 'true', 'description': 'True'}, {'id': 'false', 'description': 'False'}]}


def world_key(context):
    # Order-independent to catch duplicated worlds with a different sentence scramble.
    return digest(sorted(' '.join(s.casefold().split()) for s in context.split('.') if s.strip()))


def verification_key(world_id, statement):
    return world_id+':'+digest(' '.join(statement.casefold().strip().rstrip('.').split()))


def valid_questions(world, verified):
    selected, excluded = [], Counter()
    for q in world['questions']:
        qid = q['meta']['Qid']
        if not qid.startswith('Q') or not qid[1:].isdigit():
            raise ValueError('Unexpected question identifier')
        # Problog release suffixes can be renumbered after filtering. Join on
        # world identity AND the normalized English assertion, never the suffix.
        key = verification_key(world['id'], q['text'])
        if key not in verified:
            excluded['not_in_problog_release'] += 1
        elif type(q['label']) is not bool or q['label'] != verified[key]:
            raise ValueError('Original and Problog labels disagree')
        else:
            selected.append({'id': q['id'], 'text': q['text'], 'gold': 'true' if q['label'] else 'false',
                             'depth': q['meta']['QDep'], 'strategy': q['meta']['strategy']})
    return selected, excluded


def prepare(archive, out):
    if out.exists() or file_hash(archive) != SHA256:
        raise ValueError('Need fresh output and the exact pinned dataset release')
    seen, manifest = set(), {'source_url': URL, 'source_sha256': SHA256, 'selection_seed': SEED,
        'code_sha256': file_hash(__file__), 'max_questions_per_world': 12, 'parts': {},
        'model_inputs': 'English context, English question and true/false candidate descriptions only',
        'test_members_opened': False, 'models_trained': False,
        'source_scope': 'Original English text intersected with consistent Problog labels',
        'license_note': 'Upstream repository is Apache-2.0; archive has no separate license file. Raw data kept local.'}
    with zipfile.ZipFile(archive) as z:
        for name, depth, split, cap in PARTS:
            verified = {}
            with z.open(ROOT+f'problog/{depth}/{split}.jsonl') as stream:
                for line in stream:
                    row = json.loads(line)
                    value = row['theory_assertion_instance']
                    if value.get('exception') is not None or type(value['label']) is not bool:
                        raise ValueError('Unexpected unverified Problog example')
                    world_id, suffix = row['id'].rsplit('_', 1)
                    if not suffix.isdigit():
                        raise ValueError('Unexpected Problog ID format')
                    key = verification_key(world_id, row['english']['assertion_statement'])
                    if key in verified and verified[key] != value['label']:
                        raise ValueError('Conflicting labels for identical assertions')
                    verified[key] = value['label']
            eligible, excluded, local_seen = [], Counter(), set()
            with z.open(ROOT+f'original/{depth}/{split}.jsonl') as stream:
                for line in stream:
                    row = json.loads(line)
                    key = world_key(row['context'])
                    if key in seen or key in local_seen:
                        excluded['duplicate_world'] += 1
                        continue
                    local_seen.add(key)
                    questions, rejected = valid_questions(row, verified)
                    excluded.update(rejected)
                    filtered = [q for q in questions if q['depth'] in DEPTHS[name]]
                    excluded['question_depth_outside_selection'] += len(questions)-len(filtered)
                    questions = filtered
                    if not questions:
                        excluded['no_eligible_questions'] += 1
                        continue
                    eligible.append({'id': row['id'], 'world_sha256': key, 'context': row['context'],
                                     'questions': questions})
            seen.update(local_seen)
            rng = random.Random(SEED)
            chosen = rng.sample(sorted(eligible, key=lambda w: w['id']), min(cap, len(eligible)))
            for world in chosen:
                if len(world['questions']) > 12:
                    world['questions'] = rng.sample(world['questions'], 12)
            counts = Counter((q['gold'], str(q['depth'])) for w in chosen for q in w['questions'])
            write_new(out/f'{name}.json', chosen)
            manifest['parts'][name] = {'upstream_directory': depth, 'upstream_split': split,
                'eligible_worlds': len(eligible), 'selected_worlds': len(chosen),
                'allowed_annotated_depths': sorted(DEPTHS[name]),
                'selected_questions': sum(len(w['questions']) for w in chosen), 'excluded': dict(excluded),
                'label_by_depth': {f'{label}:{d}': n for (label, d), n in sorted(counts.items())},
                'file_sha256': file_hash(out/f'{name}.json'),
                'worlds': [{'id': w['id'], 'sha256': w['world_sha256']} for w in chosen]}
            print(json.dumps({k: v for k, v in manifest['parts'][name].items() if k != 'worlds'}), flush=True)
    write_new(out/'manifest.json', manifest)


def fetch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        h, count = hashlib.sha256(), 0
        with urllib.request.urlopen(URL, timeout=60) as response, path.open('xb') as out:
            while chunk := response.read(1024*1024):
                count += len(chunk)
                if count > 200_000_000:
                    raise ValueError('Unexpected release size')
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != SHA256:
            raise ValueError('Downloaded source differs from frozen release; do not prepare')
    elif file_hash(path) != SHA256:
        raise ValueError('Existing source is incomplete or differs from frozen release')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    fetch(args.archive)
    prepare(args.archive, args.out)
