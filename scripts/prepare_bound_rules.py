"""Reuse frozen training worlds, select previously unused development worlds."""
import argparse
import importlib.util
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from openjev.research.text_distillation import digest, file_hash, write_new

SOURCE = Path(__file__).with_name('prepare_ruletaker.py')
SPEC = importlib.util.spec_from_file_location('original_preparer', SOURCE)
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)
SEED = 60271


def prepare(args):
    if args.out.exists() or file_hash(args.archive) != OLD.SHA256:
        raise ValueError('Fresh output and pinned release required')
    previous = json.loads((args.previous/'manifest.json').read_text())
    seen = {w['sha256'] for part in previous['parts'].values() for w in part['worlds']}
    train_path = args.previous/'train.json'
    if file_hash(train_path) != previous['parts']['train']['file_sha256']:
        raise ValueError('Training packet hash changed')
    training = json.loads(train_path.read_text())
    write_new(args.out/'train.json', training)
    manifest = {'version': 'bound-rules-data-v1', 'seed': SEED, 'source_sha256': OLD.SHA256,
        'previous_manifest_sha256': digest(previous), 'code_sha256': file_hash(__file__),
        'preparer_dependency_sha256': file_hash(SOURCE), 'test_members_opened': False,
        'parts': {'train': {'worlds': len(training), 'questions': sum(len(w['questions']) for w in training),
                            'file_sha256': file_hash(args.out/'train.json'), 'selection': 'unchanged prior training worlds'}}}
    with zipfile.ZipFile(args.archive) as archive:
        for part, depth, cap, depths in [('dev_in', 'depth-2', 300, {0, 1, 2}),
                                         ('dev_shift', 'depth-5', 150, {3, 4, 5})]:
            verified = {}
            with archive.open(OLD.ROOT+f'problog/{depth}/dev.jsonl') as stream:
                for line in stream:
                    row = json.loads(line)
                    value = row['theory_assertion_instance']
                    if value.get('exception') is not None or type(value['label']) is not bool:
                        raise ValueError('Unverified source example')
                    key = OLD.verification_key(row['id'].rsplit('_', 1)[0], row['english']['assertion_statement'])
                    if key in verified and verified[key] != value['label']:
                        raise ValueError('Conflicting labels')
                    verified[key] = value['label']
            eligible, excluded, unique = [], Counter(), set()
            with archive.open(OLD.ROOT+f'original/{depth}/dev.jsonl') as stream:
                for line in stream:
                    row = json.loads(line)
                    key = OLD.world_key(row['context'])
                    if key in seen or key in unique:
                        excluded['previously_used_or_duplicate_world'] += 1
                        continue
                    unique.add(key)
                    questions, rejected = OLD.valid_questions(row, verified)
                    excluded.update(rejected)
                    questions = [q for q in questions if q['depth'] in depths]
                    if questions:
                        eligible.append({'id': row['id'], 'world_sha256': key, 'context': row['context'],
                                         'questions': questions})
            rng = random.Random(SEED)
            selected = rng.sample(sorted(eligible, key=lambda w: w['id']), min(cap, len(eligible)))
            for world in selected:
                if len(world['questions']) > 12:
                    world['questions'] = rng.sample(world['questions'], 12)
            seen.update(w['world_sha256'] for w in selected)
            write_new(args.out/f'{part}.json', selected)
            counts = Counter((q['gold'], q['depth'], 'not' in q['text'].casefold().split())
                             for w in selected for q in w['questions'])
            manifest['parts'][part] = {'worlds': len(selected), 'questions': sum(len(w['questions']) for w in selected),
                'eligible_worlds': len(eligible), 'exclusions': dict(excluded),
                'label_depth_negation_counts': {str(k): v for k, v in sorted(counts.items())},
                'file_sha256': file_hash(args.out/f'{part}.json'),
                'worlds_sha256': digest(sorted(w['world_sha256'] for w in selected))}
    write_new(args.out/'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'previous', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    prepare(p.parse_args())
