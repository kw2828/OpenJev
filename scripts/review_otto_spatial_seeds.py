"""Inspect local OTTO seed metadata without evaluating models or environments.

The result is scoped to the explicitly inventoried original ledger and source
files. It does not establish global freshness or external training independence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRST = {'lambda3': 16100001, 'lambda4': 16200001, 'lambda5': 16300001}
PROPOSED = {seed for start in FIRST.values() for seed in range(start, start+24)} | set(range(16500001, 16500004))
PRIOR = 'output/otto-conditioning-control-v1/seed-ledger-review-02.json'
LEDGER = re.compile(r'^output/otto-[^/]+/[^/]+/(?:[^/]*episodes[^/]*\.jsonl?|[^/]*cases[^/]*\.jsonl?|native-setup\.json|qualification\.jsonl?|evaluation\.jsonl)$')
DECLARATION = re.compile(r'^(?:scripts/[^/]*otto[^/]*\.py|research/otto[^/]*\.md|output/otto-[^/]+/(?:plan[^/]*\.json|[^/]+/plan[^/]*\.json))$')
TOKEN = re.compile(r'(?<![\w.])(?:161000\d\d|162000\d\d|163000\d\d|165000\d\d)(?![\w.])')


def descriptor(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError('regular source required: '+str(path))
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while part := stream.read(1024**2):
            value.update(part)
    return {'sha256': value.hexdigest(), 'bytes': path.stat().st_size}


def ranges(values):
    result = []
    for value in sorted(values):
        if result and result[-1][1]+1 == value:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def seed_fields(value, path='$', within=False):
    if isinstance(value, dict):
        for key, child in value.items():
            descriptor_only = isinstance(child, dict) and {'path', 'sha256', 'bytes'} <= child.keys()
            yield from seed_fields(child, path+'.'+key, within or ('seed' in key.lower() and not descriptor_only))
    elif isinstance(value, list):
        for child in value:
            yield from seed_fields(child, path+'[]', within)
    elif within and type(value) is int:
        yield path, value


def records(path):
    with path.open() as stream:
        if path.suffix == '.jsonl':
            for line in stream:
                yield json.loads(line)
        else:
            value = json.load(stream)
            yield from value if isinstance(value, list) else [value]


def current(name):
    return ('otto-spatial-control' in name or 'otto_spatial_control' in name
            or name == 'scripts/review_otto_spatial_seeds.py')


def inspect(output):
    if not output.is_absolute() or output.exists() or any(p.is_symlink() for p in output.parents):
        raise ValueError('exclusive absolute output required')
    start = time.perf_counter()
    paths = subprocess.check_output(['rg', '--files', '--hidden', '--no-ignore', 'scripts', 'research', 'output'],
                                    cwd=ROOT, text=True, timeout=30).splitlines()
    prior_path = ROOT/PRIOR
    prior = json.loads(prior_path.read_text())
    ledger_names = {p for p in paths if LEDGER.fullmatch(p) and not current(p)} | {f['path'] for f in prior['files']}
    declaration_names = {p for p in paths if DECLARATION.fullmatch(p)}
    report = {'version': 'otto-spatial-seed-review-v1', 'status': 'running',
              'created_utc': datetime.now(UTC).isoformat(), 'source': descriptor(Path(__file__).resolve()),
              'prior_inventory': {'path': PRIOR, **descriptor(prior_path)},
              'proposed_ranges_inclusive': {**{'eval_'+k: [v, v+23] for k, v in FIRST.items()}, 'templates': [16500001, 16500003]},
              'proposed_unique_seeds': len(PROPOSED), 'files': [], 'declarations': [],
              'overlap': [], 'overlaps': [], 'historical_declaration_matches': [], 'current_study_reservations': [],
              'model_calls': 0, 'native_calls': 0, 'training_calls': 0,
              'method': 'Fresh streaming parse of original local ledger inventory plus prior inventory; recursively collect integer values under named seed fields. Inspect proposed integer tokens in source/protocol/plan declarations. Files are hashed before and after each read.',
              'limitations': ['Only listed local original OTTO records and named integer seed fields are covered.',
                              'Declaration token checks do not solve arbitrary programmatic seed generation or prove global freshness.',
                              'No metric recomputation, numerical arrays, model inference, simulator, or external data access.']}
    try:
        for name in sorted(ledger_names):
            path = ROOT/name
            identity = descriptor(path)
            seeds, fields, count, seeded = set(), Counter(), 0, 0
            for row in records(path):
                count += 1
                entries = list(seed_fields(row))
                seeded += bool(entries)
                for field, seed in entries:
                    fields[field] += 1
                    seeds.add(seed)
            if descriptor(path) != identity:
                raise ValueError('ledger changed while scanning: '+name)
            overlap = sorted(seeds & PROPOSED)
            report['files'].append({'path': name, **identity, 'records': count, 'records_with_seed_fields': seeded,
                                    'seed_field_occurrences': sum(fields.values()), 'seed_fields': dict(fields),
                                    'unique_seeds': len(seeds), 'seed_ranges_inclusive': ranges(seeds), 'overlaps': overlap})
            if overlap:
                report['overlaps'].append({'path': name, 'seeds': overlap})
        for name in sorted(declaration_names):
            path = ROOT/name
            identity = descriptor(path)
            content = path.read_text()
            matches = [{'line': i, 'seeds': sorted({int(m.group()) for m in TOKEN.finditer(line)} & PROPOSED)}
                       for i, line in enumerate(content.splitlines(), 1)]
            matches = [m for m in matches if m['seeds']]
            named = sorted({v for row in records(path) for _, v in seed_fields(row)}) if path.suffix == '.json' else []
            if descriptor(path) != identity:
                raise ValueError('declaration changed while scanning: '+name)
            report['declarations'].append({'path': name, **identity, 'named_seed_ranges_inclusive': ranges(named),
                                           'exact_proposed_token_matches': matches, 'current_study': current(name)})
            if matches:
                report['current_study_reservations' if current(name) else 'historical_declaration_matches'].append({'path': name, 'matches': matches})
        report['overlap'] = sorted({v for row in report['overlaps'] for v in row['seeds']})
        report.update(status='completed', passed=not report['overlaps'] and not report['historical_declaration_matches'],
                      totals={'ledgers': len(report['files']), 'ledger_bytes': sum(x['bytes'] for x in report['files']),
                              'ledger_records': sum(x['records'] for x in report['files']), 'declarations': len(report['declarations'])},
                      wall_seconds=time.perf_counter()-start)
    except BaseException as error:
        report.update(status='failed', passed=False, error=repr(error), wall_seconds=time.perf_counter()-start)
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'path': str(output), **descriptor(output), 'passed': report['passed'], **report['totals']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    inspect(parser.parse_args().output)
