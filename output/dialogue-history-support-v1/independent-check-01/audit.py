"""Independent single-pass aggregation of authenticated saved endpoint metadata.

Standard library only. No public text, candidate catalog, prediction, feature,
model, or producer/helper import. Literal detection itself is inherited.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import resource
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BINS = ('first_assignment', 'revision', 'clear', 'unmentioned_retention', 'assigned_retention')
STRATA = ('all', 'changed', 'retained', 'unmentioned_retention', 'assigned_retention')
TYPES = ('none', 'dontcare', 'true', 'false', 'other')
AGES = ('current', 'recent', 'distant', 'never', 'nonliteral')
ADJ = ('adjacent', 'first_annotation', 'gapped')
PROXIES = ('delayed_system_only', 'changed_no_recent_combined_literal',
           'changed_distant_combined_literal', 'changed_never_combined_literal',
           'other_distant_combined_literal')
NONE, DC = 'reserved:NOT_MENTIONED', 'reserved:DONTCARE'
SCOPE = ('Independent aggregation and metadata consistency only: every count table, '
         'proxy support set and recorded collision diagnostic is reconstructed from '
         'saved endpoints. Text matching, source/candidate completeness, semantic '
         'support, raw public chronology, and unscored/trailing-turn structural totals '
         'inherit authenticated producer witnesses. No efficacy or memory-necessity claim.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def integer(x):
    return type(x) is int and x >= 0


def rss():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return n if sys.platform == 'darwin' else n * 1024


def value_type(cid):
    if cid == NONE:
        return 'none'
    if cid == DC:
        return 'dontcare'
    require(type(cid) is str and cid.startswith('value:') and cid[6:], 'Candidate identity')
    v = cid[6:].casefold()
    return v if v in ('true', 'false') else 'other'


def age_category(age, literal):
    if not literal:
        return 'nonliteral'
    if age is None:
        return 'never'
    return 'current' if age == 0 else 'recent' if age < 4 else 'distant'


def empty_cell():
    return {'rows': 0, 'bins': dict.fromkeys(BINS, 0), 'target_types': dict.fromkeys(TYPES, 0),
            'user_system_age': {u: dict.fromkeys(AGES, 0) for u in AGES},
            'combined_age': dict.fromkeys(AGES, 0), 'annotation_adjacency': dict.fromkeys(ADJ, 0),
            'proxies': dict.fromkeys(PROXIES, 0)}


def increment(cell, row, proxy):
    cell['rows'] += 1
    cell['bins'][row['bin']] += 1
    cell['target_types'][row['target_type']] += 1
    cell['user_system_age'][row['user_age_category']][row['system_age_category']] += 1
    cell['combined_age'][row['combined_age_category']] += 1
    cell['annotation_adjacency'][row['annotation_adjacency']] += 1
    for name, flag in proxy.items():
        cell['proxies'][name] += int(flag)


def run(args):
    started = time.perf_counter()
    out, base = Path(args.out).resolve(), Path(args.run).resolve()
    source = Path(__file__).resolve()
    require(out == source.parent and set(p.name for p in out.iterdir()) == {'audit.py'},
            'Exclusive new checker directory must initially contain only audit.py')
    source_pin = sha(source)
    inputs, rows_seen, comparisons = {}, 0, 0

    def limit(*_):
        require(time.perf_counter() - started <= 60, '60-second audit wall cap')
        require(rss() <= 1024**3, '1-GiB process RSS cap')

    def timeout(*_):
        raise TimeoutError('60-second audit wall cap')

    def eq(actual, expected, path):
        nonlocal comparisons
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), 'Key mismatch: ' + path)
            for k in expected:
                eq(actual[k], expected[k], path + '/' + k)
        else:
            require(type(actual) is type(expected) and actual == expected, 'Mismatch: ' + path)
            comparisons += 1

    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        write(out / 'started.json', {'status': 'started', 'scope': SCOPE, 'source_sha256': source_pin,
              'request': vars(args), 'limits': {'wall_seconds': 60, 'rss_bytes': 1024**3},
              'model_calls': 0, 'public_text_files_opened': 0})
        done_path = base / 'completed.json'
        require(sha(done_path) == args.completed_sha256, 'External completion pin')
        done = read(done_path)
        require(done['status'] == 'completed', 'Producer incomplete')
        names = {'started.json', 'summary.json', 'endpoint-metadata.jsonl.gz'}
        require(set(done['files']) == names and set(p.name for p in base.iterdir()) == names | {'completed.json'},
                'Exact three payloads plus completion')
        for name in sorted(names | {'completed.json'}):
            p = base / name
            require(p.is_file() and not p.is_symlink(), 'Unexpected member type')
            d = {'sha256': sha(p), 'bytes': p.stat().st_size}
            if name != 'completed.json':
                eq(d, done['files'][name], 'manifest/' + name)
            inputs[name] = d
        require(inputs['summary.json']['sha256'] == args.summary_sha256 and
                inputs['endpoint-metadata.jsonl.gz']['sha256'] == args.endpoints_sha256, 'External payload pins')
        observed = read(base / 'summary.json')
        eq(read(base / 'started.json')['plan_sha256'], done['plan_sha256'], 'Start/terminal plan')

        cells = {group: {k: empty_cell() for k in labels}
                 for group, labels in [('by_bin', BINS), ('by_stratum', STRATA), ('by_type', TYPES)]}
        services = {}
        supports = defaultdict(lambda: [0, set(), set()])
        previous, candidate_maps, query_collisions = {}, {}, {}
        identities, dialogues, dialogue_queries = set(), set(), set()
        ambiguous_rows, collision_queries, collision_dialogue_queries = 0, set(), set()
        primary = {k: Counter() for k in ('bin', 'target_type', 'previous_type', 'adjacency', 'system_age',
                                         'combined_age', 'annotated_gap', 'user_index')}
        primary_service = {}
        primary_queries = set()

        with gzip.open(base / 'endpoint-metadata.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                did, qid = r['dialogue_id'], r['query_id']
                require(type(did) is str and did and json.loads(qid) == [r['service'], r['slot']], 'Query identity')
                for k in ('turn_index', 'user_index', 'candidate_count', 'label_index',
                          'previous_label_index', 'query_literal_collision_groups', 'target_literal_match_count'):
                    require(integer(r[k]), 'Integer metadata: ' + k)
                require(r['candidate_count'] > max(r['label_index'], r['previous_label_index']), 'Candidate range')
                ident = (did, qid, r['turn_index'])
                require(ident not in identities, 'Duplicate endpoint')
                identities.add(ident)
                key = (did, qid)
                prev = previous.get(key)
                prior_id, prior_index, prior_turn, prior_user = (NONE, r['previous_label_index'], None, None) if prev is None else prev
                eq(r['previous_label_id'], prior_id, 'Previous candidate identity')
                eq(r['previous_label_index'], prior_index, 'Previous index')
                eq(r['previous_annotated_turn_index'], prior_turn, 'Previous raw turn')
                eq(r['previous_annotated_user_index'], prior_user, 'Previous USER index')
                require(prev is None or (r['turn_index'] > prior_turn and r['user_index'] > prior_user), 'Ordered endpoints')
                adjacency = 'first_annotation' if prev is None else 'adjacent' if r['user_index'] - prior_user == 1 else 'gapped'
                eq(r['annotation_adjacency'], adjacency, 'Annotation adjacency')
                eq(r['annotation_adjacent'], adjacency == 'adjacent', 'Adjacent bool')
                typ = value_type(r['label_id'])
                old_type = value_type(prior_id)
                changed = r['label_id'] != prior_id
                bin_name = ('unmentioned_retention' if typ == 'none' else 'assigned_retention') if not changed else (
                    'first_assignment' if prior_id == NONE else 'clear' if typ == 'none' else 'revision')
                eq(r['target_type'], typ, 'Target type')
                eq(r['changed'], changed, 'Changed bool')
                eq(r['bin'], bin_name, 'Transition')
                eq(r['stratum'], 'changed' if changed else bin_name, 'Stratum')
                literal = typ not in ('none', 'dontcare')
                eq(r['literal_target'], literal, 'Literal target')
                eq(r['ordinary_target'], typ == 'other', 'Ordinary target')
                mapping = candidate_maps.setdefault(qid, ({}, {}, r['candidate_count']))
                require(mapping[2] == r['candidate_count'], 'Stable schema candidate count')
                for cid, ix in [(r['label_id'], r['label_index']), (prior_id, prior_index)]:
                    require(mapping[0].setdefault(cid, ix) == ix and mapping[1].setdefault(ix, cid) == cid,
                            'Stable observed canonical candidate mapping')
                require(query_collisions.setdefault(qid, r['query_literal_collision_groups']) == r['query_literal_collision_groups'],
                        'Stable recorded query collision count')
                for role in ('user', 'system'):
                    age, ui, ti = (r[role + suffix] for suffix in ('_age', '_last_mention_user_index', '_last_mention_turn_index'))
                    if age is None:
                        require(ui is None and ti is None, 'Missing mention metadata')
                    else:
                        require(literal and integer(age) and integer(ui) and integer(ti) and
                                ui <= r['user_index'] and ti <= r['turn_index'] and age == r['user_index'] - ui,
                                'Mention age/identity arithmetic')
                        require(role != 'system' or ti < r['turn_index'], 'SYSTEM cannot be current USER endpoint')
                    require(literal or age is None, 'Reserved target cannot have literal age')
                    eq(r[role + '_age_category'], age_category(age, literal), 'Speaker age category')
                ages = [r[k + '_age'] for k in ('user', 'system') if r[k + '_age'] is not None]
                age = min(ages) if ages else None
                eq(r['combined_age'], age, 'Combined age')
                eq(r['combined_age_category'], age_category(age, literal), 'Combined category')
                eligible = changed and typ == 'other'
                proxy = {'delayed_system_only': eligible and r['user_age'] is None and r['system_age'] is not None and r['system_age'] >= 4,
                         'changed_no_recent_combined_literal': eligible and (age is None or age >= 4),
                         'changed_distant_combined_literal': eligible and age is not None and age >= 4,
                         'changed_never_combined_literal': eligible and age is None,
                         'other_distant_combined_literal': typ == 'other' and age is not None and age >= 4}
                eq(r['proxies'], proxy, 'Proxy definitions')
                require((r['target_literal_match_count'] >= 1) if literal else r['target_literal_match_count'] == 0,
                        'Recorded target collision membership')
                if r['target_literal_match_count'] > 1:
                    require(r['query_literal_collision_groups'] > 0, 'Target ambiguity needs schema collision')
                    ambiguous_rows += 1
                if r['query_literal_collision_groups']:
                    collision_queries.add(qid)
                    collision_dialogue_queries.add(key)
                previous[key] = (r['label_id'], r['label_index'], r['turn_index'], r['user_index'])
                dialogues.add(did)
                dialogue_queries.add(key)
                increment(cells['by_bin'][bin_name], r, proxy)
                increment(cells['by_type'][typ], r, proxy)
                for s in ('all', 'changed') if changed else ('all', 'retained', bin_name):
                    increment(cells['by_stratum'][s], r, proxy)
                service = services.setdefault(r['service'], {**empty_cell(), 'by_bin': {b: empty_cell() for b in BINS}})
                increment(service, r, proxy)
                increment(service['by_bin'][bin_name], r, proxy)
                for name, flag in proxy.items():
                    if flag:
                        for part in ('all', bin_name, 'changed' if changed else 'retained'):
                            v = supports[name, part]
                            v[0] += 1
                            v[1].add(did)
                            v[2].add(r['service'])
                if proxy['delayed_system_only']:
                    vals = {'bin': bin_name, 'target_type': typ, 'previous_type': old_type, 'adjacency': adjacency,
                            'system_age': str(r['system_age']), 'combined_age': str(age),
                            'annotated_gap': 'first_annotation' if prior_user is None else str(r['user_index'] - prior_user),
                            'user_index': str(r['user_index'])}
                    for k, v in vals.items():
                        primary[k][v] += 1
                    ps = primary_service.setdefault(r['service'], {'rows': 0, 'dialogues': set(),
                        'bins': Counter(), 'adjacency': Counter(), 'system_age': Counter(), 'previous_types': Counter()})
                    ps['rows'] += 1
                    ps['dialogues'].add(did)
                    ps['bins'][bin_name] += 1
                    ps['adjacency'][adjacency] += 1
                    ps['system_age'][str(r['system_age'])] += 1
                    ps['previous_types'][old_type] += 1
                    primary_queries.add(qid)
                rows_seen += 1
                if rows_seen % 512 == 0:
                    limit()

        def support(proxy, group):
            n, ds, ss = supports[proxy, group]
            return {'rows': n, 'unique_dialogues': len(ds), 'services': sorted(ss), 'service_count': len(ss)}

        expected = {**cells, 'rows': rows_seen, 'by_service': services,
            'normalized_literal_collisions': {'rows_with_ambiguous_target': ambiguous_rows,
                'distinct_queries_with_collisions': len(collision_queries),
                'dialogue_queries_with_collisions': len(collision_dialogue_queries)},
            'proxy_support': {p: {**support(p, 'all'), 'by_bin': {b: support(p, b) for b in BINS},
                'by_stratum': {s: support(p, s) for s in ('changed', 'retained')}} for p in PROXIES}}
        require(set(observed) == set(expected) | {'cohort', 'scope'}, 'Complete summary count-table coverage')
        before_tables = comparisons
        for k, v in expected.items():
            eq(observed[k], v, 'summary/' + k)
        table_checks = comparisons - before_tables
        for key, value in [('rows', rows_seen), ('dialogues', len(dialogues)), ('supplied_queries', len(dialogue_queries))]:
            eq(observed['cohort'][key], value, 'cohort/' + key)
        eq(done['rows'], rows_seen, 'completion rows')
        eq(done['dialogues'], len(dialogues), 'completion dialogues')
        inherited_keys = ('available_public_user_steps', 'available_public_query_steps',
                          'consecutive_system_pairs', 'consecutive_user_pairs')
        require(set(observed['cohort']) == {'split', 'rows', 'dialogues', 'supplied_queries', *inherited_keys}, 'Cohort schema')
        require(observed['cohort']['split'] == 'train', 'TRAIN scope')
        for v in primary_service.values():
            v['unique_dialogues'] = len(v.pop('dialogues'))
        primary_result = {**support('delayed_system_only', 'all'), 'unique_queries': len(primary_queries),
                          'counts': primary, 'by_service': primary_service}
        result = {'status': 'completed', 'agreement': True, 'scope': SCOPE,
            'rows': rows_seen, 'dialogues': len(dialogues), 'supplied_queries': len(dialogue_queries),
            'summary_count_scalar_checks': table_checks, 'total_scalar_checks': comparisons,
            'recomputed_proxy_support': expected['proxy_support'],
            'recomputed_collision_diagnostics': expected['normalized_literal_collisions'],
            'derived_primary_metadata': primary_result,
            'inherited_unreconstructed_cohort_fields': {k: observed['cohort'][k] for k in inherited_keys},
            'inherited_scope_note': 'Endpoint metadata cannot reveal omitted unscored/trailing public turns or same-role raw-turn pairs.'}
        write(out / 'summary.json', result)
        for name, item in inputs.items():
            require(sha(base / name) == item['sha256'], 'Input changed during audit')
        require(sha(source) == source_pin, 'Checker source changed')
        limit()
        files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in out.iterdir() if p.is_file()}
        receipt = {'status': 'completed', 'agreement': True, 'scope': SCOPE, 'source_sha256': source_pin,
            'producer_completed_sha256': args.completed_sha256, 'producer_summary_sha256': args.summary_sha256,
            'producer_endpoint_metadata_sha256': args.endpoints_sha256, 'authenticated_inputs': inputs,
            'producer_source_sha256_inherited': done['source_sha256'],
            'rows': rows_seen, 'summary_count_scalar_checks': table_checks, 'total_scalar_checks': comparisons,
            'files': files, 'wall_seconds': time.perf_counter() - started, 'peak_rss_bytes': rss(),
            'limits': {'wall_seconds': 60, 'rss_bytes': 1024**3}, 'model_calls': 0,
            'public_text_files_opened': 0, 'producer_helper_imports': 0, 'endpoint_decode_passes': 1,
            'no_retry': True, 'request': vars(args)}
        write(out / 'receipt.json', receipt)
        limit()
        print(json.dumps({'status': 'completed', 'receipt_sha256': sha(out / 'receipt.json'),
                          'rows': rows_seen, 'summary_count_scalar_checks': table_checks,
                          'total_scalar_checks': comparisons, 'primary': primary_result,
                          'wall_seconds': receipt['wall_seconds'], 'peak_rss_bytes': receipt['peak_rss_bytes']}))
    except BaseException as error:
        try:
            if (out / 'receipt.json').exists():
                (out / 'receipt.json').rename(out / 'completion-before-error.json')
            write(out / 'failed.json', {'status': 'failed', 'source_sha256': source_pin,
                'request': vars(args), 'rows_decoded': rows_seen, 'scalar_checks_completed': comparisons,
                'error_type': type(error).__name__, 'error': str(error), 'model_calls': 0,
                'wall_seconds': time.perf_counter() - started})
        except BaseException as secondary:
            error.add_note('Failure receipt error: ' + repr(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('run', 'completed-sha256', 'summary-sha256', 'endpoints-sha256', 'out'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
