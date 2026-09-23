"""One saved-metadata review. Standard library only; no scientific imports/data."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'output/otto-cross-query-forecast-v1'
OUT = Path(__file__).resolve().parent
PIN = '0628bb19cbaa3ad8fc8846f834654eddbf6d0bfbd4dffa531b52c865298de48d'
verified = {}
checks = 0
started = time.time_ns()


def require(ok, label):
    global checks
    checks += 1
    if not ok:
        raise ValueError(label)


def path(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require(p.is_file() and p.is_relative_to(ROOT) and '..' not in p.parts
            and not any(q.is_symlink() for q in (p, *p.parents)), 'contained regular file')
    require(p.suffix not in ('.npz', '.npy', '.h5'), 'no numerical payload access')
    return p


def descriptor(value, expected=None):
    p = path(value)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    d = {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}
    if isinstance(expected, str):
        require(d['sha256'] == expected, f'hash:{d["path"]}')
    elif expected is not None:
        require(d['sha256'] == expected['sha256'] and d['bytes'] == expected['bytes'],
                f'descriptor:{d["path"]}')
    verified[d['path']] = d
    return d


def read(value):
    return json.loads(path(value).read_text())


def closed(receipt_path, receipt):
    directory = path(receipt_path).parent
    require({p.name for p in directory.iterdir()} == set(receipt['files']) | {'receipt.json'},
            f'closed inventory:{directory.name}')
    for name, d in receipt['files'].items():
        require(Path(name).name == name, 'flat payload')
        descriptor(directory / name, d)


def source_map(mapping, frozen):
    for name, pin in mapping.items():
        require(frozen.get(name) == pin, f'prospective source join:{name}')


def publication(value, data):
    with value.open('x') as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    manifest_path = BASE / 'precollection-freeze-01.json'
    manifest_descriptor = descriptor(manifest_path, PIN)
    manifest = read(manifest_path)
    sources = manifest['sources']
    require(manifest['status'] == 'frozen_before_collection' and len(sources) == 126,
            'frozen 126 sources')
    for name, pin in sources.items():
        descriptor(name, pin)
    require(all(not (ROOT / p).exists() for p in manifest['prospective_outputs'].values()),
            'all three empirical outputs absent at review')
    for d in manifest['inputs'].values():
        descriptor(d['path'], d)
    collection = read(manifest['inputs']['collection_plan']['path'])
    require(collection['status'] == 'frozen_before_collection' and len(collection['sources']) == 114,
            'collection metadata frozen')
    source_map(collection['sources'], sources)
    for d in collection['inputs'].values():
        descriptor(d['path'], d)
    inherited = read(collection['inputs']['collection_plan']['path'])
    require(collection['native_inputs'] == inherited['native_inputs'], 'unchanged inherited native descriptors')
    source_map(inherited['sources'], sources)
    require(collection['limits'] == {'native_seconds': 7200, 'rss_bytes': 4 * 1024**3,
                                    'output_bytes': 2 * 1024**3}, 'collection resource declaration')
    expected_cohort, global_case = [], 0
    arms = ('analytic', 'neural', 'period4_hold')
    for stage, count, starts in (('train', 9, (251000001, 252000001)),
                                 ('valid', 6, (253000001, 254000001))):
        for regime, first in zip(('lambda3', 'lambda4'), starts):
            for case in range(count):
                shift = global_case % 3
                for arm in arms[shift:] + arms[:shift]:
                    expected_cohort.append(dict(stage=stage, regime=regime, case=case, arm=arm,
                        seed=first + case, initial_hit=1 + case % 3,
                        episode_index=len(expected_cohort), episode_id=f'{stage}:{regime}:{first+case}:{arm}'))
                global_case += 1
    require(collection['cohort'] == expected_cohort, 'exact prospective 54 TRAIN and 36 VALID identities')
    require(collection['call_caps']['teacher_score'] == collection['call_caps']['tensorflow_value']
            == 18 * 2188 + 36 * (547 + 8 * 3) + 36 * 2188 == 138708, 'teacher call cap')
    seed = read(manifest['inputs']['seed_review']['path'])
    seeds = [x for first, count in ((251000001, 9), (252000001, 9), (253000001, 6),
              (254000001, 6), (255000001, 3), (256000001, 54)) for x in range(first, first + count)]
    require(seed['status'] == 'reserved_before_run' and seed['hits'] == [] and seed['seeds'] == seeds,
            '87 declared reserved seeds')
    qualifications = {}
    for role, aggpath in [('collection', collection['inputs']['engineering']['path']),
                         ('fit_audit', manifest['inputs']['fit_audit_qualification']['path'])]:
        agg = read(aggpath)
        require(agg['status'] == 'passed' and agg['sources_before'] == agg['sources_after'], 'aggregate success')
        source_map(agg['sources_after'], sources)
        closed(aggpath, agg)
        for entry in agg['inputs']:
            descriptor(entry['path'], entry)
            original, copied = path(entry['path']), path(aggpath).parent / entry['copied_receipt']
            require(original.read_bytes() == copied.read_bytes(), 'exact copied qualification receipt')
            q = read(original)
            require(q['status'] == 'passed' and q['sources_before'] == q['sources_after'], 'original qualification')
            for name, pin in q['sources_after'].items():
                descriptor(name, pin)
                if name in sources:
                    require(sources[name] == pin, 'qualified current source')
            if 'files' in q:
                # Inherited receipts were copied into an older aggregate; only direct original receipts
                # assert a whole-directory inventory here.
                if original.name == 'receipt.json':
                    closed(original, q)
            prefix = entry['copied_receipt'].removesuffix('receipt.json')
            old_prefix = original.name.removesuffix('receipt.json')
            q_results = [r for r in agg['results'] if r['qualification_receipt'] == entry['copied_receipt']]
            require(len(q_results) == len(q['results']) == 2, 'pytest and lint results')
            for index, (result, orig_result) in enumerate(zip(q_results, q['results'])):
                require(result['command'] == orig_result['command'] and result['exit_code'] == orig_result['exit_code'] == 0,
                        'original successful command')
                require(not orig_result.get('timed_out', False), 'qualification not timed out')
                for key in ('log', 'stdout_file', 'stderr_file'):
                    if key in result:
                        logname = result[key]
                        require(logname.startswith(prefix), 'aggregate log prefix')
                        actual = original.parent / (old_prefix + logname[len(prefix):])
                        require(actual.read_bytes() == (path(aggpath).parent / logname).read_bytes(), 'exact original log bytes')
                        descriptor(actual)
                logfile = result.get('log', result.get('stdout_file'))
                text = (path(aggpath).parent / logfile).read_text()
                if index == 0:
                    match = re.search(r'(\d+) passed in ', text)
                    require(match is not None, 'recorded pytest pass count')
                    count = int(match.group(1))
                else:
                    require('All checks passed!' in text, 'recorded Ruff success')
            qualifications[entry['role']] = {'receipt': descriptor(original), 'passed_cases': count}
    current = {k: qualifications[k] for k in ('model', 'collector', 'data', 'metrics', 'training', 'audit')}
    require({k: v['passed_cases'] for k, v in current.items()} ==
            {'model': 30, 'collector': 22, 'data': 12, 'metrics': 73, 'training': 10, 'audit': 10},
            '157 current fabricated cases')
    require(sum(v['passed_cases'] for v in current.values()) == manifest['new_completed_fabricated_tests'] == 157,
            'manifest fixture count')
    preserved = {}
    for role, field in [('collector', 'prior_failure'), ('audit', 'prior_qualification')]:
        q = read(current[role]['receipt']['path'])
        d = descriptor(q[field]['path'], q[field]['sha256'])
        prior = read(d['path'])
        closed(d['path'], prior)
        require(prior['sources_before'] == prior['sources_after'], 'earlier attempt source stability')
        preserved[role] = {'receipt': d, 'status': prior['status']}
    require(preserved['collector']['status'] == 'failed' and preserved['audit']['status'] == 'passed',
            'earlier qualification outcomes retained')
    prior_audit = read(preserved['audit']['receipt']['path'])
    prior_log = path(preserved['audit']['receipt']['path']).parent / prior_audit['results'][0]['log']
    require(re.search(r'9 passed in ', prior_log.read_text()) is not None, 'nine earlier auditor cases retained')
    capacity_plan = read(manifest['inputs']['capacity_plan']['path'])
    cap = read(manifest['inputs']['capacity_receipt']['path'])
    terminal = read(manifest['inputs']['capacity_terminal']['path'])
    launchpath = BASE / 'capacity-supervisor-01.launch.json'
    launch = read(launchpath)
    descriptor(launchpath, cap['supervision_sha256'])
    require(capacity_plan['status'] == 'frozen_before_synthetic_work', 'capacity frozen plan')
    require(cap['status'] == 'completed' and cap['complete'] and cap['admitted'] and
            cap['pending'] is None and cap['pending_emission'] is None, 'completed synthetic capacity')
    require(cap['sources'] == capacity_plan['sources'], 'capacity source identity')
    source_map(cap['sources'], sources)
    require(cap['engineering'] == capacity_plan['engineering'], 'capacity qualification identity')
    descriptor(cap['engineering']['path'], cap['engineering'])
    require(cap['plan_sha256'] == manifest['inputs']['capacity_plan']['sha256'], 'capacity plan identity')
    closed(manifest['inputs']['capacity_receipt']['path'], cap)
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and not terminal['timed_out']
            and terminal['error'] is None and terminal['clock_error'] is None, 'original capacity process success')
    require(terminal['cleanup']['reaped'] and terminal['cleanup']['group_absent'] and terminal['group_absent']
            and terminal['cleanup']['errors'] == [], 'original process cleanup')
    require(all(terminal.get(k) == v for k, v in launch.items()), 'original launch terminal identity')
    require(launch['pid'] == launch['pgid'] and launch['pid'] != launch['parent_pid']
            and launch['cwd'] == str(ROOT) and launch['cap_seconds'] == 120, 'capacity isolated process')
    command = [str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'scripts/qualify_otto_cross_query_capacity.py'),
               'run', '--plan', str(BASE / 'capacity-plan-01.json'), '--plan-sha256', cap['plan_sha256'],
               '--supervision', str(launchpath), '--output', str(BASE / 'capacity-01')]
    require(launch['command'] == command, 'exact original capacity command')
    require(launch['clock_source_sha256'] == sources['src/openjev/research/suspend_clock.py']
            and launch['watchdog_sha256'] == sources['scripts/supervise_dialogue_observation_v2.py'], 'supervisor source pins')
    require(launch['deadline_ns'] == launch['started_ns'] + 120 * 10**9
            and launch['started_ns'] <= cap['started_ns'] <= cap['finished_ns']
            <= terminal['finished_ns'] <= launch['deadline_ns'], 'original deadline and worker enclosure')
    require(terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1e9
            and cap['wall_seconds'] == (cap['finished_ns'] - cap['started_ns']) / 1e9, 'physical times')
    require(read(BASE / 'capacity-01/started.json') == {'launch': launch, 'started_ns': cap['started_ns']},
            'worker recorded original launch')
    require(cap['limits'] == capacity_plan['limits'] and 0 < cap['peak_rss_bytes'] <= cap['limits']['rss_bytes'],
            'capacity RSS and declared limits')
    require(sum(d['bytes'] for d in cap['files'].values()) + path(manifest['inputs']['capacity_receipt']['path']).stat().st_size
            <= cap['limits']['output_bytes'], 'capacity closed bytes cap')
    runtime = read(BASE / 'capacity-01/runtime.json')
    require(all(runtime[k] == v for k, v in capacity_plan['runtime'].items()) and runtime['threads'] == 1
            and runtime['interop_threads'] == 1 and not runtime['cuda_used'] and not runtime['mps_used'], 'capacity runtime metadata')
    summary = read(BASE / 'capacity-01/summary.json')
    families = summary['families']
    require([r['family'] for r in families] == capacity_plan['configuration']['families'] == cap['completed_families'],
            'all four capacity families')
    require(all(r['chunks'] == 69 and r['forward_rows'] == 6 * 2188 and r['optimizer_updates'] == 1
                and r['backward_chunks'] + r['no_grad_chunks'] == 69
                and {'output.weight', 'output.bias'} <= set(r['changed_parameters']) for r in families),
            'full length synthetic batch and actual output update')
    seconds = math.fsum(r['batch_seconds'] for r in families)
    projected = 1.5 * 3 * 720 * seconds + 120
    require(math.isclose(seconds, summary['sum_batch_seconds'], rel_tol=1e-15)
            and math.isclose(projected, summary['projected_seconds'], rel_tol=1e-15)
            and projected <= summary['threshold_seconds'] == 5400, 'saved capacity projection')
    saved_check = read(manifest['inputs']['capacity_saved_check']['path'])
    require(saved_check['status'] == 'passed' and saved_check['admitted']
            and len(saved_check['checks']) == 1501 and all(c['passed'] is True for c in saved_check['checks']),
            '1501 saved capacity checks')
    require(saved_check['source_pins'] == cap['sources'] and saved_check['projected_seconds'] == summary['projected_seconds'],
            'capacity saved-check joins')
    for name, pin in saved_check['inputs'].items():
        descriptor(name, pin)
    require(read(BASE / 'capacity-audit-01/receipt.json')['status'] == 'failed_before_audit',
            'original saved-check setup failure retained')
    required = {'scripts/collect_otto_cross_query_forecasts.py', 'tests/test_collect_otto_cross_query_forecasts.py',
        'src/openjev/research/otto_cross_query_scores.py', 'tests/test_otto_cross_query_scores.py',
        'src/openjev/research/otto_cross_query_data.py', 'tests/test_otto_cross_query_data.py',
        'src/openjev/research/otto_cross_query_metrics.py', 'tests/test_otto_cross_query_metrics.py',
        'scripts/train_otto_cross_query_forecasts.py', 'tests/test_train_otto_cross_query_forecasts.py',
        'scripts/audit_otto_cross_query_forecasts.py', 'tests/test_audit_otto_cross_query_forecasts.py'}
    require(required <= set(sources), 'all six scientific components and tests prospectively frozen')
    require(set(sources) == set(collection['sources']) | required | set(capacity_plan['sources'])
            | {'scripts/audit_otto_score_forecasts.py'}, 'exact prospective source union')
    for name, pin in sources.items():
        descriptor(name, pin)
    descriptor(manifest_path, PIN)
    for d in manifest['inputs'].values():
        descriptor(d['path'], d)
    require(all(not (ROOT / p).exists() for p in manifest['prospective_outputs'].values()), 'review completes before empirical outputs')
    return {'status': 'passed', 'clear': True, 'version': 'otto-cross-query-precollection-review-v1',
        'manifest': manifest_descriptor, 'checks_passed': checks, 'source_count': len(sources),
        'source_scope': 'Exact 114 collection sources plus 12 downstream/capacity source additions; all current bytes checked twice.',
        'qualification_cases': {k: v['passed_cases'] for k, v in current.items()},
        'new_current_fabricated_cases': 157, 'earlier_auditor_passed_case_executions_preserved': 9,
        'total_new_passed_case_executions_including_earlier_auditor': 166,
        'qualifications': current, 'preserved_attempts': preserved,
        'capacity': {'original_parent_returncode': 0, 'worker_seconds': cap['wall_seconds'],
                     'parent_seconds': terminal['wall_seconds'], 'saved_checks': 1501,
                     'projected_seconds': summary['projected_seconds'], 'threshold_seconds': 5400,
                     'source_pins_match': True, 'admitted': True},
        'scope': 'Saved metadata, log, source-byte, descriptor and original process closure review only.',
        'limits': ['No tests rerun, scientific modules imported, empirical arrays decoded, model/native/teacher/optimizer calls.',
                   'Historical native array descriptors inherited unchanged; their numerical payloads were not opened.',
                   'Saved test execution and synthetic timing truth inherited; capacity projection is not a runtime guarantee or efficacy result.',
                   'Seed-review receipt authenticated; historical seed scan not repeated.'],
        'empirical_arrays_decoded': 0, 'model_calls': 0, 'native_calls': 0, 'teacher_calls': 0,
        'optimizer_calls': 0, 'started_unix_ns': started, 'finished_unix_ns': time.time_ns()}


if __name__ == '__main__':
    try:
        receipt = main()
    except BaseException as exc:
        publication(OUT / 'receipt.json', {'status': 'failed', 'error': {'type': type(exc).__name__, 'message': str(exc)},
                    'checks_passed_before_failure': checks - 1, 'started_unix_ns': started, 'finished_unix_ns': time.time_ns(),
                    'scope': 'Metadata review only; no tests or scientific execution.'})
        raise
    publication(OUT / 'checked-files.json', dict(sorted(verified.items())))
    receipt['files'] = {name: descriptor(OUT / name) for name in ('review.py', 'checked-files.json')}
    publication(OUT / 'receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'checks': receipt['checks_passed'],
                      'receipt': descriptor(OUT / 'receipt.json')}, sort_keys=True))
