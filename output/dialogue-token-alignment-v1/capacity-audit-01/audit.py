"""Independent, bounded saved-capacity audit. Standard library only.

No model, NumPy, producer, cache-float, prediction or checkpoint loading. Raw
geometry reconstruction and observed clock/RSS truth remain outside this scope.
Invoke only after the parent supplies the terminal completion SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METHODS = ('flat_stratum', 'token_mean', 'token_aligned')
PHASES = ('train', 'evaluation')
VERSION = 'dialogue-token-alignment-capacity-v1'
RECIPE = {'seeds': [6201, 6202, 6203], 'epochs': 20, 'batch_size': 256,
          'microbatch_size': 32, 'strata': 3, 'warm_events': 1, 'measured_events': 3,
          'synthetic_seed': 410, 'dtype': 'float32', 'threads': 4, 'interop_threads': 1,
          'learning_rate': .001, 'weight_decay': .0001, 'gradient_clip': 1.,
          'probe_seconds': 300., 'rss_bytes': 6*1024**3, 'output_bytes': 64*1024**2,
          'study_ceiling_seconds': 3600., 'admission_seconds': 2880.}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def safe(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and p.parts, 'Unsafe member')
    p = Path(root)/p
    require(not p.is_symlink() and p.resolve().is_relative_to(Path(root).resolve()), 'Escaping/symlinked member')
    return p


def number(value, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), 'Invalid nonnegative numeric value')
    return value


def integer(value):
    require(type(value) is int and value >= 0, 'Invalid nonnegative integer')
    return value


def close(actual, expected):
    number(actual)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8), 'Saved timing arithmetic mismatch')


def pin(path, expected, bindings, size=None):
    p = Path(path).resolve()
    require(p.is_relative_to(ROOT) and p.is_file(), 'Input outside repository')
    require(type(expected) is str and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected), 'Malformed digest')
    require(digest(p) == expected and (size is None or p.stat().st_size == size), 'Input hash/size: '+str(p))
    bindings[str(p.relative_to(ROOT))] = {'sha256': expected, 'bytes': p.stat().st_size}


def check_manifest(directory, entries, bindings):
    for name, value in entries.items():
        pin(safe(directory, name), value['sha256'], bindings, integer(value['bytes']))


def authenticate(args):
    run, bindings = Path(args.run).resolve(), {}
    pin(run/'completed.json', args.completed_sha256, bindings)
    done = read(run/'completed.json')
    require(done['status'] == 'completed' and done['phase'] == 'run' and done['version'] == VERSION
            and done['technical_complete'] is True and done['no_retry'] is True, 'Complete sole capacity run required')
    require(done['encoder_calls'] == done['real_feature_arrays_decoded'] == 0 and done['quality_metrics'] is False,
            'Synthetic-only execution contract')
    expected = {'started.json', 'plan.json', 'events.jsonl', 'timings.jsonl', 'projection.json'}
    expected |= {f'profile-{phase}-{s}.json' for phase in PHASES for s in range(3)}
    require(set(done['files']) == expected and {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()}
            == expected | {'completed.json'}, 'Exact twelve-file run closure')
    check_manifest(run, done['files'], bindings)
    pin(run/'plan.json', done['plan_sha256'], bindings)
    plan, started = read(run/'plan.json'), read(run/'started.json')
    require(plan['recipe'] == started['recipe'] == RECIPE and plan['version'] == started['version'] == VERSION
            and plan['runtime'] == started['runtime'] and started['phase'] == 'run', 'Frozen recipe/runtime')
    require(plan['source_sha256'] == done['source_sha256'], 'Source-map identity')
    for name, value in plan['source_sha256'].items():
        pin(safe(ROOT, name), value, bindings)
    require(plan['source_sha256']['research/dialogue-token-alignment-capacity-protocol.md'] == plan['protocol_sha256'],
            'Protocol source identity')
    request = started['request']
    require(request['plan_sha256'] == done['plan_sha256'], 'Requested external plan identity')
    freeze_plan = Path(request['plan'])
    if not freeze_plan.is_absolute():
        freeze_plan = ROOT/freeze_plan
    pin(freeze_plan, done['plan_sha256'], bindings)
    freeze = freeze_plan.parent
    receipt = read(freeze/'completed.json')
    require(receipt['status'] == 'completed' and receipt['phase'] == 'freeze'
            and receipt['plan_sha256'] == done['plan_sha256'] and receipt['model_calls'] == 0, 'Completed geometry freeze')
    frozen_members = set(plan['payloads']) | {'plan.json'}
    require(set(receipt['files']) == frozen_members
            and {p.relative_to(freeze).as_posix() for p in freeze.rglob('*') if p.is_file()}
            == frozen_members | {'completed.json'}, 'Exact frozen member closure')
    # The external run pin binds the exact plan and all plan payload hashes.
    for name, value in plan['payloads'].items():
        require(receipt['files'][name] == value, 'Frozen plan/receipt payload identity')
    check_manifest(freeze, receipt['files'], bindings)
    for name, value in plan['source_sha256'].items():
        require(plan['payloads']['sources/'+name]['sha256'] == value, 'Source snapshot identity')
    schema = Path(plan['schema_cache'])
    if not schema.is_absolute():
        schema = ROOT/schema
    pin(schema/'completed.json', plan['schema_completed_sha256'], bindings)
    schema_done = read(schema/'completed.json')
    require(schema_done['status'] == 'completed' and schema_done['phase'] == 'encode', 'Completed schema preparation')
    costs = {k: schema_done.get(k) for k in ('wall_seconds', 'setup_seconds', 'work', 'payload_bytes')}
    require(costs == plan['schema_preparation'] == done['schema_preparation'], 'Separate schema-preparation costs')
    number(costs['wall_seconds'], positive=True)
    return run, freeze, done, plan, bindings


def coverage(plan, schedule):
    require(plan['fit_rows'] == 29211 and plan['evaluation_rows'] == 13599, 'Original fixed membership counts')
    require(set(plan['geometry_strata']) == set(schedule) == set(PHASES), 'Both workload phases')
    counts = {}
    for phase, total in (('train', 6900), ('evaluation', 162)):
        records = schedule[phase]
        require(len(records) == total, 'Complete schedule population')
        identities = [(r['seed'], r['epoch'], r['start']) for r in records]
        expected = [(seed, epoch, start) for seed in RECIPE['seeds']
                    for epoch in (range(20) if phase == 'train' else [None])
                    for start in range(0, plan['fit_rows'] if phase == 'train' else plan['evaluation_rows'], 256)]
        require(identities == expected, 'Every seed/epoch/tail schedule identity in original order')
        strata = plan['geometry_strata'][phase]
        require(len(strata) == 3 and [s['stratum'] for s in strata] == [0, 1, 2], 'Three ordered strata')
        require(sum(integer(s['batches_per_arm']) for s in strata) == total, 'Stratum populations sum to phase')
        # Independently replay the small scalar partition. Token geometry itself
        # remains producer-authenticated; no row/token or NumPy array is decoded.
        ranked = sorted(range(total), key=lambda i: (number(records[i]['workload']), i))
        quotient, extra = divmod(total, 3)
        cursor = 0
        for s in strata:
            size = quotient+int(s['stratum'] < extra)
            ids = ranked[cursor:cursor+size]; cursor += size
            maximum = max(records[i]['workload'] for i in ids)
            chosen = min(i for i in ids if records[i]['workload'] == maximum)
            require(s['batches_per_arm'] == size and s['representative_index'] == chosen
                    and s['representative'] == records[chosen]
                    and s['minimum_workload'] == min(records[i]['workload'] for i in ids)
                    and s['maximum_workload'] == maximum, 'Deterministic scalar stratum/representative')
            require(s['component_maxima'] == {k: max(records[i]['geometry'][k] for i in ids)
                    for k in records[chosen]['geometry']}, 'Component extrema from saved geometries')
        counts[phase] = total
    return counts


def check_events(run, done, plan):
    events = [read_text(line) for line in (run/'timings.jsonl').read_text().splitlines()]
    journal = [read_text(line) for line in (run/'events.jsonl').read_text().splitlines()]
    require(len(events) == len(journal) == done['events'] == 72, 'Seventy-two paid events')
    profiles = {(phase, s): read(run/f'profile-{phase}-{s}.json') for phase in PHASES for s in range(3)}
    expected = [(phase, s, method, rep) for phase in PHASES for s in range(3) for rep in range(4)
                for method in METHODS[rep % 3:]+METHODS[:rep % 3]]
    identities = [(e['phase'], e['stratum'], e['method'], e['repeat']) for e in events]
    require(identities == expected, 'Every event exactly once in the prescribed balanced order')
    for event, row in zip(events, journal, strict=True):
        require({k: v for k, v in event.items() if k != 'seconds'} == row, 'Journal/timing exact join')
        number(event['seconds'], positive=True)
        phase, s, method = event['phase'], event['stratum'], event['method']
        require(event['warmup'] is (event['repeat'] == 0), 'One warm plus three measured events')
        profile = profiles[phase, s]
        geometry = plan['geometry_strata'][phase][s]['representative']['geometry']
        require(profile['geometry'] == geometry and profile['phase'] == phase and profile['stratum'] == s
                and profile['seed'] == 410+PHASES.index(phase)*3+s, 'Profile geometry/identity')
        require(event['sample_sha256'] == profile['sample_sha256'], 'All arms share one synthetic sample')
        initial = profile['initializers']
        require(set(initial['full_state_sha256']) == set(initial['common_state_sha256']) == set(METHODS)
                and initial['full_state_sha256']['token_mean'] == initial['full_state_sha256']['token_aligned']
                and len(set(initial['common_state_sha256'].values())) == 1, 'Paired initializer receipts')
        rows = integer(geometry['rows']); micro = (rows+31)//32
        require(0 < rows <= 256 and event['rows'] == rows and event['microbatches'] == geometry['microbatches'] == micro,
                'Actual tail/microbatch counts')
        inv = event['normalization']
        require(inv['rows'] == rows and inv['batches'] == micro
                and inv['supported_candidates'] == geometry['supported_candidate_positions']
                and inv['masked_candidates'] == geometry['padded_candidate_positions']-inv['supported_candidates'],
                'Normalization covers every real row and padded candidate position')
        require(number(inv['max_abs_mass_error']) <= 2e-6
                and math.isfinite(inv['min_supported_log_prob']) and math.isfinite(inv['max_supported_log_prob'])
                and inv['min_supported_log_prob'] <= inv['max_supported_log_prob'] <= 0, 'Reported finite normalization invariants')
        number(event['synthetic_weighted_loss'])
        work = event['work']
        require(work['rows'] == rows and work['supported_candidate_positions'] == inv['supported_candidates']
                and work['padded_candidate_positions'] == geometry['padded_candidate_positions']
                and work['supported_token_positions'] == geometry['supported_context_token_positions']
                and work['padded_token_positions'] == geometry['padded_context_token_positions'], 'Actor count geometry join')
        if method != 'flat_stratum':
            for key in ('supported_schema_token_positions', 'padded_schema_token_positions', 'padded_pairwise_positions'):
                require(work['schema_'+key] == geometry[key], 'Schema actor count geometry join')
        if phase == 'evaluation':
            require(event['output_materialized_bytes'] == rows*12*4
                    and len(event['synthetic_output_sha256']) == 64, 'Evaluation output materialization')
        else:
            require('output_materialized_bytes' not in event and 'synthetic_output_sha256' not in event, 'Training output scope')
    forwards = sum(e['microbatches'] for e in events)
    backwards = sum(e['microbatches'] for e in events if e['phase'] == 'train')
    expected_progress = {'events_started': 72, 'events_completed': 72, 'optimizer_attempted': 36,
                         'optimizer_returned': 36, 'forward_attempted': forwards, 'forward_returned': forwards,
                         'backward_attempted': backwards, 'backward_returned': backwards, 'active': None}
    require(done['progress'] == expected_progress, 'Attempted and returned work coverage')
    return events, expected_progress


def read_text(text):
    # Keep the same strict JSON decoder for each journal record.
    def unique(items):
        result = {}
        for k, v in items:
            require(k not in result, 'Duplicate journal key')
            result[k] = v
        return result
    def invalid(value):
        raise ValueError('Nonfinite journal scalar: '+value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def projection_check(run, done, plan, events):
    observed = read(run/'projection.json')
    require(observed == done['projection'], 'Projection file/terminal identity')
    measured = math.fsum(e['seconds'] for e in events if not e['warmup'])
    warm = math.fsum(e['seconds'] for e in events if e['warmup'])
    overhead = number(observed['observed_nonmeasured_seconds'])
    wall, peak = number(done['wall_seconds'], positive=True), integer(done['process_lifetime_peak_rss_bytes'])
    require(wall <= 300 and peak <= 6*1024**3 and overhead+1e-8 >= warm
            and measured+overhead <= wall+1e-8, 'Whole wall/RSS and remainder coverage')
    expected_cells = {}
    for phase in PHASES:
        for s in plan['geometry_strata'][phase]:
            for method in METHODS:
                times = [e['seconds'] for e in events if (e['phase'], e['stratum'], e['method']) ==
                         (phase, s['stratum'], method) and not e['warmup']]
                require(len(times) == 3, 'Exactly three measured events per cell')
                maximum = max(times)
                expected_cells[phase, s['stratum'], method] = {
                    'phase': phase, 'stratum': s['stratum'], 'method': method,
                    'batches': s['batches_per_arm'], 'maximum_measured_seconds': maximum,
                    'projected_seconds': maximum*s['batches_per_arm']}
    cells = observed['cells']
    require(len(cells) == 18 and len({(c['phase'], c['stratum'], c['method']) for c in cells}) == 18, 'Eighteen unique cells')
    for c in cells:
        key = c['phase'], c['stratum'], c['method']
        require(key in expected_cells, 'Expected projection cell')
        expected = expected_cells[key]
        require(set(c) == set(expected) and c['batches'] == expected['batches'], 'Projection cell schema/population')
        close(c['maximum_measured_seconds'], expected['maximum_measured_seconds'])
        close(c['projected_seconds'], expected['projected_seconds'])
    total = overhead+math.fsum(c['projected_seconds'] for c in expected_cells.values())
    close(observed['projected_seconds'], total)
    require(observed['admission_seconds'] == 2880 and observed['study_ceiling_seconds'] == 3600, 'Frozen limits')
    admitted = total <= 2880 and peak <= 6*1024**3
    require(observed['admitted'] is admitted and done['admitted'] is admitted, 'Unchanged admission result')
    return {'admitted': admitted, 'projected_seconds': total, 'observed_nonmeasured_seconds': overhead,
            'measured_event_seconds': measured, 'warm_event_seconds': warm, 'cells': list(expected_cells.values()),
            'actual_probe_wall_seconds': wall, 'process_lifetime_peak_rss_bytes': peak,
            'schema_preparation_separate': done['schema_preparation']}


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def execute(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); source_pin = None; previous = None
    def timeout(_sig, _frame):
        raise TimeoutError('Independent audit sixty-second cap')
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'No existing process alarm')
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, 60.)
        source_pin = digest(__file__)
        run, freeze, done, plan, bindings = authenticate(args)
        counts = coverage(plan, read(freeze/'schedule.json'))
        events, progress = check_events(run, done, plan)
        result = projection_check(run, done, plan, events)
        require(sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) <= 64*1024**2, 'Execution output cap')
        for name, item in bindings.items():
            require(digest(ROOT/name) == item['sha256'] and (ROOT/name).stat().st_size == item['bytes'], 'End input stability')
        require(digest(__file__) == source_pin, 'End auditor source stability')
        result.update(status='completed', all_scoped_quantities_match=True,
                      execution_completed_sha256=args.completed_sha256, events=72, cells_checked=18,
                      batches_per_arm=counts, progress=progress,
                      scope=['Independent standard-library arithmetic and exact saved-file/source identities; no project or model imports.',
                             'Checks72 events,36 updates, microbatch/normalization coverage,18 maximum-of-three projections and unchanged2880s/6GiB admission.',
                             'Replays scalar schedule strata and component extrema from sealed geometry; does not reconstruct token geometry, row admission, actual arrays, gradients, clock truth or RSS truth.',
                             'Normalization and initializer values are checked as source-bound receipts, not independently rerun neural calculations.',
                             'Schema preparation is authenticated and shown separately, not added to or subtracted from the frozen study projection. No real quality or training authorization is established.'])
        write(out/'summary.json', result)
        write(out/'receipt.json', {'status': 'completed', 'source_sha256': source_pin,
              'execution_completed_sha256': args.completed_sha256, 'authenticated_inputs': bindings,
              'files': {'summary.json': {'sha256': digest(out/'summary.json'), 'bytes': (out/'summary.json').stat().st_size}},
              'wall_seconds': time.monotonic()-start, 'model_calls': 0, 'rng_calls': 0, 'float_array_reads': 0})
        return {'status': 'completed', 'summary_sha256': digest(out/'summary.json'), 'receipt_sha256': digest(out/'receipt.json')}
    except BaseException as error:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/'receipt.json').exists():
                (out/'receipt.json').rename(out/'late-receipt.json')
            write(out/'failed.json', {'status': 'failed', 'error': repr(error), 'source_sha256': source_pin,
                                    'execution_completed_sha256': args.completed_sha256, 'wall_seconds': time.monotonic()-start})
        except BaseException as secondary:  # noqa: BLE001 - retain the original audit error
            if callable(getattr(error, 'add_note', None)):
                error.add_note('Failure preservation error: '+repr(secondary))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--completed-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(execute(parser.parse_args())))
